from __future__ import annotations

import json
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from data_collection.config import DataCollectionConfig
from data_collection.libero_packager import LiberoPackager
from data_collection.pi05_exporter import Pi05Exporter
from data_collection.session_manager import SessionManager
from data_collection.task_registry import TaskRegistry
from data_collection.teleop_runner import TeleopRunner
from data_collection.validation import DatasetValidator


class CollectorGUI:
    def __init__(self):
        self.config = DataCollectionConfig.default()
        self.session_manager = SessionManager(self.config)
        self.task_registry = TaskRegistry(self.config)
        self.teleop_runner = TeleopRunner(self.config, self.session_manager)
        self.packager = LiberoPackager(self.config)
        self.validator = DatasetValidator(self.config)
        self.exporter = Pi05Exporter(self.config)

        self.root = tk.Tk()
        self.root.title("LIBERO Data Collection")
        self.root.geometry("1100x720")

        self.session_id_var = tk.StringVar()
        self.device_var = tk.StringVar(value=self.config.default_device)
        self.camera_var = tk.StringVar(value=self.config.camera)
        self.target_var = tk.IntVar(value=self.config.target_per_task)
        self.search_var = tk.StringVar()
        self.notes_var = tk.StringVar()
        self.export_name_var = tk.StringVar(value="pi05_export")
        self.status_var = tk.StringVar(value="Ready")

        self.task_rows = {}
        self.selected_task_name = None
        self._build_layout()
        self.refresh_tasks()

    def _build_layout(self):
        control = ttk.Frame(self.root, padding=12)
        control.pack(fill=tk.X)

        ttk.Label(control, text="Session").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(control, textvariable=self.session_id_var, width=18).grid(row=0, column=1, sticky=tk.W)
        ttk.Button(control, text="Create Session", command=self.create_session).grid(row=0, column=2, padx=8)
        ttk.Button(control, text="Refresh Status", command=self.refresh_tasks).grid(row=0, column=3)

        ttk.Label(control, text="Device").grid(row=1, column=0, sticky=tk.W)
        ttk.Combobox(control, textvariable=self.device_var, values=["keyboard", "spacemouse"], width=15).grid(
            row=1, column=1, sticky=tk.W
        )
        ttk.Label(control, text="Camera").grid(row=1, column=2, sticky=tk.W)
        ttk.Entry(control, textvariable=self.camera_var, width=15).grid(row=1, column=3, sticky=tk.W)
        ttk.Label(control, text="Target / Task").grid(row=1, column=4, sticky=tk.W)
        ttk.Entry(control, textvariable=self.target_var, width=8).grid(row=1, column=5, sticky=tk.W)

        ttk.Label(control, text="Search").grid(row=2, column=0, sticky=tk.W)
        ttk.Entry(control, textvariable=self.search_var, width=40).grid(row=2, column=1, columnspan=2, sticky=tk.W)
        ttk.Button(control, text="Filter", command=self.refresh_tasks).grid(row=2, column=3, padx=8)
        ttk.Label(control, text="Notes").grid(row=2, column=4, sticky=tk.W)
        ttk.Entry(control, textvariable=self.notes_var, width=30).grid(row=2, column=5, sticky=tk.W)

        body = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        left = ttk.Frame(body, padding=8)
        right = ttk.Frame(body, padding=8)
        body.add(left, weight=3)
        body.add(right, weight=2)

        self.tree = ttk.Treeview(left, columns=("instruction", "dataset"), show="headings", height=25)
        self.tree.heading("instruction", text="Instruction")
        self.tree.heading("dataset", text="Packaged Dataset")
        self.tree.column("instruction", width=430)
        self.tree.column("dataset", width=330)
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind("<<TreeviewSelect>>", self.on_select_task)

        ttk.Button(right, text="Collect Selected Task", command=self.collect_selected_task).pack(fill=tk.X, pady=4)
        ttk.Button(right, text="Package Latest Raw Episode", command=self.package_selected_task).pack(fill=tk.X, pady=4)
        ttk.Button(right, text="Validate Packaged Dataset", command=self.validate_selected_task).pack(fill=tk.X, pady=4)
        ttk.Button(right, text="Export Selected Task To pi05", command=self.export_selected_task).pack(fill=tk.X, pady=4)
        ttk.Label(right, text="Export Name").pack(anchor=tk.W, pady=(12, 0))
        ttk.Entry(right, textvariable=self.export_name_var).pack(fill=tk.X)
        ttk.Label(right, text="Details").pack(anchor=tk.W, pady=(12, 0))
        self.details = tk.Text(right, height=24, wrap=tk.WORD)
        self.details.pack(fill=tk.BOTH, expand=True)

        status_bar = ttk.Label(self.root, textvariable=self.status_var, padding=8)
        status_bar.pack(fill=tk.X)

    def create_session(self):
        session = self.session_manager.create_session(
            device=self.device_var.get(),
            camera=self.camera_var.get(),
            target_per_task=self.target_var.get(),
        )
        self.session_id_var.set(session.session_id)
        self.status_var.set(f"Created session {session.session_id}")

    def refresh_tasks(self):
        search = self.search_var.get().strip().lower()
        for row in self.tree.get_children():
            self.tree.delete(row)
        for task in self.task_registry.load_tasks():
            if search and search not in task.task_name.lower() and search not in task.language_instruction.lower():
                continue
            self.tree.insert("", tk.END, iid=task.task_name, values=(task.language_instruction, task.packaged_dataset_path))

    def on_select_task(self, _event=None):
        selection = self.tree.selection()
        if not selection:
            return
        task_name = selection[0]
        self.selected_task_name = task_name
        task = self.task_registry.get_task(task_name)
        self.details.delete("1.0", tk.END)
        self.details.insert(tk.END, json.dumps(task.to_dict(), indent=2))

    def _run_background(self, target, success_message: str):
        def worker():
            try:
                result = target()
                self.root.after(0, lambda: self._show_result(success_message, result))
            except Exception as exc:  # pragma: no cover - UI error reporting
                self.root.after(0, lambda: messagebox.showerror("Error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _show_result(self, message: str, result):
        self.status_var.set(message)
        self.details.delete("1.0", tk.END)
        payload = result.to_dict() if hasattr(result, "to_dict") else result
        self.details.insert(tk.END, json.dumps(payload, indent=2))

    def _require_session_and_task(self):
        if not self.session_id_var.get():
            raise ValueError("Create or enter a session id first")
        if not self.selected_task_name:
            raise ValueError("Select a task first")
        return self.session_manager.load_session(self.session_id_var.get()), self.task_registry.get_task(self.selected_task_name)

    def collect_selected_task(self):
        session, task = self._require_session_and_task()
        notes = self.notes_var.get()
        self._run_background(
            lambda: self.teleop_runner.collect_task(session, task, num_demonstrations=1, notes=notes),
            f"Collected task {task.task_name}",
        )

    def _latest_raw_hdf5(self, session_id: str, task_name: str) -> Path:
        session = self.session_manager.load_session(session_id)
        episodes = [
            ep for ep in self.session_manager.load_episode_records(session) if ep.task_name == task_name and ep.merged_hdf5_path
        ]
        if not episodes:
            raise FileNotFoundError("No saved raw episode found for selected task")
        episodes.sort(key=lambda ep: ep.created_at, reverse=True)
        return Path(episodes[0].merged_hdf5_path)

    def package_selected_task(self):
        session, task = self._require_session_and_task()
        raw_hdf5 = self._latest_raw_hdf5(session.session_id, task.task_name)
        self._run_background(
            lambda: self.packager.package_task(task, str(raw_hdf5), overwrite=True),
            f"Packaged task {task.task_name}",
        )

    def validate_selected_task(self):
        _, task = self._require_session_and_task()
        self._run_background(
            lambda: self.validator.validate_packaged_dataset(task.packaged_dataset_path, expected_task_name=task.task_name),
            f"Validated task {task.task_name}",
        )

    def export_selected_task(self):
        _, task = self._require_session_and_task()
        export_name = self.export_name_var.get().strip() or "pi05_export"
        self._run_background(
            lambda: self.exporter.export_tasks([task], [task.packaged_dataset_path], export_name=export_name),
            f"Exported task {task.task_name}",
        )

    def run(self):
        self.root.mainloop()


def main():
    CollectorGUI().run()


if __name__ == "__main__":
    main()
