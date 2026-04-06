from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence

import h5py
import numpy as np

from data_collection.config import DataCollectionConfig
from data_collection.models import ExportResult, TaskSpec, utc_now
from data_collection.raw_storage import ensure_dir, write_json


class Pi05Exporter:
    """
    Exports packaged LIBERO HDF5s into a pi05-ready dataset directory.

    The output is organized as:
    - manifest.json
    - stats/quantiles.json
    - episodes/<task_name>__<demo_id>.npz
    - episodes/index.jsonl

    This is designed as a deterministic bridge from the repo's canonical HDF5
    archive into a compact dataset layout that can be further converted or
    ingested by downstream LeRobot / pi05 tooling.
    """

    def __init__(self, config: DataCollectionConfig):
        self.config = config

    def _build_state(self, obs_group, state_schema: str) -> np.ndarray:
        if state_schema == "ee_gripper":
            return np.concatenate([obs_group["ee_states"][()], obs_group["gripper_states"][()]], axis=1)
        if state_schema == "joint_gripper_ee":
            return np.concatenate(
                [obs_group["joint_states"][()], obs_group["gripper_states"][()], obs_group["ee_states"][()]],
                axis=1,
            )
        raise ValueError(f"Unsupported state schema: {state_schema}")

    def export_tasks(
        self,
        tasks: Sequence[TaskSpec],
        packaged_paths: Sequence[str],
        export_name: str,
        cameras: Iterable[str] = ("agentview_rgb", "eye_in_hand_rgb"),
        state_schema: str | None = None,
    ) -> ExportResult:
        if len(tasks) != len(packaged_paths):
            raise ValueError("tasks and packaged_paths must be aligned")

        state_schema = state_schema or self.config.export_state_schema
        output_dir = ensure_dir(self.config.pi05_exports_dir / export_name)
        episodes_dir = ensure_dir(output_dir / "episodes")
        stats_dir = ensure_dir(output_dir / "stats")
        index_path = output_dir / "episodes" / "index.jsonl"
        cameras = list(cameras)

        frame_count = 0
        episode_count = 0
        all_actions: List[np.ndarray] = []
        all_states: List[np.ndarray] = []
        index_lines: List[str] = []

        for task, packaged_path in zip(tasks, packaged_paths):
            with h5py.File(packaged_path, "r") as data:
                problem_info = json.loads(data["data"].attrs["problem_info"])
                task_text = "".join(problem_info["language_instruction"]).strip('"')
                for demo_name in sorted(data["data"].keys()):
                    demo = data[f"data/{demo_name}"]
                    obs = demo["obs"]
                    actions = demo["actions"][()]
                    state = self._build_state(obs, state_schema)
                    payload = {
                        "task_name": task.task_name,
                        "task_text": task_text,
                        "actions": actions,
                        "state": state,
                        "dones": demo["dones"][()],
                        "rewards": demo["rewards"][()],
                    }
                    for camera in cameras:
                        if camera in obs:
                            payload[camera] = obs[camera][()]

                    file_name = f"{task.task_name}__{demo_name}.npz"
                    np.savez_compressed(episodes_dir / file_name, **payload)
                    index_lines.append(
                        json.dumps(
                            {
                                "episode_file": file_name,
                                "task_name": task.task_name,
                                "task_text": task_text,
                                "num_frames": int(actions.shape[0]),
                                "action_dim": int(actions.shape[1]),
                                "state_dim": int(state.shape[1]),
                                "cameras": [camera for camera in cameras if camera in obs],
                            }
                        )
                    )
                    all_actions.append(actions)
                    all_states.append(state)
                    frame_count += int(actions.shape[0])
                    episode_count += 1

        index_path.write_text("\n".join(index_lines) + ("\n" if index_lines else ""))

        action_matrix = np.concatenate(all_actions, axis=0) if all_actions else np.zeros((0, 7))
        state_matrix = np.concatenate(all_states, axis=0) if all_states else np.zeros((0, 8))
        quantiles = {
            "action": self._compute_quantiles(action_matrix),
            "state": self._compute_quantiles(state_matrix),
        }
        quantiles_path = stats_dir / "quantiles.json"
        write_json(quantiles_path, quantiles)

        manifest = {
            "export_name": export_name,
            "created_at": utc_now(),
            "num_episodes": episode_count,
            "num_frames": frame_count,
            "state_schema": state_schema,
            "cameras": cameras,
            "tasks": [task.task_name for task in tasks],
            "layout": {
                "episodes_dir": "episodes",
                "index_jsonl": str(index_path.relative_to(output_dir)),
                "quantiles_json": str(quantiles_path.relative_to(output_dir)),
            },
            "notes": [
                "This export is a deterministic pi05-ready bridge built from packaged LIBERO HDF5 files.",
                "If strict LeRobot v3 ingestion is required, use this manifest and per-episode payloads as the source for the final converter.",
            ],
        }
        manifest_path = output_dir / "manifest.json"
        write_json(manifest_path, manifest)
        return ExportResult(
            export_name=export_name,
            output_dir=str(output_dir),
            num_episodes=episode_count,
            num_frames=frame_count,
            created_at=manifest["created_at"],
            task_names=[task.task_name for task in tasks],
            state_schema=state_schema,
            cameras=cameras,
            quantiles_path=str(quantiles_path),
            manifest_path=str(manifest_path),
        )

    @staticmethod
    def _compute_quantiles(values: np.ndarray) -> dict:
        if values.size == 0:
            return {"q01": [], "q10": [], "q50": [], "q90": [], "q99": []}
        return {
            "q01": np.quantile(values, 0.01, axis=0).tolist(),
            "q10": np.quantile(values, 0.10, axis=0).tolist(),
            "q50": np.quantile(values, 0.50, axis=0).tolist(),
            "q90": np.quantile(values, 0.90, axis=0).tolist(),
            "q99": np.quantile(values, 0.99, axis=0).tolist(),
        }
