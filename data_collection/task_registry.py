from __future__ import annotations

from pathlib import Path
from typing import List

from libero.libero.benchmark import get_benchmark

from data_collection.config import DataCollectionConfig
from data_collection.models import TaskSpec


class TaskRegistry:
    def __init__(self, config: DataCollectionConfig):
        self.config = config
        self._tasks: List[TaskSpec] | None = None

    def load_tasks(self) -> List[TaskSpec]:
        if self._tasks is not None:
            return self._tasks

        benchmark_cls = get_benchmark(self.config.benchmark_name)
        benchmark = benchmark_cls(task_order_index=0)
        tasks: List[TaskSpec] = []
        for task_id in range(benchmark.get_num_tasks()):
            task = benchmark.get_task(task_id)
            tasks.append(
                TaskSpec(
                    task_id=task_id,
                    benchmark_name=self.config.benchmark_name,
                    task_name=task.name,
                    language_instruction=task.language,
                    bddl_file=str(Path(benchmark.get_task_bddl_file_path(task_id))),
                    init_states_file=str(
                        self.config.init_states_root / task.problem_folder / task.init_states_file
                    ),
                    problem_folder=task.problem_folder,
                    packaged_dataset_path=str(
                        self.config.datasets_root / benchmark.get_task_demonstration(task_id)
                    ),
                )
            )
        self._tasks = tasks
        return tasks

    def get_task(self, task_name: str) -> TaskSpec:
        for task in self.load_tasks():
            if task.task_name == task_name:
                return task
        raise KeyError(f"Unknown task: {task_name}")

    def get_task_by_id(self, task_id: int) -> TaskSpec:
        tasks = self.load_tasks()
        if task_id < 0 or task_id >= len(tasks):
            raise IndexError(f"Task id {task_id} out of range")
        return tasks[task_id]

    def list_task_names(self) -> List[str]:
        return [task.task_name for task in self.load_tasks()]
