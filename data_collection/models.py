from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


@dataclass
class TaskSpec:
    task_id: int
    benchmark_name: str
    task_name: str
    language_instruction: str
    bddl_file: str
    init_states_file: str
    problem_folder: str
    packaged_dataset_path: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CollectionSession:
    session_id: str
    benchmark_name: str
    created_at: str
    updated_at: str
    root_dir: str
    device: str = "keyboard"
    camera: str = "agentview"
    target_per_task: int = 1
    operator: str = ""
    notes: str = ""
    tasks: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeRecord:
    episode_id: str
    session_id: str
    task_name: str
    benchmark_name: str
    language_instruction: str
    state: str
    created_at: str
    updated_at: str
    device: str
    camera: str
    notes: str = ""
    quality: str = ""
    failure_reason: str = ""
    raw_directory: str = ""
    merged_hdf5_path: str = ""
    packaged_dataset_path: str = ""
    export_path: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PackagingResult:
    task_name: str
    raw_hdf5_path: str
    packaged_hdf5_path: str
    num_episodes: int
    total_samples: int
    created_at: str
    manifest_path: str
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationResult:
    path: str
    passed: bool
    created_at: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExportResult:
    export_name: str
    output_dir: str
    num_episodes: int
    num_frames: int
    created_at: str
    task_names: List[str]
    state_schema: str
    cameras: List[str]
    quantiles_path: str = ""
    manifest_path: str = ""
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TeleopRunResult:
    task: TaskSpec
    episode_id: str
    raw_directory: str
    merged_hdf5_path: str
    saved: bool
    num_demos: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task.to_dict(),
            "episode_id": self.episode_id,
            "raw_directory": self.raw_directory,
            "merged_hdf5_path": self.merged_hdf5_path,
            "saved": self.saved,
            "num_demos": self.num_demos,
            "metadata": self.metadata,
        }
