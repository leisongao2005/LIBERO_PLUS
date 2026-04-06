from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from libero.libero import get_libero_path


@dataclass
class DataCollectionConfig:
    root_dir: Path
    benchmark_name: str = "libero_plus_train_subtasks"
    controller: str = "OSC_POSE"
    camera: str = "agentview"
    default_device: str = "keyboard"
    robots: tuple[str, ...] = ("Panda",)
    env_configuration: str = "single-arm-opposed"
    arm: str = "right"
    pos_sensitivity: float = 1.5
    rot_sensitivity: float = 1.0
    control_freq: int = 20
    target_per_task: int = 10
    use_camera_obs: bool = True
    camera_height: int = 128
    camera_width: int = 128
    use_depth: bool = False
    max_validation_action: float = 1.0
    min_validation_action: float = -1.0
    vendor_id: int = 9583
    product_id: int = 50734
    export_state_schema: str = "ee_gripper"

    @property
    def runs_dir(self) -> Path:
        return self.root_dir / "runs"

    @property
    def exports_dir(self) -> Path:
        return self.root_dir / "exports"

    @property
    def pi05_exports_dir(self) -> Path:
        return self.exports_dir / "pi05"

    @property
    def datasets_root(self) -> Path:
        return Path(get_libero_path("datasets"))

    @property
    def bddl_root(self) -> Path:
        return Path(get_libero_path("bddl_files"))

    @property
    def init_states_root(self) -> Path:
        return Path(get_libero_path("init_states"))

    @classmethod
    def default(cls) -> "DataCollectionConfig":
        return cls(root_dir=Path(__file__).resolve().parent)

    def ensure_directories(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        self.pi05_exports_dir.mkdir(parents=True, exist_ok=True)
