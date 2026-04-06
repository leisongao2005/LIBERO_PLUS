from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from data_collection.config import DataCollectionConfig
from data_collection.models import ValidationResult, utc_now


class DatasetValidator:
    def __init__(self, config: DataCollectionConfig):
        self.config = config

    def validate_packaged_dataset(self, dataset_path: str, expected_task_name: str | None = None) -> ValidationResult:
        path = Path(dataset_path)
        errors = []
        warnings = []
        stats = {}

        if not path.exists():
            return ValidationResult(
                path=str(path),
                passed=False,
                created_at=utc_now(),
                errors=[f"Dataset not found: {path}"],
            )

        with h5py.File(path, "r") as data:
            demos = sorted(list(data["data"].keys()))
            if not demos:
                errors.append("No demos found under /data")

            traj_lengths = []
            action_min = np.inf
            action_max = -np.inf
            for ep in demos:
                actions = data[f"data/{ep}/actions"][()]
                if actions.ndim != 2:
                    errors.append(f"{ep} actions should be rank-2, got {actions.shape}")
                    continue
                if actions.shape[1] != 7:
                    errors.append(f"{ep} action dimension should be 7, got {actions.shape[1]}")
                traj_lengths.append(actions.shape[0])
                action_min = min(action_min, float(np.min(actions)))
                action_max = max(action_max, float(np.max(actions)))

                obs = data[f"data/{ep}/obs"]
                for required_key in [
                    "agentview_rgb",
                    "eye_in_hand_rgb",
                    "joint_states",
                    "gripper_states",
                    "ee_states",
                ]:
                    if required_key not in obs:
                        errors.append(f"{ep} missing obs key: {required_key}")

                lengths = []
                for key in ["agentview_rgb", "eye_in_hand_rgb", "joint_states", "gripper_states", "ee_states"]:
                    if key in obs:
                        lengths.append(obs[key].shape[0])
                lengths.append(actions.shape[0])
                if len(set(lengths)) > 1:
                    errors.append(f"{ep} has misaligned sequence lengths: {lengths}")

                dones = data[f"data/{ep}/dones"][()]
                if len(dones) == 0 or int(dones[-1]) != 1:
                    errors.append(f"{ep} does not terminate with done=1")

                if "replay_warnings" in data[f"data/{ep}"].attrs:
                    warnings.extend(json.loads(data[f"data/{ep}"].attrs["replay_warnings"]))

            problem_info_raw = data["data"].attrs.get("problem_info", "")
            if not problem_info_raw:
                errors.append("Missing data.problem_info attribute")
            else:
                problem_info = json.loads(problem_info_raw)
                language_instruction = "".join(problem_info.get("language_instruction", ""))
                if not language_instruction.strip():
                    errors.append("Language instruction is empty")
                if expected_task_name and expected_task_name not in path.name:
                    warnings.append(
                        f"Expected task name {expected_task_name} not reflected in dataset path {path.name}"
                    )

            env_args_raw = data["data"].attrs.get("env_args", "")
            if not env_args_raw:
                errors.append("Missing data.env_args attribute")

            if action_min < self.config.min_validation_action or action_max > self.config.max_validation_action:
                errors.append(
                    f"Action bounds must stay in "
                    f"[{self.config.min_validation_action}, {self.config.max_validation_action}], "
                    f"got [{action_min}, {action_max}]"
                )

            stats = {
                "num_demos": len(demos),
                "total_transitions": int(np.sum(traj_lengths)) if traj_lengths else 0,
                "action_min": action_min if action_min is not np.inf else None,
                "action_max": action_max if action_max is not -np.inf else None,
                "traj_length_mean": float(np.mean(traj_lengths)) if traj_lengths else 0.0,
            }

        return ValidationResult(
            path=str(path),
            passed=not errors,
            created_at=utc_now(),
            errors=errors,
            warnings=warnings,
            stats=stats,
        )
