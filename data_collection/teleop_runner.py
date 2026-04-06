from __future__ import annotations

import datetime
import json
import os
import time
from glob import glob
from pathlib import Path
from typing import List

import h5py
import numpy as np
import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper

import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import TASK_MAPPING

from data_collection.config import DataCollectionConfig
from data_collection.models import TaskSpec, TeleopRunResult
from data_collection.session_manager import SessionManager


def _collect_human_trajectory(env, device, arm, env_configuration):
    reset_success = False
    while not reset_success:
        try:
            env.reset()
            reset_success = True
        except Exception:
            continue

    env.render()
    task_completion_hold_count = -1
    device.start_control()
    saving = True
    step_count = 0

    while True:
        step_count += 1
        active_robot = env.robots[0] if env_configuration == "bimanual" else env.robots[arm == "left"]
        action, _ = input2action(
            device=device,
            robot=active_robot,
            active_arm=arm,
            env_configuration=env_configuration,
        )
        if action is None:
            saving = False
            break

        env.step(action)
        env.render()

        if task_completion_hold_count == 0:
            break

        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1
            else:
                task_completion_hold_count = 10
        else:
            task_completion_hold_count = -1

    return saving, step_count


def _merge_tmp_rollouts_to_hdf5(
    tmp_directory: Path,
    out_dir: Path,
    env_info: str,
    problem_info: dict,
    bddl_file: str,
    remove_directory: List[str],
) -> Path:
    hdf5_path = out_dir / "demo.hdf5"
    with h5py.File(hdf5_path, "w") as h5_file:
        grp = h5_file.create_group("data")
        num_eps = 0
        env_name = None

        for ep_directory in os.listdir(tmp_directory):
            if ep_directory in remove_directory:
                continue
            state_paths = os.path.join(tmp_directory, ep_directory, "state_*.npz")
            states = []
            actions = []

            for state_file in sorted(glob(state_paths)):
                data = np.load(state_file, allow_pickle=True)
                env_name = str(data["env"])
                states.extend(data["states"])
                for action_info in data["action_infos"]:
                    actions.append(action_info["actions"])

            if not states:
                continue

            del states[-1]
            if len(states) != len(actions):
                raise ValueError(f"Mismatched states/actions in {ep_directory}")

            num_eps += 1
            ep_group = grp.create_group(f"demo_{num_eps}")
            xml_path = os.path.join(tmp_directory, ep_directory, "model.xml")
            with open(xml_path, "r") as xml_file:
                ep_group.attrs["model_file"] = xml_file.read()
            ep_group.create_dataset("states", data=np.array(states))
            ep_group.create_dataset("actions", data=np.array(actions))

        now = datetime.datetime.now()
        grp.attrs["date"] = f"{now.month}-{now.day}-{now.year}"
        grp.attrs["time"] = f"{now.hour}:{now.minute}:{now.second}"
        grp.attrs["repository_version"] = suite.__version__
        grp.attrs["env"] = env_name
        grp.attrs["env_info"] = env_info
        grp.attrs["problem_info"] = json.dumps(problem_info)
        grp.attrs["bddl_file_name"] = bddl_file
        with open(bddl_file, "r", encoding="utf-8") as bddl_stream:
            grp.attrs["bddl_file_content"] = bddl_stream.read()
    return hdf5_path


class TeleopRunner:
    def __init__(self, config: DataCollectionConfig, session_manager: SessionManager):
        self.config = config
        self.session_manager = session_manager

    def _make_device(self, env, device_name: str):
        if device_name == "keyboard":
            from robosuite.devices import Keyboard

            device = Keyboard(
                pos_sensitivity=self.config.pos_sensitivity,
                rot_sensitivity=self.config.rot_sensitivity,
            )
            env.viewer.add_keypress_callback("any", device.on_press)
            env.viewer.add_keyup_callback("any", device.on_release)
            env.viewer.add_keyrepeat_callback("any", device.on_press)
            return device

        if device_name == "spacemouse":
            from robosuite.devices import SpaceMouse

            return SpaceMouse(
                self.config.vendor_id,
                self.config.product_id,
                pos_sensitivity=self.config.pos_sensitivity,
                rot_sensitivity=self.config.rot_sensitivity,
            )

        raise ValueError(f"Unsupported device: {device_name}")

    def collect_task(self, session, task: TaskSpec, num_demonstrations: int = 1, notes: str = "") -> TeleopRunResult:
        episode = self.session_manager.create_episode_record(session, task, notes=notes)
        controller_config = load_controller_config(default_controller=self.config.controller)
        config = {
            "robots": list(self.config.robots),
            "controller_configs": controller_config,
        }

        problem_info = BDDLUtils.get_problem_info(task.bddl_file)
        if "TwoArm" in problem_info["problem_name"]:
            config["env_configuration"] = self.config.env_configuration

        env = TASK_MAPPING[problem_info["problem_name"]](
            bddl_file_name=task.bddl_file,
            **config,
            has_renderer=True,
            has_offscreen_renderer=False,
            render_camera=self.config.camera,
            ignore_done=True,
            use_camera_obs=False,
            reward_shaping=True,
            control_freq=self.config.control_freq,
        )
        env = VisualizationWrapper(env)

        session_root = Path(session.root_dir)
        episode_dir = session_root / "episodes" / episode.episode_id
        tmp_directory = episode_dir / "raw" / "tmp"
        tmp_directory.mkdir(parents=True, exist_ok=True)
        env = DataCollectionWrapper(env, str(tmp_directory))
        device = self._make_device(env, session.device)
        env_info = json.dumps(config)

        output_dir = episode_dir / "raw"
        output_dir.mkdir(parents=True, exist_ok=True)
        remove_directory: List[str] = []
        saved_count = 0
        try:
            while saved_count < num_demonstrations:
                saving, _ = _collect_human_trajectory(
                    env,
                    device,
                    self.config.arm,
                    self.config.env_configuration,
                )
                if saving:
                    saved_count += 1
        finally:
            env.close()

        merged_hdf5_path = _merge_tmp_rollouts_to_hdf5(
            tmp_directory=tmp_directory,
            out_dir=output_dir,
            env_info=env_info,
            problem_info=problem_info,
            bddl_file=task.bddl_file,
            remove_directory=remove_directory,
        )

        episode.state = "saved"
        episode.raw_directory = str(output_dir)
        episode.merged_hdf5_path = str(merged_hdf5_path)
        self.session_manager.save_episode_record(session, episode)
        self.session_manager.update_task_progress(session, task.task_name, delta=saved_count)
        return TeleopRunResult(
            task=task,
            episode_id=episode.episode_id,
            raw_directory=str(output_dir),
            merged_hdf5_path=str(merged_hdf5_path),
            saved=True,
            num_demos=saved_count,
            metadata={"session_id": session.session_id, "notes": notes},
        )
