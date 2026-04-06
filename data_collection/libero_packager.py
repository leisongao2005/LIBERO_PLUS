from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import h5py
import numpy as np
import robosuite.macros as macros
import robosuite.utils.transform_utils as T

import libero.libero.utils.utils as libero_utils
from libero.libero import get_libero_path
from libero.libero.envs import TASK_MAPPING

from data_collection.config import DataCollectionConfig
from data_collection.models import PackagingResult, TaskSpec, utc_now
from data_collection.raw_storage import ensure_dir, write_json


class LiberoPackager:
    def __init__(self, config: DataCollectionConfig):
        self.config = config

    def package_task(
        self,
        task: TaskSpec,
        raw_hdf5_path: str,
        overwrite: bool = False,
        output_path: str | None = None,
    ) -> PackagingResult:
        raw_hdf5 = Path(raw_hdf5_path)
        if not raw_hdf5.exists():
            raise FileNotFoundError(raw_hdf5)

        with h5py.File(raw_hdf5, "r") as source:
            env_name = source["data"].attrs["env"]
            env_kwargs = json.loads(source["data"].attrs["env_info"])
            problem_info = json.loads(source["data"].attrs["problem_info"])
            problem_name = problem_info["problem_name"]
            demos = list(source["data"].keys())
            bddl_file_name = source["data"].attrs["bddl_file_name"]

            target_path = Path(output_path) if output_path else Path(task.packaged_dataset_path)
            ensure_dir(target_path.parent)
            if target_path.exists() and not overwrite:
                raise FileExistsError(f"{target_path} already exists. Use overwrite=True to replace it.")

            env_kwargs = dict(env_kwargs)
            libero_utils.update_env_kwargs(
                env_kwargs,
                bddl_file_name=bddl_file_name,
                has_renderer=False,
                has_offscreen_renderer=self.config.use_camera_obs,
                ignore_done=True,
                use_camera_obs=self.config.use_camera_obs,
                camera_depths=self.config.use_depth,
                camera_names=["robot0_eye_in_hand", "agentview"],
                reward_shaping=True,
                control_freq=self.config.control_freq,
                camera_heights=self.config.camera_height,
                camera_widths=self.config.camera_width,
                camera_segmentations=None,
            )

            env = TASK_MAPPING[problem_name](**env_kwargs)
            total_len = 0

            try:
                with h5py.File(target_path, "w") as target:
                    grp = target.create_group("data")
                    grp.attrs["env_name"] = env_name
                    grp.attrs["problem_info"] = source["data"].attrs["problem_info"]
                    grp.attrs["macros_image_convention"] = macros.IMAGE_CONVENTION
                    grp.attrs["bddl_file_name"] = bddl_file_name
                    with open(bddl_file_name, "r") as bddl_stream:
                        grp.attrs["bddl_file_content"] = bddl_stream.read()
                    env_args = {
                        "type": 1,
                        "env_name": env_name,
                        "problem_name": problem_name,
                        "bddl_file": bddl_file_name,
                        "env_kwargs": env_kwargs,
                    }
                    grp.attrs["env_args"] = json.dumps(env_args)

                    cap_index = 5
                    for i, ep in enumerate(demos):
                        model_xml = source[f"data/{ep}"].attrs["model_file"]
                        reset_success = False
                        while not reset_success:
                            try:
                                env.reset()
                                reset_success = True
                            except Exception:
                                continue

                        model_xml = libero_utils.postprocess_model_xml(model_xml, {})
                        states = source[f"data/{ep}/states"][()]
                        actions = np.array(source[f"data/{ep}/actions"][()])
                        num_actions = actions.shape[0]
                        init_idx = 0
                        env.reset_from_xml_string(model_xml)
                        env.sim.reset()
                        env.sim.set_state_from_flattened(states[init_idx])
                        env.sim.forward()
                        model_xml = env.sim.model.get_xml()

                        ee_states = []
                        gripper_states = []
                        joint_states = []
                        robot_states = []
                        agentview_images = []
                        eye_in_hand_images = []
                        agentview_depths = []
                        eye_in_hand_depths = []
                        valid_index = []
                        replay_warnings: List[str] = []

                        for j, action in enumerate(actions):
                            obs, _, _, _ = env.step(action)
                            if j < num_actions - 1:
                                state_playback = env.sim.get_state().flatten()
                                err = np.linalg.norm(states[j + 1] - state_playback)
                                if err > 0.01:
                                    replay_warnings.append(
                                        f"Playback diverged by {err:.2f} for {ep} at step {j}"
                                    )

                            if j < cap_index:
                                continue

                            valid_index.append(j)
                            if "robot0_gripper_qpos" in obs:
                                gripper_states.append(obs["robot0_gripper_qpos"])
                            joint_states.append(obs["robot0_joint_pos"])
                            ee_state = np.hstack(
                                (obs["robot0_eef_pos"], T.quat2axisangle(obs["robot0_eef_quat"]))
                            )
                            ee_states.append(ee_state)
                            robot_states.append(env.get_robot_state_vector(obs))

                            if self.config.use_camera_obs:
                                agentview_images.append(obs["agentview_image"])
                                eye_in_hand_images.append(obs["robot0_eye_in_hand_image"])
                                if self.config.use_depth:
                                    agentview_depths.append(obs["agentview_depth"])
                                    eye_in_hand_depths.append(obs["robot0_eye_in_hand_depth"])

                        states = states[valid_index]
                        actions = actions[valid_index]
                        dones = np.zeros(len(actions), dtype=np.uint8)
                        rewards = np.zeros(len(actions), dtype=np.uint8)
                        if len(actions) > 0:
                            dones[-1] = 1
                            rewards[-1] = 1

                        ep_group = grp.create_group(f"demo_{i}")
                        obs_group = ep_group.create_group("obs")
                        obs_group.create_dataset("gripper_states", data=np.stack(gripper_states, axis=0))
                        obs_group.create_dataset("joint_states", data=np.stack(joint_states, axis=0))
                        obs_group.create_dataset("ee_states", data=np.stack(ee_states, axis=0))
                        obs_group.create_dataset("ee_pos", data=np.stack(ee_states, axis=0)[:, :3])
                        obs_group.create_dataset("ee_ori", data=np.stack(ee_states, axis=0)[:, 3:])
                        obs_group.create_dataset("agentview_rgb", data=np.stack(agentview_images, axis=0))
                        obs_group.create_dataset("eye_in_hand_rgb", data=np.stack(eye_in_hand_images, axis=0))
                        if self.config.use_depth:
                            obs_group.create_dataset("agentview_depth", data=np.stack(agentview_depths, axis=0))
                            obs_group.create_dataset(
                                "eye_in_hand_depth", data=np.stack(eye_in_hand_depths, axis=0)
                            )
                        ep_group.create_dataset("actions", data=actions)
                        ep_group.create_dataset("states", data=states)
                        ep_group.create_dataset("robot_states", data=np.stack(robot_states, axis=0))
                        ep_group.create_dataset("rewards", data=rewards)
                        ep_group.create_dataset("dones", data=dones)
                        ep_group.attrs["num_samples"] = len(agentview_images)
                        ep_group.attrs["model_file"] = model_xml
                        ep_group.attrs["init_state"] = states[init_idx]
                        if replay_warnings:
                            ep_group.attrs["replay_warnings"] = json.dumps(replay_warnings)
                        total_len += len(agentview_images)

                    grp.attrs["num_demos"] = len(demos)
                    grp.attrs["total"] = total_len
            finally:
                env.close()

        manifest = {
            "task_name": task.task_name,
            "raw_hdf5_path": str(raw_hdf5),
            "packaged_hdf5_path": str(target_path),
            "num_demos": len(demos),
            "total_samples": total_len,
            "created_at": utc_now(),
        }
        manifest_path = target_path.with_suffix(".manifest.json")
        write_json(manifest_path, manifest)
        return PackagingResult(
            task_name=task.task_name,
            raw_hdf5_path=str(raw_hdf5),
            packaged_hdf5_path=str(target_path),
            num_episodes=len(demos),
            total_samples=total_len,
            created_at=manifest["created_at"],
            manifest_path=str(manifest_path),
        )

    def package_many(
        self,
        task_to_raw_hdf5: Iterable[tuple[TaskSpec, str]],
        overwrite: bool = False,
    ) -> List[PackagingResult]:
        results = []
        for task, raw_hdf5_path in task_to_raw_hdf5:
            results.append(self.package_task(task, raw_hdf5_path, overwrite=overwrite))
        return results
