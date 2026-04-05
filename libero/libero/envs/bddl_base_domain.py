import numpy as np
import os
import robosuite.utils.transform_utils as T

from copy import deepcopy
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import SequentialCompositeSampler
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.mjcf_utils import CustomMaterial
import robosuite.macros as macros

import mujoco

import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs.robots import *
from libero.libero.envs.utils import *
from libero.libero.envs.object_states import *
from libero.libero.envs.objects import *
from libero.libero.envs.regions import *
from libero.libero.envs.arenas import *
from libero.libero.envs.predicates import eval_predicate_fn

import time

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

TASK_MAPPING = {}


def register_problem(target_class):
    """Register a problem class. Mapping is case-INsensitive."""
    TASK_MAPPING[target_class.__name__.lower()] = target_class


class BDDLBaseDomain(SingleArmEnv):
    """Base domain that parses BDDL files and provides subtask reward support."""

    def __init__(
        self,
        bddl_file_name,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        use_latch=False,
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        subtask_reward=False,
        subtask_reward_scale=0.5,
        subtask_confirmation_steps=10,
        track_subtask_info=False,
        placement_initializer=None,
        object_property_initializers=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mujoco",
        table_full_size=(1.0, 1.0, 0.05),
        workspace_offset=(0.0, 0.0, 0.0),
        arena_type="table",
        scene_xml="scenes/libero_base_style.xml",
        scene_properties={},
        **kwargs,
    ):
        t0 = time.time()
        # settings for table top (hardcoded since it's not an essential part of the environment)
        self.workspace_offset = workspace_offset
        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        # when True, reward() returns a fractional subtask reward instead of
        # a sparse 0/1 signal; step() also populates info["subtask_rewards"],
        # info["subtask_reward_increment"], and info["subtask_reward_delta"].
        self.subtask_reward = subtask_reward
        self.subtask_reward_scale = subtask_reward_scale
        self.subtask_confirmation_steps = subtask_confirmation_steps
        # when True, info["subtask_info"] is populated every step regardless of
        # subtask_reward, using dry_run=True so one-shot tracking is unaffected.
        # Useful for eval-time diagnostics without shaping the reward signal.
        self.track_subtask_info = track_subtask_info
        # Caches _evaluate_subtask_rewards() result produced inside reward() so
        # step() can copy it into info without a second physics evaluation.
        self._subtask_satisfied_cache: dict = {}
        # Tracks which subtask names have been satisfied at any point in the
        # current episode.  Once a subtask enters this set it stays credited
        # (one-shot reward), preventing reward cycling.  Cleared on reset().
        self._subtask_ever_satisfied: set = set()
        # Tracks pending confirmations for delayed On / In subtasks.
        self._subtask_confirmation_counts: dict = {}

        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer
        self.conditional_placement_initializer = None
        self.conditional_placement_on_objects_initializer = None

        # object property initializer

        if object_property_initializers is not None:
            self.object_property_initializers = object_property_initializers
        else:
            self.object_property_initializers = list()

        # Keep track of movable objects in the tasks
        self.objects_dict = {}
        # Kepp track of fixed objects in the tasks
        self.fixtures_dict = {}
        self.object_sites_dict = {}
        self.object_states_dict = {}
        self.tracking_object_states_change = []
        self.object_sites_dict = {}
        self.objects = []
        self.fixtures = []

        self.custom_asset_dir = os.path.abspath(os.path.join(DIR_PATH, "../assets"))

        self.bddl_file_name = bddl_file_name
        self.parsed_problem = BDDLUtils.robosuite_parse_problem(self.bddl_file_name)
        # Normalize once so per-step reward() has no extra division cost.
        self._normalize_subtask_weights()

        self.obj_of_interest = self.parsed_problem["obj_of_interest"]

        self._assert_problem_name()

        self._arena_type = arena_type
        self._arena_xml = os.path.join(self.custom_asset_dir, scene_xml)
        self._arena_properties = scene_properties

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            **kwargs,
        )

    def seed(self, seed):
        np.random.seed(seed)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _normalize_subtask_weights(self):
        """Normalize subtask reward weights to sum to subtask_reward_scale.

        Called once during __init__ after parsing, so per-step reward()
        just sums pre-scaled floats with no extra division.

        BDDL weights are treated as *relative* — any positive values work.
        Example: weights [1, 1] -> [0.25, 0.25] when subtask_reward_scale=0.5.
        Example: weights [2, 1] -> [0.333, 0.167] when subtask_reward_scale=0.5.
        """
        fine = self.parsed_problem.get("subtask_rewards", [])
        if not fine:
            return
        total = sum(s["reward"] for s in fine)
        if total <= 0:
            return
        scale = self.subtask_reward_scale / total
        for s in fine:
            s["reward"] = s["reward"] * scale

    def reward(self, action=None):
        """Return the task reward.

        Sparse mode (default, subtask_reward=False):
            Range [0, 1].  Returns 1.0 when all goal predicates are satisfied,
            else 0.0.

        Subtask mode (subtask_reward=True):
            Range [0, 1 + subtask_reward_scale].  Two components:

            * Subtask progress  — up to subtask_reward_scale (default 0.5).
              If the BDDL file has a (:subtask_rewards ...) section, the
              declared weights are used (normalized to subtask_reward_scale
              at init time — see _normalize_subtask_weights).  Without that
              section each goal predicate contributes subtask_reward_scale/N
              equally.

            * Task completion   — 1.0 added when all goal predicates are
              simultaneously satisfied (same condition as done=True).

            The subtask_reward_scale / 1.0 split keeps the terminal signal
            dominant. reward_scale is applied to the combined value.

        The step() interface (obs, reward, done, info) is unchanged regardless
        of mode; done always reflects full task completion.

        When subtask_reward is True, step() adds shaping diagnostics to info
        (see step()): per-subtask booleans in ``subtask_rewards``, newly
        credited weights in ``subtask_reward_increment`` (dict), and their sum
        in ``subtask_reward_delta`` (float).  We intentionally do *not* use
        the key ``subtask_reward`` in info, because training stacks often merge
        env constructor kwargs (e.g. ``subtask_reward=True``) into the info
        dict under the same name, which would shadow a dict payload and break
        aggregators that branch on ``if "subtask_reward" in info`` before
        falling back to ``subtask_info``.
        """
        if self.subtask_reward:
            satisfied = self._evaluate_subtask_rewards()
            # Cache so step() can populate info without a second evaluation.
            self._subtask_satisfied_cache = satisfied
            fine = self.parsed_problem.get("subtask_rewards", [])
            if fine:
                # Weights are already normalized to subtask_reward_scale.
                r = sum(s["reward"] for s in fine if satisfied.get(s["name"], False))
            else:
                # Coarse fallback: equal share of subtask_reward_scale per predicate.
                n = max(len(satisfied), 1)
                r = self.subtask_reward_scale * sum(1.0 for v in satisfied.values() if v) / n
            if self._check_success():
                r += 1.0
        else:
            r = 1.0 if self._check_success() else 0.0

        if self.reward_scale is not None:
            r *= self.reward_scale
        return r

    # ------------------------------------------------------------------
    # Goal / subtask evaluation
    # ------------------------------------------------------------------

    def _check_goal_state_satisfied(self, goal_state):
        """Return True iff every predicate in goal_state is satisfied.

        goal_state is the list produced by bddl_utils.robosuite_parse_problem,
        e.g. [['turnon', 'flat_stove_1'], ['on', 'moka_pot_1', 'flat_stove_1_cook_region']].
        """
        for pred in goal_state:
            pred_fn_name = pred[0]
            args = [self.object_states_dict[arg] for arg in pred[1:]]
            if not eval_predicate_fn(pred_fn_name, *args):
                return False
        return True

    def _evaluate_subtask_rewards(self, dry_run: bool = False):
        """Evaluate subtask predicates and return per-subtask satisfaction status.

        Returns a dict mapping subtask name (or predicate key) to bool.

        One-shot semantics: once a subtask is satisfied it stays True for the
        rest of the episode (_subtask_ever_satisfied), even if the predicate
        later becomes False again.  This prevents the agent from cycling a
        subtask condition to farm reward.  The set is cleared on reset().

        Fine-grained mode (BDDL has a :subtask_rewards section):
            Evaluates in declaration order.  A subtask whose :after
            prerequisites have *never* been satisfied returns False without
            evaluating its predicate (hard prerequisite gate).  Prerequisites
            use the ever-satisfied set so unlocking persists even if the
            prerequisite predicate later becomes False.

        Coarse fallback (no :subtask_rewards section):
            Each predicate in (:goal ...) is an independent subtask with equal
            reward weight.  No ordering is enforced.

        Args:
            dry_run: When True, evaluate predicates but do NOT update
                _subtask_ever_satisfied.  Use this for inspection and testing
                to avoid inadvertently crediting a subtask.
        """
        fine = self.parsed_problem.get("subtask_rewards", [])
        confirmation_counts = deepcopy(self._subtask_confirmation_counts)
        if fine:
            satisfied = {}
            # local_ever tracks intra-call propagation: if subtask A is
            # satisfied in this call, its dependents can be unlocked in the
            # same call.  In dry_run mode we intentionally do NOT write back
            # to self._subtask_ever_satisfied (avoids side-effects), but we
            # still need local propagation so :after chains resolve correctly.
            local_ever = set(self._subtask_ever_satisfied)
            for s in fine:
                name = s["name"]
                if name in local_ever:
                    # Already credited this episode — no re-evaluation needed.
                    satisfied[name] = True
                    continue
                # Prerequisites must have been satisfied at some point.
                prereqs_met = all(p in local_ever for p in s["after"])
                if prereqs_met:
                    args = [self.object_states_dict[a] for a in s["predicate_args"]]
                    if self._subtask_candidate_valid(s["predicate_name"], s["predicate_fn"], args):
                        if s["predicate_name"] in {"on", "in"}:
                            confirmation_counts[name] = confirmation_counts.get(name, 0) + 1
                            confirm_steps = s.get(
                                "confirm_steps") or self.subtask_confirmation_steps
                            is_confirmed = confirmation_counts[name] >= confirm_steps
                        else:
                            confirmation_counts[name] = 0
                            is_confirmed = True

                        if is_confirmed:
                            if not dry_run:
                                self._subtask_ever_satisfied.add(name)
                            local_ever.add(name)
                            satisfied[name] = True
                        else:
                            satisfied[name] = False
                    else:
                        confirmation_counts[name] = 0
                        satisfied[name] = False
                else:
                    confirmation_counts[name] = 0
                    satisfied[name] = False
            if not dry_run:
                self._subtask_confirmation_counts = confirmation_counts
            return satisfied
        else:
            # Coarse: every goal predicate becomes a subtask with equal weight.
            goal_state = self.parsed_problem["goal_state"]
            results = {}
            for pred in goal_state:
                key = "_".join(pred)
                if key in self._subtask_ever_satisfied:
                    results[key] = True
                    continue
                args = [self.object_states_dict[a] for a in pred[1:]]
                pred_name = pred[0].lower()
                if self._subtask_candidate_valid(
                        pred_name, lambda *call_args: eval_predicate_fn(pred_name, *call_args),
                        args):
                    if pred_name in {"on", "in"}:
                        confirmation_counts[key] = confirmation_counts.get(key, 0) + 1
                        is_confirmed = confirmation_counts[key] >= self.subtask_confirmation_steps
                    else:
                        confirmation_counts[key] = 0
                        is_confirmed = True

                    if is_confirmed:
                        if not dry_run:
                            self._subtask_ever_satisfied.add(key)
                        results[key] = True
                    else:
                        results[key] = False
                else:
                    confirmation_counts[key] = 0
                    results[key] = False
            if not dry_run:
                self._subtask_confirmation_counts = confirmation_counts
            return results

    def _resolve_contact_object(self, object_state):
        """Resolve an ObjectState / SiteObjectState to a real Mujoco object for contact checks."""
        if getattr(object_state, "object_state_type", None) == "site":
            return self.get_object(object_state.parent_name)
        return self.get_object(object_state.object_name)

    def _robot_not_in_contact(self, object_state):
        """Return True when the robot gripper is no longer touching the relevant object."""
        robot = self.robots[0]
        target_object = self._resolve_contact_object(object_state)
        if target_object is None:
            return True
        return not self.check_contact(robot.gripper, target_object)

    def _subtask_candidate_valid(self, predicate_name, predicate_fn, args):
        """Apply release-aware gating on top of the existing instantaneous predicate."""
        if not bool(predicate_fn(*args)):
            return False
        predicate_name = predicate_name.lower()
        if predicate_name in {"on", "in"}:
            return bool(args) and self._robot_not_in_contact(args[0])
        if predicate_name in {"open", "close", "turnon", "turnoff"}:
            return bool(args) and self._robot_not_in_contact(args[0])
        return True

    # ------------------------------------------------------------------
    # Abstract arena-loading hooks (implemented by problem subclasses)
    # ------------------------------------------------------------------

    def _assert_problem_name(self):
        assert (self.parsed_problem["problem_name"] == self.__class__.__name__.lower()
               ), "Problem name mismatched"

    def _load_fixtures_in_arena(self, mujoco_arena):
        raise NotImplementedError

    def _load_objects_in_arena(self, mujoco_arena):
        raise NotImplementedError

    def _load_sites_in_arena(self, mujoco_arena):
        raise NotImplementedError

    def _generate_object_state_wrapper(self,
                                       skip_object_names=[
                                           "main_table", "floor", "countertop", "coffee_table"
                                       ]):
        object_states_dict = {}
        tracking_object_states_changes = []
        for object_name in self.objects_dict.keys():
            if object_name in skip_object_names:
                continue
            object_states_dict[object_name] = ObjectState(self, object_name)
            if (self.objects_dict[object_name].category_name in VISUAL_CHANGE_OBJECTS_DICT):
                tracking_object_states_changes.append(object_states_dict[object_name])

        for object_name in self.fixtures_dict.keys():
            if object_name in skip_object_names:
                continue
            object_states_dict[object_name] = ObjectState(self, object_name, is_fixture=True)
            if (self.fixtures_dict[object_name].category_name in VISUAL_CHANGE_OBJECTS_DICT):
                tracking_object_states_changes.append(object_states_dict[object_name])

        for object_name in self.object_sites_dict.keys():
            if object_name in skip_object_names:
                continue
            object_states_dict[object_name] = SiteObjectState(
                self,
                object_name,
                parent_name=self.object_sites_dict[object_name].parent_name,
            )
        self.object_states_dict = object_states_dict
        self.tracking_object_states_change = tracking_object_states_changes

    def _load_distracting_objects(self, mujoco_arena):
        raise NotImplementedError

    def _load_custom_material(self):
        pass

    def _setup_camera(self, mujoco_arena):
        mujoco_arena.set_camera(
            camera_name="canonical_agentview",
            pos=[0.5386131746834771, 0.0, 1.4903500240372423],
            quat=[
                0.6380177736282349,
                0.3048497438430786,
                0.30484986305236816,
                0.6380177736282349,
            ],
        )
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.5886131746834771, 0.0, 1.4903500240372423],
            quat=[
                0.6380177736282349,
                0.3048497438430786,
                0.30484986305236816,
                0.6380177736282349,
            ],
        )

    def _load_model(self):
        super()._load_model()

        if self._arena_type == "table":
            xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
            self.robots[0].robot_model.set_base_xpos(xpos)
            mujoco_arena = TableArena(
                table_full_size=self.table_full_size,
                table_offset=self.workspace_offset,
                table_friction=(0.6, 0.005, 0.0001),
                xml=self._arena_xml,
                **self._arena_properties,
            )
        elif self._arena_type == "kitchen":
            xpos = self.robots[0].robot_model.base_xpos_offset["kitchen_table"](
                self.kitchen_table_full_size[0])
            self.robots[0].robot_model.set_base_xpos(xpos)
            mujoco_arena = KitchenTableArena(
                table_full_size=self.kitchen_table_full_size,
                table_offset=self.workspace_offset,
                xml=self._arena_xml,
                **self._arena_properties,
            )
        elif self._arena_type == "floor":
            xpos = self.robots[0].robot_model.base_xpos_offset["empty"]
            self.robots[0].robot_model.set_base_xpos(xpos)
            mujoco_arena = EmptyArena(
                xml=self._arena_xml,
                **self._arena_properties,
            )
        elif self._arena_type == "coffee_table":
            xpos = self.robots[0].robot_model.base_xpos_offset["coffee_table"](
                self.coffee_table_full_size[0])
            self.robots[0].robot_model.set_base_xpos(xpos)
            mujoco_arena = CoffeeTableArena(
                xml=self._arena_xml,
                **self._arena_properties,
            )
        elif self._arena_type == "living_room":
            xpos = self.robots[0].robot_model.base_xpos_offset["living_room_table"](
                self.living_room_table_full_size[0])
            self.robots[0].robot_model.set_base_xpos(xpos)
            mujoco_arena = LivingRoomTableArena(
                xml=self._arena_xml,
                **self._arena_properties,
            )
        elif self._arena_type == "study":
            xpos = self.robots[0].robot_model.base_xpos_offset["study_table"](
                self.study_table_full_size[0])
            self.robots[0].robot_model.set_base_xpos(xpos)
            mujoco_arena = StudyTableArena(
                xml=self._arena_xml,
                **self._arena_properties,
            )

        mujoco_arena.set_origin([0, 0, 0])

        self._setup_camera(mujoco_arena)
        self._load_custom_material()
        self._load_fixtures_in_arena(mujoco_arena)
        self._load_objects_in_arena(mujoco_arena)
        self._load_sites_in_arena(mujoco_arena)
        self._generate_object_state_wrapper()
        self._setup_placement_initializer(mujoco_arena)

        self.objects = list(self.objects_dict.values())
        self.fixtures = list(self.fixtures_dict.values())

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.objects + self.fixtures,
        )

        for fixture in self.fixtures:
            self.model.merge_assets(fixture)

    def _setup_placement_initializer(self, mujoco_arena):
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        self.conditional_placement_initializer = SiteSequentialCompositeSampler(
            name="ConditionalSiteSampler")
        self.conditional_placement_on_objects_initializer = SequentialCompositeSampler(
            name="ConditionalObjectSampler")
        self._add_placement_initializer()

    def _setup_references(self):
        super()._setup_references()

        self.obj_body_id = dict()

        for (object_name, object_body) in self.objects_dict.items():
            self.obj_body_id[object_name] = self.sim.model.body_name2id(object_body.root_body)

        for (fixture_name, fixture_body) in self.fixtures_dict.items():
            self.obj_body_id[fixture_name] = self.sim.model.body_name2id(fixture_body.root_body)

    def _setup_observables(self):
        observables = super()._setup_observables()

        observables["robot0_joint_pos"]._active = True

        if self.use_object_obs:
            pf = self.robots[0].robot_model.naming_prefix
            sensors = []
            names = [s.__name__ for s in sensors]

            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        pf = self.robots[0].robot_model.naming_prefix

        @sensor(modality="object")
        def world_pose_in_gripper(obs_cache):
            return (T.pose_inv(T.pose2mat((obs_cache[f"{pf}eef_pos"], obs_cache[f"{pf}eef_quat"])))
                    if f"{pf}eef_pos" in obs_cache and f"{pf}eef_quat" in obs_cache else np.eye(4))

        sensors = [world_pose_in_gripper]
        names = ["world_pose_in_gripper"]

        for (i, obj) in enumerate(self.objects):
            obj_sensors, obj_sensor_names = self._create_obj_sensors(obj_name=obj.name,
                                                                     modality="object")
            sensors += obj_sensors
            names += obj_sensor_names

        for name, s in zip(names, sensors):
            if name == "world_pose_in_gripper":
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                    enabled=True,
                    active=False,
                )
            else:
                observables[name] = Observable(name=name, sensor=s, sampling_rate=self.control_freq)

        return observables

    def _create_obj_sensors(self, obj_name, modality="object"):
        pf = self.robots[0].robot_model.naming_prefix

        @sensor(modality=modality)
        def obj_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.obj_body_id[obj_name]])

        @sensor(modality=modality)
        def obj_quat(obs_cache):
            return T.convert_quat(self.sim.data.body_xquat[self.obj_body_id[obj_name]], to="xyzw")

        @sensor(modality=modality)
        def obj_to_eef_pos(obs_cache):
            if any([
                    name not in obs_cache for name in [
                        f"{obj_name}_pos",
                        f"{obj_name}_quat",
                        "world_pose_in_gripper",
                    ]
            ]):
                return np.zeros(3)
            obj_pose = T.pose2mat((obs_cache[f"{obj_name}_pos"], obs_cache[f"{obj_name}_quat"]))
            rel_pose = T.pose_in_A_to_pose_in_B(obj_pose, obs_cache["world_pose_in_gripper"])
            rel_pos, rel_quat = T.mat2pose(rel_pose)
            obs_cache[f"{obj_name}_to_{pf}eef_quat"] = rel_quat
            return rel_pos

        @sensor(modality=modality)
        def obj_to_eef_quat(obs_cache):
            return (obs_cache[f"{obj_name}_to_{pf}eef_quat"]
                    if f"{obj_name}_to_{pf}eef_quat" in obs_cache else np.zeros(4))

        sensors = [obj_pos, obj_quat, obj_to_eef_pos, obj_to_eef_quat]
        names = [
            f"{obj_name}_pos",
            f"{obj_name}_quat",
            f"{obj_name}_to_{pf}eef_pos",
            f"{obj_name}_to_{pf}eef_quat",
        ]
        return sensors, names

    def _add_placement_initializer(self):
        mapping_inv = {}
        for k, values in self.parsed_problem["fixtures"].items():
            for v in values:
                mapping_inv[v] = k
        for k, values in self.parsed_problem["objects"].items():
            for v in values:
                mapping_inv[v] = k

        regions = self.parsed_problem["regions"]
        initial_state = self.parsed_problem["initial_state"]
        problem_name = self.parsed_problem["problem_name"]

        conditioned_initial_place_state_on_sites = []
        conditioned_initial_place_state_on_objects = []
        conditioned_initial_place_state_in_objects = []

        for state in initial_state:
            if state[0] == "on" and state[2] in self.objects_dict:
                conditioned_initial_place_state_on_objects.append(state)
                continue
            if state[0] == "in" and state[2] in regions:
                conditioned_initial_place_state_in_objects.append(state)
                continue
            if state[0] == "on" and state[2] in regions:
                object_name = state[1]
                region_name = state[2]
                target_name = regions[region_name]["target"]
                x_ranges, y_ranges = rectangle2xyrange(regions[region_name]["ranges"])
                yaw_rotation = regions[region_name]["yaw_rotation"]
                if (target_name in self.objects_dict or target_name in self.fixtures_dict):
                    conditioned_initial_place_state_on_sites.append(state)
                    continue
                if self.is_fixture(object_name):
                    fixture_sampler = MultiRegionRandomSampler(
                        f"{object_name}_sampler",
                        mujoco_objects=self.fixtures_dict[object_name],
                        x_ranges=x_ranges,
                        y_ranges=y_ranges,
                        rotation=yaw_rotation,
                        rotation_axis="z",
                        z_offset=self.z_offset,
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=False,
                        reference_pos=self.workspace_offset,
                    )
                    self.placement_initializer.append_sampler(fixture_sampler)
                else:
                    region_sampler = get_region_samplers(problem_name, mapping_inv[target_name])(
                        object_name,
                        self.objects_dict[object_name],
                        x_ranges=x_ranges,
                        y_ranges=y_ranges,
                        rotation=self.objects_dict[object_name].rotation,
                        rotation_axis=self.objects_dict[object_name].rotation_axis,
                        reference_pos=self.workspace_offset,
                    )
                    self.placement_initializer.append_sampler(region_sampler)
            if state[0] in ["open", "close"]:
                if state[1] in self.object_states_dict and hasattr(
                        self.object_states_dict[state[1]], "set_joint"):
                    obj = self.get_object(state[1])
                    if state[0] == "open":
                        joint_ranges = obj.object_properties["articulation"]["default_open_ranges"]
                    else:
                        joint_ranges = obj.object_properties["articulation"]["default_close_ranges"]
                    property_initializer = OpenCloseSampler(
                        name=obj.name,
                        state_type=state[0],
                        joint_ranges=joint_ranges,
                    )
                    self.object_property_initializers.append(property_initializer)
            elif state[0] in ["turnon", "turnoff"]:
                if state[1] in self.object_states_dict and hasattr(
                        self.object_states_dict[state[1]], "set_joint"):
                    obj = self.get_object(state[1])
                    if state[0] == "turnon":
                        joint_ranges = obj.object_properties["articulation"][
                            "default_turnon_ranges"]
                    else:
                        joint_ranges = obj.object_properties["articulation"][
                            "default_turnoff_ranges"]
                    property_initializer = TurnOnOffSampler(
                        name=obj.name,
                        state_type=state[0],
                        joint_ranges=joint_ranges,
                    )
                    self.object_property_initializers.append(property_initializer)

        for state in conditioned_initial_place_state_on_sites:
            object_name = state[1]
            region_name = state[2]
            target_name = regions[region_name]["target"]
            site_xy_size = self.object_sites_dict[region_name].size[:2]
            sampler = SiteRegionRandomSampler(
                f"{object_name}_sampler",
                mujoco_objects=self.objects_dict[object_name],
                x_ranges=[[-site_xy_size[0] / 2, site_xy_size[0] / 2]],
                y_ranges=[[-site_xy_size[1] / 2, site_xy_size[1] / 2]],
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                rotation=self.objects_dict[object_name].rotation,
                rotation_axis=self.objects_dict[object_name].rotation_axis,
            )
            self.conditional_placement_initializer.append_sampler(sampler, {
                "reference": target_name,
                "site_name": region_name
            })

        for state in conditioned_initial_place_state_on_objects:
            object_name = state[1]
            other_object_name = state[2]
            sampler = ObjectBasedSampler(
                f"{object_name}_sampler",
                mujoco_objects=self.objects_dict[object_name],
                x_ranges=[[0.0, 0.0]],
                y_ranges=[[0.0, 0.0]],
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=False,
                rotation=self.objects_dict[object_name].rotation,
                rotation_axis=self.objects_dict[object_name].rotation_axis,
            )
            self.conditional_placement_on_objects_initializer.append_sampler(
                sampler, {"reference": other_object_name})

        for state in conditioned_initial_place_state_in_objects:
            object_name = state[1]
            region_name = state[2]
            target_name = regions[region_name]["target"]
            site_xy_size = self.object_sites_dict[region_name].size[:2]
            sampler = InSiteRegionRandomSampler(
                f"{object_name}_sampler",
                mujoco_objects=self.objects_dict[object_name],
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                rotation=self.objects_dict[object_name].rotation,
                rotation_axis=self.objects_dict[object_name].rotation_axis,
            )
            self.conditional_placement_initializer.append_sampler(sampler, {
                "reference": target_name,
                "site_name": region_name
            })

    def _reset_internal(self):
        # Clear per-episode reward state so each new episode starts fresh.
        self._subtask_ever_satisfied = set()
        self._subtask_satisfied_cache = {}
        self._subtask_confirmation_counts = {}
        super()._reset_internal()

        if not self.deterministic_reset:
            for object_property_initializer in self.object_property_initializers:
                if isinstance(object_property_initializer, OpenCloseSampler):
                    joint_pos = object_property_initializer.sample()
                    self.object_states_dict[object_property_initializer.name].set_joint(joint_pos)
                elif isinstance(object_property_initializer, TurnOnOffSampler):
                    joint_pos = object_property_initializer.sample()
                    self.object_states_dict[object_property_initializer.name].set_joint(joint_pos)
                else:
                    print("Warning!!! This sampler doesn't seem to be used")

            mujoco.mj_step1(self.sim.model._model, self.sim.data._data)

            object_placements = self.placement_initializer.sample()
            object_placements = self.conditional_placement_initializer.sample(
                self.sim, object_placements)
            object_placements = (
                self.conditional_placement_on_objects_initializer.sample(object_placements))
            for obj_pos, obj_quat, obj in object_placements.values():
                if obj.name not in list(self.fixtures_dict.keys()):
                    self.sim.data.set_joint_qpos(
                        obj.joints[-1],
                        np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                    )
                else:
                    body_id = self.sim.model.body_name2id(obj.root_body)
                    self.sim.model.body_pos[body_id] = obj_pos
                    self.sim.model.body_quat[body_id] = obj_quat

    def _check_success(self):
        return False

    def visualize(self, vis_settings):
        super().visualize(vis_settings=vis_settings)

    def step(self, action):
        if self.action_dim == 4 and len(action) > 4:
            action = np.array(action)
            action = np.concatenate((action[:3], action[-1:]), axis=-1)

        # Snapshot before step so we can compute newly satisfied (incremental).
        prev_ever_satisfied = set(self._subtask_ever_satisfied) if self.subtask_reward else set()

        obs, reward, done, info = super().step(action)
        done = self._check_success()

        if self.subtask_reward:
            # reward() already evaluated and cached this during super().step().
            info["subtask_rewards"] = self._subtask_satisfied_cache
            # Incremental scalar rewards: only newly satisfied this step (one-shot).
            newly = self._subtask_ever_satisfied - prev_ever_satisfied
            incremental = {}
            fine = self.parsed_problem.get("subtask_rewards", [])
            if fine:
                name_to_reward = {s["name"]: s["reward"] for s in fine}
                for name in newly:
                    if name in name_to_reward:
                        incremental[name] = name_to_reward[name]
            else:
                # Coarse: equal share of subtask_reward_scale per subtask.
                goal_state = self.parsed_problem.get("goal_state", [])
                n_total = max(len(goal_state), 1)
                weight = self.subtask_reward_scale / n_total
                for name in newly:
                    incremental[name] = weight
            # Use names that cannot collide with merged env kwargs (e.g.
            # subtask_reward=True) — see reward() docstring.
            info["subtask_reward_increment"] = incremental
            info["subtask_reward_delta"] = float(sum(incremental.values()))

        if self.track_subtask_info:
            # Eval-mode diagnostic: evaluate subtask completion without shaping
            # the reward.  dry_run=True leaves _subtask_ever_satisfied untouched
            # so the one-shot semantics remain correct when used alongside
            # subtask_reward=True, or are simply inert when used alone.
            info["subtask_info"] = self._evaluate_subtask_rewards(dry_run=True)

        return obs, reward, done, info

    def _pre_action(self, action, policy_step=False):
        super()._pre_action(action, policy_step=policy_step)

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)
        self._post_process()
        return reward, done, info

    def _post_process(self):
        for object_state in self.tracking_object_states_change:
            object_state.update_state()

    def get_robot_state_vector(self, obs):
        return np.concatenate(
            [obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]])

    def is_fixture(self, object_name):
        return object_name in list(self.fixtures_dict.keys())

    @property
    def language_instruction(self):
        return self.parsed_problem["language_instruction"]

    def get_object(self, object_name):
        for query_dict in [
                self.fixtures_dict,
                self.objects_dict,
                self.object_sites_dict,
        ]:
            if object_name in query_dict:
                return query_dict[object_name]
