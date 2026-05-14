import numpy as np
import os
import robosuite.utils.transform_utils as T

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
from libero.libero.envs.predicates import (
    PARAMETRIC_PREDICATE_CLS,
    DefaultGraspPredicate,
    eval_predicate_fn,
    instantiate_predicate,
)
from libero.libero.bddlsim_interface import (
    RAW_PREDICATE_KEY_L4,
    raw_predicate_key_l1,
    raw_predicate_key_l2,
    raw_predicate_key_l3,
)

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

    def reward(self, action=None):
        """Return sparse task reward: 1.0 on full goal satisfaction, else 0.0.

        All reward shaping lives in the HierarchicalRewardWrapper layer.
        reward_scale is applied to the raw 0/1 value.
        """
        r = 1.0 if self._check_success() else 0.0
        if self.reward_scale is not None:
            r *= self.reward_scale
        return r

    # ------------------------------------------------------------------
    # Goal / subtask evaluation
    # ------------------------------------------------------------------

    def _eval_goal_predicate_state(self, state):
        """Evaluate one (:goal ...) conjunct: unary/binary or parametric (Near, NearEEF, …)."""
        if not state:
            return True
        head = str(state[0]).lower()
        tail = list(state[1:])
        obj_args = []
        num_params = []
        for tok in tail:
            try:
                num_params.append(float(tok))
            except (ValueError, TypeError):
                obj_args.append(tok)
        if head in PARAMETRIC_PREDICATE_CLS:
            pred_fn = instantiate_predicate(head, num_params)
            object_states = [self.object_states_dict[a] for a in obj_args]
            if len(object_states) == 1:
                return bool(pred_fn(object_states[0]))
            if len(object_states) == 2:
                return bool(pred_fn(object_states[0], object_states[1]))
            raise ValueError(
                f"Predicate {head!r} expects 1 or 2 object arguments, got {len(object_states)} "
                f"in {state!r}"
            )
        args = [self.object_states_dict[a] for a in tail]
        return bool(eval_predicate_fn(head, *args))

    def _first_obj_state_from_goal_tokens(self, pred):
        """First object-state token in a goal atom (skips numeric parameters)."""
        for tok in pred[1:]:
            try:
                float(tok)
            except (ValueError, TypeError):
                return self.object_states_dict[tok]
        return None

    def _check_goal_state_satisfied(self, goal_state):
        """Return True iff every predicate in goal_state is satisfied.

        goal_state is the list produced by bddl_utils.robosuite_parse_problem,
        e.g. [['turnon', 'flat_stove_1'], ['on', 'moka_pot_1', 'flat_stove_1_cook_region']].
        """
        for pred in goal_state:
            if not self._eval_goal_predicate_state(pred):
                return False
        return True

    # ------------------------------------------------------------------
    # Raw-predicate emission (Phase 0 contract)
    # ------------------------------------------------------------------

    # Predicates for which L2 (grasp) is meaningful: pick-and-place only.
    # For articulation predicates (turnon, turnoff, open, close) the robot
    # actuates a joint rather than grasping and carrying an object, so L2 is
    # always False (see DESIGN.md §4 "L2 grasp scope").
    _PICK_PLACE_PREDICATES: frozenset = frozenset({"on", "in"})

    def _gripper_touching_primary(self, obj_state, robot_idx: int = 0) -> bool:
        """True if the gripper has any contact with obj_state.

        Looser than DefaultGraspPredicate: does NOT require the gripper to be
        the only contact. Correct for the must-release gate when the object
        rests inside a container (basket walls also contact, making
        DefaultGraspPredicate False even while the gripper holds the object).
        Mirrors the Grasp predicate in base_predicates.py.
        """
        robot = self.robots[robot_idx]
        obj = self.get_object(obj_state.object_name)
        return bool(self.check_contact(robot.gripper, obj))

    def _l2_grasp_predicate_for_primary_object_name(self, object_name: str):
        """Return a callable(ObjectState) -> bool for the L2 grasp check.

        Uses DefaultGraspPredicate: gripper contacts the object and no other
        external geom is touching it (table, shelf, arm links outside the
        important_geoms set).
        """
        pred = DefaultGraspPredicate(robot_idx=0)
        return pred

    def _l1_near_predicate(self):
        """Return a callable(ObjectState) -> bool for the L1 localization check.

        Uses LocalizedNearEEF (registered as 'localizedneareef') — the same
        predicate used in BDDL NearEEF subtask evaluation.
        """
        return instantiate_predicate("localizedneareef", [])

    def _evaluate_raw_predicates(self) -> dict:
        """Evaluate and return the instantaneous raw predicate dict for this step.

        Returns a dict with keys (in BDDL declaration order):
            L1::<subtask_name>  — primary object near EEF (LocalizedNearEEF, transient);
                                  overridden to True when L2 is True (grasping implies
                                  nearness — LocalizedNearEEF's velocity gate can produce
                                  false-negatives during fast gripper motion).
            L2::<subtask_name>  — gripper grasping primary object (transient);
                                  always False for non-pick-place predicates
                                  (turnon/turnoff/open/close) — see DESIGN.md §4
            L3::<subtask_name>  — BDDL subtask predicate, instantaneous (no latch);
                                  False while gripper has any contact with the primary
                                  object (must-release gate). Uses a simple contact
                                  check, not DefaultGraspPredicate, so it fires
                                  correctly even when the object is inside a container.
            L4                  — full (:goal ...) satisfaction (mirrors l4_satisfied)

        Fine-grained path: uses the (:subtask_rewards ...) section when present,
        which provides named subtasks with explicit predicate specs.

        Coarse fallback: when no (:subtask_rewards ...) section exists, each
        (:goal ...) conjunct becomes an anonymous subtask keyed by its token join.
        """
        raw: dict = {}
        l1_fn = self._l1_near_predicate()

        fine = self.parsed_problem.get("subtask_rewards", [])
        if fine:
            # Fine-grained path: (:subtask_rewards ...) section present.
            for s in fine:
                name = s["name"]
                pred_name = (s.get("predicate_name") or "").lower()
                args = [self.object_states_dict[a] for a in s["predicate_args"]]

                # Primary object is the first arg (first non-numeric token).
                primary = args[0] if args else None

                # L1: primary object near EEF (always evaluated when primary exists).
                raw[raw_predicate_key_l1(name)] = bool(l1_fn(primary)) if primary is not None else False

                # L2: gripper grasping primary object — only meaningful for
                # pick-and-place predicates (on/in).  Articulation predicates
                # (turnon, turnoff, open, close) do not involve grasping and
                # carrying, so L2 is set to False unconditionally for them.
                if pred_name in self._PICK_PLACE_PREDICATES and primary is not None:
                    l2_fn = self._l2_grasp_predicate_for_primary_object_name(primary.object_name)
                    raw[raw_predicate_key_l2(name)] = bool(l2_fn(primary))
                else:
                    raw[raw_predicate_key_l2(name)] = False

                # L2 implies L1: grasping means the object is near the EEF.
                # Overrides LocalizedNearEEF's velocity gate, which can fire False
                # during fast gripper motion while the object is actively held.
                if raw[raw_predicate_key_l2(name)]:
                    raw[raw_predicate_key_l1(name)] = True

                # L3: instantaneous predicate — no one-shot latch, no :after gating.
                # Must-release gate: L3 is False while the gripper has any contact
                # with the primary object. Uses a simple any-contact check rather
                # than DefaultGraspPredicate (L2), because when an object rests
                # inside a container (e.g. basket), container walls also touch it,
                # making L2 False even while the gripper still holds the object.
                l3_val = bool(s["predicate_fn"](*args)) if args else False
                if l3_val and primary is not None and self._gripper_touching_primary(primary):
                    l3_val = False
                raw[raw_predicate_key_l3(name)] = l3_val
        else:
            # Coarse fallback: treat each (:goal ...) conjunct as an anonymous subtask.
            goal_state = self.parsed_problem.get("goal_state", [])
            for pred in goal_state:
                name = "_".join(str(x) for x in pred)
                pred_name = str(pred[0]).lower()

                # Primary object: first non-numeric token after the predicate name.
                primary = self._first_obj_state_from_goal_tokens(pred)

                # L1: primary object near EEF.
                raw[raw_predicate_key_l1(name)] = bool(l1_fn(primary)) if primary is not None else False

                # L2: only for on/in predicates; False for articulation predicates.
                # (turnon, turnoff, open, close do not require grasping the object.)
                if pred_name in self._PICK_PLACE_PREDICATES and primary is not None:
                    l2_fn = self._l2_grasp_predicate_for_primary_object_name(primary.object_name)
                    raw[raw_predicate_key_l2(name)] = bool(l2_fn(primary))
                else:
                    raw[raw_predicate_key_l2(name)] = False

                # L2 implies L1: grasping means the object is near the EEF.
                if raw[raw_predicate_key_l2(name)]:
                    raw[raw_predicate_key_l1(name)] = True

                # L3: instantaneous predicate evaluation.
                # Must-release gate: any gripper–object contact inhibits L3
                # (see fine-grained path comment for full rationale).
                l3_val = bool(self._eval_goal_predicate_state(pred))
                if l3_val and primary is not None and self._gripper_touching_primary(primary):
                    l3_val = False
                raw[raw_predicate_key_l3(name)] = l3_val

        # L4: full terminal goal satisfaction — mirrors l4_satisfied in step() info.
        raw[RAW_PREDICATE_KEY_L4] = bool(self._check_success())
        return raw

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

        obs, reward, done, info = super().step(action)
        done = self._check_success()

        # Emit the Phase 0 contract keys: instantaneous raw predicate truth and
        # terminal goal satisfaction.  All shaping lives in the wrapper layer.
        # Remove any legacy keys that may have been added by super().step().
        for _legacy_key in (
            "subtask_rewards",
            "subtask_reward_increment",
            "subtask_reward_delta",
            "subtask_info",
        ):
            info.pop(_legacy_key, None)

        l4 = bool(self._check_success())
        info["raw_predicates"] = self._evaluate_raw_predicates()
        info["l4_satisfied"] = l4

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
