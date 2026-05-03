from typing import Optional, Set

import numpy as np


class Expression:

    def __init__(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError


class UnaryAtomic(Expression):

    def __init__(self):
        pass

    def __call__(self, arg1):
        raise NotImplementedError


class BinaryAtomic(Expression):

    def __init__(self):
        pass

    def __call__(self, arg1, arg2):
        raise NotImplementedError


class MultiarayAtomic(Expression):

    def __init__(self):
        pass

    def __call__(self, *args):
        raise NotImplementedError


class TruePredicateFn(MultiarayAtomic):

    def __init__(self):
        super().__init__()

    def __call__(self, *args):
        return True


class FalsePredicateFn(MultiarayAtomic):

    def __init__(self):
        super().__init__()

    def __call__(self, *args):
        return False


class InContactPredicateFn(BinaryAtomic):

    def __call__(self, arg1, arg2):
        return arg1.check_contact(arg2)


class In(BinaryAtomic):

    def __call__(self, arg1, arg2):
        return arg2.check_contact(arg1) and arg2.check_contain(arg1)


class ExactIn(BinaryAtomic):
    """Stricter "in": inside region AND bottom is at expected vertical height.

    For site regions, compute vertical delta along world up-axis and compare to
    the site's vertical half-size projected onto world up. For non-site, fallback
    to check_ontop.
    """

    def __call__(self, arg1, arg2):
        # first: must be inside
        try:
            contain_ok = arg2.check_contain(arg1)
        except Exception:
            contain_ok = False

        bottom_ok = False
        reg_type = getattr(arg2, 'object_state_type', '?')
        if reg_type == 'site':
            try:
                env = arg2.env
                site_name = arg2.object_name
                site_obj = env.object_sites_dict[site_name]
                site_pos = env.sim.data.get_site_xpos(site_name)
                site_mat = env.sim.data.get_site_xmat(site_name)
                obj_pos = env.sim.data.body_xpos[env.obj_body_id[arg1.object_name]]

                table_top_z = float(env.workspace_offset[2])

                bottom_site_name = f"{arg1.object_name}_bottom_site"
                obj_bottom_world = None
                try:
                    obj_bottom_world = env.sim.data.get_site_xpos(bottom_site_name)
                except Exception:

                    obj_bottom_world = obj_pos

                delta_z_world = float(obj_bottom_world[2] - table_top_z)

                z_eps = 0.04
                bottom_ok = abs(delta_z_world) <= z_eps
            except Exception as e:
                print(f"[ExactInZ WARNING] geometric calc failed: {e}")
                try:
                    bottom_ok = arg2.check_ontop(arg1)
                except Exception:
                    bottom_ok = False
            else:
                # only runs if above try success
                # print(
                #     f"[ExactInZ] obj={arg1.object_name}, reg={site_name}, "
                #     f"contain={contain_ok}, delta_z_world={delta_z_world:.4f}, "
                #     f"table_top_z={table_top_z:.4f}, eps={z_eps:.4f}, bottom_ok={bottom_ok}"
                # )
                pass
        else:
            # non-site: fallback
            try:
                bottom_ok = arg2.check_ontop(arg1)
            except Exception:
                bottom_ok = False

        result = contain_ok and bottom_ok
        try:
            reg_name = getattr(arg2, "object_name", "<unknown_region>")
            obj_name = getattr(arg1, "object_name", "<unknown_obj>")
            # print(
            #     f"[DEBUG ExactIn] obj={obj_name}, region={reg_name}, "
            #     f"contain_ok={contain_ok}, bottom_ok={bottom_ok}, result={result}"
            # )
        except Exception:
            pass

        return result


class On(BinaryAtomic):

    def __call__(self, arg1, arg2):
        return arg2.check_ontop(arg1)

        # if arg2.object_state_type == "site":
        #     return arg2.check_ontop(arg1)
        # else:
        #     obj_1_pos = arg1.get_geom_state()["pos"]
        #     obj_2_pos = arg2.get_geom_state()["pos"]
        #     # arg1.on_top_of(arg2) ?
        #     # TODO (Yfeng): Add checking of center of mass are in the same regions
        #     if obj_1_pos[2] >= obj_2_pos[2] and arg2.check_contact(arg1):
        #         return True
        #     else:
        #         return False


class Up(BinaryAtomic):

    def __call__(self, arg1):
        return arg1.get_geom_state()["pos"][2] >= 1.0


class Stack(BinaryAtomic):

    def __call__(self, arg1, arg2):
        return (arg1.check_contact(arg2) and arg2.check_contain(arg1) and
                arg1.get_geom_state()["pos"][2] > arg2.get_geom_state()["pos"][2])


class PrintJointState(UnaryAtomic):
    """This is a debug predicate to allow you print the joint values of the object you care"""

    def __call__(self, arg):
        print(arg.get_joint_state())
        return True


class Open(UnaryAtomic):

    def __call__(self, arg):
        return arg.is_open()


class Close(UnaryAtomic):

    def __call__(self, arg):
        return arg.is_close()


class TurnOn(UnaryAtomic):

    def __call__(self, arg):
        return arg.turn_on()


class TurnOff(UnaryAtomic):

    def __call__(self, arg):
        return arg.turn_off()


# ---------------------------------------------------------------------------
# Geometry / robot helpers (used by localization and grasp predicates)
# ---------------------------------------------------------------------------


def _eef_world_pos(env, robot_idx: int = 0) -> np.ndarray:
    robot = env.robots[robot_idx]
    sid = robot.eef_site_id
    return np.array(env.sim.data.site_xpos[sid], dtype=np.float64)


def _eef_linear_speed(env, robot_idx: int = 0) -> float:
    """Linear speed of the end-effector (m/s), Euclidean norm of translational velocity.

    Prefer robosuite's arm Jacobian-based estimate (``robot._hand_total_velocity``),
    which matches SingleArm / OSC controller conventions. Falls back to
    ``get_site_xvelp(grip_site)`` or 0.0.

    Note: A future extension is to fuse with relative object--gripper velocity for
    stricter "correctly grasped" checks under slip.
    """
    robot = env.robots[robot_idx]
    try:
        v = np.asarray(robot._hand_total_velocity[:3], dtype=np.float64)
        return float(np.linalg.norm(v))
    except Exception:
        pass
    data = env.sim.data
    gs = getattr(robot.gripper, "important_sites", {}).get("grip_site")
    if gs is not None and hasattr(data, "get_site_xvelp"):
        try:
            v = np.asarray(data.get_site_xvelp(gs), dtype=np.float64).reshape(-1)[:3]
            return float(np.linalg.norm(v))
        except Exception:
            pass
    return 0.0


def _flatten_gripper_important_geoms(gripper) -> Set[str]:
    """All prefixed geom names listed in the gripper's important_geoms map."""
    out: Set[str] = set()
    ig = getattr(gripper, "important_geoms", None) or {}
    for geoms in ig.values():
        if isinstance(geoms, (list, tuple)):
            for g in geoms:
                out.add(str(g))
        elif geoms is not None:
            out.add(str(geoms))
    return out


def _object_up_world(env, object_name: str) -> np.ndarray:
    """Unit vector along the object's body +z axis in world frame (mug-like up)."""
    bid = env.obj_body_id[object_name]
    xmat = np.array(env.sim.data.body_xmat[bid], dtype=np.float64).reshape(3, 3)
    up = xmat[:, 2]
    n = np.linalg.norm(up)
    if n < 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return up / n


def _gripper_tool_z_world(env, robot_idx: int = 0) -> np.ndarray:
    """World-frame unit vector along the gripper tool z (prefer ``ee_z`` site frame)."""
    robot = env.robots[robot_idx]
    gripper = robot.gripper
    sites = getattr(gripper, "important_sites", None) or {}
    data = env.sim.data
    site_name = sites.get("ee_z") or sites.get("grip_site")
    if site_name is None:
        sid = robot.eef_site_id
        R = np.array(data.site_xmat[sid], dtype=np.float64).reshape(3, 3)
    elif hasattr(data, "get_site_xmat"):
        try:
            R = np.asarray(data.get_site_xmat(site_name), dtype=np.float64).reshape(3, 3)
        except Exception:
            sid = robot.eef_site_id
            R = np.array(data.site_xmat[sid], dtype=np.float64).reshape(3, 3)
    else:
        sid = env.sim.model.site_name2id(site_name)
        R = np.array(data.site_xmat[sid], dtype=np.float64).reshape(3, 3)
    z = R[:, 2]
    n = np.linalg.norm(z)
    if n < 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return z / n


class NearEEF(UnaryAtomic):
    """True if the robot end-effector is within `threshold` metres of the object."""

    def __init__(self, threshold: float = 0.15):
        self.threshold = threshold

    def __call__(self, arg):
        obj_pos = np.array(arg.get_geom_state()["pos"])
        eef_pos = np.array(arg.env.sim.data.site_xpos[arg.env.robots[0].eef_site_id])
        return float(np.linalg.norm(obj_pos - eef_pos)) < self.threshold


class Grasp(UnaryAtomic):
    """Loose grasp cue: any gripper--object contact via ``check_contact``.

    This is true for incidental brushing against links or pads. For a stricter
    signal (gripper in contact and no other geoms touching the object), use
    :class:`DefaultGraspPredicate`.
    """

    def __call__(self, arg):
        robot = arg.env.robots[0]
        obj = arg.env.get_object(arg.object_name)
        return arg.env.check_contact(robot.gripper, obj)


def _gripper_only_contacts(env, obj, robot_idx: int = 0) -> bool:
    """True if every external geom touching ``obj`` belongs to gripper important_geoms.

    When the object still rests on a surface, contacts will include the table /
    shelf; this predicate is therefore expected to be false until the object is
    isolated or lifted.     Optional future tightening: require low relative tangential
    velocity between pads and object to reject sliding "false grasps".
    """
    gripper = env.robots[robot_idx].gripper
    allowed = _flatten_gripper_important_geoms(gripper)
    if not allowed:
        return False
    getc = getattr(env, "get_contacts", None)
    if getc is None:
        return False
    try:
        touching = getc(obj)
    except Exception:
        return False
    if not touching:
        return False
    return all((g in allowed) for g in touching)


class LocalizedNearEEF(UnaryAtomic):
    """Level-1 localization: near, slow EEF, and this object is closest among movables.

    Combines (1) Euclidean distance from object root body to grip site below
    ``dist_threshold``, (2) end-effector linear speed below ``vel_threshold``, and
    (3) this object's distance to the EEF is minimal over ``env.objects_dict``.

    ``vel_threshold`` may be set to ``None`` to skip the velocity test (e.g. if
    velocities are unavailable in a custom stack).
    """

    def __init__(
        self,
        dist_threshold: float = 0.12,
        vel_threshold: Optional[float] = 0.25,
        robot_idx: int = 0,
    ):
        self.dist_threshold = dist_threshold
        self.vel_threshold = vel_threshold
        self.robot_idx = robot_idx

    def __call__(self, arg):
        env = arg.env
        name = arg.object_name
        eef = _eef_world_pos(env, self.robot_idx)
        obj_pos = np.array(arg.get_geom_state()["pos"])
        dist = float(np.linalg.norm(obj_pos - eef))
        if dist >= self.dist_threshold:
            return False
        if self.vel_threshold is not None:
            if _eef_linear_speed(env, self.robot_idx) >= self.vel_threshold:
                return False
        for other_name in env.objects_dict.keys():
            if other_name == name:
                continue
            op = np.array(
                env.sim.data.body_xpos[env.obj_body_id[other_name]],
                dtype=np.float64,
            )
            if float(np.linalg.norm(op - eef)) + 1e-9 < dist:
                return False
        return True


class DefaultGraspPredicate(UnaryAtomic):
    """Level-2 grasp: gripper contacts the object and nothing else touches the object.

    Requires (1) ``check_contact(robot.gripper, obj)`` and (2) every external geom
    contacting the object (via ``env.get_contacts``) lies in
    ``gripper.important_geoms``. So the table, shelf, or arm links outside that set
    must not be colliding with the object.

    This avoids per-fingerpad logic: if the gripper is the sole external contact,
    fingerpads are implicitly engaged for typical LIBERO grippers.

    Future work: optional EEF / relative velocity caps while closing.
    """

    def __init__(self, robot_idx: int = 0):
        self.robot_idx = robot_idx

    def __call__(self, arg):
        env = arg.env
        robot = env.robots[self.robot_idx]
        obj = env.get_object(arg.object_name)
        if not env.check_contact(robot.gripper, obj):
            return False
        return _gripper_only_contacts(env, obj, robot_idx=self.robot_idx)


class HorizontalGraspPredicate(DefaultGraspPredicate):
    """Gripper-only contact grasp plus a horizontal tool-axis check (tuned for mugs).

    After :class:`DefaultGraspPredicate` passes, requires the gripper tool
    z-axis (from ee_z if available) to be nearly perpendicular to the
    object's local +z axis (body frame), i.e. ``|dot(z_tool, z_obj)| <=
    max_abs_dot``. That matches a side-on / horizontal pick typical for
    cylindrical mugs standing upright on a table.

    **Scope note:** the axis alignment test assumes an upright, roughly
    axisymmetric mug-like object whose body z is a meaningful ``vertical''; it
    may need different object-frame axes for other categories.

    Future work: same velocity / slip extensions as :class:`DefaultGraspPredicate`.
    """

    def __init__(self, max_abs_dot: float = 0.35, robot_idx: int = 0):
        super().__init__(robot_idx=robot_idx)
        self.max_abs_dot = max_abs_dot

    def __call__(self, arg):
        if not super().__call__(arg):
            return False
        env = arg.env
        g = _gripper_tool_z_world(env, self.robot_idx)
        o = _object_up_world(env, arg.object_name)
        return abs(float(np.dot(g, o))) <= self.max_abs_dot


class Near(BinaryAtomic):
    """True if two objects are within `threshold` metres of each other."""

    def __init__(self, threshold: float = 0.10):
        self.threshold = threshold

    def __call__(self, arg1, arg2):
        pos1 = np.array(arg1.get_geom_state()["pos"])
        pos2 = np.array(arg2.get_geom_state()["pos"])
        return float(np.linalg.norm(pos1 - pos2)) < self.threshold
