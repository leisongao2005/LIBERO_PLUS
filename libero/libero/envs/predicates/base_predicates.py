from typing import List


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


import numpy as np


class NearEEF(UnaryAtomic):
    """True if the robot end-effector is within `threshold` metres of the object."""

    def __init__(self, threshold: float = 0.15):
        self.threshold = threshold

    def __call__(self, arg):
        obj_pos = np.array(arg.get_geom_state()["pos"])
        eef_pos = np.array(arg.env.sim.data.site_xpos[arg.env.robots[0].eef_site_id])
        return float(np.linalg.norm(obj_pos - eef_pos)) < self.threshold


class Grasp(UnaryAtomic):
    """True if the robot gripper is in contact with the object."""

    # TODO: we are grasping if we are touching and also add some lifting check so we actually are holding it

    def __call__(self, arg):
        robot = arg.env.robots[0]
        obj = arg.env.get_object(arg.object_name)
        return arg.env.check_contact(robot.gripper, obj)


class Near(BinaryAtomic):
    """True if two objects are within `threshold` metres of each other."""

    def __init__(self, threshold: float = 0.10):
        self.threshold = threshold

    def __call__(self, arg1, arg2):
        pos1 = np.array(arg1.get_geom_state()["pos"])
        pos2 = np.array(arg2.get_geom_state()["pos"])
        return float(np.linalg.norm(pos1 - pos2)) < self.threshold
