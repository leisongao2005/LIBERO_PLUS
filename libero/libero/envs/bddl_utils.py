from bddl.parsing import *

import itertools
import numpy as np


def parse_subtask_rewards(group):
    """Parse a (:subtask_rewards ...) token group into a structured list.

    Each element of the returned list is a dict:
      {
        "name":          str,
        "predicate_name": str,
        "predicate_fn":  callable,    # pre-instantiated with any numeric params
        "predicate_args": list[str],  # object-name tokens resolved at reward time
        "reward":        float,       # fractional weight; all weights sum to 1.0
        "after":         list[str],   # prerequisite subtask names (:after ordering)
        "confirm_steps": int | None,  # optional per-subtask delay override
      }

    Predicates with numeric thresholds (NearEEF, Near) are instantiated here so
    their __call__ only receives ObjectState arguments at evaluation time.
    """
    from libero.libero.envs.predicates import (
        VALIDATE_PREDICATE_FN_DICT,
        PARAMETRIC_PREDICATE_CLS,
        instantiate_predicate,
    )

    group.pop(0)  # discard ':subtask_rewards' keyword
    subtasks = []

    for item in group:
        assert item[0] == ":subtask", f"Expected ':subtask', got '{item[0]}'"
        name = item[1]
        pred_fn = None
        pred_name = None
        pred_args = []
        reward = None
        after = []
        confirm_steps = None

        for attr in item[2:]:
            key = attr[0]
            if key == ":predicate":
                pred_spec = attr[1]          # e.g. ['neareef', 'flat_stove_1', '0.15']
                pred_name = pred_spec[0].lower()
                obj_args, num_params = [], []
                for tok in pred_spec[1:]:
                    try:
                        num_params.append(float(tok))
                    except (ValueError, TypeError):
                        obj_args.append(tok)
                if pred_name in PARAMETRIC_PREDICATE_CLS:
                    pred_fn = instantiate_predicate(pred_name, num_params)
                elif pred_name in VALIDATE_PREDICATE_FN_DICT:
                    pred_fn = VALIDATE_PREDICATE_FN_DICT[pred_name]
                else:
                    raise ValueError(
                        f"Unknown predicate '{pred_name}' in :subtask_rewards. "
                        f"Standard predicates: {list(VALIDATE_PREDICATE_FN_DICT.keys())}. "
                        f"Parametric predicates: {list(PARAMETRIC_PREDICATE_CLS.keys())}."
                    )
                pred_args = obj_args
            elif key == ":reward":
                reward = float(attr[1])
            elif key == ":after":
                after = list(attr[1:])
            elif key == ":confirm_steps":
                confirm_steps = int(attr[1])

        assert pred_fn is not None, f"Subtask '{name}' has no :predicate attribute"
        subtasks.append(
            {
                "name": name,
                "predicate_name": pred_name,
                "predicate_fn": pred_fn,
                "predicate_args": pred_args,
                "reward": reward,
                "after": after,
                "confirm_steps": confirm_steps,
            }
        )

    # Fill equal-split default rewards for any subtask with reward=None.
    n_unspecified = sum(1 for s in subtasks if s["reward"] is None)
    allocated = sum(s["reward"] for s in subtasks if s["reward"] is not None)
    if n_unspecified > 0:
        default_reward = max(0.0, (1.0 - allocated) / n_unspecified)
        for s in subtasks:
            if s["reward"] is None:
                s["reward"] = default_reward

    return subtasks

pi = np.pi


def get_regions(t, regions, group):
    group.pop(0)
    while group:
        region = group.pop(0)
        region_name = region[0]
        target_name = None
        region_dict = {
            "target": None,
            "ranges": [],
            "extra": [],
            "yaw_rotation": [0, 0],
            "rgba": [0, 0, 1, 0],
        }
        for attribute in region[1:]:
            if attribute[0] == ":target":
                assert len(attribute) == 2
                region_dict["target"] = attribute[1]
                target_name = attribute[1]
            elif attribute[0] == ":ranges":
                for rect_range in attribute[1]:
                    assert (
                        len(rect_range) == 4
                    ), f"Dimension of rectangular range mismatched!!, supposed to be 4, only found {len(rect_range)}"
                    region_dict["ranges"].append([float(x) for x in rect_range])
            elif attribute[0] == ":yaw_rotation":
                # print(attribute[1])
                for value in attribute[1]:
                    region_dict["yaw_rotation"] = [eval(x) for x in value]
            elif attribute[0] == ":rgba":
                assert (
                    len(attribute[1]) == 4
                ), f"Missing specification for rgba color, supposed to be 4 dimension, but only got  {attribute[1]}"
                region_dict["rgba"] = [float(x) for x in attribute[1]]
            else:
                raise NotImplementedError
        regions[target_name + "_" + region_name] = region_dict


def get_scenes(t, scene_properties, group):
    group.pop(0)
    while group:
        scene_property = group.pop(0)
        scene_properties_dict = {}
        for attribute in region[1:]:
            if attribute[0] == ":floor":
                assert len(attribute) == 2
                scene_properties_dict["floor_style"] = attribute[1]
            elif attribute[0] == ":wall":
                assert len(attribute) == 2
                scene_properties_dict["wall_style"] = attribute[1]
            else:
                raise NotImplementedError


def get_problem_info(problem_filename):
    domain_name = "unknown"
    problem_filename = problem_filename
    tokens = scan_tokens(filename=problem_filename)
    if isinstance(tokens, list) and tokens.pop(0) == "define":
        problem_name = "unknown"
        language_instruction = ""
        while tokens:
            group = tokens.pop()
            t = group[0]
            if t == "problem":
                problem_name = group[-1]
            elif t == ":domain":
                domain_name = "robosuite"
            elif t == ":language":
                group.pop(0)
                language_instruction = group
    return {
        "problem_name": problem_name,
        "domain_name": domain_name,
        "language_instruction": " ".join(language_instruction),
    }


def robosuite_parse_problem(problem_filename):
    domain_name = "robosuite"
    problem_filename = problem_filename
    tokens = scan_tokens(filename=problem_filename)
    if isinstance(tokens, list) and tokens.pop(0) == "define":
        problem_name = "unknown"
        objects = {}
        obj_of_interest = []
        initial_state = []
        goal_state = []
        subtask_rewards = []
        fixtures = {}
        regions = {}
        scene_properties = {}
        language_instruction = ""
        while tokens:
            group = tokens.pop()
            t = group[0]
            if t == "problem":
                problem_name = group[-1]
            elif t == ":domain":
                if domain_name != group[-1]:
                    raise Exception("Different domain specified in problem file")
            elif t == ":requirements":
                pass
            elif t == ":objects":
                group.pop(0)
                object_list = []
                while group:
                    if group[0] == "-":
                        group.pop(0)
                        objects[group.pop(0)] = object_list
                        object_list = []
                    else:
                        object_list.append(group.pop(0))
                if object_list:
                    if not "object" in objects:
                        objects["object"] = []
                    objects["object"] += object_list
            elif t == ":obj_of_interest":
                group.pop(0)
                while group:
                    obj_of_interest.append(group.pop(0))
            elif t == ":fixtures":
                group.pop(0)
                fixture_list = []
                while group:
                    if group[0] == "-":
                        group.pop(0)
                        fixtures[group.pop(0)] = fixture_list
                        fixture_list = []
                    else:
                        fixture_list.append(group.pop(0))
                if fixture_list:
                    if not "fixture" in fixtures:
                        fixtures["fixture"] = []
                    fixtures["fixture"] += fixture_list
            elif t == ":regions":
                get_regions(t, regions, group)
            elif t == ":scene_properties":
                get_scenes(t, scene_properties, group)
            elif t == ":language":
                group.pop(0)
                language_instruction = group

            elif t == ":init":
                group.pop(0)
                initial_state = group
            elif t == ":goal":
                package_predicates(group[1], goal_state, "", "goals")
            elif t == ":subtask_rewards":
                subtask_rewards = parse_subtask_rewards(group)
            else:
                print("%s is not recognized in problem" % t)
        return {
            "problem_name": problem_name,
            "fixtures": fixtures,
            "regions": regions,
            "objects": objects,
            "scene_properties": scene_properties,
            "initial_state": initial_state,
            "goal_state": goal_state,
            "subtask_rewards": subtask_rewards,
            "language_instruction": language_instruction,
            "obj_of_interest": obj_of_interest,
        }
    else:
        raise Exception(
            f"Problem {behavior_activity} {activity_definition} does not match problem pattern"
        )
