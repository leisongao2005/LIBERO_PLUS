from .base_predicates import *


VALIDATE_PREDICATE_FN_DICT = {
    "true": TruePredicateFn(),
    "false": FalsePredicateFn(),
    "in": In(),
    # "incontact": InContactPredicateFn(),
    "on": On(),
    "up": Up(),
    # "stack":     Stack(),
    # "temporal":  TemporalPredicate(),
    "printjointstate": PrintJointState(),
    "open": Open(),
    "close": Close(),
    "turnon": TurnOn(),
    "turnoff": TurnOff(),
    "grasp": Grasp(),
}

# Predicates that require numeric parameters and must be instantiated at parse time.
# Keys are lowercased predicate names as they appear after BDDL tokenization.
PARAMETRIC_PREDICATE_CLS = {
    "neareef": NearEEF,
    "near": Near,
}


def update_predicate_fn_dict(fn_key, fn_name):
    VALIDATE_PREDICATE_FN_DICT.update({fn_key: eval(fn_name)()})


def eval_predicate_fn(predicate_fn_name, *args):
    assert predicate_fn_name in VALIDATE_PREDICATE_FN_DICT
    return VALIDATE_PREDICATE_FN_DICT[predicate_fn_name](*args)


def get_predicate_fn_dict():
    return VALIDATE_PREDICATE_FN_DICT


def get_predicate_fn(predicate_fn_name):
    return VALIDATE_PREDICATE_FN_DICT[predicate_fn_name.lower()]


def instantiate_predicate(predicate_name: str, numeric_params: list):
    """Instantiate a parametric predicate (e.g. NearEEF, Near) with numeric args."""
    name = predicate_name.lower()
    assert name in PARAMETRIC_PREDICATE_CLS, (
        f"Unknown parametric predicate '{name}'. "
        f"Available: {list(PARAMETRIC_PREDICATE_CLS.keys())}"
    )
    return PARAMETRIC_PREDICATE_CLS[name](*numeric_params)
