"""
Pure-function delta detection and shaped reward computation (Track C).

**MuJoCo-free module.** No imports from ``libero.libero.envs``; safe to import
without robosuite/MuJoCo present.

These three functions have no side effects and hold no state — all episode state
(``_l1_prev``, ``_l2_prev``, ``_l3_history``) lives in :class:`HierarchicalRewardWrapper`
and is passed in / returned as plain dicts.

See ``DESIGN.md`` §4 for the L2 scope restriction (pick-and-place only).
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

__all__ = [
    "detect_transient_deltas",
    "apply_one_shot_latch",
    "compute_shaped_reward",
]


def detect_transient_deltas(
    prev: Dict[str, bool],
    curr: Dict[str, bool],
    keys: Sequence[str],
) -> Dict[str, bool]:
    """Return which keys became True this step (transient — can re-fire each step).

    A key fires iff ``curr[key]`` is True *and* ``prev.get(key, False)`` was False.
    Used for L1 (localization) and L2 (grasp) predicates, which are transient and
    are allowed to re-fire on every step they are satisfied.

    Parameters
    ----------
    prev:
        Predicate values from the **previous** step (or empty dict at episode start).
    curr:
        Predicate values from the **current** step (``info["raw_predicates"]``).
    keys:
        Which keys to evaluate. Typically the L1/L2 keys for the active subtasks.

    Returns
    -------
    Dict[str, bool]
        Maps each key in ``keys`` to ``True`` iff the predicate is newly True this step.
    """
    result: Dict[str, bool] = {}
    for key in keys:
        was_true = prev.get(key, False)
        is_true = curr.get(key, False)
        result[key] = (not was_true) and is_true
    return result


def apply_one_shot_latch(
    history: Dict[str, bool],
    curr: Dict[str, bool],
    keys: Sequence[str],
) -> Tuple[Dict[str, bool], Dict[str, bool]]:
    """Latch keys that become True and report which fired for the first time.

    One-shot semantics: once a key becomes True in ``history`` it stays True
    forever (until the episode resets). A key is "newly fired" iff it was False in
    ``history`` and is True in ``curr``.

    Does **not** mutate ``history`` in-place — returns a fresh copy.

    Used for L3 (subtask) and L4 (terminal) predicates, which each fire at most once
    per episode.

    Parameters
    ----------
    history:
        The wrapper's current one-shot latch state (e.g. ``_l3_history``).
    curr:
        Predicate values from the current step (``info["raw_predicates"]``).
    keys:
        Which keys to evaluate. Typically the L3/L4 keys for the episode.

    Returns
    -------
    new_history:
        Copy of ``history`` updated so any key True in ``curr`` is also True.
    newly_fired:
        Keys that were False in ``history`` and are True in ``curr`` (fired this step).
    """
    new_history: Dict[str, bool] = dict(history)
    newly_fired: Dict[str, bool] = {}
    for key in keys:
        is_true = curr.get(key, False)
        if is_true and not history.get(key, False):
            newly_fired[key] = True
            new_history[key] = True
    return new_history, newly_fired


def compute_shaped_reward(
    transient_deltas: Dict[str, bool],
    one_shot_fired: Dict[str, bool],
    weights: Dict[str, float],
    subtask_names: Sequence[str],
) -> float:
    """Compute the shaped reward contribution for a single environment step.

    Reward breakdown per subtask ``s`` in ``subtask_names``:

    - ``weights["L1"]`` if ``transient_deltas["L1::{s}"]`` is True
      (localization: primary object near end-effector, newly True this step)
    - ``weights["L2"]`` if ``transient_deltas["L2::{s}"]`` is True
      (grasp: newly grasped this step; only meaningful for pick-and-place subtasks)
    - ``weights["L3"]`` if ``one_shot_fired["L3::{s}"]`` is True
      (subtask completed for the first time this episode)

    Plus globally:

    - ``weights["L4"]`` if ``one_shot_fired["L4"]`` is True
      (full goal satisfied for the first time this episode)

    The sparse reward passthrough (0/1 from the underlying sim) is **not** added here;
    that remains the wrapper's responsibility.

    Parameters
    ----------
    transient_deltas:
        Output of :func:`detect_transient_deltas` for L1/L2 keys this step.
    one_shot_fired:
        Output of :func:`apply_one_shot_latch` for L3/L4 keys this step.
    weights:
        Mapping with keys ``"L1"``, ``"L2"``, ``"L3"``, ``"L4"`` and float values.
        Default values (per the refactor spec) are
        ``{"L1": 0.1, "L2": 0.2, "L3": 0.5, "L4": 1.0}``, but this function
        does not enforce defaults — the caller (wrapper) owns the config.
    subtask_names:
        Ordered sequence of L3 subtask names for this episode (BDDL declaration order).

    Returns
    -------
    float
        Total shaped reward for this step.
    """
    total = 0.0
    for s in subtask_names:
        if transient_deltas.get(f"L1::{s}", False):
            total += weights["L1"]
        if transient_deltas.get(f"L2::{s}", False):
            total += weights["L2"]
        if one_shot_fired.get(f"L3::{s}", False):
            total += weights["L3"]
    if one_shot_fired.get("L4", False):
        total += weights["L4"]
    return total
