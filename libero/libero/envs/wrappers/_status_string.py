"""
Pure-function NL status string module for hierarchical reward wrapper (Track D).

Produces the privileged ``[Status: ...]`` observation suffix that the critic reads
during PPO training. The actor masks from the stable ``[Status:`` delimiter.

**MuJoCo-free:** no imports from ``libero.libero.envs``; depends only on stdlib.

Spec: ``.cursor/plans/hierarchical_sim_wrapper_refactor_599aa945.plan.md`` (Track D).
Design: ``libero/libero/envs/wrappers/DESIGN.md`` §8.
"""

from __future__ import annotations

from typing import Mapping, MutableMapping, Sequence

__all__ = [
    "resolve_slot_status",
    "format_status_string",
    "append_status_to_obs",
]


def resolve_slot_status(
    subtask_name: str,
    l3_history: Mapping[str, bool],
    raw_predicates: Mapping[str, bool],
) -> str:
    """Return the compressed status token for a single subtask slot.

    Uses causal-implication compression — the highest satisfied level wins:

    - ``"done"``      if L3 one-shot latch is set (``l3_history[subtask_name]``).
    - ``"grasped"``   elif ``raw_predicates["L2::<subtask_name>"]`` is True.
    - ``"localized"`` elif ``raw_predicates["L1::<subtask_name>"]`` is True.
    - ``"pending"``   otherwise.

    Parameters
    ----------
    subtask_name:
        Exact subtask name string (matches ``l3_history`` keys and ``L3::*``
        ``raw_predicates`` key suffixes).
    l3_history:
        Wrapper-owned one-shot bitmask; ``True`` means the subtask was ever satisfied
        this episode.
    raw_predicates:
        Instantaneous predicate truth dict from the sim's ``step`` info.
    """
    if l3_history.get(subtask_name, False):
        return "done"
    if raw_predicates.get(f"L2::{subtask_name}", False):
        return "grasped"
    if raw_predicates.get(f"L1::{subtask_name}", False):
        return "localized"
    return "pending"


def format_status_string(
    subtask_names_ordered: Sequence[str],
    l3_history: Mapping[str, bool],
    raw_predicates: Mapping[str, bool],
) -> str:
    """Assemble the full privileged status string for all subtask slots.

    Format: ``"[Status: <name1>=<status1>, <name2>=<status2>, ...]"``

    The ``[Status:`` prefix is a **stable delimiter** used by the external RL codebase
    to compute actor masking boundaries. Do **not** change this prefix.

    Parameters
    ----------
    subtask_names_ordered:
        Subtask names in BDDL declaration order (same sequence used for
        ``raw_predicates`` key ordering).
    l3_history:
        Wrapper-owned one-shot bitmask (see :func:`resolve_slot_status`).
    raw_predicates:
        Instantaneous predicate truth dict from the sim's ``step`` info.

    Returns
    -------
    str
        The fully-formatted status string, e.g.
        ``"[Status: pick=done, place=pending]"``.
        If ``subtask_names_ordered`` is empty, returns ``"[Status: ]"``.
    """
    slots = ", ".join(
        f"{name}={resolve_slot_status(name, l3_history, raw_predicates)}"
        for name in subtask_names_ordered
    )
    return f"[Status: {slots}]"


def append_status_to_obs(
    obs: MutableMapping[str, object],
    instruction_key: str,
    status_string: str,
) -> None:
    """Append *status_string* to ``obs[instruction_key]`` in-place.

    Modifies the observation dict in-place by appending a single space followed by
    *status_string* to the instruction string value.

    Parameters
    ----------
    obs:
        Mutable observation dict produced by the sim's ``step``.
    instruction_key:
        Key whose value receives the status suffix (typically ``"instruction"``).
    status_string:
        The formatted status string from :func:`format_status_string`.

    Raises
    ------
    KeyError
        If *instruction_key* is not present in *obs*.
    TypeError
        If ``obs[instruction_key]`` is not a ``str``.
    """
    if instruction_key not in obs:
        raise KeyError(instruction_key)
    current = obs[instruction_key]
    if not isinstance(current, str):
        raise TypeError(
            f"obs[{instruction_key!r}] must be str, got {type(current).__name__!r}"
        )
    obs[instruction_key] = current + " " + status_string
