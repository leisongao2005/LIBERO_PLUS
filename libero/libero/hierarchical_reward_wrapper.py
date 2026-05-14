"""
:class:`gym.Wrapper` for hierarchical L1–L4 shaping (Phase 2: Tracks C/D wired).

**MuJoCo-free import:** ``from libero.libero.hierarchical_reward_wrapper import …`` (or
symbols re-exported from :mod:`libero.libero.bddlsim_interface`). The plan path
``libero/libero/envs/wrappers/hierarchical_reward_wrapper.py`` re-exports the same API but
pulls in ``libero.libero.envs`` (heavy); prefer this module for tests without robosuite.

Spec: ``.cursor/plans/hierarchical_sim_wrapper_refactor_599aa945.plan.md``.
Sim ``step`` ``info`` contract: :mod:`libero.libero.bddlsim_interface`.

**Asymmetric actor–critic:** the wrapper appends a ``[Status: …]`` suffix to
``obs[instruction_key]`` every step.  Actor models must mask from the stable
``"[Status:"`` delimiter; that masking is external to this module.

**l3_history key convention:** ``_l3_history`` uses **plain subtask names** (e.g.
``"turnon_stove"``).  ``_status_string.resolve_slot_status`` requires this.
``compute_shaped_reward`` requires ``L3::`` prefixed keys in ``one_shot_fired``; the
wrapper translates via ``{f"L3::{k}": v …}``.
"""

from __future__ import annotations

import importlib.util as _ilu
import sys as _sys
from dataclasses import dataclass, field
from pathlib import Path as _Path
from typing import Dict, Literal, Mapping, MutableMapping, Optional, Sequence, Tuple, cast

import gym

from libero.libero.bddlsim_interface import validate_sim_step_info

# ---------------------------------------------------------------------------
# MuJoCo-free direct load of pure reward-function modules.
# Importing via ``libero.libero.envs.wrappers._delta`` would trigger
# ``envs/__init__.py`` (robosuite), breaking the MuJoCo-free import guarantee.
# Loading by file path bypasses the parent package init while still caching in
# sys.modules under the canonical qualified name.
# ---------------------------------------------------------------------------

def _load_pure_wrappers_module(name: str):
    qual = f"libero.libero.envs.wrappers.{name}"
    if qual in _sys.modules:
        return _sys.modules[qual]
    path = _Path(__file__).parent / "envs" / "wrappers" / f"{name}.py"
    spec = _ilu.spec_from_file_location(qual, path)
    mod = _ilu.module_from_spec(spec)
    _sys.modules[qual] = mod
    spec.loader.exec_module(mod)
    return mod


_delta = _load_pure_wrappers_module("_delta")
_status = _load_pure_wrappers_module("_status_string")

# ---------------------------------------------------------------------------

LevelName = Literal["L1", "L2", "L3", "L4"]


def _default_weights() -> Dict[LevelName, float]:
    return {"L1": 0.1, "L2": 0.2, "L3": 0.5, "L4": 1.0}


@dataclass
class RewardConfig:
    """
    Wrapper-side reward shaping configuration (replaces BDDL ``:reward`` weights).

    First-pass weights are used as-is; optional renormalization is tracked in the
    plan todo ``weight-normalization``.
    """

    weights: Dict[LevelName, float] = field(default_factory=_default_weights)
    #: Optional per–L3-subtask overrides; inner keys are level names ``L1``..``L4``.
    per_subtask_overrides: Dict[str, Dict[str, float]] = field(default_factory=dict)
    #: Locked to ``0.0`` for the first implementation (see plan).
    transient_negative_penalty: float = 0.0

    def __post_init__(self) -> None:
        if self.transient_negative_penalty != 0.0:
            raise ValueError(
                "transient_negative_penalty is locked to 0.0 in the first implementation "
                "(see hierarchical sim wrapper refactor plan)."
            )
        required: Tuple[LevelName, ...] = ("L1", "L2", "L3", "L4")
        if set(self.weights.keys()) != set(required):
            raise ValueError(
                f"weights must contain exactly {set(required)!r}, got {set(self.weights.keys())!r}"
            )
        for st, ov in self.per_subtask_overrides.items():
            if not isinstance(ov, Mapping):
                raise ValueError(f"per_subtask_overrides[{st!r}] must be a mapping")
            bad = set(ov) - set(required)
            if bad:
                raise ValueError(f"per_subtask_overrides[{st!r}] has unknown keys {bad!r}")


class HierarchicalRewardWrapper(gym.Wrapper):
    """
    Wraps a Markovian BDDL sim (or :class:`~libero.libero.bddlsim_interface.FakeBDDLEnv`).

    Each ``step`` call:

    1. Passes the action to the underlying sim and validates ``info``.
    2. Detects transient L1/L2 deltas and one-shot L3/L4 fires.
    3. Adds shaped reward (``Σ delta × weight``) on top of the sparse sim reward.
    4. Appends ``[Status: …]`` to ``obs[instruction_key]``.
    5. Exposes ``info["predicate_deltas"]``, ``info["privileged_status"]``,
       ``info["shaped_reward"]`` for downstream consumers.

    Parameters
    ----------
    env
        Must emit ``info`` satisfying :func:`~libero.libero.bddlsim_interface.validate_sim_step_info`
        on every ``step``.
    reward_config
        Shaping weights and overrides.
    instruction_key
        Observation key whose string value will receive the privileged status suffix.
    subtask_names_ordered
        If set, ``raw_predicates`` key set is validated against this sequence every step.
        If omitted, subtask order is inferred from the first post-``reset`` ``step``'s
        ``raw_predicates`` ``L3::*`` keys (Python dict insertion order, i.e. BDDL
        declaration order). Passing this explicitly is recommended for production use.
    """

    def __init__(
        self,
        env: gym.Env,
        reward_config: RewardConfig,
        instruction_key: str = "instruction",
        subtask_names_ordered: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(env)
        self.reward_config = reward_config
        self.instruction_key = instruction_key
        self._explicit_subtask_names: Optional[Tuple[str, ...]] = (
            tuple(subtask_names_ordered) if subtask_names_ordered is not None else None
        )
        self._resolved_subtask_names: Optional[Tuple[str, ...]] = self._explicit_subtask_names
        # Per-episode state
        self._l1_prev: Dict[str, bool] = {}
        self._l2_prev: Dict[str, bool] = {}
        # L3 one-shot bitmask keyed by plain subtask name (see module docstring).
        self._l3_history: Dict[str, bool] = {}
        # L4 one-shot bitmask (single key "L4").
        self._l4_history: Dict[str, bool] = {}

    def reset(self, **kwargs):  # type: ignore[override]
        self._clear_episode_state()
        return self.env.reset(**kwargs)

    def step(self, action):  # type: ignore[override]
        obs, reward, done, info = self.env.step(action)
        obs = cast(MutableMapping[str, object], obs)
        info = self._passthrough_sim_info(info)

        raw: Dict[str, bool] = info["raw_predicates"]
        subtask_names = self._resolved_subtask_names
        assert subtask_names is not None, "_resolved_subtask_names must be set before step"

        # --- L1/L2 transient delta detection ---
        l1_keys = [f"L1::{s}" for s in subtask_names]
        l2_keys = [f"L2::{s}" for s in subtask_names]
        l1_deltas = _delta.detect_transient_deltas(self._l1_prev, raw, l1_keys)
        l2_deltas = _delta.detect_transient_deltas(self._l2_prev, raw, l2_keys)

        # --- L3 one-shot latch (plain-name keys in _l3_history) ---
        # Build a plain-name dict so apply_one_shot_latch stores plain keys,
        # which format_status_string / resolve_slot_status expect.
        l3_curr = {s: bool(raw.get(f"L3::{s}", False)) for s in subtask_names}
        self._l3_history, l3_fired_plain = _delta.apply_one_shot_latch(
            self._l3_history, l3_curr, list(subtask_names)
        )
        # Translate to L3::* keys for compute_shaped_reward.
        l3_fired: Dict[str, bool] = {f"L3::{k}": v for k, v in l3_fired_plain.items()}

        # --- L4 one-shot latch ---
        self._l4_history, l4_fired = _delta.apply_one_shot_latch(
            self._l4_history, {"L4": bool(raw.get("L4", False))}, ["L4"]
        )

        # --- Shaped reward ---
        shaped = _delta.compute_shaped_reward(
            {**l1_deltas, **l2_deltas},
            {**l3_fired, **l4_fired},
            self.reward_config.weights,
            list(subtask_names),
        )
        reward = float(reward) + shaped

        # --- Privileged status string ---
        status = _status.format_status_string(list(subtask_names), self._l3_history, raw)
        _status.append_status_to_obs(obs, self.instruction_key, status)

        # --- Update prev state for next step ---
        self._l1_prev = dict(raw)
        self._l2_prev = dict(raw)

        # --- Expose wrapper outputs in info ---
        # Always include all keys (False when not fired) so downstream consumers
        # can index without checking membership.
        l3_all: Dict[str, bool] = {f"L3::{s}": l3_fired.get(f"L3::{s}", False) for s in subtask_names}
        l4_all: Dict[str, bool] = {"L4": l4_fired.get("L4", False)}
        info["predicate_deltas"] = {**l1_deltas, **l2_deltas, **l3_all, **l4_all}
        info["privileged_status"] = status
        info["shaped_reward"] = shaped

        return obs, reward, done, info

    def _clear_episode_state(self) -> None:
        if self._explicit_subtask_names is None:
            self._resolved_subtask_names = None
        else:
            self._resolved_subtask_names = self._explicit_subtask_names
        self._l1_prev.clear()
        self._l2_prev.clear()
        self._l3_history.clear()
        self._l4_history.clear()

    def _passthrough_sim_info(self, info: Mapping[str, object]) -> Dict[str, object]:
        if not isinstance(info, MutableMapping):
            info = dict(info)
        else:
            info = cast(MutableMapping[str, object], info)
            info = {k: info[k] for k in info}

        if self._resolved_subtask_names is None:
            raw = info.get("raw_predicates")
            if not isinstance(raw, Mapping):
                raise ValueError(
                    'info["raw_predicates"] must be present and mapping-like on every step '
                    "when subtask_names_ordered was not passed to the wrapper."
                )
            l3_prefix = "L3::"
            self._resolved_subtask_names = tuple(
                k[len(l3_prefix):] for k in raw if k.startswith(l3_prefix)
            )

        validate_sim_step_info(
            info,
            subtask_names_ordered=self._resolved_subtask_names,
            strict_raw_keys=True,
        )
        return info


__all__ = ["HierarchicalRewardWrapper", "LevelName", "RewardConfig"]
