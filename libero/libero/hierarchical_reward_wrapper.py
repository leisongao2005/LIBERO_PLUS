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
from typing import Dict, List, Literal, Mapping, MutableMapping, Optional, Sequence, Tuple, cast

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
        # Per-episode one-shot history — all four levels fire at most once per episode.
        # Using one-shot latches for L1/L2 (not transient edge detection) prevents reward
        # farming: the model cannot re-earn L1/L2 rewards by releasing and re-approaching.
        self._l1_history: Dict[str, bool] = {}
        self._l2_history: Dict[str, bool] = {}
        # L3 one-shot bitmask keyed by plain subtask name (see module docstring).
        self._l3_history: Dict[str, bool] = {}
        # L4 one-shot bitmask (single key "L4").
        self._l4_history: Dict[str, bool] = {}
        # Full per-step predicate trajectory for the current episode.
        # Each entry is {"step": int, "raw": Dict[str, bool], "shaped_reward": float}.
        # Accessible via .predicate_trajectory; cleared on reset.
        self._predicate_trajectory: List[Dict] = []
        # Step index at which each predicate key first fired its one-shot latch.
        # Keys are the same predicate key strings used in predicate_deltas (e.g. "L1::pick",
        # "L3::place", "L4"). Only populated for keys that have actually fired; cleared on reset.
        self.first_fire_steps: Dict[str, int] = {}
        # Episode-local step counter; incremented at the start of each step() call.
        self._step_count: int = 0

    def reset(self, **kwargs):  # type: ignore[override]
        self._clear_episode_state()
        return self.env.reset(**kwargs)

    def step(self, action):  # type: ignore[override]
        current_step = self._step_count
        self._step_count += 1

        obs, reward, done, info = self.env.step(action)
        obs = cast(MutableMapping[str, object], obs)
        info = self._passthrough_sim_info(info)

        raw: Dict[str, bool] = info["raw_predicates"]
        subtask_names = self._resolved_subtask_names
        assert subtask_names is not None, "_resolved_subtask_names must be set before step"

        # --- L1/L2 one-shot latch (same semantics as L3/L4: fire at most once per episode) ---
        l1_keys = [f"L1::{s}" for s in subtask_names]
        l2_keys = [f"L2::{s}" for s in subtask_names]
        self._l1_history, l1_fired = _delta.apply_one_shot_latch(self._l1_history, raw, l1_keys)
        self._l2_history, l2_fired = _delta.apply_one_shot_latch(self._l2_history, raw, l2_keys)

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

        # --- Record first-fire step indices (instrumentation only, no reward effect) ---
        # L1/L2 keys are stored with their full "L1::" / "L2::" prefix.
        # L3 keys are translated from plain names to "L3::" prefixed keys for consistency.
        # L4 is stored as "L4".
        for key in l1_fired:
            if key not in self.first_fire_steps:
                self.first_fire_steps[key] = current_step
        for key in l2_fired:
            if key not in self.first_fire_steps:
                self.first_fire_steps[key] = current_step
        for plain_key in l3_fired_plain:
            prefixed_key = f"L3::{plain_key}"
            if prefixed_key not in self.first_fire_steps:
                self.first_fire_steps[prefixed_key] = current_step
        for key in l4_fired:
            if key not in self.first_fire_steps:
                self.first_fire_steps[key] = current_step

        # --- Shaped reward ---
        shaped = _delta.compute_shaped_reward(
            {**l1_fired, **l2_fired},
            {**l3_fired, **l4_fired},
            self.reward_config.weights,
            list(subtask_names),
        )
        reward = float(reward) + shaped

        # --- Privileged status string ---
        status = _status.format_status_string(list(subtask_names), self._l3_history, raw)
        _status.append_status_to_obs(obs, self.instruction_key, status)

        # --- Expose wrapper outputs in info ---
        # Always include all keys (False when not fired) so downstream consumers
        # can index without checking membership.
        l1_all: Dict[str, bool] = {k: l1_fired.get(k, False) for k in l1_keys}
        l2_all: Dict[str, bool] = {k: l2_fired.get(k, False) for k in l2_keys}
        l3_all: Dict[str, bool] = {f"L3::{s}": l3_fired.get(f"L3::{s}", False) for s in subtask_names}
        l4_all: Dict[str, bool] = {"L4": l4_fired.get("L4", False)}
        info["predicate_deltas"] = {**l1_all, **l2_all, **l3_all, **l4_all}
        info["privileged_status"] = status
        info["shaped_reward"] = shaped

        # Append to per-episode trajectory (one entry per step).
        self._predicate_trajectory.append({
            "step": len(self._predicate_trajectory),
            "raw": dict(raw),
            "shaped_reward": shaped,
        })

        return obs, reward, done, info

    @property
    def predicate_trajectory(self) -> List[Dict]:
        """Read-only view of per-step predicate history for the current episode.

        Each entry: ``{"step": int, "raw": Dict[str, bool], "shaped_reward": float}``.
        Cleared on reset. Safe to read at episode end for post-hoc reward analysis.
        """
        return list(self._predicate_trajectory)

    def print_predicate_history(self) -> None:
        """Print a human-readable table of predicate values over the episode.

        One row per step. Columns: step, shaped_reward, then one column per predicate
        key (sorted). True shown as ``1``, False as ``.``. Rows where shaped_reward > 0
        are marked with ``*`` to highlight reward events.

        Also prints the one-shot first-fire summary at the bottom so you can verify
        that L1/L2 rewards were capped correctly (farming guard).
        """
        traj = self._predicate_trajectory
        if not traj:
            print("[predicate history] no steps recorded")
            return

        # Collect all predicate keys (sorted for stable column order).
        all_keys = sorted(traj[0]["raw"].keys())
        header_cols = ["step", "reward"] + all_keys
        col_w = max(len(k) for k in header_cols)

        def fmt(val):
            if isinstance(val, bool):
                return "1" if val else "."
            if isinstance(val, float):
                return f"{val:.3f}"
            return str(val)

        # Header
        print("  ".join(c.rjust(col_w) for c in header_cols))
        print("  ".join("-" * col_w for _ in header_cols))

        for entry in traj:
            step = entry["step"]
            sr = entry["shaped_reward"]
            marker = "*" if sr > 0 else " "
            row = [str(step), f"{sr:.3f}"] + [fmt(entry["raw"].get(k, False)) for k in all_keys]
            print(marker + " ".join(c.rjust(col_w) for c in row))

        # One-shot first-fire summary
        print()
        print("First-fire summary (one-shot latch state — each key earns reward at most once):")
        combined = {
            **{k: v for k, v in self._l1_history.items()},
            **{k: v for k, v in self._l2_history.items()},
            **{f"L3::{k}": v for k, v in self._l3_history.items()},
            **self._l4_history,
        }
        for k in sorted(combined):
            if combined[k] and k in self.first_fire_steps:
                print(f"  {k}: fired at step {self.first_fire_steps[k]}")
            else:
                print(f"  {k}: never")

    def _clear_episode_state(self) -> None:
        if self._explicit_subtask_names is None:
            self._resolved_subtask_names = None
        else:
            self._resolved_subtask_names = self._explicit_subtask_names
        self._l1_history.clear()
        self._l2_history.clear()
        self._l3_history.clear()
        self._l4_history.clear()
        self._predicate_trajectory.clear()
        self.first_fire_steps.clear()
        self._step_count = 0

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
