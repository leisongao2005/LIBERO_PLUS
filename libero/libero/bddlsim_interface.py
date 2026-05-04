"""
# Sim ↔ `HierarchicalRewardWrapper` step `info` contract (Phase 0, frozen)

Canonical **MuJoCo-free** module: `from libero.libero import bddlsim_interface` (or import
symbols from `libero.libero.bddlsim_interface`) does not import `libero.libero.envs`.
The planned alias path `libero.libero.envs.wrappers._interface` re-exports the same API.

This module is the **single source of truth** for the Markovian simulator `step`
payload that the hierarchical reward wrapper consumes. Track A (`BDDLBaseDomain`)
must emit exactly this shape every step, with no feature flags.

## Required `info` keys (every `step`, unconditional)

| Key | Type | Semantics |
|-----|------|-----------|
| `raw_predicates` | `dict[str, bool]` | Instantaneous predicate truth. **No** one-shot latch inside sim. |
| `subtask_history` | `dict[str, bool]` | L3 one-shot bitmask for the episode: keys are **L3 subtask names** in BDDL declaration order; values flip to `True` the first time that L3 predicate is credited (sim implementation detail) and stay `True`. |
| `l4_satisfied` | `bool` | Full `(:goal ...)` satisfaction on this physics step (before any wrapper latch). |

## `raw_predicates` key naming (locked)

- Per L3 subtask name `S` (exact string from BDDL `:subtask` / goal ordering — same strings as `subtask_history` keys):
  - `L1::<S>` — auto-derived localization predicate for `S`’s **primary object** (see below).
  - `L2::<S>` — auto-derived grasp predicate for that object.
  - `L3::<S>` — the BDDL L3 subtask predicate for `S`, instantaneous.
- Terminal goal: **`L4`** (exact key, no prefix) — same proposition as full-goal success; mirrors `l4_satisfied` as a bool entry inside `raw_predicates`.

**Ordering:** Iteration / UI order is BDDL declaration order for L1/L2/L3 triples, then the `L4` entry. Consumers should not rely on dict insertion order for logic, but sim should build dicts in this order for stable logs.

## Primary object rule (for L1/L2 derivation)

For each L3 slot, the **primary object** is the first non-numeric token in that
subtask’s `predicate_args` list (see `_first_obj_state_from_goal_tokens` in
`bddl_base_domain.py`). L1/L2 keys above refer to predicates evaluated on that object.

## Reward

Sim `reward` is **sparse** (e.g. 0/1 on full goal only). All shaping lives in the
wrapper. This fixture returns a configurable scalar for tests.

## Asymmetric actor–critic

The wrapper may append a compressed NL status string to `obs["instruction"]`.
Models may mask from a stable delimiter (e.g. `[Status:`); that masking is **not**
part of this contract.

---
"""

from __future__ import annotations

from copy import deepcopy
from typing import Callable, Dict, Mapping, MutableMapping, Optional, Sequence, TypedDict, cast

import gym
import numpy as np
from gym import spaces


class BDDLSimStepInfo(TypedDict):
    """Mandatory keys on every real (or fake) BDDL sim `step` / `info`."""

    raw_predicates: Dict[str, bool]
    subtask_history: Dict[str, bool]
    l4_satisfied: bool


def raw_predicate_key_l1(subtask_name: str) -> str:
    return f"L1::{subtask_name}"


def raw_predicate_key_l2(subtask_name: str) -> str:
    return f"L2::{subtask_name}"


def raw_predicate_key_l3(subtask_name: str) -> str:
    return f"L3::{subtask_name}"


RAW_PREDICATE_KEY_L4 = "L4"


def empty_raw_predicates(subtask_names_ordered: Sequence[str]) -> Dict[str, bool]:
    """All-false `raw_predicates` including `L4`, in stable declaration order."""
    out: Dict[str, bool] = {}
    for name in subtask_names_ordered:
        out[raw_predicate_key_l1(name)] = False
        out[raw_predicate_key_l2(name)] = False
        out[raw_predicate_key_l3(name)] = False
    out[RAW_PREDICATE_KEY_L4] = False
    return out


def make_empty_sim_step_info(subtask_names_ordered: Sequence[str]) -> BDDLSimStepInfo:
    """Default episode-initial info: no L3 progress, no L4, all raw flags false."""
    history = {name: False for name in subtask_names_ordered}
    return BDDLSimStepInfo(
        raw_predicates=empty_raw_predicates(subtask_names_ordered),
        subtask_history=history,
        l4_satisfied=False,
    )


EXAMPLE_SIM_STEP_INFO: BDDLSimStepInfo = BDDLSimStepInfo(
    raw_predicates={
        "L1::stove_on": False,
        "L2::stove_on": False,
        "L3::stove_on": True,
        "L1::open_door": True,
        "L2::open_door": False,
        "L3::open_door": False,
        RAW_PREDICATE_KEY_L4: False,
    },
    subtask_history={"stove_on": True, "open_door": False},
    l4_satisfied=False,
)


def validate_sim_step_info(
    info: Mapping[str, object],
    *,
    subtask_names_ordered: Sequence[str],
    strict_raw_keys: bool = True,
) -> None:
    """Raise `ValueError` if `info` does not satisfy the Phase 0 contract."""
    missing = [k for k in ("raw_predicates", "subtask_history", "l4_satisfied") if k not in info]
    if missing:
        raise ValueError(f"info missing keys {missing}")

    raw = info["raw_predicates"]
    hist = info["subtask_history"]
    l4_flag = info["l4_satisfied"]

    if not isinstance(raw, Mapping):
        raise ValueError("raw_predicates must be a dict-like mapping")
    if not isinstance(hist, Mapping):
        raise ValueError("subtask_history must be a dict-like mapping")
    if not isinstance(l4_flag, bool):
        raise ValueError("l4_satisfied must be bool")

    raw = cast(Mapping[str, bool], raw)
    hist = cast(Mapping[str, bool], hist)

    if list(hist.keys()) != list(subtask_names_ordered):
        raise ValueError(
            "subtask_history keys must exactly match subtask_names_ordered "
            f"(got {list(hist.keys())!r}, want {list(subtask_names_ordered)!r})"
        )

    if RAW_PREDICATE_KEY_L4 not in raw:
        raise ValueError(f'raw_predicates must contain "{RAW_PREDICATE_KEY_L4}"')

    if bool(raw[RAW_PREDICATE_KEY_L4]) != bool(l4_flag):
        raise ValueError("l4_satisfied must equal raw_predicates['L4']")

    if strict_raw_keys:
        expected = set(empty_raw_predicates(subtask_names_ordered).keys())
        if set(raw.keys()) != expected:
            raise ValueError(
                "raw_predicates key set mismatch: "
                f"expected {sorted(expected)!r}, got {sorted(raw.keys())!r}"
            )

    for k, v in raw.items():
        if not isinstance(v, bool):
            raise ValueError(f"raw_predicates[{k!r}] must be bool, got {type(v)}")
    for k, v in hist.items():
        if not isinstance(v, bool):
            raise ValueError(f"subtask_history[{k!r}] must be bool, got {type(v)}")


class InstructionStrSpace(spaces.Space):
    """`gym.Space` for a single natural-language instruction (`str`)."""

    def __init__(self) -> None:
        super().__init__(shape=(), dtype=object)

    def contains(self, x: object) -> bool:
        return isinstance(x, str)

    def sample(self) -> str:
        return ""


class FakeBDDLEnv(gym.Env):
    """
    Minimal `gym.Env` stub for wrapper / delta / NL-string tests without MuJoCo.

    - `step` returns `(obs, reward, done, info)` with `info` produced by
      `get_step_info(self._step_index)` before the step counter is incremented.
    - `reset` clears the step counter and returns `obs` built from
      `base_instruction` and optional `obs_template`.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        subtask_names_ordered: Sequence[str],
        base_instruction: str = "complete the task",
        get_step_info: Optional[Callable[[int], BDDLSimStepInfo]] = None,
        default_sparse_reward: float = 0.0,
        default_done: bool = False,
        obs_template: Optional[MutableMapping[str, object]] = None,
    ) -> None:
        super().__init__()
        self.subtask_names_ordered = list(subtask_names_ordered)
        self.base_instruction = base_instruction
        self._get_step_info = get_step_info or (
            lambda _i: make_empty_sim_step_info(self.subtask_names_ordered)
        )
        self.default_sparse_reward = float(default_sparse_reward)
        self.default_done = bool(default_done)
        self._obs_template: MutableMapping[str, object] = (
            dict(obs_template) if obs_template is not None else {}
        )
        self._step_index = 0

        obs_spaces: Dict[str, spaces.Space] = {"instruction": InstructionStrSpace()}
        for key, value in self._obs_template.items():
            if key == "instruction":
                continue
            obs_spaces[key] = infer_space_for_value(value)
        self.observation_space = spaces.Dict(obs_spaces)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

    def reset(self):  # type: ignore[override]
        self._step_index = 0
        obs = self._build_obs()
        return obs

    def step(self, action):  # type: ignore[override]
        info = self._get_step_info(self._step_index)
        validate_sim_step_info(info, subtask_names_ordered=self.subtask_names_ordered)
        self._step_index += 1
        obs = self._build_obs()
        return obs, self.default_sparse_reward, self.default_done, dict(info)

    def _build_obs(self) -> Dict[str, object]:
        obs: Dict[str, object] = {}
        for k, v in self._obs_template.items():
            obs[k] = deepcopy(v) if isinstance(v, np.ndarray) else v
        obs.setdefault("instruction", self.base_instruction)
        return obs


def infer_space_for_value(value: object) -> spaces.Space:
    if isinstance(value, np.ndarray):
        return spaces.Box(low=-np.inf, high=np.inf, shape=value.shape, dtype=value.dtype)
    if isinstance(value, (float, np.floating)):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32)
    if isinstance(value, (int, np.integer)):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.int64)
    return spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)


from libero.libero import hierarchical_reward_wrapper as _hierarchical_reward_wrapper

HierarchicalRewardWrapper = _hierarchical_reward_wrapper.HierarchicalRewardWrapper
LevelName = _hierarchical_reward_wrapper.LevelName
RewardConfig = _hierarchical_reward_wrapper.RewardConfig

__all__ = [
    "BDDLSimStepInfo",
    "EXAMPLE_SIM_STEP_INFO",
    "FakeBDDLEnv",
    "HierarchicalRewardWrapper",
    "InstructionStrSpace",
    "LevelName",
    "RAW_PREDICATE_KEY_L4",
    "RewardConfig",
    "empty_raw_predicates",
    "infer_space_for_value",
    "make_empty_sim_step_info",
    "raw_predicate_key_l1",
    "raw_predicate_key_l2",
    "raw_predicate_key_l3",
    "validate_sim_step_info",
]
