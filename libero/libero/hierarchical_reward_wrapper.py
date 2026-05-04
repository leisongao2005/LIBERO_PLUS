"""
Skeleton :class:`gym.Wrapper` for hierarchical L1–L4 shaping (Tracks C/D integrate here).

**MuJoCo-free import:** ``from libero.libero.hierarchical_reward_wrapper import …`` (or
symbols re-exported from :mod:`libero.libero.bddlsim_interface`). The plan path
``libero/libero/envs/wrappers/hierarchical_reward_wrapper.py`` re-exports the same API but
pulls in ``libero.libero.envs`` (heavy); prefer this module for tests without robosuite.

Spec: ``.cursor/plans/hierarchical_sim_wrapper_refactor_599aa945.plan.md``.
Sim ``step`` ``info`` contract: :mod:`libero.libero.bddlsim_interface`.

**Track B:** ``RewardConfig``, ``HierarchicalRewardWrapper`` lifecycle (``reset`` / ``step``),
validation of sim ``info``, and passthrough of sparse reward and sim ``info`` keys. Shaped
reward, ``predicate_deltas``, and ``privileged_status`` are Phase 2 (Tracks C/D).

**Asymmetric actor–critic:** when status appending lands (Track D), models may mask from a
stable delimiter (e.g. the literal ``"[Status:"`` prefix); masking is not implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Mapping, MutableMapping, Optional, Sequence, Tuple, cast

import gym

from libero.libero.bddlsim_interface import validate_sim_step_info

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

    Parameters
    ----------
    env
        Must emit ``info`` satisfying :func:`~libero.libero.bddlsim_interface.validate_sim_step_info`
        on every ``step``.
    reward_config
        Shaping weights and overrides (consumed in Phase 2 when reward shaping is wired).
    instruction_key
        Observation key whose string value will receive the privileged status suffix (Track D).
    subtask_names_ordered
        If set, ``subtask_history`` key order is validated against this sequence every step.
        If omitted, subtask order is taken from the first post-``reset`` ``step``'s
        ``info["subtask_history"]`` keys (Python dict insertion order).
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
        # Populated each episode; Track C will use these for transient L1/L2 deltas.
        self._l1_prev: Dict[str, bool] = {}
        self._l2_prev: Dict[str, bool] = {}

    def reset(self, **kwargs):  # type: ignore[override]
        self._clear_episode_state()
        return self.env.reset(**kwargs)

    def step(self, action):  # type: ignore[override]
        obs, reward, done, info = self.env.step(action)
        info = self._passthrough_sim_info(info)
        return obs, reward, done, info

    def _clear_episode_state(self) -> None:
        if self._explicit_subtask_names is None:
            self._resolved_subtask_names = None
        else:
            self._resolved_subtask_names = self._explicit_subtask_names
        self._l1_prev.clear()
        self._l2_prev.clear()

    def _passthrough_sim_info(self, info: Mapping[str, object]) -> Dict[str, object]:
        if not isinstance(info, MutableMapping):
            info = dict(info)
        else:
            info = cast(MutableMapping[str, object], info)
            info = {k: info[k] for k in info}

        if self._resolved_subtask_names is None:
            hist = info.get("subtask_history")
            if not isinstance(hist, Mapping):
                raise ValueError(
                    'info["subtask_history"] must be present and mapping-like on every step '
                    "when subtask_names_ordered was not passed to the wrapper."
                )
            self._resolved_subtask_names = tuple(hist.keys())

        validate_sim_step_info(
            info,
            subtask_names_ordered=self._resolved_subtask_names,
            strict_raw_keys=True,
        )
        return info


__all__ = ["HierarchicalRewardWrapper", "LevelName", "RewardConfig"]
