"""
End-to-end smoke tests for HierarchicalRewardWrapper (Phase 2 wiring).

Uses FakeBDDLEnv — no MuJoCo required.

Run with:
    conda run -n libero_data python -m pytest tests/test_hierarchical_reward_wrapper.py -v
"""

from __future__ import annotations

from typing import Dict, List

import pytest

from libero.libero.bddlsim_interface import (
    BDDLSimStepInfo,
    FakeBDDLEnv,
    HierarchicalRewardWrapper,
    RewardConfig,
    make_empty_sim_step_info,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SUBTASKS = ["pick", "place"]
WEIGHTS = {"L1": 0.1, "L2": 0.2, "L3": 0.5, "L4": 1.0}


def _make_info(
    *,
    l1_pick=False, l2_pick=False, l3_pick=False,
    l1_place=False, l2_place=False, l3_place=False,
    l4=False,
) -> BDDLSimStepInfo:
    return BDDLSimStepInfo(
        raw_predicates={
            "L1::pick": l1_pick, "L2::pick": l2_pick, "L3::pick": l3_pick,
            "L1::place": l1_place, "L2::place": l2_place, "L3::place": l3_place,
            "L4": l4,
        },
        l4_satisfied=l4,
    )


def _make_wrapped(step_infos: List[BDDLSimStepInfo]) -> HierarchicalRewardWrapper:
    idx = [0]

    def get_info(i: int) -> BDDLSimStepInfo:
        return step_infos[i] if i < len(step_infos) else make_empty_sim_step_info(SUBTASKS)

    env = FakeBDDLEnv(
        subtask_names_ordered=SUBTASKS,
        base_instruction="do the task",
        get_step_info=get_info,
    )
    return HierarchicalRewardWrapper(
        env,
        reward_config=RewardConfig(weights=WEIGHTS),
        instruction_key="instruction",
        subtask_names_ordered=SUBTASKS,
    )


# ---------------------------------------------------------------------------
# TC-W1: Info keys added by wrapper
# ---------------------------------------------------------------------------

class TestWrapperInfoKeys:
    def test_wrapper_adds_info_keys(self):
        wrapped = _make_wrapped([_make_info()])
        wrapped.reset()
        _obs, _r, _done, info = wrapped.step(None)
        assert "predicate_deltas" in info
        assert "privileged_status" in info
        assert "shaped_reward" in info

    def test_sim_keys_still_present(self):
        wrapped = _make_wrapped([_make_info()])
        wrapped.reset()
        _obs, _r, _done, info = wrapped.step(None)
        assert "raw_predicates" in info
        assert "l4_satisfied" in info


# ---------------------------------------------------------------------------
# TC-W2: Shaped reward computation
# ---------------------------------------------------------------------------

class TestShapedReward:
    def test_zero_on_empty_step(self):
        wrapped = _make_wrapped([_make_info()])
        wrapped.reset()
        _obs, reward, _done, info = wrapped.step(None)
        assert info["shaped_reward"] == 0.0
        assert reward == 0.0  # sparse base = 0

    def test_l1_fires_on_transition_to_true(self):
        """L1 delta fires when predicate goes False→True."""
        steps = [_make_info(), _make_info(l1_pick=True)]
        wrapped = _make_wrapped(steps)
        wrapped.reset()
        wrapped.step(None)  # step 0: all False
        _obs, _r, _done, info = wrapped.step(None)  # step 1: l1_pick becomes True
        assert info["shaped_reward"] == pytest.approx(WEIGHTS["L1"])

    def test_l1_does_not_refire_while_held(self):
        """L1 fires once per episode — no re-fire on consecutive True steps."""
        steps = [_make_info(l1_pick=True), _make_info(l1_pick=True)]
        wrapped = _make_wrapped(steps)
        wrapped.reset()
        _obs, _r, _done, info0 = wrapped.step(None)  # first True: fires
        _obs, _r, _done, info1 = wrapped.step(None)  # still True: no re-fire
        assert info0["shaped_reward"] == pytest.approx(WEIGHTS["L1"])
        assert info1["shaped_reward"] == 0.0

    def test_l1_does_not_refire_after_release(self):
        """L1 must not re-earn reward after True→False→True (reward farming guard)."""
        steps = [
            _make_info(l1_pick=True),   # fires
            _make_info(l1_pick=False),  # drops
            _make_info(l1_pick=True),   # re-approaches — no second reward
        ]
        wrapped = _make_wrapped(steps)
        wrapped.reset()
        _obs, _r, _done, info0 = wrapped.step(None)
        _obs, _r, _done, info1 = wrapped.step(None)
        _obs, _r, _done, info2 = wrapped.step(None)
        assert info0["shaped_reward"] == pytest.approx(WEIGHTS["L1"])
        assert info1["shaped_reward"] == 0.0
        assert info2["shaped_reward"] == 0.0

    def test_l2_does_not_refire_after_release(self):
        """L2 must not re-earn reward after grasp→release→grasp (reward farming guard)."""
        steps = [
            _make_info(l2_pick=True),   # fires
            _make_info(l2_pick=False),  # released
            _make_info(l2_pick=True),   # re-grasped — no second reward
        ]
        wrapped = _make_wrapped(steps)
        wrapped.reset()
        _obs, _r, _done, info0 = wrapped.step(None)
        _obs, _r, _done, info1 = wrapped.step(None)
        _obs, _r, _done, info2 = wrapped.step(None)
        assert info0["shaped_reward"] == pytest.approx(WEIGHTS["L2"])
        assert info1["shaped_reward"] == 0.0
        assert info2["shaped_reward"] == 0.0

    def test_l2_fires_on_grasp(self):
        steps = [_make_info(), _make_info(l2_pick=True)]
        wrapped = _make_wrapped(steps)
        wrapped.reset()
        wrapped.step(None)
        _obs, _r, _done, info = wrapped.step(None)
        assert info["shaped_reward"] == pytest.approx(WEIGHTS["L2"])

    def test_l3_fires_once_per_episode(self):
        """L3 subtask fires exactly once even if predicate stays True."""
        steps = [
            _make_info(l3_pick=True),
            _make_info(l3_pick=True),
            _make_info(l3_pick=True),
        ]
        wrapped = _make_wrapped(steps)
        wrapped.reset()
        _obs, _r, _done, info0 = wrapped.step(None)
        _obs, _r, _done, info1 = wrapped.step(None)
        _obs, _r, _done, info2 = wrapped.step(None)
        assert info0["shaped_reward"] == pytest.approx(WEIGHTS["L3"])
        assert info1["shaped_reward"] == 0.0
        assert info2["shaped_reward"] == 0.0

    def test_l3_resets_across_episodes(self):
        """L3 latch clears on reset so it can fire again next episode."""
        steps = [_make_info(l3_pick=True)]
        wrapped = _make_wrapped(steps)
        wrapped.reset()
        _obs, _r, _done, info0 = wrapped.step(None)
        assert info0["shaped_reward"] == pytest.approx(WEIGHTS["L3"])

        wrapped.reset()
        wrapped._get_step_info = lambda _: steps[0]
        _obs, _r, _done, info1 = wrapped.step(None)
        assert info1["shaped_reward"] == pytest.approx(WEIGHTS["L3"])

    def test_l4_fires_once(self):
        steps = [_make_info(l4=True), _make_info(l4=True)]
        wrapped = _make_wrapped(steps)
        wrapped.reset()
        _obs, _r, _done, info0 = wrapped.step(None)
        _obs, _r, _done, info1 = wrapped.step(None)
        assert info0["shaped_reward"] == pytest.approx(WEIGHTS["L4"])
        assert info1["shaped_reward"] == 0.0

    def test_sparse_plus_shaped(self):
        """Total reward = sparse (from FakeBDDLEnv) + shaped."""
        env = FakeBDDLEnv(
            subtask_names_ordered=SUBTASKS,
            base_instruction="task",
            get_step_info=lambda _: _make_info(l3_pick=True),
            default_sparse_reward=1.0,
        )
        wrapped = HierarchicalRewardWrapper(
            env, RewardConfig(weights=WEIGHTS), subtask_names_ordered=SUBTASKS
        )
        wrapped.reset()
        _obs, reward, _done, info = wrapped.step(None)
        assert reward == pytest.approx(1.0 + WEIGHTS["L3"])
        assert info["shaped_reward"] == pytest.approx(WEIGHTS["L3"])

    def test_combined_l1_l2_l3_same_step(self):
        """All three can fire in one step."""
        steps = [_make_info(l1_pick=True, l2_pick=True, l3_pick=True)]
        wrapped = _make_wrapped(steps)
        wrapped.reset()
        _obs, _r, _done, info = wrapped.step(None)
        expected = WEIGHTS["L1"] + WEIGHTS["L2"] + WEIGHTS["L3"]
        assert info["shaped_reward"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# TC-W3: Status string appended to obs
# ---------------------------------------------------------------------------

class TestStatusStringAppend:
    def test_status_appended_to_instruction(self):
        wrapped = _make_wrapped([_make_info()])
        wrapped.reset()
        obs, _r, _done, info = wrapped.step(None)
        instruction = obs["instruction"]
        assert isinstance(instruction, str)
        assert "[Status:" in instruction

    def test_status_reflects_l3_done(self):
        """After L3::pick fires, status string should show pick=done."""
        steps = [_make_info(l3_pick=True)]
        wrapped = _make_wrapped(steps)
        wrapped.reset()
        obs, _r, _done, info = wrapped.step(None)
        assert "pick=done" in info["privileged_status"]
        assert "place=pending" in info["privileged_status"]

    def test_status_shows_localized(self):
        steps = [_make_info(l1_pick=True)]
        wrapped = _make_wrapped(steps)
        wrapped.reset()
        obs, _r, _done, info = wrapped.step(None)
        assert "pick=localized" in info["privileged_status"]

    def test_status_shows_grasped(self):
        steps = [_make_info(l2_pick=True)]
        wrapped = _make_wrapped(steps)
        wrapped.reset()
        obs, _r, _done, info = wrapped.step(None)
        assert "pick=grasped" in info["privileged_status"]

    def test_status_resets_across_episodes(self):
        steps = [_make_info(l3_pick=True)]
        wrapped = _make_wrapped(steps)
        wrapped.reset()
        wrapped.step(None)
        # After reset, L3 latch clears → pick should be pending again
        wrapped.reset()
        wrapped._resolved_subtask_names = tuple(SUBTASKS)
        # Step with all-False info
        all_false = FakeBDDLEnv(
            subtask_names_ordered=SUBTASKS,
            get_step_info=lambda _: make_empty_sim_step_info(SUBTASKS),
        )
        wrapped2 = HierarchicalRewardWrapper(
            all_false, RewardConfig(weights=WEIGHTS), subtask_names_ordered=SUBTASKS
        )
        wrapped2.reset()
        obs2, _r, _done, info2 = wrapped2.step(None)
        assert "pick=pending" in info2["privileged_status"]


# ---------------------------------------------------------------------------
# TC-W4: predicate_deltas in info
# ---------------------------------------------------------------------------

class TestPredicateDeltas:
    def test_deltas_all_false_on_no_change(self):
        wrapped = _make_wrapped([_make_info()])
        wrapped.reset()
        _obs, _r, _done, info = wrapped.step(None)
        assert not any(info["predicate_deltas"].values())

    def test_deltas_keys_present(self):
        wrapped = _make_wrapped([_make_info()])
        wrapped.reset()
        _obs, _r, _done, info = wrapped.step(None)
        deltas = info["predicate_deltas"]
        for s in SUBTASKS:
            assert f"L1::{s}" in deltas
            assert f"L2::{s}" in deltas
            assert f"L3::{s}" in deltas
        assert "L4" in deltas

    def test_l1_delta_fires(self):
        steps = [_make_info(), _make_info(l1_pick=True)]
        wrapped = _make_wrapped(steps)
        wrapped.reset()
        wrapped.step(None)
        _obs, _r, _done, info = wrapped.step(None)
        assert info["predicate_deltas"]["L1::pick"] is True

    def test_l3_delta_fires_once(self):
        steps = [_make_info(l3_place=True), _make_info(l3_place=True)]
        wrapped = _make_wrapped(steps)
        wrapped.reset()
        _obs, _r, _done, info0 = wrapped.step(None)
        _obs, _r, _done, info1 = wrapped.step(None)
        assert info0["predicate_deltas"]["L3::place"] is True
        assert info1["predicate_deltas"]["L3::place"] is False
