"""
Integration tests for the sim-layer raw-predicate info contract (Track A).

Asserts that ``env.step()`` emits:
  - ``info["raw_predicates"]``: dict[str, bool] with keys L1::*, L2::*, L3::*, L4
  - ``info["l4_satisfied"]``: bool
  - NO legacy keys (subtask_history, subtask_rewards, subtask_reward_increment,
    subtask_reward_delta, subtask_info)

Skipped automatically when robosuite / MuJoCo is not importable.

Run with:
    conda run -n libero_data python -m pytest tests/test_sim_raw_predicates_integration.py -v
"""

import pathlib

import numpy as np
import pytest

# Skip the entire module if robosuite (MuJoCo) is not available.
robosuite = pytest.importorskip("robosuite")

import torch  # noqa: E402  (after importorskip so MuJoCo check fires first)

from libero.libero.envs.env_wrapper import ControlEnv  # noqa: E402

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).parent.parent
_BDDL_DIR = _REPO_ROOT / "libero" / "libero" / "bddl_files" / "libero_plus_train_subtasks"
_INIT_DIR = _REPO_ROOT / "libero" / "libero" / "init_files" / "libero_plus_train_subtasks"

# Task A — pure pick-and-place (both subtasks use In → L2 eligible for both)
_TASK_A_STEM = (
    "LIVING_ROOM_TABLETOP_BASKET_SCENE1030_put_the_cream_cheese_in_the_basket"
    "_and_put_the_alphabet_soup_in_the_basket"
)
_TASK_A_BDDL = str(_BDDL_DIR / f"{_TASK_A_STEM}.bddl")
_TASK_A_INIT = str(_INIT_DIR / f"{_TASK_A_STEM}.pruned_init")
_TASK_A_SUBTASKS = ["place_alphabet_soup_1", "place_cream_cheese_1"]

# Task B — mixed: Turnon + On (L2 must always be False for turnon_stove)
_TASK_B_STEM = "FLOOR_10_turn_on_the_stove_and_put_the_left_moka_pot_on_it"
_TASK_B_BDDL = str(_BDDL_DIR / f"{_TASK_B_STEM}.bddl")
_TASK_B_INIT = str(_INIT_DIR / f"{_TASK_B_STEM}.pruned_init")
_TASK_B_SUBTASKS = ["turnon_stove", "place_moka_pot_2"]

# Number of no-op steps taken per test case
_N_STEPS = 5

# Legacy info keys that must NOT appear after Track A refactor
_LEGACY_KEYS = frozenset(
    [
        "subtask_history",
        "subtask_rewards",
        "subtask_reward_increment",
        "subtask_reward_delta",
        "subtask_info",
    ]
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_env(bddl_path: str, init_path: str) -> ControlEnv:
    """Create a headless ControlEnv and load the first initial state."""
    env = ControlEnv(
        bddl_file_name=bddl_path,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )
    env.reset()
    init_states = torch.load(init_path, weights_only=False)
    # .pruned_init files may be saved as numpy arrays or torch tensors
    arr = init_states[0]
    env.set_init_state(arr.numpy() if isinstance(arr, torch.Tensor) else arr)
    return env


@pytest.fixture(scope="module")
def env_task_a():
    env = _make_env(_TASK_A_BDDL, _TASK_A_INIT)
    yield env
    env.close()


@pytest.fixture(scope="module")
def env_task_b():
    env = _make_env(_TASK_B_BDDL, _TASK_B_INIT)
    yield env
    env.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_steps(env: ControlEnv, n: int = _N_STEPS):
    """Run n no-op steps and return the list of info dicts."""
    infos = []
    for _ in range(n):
        _obs, _reward, _done, info = env.step(np.zeros(7))
        infos.append(info)
    return infos


def _expected_raw_predicate_keys(subtask_names):
    """Build the expected full key set for raw_predicates."""
    keys = set()
    for s in subtask_names:
        keys.add(f"L1::{s}")
        keys.add(f"L2::{s}")
        keys.add(f"L3::{s}")
    keys.add("L4")
    return keys


# ---------------------------------------------------------------------------
# TC-1: Info contract shape
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestInfoContractShape:
    """TC-1 — presence/absence of top-level info keys and value types."""

    def _assert_step_contract(self, info: dict, step_idx: int, task_label: str):
        ctx = f"[{task_label}] step {step_idx}"

        # Required keys present
        assert "raw_predicates" in info, f"{ctx}: missing 'raw_predicates'"
        assert "l4_satisfied" in info, f"{ctx}: missing 'l4_satisfied'"

        # Legacy keys must be absent
        for key in _LEGACY_KEYS:
            assert key not in info, f"{ctx}: legacy key '{key}' must not be present"

        raw = info["raw_predicates"]

        # All predicate values must be plain Python bool
        for k, v in raw.items():
            assert isinstance(v, bool) and type(v) is bool, (
                f"{ctx}: raw_predicates['{k}'] has type {type(v).__name__}, expected bool"
            )

        # L4 consistency
        assert raw["L4"] == info["l4_satisfied"], (
            f"{ctx}: raw_predicates['L4'] ({raw['L4']}) != "
            f"l4_satisfied ({info['l4_satisfied']})"
        )

        # L2 implies L1: grasping an object means it is near the EEF.
        for k, v in raw.items():
            if k.startswith("L2::") and v:
                l1_key = "L1::" + k[4:]
                assert raw.get(l1_key) is True, (
                    f"{ctx}: L2 implies L1 violated — {k}=True but {l1_key}={raw.get(l1_key)}"
                )
        # Note: the must-release gate (L3 False while gripper contacts primary) is
        # not assertable in zero-action steps — verified via Tier 3/4 manual testing.

    def test_contract_shape_task_a(self, env_task_a):
        infos = _collect_steps(env_task_a)
        for i, info in enumerate(infos):
            self._assert_step_contract(info, i, "TaskA")

    def test_contract_shape_task_b(self, env_task_b):
        infos = _collect_steps(env_task_b)
        for i, info in enumerate(infos):
            self._assert_step_contract(info, i, "TaskB")


# ---------------------------------------------------------------------------
# TC-2: Key-set correctness
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRawPredicateKeySet:
    """TC-2 — exact set of keys in raw_predicates matches expected schema."""

    def _assert_keyset(self, infos, subtask_names, task_label):
        expected = _expected_raw_predicate_keys(subtask_names)
        for i, info in enumerate(infos):
            actual = set(info["raw_predicates"].keys())
            assert actual == expected, (
                f"[{task_label}] step {i}: raw_predicates key set mismatch.\n"
                f"  Expected : {sorted(expected)}\n"
                f"  Got      : {sorted(actual)}\n"
                f"  Missing  : {sorted(expected - actual)}\n"
                f"  Extra    : {sorted(actual - expected)}"
            )

    def test_keyset_task_a(self, env_task_a):
        infos = _collect_steps(env_task_a)
        self._assert_keyset(infos, _TASK_A_SUBTASKS, "TaskA")

    def test_keyset_task_b(self, env_task_b):
        infos = _collect_steps(env_task_b)
        self._assert_keyset(infos, _TASK_B_SUBTASKS, "TaskB")


# ---------------------------------------------------------------------------
# TC-3: L2 gating for Turnon predicates
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestL2GatingForTurnon:
    """TC-3 — L2::turnon_stove must always be False (Turnon is not a pick predicate)."""

    def test_l2_turnon_always_false(self, env_task_b):
        infos = _collect_steps(env_task_b)
        for i, info in enumerate(infos):
            raw = info["raw_predicates"]
            assert "L2::turnon_stove" in raw, (
                f"[TaskB] step {i}: 'L2::turnon_stove' key missing from raw_predicates"
            )
            assert raw["L2::turnon_stove"] is False, (
                f"[TaskB] step {i}: L2::turnon_stove should always be False "
                f"for a Turnon predicate, got {raw['L2::turnon_stove']}"
            )


# ---------------------------------------------------------------------------
# TC-5: L4 consistency (explicit named test)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestL4Consistency:
    """TC-5 — raw_predicates['L4'] must equal info['l4_satisfied'] on every step."""

    def _assert_l4_consistent(self, infos, task_label):
        for i, info in enumerate(infos):
            l4_pred = info["raw_predicates"]["L4"]
            l4_flag = info["l4_satisfied"]
            assert l4_pred == l4_flag, (
                f"[{task_label}] step {i}: "
                f"raw_predicates['L4']={l4_pred} != l4_satisfied={l4_flag}"
            )

    def test_l4_consistency_task_a(self, env_task_a):
        infos = _collect_steps(env_task_a)
        self._assert_l4_consistent(infos, "TaskA")

    def test_l4_consistency_task_b(self, env_task_b):
        infos = _collect_steps(env_task_b)
        self._assert_l4_consistent(infos, "TaskB")
