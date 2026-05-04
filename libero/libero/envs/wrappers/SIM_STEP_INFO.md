# Phase 0 — BDDL sim `step` → `info` contract

This is a **navigation aid** for the frozen schema between a Markovian BDDL simulator and `HierarchicalRewardWrapper` (and related tests). Authoritative types and validation live in Python; this page distills what humans need when browsing the repo.

**Ground truth for the full refactor** (sim changes, wrapper tracks, parallelization): [`.cursor/plans/hierarchical_sim_wrapper_refactor_599aa945.plan.md`](../../../../.cursor/plans/hierarchical_sim_wrapper_refactor_599aa945.plan.md) — see **Phase 0 — Interface freeze**.

Legacy monolithic `info` docs under `libero/libero/envs/` (e.g. `SUBTASK_STEP_INFO.md`) describe the **old** contract, not this schema.

---

## Purpose

Every real or stubbed BDDL sim `step` must return an `info` dict that always includes the same three keys, with no feature flags. Track A (`BDDLBaseDomain`) will emit this shape unconditionally once implemented; wrappers and tests should assume it is present on every step.

---

## Where to import from

| Use case | Import |
|----------|--------|
| **Preferred (MuJoCo-free, tests)** | `from libero.libero import bddlsim_interface` or `from libero.libero.bddlsim_interface import ...` |
| **Wrapper package re-export** | `from libero.libero.envs.wrappers import ...` (same symbols; may pull more of `libero.libero.envs` when that graph loads) |
| **Planned thin alias** | `libero.libero.envs.wrappers._interface` — re-exports only; still defined in `bddlsim_interface.py` |

---

## `BDDLSimStepInfo` — required `info` keys

| Key | Type | Meaning |
|-----|------|--------|
| `raw_predicates` | `dict[str, bool]` | Instantaneous predicate truth for this physics step. **No** one-shot latch inside the sim. |
| `subtask_history` | `dict[str, bool]` | L3 one-shot bitmask for the episode: keys are **L3 subtask names** in BDDL declaration order; a value becomes `True` the first time that L3 is credited and stays `True`. |
| `l4_satisfied` | `bool` | Full `(:goal ...)` satisfied on this step (before any wrapper latch). Must agree with `raw_predicates["L4"]`. |

Sim `reward` stays **sparse** (e.g. terminal only); shaping is wrapper-owned.

---

## `raw_predicates` key naming

For each L3 subtask name `S` (exact string from BDDL `:subtask` / goal ordering — same strings as `subtask_history` keys):

- `L1::<S>` — auto-derived localization for that slot’s **primary object**
- `L2::<S>` — auto-derived grasp for that object
- `L3::<S>` — the BDDL L3 predicate, instantaneous

Terminal goal: **`L4`** (exact key, no prefix) — mirrors full-goal success; must match `l4_satisfied`.

**Ordering:** For stable logs / iteration, build triples `(L1, L2, L3)` per subtask in **BDDL declaration order**, then add `L4`. Consumers should not rely on dict order for correctness, but the sim should follow this order when practical.

**Primary object (for L1/L2):** first non-numeric token in that subtask’s `predicate_args` (see plan + `_first_obj_state_from_goal_tokens` in `bddl_base_domain.py`).

---

## Validation, examples, and tests

- **`validate_sim_step_info(info, subtask_names_ordered=..., strict_raw_keys=True)`** — raises `ValueError` if keys, types, `subtask_history` key order, `L4` presence, or `l4_satisfied` ↔ `raw_predicates["L4"]` agreement fail. Use at boundaries (sim emit, tests).
- **`EXAMPLE_SIM_STEP_INFO`** — small hand-written dict illustrating naming and history.
- **`FakeBDDLEnv`** — minimal `gym.Env` that returns contract-shaped `info` from a callable per step index; calls `validate_sim_step_info` inside `step` so wrapper / delta tests can run **without MuJoCo**.

---

## See also

- Module docstring and helpers: `libero/libero/bddlsim_interface.py` (`make_empty_sim_step_info`, `empty_raw_predicates`, `raw_predicate_key_l{1,2,3}`, `RAW_PREDICATE_KEY_L4`).
