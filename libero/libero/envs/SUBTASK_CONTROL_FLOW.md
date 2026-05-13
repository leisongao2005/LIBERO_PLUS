# Subtask Pipeline and Control Flow

> **Last updated:** Track E documentation refactor (docs/sim-wrapper-refactor).
> **Canonical design:** [`libero/libero/envs/wrappers/DESIGN.md`](./wrappers/DESIGN.md).
> **Canonical code:** `libero/libero/bddlsim_interface.py` and
> `libero/libero/hierarchical_reward_wrapper.py`.

This document describes the end-to-end data flow through the two-layer hierarchical
reward system: BDDL parsing, sim predicate evaluation, and wrapper-side shaping.

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────┐
│  1. BDDL Parsing (at task init)                 │
│     bddl_utils.robosuite_parse_problem          │
│     → subtask names, predicate specs            │
│     → goal_state (full success conditions)      │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│  2. Simulator (BDDLBaseDomain)  [per step]      │
│     Evaluates all L1/L2/L3/L4 predicates        │
│     Emits raw_predicates: dict[str, bool]       │
│     Emits l4_satisfied: bool                    │
│     Returns sparse reward: 0.0 or 1.0           │
└────────────────────┬────────────────────────────┘
                     │  info = {raw_predicates, l4_satisfied}
                     │         (+ transitional subtask_history)
┌────────────────────▼────────────────────────────┐
│  3. HierarchicalRewardWrapper  [per step]       │
│     Validates info (validate_sim_step_info)      │
│     Detects L1/L2 transient deltas              │
│     Detects L3 one-shot deltas (_l3_history)    │
│     Detects L4 one-shot delta                   │
│     Computes shaped reward via RewardConfig      │
│     Appends [Status: ...] to obs["instruction"] │
│     Returns shaped reward to training loop      │
└─────────────────────────────────────────────────┘
```

---

## Stage 1: BDDL Parsing (Task Init)

**Input:** A `.bddl` task file from `libero/libero/bddl_files/libero_plus_{train,eval}_subtasks/`.

**Parser:** `bddl_utils.robosuite_parse_problem` (and `parse_subtask_rewards` for
fine-grained subtasks).

**Output:**
- `parsed_problem["subtask_rewards"]` — list of subtask dicts with name and predicate spec.
- `parsed_problem["goal_state"]` — conjunction of predicates for full success.

**BDDL `:reward` weights** (if present) are **ignored** in the refactored design.
Use `RewardConfig` at the wrapper layer to configure shaping weights.

**BDDL `:after` prerequisites** (if present) are **ignored** in the refactored design.
Subtasks are credited independently when their predicate is instantaneously true.

---

## Stage 2: Simulator — Per-Step Predicate Evaluation

On each `env.step(action)`:

1. **Physics step** — robosuite advances the MuJoCo simulation.
2. **L1/L2/L3 evaluation** — for each subtask `S` in BDDL declaration order:
   - `raw_predicates["L1::<S>"]` — evaluate `LocalizedNearEEF` for `S`'s primary object.
   - `raw_predicates["L2::<S>"]` — evaluate `DefaultGraspPredicate` for `S`'s primary object.
   - `raw_predicates["L3::<S>"]` — evaluate the BDDL-declared predicate for `S`.
3. **L4 evaluation** — evaluate all `(:goal ...)` conjuncts:
   - `raw_predicates["L4"]` — true iff all goal atoms are satisfied.
   - `l4_satisfied` — same boolean (must agree; validated by `validate_sim_step_info`).
4. **Sparse reward** — `1.0` if `l4_satisfied`, else `0.0`.
5. **`done`** — set by `_check_success()` (full goal only).
6. **`info`** — returns `{raw_predicates: {...}, l4_satisfied: bool}` plus the
   transitional `subtask_history` field during Track A migration.

**What the sim does NOT do:**
- Does not accumulate `_subtask_ever_satisfied` or any per-episode state.
- Does not apply confirmation counts or `:after` ordering.
- Does not compute shaped rewards.
- Does not modify `obs["instruction"]`.

---

## Stage 3: Wrapper — Per-Step Delta Detection and Shaping

`HierarchicalRewardWrapper.step(action)` wraps the sim step:

### 3a. Validate Info

```python
validate_sim_step_info(info, subtask_names_ordered=self._resolved_subtask_names)
```

Raises `ValueError` if the contract is violated (wrong keys, type errors,
`L4` / `l4_satisfied` disagreement).

### 3b. Resolve Subtask Names

On the first step after `reset()`, derive `_resolved_subtask_names` from the `L3::*`
keys in `raw_predicates` (in dict insertion order = BDDL declaration order) if not
passed explicitly at construction.

### 3c. Detect L1 Transient Deltas

For each subtask `S`:
- `delta_L1[S] = raw_predicates["L1::<S>"] - _l1_prev[S]`
- Update `_l1_prev[S] = raw_predicates["L1::<S>"]`
- Apply `weight["L1"]` when `delta_L1[S] > 0` (and `weight["L1"]` * penalty when < 0,
  but `transient_negative_penalty` is locked to `0.0`).

### 3d. Detect L2 Transient Deltas

For each subtask `S` whose predicate is `on` or `in` only:
- `delta_L2[S] = raw_predicates["L2::<S>"] - _l2_prev[S]`
- Update `_l2_prev[S] = raw_predicates["L2::<S>"]`
- Apply `weight["L2"]` on positive transitions.

> **TODO (deferred):** No-contact gate — skip L2 credit if the object is still in
> contact with the gripper during a placement check.

### 3e. Detect L3 One-Shot Deltas

For each subtask `S`:
- If `raw_predicates["L3::<S>"]` is `True` and `_l3_history[S]` is `False`:
  - Fire `weight["L3"]` (or `per_subtask_overrides[S]["L3"]` if set).
  - Set `_l3_history[S] = True` (latch forever this episode).

### 3f. Detect L4 One-Shot Delta

If `l4_satisfied` is `True` and L4 has not yet fired this episode:
- Fire `weight["L4"]`.
- Latch L4 (does not re-fire).

### 3g. Compute Shaped Reward

```
shaped_reward = Σ fired_deltas × weights
```

The `RewardConfig` holds the default per-level weights and optional per-subtask
overrides. No weight normalization is applied.

### 3h. Append NL Status String (Track D)

```
obs["instruction"] += " [Status: <subtask_1> done, <subtask_2> pending, ...]"
```

This is appended for the critic's benefit. The actor must mask from `"[Status:"`.

### 3i. Return

```python
return obs, shaped_reward, done, info  # info is the passthrough from sim
```

---

## Episode Reset

`HierarchicalRewardWrapper.reset()` calls `_clear_episode_state()`:

- Clears `_l1_prev` and `_l2_prev` (empty dicts).
- Clears `_l3_history` (all `False`).
- Clears L4 latch.
- Resets `_resolved_subtask_names` if not explicitly provided at construction.

The sim's `reset()` is called by `gym.Wrapper.reset()` chain. The sim itself holds no
per-episode state to clear.

---

## Reward Summary Table

| Signal | Where computed | When fires |
|--------|----------------|------------|
| L1 shaped | Wrapper | Every step primary object is near EEF (transient positive) |
| L2 shaped | Wrapper | Every step gripper grasps primary obj for on/in subtasks (transient positive) |
| L3 shaped | Wrapper | First step each subtask predicate is true (one-shot) |
| L4 shaped | Wrapper | First step full goal is satisfied (one-shot) |
| Sparse | Sim | First step full goal is satisfied (and every subsequent step) |

---

## What Changed from the Old Design

| Old behavior | New behavior |
|--------------|--------------|
| `_subtask_confirmation_counts` — N-step window for `on`/`in` | Removed. Credits on first true step. |
| `:after` prerequisite enforcement | Removed. Subtasks credited independently. |
| `_subtask_ever_satisfied` in sim | Removed from sim; wrapper owns `_l3_history`. |
| `_normalize_subtask_weights` + BDDL `:reward` | Removed. Use `RewardConfig`. |
| `subtask_reward`, `track_subtask_info` constructor flags | Removed. Info always emitted. |
| `subtask_reward_increment`, `subtask_reward_delta` info keys | Replaced by wrapper's `predicate_deltas` dict (Track C). |
| `subtask_rewards` cumulative dict | Replaced by wrapper's `_l3_history` state. |
| Dry-run `_evaluate_subtask_rewards` | Removed. Wrapper uses the same `raw_predicates`. |

---

## Related Files

| File | Role |
|------|------|
| `libero/libero/envs/wrappers/DESIGN.md` | **Authoritative** two-layer design document. |
| `libero/libero/bddlsim_interface.py` | `BDDLSimStepInfo` contract, validation, `FakeBDDLEnv`. |
| `libero/libero/hierarchical_reward_wrapper.py` | `HierarchicalRewardWrapper`, `RewardConfig`. |
| `libero/libero/envs/bddl_base_domain.py` | Sim: `step`, `reward`, `_check_success`, `_first_obj_state_from_goal_tokens`. |
| `libero/libero/envs/bddl_utils.py` | `robosuite_parse_problem`, `parse_subtask_rewards`. |
| `libero/libero/envs/PREDICATE_LEVEL_EVALUATION.md` | L1/L2/L3/L4 predicate semantics and owner table. |
| `libero/libero/envs/wrappers/SIM_STEP_INFO.md` | `BDDLSimStepInfo` field reference. |
| `libero/libero/envs/SUBTASK_STEP_INFO.md` | Legacy pre-refactor `info` contract (historical reference only). |
