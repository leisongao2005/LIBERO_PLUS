# Subtask-related `info` fields from `BDDLBaseDomain.step()`

> **Note:** For the in-flight hierarchical sim-wrapper refactor, authoritative behavior is `.cursor/plans/hierarchical_sim_wrapper_refactor_599aa945.plan.md`; this file remains **legacy** documentation for the current monolithic `info` contract.

For the end-to-end subtask pipeline (BDDL parsing, evaluation order, reset, and `step` / `reward` ordering), see [SUBTASK_CONTROL_FLOW.md](./SUBTASK_CONTROL_FLOW.md).

This document describes the **exact auxiliary data** LIBERO’s `BDDLBaseDomain` adds to the `info` dict returned by `step()`. It applies to environments built from `TASK_MAPPING` problem classes (e.g. `Libero_*_Manipulation`) wrapped by `ControlEnv` / `OffScreenRenderEnv`, which forward `step()` to the underlying robosuite env.

## Top-level `step()` return value

Each call returns a **4-tuple**:

| Position | Name   | Type (typical) | Notes |
|----------|--------|----------------|--------|
| 0        | `obs`  | `OrderedDict` or structure used by robosuite | Unchanged by subtask logic. |
| 1        | `reward` | `float` | From `reward()`; includes subtask shaping when `subtask_reward=True` (see `reward()` docstring). |
| 2        | `done` | `bool` | After `super().step()`, LIBERO sets `done = self._check_success()` (full goal), not only horizon. |
| 3        | `info` | `dict` | Starts as `{}` from robosuite’s `_post_action`, then LIBERO **adds** the keys below when the corresponding flags are set. |

No other keys are guaranteed unless your wrappers add them.

---

## Constructor flags that control `info`

These are **not** part of `info`; they are `BDDLBaseDomain.__init__` arguments (often passed through `OffScreenRenderEnv(**kwargs)`):

| Parameter               | Default | Effect on `info` |
|-------------------------|---------|-------------------|
| `subtask_reward`        | `False` | When `True`, adds subtask shaping keys (see below). |
| `subtask_reward_scale`  | `0.5`   | Scales normalized BDDL weights; affects numeric values in `reward` and in increment dicts. |
| `subtask_confirmation_steps` | `10` | Consecutive steps required before `on` / `in` subtasks credit (see `_evaluate_subtask_rewards`). |
| `track_subtask_info`    | `False` | When `True`, adds `subtask_info` every step. |

---

## `info` keys: current interface (after the change)

When **`subtask_reward=True`**, LIBERO sets:

| Key | Type | Semantics |
|-----|------|-----------|
| `subtask_rewards` | `dict[str, bool]` | Per subtask: **cumulative** “ever satisfied this episode” (one-shot), matching the evaluation used inside `reward()` for that step. Keys are BDDL `:subtask` **names** if `(:subtask_rewards ...)` exists; otherwise goal-predicate keys `"_".join(pred)` for each flat goal atom in coarse mode. |
| `subtask_reward_increment` | `dict[str, float]` | **Only subtasks newly credited on this step** (difference in internal `_subtask_ever_satisfied` before vs after the transition). Map: subtask name (or coarse key) → weight credited this step. Empty `{}` when nothing new was satisfied. Weights are already scaled (fine: normalized from BDDL to sum to `subtask_reward_scale`; coarse: `subtask_reward_scale / N` per new goal atom). |
| `subtask_reward_delta` | `float` | `sum(subtask_reward_increment.values())`. Always `0.0` when the increment dict is empty. |

When **`track_subtask_info=True`**, LIBERO also sets:

| Key | Type | Semantics |
|-----|------|-----------|
| `subtask_info` | `dict[str, bool]` | Same shape as `subtask_rewards`, but from `_evaluate_subtask_rewards(dry_run=True)`: **does not** mutate `_subtask_ever_satisfied` or confirmation counters on disk for that call’s side effects beyond what `reward()` already did. Useful for logging / progress without changing credit state. When both flags are true, `subtask_info` is computed **after** the step and can differ slightly from `subtask_rewards` in edge cases involving dry-run confirmation accounting. |

When **`subtask_reward=False`**, LIBERO does **not** add `subtask_rewards`, `subtask_reward_increment`, or `subtask_reward_delta`.

---

## Previous interface (before the change)

Behavior was the same except for the **incremental** payload key name:

| Then | Now |
|------|-----|
| `info["subtask_reward"]` = `dict[str, float]` (newly credited weights this step) | **Removed** from LIBERO to avoid clashing with wrappers that merge `subtask_reward=True` into `info`. |
| *(no scalar sum key)* | `info["subtask_reward_delta"]` = `float` |
| *(same)* | `info["subtask_reward_increment"]` = `dict[str, float]` (same dict as old `subtask_reward`) |

`subtask_rewards` and optional `subtask_info` were already present and are **unchanged** in meaning.

---

## Why `info["subtask_reward"]` was removed

Some training stacks merge **environment constructor kwargs** (e.g. `subtask_reward=True`) into the per-step `info` dict under the same key `subtask_reward`. That produces a **boolean** where code expected a **dict**, breaks parsing, and prevents fallback branches (e.g. on `subtask_info`) from running. Renaming the env-produced dict to `subtask_reward_increment` and exposing an explicit float `subtask_reward_delta` avoids that collision.

---

## Migration for downstream code

1. Replace reads of `info["subtask_reward"]` (dict) with **`info["subtask_reward_increment"]`**.
2. For scalar logging or bonus shaping from **this step only**, use **`info["subtask_reward_delta"]`**.
3. Do not put the constructor flag into `info` as `subtask_reward`; use a different key (e.g. `use_subtask_reward`) if you need to record it.
4. If you must branch on a key, use **`isinstance(info.get("subtask_reward_increment"), dict)`** rather than **`"subtask_reward" in info`**, unless you control merges so the value cannot be a bool.

---

## Example `info` fragments

**`subtask_reward=True`, fine-grained BDDL subtasks, middle of episode, one new subtask this step:**

```python
{
    "subtask_rewards": {
        "place_moka_pot_1": True,
        "place_moka_pot_2": False,
        "turnon_stove": True,
    },
    "subtask_reward_increment": {"place_moka_pot_1": 0.16666666666666666},
    "subtask_reward_delta": 0.16666666666666666,
}
```

**Same situation, no new credit this step:**

```python
{
    "subtask_rewards": { ... },  # same cumulative booleans as appropriate
    "subtask_reward_increment": {},
    "subtask_reward_delta": 0.0,
}
```

**`subtask_reward=False`, `track_subtask_info=True`:**

```python
{
    "subtask_info": {"place_moka_pot_1": False, ...},
}
```

---

## Reference implementation

Source of truth: `libero/libero/envs/bddl_base_domain.py` — `step()`, `reward()`, and `_evaluate_subtask_rewards()`.
