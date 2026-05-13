# Subtask-Related `info` Fields — Historical Reference

> **DEPRECATED:** This document describes the **pre-refactor** monolithic `BDDLBaseDomain`
> `info` contract. It is retained for historical reference only.
>
> **For the current contract:** See
> [`libero/libero/envs/wrappers/SIM_STEP_INFO.md`](./wrappers/SIM_STEP_INFO.md) for the
> new `BDDLSimStepInfo` schema, and
> [`libero/libero/envs/wrappers/DESIGN.md`](./wrappers/DESIGN.md) for the authoritative
> two-layer design.
>
> **Last updated:** Track E documentation refactor (docs/sim-wrapper-refactor).

---

## What Changed

The LIBERO-Plus refactor moved from a monolithic `BDDLBaseDomain` that computed shaped
rewards internally to a two-layer design:

- **Sim layer** emits only `raw_predicates: dict[str, bool]` and `l4_satisfied: bool`.
  Sparse reward only. No per-episode state in the sim.
- **Wrapper layer** (`HierarchicalRewardWrapper`) owns all shaped reward computation,
  L3 one-shot history, and NL status string.

The following old `info` keys are **no longer emitted** by the sim:

| Old key | Replacement |
|---------|-------------|
| `subtask_rewards` (cumulative bool dict) | Wrapper's `_l3_history` |
| `subtask_reward_increment` (new-credit dict) | Wrapper's `predicate_deltas` (Track C) |
| `subtask_reward_delta` (scalar sum) | Wrapper's `predicate_deltas` (Track C) |
| `subtask_info` (dry-run bool dict) | No longer needed; wrapper re-uses `raw_predicates` |

The following constructor flags that gated `info` population are **removed**:

| Old flag | Status |
|----------|--------|
| `subtask_reward=True` | Removed. Info is always emitted. |
| `track_subtask_info=True` | Removed. No dry-run evaluation path. |
| `subtask_reward_scale` | Removed. Use `RewardConfig.weights`. |
| `subtask_confirmation_steps` | Removed. No confirmation window. |

---

## New `info` Contract (Current)

Every `step()` from the sim unconditionally returns:

```python
info = {
    "raw_predicates": dict[str, bool],  # L1::*, L2::*, L3::*, L4 keys
    "l4_satisfied":   bool,             # must equal raw_predicates["L4"]
    # transitional (will be removed when Track A is complete):
    "subtask_history": dict[str, bool], # L3 one-shot; being moved to wrapper
}
```

See [`libero/libero/envs/wrappers/SIM_STEP_INFO.md`](./wrappers/SIM_STEP_INFO.md) for
the full field reference and key naming rules.

After passing through `HierarchicalRewardWrapper`, the `info` is the sim's `info`
passed through unchanged (the wrapper may add `predicate_deltas` in Track C).

---

## Legacy Contract (Pre-Refactor, For Historical Reference)

The old `BDDLBaseDomain` added these keys to `info` when flags were set:

### When `subtask_reward=True`

| Key | Type | Semantics |
|-----|------|-----------|
| `subtask_rewards` | `dict[str, bool]` | Cumulative one-shot booleans for each subtask this episode. Keys were BDDL `:subtask` names (fine mode) or `"_".join(pred)` strings (coarse mode). |
| `subtask_reward_increment` | `dict[str, float]` | Subtasks **newly credited on this step** → their scaled weight. Empty `{}` when nothing new. |
| `subtask_reward_delta` | `float` | `sum(subtask_reward_increment.values())`. |

### When `track_subtask_info=True`

| Key | Type | Semantics |
|-----|------|-----------|
| `subtask_info` | `dict[str, bool]` | Result of `_evaluate_subtask_rewards(dry_run=True)`. Instantaneous check without mutating episode state. |

### Why `subtask_reward` (dict) was renamed

Some training stacks merged constructor kwargs into `info`, causing `info["subtask_reward"]`
to be a boolean (`True`/`False`) instead of the expected dict. The fix renamed the
dict to `subtask_reward_increment` and added an explicit float `subtask_reward_delta`.

---

## Migration Guide (from old to new)

If you have code consuming the old `info` keys:

| Old code | New code |
|----------|----------|
| `info["subtask_rewards"]["my_subtask"]` | `wrapper._l3_history["my_subtask"]` |
| `info["subtask_reward_increment"]` | `info.get("predicate_deltas", {})` (Track C) |
| `info["subtask_reward_delta"]` | `sum(info.get("predicate_deltas", {}).values())` (Track C) |
| `info["subtask_info"]` | `info["raw_predicates"]` — use `L3::<S>` keys |

For the shaped reward value, read the `reward` return value from
`HierarchicalRewardWrapper.step()` directly — it is the shaped reward.

---

## Example: Old vs New `info`

**Old `info` (pre-refactor, `subtask_reward=True`, one new credit this step):**

```python
{
    "subtask_rewards": {
        "place_moka_pot_1": True,
        "turnon_stove":     False,
    },
    "subtask_reward_increment": {"place_moka_pot_1": 0.5},
    "subtask_reward_delta": 0.5,
}
```

**New `info` (post-refactor, from sim passthrough):**

```python
{
    "raw_predicates": {
        "L1::place_moka_pot_1": False,
        "L2::place_moka_pot_1": False,
        "L3::place_moka_pot_1": True,
        "L1::turnon_stove":     False,
        "L2::turnon_stove":     False,
        "L3::turnon_stove":     False,
        "L4":                   False,
    },
    "l4_satisfied": False,
    # transitional:
    "subtask_history": {"place_moka_pot_1": True, "turnon_stove": False},
}
```

Shaped reward is the return value of `HierarchicalRewardWrapper.step()`, not an
`info` key. Progress state is in `wrapper._l3_history`.

---

## Reference Implementation

- **Current sim:** `libero/libero/envs/bddl_base_domain.py` — `step()`, `reward()`.
- **Current wrapper:** `libero/libero/hierarchical_reward_wrapper.py`.
- **Contract validation:** `libero/libero/bddlsim_interface.validate_sim_step_info`.
