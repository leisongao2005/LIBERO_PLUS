# Predicate-Level (Hierarchical) Evaluation

> **Last updated:** Track E documentation refactor (docs/sim-wrapper-refactor).
> **Canonical design:** [`libero/libero/envs/wrappers/DESIGN.md`](./wrappers/DESIGN.md).
> **Canonical code:** `libero/libero/bddlsim_interface.py` and
> `libero/libero/hierarchical_reward_wrapper.py`.

This document describes how LIBERO-Plus evaluates manipulation progress at four
semantic levels: **localization (L1)**, **grasping (L2)**, **subtask completion (L3)**,
and **terminal goal satisfaction (L4)**. It also describes which layer owns each level.

---

## Two-Layer Design Summary

The refactored architecture splits concerns across two layers:

| Layer | Responsibility |
|-------|----------------|
| **Simulator** (`BDDLBaseDomain`) | Evaluates raw predicates each step. Emits `raw_predicates: dict[str, bool]` and `l4_satisfied: bool`. Fully stateless — no one-shot latching. |
| **Wrapper** (`HierarchicalRewardWrapper`) | Owns L3 one-shot history, L1/L2 transient delta detection, shaped reward computation, and NL status string. |

The sim never accumulates episode state. The wrapper accumulates `_l3_history`,
`_l1_prev`, `_l2_prev`, and computes shaped reward via `RewardConfig`.

---

## Level Definitions

| Level | Name | Owner | Firing | Semantics |
|-------|------|-------|--------|-----------|
| **L1** | Localization | Wrapper (auto-derived) | Transient: re-fires every step it is true | Primary object is near the end-effector. |
| **L2** | Grasp | Wrapper (auto-derived) | Transient: re-fires every step it is true | Gripper grasping primary object. Only evaluated for `on`/`in` subtasks. |
| **L3** | Subtask | Sim (predicate) + Wrapper (latch) | One-shot: credited once per episode | BDDL-declared subtask predicate is true this step; wrapper latches in `_l3_history`. |
| **L4** | Terminal | Sim (predicate) + Wrapper (latch) | One-shot: credited once per episode | Full `(:goal ...)` satisfaction. `raw_predicates["L4"]` must equal `l4_satisfied`. |

---

## L1: Localization (Auto-Derived by Wrapper)

L1 is **not declared in BDDL** for each subtask. The wrapper derives it automatically
from the primary object of each L3 subtask using `LocalizedNearEEF` semantics.

**Primary object rule:** The primary object for subtask `S` is the first non-numeric
token in that subtask's `predicate_args` list. This is computed by
`_first_obj_state_from_goal_tokens` in `bddl_base_domain.py`.

**Predicate semantics (evaluated by sim, reported as `L1::<S>`):**

1. Euclidean distance from the object's body position to the end-effector is strictly
   less than `dist_threshold` (default `0.12`).
2. EEF linear speed is strictly less than `vel_threshold` (default `0.25`) when the
   threshold is set.
3. No other object in `objects_dict` is strictly closer to the EEF than the target
   object (closest-object tie-breaking with a small epsilon).

**Key:** `raw_predicates["L1::<S>"]` — instantaneous boolean, re-evaluated every step.

**Reward:** Wrapper computes delta vs `_l1_prev[S]` and fires `weight["L1"]` on
positive transitions (or all positive steps, depending on delta mode in Track C).

---

## L2: Grasp (Auto-Derived by Wrapper)

L2 is **not declared in BDDL** for each subtask. The wrapper derives it automatically,
but only for subtasks whose predicate is `on` or `in` (pick-and-place operations).

**L2 is skipped for:** `turnon`, `turnoff`, `open`, `close`, and any other non-placement
predicate. The corresponding `L2::<S>` key in `raw_predicates` will be present but
is only used for shaping on pick-and-place subtasks.

**Predicate semantics (evaluated by sim, reported as `L2::<S>`):** Corresponds to
`DefaultGraspPredicate`:

1. `env.check_contact(robot.gripper, obj)` is true.
2. Every external geom touching `obj.contact_geoms` appears in
   `gripper.important_geoms`. This means table/shelf contacts cause failure — L2
   requires an isolated (lifted or free-floating) grasp.

**Key:** `raw_predicates["L2::<S>"]` — instantaneous boolean, re-evaluated every step.

**Reward:** Wrapper computes delta vs `_l2_prev[S]` and fires `weight["L2"]` on
positive transitions for `on`/`in` subtasks only.

> **Deferred:** A no-contact gate (`_subtask_candidate_valid`) that additionally
> required the object not be in contact with the gripper during placement credit is
> deferred. A TODO in the wrapper's delta logic marks where this will land.

---

## L3: Subtask Completion

L3 represents the completion of a named manipulation subtask declared in BDDL with
`(:subtask_rewards ...)` or synthesized from `(:goal ...)` conjuncts.

**Sim responsibility:** Evaluates the BDDL predicate for each subtask slot on every
step. Reports `raw_predicates["L3::<S>"]` as an instantaneous boolean — no latching,
no confirmation counting.

**Wrapper responsibility:** Maintains `_l3_history: dict[str, bool]`, which starts all
`False` at episode reset. When `raw_predicates["L3::<S>"]` is `True` and
`_l3_history[S]` is `False`, the wrapper fires the L3 delta reward and sets
`_l3_history[S] = True`. Subsequent steps where L3 is true do not re-fire.

**Behaviors dropped from old design:**

- **Multi-step confirmation** (`_subtask_confirmation_counts`): L3 now credits on the
  first step the predicate is true. No N-step confirmation window.
- **`:after` ordering**: `:after` prerequisite fields are no longer enforced. Subtasks
  are credited independently.
- **Coarse mode vs fine-grained mode**: The distinction is no longer meaningful; all
  goal conjuncts are evaluated as independent subtask slots.

**Key:** `raw_predicates["L3::<S>"]` — instantaneous; `_l3_history[S]` in wrapper —
persistent one-shot for the episode.

---

## L4: Terminal Goal

L4 represents full `(:goal ...)` satisfaction.

**Sim responsibility:** Evaluates all goal conjuncts. Sets `l4_satisfied = True` and
`raw_predicates["L4"] = True` when all are satisfied. Both must agree (validated by
`validate_sim_step_info`).

**Wrapper responsibility:** Latches L4 once `l4_satisfied` is first observed as `True`.
Fires `weight["L4"]` delta reward on the first step of satisfaction, then never again.

The sim emits **sparse reward** `1.0` on terminal success; the wrapper adds the L4
shaped reward component on top of this (or replaces it, per `RewardConfig` design).

---

## raw_predicates Key Naming

For a task with subtasks `["pick_mug", "place_on_rack"]`:

```python
{
    "L1::pick_mug":     True,   # mug is near EEF this step
    "L2::pick_mug":     True,   # gripper holding mug this step
    "L3::pick_mug":     False,  # mug not yet on rack
    "L1::place_on_rack": False,
    "L2::place_on_rack": False,
    "L3::place_on_rack": False,
    "L4":               False,  # goal not yet satisfied
}
```

Keys are built in BDDL declaration order: L1/L2/L3 triple for each subtask in order,
then `L4`.

---

## Predicate Implementations

The underlying predicate classes (used by the sim) are:

| BDDL token | Python class | Notes |
|------------|-------------|-------|
| *(L1, auto-derived)* | `LocalizedNearEEF` | Parametric; see `predicates/__init__.py`. |
| *(L2, auto-derived)* | `DefaultGraspPredicate` | Fixed instance in `VALIDATE_PREDICATE_FN_DICT`. |
| `horizontalgrasppredicate` | `HorizontalGraspPredicate` | Alternative for mug-like picks; adds tool-z / object-z alignment check. |
| `on`, `in`, `open`, `close`, `turnon`, `turnoff` | Various | Declared in BDDL `:subtask_rewards`; evaluated as L3. |
| `neareef` | `NearEEF` | Parametric distance-only (weaker than L1). |
| `grasp` | `Grasp` | Weak contact check (weaker than L2). |

Full registration: `VALIDATE_PREDICATE_FN_DICT` and `PARAMETRIC_PREDICATE_CLS` in
`libero/libero/envs/predicates/__init__.py`.

---

## Related Files

| File | Role |
|------|------|
| `libero/libero/envs/wrappers/DESIGN.md` | **Authoritative** two-layer design document. |
| `libero/libero/bddlsim_interface.py` | `BDDLSimStepInfo`, `validate_sim_step_info`, `FakeBDDLEnv` (MuJoCo-free). |
| `libero/libero/hierarchical_reward_wrapper.py` | `RewardConfig`, `HierarchicalRewardWrapper` (MuJoCo-free). |
| `libero/libero/envs/predicates/base_predicates.py` | `LocalizedNearEEF`, `DefaultGraspPredicate`, `HorizontalGraspPredicate`. |
| `libero/libero/envs/predicates/__init__.py` | BDDL token → predicate class/instance registration. |
| `libero/libero/envs/bddl_base_domain.py` | Sim implementation; `_first_obj_state_from_goal_tokens`. |
| `libero/libero/envs/SUBTASK_CONTROL_FLOW.md` | End-to-end pipeline data flow. |
| `libero/libero/envs/wrappers/SIM_STEP_INFO.md` | `BDDLSimStepInfo` contract reference. |
