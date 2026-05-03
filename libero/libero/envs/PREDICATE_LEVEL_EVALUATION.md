# Predicate-level (hierarchical) evaluation

This document describes **LIBERO_PLUS** support for evaluating manipulation progress at multiple semantic levels: **localization (L1)**, **grasping (L2)**, **subtasks (L3)**, and how that relates to **task ordering (L4)**. It complements [SUBTASK_CONTROL_FLOW.md](./SUBTASK_CONTROL_FLOW.md) (orchestration and rewards) and [SUBTASK_STEP_INFO.md](./SUBTASK_STEP_INFO.md) (`info` keys).

---

## Conceptual levels

| Level | Intent | Realization in code |
|-------|--------|---------------------|
| **L1** | Object **localization**: the robot is near the target object, moving slowly enough, and this object is the **closest** among simulated manipulables. | `LocalizedNearEEF` (`localizedneareef` in BDDL). |
| **L2** | Object **grasping**: the gripper contacts the object and **no other body** (table, shelf, other links) is in contact with it; optional **horizontal** tool-axis constraint for mug-like picks. | `DefaultGraspPredicate` / `HorizontalGraspPredicate` (`defaultgrasppredicate`, `horizontalgrasppredicate`). |
| **L3** | **Subtask** progress: named steps with optional **`:after`** ordering, weights, and (for `on` / `in`) confirmation delays. | `(:subtask_rewards ...)` in BDDL + `_evaluate_subtask_rewards` in `BDDLBaseDomain`. |
| **L4** | **Out-of-order execution** (distinct scoring or detection of trajectories that violate intended stage order). | **Not implemented** as a dedicated predicate or metric; see [L4 status](#l4-out-of-order-execution) below. |

**Related unary/binary helpers** (not assigned to L1–L4 above) include `NearEEF`, `Near`, `Grasp` (loose contact), and `ExactIn` (stricter containment)—see `predicates/base_predicates.py` and `predicates/__init__.py`.

---

## L1: `LocalizedNearEEF`

**Source:** `libero/libero/envs/predicates/base_predicates.py` — class `LocalizedNearEEF`.

**BDDL:** Parametric predicate `localizedneareef` (see `PARAMETRIC_PREDICATE_CLS` in `predicates/__init__.py`). Numeric tokens after object name(s) configure thresholds:

- **0 numbers:** defaults (`dist_threshold=0.12`, `vel_threshold=0.25`).
- **1 number:** `dist_threshold` only; velocity check uses default.
- **2+ numbers:** `dist_threshold`, `vel_threshold` (additional numerics are not passed through to `robot_idx`; the active arm index stays the default `0` unless changed in Python).

**Semantics (all must hold):**

1. Euclidean distance from the object’s body pose (via `ObjectState.get_geom_state()["pos"]`) to the end-effector is **strictly less than** `dist_threshold`.
2. If `vel_threshold` is not `None`, EEF linear speed (via `_eef_linear_speed`, preferring robosuite’s hand velocity) must be **strictly less than** `vel_threshold`.
3. For every **other** name in `env.objects_dict`, that object’s body position must **not** be strictly closer to the EEF than the current object (comparison uses a small epsilon on the competing distance).

**Caveats:**

- **Distance ties:** Another object at the **same** distance as the target does not fail the “closest” test (strict inequality). Two objects can both satisfy L1 in the same step if equidistant.
- **Candidate set:** Only `objects_dict` entries are considered competitors; fixtures are not compared.
- **Subtask gating:** When used inside `(:subtask_rewards ...)`, L1 predicates are **not** subject to the multi-step `on`/`in` confirmation counter; they credit **immediately** when true and `_subtask_candidate_valid` passes (which for non-placement predicates is just the predicate value).

---

## L2: `DefaultGraspPredicate` and `HorizontalGraspPredicate`

**Source:** `base_predicates.py` — `DefaultGraspPredicate`, `HorizontalGraspPredicate`.

**BDDL:** Non-parametric entries in `VALIDATE_PREDICATE_FN_DICT`:

- `defaultgrasppredicate` — unary (one object token).
- `horizontalgrasppredicate` — unary; adds a tool-z vs object-up alignment check after the default grasp passes.

**`DefaultGraspPredicate` semantics:**

1. `env.check_contact(robot.gripper, obj)` is true.
2. `env.get_contacts(obj)` (robosuite `MujocoEnv.get_contacts`) returns the set of **external** geom names touching the object’s `contact_geoms`. **Every** such geom name must appear in the flattened `gripper.important_geoms` map. Thus table, shelf, or non-gripper robot links still touching the object cause failure—appropriate for “lifted or isolated” grasps.

**`HorizontalGraspPredicate`:** Same as default, then `|dot(z_tool, z_obj)| <= max_abs_dot` where `z_tool` is from the gripper’s `ee_z` site when available, and `z_obj` is the object body’s world +z axis. Tuned for **upright mug-like** objects; other categories may need different axes.

**Caveats:**

- If `important_geoms` is empty, the “gripper-only contacts” check returns false.
- Geom names from MuJoCo must match those listed for the gripper; version or model differences can cause false negatives.
- `max_abs_dot` for horizontal grasp is **fixed** in the registered instance; tuning from BDDL would require extending `instantiate_predicate` / BDDL parsing.

---

## L3: Subtasks (composition with L1/L2)

L3 is **not** a single predicate class. It is the **fine-grained subtask list** in BDDL:

```text
(:subtask_rewards
  (:subtask name_1 ... :predicate (...) ... :after (...))
  ...
)
```

Parsed by `parse_subtask_rewards` in `bddl_utils.py` and evaluated in `BDDLBaseDomain._evaluate_subtask_rewards`.

**Using L1/L2 inside subtasks:** The `:predicate` field may reference `localizedneareef`, `defaultgrasppredicate`, `horizontalgrasppredicate`, or other registered predicates. Ordering between subtasks is enforced with **`:after`** (prerequisite names must appear in the ever-satisfied set). **`on` / `in`** subtasks use `subtask_confirmation_steps` (or per-subtask `:confirm_steps`); most other predicates, including L1/L2, credit **on the first step** they pass validation.

**Coarse mode** (no `(:subtask_rewards ...)`): Each `(:goal ...)` conjunct is an equal-weight subtask with **no** ordering—see [SUBTASK_CONTROL_FLOW.md](./SUBTASK_CONTROL_FLOW.md).

---

## L4: Out-of-order execution

**Current status:** There is **no** separate L4 implementation (no predicate, flag, or `info` field) that **detects** or **scores** “out-of-order” execution relative to a reference ordering.

What exists today:

- **Fine-grained subtasks** with **`:after`** **gate** credit: a later subtask does not receive reward until prerequisites have been credited at least once this episode. That enforces a **partial order**; it does not by itself emit an “OOO” diagnostic or penalty signal.
- **Coarse** goal-atom subtasks allow **any** order of partial completion.

If L4 is required (e.g. logging whether the agent achieved stage B before stage A, or shaping penalties for OOO), that would be **new** logic on top of `_evaluate_subtask_rewards` or a separate evaluator.

---

## Goal evaluation vs subtask predicates

Full-episode **success** (`done`, sparse reward when `subtask_reward=False`) still requires **all** `(:goal ...)` conjuncts as today. L1/L2 predicates can appear in `(:goal ...)` as well, via `_eval_goal_predicate_state` in `bddl_base_domain.py`, which supports both dictionary predicates and parametric ones (`PARAMETRIC_PREDICATE_CLS`).

---

## Registration and parsing

| BDDL token (typical) | Python class | Notes |
|----------------------|--------------|--------|
| `localizedneareef` | `LocalizedNearEEF` | Parametric; see `instantiate_predicate`. |
| `neareef` | `NearEEF` | Parametric; distance only. |
| `near` | `Near` | Parametric binary distance. |
| `defaultgrasppredicate` | `DefaultGraspPredicate` | Fixed instance in dict. |
| `horizontalgrasppredicate` | `HorizontalGraspPredicate` | Fixed instance in dict. |
| `grasp` | `Grasp` | Weaker: any gripper–object contact. |

Full lists: `VALIDATE_PREDICATE_FN_DICT` and `PARAMETRIC_PREDICATE_CLS` in `libero/libero/envs/predicates/__init__.py`.

---

## Example BDDL fragments

**L1 as a subtask (illustrative):**

```lisp
(:subtask localize_mug
  :predicate (localizedneareef yellow_mug_1 0.12 0.25)
  :reward 0.2
)
```

**L2 after L1 (ordering via `:after`):**

```lisp
(:subtask grasp_mug
  :predicate (defaultgrasppredicate yellow_mug_1)
  :reward 0.3
  :after (localize_mug)
)
```

Exact BDDL surface syntax must match what `robosuite_parse_problem` / `parse_subtask_rewards` emit for your suite; adjust parentheses and token order to match existing task files under `libero/libero/bddl_files/`.

---

## Related files

| File | Role |
|------|------|
| `libero/libero/envs/predicates/base_predicates.py` | L1/L2 (and helper) predicate implementations. |
| `libero/libero/envs/predicates/__init__.py` | BDDL name → class / instances. |
| `libero/libero/envs/bddl_utils.py` | `parse_subtask_rewards`, parametric instantiation in subtasks. |
| `libero/libero/envs/bddl_base_domain.py` | Goal eval, subtask evaluation, `_subtask_candidate_valid`. |
| [SUBTASK_CONTROL_FLOW.md](./SUBTASK_CONTROL_FLOW.md) | End-to-end subtask pipeline. |
| [SUBTASK_STEP_INFO.md](./SUBTASK_STEP_INFO.md) | Per-step `info` contract. |

---

## Adoption note

New L1/L2 predicates are **registered and evaluable** from BDDL; many shipped `.bddl` files still use classic atoms (`on`, `in`, etc.) for goals and subtasks. Adding `localizedneareef` / `defaultgrasppredicate` to a task’s `(:subtask_rewards ...)` (or `(:goal ...)`) is how you opt into predicate-level evaluation for that task.
