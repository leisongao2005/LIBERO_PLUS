# Hierarchical Sim/Wrapper Refactor — Design Decisions

This document records the "why" behind key architecture and module-placement decisions
made during the LIBERO-Plus hierarchical reward shaping refactor. It is the companion to
the implementation spec in
`.cursor/plans/hierarchical_sim_wrapper_refactor_599aa945.plan.md`.

---

## 1. Sim is fully stateless (Option A)

**Decision:** The sim (`BDDLBaseDomain`) emits only instantaneous per-step predicate
truth in `raw_predicates` and `l4_satisfied`. It holds no per-episode state after the
refactor. `subtask_history` (the L3 one-shot bitmask) is **not** emitted by the sim;
the wrapper maintains it entirely.

**Why not Option B (sim keeps `_subtask_ever_satisfied` for `subtask_history`):**

- The wrapper must maintain `_l3_history` for its own delta detection regardless — it
  needs to know "did L3::S just flip True this step?" to emit a +1 delta. Option B
  would produce two authoritative copies of the same bitmask (sim + wrapper), creating
  a class of divergence bugs with no clear tie-breaking rule.
- Option B is architecturally incoherent: the sim would be "stateless except for this
  one set." Future readers would reasonably ask why `_subtask_ever_satisfied` survives
  when everything else was stripped, and the answer ("convenience for the wrapper") is
  exactly backwards — the wrapper is supposed to own state.
- The Phase 0 interface contract (`bddlsim_interface.py`) was authored before the
  stateless-sim requirement was clarified. Updating the contract to match the clarified
  design is the correct response; the contract is a living spec in this repo, not a
  treaty.

**Consequence:** The wrapper computes `subtask_history` from `raw_predicates["L3::S"]`
and its own `_l3_history` latch, then includes it in the **wrapper's output info** for
downstream consumers (RL trainer, logging). Sim output info contains only
`raw_predicates` and `l4_satisfied`.

---

## 2. `:after` prerequisite ordering — dropped

**Decision:** The `:after` gating in `_evaluate_subtask_rewards` is removed entirely
and not moved to the wrapper.

**Rationale:** Out-of-order (OOO) subtask detection is a separate diagnostic concern,
not a reward-shaping concern. Including `:after` gates in the shaping pipeline would
couple reward structure to a sequencing assumption that may not hold during RL
exploration. If OOO detection is added later, it belongs in a dedicated wrapper layer
or logging hook, not in the predicate evaluation path.

---

## 3. No-contact gate — deferred

**Decision:** `_subtask_candidate_valid`'s no-contact check is left out of this
refactor. A `TODO` comment marks the location in `bddl_base_domain.py`.

**Rationale:** The gate prevents crediting a subtask while the gripper is still
touching the object, which gives a cleaner continuation state for downstream subtasks.
This is a valid semantic refinement, but it requires MuJoCo contact queries inside the
sim and is not needed for the core reward-shaping architecture to function. It is
deferred to avoid scope creep; the raw-predicate approach works without it.

---

## 4. L2 grasp predicate scope restricted to pick-and-place objects

**Decision:** L2 (`DefaultGraspPredicate` / `HorizontalGraspPredicate`) is only derived
and shaped for L3 subtasks whose predicate is `on` or `in` (objects the robot physically
picks up and places). Subtasks with `turnon`, `turnoff`, `open`, or `close` predicates
have **no L2 slot**; the wrapper skips L2 reward shaping for those slots entirely.

**Rationale:** Grasp predicates are semantically meaningful only for manipulated objects.
For a `Turnon flat_stove_1` subtask, L2 would test "gripper grasping the stove," which
is never true in normal execution (you toggle a switch, not pick up the appliance). An
always-False L2 produces no useful signal and would confuse the NL status string.

**Implementation note:** The L2 derivation code should include a comment listing the
excluded predicate types so the omission is explicit.

---

## 5. Module placement for Tracks C and D

**Decision:** Pure-function modules `_delta.py` (Track C) and `_status_string.py`
(Track D) are placed under `libero/libero/envs/wrappers/`, co-located with
`hierarchical_reward_wrapper.py` in that package.

**Alternatives considered:**

- *Place under `libero/libero/` (MuJoCo-free path):* The MuJoCo-free import
  `libero.libero.hierarchical_reward_wrapper` re-exports the wrapper class from the
  heavy path. Adding `_delta.py` and `_status_string.py` to the MuJoCo-free root would
  split the implementation across two locations with no clean boundary, making the
  re-export graph harder to follow.
- *Place in a separate top-level `libero/libero/reward/` package:* Adds a new package
  with no other occupants. Premature abstraction for two utility modules that are
  exclusively consumed by `HierarchicalRewardWrapper`.

**Chosen rationale:** `_delta.py` and `_status_string.py` are implementation details of
the wrapper; co-location makes that relationship explicit. The leading underscore
signals they are private to the package, not public API. Since they are pure functions
with no MuJoCo imports, the MuJoCo-free `libero.libero.hierarchical_reward_wrapper`
module can import them from the `wrappers` sub-package without pulling in robosuite,
because Python only executes the import chain you actually call — importing
`libero.libero.envs.wrappers._delta` does not trigger `libero.libero.envs.__init__` if
that init imports robosuite conditionally or not at all.

**Caveat to verify:** Confirm `libero/libero/envs/__init__.py` does not eagerly import
robosuite/MuJoCo at module level before wiring up the MuJoCo-free import path.

---

## 6. Weight normalization at wrapper construction time

**Decision:** When the `weight-normalization` option lands, normalization is computed
once at `HierarchicalRewardWrapper.__init__`, not per step.

**Rationale:** The normalization factor depends only on the number of L3 subtasks and
the configured weights — both known at construction time. Per-step recomputation would
add CPU overhead with no benefit. Static normalization at construction also makes the
effective weight values inspectable (`wrapper.effective_weights`) without needing to
run a step.

**Interaction with `per_subtask_overrides`:** Normalization is applied to the base
`weights` dict first, then per-subtask overrides are treated as absolute values (not
subject to the global normalization). This prevents an override from being silently
rescaled and ensures the override author's intent is honored exactly.

---

## 7. `subtask_names_ordered` resolution in the wrapper

**Decision:** The wrapper infers `subtask_names_ordered` from the `L3::*` keys in
`raw_predicates` (strip prefix, preserve insertion order) on the first step after
`reset()`. Passing `subtask_names_ordered` explicitly to the constructor is recommended
for production use; the auto-resolution is a convenience for `FakeBDDLEnv`-based tests.

**Rationale:** With Option A, `subtask_history` is no longer in the sim's output, so
the previous auto-resolution path (reading `hist.keys()`) is gone. `raw_predicates`
keys are always present and ordered by BDDL declaration order per the Phase 0 spec,
making them the natural source.

**Edge case:** If `reset()` is called again before the first `step()`, `_resolved_subtask_names`
is cleared and re-derived on the next step. This is safe because episode state
(`_l1_prev`, `_l2_prev`, `_l3_history`) is also cleared on `reset()`.

---

## 8. `[Status: ...]` is critic-only; no changes to training dataset

**Decision:** The `privileged_status` string appended to `obs["instruction"]` is
intended only for the critic in the PPO RL loop. It is not stored in the SFT
demonstration dataset.

**Rationale:** This repo produces only the sim-side reward and observation signals.
The actor/critic architecture and masking strategy are the responsibility of the
external RL codebase. The stable delimiter `[Status:` is documented so the RL codebase
can compute mask boundaries.

**Future TODO:** If critic SFT training is desired, add `privileged_status` to the hdf5
dataset schema in a separate track.

---

## Open TODOs (post-refactor)

- Audit all `libero_plus_train_subtasks/` and `libero_plus_eval_subtasks/` BDDL files
  to confirm they all have `:subtask_rewards` sections in the new format. Select 5
  representative files (covering `on`, `in`, `turnon`, `open`, `close` predicate types)
  as test templates.
- Implement the no-contact gate (deferred from this refactor) once core reward shaping
  is validated end-to-end.
- Add weight normalization second pass (`weight-normalization` plan todo).
- Verify `libero/libero/envs/__init__.py` does not eagerly import MuJoCo so the
  `_delta.py` / `_status_string.py` import path stays MuJoCo-free.
