# LIBERO++ Design Intent

## Problem

SOTA models on LIBERO achieve near-perfect success rates via spatial memorization, not genuine visual grounding. Performance collapses when object locations or orientations are slightly perturbed. Current benchmarks also lack fine-grained trajectory-level evaluation needed for RL training and failure diagnosis.

Specifically:
- **Eval leakage:** Training and evaluation use the same tasks — models memorize trajectories, not tasks
- **Narrow distribution:** The fixed object placement distribution in LIBERO-10 makes sparse RL rewards accidentally sufficient, masking generalization failure
- **No failure attribution:** Existing fine-grained benchmarks (including LIBERO-X) classify by task complexity or subtask count, but cannot automatically identify *where* in the trajectory a model fails (localization vs. grasp vs. task planning)

---

## Core Differentiators vs. Prior Work

| | LIBERO-PRO | LIBERO-Plus | LIBERO-X | **LIBERO++** |
|---|---|---|---|---|
| Broader initial state distribution | ✓ limited | ✓ slight | ✓ perturbation | **✓ truly random** |
| Visual/language perturbations | — | ✓ | ✓ | out of scope |
| Fine-grained evaluation | — | — | ✓ multi-level | **✓ trajectory-level** |
| RL-compatible shaped reward | — | — | — | **✓ L1–L4 weighted** |
| Unbounded novel scene generation | — | — | — | **✓ infinite sampler** |

**Load-bearing claim:** The combination of *truly infinite scene generation* (no fixed perturbation range — semantically verified random sampling) and *automated trajectory-level failure attribution* (L1–L4 per episode). LIBERO-X has multi-level eval but not the generation pipeline; we have both.

---

## Empirical Anchor

Already validated on π-family, OpenVLA, OpenVLA-OFT:
- Models achieve ≥95% SR on LIBERO-10
- Same models collapse to <5% SR on LIBERO++ tasks, **including tasks geometrically identical to LIBERO-10 but with objects shuffled**
- Rollout inspection shows some models reproduce the original LIBERO-10 trajectory regardless of new object positions — confirming memorization, not generalization failure

---

## Key Deliverables

### 1. Infinite Scene Generation
- LIBERO-10 tasks classified into **7 generalized task types**
- **156 generalized task configurations** — each describes high-level goal + relevant objects without fixed poses
- **Random initial state sampler** + semantic verifier (no object overlaps, goal state physically achievable from sampled start)
- Ships with **156 configs × 50 pregenerated states = 7,800 novel initial states**
- Users can generate more states on demand — scenes are never exhausted and cannot be memorized

### 2. Train / Eval Split
- **41 held-out eval tasks** targeting: OOD objects, OOD target locations, modified task structures
- Remaining 115 tasks available for training
- Eval tasks specifically chosen to be impossible to generalize to via trajectory memorization

### 3. Trajectory-Level Failure Attribution (L1–L4)
Four evaluation levels recorded per episode:

| Level | Name | Trigger | Notes |
|-------|------|---------|-------|
| L1 | Localization | Primary object near end-effector | Auto-derived; overridden True when L2 is True |
| L2 | Grasp | Clean gripper contact, nothing else touching | Only for pick/place predicates; False for articulation |
| L3 | Subtask | BDDL `:subtask` predicate satisfied | Must-release gate: only fires after gripper releases object |
| L4 | Terminal | Full `(:goal ...)` satisfied | Sparse task completion |

- Per-episode one-shot latching: each level credits at most once, preventing farming
- Example failure modes this distinguishes: "never reached object (L1 fail)", "reached but couldn't grasp (L2 fail)", "grasped but dropped before placement (L3 fail)", "subtask done but full goal not met (L4 fail)"
- **Articulation tasks (open/close/turnon/turnoff):** L1 still fires (EEF proximity), L2 never fires (no grasp predicate), L3 absorbs both grasp and completion signal. This means L3 carries more weight for articulation tasks — reward weighting may need per-task-type tuning (open TODO)

### 4. RL-Compatible Shaped Reward
- Shaped reward = weighted sum of first-fire events at L1–L4
- Replaces sparse completion reward, providing learning signal on partial progress
- One-shot latching built in — reward cannot be farmed by repeated approach/release cycles
- Infrastructure delivered; RL training experiments are out of scope for this paper (enabling future work)

---

## Success Criteria

- SR collapse reproduced and documented across ≥3 SOTA model families
- L1–L4 breakdown correctly identifies the dominant failure mode for each model on eval tasks
- State generator produces valid, non-degenerate scenes across all 156 configs
- Reward weighting validated across both pick/place and articulation task types

---

## Out of Scope

- Visual/lighting/camera perturbations (LIBERO-Plus covers this)
- Language instruction perturbations
- Human demonstration collection (tentative future addition)
- RL training experiments (infrastructure shipped; experiments deferred)
- Non-LIBERO-10-derived task types (future extension)

---

---

# TODO Lists

## Benchmark TODOs — Agent Execution Plan

154 BDDL files already exist (114 train, 40 eval). Work is organized into parallel waves. Agents within the same wave are fully independent and can run simultaneously.

---

### Wave 1 — All parallel, start now

**Agent 1A · BDDL Semantic Audit**
Read all 154 BDDL files in `libero_plus_{train,eval}_subtasks/`. For each file audit:
- Undefined object/region references in `:init` (e.g., `plate_1`/`plate_init_region` referenced but not declared in `:objects` or `:regions`)
- Full yaw rotation `(0.0, 6.28)` on articulated fixtures (stove, microwave, cabinet) — if knobs/handles can face away from the robot, goal may be unreachable; flag and propose tighter bounds
- Stale `:after` tags in `(:subtask_rewards ...)` — dead code since Track A removed parser support; flag for cleanup
- Placement region overlaps between objects that could produce conflicting initial states
Output: `docs/bddl_audit_findings.md` — one entry per issue with filename, issue type, and proposed fix; sorted by severity.

**Agent 1B · Task Type Classification**
Read all 154 filenames + BDDL contents. Identify the task types implicit in the corpus (expected candidates: stove turnon+place, microwave put+close, microwave open+put, cabinet put+close, cabinet open+put, book-in-caddy, moka-pots-on-stove, mug-on-plate, basket multi-object — verify whether these collapse to exactly 7). For each type: one-paragraph description, list of all member BDDL files, and any type-specific achievability constraints (e.g., articulation orientation requirements). Output: `docs/task_type_classification.md`.

**Agent 1C · Per-Episode Trajectory Log (code)**
Extend `HierarchicalRewardWrapper` to record, for each level that fires, the step index at which it first fired. Expose as `first_fire_steps: dict[str, int]` alongside the existing `predicate_trajectory`. Update `print_predicate_history()` to include step numbers. Read-only instrumentation — no behavior changes to reward logic.

---

### Wave 2A — Requires Agent 1A findings

**Agent 2A · BDDL Fixes**
Apply all fixes from `docs/bddl_audit_findings.md`:
- Remove undefined `:init` references
- Tighten yaw bounds on articulated fixtures to keep handles/knobs in robot-facing quadrant
- Strip dead `:after` tags from `(:subtask_rewards ...)`
- Resolve placement region overlaps
One commit per fix group. Do not change range coordinates unless they are the source of an identified issue.

---

### Wave 2B — Parallel with 2A, no dependency

**Agent 2B · Sampling Code Audit + Extension Plan (human-in-loop)**
Existing sampling infrastructure lives in `libero/libero/envs/regions/`: `TableRegionSampler`, `MultiRegionRandomSampler`, `ObjectPropertySampler`. Read all three files plus `bddl_utils.py` to understand how BDDL region definitions are parsed and passed to samplers. Audit for weaknesses: correctness of yaw quaternion construction, whether `ensure_valid_placement` collision-checks against fixtures (not just other objects), FOV enforcement (none appears to exist), and failure behavior when no valid placement is found within N attempts. Write up findings and a concrete extension plan. This phase requires human review before code changes are made — output findings doc, not implementation.

---

### Wave 3 — Requires 2A + 2B

**Agent 3 · State Generation**
Run sampler over all 154 fixed BDDL configs, 50 states per config = 7,700 states. Serialize as pickles mirroring the `init_files/` directory structure. Log any configs where the sampler fails to find 50 valid states within 1,000 attempts — these are remaining achievability gaps.

---

### Standalone (no wave dependency)

- [ ] **Articulation reward reweighting** — L3 carries both grasp and completion signal for articulation tasks since L2 never fires; test whether default L1/L3/L4 weights produce well-shaped reward for these tasks, adjust per-task-type weights in `RewardConfig` if needed
- [ ] **Document 41 eval tasks** — write a one-line OOD justification for each of the 40 eval BDDL files (which axis: OOD object / OOD location / modified task)

---

## Paper / RL TODOs

- [ ] **Run SR collapse experiments** — evaluate π0, OpenVLA, OpenVLA-OFT on LIBERO++ eval set; document per-task and aggregate SR
- [ ] **Produce failure attribution breakdown** — per model: % failures at L1 / L2 / L3 / L4; the core differentiating result vs. LIBERO-X; requires Agent 1C complete
- [ ] **Design memorization ablation** — show models reproduce original LIBERO-10 trajectory even with objects displaced; rollout video + quantitative trajectory similarity metric
- [ ] **Write LIBERO-X comparison section** — make the gap between their multi-level eval and L1–L4 trajectory attribution crisp for reviewers
- [ ] **Decide RL experiment scope** — in scope for submission or infrastructure/future work only
- [ ] **Collect human demonstrations** (tentative) — decide before submission; affects benchmark characterization significantly
