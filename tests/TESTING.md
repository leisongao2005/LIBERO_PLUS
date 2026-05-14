# Predicate Smoke Testing Guide

Three verification tiers for the Track A `raw_predicates` refactor.
All commands require the `libero_data` conda environment.

---

## Tier 2 — Headless integration test (fastest, CI-safe)

**What it checks:**
- `info["raw_predicates"]` and `info["l4_satisfied"]` present on every step
- `subtask_history` and all legacy keys (`subtask_rewards`, etc.) absent
- Every predicate value is `bool` (not `np.bool_` or int)
- `raw_predicates["L4"] == l4_satisfied` on every step
- Exact key set matches subtask declarations
- `L2::turnon_stove` is always `False` (Turnon is not a pick predicate)

**Run:**
```bash
conda run -n libero_data python -m pytest tests/test_sim_raw_predicates_integration.py -v
```

**Expected output:** 8 tests across 4 classes, all green.
Skipped automatically if robosuite/MuJoCo is not importable.

**Tasks tested:**
- `basket_scene1030` — pure pick-and-place (cream cheese + alphabet soup → basket)
- `floor10` — mixed Turnon + On (turn on stove + place moka pot)

---

## Tier 3 — Demo replay with video overlay

**What it checks:**
- Predicate state shown frame-by-frame as MP4 overlay
- Visually verify L1→L2→L3→L4 progression on a real or zero-action rollout
- Shaped reward delta printed per frame
- Status string (`[Status: name=done, ...]`) shown

**Run (zero-action, no demo needed):**
```bash
# Mixed task — verify L2::turnon_stove stays False in the video
conda run -n libero_data python scripts/inspect_predicates_replay.py \
    --task floor10 --steps 200 --output floor10_debug.mp4

# Pure pick-and-place
conda run -n libero_data python scripts/inspect_predicates_replay.py \
    --task basket_scene1030 --steps 200 --output basket_debug.mp4
```

**Run (with a demo HDF5 — shows full L3/L4 completion):**
```bash
conda run -n libero_data python scripts/inspect_predicates_replay.py \
    --task floor10 \
    --demo libero/datasets/libero_plus_train_subtasks/FLOOR_10_turn_on_the_stove_and_put_the_left_moka_pot_on_it_demo.hdf5 \
    --output floor10_full.mp4
```

**What to verify in the video:**
| Predicate | When it should fire |
|-----------|-------------------|
| `L1::*`   | Gripper moves near primary object (green) |
| `L2::place_moka_pot_2` | Moka pot grasped (green) |
| `L2::turnon_stove` | **Never** — must stay gray throughout |
| `L3::*`   | Subtask physically satisfied (green, labeled `*NEW*`) |
| `L4`      | All subtasks done simultaneously |

A stdout summary table is also printed after the run — rows only shown when something fired or every 10 steps.

---

## Tier 4 — Live teleoperation with predicate dashboard

**What it checks:**
- Real-time L1/L2/L3/L4 state as you drive the robot
- Shaped reward delta per step, cumulative reward
- `[Status: ...]` string updated live
- Yellow highlight when a predicate newly fires

**Requires:** Physical display and keyboard (or SpaceMouse).

**Run:**
```bash
# Keyboard control
conda run -n libero_data python scripts/teleop_with_predicates.py \
    --task floor10 --device keyboard

# SpaceMouse (if available)
conda run -n libero_data python scripts/teleop_with_predicates.py \
    --task basket_scene1030 --device spacemouse
```

**Controls:**
- Keyboard: standard robosuite keyboard bindings (wasd + ijkl for 6-DoF)
- `q` — reset episode
- `Ctrl+C` — quit

**What to manually verify:**
1. Move gripper toward object → `L1::*` lights up green
2. Grasp object (pick task) → `L2::*` lights up green
3. Place/complete subtask → `L3::*` fires yellow ("newly fired"), then stays green
4. For `floor10`: `L2::turnon_stove` must **never** turn green regardless of what you do
5. Complete all subtasks → `L4` fires, "GOAL REACHED!" banner appears

**Dashboard layout:**
```
═══════════════════════════════════════════════════════
LIBERO-Plus Predicate Debugger  [step 42]
═══════════════════════════════════════════════════════
Task: turn on the stove and put the left moka pot on it
Status: [Status: turnon_stove=localized, place_moka_pot_2=pending]

Predicate State:
  Subtask                 L1        L2        L3 (now)  L3 ever
  ─────────────────────────────────────────────────────────────
  turnon_stove            True      False     False     False
  place_moka_pot_2        False     False     False     False

  L4 (goal): False

Reward this step:
  Sparse reward:      0.0
  Shaped (+delta):    +0.10
  Cumulative shaped:  0.45

Newly fired this step: L1::turnon_stove
═══════════════════════════════════════════════════════
Controls: SpaceNav/keyboard | 'q' reset | Ctrl+C quit
```

---

## Known design note: l3_history key convention

`apply_one_shot_latch` stores keys matching whatever you pass. The scripts maintain
a `l3_history_plain` dict (keyed by plain subtask names) separately from the
`L3::*`-keyed dict returned by `apply_one_shot_latch`. This is because:
- `compute_shaped_reward` expects `L3::<name>` keys in `one_shot_fired`
- `format_status_string` / `resolve_slot_status` expect plain subtask names in `l3_history`

When Phase 2 wires `_delta.py` into `HierarchicalRewardWrapper`, the same
translation will need to happen in the wrapper.
