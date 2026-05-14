# Predicate Smoke Testing Guide

Verification tiers for the LIBERO-Plus hierarchical reward system (Tracks A–D complete).
All commands require the `libero_data` conda environment.

**Reward farming prevention:** All four levels (L1/L2/L3/L4) are one-shot latched per episode — each predicate key earns its reward exactly once, regardless of release/re-approach cycles. This is enforced in `HierarchicalRewardWrapper` (via `apply_one_shot_latch`) and in both debug scripts.

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
# Sim-layer integration tests (requires MuJoCo; skipped automatically if not importable)
conda run -n libero_data python -m pytest tests/test_sim_raw_predicates_integration.py -v

# Wrapper unit tests (MuJoCo-free, always runnable)
conda run -n libero_data python -m pytest tests/test_hierarchical_reward_wrapper.py -v

# Both together
conda run -n libero_data python -m pytest tests/ -v
```

**Expected output:** 8 sim-layer tests + 22 wrapper tests, all green.
Sim tests skipped automatically if robosuite/MuJoCo is not importable.

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

**`--demo` is optional.** Zero actions work fine for verifying the overlay renders and L1/L2 fire near the starting pose. L3/L4 won't fire without robot motion, but the display and video pipeline are fully exercised.

**Run with live preview window (no demo needed):**
```bash
# Shows a cv2 window with predicate overlay updating in real time. Press 'q' to stop.
conda run -n libero_data python scripts/inspect_predicates_replay.py \
    --task floor10 --steps 200 --preview

conda run -n libero_data python scripts/inspect_predicates_replay.py \
    --task basket_scene1030 --steps 200 --preview
```

**Run to file only (no live window):**
```bash
conda run -n libero_data python scripts/inspect_predicates_replay.py \
    --task floor10 --steps 200 --output floor10_debug.mp4

conda run -n libero_data python scripts/inspect_predicates_replay.py \
    --task basket_scene1030 --steps 200 --output basket_debug.mp4
```

**Run (with a demo HDF5 — shows full L3/L4 completion, if demo data available):**
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
- Shaped reward delta per step, cumulative reward (each predicate earns reward at most once)
- `[Status: ...]` string updated live
- Yellow highlight when a predicate newly fires (first time only)

**Requires:** Physical display and keyboard (or SpaceMouse).

**Must use `conda activate` (not `conda run`)** — so the terminal is directly attached and ANSI screen-clearing works:
```bash
conda activate libero_data
python scripts/teleop_with_predicates.py --task floor10 --device keyboard
python scripts/teleop_with_predicates.py --task basket_scene1030 --device spacemouse
```

**Controls:**
- Keyboard: standard robosuite keyboard bindings (wasd + ijkl for 6-DoF)
- `q` — reset episode
- `Ctrl+C` — quit

**What to manually verify:**
1. Move gripper toward object → `L1::*` lights up green; moving away and returning gives **no second reward** (one-shot)
2. Grasp object (pick task) → `L2::*` lights up green; releasing and re-grasping gives **no second reward**
3. Place/complete subtask → `L3::*` fires yellow ("newly fired"), then stays green; cumulative reward stops increasing
4. For `floor10`: `L2::turnon_stove` must **never** turn green regardless of what you do
5. Complete all subtasks → `L4` fires, simulation **freezes** with a static "GOAL REACHED!" banner; the loop no longer steps the environment or overwrites stdout. Press `q` to reset.

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
  Shaped (+delta):    +0.10   ← only non-zero the FIRST time each predicate fires
  Cumulative shaped:  0.10    ← stops growing once all predicates have fired once

Newly fired this step: L1::turnon_stove  ← yellow highlight; never repeats for this key
═══════════════════════════════════════════════════════
Controls: SpaceNav/keyboard | 'q' reset | Ctrl+C quit
```

**When goal is reached (L4 fires):**
The simulation stops stepping. The screen clears and shows a static banner:
```
═══════════════════════════════════════════════════════
GOAL REACHED!  [step 87]
═══════════════════════════════════════════════════════
All subtasks complete.
Cumulative shaped reward: 0.90

Press 'q' to reset or Ctrl+C to quit.
```

---

## Known design note: l3_history key convention

`apply_one_shot_latch` stores keys matching whatever you pass. Both scripts and the wrapper maintain
a `l3_history_plain` dict (keyed by plain subtask names) separately from the
`L3::*`-keyed dict used for `compute_shaped_reward`. This is because:
- `compute_shaped_reward` expects `L3::<name>` keys in `one_shot_fired`
- `format_status_string` / `resolve_slot_status` expect plain subtask names in `l3_history`

`HierarchicalRewardWrapper` performs the same translation internally (`_l3_history` uses plain keys; it translates to `L3::*` when calling `compute_shaped_reward`).

## Known design note: `detect_transient_deltas` is unused

`_delta.py` still exports `detect_transient_deltas` (False→True edge detection) but **no code calls it for reward**. It was replaced by `apply_one_shot_latch` for all four levels to prevent reward farming. It is kept in the module for potential future use (e.g., per-step diagnostic logging) but should not be wired into any reward path.
