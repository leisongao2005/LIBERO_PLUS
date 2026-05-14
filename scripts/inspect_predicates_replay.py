"""
inspect_predicates_replay.py — Predicate-overlay video debugger for LIBERO-Plus tasks.

Replays a demonstration (or runs zero actions) and records an MP4 with L1/L2/L3/L4
predicate state overlaid on each frame, so a human can visually verify that predicates
fire at the correct moments.

Usage
-----
    python scripts/inspect_predicates_replay.py --task floor10 [--demo path/to.hdf5] [--steps 200] [--output out.mp4]
    python scripts/inspect_predicates_replay.py --task basket_scene1030

CLI args
--------
--task      {floor10 | basket_scene1030}   (required)
--demo      Path to a demo HDF5 file.  Loads actions from data/demo_0/actions.
            If omitted (or file not found), falls back to np.zeros(7) for --steps steps.
--steps     Number of zero-action steps when no demo is provided (default: 200).
--output    Output MP4 path (default: predicate_debug.mp4in the cwd).
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Repo root on path so we can import libero.* without installation
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Task catalogue
# ---------------------------------------------------------------------------

_BDDL_DIR = _REPO_ROOT / "libero" / "libero" / "bddl_files" / "libero_plus_train_subtasks"
_INIT_DIR = _REPO_ROOT / "libero" / "libero" / "init_files" / "libero_plus_train_subtasks"

TASK_CATALOGUE: Dict[str, Dict[str, Path]] = {
    "basket_scene1030": {
        "bddl": _BDDL_DIR / (
            "LIVING_ROOM_TABLETOP_BASKET_SCENE1030_put_the_cream_cheese_in_the_basket"
            "_and_put_the_alphabet_soup_in_the_basket.bddl"
        ),
        "init": _INIT_DIR / (
            "LIVING_ROOM_TABLETOP_BASKET_SCENE1030_put_the_cream_cheese_in_the_basket"
            "_and_put_the_alphabet_soup_in_the_basket.pruned_init"
        ),
    },
    "floor10": {
        "bddl": _BDDL_DIR / "FLOOR_10_turn_on_the_stove_and_put_the_left_moka_pot_on_it.bddl",
        "init": _INIT_DIR / "FLOOR_10_turn_on_the_stove_and_put_the_left_moka_pot_on_it.pruned_init",
    },
}

# ---------------------------------------------------------------------------
# BDDL subtask-name parser (no bddl_utils dependency)
# ---------------------------------------------------------------------------

def parse_subtask_names(bddl_path: Path) -> List[str]:
    """Return subtask names in BDDL declaration order using a regex."""
    content = bddl_path.read_text()
    return re.findall(r"\(:subtask\s+(\S+)", content)

# ---------------------------------------------------------------------------
# Demo loader
# ---------------------------------------------------------------------------

def load_demo_actions(demo_path: Path) -> Optional[np.ndarray]:
    """Load actions from a LIBERO HDF5 demo file.

    Tries ``data/demo_0/actions`` first, then ``actions`` directly.
    Returns ``None`` if the file does not exist or the key is missing.
    """
    try:
        import h5py  # type: ignore
    except ImportError:
        warnings.warn("h5py not installed — cannot load demo, using zero actions.")
        return None

    if not demo_path.exists():
        warnings.warn(f"Demo file not found: {demo_path} — using zero actions.")
        return None

    with h5py.File(demo_path, "r") as f:
        for key in ("data/demo_0/actions", "actions"):
            if key in f:
                actions = f[key][:]
                print(f"[demo] Loaded {len(actions)} actions from '{key}' in {demo_path}")
                return actions
        warnings.warn(
            f"Could not find 'data/demo_0/actions' or 'actions' in {demo_path} — "
            "using zero actions."
        )
        return None

# ---------------------------------------------------------------------------
# Frame rendering helpers (cv2)
# ---------------------------------------------------------------------------

def _text_color(value: bool) -> Tuple[int, int, int]:
    """BGR: green for True, gray for False."""
    return (0, 200, 0) if value else (160, 160, 160)


def _draw_overlay(
    frame: np.ndarray,
    step: int,
    sparse_reward: float,
    shaped_reward: float,
    status: str,
    subtask_names: Sequence[str],
    raw: Dict[str, bool],
    l3_history: Dict[str, bool],
    l3_fired_this_step: Dict[str, bool],
    l4_fired_this_step: bool,
    weights: Dict[str, float],
) -> np.ndarray:
    """Draw predicate overlay on *frame* (RGB uint8).  Returns a new array."""
    import cv2  # type: ignore

    img = frame.copy()
    H, W = img.shape[:2]

    # Semi-transparent black background for the overlay panel
    panel_w = min(W, 500)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, H), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    small = 0.42
    medium = 0.52
    large = 0.65
    lh = 20  # line height px
    x0 = 8
    y = 22

    def put(text: str, color=(255, 255, 255), scale=small, bold=False):
        nonlocal y
        thickness = 2 if bold else 1
        cv2.putText(img, text, (x0, y), font, scale, color, thickness, cv2.LINE_AA)
        y += lh

    def put_kv(key: str, value: bool, suffix: str = ""):
        nonlocal y
        label = f"  {key}: "
        cv2.putText(img, label, (x0, y), font, small, (220, 220, 220), 1, cv2.LINE_AA)
        tw, _ = cv2.getTextSize(label, font, small, 1)
        val_str = ("T" if value else "F") + suffix
        cv2.putText(img, val_str, (x0 + tw[0], y), font, small, _text_color(value), 1, cv2.LINE_AA)
        y += lh

    # --- Header ---
    put(f"Step {step}", scale=large, bold=True)
    put(f"sparse_reward={sparse_reward:.2f}  shaped={shaped_reward:+.3f}", scale=small)
    put(f"status: {status}", scale=small)
    y += 4

    # --- Per-subtask table ---
    for s in subtask_names:
        l1 = bool(raw.get(f"L1::{s}", False))
        l2 = bool(raw.get(f"L2::{s}", False))
        l3_raw = bool(raw.get(f"L3::{s}", False))
        l3_done = bool(l3_history.get(s, False))
        l3_new = bool(l3_fired_this_step.get(f"L3::{s}", False))

        # Delta bonus for this subtask
        delta = 0.0
        # L3 one-shot
        if l3_new:
            delta += weights["L3"]
        # L1/L2 transient are computed in _l1_delta / _l2_delta context already —
        # we show whether they're True now (not necessarily delta) for readability.

        put(f"[{s}]", scale=medium, bold=True, color=(255, 220, 80))
        put_kv("L1(loc)", l1)
        put_kv("L2(grasp)", l2)
        put_kv("L3(subtask)", l3_done, suffix=(" *NEW*" if l3_new else " (done)" if l3_done and not l3_new else ""))
        if delta > 0:
            put(f"  => L3 bonus +{delta:.2f}", color=(0, 255, 180), scale=small)
        y += 4

    # --- L4 ---
    l4_val = bool(raw.get("L4", False))
    l4_color = (0, 200, 0) if l4_val else (160, 160, 160)
    bonus_str = f" +{weights['L4']:.2f}" if l4_fired_this_step else ""
    put(f"L4(terminal): {'T' if l4_val else 'F'}{bonus_str}", color=l4_color, scale=medium, bold=True)

    return img

# ---------------------------------------------------------------------------
# Main replay loop
# ---------------------------------------------------------------------------

def run_replay(
    task: str,
    demo_path: Optional[Path],
    n_steps: int,
    output_path: Path,
) -> None:
    import cv2  # type: ignore
    import torch  # type: ignore

    from libero.libero.envs.env_wrapper import ControlEnv
    from libero.libero.envs.wrappers._delta import (
        apply_one_shot_latch,
        compute_shaped_reward,
        detect_transient_deltas,
    )
    from libero.libero.envs.wrappers._status_string import format_status_string

    catalogue = TASK_CATALOGUE[task]
    bddl_path: Path = catalogue["bddl"]
    init_path: Path = catalogue["init"]

    if not bddl_path.exists():
        sys.exit(f"[error] BDDL file not found: {bddl_path}")
    if not init_path.exists():
        sys.exit(f"[error] Init file not found: {init_path}")

    subtask_names = parse_subtask_names(bddl_path)
    print(f"[task] {task}")
    print(f"[bddl] {bddl_path.name}")
    print(f"[subtasks] {subtask_names}")

    # Load demo actions (or fall back to zeros)
    actions: Optional[np.ndarray] = None
    if demo_path is not None:
        actions = load_demo_actions(demo_path)
    if actions is None:
        print(f"[actions] Using {n_steps} zero-action steps.")
        actions = np.zeros((n_steps, 7), dtype=np.float32)
    else:
        n_steps = len(actions)
        print(f"[actions] Replaying {n_steps} demo steps.")

    # Build environment
    print("[env] Creating ControlEnv (this may take a moment)…")
    env = ControlEnv(
        bddl_file_name=str(bddl_path),
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        render_gpu_device_id=-1,
    )
    env.reset()

    # Load initial state
    init_states = torch.load(str(init_path), weights_only=False)
    init_state = init_states[0]
    if hasattr(init_state, "numpy"):
        init_state = init_state.numpy()
    env.set_init_state(init_state)
    print("[env] Init state loaded.")

    # Reward weights (default from spec)
    weights: Dict[str, float] = {"L1": 0.1, "L2": 0.2, "L3": 0.5, "L4": 1.0}

    # Key sequences
    l1_keys = [f"L1::{s}" for s in subtask_names]
    l2_keys = [f"L2::{s}" for s in subtask_names]

    # Episode state
    l1_prev: Dict[str, bool] = {}
    l2_prev: Dict[str, bool] = {}
    l3_history: Dict[str, bool] = {}
    l4_ever = False

    # Video writer — determined from first frame
    video_writer: Optional[cv2.VideoWriter] = None
    FPS = 20

    # Summary log
    summary_rows: List[str] = []

    print(f"[video] Recording {n_steps} steps → {output_path}")

    for step_idx in range(n_steps):
        action = actions[step_idx]
        obs, sparse_reward, done, info = env.step(action)

        raw: Dict[str, bool] = info.get("raw_predicates", {})
        l4_satisfied: bool = bool(info.get("l4_satisfied", False))

        # Delta detection
        l1_deltas = detect_transient_deltas(l1_prev, raw, l1_keys)
        l2_deltas = detect_transient_deltas(l2_prev, raw, l2_keys)

        # One-shot latching for L3.
        # _l3_history is keyed by plain subtask name (DESIGN.md §1); _status_string.py expects this.
        # Build a plain-name dict from raw_predicates before latching, then convert fired keys
        # back to L3::* for compute_shaped_reward (which expects that convention).
        l3_curr = {s: bool(raw.get(f"L3::{s}", False)) for s in subtask_names}
        l3_history, l3_fired_plain = apply_one_shot_latch(l3_history, l3_curr, subtask_names)
        l3_fired = {f"L3::{k}": v for k, v in l3_fired_plain.items()}

        # L4 one-shot
        l4_fired_this_step = False
        if raw.get("L4", False) and not l4_ever:
            l4_fired_this_step = True
            l4_ever = True
        l4_fired_dict: Dict[str, bool] = {"L4": True} if l4_fired_this_step else {}

        # Shaped reward
        shaped_reward = compute_shaped_reward(
            {**l1_deltas, **l2_deltas},
            {**l3_fired, **l4_fired_dict},
            weights,
            subtask_names,
        )

        # Status string
        status = format_status_string(subtask_names, l3_history, raw)

        # Update prev for next step
        l1_prev = dict(raw)
        l2_prev = dict(raw)

        # Render frame
        raw_frame = env.sim.render(camera_name="agentview", height=512, width=512)
        # MuJoCo returns (H, W, C) RGB; flip vertically (sim origin is bottom-left)
        frame_rgb: np.ndarray = raw_frame[::-1].copy()
        if frame_rgb.shape[2] == 4:
            frame_rgb = frame_rgb[:, :, :3]

        # Draw overlay
        frame_overlaid = _draw_overlay(
            frame=frame_rgb,
            step=step_idx,
            sparse_reward=float(sparse_reward),
            shaped_reward=shaped_reward,
            status=status,
            subtask_names=subtask_names,
            raw=raw,
            l3_history=l3_history,
            l3_fired_this_step=l3_fired,
            l4_fired_this_step=l4_fired_this_step,
            weights=weights,
        )

        # Convert RGB → BGR for OpenCV VideoWriter
        frame_bgr = cv2.cvtColor(frame_overlaid, cv2.COLOR_RGB2BGR)

        if video_writer is None:
            H, W = frame_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(str(output_path), fourcc, FPS, (W, H))

        video_writer.write(frame_bgr)

        # Build summary row
        l1_vals = {s: bool(raw.get(f"L1::{s}")) for s in subtask_names}
        l2_vals = {s: bool(raw.get(f"L2::{s}")) for s in subtask_names}
        l3_vals = {s: bool(l3_history.get(s)) for s in subtask_names}
        fired_str = []
        for s in subtask_names:
            if l1_deltas.get(f"L1::{s}"):
                fired_str.append(f"L1::{s}")
            if l2_deltas.get(f"L2::{s}"):
                fired_str.append(f"L2::{s}")
            if l3_fired.get(f"L3::{s}"):
                fired_str.append(f"L3::{s}")
        if l4_fired_this_step:
            fired_str.append("L4")

        summary_rows.append(
            f"step={step_idx:4d}  sparse={float(sparse_reward):.1f}  shaped={shaped_reward:+.3f}  "
            f"L1={[l1_vals[s] for s in subtask_names]}  "
            f"L2={[l2_vals[s] for s in subtask_names]}  "
            f"L3_done={[l3_vals[s] for s in subtask_names]}  "
            f"L4={raw.get('L4', False)}  "
            f"fired={fired_str or '-'}"
        )

        if done:
            print(f"[env] Episode done at step {step_idx}.")
            break

    if video_writer is not None:
        video_writer.release()
    env.close()

    # Print summary table
    print("\n" + "=" * 90)
    print(f"SUMMARY — task={task}  subtasks={subtask_names}")
    print("=" * 90)
    # Print header
    print(
        f"{'step':>6}  {'sparse':>6}  {'shaped':>7}  "
        + "  ".join(f"L1[{s}]" for s in subtask_names)
        + "  "
        + "  ".join(f"L2[{s}]" for s in subtask_names)
        + "  "
        + "  ".join(f"L3[{s}]" for s in subtask_names)
        + "  L4  fired"
    )
    print("-" * 90)

    # Only print rows where something fired or at every 10 steps to avoid spam
    for i, row in enumerate(summary_rows):
        # Always print if something fired this step
        fired_something = ("L1::" in row or "L2::" in row or "L3::" in row or "L4" in row.split("fired=")[1])
        if fired_something or i % 10 == 0:
            print(row)

    print("=" * 90)
    print(f"[done] Video saved to: {output_path.resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a LIBERO-Plus demo (or zero actions) and record a predicate-overlay MP4.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=list(TASK_CATALOGUE.keys()),
        help="Which task to run.",
    )
    parser.add_argument(
        "--demo",
        default=None,
        type=Path,
        metavar="PATH",
        help="Path to a demo HDF5 file (data/demo_0/actions).  Omit to use zero actions.",
    )
    parser.add_argument(
        "--steps",
        default=200,
        type=int,
        metavar="N",
        help="Number of zero-action steps when no demo is given (default: 200).",
    )
    parser.add_argument(
        "--output",
        default=Path("predicate_debug.mp4"),
        type=Path,
        metavar="PATH",
        help="Output MP4 path (default: predicate_debug.mp4).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_replay(
        task=args.task,
        demo_path=args.demo,
        n_steps=args.steps,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
