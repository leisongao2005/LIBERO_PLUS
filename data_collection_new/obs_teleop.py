"""
OpenCV window: live camera observation + keyboard deltas for OSC_POSE control.

Run (from repo root, with LIBERO deps installed):
  python -m data_collection_new.obs_teleop --benchmark libero_10 --task-id 0

Axis legend (position deltas, OSC dx,dy,dz; tune signs in *_DELTA below):
  W / S  — up / down (+Z / -Z)
  A / D  — left / right (-X / +X)
  Q / E  — backward / forward (-Y / +Y)

The 7D **action** vector is OSC_POSE: (dx, dy, dz, drot_0, drot_1, drot_2, gripper).
Metrics (bottom panel) include observation-space eef: 3 pos + 4 quat from obs,
not the same as the action semantics.

ESC — quit.  ,  /  .  — gripper open / close (small deltas on the 7th action dim).

Dependencies: opencv-python (cv2), robosuite, torch (via libero imports), MuJoCo.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# --- Keyboard → position delta mapping (OSC: action[0]=dx, [1]=dy, [2]=dz) ------------
# W=up, D=right, E=forward (tune signs here if inverted on your machine).
KEY_W_DELTA = np.array([0.0, 0.0, 1.0])  # up +Z
KEY_S_DELTA = np.array([0.0, 0.0, -1.0])  # down
KEY_A_DELTA = np.array([0.0, -1.0, 0.0])  # left -X
KEY_D_DELTA = np.array([0.0, 1.0, 0.0])  # right +X
KEY_Q_DELTA = np.array([1.0, 0.0, 0.0])  # backward -Y
KEY_E_DELTA = np.array([-1.0, 0.0, 0.0])  # forward +Y

GRIPPER_OPEN_DELTA = 0.08
GRIPPER_CLOSE_DELTA = -0.08


def pick_camera_obs_key(obs: Dict[str, Any], preferred: str) -> str:
    if preferred in obs:
        return preferred
    candidates = [k for k in obs if k.endswith("_image")]
    if not candidates:
        raise KeyError(
            f"No key {preferred!r} and no *_image keys in observation. Keys: {sorted(obs.keys())}"
        )
    candidates.sort()
    return candidates[0]


def observation_to_bgr_display(
    obs: Dict[str, Any], camera_key: str, display_width: int
) -> np.ndarray:
    """RGB obs → BGR uint8. If display_width > 0, scale so image width matches (up or down)."""
    img = np.asarray(obs[camera_key])
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    # robosuite RGB → OpenCV BGR; vertical flip matches video_utils.append_obs
    img_bgr = img[..., ::-1].copy()
    img_bgr = img_bgr[::-1].copy()
    h, w = img_bgr.shape[:2]
    if display_width > 0 and w != display_width:
        new_w = display_width
        new_h = max(1, int(round(h * (display_width / float(w)))))
        interp = cv2.INTER_AREA if w > display_width else cv2.INTER_CUBIC
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=interp)
    return np.ascontiguousarray(img_bgr, dtype=np.uint8)


def compose_video_and_metrics_panel(
    frame_bgr: np.ndarray,
    lines: List[str],
    *,
    panel_bg: Tuple[int, int, int] = (40, 40, 40),
    text_color: Tuple[int, int, int] = (220, 230, 220),
    font_scale: float = 0.52,
    line_height: int = 24,
    pad: int = 14,
) -> np.ndarray:
    """Stack camera on top and a solid-color metrics strip below (no text on the video)."""
    h, w = frame_bgr.shape[:2]
    panel_h = pad * 2 + len(lines) * line_height
    out = np.zeros((h + panel_h, w, 3), dtype=np.uint8)
    out[:h, :w] = frame_bgr
    out[h:, :] = panel_bg
    y = h + pad + line_height - 4
    for line in lines:
        cv2.putText(
            out,
            line,
            (pad, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            1,
            cv2.LINE_AA,
        )
        y += line_height
    return np.ascontiguousarray(out, dtype=np.uint8)


def format_eef7(obs: Dict[str, Any]) -> str:
    pos_k, quat_k = "robot0_eef_pos", "robot0_eef_quat"
    if pos_k not in obs or quat_k not in obs:
        return "eef (obs): n/a"
    p = np.asarray(obs[pos_k]).reshape(-1)
    q = np.asarray(obs[quat_k]).reshape(-1)
    parts = np.concatenate([p, q])
    if parts.size != 7:
        return f"eef (obs): len={parts.size} {parts}"
    return "eef pos+quat (obs): " + " ".join(f"{v:+.4f}" for v in parts)


def action_from_keys(
    key: int,
    delta_scale: float,
    action_dim: int,
    orient_scale: float,
) -> np.ndarray:
    """Build action from last cv2.waitKey code; rotation via bracket keys."""
    a = np.zeros(action_dim, dtype=np.float64)
    if action_dim < 7:
        return a

    d3 = np.zeros(3, dtype=np.float64)
    if key == ord("w"):
        d3 += KEY_W_DELTA * delta_scale
    elif key == ord("s"):
        d3 += KEY_S_DELTA * delta_scale
    elif key == ord("a"):
        d3 += KEY_A_DELTA * delta_scale
    elif key == ord("d"):
        d3 += KEY_D_DELTA * delta_scale
    elif key == ord("q"):
        d3 += KEY_Q_DELTA * delta_scale
    elif key == ord("e"):
        d3 += KEY_E_DELTA * delta_scale

    a[0:3] = d3

    # Optional: nudge orientation with [ and ] (indices 3–5 = small shared delta)
    if key == ord("["):
        a[3:6] = -orient_scale
    elif key == ord("]"):
        a[3:6] = orient_scale

    if key == ord(","):
        a[6] = GRIPPER_OPEN_DELTA
    elif key == ord("."):
        a[6] = GRIPPER_CLOSE_DELTA

    return a


def clip_action(action: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.clip(action, low, high)


def run_loop(
    env,
    *,
    camera_key_preferred: str,
    delta_scale: float,
    orient_scale: float,
    display_width: int,
) -> None:
    action_dim = env.action_dim
    if action_dim != 7:
        raise SystemExit(
            f"This teleop expects action_dim==7 (OSC_POSE + Panda gripper); got {action_dim}. "
            "Use a different robot/controller or extend the key mapping."
        )

    low, high = env.action_spec
    low = np.asarray(low, dtype=np.float64).reshape(-1)
    high = np.asarray(high, dtype=np.float64).reshape(-1)

    win = "LIBERO obs teleop (ESC quit)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    from data_collection_new.env_factory import reset_env_with_retry

    obs = reset_env_with_retry(env)
    cam_key = pick_camera_obs_key(obs, camera_key_preferred)
    last_action = np.zeros(action_dim, dtype=np.float64)

    running = True
    while running:
        frame = observation_to_bgr_display(obs, cam_key, display_width)
        metrics_lines = [
            f"camera: {cam_key}",
            format_eef7(obs),
            f"action (cmd): {' '.join(f'{v:+.4f}' for v in last_action)}",
            "W/S up/down   A/D left/right   E forward / Q backward   [/] rot   ,/. gripper   ESC quit",
        ]
        display = compose_video_and_metrics_panel(frame, metrics_lines)

        cv2.imshow(win, display)
        try:
            prop = cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE)
            if prop < 1:
                break
        except cv2.error:
            pass

        raw = cv2.waitKey(1)
        key = 0 if raw == -1 else raw & 0xFF
        if key == 27:  # ESC
            break

        action = action_from_keys(key, delta_scale, action_dim, orient_scale)
        action = clip_action(action, low, high)
        last_action = action.copy()

        step_ret = env.step(action)
        if isinstance(step_ret, tuple) and len(step_ret) >= 4:
            obs = step_ret[0]
        else:
            obs = step_ret

        cam_key = pick_camera_obs_key(obs, camera_key_preferred)

    cv2.destroyWindow(win)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="OpenCV observation window + WASD/QE keyboard teleop.")
    parser.add_argument("--benchmark", default="libero_10")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--task-order-index", type=int, default=0)
    parser.add_argument("--env-configuration", default="single-arm-opposed")
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--camera-key", default="agentview_image")
    parser.add_argument(
        "--delta",
        type=float,
        default=0.25,
        help="Scale for position deltas (per key event / repeat tick)",
    )
    parser.add_argument(
        "--orient-scale",
        type=float,
        default=0.03,
        help="Magnitude for [ and ] rotation nudge on action[3:6]",
    )
    parser.add_argument(
        "--display-width",
        type=int,
        default=1280,
        help="Scale camera image to this width for the window (0 = native resolution, no scaling)",
    )
    parser.add_argument(
        "--camera-size",
        type=int,
        default=512,
        help="Square env camera resolution (height and width)",
    )
    args = parser.parse_args(argv)

    from data_collection_new.env_factory import build_libero_env, resolve_task

    _b, task, bddl_path = resolve_task(args.benchmark, args.task_id, args.task_order_index)

    cam = args.camera_size
    env = build_libero_env(
        bddl_path,
        env_configuration=args.env_configuration,
        use_window=False,
        control_freq=args.control_freq,
        use_camera_obs=True,
        camera_heights=cam,
        camera_widths=cam,
        ignore_done=True,
        reward_shaping=True,
    )
    print(f"task: {task.name} | {task.language}")
    print(f"bddl: {bddl_path}")

    try:
        run_loop(
            env,
            camera_key_preferred=args.camera_key,
            delta_scale=args.delta,
            orient_scale=args.orient_scale,
            display_width=args.display_width,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main(sys.argv[1:])
