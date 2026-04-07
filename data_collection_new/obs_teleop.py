"""
OpenCV window: agent camera (left) + gripper eye-in-hand (right), metrics strip below,
and keyboard deltas for OSC_POSE control.

Run (from repo root, with LIBERO deps installed):
  python -m data_collection_new.obs_teleop --benchmark libero_10 --task-id 0

Axis legend (position deltas, OSC dx,dy,dz; tune signs in *_DELTA below):
  W / S  — up / down (+Z / -Z)
  A / D  — left / right (-X / +X)
  Q / E  — backward / forward (-Y / +Y)
  U / O  — rotation about axis 0 (OSC drot_0) + / −
  I / K  — rotation about axis 1 (drot_1) + / −
  J / L  — rotation about axis 2 (drot_2) + / −
  ] / [  — gripper open / close (also < open, > close)

Multiple movement keys work together (e.g. W+E) when **pynput** is installed; otherwise pass
``--use-waitkey`` for single-key mode (OpenCV only).

The 7D **action** vector is OSC_POSE: (dx, dy, dz, drot_0, drot_1, drot_2, gripper).
Metrics (bottom panel) include observation-space eef: 3 pos + 4 quat from obs,
not the same as the action semantics.

ESC — quit.

Dependencies: opencv-python (cv2), pynput (recommended), robosuite, torch (via libero imports), MuJoCo.
"""

from __future__ import annotations

import argparse
import sys
import threading
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import cv2
import numpy as np

# --- Keyboard → position delta mapping (OSC: action[0]=dx, [1]=dy, [2]=dz) ------------
# W=up, D=right, E=forward (tune signs here if inverted on your machine).
KEY_W_DELTA = np.array([-1.0, 0.0, 0.0])  # up +Z
KEY_S_DELTA = np.array([1.0, 0.0, 0.0])  # down
KEY_A_DELTA = np.array([0.0, -1.0, 0.0])  # left -X
KEY_D_DELTA = np.array([0.0, 1.0, 0.0])  # right +X
KEY_Q_DELTA = np.array([0.0, 0.0, 1.0])  # backward -Y
KEY_E_DELTA = np.array([0.0, 0.0, -1.0])  # forward +Y (same scale as other axes)

GRIPPER_OPEN_DELTA = 0.08
GRIPPER_CLOSE_DELTA = -0.08

# Keys forwarded into the held-key set (pynput); lowercase chars
TRACKED_CHARS: FrozenSet[str] = frozenset("wasdqeuoikjl[]<>")


class GlobalKeyState:
    """Thread-safe set of held characters (for combining W+E, etc.)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._chars: Set[str] = set()
        self._quit = False
        self._listener = None
        self._tracked: FrozenSet[str] = TRACKED_CHARS

    def snapshot(self) -> Tuple[FrozenSet[str], bool]:
        with self._lock:
            return frozenset(self._chars), self._quit

    def _char_from_event(self, key) -> Optional[str]:
        try:
            if hasattr(key, "char") and key.char is not None:
                return key.char.lower()
        except (AttributeError, UnicodeDecodeError):
            pass
        return None

    def _on_press(self, key) -> None:
        from pynput.keyboard import Key

        if key == Key.esc:
            with self._lock:
                self._quit = True
            return
        ch = self._char_from_event(key)
        if ch and ch in self._tracked:
            with self._lock:
                self._chars.add(ch)

    def _on_release(self, key) -> None:
        from pynput.keyboard import Key

        if key == Key.esc:
            return
        ch = self._char_from_event(key)
        if ch and ch in self._tracked:
            with self._lock:
                self._chars.discard(ch)

    def start(self) -> None:
        from pynput.keyboard import Listener

        self._listener = Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None


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


def pick_gripper_camera_key(obs: Dict[str, Any], preferred: str) -> Optional[str]:
    if preferred in obs:
        return preferred
    for k in sorted(obs.keys()):
        if k.endswith("_image") and "eye_in_hand" in k:
            return k
    return None


def observation_to_bgr_display(
    obs: Dict[str, Any],
    camera_key: str,
    *,
    scale_width: int = 0,
    scale_height: int = 0,
) -> np.ndarray:
    """RGB obs → BGR uint8. Scale by target width (priority) or target height; 0 = native."""
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
    if scale_width > 0 and w != scale_width:
        new_w = scale_width
        new_h = max(1, int(round(h * (scale_width / float(w)))))
        interp = cv2.INTER_AREA if w > scale_width else cv2.INTER_CUBIC
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=interp)
    elif scale_height > 0 and h != scale_height:
        new_h = scale_height
        new_w = max(1, int(round(w * (scale_height / float(h)))))
        interp = cv2.INTER_AREA if h > scale_height else cv2.INTER_CUBIC
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=interp)
    return np.ascontiguousarray(img_bgr, dtype=np.uint8)


def stack_agent_and_gripper_row(
    obs: Dict[str, Any],
    main_key: str,
    gripper_key: Optional[str],
    display_width: int,
) -> Tuple[np.ndarray, Optional[str]]:
    """Agent view on the left, gripper (eye-in-hand) POV on the right; same row height."""
    main = observation_to_bgr_display(
        obs,
        main_key,
        scale_width=display_width if display_width > 0 else 0,
    )
    if not gripper_key:
        return main, None
    mh = main.shape[0]
    gri = observation_to_bgr_display(obs, gripper_key, scale_height=mh)
    gh, gw = gri.shape[:2]
    if gh != mh:
        new_w = max(1, int(round(gw * (mh / float(gh)))))
        gri = cv2.resize(gri, (new_w, mh), interpolation=cv2.INTER_AREA)
        gri = np.ascontiguousarray(gri, dtype=np.uint8)
    # Eye-in-hand often feels inverted vs agent view: flip horizontal + vertical
    gri = np.ascontiguousarray(cv2.flip(gri, -1), dtype=np.uint8)
    sep_w = 4
    sep = np.full((mh, sep_w, 3), 48, dtype=np.uint8)
    row = np.hstack([main, sep, gri])
    return np.ascontiguousarray(row, dtype=np.uint8), gripper_key


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


def action_from_held_chars(
    chars: FrozenSet[str],
    *,
    delta_scale: float,
    orient_scale: float,
    action_dim: int,
) -> np.ndarray:
    """Sum deltas for every key currently held (W+E+… supported)."""
    a = np.zeros(action_dim, dtype=np.float64)
    if action_dim < 7:
        return a

    d3 = np.zeros(3, dtype=np.float64)
    if "w" in chars:
        d3 += KEY_W_DELTA * delta_scale
    if "s" in chars:
        d3 += KEY_S_DELTA * delta_scale
    if "a" in chars:
        d3 += KEY_A_DELTA * delta_scale
    if "d" in chars:
        d3 += KEY_D_DELTA * delta_scale
    if "q" in chars:
        d3 += KEY_Q_DELTA * delta_scale
    if "e" in chars:
        d3 += KEY_E_DELTA * delta_scale
    a[0:3] = d3

    if "u" in chars:
        a[3] += orient_scale
    if "o" in chars:
        a[3] -= orient_scale
    if "i" in chars:
        a[4] += orient_scale
    if "k" in chars:
        a[4] -= orient_scale
    if "j" in chars:
        a[5] += orient_scale
    if "l" in chars:
        a[5] -= orient_scale

    if "]" in chars or "<" in chars:
        a[6] += GRIPPER_OPEN_DELTA
    if "[" in chars or ">" in chars:
        a[6] += GRIPPER_CLOSE_DELTA

    return a


def action_from_waitkey(
    key: int,
    *,
    delta_scale: float,
    orient_scale: float,
    action_dim: int,
) -> np.ndarray:
    """Single-key fallback when pynput is disabled (no simultaneous keys)."""
    if key == 0:
        return np.zeros(action_dim, dtype=np.float64)
    ch = chr(key) if 0 < key < 128 else ""
    if len(ch) == 1 and ch.isalpha():
        ch = ch.lower()
    if ch not in TRACKED_CHARS:
        return np.zeros(action_dim, dtype=np.float64)
    return action_from_held_chars(
        frozenset({ch}),
        delta_scale=delta_scale,
        orient_scale=orient_scale,
        action_dim=action_dim,
    )


def clip_action(action: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.clip(action, low, high)


def pynput_available() -> bool:
    try:
        import pynput  # noqa: F401

        return True
    except ImportError:
        return False


def run_loop(
    env,
    *,
    camera_key_preferred: str,
    gripper_camera_preferred: str,
    show_gripper_panel: bool,
    delta_scale: float,
    orient_scale: float,
    display_width: int,
    use_pynput_keys: bool,
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
    if show_gripper_panel and pick_gripper_camera_key(obs, gripper_camera_preferred) is None:
        print(
            "[obs_teleop] No gripper camera in obs (expected e.g. robot0_eye_in_hand_image). "
            "Right panel omitted until present. Ensure env uses camera_names with robot0_eye_in_hand."
        )

    last_action = np.zeros(action_dim, dtype=np.float64)

    key_state: Optional[GlobalKeyState] = None
    if use_pynput_keys:
        key_state = GlobalKeyState()
        key_state.start()

    running = True
    try:
        while running:
            gk = pick_gripper_camera_key(obs, gripper_camera_preferred) if show_gripper_panel else None
            frame, _ = stack_agent_and_gripper_row(obs, cam_key, gk, display_width)
            metrics_lines = [
                f"views: {cam_key}" + (f"  |  {gk}" if gk else ""),
                format_eef7(obs),
                f"action (cmd): {' '.join(f'{v:+.4f}' for v in last_action)}",
                "Pos: WASD QE   Rot: U/O I/K J/L   Grip: ]/[ or </> open/close   ESC quit"
                + ("   (pynput: multi-key)" if use_pynput_keys else "   (--use-waitkey: one key)"),
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

            if key_state is not None:
                held, quit_from_pynput = key_state.snapshot()
                if quit_from_pynput:
                    break
                action = action_from_held_chars(
                    held,
                    delta_scale=delta_scale,
                    orient_scale=orient_scale,
                    action_dim=action_dim,
                )
            else:
                action = action_from_waitkey(
                    key,
                    delta_scale=delta_scale,
                    orient_scale=orient_scale,
                    action_dim=action_dim,
                )

            action = clip_action(action, low, high)
            last_action = action.copy()

            step_ret = env.step(action)
            if isinstance(step_ret, tuple) and len(step_ret) >= 4:
                obs = step_ret[0]
            else:
                obs = step_ret

            cam_key = pick_camera_obs_key(obs, camera_key_preferred)

    finally:
        if key_state is not None:
            key_state.stop()

    cv2.destroyWindow(win)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="OpenCV observation window + WASD/QE keyboard teleop.")
    parser.add_argument("--benchmark", default="libero_10")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--task-order-index", type=int, default=0)
    parser.add_argument("--env-configuration", default="single-arm-opposed")
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--camera-key", default="agentview_image", help="Left (main) camera obs key")
    parser.add_argument(
        "--gripper-camera-key",
        default="robot0_eye_in_hand_image",
        help="Right panel: eye-in-hand image key",
    )
    parser.add_argument(
        "--no-gripper-panel",
        action="store_true",
        help="Disable right-hand gripper POV (agent view only)",
    )
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
        help="Per-step rotation magnitude for U/O (axis0), I/K (axis1), J/L (axis2)",
    )
    parser.add_argument(
        "--use-waitkey",
        action="store_true",
        help="Disable pynput; only one key per frame (no W+E combos). ESC still works.",
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
    cam_names: Optional[List[str]] = None
    if not args.no_gripper_panel:
        cam_names = ["agentview", "robot0_eye_in_hand"]

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
        camera_names=cam_names,
    )
    print(f"task: {task.name} | {task.language}")
    print(f"bddl: {bddl_path}")

    use_pynput_keys = not args.use_waitkey and pynput_available()
    if not args.use_waitkey and not pynput_available():
        print(
            "[obs_teleop] pynput not installed — install with `pip install pynput` for combined keys "
            "(e.g. W+E). Using single-key mode until then."
        )

    try:
        run_loop(
            env,
            camera_key_preferred=args.camera_key,
            gripper_camera_preferred=args.gripper_camera_key,
            show_gripper_panel=not args.no_gripper_panel,
            delta_scale=args.delta,
            orient_scale=args.orient_scale,
            display_width=args.display_width,
            use_pynput_keys=use_pynput_keys,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main(sys.argv[1:])
