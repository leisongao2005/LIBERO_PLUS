"""
LIBERO-Plus interactive teleoperation with live predicate state display.

Like collect_demonstration.py but prints a live L1/L2/L3/L4 predicate dashboard
to the terminal every step. No demonstrations are saved — this is a debug/verify
tool for confirming that hierarchical reward signals fire correctly.

Usage:
    python scripts/teleop_with_predicates.py --task floor10 [--device keyboard]
    python scripts/teleop_with_predicates.py --task basket_scene1030 [--device spacemouse]

Tasks:
    basket_scene1030  — pure pick-and-place (L2 can fire for both subtasks)
    floor10           — mixed Turnon + On (L2::turnon_stove always False)

Controls:
    keyboard / spacemouse  — drive the robot
    'q' key or action=None — reset the episode
    Ctrl+C                 — quit
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
import time
from typing import Dict, List, Sequence

import init_path  # noqa: F401  — inserts repo root onto sys.path
import torch

import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import TASK_MAPPING
from libero.libero.envs.wrappers._delta import (
    apply_one_shot_latch,
    compute_shaped_reward,
)
from libero.libero.envs.wrappers._status_string import format_status_string
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import VisualizationWrapper

# ---------------------------------------------------------------------------
# ANSI colour helpers
# ---------------------------------------------------------------------------

_RESET = "\033[0m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_CYAN = "\033[36m"


def _coloured_bool(value: bool, newly_fired: bool = False) -> str:
    """Return a coloured string for a boolean predicate value."""
    if newly_fired and value:
        return f"{_YELLOW}True {_RESET}"
    if value:
        return f"{_GREEN}True {_RESET}"
    return f"{_DIM}False{_RESET}"


def _banner(text: str, width: int = 55) -> str:
    bar = "═" * width
    return f"{_BOLD}{bar}\n{text}\n{bar}{_RESET}"


# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).parent.parent

TASK_CONFIGS = {
    "basket_scene1030": {
        "bddl": (
            REPO_ROOT
            / "libero/libero/bddl_files/libero_plus_train_subtasks"
            / "LIVING_ROOM_TABLETOP_BASKET_SCENE1030"
            "_put_the_cream_cheese_in_the_basket"
            "_and_put_the_alphabet_soup_in_the_basket.bddl"
        ),
        "init": (
            REPO_ROOT
            / "libero/libero/init_files/libero_plus_train_subtasks"
            / "LIVING_ROOM_TABLETOP_BASKET_SCENE1030"
            "_put_the_cream_cheese_in_the_basket"
            "_and_put_the_alphabet_soup_in_the_basket.pruned_init"
        ),
    },
    "floor10": {
        "bddl": (
            REPO_ROOT
            / "libero/libero/bddl_files/libero_plus_train_subtasks"
            / "FLOOR_10_turn_on_the_stove_and_put_the_left_moka_pot_on_it.bddl"
        ),
        "init": (
            REPO_ROOT
            / "libero/libero/init_files/libero_plus_train_subtasks"
            / "FLOOR_10_turn_on_the_stove_and_put_the_left_moka_pot_on_it.pruned_init"
        ),
    },
}


# ---------------------------------------------------------------------------
# BDDL parsing helpers
# ---------------------------------------------------------------------------


def parse_subtask_names(bddl_path: pathlib.Path) -> List[str]:
    """Extract subtask names in BDDL declaration order via regex."""
    content = bddl_path.read_text()
    return re.findall(r"\(:subtask\s+(\S+)", content)


# ---------------------------------------------------------------------------
# Episode state helpers
# ---------------------------------------------------------------------------


def _make_fresh_episode_state(subtask_names: Sequence[str]) -> dict:
    """Return a clean per-episode mutable state dict."""
    return {
        "l3_history": {},
        "l1_history": {},
        "l2_history": {},
        "l4_ever_fired": False,
        "cumulative_shaped": 0.0,
        "step": 0,
    }


# ---------------------------------------------------------------------------
# Dashboard renderer
# ---------------------------------------------------------------------------

_WEIGHTS: Dict[str, float] = {"L1": 0.1, "L2": 0.2, "L3": 0.5, "L4": 1.0}


def render_dashboard(
    *,
    step: int,
    language_instruction: str,
    subtask_names: Sequence[str],
    raw: Dict[str, bool],
    l3_history: Dict[str, bool],
    sparse_reward: float,
    step_shaped: float,
    cumulative_shaped: float,
    newly_fired_this_step: Dict[str, bool],
    status: str,
) -> None:
    """Clear the terminal and print a live predicate dashboard."""
    sys.stdout.write("\033[H\033[J")
    sys.stdout.flush()

    width = 55
    print(_banner(f"LIBERO-Plus Predicate Debugger  [step {step}]", width))
    print(f"Task: {_CYAN}{language_instruction}{_RESET}")
    print(f"Status: {_YELLOW}{status}{_RESET}")
    print()

    # --- Predicate table ---
    col_w = 22
    hdr = f"  {'Subtask':<{col_w}} {'L1':<9} {'L2':<9} {'L3 (now)':<9} {'L3 ever'}"
    print("Predicate State:")
    print(hdr)
    print("  " + "─" * (width - 2))
    for s in subtask_names:
        l1_val = raw.get(f"L1::{s}", False)
        l2_val = raw.get(f"L2::{s}", False)
        l3_val = raw.get(f"L3::{s}", False)
        l3_ever = l3_history.get(s, False)

        l1_new = newly_fired_this_step.get(f"L1::{s}", False)
        l2_new = newly_fired_this_step.get(f"L2::{s}", False)
        l3_new = newly_fired_this_step.get(f"L3::{s}", False)
        l3_ever_new = newly_fired_this_step.get(f"L3::{s}", False)

        row = (
            f"  {s:<{col_w}} "
            f"{_coloured_bool(l1_val, l1_new):<18} "
            f"{_coloured_bool(l2_val, l2_new):<18} "
            f"{_coloured_bool(l3_val, l3_new):<18} "
            f"{_coloured_bool(l3_ever, l3_ever_new)}"
        )
        print(row)

    l4_val = raw.get("L4", False)
    l4_new = newly_fired_this_step.get("L4", False)
    print()
    print(f"  L4 (goal): {_coloured_bool(l4_val, l4_new)}")
    print()

    # --- Reward summary ---
    sign = "+" if step_shaped >= 0 else ""
    print("Reward this step:")
    print(f"  Sparse reward:      {sparse_reward:.1f}")
    print(f"  Shaped (+delta):    {sign}{step_shaped:.2f}")
    print(f"  Cumulative shaped:  {cumulative_shaped:.2f}")
    print()

    # --- Newly fired labels ---
    fired_labels = [k for k, v in newly_fired_this_step.items() if v]
    if fired_labels:
        print(f"Newly fired this step: {_YELLOW}{', '.join(fired_labels)}{_RESET}")
    else:
        print(f"Newly fired this step: {_DIM}(none){_RESET}")

    print()
    print(_banner("Controls: SpaceNav/keyboard | 'q' reset | Ctrl+C quit", width))
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main teleoperation loop
# ---------------------------------------------------------------------------


def run_teleop(
    env,
    device,
    arm: str,
    env_configuration: str,
    init_states,
    subtask_names: List[str],
    language_instruction: str,
) -> None:
    """Interactive teleoperation loop with live predicate display.

    Does NOT save demonstrations. Runs until Ctrl+C.
    """
    episode_state = _make_fresh_episode_state(subtask_names)
    l1_keys = [f"L1::{s}" for s in subtask_names]
    l2_keys = [f"L2::{s}" for s in subtask_names]
    l3_keys = [f"L3::{s}" for s in subtask_names]

    def _apply_init_state(state_vec) -> None:
        """Load a flattened MuJoCo state into the env and re-derive observations.

        Mirrors what ``ControlEnv.set_init_state`` does, but works on the raw
        robosuite env exposed through VisualizationWrapper's __getattr__ passthrough.
        """
        env.sim.set_state_from_flattened(state_vec)
        env.sim.forward()
        env._post_process()
        env._update_observables(force=True)

    def _reset_episode() -> None:
        nonlocal episode_state
        episode_state = _make_fresh_episode_state(subtask_names)
        reset_ok = False
        while not reset_ok:
            try:
                env.reset()
                reset_ok = True
            except Exception:
                continue
        # .pruned_init files may be saved as numpy arrays or torch tensors             
        arr = init_states[0]                                                           
        _apply_init_state(arr.numpy() if isinstance(arr, torch.Tensor) else arr)
        env.render()
        device.start_control()
        print("Episode reset. Ready.")

    _reset_episode()

    try:
        while True:
            # --- Determine active robot ---
            active_robot = (
                env.robots[0]
                if env_configuration == "bimanual"
                else env.robots[arm == "left"]
            )

            # --- Get action ---
            action, _grasp = input2action(
                device=device,
                robot=active_robot,
                active_arm=arm,
                env_configuration=env_configuration,
            )

            # action=None means the user pressed the reset key ('q' on keyboard)
            if action is None:
                print("Resetting episode...")
                time.sleep(0.5)
                _reset_episode()
                continue

            # --- Step environment ---
            _obs, sparse_reward, _done, info = env.step(action)

            raw: Dict[str, bool] = info.get("raw_predicates", {})

            # --- L1/L2 one-shot latch (fire at most once per episode, prevents farming) ---
            episode_state["l1_history"], l1_fired = apply_one_shot_latch(
                episode_state["l1_history"], raw, l1_keys
            )
            episode_state["l2_history"], l2_fired = apply_one_shot_latch(
                episode_state["l2_history"], raw, l2_keys
            )

            # --- One-shot latch (L3 subtasks) ---
            # apply_one_shot_latch uses the full "L3::name" keys;
            # format_status_string expects plain "name" keys, so we keep a parallel
            # plain-key copy for the status string.
            new_l3_history, l3_fired = apply_one_shot_latch(
                episode_state["l3_history"], raw, l3_keys
            )
            episode_state["l3_history"] = new_l3_history
            # Build plain-key history for format_status_string
            l3_history_plain = {
                k[len("L3::"):]: v for k, v in new_l3_history.items()
            }

            # --- One-shot latch (L4 terminal) ---
            l4_this_step_fired: Dict[str, bool] = {}
            if raw.get("L4", False) and not episode_state["l4_ever_fired"]:
                l4_this_step_fired["L4"] = True
                episode_state["l4_ever_fired"] = True

            # --- Shaped reward ---
            step_shaped = compute_shaped_reward(
                {**l1_fired, **l2_fired},
                {**l3_fired, **l4_this_step_fired},
                _WEIGHTS,
                subtask_names,
            )
            episode_state["cumulative_shaped"] += step_shaped

            # --- Status string ---
            # format_status_string expects plain subtask names as l3_history keys
            status = format_status_string(
                subtask_names, l3_history_plain, raw
            )

            episode_state["step"] += 1

            # --- Collect newly-fired events for highlighting ---
            newly_fired: Dict[str, bool] = {}
            newly_fired.update({k: v for k, v in l1_fired.items() if v})
            newly_fired.update({k: v for k, v in l2_fired.items() if v})
            newly_fired.update({k: v for k, v in l3_fired.items() if v})
            newly_fired.update({k: v for k, v in l4_this_step_fired.items() if v})

            # --- Render dashboard ---
            render_dashboard(
                step=episode_state["step"],
                language_instruction=language_instruction,
                subtask_names=subtask_names,
                raw=raw,
                l3_history=l3_history_plain,
                sparse_reward=float(sparse_reward),
                step_shaped=step_shaped,
                cumulative_shaped=episode_state["cumulative_shaped"],
                newly_fired_this_step=newly_fired,
                status=status,
            )

            env.render()

            # --- Goal reached: freeze display, stop stepping, wait for reset ---
            if episode_state["l4_ever_fired"]:
                sys.stdout.write("\033[H\033[J")
                sys.stdout.flush()
                total = episode_state["cumulative_shaped"]
                print(_banner(f"GOAL REACHED!  [step {episode_state['step']}]", 55))
                print(f"{_GREEN}{_BOLD}All subtasks complete.{_RESET}")
                print(f"Cumulative shaped reward: {_CYAN}{total:.2f}{_RESET}")
                print()
                print("Press 'q' to reset or Ctrl+C to quit.")
                sys.stdout.flush()
                # Wait without calling env.step() or render_dashboard again.
                while True:
                    wait_robot = (
                        env.robots[0]
                        if env_configuration == "bimanual"
                        else env.robots[arm == "left"]
                    )
                    wait_action, _ = input2action(
                        device=device,
                        robot=wait_robot,
                        active_arm=arm,
                        env_configuration=env_configuration,
                    )
                    if wait_action is None:
                        break
                    env.render()
                _reset_episode()
                continue

    except KeyboardInterrupt:
        print("\nExiting teleop. Goodbye.")
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LIBERO-Plus teleoperation with live predicate display."
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=list(TASK_CONFIGS.keys()),
        help="Which task to run: 'basket_scene1030' or 'floor10'.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="keyboard",
        choices=["keyboard", "spacemouse"],
        help="Input device. Default: keyboard.",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="right",
        help="Which arm to control for bimanual envs. Default: right.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="single-arm-opposed",
        help="Robosuite env_configuration. Default: single-arm-opposed.",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="OSC_POSE",
        help="Controller type. Default: OSC_POSE.",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="agentview",
        help="Render camera name. Default: agentview.",
    )
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=1.5,
        help="Position scaling for input device. Default: 1.5.",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=3.0,
        help="Rotation scaling for input device. Default: 3.0.",
    )
    parser.add_argument(
        "--vendor-id",
        type=int,
        default=9583,
        help="USB vendor ID for SpaceMouse. Default: 9583.",
    )
    parser.add_argument(
        "--product-id",
        type=int,
        default=50734,
        help="USB product ID for SpaceMouse. Default: 50734.",
    )
    args = parser.parse_args()

    cfg = TASK_CONFIGS[args.task]
    bddl_path: pathlib.Path = cfg["bddl"]
    init_path_file: pathlib.Path = cfg["init"]

    if not bddl_path.exists():
        sys.exit(f"ERROR: BDDL file not found: {bddl_path}")
    if not init_path_file.exists():
        sys.exit(f"ERROR: Init file not found: {init_path_file}")

    # --- Parse subtask names ---
    subtask_names = parse_subtask_names(bddl_path)
    if not subtask_names:
        sys.exit(f"ERROR: No :subtask entries found in {bddl_path}")
    print(f"Subtask names parsed from BDDL: {subtask_names}")

    # --- Parse problem info ---
    problem_info = BDDLUtils.get_problem_info(str(bddl_path))
    problem_name = problem_info["problem_name"]
    domain_name = problem_info["domain_name"]
    language_instruction = problem_info["language_instruction"]
    print(f"Task: {language_instruction}")

    # --- Build controller config ---
    controller_config = load_controller_config(default_controller=args.controller)
    env_config = {
        "robots": ["Panda"],
        "controller_configs": controller_config,
    }
    if "TwoArm" in problem_name:
        env_config["env_configuration"] = args.config

    # --- Create environment (same pattern as collect_demonstration.py) ---
    env = TASK_MAPPING[problem_name](
        bddl_file_name=str(bddl_path),
        **env_config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=False,
        control_freq=20,
    )

    # Wrap with VisualizationWrapper for on-screen rendering
    env = VisualizationWrapper(env)

    # --- Load init states ---
    init_states = torch.load(str(init_path_file), weights_only=False)

    # --- Initialise input device ---
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
        env.viewer.add_keypress_callback(device.on_press)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(
            args.vendor_id,
            args.product_id,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
    else:
        sys.exit(f"Unknown device: {args.device}")

    # --- Run ---
    run_teleop(
        env=env,
        device=device,
        arm=args.arm,
        env_configuration=args.config,
        init_states=init_states,
        subtask_names=subtask_names,
        language_instruction=language_instruction,
    )


if __name__ == "__main__":
    main()
