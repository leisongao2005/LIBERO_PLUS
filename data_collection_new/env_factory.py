"""Shared LIBERO benchmark resolution and robosuite env construction."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union


def resolve_task(
    benchmark_name: str, task_id: int, task_order_index: int
) -> Tuple[Any, Any, str]:
    from libero.libero.benchmark import get_benchmark

    benchmark_cls = get_benchmark(benchmark_name)
    benchmark = benchmark_cls(task_order_index=task_order_index)
    n = benchmark.get_num_tasks()
    if task_id < 0 or task_id >= n:
        raise SystemExit(
            f"task_id must be in [0, {n - 1}] for benchmark {benchmark_name!r} (got {task_id})"
        )
    task = benchmark.get_task(task_id)
    bddl_path = benchmark.get_task_bddl_file_path(task_id)
    return benchmark, task, bddl_path


def build_libero_env(
    bddl_path: str,
    *,
    env_configuration: str = "single-arm-opposed",
    use_window: bool = False,
    control_freq: int = 20,
    use_camera_obs: bool = True,
    camera_heights: int = 128,
    camera_widths: int = 128,
    ignore_done: bool = False,
    reward_shaping: bool = False,
    controller: str = "OSC_POSE",
    camera_names: Optional[Union[str, List[str], Tuple[str, ...]]] = None,
):
    import libero.libero.envs.bddl_utils as BDDLUtils
    from libero.libero.envs import TASK_MAPPING
    import robosuite as suite

    problem_info = BDDLUtils.get_problem_info(bddl_path)
    problem_name = problem_info["problem_name"]

    controller_configs = suite.load_controller_config(default_controller=controller)

    kwargs: dict = {
        "bddl_file_name": bddl_path,
        "robots": ["Panda"],
        "controller_configs": controller_configs,
        "has_renderer": use_window,
        "has_offscreen_renderer": not use_window,
        "control_freq": control_freq,
        "use_camera_obs": use_camera_obs,
        "camera_heights": camera_heights,
        "camera_widths": camera_widths,
        "ignore_done": ignore_done,
        "reward_shaping": reward_shaping,
    }

    if camera_names is not None:
        kwargs["camera_names"] = camera_names

    if "TwoArm" in problem_name:
        kwargs["env_configuration"] = env_configuration

    return TASK_MAPPING[problem_name](**kwargs)


def reset_env_with_retry(env):
    """Reset until placement succeeds (same idea as ControlEnv.reset)."""
    from robosuite.utils.errors import RandomizationError

    while True:
        try:
            return env.reset()
        except RandomizationError:
            continue
