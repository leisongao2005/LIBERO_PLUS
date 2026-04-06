"""
Create a LIBERO environment for a benchmark task index.

Example:
  python -m data_collection_new.make_env --benchmark libero_10 --task-id 0 --dry-run
  python -m data_collection_new.make_env --benchmark libero_plus_train_subtasks --task-id 3
"""

from __future__ import annotations

import argparse
import sys

from data_collection_new.env_factory import build_libero_env, resolve_task


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Instantiate a LIBERO robosuite env for benchmark task_id (ordered index)."
    )
    parser.add_argument(
        "--benchmark",
        default="libero_10",
        help="Registered benchmark name, e.g. libero_10, libero_plus_train_subtasks",
    )
    parser.add_argument("--task-id", type=int, required=True, help="Index into the ordered task list")
    parser.add_argument(
        "--task-order-index",
        type=int,
        default=0,
        help="Benchmark task permutation (see libero.libero.benchmark.task_orders)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only resolve BDDL path and task metadata (no robosuite / MuJoCo)",
    )
    parser.add_argument(
        "--window",
        action="store_true",
        help="Open an on-screen renderer (has_renderer=True); default is offscreen only",
    )
    parser.add_argument(
        "--env-configuration",
        default="single-arm-opposed",
        help="Passed to TwoArm-style problems only",
    )
    parser.add_argument(
        "--control-freq",
        type=int,
        default=20,
    )
    args = parser.parse_args(argv)

    _, task, bddl_path = resolve_task(args.benchmark, args.task_id, args.task_order_index)

    print(f"benchmark:        {args.benchmark}")
    print(f"task_order_index: {args.task_order_index}")
    print(f"task_id:          {args.task_id}")
    print(f"task_name:        {task.name}")
    print(f"language:         {task.language}")
    print(f"bddl_file:        {bddl_path}")

    if args.dry_run:
        return

    env = build_libero_env(
        bddl_path,
        env_configuration=args.env_configuration,
        use_window=args.window,
        control_freq=args.control_freq,
        camera_heights=512,
        camera_widths=512,
    )
    try:
        obs = env.reset()
        print(f"reset ok; observation keys: {list(obs.keys())}")
    finally:
        env.close()


if __name__ == "__main__":
    main(sys.argv[1:])
