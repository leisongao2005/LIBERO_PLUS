from __future__ import annotations

import argparse
import json

from data_collection.config import DataCollectionConfig
from data_collection.session_manager import SessionManager
from data_collection.task_registry import TaskRegistry


def _build_runtime(include_collection: bool = False):
    config = DataCollectionConfig.default()
    session_manager = SessionManager(config)
    task_registry = TaskRegistry(config)
    teleop_runner = None
    packager = None
    validator = None
    exporter = None
    if include_collection:
        from data_collection.libero_packager import LiberoPackager
        from data_collection.pi05_exporter import Pi05Exporter
        from data_collection.teleop_runner import TeleopRunner
        from data_collection.validation import DatasetValidator

        teleop_runner = TeleopRunner(config, session_manager)
        packager = LiberoPackager(config)
        validator = DatasetValidator(config)
        exporter = Pi05Exporter(config)
    return config, session_manager, task_registry, teleop_runner, packager, validator, exporter


def main():
    parser = argparse.ArgumentParser(description="Data collection framework CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-tasks")

    start_session = subparsers.add_parser("start-session")
    start_session.add_argument("--device", default="keyboard")
    start_session.add_argument("--camera", default="agentview")
    start_session.add_argument("--target-per-task", type=int, default=10)
    start_session.add_argument("--operator", default="")
    start_session.add_argument("--notes", default="")

    collect = subparsers.add_parser("collect-task")
    collect.add_argument("--session-id", required=True)
    collect.add_argument("--task-name", required=True)
    collect.add_argument("--num-demonstrations", type=int, default=1)
    collect.add_argument("--notes", default="")

    package = subparsers.add_parser("package-task")
    package.add_argument("--task-name", required=True)
    package.add_argument("--raw-hdf5", required=True)
    package.add_argument("--overwrite", action="store_true")

    validate = subparsers.add_parser("validate-task")
    validate.add_argument("--dataset", required=True)
    validate.add_argument("--task-name")

    export = subparsers.add_parser("export-pi05")
    export.add_argument("--export-name", required=True)
    export.add_argument("--task-name", action="append", required=True)
    export.add_argument("--dataset", action="append", required=True)
    export.add_argument("--state-schema", default="ee_gripper")

    subparsers.add_parser("show-status")

    args = parser.parse_args()
    include_collection = args.command in {"collect-task", "package-task", "validate-task", "export-pi05"}
    _, session_manager, task_registry, teleop_runner, packager, validator, exporter = _build_runtime(
        include_collection=include_collection
    )

    if args.command == "list-tasks":
        print(json.dumps([task.to_dict() for task in task_registry.load_tasks()], indent=2))
        return

    if args.command == "start-session":
        session = session_manager.create_session(
            device=args.device,
            camera=args.camera,
            target_per_task=args.target_per_task,
            operator=args.operator,
            notes=args.notes,
        )
        print(json.dumps(session.to_dict(), indent=2))
        return

    if args.command == "collect-task":
        session = session_manager.load_session(args.session_id)
        task = task_registry.get_task(args.task_name)
        result = teleop_runner.collect_task(
            session=session,
            task=task,
            num_demonstrations=args.num_demonstrations,
            notes=args.notes,
        )
        print(json.dumps(result.to_dict(), indent=2))
        return

    if args.command == "package-task":
        task = task_registry.get_task(args.task_name)
        result = packager.package_task(task, args.raw_hdf5, overwrite=args.overwrite)
        print(json.dumps(result.to_dict(), indent=2))
        return

    if args.command == "validate-task":
        result = validator.validate_packaged_dataset(args.dataset, expected_task_name=args.task_name)
        print(json.dumps(result.to_dict(), indent=2))
        return

    if args.command == "export-pi05":
        tasks = [task_registry.get_task(task_name) for task_name in args.task_name]
        result = exporter.export_tasks(
            tasks=tasks,
            packaged_paths=args.dataset,
            export_name=args.export_name,
            state_schema=args.state_schema,
        )
        print(json.dumps(result.to_dict(), indent=2))
        return

    if args.command == "show-status":
        payload = {
            "sessions": [session.to_dict() for session in session_manager.list_sessions()],
            "tasks": [task.to_dict() for task in task_registry.load_tasks()],
        }
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
