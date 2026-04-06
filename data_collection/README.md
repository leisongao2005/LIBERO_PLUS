# Data Collection Framework

This directory contains the manual teleoperation collection framework for `libero_plus_train_subtasks`.

## Documentation

- `docs/SETUP.md`: environment setup, dependencies, and first-run checks
- `docs/USAGE.md`: GUI and CLI workflows for collection, packaging, validation, and export

## What Lives Here

- `config.py`: shared configuration and path resolution
- `models.py`: dataclasses shared across runtime, GUI, packaging, validation, and export
- `task_registry.py`: benchmark-backed task lookup
- `session_manager.py`: session and episode bookkeeping
- `teleop_runner.py`: library wrapper around robosuite keyboard / SpaceMouse collection
- `libero_packager.py`: conversion from raw teleop HDF5 to benchmark-compatible LIBERO HDF5
- `validation.py`: QC and dataset checks
- `pi05_exporter.py`: pi05-ready export bridge with per-episode payloads and quantile stats
- `gui_app.py`: Tkinter GUI for operators
- `cli.py`: scriptable command-line interface

## Storage Layout

- `runs/<session_id>/`: one collection session
- `runs/<session_id>/episodes/<episode_id>/raw/`: raw teleop artifacts and merged raw `demo.hdf5`
- `exports/pi05/<export_name>/`: exported pi05-ready dataset bridge

## Typical Workflow

1. Create a session.
2. Select a task from `libero_plus_train_subtasks`.
3. Launch teleoperation with keyboard or SpaceMouse.
4. Save the raw episode.
5. Package the raw episode into `datasets/libero_plus_train_subtasks/<task_name>_demo.hdf5`.
6. Validate the packaged dataset.
7. Export the packaged dataset into a pi05-ready dataset directory.

## CLI Examples

List tasks:

```bash
python -m data_collection.cli list-tasks
```

Create a session:

```bash
python -m data_collection.cli start-session --device keyboard --target-per-task 5
```

Collect one task:

```bash
python -m data_collection.cli collect-task \
  --session-id <session_id> \
  --task-name <task_name>
```

Package a raw run:

```bash
python -m data_collection.cli package-task \
  --task-name <task_name> \
  --raw-hdf5 <path_to_raw_demo_hdf5> \
  --overwrite
```

Validate:

```bash
python -m data_collection.cli validate-task --dataset <packaged_hdf5>
```

Export to pi05:

```bash
python -m data_collection.cli export-pi05 \
  --export-name pi05_export \
  --task-name <task_name> \
  --dataset <packaged_hdf5>
```

## GUI

Launch the GUI with:

```bash
python -m data_collection.gui_app
```

The GUI is a session controller and dashboard. Live teleoperation still happens in the robosuite viewer window.

## Notes

- The intended action space is 7D in `OSC_POSE`: `3` translation, `3` orientation, `1` gripper.
- Keep collection modes consistent. Do not mix 7D `OSC_POSE` data with reduced 4D action data in the same task dataset.
- The pi05 exporter writes a deterministic bridge dataset with manifests, quantiles, and per-episode payloads so downstream conversion / ingestion stays reproducible.
