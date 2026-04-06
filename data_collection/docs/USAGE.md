# Data Collection Usage

This guide covers the normal operator workflow for collecting data, packaging it, validating it, and exporting it for `pi05`.

## Concepts

The framework has three main artifact types:

- raw collection artifacts:
  - stored under `data_collection/runs/<session_id>/episodes/<episode_id>/raw/`
  - includes the merged raw `demo.hdf5`
- packaged LIBERO datasets:
  - stored under `datasets/libero_plus_train_subtasks/<task_name>_demo.hdf5`
  - used by the existing LIBERO training stack
- pi05 export datasets:
  - stored under `data_collection/exports/pi05/<export_name>/`
  - used as the bridge format for downstream `pi05` fine-tuning

## Action Convention

The intended collection action is 7D:

- `3` end-effector translation values
- `3` end-effector orientation values
- `1` gripper value

Do not mix reduced 4D action data with this 7D mode in the same dataset.

## GUI Workflow

Launch the GUI:

```bash
python -m data_collection.gui_app
```

Recommended operator flow:

1. Click `Create Session`
2. Select device:
   - `keyboard` for default teleop
   - `spacemouse` if available
3. Select a task in the task table
4. Click `Collect Selected Task`
5. Use the robosuite window to perform the task
6. After collection finishes, click:
   - `Package Latest Raw Episode`
   - `Validate Packaged Dataset`
   - `Export Selected Task To pi05`

The right-hand details panel shows the latest result payload.

## CLI Workflow

### 1. List tasks

```bash
python -m data_collection.cli list-tasks
```

### 2. Start a session

```bash
python -m data_collection.cli start-session \
  --device keyboard \
  --camera agentview \
  --target-per-task 10
```

This returns a `session_id`.

### 3. Collect a task

```bash
python -m data_collection.cli collect-task \
  --session-id <session_id> \
  --task-name <task_name> \
  --num-demonstrations 1 \
  --notes "first pass"
```

This creates:

- an episode directory under the session
- raw teleop fragments
- a merged raw `demo.hdf5`

### 4. Package the raw run

```bash
python -m data_collection.cli package-task \
  --task-name <task_name> \
  --raw-hdf5 data_collection/runs/<session_id>/episodes/<episode_id>/raw/demo.hdf5 \
  --overwrite
```

This writes the benchmark-compatible output:

```text
datasets/libero_plus_train_subtasks/<task_name>_demo.hdf5
```

### 5. Validate the packaged dataset

```bash
python -m data_collection.cli validate-task \
  --dataset datasets/libero_plus_train_subtasks/<task_name>_demo.hdf5 \
  --task-name <task_name>
```

Validation checks include:

- action dimension is `7`
- action values remain in `[-1, 1]`
- sequence lengths are aligned
- required observation keys are present
- final timestep terminates with `done = 1`

### 6. Export to pi05

```bash
python -m data_collection.cli export-pi05 \
  --export-name libero_plus_pi05 \
  --task-name <task_name> \
  --dataset datasets/libero_plus_train_subtasks/<task_name>_demo.hdf5 \
  --state-schema ee_gripper
```

The default state schema is:

- `ee_gripper`

This means:

- `state = concat(ee_states, gripper_states)`

Alternative:

- `joint_gripper_ee`

This means:

- `state = concat(joint_states, gripper_states, ee_states)`

## Output Layout

### Session output

```text
data_collection/runs/<session_id>/
  session.json
  episodes/
    <episode_id>/
      episode.json
      raw/
        demo.hdf5
        tmp/
```

### Packaged LIBERO output

```text
datasets/libero_plus_train_subtasks/<task_name>_demo.hdf5
datasets/libero_plus_train_subtasks/<task_name>_demo.manifest.json
```

### pi05 export output

```text
data_collection/exports/pi05/<export_name>/
  manifest.json
  stats/
    quantiles.json
  episodes/
    index.jsonl
    <task_name>__demo_0.npz
    ...
```

## Recommended Operating Procedure

For each task:

1. Collect one or more clean raw episodes
2. Package them immediately
3. Run validation immediately
4. Only export to pi05 after validation passes
5. Keep notes on failed attempts outside the final dataset if they are not usable

## Troubleshooting

## `list-tasks` works but collection fails

Likely causes:

- robosuite renderer not available
- display configuration issue
- missing control device support

## Packaging fails

Likely causes:

- `h5py` missing
- raw `demo.hdf5` missing or incomplete
- environment replay diverges badly

## Validation fails on action range

Likely causes:

- collected actions are not normalized
- a non-standard controller configuration was used

## Export completes but downstream training still needs conversion

The current exporter writes a deterministic bridge dataset with episode payloads, index metadata, and quantile stats. If your exact downstream `pi05` tool expects a strict LeRobot v3 writer, use this export as the canonical source for that final conversion step.
