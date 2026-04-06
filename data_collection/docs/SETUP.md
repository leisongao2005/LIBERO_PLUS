# Data Collection Setup

This guide describes how to prepare the environment for the `data_collection/` framework.

## Prerequisites

- Python environment with the repo dependencies installed
- Working LIBERO installation and config
- MuJoCo / robosuite runtime working on the machine
- Keyboard control, or an attached SpaceMouse if you plan to use it

## Install Dependencies

From the repo root:

```bash
pip install -r requirements.txt
pip install -e .
```

Important packages for this framework:

- `robosuite`
- `robomimic`
- `opencv-python`
- `h5py`
- `numpy`
- `bddl`

If `h5py` is missing, packaging, validation, and export commands will fail.

## Verify LIBERO Paths

The framework uses the same LIBERO path configuration as the rest of the repo.

Check that your LIBERO config exists and points to valid locations:

```bash
python - <<'PY'
from libero.libero import get_libero_path
for key in ["bddl_files", "init_states", "datasets", "assets"]:
    print(key, "->", get_libero_path(key))
PY
```

You should verify:

- `bddl_files` exists
- `init_states` exists
- `datasets` exists or is writable

## Verify Task Discovery

Make sure the new framework can see the `libero_plus_train_subtasks` benchmark:

```bash
python -m data_collection.cli list-tasks
```

Expected result:

- a JSON list of tasks
- task names from `libero_plus_train_subtasks`

## Verify GUI Availability

The GUI uses Tkinter. On most Linux installations it is already available. Test it with:

```bash
python -m data_collection.gui_app
```

Expected result:

- a desktop window titled `LIBERO Data Collection`

If Tkinter is missing, install the system package that provides Python Tk support for your distro.

## Verify Teleoperation Runtime

Before a full data collection run, confirm robosuite rendering and device control work:

1. Create a session:

```bash
python -m data_collection.cli start-session --device keyboard
```

2. Copy the returned `session_id`.
3. Use the GUI or CLI to launch collection for one task.

If the robosuite window does not appear, check:

- display / X forwarding setup
- MuJoCo runtime
- OpenGL availability

## Verify Packaging Runtime

Once you have one raw `demo.hdf5`, validate the packaging path:

```bash
python -m data_collection.cli package-task \
  --task-name <task_name> \
  --raw-hdf5 <raw_demo_hdf5> \
  --overwrite
```

Then validate the packaged output:

```bash
python -m data_collection.cli validate-task \
  --dataset <packaged_hdf5> \
  --task-name <task_name>
```

## Verify Export Runtime

Export one packaged task into the pi05 bridge dataset:

```bash
python -m data_collection.cli export-pi05 \
  --export-name test_export \
  --task-name <task_name> \
  --dataset <packaged_hdf5>
```

Expected output tree:

- `data_collection/exports/pi05/test_export/manifest.json`
- `data_collection/exports/pi05/test_export/stats/quantiles.json`
- `data_collection/exports/pi05/test_export/episodes/index.jsonl`
- `data_collection/exports/pi05/test_export/episodes/*.npz`

## First-Run Checklist

- `list-tasks` works
- GUI opens
- robosuite viewer opens during collection
- one raw `demo.hdf5` is created
- one packaged LIBERO HDF5 is created
- validation passes
- one pi05 export directory is created

After these checks pass, the framework is ready for larger-scale collection.
