# Collecting LIBERO-Plus demonstrations with `collect_demonstration.py`

This guide describes how to run [`collect_demonstration.py`](collect_demonstration.py) to record human teleop for tasks whose BDDL lives under the LIBERO-Plus suites (e.g. `libero_plus_train_subtasks`, `libero_plus_eval_subtasks`).

## Prerequisites

- Working **LIBERO** install: MuJoCo, robosuite, and this repo on `PYTHONPATH` (the script imports `init_path`, which adds the repo root).
- Valid **`~/.libero/config.yaml`** (or `LIBERO_CONFIG_PATH`) so `bddl_files`, `init_states`, and assets resolve.
- A **display** for the MuJoCo viewer (the script uses `has_renderer=True`).

## BDDL path

Tasks are defined by a **`.bddl` file**. For Plus training tasks, files typically look like:

```text
<bddl_files from config>/libero_plus_train_subtasks/<task_name>.bddl
```

Use the same paths your benchmark uses (resolve `<bddl_files>` from `libero.libero.get_libero_path("bddl_files")` or open your config YAML).

## Command

From the **LIBERO_PLUS** repository root:

```bash
python scripts/collect_demonstration.py \
  --bddl-file /path/to/<task_name>.bddl \
  --device keyboard \
  --num-demonstration 10 \
  --directory demonstration_data
```

### Useful flags

| Flag | Default | Notes |
|------|---------|--------|
| `--bddl-file` | (required) | Absolute or relative path to the task BDDL. |
| `--device` | `spacemouse` | Use `keyboard` if you do not have a SpaceMouse. |
| `--num-demonstration` | `50` | Number of episodes to record into one output run. |
| `--directory` | `demonstration_data` | Parent folder for timestamped run directories. |
| `--controller` | `OSC_POSE` | Alternative: `IK_POSE`. |
| `--camera` | `agentview` | Viewer camera. |
| `--config` | `single-arm-opposed` | Bimanual / two-arm envs may need this set appropriately. |
| `--arm` | `right` | Active arm for dual-arm setups. |

## What gets written

1. **While collecting:** `DataCollectionWrapper` writes rollouts under a temp tree, e.g.  
   `demonstration_data/tmp/<problem>_ln_<instruction>/...`

2. **After each saved episode:** rollouts are merged into a single HDF5:

   ```text
   <directory>/<domain>_ln_<problem>_<timestamp>_<instruction>/demo.hdf5
   ```

   The file has a `data/` group with `demo_1`, `demo_2`, … each containing `states`, `actions`, and `model_file` metadata, plus top-level attrs such as `env_info`, `bddl_file_name`, `bddl_file_content`, and `problem_info`.

## Controls

- **Keyboard / SpaceMouse** are wired through robosuite’s `input2action` and the **native viewer** (not a separate OpenCV window).
- Ending an episode without saving follows the device’s reset behavior (see robosuite docs for your device).

## Next steps in this repo

Raw `demo.hdf5` is **not** yet in the packaged LIBERO dataset layout under `datasets/`. To match the rest of **LIBERO_PLUS** (training, validation, **pi05** export):

1. Run the **`data_collection`** packager on the raw file (same role as `package-task` in the CLI) so it becomes a canonical per-task dataset under your configured datasets root.
2. Optionally **validate** and **export-pi05** per [`data_collection/docs/USAGE.md`](../data_collection/docs/USAGE.md).

## Troubleshooting

- **`NameError: problem_info` when writing `demo.hdf5`:** `gather_demonstrations_as_hdf5` must receive `problem_info` (the dict from `BDDLUtils.get_problem_info`) and use it for `grp.attrs["problem_info"]`. Pass it from `main` into that function.
- **HDF5 not flushing / corrupt file:** avoid reusing the variable name `f` for the inner `open(model.xml)` while the outer `f` is the `h5py.File`; use a different name (e.g. `xml_f`) so `f.close()` closes the HDF5 file.

## See also

- [`data_collection/docs/USAGE.md`](../data_collection/docs/USAGE.md) — session-based collection and export.
- [`data_collection/SFT_DATA_COLLECTION_FINDINGS.md`](../data_collection/SFT_DATA_COLLECTION_FINDINGS.md) — design notes on formats and pi05.
