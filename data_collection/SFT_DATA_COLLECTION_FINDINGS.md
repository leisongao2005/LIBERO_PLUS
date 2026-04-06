# SFT Data Collection Framework for `libero_plus_train_subtasks`

## Goal

Build a framework to let a human manually operate the robot, collect demonstrations for the tasks in `libero_plus_train_subtasks`, and export them into a format suitable for `pi 0.5` (`pi05`) fine-tuning.

This document summarizes:

- what already exists in this repo,
- what is verified about the action interface,
- what is missing for a practical teleoperation workflow,
- and the recommended architecture for collection and export.

## Executive Summary

The repo already has most of the low-level pieces needed for teleoperation-based data collection:

- `libero_plus_train_subtasks` is already registered as a benchmark and mapped to concrete task names.
- A robosuite-based human teleoperation collector already exists and supports `keyboard` and `spacemouse`.
- A dataset packaging script already converts raw teleop rollouts into the per-task LIBERO HDF5 format used by this training codebase.

The main gaps are productization and interoperability:

- there is no task-selection GUI or collection dashboard yet,
- the current teleop scripts are CLI-first and not designed for rapid multi-task collection,
- and the packaged LIBERO HDF5 format in this repo is not the same as the public `pi05` fine-tuning format.

Recommendation:

1. Reuse the existing robosuite teleop + MuJoCo viewer as the first GUI.
2. Build a thin session manager around it for task selection, episode bookkeeping, and operator controls.
3. Continue storing raw rollouts and packaged LIBERO HDF5 files for local validation.
4. Add a separate exporter that converts the packaged LIBERO task datasets into LeRobot v3 datasets for `pi05`.

## Relevant Existing Code

### Task suite

- `libero/libero/benchmark/__init__.py`
  - Registers `LIBERO_PLUS_TRAIN_SUBTASKS`.
  - Resolves each task's dataset path as:
    - ``libero_plus_train_subtasks/<task_name>_demo.hdf5``
- `libero/libero/benchmark/libero_suite_task_map.py`
  - Defines the actual list of tasks in `libero_plus_train_subtasks`.

### Human teleoperation / keyboard control

- `scripts/collect_demonstration.py`
  - Existing single-task teleoperation collector.
  - Uses robosuite `VisualizationWrapper`, `DataCollectionWrapper`, and `input2action`.
  - Supports `--device keyboard` and `--device spacemouse`.
- `scripts/libero_100_collect_demonstrations.py`
  - Similar collection flow for bulk LIBERO data collection.

### Dataset packaging for this repo

- `scripts/create_dataset.py`
  - Replays raw demonstrations and writes final task datasets with:
    - `actions`
    - `states`
    - `robot_states`
    - `rewards`
    - `dones`
    - `obs/agentview_rgb`
    - `obs/eye_in_hand_rgb`
    - `obs/joint_states`
    - `obs/gripper_states`
    - `obs/ee_states`
    - `obs/ee_pos`
    - `obs/ee_ori`
- `scripts/get_dataset_info.py`
  - Verifies structure and checks that actions stay in `[-1, 1]`.
- `libero/lifelong/datasets.py`
  - Training reads action dimensionality from dataset metadata via `shape_meta["ac_dim"]`.

## Verified Action Interface

## Conclusion

For the normal LIBERO path in this repo, the policy action is effectively a 7D vector:

- `3` values for end-effector position control,
- `3` values for end-effector orientation control,
- `1` value for gripper open / close.

In practice, this is the interface you should target for teleoperation and collection.

## Evidence from this repo

### 1. Standard evaluation assumes 7D actions

- `libero/lifelong/evaluate.py`
  - Advances physics with `np.zeros((env_num, 7))`.
- `libero/lifelong/metric.py`
  - Also uses dummy actions of shape `(env_num, 7)`.
- `README.md`
  - Demonstrates stepping the environment with `dummy_action = [0.] * 7`.

This is the strongest repo-local evidence that the default single-arm LIBERO action space is 7D.

### 2. The learned action size comes from the dataset

- `libero/lifelong/datasets.py`
  - Reads `shape_meta` from the HDF5 dataset.
- Policy implementations set output size from `shape_meta["ac_dim"]`:
  - `libero/lifelong/models/bc_rnn_policy.py`
  - `libero/lifelong/models/bc_transformer_policy.py`
  - `libero/lifelong/models/bc_vilt_policy.py`

So the model does not hardcode `7`, but the standard datasets and evaluation flow clearly expect it.

### 3. Actions are normalized

- `libero/lifelong/models/policy_head.py`
  - `GMMHead` applies `tanh` to action means.
- `scripts/get_dataset_info.py`
  - Raises an error if action values leave `[-1, 1]`.

This strongly suggests collected actions should remain normalized within `[-1, 1]`.

### 4. There is also a 4D fallback mode

- `libero/libero/envs/bddl_base_domain.py`
  - If `self.action_dim == 4` and a longer action is passed, the env keeps:
    - first `3` dims
    - last `1` dim

That strongly implies a reduced control mode of:

- position-only `xyz`
- gripper

with orientation omitted.

## Evidence from robosuite upstream

The current collector uses robosuite `OSC_POSE` and `input2action`.

Public robosuite controller docs describe `OSC_POSE` with:

- `3` translational controls,
- `3` rotational controls,
- and a gripper command when the gripper is present.

The controller config also exposes six pose outputs:

- `output_max = [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]`

which matches:

- `xyz` translation
- `xyz` rotational / axis-angle-style orientation control

Combined with the repo's repeated 7D assumptions, this is sufficient to treat the collected action as:

`[dx, dy, dz, d_rx, d_ry, d_rz, gripper]`

For the doc below, I treat that as verified enough for implementation planning.

## What the Current Collector Already Gives You

The existing flow is:

1. Run `scripts/collect_demonstration.py` on one BDDL task.
2. Teleoperate with keyboard or SpaceMouse through the robosuite viewer.
3. Save raw demonstration fragments under a temporary directory.
4. Merge them into `demo.hdf5`.
5. Run `scripts/create_dataset.py` to create the final task dataset at:
   - `<datasets>/<suite>/<task_name>_demo.hdf5`

This means the repo already supports:

- manual control,
- raw action/state capture,
- replay validation,
- and packaging into the format used by the existing LIBERO imitation-learning stack.

## What Is Missing

## 1. A real operator-facing GUI

Today the "GUI" is mainly the robosuite / MuJoCo viewer plus keyboard callbacks.

That is enough to start collecting data, but it is missing:

- task browsing for `libero_plus_train_subtasks`,
- demo counters and completion status,
- explicit save / discard controls,
- quality-control notes,
- per-task target counts,
- failure / retry tracking,
- and quick launch for the next task.

## 2. A collection manager across the whole suite

The current collector is single-task oriented. For your use case, you want a manager that can:

- enumerate all tasks in `libero_plus_train_subtasks`,
- launch collection for one selected task,
- record episode metadata,
- show progress toward target demo counts,
- and validate that the final file lands at the correct per-task path.

## 3. A `pi05` export path

The repo's packaged HDF5 format is suitable for this repo's own training code, but public `pi05` fine-tuning expects a LeRobot dataset, not this custom HDF5 layout.

So there should be two outputs:

- LIBERO-format task HDF5 for local training / validation here
- LeRobot v3 dataset export for `pi05`

## Recommended Framework

## Layer 1: Teleop runner

Wrap the existing `scripts/collect_demonstration.py` logic instead of replacing it.

Recommended responsibilities:

- load a task by benchmark name + task id,
- resolve the BDDL file automatically,
- start robosuite with `OSC_POSE`,
- support `keyboard` as the default operator device,
- allow optional `spacemouse`,
- and expose a stable save / discard protocol per episode.

Recommended output:

- raw teleop episode directory
- merged `demo.hdf5`

## Layer 2: Collection GUI / session manager

Build a lightweight launcher GUI around the teleop runner.

Minimum GUI features:

- task list for `libero_plus_train_subtasks`
- search / filter
- target demos per task
- completed demos per task
- buttons:
  - `Launch`
  - `Save episode`
  - `Discard episode`
  - `Replay latest`
  - `Package dataset`
- operator notes field
- optional success / quality tags

Implementation options:

- fastest: Tkinter or PySimpleGUI-style desktop wrapper around subprocess calls
- more flexible: small web UI that launches local Python jobs

For a first version, a small Python desktop app is enough. The robosuite viewer itself can remain the live control window.

## Layer 3: Dataset packager

Keep using `scripts/create_dataset.py` as the canonical packager for this repo.

This guarantees the output remains compatible with:

- `libero/lifelong/datasets.py`
- the existing benchmark loader
- current evaluation / training code

Expected packaged file path per task:

- `datasets/libero_plus_train_subtasks/<task_name>_demo.hdf5`

## Layer 4: `pi05` exporter

Add a new exporter that reads each packaged LIBERO HDF5 and writes a LeRobot v3 dataset.

That exporter should map:

- episode task text:
  - from `problem_info["language_instruction"]`
- images:
  - from `obs/agentview_rgb`
  - optionally `obs/eye_in_hand_rgb`
- state:
  - from `obs/ee_states`
  - optionally concatenate `joint_states` and `gripper_states` if desired
- action:
  - from `actions`
- episode boundaries:
  - from each `demo_*` group

This keeps the internal collection format decoupled from the external fine-tuning format.

## Suggested Data Schema

## Canonical internal collection record

For each timestep, preserve:

- `action`
- `state` (flattened sim state, if available)
- `robot_states`
- `joint_states`
- `gripper_states`
- `ee_states`
- `agentview_rgb`
- `eye_in_hand_rgb`
- `reward`
- `done`
- `language_instruction`
- `task_name`
- `benchmark_name`
- `episode_id`
- `operator_id` or note

This is already close to what `create_dataset.py` emits.

## Suggested `pi05` state choice

For `pi05`, keep the state small and consistent.

Best first choice:

- `state = concat(ee_states, gripper_states)`

Reasons:

- matches the action semantics closely,
- should transfer better across embodiments than full joint positions,
- and stays well below the public `pi05` max state dimension guidance.

Alternative:

- `state = concat(joint_states, gripper_states, ee_states)`

This may help in simulation, but it couples the dataset more tightly to Panda-specific kinematics.

## `pi05` Fine-Tuning Requirements

Based on public LeRobot `pi05` documentation:

- dataset format should be LeRobot v3
- task descriptions are required
- actions are required
- state features are required
- images are optional but recommended
- datasets should have at least `600` rows total
- default normalization expects quantile stats (`q01`, `q99`)
- action and state dimensions should fit within the model limits
- public guidance states smaller state / action vectors are padded up to model limits

Additional public details often cited for `pi05`:

- chunked action prediction
- image resizing handled in the policy pipeline
- support for multiple cameras

For this project, the important consequence is:

the exporter must produce a proper LeRobot dataset with episode metadata and feature stats, or training must override normalization to `MEAN_STD`.

## Compatibility Assessment

## Compatible today

You can already:

- collect teleop demonstrations,
- package them into LIBERO task datasets,
- train / fine-tune the existing policies in this repo,
- and evaluate them on the `libero_plus_train_subtasks` tasks.

## Not directly compatible today

You cannot directly point public `pi05` fine-tuning tooling at the current repo HDF5 files without conversion.

The missing piece is a dataset converter / exporter.

## Recommended Implementation Plan

## Phase 1: Prove out collection

Build a minimal collector for one task using existing scripts:

- keyboard control
- episode save / discard
- packaged LIBERO HDF5 output
- dataset integrity check

Success criteria:

- one task in `libero_plus_train_subtasks`
- at least a few clean demos
- output opens in `scripts/get_dataset_info.py`

## Phase 2: Multi-task GUI

Add the collection manager GUI:

- list all tasks
- show demo counts
- launch task collection
- package datasets automatically

Success criteria:

- operator can collect across many tasks without editing command lines

## Phase 3: `pi05` exporter

Add a converter from packaged LIBERO task HDF5 to LeRobot v3.

Success criteria:

- output dataset includes:
  - task text
  - actions
  - state
  - images
  - episode boundaries
  - stats / normalization metadata

## Phase 4: `pi05` fine-tuning validation

Validate with a small subset first:

- export one task
- run LeRobot dataset checks
- add quantile stats
- run a short `pi05` fine-tuning job

## Risks and Design Choices

## 1. Action semantics must stay consistent

Do not collect one dataset with 7D `OSC_POSE` actions and another with a reduced 4D mode unless you explicitly separate them.

Mixing action parameterizations will make the fine-tuning data inconsistent.

## 2. Keyboard teleop may be slow for dense collection

Keyboard control is feasible, but SpaceMouse is usually more efficient for pose control.

Recommendation:

- support both
- default to keyboard first because it is more accessible

## 3. Keep one internal source of truth

The repo's packaged LIBERO HDF5 should remain the canonical archive format for this codebase.

Then derive:

- training subsets
- QC reports
- `pi05` exports

from that packaged version.

## 4. State feature choice matters for `pi05`

If the target is cross-task SFT rather than Panda-specific imitation only, prefer a compact task-relevant state definition centered on:

- end-effector pose
- gripper state

instead of very robot-specific full-joint features.

## Bottom Line

The repo already contains the core teleoperation and dataset-packaging machinery needed to collect manual demonstrations for `libero_plus_train_subtasks`.

The 7D action interface is sufficiently verified for planning and implementation as:

- end-effector position `3`
- end-effector orientation `3`
- gripper command `1`

The most effective path is not to replace the current collector, but to wrap it with:

- a task-oriented GUI / session manager
- better bookkeeping and QC
- and a new LeRobot v3 exporter for `pi05`

That gives you one collection pipeline that serves both:

- this repo's native LIBERO training format
- and external `pi 0.5` fine-tuning.
