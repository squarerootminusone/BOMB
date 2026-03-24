# Go + MimicGen Benchmark (OpenVLA Prep)

This benchmark is a standalone workspace for creating a future OpenVLA fine-tuning dataset from Go manipulation scenes.

It is implemented entirely under `benchmarks/go_vla_benchmark` so dependency packages stay untouched.

## What It Includes

- Backend-selectable Go wrappers:
  - DeepMind `physics_planning_games` (`GoJacoBenchmarkEnv`)
  - robosuite rigid-body 5x5 task (`GoRobosuiteBenchmarkEnv`)
- A custom MimicGen interface (`MG_GoJacoSingleMove`) for extracting `datagen_info`
- Source demo collection script (scripted control for initial MimicGen seeds)
- MimicGen augmentation script using `mimicgen.datagen.DataGenerator`
- Dataset inspection script
- Task config template with subtask boundaries, action noise, interpolation, and selection strategy

## Folder Layout

- `go_vla_benchmark/go_env.py`: DeepMind simulator wrapper
- `go_vla_benchmark/robosuite_go_env.py`: robosuite rigid-body 5x5 wrapper
- `go_vla_benchmark/env_factory.py`: backend selection by `--environment-name`
- `go_vla_benchmark/mimicgen_interface.py`: MimicGen environment interface (registered via import)
- `go_vla_benchmark/collect.py`: source demo collection pipeline
- `go_vla_benchmark/generate.py`: MimicGen augmentation pipeline
- `configs/go_single_move_task.json`: task spec and default generation settings
- `scripts/*.py`: CLI entrypoints

## Prerequisites

- Python `3.10` or `3.11` (validated)
- MuJoCo-compatible rendering stack
- `gnugo` only if you use the DeepMind Go backend

Install benchmark dependencies:

```bash
pip install -r benchmarks/go_vla_benchmark/requirements.txt
pip install --no-deps -e "git+https://github.com/ARISE-Initiative/robomimic.git@d0b37cf214bd24fb590d182edb6384333f67b661#egg=robomimic"
pip install -e mimicgen
```

For the robosuite backend, use the local `./robosuite` checkout (added to `PYTHONPATH` by benchmark scripts).

If you hit `Failed building wheel for egl_probe` with a CMake compatibility error, force CMake 3.x and retry:

```bash
pip install --upgrade "cmake<4"
pip install --no-cache-dir -r benchmarks/go_vla_benchmark/requirements.txt
pip install --no-deps -e "git+https://github.com/ARISE-Initiative/robomimic.git@d0b37cf214bd24fb590d182edb6384333f67b661#egg=robomimic"
```

Set `PYTHONPATH` so local repos are imported:

```bash
export PYTHONPATH="$PWD/deepmind-research:$PWD/mimicgen:$PWD/robosuite:$PWD/benchmarks/go_vla_benchmark:$PYTHONPATH"
```

Run setup doctor (fails with non-zero exit if generation is not ready):

```bash
python benchmarks/go_vla_benchmark/scripts/check_setup.py --strict
```

If using the DeepMind backend and `gnugo` is not on `PATH`, pass `--gnugo-path /abs/path/to/gnugo` (or set `GNUGO_PATH`).

## Workflow

## 1) Launch Go Simulator Viewer (optional sanity check)

```bash
python benchmarks/go_vla_benchmark/scripts/run_go_viewer.py --env go_7x7 --seed 0
```

Rigid-stone 5x5 variant:

```bash
python benchmarks/go_vla_benchmark/scripts/run_go_viewer.py --env go_5x5_rigid_bodies --seed 0
```

## 2) Collect Source Demos (Initial MimicGen Seeds)

```bash
python benchmarks/go_vla_benchmark/scripts/collect_source_demos.py \
  --output benchmarks/go_vla_benchmark/data/source_go.hdf5 \
  --environment-name robosuite_go_5x5_rigid_bodies \
  --num-demos 5 \
  --camera-size 512 \
  --opening-min 0 \
  --opening-max 8 \
  --action-scale 0.03 \
  --controller-divisor 2.5 \
  --success-hold-steps 40 \
  --detour-steps 20 \
  --detour-radius 0.18 \
  --approach-steps 30 \
  --press-steps 16 \
  --retreat-steps 22 \
  --side-transfer-steps 28 \
  --side-margin 0.18 \
  --no-carried-stone
```

```bash
python benchmarks/go_vla_benchmark/scripts/collect_source_demos.py \
  --output benchmarks/go_vla_benchmark/data/source_go.hdf5 \
  --environment-name robosuite_go_5x5_rigid_bodies \
  --num-demos 5 \
  --camera-size 512 \
  --action-scale 0.03 \
  --controller-divisor 2.5 \
  --success-hold-steps 40 \
  --detour-steps 0 \
  --detour-radius 0.01 \
  --approach-steps 30 \
  --press-steps 16 \
  --retreat-steps 22 \
  --side-transfer-steps 0 \
  --side-margin 0.01 \
  --no-carried-stone
```

```bash
conda run --no-capture-output -n main   python benchmarks/go_vla_benchmark/scripts/convert_to_rlds.py   --input /root/dsait4125/benchmarks/go_vla_benchmark/data/source_go.hdf5
```

```bash
 python viewer_video.py   --data-dir /root/tensorflow_datasets   --start-episode 0   --num-episodes 5   --output /root/dsait4125/all_episodes.mp4
```

Other installs
```bash
pip install huggingface_hub
```
Juicy juicy
```bash
```


This creates MimicGen-compatible source trajectories with:

- `datagen_info/eef_pose`
- `datagen_info/object_poses/{target_intersection,board_origin}`
- `datagen_info/subtask_term_signals/reach_target`

The `--camera-size` (or `--camera-height` + `--camera-width`) flag controls native simulator render resolution written into HDF5 frames (for example `--camera-size 512` for native `512x512`). The `--controller-divisor`, `--success-hold-steps`, `--detour-steps`, `--detour-radius`, `--approach-steps`, `--press-steps`, `--retreat-steps`, `--side-transfer-steps`, and `--side-margin` flags control trajectory length and visible motion in the HDF5 itself (not just video playback speed). By default, rendered images include an EEF + target overlay; disable it with `--no-eef-overlay`.

For OpenVLA-style manipulation data, the default disables opponent responses during scripted moves (to avoid random extra stones appearing mid-trajectory). Re-enable with `--enable-opponent-moves`.
For robosuite backend robot swaps, pass `--robot <RobotName>` (for example `--robot UR5e`).

## Visualize What Is In HDF5 (Exact `obs/agentview_image`)

```bash
python benchmarks/go_vla_benchmark/scripts/export_agentview_video.py \
  --dataset benchmarks/go_vla_benchmark/data/source_go.hdf5 \
  --output benchmarks/go_vla_benchmark/data/source_go_preview.mp4 \
  --num-demos 5 \
  --fps 4 \
  --macro-block-size 1
```

## 3) Validate Source Dataset (recommended)

```bash
python mimicgen/mimicgen/scripts/get_source_info.py \
  --dataset benchmarks/go_vla_benchmark/data/source_go.hdf5
```

## 4) Generate Augmented Demos with MimicGen

```bash
python benchmarks/go_vla_benchmark/scripts/generate_augmented_demos.py \
  --source benchmarks/go_vla_benchmark/data/source_go.hdf5 \
  --output benchmarks/go_vla_benchmark/data/augmented_go.hdf5 \
  --task-config benchmarks/go_vla_benchmark/configs/go_single_move_task.json \
  --environment-name robosuite_go_5x5_rigid_bodies \
  --num-demos 200 \
  --max-attempts 500 \
  --camera-size 512 \
  --action-scale 0.03
```

Augmentations come from:

- MimicGen object-centric segment recombination
- subtask boundary offsets (`subtask_term_offset_range`)
- action noise and interpolation
- random opening board states (`opening-min`, `opening-max`)
- varied legal move targets per episode

## 5) Inspect Generated Dataset

```bash
python benchmarks/go_vla_benchmark/scripts/inspect_dataset.py \
  --dataset benchmarks/go_vla_benchmark/data/augmented_go.hdf5
```

## 6) Convert Generated HDF5 to RLDS (for OpenVLA)

```bash
python benchmarks/go_vla_benchmark/scripts/convert_to_rlds.py \
  --input benchmarks/go_vla_benchmark/data/augmented_go.hdf5
```

This conversion step does more than repackage files. The benchmark still writes
raw MimicGen-style HDF5 during collection and augmentation, while the RLDS
builder applies OpenVLA-oriented preprocessing:

- keeps the raw collection / MimicGen pipeline unchanged
- downsamples robosuite trajectories from 20 Hz to about 5 Hz
- filters near-no-op actions
- exports native 4-DoF actions `[dx, dy, dz, gripper]`
- remaps gripper values from `{0, 1}` to `{-1, +1}`
- derives language instructions from the board-state delta
- preserves the HDF5 image resolution instead of hard-coding `256x256`

The converted dataset is written to `~/tensorflow_datasets/go_vla_dataset/`.
Verify it with:

```bash
python benchmarks/go_vla_benchmark/scripts/verify_rlds.py
```

If you want RLDS for a smaller debugging set, you can point `--input` at
`benchmarks/go_vla_benchmark/data/source_go.hdf5` instead.

## Local Explanation Pipeline

The explainability path is now split into a reusable collector plus downstream
renderers, so you can swap checkpoints and datasets without changing the
rendering code.

For the remote GPU workflow with fish shell setup, SSH / rsync upload, LoRA
merge on the cloud, tmux execution, and result download, see
[CLOUD_EXPORT.md](/Users/rafael/Repos/DSAIT/4125.%20Computer%20Vision/dsait4125/benchmarks/go_vla_benchmark/CLOUD_EXPORT.md).

Collect local explanations with:

```bash
python benchmarks/go_vla_benchmark/scripts/collect_openvla_local_explanations.py \
  --dataset benchmarks/go_vla_benchmark/data/source_go.hdf5 \
  --checkpoint /abs/path/to/openvla-checkpoint \
  --trace-output benchmarks/go_vla_benchmark/data/local_explanations/source_go_trace.npz \
  --summary-output benchmarks/go_vla_benchmark/data/local_explanations/source_go_summary.json \
  --num-demos 10 \
  --stride 4 \
  --attention-layers 4 \
  --attn-implementation eager
```

Each step in the saved trace now includes:

- image patch attribution for every generated action token plus a mean patch map
- text token attribution over the instruction tokens for every generated action token
- target output token probabilities for the generated action-token sequence

Turn that trace into per-step artifacts with:

```bash
python benchmarks/go_vla_benchmark/scripts/export_openvla_local_explanation_report.py \
  --trace-input benchmarks/go_vla_benchmark/data/local_explanations/source_go_trace.npz \
  --output-dir benchmarks/go_vla_benchmark/data/local_explanations/report \
  --token-top-k 8
```

This report export creates, for each step:

- a patch-attribution panel with the mean map plus one heatmap per generated action token
- a Markdown + JSON summary of target output token ids and probabilities
- ranked instruction-token saliency tables, both mean and per output token

You can run this in two ways:

- separate: collect the reusable trace first, then render videos later from `--trace-input`
- chained: run the exporter directly from `--checkpoint` and save the same trace with `--trace-output`

The trace format is designed to grow into later explainability stages:

- intervention tests can reuse the same dataset clips and saved prompts
- activation patching can reuse the same model adapter boundary
- global semantic analysis can consume the same saved trace metadata

## Intervention Tests

Stage 2 runs counterfactual perturbations on the same RLDS-aligned clips and
stores them in a separate reusable trace. The collection script supports running
all interventions together or only a subset via `--runs`.

Collect the full intervention suite:

```bash
python benchmarks/go_vla_benchmark/scripts/collect_openvla_intervention_tests.py \
  --dataset benchmarks/go_vla_benchmark/data/source_go.hdf5 \
  --checkpoint /abs/path/to/openvla-checkpoint \
  --trace-output benchmarks/go_vla_benchmark/data/interventions/source_go_interventions.npz \
  --summary-output benchmarks/go_vla_benchmark/data/interventions/source_go_interventions.json \
  --num-demos 10 \
  --stride 4 \
  --top-k 8 \
  --max-counterfactual-edits 4 \
  --attn-implementation eager
```

Export the human-readable report:

```bash
python benchmarks/go_vla_benchmark/scripts/export_openvla_intervention_report.py \
  --trace-input benchmarks/go_vla_benchmark/data/interventions/source_go_interventions.npz \
  --output-dir benchmarks/go_vla_benchmark/data/interventions/report
```

Detailed commands and pseudocode for patch occlusion, text masking, and greedy
minimal counterfactual edits live in
[INTERVENTION_TESTS.md](/Users/rafael/Repos/DSAIT/4125.%20Computer%20Vision/dsait4125/benchmarks/go_vla_benchmark/INTERVENTION_TESTS.md).

## Internal Causal Localization

Stage 3 performs activation patching on top of the same RLDS-aligned clips. It
stores layer-level and head-level restoration scores in a separate reusable
trace, and can optionally patch any discovered cross-attention-style blocks.

Collect causal localization with patch occlusion as the corruption source:

```bash
python benchmarks/go_vla_benchmark/scripts/collect_openvla_causal_localization.py \
  --dataset benchmarks/go_vla_benchmark/data/source_go.hdf5 \
  --checkpoint /abs/path/to/openvla-checkpoint \
  --trace-output benchmarks/go_vla_benchmark/data/causal/source_go_patch_causal.npz \
  --summary-output benchmarks/go_vla_benchmark/data/causal/source_go_patch_causal.json \
  --num-demos 4 \
  --stride 4 \
  --corruption-type patch-occlusion \
  --attn-implementation eager
```

Export the human-readable report:

```bash
python benchmarks/go_vla_benchmark/scripts/export_openvla_causal_localization_report.py \
  --trace-input benchmarks/go_vla_benchmark/data/causal/source_go_patch_causal.npz \
  --output-dir benchmarks/go_vla_benchmark/data/causal/report
```

Detailed commands and pseudocode for layer, head, and optional cross-attention
patching live in
[CAUSAL_LOCALIZATION.md](/Users/rafael/Repos/DSAIT/4125.%20Computer%20Vision/dsait4125/benchmarks/go_vla_benchmark/CAUSAL_LOCALIZATION.md).

## Attention Failure Videos

To export a side-by-side video with the original inference frames on the left and
an attention-map video on the right, use the same local-explanation trace:

```bash
python benchmarks/go_vla_benchmark/scripts/export_openvla_attention_videos.py \
  --trace-input benchmarks/go_vla_benchmark/data/local_explanations/source_go_trace.npz \
  --output-dir benchmarks/go_vla_benchmark/data/attention_failures \
  --manifest-output benchmarks/go_vla_benchmark/data/attention_failures_manifest.json \
  --top-k 3
```

This renders:

- the language instruction in the header
- the original inference video next to the attention-map video
- `gt` and `pred` action cards with `x / y / z / gripper`
- an attention-history strip that shows how the spatial attention shifts over time

If you already have a saved trace file with predictions + attention grids, render
without rerunning the model:

```bash
python benchmarks/go_vla_benchmark/scripts/export_openvla_attention_videos.py \
  --trace-input benchmarks/go_vla_benchmark/data/local_explanations/source_go_trace.npz \
  --output-dir benchmarks/go_vla_benchmark/data/attention_failures
```

If you want the exporter to collect and render in one step, it still supports the
direct checkpoint path and now writes the richer reusable trace format:

```bash
python benchmarks/go_vla_benchmark/scripts/export_openvla_attention_videos.py \
  --dataset benchmarks/go_vla_benchmark/data/source_go.hdf5 \
  --checkpoint /abs/path/to/openvla-checkpoint \
  --trace-output benchmarks/go_vla_benchmark/data/local_explanations/source_go_trace.npz \
  --output-dir benchmarks/go_vla_benchmark/data/attention_failures \
  --top-k 3 \
  --stride 4 \
  --attn-implementation eager
```

## Diversity Controls

Use these controls to create broader scenarios (different board states / moves):

- Increase `--opening-min` and `--opening-max` in both collection and generation.
- Increase `action_noise`, `num_interpolation_steps`, and `subtask_term_offset_range` in `configs/go_single_move_task.json`.
- Increase `--num-demos` for both source and generated datasets.

## OpenVLA Next Step

OpenVLA fine-tuning expects RLDS. The collection and MimicGen generation steps
above should still be followed as before; the additional step is the HDF5 ->
RLDS conversion pass, which now also performs the action / timing / language
preprocessing listed in Step 6.
