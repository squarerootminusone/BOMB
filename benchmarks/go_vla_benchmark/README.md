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
  --num-demos 40 \
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
  --num-demos 40 \
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

## Diversity Controls

Use these controls to create broader scenarios (different board states / moves):

- Increase `--opening-min` and `--opening-max` in both collection and generation.
- Increase `action_noise`, `num_interpolation_steps`, and `subtask_term_offset_range` in `configs/go_single_move_task.json`.
- Increase `--num-demos` for both source and generated datasets.

## OpenVLA Next Step

OpenVLA fine-tuning expects RLDS. This benchmark outputs MimicGen-style HDF5 first, so the next stage is an HDF5 -> RLDS conversion pass with task instructions and camera/image selection policy.
