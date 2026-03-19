# MuJoGo Runtime

## Prerequisites

- Python 3.11
- Conda (Miniconda or Anaconda)
- Homebrew (macOS)

## Setup

### 1. Install GNU Go

```bash
brew install gnu-go
```

### 2. Create conda environment

```bash
conda create -n mujogo python=3.11 -y
conda activate mujogo
pip install -r requirements.txt
```

### 3. Clone the source

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/deepmind-research.git
cd deepmind-research
git sparse-checkout set physics_planning_games
```

## Running

```bash
conda activate mujogo
PYTHONPATH=deepmind-research python deepmind-research/physics_planning_games/explore.py --environment_name=go_7x7
```

### Available environments

| Name | Description |
|------|-------------|
| `go_7x7` | 7x7 Go with GnuGo opponent |
| `tic_tac_toe_markers_features` | Tic-tac-toe with random opponent |
| `tic_tac_toe_mixture_opponent_markers_features` | Tic-tac-toe with mixture opponent |
| `tic_tac_toe_optimal_opponent_markers_features` | Tic-tac-toe with optimal opponent |

## Notes

- The benchmark now resolves `gnugo` from `--gnugo-path`, `GNUGO_PATH`, or `PATH`; no dependency-package edits are needed.
- The viewer opens a MuJoCo interactive window where a Jaco robot arm interacts with the board.
- The `explore.py` script above is focused on board games in this setup. The mujoban environment requires additional dependencies (`six`, `boxoban`).

## Go + MimicGen Benchmark

Standalone benchmark code (outside dependencies) is in:
`benchmarks/go_vla_benchmark`

Quick run:

```bash
conda activate mujogo
pip install -r benchmarks/go_vla_benchmark/requirements.txt
pip install --no-deps -e "git+https://github.com/ARISE-Initiative/robomimic.git@d0b37cf214bd24fb590d182edb6384333f67b661#egg=robomimic"
pip install -e mimicgen
export PYTHONPATH="$PWD/deepmind-research:$PWD/mimicgen:$PWD/benchmarks/go_vla_benchmark:$PYTHONPATH"

python benchmarks/go_vla_benchmark/scripts/check_setup.py --strict

python benchmarks/go_vla_benchmark/scripts/collect_source_demos.py \
  --output benchmarks/go_vla_benchmark/data/source_go.hdf5 \
  --num-demos 40 \
  --action-scale 0.01 \
  --controller-divisor 10 \
  --success-hold-steps 20 \
  --detour-steps 25 \
  --detour-radius 0.20 \
  --approach-steps 40 \
  --press-steps 20 \
  --retreat-steps 25

python benchmarks/go_vla_benchmark/scripts/generate_augmented_demos.py \
  --source benchmarks/go_vla_benchmark/data/source_go.hdf5 \
  --output benchmarks/go_vla_benchmark/data/augmented_go.hdf5 \
  --task-config benchmarks/go_vla_benchmark/configs/go_single_move_task.json \
  --num-demos 200 \
  --action-scale 0.01 \
  --success-hold-steps 20
```

If install fails on `egl_probe`, run:

```bash
pip install --upgrade "cmake<4"
pip install --no-cache-dir -r benchmarks/go_vla_benchmark/requirements.txt
pip install --no-deps -e "git+https://github.com/ARISE-Initiative/robomimic.git@d0b37cf214bd24fb590d182edb6384333f67b661#egg=robomimic"
```

To visually inspect exactly what is stored in HDF5:

```bash
python benchmarks/go_vla_benchmark/scripts/export_agentview_video.py \
  --dataset benchmarks/go_vla_benchmark/data/source_go.hdf5 \
  --output benchmarks/go_vla_benchmark/data/source_go_preview.mp4 \
  --num-demos 40 \
  --fps 4 \
  --macro-block-size 1
```
