#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
BENCH_DIR="$ROOT_DIR/benchmarks/go_vla_benchmark"
CAMERA_SIZE="${CAMERA_SIZE:-84}"
ENV_NAME="${ENV_NAME:-go_7x7}"

# Install benchmark dependencies in the currently active environment.
python -m pip install -r "$BENCH_DIR/requirements.txt"
python -m pip install --no-deps -e "git+https://github.com/ARISE-Initiative/robomimic.git@d0b37cf214bd24fb590d182edb6384333f67b661#egg=robomimic"
python -m pip install -e "$ROOT_DIR/mimicgen"

action_python() {
  PYTHONPATH="$ROOT_DIR/deepmind-research:$ROOT_DIR/mimicgen:$ROOT_DIR/robosuite:$BENCH_DIR:${PYTHONPATH:-}" \
    python "$@"
}

action_python "$BENCH_DIR/scripts/check_setup.py" --strict

action_python "$BENCH_DIR/scripts/collect_source_demos.py" \
  --output "$BENCH_DIR/data/source_go.hdf5" \
  --environment-name "$ENV_NAME" \
  --num-demos 40 \
  --camera-size "$CAMERA_SIZE" \
  --action-scale 0.03 \
  --controller-divisor 2.5 \
  --success-hold-steps 40 \
  --detour-steps 20 \
  --detour-radius 0.18 \
  --approach-steps 30 \
  --press-steps 16 \
  --retreat-steps 22 \
  --side-transfer-steps 28 \
  --side-margin 0.18

action_python "$BENCH_DIR/scripts/generate_augmented_demos.py" \
  --source "$BENCH_DIR/data/source_go.hdf5" \
  --output "$BENCH_DIR/data/augmented_go.hdf5" \
  --task-config "$BENCH_DIR/configs/go_single_move_task.json" \
  --environment-name "$ENV_NAME" \
  --num-demos 200 \
  --camera-size "$CAMERA_SIZE" \
  --action-scale 0.03 \
  --success-hold-steps 20

action_python "$BENCH_DIR/scripts/inspect_dataset.py" \
  --dataset "$BENCH_DIR/data/augmented_go.hdf5"
