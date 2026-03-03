# Project Rules

## Go VLA Benchmark — Absolute Rules
- **NEVER teleport stones.** All stone placement must happen through physics simulation. No `set_stone_pose` or `_place_stone_at_intersection` after the episode starts.
- **NEVER pin stone positions.** Stones must always remain fully dynamic rigid bodies.
- Use `conda run -n mujogo` to run all Python scripts in this project.

## Collecting & Generating Trajectories

```bash
# Collect source demonstrations (from repo root)
MUJOCO_GL=glfw conda run -n mujogo python benchmarks/go_vla_benchmark/scripts/collect_source_demos.py \
  --num-demos 40 \
  --camera-size 256 \
  --environment-name robosuite_go_5x5_rigid_bodies

# Generate augmented demos via MimicGen (from repo root)
MUJOCO_GL=glfw conda run -n mujogo python benchmarks/go_vla_benchmark/scripts/generate_augmented_demos.py \
  --input benchmarks/go_vla_benchmark/data/source_go.hdf5 \
  --camera-size 256
```

Key flags: `--num-demos` (number of source demos), `--camera-size` (image resolution), `--environment-name` (env variant).
