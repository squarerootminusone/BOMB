# Go VLA Benchmark

## Overview

This project builds an OpenVLA fine-tuning dataset from simulated Go stone manipulation. The pipeline:

1. **Robosuite env** (`robosuite_go_env.py`): A 5×5 Go board with rigid-body stones on a table, a Panda arm, randomized board position/rotation, and domain-randomized lighting/textures.
2. **Collect source demos** (`collect_source_demos.py`): A scripted controller picks up a stone from the source tray and places it at a legal intersection. Produces ~40 MimicGen-compatible seed trajectories with `datagen_info`.
3. **MimicGen augmentation** (`generate_augmented_demos.py`): Recombines trajectory segments across different board states/targets to produce hundreds of augmented demos.
4. **RLDS conversion** (`convert_to_rlds.py`): Converts the HDF5 dataset to RLDS format for OpenVLA fine-tuning.
5. **Eval** (`eval_openvla.py`): Runs a fine-tuned OpenVLA checkpoint against the env and records success rate + videos.

Each episode = one stone placement (pick from source tray → place on board intersection). The env wrapper (`GoRobosuiteBenchmarkEnv`) handles the game logic, opening positions, move legality (via OpenSpiel), and observation/action space.

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

## Coordinate System Gotchas

- **Board intersection positions must come from MuJoCo's forward kinematics** (`geom_xpos` + `geom_xmat`), not from manual reconstruction with `board_center_xy` + `_board_rotation_rad`. The manual math double-counts transforms that MuJoCo already applies through the table body hierarchy.
- **`sim.render()` returns bottom-to-top (OpenGL convention)** and is flipped with `[::-1]`. Any world-to-pixel projection must use `(height - 1) * 0.5` (not `height * 0.5`) as the principal point to account for the flip's off-by-one.
- **Robosuite scene cameras**: `agentview` (default), `birdview` (top-down), `frontview`, `sideview`, `robot0_eye_in_hand`. Use `sim.render(camera_name=...)` to switch.
- **Visual vs collision geoms share the same `pos`/`quat`** — they are co-located. Grid lines are painted via texture on the visual geom, not via the invisible `go_line_*` geoms (which exist only for broadphase stability).

## Remote Setup

- OpenVLA-OFT: [openvla-oft/SETUP_REMOTE.md](openvla-oft/SETUP_REMOTE.md)
- RoboTwin: [benchmarks/RoboTwin/SETUP_REMOTE.md](benchmarks/RoboTwin/SETUP_REMOTE.md)
- Motus training: [benchmarks/RoboTwin/SETUP_MOTUS.md](benchmarks/RoboTwin/SETUP_MOTUS.md)

---

# RoboTwin 2.0 — Go Stone Placement

## Overview

The `robotwin-2` branch ports the Go stone manipulation task to **RoboTwin 2.0** (SAPIEN-based), outputting data in **Motus format** for Motus training/inference. Uses ALOHA dual-arm robot with left arm active, right arm at home pose (14D qpos native to Motus).

## RoboTwin Gotchas

- **SAPIEN/PhysX cylinder primitives have their height axis along X.** Both collision and render shapes need a 90° Y rotation (`set_local_pose`) so the flat faces are parallel to the table. Always rotate both together or physics and visuals will disagree.
- **Board intersection positions must come from SAPIEN's `actor.get_pose().to_transformation_matrix()`** — same principle as MuJoCo's `geom_xpos`.
- **EE-to-stone offset**: After grasping, the stone hangs below the end-effector. Target poses must compensate for this offset or the stone will land ~3-5cm off.
- **Blackwell (SM 120) GPUs** need PyTorch 2.7+ and CUDA 12.8+ nvcc. PyTorch 2.4.x only supports up to SM 90.
- **OIDN denoiser errors** on Blackwell are cosmetic — rendering still works fine.
- **Table color must stay mid-tone** (never pure white or black) to avoid confusion with stones.
