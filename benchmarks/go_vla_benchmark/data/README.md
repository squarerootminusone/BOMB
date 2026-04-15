# Go VLA Benchmark — Data Directory

## Structure

```
data/
├── datasets/          Production HDF5 source datasets
├── test/              Test/debug HDF5s + paired preview media
├── eval_videos/       Checkpoint evaluation recordings
├── example_videos/    5-demo sample videos per RLDS version
├── motus/             Motion trajectories in Motus format
├── previews/          Dataset preview videos/images
└── debug/             One-off investigation artifacts
```

## datasets/

Production HDF5 files used to build RLDS datasets for OpenVLA fine-tuning.

| File | Demos | Actions | Images | RLDS Version | Description |
|------|-------|---------|--------|--------------|-------------|
| `source_go.hdf5` | 250 | (T, 4) | (T, 256, 256, 3) | 3.0.0 | 8 Hz, board/lighting randomization, dedup |
| `source_go_200.hdf5` | 200 | (T, 4) | (T, 256, 256, 3) | 2.1.0 | 8 Hz, taller stones (12mm), no perturbations |
| `augmented_go.hdf5` | 250 | (T, 4) | (T, 256, 256, 3) | 2.0.0 | MimicGen-augmented demos |

## test/

Test/debug HDF5 datasets and their paired preview videos/images.

| File | Demos | Description |
|------|-------|-------------|
| `test_5_demos.hdf5` | 5 | Standard OpenVLA test set |
| `test_5_demos_oft.hdf5` | 5 | OpenVLA-OFT test set |
| `test_5_no_perturb.hdf5` | 5 | No perturbations |
| `test_5_taller.hdf5` | 5 | Taller (12mm) stones |
| `test_224.hdf5` | 5 | 224px resolution |
| `test5.hdf5` | 5 | Early test file |
| `rotation_test.hdf5` | 5 | Board rotation sweep (7-DOF) |
| `perturbation_preview{,2,3}.hdf5` | 5 each | Perturbation visualization |

Paired media: `test_oft_demo_*.mp4`, `test_noperturb_demo_*.mp4`, `test_taller_demo_*.mp4`, `*_preview.mp4`, `*_frame0.png`.

## eval_videos/

Checkpoint evaluation recordings, normalized to `step_NNNN` naming.

| Directory | Step | Episodes | Notes |
|-----------|------|----------|-------|
| `step_0750/` | 750 | 6 | Early checkpoint |
| `step_1500/` | 1500 | 6 | |
| `step_3000_ol2/` | 3000 | 10 | OL2 variant |
| `step_5000/` | 5000 | 8 | |
| `step_5250/` | 5250 | 5 | |
| `step_6000/` | 6000 | 5 | |
| `step_20000_film/` | 20000 | 10 | FiLM conditioning |

## example_videos/

5 sample episode recordings per RLDS dataset version, rendered at 8 fps.

| Version | Avg Frames | Description |
|---------|------------|-------------|
| `v1.0.0/` | ~187 | 7-DOF, no dedup |
| `v2.0.0/` | ~178 | 4-DOF, MimicGen augmented |
| `v2.1.0/` | ~63 | 4-DOF, taller stones, no perturbations |
| `v3.0.0/` | ~83 | 4-DOF, randomization + dedup |

## motus/

Motion trajectories in Motus format (RoboTwin ecosystem).

| Directory | Trajectories | Contents |
|-----------|-------------|----------|
| `50/` | 50 | qpos (.pt), videos (.mp4), metas (.txt) |
| `500/` | 500 | qpos (.pt), videos (.mp4), metas (.txt) |
| `sample_videos/` | 2 | Example episode recordings |

## previews/

Dataset preview media for documentation/reference.

- `source_go_preview.mp4` — preview of source_go.hdf5
- `source_go_agentview.mp4` — agentview of source_go.hdf5
- `augmented_go_preview.mp4` — preview of augmented_go.hdf5
- `rlds_sample_frame.png` — sample RLDS frame
- `preview_frames/` — 20 PNGs (demo start/mid/spawn frames)
- `preview_demos/` — 7 MP4s (success/fail recordings)

## debug/

One-off investigation and analysis artifacts.

- `demo_attempts.mp4`, `failed_demos.mp4` — collection debug recordings
- `robosuite_failed_preview.mp4` — env test
- `augment_debug.mp4` — augmentation debug
- `intersection_rotation_sweep.mp4` — rotation analysis
- `intersection_plots/` — board intersection analysis under rotation (4 PNGs)
