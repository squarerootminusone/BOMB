# Trajectory Intervention

This document covers the online task-level intervention path for the Go VLA
benchmark.

The offline per-action intervention pipeline still lives in
[`ACTION_INTERVENTION.md`](./ACTION_INTERVENTION.md). That path perturbs a
recorded frame and measures how the predicted action tokens change. The
trajectory path instead reruns the simulator and measures how masking changes
the full rollout.

## Feature Summary

There are now two explainability paths:

- the original offline per-action intervention pipeline in
  [`ACTION_INTERVENTION.md`](./ACTION_INTERVENTION.md)
- the newer online task-level intervention pipeline described here

The online path is implemented through:

- rollout collection in
  [`go_vla_benchmark/explainability/online_interventions.py`](./go_vla_benchmark/explainability/online_interventions.py)
- publication PNG export in
  [`go_vla_benchmark/explainability/reporting_online_interventions.py`](./go_vla_benchmark/explainability/reporting_online_interventions.py)
- the CLI entrypoint in
  [`scripts/export_openvla_online_intervention_report.py`](./scripts/export_openvla_online_intervention_report.py)

## Current Behavior

The online report uses the real simulator and online inference. It supports:

- text masking
- patch masking

The report layout stays publication-style:

- the unmasked trajectory is shown on the left
- masked attempts are overlaid in one shared plot on the right
- the legend stays at the bottom
- the exported PNG contains no extra GUI text

The plotted trajectory is the arm / end-effector path shown in a true top-down
`x,y` view. Height is no longer mapped into image position. Instead, `z`
controls trace opacity: the path is most opaque near table level and fades to
30% opacity at the highest point reached in the rollout.

Each segment is colored by task phase:

- `move_to_puck`
- `pick_up_puck`
- `move_puck`
- `drop_puck`

Phase coloring reflects environment events rather than a heuristic:

- `pick_up_puck` starts only after a real grasp
- `move_puck` starts only after the grasped puck leaves the source region or
  lifts away from it
- `drop_puck` starts on release or commit

## Dataset-Driven Masking And Reset

The trajectory exporter can now take its intervention source from the same Go
HDF5 dataset used by the offline action-intervention pipeline.

For each selected demo it:

- keeps demo order in the order exposed by the HDF5 file
- rebuilds the natural-language instruction from the dataset board state
- derives the target row and column from the board-state delta
- restores the stone color from the dataset metadata
- replays the recorded opening move history when `obs/move_history` is present
- falls back to the opening-stone count from the first board state when move
  history is missing
- chooses the masking source frame from the same RLDS-style keep-index logic
  used by the offline collectors

Mask selection is dataset-driven as well:

- text masking is chosen from the reference HDF5 frame by running the offline
  text-masking scan and taking the strongest candidate
- patch masking is chosen from the reference HDF5 frame by running the offline
  patch-occlusion scan and taking the strongest candidate

This keeps the online and offline intervention paths anchored to the same
recorded demo content instead of relying on a hand-written target phrase only.

## Trajectory Markers

The rollout format did not need a structural rewrite. It already records

- `gripper_action`
- `stone_grasped`
- `move_committed`
- per-step `phase`

The renderer now overlays three marker types on both panels:

- hollow markers where the gripper is closing but the stone is not grasped yet:
  pickup attempt
- filled markers where `stone_grasped=True`: actual hold / transport
- release markers where grasp changes `True -> False` or
  `move_committed=True`: release / commit

## Manifest Contents

The JSON manifest includes:

- full trajectories
- phase transition steps
- event steps
- `ever_grasped`
- `ever_moved_puck`
- `ever_released`
- the resolved intervention kind and selected text or patch mask
- the dataset demo key and reference frame index when dataset mode is used

## Video

The unmasked rollout writes an MP4 by default from the same export command.

- default video path: `<output-png stem>_baseline.mp4`
- override with `--baseline-video-output`
- control playback speed with `--video-fps`

The rollout collector already has a frame callback hook, so baseline frames are
streamed directly to the video writer.

## Manual Run

Manual mode still works when you want to specify the target directly and mask a
text span by label or index:

```bash
python benchmarks/go_vla_benchmark/scripts/export_openvla_online_intervention_report.py \
  --checkpoint /abs/path/to/openvla-checkpoint \
  --output-png benchmarks/go_vla_benchmark/data/interventions/online_task_report.png \
  --summary-output benchmarks/go_vla_benchmark/data/interventions/online_task_report.json \
  --target-row 3 \
  --target-col 4 \
  --attempts 5 \
  --max-steps 200
```

## Dataset-Driven Run

Dataset mode resolves the instruction, reset state, and intervention candidate
from the HDF5 demos:

```bash
python benchmarks/go_vla_benchmark/scripts/export_openvla_online_intervention_report.py \
  --checkpoint /abs/path/to/openvla-checkpoint \
  --dataset benchmarks/go_vla_benchmark/data/source_go.hdf5 \
  --output-png benchmarks/go_vla_benchmark/data/interventions/online_task_report.png \
  --summary-output benchmarks/go_vla_benchmark/data/interventions/online_task_report.json \
  --intervention-kind text \
  --attempts 5 \
  --max-steps 200
```

Use patch masking instead:

```bash
python benchmarks/go_vla_benchmark/scripts/export_openvla_online_intervention_report.py \
  --checkpoint /abs/path/to/openvla-checkpoint \
  --dataset benchmarks/go_vla_benchmark/data/source_go.hdf5 \
  --output-png benchmarks/go_vla_benchmark/data/interventions/online_task_report.png \
  --summary-output benchmarks/go_vla_benchmark/data/interventions/online_task_report.json \
  --intervention-kind patch \
  --attempts 5 \
  --max-steps 200
```

Useful selection flags:

- `--demos` to restrict to specific demo keys
- `--start` and `--num-demos` to take a dataset-order slice
- `--stride` to match the reference-frame sampling used by the offline
  intervention collector
- `--reference-step` to choose which kept frame supplies the mask

If multiple demos are selected, the exporter writes one PNG / JSON / MP4 set per
demo with a demo-key suffix while keeping the dataset order.

## Debugging Note

The earlier "200 vs 1000 steps look identical" issue was real but was not
caused by `max_steps` being ignored. In the inspected case the first 200 steps
were deterministic, and the longer rollout only jittered after that by roughly
`7.3e-05 m`, so the PNGs looked visually identical. The robot also got stuck
early, never grasped the puck, and never committed the move.
