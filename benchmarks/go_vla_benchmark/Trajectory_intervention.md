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
- dataset-driven report export from recorded Go demos
- simulator-driven random report export with `--simulator-demos`
- policy inputs that now match the fine-tune rollout view more closely by using
  the simulator end-effector overlay plus the same 90% center crop before
  model inference

The report layout stays paper-style:

- the exported PNG is a clean Matplotlib-style figure rather than a UI card
- the unmasked trajectory is shown on the left
- masked attempts are overlaid in one shared plot on the right
- in dataset-driven mode the right panel uses the top `--attempts` ranked text
  spans or patch candidates from the reference frame instead of rerunning one
  identical mask repeatedly
- each masked rollout stays identifiable through its legend entry and color
- the bottom legends group task phases, event markers, and the unmasked plus
  masked attempt labels in a paper-friendly layout
- the exported PNG contains no extra GUI text

The plotted trajectory is the arm / end-effector path shown in a true top-down
`x,y` view. By default opacity fades by time: the run starts at 70% opacity and
ramps up to full opacity at the end, so older trajectory segments are more
transparent and newer segments are more opaque.

If you prefer a height-based fade instead, pass
`--height-trajectory-color-degradation`. In that mode the path is most opaque
near table level and fades to 70% opacity at the highest point reached in the
rollout.

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
  text-masking scan over the fixed RLDS 7-mask shortlist derived from the
  builder templates: `{color}`, `{r}`, `{c}`, the combined `{r}/{c}` phrase,
  plus three multi-word template phrases
- the exporter still runs at most `--attempts` masked rollouts from that
  ranking, so the default setting produces 5 per-demo PNG / JSON / MP4 sets
- patch masking is chosen from the reference HDF5 frame by running the offline
  patch-occlusion scan and taking the top ranked candidates up to `--attempts`

This keeps the online and offline intervention paths anchored to the same
recorded demo content instead of relying on a hand-written target phrase only.

## Simulator-Driven Random Runs

You can also skip HDF5 input entirely and ask the exporter to sample fresh
simulator scenes with `--simulator-demos`.

For each sampled simulator demo it:

- resets a fresh random Go scene directly in the simulator
- samples a random opening length in the configured simulator opening range
- samples a random stone color
- lets the environment choose a legal target move
- uses the reset frame itself as the reference frame for ranking text or patch
  masks
- reuses that exact reset state for the baseline and masked rollouts

This is the easiest way to run text or patch masking online when you do not
have an HDF5 dataset on hand.

Reset-time randomness is also anchored per report now:

- different demos no longer restart from the same reset-1 RNG state when they
  share the same CLI `--seed`
- the exporter derives a stable per-demo reset seed from the base seed and the
  dataset demo identity
- baseline and masked attempts for one demo reuse that same reset seed, so
  board pose, lighting, camera, source-stone placement, fallback opening
  sampling, and other reset-time random variables stay matched for fair
  comparison
- rerunning with the same CLI `--seed` reproduces the same per-demo scenes

## Trajectory Markers

The rollout format did not need a structural rewrite. It already records

- `gripper_action`
- `stone_grasped`
- `move_committed`
- per-step `phase`

The renderer now overlays three marker types on both panels:

- hollow markers where the gripper is closing very aggressively but the stone is
  not grasped yet: pickup attempt
- filled markers where `stone_grasped=True`: actual hold / transport
- release markers where grasp changes `True -> False` or
  `move_committed=True`: release / commit

On the masked panel the renderer also aligns the masked rollout starting points
to the same `x,y` origin before drawing, so the overlaid paths compare how the
interventions diverge after the shared start rather than showing reset-offset
jitter.

## Manifest Contents

The JSON manifest includes:

- full trajectories
- phase transition steps
- event steps
- `ever_grasped`
- `ever_moved_puck`
- `ever_released`
- the resolved per-report `reset_seed`
- the resolved intervention kind and the leading selected text or patch mask
- the per-attempt mask attached to each masked rollout entry
- the dataset demo key and reference frame index when dataset mode is used

## Video

The export command now writes an MP4 for every rollout attempt, not just the
unmasked baseline.

- default baseline path: `<output-png stem>_baseline.mp4`
- default masked-attempt paths: `<output-png stem>_masked_attempt_01.mp4`,
  `<output-png stem>_masked_attempt_02.mp4`, and so on
- `--baseline-video-output` still sets the baseline path; masked-attempt files
  are written beside it using the same stem
- control playback speed with `--video-fps`
- each MP4 frame is written directly from the simulator camera feed, including
  the environment's own end-effector overlay when enabled
- the JSON manifest now includes an `output_videos` list with every saved MP4

The rollout collector already had a frame callback hook for the baseline, and
the masked attempts now use the same callback path so every rollout is streamed
directly to disk during export. The policy path uses those same frames, but
applies the same 90% center crop used in the fine-tune validation rollouts
before sending images to OpenVLA.

## Manual Run

Manual mode still works when you want to specify the target directly and mask a
text span by label or index. Manual patch masking now works as well: the script
first grabs the reset-frame image from the simulator, ranks patch masks on that
frame, and then reruns the rollout with the selected mask.

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

## Simulator-Driven Run

If you just want random simulator scenes instead of dataset-derived resets, use
`--simulator-demos`:

```bash
python benchmarks/go_vla_benchmark/scripts/export_openvla_online_intervention_report.py \
  --checkpoint /abs/path/to/openvla-checkpoint \
  --output-png benchmarks/go_vla_benchmark/data/interventions/online_task_report.png \
  --summary-output benchmarks/go_vla_benchmark/data/interventions/online_task_report.json \
  --intervention-kind text \
  --simulator-demos 3 \
  --attempts 7 \
  --max-steps 200
```

Switch to patch masking by changing `--intervention-kind patch`. When multiple
simulator demos are requested, the exporter writes one PNG / JSON / MP4 set per
sampled scene with a simulator-demo suffix.

## Debugging Note

The earlier "200 vs 1000 steps look identical" issue was real but was not
caused by `max_steps` being ignored. In the inspected case the first 200 steps
were deterministic, and the longer rollout only jittered after that by roughly
`7.3e-05 m`, so the PNGs looked visually identical. The robot also got stuck
early, never grasped the puck, and never committed the move.
