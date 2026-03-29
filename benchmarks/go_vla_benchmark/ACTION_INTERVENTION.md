# Intervention Tests

Basic implementation of the fundaments from: https://www.alphaxiv.org/abs/2509.22496
The interventions are done offline using recorded frames from a .hdf5 file.

This has nothing to do with the interanal weight of hte model/attention maps. It reruns the model with an intervention (text of image patch) and sees how the original(correct) action likelihoods change.

This stage extends the existing local-explanation pipeline with counterfactual
input perturbations over the same RLDS-aligned clips:

- patch occlusion
- text masking
- minimal counterfactual edits
## Inputs

- an HDF5 Go dataset loaded through the existing RLDS-style clip adapter
- an OpenVLA checkpoint
- the same prompt style and action de-normalization settings used in local explanations

## Outputs

Collection writes one reusable trace:

- `openvla_intervention_tests_v1` `.npz`
- optional summary JSON manifest

Report export writes:

- one folder per demo
- one JSON and one Markdown file per step
- one intervention-panel PNG per step when patch occlusion or text masking is enabled

Both collection and report export keep demos in the same order exposed by the
source HDF5 file, so the `.npz`, report index, and per-demo folders stay easy
to compare side by side. If you use `--top-k` during export, it keeps that
subset in dataset order instead of re-sorting the report by score.

## Run Collection

**For the reports, you can use --cross-step-comparison to get normalisation over all of the steps, rather than stepwise. The exporter always writes a third stitched panel with a plain bilinear-smoothed overlay in addition to the raw frame and the gridded patch view.**


Run all three intervention families:

```bash
conda run --no-capture-output -n main \
  python benchmarks/go_vla_benchmark/scripts/collect_openvla_intervention_tests.py \
  --dataset /root/dsait4125/benchmarks/go_vla_benchmark/data/source_go.hdf5 \
  --checkpoint /root/16-18-14/checkpoints/best-merged \
  --trace-output /root/dsait4125/benchmarks/go_vla_benchmark/data/interventions/source_go_interventions.npz \
  --summary-output /root/dsait4125/benchmarks/go_vla_benchmark/data/interventions/source_go_interventions.json \
  --num-demos 5 \
  --stride 4 \
  --top-k 8 \
  --max-counterfactual-edits 4 \
  --device cuda:0 \
  --attn-implementation eager

python benchmarks/go_vla_benchmark/scripts/export_openvla_intervention_report.py \
  --trace-input benchmarks/go_vla_benchmark/data/interventions/source_go_interventions.npz \
  --output-dir benchmarks/go_vla_benchmark/data/interventions/report
```

Run only patch occlusion:

```bash
conda run --no-capture-output -n main \
  python benchmarks/go_vla_benchmark/scripts/collect_openvla_intervention_tests.py \
  --dataset /root/dsait4125/benchmarks/go_vla_benchmark/data/source_go.hdf5 \
  --checkpoint /root/16-18-14/checkpoints/best-merged \
  --trace-output /root/dsait4125/benchmarks/go_vla_benchmark/data/interventions/source_go_interventions_patches.npz \
  --summary-output /root/dsait4125/benchmarks/go_vla_benchmark/data/interventions/source_go_interventions_patches.json \
  --runs patch-occlusion \
  --num-demos 5 \
  --stride 4 \
  --device cuda:0 \
  --attn-implementation eager

python benchmarks/go_vla_benchmark/scripts/export_openvla_intervention_report.py \
  --trace-input benchmarks/go_vla_benchmark/data/interventions/source_go_interventions_patches.npz \
  --output-dir benchmarks/go_vla_benchmark/data/interventions/report_patches
```

Run only text masking:

```bash
conda run --no-capture-output -n main \
  python benchmarks/go_vla_benchmark/scripts/collect_openvla_intervention_tests.py \
  --dataset /root/dsait4125/benchmarks/go_vla_benchmark/data/source_go.hdf5 \
  --checkpoint /root/16-18-14/checkpoints/best-merged \
  --trace-output /root/dsait4125/benchmarks/go_vla_benchmark/data/interventions/source_go_interventions_text.npz \
  --summary-output /root/dsait4125/benchmarks/go_vla_benchmark/data/interventions/source_go_interventions_text.json \
  --runs text-masking \
  --num-demos 5 \
  --stride 4 \
  --device cuda:0 \
  --attn-implementation eager

python benchmarks/go_vla_benchmark/scripts/export_openvla_intervention_report.py \
  --trace-input benchmarks/go_vla_benchmark/data/interventions/source_go_interventions_text.npz \
  --output-dir benchmarks/go_vla_benchmark/data/interventions/report_text
```

## Main Algorithms
the intuition would be: 
effect(patch j) = baseline_logprob - occluded_logprob_j



In more detail, for the probabilities, this is how we dod it
1. Greedy decoded-token probabilities
   p_t = P(y_t | image, prompt, previous decoded tokens)
   where y_t is the token chosen by argmax at step t

2. Teacher-forced reference-sequence probabilities
   p_t^ref = P(y_t^clean | modified image/prompt, previous clean tokens)
   where y_t^clean is the token from the original clean baseline decode

So we're just lookingfor any action changes under the modified output.


For the .png outputs: when patch occlusion is present, the exporter stitches three images side by side: left = raw frame; middle = the patch effect map resized with nearest-neighbor, colorized, alpha-blended, and annotated with the black patch grid / indices; right = a plain bilinear-smoothed version of that same overlay, with no grid and no numbers. For text-masking-only exports, the step PNG is just the raw frame.
For the text masking we just set the attention_mask = 0 during inference for those specific tokens


### Patch Occlusion

```text
for each RLDS-selected frame:
    run clean OpenVLA decoding once
    keep the predicted action-token sequence as the baseline target
    split the image into the same square patch grid used by the VLM projector
    for each patch:
        replace that patch with a frame-mean color block
        teacher-force the baseline target tokens on the occluded image
        measure log-prob drop relative to the clean target-token probabilities
    rank patches by log-prob drop
    re-decode the top-k occluded variants and store their predicted tokens/actions
```

### Text Masking

```text
for each RLDS-selected frame:
    run clean OpenVLA decoding once
    build word / phrase masking candidates from the instruction text
    always include meaningful spans such as colors, "stone", "column", and
    coordinate phrases like "(4, 4)" alongside the individual number words
    for each candidate span:
        zero the attention-mask entries for all prompt tokens covered by that span
        teacher-force the baseline target tokens with that masked prompt
        measure log-prob drop relative to the clean target-token probabilities
    rank masked spans by log-prob drop
    re-decode the top-k masked variants and store their predicted tokens/actions
```

### Minimal Counterfactual Edits

```text
for each RLDS-selected frame:
    start from the single-edit rankings from patch occlusion and text masking
    merge both candidate lists into one score-sorted queue
    greedily add the strongest remaining edit to the active edit set
    after each added edit:
        apply all selected image occlusions and text masks together
        re-decode the frame with OpenVLA
        teacher-force the original baseline target tokens to measure cumulative log-prob drop
        stop when the predicted action-token sequence changes
    save the full greedy trajectory, not just the final edit set
```

## Notes

- The collection script now prints a startup JSON block to `stderr` with the
  requested device, resolved device, and first model-parameter device. On a GPU
  run these should all report `cuda:0`.
- The counterfactual search is a greedy approximation to a minimal edit set. It
  is intentionally transparent: every cumulative edit is stored in the trace.
- `--runs` lets you execute only the expensive part you need.
- All traces stay self-contained, so the report exporter does not rerun the model.

## Online Task-Level Report

The simulator-rerun trajectory intervention path has grown beyond a short
appendix here. Its documentation now lives in
[`Trajectory_intervention.md`](./Trajectory_intervention.md).

That document covers:

- dataset-driven text and patch mask selection from the Go HDF5 demos
- simulator reset from recorded demo opening history
- publication PNG rendering and rollout markers
- baseline MP4 export
- dataset-order multi-demo export behavior
