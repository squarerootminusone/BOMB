# Intervention Tests

This stage extends the existing local-explanation pipeline with counterfactual
input perturbations over the same RLDS-aligned clips:

- patch occlusion
- text masking
- minimal counterfactual edits

It reuses the same Go HDF5 loader, RLDS-style frame selection, prompt builder,
and OpenVLA action decoding already used in stage 1.

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

## Run Collection

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
  --attn-implementation eager

python benchmarks/go_vla_benchmark/scripts/export_openvla_intervention_report.py \
  --trace-input benchmarks/go_vla_benchmark/data/interventions/source_go_interventions_text.npz \
  --output-dir benchmarks/go_vla_benchmark/data/interventions/report_text
```

## Main Algorithms

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
    find the instruction-token positions inside the prompt
    for each instruction token:
        zero its attention-mask entry while keeping the rest of the prompt fixed
        teacher-force the baseline target tokens with that masked prompt
        measure log-prob drop relative to the clean target-token probabilities
    rank masked tokens by log-prob drop
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

- The counterfactual search is a greedy approximation to a minimal edit set. It
  is intentionally transparent: every cumulative edit is stored in the trace.
- `--runs` lets you execute only the expensive part you need.
- All traces stay self-contained, so the report exporter does not rerun the model.
