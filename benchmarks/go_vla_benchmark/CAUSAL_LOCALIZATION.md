# Internal Causal Localization

This stage runs activation patching on the same RLDS-aligned clips used by the
local explanation and intervention stages. It localizes which internal decoder
layers and attention heads restore the clean target-token probability when a
corrupted input is patched with clean activations.

It supports:

- activation patching over decoder layers
- activation patching over individual attention heads
- optional patching over discovered cross-attention-style blocks, if the loaded
  checkpoint exposes them

## Inputs

- an HDF5 Go dataset loaded through the existing RLDS-style clip adapter
- an OpenVLA checkpoint
- a corruption family:
  - `patch-occlusion`
  - `text-masking`

## Outputs

Collection writes one reusable trace:

- `openvla_causal_localization_v1` `.npz`
- optional summary JSON manifest

Report export writes:

- one folder per demo
- one JSON and one Markdown file per step
- one restoration heatmap PNG per step

## Run Collection

Use the strongest patch occlusion per step as the corrupted input:

```bash
python benchmarks/go_vla_benchmark/scripts/collect_openvla_causal_localization.py \
  --dataset benchmarks/go_vla_benchmark/data/source_go.hdf5 \
  --checkpoint /abs/path/to/openvla-checkpoint \
  --trace-output benchmarks/go_vla_benchmark/data/causal/source_go_patch_causal.npz \
  --summary-output benchmarks/go_vla_benchmark/data/causal/source_go_patch_causal.json \
  --num-demos 4 \
  --stride 4 \
  --corruption-type patch-occlusion \
  --attn-implementation eager
```

Use the strongest text mask per step instead:

```bash
python benchmarks/go_vla_benchmark/scripts/collect_openvla_causal_localization.py \
  --dataset benchmarks/go_vla_benchmark/data/source_go.hdf5 \
  --checkpoint /abs/path/to/openvla-checkpoint \
  --trace-output benchmarks/go_vla_benchmark/data/causal/source_go_text_causal.npz \
  --num-demos 4 \
  --stride 4 \
  --corruption-type text-masking \
  --attn-implementation eager
```

Pin a specific corrupted patch or token index:

```bash
python benchmarks/go_vla_benchmark/scripts/collect_openvla_causal_localization.py \
  --dataset benchmarks/go_vla_benchmark/data/source_go.hdf5 \
  --checkpoint /abs/path/to/openvla-checkpoint \
  --trace-output benchmarks/go_vla_benchmark/data/causal/manual_index.npz \
  --corruption-type patch-occlusion \
  --corruption-index 0 \
  --num-demos 1 \
  --max-steps 1 \
  --attn-implementation eager
```

Also patch cross-attention-style blocks when the checkpoint exposes them:

```bash
python benchmarks/go_vla_benchmark/scripts/collect_openvla_causal_localization.py \
  --dataset benchmarks/go_vla_benchmark/data/source_go.hdf5 \
  --checkpoint /abs/path/to/openvla-checkpoint \
  --trace-output benchmarks/go_vla_benchmark/data/causal/with_cross_blocks.npz \
  --corruption-type patch-occlusion \
  --per-cross-attention \
  --num-demos 1 \
  --max-steps 1 \
  --attn-implementation eager
```

## Export Report

```bash
python benchmarks/go_vla_benchmark/scripts/export_openvla_causal_localization_report.py \
  --trace-input benchmarks/go_vla_benchmark/data/causal/source_go_patch_causal.npz \
  --output-dir benchmarks/go_vla_benchmark/data/causal/report
```

## Main Algorithms

### Activation Patching Over Layers

```text
for each RLDS-selected frame:
    run clean decoding once and keep the clean action-token sequence
    build a corrupted input by either:
        occluding the strongest image patch, or
        masking the strongest instruction token
    teacher-force the clean target tokens on both clean and corrupted inputs
    cache every decoder-layer output on the clean run
    for each decoder layer:
        rerun the corrupted forward pass
        replace that layer's action-token activations with the clean cached activations
        score how much the clean target-token log-probability is restored
```

### Activation Patching Over Heads

```text
for each RLDS-selected frame:
    reuse the same clean and corrupted teacher-forced passes
    cache the tensor entering each attention o_proj layer on the clean run
    for each layer and each head slice:
        rerun the corrupted forward pass
        replace only that head slice at the action-token positions
        score the restoration of the clean target-token log-probability
```

### Optional Cross-Attention Blocks

```text
discover modules whose names look like cross-attention blocks
if any are present:
    cache their clean outputs
    rerun the corrupted forward pass once per block
    replace that block's action-token activations with the clean cached activations
    record the same normalized restoration score
if none are present:
    keep the decoder-layer and attention-head localization only
```

## Notes

- OpenVLA is typically a decoder-only VLM, so the most important localization
  signal usually comes from decoder layers and self-attention heads.
- `--per-cross-attention` is optional by design because some checkpoints will
  not expose separate cross-attention blocks.
- The stored score is a normalized restoration ratio:
  `(patched - corrupted) / (clean - corrupted)` on target-token log-probability.

