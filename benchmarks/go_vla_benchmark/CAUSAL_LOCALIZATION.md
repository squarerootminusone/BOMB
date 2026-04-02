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

- either:
  - an HDF5 Go dataset loaded through the existing RLDS-style clip adapter
  - random successful simulator demos collected on the fly with
    `--simulator-demos`
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

Both collection and report export keep demos in a stable input order:

- HDF5 runs keep the order exposed by the source dataset
- simulator runs keep the successful simulator-sampling order

That keeps the `.npz`, report index, and per-demo folders lined up for easy
comparison. If you use `--top-k` during export, it keeps that subset in input
order instead of re-sorting the report by score.

The exported PNG now annotates the layer rows and head columns directly in the
image:

- the left panel is one cell per decoder layer, labeled with `layer_idx`
- the right panel is one cell per `(layer_idx, head_idx)` pair, with the head
  axis numbered across the top
- color uses a fixed diverging restoration scale instead of per-panel min-max
  normalization:
  - blue: restoration `<= -1`
  - white: restoration `= 0`
  - red: restoration `>= +1`
  - values outside `[-1, +1]` are clipped before coloring

## Run Collection

Use the strongest patch occlusion per step as the corrupted input:

```bash
conda run --no-capture-output -n main \
  python benchmarks/go_vla_benchmark/scripts/collect_openvla_causal_localization.py \
  --checkpoint /root/16-18-14/checkpoints/best-merged \
  --trace-output /root/dsait4125/benchmarks/go_vla_benchmark/data/causal/simulator_patch_causal.npz \
  --summary-output /root/dsait4125/benchmarks/go_vla_benchmark/data/causal/simulator_patch_causal.json \
  --simulator-demos 4 \
  --stride 4 \
  --device cuda:0 \
  --corruption-type patch-occlusion \
  --attn-implementation eager

python benchmarks/go_vla_benchmark/scripts/export_openvla_causal_localization_report.py \
  --trace-input benchmarks/go_vla_benchmark/data/causal/simulator_patch_causal.npz \
  --output-dir benchmarks/go_vla_benchmark/data/causal/report_patch
```

Use the strongest text mask per step instead:

```bash
conda run --no-capture-output -n main \
  python benchmarks/go_vla_benchmark/scripts/collect_openvla_causal_localization.py \
  --dataset /root/dsait4125/benchmarks/go_vla_benchmark/data/source_go.hdf5 \
  --checkpoint /root/16-18-14/checkpoints/best-merged \
  --trace-output /root/dsait4125/benchmarks/go_vla_benchmark/data/causal/source_go_text_causal.npz \
  --summary-output /root/dsait4125/benchmarks/go_vla_benchmark/data/causal/source_go_text_causal.json \
  --num-demos 4 \
  --stride 4 \
  --device cuda:0 \
  --corruption-type text-masking \
  --attn-implementation eager

python benchmarks/go_vla_benchmark/scripts/export_openvla_causal_localization_report.py \
  --trace-input benchmarks/go_vla_benchmark/data/causal/source_go_text_causal.npz \
  --output-dir benchmarks/go_vla_benchmark/data/causal/report_text
```

Match an existing intervention trace exactly so the causal run reuses the same
demo order, the same per-demo frame indices, and the same best per-step
patch/text choice chosen during intervention testing:

```bash
conda run --no-capture-output -n main \
  python benchmarks/go_vla_benchmark/scripts/collect_openvla_causal_localization.py \
  --dataset /root/dsait4125/benchmarks/go_vla_benchmark/data/source_go.hdf5 \
  --checkpoint /root/16-18-14/checkpoints/best-merged \
  --trace-output /root/dsait4125/benchmarks/go_vla_benchmark/data/causal/source_go_patch_causal_matched.npz \
  --summary-output /root/dsait4125/benchmarks/go_vla_benchmark/data/causal/source_go_patch_causal_matched.json \
  --corruption-type patch-occlusion \
  --intervention-match \
  --device cuda:0 \
  --attn-implementation eager

python benchmarks/go_vla_benchmark/scripts/export_openvla_causal_localization_report.py \
  --trace-input benchmarks/go_vla_benchmark/data/causal/source_go_patch_causal_matched.npz \
  --output-dir benchmarks/go_vla_benchmark/data/causal/report_patch_matched
```

When `--intervention-match` is set, the causal collector ignores manual demo and
frame selection and instead follows the intervention `.npz` exactly. By
default it looks for a matching trace under
`benchmarks/go_vla_benchmark/data/interventions/` based on the dataset stem and
`--corruption-type`, but you can also pass an explicit intervention `.npz`
path as the value of `--intervention-match`. For simulator-backed intervention
traces, pass that explicit path because there is no dataset stem to infer from.

Pin a specific corrupted patch or token index:

```bash
conda run --no-capture-output -n main \
  python benchmarks/go_vla_benchmark/scripts/collect_openvla_causal_localization.py \
  --dataset /root/dsait4125/benchmarks/go_vla_benchmark/data/source_go.hdf5 \
  --checkpoint /root/16-18-14/checkpoints/best-merged \
  --trace-output /root/dsait4125/benchmarks/go_vla_benchmark/data/causal/source_go_patch_causal_manual_index.npz \
  --summary-output /root/dsait4125/benchmarks/go_vla_benchmark/data/causal/source_go_patch_causal_manual_index.json \
  --corruption-type patch-occlusion \
  --corruption-index 0 \
  --num-demos 1 \
  --max-steps 1 \
  --device cuda:0 \
  --attn-implementation eager

python benchmarks/go_vla_benchmark/scripts/export_openvla_causal_localization_report.py \
  --trace-input benchmarks/go_vla_benchmark/data/causal/source_go_patch_causal_manual_index.npz \
  --output-dir benchmarks/go_vla_benchmark/data/causal/report_manual_index
```

Also patch cross-attention-style blocks when the checkpoint exposes them:

```bash
conda run --no-capture-output -n main \
  python benchmarks/go_vla_benchmark/scripts/collect_openvla_causal_localization.py \
  --dataset /root/dsait4125/benchmarks/go_vla_benchmark/data/source_go.hdf5 \
  --checkpoint /root/16-18-14/checkpoints/best-merged \
  --trace-output /root/dsait4125/benchmarks/go_vla_benchmark/data/causal/source_go_patch_causal_cross_blocks.npz \
  --summary-output /root/dsait4125/benchmarks/go_vla_benchmark/data/causal/source_go_patch_causal_cross_blocks.json \
  --corruption-type patch-occlusion \
  --per-cross-attention \
  --num-demos 1 \
  --max-steps 1 \
  --device cuda:0 \
  --attn-implementation eager

python benchmarks/go_vla_benchmark/scripts/export_openvla_causal_localization_report.py \
  --trace-input benchmarks/go_vla_benchmark/data/causal/source_go_patch_causal_cross_blocks.npz \
  --output-dir benchmarks/go_vla_benchmark/data/causal/report_cross_blocks
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

- The collection script now prints a startup JSON block to `stderr` with the
  requested device, resolved device, and first model-parameter device. On a GPU
  run these should all report `cuda:0`.
- OpenVLA is typically a decoder-only VLM, so the most important localization
  signal usually comes from decoder layers and self-attention heads.
- `--per-cross-attention` is optional by design because some checkpoints will
  not expose separate cross-attention blocks.
- Simulator-backed traces embed their clips directly, so report export works
  without `--dataset`.
- The stored score is a normalized restoration ratio:
  `(patched - corrupted) / (clean - corrupted)` on target-token log-probability.
- The PNG uses that same scalar score directly for coloring:
  `u = clip((restoration + 1) / 2, 0, 1)`, then `u = 0` is blue, `u = 0.5` is
  white, and `u = 1` is red.
- If a heatmap looks like `32 x 32`, that means `32` decoder layers and `32`
  attention heads per layer, not a separate `32 x 32` head-to-head matrix.
