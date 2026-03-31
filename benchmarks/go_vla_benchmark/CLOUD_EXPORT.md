# Cloud Export Walkthrough

Use this when you want to run the OpenVLA local-explanation / attention-video
pipeline on a remote GPU box and pull the artifacts back afterwards.

This version assumes the cloud VM layout you described:

- `/root/dsait4125` is a full clone of the repo
- `/root/16-18-14` is the full fine-tuning run folder copied from
  `data-analysis/16-18-14`

The important consequence is that your LoRA weights are not a standalone
checkpoint yet. The tree you shared shows an adapter-only PEFT export under:

- `/root/16-18-14/checkpoints/best`
- `/root/16-18-14/checkpoints/latest`

Those directories contain:

- `adapter_model.safetensors`: the learned LoRA delta weights
- `adapter_config.json`: points back to the base model, here `openvla/openvla-7b`
- tokenizer / preprocessor files
- `dataset_statistics.json`: the action un-normalization stats for your Go run

The benchmark explainability scripts still expect a full Hugging Face checkpoint
for `--checkpoint`, so the cloud flow is:

1. clone `dsait4125` on the VM
2. rsync `16-18-14` to the VM
3. download the base Hugging Face checkpoint `openvla/openvla-7b`
4. use `viewer.py`'s `ModelRunner` to export `/root/16-18-14/checkpoints/best`
   or `latest` as a merged checkpoint
5. point `--checkpoint` at the merged directory, not at the adapter directory

## Local Fish Setup

Run these from fish on your laptop:

```fish
set -x HOST 192.165.134.28
set -x CLOUD root@$HOST
set -x PORT 16825
set -x KEY ~/.ssh/id_ed25519_cloud_instances
set -x ADDRESS 8080:localhost:8080

set -x REMOTE_ROOT /root
set -x REMOTE_REPO $REMOTE_ROOT/dsait4125
set -x REMOTE_RUN_DIR $REMOTE_ROOT/16-18-14
set -x REMOTE_ADAPTER_DIR $REMOTE_RUN_DIR/checkpoints/best
set -x REMOTE_MERGED_DIR $REMOTE_RUN_DIR/checkpoints/best-merged
set -x REMOTE_EXPORT_DIR $REMOTE_RUN_DIR/cloud_export
set -x REMOTE_BASE_DIR $REMOTE_ROOT/models/openvla-7b

set -x REPO_GIT git@github.com:squarerootminusone/dsait4125.git

set -x LOCAL_REPO "$HOME/repos/DSAIT/4125. Computer Vision/dsait4125"
set -x LOCAL_RUN_DIR "$HOME/repos/DSAIT/4125. Computer Vision/data-analysis/16-18-14"
set -x LOCAL_DATASET "$LOCAL_REPO/benchmarks/go_vla_benchmark/data/source_go.hdf5"
set -x CLOUD_DATASET "$REMOTE_REPO/benchmarks/go_vla_benchmark/data/source_go.hdf5"
set -x CLOUD_VIDEO "$REMOTE_REPO/benchmarks/go_vla_benchmark/data/source_go_preview.mp4"
set -x CLOUD_VIDEO_RLDS "$REMOTE_REPO/all_episodes.mp4"
set -x CLOUD_ALL_DATA "$REMOTE_REPO/benchmarks/go_vla_benchmark/data"

set -x LOCAL_EXPORT_DIR "$LOCAL_RUN_DIR/cloud_export"
```

```bash
rsync -avP -e "ssh -i $KEY -p $PORT" \
  "$CLOUD:$CLOUD_VIDEO" \
  "$LOCAL_REPO/"
```


```bash
rsync -avP -e "ssh -i $KEY -p $PORT" \
  "$CLOUD:$CLOUD_VIDEO_RLDS" \
  "$LOCAL_REPO/"
```

```bash
rsync -avP -e "ssh -i $KEY -p $PORT" \
  --exclude='/**/.*' \
  --exclude='.*' \
  "$CLOUD:$CLOUD_ALL_DATA" \
  "$LOCAL_REPO/data"
```

```bash
rsync -avP -e "ssh -i $KEY -p $PORT" "$LOCAL_REPO/tensorflow_datasets" "$CLOUD:tensorflow_datasets" 
```

```
conda run --no-capture-output -n main \
  python benchmarks/go_vla_benchmark/scripts/convert_to_rlds.py \
  --input /root/dsait4125/benchmarks/go_vla_benchmark/data/source_go.hdf5

```


If you want the most recent adapter instead of the best one, swap both of these:

- `checkpoints/best` -> `checkpoints/latest`
- `checkpoints/best-merged` -> `checkpoints/latest-merged`

Verify SSH access:

```fish
ssh -i $KEY -p $PORT $CLOUD "echo connected"
```

Optional port forward:

```fish
ssh -i $KEY -p $PORT -L $ADDRESS $CLOUD
```

## Clone The Repo On The Cloud

Still on your laptop:

```fish
ssh -i $KEY -p $PORT $CLOUD "apt-get update && apt-get install -y git rsync python3-venv tmux"
```

```fish
ssh -i $KEY -p $PORT $CLOUD "cd $REMOTE_ROOT && git clone $REPO_GIT dsait4125"
```

If the repo is already there, update it instead:

```fish
ssh -i $KEY -p $PORT $CLOUD "git -C $REMOTE_REPO pull --ff-only"
```

## Upload The Full Fine-Tune Run Folder

Copy the entire `16-18-14` directory so it sits next to the repo at `/root`:

```fish
rsync -avP -e "ssh -i $KEY -p $PORT" \
  "$LOCAL_RUN_DIR" \
  "$CLOUD:$REMOTE_ROOT/"
```

That gives you this layout on the VM:

```text
/root/
├── dsait4125/
└── 16-18-14/
    ├── checkpoints/
    │   ├── best/
    │   └── latest/
    ├── finetune.log
    └── wandb/
```

## Ensure The HDF5 Dataset Exists On The VM

The exporter reads the HDF5 file `source_go.hdf5`. That file is usually not part
of a normal git clone, so copy it if it is missing.

First create the target directory:

```fishv
ssh -i $KEY -p $PORT $CLOUD "mkdir -p $REMOTE_REPO/benchmarks/go_vla_benchmark/data"
```

Then upload the dataset:

```fish
rsync -avP -e "ssh -i $KEY -p $PORT" \
  "$LOCAL_DATASET" \
  "$CLOUD:$REMOTE_REPO/benchmarks/go_vla_benchmark/data/source_go.hdf5"
```

Important: the exporter uses `source_go.hdf5`, not the RLDS `1.0.0/` data.

## Cloud Python Environment

SSH in:

```fish
ssh -i $KEY -p $PORT $CLOUD
```

On the cloud instance:

```bash
python3 -m venv /root/venvs/go-vla
source /root/venvs/go-vla/bin/activate
pip install --upgrade pip

pip install -r /root/dsait4125/openvla/requirements-min.txt
pip install peft==0.11.1 accelerate huggingface_hub h5py imageio imageio-ffmpeg pillow sentencepiece protobuf safetensors einops
```

If Hugging Face download fails because of auth or license gating:

```bash
huggingface-cli login
```

## Fetch The Base Hugging Face Checkpoint

Download the base checkpoint directly on the cloud:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="openvla/openvla-7b",
    local_dir="/root/models/openvla-7b",
)

print("/root/models/openvla-7b")
PY
```

After this step you have:

- `/root/models/openvla-7b`: full base OpenVLA checkpoint
- `/root/16-18-14/checkpoints/best`: your LoRA adapter-only checkpoint

## Export A Full Checkpoint Via `viewer.py`

This is still the key connection step between the Hugging Face base model and
your fine-tuned LoRA run, but instead of duplicating all of the OpenVLA / PEFT
wiring inline, reuse the shared `ModelRunner` implementation in the repo root
[`viewer.py`](../../viewer.py).

The export step below:

- points `ModelRunner` at the local base model you already downloaded to
  `/root/models/openvla-7b`
- loads the LoRA adapter from `/root/16-18-14/checkpoints/best`
- uses the shared export helper to merge the LoRA weights into the base weights
- carries over `dataset_statistics.json` so the benchmark uses your fine-tuned
  action stats
- writes a benchmark-ready full checkpoint to
  `/root/16-18-14/checkpoints/best-merged`

Run:

```bash
python - <<'PY'
import os
import sys
from pathlib import Path

ROOT = Path("/root")
REPO = ROOT / "dsait4125"
BASE_DIR = ROOT / "models" / "openvla-7b"
ADAPTER_DIR = ROOT / "16-18-14" / "checkpoints" / "best"
OUT_DIR = ROOT / "16-18-14" / "checkpoints" / "best-merged"

os.environ["OPENVLA_BASE_MODEL"] = str(BASE_DIR)
sys.path.insert(0, str(REPO))

from new_viewer import ModelRunner

runner = ModelRunner()
runner.load_adapter(str(ADAPTER_DIR))
runner.export_merged_checkpoint(
    output_dir=str(OUT_DIR),
    adapter_dir=str(ADAPTER_DIR),
    base_model=str(BASE_DIR),
)

print(f"base      : {BASE_DIR}")
print(f"adapter   : {ADAPTER_DIR}")
print(f"merged out: {OUT_DIR}")
PY
```

If you want the latest adapter instead, change only:

- `ADAPTER_DIR = ROOT / "16-18-14" / "checkpoints" / "latest"`
- `OUT_DIR = ROOT / "16-18-14" / "checkpoints" / "latest-merged"`

## Run The Local Explanation Workflow

Start tmux so the job survives disconnects:

```bash
tmux new -s go_attn
```

Inside tmux:

```bash
cd /root/dsait4125
source /root/venvs/go-vla/bin/activate
mkdir -p /root/16-18-14/cloud_export
```

### Option A: separate collection, then rendering

This is the cleanest workflow if you want reusable traces plus downstream
reports and videos.

Small test first:

```bash
python benchmarks/go_vla_benchmark/scripts/collect_openvla_local_explanations.py \
  --dataset /root/dsait4125/benchmarks/go_vla_benchmark/data/source_go.hdf5 \
  --checkpoint /root/16-18-14/checkpoints/best-merged \
  --trace-output /root/16-18-14/cloud_export/local_explanations_test_trace.npz \
  --summary-output /root/16-18-14/cloud_export/local_explanations_test_summary.json \
  --num-demos 20 \
  --stride 4 \
  --unnorm-key go_vla_dataset \
  --attn-implementation eager \
  --device cuda:0
```

Render failure videos from that trace:

```bash
python benchmarks/go_vla_benchmark/scripts/export_openvla_attention_videos.py \
  --trace-input /root/16-18-14/cloud_export/local_explanations_test_trace.npz \
  --output-dir /root/16-18-14/cloud_export/attention_failures_test \
  --manifest-output /root/16-18-14/cloud_export/attention_failures_test_manifest.json \
  --top-k 3
```

Render the local explanation report from the same trace:

```bash
python benchmarks/go_vla_benchmark/scripts/export_openvla_local_explanation_report.py \
  --trace-input /root/16-18-14/cloud_export/local_explanations_test_trace.npz \
  --output-dir /root/16-18-14/cloud_export/local_explanations_test_report \
  --token-top-k 8
```

Full separate run:

```bash
python benchmarks/go_vla_benchmark/scripts/collect_openvla_local_explanations.py \
  --dataset /root/dsait4125/benchmarks/go_vla_benchmark/data/source_go.hdf5 \
  --checkpoint /root/16-18-14/checkpoints/best-merged \
  --trace-output /root/16-18-14/cloud_export/local_explanations_trace.npz \
  --summary-output /root/16-18-14/cloud_export/local_explanations_summary.json \
  --stride 4 \
  --unnorm-key go_vla_dataset \
  --attn-implementation eager \
  --device cuda:0
```

```bash
python benchmarks/go_vla_benchmark/scripts/export_openvla_attention_videos.py \
  --trace-input /root/16-18-14/cloud_export/local_explanations_trace.npz \
  --output-dir /root/16-18-14/cloud_export/attention_failures \
  --manifest-output /root/16-18-14/cloud_export/attention_failures_manifest.json \
  --top-k 3
```

```bash
python benchmarks/go_vla_benchmark/scripts/export_openvla_local_explanation_report.py \
  --trace-input /root/16-18-14/cloud_export/local_explanations_trace.npz \
  --output-dir /root/16-18-14/cloud_export/local_explanations_report \
  --token-top-k 8
```

### Option B: chained collect + render

Use this when the main goal is still the failure videos, but you also want the
same reusable trace saved on disk.

Small test first:

```bash
python benchmarks/go_vla_benchmark/scripts/export_openvla_attention_videos.py \
  --dataset /root/dsait4125/benchmarks/go_vla_benchmark/data/source_go.hdf5 \
  --checkpoint /root/16-18-14/checkpoints/best-merged \
  --output-dir /root/16-18-14/cloud_export/attention_failures_test \
  --trace-output /root/16-18-14/cloud_export/attention_failures_test_trace.npz \
  --manifest-output /root/16-18-14/cloud_export/attention_failures_test_manifest.json \
  --num-demos 20 \
  --top-k 3 \
  --stride 4 \
  --unnorm-key go_vla_dataset \
  --attn-implementation eager \
  --device cuda:0
```

Full chained run:

```bash
python benchmarks/go_vla_benchmark/scripts/export_openvla_attention_videos.py \
  --dataset /root/dsait4125/benchmarks/go_vla_benchmark/data/source_go.hdf5 \
  --checkpoint /root/16-18-14/checkpoints/best-merged \
  --output-dir /root/16-18-14/cloud_export/attention_failures \
  --trace-output /root/16-18-14/cloud_export/attention_failures_trace.npz \
  --manifest-output /root/16-18-14/cloud_export/attention_failures_manifest.json \
  --top-k 3 \
  --stride 4 \
  --unnorm-key go_vla_dataset \
  --attn-implementation eager \
  --device cuda:0
```

Detach from tmux with `Ctrl-b` then `d`.

## Pull Results Back To Your Laptop

Back on your laptop in fish:

```fish
mkdir -p "$LOCAL_EXPORT_DIR"
rsync -avP -e "ssh -i $KEY -p $PORT" \
  "$CLOUD:$REMOTE_EXPORT_DIR/" \
  "$LOCAL_EXPORT_DIR/"
```

## Quick Mental Model

If you want to remember just one thing, it is this:

- base checkpoint: `/root/models/openvla-7b`
- LoRA adapter: `/root/16-18-14/checkpoints/best`
- benchmark-ready merged checkpoint: `/root/16-18-14/checkpoints/best-merged`

The benchmark scripts should use the merged checkpoint path, not the adapter
path.
