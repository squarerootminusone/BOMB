# Go VLA Benchmark

Go benchmark scaffolding for DeepMind Go + MimicGen is at:
`benchmarks/go_vla_benchmark/README.md`

## Environment Setup

### 1. Create conda environment

```bash
conda create -n mujogo python=3.11 -y
conda activate mujogo
```

### 2. Install PyTorch (CUDA 12.x)

```bash
pip install torch torchvision torchaudio
```

### 3. Install all dependencies

```bash
pip install -e openvla --no-deps
pip install -r openvla/requirements-finetune.txt
pip install -e mimicgen --no-deps
```

## Data Generation

### Collect source demonstrations

```bash
MUJOCO_GL=glfw conda run --no-capture-output -n mujogo \
  python benchmarks/go_vla_benchmark/scripts/collect_source_demos.py \
    --num-demos 40 --camera-size 256 \
    --environment-name robosuite_go_5x5_rigid_bodies
```

### Generate augmented demos via MimicGen

```bash
MUJOCO_GL=glfw conda run --no-capture-output -n mujogo \
  python benchmarks/go_vla_benchmark/scripts/generate_augmented_demos.py \
    --source benchmarks/go_vla_benchmark/data/source_go.hdf5 \
    --camera-size 256 --num-workers 8
```

### Convert to RLDS format

```bash
conda run --no-capture-output -n mujogo \
  python benchmarks/go_vla_benchmark/scripts/convert_to_rlds.py
```

The RLDS dataset is saved to `~/tensorflow_datasets/go_vla_dataset/`.

## Finetuning

### LoRA finetuning (default — paper config)

This is the configuration from the [OpenVLA paper](https://arxiv.org/abs/2406.09246): LoRA r=32, batch size 16, lr 5e-4.

```bash
conda run --no-capture-output -n mujogo \
  python openvla/vla-scripts/finetune.py \
    training.batch_size=16
```

### Full bf16 finetuning

```bash
conda run --no-capture-output -n mujogo \
  python openvla/vla-scripts/finetune.py \
    lora.enabled=false \
    training.batch_size=12
```

### QLoRA finetuning

```bash
conda run --no-capture-output -n mujogo \
  python openvla/vla-scripts/finetune.py \
    training.batch_size=1 \
    training.grad_accumulation_steps=4 \
    lora.quantization=true
```

### Config overrides

All config values in `openvla/configs/finetune.yaml` can be overridden on the CLI:

```bash
python openvla/vla-scripts/finetune.py \
  training.epochs=50 \
  training.learning_rate=2e-5 \
  lora.rank=16 \
  wandb.project=my_project
```

### Resume training

```bash
python openvla/vla-scripts/finetune.py \
  resume_from=outputs/<date>/<time>/resume
```

### Outputs

Checkpoints are saved to `outputs/<date>/<time>/checkpoints/`:
- `best/` — best model by validation loss
- `latest/` — most recent model at validation time

## Evaluation

Run the trained model in the Go environment:

```bash
MUJOCO_GL=egl conda run --no-capture-output -n mujogo \
  python openvla/vla-scripts/eval_go.py \
    --model-path outputs/<date>/<time>/checkpoints/best \
    --num-episodes 20 \
    --save-videos \
    --load-4bit
```

Results are saved to `<model-path>/eval/eval_results.json`.
