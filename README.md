# BOMB: BOard Manipulation Benchmark

**[Read the full blog post](https://hackmd.io/@soZ5hy2XQ5ia8pihx7AdHQ/r1gbwuUibe)**

A simulation benchmark for evaluating Vision-Language-Action (VLA) models on fine-grained spatial manipulation tasks using Go stone placement on a 5×5 board.

## Overview

BOMB evaluates how well VLA models can ground language-specified spatial destinations in visual space. Models must locate and grasp a stone, then transport it to a specified board position. The benchmark is implemented in MuJoCo and RoboTwin2 simulators.

## Repository Structure

### Models

| Model | Location | Training Script |
|-------|----------|-----------------|
| **OpenVLA** (7B) | `openvla/` | `openvla/vla-scripts/finetune.py` |
| **OpenVLA-OFT** | `openvla-oft-repo/` | `openvla-oft-repo/vla-scripts/finetune.py` |
| **SpatialVLA** (4B) | `spatialvla/` | `spatialvla/scripts/finetune_hydra.py` |
| **π₀** (3.3B) | `openpi/` | `openpi/scripts/train.py` |

### Benchmark & Evaluation

| Component | Location |
|-----------|----------|
| GO Environment | `benchmarks/go_vla_benchmark/go_vla_benchmark/` |
| RoboTwin2 Simulator | `benchmarks/RoboTwin/` |
| Evaluation Script | `scripts/evaluation/eval_openvla.py` |

### Explainability

All explainability methods are in `benchmarks/go_vla_benchmark/go_vla_benchmark/explainability/`:

| Method | Description |
|--------|-------------|
| Patch Masking | Occlude visual regions to measure action prediction changes |
| Text Intervention | Mask instruction tokens via attention masks |
| Activation Patching | Restore clean activations at specific layers/heads |

Related scripts:
- `scripts/collection/collect_openvla_causal_localization.py`
- `scripts/collection/collect_openvla_intervention_tests.py`
- `scripts/export/export_openvla_attention_videos.py`
- `scripts/export/export_openvla_causal_localization_report.py`
- `scripts/export/export_openvla_intervention_report.py`

### Scripts

```
scripts/
├── collection/      # Data collection (demos, explanations)
├── processing/      # Data conversion (RLDS, augmentation)
├── evaluation/      # Model evaluation
├── export/          # Video and report generation
├── analysis/        # Dataset inspection and statistics
├── setup/           # Environment setup and diagnostics
└── utilities/       # Self-play, verification
```

## Quick Start

### Environment Setup

```bash
conda create -n mujogo python=3.11 -y
conda activate mujogo
pip install torch torchvision torchaudio
pip install -e openvla --no-deps
pip install -r openvla/requirements-finetune.txt
```

### Data Collection

```bash
MUJOCO_GL=glfw python scripts/collection/collect_source_demos.py \
    --num-demos 100 --camera-size 256
```

### Training (OpenVLA example)

```bash
MUJOCO_GL=egl python openvla/vla-scripts/finetune.py \
    --dataset_name go_vla_dataset \
    --batch_size 4 --lora_rank 32 --use_quantization True
```

### Evaluation

```bash
MUJOCO_GL=egl python scripts/evaluation/eval_openvla.py \
    --checkpoint <path_to_checkpoint> --num-episodes 50
```

## Perturbations

The benchmark tests robustness across multiple perturbation dimensions:
- Table color/texture and reflectivity
- Stone layouts and initial positions
- Color balance and brightness
- Camera FoV and angle

## Pre-trained Checkpoints

Available on HuggingFace: [MJ22x/go-vla-benchmark](https://huggingface.co/datasets/MJ22x/go-vla-benchmark)

## Contributors

- Paul ([@squarerootminusone](https://github.com/squarerootminusone))
- Marcin ([@marjarai](https://github.com/marjarai))
- Rafael ([@rafael-alani](https://github.com/rafael-alani))

## Acknowledgments

This project builds upon several open-source projects. We thank the original authors for their contributions:

- **[OpenVLA](https://github.com/openvla/openvla)** (MIT License) — Kim, Pertsch, Karamcheti et al.
- **[OpenVLA-OFT](https://github.com/moojink/openvla-oft)** (MIT License) — Kim, Finn, Liang et al.
- **[π₀ / OpenPI](https://github.com/Physical-Intelligence/openpi)** (Apache 2.0) — Physical Intelligence
- **[SpatialVLA](https://huggingface.co/IPEC-COMMUNITY/spatialvla-4b-224-pt)** (MIT License) — IPEC Community
- **[MimicGen](https://github.com/NVlabs/mimicgen)** (NVIDIA License) — NVlabs
- **[robosuite](https://github.com/ARISE-Initiative/robosuite)** — ARISE Initiative
- **[DeepMind Research](https://github.com/deepmind/deepmind-research)** — DeepMind
- **[RoboTwin](https://github.com/TianxingChen/RoboTwin)** — Chen et al.
