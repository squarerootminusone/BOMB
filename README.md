sup boys lets get this going

Go benchmark scaffolding for DeepMind Go + MimicGen is at:
`benchmarks/go_vla_benchmark/README.md`

## Fine-tuning (OpenVLA-OFT)

Requires NVIDIA GPU with Ada Lovelace+ (RTX 6000 Pro, RTX 4090, etc.) and conda.

```bash
# Set up conda env (Python 3.10, PyTorch 2.4, CUDA 12.4)
cd openvla-oft
bash scripts/setup_env.sh
conda activate openvla-oft

# Authenticate wandb
wandb login

# Fine-tune on Go VLA dataset (DATA_DIR must contain go_vla_dataset/)
bash scripts/finetune_go_vla.sh <DATA_DIR>
```

See `openvla-oft/README.md` for full details.
