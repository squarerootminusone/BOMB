"""Hydra wrapper for SpatialVLA fine-tuning.

Reads the same YAML config format, then launches SpatialVLA's native
spatialvla_finetune.py via subprocess with the correct CLI args.

Usage:
    PYTHONPATH=spatialvla conda run --no-capture-output -n mujogo-spatialvla \
        python spatialvla/scripts/finetune_hydra.py
"""

import os
import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]


@hydra.main(version_base=None, config_path="../configs", config_name="finetune")
def main(cfg: DictConfig) -> None:
    output_dir = Path(os.getcwd())  # Hydra sets cwd to output dir

    model_tag = cfg.vla_path.split("/")[-1]
    exp_id = (
        f"spatialvla+{model_tag}+{cfg.data.dataset_name}"
        f"+b{cfg.training.batch_size}+lr-{cfg.training.learning_rate}"
    )
    if cfg.lora.enabled:
        exp_id += f"+lora-r{cfg.lora.rank}"
    run_name = cfg.wandb.get("run_name") or f"ft+{exp_id}"

    cmd = [
        sys.executable, str(REPO_ROOT / "train" / "spatialvla_finetune.py"),
        f"--model_name_or_path={cfg.vla_path}",
        f"--data_root_dir={Path(cfg.data.root_dir).expanduser()}",
        f"--data_mix={cfg.data.dataset_name}",
        f"--lora={cfg.lora.rank if cfg.lora.enabled else 0}",
        f"--lora_alpha={cfg.lora.alpha}",
        f"--lora_target={cfg.lora.target}",
        f"--load_in_4bit={cfg.lora.get('quantization', False)}",
        f"--mask_rotation_loss={cfg.action.mask_rotation_loss}",
        f"--action_forward_steps={cfg.action.action_forward_steps}",
        f"--flash_attn=True",
        f"--grad_checkpoint={cfg.training.gradient_checkpointing}",
        f"--output_dir={output_dir / 'checkpoints'}",
        "--overwrite_output_dir=False",
        "--freeze_vision_tower=False",
        "--bf16=True",
        "--tf32=True",
        f"--num_train_epochs={cfg.training.epochs}",
        f"--per_device_train_batch_size={cfg.training.batch_size}",
        f"--gradient_accumulation_steps={cfg.training.grad_accumulation_steps}",
        "--save_strategy=steps",
        f"--save_steps={cfg.training.save_steps}",
        "--save_total_limit=2",
        "--save_only_model=True",
        f"--learning_rate={cfg.training.learning_rate}",
        "--weight_decay=0.0",
        "--warmup_ratio=0.005",
        "--lr_scheduler_type=linear",
        "--logging_steps=10",
        "--do_train=True",
        f"--deepspeed={REPO_ROOT / 'scripts' / 'zero1.json'}",
        "--report_to=wandb",
        "--log_level=warning",
    ]

    print(f"SpatialVLA Fine-tuning: {run_name}")
    print(f"  Output: {output_dir}")
    print(f"  Command: {' '.join(cmd)}")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + ":" + env.get("PYTHONPATH", "")
    env["WANDB_PROJECT"] = cfg.wandb.project
    env["WANDB_NAME"] = run_name
    if cfg.wandb.entity:
        env["WANDB_ENTITY"] = cfg.wandb.entity
    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
