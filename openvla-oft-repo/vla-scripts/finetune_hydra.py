"""Hydra wrapper for OpenVLA-OFT fine-tuning.

Reads the same YAML config format as vanilla OpenVLA, then launches OFT's native
finetune.py via subprocess with the correct CLI args.

Usage:
    PYTHONPATH=openvla-oft-repo conda run --no-capture-output -n mujogo-oft \
        python openvla-oft-repo/vla-scripts/finetune_hydra.py
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
    exp_id = f"oft+{model_tag}+{cfg.data.dataset_name}+b{cfg.training.batch_size}+lr-{cfg.training.learning_rate}"
    if cfg.lora.enabled:
        exp_id += f"+lora-r{cfg.lora.rank}"
    run_name = cfg.wandb.get("run_name") or f"ft+{exp_id}"

    cmd = [
        sys.executable, str(REPO_ROOT / "vla-scripts" / "finetune.py"),
        f"--vla_path={cfg.vla_path}",
        f"--data_root_dir={Path(cfg.data.root_dir).expanduser()}",
        f"--dataset_name={cfg.data.dataset_name}",
        f"--run_root_dir={output_dir}",
        f"--shuffle_buffer_size={cfg.data.shuffle_buffer_size}",
        f"--use_l1_regression={cfg.action.use_l1_regression}",
        f"--use_diffusion={cfg.action.use_diffusion}",
        f"--batch_size={cfg.training.batch_size}",
        f"--learning_rate={cfg.training.learning_rate}",
        f"--max_steps={cfg.training.max_steps}",
        f"--grad_accumulation_steps={cfg.training.grad_accumulation_steps}",
        f"--use_val_set=True",
        f"--val_freq={cfg.training.val_steps}",
        f"--save_freq={cfg.training.save_steps}",
        f"--num_steps_before_decay={cfg.training.num_steps_before_decay}",
        f"--image_aug={cfg.data.image_aug}",
        f"--use_lora={cfg.lora.enabled}",
        f"--lora_rank={cfg.lora.rank}",
        f"--wandb_project={cfg.wandb.project}",
        f"--run_id_override={run_name}",
    ]
    if cfg.wandb.entity:
        cmd.append(f"--wandb_entity={cfg.wandb.entity}")

    print(f"OpenVLA-OFT Fine-tuning: {run_name}")
    print(f"  Output: {output_dir}")
    print(f"  Command: {' '.join(cmd)}")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + ":" + env.get("PYTHONPATH", "")
    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
