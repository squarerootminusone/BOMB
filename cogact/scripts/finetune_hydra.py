"""Hydra wrapper for CogACT fine-tuning.

Reads the same YAML config format, then launches CogACT's native train.py
via subprocess with the correct CLI args.

Usage:
    PYTHONPATH=openvla:cogact conda run --no-capture-output -n mujogo-cogact \
        python cogact/scripts/finetune_hydra.py
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
        f"cogact+{model_tag}+{cfg.data.dataset_name}"
        f"+b{cfg.training.batch_size}+lr-{cfg.training.learning_rate}"
        f"+{cfg.action.action_model_type}+dim{cfg.action.action_dim}"
    )
    run_name = cfg.wandb.get("run_name") or f"ft+{exp_id}"

    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "train.py"),
        f"--pretrained_checkpoint={cfg.vla_path}",
        "--vla.type=prism-dinosiglip-224px+oxe+diffusion",
        f"--vla.data_mix={cfg.data.dataset_name}",
        "--vla.expected_world_size=1",
        f"--vla.global_batch_size={cfg.training.batch_size}",
        f"--vla.per_device_batch_size={cfg.training.batch_size}",
        f"--vla.learning_rate={cfg.training.learning_rate}",
        f"--data_root_dir={Path(cfg.data.root_dir).expanduser()}",
        f"--run_root_dir={output_dir}",
        f"--action_dim={cfg.action.action_dim}",
        f"--action_model_type={cfg.action.action_model_type}",
        f"--future_action_window_size={cfg.action.future_action_window_size}",
        f"--past_action_window_size={cfg.action.past_action_window_size}",
        f"--repeated_diffusion_steps={cfg.action.repeated_diffusion_steps}",
        f"--save_interval={cfg.training.save_steps}",
        f"--image_aug={cfg.data.image_aug}",
        f"--wandb_project={cfg.wandb.project}",
        f"--run_id={run_name}",
        "--is_resume=False",
    ]
    if cfg.wandb.entity:
        cmd.append(f"--wandb_entity={cfg.wandb.entity}")

    print(f"CogACT Fine-tuning: {run_name}")
    print(f"  Output: {output_dir}")
    print(f"  Command: {' '.join(cmd)}")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + ":" + env.get("PYTHONPATH", "")
    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
