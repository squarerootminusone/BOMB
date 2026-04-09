#!/usr/bin/env python3
"""
Evaluate a fine-tuned OpenVLA-OFT checkpoint on the Go VLA benchmark.

Loads the checkpoint, runs episodes in the robosuite Go environment,
and reports stone placement success rate.

Usage:
    MUJOCO_GL=glfw conda run -n mujogo python scripts/evaluation/eval_openvla.py \
        --checkpoint <path_to_checkpoint_dir> \
        --num-episodes 50

Requires both the mujogo env (for robosuite/mujoco) and the openvla-oft
package to be installed (for model loading).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import imageio
import numpy as np

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "go_vla_benchmark"))
sys.path.insert(0, str(REPO_ROOT / "openvla-oft"))

from go_vla_benchmark.paths import bootstrap_pythonpath

bootstrap_pythonpath(REPO_ROOT)

import torch
from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_vla,
    get_vla_action,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    get_image_resize_size,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK

from go_vla_benchmark.openvla_action_utils import openvla_action_to_benchmark
from go_vla_benchmark.env_factory import create_benchmark_env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Instruction templates matching the RLDS builder
_INSTRUCTION_TEMPLATES = [
    "Place a black stone on the Go board at row {r}, column {c}.",
    "Put a black stone at position ({r}, {c}) on the Go board.",
    "Move the black stone to row {r}, column {c} on the board.",
    "Set a black stone at ({r}, {c}).",
]


class EvalConfig:
    """Minimal config object matching what openvla_utils expects."""

    def __init__(
        self,
        pretrained_checkpoint: str,
        lora_rank: int = 32,
        center_crop: bool = True,
        num_open_loop_steps: int = 8,
        num_images_in_input: int = 1,
        use_proprio: bool = False,
        use_l1_regression: bool = True,
        use_diffusion: bool = False,
        use_film: bool = False,
        num_diffusion_steps_train: int = 50,
        num_diffusion_steps_inference: int = 50,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        model_family: str = "openvla",
        unnorm_key: str = "go_vla_dataset",
    ):
        self.pretrained_checkpoint = pretrained_checkpoint
        self.lora_rank = lora_rank
        self.center_crop = center_crop
        self.num_open_loop_steps = num_open_loop_steps
        self.num_images_in_input = num_images_in_input
        self.use_proprio = use_proprio
        self.use_l1_regression = use_l1_regression
        self.use_diffusion = use_diffusion
        self.use_film = use_film
        self.num_diffusion_steps_train = num_diffusion_steps_train
        self.num_diffusion_steps_inference = num_diffusion_steps_inference
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.model_family = model_family
        self.unnorm_key = unnorm_key


def make_instruction(row: int, col: int, seed: int = 0) -> str:
    """Generate a language instruction for the target position."""
    rng = np.random.RandomState(seed=seed)
    template = _INSTRUCTION_TEMPLATES[rng.randint(len(_INSTRUCTION_TEMPLATES))]
    return template.format(r=row, c=col)


def process_action_for_env(action: np.ndarray) -> np.ndarray:
    """Convert an OpenVLA action into benchmark env semantics."""
    return openvla_action_to_benchmark(action, binarize=True)


def run_episode(
    cfg: EvalConfig,
    env,
    vla,
    processor,
    action_head,
    target_row: int,
    target_col: int,
    max_steps: int = 400,
    num_wait_steps: int = 10,
    episode_idx: int = 0,
    save_video: bool = False,
    video_dir: Optional[str] = None,
) -> dict:
    """Run a single evaluation episode."""
    from go_vla_benchmark.robosuite_go_env import GoResetOptions

    # Reset env with specific target
    options = GoResetOptions(
        opening_moves=np.random.randint(0, 5),
        target_row=target_row,
        target_col=target_col,
    )
    env.reset(options=options)

    instruction = make_instruction(target_row, target_col, seed=episode_idx)
    logger.info(f"  Instruction: {instruction}")

    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    replay_images = []
    resize_size = get_image_resize_size(cfg)

    for t in range(max_steps + num_wait_steps):
        obs = env.get_observation()

        # Wait steps for physics to settle
        if t < num_wait_steps:
            dummy_action = np.zeros(4, dtype=np.float32)
            obs, reward, done, info = env.step(dummy_action)
            continue

        # Get image for model
        image = obs.get("agentview_image")
        if image is None:
            raise RuntimeError("Environment did not return agentview_image")

        if save_video:
            replay_images.append(image.copy())

        # Requery model when action queue is empty
        if len(action_queue) == 0:
            image_resized = resize_image_for_policy(image, resize_size)
            observation = {"full_image": image_resized}
            actions = get_vla_action(
                cfg=cfg,
                vla=vla,
                processor=processor,
                obs=observation,
                task_label=instruction,
                action_head=action_head,
            )
            action_queue.extend(actions)

        action = action_queue.popleft()
        action = process_action_for_env(action)

        obs, reward, done, info = env.step(action)

        if done:
            break

    success = info.get("move_committed", False)

    # Save video
    if save_video and replay_images and video_dir:
        os.makedirs(video_dir, exist_ok=True)
        tag = "success" if success else "fail"
        video_path = os.path.join(video_dir, f"ep{episode_idx:03d}_{tag}.mp4")
        imageio.mimsave(video_path, replay_images, fps=20)
        logger.info(f"  Saved video: {video_path}")

    return {
        "success": success,
        "steps": t + 1,
        "target_row": target_row,
        "target_col": target_col,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OpenVLA-OFT on Go VLA benchmark")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to fine-tuned checkpoint directory",
    )
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--board-size", type=int, default=5)
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--save-video", action="store_true", help="Save rollout videos")
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Directory for rollout videos (default: <checkpoint>/eval_videos)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to save eval results JSON (default: <checkpoint>/eval_results.json)",
    )
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--num-open-loop-steps", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed_everywhere(args.seed)

    checkpoint_dir = str(Path(args.checkpoint).resolve())
    video_dir = args.video_dir or os.path.join(checkpoint_dir, "eval_videos")
    log_file = args.log_file or os.path.join(checkpoint_dir, "eval_results.json")

    # ---- Load model ----
    logger.info(f"Loading checkpoint from: {checkpoint_dir}")
    cfg = EvalConfig(
        pretrained_checkpoint=checkpoint_dir,
        lora_rank=args.lora_rank,
        num_open_loop_steps=args.num_open_loop_steps,
    )

    vla = get_vla(cfg)
    processor = get_processor(cfg)
    action_head = get_action_head(cfg, llm_dim=vla.llm_dim)

    # ---- Create environment ----
    logger.info("Creating Go environment...")
    env = create_benchmark_env(
        environment_name="robosuite_go_5x5_rigid_bodies",
        seed=args.seed,
        include_image_obs=True,
        camera_height=args.camera_size,
        camera_width=args.camera_size,
        max_steps=args.max_steps,
    )

    # ---- Run evaluation ----
    logger.info(f"Running {args.num_episodes} evaluation episodes...")
    results = []
    total_success = 0

    for ep_idx in range(args.num_episodes):
        # Random target position on the board
        target_row = np.random.randint(0, args.board_size)
        target_col = np.random.randint(0, args.board_size)

        logger.info(f"Episode {ep_idx + 1}/{args.num_episodes} | target=({target_row}, {target_col})")

        ep_result = run_episode(
            cfg=cfg,
            env=env,
            vla=vla,
            processor=processor,
            action_head=action_head,
            target_row=target_row,
            target_col=target_col,
            max_steps=args.max_steps,
            episode_idx=ep_idx,
            save_video=args.save_video,
            video_dir=video_dir,
        )

        results.append(ep_result)
        if ep_result["success"]:
            total_success += 1

        success_rate = total_success / (ep_idx + 1)
        logger.info(
            f"  {'SUCCESS' if ep_result['success'] else 'FAIL'} | "
            f"steps={ep_result['steps']} | "
            f"running success rate: {success_rate:.1%} ({total_success}/{ep_idx + 1})"
        )

    # ---- Report ----
    final_success_rate = total_success / args.num_episodes if args.num_episodes > 0 else 0.0
    logger.info("=" * 60)
    logger.info(f"FINAL SUCCESS RATE: {final_success_rate:.1%} ({total_success}/{args.num_episodes})")
    logger.info("=" * 60)

    # Save results
    summary = {
        "checkpoint": checkpoint_dir,
        "num_episodes": args.num_episodes,
        "total_successes": total_success,
        "success_rate": final_success_rate,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "episodes": results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Results saved to: {log_file}")


if __name__ == "__main__":
    main()
