#!/usr/bin/env python3
"""Evaluate a fine-tuned OpenVLA model on the Go benchmark environment.

Usage:
    conda run -n mujogo python openvla/vla-scripts/eval_go.py \
        --model-path outputs/<date>/<time>/checkpoints/best \
        --num-episodes 20 \
        --save-videos
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import tqdm
from transformers import BitsAndBytesConfig
from PIL import Image

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "openvla"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "go_vla_benchmark"))

from go_vla_benchmark.paths import bootstrap_pythonpath

bootstrap_pythonpath(REPO_ROOT)

from go_vla_benchmark.robosuite_compat import ensure_robosuite_compat

ensure_robosuite_compat()

from go_vla_benchmark.common import GoResetOptions
from go_vla_benchmark.env_factory import create_benchmark_env

from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def get_openvla_prompt(instruction: str, model_path: str) -> str:
    if "v01" in model_path:
        return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OpenVLA on Go benchmark")
    parser.add_argument("--model-path", type=str, required=True, help="path to fine-tuned model checkpoint")
    parser.add_argument("--num-episodes", type=int, default=20, help="number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=200, help="max steps per episode")
    parser.add_argument("--environment-name", type=str, default="robosuite_go_5x5_rigid_bodies")
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument("--opening-min", type=int, default=0)
    parser.add_argument("--opening-max", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-videos", action="store_true", help="save rollout videos as MP4")
    parser.add_argument("--output-dir", type=str, default=None, help="directory for results (default: model-path/eval)")
    parser.add_argument("--unnorm-key", type=str, default="go_vla_dataset")
    parser.add_argument("--load-4bit", action="store_true", help="load model in 4-bit quantization (for 16GB GPUs)")
    parser.add_argument("--wandb", action="store_true", help="log metrics to wandb")
    parser.add_argument("--wandb-project", type=str, default="openvla", help="wandb project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="wandb entity")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="wandb run name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path).resolve()
    output_dir = Path(args.output_dir) if args.output_dir else model_path / "eval"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Optional wandb logging
    use_wandb = False
    if args.wandb:
        try:
            import wandb

            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name or f"eval-{model_path.name}",
                config=vars(args),
            )
            use_wandb = True
        except Exception as e:
            print(f"[W&B] Failed to initialize: {e}. Continuing without W&B.")

    # Register OpenVLA model
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load model and processor
    print(f"Loading model from {model_path}")
    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)

    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if args.load_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    vla = AutoModelForVision2Seq.from_pretrained(str(model_path), **load_kwargs)
    if not args.load_4bit:
        vla = vla.to(device)

    # Load dataset statistics for action denormalization
    stats_path = model_path / "dataset_statistics.json"
    if stats_path.exists():
        with open(stats_path) as f:
            vla.norm_stats = json.load(f)
        print(f"Loaded norm stats from {stats_path}")
    else:
        print(f"WARNING: {stats_path} not found, actions will not be denormalized correctly")

    vla.eval()

    # Create environment
    print(f"Creating environment: {args.environment_name}")
    env = create_benchmark_env(
        environment_name=args.environment_name,
        seed=args.seed,
        include_image_obs=True,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        action_scale=0.03,
        drive_physical_arm=True,
        render_carried_stone=True,
        render_eef_overlay=False,
    )

    # Same instruction templates as training data (with {color} placeholder)
    _INSTRUCTION_TEMPLATES = [
        "Place a {color} stone on the Go board at row {r}, column {c}.",
        "Put a {color} stone at position ({r}, {c}) on the Go board.",
        "Move the {color} stone to row {r}, column {c} on the board.",
        "Set a {color} stone at ({r}, {c}).",
    ]

    rng = np.random.RandomState(args.seed)

    results = []
    successes = 0

    for ep_idx in tqdm.tqdm(range(args.num_episodes), desc="Evaluating", unit="ep"):
        opening_moves = int(rng.randint(args.opening_min, args.opening_max + 1))
        env.queue_reset_options(GoResetOptions(opening_moves=opening_moves))
        obs = env.reset()

        # Build prompt from target position (matching training data format)
        r, c = env._target_rc
        color = env._stone_color
        instruction = _INSTRUCTION_TEMPLATES[rng.randint(len(_INSTRUCTION_TEMPLATES))].format(color=color, r=r, c=c)
        prompt = get_openvla_prompt(instruction, str(model_path))

        frames = []
        episode_reward = 0.0
        done = False

        for step_idx in range(args.max_steps):
            # Get image observation
            image = obs.get("agentview_image", obs.get("image"))
            if image is None:
                raise RuntimeError(f"No image in observation keys: {list(obs.keys())}")

            # Image might be (H, W, C) uint8
            if image.dtype != np.uint8:
                image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

            if args.save_videos:
                frames.append(image.copy())

            # VLA inference — apply center 90% crop (matches training augmentation)
            pil_image = Image.fromarray(image).convert("RGB")
            w, h = pil_image.size
            crop_size = int(min(w, h) * 0.9)
            left = (w - crop_size) // 2
            top = (h - crop_size) // 2
            pil_image = pil_image.crop((left, top, left + crop_size, top + crop_size))
            inputs = processor(prompt, pil_image).to(device, dtype=torch.bfloat16)
            with torch.no_grad():
                action = vla.predict_action(**inputs, unnorm_key=args.unnorm_key, do_sample=False)

            # Step environment
            obs, reward, done, info = env.step(action)
            episode_reward += reward

            # Check success
            success_info = env.is_success()
            if success_info.get("task", False):
                successes += 1
                break

        ep_result = {
            "episode": ep_idx,
            "success": bool(success_info.get("task", False)),
            "steps": step_idx + 1,
            "reward": float(episode_reward),
            "opening_moves": opening_moves,
        }
        results.append(ep_result)
        tqdm.tqdm.write(
            f"  ep {ep_idx}: success={ep_result['success']}, "
            f"steps={ep_result['steps']}, reward={ep_result['reward']:.3f}"
        )

        # Save video
        if args.save_videos and frames:
            _save_video(frames, output_dir / f"episode_{ep_idx:03d}.mp4")

        # Log per-episode metrics to wandb
        if use_wandb:
            log_data = {
                "episode": ep_idx,
                "success": int(ep_result["success"]),
                "steps": ep_result["steps"],
                "reward": ep_result["reward"],
            }
            if args.save_videos and frames:
                try:
                    log_data["video"] = wandb.Video(
                        str(output_dir / f"episode_{ep_idx:03d}.mp4"), format="mp4"
                    )
                except Exception:
                    pass
            wandb.log(log_data, step=ep_idx)

    # Summary
    success_rate = successes / args.num_episodes
    avg_steps = np.mean([r["steps"] for r in results])
    avg_reward = np.mean([r["reward"] for r in results])

    summary = {
        "model_path": str(model_path),
        "num_episodes": args.num_episodes,
        "success_rate": success_rate,
        "successes": successes,
        "avg_steps": float(avg_steps),
        "avg_reward": float(avg_reward),
        "results": results,
    }

    print(f"\n{'='*50}")
    print(f"Success rate: {successes}/{args.num_episodes} ({success_rate:.1%})")
    print(f"Avg steps: {avg_steps:.1f}")
    print(f"Avg reward: {avg_reward:.3f}")
    print(f"{'='*50}")

    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {results_path}")

    # Log aggregate metrics to wandb
    if use_wandb:
        wandb.log({
            "success_rate": success_rate,
            "avg_steps": float(avg_steps),
            "avg_reward": float(avg_reward),
        })
        wandb.finish()


def _save_video(frames: list, path: Path, fps: int = 30) -> None:
    """Save frames as MP4 video using imageio."""
    try:
        import imageio
    except ImportError:
        print(f"  [skip video] pip install imageio[ffmpeg] to save videos")
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264", pixelformat="yuv420p")
    for frame in frames:
        writer.append_data(frame)
    writer.close()


if __name__ == "__main__":
    main()
