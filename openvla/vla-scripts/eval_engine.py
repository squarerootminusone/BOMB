#!/usr/bin/env python3
"""Evaluate a fine-tuned OpenVLA LoRA adapter in the physics engine.

Loads the base model + LoRA adapter, runs rollouts with compounding error.

Usage:
    MUJOCO_GL=egl PYTHONPATH=openvla conda run --no-capture-output -n mujogo \
        python openvla/vla-scripts/eval_engine.py \
        --adapter-path /path/to/adapter_dir \
        --num-episodes 5 --save-videos --load-4bit
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
from PIL import Image
from peft import PeftModel
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

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
from go_vla_benchmark.openvla_action_utils import standardized_action_to_benchmark

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

_INSTRUCTION_TEMPLATES = [
    "Place a {color} stone on the Go board at row {r}, column {c}.",
    "Put a {color} stone at position ({r}, {c}) on the Go board.",
    "Move the {color} stone to row {r}, column {c} on the board.",
    "Set a {color} stone at ({r}, {c}).",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-path", type=str, required=True)
    p.add_argument("--base-model", type=str, default="openvla/openvla-7b")
    p.add_argument("--num-episodes", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-videos", action="store_true")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--unnorm-key", type=str, default="go_vla_dataset")
    p.add_argument("--load-4bit", action="store_true")
    p.add_argument("--opening-min", type=int, default=0)
    p.add_argument("--opening-max", type=int, default=4)
    return p.parse_args()


def _save_video(frames, path, fps=30):
    import imageio
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264", pixelformat="yuv420p")
    for f in frames:
        writer.append_data(f)
    writer.close()


def load_openvla_with_adapter(base_model_path, adapter_path, load_4bit, device):
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    processor = AutoProcessor.from_pretrained(adapter_path, trust_remote_code=True)

    load_kwargs = dict(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)
    if load_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    base_vla = AutoModelForVision2Seq.from_pretrained(base_model_path, **load_kwargs)
    vla = PeftModel.from_pretrained(base_vla, adapter_path)
    if not load_4bit:
        vla = vla.merge_and_unload()
        vla = vla.to(device)

    # Load norm stats — set on the inner base model for PEFT compatibility
    target = vla
    if hasattr(target, "base_model"):
        target = target.base_model
    if hasattr(target, "model"):
        target = target.model
    for search_path in [Path(adapter_path) / "dataset_statistics.json"] + [p / "dataset_statistics.json" for p in Path(adapter_path).parents]:
        if search_path.exists():
            with open(search_path) as f:
                target.norm_stats = json.load(f)
            print(f"Loaded norm stats from {search_path}")
            break

    vla.eval()
    return vla, processor


def main():
    args = parse_args()
    adapter_path = Path(args.adapter_path).resolve()
    output_dir = Path(args.output_dir) if args.output_dir else adapter_path / "eval_engine"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vla, processor = load_openvla_with_adapter(args.base_model, str(adapter_path), args.load_4bit, device)

    env = create_benchmark_env(
        environment_name="robosuite_go_5x5_rigid_bodies",
        seed=args.seed, include_image_obs=True,
        camera_height=256, camera_width=256,
        action_scale=0.03, drive_physical_arm=True,
        render_carried_stone=True, render_eef_overlay=True,
    )

    rng = np.random.RandomState(args.seed)
    results = []
    successes = 0

    for ep_idx in tqdm.tqdm(range(args.num_episodes), desc="Engine eval"):
        opening_moves = int(rng.randint(args.opening_min, args.opening_max + 1))
        env.queue_reset_options(GoResetOptions(opening_moves=opening_moves))
        obs = env.reset()

        r, c = env._target_rc
        color = env._stone_color
        instruction = _INSTRUCTION_TEMPLATES[rng.randint(len(_INSTRUCTION_TEMPLATES))].format(color=color, r=r, c=c)
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"

        frames = []
        episode_reward = 0.0

        for step_idx in range(args.max_steps):
            image = obs.get("agentview_image", obs.get("image"))
            if image.dtype != np.uint8:
                image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
            if args.save_videos:
                frames.append(image.copy())

            pil_image = Image.fromarray(image).convert("RGB")
            w, h = pil_image.size
            crop_size = int(min(w, h) * 0.9)
            left, top = (w - crop_size) // 2, (h - crop_size) // 2
            pil_image = pil_image.crop((left, top, left + crop_size, top + crop_size))
            inputs = processor(prompt, pil_image).to(device, dtype=torch.bfloat16)
            with torch.no_grad():
                action = vla.predict_action(**inputs, unnorm_key=args.unnorm_key, do_sample=False)

            action = standardized_action_to_benchmark(action, binarize=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            if env.is_success().get("task", False):
                successes += 1
                break

        ep_result = {
            "episode": ep_idx, "success": bool(env.is_success().get("task", False)),
            "steps": step_idx + 1, "reward": float(episode_reward), "opening_moves": opening_moves,
        }
        results.append(ep_result)
        tqdm.tqdm.write(f"  ep {ep_idx}: success={ep_result['success']}, steps={ep_result['steps']}")

        if args.save_videos and frames:
            _save_video(frames, output_dir / f"episode_{ep_idx:03d}.mp4")

    success_rate = successes / args.num_episodes
    summary = {
        "adapter_path": str(adapter_path), "num_episodes": args.num_episodes,
        "success_rate": success_rate, "successes": successes,
        "avg_steps": float(np.mean([r["steps"] for r in results])),
        "results": results,
    }
    print(f"\nSuccess rate: {successes}/{args.num_episodes} ({success_rate:.1%})")
    with open(output_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
