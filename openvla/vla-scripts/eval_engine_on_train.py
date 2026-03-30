#!/usr/bin/env python3
"""Evaluate a fine-tuned OpenVLA on a specific RLDS train sample in the actual engine.

Parses the instruction from the RLDS episode to get (color, row, col),
resets the engine with those exact parameters, then runs the VLA with
compounding error.

Usage:
    MUJOCO_GL=egl PYTHONPATH=openvla conda run --no-capture-output -n mujogo \
        python openvla/vla-scripts/eval_engine_on_train.py \
        --adapter-path /path/to/adapter \
        --episode-idx 0 --save-video --load-4bit
"""

from __future__ import annotations

import argparse
import json
import os
import re
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

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


def parse_instruction(instruction: str):
    """Extract color, row, col from instruction string."""
    color_match = re.search(r"(black|white)", instruction, re.IGNORECASE)
    color = color_match.group(1).lower() if color_match else "black"

    # Match patterns like "row 3, column 1" or "(3, 1)" or "at (3, 1)"
    rc_match = re.search(r"(?:row\s+(\d+),?\s*column\s+(\d+)|\((\d+),?\s*(\d+)\))", instruction)
    if rc_match:
        r = int(rc_match.group(1) or rc_match.group(3))
        c = int(rc_match.group(2) or rc_match.group(4))
    else:
        raise ValueError(f"Cannot parse row/col from instruction: {instruction}")
    return color, r, c


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-path", type=str, required=True)
    p.add_argument("--base-model", type=str, default="openvla/openvla-7b")
    p.add_argument("--data-dir", type=str, default="~/tensorflow_datasets")
    p.add_argument("--dataset-name", type=str, default="go_vla_dataset")
    p.add_argument("--episode-idx", type=int, default=0, help="which train episode to replay")
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--unnorm-key", type=str, default="go_vla_dataset")
    p.add_argument("--load-4bit", action="store_true")
    return p.parse_args()


def _save_video(frames, path, fps=30):
    import imageio
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264", pixelformat="yuv420p")
    for f in frames:
        writer.append_data(f)
    writer.close()


def main():
    args = parse_args()
    adapter_path = Path(args.adapter_path).resolve()
    output_dir = Path(args.output_dir) if args.output_dir else adapter_path / "eval_engine_train"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load RLDS episode to get the instruction
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow_datasets as tfds
    data_dir = os.path.expanduser(args.data_dir)
    ds = tfds.load(args.dataset_name, split=f"train[{args.episode_idx}:{args.episode_idx + 1}]", data_dir=data_dir)
    ep = next(iter(ds))
    steps = list(ep["steps"])
    instruction = steps[0]["language_instruction"].numpy().decode()
    gt_actions = np.array([s["action"].numpy()[:4] for s in steps])
    color, row, col = parse_instruction(instruction)
    print(f"Train episode {args.episode_idx}: '{instruction}'")
    print(f"  Parsed: color={color}, row={row}, col={col}, gt_steps={len(steps)}")

    # Load model
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    processor = AutoProcessor.from_pretrained(str(adapter_path), trust_remote_code=True)
    load_kwargs = dict(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)
    if args.load_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    base_vla = AutoModelForVision2Seq.from_pretrained(args.base_model, **load_kwargs)
    vla = PeftModel.from_pretrained(base_vla, str(adapter_path))
    if not args.load_4bit:
        vla = vla.merge_and_unload()
        vla = vla.to(device)

    # Set norm stats on inner model
    target = vla
    if hasattr(target, "base_model"):
        target = target.base_model
    if hasattr(target, "model"):
        target = target.model
    for sp in [adapter_path / "dataset_statistics.json"] + [p / "dataset_statistics.json" for p in adapter_path.parents]:
        if sp.exists():
            with open(sp) as f:
                target.norm_stats = json.load(f)
            print(f"Loaded norm stats from {sp}")
            break

    vla.eval()

    # Create environment and reset with the EXACT task from the training sample
    env = create_benchmark_env(
        environment_name="robosuite_go_5x5_rigid_bodies",
        seed=args.seed, include_image_obs=True,
        camera_height=256, camera_width=256,
        action_scale=0.03, drive_physical_arm=True,
        render_carried_stone=True, render_eef_overlay=False,
    )

    env.queue_reset_options(GoResetOptions(
        opening_moves=0,
        stone_color=color,
        target_row=row,
        target_col=col,
    ))
    obs = env.reset()
    print(f"  Env target: ({env._target_rc[0]}, {env._target_rc[1]}), color: {env._stone_color}")

    # Build prompt with the EXACT same instruction from training
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"

    frames = []
    episode_reward = 0.0

    for step_idx in tqdm.tqdm(range(args.max_steps), desc="Engine rollout"):
        image = obs.get("agentview_image", obs.get("image"))
        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        if args.save_video:
            frames.append(image.copy())

        pil_image = Image.fromarray(image).convert("RGB")
        w, h = pil_image.size
        crop_size = int(min(w, h) * 0.9)
        left, top = (w - crop_size) // 2, (h - crop_size) // 2
        pil_image = pil_image.crop((left, top, left + crop_size, top + crop_size))
        inputs = processor(prompt, pil_image).to(device, dtype=torch.bfloat16)
        with torch.no_grad():
            action = vla.predict_action(**inputs, unnorm_key=args.unnorm_key, do_sample=False)

        obs, reward, done, info = env.step(action)
        episode_reward += reward

        if env.is_success().get("task", False):
            print(f"\n  SUCCESS at step {step_idx + 1}!")
            break

    success = env.is_success().get("task", False)
    print(f"\n  Result: success={success}, steps={step_idx + 1}, reward={episode_reward:.3f}")
    print(f"  (GT demo completed in {len(steps)} steps)")

    if args.save_video and frames:
        video_path = output_dir / f"train_ep{args.episode_idx}_{color}_r{row}c{col}.mp4"
        _save_video(frames, video_path)
        print(f"  Video saved to {video_path}")

    result = {
        "episode_idx": args.episode_idx,
        "instruction": instruction,
        "color": color, "row": row, "col": col,
        "success": success,
        "steps": step_idx + 1,
        "gt_steps": len(steps),
        "reward": float(episode_reward),
    }
    with open(output_dir / f"result_ep{args.episode_idx}.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
