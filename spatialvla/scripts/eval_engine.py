#!/usr/bin/env python3
"""Evaluate a fine-tuned SpatialVLA LoRA adapter in the physics engine.

Usage:
    MUJOCO_GL=egl PYTHONPATH=spatialvla conda run --no-capture-output -n mujogo \
        python spatialvla/scripts/eval_engine.py \
        --adapter-path /path/to/checkpoint-XXXX \
        --num-episodes 5 --save-videos
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

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "spatialvla"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "go_vla_benchmark"))

from go_vla_benchmark.paths import bootstrap_pythonpath
bootstrap_pythonpath(REPO_ROOT)
from go_vla_benchmark.openvla_action_utils import standardized_action_to_benchmark
from go_vla_benchmark.robosuite_compat import ensure_robosuite_compat
ensure_robosuite_compat()
from go_vla_benchmark.common import GoResetOptions
from go_vla_benchmark.env_factory import create_benchmark_env

_INSTRUCTION_TEMPLATES = [
    "Place a {color} stone on the Go board at row {r}, column {c}.",
    "Put a {color} stone at position ({r}, {c}) on the Go board.",
    "Move the {color} stone to row {r}, column {c} on the board.",
    "Set a {color} stone at ({r}, {c}).",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-path", type=str, required=True)
    p.add_argument("--base-model", type=str, default="IPEC-COMMUNITY/spatialvla-4b-224-pt")
    p.add_argument("--num-episodes", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-videos", action="store_true")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--unnorm-key", type=str, default="go_vla_dataset/6.0.0")
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


def _7dof_to_4dof(action_7d):
    """Extract 4-DoF [dx, dy, dz, gripper] from 7-DoF [dx, dy, dz, roll, pitch, yaw, gripper]."""
    return np.concatenate([action_7d[:3], action_7d[6:7]])


def load_spatialvla_with_adapter(base_model_path, adapter_path, device):
    from model import SpatialVLAConfig, SpatialVLAForConditionalGeneration, SpatialVLAProcessor

    processor = SpatialVLAProcessor.from_pretrained(adapter_path, trust_remote_code=True)
    base_model = SpatialVLAForConditionalGeneration.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()
    model = model.to(device).eval()
    return model, processor


def main():
    args = parse_args()
    adapter_path = Path(args.adapter_path).resolve()
    output_dir = Path(args.output_dir) if args.output_dir else adapter_path / "eval_engine"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, processor = load_spatialvla_with_adapter(args.base_model, str(adapter_path), device)

    env = create_benchmark_env(
        environment_name="robosuite_go_5x5_rigid_bodies",
        seed=args.seed, include_image_obs=True,
        camera_height=256, camera_width=256,
        action_scale=0.03, drive_physical_arm=True,
        render_carried_stone=True, render_eef_overlay=False,
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
        prompt = f"What action should the robot take to {instruction.lower()}?"

        frames = []
        episode_reward = 0.0

        for step_idx in range(args.max_steps):
            image = obs.get("agentview_image", obs.get("image"))
            if image.dtype != np.uint8:
                image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
            if args.save_videos:
                frames.append(image.copy())

            pil_image = Image.fromarray(image).convert("RGB")
            inputs = processor(images=[pil_image], text=prompt, return_tensors="pt")
            with torch.no_grad():
                generation_outputs = model.predict_action(inputs)
            result = processor.decode_actions(generation_outputs, unnorm_key=args.unnorm_key)
            action_7d = result["actions"][0]  # first action in chunk
            action = _7dof_to_4dof(action_7d)

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
