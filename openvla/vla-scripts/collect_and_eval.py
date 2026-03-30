#!/usr/bin/env python3
"""Collect one scripted demo using the REAL collection pipeline, then evaluate VLA on BOTH:
1. Isolated timesteps (GT images -> predict actions, no compounding error)
2. Engine rollout (restore exact sim state -> VLA controls arm, compounding error)

Guarantees both evals use the EXACT same board state, lighting, camera, etc.

Usage:
    MUJOCO_GL=egl uv run python openvla/vla-scripts/collect_and_eval.py \
        --adapter-path /path/to/adapter --load-4bit
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
from PIL import Image, ImageDraw, ImageFont
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
from go_vla_benchmark.collect import _collect_single_episode
from go_vla_benchmark.mimicgen_interface import MG_GoJacoSingleMove
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
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--opening-moves", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--unnorm-key", type=str, default="go_vla_dataset")
    p.add_argument("--load-4bit", action="store_true")
    return p.parse_args()


def _save_video(frames, path, fps=30):
    import imageio
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=1)
    for f in frames:
        writer.append_data(f)
    writer.close()


def _overlay_text(image, lines):
    pil = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 9)
    except (OSError, IOError):
        font = ImageFont.load_default()
    for i, line in enumerate(lines):
        draw.text((2, 2 + i * 12), line, fill=(255, 255, 255), font=font)
    return np.array(pil)


def main():
    args = parse_args()
    adapter_path = Path(args.adapter_path).resolve()
    output_dir = Path(args.output_dir) if args.output_dir else adapter_path / "collect_and_eval"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- Step 1: Create env and collect scripted demo using the REAL pipeline ---
    print("Creating environment...")
    env = create_benchmark_env(
        environment_name="robosuite_go_5x5_rigid_bodies",
        seed=args.seed, include_image_obs=True,
        camera_height=256, camera_width=256,
        action_scale=0.03, drive_physical_arm=True,
        render_carried_stone=True, render_eef_overlay=False,
    )
    env_interface = MG_GoJacoSingleMove(env=env)

    rng = np.random.RandomState(args.seed)
    color = rng.choice(["black", "white"])
    env.queue_reset_options(GoResetOptions(opening_moves=args.opening_moves, stone_color=color))
    env.reset()
    r, c = env._target_rc
    instruction = _INSTRUCTION_TEMPLATES[rng.randint(len(_INSTRUCTION_TEMPLATES))].format(color=color, r=r, c=c)
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"

    print(f"Task: {instruction}")

    # Save sim state BEFORE collecting the demo
    from robosuite.utils.binding_utils import MjSimState
    sim_state_initial = env._rs_env.sim.get_state().flatten().copy()

    print("Collecting scripted demo (real pipeline)...")
    episode, demo_success = _collect_single_episode(
        env=env,
        env_interface=env_interface,
        controller_divisor=2.0,
        detour_steps=0,
        detour_radius=0.0,
        approach_steps=10,
        press_steps=6,
        retreat_steps=6,
        side_transfer_steps=0,
        side_margin=0.16,
        recovery_steps=3,
    )
    gt_images = episode.observations["agentview_image"]
    gt_actions = episode.actions[:, :4]
    print(f"  Demo: {len(gt_images)} steps, success={demo_success}")

    if not demo_success:
        print("  WARNING: Scripted demo failed! Retrying with different seed...")
        # Try a few more seeds
        for retry_seed in range(args.seed + 1, args.seed + 10):
            env2 = create_benchmark_env(
                environment_name="robosuite_go_5x5_rigid_bodies",
                seed=retry_seed, include_image_obs=True,
                camera_height=256, camera_width=256,
                action_scale=0.03, drive_physical_arm=True,
                render_carried_stone=True, render_eef_overlay=False,
            )
            env_interface2 = MG_GoJacoSingleMove(env=env2)
            rng2 = np.random.RandomState(retry_seed)
            color = rng2.choice(["black", "white"])
            env2.queue_reset_options(GoResetOptions(opening_moves=args.opening_moves, stone_color=color))
            env2.reset()
            r, c = env2._target_rc
            instruction = _INSTRUCTION_TEMPLATES[rng2.randint(len(_INSTRUCTION_TEMPLATES))].format(color=color, r=r, c=c)
            prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
            sim_state_initial = env2._rs_env.sim.get_state().flatten().copy()
            episode, demo_success = _collect_single_episode(
                env=env2, env_interface=env_interface2,
                controller_divisor=2.0, detour_steps=0, detour_radius=0.0,
                approach_steps=10, press_steps=6, retreat_steps=6,
                side_transfer_steps=0, side_margin=0.16, recovery_steps=3,
            )
            if demo_success:
                env = env2
                env_interface = env_interface2
                gt_images = episode.observations["agentview_image"]
                gt_actions = episode.actions[:, :4]
                print(f"  Retry seed={retry_seed}: success! {len(gt_images)} steps, task: {instruction}")
                break
        else:
            print("  All retries failed. Exiting.")
            return

    # Save demo video
    _save_video(gt_images, output_dir / "demo_scripted.mp4")
    print(f"  Saved demo video ({len(gt_images)} frames)")

    # --- Step 2: Load model ---
    print("Loading model...")
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

    # --- Step 3: Isolated timestep eval (no compounding error) ---
    print(f"\n=== Isolated timestep eval ({len(gt_images)} frames) ===")
    isolated_preds = []
    isolated_frames = []
    for t in tqdm.tqdm(range(len(gt_images)), desc="Isolated eval"):
        img = gt_images[t]
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        pil_image = Image.fromarray(img).convert("RGB")
        w, h = pil_image.size
        cs = int(min(w, h) * 0.9)
        l, tp = (w - cs) // 2, (h - cs) // 2
        pil_image = pil_image.crop((l, tp, l + cs, tp + cs))
        inputs = processor(prompt, pil_image)
        inputs["pixel_values"] = inputs["pixel_values"].to(device, dtype=torch.bfloat16)
        inputs["input_ids"] = inputs["input_ids"].to(device)
        inputs["attention_mask"] = inputs["attention_mask"].to(device)
        with torch.no_grad():
            pred = vla.predict_action(**inputs, unnorm_key=args.unnorm_key, do_sample=False)
        isolated_preds.append(pred[:4])
        gt_str = " ".join(f"{v:+.3f}" for v in gt_actions[t])
        pr_str = " ".join(f"{v:+.3f}" for v in pred[:4])
        l1 = np.abs(gt_actions[t][:3] - pred[:3]).mean()
        isolated_frames.append(_overlay_text(img, [
            f"ISOLATED t={t}", f"GT:   {gt_str}", f"Pred: {pr_str}", f"xyz_L1={l1:.4f}"
        ]))

    isolated_preds = np.array(isolated_preds)

    # RLDS and model outputs both use standardized absolute gripper values:
    # `1=open`, `0=close`.
    gt_compare = gt_actions.copy()
    pred_compare = isolated_preds.copy()
    gt_compare[:, 3] = (gt_compare[:, 3] > 0.5).astype(np.float32)
    pred_compare[:, 3] = (pred_compare[:, 3] > 0.5).astype(np.float32)
    isolated_l1 = np.abs(pred_compare - gt_compare).mean(axis=1)
    isolated_l1_xyz = np.abs(isolated_preds[:, :3] - gt_actions[:, :3]).mean()
    print(f"  Mean L1: {isolated_l1.mean():.4f} (xyz only: {isolated_l1_xyz:.4f})")
    print(f"  Per-dim: x={np.abs(pred_compare - gt_compare)[:, 0].mean():.4f} "
          f"y={np.abs(pred_compare - gt_compare)[:, 1].mean():.4f} "
          f"z={np.abs(pred_compare - gt_compare)[:, 2].mean():.4f} "
          f"g={np.abs(pred_compare - gt_compare)[:, 3].mean():.4f}")
    _save_video(isolated_frames, output_dir / "eval_isolated.mp4", fps=5)

    # --- Step 4: Engine rollout (restore EXACT same sim state, compounding error) ---
    print(f"\n=== Engine rollout eval (max {args.max_steps} steps) ===")

    sim = env._rs_env.sim
    restored_state = MjSimState.from_flattened(sim_state_initial, sim)
    sim.set_state(restored_state)
    sim.forward()

    # Reset internal env flags so is_success() doesn't carry over from the demo
    env._move_committed = False
    env._step_count = 0
    env._success_step = None

    obs = env.get_observation()

    rollout_frames = []
    rollout_reward = 0.0

    for step_idx in tqdm.tqdm(range(args.max_steps), desc="Engine rollout"):
        img = obs.get("agentview_image", obs.get("image"))
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)

        pil_image = Image.fromarray(img).convert("RGB")
        w, h = pil_image.size
        cs = int(min(w, h) * 0.9)
        l, tp = (w - cs) // 2, (h - cs) // 2
        pil_image = pil_image.crop((l, tp, l + cs, tp + cs))
        inputs = processor(prompt, pil_image)
        inputs["pixel_values"] = inputs["pixel_values"].to(device, dtype=torch.bfloat16)
        inputs["input_ids"] = inputs["input_ids"].to(device)
        inputs["attention_mask"] = inputs["attention_mask"].to(device)
        with torch.no_grad():
            action = vla.predict_action(**inputs, unnorm_key=args.unnorm_key, do_sample=False)

        rollout_frames.append(_overlay_text(img, [
            f"ENGINE t={step_idx}",
            f"Action: {' '.join(f'{v:+.3f}' for v in action[:4])}",
        ]))

        env_action = standardized_action_to_benchmark(action[:4], binarize=True)
        obs, reward, done, info = env.step(env_action)
        rollout_reward += reward
        if env.is_success().get("task", False):
            print(f"  SUCCESS at step {step_idx + 1}!")
            break

    rollout_success = env.is_success().get("task", False)
    print(f"  Result: success={rollout_success}, steps={step_idx + 1}, reward={rollout_reward:.3f}")
    print(f"  (Scripted demo completed in {len(gt_images)} steps, success={demo_success})")
    _save_video(rollout_frames, output_dir / "eval_engine.mp4")

    # --- Step 5: Save results ---
    results = {
        "instruction": instruction,
        "color": color, "row": r, "col": c,
        "seed": args.seed,
        "opening_moves": args.opening_moves,
        "demo_steps": len(gt_images),
        "demo_success": demo_success,
        "isolated": {
            "mean_l1": float(isolated_l1.mean()),
            "mean_l1_xyz": float(isolated_l1_xyz),
            "l1_x": float(np.abs(pred_compare - gt_compare)[:, 0].mean()),
            "l1_y": float(np.abs(pred_compare - gt_compare)[:, 1].mean()),
            "l1_z": float(np.abs(pred_compare - gt_compare)[:, 2].mean()),
            "l1_grip": float(np.abs(pred_compare - gt_compare)[:, 3].mean()),
        },
        "engine": {
            "success": rollout_success,
            "steps": step_idx + 1,
            "reward": float(rollout_reward),
        },
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Isolated L1: {isolated_l1.mean():.4f} (xyz: {isolated_l1_xyz:.4f})")
    print(f"Engine:       success={rollout_success}, {step_idx+1} steps (demo={len(gt_images)})")
    print(f"{'='*60}")
    print(f"Videos: {output_dir}/")
    print(f"  demo_scripted.mp4  — GT scripted controller")
    print(f"  eval_isolated.mp4  — VLA on GT images (no compounding)")
    print(f"  eval_engine.mp4    — VLA in engine (compounding error)")


if __name__ == "__main__":
    main()
