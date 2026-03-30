#!/usr/bin/env python3
"""Evaluate a fine-tuned OpenVLA model on the RLDS training/val data.

For each episode, runs inference frame-by-frame (no compounding error),
computes L1 action error and token accuracy, and saves side-by-side videos.

Usage:
    PYTHONPATH=openvla conda run --no-capture-output -n mujogo \
        python openvla/vla-scripts/eval_traindata.py \
        --adapter-path /path/to/adapter_dir \
        --split train --max-episodes 5 --save-videos --load-4bit
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

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-path", type=str, required=True)
    p.add_argument("--base-model", type=str, default="openvla/openvla-7b")
    p.add_argument("--data-dir", type=str, default="~/tensorflow_datasets")
    p.add_argument("--dataset-name", type=str, default="go_vla_dataset")
    p.add_argument("--split", type=str, default="train", choices=["train", "val", "all"])
    p.add_argument("--max-episodes", type=int, default=None, help="limit episodes (None=all)")
    p.add_argument("--save-videos", action="store_true")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--unnorm-key", type=str, default="go_vla_dataset")
    p.add_argument("--load-4bit", action="store_true")
    return p.parse_args()


def _save_video(frames, path, fps=5):
    import imageio
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=1)
    for f in frames:
        writer.append_data(f)
    writer.close()


def _make_comparison_frame(image, gt_action, pred_action):
    """Draw GT vs predicted action text onto a copy of the image."""
    from PIL import ImageDraw, ImageFont
    pil = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 9)
    except (OSError, IOError):
        font = ImageFont.load_default()
    h = image.shape[0]
    gt_str = "GT:  " + " ".join(f"{v:+.3f}" for v in gt_action)
    pr_str = "Pred:" + " ".join(f"{v:+.3f}" for v in pred_action)
    l1 = np.abs(gt_action - pred_action).mean()
    draw.text((2, h - 32), gt_str, fill=(0, 255, 0), font=font)
    draw.text((2, h - 20), pr_str, fill=(255, 100, 100), font=font)
    draw.text((2, 2), f"L1={l1:.4f}", fill=(255, 255, 255), font=font)
    return np.array(pil)


def load_rlds_episodes(data_dir, dataset_name, split):
    """Load episodes from RLDS dataset."""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow_datasets as tfds

    data_dir = os.path.expanduser(data_dir)
    if split == "all":
        splits = ["train", "val"]
    else:
        splits = [split]

    episodes = []
    for s in splits:
        ds = tfds.load(dataset_name, split=s, data_dir=data_dir)
        for ep_idx, ep in enumerate(ds):
            steps = list(ep["steps"])
            images = np.array([step["observation"]["image"].numpy() for step in steps])
            actions = np.array([step["action"].numpy() for step in steps])
            instruction = steps[0]["language_instruction"].numpy().decode()
            episodes.append({
                "images": images,
                "actions": actions[:, :4],  # 4-DoF
                "instruction": instruction,
                "split": s,
                "idx": ep_idx,
            })
    return episodes


def main():
    args = parse_args()
    adapter_path = Path(args.adapter_path).resolve()
    output_dir = Path(args.output_dir) if args.output_dir else adapter_path / f"eval_traindata_{args.split}"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    # Load norm stats — set on the inner base model for PEFT compatibility
    target = vla
    if hasattr(target, "base_model"):
        target = target.base_model
    if hasattr(target, "model"):
        target = target.model
    for search_path in [adapter_path / "dataset_statistics.json"] + [p / "dataset_statistics.json" for p in adapter_path.parents]:
        if search_path.exists():
            with open(search_path) as f:
                target.norm_stats = json.load(f)
            print(f"Loaded norm stats from {search_path}")
            break

    vla.eval()

    # Load RLDS episodes
    print(f"Loading RLDS episodes ({args.split})...")
    episodes = load_rlds_episodes(args.data_dir, args.dataset_name, args.split)
    if args.max_episodes:
        episodes = episodes[:args.max_episodes]
    print(f"Loaded {len(episodes)} episodes")

    all_results = []

    for ep in tqdm.tqdm(episodes, desc=f"Eval {args.split}"):
        images = ep["images"]
        gt_actions = ep["actions"]
        instruction = ep["instruction"]
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"

        pred_actions = []
        frames = []

        for t in range(len(images)):
            image = images[t]
            if image.dtype != np.uint8:
                image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

            pil_image = Image.fromarray(image).convert("RGB")
            w, h = pil_image.size
            crop_size = int(min(w, h) * 0.9)
            left, top = (w - crop_size) // 2, (h - crop_size) // 2
            pil_image = pil_image.crop((left, top, left + crop_size, top + crop_size))
            inputs = processor(prompt, pil_image).to(device, dtype=torch.bfloat16)
            with torch.no_grad():
                action = vla.predict_action(**inputs, unnorm_key=args.unnorm_key, do_sample=False)

            pred_actions.append(action[:4])

            if args.save_videos:
                frames.append(_make_comparison_frame(image, gt_actions[t], action[:4]))

        pred_actions = np.array(pred_actions)

        # RLDS and model outputs both use standardized absolute gripper values:
        # `1=open`, `0=close`.
        gt_compare = gt_actions.copy()
        pred_compare = pred_actions.copy()
        gt_compare[:, 3] = (gt_compare[:, 3] > 0.5).astype(np.float32)
        pred_compare[:, 3] = (pred_compare[:, 3] > 0.5).astype(np.float32)

        # Metrics
        l1_per_step = np.abs(pred_compare - gt_compare).mean(axis=1)
        l1_per_dim = np.abs(pred_compare - gt_compare).mean(axis=0)
        l1_total = l1_per_step.mean()

        ep_result = {
            "split": ep["split"], "idx": ep["idx"],
            "instruction": instruction,
            "num_steps": len(images),
            "l1_total": float(l1_total),
            "l1_x": float(l1_per_dim[0]), "l1_y": float(l1_per_dim[1]),
            "l1_z": float(l1_per_dim[2]), "l1_grip": float(l1_per_dim[3]),
        }
        all_results.append(ep_result)

        tqdm.tqdm.write(
            f"  [{ep['split']}] ep {ep['idx']}: L1={l1_total:.4f} "
            f"(x={l1_per_dim[0]:.4f} y={l1_per_dim[1]:.4f} z={l1_per_dim[2]:.4f} g={l1_per_dim[3]:.4f}) "
            f"| {instruction[:50]}"
        )

        if args.save_videos and frames:
            _save_video(frames, output_dir / f"ep_{ep['split']}_{ep['idx']:03d}.mp4")

    # Summary
    mean_l1 = np.mean([r["l1_total"] for r in all_results])
    print(f"\n{'='*60}")
    print(f"Mean L1: {mean_l1:.4f} over {len(all_results)} episodes")
    for dim_name in ["l1_x", "l1_y", "l1_z", "l1_grip"]:
        print(f"  {dim_name}: {np.mean([r[dim_name] for r in all_results]):.4f}")
    print(f"{'='*60}")

    summary = {
        "adapter_path": str(adapter_path),
        "split": args.split,
        "num_episodes": len(all_results),
        "mean_l1": float(mean_l1),
        "mean_l1_x": float(np.mean([r["l1_x"] for r in all_results])),
        "mean_l1_y": float(np.mean([r["l1_y"] for r in all_results])),
        "mean_l1_z": float(np.mean([r["l1_z"] for r in all_results])),
        "mean_l1_grip": float(np.mean([r["l1_grip"] for r in all_results])),
        "results": all_results,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
