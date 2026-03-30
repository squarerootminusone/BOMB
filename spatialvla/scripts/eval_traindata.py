#!/usr/bin/env python3
"""Evaluate a fine-tuned SpatialVLA model on the RLDS training/val data.

Frame-by-frame inference (no compounding error), computes L1 action error.

Usage:
    PYTHONPATH=spatialvla conda run --no-capture-output -n mujogo \
        python spatialvla/scripts/eval_traindata.py \
        --adapter-path /path/to/checkpoint-XXXX \
        --split train --max-episodes 5 --save-videos
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

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "spatialvla"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-path", type=str, required=True)
    p.add_argument("--base-model", type=str, default="IPEC-COMMUNITY/spatialvla-4b-224-pt")
    p.add_argument("--data-dir", type=str, default="~/tensorflow_datasets")
    p.add_argument("--dataset-name", type=str, default="go_vla_dataset")
    p.add_argument("--split", type=str, default="train", choices=["train", "val", "all"])
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--save-videos", action="store_true")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--unnorm-key", type=str, default="go_vla_dataset/4.0.0")
    return p.parse_args()


def _save_video(frames, path, fps=5):
    import imageio
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=1)
    for f in frames:
        writer.append_data(f)
    writer.close()


def _make_comparison_frame(image, gt_action, pred_action):
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


def _7dof_to_4dof(action_7d):
    return np.concatenate([action_7d[:3], action_7d[6:7]])


def load_rlds_episodes(data_dir, dataset_name, split):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow_datasets as tfds
    data_dir = os.path.expanduser(data_dir)
    splits = ["train", "val"] if split == "all" else [split]
    episodes = []
    for s in splits:
        ds = tfds.load(dataset_name, split=s, data_dir=data_dir)
        for ep_idx, ep in enumerate(ds):
            steps = list(ep["steps"])
            images = np.array([step["observation"]["image"].numpy() for step in steps])
            actions = np.array([step["action"].numpy() for step in steps])
            instruction = steps[0]["language_instruction"].numpy().decode()
            episodes.append({"images": images, "actions": actions[:, :4], "instruction": instruction, "split": s, "idx": ep_idx})
    return episodes


def main():
    args = parse_args()
    adapter_path = Path(args.adapter_path).resolve()
    output_dir = Path(args.output_dir) if args.output_dir else adapter_path / f"eval_traindata_{args.split}"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    from model import SpatialVLAConfig, SpatialVLAForConditionalGeneration, SpatialVLAProcessor

    processor = SpatialVLAProcessor.from_pretrained(str(adapter_path), trust_remote_code=True)
    base_model = SpatialVLAForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model = model.merge_and_unload()
    model = model.to(device).eval()

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
        prompt = f"What action should the robot take to {instruction.lower()}?"

        pred_actions = []
        frames = []

        for t in range(len(images)):
            image = images[t]
            if image.dtype != np.uint8:
                image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

            pil_image = Image.fromarray(image).convert("RGB")
            inputs = processor(images=[pil_image], text=prompt, return_tensors="pt")
            with torch.no_grad():
                generation_outputs = model.predict_action(inputs)
            result = processor.decode_actions(generation_outputs, unnorm_key=args.unnorm_key)
            action_7d = result["actions"][0]
            action_4d = _7dof_to_4dof(action_7d)
            pred_actions.append(action_4d)

            if args.save_videos:
                frames.append(_make_comparison_frame(image, gt_actions[t], action_4d))

        pred_actions = np.array(pred_actions)
        l1_per_dim = np.abs(pred_actions - gt_actions).mean(axis=0)
        l1_total = l1_per_dim.mean()

        ep_result = {
            "split": ep["split"], "idx": ep["idx"], "instruction": instruction,
            "num_steps": len(images), "l1_total": float(l1_total),
            "l1_x": float(l1_per_dim[0]), "l1_y": float(l1_per_dim[1]),
            "l1_z": float(l1_per_dim[2]), "l1_grip": float(l1_per_dim[3]),
        }
        all_results.append(ep_result)
        tqdm.tqdm.write(
            f"  [{ep['split']}] ep {ep['idx']}: L1={l1_total:.4f} "
            f"(x={l1_per_dim[0]:.4f} y={l1_per_dim[1]:.4f} z={l1_per_dim[2]:.4f} g={l1_per_dim[3]:.4f})"
        )

        if args.save_videos and frames:
            _save_video(frames, output_dir / f"ep_{ep['split']}_{ep['idx']:03d}.mp4")

    mean_l1 = np.mean([r["l1_total"] for r in all_results])
    print(f"\n{'='*60}")
    print(f"Mean L1: {mean_l1:.4f} over {len(all_results)} episodes")
    for d in ["l1_x", "l1_y", "l1_z", "l1_grip"]:
        print(f"  {d}: {np.mean([r[d] for r in all_results]):.4f}")
    print(f"{'='*60}")

    summary = {
        "adapter_path": str(adapter_path), "split": args.split,
        "num_episodes": len(all_results), "mean_l1": float(mean_l1),
        "results": all_results,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
