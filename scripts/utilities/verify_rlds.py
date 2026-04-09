#!/usr/bin/env python3
"""Verify a built Go VLA RLDS dataset: print stats and optionally save a frame."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
BUILDER_DIR = THIS_FILE.parents[1] / "rlds_builder"

sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "go_vla_benchmark"))
sys.path.insert(0, str(BUILDER_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a built Go VLA RLDS dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.expanduser("~/tensorflow_datasets"),
        help="TFDS data directory (default: ~/tensorflow_datasets)",
    )
    parser.add_argument(
        "--save-frame",
        type=str,
        default=None,
        help="If set, save the first frame of the first episode as a PNG.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from go_vla_dataset.go_vla_dataset_dataset_builder import Builder

    builder = Builder(data_dir=args.data_dir)
    ds = builder.as_dataset(split="train")

    num_episodes = 0
    total_steps = 0
    all_actions: list[np.ndarray] = []
    first_instruction: str | None = None
    first_image_shape: tuple | None = None
    first_frame: np.ndarray | None = None

    for episode in ds:
        num_episodes += 1
        for step in episode["steps"]:
            action = step["action"].numpy()
            all_actions.append(action)
            total_steps += 1

            if first_instruction is None:
                first_instruction = step["language_instruction"].numpy().decode("utf-8")
                img = step["observation"]["image"].numpy()
                first_image_shape = img.shape
                first_frame = img

    if not all_actions:
        print("No data found in the dataset.")
        return

    actions = np.stack(all_actions)

    print("=== Go VLA RLDS Dataset Verification ===")
    print(f"  Episodes        : {num_episodes}")
    print(f"  Total steps     : {total_steps}")
    print(f"  Action shape    : {actions.shape[1:]}")
    print()
    print("  Action stats per dimension:")
    for d in range(actions.shape[1]):
        col = actions[:, d]
        print(
            f"    dim {d}: min={col.min():.4f}  max={col.max():.4f}  "
            f"mean={col.mean():.4f}"
        )
    print()
    print(f"  First episode instruction : {first_instruction}")
    print(f"  First episode image shape : {first_image_shape}")

    # Sanity checks
    print()

    xyz = actions[:, :3]
    if xyz.min() >= -1.0 and xyz.max() <= 1.0:
        print("  [OK] XYZ action dims 0-2 are within [-1, 1].")
    else:
        print(
            f"  [INFO] XYZ action dims 0-2 range: [{xyz.min():.4f}, {xyz.max():.4f}]"
        )

    gripper = actions[:, 3]
    if gripper.min() >= 0.0 and gripper.max() <= 1.0:
        print(
            "  [OK] Gripper dim 3 uses OpenVLA convention [0, 1] "
            "(1=open, 0=close)."
        )
    elif gripper.min() >= -1.0 and gripper.max() <= 1.0:
        print(
            "  [WARN] Gripper dim 3 looks like raw benchmark/env semantics "
            "[-1, 1] instead of OpenVLA's [0, 1] convention."
        )
    else:
        print(
            f"  [INFO] Gripper dim 3 range: [{gripper.min():.4f}, {gripper.max():.4f}]"
        )

    unique_gripper = np.unique(gripper)
    gripper_var = gripper.var()
    print(f"  Gripper unique values: {unique_gripper}")
    print(f"  Gripper variance: {gripper_var:.6f}")
    if gripper_var > 0:
        print("  [OK] Gripper has non-zero variance (not stuck).")
    else:
        print("  [WARN] Gripper variance is zero — model may learn constant gripper!")

    if args.save_frame and first_frame is not None:
        from PIL import Image

        img = Image.fromarray(first_frame)
        save_path = Path(args.save_frame)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(save_path))
        print(f"\n  Saved first frame to {save_path}")


if __name__ == "__main__":
    main()
