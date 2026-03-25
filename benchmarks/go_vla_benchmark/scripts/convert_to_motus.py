"""Convert RoboTwin HDF5 episodes to Motus data format.

Motus format per episode:
  {subset}/{task}/qpos/{id}.pt    — (T, 14) float32 joint positions
  {subset}/{task}/videos/{id}.mp4 — T-shaped multi-camera concat, 320x360, 30fps
  {subset}/{task}/metas/{id}.txt  — text instruction
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import h5py
import numpy as np
import torch


def decode_images(encoded_data: np.ndarray) -> list[np.ndarray]:
    """Decode JPEG-encoded images from HDF5 byte strings."""
    images = []
    for i in range(len(encoded_data)):
        raw = encoded_data[i]
        if isinstance(raw, bytes):
            buf = np.frombuffer(raw.rstrip(b"\0"), dtype=np.uint8)
        else:
            buf = np.frombuffer(raw.tobytes().rstrip(b"\0"), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)
    return images


def make_t_shape_frame(
    head: np.ndarray,
    left_wrist: np.ndarray,
    right_wrist: np.ndarray,
    target_w: int = 320,
    target_h: int = 360,
) -> np.ndarray:
    """Concat cameras in T-shape layout and resize.

    Top row: head camera (full width)
    Bottom row: left_wrist + right_wrist (each half-width, side by side)
    """
    h, w = head.shape[:2]

    # Resize all to consistent widths
    head_resized = cv2.resize(head, (target_w, target_h // 2))
    half_w = target_w // 2
    bottom_h = target_h - target_h // 2
    left_resized = cv2.resize(left_wrist, (half_w, bottom_h))
    right_resized = cv2.resize(right_wrist, (target_w - half_w, bottom_h))

    bottom = np.concatenate([left_resized, right_resized], axis=1)
    frame = np.concatenate([head_resized, bottom], axis=0)
    return frame


def convert_episode(
    hdf5_path: str,
    episode_id: int,
    output_dir: str,
    scene_info: dict | None = None,
) -> bool:
    """Convert a single RoboTwin HDF5 episode to Motus format.

    Returns True if conversion succeeded, False if skipped.
    """
    with h5py.File(hdf5_path, "r") as f:
        # Extract qpos: joint_action/vector -> (T, 14)
        if "joint_action" not in f or "vector" not in f["joint_action"]:
            print(f"  Skipping {hdf5_path}: no joint_action/vector")
            return False
        qpos_data = f["joint_action/vector"][()]
        qpos_tensor = torch.from_numpy(qpos_data).float()

        # Validate: reject if max(abs(qpos)) > 4.0
        if torch.max(torch.abs(qpos_tensor)) > 4.0:
            print(f"  Skipping {hdf5_path}: qpos out of range (max={torch.max(torch.abs(qpos_tensor)):.3f})")
            return False

        # Extract camera images
        head_imgs = decode_images(f["observation/head_camera/rgb"][()])

        # Try to get wrist cameras; fall back to head if not available
        if "observation/left_camera/rgb" in f:
            left_imgs = decode_images(f["observation/left_camera/rgb"][()])
        else:
            left_imgs = head_imgs

        if "observation/right_camera/rgb" in f:
            right_imgs = decode_images(f["observation/right_camera/rgb"][()])
        else:
            right_imgs = head_imgs

    # Ensure frame counts match
    n_frames = min(len(head_imgs), len(left_imgs), len(right_imgs), qpos_tensor.shape[0])
    if n_frames == 0:
        print(f"  Skipping {hdf5_path}: no frames")
        return False

    # Save qpos
    qpos_dir = os.path.join(output_dir, "qpos")
    os.makedirs(qpos_dir, exist_ok=True)
    torch.save(qpos_tensor[:n_frames], os.path.join(qpos_dir, f"{episode_id}.pt"))

    # Create video
    video_dir = os.path.join(output_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, f"{episode_id}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, 30, (320, 360))

    for i in range(n_frames):
        frame = make_t_shape_frame(head_imgs[i], left_imgs[i], right_imgs[i])
        writer.write(frame)
    writer.release()

    # Create meta text
    meta_dir = os.path.join(output_dir, "metas")
    os.makedirs(meta_dir, exist_ok=True)

    # Generate instruction from scene info if available
    instruction = "Place a stone on the Go board intersection."
    if scene_info:
        ep_info = scene_info.get(f"episode_{episode_id}", {}).get("info", {})
        color = ep_info.get("stone_color", "")
        row = ep_info.get("target_row", "")
        col = ep_info.get("target_col", "")
        if color and row != "" and col != "":
            instruction = f"Place a {color} stone at row {row}, column {col} on the Go board."

    with open(os.path.join(meta_dir, f"{episode_id}.txt"), "w") as f:
        f.write(instruction)

    return True


def main():
    parser = argparse.ArgumentParser(description="Convert RoboTwin HDF5 to Motus format")
    parser.add_argument("--input", required=True, help="Path to RoboTwin task data directory")
    parser.add_argument("--output", required=True, help="Output directory for Motus format data")
    parser.add_argument("--subset", default="randomized", help="Subset name (clean or randomized)")
    parser.add_argument("--task", default="go_stone_placement", help="Task name")
    args = parser.parse_args()

    data_dir = os.path.join(args.input, "data")
    if not os.path.isdir(data_dir):
        print(f"Error: data directory not found at {data_dir}")
        sys.exit(1)

    # Load scene info if available
    scene_info_path = os.path.join(args.input, "scene_info.json")
    scene_info = None
    if os.path.exists(scene_info_path):
        with open(scene_info_path, "r") as f:
            scene_info = json.load(f)

    output_dir = os.path.join(args.output, args.subset, args.task)
    os.makedirs(output_dir, exist_ok=True)

    # Find all HDF5 episode files
    hdf5_files = []
    for fname in os.listdir(data_dir):
        if fname.startswith("episode") and fname.endswith(".hdf5"):
            idx = int(fname.replace("episode", "").replace(".hdf5", ""))
            hdf5_files.append((idx, os.path.join(data_dir, fname)))
    hdf5_files.sort()

    if not hdf5_files:
        print(f"No HDF5 episode files found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(hdf5_files)} episodes")
    success_count = 0

    for idx, hdf5_path in hdf5_files:
        print(f"Converting episode {idx}...")
        if convert_episode(hdf5_path, idx, output_dir, scene_info):
            success_count += 1

    print(f"\nDone: {success_count}/{len(hdf5_files)} episodes converted")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
