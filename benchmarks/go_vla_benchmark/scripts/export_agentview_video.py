#!/usr/bin/env python3
"""Export agentview frames from dataset episodes into a single MP4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import h5py
import imageio
import numpy as np


def _sorted_demo_keys(data_group: h5py.Group) -> List[str]:
    demos = list(data_group.keys())
    order = np.argsort([int(x.split("_")[1]) for x in demos])
    return [demos[i] for i in order]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="input hdf5 dataset path")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="output mp4 path (default: dataset path with _agentview.mp4 suffix)",
    )
    parser.add_argument("--start", type=int, default=0, help="start demo index in sorted order")
    parser.add_argument("--num-demos", type=int, default=10, help="number of demos to export; <= 0 means all")
    parser.add_argument(
        "--demos",
        type=str,
        default=None,
        help="comma-separated explicit demo keys (e.g., demo_0,demo_7). Overrides start/num-demos.",
    )
    parser.add_argument("--fps", type=int, default=12, help="output video fps")
    parser.add_argument("--stride", type=int, default=1, help="frame stride per demo (>=1)")
    parser.add_argument("--separator-frames", type=int, default=4, help="black frames inserted between demos")
    parser.add_argument(
        "--camera-size",
        type=int,
        default=None,
        help="optional validation: assert dataset frames are square and match this native size",
    )
    parser.add_argument(
        "--macro-block-size",
        type=int,
        default=1,
        help="ffmpeg macro block size; use 1 to keep native 84x84 without resizing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    output_path = Path(args.output) if args.output else dataset_path.with_name(dataset_path.stem + "_agentview.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stride = max(1, int(args.stride))
    separator_frames = max(0, int(args.separator_frames))

    with h5py.File(dataset_path, "r") as f:
        if "data" not in f:
            raise RuntimeError(f"{dataset_path} is missing 'data' group")
        data_group = f["data"]
        all_demos = _sorted_demo_keys(data_group)
        if len(all_demos) == 0:
            raise RuntimeError(f"no demos found in {dataset_path}")

        if args.demos:
            selected = [x.strip() for x in args.demos.split(",") if x.strip()]
            missing = [x for x in selected if x not in data_group]
            if missing:
                raise RuntimeError(f"requested demos not found: {missing}")
            demo_keys = selected
        else:
            start = max(0, int(args.start))
            if args.num_demos <= 0:
                demo_keys = all_demos[start:]
            else:
                demo_keys = all_demos[start : start + int(args.num_demos)]

        first_obs = data_group[f"{demo_keys[0]}/obs/agentview_image"]
        frame_h, frame_w = int(first_obs.shape[1]), int(first_obs.shape[2])
        if args.camera_size is not None:
            expected = int(args.camera_size)
            if expected <= 0:
                raise ValueError("--camera-size must be > 0")
            if (frame_h != frame_w) or (frame_h != expected):
                raise RuntimeError(
                    f"dataset frames are {frame_h}x{frame_w}, expected square {expected}x{expected}"
                )
        blank = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

        total_frames = 0
        frames_per_demo = {}
        with imageio.get_writer(
            str(output_path),
            fps=int(args.fps),
            macro_block_size=int(args.macro_block_size),
        ) as writer:
            for idx, demo_key in enumerate(demo_keys):
                obs_key = f"{demo_key}/obs/agentview_image"
                if obs_key not in data_group:
                    raise RuntimeError(f"missing {obs_key} in dataset")
                frames = data_group[obs_key][:]

                if frames.ndim != 4 or frames.shape[-1] != 3:
                    raise RuntimeError(f"unexpected shape for {obs_key}: {frames.shape}")

                count = 0
                for frame in frames[::stride]:
                    writer.append_data(frame)
                    count += 1

                frames_per_demo[demo_key] = count
                total_frames += count

                if idx < len(demo_keys) - 1:
                    for _ in range(separator_frames):
                        writer.append_data(blank)
                        total_frames += 1

    stats = {
        "dataset": str(dataset_path),
        "output_video": str(output_path),
        "num_demos": len(demo_keys),
        "demo_keys": demo_keys,
        "fps": int(args.fps),
        "stride": stride,
        "frames_per_demo": frames_per_demo,
        "total_video_frames": total_frames,
    }
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
