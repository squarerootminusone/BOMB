#!/usr/bin/env python3
"""Export the exact image sequence consumed by the RLDS builder from HDF5.

This script starts from a raw MimicGen-compatible HDF5 file and reproduces the
same frame-selection logic used during HDF5 -> RLDS conversion:

1. Extract 4D actions ``[dx, dy, dz, gripper]`` from the stored action array.
2. Compute the RLDS keep mask with the same near-duplicate removal rule.
3. Filter ``obs/agentview_image`` with that mask and write the kept frames to MP4.

The output video therefore shows the per-step images that make it into RLDS,
which is closer to what the model is actually trained on than the raw HDF5
trajectory preview.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import h5py
import imageio
import numpy as np

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]


def _sorted_demo_keys(data_group: h5py.Group) -> List[str]:
    demos = list(data_group.keys())
    order = np.argsort([int(x.split("_")[1]) for x in demos])
    return [demos[i] for i in order]


def _dedupe_paths(paths: List[Path]) -> List[Path]:
    deduped: List[Path] = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def _resolve_dataset_path(dataset: str) -> Path:
    requested = Path(dataset).expanduser()
    if requested.is_absolute():
        if requested.is_file():
            return requested.resolve()
        raise FileNotFoundError(f"dataset not found at absolute path: {requested}")

    direct_candidates = _dedupe_paths(
        [
            Path.cwd() / requested,
            REPO_ROOT / requested,
        ]
    )
    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate

    search_roots = [
        REPO_ROOT / "benchmarks" / "go_vla_benchmark" / "data",
        REPO_ROOT / "benchmarks",
    ]
    discovered: List[Path] = []
    for root in search_roots:
        if root.exists():
            discovered.extend(root.rglob(requested.name))
    discovered = _dedupe_paths([path for path in discovered if path.is_file()])

    if len(discovered) == 1:
        return discovered[0]

    tried_lines = "\n".join(f"  - {path}" for path in direct_candidates)
    if discovered:
        found_lines = "\n".join(f"  - {path}" for path in discovered)
        raise FileNotFoundError(
            f"dataset '{dataset}' was not found at expected locations.\n"
            f"Tried:\n{tried_lines}\n"
            f"Found multiple files with the same name:\n{found_lines}\n"
            "Pass an absolute --dataset path to disambiguate."
        )

    raise FileNotFoundError(
        f"dataset '{dataset}' was not found.\n"
        f"Tried:\n{tried_lines}\n"
        "Tip: pass an absolute --dataset path from the collector output."
    )


def _resolve_output_path(output: str | None, dataset_path: Path) -> Path:
    if output is None:
        return dataset_path.with_name(dataset_path.stem + "_model_input.mp4")
    output_path = Path(output).expanduser()
    if output_path.is_absolute():
        return output_path.resolve()
    return (REPO_ROOT / output_path).resolve()


def _extract_action_4d(actions: np.ndarray) -> np.ndarray:
    """Mirror the RLDS builder's 4D action extraction logic."""
    if actions.shape[1] >= 7:
        out = np.zeros((actions.shape[0], 4), dtype=np.float32)
        out[:, :3] = actions[:, :3]
        out[:, 3] = actions[:, 6]
        return out
    return actions[:, :4].astype(np.float32)


def _deduplicate_indices(
    actions_4d: np.ndarray,
    eef_pos: np.ndarray,
    action_norm_thresh: float = 0.02,
    eef_disp_thresh: float = 0.001,
) -> np.ndarray:
    """Mirror the RLDS builder's frame-selection logic exactly."""
    n = actions_4d.shape[0]
    keep = np.zeros(n, dtype=bool)
    keep[0] = True
    keep[-1] = True
    last_kept_pos = eef_pos[0].copy()
    for t in range(1, n - 1):
        action_norm = float(np.linalg.norm(actions_4d[t, :3]))
        eef_disp = float(np.linalg.norm(eef_pos[t] - last_kept_pos))
        if action_norm > action_norm_thresh or eef_disp > eef_disp_thresh:
            keep[t] = True
            last_kept_pos = eef_pos[t].copy()
    return keep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the exact RLDS/model input image sequence from an HDF5 dataset."
    )
    parser.add_argument("--dataset", type=str, required=True, help="input hdf5 dataset path")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="output mp4 path (default: dataset path with _model_input.mp4 suffix)",
    )
    parser.add_argument("--start", type=int, default=0, help="start demo index in sorted order")
    parser.add_argument("--num-demos", type=int, default=1, help="number of demos to export; <= 0 means all")
    parser.add_argument(
        "--demos",
        type=str,
        default=None,
        help="comma-separated explicit demo keys (e.g. demo_0,demo_7). Overrides start/num-demos.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        help="output video fps (default: 8 to match the control frequency)",
    )
    parser.add_argument(
        "--separator-frames",
        type=int,
        default=0,
        help="black frames inserted between demos (default: 0 to preserve exact frame order)",
    )
    parser.add_argument(
        "--hold-last-frames",
        type=int,
        default=0,
        help="extra copies of each demo's last kept frame for easier viewing (default: 0)",
    )
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
        help="ffmpeg macro block size; use 1 to keep native frame size",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = _resolve_dataset_path(args.dataset)
    output_path = _resolve_output_path(args.output, dataset_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    separator_frames = max(0, int(args.separator_frames))
    hold_last_frames = max(0, int(args.hold_last_frames))

    with h5py.File(dataset_path, "r") as f:
        if "data" not in f:
            raise RuntimeError(f"{dataset_path} is missing 'data' group")
        data_group = f["data"]
        all_demos = _sorted_demo_keys(data_group)
        if len(all_demos) == 0:
            raise RuntimeError(f"no demos found in {dataset_path}")

        if args.demos:
            demo_keys = [x.strip() for x in args.demos.split(",") if x.strip()]
            missing = [x for x in demo_keys if x not in data_group]
            if missing:
                raise RuntimeError(f"requested demos not found: {missing}")
        else:
            start = max(0, int(args.start))
            if args.num_demos <= 0:
                demo_keys = all_demos[start:]
            else:
                demo_keys = all_demos[start : start + int(args.num_demos)]

        if not demo_keys:
            raise RuntimeError("no demos selected")

        first_obs = data_group[f"{demo_keys[0]}/obs/agentview_image"]
        frame_h, frame_w = int(first_obs.shape[1]), int(first_obs.shape[2])
        if args.camera_size is not None:
            expected = int(args.camera_size)
            if expected <= 0:
                raise ValueError("--camera-size must be > 0")
            if (frame_h != frame_w) or (frame_h != expected):
                raise RuntimeError(
                    f"dataset frames are {frame_h}x{frame_w}. "
                    f"--camera-size must match {frame_h} (or be omitted)."
                )

        blank = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        demo_stats = []
        total_video_frames = 0

        with imageio.get_writer(
            str(output_path),
            fps=int(args.fps),
            macro_block_size=int(args.macro_block_size),
        ) as writer:
            for idx, demo_key in enumerate(demo_keys):
                ep = data_group[demo_key]
                images = np.asarray(ep["obs/agentview_image"], dtype=np.uint8)
                eef_pos = np.asarray(ep["obs/eef_pos"], dtype=np.float32)
                actions = np.asarray(ep["actions"], dtype=np.float32)

                if images.shape[0] == 0:
                    raise RuntimeError(f"{demo_key} contains no frames")
                if eef_pos.shape[0] != images.shape[0] or actions.shape[0] != images.shape[0]:
                    raise RuntimeError(
                        f"{demo_key} has inconsistent lengths: "
                        f"images={images.shape[0]}, eef_pos={eef_pos.shape[0]}, actions={actions.shape[0]}"
                    )

                actions_4d = _extract_action_4d(actions)
                keep_mask = _deduplicate_indices(actions_4d=actions_4d, eef_pos=eef_pos)
                kept_images = images[keep_mask]

                for frame in kept_images:
                    writer.append_data(frame)
                    total_video_frames += 1

                if hold_last_frames > 0:
                    last_frame = kept_images[-1]
                    for _ in range(hold_last_frames):
                        writer.append_data(last_frame)
                        total_video_frames += 1

                if idx < len(demo_keys) - 1:
                    for _ in range(separator_frames):
                        writer.append_data(blank)
                        total_video_frames += 1

                demo_stats.append(
                    {
                        "demo_key": demo_key,
                        "raw_frames": int(images.shape[0]),
                        "kept_frames": int(kept_images.shape[0]),
                        "dropped_frames": int(images.shape[0] - kept_images.shape[0]),
                        "keep_ratio": float(kept_images.shape[0] / images.shape[0]),
                    }
                )

    print(
        json.dumps(
            {
                "dataset": str(dataset_path),
                "output_video": str(output_path),
                "num_demos": len(demo_stats),
                "fps": int(args.fps),
                "separator_frames": separator_frames,
                "hold_last_frames": hold_last_frames,
                "total_video_frames": total_video_frames,
                "demos": demo_stats,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
