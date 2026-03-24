"""Dataset adapter for Go benchmark MimicGen-compatible HDF5 files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import h5py
import numpy as np

from ..rlds_preprocessing import (
    build_instruction,
    compute_rlds_keep_indices,
    extract_action_4d,
    remap_gripper_to_openvla,
)
from .core import EpisodeClip


def _sorted_demo_keys(data_group: h5py.Group) -> List[str]:
    demos = list(data_group.keys())
    order = np.argsort([int(x.split("_")[1]) for x in demos])
    return [demos[i] for i in order]


def _build_instruction(ep_group: h5py.Group, demo_idx: int) -> str:
    if "obs/board_state" not in ep_group:
        return f"Demo {demo_idx}"

    board_state = np.asarray(ep_group["obs/board_state"], dtype=np.float32)
    stone_color = ep_group["stone_color"][()].decode() if "stone_color" in ep_group else "black"
    return build_instruction(
        board_state=board_state,
        demo_idx=demo_idx,
        stone_color=stone_color,
    )


def _select_demo_keys(
    all_demo_keys: Sequence[str],
    demos: Optional[str],
    start: int,
    num_demos: int,
) -> List[str]:
    if demos:
        requested = [item.strip() for item in demos.split(",") if item.strip()]
        missing = [demo for demo in requested if demo not in all_demo_keys]
        if missing:
            raise ValueError(f"requested demos not found: {missing}")
        return requested

    start = max(0, int(start))
    if num_demos <= 0:
        return list(all_demo_keys[start:])
    return list(all_demo_keys[start : start + int(num_demos)])


@dataclass
class GoHDF5DatasetAdapter:
    """Load RLDS-aligned explainability clips from the Go benchmark HDF5 dataset."""

    camera_key: str = "obs/agentview_image"

    def load_episode_clips(
        self,
        dataset_path: Path,
        demos: Optional[str],
        start: int,
        num_demos: int,
        stride: int,
        max_steps: int,
    ) -> List[EpisodeClip]:
        stride = max(1, int(stride))

        with h5py.File(dataset_path, "r") as handle:
            if "data" not in handle:
                raise RuntimeError(f"{dataset_path} is missing 'data' group")

            all_demo_keys = _sorted_demo_keys(handle["data"])
            selected_demo_keys = _select_demo_keys(all_demo_keys, demos=demos, start=start, num_demos=num_demos)

            clips: List[EpisodeClip] = []
            for demo_key in selected_demo_keys:
                ep_group = handle[f"data/{demo_key}"]
                images = np.asarray(ep_group[self.camera_key], dtype=np.uint8)
                gt_actions = remap_gripper_to_openvla(
                    extract_action_4d(np.asarray(ep_group["actions"], dtype=np.float32))
                )
                demo_idx = int(demo_key.split("_")[1])
                instruction = _build_instruction(ep_group, demo_idx=demo_idx)

                frame_indices = compute_rlds_keep_indices(gt_actions)
                frame_indices = frame_indices[::stride]
                if max_steps > 0:
                    frame_indices = frame_indices[: max_steps]
                if frame_indices.size == 0:
                    continue

                clips.append(
                    EpisodeClip(
                        demo_key=demo_key,
                        instruction=instruction,
                        images=images[frame_indices],
                        gt_actions=gt_actions[frame_indices],
                        frame_indices=frame_indices,
                    )
                )

        return clips
