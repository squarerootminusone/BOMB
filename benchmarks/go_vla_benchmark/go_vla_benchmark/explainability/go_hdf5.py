"""Dataset adapter for Go benchmark MimicGen-compatible HDF5 files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import h5py
import numpy as np

from ..rlds_preprocessing import (
    build_instruction,
    derive_target_from_board_state,
    compute_rlds_keep_indices,
    extract_action_4d,
    remap_gripper_to_openvla,
)
from .core import EpisodeClip


def _dataset_demo_keys(data_group: h5py.Group) -> List[str]:
    # Respect the order exposed by the HDF5 group so downstream traces and reports
    # line up with the source dataset. When the file was created with track_order,
    # h5py preserves insertion order here.
    return list(data_group.keys())


def _build_instruction(ep_group: h5py.Group, demo_idx: int) -> str:
    if "obs/board_state" not in ep_group:
        return f"Demo {demo_idx}"

    board_state = np.asarray(ep_group["obs/board_state"], dtype=np.float32)
    stone_color = _stone_color_from_group(ep_group)
    return build_instruction(
        board_state=board_state,
        demo_idx=demo_idx,
        stone_color=stone_color,
    )


def _stone_color_from_group(ep_group: h5py.Group) -> str:
    if "stone_color" not in ep_group:
        return "black"
    value = ep_group["stone_color"][()]
    if isinstance(value, bytes):
        return value.decode()
    if hasattr(value, "decode"):
        return value.decode()
    return str(value)


def _opening_move_history_from_group(ep_group: h5py.Group) -> Optional[np.ndarray]:
    if "obs/move_history" not in ep_group:
        return None
    move_history = np.asarray(ep_group["obs/move_history"], dtype=np.int32)
    if move_history.ndim == 0 or move_history.shape[0] == 0:
        return None
    initial_history = move_history[0].reshape(-1)
    valid_history = initial_history[initial_history >= 0]
    if valid_history.size == 0:
        return None
    return valid_history.astype(np.int32)


def _opening_moves_from_board_state(board_state: np.ndarray) -> int:
    board_state = np.asarray(board_state, dtype=np.float32)
    if board_state.ndim < 4 or board_state.shape[0] == 0:
        return 0
    initial_board = board_state[0]
    occupancy_channels = initial_board[:, :, 1:3] if initial_board.shape[-1] >= 3 else initial_board[:, :, 1:]
    if occupancy_channels.size == 0:
        return 0
    occupied = np.any(occupancy_channels > 0.5, axis=-1)
    return int(np.count_nonzero(occupied))


def _select_demo_keys_preserving_dataset_order(
    all_demo_keys: Sequence[str],
    demos: Optional[str],
    start: int,
    num_demos: int,
) -> List[str]:
    if demos:
        requested = {item.strip() for item in demos.split(",") if item.strip()}
        missing = [demo for demo in requested if demo not in all_demo_keys]
        if missing:
            raise ValueError(f"requested demos not found: {missing}")
        return [demo_key for demo_key in all_demo_keys if demo_key in requested]
    return _select_demo_keys(all_demo_keys, demos=None, start=start, num_demos=num_demos)


@dataclass
class GoHDF5OnlineDemoSpec:
    demo_key: str
    instruction: str
    target_row: int
    target_col: int
    stone_color: str
    opening_moves: int
    opening_move_history: Optional[np.ndarray]
    reference_frame_index: int
    reference_image: np.ndarray


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

            all_demo_keys = _dataset_demo_keys(handle["data"])
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

    def load_online_demo_specs(
        self,
        dataset_path: Path,
        demos: Optional[str],
        start: int,
        num_demos: int,
        stride: int,
        reference_step: int = 0,
    ) -> List[GoHDF5OnlineDemoSpec]:
        stride = max(1, int(stride))
        reference_step = max(0, int(reference_step))

        with h5py.File(dataset_path, "r") as handle:
            if "data" not in handle:
                raise RuntimeError(f"{dataset_path} is missing 'data' group")

            all_demo_keys = _dataset_demo_keys(handle["data"])
            selected_demo_keys = _select_demo_keys_preserving_dataset_order(
                all_demo_keys,
                demos=demos,
                start=start,
                num_demos=num_demos,
            )

            demos_out: List[GoHDF5OnlineDemoSpec] = []
            for demo_key in selected_demo_keys:
                ep_group = handle[f"data/{demo_key}"]
                images = np.asarray(ep_group[self.camera_key], dtype=np.uint8)
                gt_actions = remap_gripper_to_openvla(
                    extract_action_4d(np.asarray(ep_group["actions"], dtype=np.float32))
                )
                frame_indices = compute_rlds_keep_indices(gt_actions)[::stride]
                if frame_indices.size == 0:
                    continue
                ref_idx = frame_indices[min(reference_step, int(frame_indices.size) - 1)]

                demo_idx = int(demo_key.split("_")[1])
                board_state = (
                    np.asarray(ep_group["obs/board_state"], dtype=np.float32)
                    if "obs/board_state" in ep_group
                    else np.zeros((0,), dtype=np.float32)
                )
                if board_state.ndim >= 4 and board_state.shape[0] > 0:
                    target_row, target_col = derive_target_from_board_state(board_state)
                    opening_moves = _opening_moves_from_board_state(board_state)
                else:
                    target_row, target_col = 0, 0
                    opening_moves = 0
                opening_move_history = _opening_move_history_from_group(ep_group)
                if opening_move_history is not None:
                    opening_moves = int(opening_move_history.shape[0])

                demos_out.append(
                    GoHDF5OnlineDemoSpec(
                        demo_key=demo_key,
                        instruction=_build_instruction(ep_group, demo_idx=demo_idx),
                        target_row=int(target_row),
                        target_col=int(target_col),
                        stone_color=_stone_color_from_group(ep_group),
                        opening_moves=int(opening_moves),
                        opening_move_history=None
                        if opening_move_history is None
                        else np.asarray(opening_move_history, dtype=np.int32),
                        reference_frame_index=int(ref_idx),
                        reference_image=np.asarray(images[int(ref_idx)], dtype=np.uint8),
                    )
                )

        return demos_out
