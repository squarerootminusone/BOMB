"""Shared RLDS-style preprocessing helpers for Go benchmark HDF5 episodes."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .openvla_action_utils import canonicalize_action_to_benchmark


INSTRUCTION_TEMPLATES = [
    "Place a {color} stone on the Go board at row {r}, column {c}.",
    "Put a {color} stone at position ({r}, {c}) on the Go board.",
    "Move the {color} stone to row {r}, column {c} on the board.",
    "Set a {color} stone at ({r}, {c}).",
]

RLDS_SUBSAMPLE_STRIDE = 4
RLDS_NOOP_THRESHOLD = 1e-4


def derive_target_from_board_state(board_state: np.ndarray) -> tuple[int, int]:
    """Infer the placed stone location from the board-state delta."""
    for channel_idx in (1, 2):
        diff = board_state[-1, :, :, channel_idx] - board_state[0, :, :, channel_idx]
        if diff.max() > 0.5:
            row, col = divmod(int(np.argmax(diff)), board_state.shape[2])
            return row, col

    diff = board_state[-1, :, :, 1] - board_state[0, :, :, 1]
    row, col = divmod(int(np.argmax(diff)), board_state.shape[2])
    return row, col


def extract_action_4d(actions: np.ndarray) -> np.ndarray:
    """Convert legacy action layouts to `[dx, dy, dz, gripper]`."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"expected 2D action array, got shape {actions.shape}")

    if actions.shape[1] >= 7:
        out = np.zeros((actions.shape[0], 4), dtype=np.float32)
        out[:, :3] = actions[:, :3]
        out[:, 3] = actions[:, 6]
        return out

    if actions.shape[1] >= 4:
        return actions[:, :4].astype(np.float32)

    raise ValueError(f"expected action dimension >= 4, got shape {actions.shape}")


def remap_gripper_to_openvla(actions_4d: np.ndarray) -> np.ndarray:
    """Historical alias that canonicalizes actions to benchmark gripper semantics.

    Despite the legacy name, callers rely on this returning benchmark-style
    actions with `-1=open, +1=close`, converting from OpenVLA `[1, 0]` only
    when needed.
    """
    return canonicalize_action_to_benchmark(actions_4d, binarize=False)


def compute_rlds_keep_indices(
    actions_4d: np.ndarray,
    subsample_stride: int = RLDS_SUBSAMPLE_STRIDE,
    noop_threshold: float = RLDS_NOOP_THRESHOLD,
) -> np.ndarray:
    """Return the RLDS-style frame indices kept after subsampling + no-op filtering."""
    actions_4d = np.asarray(actions_4d, dtype=np.float32)
    if actions_4d.ndim != 2 or actions_4d.shape[1] < 4:
        raise ValueError(f"expected action array shaped [T, >=4], got {actions_4d.shape}")
    if actions_4d.shape[0] == 0:
        return np.zeros((0,), dtype=np.int32)

    stride = max(1, int(subsample_stride))
    threshold = float(noop_threshold)

    keep: list[int] = []
    for step_idx in range(0, actions_4d.shape[0], stride):
        if np.linalg.norm(actions_4d[step_idx, :3]) >= threshold:
            keep.append(step_idx)

    last_idx = actions_4d.shape[0] - 1
    if last_idx not in keep:
        keep.append(last_idx)
    return np.asarray(keep, dtype=np.int32)


def build_instruction(
    board_state: np.ndarray,
    demo_idx: int,
    stone_color: Optional[str] = None,
) -> str:
    """Build the same language instruction template used by the RLDS builder."""
    row, col = derive_target_from_board_state(board_state)
    color = stone_color or "black"
    rng = np.random.RandomState(seed=int(demo_idx))
    template = INSTRUCTION_TEMPLATES[rng.randint(len(INSTRUCTION_TEMPLATES))]
    return template.format(color=color, r=row, c=col)
