"""OpenVLA-specific gripper convention helpers for the Go benchmark.

The benchmark environment uses raw robot gripper commands:
  - `-1.0` = open
  - `+1.0` = close

For OpenVLA fine-tuning we follow the upstream absolute-gripper convention:
  - `1.0` = open
  - `0.0` = close

The gripper dimension is excluded from action normalization, so inference
returns that convention directly and we must convert it back before stepping
the environment.
"""

from __future__ import annotations

import numpy as np


def benchmark_gripper_to_openvla(gripper: np.ndarray | float) -> np.ndarray:
    """Map benchmark/env gripper commands to the OpenVLA training convention."""
    gripper_arr = np.asarray(gripper, dtype=np.float32)
    return np.where(gripper_arr <= 0.0, 1.0, 0.0).astype(np.float32)


def benchmark_action_to_openvla(action: np.ndarray) -> np.ndarray:
    """Convert a benchmark action `[dx, dy, dz, gripper]` for OpenVLA training."""
    action_arr = np.asarray(action, dtype=np.float32)
    if action_arr.shape[-1] < 4:
        raise ValueError(f"expected action with at least 4 dims, got shape {action_arr.shape}")

    converted = action_arr.copy()
    converted[..., 3] = benchmark_gripper_to_openvla(converted[..., 3])
    return converted


def openvla_action_to_benchmark(action: np.ndarray, *, binarize: bool = True) -> np.ndarray:
    """Convert an OpenVLA-predicted action into benchmark env semantics.

    Mirrors the stock OpenVLA rollout helpers:
      1. gripper `[0, 1]` -> `[-1, 1]`
      2. optional binarization to `{-1, 0, 1}`
      3. sign flip so env uses `-1=open, +1=close`
    """
    action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if action_arr.shape[0] < 4:
        raise ValueError(f"expected at least 4 action dims, got shape {action_arr.shape}")

    converted = action_arr[:4].copy()
    converted[3] = 2.0 * converted[3] - 1.0
    if binarize:
        converted[3] = np.sign(converted[3])
    converted[3] *= -1.0
    return converted
