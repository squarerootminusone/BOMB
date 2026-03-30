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


def _absolute_gripper_to_benchmark(gripper: np.ndarray | float, *, binarize: bool) -> np.ndarray:
    """Map standardized absolute gripper values `[1=open, 0=close]` into benchmark semantics."""
    converted = 1.0 - (2.0 * np.asarray(gripper, dtype=np.float32))
    if binarize:
        converted = np.sign(converted)
    return converted.astype(np.float32)


def benchmark_action_to_openvla(action: np.ndarray) -> np.ndarray:
    """Convert a benchmark action `[dx, dy, dz, gripper]` for OpenVLA training."""
    action_arr = np.asarray(action, dtype=np.float32)
    if action_arr.shape[-1] < 4:
        raise ValueError(f"expected action with at least 4 dims, got shape {action_arr.shape}")

    converted = action_arr.copy()
    converted[..., 3] = benchmark_gripper_to_openvla(converted[..., 3])
    return converted


def canonicalize_action_to_benchmark(action: np.ndarray, *, binarize: bool = False) -> np.ndarray:
    """Return actions using benchmark gripper semantics `-1=open, +1=close`.

    Mixed sources appear in the benchmark tooling:
    - raw HDF5 / env actions already use benchmark semantics
    - RLDS/OpenVLA-standardized actions may use `[1=open, 0=close]`

    This helper only converts when the gripper channel looks OpenVLA-like.
    """
    action_arr = np.asarray(action, dtype=np.float32)
    if action_arr.shape[-1] < 4:
        raise ValueError(f"expected action with at least 4 dims, got shape {action_arr.shape}")

    converted = action_arr.copy()
    gripper = converted[..., 3]
    if gripper.size > 0 and np.all((gripper >= 0.0) & (gripper <= 1.0)):
        converted[..., 3] = _absolute_gripper_to_benchmark(gripper, binarize=binarize)
    return converted


def standardized_action_to_benchmark(action: np.ndarray, *, binarize: bool = True) -> np.ndarray:
    """Convert a standardized absolute-gripper action into benchmark env semantics.

    This is suitable for any model trained against Go RLDS actions that use:
      - `1.0` = open
      - `0.0` = close

    Mirrors the stock OpenVLA rollout helpers:
      1. gripper `[0, 1]` -> `[-1, 1]`
      2. optional binarization to `{-1, 0, 1}`
      3. sign flip so env uses `-1=open, +1=close`
    """
    action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if action_arr.shape[0] < 4:
        raise ValueError(f"expected at least 4 action dims, got shape {action_arr.shape}")

    converted = action_arr[:4].copy()
    converted[3] = _absolute_gripper_to_benchmark(converted[3], binarize=binarize)
    return converted


def openvla_action_to_benchmark(action: np.ndarray, *, binarize: bool = True) -> np.ndarray:
    """Backward-compatible alias for OpenVLA rollouts using standardized gripper outputs."""
    return standardized_action_to_benchmark(action, binarize=binarize)
