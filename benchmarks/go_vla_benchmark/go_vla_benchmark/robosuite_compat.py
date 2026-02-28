"""Minimal robosuite compatibility for MimicGen pose utilities.

This benchmark only needs quaternion / rotation conversions from
`robosuite.utils.transform_utils`. To avoid bringing the full robosuite
dependency stack (which can pull `egl_probe`), we provide a lightweight
fallback module when robosuite is not installed.
"""

from __future__ import annotations

import math
import sys
import types

import numpy as np


def _normalize_quat_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


def _axisangle2quat(axis_angle: np.ndarray) -> np.ndarray:
    aa = np.asarray(axis_angle, dtype=np.float64).reshape(3)
    angle = np.linalg.norm(aa)
    if angle < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    axis = aa / angle
    half = 0.5 * angle
    s = math.sin(half)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(half)], dtype=np.float64)


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    q = _normalize_quat_xyzw(quat)
    w = float(np.clip(q[3], -1.0, 1.0))
    angle = 2.0 * math.acos(w)
    s = math.sqrt(max(1.0 - w * w, 0.0))
    if s < 1e-12:
        return np.zeros(3, dtype=np.float64)
    axis = q[:3] / s
    return axis * angle


def _quat2mat(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = _normalize_quat_xyzw(quat)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _mat2quat(mat: np.ndarray) -> np.ndarray:
    m = np.asarray(mat, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1e-12)) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 1e-12)) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 1e-12)) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return _normalize_quat_xyzw(np.array([x, y, z, w], dtype=np.float64))


def _install_fallback_robosuite() -> None:
    robosuite_mod = types.ModuleType("robosuite")
    robosuite_mod.__version__ = "1.4.1-compat"

    utils_mod = types.ModuleType("robosuite.utils")
    transform_mod = types.ModuleType("robosuite.utils.transform_utils")
    transform_mod.axisangle2quat = _axisangle2quat
    transform_mod.quat2axisangle = _quat2axisangle
    transform_mod.quat2mat = _quat2mat
    transform_mod.mat2quat = _mat2quat

    utils_mod.transform_utils = transform_mod
    robosuite_mod.utils = utils_mod

    sys.modules["robosuite"] = robosuite_mod
    sys.modules["robosuite.utils"] = utils_mod
    sys.modules["robosuite.utils.transform_utils"] = transform_mod


def ensure_robosuite_compat() -> None:
    """Install fallback module only when robosuite is unavailable."""
    try:
        import robosuite  # noqa: F401
    except Exception:
        _install_fallback_robosuite()
