"""Reusable helpers for board-adjacent distractor placement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class PrimitiveDistractorSpec:
    """Specification for a simple rigid distractor object."""

    primitive: str
    size: Tuple[float, ...]
    rgba: Tuple[float, float, float, float]
    density: float = 700.0
    friction: Tuple[float, float, float] = (1.0, 0.02, 0.01)
    solref: Tuple[float, float] = (0.01, 0.8)
    solimp: Tuple[float, float, float] = (0.998, 0.998, 0.001)

    def horizontal_radius(self) -> float:
        primitive = str(self.primitive).lower().strip()
        if primitive == "box":
            if len(self.size) != 3:
                raise ValueError("box distractor size must have 3 values")
            return float(np.linalg.norm(np.asarray(self.size[:2], dtype=np.float64), ord=2))
        if primitive in ("cylinder", "capsule"):
            if len(self.size) != 2:
                raise ValueError(f"{primitive} distractor size must have 2 values")
            return float(self.size[0])
        if primitive == "ball":
            if len(self.size) != 1:
                raise ValueError("ball distractor size must have 1 value")
            return float(self.size[0])
        raise ValueError(f"unsupported distractor primitive: {self.primitive}")

    def half_height(self) -> float:
        primitive = str(self.primitive).lower().strip()
        if primitive == "box":
            if len(self.size) != 3:
                raise ValueError("box distractor size must have 3 values")
            return float(self.size[2])
        if primitive == "cylinder":
            if len(self.size) != 2:
                raise ValueError("cylinder distractor size must have 2 values")
            return float(self.size[1])
        if primitive == "capsule":
            if len(self.size) != 2:
                raise ValueError("capsule distractor size must have 2 values")
            return float(self.size[0] + self.size[1])
        if primitive == "ball":
            if len(self.size) != 1:
                raise ValueError("ball distractor size must have 1 value")
            return float(self.size[0])
        raise ValueError(f"unsupported distractor primitive: {self.primitive}")


@dataclass(frozen=True)
class BoardAdjacentPlacementPolicy:
    """Samples XY placements around (but outside) a board footprint."""

    board_center_xy: Tuple[float, float]
    board_half_extent_xy: Tuple[float, float]
    table_low_xy: Tuple[float, float]
    table_high_xy: Tuple[float, float]
    forbidden_margin: float = 0.02
    ring_width: float = 0.14
    min_object_spacing: float = 0.01
    max_attempts_per_object: int = 500

    def _candidate_in_ring(self, candidate_xy: np.ndarray, radius: float) -> bool:
        center = np.asarray(self.board_center_xy, dtype=np.float64)
        half = np.asarray(self.board_half_extent_xy, dtype=np.float64)
        inner_half = half + float(self.forbidden_margin) + float(radius)
        outer_half = inner_half + float(self.ring_width)
        delta = np.abs(np.asarray(candidate_xy, dtype=np.float64) - center)
        inside_outer = bool(np.all(delta <= outer_half))
        inside_inner = bool(np.all(delta <= inner_half))
        return inside_outer and (not inside_inner)

    def _candidate_on_table(self, candidate_xy: np.ndarray, radius: float) -> bool:
        low = np.asarray(self.table_low_xy, dtype=np.float64) + float(radius)
        high = np.asarray(self.table_high_xy, dtype=np.float64) - float(radius)
        return bool(np.all(candidate_xy >= low) and np.all(candidate_xy <= high))

    def _candidate_far_from_placed(
        self,
        candidate_xy: np.ndarray,
        radius: float,
        placed_xy: np.ndarray,
        placed_radii: np.ndarray,
    ) -> bool:
        if placed_xy.shape[0] == 0:
            return True
        required = placed_radii + float(radius) + float(self.min_object_spacing)
        dists = np.linalg.norm(placed_xy - candidate_xy.reshape(1, 2), axis=1)
        return bool(np.all(dists >= required))

    def sample_xy_positions(self, rng: np.random.RandomState, radii: Sequence[float]) -> np.ndarray:
        radii = np.asarray([float(r) for r in radii], dtype=np.float64)
        if radii.ndim != 1:
            raise ValueError("radii must be a 1D sequence")
        if radii.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        if np.any(radii < 0.0):
            raise ValueError("radii values must be >= 0")

        sampled = np.zeros((radii.shape[0], 2), dtype=np.float64)
        order = np.argsort(-radii)
        placed_indices = []

        table_low = np.asarray(self.table_low_xy, dtype=np.float64)
        table_high = np.asarray(self.table_high_xy, dtype=np.float64)

        for idx in order:
            radius = float(radii[idx])
            low = table_low + radius
            high = table_high - radius
            if np.any(low >= high):
                raise RuntimeError("table bounds too small for distractor placement")

            accepted = False
            for _ in range(int(max(1, self.max_attempts_per_object))):
                candidate = rng.uniform(low=low, high=high).astype(np.float64)
                if not self._candidate_in_ring(candidate_xy=candidate, radius=radius):
                    continue
                prev = np.asarray(placed_indices, dtype=np.int32)
                prev_xy = sampled[prev] if prev.size > 0 else np.zeros((0, 2), dtype=np.float64)
                prev_r = radii[prev] if prev.size > 0 else np.zeros((0,), dtype=np.float64)
                if not self._candidate_on_table(candidate_xy=candidate, radius=radius):
                    continue
                if not self._candidate_far_from_placed(
                    candidate_xy=candidate,
                    radius=radius,
                    placed_xy=prev_xy,
                    placed_radii=prev_r,
                ):
                    continue
                sampled[idx] = candidate
                placed_indices.append(int(idx))
                accepted = True
                break

            if not accepted:
                raise RuntimeError(
                    "failed to sample distractor placement outside board after "
                    f"{self.max_attempts_per_object} attempts"
                )

        return sampled.astype(np.float32)
