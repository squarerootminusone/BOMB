"""Shared types for Go benchmark environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class GoResetOptions:
    opening_moves: int = 0
    target_row: Optional[int] = None
    target_col: Optional[int] = None
    stone_color: Optional[str] = None  # "black", "white", or None (random)
