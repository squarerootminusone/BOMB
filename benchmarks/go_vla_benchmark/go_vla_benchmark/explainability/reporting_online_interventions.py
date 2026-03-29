"""Publication-style PNG rendering for online task-level intervention reports."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .online_interventions import (
    OnlineInterventionAttempt,
    OnlineInterventionReport,
    TASK_PHASE_COLORS,
    TASK_PHASE_LABELS,
    TASK_PHASE_ORDER,
)


BG_COLOR = (248, 249, 251, 255)
PANEL_BG = (255, 255, 255, 255)
PANEL_BORDER = (212, 218, 226, 255)
LEGEND_TEXT = (32, 40, 52, 255)
_ISO_PROJECTION = np.asarray(
    [
        [0.86, -0.86, 0.0],
        [0.46, 0.46, -1.2],
    ],
    dtype=np.float32,
)


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Menlo.ttc",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _project_xyz(xyz: np.ndarray) -> np.ndarray:
    points = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    return points @ _ISO_PROJECTION.T


def _all_report_points(report: OnlineInterventionReport) -> np.ndarray:
    points: list[np.ndarray] = []
    for attempt in [report.baseline, *report.masked_attempts]:
        if attempt.trajectory:
            points.append(
                np.asarray([np.asarray(step.eef_xyz, dtype=np.float32) for step in attempt.trajectory], dtype=np.float32)
            )
    if not points:
        return np.zeros((1, 3), dtype=np.float32)
    return np.concatenate(points, axis=0)


def _projection_bounds(report: OnlineInterventionReport) -> tuple[np.ndarray, np.ndarray]:
    projected = _project_xyz(_all_report_points(report))
    minimum = projected.min(axis=0)
    maximum = projected.max(axis=0)
    span = np.maximum(maximum - minimum, 1e-4)
    pad = span * 0.06
    return minimum - pad, maximum + pad


def _map_projected_point(
    xyz: np.ndarray,
    *,
    box: tuple[int, int, int, int],
    projection_min: np.ndarray,
    projection_max: np.ndarray,
    padding: int = 28,
) -> tuple[int, int]:
    left, top, right, bottom = box
    usable_w = max(1, right - left - (2 * padding))
    usable_h = max(1, bottom - top - (2 * padding))
    projected = _project_xyz(np.asarray(xyz, dtype=np.float32).reshape(1, 3))[0]
    span = np.maximum(projection_max - projection_min, 1e-4)
    scale = min(usable_w / span[0], usable_h / span[1])
    draw_w = span[0] * scale
    draw_h = span[1] * scale
    offset_x = left + padding + ((usable_w - draw_w) * 0.5)
    offset_y = top + padding + ((usable_h - draw_h) * 0.5)
    x = offset_x + ((projected[0] - projection_min[0]) * scale)
    y = offset_y + draw_h - ((projected[1] - projection_min[1]) * scale)
    return int(round(x)), int(round(y))


def _line_color(phase: str, alpha: int) -> tuple[int, int, int, int]:
    rgb = TASK_PHASE_COLORS.get(phase, (120, 125, 132))
    return (int(rgb[0]), int(rgb[1]), int(rgb[2]), int(alpha))


def _draw_attempt_trajectory(
    draw: ImageDraw.ImageDraw,
    *,
    attempt: OnlineInterventionAttempt,
    plot_box: tuple[int, int, int, int],
    projection_min: np.ndarray,
    projection_max: np.ndarray,
    alpha: int,
    width: int,
) -> None:
    if len(attempt.trajectory) < 2:
        return
    for prev_step, step in zip(attempt.trajectory[:-1], attempt.trajectory[1:]):
        prev_point = _map_projected_point(
            prev_step.eef_xyz,
            box=plot_box,
            projection_min=projection_min,
            projection_max=projection_max,
        )
        next_point = _map_projected_point(
            step.eef_xyz,
            box=plot_box,
            projection_min=projection_min,
            projection_max=projection_max,
        )
        draw.line([prev_point, next_point], fill=_line_color(step.phase, alpha=alpha), width=width)


def _draw_plot_panel(
    draw: ImageDraw.ImageDraw,
    *,
    panel_box: tuple[int, int, int, int],
    attempts: Sequence[OnlineInterventionAttempt],
    projection_min: np.ndarray,
    projection_max: np.ndarray,
    alpha: int,
    width: int,
) -> None:
    draw.rounded_rectangle(panel_box, radius=24, fill=PANEL_BG, outline=PANEL_BORDER, width=2)
    plot_box = (
        panel_box[0] + 8,
        panel_box[1] + 8,
        panel_box[2] - 8,
        panel_box[3] - 8,
    )
    for attempt in attempts:
        _draw_attempt_trajectory(
            draw,
            attempt=attempt,
            plot_box=plot_box,
            projection_min=projection_min,
            projection_max=projection_max,
            alpha=alpha,
            width=width,
        )


def _legend_items() -> Iterable[tuple[str, tuple[int, int, int]]]:
    for phase in TASK_PHASE_ORDER:
        yield TASK_PHASE_LABELS[phase], TASK_PHASE_COLORS[phase]


def export_online_intervention_report_png(
    report: OnlineInterventionReport,
    output_path: Path,
) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    canvas_w = 2200
    canvas_h = 1200
    margin_x = 54
    margin_top = 50
    margin_bottom = 46
    panel_gap = 40
    legend_h = 132
    panel_h = canvas_h - margin_top - margin_bottom - legend_h
    panel_w = int((canvas_w - (2 * margin_x) - panel_gap) / 2)

    left_panel = (
        margin_x,
        margin_top,
        margin_x + panel_w,
        margin_top + panel_h,
    )
    right_panel = (
        left_panel[2] + panel_gap,
        margin_top,
        left_panel[2] + panel_gap + panel_w,
        margin_top + panel_h,
    )

    image = Image.new("RGBA", (canvas_w, canvas_h), color=BG_COLOR)
    draw = ImageDraw.Draw(image, "RGBA")
    projection_min, projection_max = _projection_bounds(report)

    _draw_plot_panel(
        draw,
        panel_box=left_panel,
        attempts=[report.baseline],
        projection_min=projection_min,
        projection_max=projection_max,
        alpha=240,
        width=7,
    )
    _draw_plot_panel(
        draw,
        panel_box=right_panel,
        attempts=report.masked_attempts,
        projection_min=projection_min,
        projection_max=projection_max,
        alpha=112,
        width=6,
    )

    legend_font = _load_font(32)
    legend_y = canvas_h - legend_h + 34
    legend_x = margin_x + 10
    swatch_w = 78
    swatch_gap = 26
    item_gap = 92
    for label, color in _legend_items():
        draw.line(
            [(legend_x, legend_y + 18), (legend_x + swatch_w, legend_y + 18)],
            fill=(int(color[0]), int(color[1]), int(color[2]), 255),
            width=10,
        )
        draw.text((legend_x + swatch_w + swatch_gap, legend_y), label, fill=LEGEND_TEXT, font=legend_font)
        text_w, _ = _text_size(draw, label, font=legend_font)
        legend_x += swatch_w + swatch_gap + text_w + item_gap

    image.convert("RGB").save(output_path)
    return output_path
