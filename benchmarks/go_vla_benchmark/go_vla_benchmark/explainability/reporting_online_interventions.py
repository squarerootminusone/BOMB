"""PNG rendering for online task-level intervention reports."""

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


BG_COLOR = (245, 247, 251)
PANEL_BG = (255, 255, 255)
PANEL_BORDER = (213, 220, 231)
TEXT_PRIMARY = (27, 35, 48)
TEXT_MUTED = (94, 108, 127)
TARGET_COLOR = (34, 116, 214)
SOURCE_COLOR = (122, 128, 138)
STONE_COLOR = (54, 59, 70)
SUCCESS_COLOR = (66, 154, 97)
FAIL_COLOR = (196, 67, 64)
START_COLOR = (34, 34, 34)
_ISO_PROJECTION = np.asarray(
    [
        [0.86, -0.86, 0.0],
        [0.46, 0.46, -1.2],
    ],
    dtype=np.float32,
)


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Menlo.ttc",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
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
        points.append(np.asarray(attempt.target_xyz, dtype=np.float32).reshape(1, 3))
        points.append(np.asarray(attempt.source_xyz, dtype=np.float32).reshape(1, 3))
        if attempt.final_stone_xyz is not None:
            points.append(np.asarray(attempt.final_stone_xyz, dtype=np.float32).reshape(1, 3))
        if attempt.trajectory:
            points.append(
                np.asarray([np.asarray(step.eef_xyz, dtype=np.float32) for step in attempt.trajectory], dtype=np.float32)
            )
            stone_points = [
                np.asarray(step.stone_xyz, dtype=np.float32)
                for step in attempt.trajectory
                if step.stone_xyz is not None
            ]
            if stone_points:
                points.append(np.asarray(stone_points, dtype=np.float32))
    if not points:
        return np.zeros((1, 3), dtype=np.float32)
    return np.concatenate(points, axis=0)


def _projection_bounds(report: OnlineInterventionReport) -> tuple[np.ndarray, np.ndarray]:
    projected = _project_xyz(_all_report_points(report))
    minimum = projected.min(axis=0)
    maximum = projected.max(axis=0)
    span = np.maximum(maximum - minimum, 1e-4)
    return minimum, minimum + span


def _map_projected_point(
    xyz: np.ndarray,
    *,
    box: tuple[int, int, int, int],
    projection_min: np.ndarray,
    projection_max: np.ndarray,
    padding: int = 20,
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


def _draw_circle(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    radius: int,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int] | None = None,
    width: int = 1,
) -> None:
    x, y = center
    draw.ellipse(
        [(x - radius, y - radius), (x + radius, y + radius)],
        fill=fill,
        outline=outline,
        width=width,
    )


def _draw_cross(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
    width: int = 2,
) -> None:
    x, y = center
    draw.line([(x - radius, y), (x + radius, y)], fill=color, width=width)
    draw.line([(x, y - radius), (x, y + radius)], fill=color, width=width)


def _draw_panel(
    draw: ImageDraw.ImageDraw,
    *,
    panel_box: tuple[int, int, int, int],
    attempt: OnlineInterventionAttempt,
    projection_min: np.ndarray,
    projection_max: np.ndarray,
    title_font: ImageFont.ImageFont,
    text_font: ImageFont.ImageFont,
) -> None:
    left, top, right, bottom = panel_box
    draw.rounded_rectangle(panel_box, radius=18, fill=PANEL_BG, outline=PANEL_BORDER, width=2)

    status_color = SUCCESS_COLOR if attempt.success else FAIL_COLOR
    subtitle = f"{'success' if attempt.success else 'timeout'} in {attempt.steps_taken} steps"
    if attempt.mask is not None:
        subtitle = f"{subtitle} | mask: {attempt.mask.label}"

    draw.text((left + 18, top + 16), attempt.title, fill=TEXT_PRIMARY, font=title_font)
    draw.text((left + 18, top + 42), subtitle, fill=status_color if attempt.success else TEXT_MUTED, font=text_font)

    plot_box = (left + 14, top + 68, right - 14, bottom - 14)
    draw.rounded_rectangle(plot_box, radius=14, outline=PANEL_BORDER, width=1)

    source_point = _map_projected_point(
        attempt.source_xyz,
        box=plot_box,
        projection_min=projection_min,
        projection_max=projection_max,
    )
    target_point = _map_projected_point(
        attempt.target_xyz,
        box=plot_box,
        projection_min=projection_min,
        projection_max=projection_max,
    )
    _draw_circle(draw, source_point, radius=4, fill=SOURCE_COLOR)
    _draw_cross(draw, target_point, radius=7, color=TARGET_COLOR, width=2)

    if attempt.final_stone_xyz is not None:
        stone_point = _map_projected_point(
            attempt.final_stone_xyz,
            box=plot_box,
            projection_min=projection_min,
            projection_max=projection_max,
        )
        _draw_circle(draw, stone_point, radius=5, fill=STONE_COLOR, outline=(255, 255, 255), width=1)

    if len(attempt.trajectory) >= 2:
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
            color = TASK_PHASE_COLORS.get(step.phase, TEXT_MUTED)
            draw.line([prev_point, next_point], fill=color, width=4)

    if attempt.trajectory:
        start_point = _map_projected_point(
            attempt.trajectory[0].eef_xyz,
            box=plot_box,
            projection_min=projection_min,
            projection_max=projection_max,
        )
        end_point = _map_projected_point(
            attempt.trajectory[-1].eef_xyz,
            box=plot_box,
            projection_min=projection_min,
            projection_max=projection_max,
        )
        _draw_circle(draw, start_point, radius=4, fill=START_COLOR)
        _draw_circle(
            draw,
            end_point,
            radius=5,
            fill=SUCCESS_COLOR if attempt.success else FAIL_COLOR,
            outline=(255, 255, 255),
            width=1,
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

    margin = 28
    header_h = 92
    footer_h = 76
    panel_gap = 24
    baseline_w = 760
    attempt_w = 600
    attempt_gap = 14
    attempt_count = max(1, len(report.masked_attempts))
    attempt_h = 148 if attempt_count >= 4 else 176
    stack_h = (attempt_count * attempt_h) + ((attempt_count - 1) * attempt_gap)
    canvas_h = margin + header_h + stack_h + footer_h + margin
    canvas_w = margin + baseline_w + panel_gap + attempt_w + margin

    image = Image.new("RGB", (canvas_w, canvas_h), color=BG_COLOR)
    draw = ImageDraw.Draw(image)
    title_font = _load_font(28)
    panel_title_font = _load_font(22)
    text_font = _load_font(16)
    small_font = _load_font(14)

    projection_min, projection_max = _projection_bounds(report)

    header_y = margin
    draw.text((margin, header_y), "Online Task Intervention Report", fill=TEXT_PRIMARY, font=title_font)
    draw.text(
        (margin, header_y + 36),
        f"Target: row {report.target_row}, column {report.target_col} | env: {report.environment_name}",
        fill=TEXT_MUTED,
        font=text_font,
    )
    draw.text(
        (margin, header_y + 58),
        f"Masked text span: {report.text_mask.label}",
        fill=TEXT_MUTED,
        font=text_font,
    )

    panels_top = margin + header_h
    baseline_box = (margin, panels_top, margin + baseline_w, panels_top + stack_h)
    _draw_panel(
        draw,
        panel_box=baseline_box,
        attempt=report.baseline,
        projection_min=projection_min,
        projection_max=projection_max,
        title_font=panel_title_font,
        text_font=text_font,
    )

    right_left = margin + baseline_w + panel_gap
    for attempt_index, attempt in enumerate(report.masked_attempts):
        top = panels_top + (attempt_index * (attempt_h + attempt_gap))
        box = (right_left, top, right_left + attempt_w, top + attempt_h)
        _draw_panel(
            draw,
            panel_box=box,
            attempt=attempt,
            projection_min=projection_min,
            projection_max=projection_max,
            title_font=_load_font(18),
            text_font=small_font,
        )

    legend_y = canvas_h - footer_h + 12
    legend_x = margin
    for label, color in _legend_items():
        draw.rounded_rectangle(
            (legend_x, legend_y + 4, legend_x + 20, legend_y + 20),
            radius=4,
            fill=color,
        )
        draw.text((legend_x + 28, legend_y), label, fill=TEXT_PRIMARY, font=text_font)
        legend_x += 190

    image.save(output_path)
    return output_path
