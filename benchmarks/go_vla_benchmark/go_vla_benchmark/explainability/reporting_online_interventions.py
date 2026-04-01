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
LABEL_BG = (255, 255, 255, 244)
MARKER_OUTLINE = (28, 35, 44, 236)
MARKER_FILL = (255, 255, 255, 244)
MARKER_SOLID = (28, 35, 44, 236)
RELEASE_MARKER_OUTLINE = (150, 55, 52, 236)
PHASE_MARKER_OUTLINE = (255, 255, 255, 244)
START_MARKER_FILL = (255, 255, 255, 248)
START_MARKER_OUTLINE = (28, 35, 44, 232)
GRIPPER_CLOSE_THRESHOLD = 0.25
MIN_TRAJECTORY_OPACITY = 0.7
ATTEMPT_ACCENT_COLORS = (
    (37, 99, 235, 255),
    (219, 39, 119, 255),
    (14, 165, 233, 255),
    (234, 88, 12, 255),
    (22, 163, 74, 255),
)
MASK_LEGEND_HEADER = (62, 74, 89, 255)


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


def _project_xy(xyz: np.ndarray) -> np.ndarray:
    points = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    return points[:, :2].copy()


def _apply_xy_offset(xyz: np.ndarray, *, xy_offset: np.ndarray | None = None) -> np.ndarray:
    point = np.asarray(xyz, dtype=np.float32).reshape(-1)[:3].copy()
    if xy_offset is not None:
        point[:2] += np.asarray(xy_offset, dtype=np.float32).reshape(-1)[:2]
    return point


def _projection_extents_for_attempts(
    attempts: Sequence[OnlineInterventionAttempt],
    *,
    attempt_xy_offsets: Sequence[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if not attempts:
        zero = np.zeros((2,), dtype=np.float32)
        return zero, zero.copy()

    if attempt_xy_offsets is None:
        offsets = [np.zeros((2,), dtype=np.float32) for _ in attempts]
    else:
        offsets = [
            np.asarray(item, dtype=np.float32).reshape(-1)[:2].copy()
            for item in attempt_xy_offsets[: len(attempts)]
        ]
        if len(offsets) < len(attempts):
            offsets.extend(np.zeros((2,), dtype=np.float32) for _ in range(len(attempts) - len(offsets)))

    points: list[np.ndarray] = []
    for attempt, xy_offset in zip(attempts, offsets):
        if not attempt.trajectory:
            continue
        points.append(
            np.asarray(
                [_apply_xy_offset(step.eef_xyz, xy_offset=xy_offset) for step in attempt.trajectory],
                dtype=np.float32,
            )
        )
    if not points:
        zero = np.zeros((2,), dtype=np.float32)
        return zero, zero.copy()

    projected = _project_xy(np.concatenate(points, axis=0))
    return projected.min(axis=0), projected.max(axis=0)


def _projection_bounds_from_extents(
    minimum: np.ndarray,
    maximum: np.ndarray,
    *,
    span_override: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    minimum = np.asarray(minimum, dtype=np.float32).reshape(-1)[:2]
    maximum = np.asarray(maximum, dtype=np.float32).reshape(-1)[:2]
    center = 0.5 * (minimum + maximum)
    span = np.maximum(maximum - minimum, 1e-4)
    if span_override is not None:
        span = np.maximum(np.asarray(span_override, dtype=np.float32).reshape(-1)[:2], 1e-4)
    pad = span * 0.06
    half = 0.5 * span
    return center - half - pad, center + half + pad


def _masked_attempt_xy_offsets(report: OnlineInterventionReport) -> list[np.ndarray]:
    if not report.masked_attempts:
        return []
    if report.baseline.trajectory:
        anchor_xy = _project_xy(np.asarray(report.baseline.trajectory[0].eef_xyz, dtype=np.float32).reshape(1, 3))[0]
    else:
        anchor_xy = None

    offsets: list[np.ndarray] = []
    for attempt in report.masked_attempts:
        if not attempt.trajectory:
            offsets.append(np.zeros((2,), dtype=np.float32))
            continue
        start_xy = _project_xy(np.asarray(attempt.trajectory[0].eef_xyz, dtype=np.float32).reshape(1, 3))[0]
        if anchor_xy is None:
            anchor_xy = start_xy.copy()
        offsets.append(np.asarray(anchor_xy - start_xy, dtype=np.float32))
    return offsets


def _height_bounds(report: OnlineInterventionReport) -> tuple[float, float]:
    z_values: list[np.ndarray] = []
    for attempt in [report.baseline, *report.masked_attempts]:
        if attempt.trajectory:
            z_values.append(
                np.asarray([float(np.asarray(step.eef_xyz, dtype=np.float32)[2]) for step in attempt.trajectory], dtype=np.float32)
            )
        z_values.append(np.asarray([float(np.asarray(attempt.source_xyz, dtype=np.float32)[2])], dtype=np.float32))
        z_values.append(np.asarray([float(np.asarray(attempt.target_xyz, dtype=np.float32)[2])], dtype=np.float32))
        if attempt.final_stone_xyz is not None:
            z_values.append(np.asarray([float(np.asarray(attempt.final_stone_xyz, dtype=np.float32)[2])], dtype=np.float32))
    stacked = np.concatenate(z_values, axis=0) if z_values else np.asarray([0.0], dtype=np.float32)
    return float(stacked.min()), float(stacked.max())


def _height_alpha(z: float, *, z_min: float, z_max: float, alpha: int) -> int:
    span = max(float(z_max) - float(z_min), 1e-4)
    normalized = float(np.clip((float(z) - float(z_min)) / span, 0.0, 1.0))
    opacity = 1.0 - ((1.0 - MIN_TRAJECTORY_OPACITY) * normalized)
    return int(round(float(alpha) * opacity))


def _time_alpha(step_index: int, *, step_count: int, alpha: int) -> int:
    if int(step_count) <= 1:
        return int(alpha)
    normalized = float(np.clip(float(step_index) / float(max(1, step_count - 1)), 0.0, 1.0))
    opacity = MIN_TRAJECTORY_OPACITY + ((1.0 - MIN_TRAJECTORY_OPACITY) * normalized)
    return int(round(float(alpha) * opacity))


def _resolve_alpha(
    *,
    alpha_mode: str,
    alpha: int,
    z_value: float,
    z_min: float,
    z_max: float,
    step_index: int,
    step_count: int,
) -> int:
    if alpha_mode == "time":
        return _time_alpha(step_index, step_count=step_count, alpha=alpha)
    return _height_alpha(z_value, z_min=z_min, z_max=z_max, alpha=alpha)


def _map_projected_point(
    xyz: np.ndarray,
    *,
    box: tuple[int, int, int, int],
    projection_min: np.ndarray,
    projection_max: np.ndarray,
    padding: int = 28,
    xy_offset: np.ndarray | None = None,
) -> tuple[int, int]:
    left, top, right, bottom = box
    usable_w = max(1, right - left - (2 * padding))
    usable_h = max(1, bottom - top - (2 * padding))
    projected = _project_xy(np.asarray(xyz, dtype=np.float32).reshape(1, 3))[0]
    if xy_offset is not None:
        projected = projected + np.asarray(xy_offset, dtype=np.float32).reshape(-1)[:2]
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


def _accent_color(color: Sequence[int], alpha: int) -> tuple[int, int, int, int]:
    rgb = tuple(int(value) for value in color[:3])
    return (rgb[0], rgb[1], rgb[2], int(alpha))


def _marker_events(
    attempt: OnlineInterventionAttempt,
) -> tuple[list[tuple[np.ndarray, int]], list[tuple[np.ndarray, int]], list[tuple[np.ndarray, int]]]:
    pickup_attempts: list[tuple[np.ndarray, int]] = []
    grasped_points: list[tuple[np.ndarray, int]] = []
    releases: list[tuple[np.ndarray, int]] = []
    previous_step = None
    for step_idx, step in enumerate(attempt.trajectory):
        if float(step.gripper_action) > GRIPPER_CLOSE_THRESHOLD and not bool(step.stone_grasped):
            pickup_attempts.append((np.asarray(step.eef_xyz, dtype=np.float32), int(step_idx)))
        if bool(step.stone_grasped):
            grasped_points.append((np.asarray(step.eef_xyz, dtype=np.float32), int(step_idx)))
        release_now = False
        if previous_step is not None:
            release_now = (bool(previous_step.stone_grasped) and not bool(step.stone_grasped)) or (
                bool(step.move_committed) and not bool(previous_step.move_committed)
            )
        elif bool(step.move_committed):
            release_now = True
        if release_now:
            releases.append((np.asarray(step.eef_xyz, dtype=np.float32), int(step_idx)))
        previous_step = step
    return pickup_attempts, grasped_points, releases


def _draw_circle_marker(
    draw: ImageDraw.ImageDraw,
    *,
    center: tuple[int, int],
    radius: int,
    outline: tuple[int, int, int, int],
    fill: tuple[int, int, int, int],
    width: int,
) -> None:
    draw.ellipse(
        (
            center[0] - radius,
            center[1] - radius,
            center[0] + radius,
            center[1] + radius,
        ),
        outline=outline,
        fill=fill,
        width=width,
    )


def _draw_diamond_marker(
    draw: ImageDraw.ImageDraw,
    *,
    center: tuple[int, int],
    radius: int,
    outline: tuple[int, int, int, int],
    fill: tuple[int, int, int, int],
    width: int,
) -> None:
    points = [
        (center[0], center[1] - radius),
        (center[0] + radius, center[1]),
        (center[0], center[1] + radius),
        (center[0] - radius, center[1]),
    ]
    draw.polygon(points, fill=fill)
    draw.line(points + [points[0]], fill=outline, width=width)


def _draw_square_marker(
    draw: ImageDraw.ImageDraw,
    *,
    center: tuple[int, int],
    radius: int,
    outline: tuple[int, int, int, int],
    fill: tuple[int, int, int, int],
    width: int,
) -> None:
    draw.rectangle(
        (
            center[0] - radius,
            center[1] - radius,
            center[0] + radius,
            center[1] + radius,
        ),
        outline=outline,
        fill=fill,
        width=width,
    )


def _phase_change_markers(attempt: OnlineInterventionAttempt) -> list[tuple[np.ndarray, int, str]]:
    markers: list[tuple[np.ndarray, int, str]] = []
    previous_phase = None
    for step_idx, step in enumerate(attempt.trajectory):
        phase = str(step.phase)
        if previous_phase != phase:
            markers.append((np.asarray(step.eef_xyz, dtype=np.float32), int(step_idx), phase))
            previous_phase = phase
    return markers


def _draw_phase_change_markers(
    draw: ImageDraw.ImageDraw,
    *,
    attempt: OnlineInterventionAttempt,
    plot_box: tuple[int, int, int, int],
    projection_min: np.ndarray,
    projection_max: np.ndarray,
    z_min: float,
    z_max: float,
    alpha: int,
    alpha_mode: str,
    xy_offset: np.ndarray | None = None,
) -> None:
    step_count = max(1, len(attempt.trajectory))
    minimum_alpha = min(255, max(96, int(round(float(alpha) * 0.75))))
    for xyz, step_idx, phase in _phase_change_markers(attempt):
        marker_alpha = max(
            minimum_alpha,
            _resolve_alpha(
                alpha_mode=alpha_mode,
                alpha=alpha,
                z_value=float(np.asarray(xyz, dtype=np.float32)[2]),
                z_min=z_min,
                z_max=z_max,
                step_index=int(step_idx),
                step_count=step_count,
            ),
        )
        _draw_square_marker(
            draw,
            center=_map_projected_point(
                xyz,
                box=plot_box,
                projection_min=projection_min,
                projection_max=projection_max,
                xy_offset=xy_offset,
            ),
            radius=8,
            outline=(
                PHASE_MARKER_OUTLINE[0],
                PHASE_MARKER_OUTLINE[1],
                PHASE_MARKER_OUTLINE[2],
                min(255, marker_alpha + 28),
            ),
            fill=_line_color(phase, alpha=marker_alpha),
            width=2,
        )


def _draw_attempt_markers(
    draw: ImageDraw.ImageDraw,
    *,
    attempt: OnlineInterventionAttempt,
    plot_box: tuple[int, int, int, int],
    projection_min: np.ndarray,
    projection_max: np.ndarray,
    z_min: float,
    z_max: float,
    alpha: int,
    alpha_mode: str,
    xy_offset: np.ndarray | None = None,
) -> None:
    pickup_attempts, grasped_points, releases = _marker_events(attempt)
    step_count = max(1, len(attempt.trajectory))

    for xyz, step_idx in pickup_attempts:
        marker_alpha = _resolve_alpha(
            alpha_mode=alpha_mode,
            alpha=alpha,
            z_value=float(np.asarray(xyz, dtype=np.float32)[2]),
            z_min=z_min,
            z_max=z_max,
            step_index=int(step_idx),
            step_count=step_count,
        )
        attempt_outline = (
            MARKER_OUTLINE[0],
            MARKER_OUTLINE[1],
            MARKER_OUTLINE[2],
            marker_alpha,
        )
        attempt_fill = (
            MARKER_FILL[0],
            MARKER_FILL[1],
            MARKER_FILL[2],
            min(255, marker_alpha + 20),
        )
        _draw_circle_marker(
            draw,
            center=_map_projected_point(
                xyz,
                box=plot_box,
                projection_min=projection_min,
                projection_max=projection_max,
                xy_offset=xy_offset,
            ),
            radius=8,
            outline=attempt_outline,
            fill=attempt_fill,
            width=2,
        )
    for xyz, step_idx in grasped_points:
        marker_alpha = _resolve_alpha(
            alpha_mode=alpha_mode,
            alpha=alpha,
            z_value=float(np.asarray(xyz, dtype=np.float32)[2]),
            z_min=z_min,
            z_max=z_max,
            step_index=int(step_idx),
            step_count=step_count,
        )
        solid_fill = (
            MARKER_SOLID[0],
            MARKER_SOLID[1],
            MARKER_SOLID[2],
            marker_alpha,
        )
        _draw_circle_marker(
            draw,
            center=_map_projected_point(
                xyz,
                box=plot_box,
                projection_min=projection_min,
                projection_max=projection_max,
                xy_offset=xy_offset,
            ),
            radius=6,
            outline=solid_fill,
            fill=solid_fill,
            width=1,
        )
    for xyz, step_idx in releases:
        marker_alpha = _resolve_alpha(
            alpha_mode=alpha_mode,
            alpha=alpha,
            z_value=float(np.asarray(xyz, dtype=np.float32)[2]),
            z_min=z_min,
            z_max=z_max,
            step_index=int(step_idx),
            step_count=step_count,
        )
        attempt_fill = (
            MARKER_FILL[0],
            MARKER_FILL[1],
            MARKER_FILL[2],
            min(255, marker_alpha + 20),
        )
        release_outline = (
            RELEASE_MARKER_OUTLINE[0],
            RELEASE_MARKER_OUTLINE[1],
            RELEASE_MARKER_OUTLINE[2],
            marker_alpha,
        )
        _draw_diamond_marker(
            draw,
            center=_map_projected_point(
                xyz,
                box=plot_box,
                projection_min=projection_min,
                projection_max=projection_max,
                xy_offset=xy_offset,
            ),
            radius=9,
            outline=release_outline,
            fill=attempt_fill,
            width=2,
        )


def _draw_attempt_trajectory(
    draw: ImageDraw.ImageDraw,
    *,
    attempt: OnlineInterventionAttempt,
    plot_box: tuple[int, int, int, int],
    projection_min: np.ndarray,
    projection_max: np.ndarray,
    z_min: float,
    z_max: float,
    alpha: int,
    width: int,
    alpha_mode: str,
    xy_offset: np.ndarray | None = None,
    trajectory_color: Sequence[int] | None = None,
) -> None:
    if len(attempt.trajectory) < 2:
        return
    segment_count = max(1, len(attempt.trajectory) - 1)
    for segment_idx, (prev_step, step) in enumerate(zip(attempt.trajectory[:-1], attempt.trajectory[1:]), start=1):
        prev_point = _map_projected_point(
            prev_step.eef_xyz,
            box=plot_box,
            projection_min=projection_min,
            projection_max=projection_max,
            xy_offset=xy_offset,
        )
        next_point = _map_projected_point(
            step.eef_xyz,
            box=plot_box,
            projection_min=projection_min,
            projection_max=projection_max,
            xy_offset=xy_offset,
        )
        segment_z = 0.5 * (
            float(np.asarray(prev_step.eef_xyz, dtype=np.float32)[2]) + float(np.asarray(step.eef_xyz, dtype=np.float32)[2])
        )
        segment_alpha = _resolve_alpha(
            alpha_mode=alpha_mode,
            alpha=alpha,
            z_value=segment_z,
            z_min=z_min,
            z_max=z_max,
            step_index=int(segment_idx),
            step_count=segment_count + 1,
        )
        draw.line(
            [prev_point, next_point],
            fill=(
                _line_color(step.phase, alpha=segment_alpha)
                if trajectory_color is None
                else _accent_color(trajectory_color, alpha=segment_alpha)
            ),
            width=width,
        )


def _attempt_accent_color(attempt_index: int) -> tuple[int, int, int, int]:
    return ATTEMPT_ACCENT_COLORS[max(0, int(attempt_index) - 1) % len(ATTEMPT_ACCENT_COLORS)]


def _fit_label_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    font: ImageFont.ImageFont,
    max_width: int,
) -> str:
    if _text_size(draw, text, font=font)[0] <= int(max_width):
        return text
    stripped = text.strip()
    while len(stripped) > 4:
        stripped = stripped[:-1].rstrip()
        candidate = f"{stripped}..."
        if _text_size(draw, candidate, font=font)[0] <= int(max_width):
            return candidate
    return "..."


def _attempt_label_text(attempt: OnlineInterventionAttempt) -> str:
    mask_label = attempt.title if attempt.mask is None else str(attempt.mask.label)
    return f"{int(attempt.attempt_index)}. {mask_label}"


def _attempt_legend_swatch_colors(
    attempt: OnlineInterventionAttempt,
) -> tuple[tuple[int, int, int, int], ...]:
    if attempt.mask is None:
        return tuple(
            (int(color[0]), int(color[1]), int(color[2]), 255)
            for color in (TASK_PHASE_COLORS[phase] for phase in TASK_PHASE_ORDER)
        )
    return (_attempt_accent_color(attempt.attempt_index),)


def _draw_attempt_legend_swatch(
    image: Image.Image,
    *,
    box: tuple[int, int, int, int],
    colors: Sequence[Sequence[int]],
) -> None:
    left, top, right, bottom = box
    swatch_w = max(1, int(right) - int(left))
    swatch_h = max(1, int(bottom) - int(top))
    radius = min(5, max(1, swatch_h // 2))
    swatch = Image.new("RGBA", (swatch_w, swatch_h), color=(0, 0, 0, 0))
    swatch_draw = ImageDraw.Draw(swatch, "RGBA")
    resolved_colors = [
        tuple(int(value) for value in color[:3]) + (255,)
        for color in colors
    ]
    segment_edges = [int(round((idx * swatch_w) / max(1, len(resolved_colors)))) for idx in range(len(resolved_colors) + 1)]
    for color, segment_left, segment_right in zip(resolved_colors, segment_edges[:-1], segment_edges[1:]):
        swatch_draw.rectangle(
            (
                int(segment_left),
                0,
                max(int(segment_left), int(segment_right) - 1),
                swatch_h - 1,
            ),
            fill=color,
        )
    swatch_mask = Image.new("L", (swatch_w, swatch_h), color=0)
    mask_draw = ImageDraw.Draw(swatch_mask)
    mask_draw.rounded_rectangle((0, 0, swatch_w - 1, swatch_h - 1), radius=radius, fill=255)
    swatch.putalpha(swatch_mask)
    swatch_draw.rounded_rectangle(
        (0, 0, swatch_w - 1, swatch_h - 1),
        radius=radius,
        outline=(255, 255, 255, 255),
        width=1,
    )
    image.alpha_composite(swatch, dest=(int(left), int(top)))


def _draw_attempt_legend(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    *,
    attempts: Sequence[OnlineInterventionAttempt],
    box: tuple[int, int, int, int],
) -> None:
    if not attempts:
        return

    header_font = _load_font(28)
    item_font = _load_font(24)
    left, top, right, bottom = box
    draw.text((left, top), "Attempts", fill=MASK_LEGEND_HEADER, font=header_font)
    _header_w, header_h = _text_size(draw, "Attempts", font=header_font)

    item_top = top + header_h + 16
    item_height = 34
    row_gap = 16
    x = left
    y = item_top
    max_item_width = min(520, max(220, right - left))

    for attempt in attempts:
        swatch_colors = _attempt_legend_swatch_colors(attempt)
        label = _fit_label_text(
            draw,
            _attempt_label_text(attempt),
            font=item_font,
            max_width=max_item_width - 42,
        )
        label_w, label_h = _text_size(draw, label, font=item_font)
        item_width = min(max_item_width, label_w + 42)

        if (x + item_width) > right and x > left:
            x = left
            y += item_height + row_gap
        if (y + item_height) > bottom:
            break

        swatch_y = y + int(round((item_height - 10) * 0.5))
        _draw_attempt_legend_swatch(
            image,
            box=(x, swatch_y, x + 24, swatch_y + 10),
            colors=swatch_colors,
        )
        draw.text(
            (x + 34, y + int(round((item_height - label_h) * 0.5)) - 1),
            label,
            fill=LEGEND_TEXT,
            font=item_font,
        )
        x += item_width + 28


def _draw_attempt_labels(
    draw: ImageDraw.ImageDraw,
    *,
    attempts: Sequence[OnlineInterventionAttempt],
    plot_box: tuple[int, int, int, int],
    projection_min: np.ndarray,
    projection_max: np.ndarray,
    attempt_xy_offsets: Sequence[np.ndarray],
    label_font: ImageFont.ImageFont,
) -> None:
    entries: list[dict[str, object]] = []
    for attempt, xy_offset in zip(attempts, attempt_xy_offsets):
        if not attempt.trajectory:
            continue
        endpoint = _map_projected_point(
            attempt.trajectory[-1].eef_xyz,
            box=plot_box,
            projection_min=projection_min,
            projection_max=projection_max,
            xy_offset=xy_offset,
        )
        entries.append(
            {
                "attempt": attempt,
                "endpoint": endpoint,
                "accent": _attempt_accent_color(attempt.attempt_index),
                "label": _attempt_label_text(attempt),
            }
        )
    if not entries:
        return

    sorted_entries = sorted(entries, key=lambda item: int(item["endpoint"][1]))
    slot_top = plot_box[1] + 28
    slot_bottom = plot_box[3] - 28
    max_box_width = max(180, int((plot_box[2] - plot_box[0]) * 0.42))
    for slot_idx, entry in enumerate(sorted_entries):
        if len(sorted_entries) == 1:
            slot_y = int(round((slot_top + slot_bottom) * 0.5))
        else:
            slot_y = int(
                round(
                    slot_top + (slot_idx * (slot_bottom - slot_top) / float(max(1, len(sorted_entries) - 1)))
                )
            )
        endpoint = tuple(int(value) for value in entry["endpoint"])
        accent = tuple(int(value) for value in entry["accent"])
        label = _fit_label_text(draw, str(entry["label"]), font=label_font, max_width=max_box_width - 24)
        text_w, text_h = _text_size(draw, label, font=label_font)
        box_w = min(max_box_width, text_w + 24)
        box_h = text_h + 16
        box_right = plot_box[2] - 18
        box_left = max(plot_box[0] + 20, box_right - box_w)
        box_top = int(np.clip(slot_y - (box_h * 0.5), plot_box[1] + 12, plot_box[3] - box_h - 12))
        box_bottom = box_top + box_h
        connector_mid_x = int(round((endpoint[0] + box_left - 12) * 0.5))
        connector_end = (box_left - 10, int(round((box_top + box_bottom) * 0.5)))

        draw.line([endpoint, (connector_mid_x, endpoint[1]), connector_end], fill=accent, width=3)
        draw.ellipse(
            (
                endpoint[0] - 7,
                endpoint[1] - 7,
                endpoint[0] + 7,
                endpoint[1] + 7,
            ),
            fill=accent,
            outline=(255, 255, 255, 255),
            width=2,
        )
        draw.rounded_rectangle(
            (box_left, box_top, box_right, box_bottom),
            radius=14,
            fill=LABEL_BG,
            outline=accent,
            width=2,
        )
        draw.text((box_left + 12, box_top + 8), label, fill=LEGEND_TEXT, font=label_font)


def _draw_aligned_start_marker(
    draw: ImageDraw.ImageDraw,
    *,
    point: tuple[int, int],
) -> None:
    radius = 9
    draw.ellipse(
        (
            point[0] - radius,
            point[1] - radius,
            point[0] + radius,
            point[1] + radius,
        ),
        fill=START_MARKER_FILL,
        outline=START_MARKER_OUTLINE,
        width=2,
    )
    draw.line([(point[0] - 6, point[1]), (point[0] + 6, point[1])], fill=START_MARKER_OUTLINE, width=2)
    draw.line([(point[0], point[1] - 6), (point[0], point[1] + 6)], fill=START_MARKER_OUTLINE, width=2)


def _draw_plot_panel(
    image: Image.Image,
    *,
    panel_box: tuple[int, int, int, int],
    attempts: Sequence[OnlineInterventionAttempt],
    projection_min: np.ndarray,
    projection_max: np.ndarray,
    z_min: float,
    z_max: float,
    alpha: int,
    width: int,
    alpha_mode: str,
    attempt_xy_offsets: Sequence[np.ndarray] | None = None,
    show_attempt_labels: bool = False,
    show_aligned_start_marker: bool = False,
    trajectory_colors: Sequence[Sequence[int] | None] | None = None,
    show_phase_change_markers: bool = False,
) -> Image.Image:
    base_draw = ImageDraw.Draw(image, "RGBA")
    base_draw.rounded_rectangle(panel_box, radius=24, fill=PANEL_BG, outline=PANEL_BORDER, width=2)
    plot_box = (
        panel_box[0] + 8,
        panel_box[1] + 8,
        panel_box[2] - 8,
        panel_box[3] - 8,
    )
    if attempt_xy_offsets is None:
        xy_offsets = [np.zeros((2,), dtype=np.float32) for _ in attempts]
    else:
        xy_offsets = [
            np.asarray(item, dtype=np.float32).reshape(-1)[:2].copy()
            for item in attempt_xy_offsets[: len(attempts)]
        ]
        if len(xy_offsets) < len(attempts):
            xy_offsets.extend(np.zeros((2,), dtype=np.float32) for _ in range(len(attempts) - len(xy_offsets)))
    if trajectory_colors is None:
        line_colors = [None for _ in attempts]
    else:
        line_colors = list(trajectory_colors[: len(attempts)])
        if len(line_colors) < len(attempts):
            line_colors.extend([None for _ in range(len(attempts) - len(line_colors))])

    for attempt, xy_offset, line_color in zip(attempts, xy_offsets, line_colors):
        attempt_layer = Image.new("RGBA", image.size, color=(0, 0, 0, 0))
        attempt_draw = ImageDraw.Draw(attempt_layer, "RGBA")
        _draw_attempt_trajectory(
            attempt_draw,
            attempt=attempt,
            plot_box=plot_box,
            projection_min=projection_min,
            projection_max=projection_max,
            z_min=z_min,
            z_max=z_max,
            alpha=alpha,
            width=width,
            alpha_mode=alpha_mode,
            xy_offset=xy_offset,
            trajectory_color=line_color,
        )
        if show_phase_change_markers:
            _draw_phase_change_markers(
                attempt_draw,
                attempt=attempt,
                plot_box=plot_box,
                projection_min=projection_min,
                projection_max=projection_max,
                z_min=z_min,
                z_max=z_max,
                alpha=min(255, alpha + 48),
                alpha_mode=alpha_mode,
                xy_offset=xy_offset,
            )
        _draw_attempt_markers(
            attempt_draw,
            attempt=attempt,
            plot_box=plot_box,
            projection_min=projection_min,
            projection_max=projection_max,
            z_min=z_min,
            z_max=z_max,
            alpha=min(255, alpha + 36),
            alpha_mode=alpha_mode,
            xy_offset=xy_offset,
        )
        image = Image.alpha_composite(image, attempt_layer)
    draw = ImageDraw.Draw(image, "RGBA")
    if show_aligned_start_marker:
        for attempt, xy_offset in zip(attempts, xy_offsets):
            if not attempt.trajectory:
                continue
            _draw_aligned_start_marker(
                draw,
                point=_map_projected_point(
                    attempt.trajectory[0].eef_xyz,
                    box=plot_box,
                    projection_min=projection_min,
                    projection_max=projection_max,
                    xy_offset=xy_offset,
                ),
            )
            break
    if show_attempt_labels:
        _draw_attempt_labels(
            draw,
            attempts=attempts,
            plot_box=plot_box,
            projection_min=projection_min,
            projection_max=projection_max,
            attempt_xy_offsets=xy_offsets,
            label_font=_load_font(24),
        )
    return image


def _legend_items() -> Iterable[tuple[str, tuple[int, int, int]]]:
    for phase in TASK_PHASE_ORDER:
        yield TASK_PHASE_LABELS[phase], TASK_PHASE_COLORS[phase]


def _draw_legend_marker(
    draw: ImageDraw.ImageDraw,
    *,
    kind: str,
    center: tuple[int, int],
) -> None:
    if kind == "phase":
        _draw_square_marker(
            draw,
            center=center,
            radius=10,
            outline=PHASE_MARKER_OUTLINE,
            fill=MARKER_FILL,
            width=2,
        )
        return
    if kind == "pickup_attempt":
        _draw_circle_marker(
            draw,
            center=center,
            radius=9,
            outline=MARKER_OUTLINE,
            fill=MARKER_FILL,
            width=2,
        )
        return
    if kind == "grasp_hold":
        _draw_circle_marker(
            draw,
            center=center,
            radius=7,
            outline=MARKER_SOLID,
            fill=MARKER_SOLID,
            width=1,
        )
        return
    _draw_diamond_marker(
        draw,
        center=center,
        radius=10,
        outline=RELEASE_MARKER_OUTLINE,
        fill=MARKER_FILL,
        width=2,
    )


def export_online_intervention_report_png(
    report: OnlineInterventionReport,
    output_path: Path,
    *,
    trajectory_alpha_mode: str = "time",
) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    alpha_mode = str(trajectory_alpha_mode).strip().lower()
    if alpha_mode not in {"height", "time"}:
        raise ValueError(f"unsupported trajectory alpha mode: {trajectory_alpha_mode}")

    canvas_w = 2200
    canvas_h = 1320
    margin_x = 54
    margin_top = 50
    margin_bottom = 46
    panel_gap = 40
    legend_h = 310
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
    z_min, z_max = _height_bounds(report)
    baseline_offsets = [np.zeros((2,), dtype=np.float32)]
    masked_offsets = _masked_attempt_xy_offsets(report)
    left_raw_min, left_raw_max = _projection_extents_for_attempts([report.baseline], attempt_xy_offsets=baseline_offsets)
    if report.masked_attempts:
        right_raw_min, right_raw_max = _projection_extents_for_attempts(
            report.masked_attempts,
            attempt_xy_offsets=masked_offsets,
        )
    else:
        right_raw_min, right_raw_max = left_raw_min.copy(), left_raw_max.copy()
    shared_span = np.maximum(left_raw_max - left_raw_min, right_raw_max - right_raw_min)
    left_projection_min, left_projection_max = _projection_bounds_from_extents(
        left_raw_min,
        left_raw_max,
        span_override=shared_span,
    )
    right_projection_min, right_projection_max = _projection_bounds_from_extents(
        right_raw_min,
        right_raw_max,
        span_override=shared_span,
    )

    image = _draw_plot_panel(
        image,
        panel_box=left_panel,
        attempts=[report.baseline],
        projection_min=left_projection_min,
        projection_max=left_projection_max,
        z_min=z_min,
        z_max=z_max,
        alpha=240,
        width=7,
        alpha_mode=alpha_mode,
        attempt_xy_offsets=baseline_offsets,
        show_phase_change_markers=True,
    )
    masked_trajectory_colors = [_attempt_accent_color(attempt.attempt_index) for attempt in report.masked_attempts]
    image = _draw_plot_panel(
        image,
        panel_box=right_panel,
        attempts=report.masked_attempts,
        projection_min=right_projection_min,
        projection_max=right_projection_max,
        z_min=z_min,
        z_max=z_max,
        alpha=112,
        width=6,
        alpha_mode=alpha_mode,
        attempt_xy_offsets=masked_offsets,
        trajectory_colors=masked_trajectory_colors,
        show_phase_change_markers=bool(report.masked_attempts),
        show_aligned_start_marker=bool(report.masked_attempts),
    )

    draw = ImageDraw.Draw(image, "RGBA")
    legend_font = _load_font(32)
    legend_y = canvas_h - legend_h + 26
    legend_x = margin_x + 10
    item_gap = 92
    for label, color in _legend_items():
        _draw_square_marker(
            draw,
            center=(legend_x + 14, legend_y + 17),
            radius=11,
            outline=PHASE_MARKER_OUTLINE,
            fill=(int(color[0]), int(color[1]), int(color[2]), 255),
            width=2,
        )
        draw.text((legend_x + 36, legend_y), label, fill=LEGEND_TEXT, font=legend_font)
        text_w, _ = _text_size(draw, label, font=legend_font)
        legend_x += text_w + 36 + item_gap

    marker_font = _load_font(30)
    marker_y = legend_y + 66
    marker_x = margin_x + 22
    marker_items = (
        ("pickup_attempt", "Pickup Attempt"),
        ("grasp_hold", "Grasped / Hold"),
        ("release", "Release / Commit"),
    )
    for kind, label in marker_items:
        _draw_legend_marker(draw, kind=kind, center=(marker_x, marker_y + 14))
        draw.text((marker_x + 28, marker_y), label, fill=LEGEND_TEXT, font=marker_font)
        text_w, _ = _text_size(draw, label, font=marker_font)
        marker_x += text_w + 172

    _draw_attempt_legend(
        image,
        draw,
        attempts=[report.baseline, *report.masked_attempts],
        box=(
            margin_x + 20,
            legend_y + 112,
            canvas_w - margin_x - 20,
            canvas_h - margin_bottom + 8,
        ),
    )

    image.convert("RGB").save(output_path)
    return output_path
