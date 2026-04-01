"""Paper-style rendering for online task-level intervention reports."""

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


def _mpl_rgba(color: Sequence[int], alpha: float = 1.0) -> tuple[float, float, float, float]:
    rgb = [float(int(value)) / 255.0 for value in color[:3]]
    return (
        float(rgb[0]),
        float(rgb[1]),
        float(rgb[2]),
        float(np.clip(alpha, 0.0, 1.0)),
    )


def _attempt_xy_points(
    attempt: OnlineInterventionAttempt,
    *,
    xy_offset: np.ndarray | None = None,
) -> np.ndarray:
    if not attempt.trajectory:
        return np.zeros((0, 2), dtype=np.float32)
    xyz = np.asarray(
        [_apply_xy_offset(step.eef_xyz, xy_offset=xy_offset) for step in attempt.trajectory],
        dtype=np.float32,
    )
    return _project_xy(xyz)


def _offset_projected_xy(xyz: np.ndarray, *, xy_offset: np.ndarray | None = None) -> np.ndarray:
    adjusted = _apply_xy_offset(xyz, xy_offset=xy_offset)
    return _project_xy(np.asarray(adjusted, dtype=np.float32).reshape(1, 3))[0]


def _draw_plot_content(
    image: Image.Image,
    *,
    plot_box: tuple[int, int, int, int],
    attempts: Sequence[OnlineInterventionAttempt],
    projection_min: np.ndarray,
    projection_max: np.ndarray,
    z_min: float,
    z_max: float,
    alpha: int,
    width: int,
    alpha_mode: str,
    attempt_xy_offsets: Sequence[np.ndarray] | None = None,
    show_aligned_start_marker: bool = False,
    trajectory_colors: Sequence[Sequence[int] | None] | None = None,
    show_phase_change_markers: bool = False,
) -> Image.Image:
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
    if show_aligned_start_marker:
        draw = ImageDraw.Draw(image, "RGBA")
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
    return image


def _paper_tick_values(minimum: float, maximum: float, count: int = 5) -> np.ndarray:
    if abs(float(maximum) - float(minimum)) < 1e-6:
        return np.asarray([float(minimum)] * int(count), dtype=np.float32)
    return np.linspace(float(minimum), float(maximum), int(count), dtype=np.float32)


def _format_axis_tick(value: float) -> str:
    text = f"{float(value):.2f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _draw_rotated_text(
    image: Image.Image,
    *,
    position: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
    angle: int,
) -> None:
    temp = Image.new("RGBA", (max(8, len(text) * 32), 80), color=(0, 0, 0, 0))
    temp_draw = ImageDraw.Draw(temp, "RGBA")
    temp_draw.text((4, 4), text, fill=fill, font=font)
    rotated = temp.rotate(int(angle), expand=True)
    image.alpha_composite(rotated, dest=position)


def _draw_target_marker(draw: ImageDraw.ImageDraw, *, center: tuple[int, int]) -> None:
    radius = 8
    draw.line(
        [(center[0] - radius, center[1] - radius), (center[0] + radius, center[1] + radius)],
        fill=MARKER_SOLID,
        width=3,
    )
    draw.line(
        [(center[0] - radius, center[1] + radius), (center[0] + radius, center[1] - radius)],
        fill=MARKER_SOLID,
        width=3,
    )


def _draw_reference_markers(
    draw: ImageDraw.ImageDraw,
    *,
    attempt: OnlineInterventionAttempt,
    plot_box: tuple[int, int, int, int],
    projection_min: np.ndarray,
    projection_max: np.ndarray,
    xy_offset: np.ndarray | None = None,
) -> None:
    source_point = _map_projected_point(
        attempt.source_xyz,
        box=plot_box,
        projection_min=projection_min,
        projection_max=projection_max,
        xy_offset=xy_offset,
    )
    _draw_circle_marker(
        draw,
        center=source_point,
        radius=8,
        outline=MARKER_SOLID,
        fill=(255, 255, 255, 255),
        width=2,
    )
    target_point = _map_projected_point(
        attempt.target_xyz,
        box=plot_box,
        projection_min=projection_min,
        projection_max=projection_max,
        xy_offset=xy_offset,
    )
    _draw_target_marker(draw, center=target_point)


def _draw_event_legend_row(
    draw: ImageDraw.ImageDraw,
    *,
    image: Image.Image,
    origin: tuple[int, int],
    font: ImageFont.ImageFont,
    include_aligned_start: bool,
) -> None:
    x, y = origin
    items = [
        ("pickup_attempt", "Pickup Attempt"),
        ("grasp_hold", "Grasped / Hold"),
        ("release", "Release / Commit"),
        ("source", "Source"),
        ("target", "Target"),
    ]
    if include_aligned_start:
        items.append(("aligned_start", "Aligned Start"))

    for kind, label in items:
        center = (x + 14, y + 16)
        if kind in {"pickup_attempt", "grasp_hold", "release"}:
            _draw_legend_marker(draw, kind=kind, center=center)
        elif kind == "source":
            _draw_circle_marker(
                draw,
                center=center,
                radius=8,
                outline=MARKER_SOLID,
                fill=(255, 255, 255, 255),
                width=2,
            )
        elif kind == "target":
            _draw_target_marker(draw, center=center)
        else:
            _draw_aligned_start_marker(draw, point=center)
        draw.text((x + 34, y + 1), label, fill=LEGEND_TEXT, font=font)
        text_w, _ = _text_size(draw, label, font=font)
        x += text_w + 102


def _draw_paper_axes(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    *,
    axis_box: tuple[int, int, int, int],
    plot_box: tuple[int, int, int, int],
    projection_min: np.ndarray,
    projection_max: np.ndarray,
    title: str,
) -> None:
    axis_color = (107, 114, 128, 255)
    grid_color = (229, 231, 235, 255)
    title_font = _load_font(34)
    tick_font = _load_font(20)
    label_font = _load_font(24)
    draw.text((axis_box[0], axis_box[1]), title, fill=(17, 24, 39, 255), font=title_font)
    draw.rectangle(plot_box, outline=axis_color, width=2)

    x_ticks = _paper_tick_values(float(projection_min[0]), float(projection_max[0]), count=5)
    y_ticks = _paper_tick_values(float(projection_min[1]), float(projection_max[1]), count=5)
    for tick_value in x_ticks:
        tick_point = _map_projected_point(
            np.asarray([tick_value, float(projection_min[1]), 0.0], dtype=np.float32),
            box=plot_box,
            projection_min=projection_min,
            projection_max=projection_max,
        )
        draw.line([(tick_point[0], plot_box[1]), (tick_point[0], plot_box[3])], fill=grid_color, width=1)
        draw.line([(tick_point[0], plot_box[3]), (tick_point[0], plot_box[3] + 8)], fill=axis_color, width=1)
        label = _format_axis_tick(float(tick_value))
        label_w, _ = _text_size(draw, label, font=tick_font)
        draw.text((tick_point[0] - (label_w // 2), plot_box[3] + 14), label, fill=LEGEND_TEXT, font=tick_font)

    for tick_value in y_ticks:
        tick_point = _map_projected_point(
            np.asarray([float(projection_min[0]), tick_value, 0.0], dtype=np.float32),
            box=plot_box,
            projection_min=projection_min,
            projection_max=projection_max,
        )
        draw.line([(plot_box[0], tick_point[1]), (plot_box[2], tick_point[1])], fill=grid_color, width=1)
        draw.line([(plot_box[0] - 8, tick_point[1]), (plot_box[0], tick_point[1])], fill=axis_color, width=1)
        label = _format_axis_tick(float(tick_value))
        label_w, label_h = _text_size(draw, label, font=tick_font)
        draw.text(
            (plot_box[0] - 18 - label_w, tick_point[1] - (label_h // 2)),
            label,
            fill=LEGEND_TEXT,
            font=tick_font,
        )

    x_label = "x (m)"
    x_label_w, _ = _text_size(draw, x_label, font=label_font)
    draw.text(
        (plot_box[0] + ((plot_box[2] - plot_box[0] - x_label_w) // 2), plot_box[3] + 48),
        x_label,
        fill=LEGEND_TEXT,
        font=label_font,
    )
    _draw_rotated_text(
        image,
        position=(axis_box[0], plot_box[1] + ((plot_box[3] - plot_box[1]) // 2) + 26),
        text="y (m)",
        font=label_font,
        fill=LEGEND_TEXT,
        angle=90,
    )


def _export_online_intervention_report_png_paper_style_pil(
    report: OnlineInterventionReport,
    output_path: Path,
    *,
    alpha_mode: str,
    z_min: float,
    z_max: float,
    baseline_offsets: Sequence[np.ndarray],
    masked_offsets: Sequence[np.ndarray],
    left_projection_min: np.ndarray,
    left_projection_max: np.ndarray,
    right_projection_min: np.ndarray,
    right_projection_max: np.ndarray,
) -> Path:
    canvas_w = 2200
    canvas_h = 1500
    margin_x = 64
    margin_top = 48
    margin_bottom = 42
    panel_gap = 52
    legend_h = 360
    panel_h = canvas_h - margin_top - margin_bottom - legend_h
    panel_w = int((canvas_w - (2 * margin_x) - panel_gap) / 2)

    left_axis = (
        margin_x,
        margin_top,
        margin_x + panel_w,
        margin_top + panel_h,
    )
    right_axis = (
        left_axis[2] + panel_gap,
        margin_top,
        left_axis[2] + panel_gap + panel_w,
        margin_top + panel_h,
    )
    left_plot = (
        left_axis[0] + 90,
        left_axis[1] + 62,
        left_axis[2] - 24,
        left_axis[3] - 88,
    )
    right_plot = (
        right_axis[0] + 90,
        right_axis[1] + 62,
        right_axis[2] - 24,
        right_axis[3] - 88,
    )

    image = Image.new("RGBA", (canvas_w, canvas_h), color=(255, 255, 255, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    _draw_paper_axes(
        image,
        draw,
        axis_box=left_axis,
        plot_box=left_plot,
        projection_min=left_projection_min,
        projection_max=left_projection_max,
        title="Unmasked Rollout",
    )
    _draw_paper_axes(
        image,
        draw,
        axis_box=right_axis,
        plot_box=right_plot,
        projection_min=right_projection_min,
        projection_max=right_projection_max,
        title="Masked Rollouts (Aligned Start)",
    )

    image = _draw_plot_content(
        image,
        plot_box=left_plot,
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
    draw = ImageDraw.Draw(image, "RGBA")
    _draw_reference_markers(
        draw,
        attempt=report.baseline,
        plot_box=left_plot,
        projection_min=left_projection_min,
        projection_max=left_projection_max,
        xy_offset=baseline_offsets[0] if baseline_offsets else None,
    )

    masked_trajectory_colors = [_attempt_accent_color(attempt.attempt_index) for attempt in report.masked_attempts]
    image = _draw_plot_content(
        image,
        plot_box=right_plot,
        attempts=report.masked_attempts,
        projection_min=right_projection_min,
        projection_max=right_projection_max,
        z_min=z_min,
        z_max=z_max,
        alpha=240,
        width=6,
        alpha_mode=alpha_mode,
        attempt_xy_offsets=masked_offsets,
        trajectory_colors=masked_trajectory_colors,
        show_phase_change_markers=bool(report.masked_attempts),
        show_aligned_start_marker=bool(report.masked_attempts),
    )

    draw = ImageDraw.Draw(image, "RGBA")
    header_font = _load_font(26)
    phase_font = _load_font(24)
    legend_x = margin_x + 8
    legend_y = canvas_h - legend_h + 18
    draw.text((legend_x, legend_y), "Task Phase", fill=MASK_LEGEND_HEADER, font=header_font)
    phase_row_y = legend_y + 40
    phase_row_x = legend_x
    for label, color in _legend_items():
        _draw_square_marker(
            draw,
            center=(phase_row_x + 11, phase_row_y + 14),
            radius=9,
            outline=PHASE_MARKER_OUTLINE,
            fill=(int(color[0]), int(color[1]), int(color[2]), 255),
            width=2,
        )
        draw.text((phase_row_x + 30, phase_row_y), label, fill=LEGEND_TEXT, font=phase_font)
        text_w, _ = _text_size(draw, label, font=phase_font)
        phase_row_x += text_w + 90

    event_header_y = phase_row_y + 56
    draw.text((legend_x, event_header_y), "Events", fill=MASK_LEGEND_HEADER, font=header_font)
    _draw_event_legend_row(
        draw,
        image=image,
        origin=(legend_x, event_header_y + 38),
        font=phase_font,
        include_aligned_start=bool(report.masked_attempts),
    )

    _draw_attempt_legend(
        image,
        draw,
        attempts=[report.baseline, *report.masked_attempts],
        box=(
            legend_x,
            event_header_y + 96,
            canvas_w - margin_x,
            canvas_h - 24,
        ),
    )

    image.convert("RGB").save(output_path)
    return output_path


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

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.legend_handler import HandlerTuple
        from matplotlib.lines import Line2D
    except Exception:
        return _export_online_intervention_report_png_paper_style_pil(
            report,
            output_path,
            alpha_mode=alpha_mode,
            z_min=z_min,
            z_max=z_max,
            baseline_offsets=baseline_offsets,
            masked_offsets=masked_offsets,
            left_projection_min=left_projection_min,
            left_projection_max=left_projection_max,
            right_projection_min=right_projection_min,
            right_projection_max=right_projection_max,
        )

    def _style_axis(ax, *, title: str, projection_min: np.ndarray, projection_max: np.ndarray) -> None:
        ax.set_title(title, loc="left", fontsize=15, fontweight="bold", color="#111827", pad=10)
        ax.set_xlim(float(projection_min[0]), float(projection_max[0]))
        ax.set_ylim(float(projection_min[1]), float(projection_max[1]))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (m)", fontsize=11, color="#374151")
        ax.set_ylabel("y (m)", fontsize=11, color="#374151")
        ax.set_facecolor("white")
        ax.grid(True, color="#E5E7EB", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=10, colors="#4B5563")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#9CA3AF")
        ax.spines["bottom"].set_color("#9CA3AF")
        ax.spines["left"].set_linewidth(0.9)
        ax.spines["bottom"].set_linewidth(0.9)

    def _plot_reference_markers(ax, attempt: OnlineInterventionAttempt, *, xy_offset: np.ndarray | None = None) -> None:
        source_xy = _offset_projected_xy(attempt.source_xyz, xy_offset=xy_offset)
        target_xy = _offset_projected_xy(attempt.target_xyz, xy_offset=xy_offset)
        ax.scatter(
            [source_xy[0]],
            [source_xy[1]],
            s=72,
            marker="o",
            facecolors="white",
            edgecolors=[_mpl_rgba((17, 24, 39), 1.0)],
            linewidths=1.5,
            zorder=6,
        )
        ax.scatter(
            [target_xy[0]],
            [target_xy[1]],
            s=82,
            marker="X",
            c=[_mpl_rgba((17, 24, 39), 1.0)],
            linewidths=0.0,
            zorder=6,
        )

    def _plot_attempt(
        ax,
        *,
        attempt: OnlineInterventionAttempt,
        xy_offset: np.ndarray | None = None,
        trajectory_color: Sequence[int] | None = None,
        line_width: float,
        show_phase_change_markers: bool,
    ) -> None:
        points_xy = _attempt_xy_points(attempt, xy_offset=xy_offset)
        if len(points_xy) >= 2:
            segments = np.stack([points_xy[:-1], points_xy[1:]], axis=1)
            colors: list[tuple[float, float, float, float]] = []
            segment_count = max(1, len(attempt.trajectory) - 1)
            for segment_idx, (prev_step, step) in enumerate(zip(attempt.trajectory[:-1], attempt.trajectory[1:]), start=1):
                segment_z = 0.5 * (
                    float(np.asarray(prev_step.eef_xyz, dtype=np.float32)[2])
                    + float(np.asarray(step.eef_xyz, dtype=np.float32)[2])
                )
                segment_alpha = _resolve_alpha(
                    alpha_mode=alpha_mode,
                    alpha=255,
                    z_value=segment_z,
                    z_min=z_min,
                    z_max=z_max,
                    step_index=int(segment_idx),
                    step_count=segment_count + 1,
                )
                segment_rgb = TASK_PHASE_COLORS.get(step.phase, (120, 125, 132)) if trajectory_color is None else trajectory_color
                colors.append(_mpl_rgba(segment_rgb, alpha=float(segment_alpha) / 255.0))
            collection = LineCollection(
                segments,
                colors=colors,
                linewidths=line_width,
                capstyle="round",
                joinstyle="round",
                antialiaseds=False,
                zorder=3,
            )
            ax.add_collection(collection)

        if show_phase_change_markers:
            step_count = max(1, len(attempt.trajectory))
            minimum_alpha = 0.82
            for xyz, step_idx, phase in _phase_change_markers(attempt):
                marker_alpha = max(
                    minimum_alpha,
                    float(
                        _resolve_alpha(
                            alpha_mode=alpha_mode,
                            alpha=255,
                            z_value=float(np.asarray(xyz, dtype=np.float32)[2]),
                            z_min=z_min,
                            z_max=z_max,
                            step_index=int(step_idx),
                            step_count=step_count,
                        )
                    )
                    / 255.0,
                )
                xy = _offset_projected_xy(xyz, xy_offset=xy_offset)
                ax.scatter(
                    [xy[0]],
                    [xy[1]],
                    s=52,
                    marker="s",
                    c=[_mpl_rgba(TASK_PHASE_COLORS.get(phase, (120, 125, 132)), marker_alpha)],
                    edgecolors=[_mpl_rgba(PHASE_MARKER_OUTLINE, min(1.0, marker_alpha + 0.1))],
                    linewidths=1.1,
                    zorder=4,
                )

        pickup_attempts, grasped_points, releases = _marker_events(attempt)
        for xyz, _step_idx in pickup_attempts:
            xy = _offset_projected_xy(xyz, xy_offset=xy_offset)
            ax.scatter(
                [xy[0]],
                [xy[1]],
                s=58,
                marker="o",
                facecolors="white",
                edgecolors=[_mpl_rgba(MARKER_OUTLINE, 0.95)],
                linewidths=1.2,
                zorder=5,
            )
        for xyz, _step_idx in grasped_points:
            xy = _offset_projected_xy(xyz, xy_offset=xy_offset)
            ax.scatter(
                [xy[0]],
                [xy[1]],
                s=34,
                marker="o",
                c=[_mpl_rgba(MARKER_SOLID, 0.95)],
                linewidths=0.0,
                zorder=5,
            )
        for xyz, _step_idx in releases:
            xy = _offset_projected_xy(xyz, xy_offset=xy_offset)
            ax.scatter(
                [xy[0]],
                [xy[1]],
                s=64,
                marker="D",
                facecolors="white",
                edgecolors=[_mpl_rgba(RELEASE_MARKER_OUTLINE, 0.95)],
                linewidths=1.3,
                zorder=5,
            )

    fig = plt.figure(figsize=(13.6, 8.4), dpi=220, facecolor="white")
    grid = fig.add_gridspec(2, 2, height_ratios=[4.2, 1.5], hspace=0.25, wspace=0.18)
    baseline_ax = fig.add_subplot(grid[0, 0])
    masked_ax = fig.add_subplot(grid[0, 1])
    legend_ax = fig.add_subplot(grid[1, :])
    legend_ax.axis("off")

    _style_axis(
        baseline_ax,
        title="Unmasked Rollout",
        projection_min=left_projection_min,
        projection_max=left_projection_max,
    )
    _style_axis(
        masked_ax,
        title="Masked Rollouts (Aligned Start)",
        projection_min=right_projection_min,
        projection_max=right_projection_max,
    )

    _plot_attempt(
        baseline_ax,
        attempt=report.baseline,
        xy_offset=baseline_offsets[0],
        trajectory_color=None,
        line_width=3.2,
        show_phase_change_markers=True,
    )
    _plot_reference_markers(baseline_ax, report.baseline, xy_offset=baseline_offsets[0])

    masked_trajectory_colors = [_attempt_accent_color(attempt.attempt_index) for attempt in report.masked_attempts]
    for attempt, xy_offset, trajectory_color in zip(report.masked_attempts, masked_offsets, masked_trajectory_colors):
        _plot_attempt(
            masked_ax,
            attempt=attempt,
            xy_offset=xy_offset,
            trajectory_color=trajectory_color,
            line_width=2.4,
            show_phase_change_markers=True,
        )
    if report.masked_attempts and report.masked_attempts[0].trajectory:
        aligned_start_xy = _attempt_xy_points(report.masked_attempts[0], xy_offset=masked_offsets[0])[0]
        masked_ax.scatter(
            [aligned_start_xy[0]],
            [aligned_start_xy[1]],
            s=84,
            marker="+",
            c=[_mpl_rgba(START_MARKER_OUTLINE, 0.95)],
            linewidths=1.6,
            zorder=6,
        )

    phase_handles = [
        Line2D([0], [0], color=_mpl_rgba(color, 1.0), lw=3.0, label=label)
        for label, color in _legend_items()
    ]
    event_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=7.0,
            markerfacecolor="white",
            markeredgecolor=_mpl_rgba(MARKER_OUTLINE, 0.95),
            markeredgewidth=1.2,
            label="Pickup Attempt",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=6.0,
            markerfacecolor=_mpl_rgba(MARKER_SOLID, 0.95),
            markeredgecolor=_mpl_rgba(MARKER_SOLID, 0.95),
            label="Grasped / Hold",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="None",
            markersize=7.0,
            markerfacecolor="white",
            markeredgecolor=_mpl_rgba(RELEASE_MARKER_OUTLINE, 0.95),
            markeredgewidth=1.2,
            label="Release / Commit",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=7.0,
            markerfacecolor="white",
            markeredgecolor=_mpl_rgba((17, 24, 39), 1.0),
            markeredgewidth=1.3,
            label="Source",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            linestyle="None",
            markersize=7.2,
            markerfacecolor=_mpl_rgba((17, 24, 39), 1.0),
            markeredgecolor=_mpl_rgba((17, 24, 39), 1.0),
            label="Target",
        ),
    ]
    if report.masked_attempts:
        event_handles.append(
            Line2D(
                [0],
                [0],
                marker="+",
                linestyle="None",
                markersize=9.0,
                markeredgecolor=_mpl_rgba(START_MARKER_OUTLINE, 0.95),
                markeredgewidth=1.6,
                label="Aligned Start",
            )
        )

    baseline_handle = tuple(
        Line2D([0], [0], color=_mpl_rgba(TASK_PHASE_COLORS[phase], 1.0), lw=3.0)
        for phase in TASK_PHASE_ORDER
    )
    attempt_handles: list[object] = [baseline_handle]
    attempt_labels = [_attempt_label_text(report.baseline)]
    for attempt in report.masked_attempts:
        attempt_handles.append(
            Line2D(
                [0],
                [0],
                color=_mpl_rgba(_attempt_accent_color(attempt.attempt_index), 1.0),
                lw=3.0,
            )
        )
        attempt_labels.append(_attempt_label_text(attempt))

    phase_legend = legend_ax.legend(
        phase_handles,
        [handle.get_label() for handle in phase_handles],
        title="Task Phase",
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=4,
        frameon=False,
        fontsize=10,
        title_fontsize=11,
        handlelength=2.5,
        columnspacing=1.6,
        borderaxespad=0.0,
    )
    legend_ax.add_artist(phase_legend)

    event_legend = legend_ax.legend(
        event_handles,
        [handle.get_label() for handle in event_handles],
        title="Events",
        loc="upper left",
        bbox_to_anchor=(0.0, 0.60),
        ncol=min(6, max(3, len(event_handles))),
        frameon=False,
        fontsize=10,
        title_fontsize=11,
        handlelength=1.8,
        columnspacing=1.4,
        borderaxespad=0.0,
    )
    legend_ax.add_artist(event_legend)

    attempt_legend = legend_ax.legend(
        attempt_handles,
        attempt_labels,
        title="Attempts",
        loc="upper left",
        bbox_to_anchor=(0.0, 0.18),
        ncol=min(3, max(1, len(attempt_handles))),
        frameon=False,
        fontsize=10,
        title_fontsize=11,
        handlelength=2.5,
        columnspacing=1.4,
        borderaxespad=0.0,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.25)},
    )
    legend_ax.add_artist(attempt_legend)

    fig.subplots_adjust(left=0.07, right=0.985, top=0.94, bottom=0.10)
    try:
        fig.savefig(output_path, dpi=220, facecolor="white")
    finally:
        plt.close(fig)

    Image.open(output_path).convert("RGB").save(output_path)
    return output_path
