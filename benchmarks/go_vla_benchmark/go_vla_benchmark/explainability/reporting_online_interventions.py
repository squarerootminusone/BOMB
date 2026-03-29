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
MARKER_OUTLINE = (28, 35, 44, 236)
MARKER_FILL = (255, 255, 255, 244)
MARKER_SOLID = (28, 35, 44, 236)
RELEASE_MARKER_OUTLINE = (150, 55, 52, 236)
GRIPPER_CLOSE_THRESHOLD = 0.25
MIN_TRAJECTORY_OPACITY = 0.3


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
    projected = _project_xy(_all_report_points(report))
    minimum = projected.min(axis=0)
    maximum = projected.max(axis=0)
    span = np.maximum(maximum - minimum, 1e-4)
    pad = span * 0.06
    return minimum - pad, maximum + pad


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


def _composite_rgba_over_panel(color: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    alpha = float(color[3]) / 255.0
    blended = []
    for channel, bg_channel in zip(color[:3], PANEL_BG[:3]):
        blended.append(int(round((float(bg_channel) * (1.0 - alpha)) + (float(channel) * alpha))))
    return (blended[0], blended[1], blended[2], 255)


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
    projected = _project_xy(np.asarray(xyz, dtype=np.float32).reshape(1, 3))[0]
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
    return _composite_rgba_over_panel((int(rgb[0]), int(rgb[1]), int(rgb[2]), int(alpha)))


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
        attempt_outline = _composite_rgba_over_panel(
            (MARKER_OUTLINE[0], MARKER_OUTLINE[1], MARKER_OUTLINE[2], marker_alpha)
        )
        attempt_fill = _composite_rgba_over_panel(
            (MARKER_FILL[0], MARKER_FILL[1], MARKER_FILL[2], min(255, marker_alpha + 20))
        )
        _draw_circle_marker(
            draw,
            center=_map_projected_point(
                xyz,
                box=plot_box,
                projection_min=projection_min,
                projection_max=projection_max,
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
        solid_fill = _composite_rgba_over_panel(
            (MARKER_SOLID[0], MARKER_SOLID[1], MARKER_SOLID[2], marker_alpha)
        )
        _draw_circle_marker(
            draw,
            center=_map_projected_point(
                xyz,
                box=plot_box,
                projection_min=projection_min,
                projection_max=projection_max,
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
        attempt_fill = _composite_rgba_over_panel(
            (MARKER_FILL[0], MARKER_FILL[1], MARKER_FILL[2], min(255, marker_alpha + 20))
        )
        release_outline = _composite_rgba_over_panel(
            (
                RELEASE_MARKER_OUTLINE[0],
                RELEASE_MARKER_OUTLINE[1],
                RELEASE_MARKER_OUTLINE[2],
                marker_alpha,
            )
        )
        _draw_diamond_marker(
            draw,
            center=_map_projected_point(
                xyz,
                box=plot_box,
                projection_min=projection_min,
                projection_max=projection_max,
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
        )
        next_point = _map_projected_point(
            step.eef_xyz,
            box=plot_box,
            projection_min=projection_min,
            projection_max=projection_max,
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
        draw.line([prev_point, next_point], fill=_line_color(step.phase, alpha=segment_alpha), width=width)


def _draw_plot_panel(
    draw: ImageDraw.ImageDraw,
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
            z_min=z_min,
            z_max=z_max,
            alpha=alpha,
            width=width,
            alpha_mode=alpha_mode,
        )
    for attempt in attempts:
        _draw_attempt_markers(
            draw,
            attempt=attempt,
            plot_box=plot_box,
            projection_min=projection_min,
            projection_max=projection_max,
            z_min=z_min,
            z_max=z_max,
            alpha=min(255, alpha + 36),
            alpha_mode=alpha_mode,
        )


def _legend_items() -> Iterable[tuple[str, tuple[int, int, int]]]:
    for phase in TASK_PHASE_ORDER:
        yield TASK_PHASE_LABELS[phase], TASK_PHASE_COLORS[phase]


def _draw_legend_marker(
    draw: ImageDraw.ImageDraw,
    *,
    kind: str,
    center: tuple[int, int],
) -> None:
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
    trajectory_alpha_mode: str = "height",
) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    alpha_mode = str(trajectory_alpha_mode).strip().lower()
    if alpha_mode not in {"height", "time"}:
        raise ValueError(f"unsupported trajectory alpha mode: {trajectory_alpha_mode}")

    canvas_w = 2200
    canvas_h = 1200
    margin_x = 54
    margin_top = 50
    margin_bottom = 46
    panel_gap = 40
    legend_h = 190
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
    z_min, z_max = _height_bounds(report)

    _draw_plot_panel(
        draw,
        panel_box=left_panel,
        attempts=[report.baseline],
        projection_min=projection_min,
        projection_max=projection_max,
        z_min=z_min,
        z_max=z_max,
        alpha=240,
        width=7,
        alpha_mode=alpha_mode,
    )
    _draw_plot_panel(
        draw,
        panel_box=right_panel,
        attempts=report.masked_attempts,
        projection_min=projection_min,
        projection_max=projection_max,
        z_min=z_min,
        z_max=z_max,
        alpha=112,
        width=6,
        alpha_mode=alpha_mode,
    )

    legend_font = _load_font(32)
    legend_y = canvas_h - legend_h + 26
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

    image.convert("RGB").save(output_path)
    return output_path
