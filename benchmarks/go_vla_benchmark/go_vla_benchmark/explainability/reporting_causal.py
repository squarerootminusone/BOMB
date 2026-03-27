"""Human-readable report export for causal-localization traces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .causal_localization import CausalLocalizationStep, EpisodeCausalTrace
from .core import EpisodeClip


BG_COLOR = (246, 248, 252)
PANEL_BG = (255, 255, 255)
PANEL_BORDER = (211, 218, 230)
TEXT_PRIMARY = (24, 32, 45)
TEXT_MUTED = (95, 108, 126)
ACCENT = (50, 102, 184)
GRID_LINE = (228, 233, 242)
NEGATIVE_COLOR = (56, 118, 217)
NEUTRAL_COLOR = (245, 247, 250)
POSITIVE_COLOR = (212, 61, 61)
HEAT_CELL_H = 24
LAYER_CELL_W = 88
HEAD_CELL_W = 28
HEADER_MIN_W = 560
PANEL_GAP = 32
HEADER_HEIGHT = 132
RIGHT_MARGIN = 18
LEGEND_BAR_W = 160
LEGEND_BAR_H = 14
RESTORATION_MIN = -1.0
RESTORATION_MAX = 1.0


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


def _normalize_restoration_scores(
    scores: np.ndarray,
    score_min: float = RESTORATION_MIN,
    score_max: float = RESTORATION_MAX,
    eps: float = 1e-8,
) -> np.ndarray:
    scores = np.nan_to_num(np.asarray(scores, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if abs(score_max - score_min) <= eps:
        return np.zeros_like(scores)
    clipped = np.clip(scores, score_min, score_max)
    return (clipped - score_min) / (score_max - score_min)


def _apply_colormap(values: np.ndarray) -> np.ndarray:
    values = np.clip(values.astype(np.float32), 0.0, 1.0)
    values_expanded = values[..., None]
    negative = np.asarray(NEGATIVE_COLOR, dtype=np.float32)
    neutral = np.asarray(NEUTRAL_COLOR, dtype=np.float32)
    positive = np.asarray(POSITIVE_COLOR, dtype=np.float32)

    lower_mix = np.clip(values_expanded * 2.0, 0.0, 1.0)
    upper_mix = np.clip((values_expanded - 0.5) * 2.0, 0.0, 1.0)
    lower = negative + (neutral - negative) * lower_mix
    upper = neutral + (positive - neutral) * upper_mix
    rgb = np.where(values_expanded <= 0.5, lower, upper)
    return np.rint(rgb).astype(np.uint8)


def _render_score_grid(scores: np.ndarray, cell_w: int, cell_h: int) -> Image.Image:
    score_grid = np.asarray(scores, dtype=np.float32)
    if score_grid.ndim != 2:
        raise ValueError(f"expected a 2D score grid, got shape {score_grid.shape}")
    color_grid = _apply_colormap(_normalize_restoration_scores(score_grid))
    return Image.fromarray(color_grid).resize(
        (max(1, cell_w * score_grid.shape[1]), max(cell_h, cell_h * score_grid.shape[0])),
        resample=Image.Resampling.NEAREST,
    )


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    center: Tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill: Tuple[int, int, int],
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    draw.text((center[0] - width / 2.0, center[1] - height / 2.0), text, font=font, fill=fill)


def _cell_text_fill(rgb: np.ndarray) -> Tuple[int, int, int]:
    luminance = 0.299 * float(rgb[0]) + 0.587 * float(rgb[1]) + 0.114 * float(rgb[2])
    return PANEL_BG if luminance < 150.0 else TEXT_PRIMARY


def _draw_panel_grid(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    rows: int,
    cols: int,
    cell_w: int,
    cell_h: int,
) -> None:
    for row_idx in range(rows + 1):
        y = box[1] + row_idx * cell_h
        draw.line((box[0], y, box[2], y), fill=GRID_LINE, width=1)
    for col_idx in range(cols + 1):
        x = box[0] + col_idx * cell_w
        draw.line((x, box[1], x, box[3]), fill=GRID_LINE, width=1)


def _draw_color_legend(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    origin_x: int,
    origin_y: int,
    fonts: Dict[str, ImageFont.ImageFont],
) -> None:
    draw.text((origin_x, origin_y), "restoration scale", font=fonts["meta"], fill=ACCENT)
    legend_bar = _render_score_grid(
        np.linspace(RESTORATION_MIN, RESTORATION_MAX, num=LEGEND_BAR_W, dtype=np.float32).reshape(1, -1),
        cell_w=1,
        cell_h=LEGEND_BAR_H,
    )
    bar_box = (origin_x, origin_y + 18, origin_x + LEGEND_BAR_W, origin_y + 18 + LEGEND_BAR_H)
    draw.rounded_rectangle(
        (bar_box[0] - 1, bar_box[1] - 1, bar_box[2] + 1, bar_box[3] + 1),
        radius=6,
        fill=PANEL_BG,
        outline=PANEL_BORDER,
        width=1,
    )
    canvas.paste(legend_bar, (bar_box[0], bar_box[1]))
    label_y = bar_box[3] + 10
    _draw_centered_text(draw, (bar_box[0] + 10, label_y), "-1", font=fonts["axis"], fill=TEXT_MUTED)
    _draw_centered_text(draw, ((bar_box[0] + bar_box[2]) / 2.0, label_y), "0", font=fonts["axis"], fill=TEXT_MUTED)
    _draw_centered_text(draw, (bar_box[2] - 10, label_y), "+1", font=fonts["axis"], fill=TEXT_MUTED)


def _sanitize_filename(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_") or "episode"


def _format_markdown_table(rows: Sequence[Sequence[object]], headers: Sequence[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def _render_heatmap(step: CausalLocalizationStep, output_path: Path) -> None:
    layer_scores = np.asarray(step.layer_restoration_scores, dtype=np.float32).reshape(-1, 1)
    head_scores = np.asarray(step.head_restoration_scores, dtype=np.float32)
    if head_scores.ndim != 2:
        raise ValueError(f"expected per-head scores to be 2D, got shape {head_scores.shape}")

    layer_count = int(layer_scores.shape[0])
    head_count = int(head_scores.shape[1]) if head_scores.size else 0
    if head_count <= 0:
        raise ValueError("expected at least one attention head in the per-head score grid")
    head_cell_w = max(HEAD_CELL_W, (120 + max(1, head_count) - 1) // max(1, head_count))

    layer_heat = _render_score_grid(layer_scores, cell_w=LAYER_CELL_W, cell_h=HEAT_CELL_H)
    head_heat = _render_score_grid(head_scores, cell_w=head_cell_w, cell_h=HEAT_CELL_H)

    canvas_w = max(HEADER_MIN_W, 18 + layer_heat.width + PANEL_GAP + head_heat.width + RIGHT_MARGIN)
    canvas_h = max(layer_heat.height, head_heat.height) + HEADER_HEIGHT + 28
    canvas = Image.new("RGB", (canvas_w, canvas_h), BG_COLOR)
    draw = ImageDraw.Draw(canvas)
    fonts = {
        "title": _load_font(20),
        "body": _load_font(15),
        "meta": _load_font(13),
        "axis": _load_font(10),
        "layer": _load_font(12),
    }

    draw.text((18, 16), "Activation Patching Restoration", font=fonts["title"], fill=TEXT_PRIMARY)
    draw.text((18, 44), f"corruption: {step.corruption_type} -> {step.corruption_label}", font=fonts["meta"], fill=TEXT_MUTED)
    draw.text((18, 64), "left: whole-layer restoration, right: per-head restoration", font=fonts["body"], fill=TEXT_PRIMARY)
    _draw_color_legend(canvas, draw, canvas_w - RIGHT_MARGIN - LEGEND_BAR_W, 16, fonts)

    layer_box = (18, HEADER_HEIGHT, 18 + layer_heat.width, HEADER_HEIGHT + layer_heat.height)
    head_box = (18 + layer_heat.width + PANEL_GAP, HEADER_HEIGHT, 18 + layer_heat.width + PANEL_GAP + head_heat.width, HEADER_HEIGHT + head_heat.height)
    draw.rounded_rectangle(layer_box, radius=12, fill=PANEL_BG, outline=PANEL_BORDER, width=1)
    draw.rounded_rectangle(head_box, radius=12, fill=PANEL_BG, outline=PANEL_BORDER, width=1)
    canvas.paste(layer_heat, (layer_box[0], layer_box[1]))
    canvas.paste(head_heat, (head_box[0], head_box[1]))
    _draw_panel_grid(draw, layer_box, rows=layer_count, cols=1, cell_w=LAYER_CELL_W, cell_h=HEAT_CELL_H)
    _draw_panel_grid(draw, head_box, rows=layer_count, cols=head_count, cell_w=head_cell_w, cell_h=HEAT_CELL_H)

    draw.text((layer_box[0], layer_box[1] - 38), "layer index", font=fonts["meta"], fill=ACCENT)
    draw.text((head_box[0], head_box[1] - 38), "head index", font=fonts["meta"], fill=ACCENT)

    layer_pixels = np.asarray(layer_heat)
    for layer_idx in range(layer_count):
        center_y = layer_box[1] + layer_idx * HEAT_CELL_H + HEAT_CELL_H / 2.0
        row_rgb = layer_pixels[layer_idx * HEAT_CELL_H + HEAT_CELL_H // 2, layer_heat.width // 2]
        _draw_centered_text(
            draw,
            (layer_box[0] + layer_heat.width / 2.0, center_y),
            str(layer_idx),
            font=fonts["layer"],
            fill=_cell_text_fill(row_rgb),
        )

    for head_idx in range(head_count):
        center_x = head_box[0] + head_idx * head_cell_w + head_cell_w / 2.0
        _draw_centered_text(draw, (center_x, head_box[1] - 14), str(head_idx), font=fonts["axis"], fill=TEXT_MUTED)
        draw.line((center_x, head_box[1] - 4, center_x, head_box[1]), fill=PANEL_BORDER, width=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _step_payload(
    clip: EpisodeClip,
    trace: EpisodeCausalTrace,
    step: CausalLocalizationStep,
    step_idx: int,
    heatmap_name: str,
) -> Dict[str, object]:
    layer_scores = np.asarray(step.layer_restoration_scores, dtype=np.float32)
    head_scores = np.asarray(step.head_restoration_scores, dtype=np.float32)
    best_layer = int(np.argmax(layer_scores)) if layer_scores.size else -1
    best_head_flat = int(np.argmax(head_scores)) if head_scores.size else -1
    best_head = None if best_head_flat < 0 else np.unravel_index(best_head_flat, head_scores.shape)

    cross_attention_entries = []
    if step.cross_attention_restoration_scores is not None and step.cross_attention_labels is not None:
        for label, score in zip(step.cross_attention_labels, np.asarray(step.cross_attention_restoration_scores, dtype=np.float32).tolist()):
            cross_attention_entries.append({"label": label, "restoration": float(score)})

    return {
        "demo_key": clip.demo_key,
        "instruction": clip.instruction,
        "step_index": step_idx,
        "frame_index": int(clip.frame_indices[step_idx]),
        "step_l1": float(trace.step_errors[step_idx]),
        "corruption_type": step.corruption_type,
        "corruption_index": int(step.corruption_index),
        "corruption_label": step.corruption_label,
        "clean_sequence_logprob": float(step.clean_sequence_logprob),
        "corrupted_sequence_logprob": float(step.corrupted_sequence_logprob),
        "layer_restoration_scores": layer_scores.tolist(),
        "head_restoration_scores": head_scores.tolist(),
        "best_layer": best_layer,
        "best_head": None if best_head is None else {"layer": int(best_head[0]), "head": int(best_head[1])},
        "cross_attention": cross_attention_entries,
        "artifacts": {"restoration_heatmap": heatmap_name},
    }


def _write_step_markdown(payload: Dict[str, object], output_path: Path) -> None:
    layer_rows = [(idx, f"{float(score):.4f}") for idx, score in enumerate(payload["layer_restoration_scores"])]
    head_rows = []
    head_scores = np.asarray(payload["head_restoration_scores"], dtype=np.float32)
    head_shape = tuple(int(dim) for dim in head_scores.shape) if head_scores.size else (0, 0)
    if head_scores.size:
        for layer_idx in range(head_scores.shape[0]):
            top_head = int(np.argmax(head_scores[layer_idx]))
            head_rows.append((layer_idx, top_head, f"{float(head_scores[layer_idx, top_head]):.4f}"))
    cross_rows = [(item["label"], f"{float(item['restoration']):.4f}") for item in payload["cross_attention"]]

    lines = [
        f"# {payload['demo_key']} step {int(payload['step_index'])}",
        "",
        f"Instruction: `{payload['instruction']}`",
        "",
        f"Frame index: `{payload['frame_index']}`",
        "",
        f"Corruption: `{payload['corruption_type']}` / `{payload['corruption_label']}`",
        "",
        f"Clean log-prob: `{float(payload['clean_sequence_logprob']):.4f}`   Corrupted log-prob: `{float(payload['corrupted_sequence_logprob']):.4f}`",
        "",
        f"![Restoration heatmap]({payload['artifacts']['restoration_heatmap']})",
        "",
        "Color scale in PNG: `blue <= -1`, `white = 0`, `red >= +1` with scores clipped into `[-1, +1]` before coloring.",
        "",
        f"Head grid shape: `{head_shape[0]} layers x {head_shape[1]} heads` (`layer_idx`, `head_idx`).",
        "",
        "## Layer Restoration",
        "",
        _format_markdown_table(layer_rows, headers=["layer", "restoration"]),
        "",
        "## Best Head Per Layer",
        "",
        _format_markdown_table(head_rows, headers=["layer", "head", "restoration"]) if head_rows else "No head scores were stored.",
        "",
        "## Cross-Attention Blocks",
        "",
        _format_markdown_table(cross_rows, headers=["block", "restoration"]) if cross_rows else "No cross-attention blocks were patched.",
    ]
    output_path.write_text("\n".join(lines))


def export_causal_localization_report(
    output_dir: Path,
    clips: Sequence[EpisodeClip],
    traces: Sequence[EpisodeCausalTrace],
    manifest_path: Optional[Path] = None,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_by_key = {clip.demo_key: clip for clip in clips}

    manifest_items = []
    root_lines = [
        "# Internal Causal Localization Report",
        "",
        "| demo_key | score | mean_l1 | report |",
        "| --- | --- | --- | --- |",
    ]

    for trace in traces:
        clip = clip_by_key[trace.demo_key]
        demo_dir = output_dir / _sanitize_filename(trace.demo_key)
        demo_dir.mkdir(parents=True, exist_ok=True)

        step_items = []
        demo_lines = [
            f"# {trace.demo_key}",
            "",
            f"Instruction: `{clip.instruction}`",
            "",
            f"Score: `{trace.score:.4f}`   Mean L1: `{trace.mean_l1:.4f}`   Max L1: `{trace.max_l1:.4f}`",
            "",
            "| step | frame_index | step_l1 | details |",
            "| --- | --- | --- | --- |",
        ]

        for step_idx, step in enumerate(trace.steps):
            step_name = f"step_{step_idx:03d}"
            heatmap_name = f"{step_name}_restoration_heatmap.png"
            _render_heatmap(step=step, output_path=demo_dir / heatmap_name)

            payload = _step_payload(
                clip=clip,
                trace=trace,
                step=step,
                step_idx=step_idx,
                heatmap_name=heatmap_name,
            )
            json_path = demo_dir / f"{step_name}.json"
            md_path = demo_dir / f"{step_name}.md"
            json_path.write_text(json.dumps(payload, indent=2))
            _write_step_markdown(payload, md_path)

            step_items.append(
                {
                    "step_index": step_idx,
                    "frame_index": int(clip.frame_indices[step_idx]),
                    "step_l1": float(trace.step_errors[step_idx]),
                    "json_path": str(json_path),
                    "markdown_path": str(md_path),
                }
            )
            demo_lines.append(
                f"| {step_idx} | {int(clip.frame_indices[step_idx])} | {float(trace.step_errors[step_idx]):.4f} | [{md_path.name}]({md_path.name}) |"
            )

        summary = {
            "demo_key": trace.demo_key,
            "instruction": clip.instruction,
            "score": float(trace.score),
            "mean_l1": float(trace.mean_l1),
            "max_l1": float(trace.max_l1),
            "num_steps": int(len(trace.steps)),
            "steps": step_items,
        }
        summary_path = demo_dir / "summary.json"
        readme_path = demo_dir / "README.md"
        summary_path.write_text(json.dumps(summary, indent=2))
        readme_path.write_text("\n".join(demo_lines))

        manifest_items.append(
            {
                "demo_key": trace.demo_key,
                "instruction": clip.instruction,
                "score": float(trace.score),
                "mean_l1": float(trace.mean_l1),
                "max_l1": float(trace.max_l1),
                "num_steps": int(len(trace.steps)),
                "report_dir": str(demo_dir),
                "summary_path": str(summary_path),
                "readme_path": str(readme_path),
            }
        )
        root_lines.append(
            f"| {trace.demo_key} | {trace.score:.4f} | {trace.mean_l1:.4f} | [{readme_path.parent.name}/README.md]({readme_path.parent.name}/README.md) |"
        )

    manifest = {"output_dir": str(output_dir), "episodes": manifest_items}
    readme_path = output_dir / "README.md"
    readme_path.write_text("\n".join(root_lines))
    manifest["readme_path"] = str(readme_path)

    target_manifest_path = output_dir / "manifest.json" if manifest_path is None else manifest_path
    target_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest["manifest_path"] = str(target_manifest_path)
    target_manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest
