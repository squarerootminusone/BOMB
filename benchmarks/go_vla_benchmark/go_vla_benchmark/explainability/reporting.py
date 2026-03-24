"""Human-readable local explanation report export."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .core import EpisodeClip, EpisodeTrace


BG_COLOR = (246, 248, 252)
PANEL_BG = (255, 255, 255)
PANEL_BORDER = (211, 218, 230)
TEXT_PRIMARY = (24, 32, 45)
TEXT_MUTED = (95, 108, 126)
ACCENT = (50, 102, 184)


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


def _clip_attention_grid(grid: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    grid = np.nan_to_num(np.asarray(grid, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    grid = np.maximum(grid, 0.0)
    peak = float(grid.max()) if grid.size else 0.0
    if peak <= eps:
        return np.zeros_like(grid)
    return grid / peak


def _apply_colormap(values: np.ndarray) -> np.ndarray:
    values = np.clip(values.astype(np.float32), 0.0, 1.0)
    r = np.clip(1.8 * values - 0.55, 0.0, 1.0)
    g = np.clip(1.7 * np.sin(values * math.pi), 0.0, 1.0)
    b = np.clip(1.65 - 1.8 * values, 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def _resize_grid_to_frame(grid: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    image = Image.fromarray((np.clip(grid, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    return np.asarray(image.resize((target_w, target_h), resample=Image.Resampling.BILINEAR), dtype=np.uint8)


def _attention_overlay(frame: np.ndarray, attention_grid: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    heat = _resize_grid_to_frame(_clip_attention_grid(attention_grid), frame.shape[0], frame.shape[1])
    heat_rgb = _apply_colormap(heat.astype(np.float32) / 255.0)
    mixed = (frame.astype(np.float32) * (1.0 - alpha)) + (heat_rgb.astype(np.float32) * alpha)
    return np.clip(mixed, 0.0, 255.0).astype(np.uint8)


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def _draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    x: int,
    y: int,
    max_width: int,
    line_spacing: int = 4,
) -> int:
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if _text_size(draw, candidate, font)[0] <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
    if current:
        lines.append(current)

    cursor_y = y
    for line in lines:
        draw.text((x, cursor_y), line, font=font, fill=fill)
        cursor_y += _text_size(draw, line, font)[1] + line_spacing
    return cursor_y


def _sanitize_filename(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_") or "episode"


def _token_entries(
    token_ids: Optional[np.ndarray],
    token_probs: Optional[np.ndarray],
) -> List[Dict[str, object]]:
    if token_ids is None or token_probs is None:
        return []
    token_ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
    token_probs = np.asarray(token_probs, dtype=np.float32).reshape(-1)
    count = int(min(token_ids.shape[0], token_probs.shape[0]))
    return [
        {
            "position": idx,
            "token_id": int(token_ids[idx]),
            "probability": float(token_probs[idx]),
        }
        for idx in range(count)
    ]


def _ranked_text_entries(
    token_ids: Optional[np.ndarray],
    token_texts: Optional[Sequence[str]],
    scores: Optional[np.ndarray],
    top_k: int,
) -> List[Dict[str, object]]:
    if token_ids is None or token_texts is None or scores is None:
        return []

    token_ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    count = int(min(token_ids.shape[0], scores.shape[0], len(token_texts)))
    if count == 0:
        return []

    order = np.argsort(-scores[:count], kind="stable")
    limit = count if top_k <= 0 else min(count, int(top_k))
    ranked = []
    for rank, idx in enumerate(order[:limit], start=1):
        ranked.append(
            {
                "rank": rank,
                "token_index": int(idx),
                "token_id": int(token_ids[idx]),
                "token": str(token_texts[idx]),
                "attribution": float(scores[idx]),
            }
        )
    return ranked


def _render_patch_panel(
    clip: EpisodeClip,
    trace: EpisodeTrace,
    step_idx: int,
    output_path: Path,
) -> None:
    frame = clip.images[step_idx]
    per_token_patches = None if trace.image_patch_attributions is None else np.asarray(trace.image_patch_attributions[step_idx], dtype=np.float32)
    mean_patch = np.asarray(trace.attention_grids[step_idx], dtype=np.float32)
    token_entries = _token_entries(
        None if trace.target_token_ids is None else trace.target_token_ids[step_idx],
        None if trace.target_token_probs is None else trace.target_token_probs[step_idx],
    )

    frame_h, frame_w = frame.shape[:2]
    thumb_size = 176
    columns = 4
    tile_gap = 14
    top_margin = 18
    side_margin = 18
    header_h = 90

    num_tiles = 1 + (0 if per_token_patches is None else int(per_token_patches.shape[0]))
    rows = max(1, math.ceil(num_tiles / columns))
    grid_h = rows * (thumb_size + 46) + max(0, rows - 1) * tile_gap
    canvas_w = side_margin * 2 + frame_w * 2 + tile_gap
    canvas_w = max(canvas_w, side_margin * 2 + (columns * thumb_size) + ((columns - 1) * tile_gap))
    canvas_h = header_h + top_margin + frame_h + top_margin + grid_h + side_margin

    canvas = Image.new("RGB", (canvas_w, canvas_h), BG_COLOR)
    draw = ImageDraw.Draw(canvas)
    fonts = {
        "title": _load_font(20),
        "body": _load_font(15),
        "meta": _load_font(13),
    }

    draw.text((side_margin, 16), f"{clip.demo_key}  step {step_idx + 1}/{len(clip.images)}", font=fonts["title"], fill=TEXT_PRIMARY)
    meta = f"frame_index {int(clip.frame_indices[step_idx])}   L1 {float(trace.step_errors[step_idx]):.3f}   score {trace.score:.3f}"
    draw.text((side_margin, 42), meta, font=fonts["meta"], fill=TEXT_MUTED)
    _draw_wrapped_text(
        draw,
        text=f"instruction: {clip.instruction}",
        font=fonts["body"],
        fill=TEXT_PRIMARY,
        x=side_margin,
        y=60,
        max_width=canvas_w - (2 * side_margin),
        line_spacing=2,
    )

    panels_top = header_h
    original_box = (side_margin, panels_top, side_margin + frame_w, panels_top + frame_h)
    overlay_box = (side_margin + frame_w + tile_gap, panels_top, side_margin + (2 * frame_w) + tile_gap, panels_top + frame_h)
    draw.rounded_rectangle(original_box, radius=14, fill=PANEL_BG, outline=PANEL_BORDER, width=1)
    draw.rounded_rectangle(overlay_box, radius=14, fill=PANEL_BG, outline=PANEL_BORDER, width=1)
    canvas.paste(Image.fromarray(frame), (original_box[0], original_box[1]))
    canvas.paste(Image.fromarray(_attention_overlay(frame, mean_patch)), (overlay_box[0], overlay_box[1]))
    draw.text((original_box[0] + 12, original_box[1] + 10), "original frame", font=fonts["meta"], fill=TEXT_MUTED)
    draw.text((overlay_box[0] + 12, overlay_box[1] + 10), "mean patch attribution", font=fonts["meta"], fill=TEXT_MUTED)

    grid_top = panels_top + frame_h + top_margin
    tiles: List[tuple[str, Optional[float], np.ndarray]] = [("mean", None, mean_patch)]
    if per_token_patches is not None:
        for idx in range(int(per_token_patches.shape[0])):
            probability = None if idx >= len(token_entries) else float(token_entries[idx]["probability"])
            tiles.append((f"token {idx}", probability, per_token_patches[idx]))

    for idx, (label, probability, patch_grid) in enumerate(tiles):
        row = idx // columns
        col = idx % columns
        x = side_margin + col * (thumb_size + tile_gap)
        y = grid_top + row * (thumb_size + 46 + tile_gap)
        box = (x, y, x + thumb_size, y + thumb_size)
        draw.rounded_rectangle(box, radius=12, fill=PANEL_BG, outline=PANEL_BORDER, width=1)
        heat = Image.fromarray(_apply_colormap(_clip_attention_grid(patch_grid))).resize(
            (thumb_size - 14, thumb_size - 14),
            resample=Image.Resampling.BILINEAR,
        )
        canvas.paste(heat, (x + 7, y + 7))
        label_text = label if probability is None else f"{label}  p={probability:.3f}"
        draw.text((x, y + thumb_size + 8), label_text, font=fonts["meta"], fill=ACCENT if idx == 0 else TEXT_MUTED)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _format_markdown_table(rows: Sequence[Sequence[object]], headers: Sequence[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def _step_payload(
    clip: EpisodeClip,
    trace: EpisodeTrace,
    step_idx: int,
    token_top_k: int,
    patch_panel_name: str,
) -> Dict[str, object]:
    token_entries = _token_entries(
        None if trace.target_token_ids is None else trace.target_token_ids[step_idx],
        None if trace.target_token_probs is None else trace.target_token_probs[step_idx],
    )
    mean_text_scores = None
    if trace.mean_text_token_attributions is not None and step_idx < len(trace.mean_text_token_attributions):
        mean_text_scores = np.asarray(trace.mean_text_token_attributions[step_idx], dtype=np.float32)

    mean_text_entries = _ranked_text_entries(
        token_ids=trace.text_token_ids,
        token_texts=trace.text_tokens,
        scores=mean_text_scores,
        top_k=token_top_k,
    )

    per_output_token_entries = []
    if trace.text_token_attributions is not None:
        text_token_attributions = np.asarray(trace.text_token_attributions[step_idx], dtype=np.float32)
        token_count = int(text_token_attributions.shape[0]) if text_token_attributions.ndim >= 2 else 0
        for token_idx in range(token_count):
            top_text_entries = _ranked_text_entries(
                token_ids=trace.text_token_ids,
                token_texts=trace.text_tokens,
                scores=text_token_attributions[token_idx],
                top_k=token_top_k,
            )
            probability = None if token_idx >= len(token_entries) else float(token_entries[token_idx]["probability"])
            token_id = None if token_idx >= len(token_entries) else int(token_entries[token_idx]["token_id"])
            per_output_token_entries.append(
                {
                    "target_position": token_idx,
                    "target_token_id": token_id,
                    "probability": probability,
                    "top_instruction_tokens": top_text_entries,
                }
            )

    payload = {
        "demo_key": clip.demo_key,
        "instruction": clip.instruction,
        "step_index": step_idx,
        "frame_index": int(clip.frame_indices[step_idx]),
        "score": float(trace.score),
        "mean_l1": float(trace.mean_l1),
        "max_l1": float(trace.max_l1),
        "step_l1": float(trace.step_errors[step_idx]),
        "gt_action_xyzg": np.asarray(clip.gt_actions[step_idx], dtype=np.float32).tolist(),
        "pred_action_xyzg": np.asarray(trace.pred_actions[step_idx], dtype=np.float32).tolist(),
        "raw_pred_action": None if trace.raw_pred_actions is None else np.asarray(trace.raw_pred_actions[step_idx], dtype=np.float32).tolist(),
        "prompt": trace.prompt,
        "task_text": trace.task_text,
        "target_output_tokens": token_entries,
        "mean_text_token_attribution": mean_text_entries,
        "per_output_token_text_attribution": per_output_token_entries,
        "artifacts": {
            "patch_attribution_panel": patch_panel_name,
        },
    }
    return payload


def _write_step_markdown(payload: Dict[str, object], output_path: Path) -> None:
    token_rows = [
        (entry["position"], entry["token_id"], f"{float(entry['probability']):.4f}")
        for entry in payload["target_output_tokens"]
    ]
    mean_rows = [
        (entry["rank"], entry["token"], entry["token_id"], f"{float(entry['attribution']):.4f}")
        for entry in payload["mean_text_token_attribution"]
    ]

    lines = [
        f"# {payload['demo_key']} step {int(payload['step_index'])}",
        "",
        f"Instruction: `{payload['instruction']}`",
        "",
        f"Frame index: `{payload['frame_index']}`",
        "",
        f"Step L1: `{float(payload['step_l1']):.4f}`",
        "",
        f"![Patch attribution]({payload['artifacts']['patch_attribution_panel']})",
        "",
        "## Target Output Token Probability",
        "",
    ]
    if token_rows:
        lines.append(_format_markdown_table(token_rows, headers=["position", "token_id", "probability"]))
    else:
        lines.append("No target token probabilities were stored in this trace.")

    lines.extend(
        [
            "",
            "## Mean Text Token Attribution",
            "",
        ]
    )
    if mean_rows:
        lines.append(_format_markdown_table(mean_rows, headers=["rank", "token", "token_id", "attribution"]))
    else:
        lines.append("No text token attributions were stored in this trace.")

    lines.extend(
        [
            "",
            "## Per Output Token Text Attribution",
            "",
        ]
    )
    per_output = payload["per_output_token_text_attribution"]
    if per_output:
        for entry in per_output:
            probability = entry["probability"]
            probability_text = "n/a" if probability is None else f"{float(probability):.4f}"
            lines.append(
                f"### Target token {entry['target_position']} (`id={entry['target_token_id']}`, `p={probability_text}`)"
            )
            lines.append("")
            rows = [
                (item["rank"], item["token"], item["token_id"], f"{float(item['attribution']):.4f}")
                for item in entry["top_instruction_tokens"]
            ]
            if rows:
                lines.append(_format_markdown_table(rows, headers=["rank", "token", "token_id", "attribution"]))
            else:
                lines.append("No instruction-token attribution available for this output token.")
            lines.append("")
    else:
        lines.append("No per-output-token text attributions were stored in this trace.")

    output_path.write_text("\n".join(lines))


def export_local_explanation_report(
    output_dir: Path,
    clips: Sequence[EpisodeClip],
    traces: Sequence[EpisodeTrace],
    token_top_k: int = 8,
    manifest_path: Optional[Path] = None,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_by_key = {clip.demo_key: clip for clip in clips}

    manifest_items = []
    root_readme_lines = [
        "# Local Explanation Report",
        "",
        "| demo_key | score | mean_l1 | max_l1 | report |",
        "| --- | --- | --- | --- | --- |",
    ]

    for trace in traces:
        clip = clip_by_key[trace.demo_key]
        demo_dir = output_dir / _sanitize_filename(trace.demo_key)
        demo_dir.mkdir(parents=True, exist_ok=True)

        step_items = []
        demo_readme_lines = [
            f"# {trace.demo_key}",
            "",
            f"Instruction: `{clip.instruction}`",
            "",
            f"Score: `{trace.score:.4f}`   Mean L1: `{trace.mean_l1:.4f}`   Max L1: `{trace.max_l1:.4f}`",
            "",
            "| step | frame_index | step_l1 | patch panel | details |",
            "| --- | --- | --- | --- | --- |",
        ]

        for step_idx in range(len(clip.images)):
            step_name = f"step_{step_idx:03d}"
            patch_panel_name = f"{step_name}_patch_attribution.png"
            patch_panel_path = demo_dir / patch_panel_name
            _render_patch_panel(clip=clip, trace=trace, step_idx=step_idx, output_path=patch_panel_path)

            payload = _step_payload(
                clip=clip,
                trace=trace,
                step_idx=step_idx,
                token_top_k=token_top_k,
                patch_panel_name=patch_panel_name,
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
                    "patch_panel_path": str(patch_panel_path),
                    "json_path": str(json_path),
                    "markdown_path": str(md_path),
                }
            )
            demo_readme_lines.append(
                f"| {step_idx} | {int(clip.frame_indices[step_idx])} | {float(trace.step_errors[step_idx]):.4f} | [{patch_panel_name}]({patch_panel_name}) | [{md_path.name}]({md_path.name}) |"
            )

        demo_summary = {
            "demo_key": trace.demo_key,
            "instruction": clip.instruction,
            "score": float(trace.score),
            "mean_l1": float(trace.mean_l1),
            "max_l1": float(trace.max_l1),
            "num_steps": int(len(clip.images)),
            "steps": step_items,
        }
        summary_path = demo_dir / "summary.json"
        readme_path = demo_dir / "README.md"
        summary_path.write_text(json.dumps(demo_summary, indent=2))
        readme_path.write_text("\n".join(demo_readme_lines))

        manifest_items.append(
            {
                "demo_key": trace.demo_key,
                "instruction": clip.instruction,
                "score": float(trace.score),
                "mean_l1": float(trace.mean_l1),
                "max_l1": float(trace.max_l1),
                "num_steps": int(len(clip.images)),
                "report_dir": str(demo_dir),
                "summary_path": str(summary_path),
                "readme_path": str(readme_path),
            }
        )
        root_readme_lines.append(
            f"| {trace.demo_key} | {trace.score:.4f} | {trace.mean_l1:.4f} | {trace.max_l1:.4f} | [{readme_path.parent.name}/README.md]({readme_path.parent.name}/README.md) |"
        )

    manifest = {
        "output_dir": str(output_dir),
        "episodes": manifest_items,
    }

    readme_path = output_dir / "README.md"
    readme_path.write_text("\n".join(root_readme_lines))
    manifest["readme_path"] = str(readme_path)

    target_manifest_path = output_dir / "manifest.json" if manifest_path is None else manifest_path
    target_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest["manifest_path"] = str(target_manifest_path)
    target_manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest
