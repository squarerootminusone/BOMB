"""Human-readable report export for causal-localization traces."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

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


def _clip_grid(grid: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    grid = np.nan_to_num(np.asarray(grid, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    minimum = float(grid.min()) if grid.size else 0.0
    maximum = float(grid.max()) if grid.size else 0.0
    if maximum - minimum <= eps:
        return np.zeros_like(grid)
    return (grid - minimum) / (maximum - minimum)


def _apply_colormap(values: np.ndarray) -> np.ndarray:
    values = np.clip(values.astype(np.float32), 0.0, 1.0)
    r = np.clip(1.8 * values - 0.55, 0.0, 1.0)
    g = np.clip(1.7 * np.sin(values * math.pi), 0.0, 1.0)
    b = np.clip(1.65 - 1.8 * values, 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


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
    layer_heat = Image.fromarray(_apply_colormap(_clip_grid(layer_scores))).resize((80, max(24, 24 * layer_scores.shape[0])), resample=Image.Resampling.NEAREST)
    head_heat = Image.fromarray(_apply_colormap(_clip_grid(head_scores))).resize((max(120, 28 * head_scores.shape[1]), max(24, 24 * head_scores.shape[0])), resample=Image.Resampling.NEAREST)

    canvas_w = layer_heat.width + head_heat.width + 90
    canvas_h = max(layer_heat.height, head_heat.height) + 110
    canvas = Image.new("RGB", (canvas_w, canvas_h), BG_COLOR)
    draw = ImageDraw.Draw(canvas)
    fonts = {"title": _load_font(20), "body": _load_font(15), "meta": _load_font(13)}

    draw.text((18, 16), "Activation Patching Restoration", font=fonts["title"], fill=TEXT_PRIMARY)
    draw.text((18, 44), f"corruption: {step.corruption_type} -> {step.corruption_label}", font=fonts["meta"], fill=TEXT_MUTED)
    draw.text((18, 64), "left: whole-layer restoration, right: per-head restoration", font=fonts["body"], fill=TEXT_PRIMARY)

    layer_box = (18, 92, 18 + layer_heat.width, 92 + layer_heat.height)
    head_box = (48 + layer_heat.width, 92, 48 + layer_heat.width + head_heat.width, 92 + head_heat.height)
    draw.rounded_rectangle(layer_box, radius=12, fill=PANEL_BG, outline=PANEL_BORDER, width=1)
    draw.rounded_rectangle(head_box, radius=12, fill=PANEL_BG, outline=PANEL_BORDER, width=1)
    canvas.paste(layer_heat, (layer_box[0], layer_box[1]))
    canvas.paste(head_heat, (head_box[0], head_box[1]))
    draw.text((layer_box[0], layer_box[1] - 18), "layers", font=fonts["meta"], fill=ACCENT)
    draw.text((head_box[0], head_box[1] - 18), "heads", font=fonts["meta"], fill=ACCENT)

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

