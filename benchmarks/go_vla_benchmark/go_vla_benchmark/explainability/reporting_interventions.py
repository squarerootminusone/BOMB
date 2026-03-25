"""Human-readable report export for intervention-test traces."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .core import EpisodeClip
from .interventions import EpisodeInterventionTrace, InterventionCandidateEffect, InterventionStepTrace


BG_COLOR = (246, 248, 252)
PANEL_BG = (255, 255, 255)
PANEL_BORDER = (211, 218, 230)
TEXT_PRIMARY = (24, 32, 45)
TEXT_MUTED = (95, 108, 126)
ACCENT = (50, 102, 184)
OPENVLA_ACTION_VOCAB_SIZE = 32000
OPENVLA_ACTION_BINS = 256
OPENVLA_BIN_CENTERS = ((np.linspace(-1.0, 1.0, OPENVLA_ACTION_BINS)[:-1] + np.linspace(-1.0, 1.0, OPENVLA_ACTION_BINS)[1:]) / 2.0).astype(np.float32)


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


def _overlay(frame: np.ndarray, heat_grid: np.ndarray, alpha: float = 0.58) -> np.ndarray:
    heat = _resize_grid_to_frame(_clip_grid(heat_grid), frame.shape[0], frame.shape[1])
    heat_rgb = _apply_colormap(heat.astype(np.float32) / 255.0)
    mixed = (frame.astype(np.float32) * (1.0 - alpha)) + (heat_rgb.astype(np.float32) * alpha)
    return np.clip(mixed, 0.0, 255.0).astype(np.uint8)


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


def _format_int_sequence(values: Optional[Sequence[object]]) -> str:
    if values is None:
        return ""
    items = list(values)
    if not items:
        return ""
    return " ".join(str(int(item)) for item in items)


def _format_float_sequence(values: Optional[Sequence[object]], precision: int = 6) -> str:
    if values is None:
        return ""
    items = list(values)
    if not items:
        return ""
    return " ".join(f"{float(item):.{precision}f}" for item in items)


def _resolve_action_bins(
    token_ids: Optional[Sequence[object]],
    bin_indices: Optional[Sequence[object]],
    bin_centers: Optional[Sequence[object]],
) -> tuple[Optional[List[int]], Optional[List[float]]]:
    if bin_indices is not None and bin_centers is not None:
        return (
            np.asarray(bin_indices, dtype=np.int64).tolist(),
            np.asarray(bin_centers, dtype=np.float32).tolist(),
        )
    if token_ids is None:
        return None, None

    token_array = np.asarray(token_ids, dtype=np.int64).reshape(-1)
    if token_array.size == 0:
        return [], []
    derived_indices = np.clip(
        OPENVLA_ACTION_VOCAB_SIZE - token_array - 1,
        a_min=0,
        a_max=OPENVLA_BIN_CENTERS.shape[0] - 1,
    ).astype(np.int64)
    derived_centers = np.asarray(OPENVLA_BIN_CENTERS[derived_indices], dtype=np.float32)
    return derived_indices.tolist(), derived_centers.tolist()


def _candidate_rows(candidates: Sequence[InterventionCandidateEffect]) -> List[tuple[object, ...]]:
    rows = []
    for candidate in candidates:
        pred_tokens = [] if candidate.target_token_ids is None else np.asarray(candidate.target_token_ids, dtype=np.int64).tolist()
        rows.append(
            (
                candidate.index,
                candidate.label,
                f"{float(candidate.score):.4f}",
                " ".join(str(item) for item in pred_tokens),
            )
        )
    return rows


def _token_to_display_piece(token: str) -> str:
    piece = str(token)
    piece = piece.replace("▁", " ")
    piece = piece.replace("Ġ", " ")
    piece = piece.replace("Ċ", "\n")
    return piece


def _normalize_display_text(text: str) -> str:
    normalized = re.sub(r"[ \t]+", " ", text)
    normalized = re.sub(r" *\n *", "\n", normalized)
    return normalized.strip()


def _tokens_to_text(tokens: Sequence[str]) -> str:
    return _normalize_display_text("".join(_token_to_display_piece(token) for token in tokens))


def _mask_display_token(token: str) -> str:
    piece = _token_to_display_piece(token)
    leading_ws = len(piece) - len(piece.lstrip(" "))
    body = piece.lstrip(" ")
    body_len = max(1, len(body.strip()))
    return (" " * leading_ws) + ("█" * body_len)


def _mask_display_span(text: str, start: int, end: int) -> str:
    chars = list(text)
    start = max(0, int(start))
    end = min(len(chars), int(end))
    for idx in range(start, end):
        if chars[idx].isalnum():
            chars[idx] = "█"
    return _normalize_display_text("".join(chars))


def _masked_query_text(step: InterventionStepTrace, candidate: InterventionCandidateEffect) -> str:
    if (
        candidate.task_char_start is not None
        and candidate.task_char_end is not None
        and int(candidate.task_char_end) > int(candidate.task_char_start)
        and int(candidate.task_char_start) >= 0
    ):
        return _mask_display_span(_display_query_text(step), int(candidate.task_char_start), int(candidate.task_char_end))

    masked_index = int(candidate.index)
    pieces: List[str] = []
    for token_index, token in enumerate(step.text_tokens):
        pieces.append(_mask_display_token(token) if token_index == masked_index else _token_to_display_piece(token))
    return _normalize_display_text("".join(pieces))


def _display_query_text(step: InterventionStepTrace) -> str:
    if step.task_text:
        return _normalize_display_text(step.task_text)
    if step.text_tokens:
        return _tokens_to_text(step.text_tokens)
    return _normalize_display_text(step.task_text)


def _format_mask_label(label: str, index: int) -> str:
    display = _tokens_to_text([label]).strip()
    if display:
        return display
    return f"span {index}"


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    paragraphs = text.splitlines() or [""]
    wrapped: List[str] = []
    for paragraph in paragraphs:
        words = paragraph.split()
        if not words:
            wrapped.append("")
            continue
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            bbox = draw.textbbox((0, 0), candidate, font=font)
            if (bbox[2] - bbox[0]) <= max_width:
                current = candidate
            else:
                wrapped.append(current)
                current = word
        wrapped.append(current)
    return wrapped


def _text_block_height(draw: ImageDraw.ImageDraw, lines: Sequence[str], font: ImageFont.ImageFont, spacing: int = 4) -> int:
    if not lines:
        return 0
    bbox = draw.multiline_textbbox((0, 0), "\n".join(lines), font=font, spacing=spacing)
    return bbox[3] - bbox[1]


def _text_mask_candidates(step: InterventionStepTrace, limit: Optional[int] = None) -> List[InterventionCandidateEffect]:
    if step.text_masking is None:
        return []
    candidates = list(step.text_masking.top_candidates if limit is None else step.text_masking.top_candidates[:limit])
    if not candidates:
        scores = np.asarray(step.text_masking.effect_map, dtype=np.float32).reshape(-1)
        order = np.argsort(-scores, kind="stable")
        if limit is not None:
            order = order[:limit]
        for token_index in order.tolist():
            label = step.text_tokens[token_index] if token_index < len(step.text_tokens) else f"span {token_index}"
            candidates.append(
                InterventionCandidateEffect(
                    index=int(token_index),
                    label=str(label),
                    score=float(scores[token_index]),
                )
            )
    return candidates


def _text_mask_rows(step: InterventionStepTrace, limit: int = 5) -> List[tuple[str, str]]:
    candidates = _text_mask_candidates(step, limit=limit)
    rows: List[tuple[str, str]] = []
    for rank, candidate in enumerate(candidates, start=1):
        label = _format_mask_label(candidate.label, int(candidate.index))
        title = f"Mask {rank}: {label} ({float(candidate.score):.4f})"
        rows.append((title, _masked_query_text(step, candidate)))
    return rows


def _render_intervention_panel(
    clip: EpisodeClip,
    step: InterventionStepTrace,
    step_idx: int,
    output_path: Path,
) -> None:
    if step.patch_occlusion is None:
        return

    frame = clip.images[step_idx]
    frame_h, frame_w = frame.shape[:2]
    frame_gap = 12
    margin = 12
    canvas_w = (2 * frame_w) + frame_gap + (2 * margin)
    canvas_h = frame_h + (2 * margin)
    canvas = Image.new("RGB", (canvas_w, canvas_h), BG_COLOR)
    heat_grid = np.asarray(step.patch_occlusion.effect_map, dtype=np.float32)
    canvas.paste(Image.fromarray(frame), (margin, margin))
    canvas.paste(Image.fromarray(_overlay(frame, heat_grid)), (margin + frame_w + frame_gap, margin))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _step_payload(
    clip: EpisodeClip,
    trace: EpisodeInterventionTrace,
    step: InterventionStepTrace,
    step_idx: int,
    panel_name: Optional[str],
) -> Dict[str, object]:
    baseline_bin_indices, baseline_bin_centers = _resolve_action_bins(
        token_ids=step.baseline_target_token_ids,
        bin_indices=step.baseline_target_token_bin_indices,
        bin_centers=step.baseline_target_token_bin_centers,
    )

    counterfactual_rows = []
    for edit in step.counterfactual_edits:
        predicted_bin_indices, predicted_bin_centers = _resolve_action_bins(
            token_ids=edit.target_token_ids,
            bin_indices=edit.target_token_bin_indices,
            bin_centers=edit.target_token_bin_centers,
        )
        counterfactual_rows.append(
            {
                "rank": edit.step_rank,
                "edit_type": edit.edit_type,
                "index": edit.index,
                "label": edit.label,
                "single_effect_score": float(edit.single_effect_score),
                "cumulative_sequence_logprob": float(edit.cumulative_sequence_logprob),
                "cumulative_sequence_logprob_drop": float(edit.cumulative_sequence_logprob_drop),
                "changed_prediction": bool(edit.changed_prediction),
                "predicted_token_ids": np.asarray(edit.target_token_ids, dtype=np.int64).tolist(),
                "predicted_token_bin_indices": predicted_bin_indices,
                "predicted_token_bin_centers": predicted_bin_centers,
            }
        )

    return {
        "demo_key": clip.demo_key,
        "instruction": clip.instruction,
        "task_text": _display_query_text(step),
        "step_index": step_idx,
        "frame_index": int(clip.frame_indices[step_idx]),
        "step_l1": float(trace.step_errors[step_idx]),
        "baseline_pred_action_xyzg": np.asarray(step.baseline_pred_action_xyzg, dtype=np.float32).tolist(),
        "baseline_raw_pred_action": np.asarray(step.baseline_raw_pred_action, dtype=np.float32).tolist(),
        "baseline_target_token_ids": np.asarray(step.baseline_target_token_ids, dtype=np.int64).tolist(),
        "baseline_target_token_probs": np.asarray(step.baseline_target_token_probs, dtype=np.float32).tolist(),
        "baseline_target_token_bin_indices": baseline_bin_indices,
        "baseline_target_token_bin_centers": baseline_bin_centers,
        "patch_occlusion": None
        if step.patch_occlusion is None
        else {
            "effect_map": np.asarray(step.patch_occlusion.effect_map, dtype=np.float32).tolist(),
            "top_candidates": [
                {
                    "index": item.index,
                    "label": item.label,
                    "score": float(item.score),
                    "predicted_token_ids": None if item.target_token_ids is None else np.asarray(item.target_token_ids, dtype=np.int64).tolist(),
                    "predicted_token_bin_indices": _resolve_action_bins(
                        token_ids=item.target_token_ids,
                        bin_indices=item.target_token_bin_indices,
                        bin_centers=item.target_token_bin_centers,
                    )[0],
                    "predicted_token_bin_centers": _resolve_action_bins(
                        token_ids=item.target_token_ids,
                        bin_indices=item.target_token_bin_indices,
                        bin_centers=item.target_token_bin_centers,
                    )[1],
                }
                for item in step.patch_occlusion.top_candidates
            ],
        },
        "text_masking": None
        if step.text_masking is None
        else {
            "original_query": _display_query_text(step),
            "effect_map": np.asarray(step.text_masking.effect_map, dtype=np.float32).tolist(),
            "top_candidates": [
                {
                    "index": item.index,
                    "label": item.label,
                    "score": float(item.score),
                    "task_char_start": item.task_char_start,
                    "task_char_end": item.task_char_end,
                    "masked_query": _masked_query_text(step, item),
                    "predicted_token_ids": None if item.target_token_ids is None else np.asarray(item.target_token_ids, dtype=np.int64).tolist(),
                    "predicted_token_bin_indices": _resolve_action_bins(
                        token_ids=item.target_token_ids,
                        bin_indices=item.target_token_bin_indices,
                        bin_centers=item.target_token_bin_centers,
                    )[0],
                    "predicted_token_bin_centers": _resolve_action_bins(
                        token_ids=item.target_token_ids,
                        bin_indices=item.target_token_bin_indices,
                        bin_centers=item.target_token_bin_centers,
                    )[1],
                }
                for item in _text_mask_candidates(step, limit=None)
            ],
        },
        "counterfactual_success": bool(step.counterfactual_success),
        "counterfactual_edits": counterfactual_rows,
        "artifacts": {
            "intervention_panel": panel_name,
            "patch_occlusion_panel": panel_name if step.patch_occlusion is not None else None,
        },
    }


def _write_step_markdown(payload: Dict[str, object], output_path: Path) -> None:
    baseline_bin_indices = payload.get("baseline_target_token_bin_indices") or []
    baseline_bin_centers = payload.get("baseline_target_token_bin_centers") or []
    baseline_rows = [
        (
            idx,
            token_id,
            "" if idx >= len(baseline_bin_indices) else int(baseline_bin_indices[idx]),
            "" if idx >= len(baseline_bin_centers) else f"{float(baseline_bin_centers[idx]):.6f}",
            f"{float(prob):.4f}",
        )
        for idx, (token_id, prob) in enumerate(zip(payload["baseline_target_token_ids"], payload["baseline_target_token_probs"]))
    ]
    patch_rows = []
    if payload["patch_occlusion"] is not None:
        patch_rows = [
            (
                item["index"],
                item["label"],
                f"{float(item['score']):.4f}",
                _format_int_sequence(item.get("predicted_token_ids")),
                _format_int_sequence(item.get("predicted_token_bin_indices")),
                _format_float_sequence(item.get("predicted_token_bin_centers")),
            )
            for item in payload["patch_occlusion"]["top_candidates"]
        ]
    text_rows = []
    if payload["text_masking"] is not None:
        text_rows = [
            (
                item["index"],
                item["label"],
                f"{float(item['score']):.4f}",
                item.get("masked_query", ""),
                _format_int_sequence(item.get("predicted_token_ids")),
                _format_int_sequence(item.get("predicted_token_bin_indices")),
                _format_float_sequence(item.get("predicted_token_bin_centers")),
            )
            for item in payload["text_masking"]["top_candidates"]
        ]
    counterfactual_rows = [
        (
            item["rank"],
            item["edit_type"],
            item["label"],
            f"{float(item['single_effect_score']):.4f}",
            f"{float(item['cumulative_sequence_logprob_drop']):.4f}",
            item["changed_prediction"],
            _format_int_sequence(item.get("predicted_token_ids")),
            _format_int_sequence(item.get("predicted_token_bin_indices")),
            _format_float_sequence(item.get("predicted_token_bin_centers")),
        )
        for item in payload["counterfactual_edits"]
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
        f"Baseline raw action: `{payload['baseline_raw_pred_action']}`",
        "",
        "## Baseline Output Tokens",
        "",
        _format_markdown_table(baseline_rows, headers=["position", "token_id", "bin_index", "bin_center", "probability"]),
        "",
        "## Patch Occlusion",
        "",
    ]
    panel_name = payload["artifacts"].get("intervention_panel") or payload["artifacts"].get("patch_occlusion_panel")
    if panel_name:
        lines.extend([f"![Intervention panel]({panel_name})", ""])
    lines.append(
        _format_markdown_table(
            patch_rows,
            headers=["index", "label", "logprob_drop", "predicted_token_ids", "predicted_bin_indices", "predicted_bin_centers"],
        )
        if patch_rows
        else "Patch occlusion was not run for this step."
    )
    lines.extend(["", "## Text Masking", ""])
    if payload["text_masking"] is not None:
        lines.extend(
            [
                f"Original query: `{payload['text_masking']['original_query']}`",
                "",
                _format_markdown_table(
                    text_rows,
                    headers=[
                        "index",
                        "label",
                        "logprob_drop",
                        "masked_query",
                        "predicted_token_ids",
                        "predicted_bin_indices",
                        "predicted_bin_centers",
                    ],
                )
                if text_rows
                else "Text masking was not run for this step.",
            ]
        )
    else:
        lines.append("Text masking was not run for this step.")
    lines.extend(["", "## Minimal Counterfactual Edits", ""])
    lines.append(
        _format_markdown_table(
            counterfactual_rows,
            headers=[
                "rank",
                "edit_type",
                "label",
                "single_effect",
                "cumulative_drop",
                "changed",
                "predicted_token_ids",
                "predicted_bin_indices",
                "predicted_bin_centers",
            ],
        )
        if counterfactual_rows
        else "No counterfactual edit trajectory was stored."
    )
    output_path.write_text("\n".join(lines))


def export_intervention_report(
    output_dir: Path,
    clips: Sequence[EpisodeClip],
    traces: Sequence[EpisodeInterventionTrace],
    manifest_path: Optional[Path] = None,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_by_key = {clip.demo_key: clip for clip in clips}

    manifest_items = []
    root_lines = [
        "# Intervention Test Report",
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
            panel_name = None
            if step.patch_occlusion is not None:
                panel_name = f"{step_name}_intervention_panel.png"
                _render_intervention_panel(
                    clip=clip,
                    step=step,
                    step_idx=step_idx,
                    output_path=demo_dir / panel_name,
                )

            payload = _step_payload(
                clip=clip,
                trace=trace,
                step=step,
                step_idx=step_idx,
                panel_name=panel_name,
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
