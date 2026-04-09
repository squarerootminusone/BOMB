#!/usr/bin/env python3
"""Export side-by-side inference and attention videos for Go VLA episodes.

This script supports two workflows:

1. Render from an existing trace file containing predicted actions and attention maps.
2. Run offline OpenVLA inference on Go HDF5 demos, extract image-patch attentions, rank
   the worst episodes by GT-vs-prediction mismatch, and export videos for those failures.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "go_vla_benchmark"))
sys.path.insert(0, str(REPO_ROOT / "openvla"))

from go_vla_benchmark.explainability import (  # noqa: E402
    DATASET_ADAPTERS,
    MODEL_ADAPTERS,
    EpisodeClip,
    EpisodeTrace,
    collect_episode_traces,
    load_trace_file,
    resolve_dataset_path,
    resolve_optional_path,
    resolve_repo_relative_path,
    save_trace_file,
    select_ranked_traces,
)


PANEL_MARGIN = 18
PANEL_GAP = 18
TOP_BANNER_HEIGHT = 84
FOOTER_HEIGHT = 166
CARD_GAP = 14
CARD_PADDING = 12
TAG_HEIGHT = 24
HISTORY_CELL = 28
HISTORY_GAP = 4
HISTORY_STEPS = 10

BG_COLOR = (10, 14, 22)
PANEL_BG = (22, 28, 39)
CARD_BG = (14, 18, 28, 220)
TEXT_PRIMARY = (240, 244, 250)
TEXT_SECONDARY = (186, 196, 210)
TEXT_MUTED = (126, 138, 154)
ACCENT = (88, 187, 255)
GT_COLOR = (91, 214, 155)
PRED_COLOR = (255, 184, 77)
ERROR_COLOR = (255, 112, 112)
GRID_LINE = (85, 95, 112, 220)

AXIS_SPECS = [
    ("x", "depth", 0, (255, 103, 103), (91, 208, 208)),
    ("y", "horiz", 1, (240, 110, 222), (242, 232, 102)),
    ("z", "height", 2, (95, 150, 255), (255, 163, 88)),
]


def _resolve_output_dir(output_dir: str) -> Path:
    return resolve_repo_relative_path(output_dir, repo_root=REPO_ROOT)


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Menlo.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
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
    if float(grid.max()) <= eps:
        return np.zeros_like(grid)
    grid = grid / float(grid.max())
    return grid


def _apply_colormap(values: np.ndarray) -> np.ndarray:
    values = np.clip(values.astype(np.float32), 0.0, 1.0)
    r = np.clip(1.8 * values - 0.55, 0.0, 1.0)
    g = np.clip(1.7 * np.sin(values * math.pi), 0.0, 1.0)
    b = np.clip(1.65 - 1.8 * values, 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def _resize_grid_to_frame(grid: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    image = Image.fromarray((np.clip(grid, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    return np.asarray(image.resize((target_w, target_h), resample=Image.Resampling.BILINEAR), dtype=np.uint8)


def _attention_overlay(frame: np.ndarray, attention_grid: np.ndarray, alpha: float = 0.58) -> np.ndarray:
    heat = _resize_grid_to_frame(_clip_attention_grid(attention_grid), frame.shape[0], frame.shape[1])
    heat_rgb = _apply_colormap(heat.astype(np.float32) / 255.0)
    mixed = (frame.astype(np.float32) * (1.0 - alpha)) + (heat_rgb.astype(np.float32) * alpha)
    return np.clip(mixed, 0.0, 255.0).astype(np.uint8)


def _fit_attention_grid(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    side = int(round(math.sqrt(vector.size)))
    if side * side != vector.size:
        raise ValueError(f"expected square number of image patches, got {vector.size}")
    return vector.reshape(side, side)


def _format_gripper(gripper_value: float) -> str:
    return "open" if float(gripper_value) < 0.0 else "closed"


def _sanitize_filename(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_") or "episode"


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return (box[2] - box[0], box[3] - box[1])


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


def _draw_tag(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    label: str,
    font: ImageFont.ImageFont,
    bg_fill: tuple[int, int, int, int],
) -> None:
    text_w, text_h = _text_size(draw, label, font)
    tag_w = text_w + 16
    draw.rounded_rectangle((x, y, x + tag_w, y + TAG_HEIGHT), radius=7, fill=bg_fill)
    text_y = y + max(1, (TAG_HEIGHT - text_h) // 2 - 1)
    draw.text((x + 8, text_y), label, font=font, fill=TEXT_PRIMARY)


def _draw_action_card(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    w: int,
    h: int,
    label: str,
    label_color: tuple[int, int, int],
    action: np.ndarray,
    error: Optional[float],
    fonts: Dict[str, ImageFont.ImageFont],
) -> None:
    draw.rounded_rectangle((x, y, x + w, y + h), radius=14, fill=CARD_BG, outline=(52, 61, 78, 255), width=1)
    draw.text((x + CARD_PADDING, y + CARD_PADDING - 1), label, font=fonts["label"], fill=label_color)

    header_meta = f"L1 {error:.3f}" if error is not None else ""
    if header_meta:
        meta_w, _ = _text_size(draw, header_meta, fonts["meta"])
        draw.text((x + w - CARD_PADDING - meta_w, y + CARD_PADDING), header_meta, font=fonts["meta"], fill=TEXT_MUTED)

    bar_left = x + 118
    bar_right = x + w - CARD_PADDING
    zero_x = bar_left + (bar_right - bar_left) // 2

    cursor_y = y + 42
    for axis_name, axis_desc, axis_idx, pos_color, neg_color in AXIS_SPECS:
        value = float(action[axis_idx])
        line = f"{axis_name} {axis_desc:<6} {value:+.2f}"
        draw.text((x + CARD_PADDING, cursor_y - 2), line, font=fonts["body"], fill=TEXT_SECONDARY)

        bar_top = cursor_y + 2
        bar_bottom = bar_top + 10
        draw.rounded_rectangle((bar_left, bar_top, bar_right, bar_bottom), radius=4, fill=(36, 43, 57, 255))
        draw.line((zero_x, bar_top - 2, zero_x, bar_bottom + 2), fill=GRID_LINE, width=1)

        bar_half = (bar_right - bar_left) // 2
        delta = int(round(abs(value) * bar_half))
        if value >= 0:
            fill_box = (zero_x, bar_top, zero_x + delta, bar_bottom)
            fill_color = pos_color
        else:
            fill_box = (zero_x - delta, bar_top, zero_x, bar_bottom)
            fill_color = neg_color
        draw.rounded_rectangle(fill_box, radius=4, fill=fill_color)
        cursor_y += 22

    grip_text = f"grip {_format_gripper(float(action[3]))}"
    grip_fill = GT_COLOR if float(action[3]) < 0.0 else ERROR_COLOR
    draw.ellipse((x + CARD_PADDING, y + h - 30, x + CARD_PADDING + 12, y + h - 18), fill=grip_fill)
    draw.text((x + CARD_PADDING + 20, y + h - 33), grip_text, font=fonts["body"], fill=TEXT_SECONDARY)


def _build_history_strip(grids: np.ndarray, step_idx: int, max_items: int) -> np.ndarray:
    start_idx = max(0, step_idx - max_items + 1)
    items: List[np.ndarray] = []
    for idx in range(start_idx, step_idx + 1):
        grid = _clip_attention_grid(grids[idx])
        thumb = Image.fromarray(_apply_colormap(grid)).resize(
            (HISTORY_CELL, HISTORY_CELL),
            resample=Image.Resampling.BILINEAR,
        )
        thumb_arr = np.asarray(thumb, dtype=np.uint8).copy()

        fade = 0.35 + 0.65 * ((idx - start_idx + 1) / max(1, step_idx - start_idx + 1))
        thumb_arr = np.clip(thumb_arr.astype(np.float32) * fade, 0.0, 255.0).astype(np.uint8)
        if idx == step_idx:
            thumb_arr[:, -3:, :] = np.asarray(ACCENT, dtype=np.uint8)
            thumb_arr[:3, :, :] = np.asarray(ACCENT, dtype=np.uint8)
        items.append(thumb_arr)

    if not items:
        return np.zeros((HISTORY_CELL, HISTORY_CELL, 3), dtype=np.uint8)

    width = (len(items) * HISTORY_CELL) + (max(0, len(items) - 1) * HISTORY_GAP)
    strip = np.zeros((HISTORY_CELL, width, 3), dtype=np.uint8)
    cursor_x = 0
    for item in items:
        strip[:, cursor_x : cursor_x + HISTORY_CELL] = item
        cursor_x += HISTORY_CELL + HISTORY_GAP
    return strip


def _draw_episode_frame(
    clip: EpisodeClip,
    trace: EpisodeTrace,
    step_idx: int,
    fonts: Dict[str, ImageFont.ImageFont],
) -> np.ndarray:
    frame = clip.images[step_idx]
    gt_action = clip.gt_actions[step_idx]
    pred_action = trace.pred_actions[step_idx]
    attention_grid = trace.attention_grids[step_idx]

    frame_h, frame_w = frame.shape[:2]
    canvas_w = (2 * frame_w) + PANEL_GAP + (2 * PANEL_MARGIN)
    canvas_h = TOP_BANNER_HEIGHT + frame_h + FOOTER_HEIGHT + PANEL_MARGIN

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=BG_COLOR)
    draw = ImageDraw.Draw(canvas, "RGBA")

    draw.rounded_rectangle(
        (PANEL_MARGIN, 12, canvas_w - PANEL_MARGIN, TOP_BANNER_HEIGHT - 8),
        radius=18,
        fill=(16, 22, 31, 255),
        outline=(36, 48, 67, 255),
        width=1,
    )

    meta_line = (
        f"{clip.demo_key}   step {step_idx + 1}/{len(clip.images)}   "
        f"score {trace.score:.3f}   mean {trace.mean_l1:.3f}   max {trace.max_l1:.3f}"
    )
    draw.text((PANEL_MARGIN + 16, 18), meta_line, font=fonts["meta"], fill=TEXT_MUTED)
    _draw_wrapped_text(
        draw,
        text=f"instruction: {clip.instruction}",
        font=fonts["title"],
        fill=TEXT_PRIMARY,
        x=PANEL_MARGIN + 16,
        y=36,
        max_width=canvas_w - (2 * PANEL_MARGIN) - 32,
        line_spacing=3,
    )

    left_x = PANEL_MARGIN
    right_x = PANEL_MARGIN + frame_w + PANEL_GAP
    panel_y = TOP_BANNER_HEIGHT

    draw.rounded_rectangle(
        (left_x, panel_y, left_x + frame_w, panel_y + frame_h),
        radius=16,
        fill=PANEL_BG,
    )
    draw.rounded_rectangle(
        (right_x, panel_y, right_x + frame_w, panel_y + frame_h),
        radius=16,
        fill=PANEL_BG,
    )

    canvas.paste(Image.fromarray(frame), (left_x, panel_y))
    canvas.paste(Image.fromarray(_attention_overlay(frame, attention_grid)), (right_x, panel_y))

    _draw_tag(draw, left_x + 12, panel_y + 12, "original inference", fonts["meta"], (12, 16, 24, 210))
    _draw_tag(draw, right_x + 12, panel_y + 12, "attention map", fonts["meta"], (12, 16, 24, 210))

    footer_top = panel_y + frame_h + 10
    left_footer_x = left_x
    right_footer_x = right_x

    card_width = (frame_w - CARD_GAP) // 2
    card_height = 136
    current_error = float(trace.step_errors[step_idx]) if step_idx < len(trace.step_errors) else None
    _draw_action_card(
        draw,
        x=left_footer_x,
        y=footer_top,
        w=card_width,
        h=card_height,
        label="gt",
        label_color=GT_COLOR,
        action=gt_action,
        error=None,
        fonts=fonts,
    )
    _draw_action_card(
        draw,
        x=left_footer_x + card_width + CARD_GAP,
        y=footer_top,
        w=card_width,
        h=card_height,
        label="pred",
        label_color=PRED_COLOR,
        action=pred_action,
        error=current_error,
        fonts=fonts,
    )

    draw.rounded_rectangle(
        (right_footer_x, footer_top, right_footer_x + frame_w, footer_top + card_height),
        radius=14,
        fill=CARD_BG,
        outline=(52, 61, 78, 255),
        width=1,
    )
    draw.text((right_footer_x + CARD_PADDING, footer_top + CARD_PADDING - 1), "attention history", font=fonts["label"], fill=ACCENT)
    draw.text(
        (right_footer_x + CARD_PADDING, footer_top + 36),
        "past -> present",
        font=fonts["meta"],
        fill=TEXT_MUTED,
    )

    history_strip = _build_history_strip(trace.attention_grids, step_idx=step_idx, max_items=HISTORY_STEPS)
    strip_x = right_footer_x + CARD_PADDING
    strip_y = footer_top + 58
    canvas.paste(Image.fromarray(history_strip), (strip_x, strip_y))

    current_stats = [
        f"current L1 {current_error:.3f}" if current_error is not None else "current L1 n/a",
        f"gt grip {_format_gripper(float(gt_action[3]))}",
        f"pred grip {_format_gripper(float(pred_action[3]))}",
    ]
    stats_text = "   ".join(current_stats)
    draw.text((right_footer_x + CARD_PADDING, footer_top + card_height - 28), stats_text, font=fonts["body"], fill=TEXT_SECONDARY)

    return np.asarray(canvas, dtype=np.uint8)


def _build_fonts() -> Dict[str, ImageFont.ImageFont]:
    return {
        "title": _load_font(20),
        "label": _load_font(18),
        "body": _load_font(15),
        "meta": _load_font(13),
    }


def export_videos(
    output_dir: Path,
    clips: Sequence[EpisodeClip],
    traces: Sequence[EpisodeTrace],
    fps: int,
    macro_block_size: int,
    manifest_path: Optional[Path],
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_by_key = {clip.demo_key: clip for clip in clips}
    fonts = _build_fonts()

    manifest_items = []
    for trace in traces:
        clip = clip_by_key[trace.demo_key]
        video_name = f"{_sanitize_filename(trace.demo_key)}_score_{trace.score:.3f}.mp4"
        video_path = output_dir / video_name
        with imageio.get_writer(str(video_path), fps=int(fps), macro_block_size=int(macro_block_size)) as writer:
            for step_idx in range(len(clip.images)):
                writer.append_data(_draw_episode_frame(clip=clip, trace=trace, step_idx=step_idx, fonts=fonts))

        manifest_items.append(
            {
                "demo_key": trace.demo_key,
                "instruction": clip.instruction,
                "score": trace.score,
                "mean_l1": trace.mean_l1,
                "max_l1": trace.max_l1,
                "num_steps": int(len(clip.images)),
                "video_path": str(video_path),
            }
        )

    manifest = {
        "output_dir": str(output_dir),
        "videos": manifest_items,
    }

    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=False, type=str, help="Go benchmark HDF5 dataset")
    parser.add_argument("--output-dir", required=True, type=str, help="directory for exported MP4 videos")
    parser.add_argument("--trace-input", type=str, default=None, help="optional .npz trace file to render from")
    parser.add_argument("--trace-output", type=str, default=None, help="optional .npz path to save collected traces")
    parser.add_argument("--manifest-output", type=str, default=None, help="optional JSON manifest path")
    parser.add_argument("--checkpoint", type=str, default=None, help="OpenVLA checkpoint path or HF id")
    parser.add_argument("--dataset-adapter", choices=sorted(DATASET_ADAPTERS.keys()), default="go-hdf5")
    parser.add_argument("--model-adapter", choices=sorted(MODEL_ADAPTERS.keys()), default="openvla")
    parser.add_argument("--prompt-style", choices=["openvla", "openvla-v01"], default=None)
    parser.add_argument("--unnorm-key", type=str, default=None, help="dataset statistics key for de-normalizing actions")
    parser.add_argument("--action-dim", type=int, default=None, help="fallback action dimension when norm stats are absent")
    parser.add_argument("--start", type=int, default=0, help="start demo index in sorted order")
    parser.add_argument("--num-demos", type=int, default=0, help="number of demos to process; <= 0 means all")
    parser.add_argument(
        "--rank-demos",
        type=str,
        default=None,
        help="optional comma-separated demo keys to export directly instead of top-k failures",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="extra frame stride after RLDS-style subsampling/filtering",
    )
    parser.add_argument("--max-steps", type=int, default=0, help="max timesteps per demo after applying stride")
    parser.add_argument("--top-k", type=int, default=3, help="number of highest-error demos to export")
    parser.add_argument("--fps", type=int, default=6, help="output video fps")
    parser.add_argument("--macro-block-size", type=int, default=1, help="ffmpeg macro block size")
    parser.add_argument("--attention-layers", type=int, default=4, help="number of last decoder layers to average")
    parser.add_argument(
        "--attn-implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default="eager",
        help="attention backend; use eager for reliable output_attentions",
    )
    parser.add_argument("--device", type=str, default=None, help="torch device, e.g. cuda:0 or cpu")
    parser.add_argument("--load-in-8bit", action="store_true", help="load model with bitsandbytes 8-bit weights")
    parser.add_argument("--load-in-4bit", action="store_true", help="load model with bitsandbytes 4-bit weights")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = _resolve_output_dir(args.output_dir)
    manifest_path = resolve_optional_path(args.manifest_output, repo_root=REPO_ROOT)
    trace_input_path = resolve_optional_path(args.trace_input, repo_root=REPO_ROOT)
    trace_output_path = resolve_optional_path(args.trace_output, repo_root=REPO_ROOT)

    if trace_input_path is None and args.checkpoint is None:
        raise ValueError("provide either --trace-input or --checkpoint")

    if trace_input_path is not None:
        trace_bundle = load_trace_file(trace_input_path)
        dataset_path = resolve_dataset_path(args.dataset, repo_root=REPO_ROOT) if args.dataset else trace_bundle.dataset_path
        dataset_adapter_name = trace_bundle.dataset_adapter or args.dataset_adapter
        dataset_adapter = DATASET_ADAPTERS[dataset_adapter_name]()

        clips = dataset_adapter.load_episode_clips(
            dataset_path=dataset_path,
            demos=",".join(trace_bundle.demo_keys),
            start=0,
            num_demos=0,
            stride=1,
            max_steps=0,
        )
        frame_indices_by_key = trace_bundle.frame_indices_by_key
        adjusted_clips: List[EpisodeClip] = []
        for clip in clips:
            frame_indices = frame_indices_by_key[clip.demo_key]
            if frame_indices.size == 0:
                continue
            frame_index_to_pos = {int(frame_idx): pos for pos, frame_idx in enumerate(np.asarray(clip.frame_indices, dtype=np.int32).tolist())}
            selected_positions = np.asarray([frame_index_to_pos[int(frame_idx)] for frame_idx in frame_indices], dtype=np.int32)
            adjusted_clips.append(
                EpisodeClip(
                    demo_key=clip.demo_key,
                    instruction=clip.instruction,
                    images=clip.images[selected_positions],
                    gt_actions=clip.gt_actions[selected_positions],
                    frame_indices=np.asarray(frame_indices, dtype=np.int32),
                )
            )

        selected_traces = select_ranked_traces(
            traces=[trace_bundle.traces[clip.demo_key] for clip in adjusted_clips],
            demos=args.rank_demos,
            top_k=args.top_k,
        )
        manifest = export_videos(
            output_dir=output_dir,
            clips=adjusted_clips,
            traces=selected_traces,
            fps=args.fps,
            macro_block_size=args.macro_block_size,
            manifest_path=manifest_path,
        )
        print(json.dumps(manifest, indent=2))
        return

    if args.dataset is None:
        raise ValueError("--dataset is required when collecting traces from a checkpoint")

    dataset_path = resolve_dataset_path(args.dataset, repo_root=REPO_ROOT)
    dataset_adapter = DATASET_ADAPTERS[args.dataset_adapter]()
    clips = dataset_adapter.load_episode_clips(
        dataset_path=dataset_path,
        demos=args.rank_demos,
        start=args.start,
        num_demos=args.num_demos,
        stride=args.stride,
        max_steps=args.max_steps,
    )
    if not clips:
        raise RuntimeError("no demos selected from dataset")

    model_adapter = MODEL_ADAPTERS[args.model_adapter](
        checkpoint=args.checkpoint,
        prompt_style=args.prompt_style,
        unnorm_key=args.unnorm_key,
        action_dim=args.action_dim,
        attention_layers=args.attention_layers,
        attn_implementation=args.attn_implementation,
        device=args.device,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )
    traces = collect_episode_traces(clips=clips, model_adapter=model_adapter)
    if trace_output_path is not None:
        save_trace_file(
            trace_path=trace_output_path,
            dataset_path=dataset_path,
            clips=clips,
            traces=traces,
            checkpoint=args.checkpoint,
            prompt_style=model_adapter.prompt_style,
            dataset_adapter=args.dataset_adapter,
            model_adapter=args.model_adapter,
        )

    selected_traces = select_ranked_traces(traces=traces, demos=args.rank_demos, top_k=args.top_k)
    manifest = export_videos(
        output_dir=output_dir,
        clips=clips,
        traces=selected_traces,
        fps=args.fps,
        macro_block_size=args.macro_block_size,
        manifest_path=manifest_path,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
