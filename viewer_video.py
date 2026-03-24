"""Export RLDS episodes to an MP4 with the same HUD used by viewer.py.

Usage:
    conda run --no-capture-output -n mujogo python viewer_video.py
    conda run --no-capture-output -n mujogo python viewer_video.py --episode 3 --output /tmp/episode3.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from viewer import (
    AXES,
    BAR_CENTER,
    BAR_MAX_W,
    GRIP_RADIUS,
    HUD_LEFT,
    VALUE_COL,
    build_overlays,
    load_episodes,
)


TEXT_COLOR = (255, 255, 255, 255)
MUTED_TEXT_COLOR = (196, 196, 196, 255)
BG_COLOR = (0, 0, 0, 255)
ZERO_LINE_COLOR = (255, 255, 255, 128)
HEADER_PAD_X = 8
HEADER_PAD_Y = 8
HEADER_GAP = 4
HUD_RIGHT_PAD = 28


def _round_up(value: int, multiple: int) -> int:
    if multiple <= 1:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export RLDS episodes from viewer.py to a non-interactive MP4."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path("~/tensorflow_datasets").expanduser()),
        help="TFDS data directory (default: ~/tensorflow_datasets)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="viewer_preview.mp4",
        help="output MP4 path (default: ./viewer_preview.mp4)",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="single episode index to export; overrides --start-episode/--num-episodes",
    )
    parser.add_argument(
        "--start-episode",
        type=int,
        default=0,
        help="start episode index when exporting a range",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1,
        help="number of episodes to export; <= 0 means all from --start-episode",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="output video FPS (default: 5, matching the RLDS downsample target)",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="frame stride within each episode (default: 1)",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=0.5,
        help="seconds to hold the last frame of each episode (default: 0.5)",
    )
    parser.add_argument(
        "--separator-frames",
        type=int,
        default=4,
        help="black frames inserted between episodes (default: 4)",
    )
    parser.add_argument(
        "--macro-block-size",
        type=int,
        default=1,
        help="ffmpeg macro block size; use 1 to preserve native dimensions",
    )
    return parser.parse_args()


def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _rgba(color) -> tuple[int, int, int, int]:
    arr = np.asarray(color, dtype=np.float32)
    if arr.ndim != 1 or arr.size != 4:
        raise ValueError(f"expected RGBA color, got shape {arr.shape}")
    if np.max(arr) <= 1.0:
        arr = np.clip(arr * 255.0, 0, 255)
    return tuple(int(round(v)) for v in arr.tolist())


def _xy(point: np.ndarray, y_offset: int = 0) -> tuple[float, float]:
    row, col = point
    return float(col), float(row + y_offset)


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> float:
    left, _, right, _ = draw.textbbox((0, 0), text, font=font)
    return float(right - left)


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    if not text:
        return [""]

    words = text.split()
    lines: list[str] = []
    current = words[0]

    for word in words[1:]:
        candidate = f"{current} {word}"
        if _text_width(draw, candidate, font=font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word

    lines.append(current)
    return lines


def _line_height(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont) -> int:
    top, bottom = draw.textbbox((0, 0), "Ag", font=font)[1::2]
    return max(1, bottom - top)


def _hud_canvas_width(img_w: int) -> int:
    hud_min_width = int(max(VALUE_COL + 48, BAR_CENTER + BAR_MAX_W + 90) + HUD_RIGHT_PAD)
    return max(img_w, hud_min_width)


def _build_header_lines(
    draw: ImageDraw.ImageDraw,
    instruction: str,
    canvas_w: int,
    meta_font: ImageFont.ImageFont,
    instruction_font: ImageFont.ImageFont,
) -> tuple[list[str], int]:
    max_text_width = max(40, canvas_w - 2 * HEADER_PAD_X)
    instruction_lines = _wrap_text(draw, instruction, instruction_font, max_text_width)
    meta_h = _line_height(draw, meta_font)
    instruction_h = _line_height(draw, instruction_font)
    header_h = HEADER_PAD_Y + meta_h + HEADER_GAP + instruction_h * len(instruction_lines) + HEADER_PAD_Y
    return instruction_lines, header_h


def _compute_layout(
    episodes: list[dict],
    episode_indices: list[int],
    meta_font: ImageFont.ImageFont,
    instruction_font: ImageFont.ImageFont,
) -> tuple[int, int, int, dict[int, list[str]]]:
    max_img_w = max(int(episodes[idx]["images"].shape[2]) for idx in episode_indices)
    max_img_h = max(int(episodes[idx]["images"].shape[1]) for idx in episode_indices)
    canvas_w = _round_up(_hud_canvas_width(max_img_w), 2)

    dummy_draw = ImageDraw.Draw(Image.new("RGBA", (canvas_w, 1), BG_COLOR))
    header_h = 0
    instruction_lines_by_episode: dict[int, list[str]] = {}
    for idx in episode_indices:
        instruction = episodes[idx]["instruction"]
        lines, candidate_h = _build_header_lines(
            dummy_draw,
            instruction,
            canvas_w,
            meta_font,
            instruction_font,
        )
        instruction_lines_by_episode[idx] = lines
        header_h = max(header_h, candidate_h)

    canvas_h = _round_up(header_h + max_img_h, 2)
    return canvas_w, canvas_h, header_h, instruction_lines_by_episode


def _render_frame(
    frame: np.ndarray,
    overlay: dict,
    frame_idx: int,
    episode_idx: int,
    total_episodes: int,
    instruction_lines: list[str],
    canvas_w: int,
    canvas_h: int,
    header_h: int,
    meta_font: ImageFont.ImageFont,
    instruction_font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
) -> np.ndarray:
    canvas = Image.new("RGBA", (canvas_w, canvas_h), BG_COLOR)
    canvas.paste(Image.fromarray(frame).convert("RGBA"), (0, header_h))
    draw = ImageDraw.Draw(canvas, "RGBA")

    meta_text = f"Episode {episode_idx}/{total_episodes - 1} | Frame {frame_idx + 1}/{overlay['T']}"
    draw.text((HEADER_PAD_X, HEADER_PAD_Y), meta_text, font=meta_font, fill=MUTED_TEXT_COLOR)

    meta_h = _line_height(draw, meta_font)
    text_y = HEADER_PAD_Y + meta_h + HEADER_GAP
    instruction_text = "\n".join(instruction_lines)
    draw.multiline_text(
        (HEADER_PAD_X, text_y),
        instruction_text,
        font=instruction_font,
        fill=TEXT_COLOR,
        spacing=2,
    )

    y_offset = header_h

    for line in overlay["zero_lines"]:
        draw.line([_xy(pt, y_offset=y_offset) for pt in line], fill=ZERO_LINE_COLOR, width=1)

    for name, desc, _, _, _ in AXES:
        row = overlay["axis_rows"][name]
        label_y = row - 5 + y_offset
        draw.text((HUD_LEFT, label_y), f"{name} ({desc})", font=body_font, fill=TEXT_COLOR)

        polygon = overlay["bars"][name][frame_idx]
        draw.polygon([_xy(pt, y_offset=y_offset) for pt in polygon], fill=_rgba(overlay["bar_colors"][name][frame_idx]))
        draw.text((VALUE_COL, label_y), overlay["value_strs"][name][frame_idx], font=body_font, fill=TEXT_COLOR)

    grip_center_x = overlay["grip_col"]
    grip_center_y = overlay["grip_row"] + y_offset
    grip_fill = _rgba(overlay["grip_colors"][frame_idx])
    draw.ellipse(
        (
            grip_center_x - GRIP_RADIUS,
            grip_center_y - GRIP_RADIUS,
            grip_center_x + GRIP_RADIUS,
            grip_center_y + GRIP_RADIUS,
        ),
        fill=grip_fill,
    )
    draw.text((grip_center_x + 10, grip_center_y - 5), overlay["grip_strs"][frame_idx], font=body_font, fill=TEXT_COLOR)

    return np.asarray(canvas.convert("RGB"))


def _resolve_output_path(output: str) -> Path:
    return Path(output).expanduser().resolve()


def _select_episode_indices(num_episodes: int, args: argparse.Namespace) -> list[int]:
    if args.episode is not None:
        if not 0 <= args.episode < num_episodes:
            raise ValueError(f"--episode must be in [0, {num_episodes - 1}], got {args.episode}")
        return [args.episode]

    start = max(0, int(args.start_episode))
    if start >= num_episodes:
        raise ValueError(f"--start-episode must be smaller than {num_episodes}, got {start}")

    if args.num_episodes <= 0:
        return list(range(start, num_episodes))

    end = min(num_episodes, start + int(args.num_episodes))
    return list(range(start, end))


def main() -> None:
    args = parse_args()
    output_path = _resolve_output_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading RLDS dataset...")
    episodes = load_episodes(data_dir=args.data_dir)
    print(f"Loaded {len(episodes)} episodes from {args.data_dir}.")

    if not episodes:
        print("No episodes found.")
        return

    episode_indices = _select_episode_indices(len(episodes), args)
    frame_stride = max(1, int(args.frame_stride))
    separator_frames = max(0, int(args.separator_frames))
    hold_frames = max(0, int(round(max(0.0, float(args.hold_seconds)) * max(1, int(args.fps)))))

    meta_font = _load_font(12)
    instruction_font = _load_font(14)
    body_font = _load_font(12)

    canvas_w, canvas_h, header_h, instruction_lines_by_episode = _compute_layout(
        episodes,
        episode_indices,
        meta_font,
        instruction_font,
    )
    blank_frame = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    total_written = 0
    frames_per_episode: dict[int, int] = {}

    with imageio.get_writer(
        str(output_path),
        fps=int(args.fps),
        macro_block_size=int(args.macro_block_size),
    ) as writer:
        for offset, episode_idx in enumerate(episode_indices):
            episode = episodes[episode_idx]
            images = episode["images"]
            overlay = build_overlays(episode["actions"], images.shape[1], images.shape[2])

            written_for_episode = 0
            rendered_last_frame = None

            for frame_idx in range(0, overlay["T"], frame_stride):
                rendered = _render_frame(
                    images[frame_idx],
                    overlay,
                    frame_idx=frame_idx,
                    episode_idx=episode_idx,
                    total_episodes=len(episodes),
                    instruction_lines=instruction_lines_by_episode[episode_idx],
                    canvas_w=canvas_w,
                    canvas_h=canvas_h,
                    header_h=header_h,
                    meta_font=meta_font,
                    instruction_font=instruction_font,
                    body_font=body_font,
                )
                writer.append_data(rendered)
                rendered_last_frame = rendered
                written_for_episode += 1
                total_written += 1

            if rendered_last_frame is not None:
                for _ in range(hold_frames):
                    writer.append_data(rendered_last_frame)
                    written_for_episode += 1
                    total_written += 1

            frames_per_episode[episode_idx] = written_for_episode

            if offset < len(episode_indices) - 1:
                for _ in range(separator_frames):
                    writer.append_data(blank_frame)
                    total_written += 1

    print(f"Wrote {output_path}")
    print(f"Episodes exported: {episode_indices}")
    print(f"Frames written: {total_written}")
    print(f"Frames per episode: {frames_per_episode}")


if __name__ == "__main__":
    main()
