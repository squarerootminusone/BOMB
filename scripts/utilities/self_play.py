#!/usr/bin/env python3
"""VLA Self-Play: two arms play Go against each other.

A move selector (random, MCTS, or KataGo) picks WHERE to play;
the finetuned OpenVLA model physically places each stone via the
robot arm. Outputs a continuous video and game record JSON.

Usage:
    MUJOCO_GL=egl conda run --no-capture-output -n mujogo python \
        scripts/utilities/self_play.py \
        --model-path outputs/<date>/<time>/checkpoints/best \
        --max-moves 15 --load-4bit --player1 random --player2 mcts
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "openvla"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "go_vla_benchmark"))

from go_vla_benchmark.paths import bootstrap_pythonpath

bootstrap_pythonpath(REPO_ROOT)

from go_vla_benchmark.robosuite_compat import ensure_robosuite_compat

ensure_robosuite_compat()

from go_vla_benchmark.common import GoResetOptions
from go_vla_benchmark.env_factory import create_benchmark_env
from go_vla_benchmark.move_selection import KataGoMoveSelector, make_selector
from go_vla_benchmark.openvla_action_utils import openvla_action_to_benchmark

from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from transformers import BitsAndBytesConfig

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Same instruction templates as training data
_INSTRUCTION_TEMPLATES = [
    "Place a {color} stone on the Go board at row {r}, column {c}.",
    "Put a {color} stone at position ({r}, {c}) on the Go board.",
    "Move the {color} stone to row {r}, column {c} on the board.",
    "Set a {color} stone at ({r}, {c}).",
]

SELF, OPPONENT = 0, 1


def _quat_mult_wxyz(q1, q2):
    """Multiply two quaternions in (w,x,y,z) format."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VLA Self-Play")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--max-moves", type=int, default=20)
    p.add_argument("--max-steps-per-move", type=int, default=200)
    p.add_argument("--player1", type=str, default="random", help="black player: random|mcts|katago")
    p.add_argument("--player2", type=str, default="random", help="white player: random|mcts|katago")
    p.add_argument("--katago-path", type=str, default="katago")
    p.add_argument("--katago-model", type=str, default=None)
    p.add_argument("--mcts-simulations", type=int, default=1000)
    p.add_argument("--camera-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--unnorm-key", type=str, default="go_vla_dataset")
    p.add_argument("--load-4bit", action="store_true")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--force-on-failure", action="store_true", default=False)
    p.add_argument("--force-trajectory", action="store_true", help="use scripted controller instead of VLA")
    p.add_argument("--two-arm", action="store_true")
    p.add_argument("--overlay", action="store_true", default=True)
    p.add_argument("--no-overlay", action="store_false", dest="overlay")
    return p.parse_args()


def _draw_overlay(img: np.ndarray, move_num: int, color: str,
                  row: int, col: int, placed: int, total: int) -> np.ndarray:
    """Render text overlay onto a copy of the image."""
    out = img.copy()
    pil = Image.fromarray(out)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 10)
    except (OSError, IOError):
        font = ImageFont.load_default()
    fg = (255, 255, 255)
    draw.text((4, 4), f"Move {move_num} | {color.upper()}", fill=fg, font=font)
    draw.text((4, 18), f"Target: ({row},{col})", fill=fg, font=font)
    h = img.shape[0]
    draw.text((4, h - 16), f"{placed}/{total} placed by VLA", fill=fg, font=font)
    return np.array(pil)


def _build_selector(strategy: str, seed: int, args: argparse.Namespace):
    kwargs = {}
    if strategy == "mcts":
        kwargs["num_simulations"] = args.mcts_simulations
    elif strategy == "katago":
        kwargs["katago_path"] = args.katago_path
        if args.katago_model:
            kwargs["model_path"] = args.katago_model
    return make_selector(strategy, seed=seed, **kwargs)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path).resolve()
    output_dir = Path(args.output_dir) if args.output_dir else model_path / "self_play"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rng = np.random.RandomState(args.seed)

    # --- Load model (base + LoRA adapter) ---
    print(f"Loading model from {model_path}")
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    load_kwargs = dict(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)
    if args.load_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    from peft import PeftModel
    base_vla = AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b", **load_kwargs)
    vla = PeftModel.from_pretrained(base_vla, str(model_path))
    if not args.load_4bit:
        vla = vla.merge_and_unload()
        vla = vla.to(device)

    # Set norm stats on the inner model for PEFT compatibility
    target = vla
    if hasattr(target, "base_model"):
        target = target.base_model
    if hasattr(target, "model"):
        target = target.model
    for sp in [model_path / "dataset_statistics.json"] + [p / "dataset_statistics.json" for p in model_path.parents]:
        if sp.exists():
            with open(sp) as f:
                target.norm_stats = json.load(f)
            print(f"Loaded norm stats from {sp}")
            break
    vla.eval()

    # --- Create environment ---
    print("Creating environment")
    env = create_benchmark_env(
        environment_name="robosuite_go_5x5_rigid_bodies",
        seed=args.seed,
        include_image_obs=True,
        camera_height=args.camera_size,
        camera_width=args.camera_size,
        action_scale=0.03,
        drive_physical_arm=True,
        render_carried_stone=True,
        render_eef_overlay=False,
        enable_opponent_moves=False,
    )

    # --- Move selectors ---
    selectors = {
        0: _build_selector(args.player1, args.seed, args),
        1: _build_selector(args.player2, args.seed + 1, args),
    }
    print(f"Player 1 (black): {args.player1}")
    print(f"Player 2 (white): {args.player2}")

    # --- Initial reset ---
    obs = env.reset(GoResetOptions(opening_moves=0, stone_color="black"))
    # Set self-play player for the first move (black = OpenSpiel player 0)
    env._self_play_player = int(env._logic._state.current_player())

    colors = ["black", "white"]
    frames = []
    game_record = []
    vla_placed = 0

    print(f"\n=== Starting self-play (max {args.max_moves} moves) ===\n")

    for move_num in range(args.max_moves):
        color = colors[move_num % 2]
        player = move_num % 2  # OpenSpiel player ID

        # Set up turn
        if move_num == 0:
            selector = selectors[player]
            row, col = selector.select_move(env._logic, env.board_size)
            env.set_target_intersection(row, col)
            obs = env.get_observation()
        elif hasattr(env, "_next_move_ready") and env._next_move_ready is not None:
            # Already set up by previous iteration's teleport
            color, row, col = env._next_move_ready
            env._next_move_ready = None
        else:
            selector = selectors[player]
            row, col = selector.select_move(env._logic, env.board_size)
            obs = env.continue_game(target_row=row, target_col=col, stone_color=color)

        print(f"Move {move_num}: {color} -> ({row}, {col})")

        # Notify KataGo selectors about the chosen move (both players need to know)
        for sel in selectors.values():
            if isinstance(sel, KataGoMoveSelector):
                sel.notify_move(row, col, color, env.board_size)

        # Build prompt
        template = _INSTRUCTION_TEMPLATES[rng.randint(len(_INSTRUCTION_TEMPLATES))]
        instruction = template.format(color=color, r=row, c=col)
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"

        # Execution loop
        placed = False
        if args.force_trajectory:
            # Use the real scripted collection pipeline
            from go_vla_benchmark.collect import _collect_single_episode
            from go_vla_benchmark.mimicgen_interface import MG_GoJacoSingleMove
            ei = MG_GoJacoSingleMove(env=env)
            episode, placed = _collect_single_episode(
                env=env, env_interface=ei,
                controller_divisor=2.0, detour_steps=0, detour_radius=0.0,
                approach_steps=10, press_steps=6, retreat_steps=6,
                side_transfer_steps=0, side_margin=0.16, recovery_steps=3,
                hover_height_noise=0.2,
            )
            # Extract frames from the episode
            ep_images = episode.observations["agentview_image"]
            step_idx = len(ep_images) - 1
            for t in range(len(ep_images)):
                img = ep_images[t]
                if img.dtype != np.uint8:
                    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                frame = _draw_overlay(img, move_num, color, row, col,
                                      vla_placed, move_num) if args.overlay else img.copy()
                frames.append(frame)
            if placed:
                vla_placed += 1
            obs = env.get_observation()
        else:
            for step_idx in range(args.max_steps_per_move):
                image = obs.get("agentview_image", obs.get("image"))
                if image is not None and image.dtype != np.uint8:
                    image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

                if image is not None:
                    frame = _draw_overlay(image, move_num, color, row, col,
                                          vla_placed, move_num) if args.overlay else image.copy()
                    frames.append(frame)

                # VLA inference
                pil_image = Image.fromarray(image).convert("RGB")
                w, h = pil_image.size
                crop_size = int(min(w, h) * 0.9)
                left = (w - crop_size) // 2
                top = (h - crop_size) // 2
                pil_image = pil_image.crop((left, top, left + crop_size, top + crop_size))
                inputs = processor(prompt, pil_image).to(device, dtype=torch.bfloat16)
                with torch.no_grad():
                    action = vla.predict_action(**inputs, unnorm_key=args.unnorm_key, do_sample=False)

                action = openvla_action_to_benchmark(action, binarize=True)
                obs, reward, done, info = env.step(action)

                if env._move_committed:
                    placed = True
                    vla_placed += 1
                    break

        if not placed:
            if args.force_on_failure:
                print(f"  Force-committing move {move_num}")
                env._commit_target_move_fallback()
            else:
                print(f"  Move {move_num} failed (VLA could not place) — stopping.")
                game_record.append({
                    "move": move_num, "color": color, "row": row, "col": col,
                    "placed_by_vla": False, "steps": step_idx + 1,
                })
                break

        game_record.append({
            "move": move_num,
            "color": color,
            "row": row,
            "col": col,
            "placed_by_vla": placed,
            "steps": step_idx + 1,
        })

        status = "VLA" if placed else "FORCED"
        print(f"  [{status}] steps={step_idx + 1}")

        # Teleport arm home immediately, then hold frames showing the reset board
        if move_num < args.max_moves - 1:
            # Peek at next move to set up continue_game now (teleport happens here)
            next_color = colors[(move_num + 1) % 2]
            next_selector = selectors[(move_num + 1) % 2]
            next_row, next_col = next_selector.select_move(env._logic, env.board_size)
            obs = env.continue_game(target_row=next_row, target_col=next_col, stone_color=next_color)
            # Store for next iteration
            env._next_move_ready = (next_color, next_row, next_col)

        # Hold frames after teleport (pause so viewer can see the reset)
        for _ in range(45):
            hold_img = env.render(mode="rgb_array", height=args.camera_size, width=args.camera_size)
            if args.overlay:
                hold_img = _draw_overlay(hold_img, move_num, color, row, col,
                                         vla_placed, move_num + 1)
            frames.append(hold_img)

        if env._logic.is_game_over:
            print("Game over (OpenSpiel).")
            break

    # --- Save outputs ---
    print(f"\nSaving {len(frames)} frames to video...")
    video_path = str(output_dir / "self_play.mp4")
    writer = imageio.get_writer(video_path, fps=args.fps,
                                codec="libx264", pixelformat="yuv420p",
                                macro_block_size=1)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f"Video saved to {video_path}")

    record_path = str(output_dir / "game_record.json")
    summary = {
        "total_moves": len(game_record),
        "vla_placed": vla_placed,
        "vla_success_rate": vla_placed / max(len(game_record), 1),
        "player1_strategy": args.player1,
        "player2_strategy": args.player2,
        "two_arm": args.two_arm,
        "moves": game_record,
    }
    with open(record_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Game record saved to {record_path}")

    # Cleanup
    for sel in selectors.values():
        sel.close()

    print(f"\nDone. {vla_placed}/{len(game_record)} stones placed by VLA.")


if __name__ == "__main__":
    main()
