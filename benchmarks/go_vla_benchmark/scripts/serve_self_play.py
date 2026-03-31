#!/usr/bin/env python3
"""FastAPI server that wraps the VLA self-play loop for real-time WebSocket streaming.

Why: Enables a web frontend to observe and control Go self-play games in
real time, without requiring the viewer to run heavy ML inference locally.

How: Loads the VLA model and robosuite environment once at startup. A
GameSession thread runs the same inference loop as self_play.py and pushes
JPEG-encoded frames (with JSON metadata) into a thread-safe deque. An
asyncio broadcast loop drains the deque and fans frames out to all
connected WebSocket clients.

Usage:
    MUJOCO_GL=egl conda run --no-capture-output -n mujogo python \
        benchmarks/go_vla_benchmark/scripts/serve_self_play.py \
        --model-path outputs/<date>/<time>/checkpoints/best \
        --load-4bit --port 8765
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import io
import json
import os
import struct
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Path setup (mirrors self_play.py exactly)
# ---------------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
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

from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
)

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import (
    PrismaticImageProcessor,
    PrismaticProcessor,
)

# ---------------------------------------------------------------------------
# Instruction templates (must match training data exactly)
# ---------------------------------------------------------------------------

_INSTRUCTION_TEMPLATES = [
    "Place a {color} stone on the Go board at row {r}, column {c}.",
    "Put a {color} stone at position ({r}, {c}) on the Go board.",
    "Move the {color} stone to row {r}, column {c} on the board.",
    "Set a {color} stone at ({r}, {c}).",
]

SELF, OPPONENT = 0, 1


def _find_dataset_statistics_path(model_path: Path) -> Path | None:
    candidates = [model_path / "dataset_statistics.json"]
    candidates.extend(parent / "dataset_statistics.json" for parent in model_path.parents)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the self-play server.

    Why: Reuses the same knobs as self_play.py so operators can configure
    the model and environment identically.
    """
    p = argparse.ArgumentParser(description="VLA Self-Play WebSocket Server")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--load-4bit", action="store_true")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--camera-size", type=int, default=256)
    p.add_argument("--unnorm-key", type=str, default="go_vla_dataset")
    p.add_argument("--max-steps-per-move", type=int, default=200)
    p.add_argument("--force-on-failure", action="store_true", default=True)
    p.add_argument("--no-force-on-failure", action="store_false", dest="force_on_failure")
    p.add_argument("--katago-path", type=str, default="katago")
    p.add_argument("--katago-model", type=str, default=None)
    p.add_argument("--mcts-simulations", type=int, default=1000)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Frame encoding
# ---------------------------------------------------------------------------


def encode_frame(image_np: np.ndarray, metadata: dict) -> str:
    """Encode a frame as a JSON string with base64-embedded JPEG.

    Returns a JSON text message (not binary) so WebSocket transport is simpler.
    """
    import base64
    buf = io.BytesIO()
    Image.fromarray(image_np).save(buf, format="JPEG", quality=80)
    jpeg_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    metadata["image"] = jpeg_b64
    return json.dumps(metadata, separators=(",", ":"))


# ---------------------------------------------------------------------------
# GameSession (runs in a background thread)
# ---------------------------------------------------------------------------


class GameSession:
    """Manages a single self-play game as an async coroutine.

    Runs entirely on the main thread so MuJoCo rendering (which requires
    the OpenGL/EGL context) works correctly. VLA inference blocks for
    ~300ms per step but that's acceptable — frames are sent between steps.
    """

    def __init__(
        self,
        vla: Any,
        processor: Any,
        env: Any,
        config: dict,
        broadcast_fn: Callable[[str], Any],
        device: torch.device,
        cli_args: argparse.Namespace,
    ) -> None:
        self._vla = vla
        self._processor = processor
        self._env = env
        self._config = config
        self._broadcast_fn = broadcast_fn
        self._device = device
        self._cli_args = cli_args

        self._paused = False
        self._stop_flag = False

        self._status = "idle"
        self._move_num = 0
        self._vla_placed = 0
        self._total_moves = 0

        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        self._task = asyncio.ensure_future(self.run())

    def pause(self) -> None:
        self._paused = True
        self._status = "paused"

    def resume(self) -> None:
        self._paused = False
        if self._status == "paused":
            self._status = "running"

    def stop(self) -> None:
        self._stop_flag = True
        self._paused = False

    @property
    def status(self) -> str:
        return self._status

    @property
    def move_num(self) -> int:
        return self._move_num

    @property
    def vla_placed(self) -> int:
        return self._vla_placed

    @property
    def total_moves(self) -> int:
        return self._total_moves

    async def _send_frame(self, image: np.ndarray, meta: dict) -> None:
        """Encode and broadcast a frame, then yield to the event loop."""
        encoded = encode_frame(image, meta)
        manager.latest_frame = encoded
        await manager.broadcast(encoded)

    async def run(self) -> None:
        self._status = "running"
        env = self._env
        cfg = self._config
        max_moves: int = cfg.get("max_moves", 20)
        seed: int = cfg.get("seed", 42)
        player1: str = cfg.get("player1", "random")
        player2: str = cfg.get("player2", "random")

        rng = np.random.RandomState(seed)
        camera_size = self._cli_args.camera_size

        selectors = {
            0: self._build_selector(player1, seed),
            1: self._build_selector(player2, seed + 1),
        }

        colors = ["black", "white"]
        self._vla_placed = 0

        try:
            for move_num in range(max_moves):
                if self._stop_flag:
                    break

                # Pause support
                while self._paused and not self._stop_flag:
                    await asyncio.sleep(0.1)
                if self._stop_flag:
                    break

                self._move_num = move_num
                color = colors[move_num % 2]
                player = move_num % 2

                selector = selectors[player]
                row, col = selector.select_move(env._logic, env.board_size)

                if move_num == 0:
                    env.set_target_intersection(row, col)
                else:
                    obs = env.continue_game(target_row=row, target_col=col, stone_color=color)

                for sel in selectors.values():
                    if isinstance(sel, KataGoMoveSelector):
                        sel.notify_move(row, col, color, env.board_size)

                template = _INSTRUCTION_TEMPLATES[rng.randint(len(_INSTRUCTION_TEMPLATES))]
                instruction = template.format(color=color, r=row, c=col)
                prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"

                if move_num == 0:
                    obs = env.get_observation()

                placed = False
                for step_idx in range(self._cli_args.max_steps_per_move):
                    if self._stop_flag:
                        break
                    while self._paused and not self._stop_flag:
                        await asyncio.sleep(0.1)
                    if self._stop_flag:
                        break

                    image = obs.get("agentview_image", obs.get("image"))
                    if image is not None and image.dtype != np.uint8:
                        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

                    # Broadcast frame then yield to event loop
                    if image is not None:
                        meta = {
                            "type": "frame",
                            "board_state": env.get_board_state().tolist(),
                            "current_player": int(env._logic._state.current_player()),
                            "move_num": move_num,
                            "color": color,
                            "target_rc": [row, col],
                            "steps_taken": step_idx,
                            "is_game_over": env._logic.is_game_over,
                            "vla_placed_count": self._vla_placed,
                            "total_moves": move_num + 1,
                            "frame_type": "inference",
                            "game_status": "running",
                            "move_committed": env._move_committed,
                        }
                        await self._send_frame(image, meta)

                    # VLA inference (blocking, ~300ms)
                    pil_image = Image.fromarray(image).convert("RGB")
                    w, h = pil_image.size
                    crop_size = int(min(w, h) * 0.9)
                    left = (w - crop_size) // 2
                    top = (h - crop_size) // 2
                    pil_image = pil_image.crop((left, top, left + crop_size, top + crop_size))
                    inputs = self._processor(prompt, pil_image).to(
                        self._device, dtype=torch.bfloat16
                    )
                    with torch.no_grad():
                        action = self._vla.predict_action(
                            **inputs,
                            unnorm_key=self._cli_args.unnorm_key,
                            do_sample=False,
                        )

                    action = openvla_action_to_benchmark(action, binarize=True)
                    obs, reward, done, info = env.step(action)

                    if env._move_committed:
                        placed = True
                        self._vla_placed += 1
                        break

                if not placed and not self._stop_flag:
                    if self._cli_args.force_on_failure:
                        env._commit_target_move_fallback()

                # Hold frames
                for _ in range(15):
                    if self._stop_flag:
                        break
                    hold_img = env.render(mode="rgb_array", height=camera_size, width=camera_size)
                    meta = {
                        "type": "frame",
                        "board_state": env.get_board_state().tolist(),
                        "current_player": int(env._logic._state.current_player()),
                        "move_num": move_num,
                        "color": color,
                        "target_rc": [row, col],
                        "steps_taken": step_idx + 1 if not self._stop_flag else 0,
                        "is_game_over": env._logic.is_game_over,
                        "vla_placed_count": self._vla_placed,
                        "total_moves": move_num + 1,
                        "frame_type": "hold",
                        "game_status": "running",
                        "move_committed": True,
                    }
                    await self._send_frame(hold_img, meta)
                    await asyncio.sleep(1.0 / 30)

                self._total_moves = move_num + 1
                if env._logic.is_game_over:
                    break

            final_img = env.render(mode="rgb_array", height=camera_size, width=camera_size)
            final_meta = {
                "type": "frame",
                "board_state": env.get_board_state().tolist(),
                "current_player": int(env._logic._state.current_player()),
                "move_num": self._move_num,
                "color": colors[self._move_num % 2],
                "target_rc": [0, 0],
                "steps_taken": 0,
                "is_game_over": env._logic.is_game_over,
                "vla_placed_count": self._vla_placed,
                "total_moves": self._total_moves,
                "frame_type": "final",
                "game_status": "finished",
                "move_committed": False,
            }
            await self._send_frame(final_img, final_meta)

        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"GameSession error: {exc}", flush=True)
        finally:
            for sel in selectors.values():
                sel.close()
            self._status = "finished"

    def _build_selector(self, strategy: str, seed: int) -> Any:
        kwargs: dict[str, Any] = {}
        if strategy == "mcts":
            kwargs["num_simulations"] = self._cli_args.mcts_simulations
        elif strategy == "katago":
            kwargs["katago_path"] = self._cli_args.katago_path
            if self._cli_args.katago_model:
                kwargs["model_path"] = self._cli_args.katago_model
        return make_selector(strategy, seed=seed, **kwargs)


# ---------------------------------------------------------------------------
# ConnectionManager
# ---------------------------------------------------------------------------


class ConnectionManager:
    """Track active WebSocket connections and broadcast binary frames.

    Why: Multiple browser tabs may observe the same game simultaneously.
    A central manager avoids per-client polling logic.

    How: Maintains a set of active connections. On broadcast, iterates
    the set and silently removes any connection that raises an exception.
    Caches the latest frame so newly-connected clients see something
    immediately.
    """

    def __init__(self) -> None:
        self.active_connections: set[WebSocket] = set()
        self.latest_frame: Optional[str] = None

    async def connect(self, websocket: WebSocket) -> None:
        """Accept a WebSocket and send the most recent frame if available."""
        await websocket.accept()
        self.active_connections.add(websocket)
        if self.latest_frame is not None:
            try:
                await websocket.send_text(self.latest_frame)
            except Exception:
                self.active_connections.discard(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket from the active set."""
        self.active_connections.discard(websocket)

    async def broadcast(self, data: str) -> None:
        """Send text data to every connected client.

        Silently removes clients that have disconnected.
        """
        dead: list[WebSocket] = []
        for ws in list(self.active_connections):
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.active_connections.discard(ws)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class GameStartRequest(BaseModel):
    """Body for POST /api/game/start."""

    player1: str = "random"
    player2: str = "random"
    max_moves: int = 20
    two_arm: bool = False
    seed: int = 42


class StatusResponse(BaseModel):
    """Response for GET /api/game/status."""

    status: str
    move_num: int
    vla_placed: int
    total_moves: int


# ---------------------------------------------------------------------------
# Global state (populated at startup, before requests are served)
# ---------------------------------------------------------------------------

args: argparse.Namespace
vla: Any = None
processor: Any = None
env: Any = None
device: torch.device
manager = ConnectionManager()
session: Optional[GameSession] = None
frame_queue: collections.deque = collections.deque(maxlen=2)

# Single-operator lock: only one WebSocket client can control at a time
_active_ws: Optional[WebSocket] = None
_active_ws_last_activity: float = 0.0
_INACTIVITY_TIMEOUT = 300  # 5 minutes


def _refresh_activity() -> None:
    global _active_ws_last_activity
    _active_ws_last_activity = time.monotonic()


# ---------------------------------------------------------------------------
# Broadcast loop (asyncio background task)
# ---------------------------------------------------------------------------


async def broadcast_loop() -> None:
    """Drain the frame deque and push frames to all WebSocket clients.

    Why: The GameSession thread cannot await coroutines. This async task
    bridges the thread-safe deque to the async WebSocket layer.

    How: Polls the deque at ~60 Hz. When no game is running, sends an
    idle frame once per second so clients always have a live camera view.
    """
    while True:
        # Send idle frames when no game is active
        if (session is None or session.status in ("idle", "finished")) and env is not None and manager.active_connections:
            try:
                idle_img = env.render(
                    mode="rgb_array",
                    height=args.camera_size,
                    width=args.camera_size,
                )
                idle_meta = {
                    "type": "frame",
                    "board_state": env.get_board_state().tolist(),
                    "current_player": int(env._logic._state.current_player()),
                    "move_num": 0,
                    "color": "black",
                    "target_rc": [0, 0],
                    "steps_taken": 0,
                    "is_game_over": False,
                    "vla_placed_count": 0,
                    "total_moves": 0,
                    "frame_type": "idle",
                    "game_status": "idle",
                    "move_committed": False,
                }
                idle_frame = encode_frame(idle_img, idle_meta)
                manager.latest_frame = idle_frame
                await manager.broadcast(idle_frame)
            except Exception:
                pass
        await asyncio.sleep(1.0)


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model/env, then start the broadcast loop."""
    _startup()
    task = asyncio.create_task(broadcast_loop())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -- HTTP endpoints ---------------------------------------------------------


@app.get("/api/health")
async def health() -> dict:
    """Basic liveness probe."""
    return {"status": "ok", "model_loaded": vla is not None, "env_ready": env is not None}


@app.get("/api/debug/frame")
async def debug_frame():
    """Return a single JPEG frame for debugging."""
    from fastapi.responses import Response
    if env is None:
        return Response(content=b"no env", status_code=503)
    img = env.render(mode="rgb_array", height=args.camera_size, width=args.camera_size)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=80)
    return Response(content=buf.getvalue(), media_type="image/jpeg")


@app.get("/api/game/status", response_model=StatusResponse)
async def game_status() -> StatusResponse:
    """Return the current game session status."""
    if session is None:
        return StatusResponse(status="idle", move_num=0, vla_placed=0, total_moves=0)
    return StatusResponse(
        status=session.status,
        move_num=session.move_num,
        vla_placed=session.vla_placed,
        total_moves=session.total_moves,
    )


@app.post("/api/game/start")
async def start_game(config: GameStartRequest) -> dict:
    """Start a new self-play game (stops any running game first).

    Why: The viewer UI needs a single endpoint to kick off a game with
    configurable players, move limits, and RNG seed.

    How: Stops any existing session, resets the environment, then spawns
    a new GameSession thread.
    """
    global session

    _refresh_activity()

    if session is not None and session.status in ("running", "paused"):
        session.stop()
        await asyncio.sleep(0.5)  # let the coroutine finish

    # Reset environment
    env.reset(GoResetOptions(opening_moves=0, stone_color="black"))
    env._self_play_player = int(env._logic._state.current_player())

    session = GameSession(
        vla=vla,
        processor=processor,
        env=env,
        config=config.model_dump(),
        broadcast_fn=None,  # unused — session broadcasts directly
        device=device,
        cli_args=args,
    )
    session.start()
    return {"status": "starting"}


@app.post("/api/game/pause")
async def pause_game() -> dict:
    """Pause the running game."""
    _refresh_activity()
    if session is None or session.status != "running":
        return {"status": "no_game_running"}
    session.pause()
    return {"status": "paused"}


@app.post("/api/game/resume")
async def resume_game() -> dict:
    """Resume a paused game."""
    _refresh_activity()
    if session is None or session.status != "paused":
        return {"status": "not_paused"}
    session.resume()
    return {"status": "running"}


@app.post("/api/game/stop")
async def stop_game() -> dict:
    """Stop the running game."""
    _refresh_activity()
    if session is None or session.status not in ("running", "paused"):
        return {"status": "no_game_running"}
    session.stop()
    return {"status": "stopped"}


# -- WebSocket endpoint -----------------------------------------------------


@app.websocket("/ws/stream")
async def stream(websocket: WebSocket) -> None:
    """Single-operator WebSocket: only one client can connect at a time.

    If the slot is occupied (and not timed out), the new client receives
    an error message and is disconnected. After 5 min of inactivity the
    slot auto-releases.
    """
    global _active_ws, _active_ws_last_activity

    # Check if slot is occupied and not timed out
    if _active_ws is not None:
        elapsed = time.monotonic() - _active_ws_last_activity
        if elapsed < _INACTIVITY_TIMEOUT:
            await websocket.accept()
            await websocket.send_text(json.dumps({"type": "error", "message": "occupied"}))
            await websocket.close(code=4001, reason="occupied")
            return
        else:
            # Stale connection — evict it
            try:
                await _active_ws.close(code=4002, reason="inactivity_timeout")
            except Exception:
                pass
            manager.disconnect(_active_ws)
            _active_ws = None

    # Claim the slot
    _active_ws = websocket
    _active_ws_last_activity = time.monotonic()
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            _active_ws_last_activity = time.monotonic()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        if _active_ws is websocket:
            _active_ws = None
    except Exception:
        manager.disconnect(websocket)
        if _active_ws is websocket:
            _active_ws = None


# ---------------------------------------------------------------------------
# Startup: load model and create environment
# ---------------------------------------------------------------------------


def _startup() -> None:
    """Load the VLA model, processor, and robosuite environment.

    Why: These are expensive to initialise (several GB of GPU memory, MuJoCo
    compilation) and must only happen once.

    How: Mirrors self_play.py's model loading and env creation exactly.
    Populates module-level globals so route handlers can use them.
    """
    global args, vla, processor, env, device

    args = parse_args()
    model_path = Path(args.model_path).resolve()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- Load model ---
    print(f"Loading model from {model_path}")
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    load_kwargs: dict[str, Any] = dict(
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    )
    if args.load_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    vla = AutoModelForVision2Seq.from_pretrained(str(model_path), **load_kwargs)
    if not args.load_4bit:
        vla = vla.to(device)
    stats_path = _find_dataset_statistics_path(model_path)
    if stats_path is not None:
        with open(stats_path) as f:
            vla.norm_stats = json.load(f)
        print(f"Loaded norm stats from {stats_path}")
    else:
        print("WARNING: dataset_statistics.json not found; actions may not be denormalized correctly")
    vla.eval()

    # --- Create environment ---
    print("Creating environment")
    env = create_benchmark_env(
        environment_name="robosuite_go_5x5_rigid_bodies",
        seed=42,
        include_image_obs=True,
        camera_height=args.camera_size,
        camera_width=args.camera_size,
        action_scale=0.03,
        drive_physical_arm=True,
        render_carried_stone=True,
        render_eef_overlay=False,
        enable_opponent_moves=False,
        robot="Panda",
        env_configuration="default",
    )

    # Initial reset so the env is in a valid state for idle frames
    env.reset(GoResetOptions(opening_moves=0, stone_color="black"))
    env._self_play_player = int(env._logic._state.current_player())

    print(f"Server ready. Listening on port {args.port}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    # Parse args early so we have the port, but defer heavy loading to lifespan
    # (uvicorn binds the port first, then lifespan runs _startup)
    args = parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
