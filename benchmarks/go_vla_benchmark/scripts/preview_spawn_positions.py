#!/usr/bin/env python3
"""Render 5 demo resets with markers at all initial stone spawn positions.

Each image shows the board after settling, with coloured markers at the
*pre-settle* intersection coordinates where stones were placed during the
random opening.  White-stone spawns are marked in cyan, black-stone spawns
in magenta.  The active (source) stone spawn is marked in orange.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "go_vla_benchmark"))

from go_vla_benchmark.paths import bootstrap_pythonpath

bootstrap_pythonpath(REPO_ROOT)

from go_vla_benchmark.common import GoResetOptions
from go_vla_benchmark.env_factory import create_benchmark_env

SELF = 0
OPPONENT = 1

CYAN = np.array([0, 220, 220], dtype=np.uint8)
MAGENTA = np.array([220, 0, 220], dtype=np.uint8)
ORANGE = np.array([255, 140, 30], dtype=np.uint8)
GREEN = np.array([0, 220, 0], dtype=np.uint8)
MARKER_RADIUS = 5
CAMERA_NAME = "birdview"


def _world_to_image_rc(sim, xyz: np.ndarray, height: int, width: int, cam_name: str = CAMERA_NAME):
    """Project a 3D world point to image (row, col) using the named MuJoCo camera."""
    cam_id = sim.model.camera_name2id(cam_name)
    cam_pos = sim.data.cam_xpos[cam_id]
    cam_mat = sim.data.cam_xmat[cam_id].reshape(3, 3)

    p_world = np.asarray(xyz, dtype=np.float64).ravel()[:3]
    p_cam = cam_mat.T @ (p_world - cam_pos)

    depth = -p_cam[2]
    if depth < 1e-8:
        return height // 2, width // 2

    fovy = sim.model.cam_fovy[cam_id]
    f = (0.5 * height) / np.tan(np.radians(fovy) * 0.5)

    col = f * p_cam[0] / depth + (width - 1) * 0.5
    row = f * (-p_cam[1]) / depth + (height - 1) * 0.5

    col = int(np.clip(round(col), 0, width - 1))
    row = int(np.clip(round(row), 0, height - 1))
    return row, col


def _draw_disk(image: np.ndarray, row: int, col: int, radius: int, color: np.ndarray) -> None:
    h, w = image.shape[:2]
    r0 = max(0, row - radius)
    r1 = min(h - 1, row + radius)
    c0 = max(0, col - radius)
    c1 = min(w - 1, col + radius)
    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            if (rr - row) * (rr - row) + (cc - col) * (cc - col) <= radius * radius:
                image[rr, cc] = color


def main() -> None:
    outdir = REPO_ROOT / "benchmarks" / "go_vla_benchmark" / "data" / "preview_frames"
    outdir.mkdir(exist_ok=True)

    env = create_benchmark_env(
        seed=7,
        environment_name="robosuite_go_5x5_rigid_bodies",
        include_image_obs=True,
        camera_height=600,
        camera_width=600,
        action_scale=0.03,
        success_hold_steps=0,
        drive_physical_arm=True,
        enable_opponent_moves=False,
        opening_with_opponent=True,
        render_carried_stone=False,
        render_eef_overlay=False,
        eef_overlay_trail=0,
        robot="Panda",
        gripper_types="default",
    )

    import imageio

    h, w = 600, 600

    for demo_idx in range(5):
        rng = np.random.RandomState(demo_idx + 100)
        opening_moves = rng.randint(1, 9)

        env.reset(options=GoResetOptions(opening_moves=opening_moves))

        rs = env._rs_env
        sim = rs.sim
        image = sim.render(camera_name=CAMERA_NAME, width=w, height=h)[::-1]

        # Get board visual Z for projection (so markers land on the visible grid)
        try:
            board_gid = sim.model.geom_name2id("go_board_surface_visual")
            line_z = float(sim.data.geom_xpos[board_gid, 2])
        except Exception:
            line_z = env.board_surface_z

        # Draw green dots at ALL intersections (to verify grid alignment)
        for row in range(env.board_size):
            for col in range(env.board_size):
                ix_xyz = env._intersection_xyz[row, col].copy()
                ix_xyz[2] = line_z
                pr, pc = _world_to_image_rc(sim, ix_xyz, h, w)
                _draw_disk(image, row=pr, col=pc, radius=2, color=GREEN)

        # Draw spawn markers for each placed stone
        for (player_id, row, col), _stone_idx in env._stone_assignments.items():
            spawn_xyz = env._intersection_xyz[row, col].copy()
            spawn_xyz[2] = line_z

            pr, pc = _world_to_image_rc(sim, spawn_xyz, h, w)
            color = CYAN if int(player_id) == SELF else MAGENTA
            _draw_disk(image, row=pr, col=pc, radius=MARKER_RADIUS, color=color)

        # Mark active (source) stone spawn position
        if env._active_white_stone_idx is not None:
            src_xyz = env._source_xyz.copy()
            src_xyz[2] = line_z
            pr, pc = _world_to_image_rc(sim, src_xyz, h, w)
            _draw_disk(image, row=pr, col=pc, radius=MARKER_RADIUS, color=ORANGE)

        outpath = outdir / f"demo_{demo_idx}_spawn.png"
        imageio.imwrite(str(outpath), image)
        n_stones = len(env._stone_assignments)
        print(f"Demo {demo_idx}: {opening_moves} opening moves, {n_stones} stones placed -> {outpath.name}")

    env.close()
    print(f"\nAll previews saved to {outdir}")


if __name__ == "__main__":
    main()
