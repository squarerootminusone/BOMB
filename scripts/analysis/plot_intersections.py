#!/usr/bin/env python3
"""Plot all 25 board intersections overlaid on the rendered board.

Produces 3 images: 0deg, -3deg, and +3deg board rotation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "go_vla_benchmark"))

from go_vla_benchmark.paths import bootstrap_pythonpath

bootstrap_pythonpath(REPO_ROOT)

from go_vla_benchmark.common import GoResetOptions
from go_vla_benchmark.env_factory import create_benchmark_env


def render_with_intersections(env, label: str, rotation_rad: float, outdir: Path):
    """Reset env with a forced rotation, render frame with all 25 intersections marked."""
    # Reset env normally
    env.reset(options=GoResetOptions(opening_moves=3))

    # Force a specific board rotation by overriding the inner env's value
    # and recomputing the intersection grid
    rs = env._rs_env
    model = rs.sim.model

    # Undo the random rotation and apply our desired one
    old_rot = rs._board_rotation_rad
    desired_rot = rotation_rad

    # Re-run board position randomization with fixed rotation
    # Reset geom positions to shifted (unrotated) state first
    for name, default_pos in rs._default_geom_pos.items():
        try:
            gid = model.geom_name2id(name)
            model.geom_pos[gid, 0] = default_pos[0] + (rs.board_center_xy[0] - rs._board_center_xy_default[0])
            model.geom_pos[gid, 1] = default_pos[1] + (rs.board_center_xy[1] - rs._board_center_xy_default[1])
            if "collision" in name or "visual" in name:
                default_quat = rs._default_geom_quat.get(name)
                if default_quat is not None:
                    model.geom_quat[gid] = default_quat.copy()
        except Exception:
            pass

    # Apply desired rotation
    cos_r = np.cos(desired_rot)
    sin_r = np.sin(desired_rot)
    board_cx = float(rs.board_center_xy[0])
    board_cy = float(rs.board_center_xy[1])
    for name in rs._default_geom_pos.keys():
        try:
            gid = model.geom_name2id(name)
            px = model.geom_pos[gid, 0] - board_cx
            py = model.geom_pos[gid, 1] - board_cy
            model.geom_pos[gid, 0] = cos_r * px - sin_r * py + board_cx
            model.geom_pos[gid, 1] = sin_r * px + cos_r * py + board_cy
            if "collision" in name or "visual" in name:
                default_quat = rs._default_geom_quat.get(name)
                if default_quat is not None:
                    z_rot_quat = np.array([
                        np.cos(desired_rot * 0.5), 0, 0, np.sin(desired_rot * 0.5),
                    ], dtype=np.float64)
                    model.geom_quat[gid] = rs._quat_mul(z_rot_quat, default_quat)
        except Exception:
            pass

    rs._board_rotation_rad = desired_rot
    rs.sim.forward()

    # board_intersections_xyz now reads from geom_xpos directly,
    # so rotation is already included — no manual sync needed.
    env._intersection_xyz = rs.board_intersections_xyz.copy()

    # Render frame
    h, w = env.camera_height, env.camera_width
    image = env.render(mode="rgb_array", height=h, width=w)

    # Get line Z for projection
    board_gid = rs.sim.model.geom_name2id("go_board_surface_visual")
    line_z = float(rs.sim.data.geom_xpos[board_gid, 2])

    # Draw all 25 intersections as colored dots
    for r in range(env.board_size):
        for c in range(env.board_size):
            vis_xyz = env._intersection_xyz[r, c].copy()
            vis_xyz[2] = line_z
            pr, pc = env._world_to_image_rc(vis_xyz, h, w)

            # Color: green for corners, yellow for edges, red for center
            if (r, c) == (2, 2):
                color = np.array([255, 0, 0], dtype=np.uint8)  # center = red
            elif r in (0, 4) and c in (0, 4):
                color = np.array([0, 255, 0], dtype=np.uint8)  # corners = green
            else:
                color = np.array([255, 255, 0], dtype=np.uint8)  # others = yellow

            env._draw_disk(image, row=pr, col=pc, radius=3, color=color)

            # Label with (r,c) as tiny text - just put a smaller dot offset
            # to indicate row: shift dot up by row index
            label_color = np.array([0, 180, 255], dtype=np.uint8)
            env._draw_disk(image, row=max(0, pr - 5), col=pc, radius=1, color=label_color)

    import imageio
    outpath = outdir / f"intersections_{label}.png"
    imageio.imwrite(str(outpath), image)
    print(f"Saved {outpath} (rotation={np.degrees(desired_rot):.1f}deg)")


def main():
    outdir = Path(REPO_ROOT / "benchmarks" / "go_vla_benchmark" / "data" / "intersection_plots")
    outdir.mkdir(exist_ok=True)

    env = create_benchmark_env(
        seed=42,
        environment_name="robosuite_go_5x5_rigid_bodies",
        include_image_obs=True,
        camera_height=256,
        camera_width=256,
        action_scale=0.03,
        success_hold_steps=0,
        drive_physical_arm=True,
        enable_opponent_moves=False,
        opening_with_opponent=True,
        render_carried_stone=True,
        render_eef_overlay=False,
        eef_overlay_trail=0,
        robot="Panda",
        gripper_types="default",
    )

    render_with_intersections(env, "0deg", rotation_rad=0.0, outdir=outdir)
    render_with_intersections(env, "neg3deg", rotation_rad=np.radians(-3.0), outdir=outdir)
    render_with_intersections(env, "pos3deg", rotation_rad=np.radians(3.0), outdir=outdir)

    env.close()
    print(f"\nAll images saved to {outdir}")


if __name__ == "__main__":
    main()
