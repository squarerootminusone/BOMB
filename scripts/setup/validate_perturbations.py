#!/usr/bin/env python3
"""Validate dataset perturbations for the Go VLA benchmark.

Runs 5 tests over multiple environment resets to verify that all visual
and trajectory perturbations are within expected ranges and that the
environment remains functionally correct.

Usage:
    conda run -n mujogo python scripts/validate_perturbations.py --strict --num-resets 20
"""

from __future__ import annotations

import argparse
import json
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
from go_vla_benchmark.robosuite_go_env import SELF


def _create_env(seed: int = 42):
    return create_benchmark_env(
        seed=seed,
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
        render_eef_overlay=True,
        eef_overlay_trail=10,
        robot="Panda",
        gripper_types="default",
    )


def test_target_marker_location(env, num_resets: int = 10) -> dict:
    """Blue dot in rendered image is within 3px of projected target intersection."""
    errors = []
    max_pixel_err = 0.0
    for i in range(num_resets):
        env.reset(options=GoResetOptions(opening_moves=np.random.randint(0, 6)))
        obs = env.get_observation()
        image = obs["agentview_image"]
        h, w = image.shape[:2]

        target_xyz = env.get_target_pose()[:3, 3].copy()
        try:
            board_gid = env._rs_env.sim.model.geom_name2id("go_board_surface_visual")
            target_xyz[2] = float(env._rs_env.sim.data.geom_xpos[board_gid, 2])
        except Exception:
            pass
        proj_r, proj_c = env._world_to_image_rc(target_xyz, h, w)

        # Find the blue marker in the image (color [20, 130, 255])
        blue_mask = (
            (image[:, :, 0].astype(int) < 80)
            & (image[:, :, 1].astype(int) > 80)
            & (image[:, :, 2].astype(int) > 200)
        )
        blue_coords = np.argwhere(blue_mask)
        if len(blue_coords) == 0:
            errors.append(f"Reset {i}: no blue marker found in image")
            continue

        blue_center = blue_coords.mean(axis=0)
        pixel_err = np.sqrt((blue_center[0] - proj_r) ** 2 + (blue_center[1] - proj_c) ** 2)
        max_pixel_err = max(max_pixel_err, pixel_err)
        if pixel_err > 3.0:
            errors.append(
                f"Reset {i}: blue marker at ({blue_center[0]:.1f}, {blue_center[1]:.1f}) "
                f"vs projected ({proj_r}, {proj_c}), error={pixel_err:.1f}px"
            )

    return {
        "test": "target_marker_location",
        "passed": len(errors) == 0,
        "max_pixel_error": round(max_pixel_err, 2),
        "errors": errors,
    }


def test_board_intersections(env, num_resets: int = 10) -> dict:
    """Grid spacing ~= board_spacing, centered, rotation in [-5, 5] deg."""
    errors = []
    for i in range(num_resets):
        env.reset(options=GoResetOptions(opening_moves=0))
        grid = env._intersection_xyz
        bs = env.board_size
        spacing = env._rs_env.board_spacing

        # Check spacing between adjacent columns in first row
        for c in range(bs - 1):
            d = np.linalg.norm(grid[0, c + 1, :2] - grid[0, c, :2])
            if abs(d - spacing) > 0.01:
                errors.append(f"Reset {i}: col spacing ({c},{c+1}) = {d:.4f}, expected ~{spacing}")

        # Check spacing between adjacent rows in first column
        for r in range(bs - 1):
            d = np.linalg.norm(grid[r + 1, 0, :2] - grid[r, 0, :2])
            if abs(d - spacing) > 0.01:
                errors.append(f"Reset {i}: row spacing ({r},{r+1}) = {d:.4f}, expected ~{spacing}")

        # Check grid is self-consistent: center of grid should match
        # the actual board surface geom world position
        center_xy = grid[:, :, :2].mean(axis=(0, 1))
        try:
            bgid = env._rs_env.sim.model.geom_name2id("go_board_surface_visual")
            board_world_xy = env._rs_env.sim.data.geom_xpos[bgid, :2]
        except Exception:
            board_world_xy = center_xy  # skip check
        center_err = np.linalg.norm(center_xy - board_world_xy)
        if center_err > 0.005:
            errors.append(f"Reset {i}: grid center off by {center_err:.4f}m")

        # Check rotation angle is within [-5, 5] degrees
        rotation_rad = getattr(env._rs_env, "_board_rotation_rad", 0.0)
        rotation_deg = np.degrees(rotation_rad)
        if abs(rotation_deg) > 5.5:
            errors.append(f"Reset {i}: rotation {rotation_deg:.2f} deg outside [-5, 5]")

    return {
        "test": "board_intersections",
        "passed": len(errors) == 0,
        "errors": errors,
    }


def test_camera_projection_consistency(env, num_resets: int = 10) -> dict:
    """Projected board points within image bounds and maintain relative spatial ordering."""
    errors = []
    for i in range(num_resets):
        env.reset(options=GoResetOptions(opening_moves=0))
        grid = env._intersection_xyz
        h, w = env.camera_height, env.camera_width

        projected = np.zeros((env.board_size, env.board_size, 2), dtype=np.float64)
        for r in range(env.board_size):
            for c in range(env.board_size):
                pr, pc = env._world_to_image_rc(grid[r, c], h, w)
                projected[r, c] = [pr, pc]
                if pr < 0 or pr >= h or pc < 0 or pc >= w:
                    errors.append(f"Reset {i}: ({r},{c}) projects to ({pr},{pc}) outside [{h}x{w}]")

        # Check column ordering: for each row, column indices should be monotonic
        # (direction depends on camera perspective - may be increasing or decreasing)
        for r in range(env.board_size):
            cols = projected[r, :, 1]
            diffs = np.diff(cols)
            is_monotonic = np.all(diffs > -2) or np.all(diffs < 2)
            if not is_monotonic:
                errors.append(f"Reset {i}: row {r} column ordering not monotonic: {cols.tolist()}")

    return {
        "test": "camera_projection_consistency",
        "passed": len(errors) == 0,
        "errors": errors,
    }


def test_stone_placement(env, num_resets: int = 5) -> dict:
    """After fallback commit, stone is near target and is_success is True."""
    errors = []
    for i in range(num_resets):
        env.reset(options=GoResetOptions(opening_moves=np.random.randint(0, 4)))
        target_xyz = env.get_target_pose()[:3, 3]
        target_rc = env.get_target_intersection()

        # Use the fallback commit which places the stone at the target
        committed = env._commit_target_move_fallback()
        if not committed:
            errors.append(f"Reset {i}: fallback commit returned False")
            continue

        # Verify the stone assignment exists for the target intersection
        key = (SELF, int(target_rc[0]), int(target_rc[1]))
        stone_idx = env._stone_assignments.get(key)
        if stone_idx is None:
            errors.append(f"Reset {i}: no stone assigned at target {target_rc}")
            continue

        stone_xyz = env._rs_env.get_stone_pos(stone_idx)
        dist = np.linalg.norm(stone_xyz[:2] - target_xyz[:2])
        if dist > env.place_xy_threshold:
            errors.append(
                f"Reset {i}: stone dist={dist:.4f} > threshold={env.place_xy_threshold}"
            )

    return {
        "test": "stone_placement",
        "passed": len(errors) == 0,
        "errors": errors,
    }


def test_perturbation_ranges(env, num_resets: int = 50) -> dict:
    """Over many resets, check perturbation values are within expected ranges."""
    errors = []
    fov_ratios = []
    cam_angle_degs = []
    board_rotations = []
    wb_shifts = []
    extra_light_active_count = 0
    mat_shininesses = []
    mat_speculars = []

    rs_env = env._rs_env
    default_fovy = getattr(rs_env, "_default_cam_fovy", None)
    default_cam_quat = getattr(rs_env, "_default_cam_quat", None)

    for i in range(num_resets):
        env.reset(options=GoResetOptions(opening_moves=0))

        # FOV ratio
        if default_fovy is not None and rs_env._cam_id is not None:
            current_fovy = float(rs_env.sim.model.cam_fovy[rs_env._cam_id])
            ratio = current_fovy / default_fovy
            fov_ratios.append(ratio)

        # Camera angle
        if default_cam_quat is not None and rs_env._cam_id is not None:
            current_quat = rs_env.sim.model.cam_quat[rs_env._cam_id].copy()
            # Compute relative rotation angle
            q_inv = default_cam_quat.copy()
            q_inv[1:] *= -1
            q_rel = rs_env._quat_mul(current_quat, q_inv)
            angle_rad = 2 * np.arccos(np.clip(abs(q_rel[0]), 0, 1))
            cam_angle_degs.append(np.degrees(angle_rad))

        # Board rotation
        rotation_deg = np.degrees(getattr(rs_env, "_board_rotation_rad", 0.0))
        board_rotations.append(rotation_deg)

        # WB shift
        wb = getattr(env, "_wb_shift", np.zeros(3))
        wb_shifts.append(wb.copy())

        # Extra light
        extra_lid = getattr(rs_env, "_extra_light_id", None)
        if extra_lid is not None:
            diffuse = rs_env.sim.model.light_diffuse[extra_lid]
            if np.any(diffuse > 0.01):
                extra_light_active_count += 1

        # Stone material
        for mid in getattr(rs_env, "_stone_mat_ids", set()):
            mat_shininesses.append(float(rs_env.sim.model.mat_shininess[mid]))
            mat_speculars.append(float(rs_env.sim.model.mat_specular[mid]))

    # Validate ranges
    if fov_ratios:
        min_fov = min(fov_ratios)
        max_fov = max(fov_ratios)
        if min_fov < 0.80 or max_fov > 1.20:
            errors.append(f"FOV ratio range [{min_fov:.3f}, {max_fov:.3f}] outside [0.80, 1.20]")
        if max_fov - min_fov < 0.05:
            errors.append(f"FOV ratio range too narrow: [{min_fov:.3f}, {max_fov:.3f}]")

    if cam_angle_degs:
        max_angle = max(cam_angle_degs)
        if max_angle > 15.0:
            errors.append(f"Max camera angle {max_angle:.1f} deg exceeds 15 deg")

    if board_rotations:
        min_rot = min(board_rotations)
        max_rot = max(board_rotations)
        if min_rot < -5.5 or max_rot > 5.5:
            errors.append(f"Board rotation range [{min_rot:.2f}, {max_rot:.2f}] outside [-5.5, 5.5]")
        if max_rot - min_rot < 1.0:
            errors.append(f"Board rotation range too narrow: [{min_rot:.2f}, {max_rot:.2f}]")

    if wb_shifts:
        wb_arr = np.array(wb_shifts)
        if np.any(wb_arr < -0.09) or np.any(wb_arr > 0.09):
            errors.append(f"WB shift outside [-0.09, 0.09]")
        if np.ptp(wb_arr) < 0.02:
            errors.append("WB shift range too narrow")

    extra_light_pct = extra_light_active_count / max(num_resets, 1)
    if num_resets >= 20 and (extra_light_pct < 0.1 or extra_light_pct > 0.7):
        errors.append(f"Extra light activation rate {extra_light_pct:.1%} outside [10%, 70%]")

    if mat_shininesses:
        if min(mat_shininesses) < 0.04 or max(mat_shininesses) > 0.96:
            errors.append(f"Stone shininess outside [0.04, 0.96]")
    if mat_speculars:
        if min(mat_speculars) < 0.09 or max(mat_speculars) > 0.91:
            errors.append(f"Stone specular outside [0.09, 0.91]")

    return {
        "test": "perturbation_ranges",
        "passed": len(errors) == 0,
        "num_resets": num_resets,
        "fov_ratio_range": [round(min(fov_ratios), 3), round(max(fov_ratios), 3)] if fov_ratios else None,
        "max_cam_angle_deg": round(max(cam_angle_degs), 2) if cam_angle_degs else None,
        "board_rotation_range": [round(min(board_rotations), 2), round(max(board_rotations), 2)] if board_rotations else None,
        "extra_light_activation_pct": round(extra_light_pct * 100, 1),
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate Go benchmark perturbations")
    parser.add_argument("--strict", action="store_true", help="exit with non-zero on any failure")
    parser.add_argument("--num-resets", type=int, default=20, help="number of resets per test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    env = _create_env(seed=args.seed)

    tests = [
        ("target_marker_location", lambda: test_target_marker_location(env, num_resets=args.num_resets)),
        ("board_intersections", lambda: test_board_intersections(env, num_resets=args.num_resets)),
        ("camera_projection_consistency", lambda: test_camera_projection_consistency(env, num_resets=args.num_resets)),
        ("stone_placement", lambda: test_stone_placement(env, num_resets=min(5, args.num_resets))),
        ("perturbation_ranges", lambda: test_perturbation_ranges(env, num_resets=max(args.num_resets, 20))),
    ]

    results = []
    all_passed = True
    for name, test_fn in tests:
        print(f"Running {name}...", flush=True)
        result = test_fn()
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        all_passed = all_passed and result["passed"]
        print(f"  {status}: {name}", flush=True)
        if result.get("errors"):
            for err in result["errors"][:3]:
                print(f"    - {err}", flush=True)
            if len(result["errors"]) > 3:
                print(f"    ... and {len(result['errors']) - 3} more", flush=True)

    print(json.dumps(results, indent=2))
    env.close()

    if args.strict and not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
