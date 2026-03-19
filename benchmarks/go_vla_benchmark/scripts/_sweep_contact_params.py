#!/usr/bin/env python3
"""Sweep contact parameters and measure stone sinking + lateral drift."""
from __future__ import annotations
import argparse, json, subprocess, sys, numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "benchmarks" / "go_vla_benchmark"))
from go_vla_benchmark.paths import bootstrap_pythonpath
bootstrap_pythonpath(REPO)


def run(
    solref: float,
    solimp: float,
    margin: float,
    rolling_friction: float,
    torsional_friction: float,
    damping: float,
    seed: int = 42,
):
    import go_vla_benchmark.robosuite_go_env as mod

    # Monkey-patch _patch_physics_params with our values
    orig_patch = mod._Go5x5RigidRobosuite._patch_physics_params
    def custom_patch(self):
        model = self.sim.model
        model.opt.integrator = 2
        model.opt.timestep = 0.0005
        self.model_timestep = model.opt.timestep
        model.opt.solver = 2
        model.opt.iterations = 500
        model.opt.tolerance = 1e-10
        model.opt.noslip_iterations = 100
        model.opt.noslip_tolerance = 1e-9

        solref_arr = [solref, 1.0]
        solimp_arr = [solimp, solimp, 0.001, 0.5, 2.0]

        for obj in self._stone_objects:
            body_id = model.body_name2id(obj.root_body)
            for gid in range(model.ngeom):
                if model.geom_bodyid[gid] == body_id:
                    model.geom_condim[gid] = 4
                    model.geom_friction[gid] = [1.5, torsional_friction, rolling_friction]
                    model.geom_solref[gid] = solref_arr
                    model.geom_solimp[gid] = solimp_arr
                    model.geom_margin[gid] = margin
                    model.geom_gap[gid] = 0.0
        try:
            bid = model.geom_name2id("go_board_surface_collision")
            model.geom_condim[bid] = 4
            model.geom_friction[bid] = [1.5, torsional_friction, rolling_friction]
            model.geom_solref[bid] = solref_arr
            model.geom_solimp[bid] = solimp_arr
        except Exception:
            pass
        try:
            for n in ["table_collision", "table_visual"]:
                gid = model.geom_name2id(n)
                model.geom_condim[gid] = 4
                model.geom_friction[gid] = [1.5, torsional_friction, rolling_friction]
                model.geom_solref[gid] = solref_arr
                model.geom_solimp[gid] = solimp_arr
        except Exception:
            pass
        for jn in self._stone_joint_names:
            jid = model.joint_name2id(jn)
            dof = model.jnt_dofadr[jid]
            for i in range(6):
                model.dof_damping[dof + i] = damping

    mod._Go5x5RigidRobosuite._patch_physics_params = custom_patch

    from go_vla_benchmark.env_factory import create_benchmark_env
    from go_vla_benchmark.common import GoResetOptions

    env = create_benchmark_env(
        seed=seed, environment_name="robosuite_go_5x5_rigid_bodies",
        include_image_obs=False, camera_height=64, camera_width=64,
        action_scale=0.03, success_hold_steps=0,
        drive_physical_arm=True, enable_opponent_moves=False,
        opening_with_opponent=True, render_carried_stone=True,
        render_eef_overlay=False, eef_overlay_trail=0,
        robot="Panda", gripper_types="default",
    )
    rs = env._rs_env
    model = rs.sim.model
    coll_gid = model.geom_name2id("go_board_surface_collision")

    sinks = []
    drifts = []
    drop_drifts_50 = []
    drop_drifts_100 = []
    drop_drifts_200 = []
    NUM_RESETS = 20
    for reset_i in range(NUM_RESETS):
        obs = env.reset(options=GoResetOptions(opening_moves=3))
        coll_top = rs.sim.data.geom_xpos[coll_gid, 2] + model.geom_size[coll_gid, 2]
        grid = rs.board_intersections_xyz

        # --- Existing sink/drift test: 10mm drop ---
        target_xyz = grid[2, 2].copy()
        drop_pos = target_xyz.copy()
        drop_pos[2] = coll_top + rs.stone_half_height + 0.01
        rs.set_stone_pose(0, drop_pos)
        rs.sim.data.set_joint_qvel(rs._stone_joint_names[0], np.zeros(6))
        rs.sim.forward()
        start_xy = drop_pos[:2].copy()

        zero = np.zeros(rs.action_dim, dtype=np.float32)
        for _ in range(200):
            rs.step(zero)
        for jn in rs._stone_joint_names:
            rs.sim.data.set_joint_qvel(jn, np.zeros(6, dtype=np.float64))
        rs.sim.forward()

        final = rs.sim.data.body_xpos[rs._stone_body_ids[0]]
        sink = coll_top + rs.stone_half_height - final[2]  # positive = sinking
        drift = np.linalg.norm(final[:2] - start_xy)
        sinks.append(float(sink))
        drifts.append(float(drift))

        # --- Drop-drift test: release from 3× stone height (mimics collect.py) ---
        stone_height = rs.stone_half_height * 2
        drop_pos2 = target_xyz.copy()
        drop_pos2[2] = coll_top + rs.stone_half_height + stone_height * 3
        rs.set_stone_pose(0, drop_pos2)
        rs.sim.data.set_joint_qvel(rs._stone_joint_names[0], np.zeros(6))
        rs.sim.forward()
        drop_xy = drop_pos2[:2].copy()

        for _ in range(50):
            rs.step(zero)
        pos_50 = rs.sim.data.body_xpos[rs._stone_body_ids[0]]
        drop_drifts_50.append(float(np.linalg.norm(pos_50[:2] - drop_xy)))

        for _ in range(50):
            rs.step(zero)
        pos_100 = rs.sim.data.body_xpos[rs._stone_body_ids[0]]
        drop_drifts_100.append(float(np.linalg.norm(pos_100[:2] - drop_xy)))

        for _ in range(100):
            rs.step(zero)
        pos_200 = rs.sim.data.body_xpos[rs._stone_body_ids[0]]
        drop_drifts_200.append(float(np.linalg.norm(pos_200[:2] - drop_xy)))

    env.close()
    # Restore
    mod._Go5x5RigidRobosuite._patch_physics_params = orig_patch

    def pct(arr):
        q5, q50, q95 = np.percentile(arr, [5, 50, 95])
        return round(q5 * 1000, 3), round(q50 * 1000, 3), round(q95 * 1000, 3)

    s5, s50, s95 = pct(sinks)
    d5, d50, d95 = pct(drifts)
    dd50_5, dd50_50, dd50_95 = pct(drop_drifts_50)
    dd100_5, dd100_50, dd100_95 = pct(drop_drifts_100)
    dd200_5, dd200_50, dd200_95 = pct(drop_drifts_200)

    result = {
        "solref": solref,
        "solimp": solimp,
        "margin": margin,
        "rolling_friction": rolling_friction,
        "torsional_friction": torsional_friction,
        "damping": damping,
        "sink_q5_mm": s5, "sink_q50_mm": s50, "sink_q95_mm": s95,
        "drift_q5_mm": d5, "drift_q50_mm": d50, "drift_q95_mm": d95,
        "drop50_q5_mm": dd50_5, "drop50_q50_mm": dd50_50, "drop50_q95_mm": dd50_95,
        "drop100_q5_mm": dd100_5, "drop100_q50_mm": dd100_50, "drop100_q95_mm": dd100_95,
        "drop200_q5_mm": dd200_5, "drop200_q50_mm": dd200_50, "drop200_q95_mm": dd200_95,
    }
    print(json.dumps(result))


def _run_one_trial(trial_idx, params):
    """Run a single trial as a subprocess and return parsed JSON result."""
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--solref", str(params["solref"]),
        "--solimp", str(params["solimp"]),
        "--margin", str(params["margin"]),
        "--rolling-friction", str(params["rolling_friction"]),
        "--torsional-friction", str(params["torsional_friction"]),
        "--damping", str(params["damping"]),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return {"trial": trial_idx, "error": proc.stderr.strip(), **params}
    # Parse the last line of stdout as JSON (in case of import warnings)
    for line in reversed(proc.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return {"trial": trial_idx, **json.loads(line)}
    return {"trial": trial_idx, "error": "no JSON output", **params}


def run_sweep(n_trials: int, max_workers: int = 8):
    """Generate random param combos and run them in parallel."""
    rng = np.random.default_rng(seed=0)
    combos = []
    for _ in range(n_trials):
        combos.append({
            "rolling_friction": round(float(rng.uniform(0.03, 0.15)), 5),
            "torsional_friction": round(float(rng.uniform(0.05, 0.15)), 5),
            "margin": round(float(rng.uniform(0.00015, 0.0007)), 6),
            "solref": round(float(rng.uniform(0.0014, 0.0025)), 5),
            "solimp": round(float(rng.uniform(0.992, 0.996)), 5),
            "damping": round(float(rng.uniform(0.1, 0.5)), 4),
        })

    results = []
    print(f"Launching {n_trials} trials across {max_workers} workers...", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_run_one_trial, i, combo): i
            for i, combo in enumerate(combos)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            res = fut.result()
            status = "ERR" if "error" in res else "ok"
            print(f"  trial {idx:3d} {status}", file=sys.stderr)
            results.append(res)

    # Sort by drop200_q50_mm ascending (best settling first)
    ok = [r for r in results if "error" not in r]
    err = [r for r in results if "error" in r]
    ok.sort(key=lambda r: r.get("drop200_q50_mm", 1e9))

    # Print table
    hdr = (
        f"{'#':>3} | {'roll_f':>7} | {'tors_f':>7} | {'margin':>8} | {'solref':>7} | "
        f"{'solimp':>7} | {'damp':>6} | {'sink_q50':>8} | {'drift_q50':>9} | "
        f"{'d50_q50':>7} | {'d100_q50':>8} | {'d200_q50':>8} | {'d200_q95':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in ok:
        print(
            f"{r['trial']:3d} | "
            f"{r['rolling_friction']:7.4f} | "
            f"{r['torsional_friction']:7.4f} | "
            f"{r['margin']:8.5f} | "
            f"{r['solref']:7.4f} | "
            f"{r['solimp']:7.4f} | "
            f"{r['damping']:6.3f} | "
            f"{r['sink_q50_mm']:8.3f} | "
            f"{r['drift_q50_mm']:9.3f} | "
            f"{r['drop50_q50_mm']:7.3f} | "
            f"{r['drop100_q50_mm']:8.3f} | "
            f"{r['drop200_q50_mm']:8.3f} | "
            f"{r['drop200_q95_mm']:8.3f}"
        )
    for r in err:
        print(f"{r['trial']:3d} | ERROR: {r['error'][:80]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", type=int, default=None,
                   help="Run N random trials in parallel (e.g. --sweep 50)")
    p.add_argument("--solref", type=float)
    p.add_argument("--solimp", type=float)
    p.add_argument("--margin", type=float)
    p.add_argument("--rolling-friction", type=float)
    p.add_argument("--torsional-friction", type=float)
    p.add_argument("--damping", type=float)
    args = p.parse_args()

    if args.sweep is not None:
        run_sweep(args.sweep)
    else:
        # All params required in single-trial mode
        missing = []
        for name in ["solref", "solimp", "margin", "rolling_friction",
                      "torsional_friction", "damping"]:
            if getattr(args, name) is None:
                missing.append(f"--{name.replace('_', '-')}")
        if missing:
            p.error(f"single-trial mode requires: {', '.join(missing)}")
        run(
            args.solref, args.solimp, args.margin,
            rolling_friction=args.rolling_friction,
            torsional_friction=args.torsional_friction,
            damping=args.damping,
        )
