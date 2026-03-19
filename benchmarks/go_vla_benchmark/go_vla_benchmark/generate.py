"""MimicGen augmentation pipeline for the Go benchmark."""

from __future__ import annotations

import math
import multiprocessing
from typing import Dict, List, Optional

import numpy as np

from .robosuite_compat import ensure_robosuite_compat

ensure_robosuite_compat()

from mimicgen.datagen.data_generator import DataGenerator

from .config import GenerationSettings, load_task_config
from .common import GoResetOptions
from .dataset_io import EpisodeRecord, read_demo_keys, write_dataset
from .env_factory import create_benchmark_env
from .mimicgen_interface import MG_GoJacoSingleMove


def _stack_obs(obs_seq: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    keys = obs_seq[0].keys()
    return {k: np.asarray([obs[k] for obs in obs_seq]) for k in keys}


def _episode_from_generated(generated: Dict[str, object]) -> EpisodeRecord:
    observations = generated["observations"]
    if len(observations) == 0:
        raise RuntimeError("generated trajectory is empty")

    extras = {
        "src_demo_inds": np.asarray(generated["src_demo_inds"], dtype=np.int32),
        "src_demo_labels": np.asarray(generated["src_demo_labels"], dtype=np.int32),
    }

    return EpisodeRecord(
        actions=np.asarray(generated["actions"], dtype=np.float32),
        states=np.asarray(generated["states"], dtype=np.float32),
        observations=_stack_obs(observations),
        datagen_infos=generated["datagen_infos"],
        initial_state=generated["initial_state"],
        extras=extras,
    )


def _generate_worker(config: dict) -> dict:
    """Worker function for parallel MimicGen generation (must be top-level for pickle)."""
    import os
    os.environ["MUJOCO_GL"] = "glfw"
    worker_id = config["worker_id"]
    task_spec, _, raw_cfg = load_task_config(config["task_config_path"])

    rng = np.random.RandomState(config["seed"])
    env = create_benchmark_env(
        seed=config["seed"],
        environment_name=config["environment_name"],
        include_image_obs=config["include_image_obs"],
        camera_height=config["camera_height"],
        camera_width=config["camera_width"],
        gnugo_path=config["gnugo_path"],
        action_scale=config["action_scale"],
        success_hold_steps=config["success_hold_steps"],
        drive_physical_arm=config["drive_physical_arm"],
        enable_opponent_moves=config["enable_opponent_moves"],
        opening_with_opponent=config["opening_with_opponent"],
        render_carried_stone=config["render_carried_stone"],
        render_eef_overlay=config["render_eef_overlay"],
        eef_overlay_trail=config["eef_overlay_trail"],
        robot=config["robot"],
        gripper_types=config["gripper_types"],
    )
    env_interface = MG_GoJacoSingleMove(env=env)

    demo_keys = read_demo_keys(config["source_dataset_path"])
    data_generator = DataGenerator(
        task_spec=task_spec,
        dataset_path=config["source_dataset_path"],
        demo_keys=demo_keys,
    )

    episodes: List[EpisodeRecord] = []
    attempts = 0

    while (len(episodes) < config["num_demos"]) and (attempts < config["max_attempts"]):
        attempts += 1
        print(
            f"[generate][worker {worker_id}] attempt {attempts}/{config['max_attempts']} "
            f"demos={len(episodes)}/{config['num_demos']}",
            flush=True,
        )

        opening_moves = int(rng.randint(config["opening_moves_min"], config["opening_moves_max"] + 1))
        env.queue_reset_options(GoResetOptions(opening_moves=opening_moves))

        generated = data_generator.generate(
            env=env,
            env_interface=env_interface,
            select_src_per_subtask=config["select_src_per_subtask"],
            transform_first_robot_pose=config["transform_first_robot_pose"],
            interpolate_from_last_target_pose=config["interpolate_from_last_target_pose"],
            render=False,
            video_writer=None,
            video_skip=5,
            camera_names=None,
            pause_subtask=False,
        )

        if bool(generated["success"]):
            episodes.append(_episode_from_generated(generated))
            print(
                f"[generate][worker {worker_id}] success demos={len(episodes)}/{config['num_demos']}",
                flush=True,
            )

    return {
        "episodes": episodes,
        "attempts": attempts,
        "env_meta": {
            **env.serialize(),
            "source_dataset": config["source_dataset_path"],
            "task_config": raw_cfg,
        },
        "env_interface_name": type(env_interface).__name__,
        "env_interface_type": type(env_interface).INTERFACE_TYPE,
    }


def _resolve_settings(
    defaults: GenerationSettings,
    num_demos: Optional[int],
    max_attempts: Optional[int],
    seed: Optional[int],
    opening_moves_min: Optional[int],
    opening_moves_max: Optional[int],
) -> GenerationSettings:
    return GenerationSettings(
        num_demos=defaults.num_demos if num_demos is None else int(num_demos),
        max_attempts=defaults.max_attempts if max_attempts is None else int(max_attempts),
        select_src_per_subtask=defaults.select_src_per_subtask,
        transform_first_robot_pose=defaults.transform_first_robot_pose,
        interpolate_from_last_target_pose=defaults.interpolate_from_last_target_pose,
        seed=defaults.seed if seed is None else int(seed),
        opening_moves_min=defaults.opening_moves_min if opening_moves_min is None else int(opening_moves_min),
        opening_moves_max=defaults.opening_moves_max if opening_moves_max is None else int(opening_moves_max),
    )


def generate_augmented_demonstrations(
    source_dataset_path: str,
    output_path: str,
    task_config_path: str,
    environment_name: str = "go_7x7",
    num_demos: Optional[int] = None,
    max_attempts: Optional[int] = None,
    seed: Optional[int] = None,
    opening_moves_min: Optional[int] = None,
    opening_moves_max: Optional[int] = None,
    include_image_obs: bool = True,
    camera_height: int = 84,
    camera_width: int = 84,
    gnugo_path: str | None = None,
    action_scale: float = 0.03,
    success_hold_steps: int = 0,
    drive_physical_arm: bool = True,
    enable_opponent_moves: bool = False,
    opening_with_opponent: bool = True,
    render_carried_stone: bool = True,
    render_eef_overlay: bool = True,
    eef_overlay_trail: int = 10,
    robot: str = "Panda",
    gripper_types: str = "default",
    num_workers: int = 1,
) -> Dict[str, object]:
    """Generate new trajectories via MimicGen using Go source demonstrations."""
    task_spec, default_settings, raw_cfg = load_task_config(task_config_path)
    settings = _resolve_settings(
        defaults=default_settings,
        num_demos=num_demos,
        max_attempts=max_attempts,
        seed=seed,
        opening_moves_min=opening_moves_min,
        opening_moves_max=opening_moves_max,
    )

    if settings.num_demos <= 0:
        raise ValueError("num_demos must be > 0")
    if settings.max_attempts < settings.num_demos:
        raise ValueError("max_attempts must be >= num_demos")

    if num_workers > 1:
        per_worker_demos = math.ceil(settings.num_demos / num_workers)
        per_worker_attempts = math.ceil(settings.max_attempts / num_workers)
        worker_configs = []
        for i in range(num_workers):
            worker_configs.append({
                "worker_id": i,
                "num_demos": per_worker_demos,
                "max_attempts": per_worker_attempts,
                "seed": settings.seed + i,
                "source_dataset_path": source_dataset_path,
                "task_config_path": task_config_path,
                "opening_moves_min": settings.opening_moves_min,
                "opening_moves_max": settings.opening_moves_max,
                "environment_name": environment_name,
                "include_image_obs": include_image_obs,
                "camera_height": camera_height,
                "camera_width": camera_width,
                "gnugo_path": gnugo_path,
                "action_scale": action_scale,
                "success_hold_steps": success_hold_steps,
                "drive_physical_arm": drive_physical_arm,
                "enable_opponent_moves": enable_opponent_moves,
                "opening_with_opponent": opening_with_opponent,
                "render_carried_stone": render_carried_stone,
                "render_eef_overlay": render_eef_overlay,
                "eef_overlay_trail": eef_overlay_trail,
                "robot": robot,
                "gripper_types": gripper_types,
                "select_src_per_subtask": settings.select_src_per_subtask,
                "transform_first_robot_pose": settings.transform_first_robot_pose,
                "interpolate_from_last_target_pose": settings.interpolate_from_last_target_pose,
            })

        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(num_workers) as pool:
            results = pool.map(_generate_worker, worker_configs)

        episodes: List[EpisodeRecord] = []
        total_attempts = 0
        for r in results:
            episodes.extend(r["episodes"])
            total_attempts += r["attempts"]
        episodes = episodes[:settings.num_demos]

        if len(episodes) < settings.num_demos:
            raise RuntimeError(
                f"MimicGen generation incomplete: got {len(episodes)} / {settings.num_demos} "
                f"successes after {total_attempts} attempts across {num_workers} workers"
            )

        env_meta = results[0]["env_meta"]
        env_interface_name = results[0]["env_interface_name"]
        env_interface_type = results[0]["env_interface_type"]

        write_dataset(
            output_path=output_path,
            episodes=episodes,
            env_meta=env_meta,
            env_interface_name=env_interface_name,
            env_interface_type=env_interface_type,
        )

        return {
            "source_dataset": source_dataset_path,
            "output_path": output_path,
            "num_generated": len(episodes),
            "attempts": total_attempts,
            "success_rate": float(len(episodes) / total_attempts),
            "settings": settings.__dict__,
            "num_workers": num_workers,
        }

    demo_keys = read_demo_keys(source_dataset_path)
    if not demo_keys:
        raise RuntimeError(f"no demos found in source dataset: {source_dataset_path}")

    rng = np.random.RandomState(settings.seed)

    env = create_benchmark_env(
        seed=settings.seed,
        environment_name=environment_name,
        include_image_obs=include_image_obs,
        camera_height=camera_height,
        camera_width=camera_width,
        gnugo_path=gnugo_path,
        action_scale=action_scale,
        success_hold_steps=success_hold_steps,
        drive_physical_arm=drive_physical_arm,
        enable_opponent_moves=enable_opponent_moves,
        opening_with_opponent=opening_with_opponent,
        render_carried_stone=render_carried_stone,
        render_eef_overlay=render_eef_overlay,
        eef_overlay_trail=eef_overlay_trail,
        robot=robot,
        gripper_types=gripper_types,
    )
    env_interface = MG_GoJacoSingleMove(env=env)

    data_generator = DataGenerator(
        task_spec=task_spec,
        dataset_path=source_dataset_path,
        demo_keys=demo_keys,
    )

    episodes: List[EpisodeRecord] = []
    attempts = 0

    while (len(episodes) < settings.num_demos) and (attempts < settings.max_attempts):
        attempts += 1

        opening_moves = int(rng.randint(settings.opening_moves_min, settings.opening_moves_max + 1))
        # DataGenerator internally calls env.reset(); queue options so they are applied there.
        env.queue_reset_options(GoResetOptions(opening_moves=opening_moves))

        generated = data_generator.generate(
            env=env,
            env_interface=env_interface,
            select_src_per_subtask=settings.select_src_per_subtask,
            transform_first_robot_pose=settings.transform_first_robot_pose,
            interpolate_from_last_target_pose=settings.interpolate_from_last_target_pose,
            render=False,
            video_writer=None,
            video_skip=5,
            camera_names=None,
            pause_subtask=False,
        )

        if bool(generated["success"]):
            episodes.append(_episode_from_generated(generated))

    if len(episodes) < settings.num_demos:
        raise RuntimeError(
            f"MimicGen generation incomplete: got {len(episodes)} / {settings.num_demos} "
            f"successes after {attempts} attempts"
        )

    write_dataset(
        output_path=output_path,
        episodes=episodes,
        env_meta={
            **env.serialize(),
            "source_dataset": source_dataset_path,
            "task_config": raw_cfg,
        },
        env_interface_name=type(env_interface).__name__,
        env_interface_type=type(env_interface).INTERFACE_TYPE,
    )

    return {
        "source_dataset": source_dataset_path,
        "output_path": output_path,
        "num_generated": len(episodes),
        "attempts": attempts,
        "success_rate": float(len(episodes) / attempts),
        "settings": settings.__dict__,
    }
