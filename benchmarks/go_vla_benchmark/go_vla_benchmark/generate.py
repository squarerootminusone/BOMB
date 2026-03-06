"""MimicGen augmentation pipeline for the Go benchmark."""

from __future__ import annotations

import multiprocessing as mp
import os
from typing import Dict, List, Optional

import numpy as np
import tqdm

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


# ---------------------------------------------------------------------------
# Worker for parallel generation
# ---------------------------------------------------------------------------

def _worker_generate(args):
    """Run in a spawned process. Creates its own env + DataGenerator, attempts
    one generation, and returns the EpisodeRecord on success or None."""
    (
        worker_seed,
        opening_moves,
        source_dataset_path,
        demo_keys,
        task_spec,
        settings,
        env_kwargs,
    ) = args

    ensure_robosuite_compat()

    env = create_benchmark_env(seed=worker_seed, **env_kwargs)
    env_interface = MG_GoJacoSingleMove(env=env)
    data_generator = DataGenerator(
        task_spec=task_spec,
        dataset_path=source_dataset_path,
        demo_keys=demo_keys,
    )

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
        return _episode_from_generated(generated)
    return None


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
    num_workers: int = 0,
) -> Dict[str, object]:
    """Generate new trajectories via MimicGen using Go source demonstrations.

    Args:
        num_workers: Number of parallel worker processes. 0 = serial (original
            behaviour), >0 = spawn that many workers.
    """
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

    demo_keys = read_demo_keys(source_dataset_path)
    if not demo_keys:
        raise RuntimeError(f"no demos found in source dataset: {source_dataset_path}")

    rng = np.random.RandomState(settings.seed)

    env_kwargs = dict(
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

    if num_workers > 0:
        episodes, attempts = _generate_parallel(
            settings=settings,
            rng=rng,
            source_dataset_path=source_dataset_path,
            demo_keys=demo_keys,
            task_spec=task_spec,
            env_kwargs=env_kwargs,
            num_workers=num_workers,
        )
    else:
        episodes, attempts = _generate_serial(
            settings=settings,
            rng=rng,
            source_dataset_path=source_dataset_path,
            demo_keys=demo_keys,
            task_spec=task_spec,
            env_kwargs=env_kwargs,
        )

    if len(episodes) < settings.num_demos:
        raise RuntimeError(
            f"MimicGen generation incomplete: got {len(episodes)} / {settings.num_demos} "
            f"successes after {attempts} attempts"
        )

    write_dataset(
        output_path=output_path,
        episodes=episodes,
        env_meta={
            **create_benchmark_env(seed=settings.seed, **env_kwargs).serialize(),
            "source_dataset": source_dataset_path,
            "task_config": raw_cfg,
        },
        env_interface_name=MG_GoJacoSingleMove.__name__,
        env_interface_type=MG_GoJacoSingleMove.INTERFACE_TYPE,
    )

    return {
        "source_dataset": source_dataset_path,
        "output_path": output_path,
        "num_generated": len(episodes),
        "attempts": attempts,
        "success_rate": float(len(episodes) / attempts) if attempts else 0.0,
        "settings": settings.__dict__,
    }


def _generate_serial(
    settings: GenerationSettings,
    rng: np.random.RandomState,
    source_dataset_path: str,
    demo_keys: List[str],
    task_spec: dict,
    env_kwargs: dict,
) -> tuple[List[EpisodeRecord], int]:
    """Original serial generation loop with tqdm."""
    env = create_benchmark_env(seed=settings.seed, **env_kwargs)
    env_interface = MG_GoJacoSingleMove(env=env)

    data_generator = DataGenerator(
        task_spec=task_spec,
        dataset_path=source_dataset_path,
        demo_keys=demo_keys,
    )

    episodes: List[EpisodeRecord] = []
    attempts = 0

    pbar = tqdm.tqdm(total=settings.num_demos, desc="Generating demos", unit="demo")
    while (len(episodes) < settings.num_demos) and (attempts < settings.max_attempts):
        attempts += 1

        opening_moves = int(rng.randint(settings.opening_moves_min, settings.opening_moves_max + 1))
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

        success = bool(generated["success"])
        if success:
            episodes.append(_episode_from_generated(generated))
            pbar.update(1)

        pbar.set_postfix(attempts=attempts, success_rate=f"{len(episodes)}/{attempts}")

    pbar.close()
    return episodes, attempts


def _generate_parallel(
    settings: GenerationSettings,
    rng: np.random.RandomState,
    source_dataset_path: str,
    demo_keys: List[str],
    task_spec: dict,
    env_kwargs: dict,
    num_workers: int,
) -> tuple[List[EpisodeRecord], int]:
    """Parallel generation using a process pool."""
    # Pre-generate all attempt parameters (seeds + opening moves)
    worker_seeds = rng.randint(0, 2**31, size=settings.max_attempts).tolist()
    opening_moves_list = [
        int(rng.randint(settings.opening_moves_min, settings.opening_moves_max + 1))
        for _ in range(settings.max_attempts)
    ]

    episodes: List[EpisodeRecord] = []
    attempts = 0

    # Limit MuJoCo threads per worker so they don't fight over cores
    threads_per_worker = max(1, os.cpu_count() // num_workers)

    def make_args(i):
        return (
            worker_seeds[i],
            opening_moves_list[i],
            source_dataset_path,
            demo_keys,
            task_spec,
            settings,
            env_kwargs,
        )

    ctx = mp.get_context("spawn")
    pbar = tqdm.tqdm(total=settings.num_demos, desc=f"Generating demos ({num_workers}w)", unit="demo")

    # Submit in chunks to allow early stopping
    chunk_size = num_workers * 2
    with ctx.Pool(
        processes=num_workers,
        initializer=_pool_initializer,
        initargs=(threads_per_worker,),
    ) as pool:
        idx = 0
        while len(episodes) < settings.num_demos and idx < settings.max_attempts:
            end = min(idx + chunk_size, settings.max_attempts)
            batch_args = [make_args(i) for i in range(idx, end)]

            for result in pool.imap_unordered(_worker_generate, batch_args):
                attempts += 1
                if result is not None:
                    episodes.append(result)
                    pbar.update(1)
                pbar.set_postfix(attempts=attempts, success_rate=f"{len(episodes)}/{attempts}")

                if len(episodes) >= settings.num_demos:
                    break

            idx = end

    pbar.close()
    return episodes, attempts


def _pool_initializer(mj_threads: int):
    """Set MuJoCo thread count per worker to avoid oversubscription."""
    os.environ["MJ_NUM_THREADS"] = str(mj_threads)
    os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "egl")
