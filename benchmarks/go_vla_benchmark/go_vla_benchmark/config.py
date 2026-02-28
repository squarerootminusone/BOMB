"""Config loading utilities for the Go benchmark."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple


@dataclass
class GenerationSettings:
    num_demos: int
    max_attempts: int
    select_src_per_subtask: bool
    transform_first_robot_pose: bool
    interpolate_from_last_target_pose: bool
    seed: int
    opening_moves_min: int
    opening_moves_max: int


def load_task_config(path: str | Path) -> Tuple[Any, GenerationSettings, Dict[str, Any]]:
    """
    Load task spec json and return (task_spec, generation_settings, raw_dict).

    Returns:
        task_spec: mimicgen.configs.task_spec.MG_TaskSpec
        generation_settings: defaults parsed from config
        raw_dict: full raw json config
    """
    from mimicgen.configs.task_spec import MG_TaskSpec

    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    task_spec = MG_TaskSpec.from_json(json_dict=cfg["task_spec"])

    defaults = cfg.get("defaults", {})
    settings = GenerationSettings(
        num_demos=int(defaults.get("num_demos", 200)),
        max_attempts=int(defaults.get("max_attempts", 500)),
        select_src_per_subtask=bool(defaults.get("select_src_per_subtask", False)),
        transform_first_robot_pose=bool(defaults.get("transform_first_robot_pose", True)),
        interpolate_from_last_target_pose=bool(defaults.get("interpolate_from_last_target_pose", True)),
        seed=int(defaults.get("seed", 0)),
        opening_moves_min=int(defaults.get("opening_moves_min", 0)),
        opening_moves_max=int(defaults.get("opening_moves_max", 8)),
    )

    return task_spec, settings, cfg
