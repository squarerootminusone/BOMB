"""Factory for selecting the benchmark environment backend."""

from __future__ import annotations

from typing import Any


_ROBOSUITE_ENV_ALIASES = {
    "robosuite_go_5x5",
    "robosuite_go_5x5_rigid",
    "robosuite_go_5x5_rigid_bodies",
    "go_5x5_robosuite",
    "go_5x5_robosuite_rigid",
}


def is_robosuite_backend(environment_name: str) -> bool:
    """True if @environment_name should be instantiated with robosuite."""
    env_name = str(environment_name).strip().lower()
    return env_name in _ROBOSUITE_ENV_ALIASES or env_name.startswith("robosuite_")


def create_benchmark_env(environment_name: str, **kwargs: Any) -> Any:
    """Instantiate either the dm_control or robosuite Go benchmark backend."""
    if is_robosuite_backend(environment_name):
        from .robosuite_go_env import GoRobosuiteBenchmarkEnv

        return GoRobosuiteBenchmarkEnv(environment_name=environment_name, **kwargs)

    kwargs = dict(kwargs)
    kwargs.pop("robot", None)
    kwargs.pop("gripper_types", None)

    from .go_env import GoJacoBenchmarkEnv

    return GoJacoBenchmarkEnv(environment_name=environment_name, **kwargs)
