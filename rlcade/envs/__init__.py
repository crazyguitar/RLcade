from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

import gymnasium as gym

from rlcade.envs.smb import (
    SuperMarioBrosEnv,
    RENDER_HUMAN,
    RENDER_RGB_ARRAY,
)
from rlcade.envs.register import register_envs
from rlcade.logger import get_logger

# Register all environments with gymnasium on import
register_envs()

logger = get_logger(__name__)


@dataclass
class SuperMarioBrosConfig:
    rom_path: str = ""
    actions: str = "simple"
    world: int | None = None
    stage: int | None = None
    render_mode: str | None = None
    skip: int = 4
    episodic_life: bool = True
    clip_rewards: bool = True
    frame_stack: int = 4
    custom_reward: bool = True

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> SuperMarioBrosConfig:
        return cls(
            rom_path=args.rom,
            world=args.world,
            stage=args.stage,
            actions=args.actions,
            render_mode=getattr(args, "render_mode", None),
            custom_reward=getattr(args, "custom_reward", False),
        )


_ENV_CONFIGS = {
    "rlcade/SuperMarioBros-v0": SuperMarioBrosConfig,
}


# Super Mario Bros constants
SMB_TOTAL_WORLDS = 8
SMB_STAGES_PER_WORLD = 4
SMB_FIRST_WORLD = 1
SMB_FIRST_STAGE = 1


def get_world_stage_pairs(world: Optional[int], stage: Optional[int]) -> List[Tuple[int, int]]:
    """Generate world/stage pairs based on selection criteria."""
    if stage is not None and world is None:
        raise ValueError("Cannot specify stage without world")

    if world is not None and stage is not None:
        return [(world, stage)]
    elif world is not None:
        return [(world, s) for s in range(SMB_FIRST_STAGE, SMB_STAGES_PER_WORLD + 1)]
    else:
        return [
            (w, s)
            for w in range(SMB_FIRST_WORLD, SMB_TOTAL_WORLDS + 1)
            for s in range(SMB_FIRST_STAGE, SMB_STAGES_PER_WORLD + 1)
        ]


def create_env(args: argparse.Namespace) -> gym.Env:
    """Create an environment from CLI args."""
    config_cls = _ENV_CONFIGS[args.env]
    config = config_cls.from_args(args)
    return gym.make(args.env, config=config, disable_env_checker=True)


def create_vector_env(
    args: argparse.Namespace,
    *,
    use_gym: bool = False,
    rank: int | None = None,
    world_size: int | None = None,
    label: str = "env",
) -> gym.Env:
    """Create vector env for multi-world/stage or single env.

    When *rank* and *world_size* are given the world/stage combinations are
    partitioned across ranks so each rank only creates its own subset.
    """
    combinations = get_world_stage_pairs(args.world, args.stage)

    if rank is not None and world_size is not None and world_size > 1:
        per_rank = len(combinations) // world_size
        start = rank * per_rank
        combinations = combinations[start : start + per_rank]

    if len(combinations) == 1:
        return create_env(args)

    if use_gym:
        return _create_async_vector_env(args, combinations, label)
    return _create_inprocess_vector_env(args, combinations, label)


def _create_inprocess_vector_env(args: argparse.Namespace, combinations: List[Tuple[int, int]], label: str):
    """Create in-process Rust vector env — zero IPC overhead."""
    from rlcade.envs.smb import SuperMarioBrosVecEnv

    config = _ENV_CONFIGS[args.env].from_args(args)
    configs = [
        dict(
            rom=config.rom_path,
            actions=config.actions,
            world=w,
            stage=s,
            skip=config.skip,
            episodic_life=config.episodic_life,
            custom_reward=config.custom_reward,
            clip_rewards=config.clip_rewards,
            frame_stack=config.frame_stack,
        )
        for w, s in combinations
    ]
    logger.info("Creating SuperMarioBrosVecEnv [%s] with %d envs (in-process)", label, len(configs))
    return SuperMarioBrosVecEnv(configs)


def _create_async_vector_env(args: argparse.Namespace, combinations: List[Tuple[int, int]], label: str):
    """Create subprocess-based AsyncVectorEnv."""
    logger.info("Creating AsyncVectorEnv [%s] with %d envs", label, len(combinations))
    env_fns = _create_env_factories(args, combinations)
    return gym.vector.AsyncVectorEnv(env_fns)


def _create_env_factories(args: argparse.Namespace, combinations: List[Tuple[int, int]]):
    """Create environment factory functions."""
    config_cls = _ENV_CONFIGS[args.env]

    def make_env_factory(world: int, stage: int):
        def factory():
            env_args = argparse.Namespace(**vars(args))
            env_args.world = world
            env_args.stage = stage
            config = config_cls.from_args(env_args)
            return gym.make(args.env, config=config, disable_env_checker=True)

        return factory

    return [make_env_factory(w, s) for w, s in combinations]


def get_env_info(env) -> tuple[tuple[int, ...], int]:
    """Return (obs_shape, n_actions) for single or vector env."""
    if hasattr(env, "num_envs") and env.num_envs > 1:
        return env.observation_space.shape[1:], env.action_space.n
    return env.observation_space.shape, env.action_space.n


__all__ = [
    "SuperMarioBrosConfig",
    "SuperMarioBrosEnv",
    "create_env",
    "create_vector_env",
    "get_env_info",
    "get_world_stage_pairs",
    "RENDER_HUMAN",
    "RENDER_RGB_ARRAY",
]
