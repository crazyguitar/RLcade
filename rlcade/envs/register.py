"""Register NES environments with gymnasium."""

import gymnasium as gym

ENVS = {
    "rlcade/SuperMarioBros-v0": "rlcade.envs.smb:SuperMarioBrosEnv",
}


def register_envs(envs: dict[str, str] | None = None):
    """Register environments with gymnasium. Uses built-in registry if envs is None."""
    for env_id, entry_point in (envs or ENVS).items():
        if env_id not in gym.registry:
            gym.register(id=env_id, entry_point=entry_point)
