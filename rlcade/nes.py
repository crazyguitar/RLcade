"""NES inference engine — load a trained agent and play."""

import time

import gymnasium as gym
import torch

from rlcade.logger import get_logger

logger = get_logger(__name__)

FRAME_DURATION = 1.0 / 60


class Nes:
    """NES inference engine — load a trained agent and play."""

    def __init__(self, env: gym.Env, agent, device: str = "cpu"):
        self.device = device
        self.env = env
        self.rust_env = env.unwrapped._env
        self.obs_shape = env.observation_space.shape
        self.agent = agent
        self.action = 0

    def play(self):
        """Play episodes until the user quits."""
        obs, _ = self.env.reset()
        self.action = self._choose_action(obs)
        episode = 0

        while not self.env.unwrapped.poll_quit():
            frame_start = time.time()
            obs, episode = self._step_frame(obs, episode)
            self.limit_fps(frame_start)

        self.env.close()

    def _step_frame(self, obs, episode: int):
        """Advance one NES frame, render it, and update the agent when ready."""
        step_obs, _, terminated, truncated, info = self.rust_env.step_frame(self.action)
        self.env.render()

        if terminated or truncated:
            episode += 1
            logger.info("Episode %d: %s", episode, info)
            obs, _ = self.env.reset()
            self.action = self._choose_action(obs)
        elif info["ready"]:
            obs = step_obs.reshape(self.obs_shape)
            self.action = self._choose_action(obs)

        return obs, episode

    def _choose_action(self, obs):
        obs_tensor = torch.from_numpy(obs).float().to(self.device)
        return int(self.agent.act(obs_tensor, deterministic=True))

    @staticmethod
    def limit_fps(frame_start: float):
        """Sleep for the remainder of the frame to maintain ~60 FPS."""
        elapsed = time.time() - frame_start
        if elapsed < FRAME_DURATION:
            time.sleep(FRAME_DURATION - elapsed)
