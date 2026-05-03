"""On-Policy Trainer — shared base for rollout-based algorithms (PPO, etc.)."""

import time

import torch

from rlcade.plugins import TrainerPlugin
from rlcade.training.trainer import Trainer
from rlcade.logger import get_logger

logger = get_logger(__name__)


class OnPolicyTrainer(Trainer):
    """Base trainer for on-policy algorithms.

    Implements the collect → learn loop, episode score extraction,
    and checkpoint resume by num_steps. Subclasses only need to set
    ``_loss_key`` and optionally override ``collect`` or ``step``.
    """

    _loss_key: str = "loss"

    def __init__(self, args, *, plugins: list[TrainerPlugin] | None = None):
        super().__init__(args, plugins=plugins)
        self.num_steps = args.num_steps
        self.obs: torch.Tensor | None = None

    @property
    def config(self) -> dict:
        return {**super().config, "num_steps": self.num_steps}

    def step(self, iteration: int) -> None:
        """Run one iteration: collect → learn. Updates self.metrics."""
        t0 = time.time()
        rollout, self.obs, scores = self.collect(self.obs)
        t1 = time.time()
        learn_metrics = self.agent.learn(rollout)
        t2 = time.time()

        self.metrics.advance(self.num_steps)
        self.metrics.record_loss(learn_metrics[self._loss_key])
        self.metrics.record_episodes(scores)
        self.metrics.sps = self.num_steps / max(t2 - t0, 1e-12)
        self.metrics.extras.update(learn_metrics)
        self.metrics.extras["rollout"] = t1 - t0
        self.metrics.extras["train"] = t2 - t1

    def collect(self, obs: torch.Tensor | None = None):
        """Collect rollout and extract episode scores."""
        rollout, obs = self.agent.collect_rollout(self.env, self.num_steps, obs)
        scores = self.extract_episode_scores(rollout["rewards"], rollout["dones"])
        return rollout, obs, scores

    @staticmethod
    def extract_episode_scores(rewards: torch.Tensor, dones: torch.Tensor) -> list[float]:
        """Sum rewards between done flags to get completed episode scores.

        With vec envs, rewards are (T, N) where T=timesteps and N=num_envs.
        Each env runs a different world/stage, so we iterate per-env to
        avoid mixing rewards across stages.

        Example with 2 envs, 3 timesteps:
            rewards = [[10, 5], [20, 3], [15, 8]]
            dones   = [[ 0, 1], [ 1, 0], [ 0, 0]]

            env0: 10+20=30 (done at t1) — world 1-1 episode
            env1: 5 (done at t0)        — world 1-2 episode
            result: [30, 5]
        """
        if rewards.dim() == 1:
            scores = []
            episode_reward = 0.0
            for reward, done in zip(rewards, dones):
                episode_reward += reward.item()
                if done:
                    scores.append(episode_reward)
                    episode_reward = 0.0
            return scores

        T, N = rewards.shape
        scores = []
        for env_idx in range(N):
            episode_reward = 0.0
            for t in range(T):
                episode_reward += rewards[t, env_idx].item()
                if dones[t, env_idx]:
                    scores.append(episode_reward)
                    episode_reward = 0.0
        return scores

    def swap(self, new_env) -> None:
        super().swap(new_env)
        self.obs = None
        self.agent.reset()
