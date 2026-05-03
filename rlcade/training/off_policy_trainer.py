"""Off-Policy Trainer — shared base for replay-buffer algorithms (DQN, SAC, etc.)."""

import time

import numpy as np

from rlcade.utils import PinMemory
from rlcade.plugins import TrainerPlugin
from rlcade.training.trainer import Trainer
from rlcade.logger import get_logger

logger = get_logger(__name__)


class OffPolicyTrainer(Trainer):
    """Shared base for off-policy trainers (DQN, SAC, Rainbow).

    Subclasses only need to define ``_loss_key`` and optionally override ``step``.
    """

    _loss_key: str = "loss"

    def __init__(self, args, *, plugins: list[TrainerPlugin] | None = None):
        super().__init__(args, plugins=plugins)
        self.num_envs = self.env.num_envs if hasattr(self.env, "num_envs") else 1
        self.obs = None
        self.episode_rewards = np.zeros(self.num_envs)
        self.t0 = None
        # Trainer owns PinMemory because the H2D happens here (step()).
        # In PPO the H2D is inside the agent's collect_rollout, so the PPO
        # agent owns its own PinMemory instead.
        self._pin_memory_enabled = getattr(args, "pin_memory", True)
        self._pin_memory = PinMemory(enabled=self._pin_memory_enabled)

    @property
    def config(self) -> dict:
        return {
            **super().config,
            "num_envs": self.num_envs,
        }

    def swap(self, new_env) -> None:
        super().swap(new_env)
        self.num_envs = new_env.num_envs if hasattr(new_env, "num_envs") else 1
        self.episode_rewards = np.zeros(self.num_envs)
        self.obs, _ = new_env.reset()
        self.t0 = time.time()
        # Obs shape may change on swap — drop cached pinned buffers.
        self._pin_memory.reset()

    def setup(self) -> None:
        super().setup()
        self.obs, _ = self.env.reset()
        self.t0 = time.time()

    def step(self, iteration: int) -> None:
        actions = self.agent.get_action(self._pin_memory.to(self.obs, self.agent.device))
        next_obs, rewards, terminated, truncated, _ = self.env.step(actions)
        dones = np.atleast_1d(np.asarray(terminated) | np.asarray(truncated))

        self.agent.store(self.obs, actions, rewards, next_obs, dones)
        self._record_episodes(rewards, dones)
        self.obs = self._maybe_reset(next_obs, dones)
        self._maybe_learn()

        self.metrics.advance(self.num_envs)
        elapsed = time.time() - self.t0
        self.metrics.sps = self.metrics.total_steps / max(elapsed, 1e-12)
        self.metrics.extras["steps"] = self.metrics.total_steps

    def _maybe_reset(self, next_obs, dones):
        """Vec envs auto-reset; single env needs manual reset."""
        if self.num_envs == 1 and dones.any():
            next_obs, _ = self.env.reset()
        return next_obs

    def _maybe_learn(self) -> None:
        if not self.agent.can_learn():
            return
        learn_metrics = self.agent.learn()
        self.metrics.record_loss(learn_metrics[self._loss_key])
        self.metrics.extras.update(learn_metrics)

    def _record_episodes(self, rewards, dones):
        self.episode_rewards += np.atleast_1d(rewards)
        for i in np.flatnonzero(np.atleast_1d(dones)):
            self.metrics.record_episodes([float(self.episode_rewards[i])])
            self.episode_rewards[i] = 0.0
