"""SAC-Discrete agent — Soft Actor-Critic for discrete action spaces.

Off-policy, entropy-regularized actor-critic with:
- Categorical policy (actor)
- Dual Q-networks with soft target updates (critics)
- Automatic entropy temperature tuning

Paper: Christodoulou 2019, "Soft Actor-Critic for Discrete Action Settings"
       https://arxiv.org/abs/1910.07207

Follows the same EnvSAC/VecSAC pattern as DQN for single vs vector env support.
"""

from __future__ import annotations

import argparse
from abc import abstractmethod
from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from rlcade.agent.base import AgentBase, EnvAgentMixin, VecAgentMixin, Agent, is_vector_env
from rlcade.modules import create_actor, create_qnet, build_encoder_kwargs, parse_channels
from rlcade.utils import to_tensor, clip_grad_norm_, soft_update
from rlcade.utils.amp import resolve_amp_device_type, create_grad_scaler
from rlcade.utils.replay_buffer import ReplayBuffer


@dataclass
class SACConfig:
    obs_shape: tuple[int, ...]
    n_actions: int
    actor: str = "actor"
    qnet: str = "qnet"
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    gamma: float = 0.99
    tau: float = 5e-3
    batch_size: int = 64
    buffer_size: int = 100_000
    learn_start: int = 10_000
    learn_freq: int = 4
    init_alpha: float = 0.2
    max_grad_norm: float = 10.0
    device: str = "cpu"
    checkpoint: str | None = None
    encoder: str = "cnn"
    resnet_channels: list[int] | None = None
    resnet_out_dim: int = 256
    target_entropy_ratio: float = 0.98
    amp: bool = False
    grad_accum_steps: int = 1

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> SACConfig:
        return cls(
            obs_shape=args.obs_shape,
            n_actions=args.n_actions,
            actor=getattr(args, "actor", "actor"),
            qnet=getattr(args, "qnet", "qnet"),
            lr_actor=getattr(args, "lr_actor", 3e-4),
            lr_critic=getattr(args, "lr_critic", 3e-4),
            lr_alpha=getattr(args, "lr_alpha", 3e-4),
            gamma=getattr(args, "gamma", 0.99),
            tau=getattr(args, "tau", 5e-3),
            batch_size=getattr(args, "batch_size", 64),
            buffer_size=getattr(args, "buffer_size", 100_000),
            learn_start=getattr(args, "learn_start", 10_000),
            learn_freq=getattr(args, "learn_freq", 4),
            init_alpha=getattr(args, "init_alpha", 0.2),
            max_grad_norm=getattr(args, "max_grad_norm", 10.0),
            device=getattr(args, "device", "cpu"),
            checkpoint=getattr(args, "checkpoint", None),
            encoder=getattr(args, "encoder", "cnn"),
            resnet_channels=parse_channels(getattr(args, "resnet_channels", "16,32,32")),
            resnet_out_dim=getattr(args, "resnet_out_dim", 256),
            target_entropy_ratio=getattr(args, "target_entropy_ratio", 0.98),
            amp=getattr(args, "amp", False),
            grad_accum_steps=getattr(args, "grad_accum_steps", 1),
        )


class SACBase(AgentBase):
    def __init__(self, config: SACConfig):
        self.device = torch.device(config.device)
        self.n_actions = config.n_actions
        self.gamma = config.gamma
        self.tau = config.tau
        self.batch_size = config.batch_size
        self.learn_start = config.learn_start
        self.learn_freq = config.learn_freq
        self.max_grad_norm = config.max_grad_norm
        self.step_count = 0

        # Actor: categorical policy
        enc_kwargs = build_encoder_kwargs(config)
        self.actor = create_actor(
            config.actor,
            config.obs_shape,
            config.n_actions,
            **enc_kwargs,
        ).to(self.device)

        # Dual Q-networks
        self.q1 = create_qnet(
            config.qnet,
            config.obs_shape,
            config.n_actions,
            **enc_kwargs,
        ).to(self.device)
        self.q2 = create_qnet(
            config.qnet,
            config.obs_shape,
            config.n_actions,
            **enc_kwargs,
        ).to(self.device)
        self.q1_target = create_qnet(
            config.qnet,
            config.obs_shape,
            config.n_actions,
            **enc_kwargs,
        ).to(self.device)
        self.q2_target = create_qnet(
            config.qnet,
            config.obs_shape,
            config.n_actions,
            **enc_kwargs,
        ).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.q1_target.eval()
        self.q2_target.eval()
        self.q1_target.requires_grad_(False)
        self.q2_target.requires_grad_(False)

        # Automatic entropy temperature
        self.target_entropy = -np.log(1.0 / config.n_actions) * config.target_entropy_ratio
        self.log_alpha = torch.tensor(
            np.log(config.init_alpha), dtype=torch.float32, device=self.device, requires_grad=True
        )

        self._lr_actor = config.lr_actor
        self._lr_critic = config.lr_critic
        self._lr_alpha = config.lr_alpha
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.alpha_optimizer = None

        self.buffer = ReplayBuffer(config.buffer_size, config.obs_shape, config.device)

        self._amp_enabled = config.amp
        self._amp_device_type = resolve_amp_device_type(config.device)
        self._grad_accum_steps = config.grad_accum_steps

    def models(self):
        return [("actor", self.actor), ("q1", self.q1), ("q2", self.q2)]

    def target_models(self):
        return [("q1_target", self.q1_target), ("q2_target", self.q2_target)]

    def optimizers(self):
        if self.actor_optimizer is None:
            return []
        critic_container = nn.ModuleDict({"q1": self.q1, "q2": self.q2})
        return [
            ("actor_optimizer", self.actor_optimizer, self.actor),
            ("critic_optimizer", self.critic_optimizer, critic_container),
        ]

    def create_optimizers(self):
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self._lr_actor)
        self._critic_params = list(self.q1.parameters()) + list(self.q2.parameters())
        self.critic_optimizer = torch.optim.Adam(self._critic_params, lr=self._lr_critic)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self._lr_alpha)
        self.critic_scaler = create_grad_scaler(self._amp_device_type, self._amp_enabled)
        self.actor_scaler = create_grad_scaler(self._amp_device_type, self._amp_enabled)
        self.alpha_scaler = create_grad_scaler(self._amp_device_type, self._amp_enabled)
        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        self.alpha_optimizer.zero_grad()

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _select_actions(self, obs: torch.Tensor, deterministic: bool) -> torch.Tensor:
        with torch.no_grad():
            dist = Categorical(logits=self.actor(obs))
        if deterministic:
            return dist.probs.argmax(dim=-1)
        return dist.sample()

    @abstractmethod
    def get_action(self, obs: torch.Tensor, *, deterministic: bool = False):
        pass

    @abstractmethod
    def store(self, obs, action, reward, next_obs, done):
        pass

    def can_learn(self) -> bool:
        return len(self.buffer) >= self.learn_start and self.step_count % self.learn_freq == 0

    def _update_critics(self, obs, actions, rewards, next_obs, dones) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute TD targets and backward on critic loss. Returns (critic_loss, q1_values)."""
        with torch.amp.autocast(self._amp_device_type, enabled=self._amp_enabled):
            with torch.no_grad():
                next_dist = Categorical(logits=self.actor(next_obs))
                next_probs = next_dist.probs
                next_log_probs = next_dist.logits  # Categorical normalizes internally
                next_q_min = torch.min(self.q1_target(next_obs), self.q2_target(next_obs))
                next_v = (next_probs * (next_q_min - self.alpha * next_log_probs)).sum(dim=-1)
                target_q = rewards + self.gamma * next_v * (1.0 - dones)

            acts = actions.long().unsqueeze(1)
            q1_values = self.q1(obs).gather(1, acts).squeeze(1)
            q2_values = self.q2(obs).gather(1, acts).squeeze(1)
            loss = F.mse_loss(q1_values, target_q) + F.mse_loss(q2_values, target_q)

        self.critic_scaler.scale(loss / self._grad_accum_steps).backward()
        return loss, q1_values

    def _update_actor(self, obs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Backward on actor loss. Returns (actor_loss, probs, log_probs)."""
        with torch.amp.autocast(self._amp_device_type, enabled=self._amp_enabled):
            dist = Categorical(logits=self.actor(obs))
            probs = dist.probs
            log_probs = dist.logits  # Categorical normalizes internally
            q_min = torch.min(self.q1(obs), self.q2(obs)).detach()
            loss = (probs * (self.alpha.detach() * log_probs - q_min)).sum(dim=-1).mean()

        self.actor_scaler.scale(loss / self._grad_accum_steps).backward()
        return loss, probs, log_probs

    def _update_alpha(self, probs: torch.Tensor, log_probs: torch.Tensor) -> None:
        """Backward on alpha loss."""
        entropy_gap = (probs * (log_probs + self.target_entropy)).sum(dim=-1).mean()
        alpha_loss = -(self.log_alpha * entropy_gap.detach())
        self.alpha_scaler.scale(alpha_loss / self._grad_accum_steps).backward()

    def _step_optimizer(self, scaler, optimizer, parameters=None):
        """Unscale, optionally clip, step, and update a single optimizer."""
        scaler.unscale_(optimizer)
        if parameters is not None:
            clip_grad_norm_(parameters, self.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    def _accumulate_grads(self, obs, actions, rewards, next_obs, dones):
        """Split batch into micro-batches, forward + backward each to reduce peak memory."""
        micro = self.batch_size // self._grad_accum_steps
        total_critic_loss = 0.0
        total_actor_loss = 0.0
        total_q_mean = 0.0
        n_micro = 0
        for i in range(0, self.batch_size, micro):
            j = min(i + micro, self.batch_size)
            critic_loss, q1_values = self._update_critics(
                obs[i:j], actions[i:j], rewards[i:j], next_obs[i:j], dones[i:j]
            )
            actor_loss, probs, log_probs = self._update_actor(obs[i:j])
            self._update_alpha(probs, log_probs)
            total_critic_loss += critic_loss.item()
            total_actor_loss += actor_loss.item()
            total_q_mean += q1_values.mean().item()
            n_micro += 1
        return total_critic_loss / n_micro, total_actor_loss / n_micro, total_q_mean / n_micro

    def learn(self) -> dict[str, float]:
        batch = self.buffer.sample(self.batch_size)
        obs, actions, rewards, next_obs, dones = (
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["next_obs"],
            batch["dones"],
        )

        critic_loss, actor_loss, q_mean = self._accumulate_grads(obs, actions, rewards, next_obs, dones)
        self._step_optimizer(self.critic_scaler, self.critic_optimizer, self._critic_params)
        self._step_optimizer(self.actor_scaler, self.actor_optimizer, self.actor.parameters())
        self._step_optimizer(self.alpha_scaler, self.alpha_optimizer)
        self._soft_update()

        return {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "alpha": self.alpha.item(),
            "q_mean": q_mean,
        }

    def _soft_update(self):
        soft_update(self.q1, self.q1_target, self.tau)
        soft_update(self.q2, self.q2_target, self.tau)

    def _state(self, step: int = 0) -> dict:
        data = {
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "step": step,
            "step_count": self.step_count,
        }
        if self.actor_optimizer is not None:
            data["actor_optimizer"] = self.actor_optimizer.state_dict()
            data["critic_optimizer"] = self.critic_optimizer.state_dict()
            data["alpha_optimizer"] = self.alpha_optimizer.state_dict()
        for name in ("critic_scaler", "actor_scaler", "alpha_scaler"):
            scaler = getattr(self, name, None)
            if scaler is not None and scaler.is_enabled():
                data[name] = scaler.state_dict()
        return data

    def _load_optimizer_state(self, state: dict) -> int:
        """Restore optimizer, scaler, and alpha state from checkpoint."""
        self.log_alpha = state["log_alpha"].to(self.device).requires_grad_(True)
        if self.alpha_optimizer is not None:
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self._lr_alpha)
            if "alpha_optimizer" in state:
                self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
        if self.actor_optimizer is not None and "actor_optimizer" in state:
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        if self.critic_optimizer is not None and "critic_optimizer" in state:
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        for name in ("critic_scaler", "actor_scaler", "alpha_scaler"):
            scaler = getattr(self, name, None)
            if scaler is not None and name in state:
                scaler.load_state_dict(state[name])
        self.step_count = state.get("step_count", 0)
        return state.get("step", 0)

    def _load_state(self, state: dict) -> int:
        self.actor.load_state_dict(state["actor"])
        self.q1.load_state_dict(state["q1"])
        self.q2.load_state_dict(state["q2"])
        self.q1_target.load_state_dict(state["q1_target"])
        self.q2_target.load_state_dict(state["q2_target"])
        return self._load_optimizer_state(state)

    def load_non_model_state(self, state: dict) -> int:
        return self._load_optimizer_state(state)


class EnvSAC(EnvAgentMixin, SACBase):
    def get_action(self, obs: torch.Tensor, *, deterministic: bool = False) -> int:
        obs = to_tensor(obs, self.device)
        if obs.dim() < 4:
            obs = obs.unsqueeze(0)
        return self._select_actions(obs, deterministic).item()

    def store(self, obs, action, reward, next_obs, done):
        self.buffer.add(
            torch.as_tensor(obs, dtype=torch.float32),
            action,
            reward,
            torch.as_tensor(next_obs, dtype=torch.float32),
            done,
        )
        self.step_count += 1


class VecSAC(VecAgentMixin, SACBase):
    def get_action(self, obs: torch.Tensor, *, deterministic: bool = False) -> np.ndarray:
        obs = to_tensor(obs, self.device)
        return self._select_actions(obs, deterministic).cpu().numpy()

    def store(self, obs, action, reward, next_obs, done):
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
        next_obs_t = torch.as_tensor(np.asarray(next_obs), dtype=torch.float32)
        actions = np.atleast_1d(action)
        rewards = np.atleast_1d(reward)
        dones = np.atleast_1d(done)
        for i in range(obs_t.shape[0]):
            self.buffer.add(obs_t[i], actions[i], rewards[i], next_obs_t[i], dones[i])
        self.step_count += obs_t.shape[0]


def _create_sac(config: SACConfig, env=None):
    num_envs = env.num_envs if env is not None and is_vector_env(env) else 1
    config = replace(
        config,
        learn_start=config.learn_start * num_envs,
        learn_freq=config.learn_freq * num_envs,
    )

    if env is None or is_vector_env(env):
        return VecSAC(config)
    return EnvSAC(config)


class SAC(Agent):
    def __init__(self, config: SACConfig, env=None):
        self.config = config
        self.sac = _create_sac(config, env)
        super().__init__(self.sac)

    def store(self, obs, action, reward, next_obs, done):
        self.sac.store(obs, action, reward, next_obs, done)

    def can_learn(self) -> bool:
        return self.sac.can_learn()

    def learn(self) -> dict[str, float]:
        return self.sac.learn()

    @property
    def step_count(self):
        return self.sac.step_count

    @classmethod
    def restore(cls, config: SACConfig, f, env=None) -> SAC:
        agent = cls(config, env)
        agent.sac.actor.eval()
        agent.sac.q1.eval()
        agent.sac.q2.eval()
        agent.sac.load(f)
        return agent
