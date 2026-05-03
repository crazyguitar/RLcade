"""DQN agents — Double DQN and Rainbow DQN (C51 + PER + NoisyNet + Dueling + N-step).

Follows the same Env*/Vec* pattern as PPO for single vs vector env support.
"""

from __future__ import annotations

import argparse
from abc import abstractmethod
from collections import deque
from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn as nn

from rlcade.agent.base import AgentBase, EnvAgentMixin, VecAgentMixin, Agent, is_vector_env
from rlcade.modules import create_qnet, build_encoder_kwargs, parse_channels
from rlcade.utils import to_tensor, clip_grad_norm_, soft_update
from rlcade.utils.amp import resolve_amp_device_type, create_grad_scaler
from rlcade.utils.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

# DQN (Double DQN with dueling architecture and soft target updates)


@dataclass
class DQNConfig:
    obs_shape: tuple[int, ...]
    n_actions: int
    qnet: str = "qnet"
    lr: float = 1e-4
    gamma: float = 0.99
    tau: float = 1e-3
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 100_000
    batch_size: int = 32
    buffer_size: int = 100_000
    learn_start: int = 10_000
    learn_freq: int = 4
    double: bool = True
    max_grad_norm: float = 10.0
    device: str = "cpu"
    checkpoint: str | None = None
    encoder: str = "cnn"
    resnet_channels: list[int] | None = None
    resnet_out_dim: int = 256
    amp: bool = False
    grad_accum_steps: int = 1

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> DQNConfig:
        return cls(
            obs_shape=args.obs_shape,
            n_actions=args.n_actions,
            qnet=getattr(args, "qnet", "qnet"),
            lr=getattr(args, "lr", 1e-4),
            gamma=getattr(args, "gamma", 0.99),
            tau=getattr(args, "tau", 1e-3),
            epsilon_start=getattr(args, "epsilon_start", 1.0),
            epsilon_end=getattr(args, "epsilon_end", 0.01),
            epsilon_decay=getattr(args, "epsilon_decay", 100_000),
            batch_size=getattr(args, "batch_size", 32),
            buffer_size=getattr(args, "buffer_size", 100_000),
            learn_start=getattr(args, "learn_start", 10_000),
            learn_freq=getattr(args, "learn_freq", 4),
            double=getattr(args, "double", True),
            max_grad_norm=getattr(args, "max_grad_norm", 10.0),
            device=getattr(args, "device", "cpu"),
            checkpoint=getattr(args, "checkpoint", None),
            encoder=getattr(args, "encoder", "cnn"),
            resnet_channels=parse_channels(getattr(args, "resnet_channels", "16,32,32")),
            resnet_out_dim=getattr(args, "resnet_out_dim", 256),
            amp=getattr(args, "amp", False),
            grad_accum_steps=getattr(args, "grad_accum_steps", 1),
        )


class DQNBase(AgentBase):
    def __init__(self, config: DQNConfig):
        self.device = torch.device(config.device)
        self.n_actions = config.n_actions
        self.gamma = config.gamma
        self.tau = config.tau
        self.double = config.double
        self.batch_size = config.batch_size
        self.learn_start = config.learn_start
        self.learn_freq = config.learn_freq
        self.max_grad_norm = config.max_grad_norm

        self.epsilon_start = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay = config.epsilon_decay
        self.step_count = 0

        enc_kwargs = build_encoder_kwargs(config)
        self.qnet = create_qnet(
            config.qnet,
            config.obs_shape,
            config.n_actions,
            **enc_kwargs,
        ).to(self.device)
        self.target = create_qnet(
            config.qnet,
            config.obs_shape,
            config.n_actions,
            **enc_kwargs,
        ).to(self.device)
        self.target.load_state_dict(self.qnet.state_dict())
        self.target.eval()

        self.optimizer = None
        self._lr = config.lr
        self.criterion = nn.SmoothL1Loss()
        self.buffer = ReplayBuffer(config.buffer_size, config.obs_shape, config.device)
        self._amp_enabled = config.amp
        self._amp_device_type = resolve_amp_device_type(config.device)
        self._grad_accum_steps = config.grad_accum_steps

    def models(self):
        return [("qnet", self.qnet)]

    def target_models(self):
        return [("target", self.target)]

    def optimizers(self):
        if self.optimizer is None:
            return []
        return [("optimizer", self.optimizer, self.qnet)]

    def create_optimizers(self):
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self._lr)
        self.scaler = create_grad_scaler(self._amp_device_type, self._amp_enabled)
        self.optimizer.zero_grad()

    @property
    def epsilon(self) -> float:
        """Linear epsilon decay based on total transitions seen.

        With vectorized envs, step_count grows by num_envs per trainer step,
        so epsilon_decay should be scaled accordingly (e.g. 100K * num_envs).
        """
        frac = min(1.0, self.step_count / max(1, self.epsilon_decay))
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def _q_actions(self, obs: torch.Tensor, deterministic: bool) -> torch.Tensor:
        """Return greedy or epsilon-greedy actions for a batch of obs."""
        with torch.no_grad():
            q_values = self.qnet(obs)
        if deterministic:
            return q_values.argmax(dim=-1)
        batch = q_values.shape[0]
        greedy = q_values.argmax(dim=-1).cpu().numpy()
        random = np.random.randint(0, self.n_actions, size=batch)
        mask = np.random.random(batch) < self.epsilon
        return torch.as_tensor(np.where(mask, random, greedy))

    @abstractmethod
    def get_action(self, obs: torch.Tensor, *, deterministic: bool = False):
        pass

    @abstractmethod
    def store(self, obs, action, reward, next_obs, done):
        pass

    def can_learn(self) -> bool:
        return len(self.buffer) >= self.learn_start and self.step_count % self.learn_freq == 0

    @torch.no_grad()
    def _compute_target(self, rewards, next_obs, dones):
        """Compute TD target using double DQN or standard DQN."""
        if self.double:
            next_actions = self.qnet(next_obs).argmax(dim=1, keepdim=True)
            next_q = self.target(next_obs).gather(1, next_actions).squeeze(1)
        else:
            next_q = self.target(next_obs).max(dim=1).values
        return rewards + self.gamma * next_q * (1.0 - dones)

    def _forward_backward(self, obs, actions, rewards, next_obs, dones):
        """Forward + scaled backward on one micro-batch for gradient accumulation."""
        with torch.amp.autocast(self._amp_device_type, enabled=self._amp_enabled):
            q_values = self.qnet(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
            target = self._compute_target(rewards, next_obs, dones)
            loss = self.criterion(q_values, target)
        self.scaler.scale(loss / self._grad_accum_steps).backward()
        return loss

    def learn(self) -> dict[str, float]:
        batch = self.buffer.sample(self.batch_size)
        obs, actions, rewards, next_obs, dones = (
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["next_obs"],
            batch["dones"],
        )

        # Gradient accumulation: split batch into micro-batches to reduce
        # peak memory while keeping the same effective batch size.
        self.optimizer.zero_grad()
        micro = self.batch_size // self._grad_accum_steps
        for i in range(0, self.batch_size, micro):
            j = min(i + micro, self.batch_size)
            loss = self._forward_backward(obs[i:j], actions[i:j], rewards[i:j], next_obs[i:j], dones[i:j])

        self.scaler.unscale_(self.optimizer)
        clip_grad_norm_(self.qnet.parameters(), self.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self._soft_update()

        with torch.no_grad():
            q_values = self.qnet(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        return {"loss": loss.item(), "q_mean": q_values.mean().item(), "epsilon": self.epsilon}

    def _soft_update(self):
        soft_update(self.qnet, self.target, self.tau)

    def _state(self, step: int = 0) -> dict:
        data = {
            "qnet": self.qnet.state_dict(),
            "target": self.target.state_dict(),
            "step": step,
            "step_count": self.step_count,
        }
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if hasattr(self, "scaler") and self.scaler is not None and self.scaler.is_enabled():
            data["grad_scaler"] = self.scaler.state_dict()
        return data

    def _load_state(self, state: dict) -> int:
        self.qnet.load_state_dict(state["qnet"])
        self.target.load_state_dict(state["target"])
        if "optimizer" in state and self.optimizer is not None:
            self.optimizer.load_state_dict(state["optimizer"])
        if hasattr(self, "scaler") and self.scaler is not None and "grad_scaler" in state:
            self.scaler.load_state_dict(state["grad_scaler"])
        self.step_count = state.get("step_count", 0)
        return state.get("step", 0)

    def load_non_model_state(self, state: dict) -> int:
        if "optimizer" in state and self.optimizer is not None:
            self.optimizer.load_state_dict(state["optimizer"])
        self.step_count = state.get("step_count", 0)
        return state.get("step", 0)


class EnvDQN(EnvAgentMixin, DQNBase):
    def get_action(self, obs: torch.Tensor, *, deterministic: bool = False) -> int:
        obs = to_tensor(obs, self.device)
        if obs.dim() < 4:
            obs = obs.unsqueeze(0)
        return self._q_actions(obs, deterministic).item()

    def store(self, obs, action, reward, next_obs, done):
        self.buffer.add(
            torch.as_tensor(obs, dtype=torch.float32),
            action,
            reward,
            torch.as_tensor(next_obs, dtype=torch.float32),
            done,
        )
        self.step_count += 1


class VecDQN(VecAgentMixin, DQNBase):
    def get_action(self, obs: torch.Tensor, *, deterministic: bool = False) -> np.ndarray:
        obs = to_tensor(obs, self.device)
        return self._q_actions(obs, deterministic).cpu().numpy()

    def store(self, obs, action, reward, next_obs, done):
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
        next_obs_t = torch.as_tensor(np.asarray(next_obs), dtype=torch.float32)
        actions = np.atleast_1d(action)
        rewards = np.atleast_1d(reward)
        dones = np.atleast_1d(done)
        for i in range(obs_t.shape[0]):
            self.buffer.add(obs_t[i], actions[i], rewards[i], next_obs_t[i], dones[i])
        # Count total transitions, not calls — epsilon_decay should scale with num_envs
        self.step_count += obs_t.shape[0]


def _create_dqn(config: DQNConfig, env=None):
    # Scale by num_envs since step_count/buffer grow by num_envs per trainer step
    num_envs = env.num_envs if env is not None and is_vector_env(env) else 1
    config = replace(
        config,
        epsilon_decay=config.epsilon_decay * num_envs,
        learn_start=config.learn_start * num_envs,
        learn_freq=config.learn_freq * num_envs,
    )

    if env is None or is_vector_env(env):
        return VecDQN(config)
    return EnvDQN(config)


class DQN(Agent):
    def __init__(self, config: DQNConfig, env=None):
        self.config = config
        self.dqn = _create_dqn(config, env)
        super().__init__(self.dqn)

    def store(self, obs, action, reward, next_obs, done):
        self.dqn.store(obs, action, reward, next_obs, done)

    def can_learn(self) -> bool:
        return self.dqn.can_learn()

    def learn(self) -> dict[str, float]:
        return self.dqn.learn()

    @property
    def step_count(self):
        return self.dqn.step_count

    @property
    def epsilon(self):
        return self.dqn.epsilon

    @classmethod
    def restore(cls, config: DQNConfig, f, env=None) -> DQN:
        agent = cls(config, env)
        agent.dqn.qnet.eval()
        agent.dqn.target.eval()
        agent.dqn.load(f)
        return agent


# Rainbow DQN (C51 + PER + NoisyNet + Dueling + Double + N-step)


@dataclass
class RainbowDQNConfig:
    obs_shape: tuple[int, ...]
    n_actions: int
    qnet: str = "rainbow_qnet"
    lr: float = 6.25e-5
    gamma: float = 0.99
    tau: float = 1e-3
    batch_size: int = 32
    buffer_size: int = 100_000
    learn_start: int = 10_000
    learn_freq: int = 4
    # PER
    alpha: float = 0.6
    beta_start: float = 0.4
    beta_end: float = 1.0
    prior_eps: float = 1e-6
    # Distributional
    num_atoms: int = 51
    v_min: float = -200.0
    v_max: float = 200.0
    # NoisyNet
    noise_std: float = 0.5
    # N-step
    n_step: int = 3
    max_grad_norm: float = 10.0
    device: str = "cpu"
    checkpoint: str | None = None
    encoder: str = "cnn"
    resnet_channels: list[int] | None = None
    resnet_out_dim: int = 256
    amp: bool = False
    grad_accum_steps: int = 1

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> RainbowDQNConfig:
        return cls(
            obs_shape=args.obs_shape,
            n_actions=args.n_actions,
            qnet=getattr(args, "qnet", "rainbow_qnet"),
            lr=getattr(args, "lr", 6.25e-5),
            gamma=getattr(args, "gamma", 0.99),
            tau=getattr(args, "tau", 1e-3),
            batch_size=getattr(args, "batch_size", 32),
            buffer_size=getattr(args, "buffer_size", 100_000),
            learn_start=getattr(args, "learn_start", 10_000),
            learn_freq=getattr(args, "learn_freq", 4),
            alpha=getattr(args, "alpha", 0.6),
            beta_start=getattr(args, "beta_start", 0.4),
            beta_end=getattr(args, "beta_end", 1.0),
            prior_eps=getattr(args, "prior_eps", 1e-6),
            num_atoms=getattr(args, "num_atoms", 51),
            v_min=getattr(args, "v_min", -200.0),
            v_max=getattr(args, "v_max", 200.0),
            noise_std=getattr(args, "noise_std", 0.5),
            n_step=getattr(args, "n_step", 3),
            max_grad_norm=getattr(args, "max_grad_norm", 10.0),
            device=getattr(args, "device", "cpu"),
            checkpoint=getattr(args, "checkpoint", None),
            encoder=getattr(args, "encoder", "cnn"),
            resnet_channels=parse_channels(getattr(args, "resnet_channels", "16,32,32")),
            resnet_out_dim=getattr(args, "resnet_out_dim", 256),
            amp=getattr(args, "amp", False),
            grad_accum_steps=getattr(args, "grad_accum_steps", 1),
        )


class NStepBuffer:
    """Accumulates n-step returns before pushing to the main replay buffer."""

    def __init__(self, n_step: int, gamma: float):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer: deque[tuple] = deque()

    def append(self, transition: tuple) -> tuple | list[tuple] | None:
        """Append transition; return n-step transition(s) when ready or on episode end."""
        self.buffer.append(transition)
        obs, action, reward, next_obs, done = transition
        if done:
            return self._flush()
        if len(self.buffer) >= self.n_step:
            return self._pop()
        return None

    def _pop(self) -> tuple:
        obs, action, _, _, _ = self.buffer[0]
        reward = sum(self.gamma**i * t[2] for i, t in enumerate(self.buffer))
        _, _, _, next_obs, done = self.buffer[-1]
        self.buffer.popleft()
        return obs, action, reward, next_obs, done

    def _flush(self) -> list[tuple]:
        """Drain all remaining transitions as n-step returns."""
        results = []
        while self.buffer:
            obs, action, _, _, _ = self.buffer[0]
            reward = sum(self.gamma**i * t[2] for i, t in enumerate(self.buffer))
            _, _, _, next_obs, done = self.buffer[-1]
            results.append((obs, action, reward, next_obs, done))
            self.buffer.popleft()
        return results


class RainbowDQNBase(AgentBase):
    def __init__(self, config: RainbowDQNConfig):
        self.device = torch.device(config.device)
        self.n_actions = config.n_actions
        self.gamma = config.gamma
        self.tau = config.tau
        self.batch_size = config.batch_size
        self.learn_start = config.learn_start
        self.learn_freq = config.learn_freq
        self.max_grad_norm = config.max_grad_norm
        self.prior_eps = config.prior_eps
        self.n_step = config.n_step
        self.num_atoms = config.num_atoms
        self.v_min = config.v_min
        self.v_max = config.v_max
        self.step_count = 0

        # PER beta annealing state
        self.beta_start = config.beta_start
        self.beta_end = config.beta_end
        self.beta = config.beta_start

        enc_kwargs = build_encoder_kwargs(config)
        self.qnet = create_qnet(
            config.qnet,
            config.obs_shape,
            config.n_actions,
            num_atoms=config.num_atoms,
            v_min=config.v_min,
            v_max=config.v_max,
            noise_std=config.noise_std,
            **enc_kwargs,
        ).to(self.device)
        self.target = create_qnet(
            config.qnet,
            config.obs_shape,
            config.n_actions,
            num_atoms=config.num_atoms,
            v_min=config.v_min,
            v_max=config.v_max,
            noise_std=config.noise_std,
            **enc_kwargs,
        ).to(self.device)
        self.target.load_state_dict(self.qnet.state_dict())
        self.target.eval()

        self._lr = config.lr
        self.optimizer = None
        self.buffer = PrioritizedReplayBuffer(
            config.buffer_size,
            config.obs_shape,
            alpha=config.alpha,
            device=config.device,
        )

        self.support = self.qnet.support
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        self._amp_enabled = config.amp
        self._amp_device_type = resolve_amp_device_type(config.device)
        self._grad_accum_steps = config.grad_accum_steps

    def models(self):
        return [("qnet", self.qnet)]

    def target_models(self):
        return [("target", self.target)]

    def optimizers(self):
        if self.optimizer is None:
            return []
        return [("optimizer", self.optimizer, self.qnet)]

    def create_optimizers(self):
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self._lr, eps=1.5e-4)
        self.scaler = create_grad_scaler(self._amp_device_type, self._amp_enabled)
        self.optimizer.zero_grad()

    @abstractmethod
    def get_action(self, obs: torch.Tensor, *, deterministic: bool = False):
        pass

    @abstractmethod
    def store(self, obs, action, reward, next_obs, done):
        pass

    def _greedy_actions(self, obs: torch.Tensor) -> torch.Tensor:
        """NoisyNet provides exploration — no epsilon needed."""
        with torch.no_grad():
            q_values = self.qnet(obs)
        return q_values.argmax(dim=-1)

    def can_learn(self) -> bool:
        return len(self.buffer) >= self.learn_start and self.step_count % self.learn_freq == 0

    @torch.no_grad()
    def _project_target_distribution(self, rewards, next_obs, dones, gamma_n):
        """Project the target distribution onto the fixed support (C51)."""
        n = rewards.shape[0]
        next_actions = self.qnet(next_obs).argmax(dim=1)
        target_dist = self.target.dist(next_obs)
        target_dist = target_dist[range(n), next_actions]

        # index_add_ requires matching dtypes; run projection in float32
        with torch.amp.autocast(self._amp_device_type, enabled=False):
            target_dist = target_dist.float()
            t_z = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * gamma_n * self.support
            t_z = t_z.clamp(self.v_min, self.v_max)
            b = (t_z - self.v_min) / self.delta_z
            lo = b.floor().long()
            hi = b.ceil().long()

            lo[(hi > 0) * (hi == lo)] -= 1
            hi[((self.num_atoms - 1) > lo) * (hi == lo)] += 1

            proj_dist = torch.zeros_like(target_dist)
            offset = (
                torch.linspace(0, (n - 1) * self.num_atoms, n, device=self.device).long().unsqueeze(1).expand_as(lo)
            )
            proj_dist.view(-1).index_add_(0, (lo + offset).view(-1), (target_dist * (hi.float() - b)).view(-1))
            proj_dist.view(-1).index_add_(0, (hi + offset).view(-1), (target_dist * (b - lo.float())).view(-1))

        return proj_dist

    def _compute_loss(self, obs, actions, rewards, next_obs, dones, weights):
        """Compute elementwise and weighted C51 cross-entropy loss under AMP."""
        n = obs.shape[0]
        gamma_n = self.gamma**self.n_step

        with torch.amp.autocast(self._amp_device_type, enabled=self._amp_enabled):
            proj_dist = self._project_target_distribution(rewards, next_obs, dones, gamma_n)

            log_p = self.qnet(obs, log=True)
            log_p_actions = log_p[range(n), actions.long()]

            elementwise_loss = -(proj_dist * log_p_actions).sum(dim=1)
            loss = (weights * elementwise_loss).mean()

        return loss, elementwise_loss, log_p

    def _accumulate_grads(self, batch):
        """Split batch into micro-batches, forward + backward each to reduce peak memory."""
        micro = self.batch_size // self._grad_accum_steps
        all_elem_loss = []
        total_loss = 0.0
        all_log_p = []
        n_micro = 0
        for i in range(0, self.batch_size, micro):
            j = min(i + micro, self.batch_size)
            mb = {k: batch[k][i:j] for k in ("obs", "actions", "rewards", "next_obs", "dones", "weights")}
            loss, elementwise_loss, log_p = self._compute_loss(
                mb["obs"],
                mb["actions"],
                mb["rewards"],
                mb["next_obs"],
                mb["dones"],
                mb["weights"],
            )
            self.scaler.scale(loss / self._grad_accum_steps).backward()
            all_elem_loss.append(elementwise_loss.detach())
            total_loss += loss.item()
            all_log_p.append(log_p.detach())
            n_micro += 1
        return total_loss / n_micro, torch.cat(all_elem_loss), torch.cat(all_log_p)

    def learn(self) -> dict[str, float]:
        batch = self.buffer.sample(self.batch_size, beta=self.beta)

        self.optimizer.zero_grad()
        loss, elementwise_loss, log_p = self._accumulate_grads(batch)
        self.scaler.unscale_(self.optimizer)
        clip_grad_norm_(self.qnet.parameters(), self.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self._soft_update()
        self.qnet.reset_noise()
        self.target.reset_noise()

        priorities = elementwise_loss.cpu().numpy() + self.prior_eps
        self.buffer.update_priorities(batch["indices"], priorities)

        return {"loss": loss, "q_mean": log_p.exp().mul(self.support).sum(-1).mean().item()}

    def _soft_update(self):
        soft_update(self.qnet, self.target, self.tau)

    def _state(self, step: int = 0) -> dict:
        data = {
            "qnet": self.qnet.state_dict(),
            "target": self.target.state_dict(),
            "step": step,
            "step_count": self.step_count,
        }
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if hasattr(self, "scaler") and self.scaler is not None and self.scaler.is_enabled():
            data["grad_scaler"] = self.scaler.state_dict()
        return data

    def _load_state(self, state: dict) -> int:
        self.qnet.load_state_dict(state["qnet"])
        self.target.load_state_dict(state["target"])
        if "optimizer" in state and self.optimizer is not None:
            self.optimizer.load_state_dict(state["optimizer"])
        if hasattr(self, "scaler") and self.scaler is not None and "grad_scaler" in state:
            self.scaler.load_state_dict(state["grad_scaler"])
        self.step_count = state.get("step_count", 0)
        return state.get("step", 0)

    def load_non_model_state(self, state: dict) -> int:
        if "optimizer" in state and self.optimizer is not None:
            self.optimizer.load_state_dict(state["optimizer"])
        self.step_count = state.get("step_count", 0)
        return state.get("step", 0)


class EnvRainbowDQN(EnvAgentMixin, RainbowDQNBase):
    def __init__(self, config: RainbowDQNConfig):
        super().__init__(config)
        self.n_step_buffer = NStepBuffer(config.n_step, config.gamma)

    def get_action(self, obs: torch.Tensor, *, deterministic: bool = False) -> int:
        obs = to_tensor(obs, self.device)
        if obs.dim() < 4:
            obs = obs.unsqueeze(0)
        if deterministic:
            self.qnet.eval()
        action = self._greedy_actions(obs).item()
        if deterministic:
            self.qnet.train()
        return action

    def store(self, obs, action, reward, next_obs, done):
        transition = (
            torch.as_tensor(obs, dtype=torch.float32),
            action,
            reward,
            torch.as_tensor(next_obs, dtype=torch.float32),
            done,
        )
        result = self.n_step_buffer.append(transition)
        if isinstance(result, list):
            for r in result:
                self.buffer.add(*r)
        elif result is not None:
            self.buffer.add(*result)
        self.step_count += 1


class VecRainbowDQN(VecAgentMixin, RainbowDQNBase):
    def __init__(self, config: RainbowDQNConfig, num_envs: int):
        super().__init__(config)
        self.n_step_buffers = [NStepBuffer(config.n_step, config.gamma) for _ in range(num_envs)]

    def get_action(self, obs: torch.Tensor, *, deterministic: bool = False) -> np.ndarray:
        obs = to_tensor(obs, self.device)
        if deterministic:
            self.qnet.eval()
        actions = self._greedy_actions(obs).cpu().numpy()
        if deterministic:
            self.qnet.train()
        return actions

    def store(self, obs, action, reward, next_obs, done):
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
        next_obs_t = torch.as_tensor(np.asarray(next_obs), dtype=torch.float32)
        actions = np.atleast_1d(action)
        rewards = np.atleast_1d(reward)
        dones = np.atleast_1d(done)
        for i in range(obs_t.shape[0]):
            transition = (obs_t[i], actions[i], rewards[i], next_obs_t[i], dones[i])
            result = self.n_step_buffers[i].append(transition)
            if isinstance(result, list):
                for r in result:
                    self.buffer.add(*r)
            elif result is not None:
                self.buffer.add(*result)
        self.step_count += obs_t.shape[0]


def _create_rainbow_dqn(config: RainbowDQNConfig, env=None):
    num_envs = env.num_envs if env is not None and is_vector_env(env) else 1
    config = replace(
        config,
        learn_start=config.learn_start * num_envs,
        learn_freq=config.learn_freq * num_envs,
    )

    if env is None or is_vector_env(env):
        return VecRainbowDQN(config, num_envs)
    return EnvRainbowDQN(config)


class RainbowDQN(Agent):
    def __init__(self, config: RainbowDQNConfig, env=None):
        self.config = config
        self.rainbow = _create_rainbow_dqn(config, env)
        super().__init__(self.rainbow)

    def store(self, obs, action, reward, next_obs, done):
        self.rainbow.store(obs, action, reward, next_obs, done)

    def can_learn(self) -> bool:
        return self.rainbow.can_learn()

    def learn(self) -> dict[str, float]:
        return self.rainbow.learn()

    @property
    def step_count(self):
        return self.rainbow.step_count

    @property
    def beta(self):
        return self.rainbow.beta

    @beta.setter
    def beta(self, value: float):
        self.rainbow.beta = value

    @property
    def beta_start(self):
        return self.rainbow.beta_start

    @property
    def beta_end(self):
        return self.rainbow.beta_end

    @classmethod
    def restore(cls, config: RainbowDQNConfig, f, env=None) -> RainbowDQN:
        agent = cls(config, env)
        agent.rainbow.qnet.eval()
        agent.rainbow.target.eval()
        agent.rainbow.load(f)
        return agent
