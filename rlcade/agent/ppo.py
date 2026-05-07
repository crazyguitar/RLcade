from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
from abc import abstractmethod
from torch.distributions import Categorical
from rlcade.agent.base import AgentBase, EnvAgentMixin, VecAgentMixin, Agent, is_vector_env, strip_wrapper_prefixes
from rlcade.modules import create_actor, create_critic, build_encoder_kwargs, parse_channels
from rlcade.modules.lstm import LstmActorCritic
from rlcade.modules.icm import ICM
from rlcade.utils import PinMemory, clip_grad_norm_
from rlcade.utils.amp import resolve_amp_device_type, create_grad_scaler


def compute_gae(rewards, values, dones, next_value, next_done, gamma, gae_lambda):
    """Compute GAE advantages. Works for both scalar (T,) and vector (T, N) shapes."""
    n = len(rewards)
    advantages = torch.zeros_like(rewards)
    is_vec = rewards.dim() >= 2
    last_gae = torch.zeros_like(next_done) if is_vec else 0.0
    for t in reversed(range(n)):
        next_nonterminal = 1.0 - (next_done if t == n - 1 else dones[t] if is_vec else dones[t].item())
        nextval = next_value if t == n - 1 else values[t + 1]
        delta = rewards[t] + gamma * nextval * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae
    return advantages


def compute_policy_loss(ratio, advantages, clip_coef):
    pg1 = -advantages * ratio
    pg2 = -advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
    return torch.max(pg1, pg2).mean()


def compute_value_loss(new_values, old_values, returns, clip_coef):
    v_clipped = old_values + (new_values - old_values).clamp(-clip_coef, clip_coef)
    loss_unclipped = (new_values - returns).square()
    loss_clipped = (v_clipped - returns).square()
    return 0.5 * torch.max(loss_unclipped, loss_clipped).mean()


def accumulate_metrics(agg: dict[str, float], metrics: dict[str, float]) -> None:
    for k, v in metrics.items():
        agg[k] = agg.get(k, 0.0) + v


_ICM_CHUNK = 4096  # max frames per ICM forward pass to bound GPU memory


@torch.no_grad()
def icm_intrinsic_reward(icm: ICM, obs: torch.Tensor, next_obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Compute ICM intrinsic reward in chunks to avoid OOM on large rollouts."""
    n = obs.shape[0]
    rewards = torch.empty(n, device=obs.device)
    for i in range(0, n, _ICM_CHUNK):
        j = min(i + _ICM_CHUNK, n)
        rewards[i:j], _, _ = icm(obs[i:j], next_obs[i:j], actions[i:j])
    return rewards


def icm_loss(icm: ICM, obs: torch.Tensor, next_obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Compute ICM forward+inverse loss in chunks to bound activation memory."""
    n = obs.shape[0]
    total_fwd = torch.tensor(0.0, device=obs.device)
    total_inv = torch.tensor(0.0, device=obs.device)
    for i in range(0, n, _ICM_CHUNK):
        j = min(i + _ICM_CHUNK, n)
        _, fwd, inv = icm(obs[i:j], next_obs[i:j], actions[i:j])
        total_fwd = total_fwd + fwd * (j - i)
        total_inv = total_inv + inv * (j - i)
    return (total_fwd + total_inv) / n


@dataclass
class PPOConfig:
    obs_shape: tuple[int, ...]
    n_actions: int
    actor: str = "actor"
    critic: str = "critic"
    lr: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    batch_size: int = 256
    device: str = "cpu"
    checkpoint: str | None = None
    lr_schedule: bool = False
    max_iterations: int = 5000
    encoder: str = "cnn"
    resnet_channels: list[int] | None = None
    resnet_out_dim: int = 256
    # ICM
    icm: bool = False
    icm_coef: float = 0.01
    icm_feature_dim: int = 256
    # AMP and gradient accumulation
    amp: bool = False
    grad_accum_steps: int = 1
    # Host/device transfer optimization
    pin_memory: bool = True

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> PPOConfig:
        return cls(
            obs_shape=args.obs_shape,
            n_actions=args.n_actions,
            actor=args.actor,
            critic=args.critic,
            lr=getattr(args, "lr", 2.5e-4),
            gamma=getattr(args, "gamma", 0.99),
            gae_lambda=getattr(args, "gae_lambda", 0.95),
            clip_coef=getattr(args, "clip_coef", 0.2),
            ent_coef=getattr(args, "ent_coef", 0.01),
            vf_coef=getattr(args, "vf_coef", 0.5),
            max_grad_norm=getattr(args, "max_grad_norm", 0.5),
            update_epochs=getattr(args, "update_epochs", 4),
            batch_size=getattr(args, "batch_size", 256),
            device=getattr(args, "device", "cpu"),
            checkpoint=getattr(args, "checkpoint", None),
            lr_schedule=getattr(args, "lr_schedule", False),
            max_iterations=getattr(args, "max_iterations", 5000),
            encoder=getattr(args, "encoder", "cnn"),
            resnet_channels=parse_channels(getattr(args, "resnet_channels", "16,32,32")),
            resnet_out_dim=getattr(args, "resnet_out_dim", 256),
            icm=getattr(args, "icm", False),
            icm_coef=getattr(args, "icm_coef", 0.01),
            icm_feature_dim=getattr(args, "icm_feature_dim", 256),
            amp=getattr(args, "amp", False),
            grad_accum_steps=getattr(args, "grad_accum_steps", 1),
            pin_memory=getattr(args, "pin_memory", True),
        )


class PPOBase(AgentBase):
    def __init__(self, config: PPOConfig):
        self.device = torch.device(config.device)
        enc_kwargs = build_encoder_kwargs(config)
        self.actor = create_actor(config.actor, config.obs_shape, config.n_actions, **enc_kwargs).to(self.device)
        self.critic = create_critic(config.critic, config.obs_shape, **enc_kwargs).to(self.device)
        self.icm = None
        self.icm_coef = config.icm_coef
        if config.icm:
            self.icm = ICM(config.obs_shape, config.n_actions, config.icm_feature_dim).to(self.device)
        self._lr = config.lr
        self._lr_schedule = config.lr_schedule
        self._max_iterations = config.max_iterations
        self.optimizer = None
        self.scheduler = None
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.clip_coef = config.clip_coef
        self.ent_coef = config.ent_coef
        self.vf_coef = config.vf_coef
        self.max_grad_norm = config.max_grad_norm
        self.update_epochs = config.update_epochs
        self.batch_size = config.batch_size
        self.target_kl: float | None = None
        self._amp_enabled = config.amp
        self._amp_device_type = resolve_amp_device_type(config.device)
        self._grad_accum_steps = config.grad_accum_steps
        # Agent owns PinMemory because rollout H2D happens inside collect_rollout.
        # (Off-policy trainers own their own PinMemory — H2D is at trainer layer.)
        self._pin_memory = PinMemory(enabled=config.pin_memory)

    def create_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters, lr=self._lr, eps=1e-5)
        self.scheduler = None
        if self._lr_schedule:
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1.0, end_factor=0.0, total_iters=self._max_iterations
            )
        self.scaler = create_grad_scaler(self._amp_device_type, self._amp_enabled)

    @property
    def parameters(self):
        params = list(self.actor.parameters()) + list(self.critic.parameters())
        if self.icm is not None:
            params += list(self.icm.parameters())
        return params

    def models(self):
        ms = [("actor", self.actor), ("critic", self.critic)]
        if self.icm is not None:
            ms.append(("icm", self.icm))
        return ms

    def optimizers(self):
        if self.optimizer is None:
            return []
        container = nn.ModuleDict({k: m for k, m in self.models()})
        return [("optimizer", self.optimizer, container)]

    @abstractmethod
    def collect_rollout(self, env, num_steps: int, obs: torch.Tensor | None = None):
        pass

    @abstractmethod
    def get_action(
        self, obs: torch.Tensor, *, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    @staticmethod
    def _sample_action(dist, deterministic: bool):
        action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        dist = Categorical(logits=self.actor(obs))
        value = self.critic(obs)
        return dist.log_prob(actions), dist.entropy(), value

    def compute_gae(self, rewards, values, dones, next_value, next_done):
        return compute_gae(rewards, values, dones, next_value, next_done, self.gamma, self.gae_lambda)

    def bootstrap_value(self, next_obs: torch.Tensor):
        with torch.no_grad():
            if next_obs.dim() < 4:
                next_obs = next_obs.unsqueeze(0)
            return self.critic(next_obs).squeeze()

    def process_trajectory(self, rollout: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        next_value = self.bootstrap_value(rollout["next_obs"])

        rewards = rollout["rewards"]
        # Augment rewards with ICM intrinsic curiosity
        if self.icm is not None:
            obs = rollout["obs"]
            next_obs_seq = torch.cat([obs[1:], rollout["next_obs"].unsqueeze(0)], dim=0)

            flat_obs = obs.reshape(-1, *obs.shape[-3:])
            flat_next = next_obs_seq.reshape(-1, *obs.shape[-3:])
            flat_actions = rollout["actions"].reshape(-1)
            intrinsic = icm_intrinsic_reward(self.icm, flat_obs, flat_next, flat_actions)
            rewards = rewards + self.icm_coef * intrinsic.reshape(rewards.shape)

        advantages = self.compute_gae(
            rewards,
            rollout["values"],
            rollout["dones"],
            next_value,
            rollout["next_done"],
        )

        returns = advantages + rollout["values"]

        # Normalize advantages per-rollout (not per-minibatch)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        trajectory = dict(
            obs=rollout["obs"],
            actions=rollout["actions"],
            old_log_probs=rollout["log_probs"],
            values=rollout["values"],
            advantages=advantages,
            returns=returns,
        )

        # Store next_obs for ICM training loss
        if self.icm is not None:
            trajectory["next_obs"] = next_obs_seq

        return trajectory

    def update_step(self, batch: dict[str, torch.Tensor], scale: float = 1.0) -> dict[str, float]:
        with torch.amp.autocast(self._amp_device_type, enabled=self._amp_enabled):
            new_log_probs, entropy, new_values = self.evaluate_actions(batch["obs"], batch["actions"])

            ratio = (new_log_probs - batch["old_log_probs"]).exp()
            adv = batch["advantages"]

            policy_loss = compute_policy_loss(ratio, adv, self.clip_coef)
            value_loss = compute_value_loss(new_values, batch["values"], batch["returns"], self.clip_coef)
            entropy_mean = entropy.mean()
            loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_mean

            # ICM forward/inverse loss
            icm_loss_val = torch.tensor(0.0, device=loss.device)
            if self.icm is not None and "next_obs" in batch:
                icm_loss_val = icm_loss(self.icm, batch["obs"], batch["next_obs"], batch["actions"])
                loss = loss + icm_loss_val

        self.scaler.scale(loss * scale).backward()

        with torch.no_grad():
            clip_fraction = ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
            approx_kl = self.compute_kl(ratio, new_log_probs, batch["old_log_probs"])

        metrics = {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy_mean.item(),
            "kl": approx_kl,
            "clip_fraction": clip_fraction,
        }
        if self.icm is not None:
            metrics["icm_loss"] = icm_loss_val.item()
        return metrics

    @staticmethod
    @torch.no_grad()
    def compute_kl(ratio, new_log_probs, old_log_probs) -> float:
        log_ratio = new_log_probs - old_log_probs
        return ((ratio - 1) - log_ratio).mean().item()

    def _accumulate_minibatch(self, batch):
        """Split a minibatch into micro-batches to reduce peak memory."""
        mb_size = len(batch["obs"])
        scale = 1.0 / self._grad_accum_steps
        micro = max(1, mb_size // self._grad_accum_steps)
        agg: dict[str, float] = {}
        n_micro = 0
        for mi in range(0, mb_size, micro):
            micro_batch = {k: v[mi : mi + micro] for k, v in batch.items()}
            metrics = self.update_step(micro_batch, scale=scale)
            accumulate_metrics(agg, metrics)
            n_micro += 1
        return {k: v / n_micro for k, v in agg.items()}

    def update_epoch(self, trajectory: dict[str, torch.Tensor]):
        n = len(trajectory["obs"])
        idxs = torch.randperm(n, device=self.device)
        agg: dict[str, float] = {}
        num_updates = 0
        last_kl = 0.0

        for start in range(0, n, self.batch_size):
            mb = idxs[start : start + self.batch_size]
            batch = {k: v[mb] for k, v in trajectory.items()}

            self.optimizer.zero_grad()
            metrics = self._accumulate_minibatch(batch)
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.parameters, self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            accumulate_metrics(agg, metrics)
            last_kl = metrics["kl"]
            num_updates += 1

        return agg, num_updates, last_kl

    def learn(self, rollout: dict[str, torch.Tensor]) -> dict[str, float]:
        # Batch-transfer rollout to training device (eliminates per-step GPU syncs)
        rollout = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in rollout.items()}
        trajectory = self.process_trajectory(rollout)
        agg: dict[str, float] = {}
        num_updates = 0

        for _ in range(self.update_epochs):
            epoch_agg, epoch_updates, epoch_kl = self.update_epoch(trajectory)
            accumulate_metrics(agg, epoch_agg)
            num_updates += epoch_updates
            if self.target_kl is not None and epoch_kl > self.target_kl:
                break

        if self.scheduler is not None:
            self.scheduler.step()

        return {k: v / max(1, num_updates) for k, v in agg.items()}

    def _state(self, step: int = 0) -> dict:
        data = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "step": step,
        }
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.icm is not None:
            data["icm"] = self.icm.state_dict()
        if hasattr(self, "scaler") and self.scaler is not None and self.scaler.is_enabled():
            data["grad_scaler"] = self.scaler.state_dict()
        return data

    def _load_state(self, state: dict) -> int:
        self.actor.load_state_dict(strip_wrapper_prefixes(state["actor"]))
        self.critic.load_state_dict(strip_wrapper_prefixes(state["critic"]))
        if self.optimizer is not None and "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        if self.icm is not None and "icm" in state:
            self.icm.load_state_dict(strip_wrapper_prefixes(state["icm"]))
        if hasattr(self, "scaler") and self.scaler is not None and "grad_scaler" in state:
            self.scaler.load_state_dict(state["grad_scaler"])
        return state.get("step", 0)

    def load_non_model_state(self, state: dict) -> int:
        if self.optimizer is not None and "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        return state.get("step", 0)

    def reset(self) -> None:
        # Obs shape may change on env swap — drop cached pinned buffers.
        self._pin_memory.reset()


class EnvPPO(EnvAgentMixin, PPOBase):
    def collect_rollout(self, env, num_steps: int, obs: torch.Tensor | None = None):
        if obs is None:
            obs, _ = env.reset()
            obs = self._pin_memory.to(obs, self.device)

        autoreset = getattr(env, "autoreset", False)
        all_obs, actions, log_probs, rewards, dones, values = [], [], [], [], [], []

        for _ in range(num_steps):
            all_obs.append(obs)
            action, log_prob, value = self.get_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)
            values.append(value)

            obs = self._pin_memory.to(next_obs, self.device)
            if done and not autoreset:
                next_reset, _ = env.reset()
                obs = self._pin_memory.to(next_reset, self.device)

        rollout = dict(
            obs=torch.stack(all_obs),
            actions=torch.stack(actions),
            log_probs=torch.stack(log_probs),
            rewards=torch.tensor(rewards, dtype=torch.float32, device=self.device),
            dones=torch.tensor(dones, dtype=torch.float32, device=self.device),
            values=torch.stack(values),
            next_obs=obs,
            next_done=float(done),
        )
        return rollout, obs

    def get_action(
        self, obs: torch.Tensor, *, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            if obs.dim() < 4:
                obs = obs.unsqueeze(0)
            dist = Categorical(logits=self.actor(obs))
            value = self.critic(obs)
            action, log_prob = self._sample_action(dist, deterministic)
        return action.squeeze(0), log_prob.squeeze(0), value.squeeze(0)

    def act(self, obs: torch.Tensor, *, deterministic: bool = False):
        return self.get_action(obs, deterministic=deterministic)[0].item()


class VecPPO(VecAgentMixin, PPOBase):
    def collect_rollout(self, env, num_steps: int, obs: torch.Tensor | None = None):
        if obs is None:
            obs = torch.as_tensor(env.reset()[0], dtype=torch.float32)
        elif obs.device.type != "cpu":
            obs = obs.cpu()

        all_obs, actions, log_probs, rewards, dones, values = [], [], [], [], [], []

        for _ in range(num_steps):
            all_obs.append(obs)
            action, log_prob, value = self.get_action(obs)
            next_obs_array, rewards_array, terminated, truncated, _ = env.step(action.numpy())
            dones_array = terminated | truncated

            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(torch.as_tensor(rewards_array, dtype=torch.float32))
            dones.append(torch.as_tensor(dones_array, dtype=torch.float32))
            values.append(value)

            obs = torch.as_tensor(next_obs_array, dtype=torch.float32)

        # All tensors on CPU — transferred to self.device in learn()
        rollout = dict(
            obs=torch.stack(all_obs),  # (T, N, *obs_shape)
            actions=torch.stack(actions),  # (T, N)
            log_probs=torch.stack(log_probs),  # (T, N)
            rewards=torch.stack(rewards),  # (T, N)
            dones=torch.stack(dones),  # (T, N)
            values=torch.stack(values),  # (T, N)
            next_obs=obs,  # (N, *obs_shape)
            next_done=torch.as_tensor(dones_array, dtype=torch.float32),
        )
        return rollout, obs

    def get_action(
        self, obs: torch.Tensor, *, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            obs_dev = self._pin_memory.to(obs, self.device)
            dist = Categorical(logits=self.actor(obs_dev))
            value = self.critic(obs_dev)
            action, log_prob = self._sample_action(dist, deterministic)
        # Return on CPU to avoid per-step GPU→CPU syncs in collect_rollout
        return action.cpu(), log_prob.cpu(), value.cpu()

    def act(self, obs: torch.Tensor, *, deterministic: bool = False):
        return self.get_action(obs, deterministic=deterministic)[0].numpy()

    def compute_gae(self, rewards, values, dones, next_value, next_done):
        """Per-env GAE on (T, N) shaped tensors."""
        return compute_gae(rewards, values, dones, next_value, next_done, self.gamma, self.gae_lambda)

    def bootstrap_value(self, next_obs: torch.Tensor):
        with torch.no_grad():
            return self.critic(next_obs)  # (N,)

    def process_trajectory(self, rollout: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        next_value = self.bootstrap_value(rollout["next_obs"])

        rewards = rollout["rewards"]
        obs = rollout["obs"]

        # Build next_obs sequence for ICM
        next_obs_seq = None
        if self.icm is not None:
            next_obs_seq = torch.cat([obs[1:], rollout["next_obs"].unsqueeze(0)], dim=0)
            flat_obs = obs.reshape(-1, *obs.shape[2:])
            flat_next = next_obs_seq.reshape(-1, *obs.shape[2:])
            flat_actions = rollout["actions"].reshape(-1)
            intrinsic = icm_intrinsic_reward(self.icm, flat_obs, flat_next, flat_actions)
            rewards = rewards + self.icm_coef * intrinsic.reshape(rewards.shape)

        advantages = self.compute_gae(
            rewards,
            rollout["values"],
            rollout["dones"],
            next_value,
            rollout["next_done"],
        )
        returns = advantages + rollout["values"]

        # Normalize advantages per-rollout (not per-minibatch)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Flatten (T, N, ...) -> (T*N, ...) for minibatch training
        obs_shape = obs.shape[2:]
        trajectory = dict(
            obs=obs.reshape(-1, *obs_shape),
            actions=rollout["actions"].reshape(-1),
            old_log_probs=rollout["log_probs"].reshape(-1),
            values=rollout["values"].reshape(-1),
            advantages=advantages.reshape(-1),
            returns=returns.reshape(-1),
        )
        if self.icm is not None and next_obs_seq is not None:
            trajectory["next_obs"] = next_obs_seq.reshape(-1, *obs_shape)
        return trajectory


def _create_ppo(config: PPOConfig, env=None):
    if env is None or is_vector_env(env):
        return VecPPO(config)
    return EnvPPO(config)


class PPO(Agent):
    def __init__(self, config: PPOConfig, env=None):
        self.config = config
        self.ppo = _create_ppo(config, env)
        super().__init__(self.ppo)

    def collect_rollout(self, env, num_steps: int, obs: torch.Tensor | None = None):
        return self.ppo.collect_rollout(env, num_steps, obs)

    def learn(self, rollout: dict[str, torch.Tensor]):
        return self.ppo.learn(rollout)

    @classmethod
    def restore(cls, config: PPOConfig, f, env=None) -> PPO:
        agent = cls(config, env)
        agent.ppo.actor.eval()
        agent.ppo.critic.eval()
        agent.ppo.load(f)
        return agent


# LSTM-PPO


@dataclass
class LstmPPOConfig:
    obs_shape: tuple[int, ...]
    n_actions: int
    lstm_hidden: int = 256
    lr: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    batch_size: int = 256
    device: str = "cpu"
    checkpoint: str | None = None
    lr_schedule: bool = False
    max_iterations: int = 5000
    # ICM
    icm: bool = False
    icm_coef: float = 0.01
    icm_feature_dim: int = 256
    # AMP and gradient accumulation
    amp: bool = False
    grad_accum_steps: int = 1
    # Host/device transfer optimization
    pin_memory: bool = True

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> LstmPPOConfig:
        return cls(
            obs_shape=args.obs_shape,
            n_actions=args.n_actions,
            lstm_hidden=getattr(args, "lstm_hidden", 256),
            lr=getattr(args, "lr", 2.5e-4),
            gamma=getattr(args, "gamma", 0.99),
            gae_lambda=getattr(args, "gae_lambda", 0.95),
            clip_coef=getattr(args, "clip_coef", 0.2),
            ent_coef=getattr(args, "ent_coef", 0.01),
            vf_coef=getattr(args, "vf_coef", 0.5),
            max_grad_norm=getattr(args, "max_grad_norm", 0.5),
            update_epochs=getattr(args, "update_epochs", 4),
            batch_size=getattr(args, "batch_size", 256),
            device=getattr(args, "device", "cpu"),
            checkpoint=getattr(args, "checkpoint", None),
            lr_schedule=getattr(args, "lr_schedule", False),
            max_iterations=getattr(args, "max_iterations", 5000),
            icm=getattr(args, "icm", False),
            icm_coef=getattr(args, "icm_coef", 0.01),
            icm_feature_dim=getattr(args, "icm_feature_dim", 256),
            amp=getattr(args, "amp", False),
            grad_accum_steps=getattr(args, "grad_accum_steps", 1),
            pin_memory=getattr(args, "pin_memory", True),
        )


class LstmPPOBase(AgentBase):
    """Shared PPO+LSTM logic. Sequential updates, no shuffle."""

    def __init__(self, config: LstmPPOConfig):
        self.device = torch.device(config.device)
        self.model = LstmActorCritic(
            config.obs_shape,
            config.n_actions,
            config.lstm_hidden,
        ).to(self.device)
        self.icm = None
        self.icm_coef = config.icm_coef
        if config.icm:
            self.icm = ICM(config.obs_shape, config.n_actions, config.icm_feature_dim).to(self.device)
        self._lr = config.lr
        self._lr_schedule = config.lr_schedule
        self._max_iterations = config.max_iterations
        self.optimizer = None
        self.scheduler = None
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.clip_coef = config.clip_coef
        self.ent_coef = config.ent_coef
        self.vf_coef = config.vf_coef
        self.max_grad_norm = config.max_grad_norm
        self.update_epochs = config.update_epochs
        self.batch_size = config.batch_size
        self.hidden = None
        self._amp_enabled = config.amp
        self._amp_device_type = resolve_amp_device_type(config.device)
        self._grad_accum_steps = config.grad_accum_steps
        # Agent owns PinMemory because rollout H2D happens inside collect_rollout.
        # (Off-policy trainers own their own PinMemory — H2D is at trainer layer.)
        self._pin_memory = PinMemory(enabled=config.pin_memory)

    def models(self):
        ms = [("model", self.model)]
        if self.icm is not None:
            ms.append(("icm", self.icm))
        return ms

    def target_models(self):
        return []

    def optimizers(self):
        if self.optimizer is None:
            return []
        container = nn.ModuleDict({k: m for k, m in self.models()})
        return [("optimizer", self.optimizer, container)]

    def create_optimizers(self):
        params = list(self.model.parameters())
        if self.icm is not None:
            params += list(self.icm.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self._lr, eps=1e-5)
        self.scheduler = None
        if self._lr_schedule:
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1.0, end_factor=0.0, total_iters=self._max_iterations
            )
        self.scaler = create_grad_scaler(self._amp_device_type, self._amp_enabled)

    def _zero_hidden(self, batch_size):
        return self.model.initial_state(batch_size, self.device)

    @staticmethod
    def _zero_done_hidden(hx, cx, dones, device):
        """Zero out hidden state for terminated envs."""
        if not dones.any():
            return hx, cx
        mask = torch.as_tensor(~dones, dtype=torch.float32, device=device).unsqueeze(-1)
        return hx * mask, cx * mask

    @torch.no_grad()
    def _forward(self, obs, hx, cx, deterministic=False):
        """Single forward pass. Returns (action, log_prob, value, hx, cx)."""
        dist, value, hx, cx = self.model(obs, (hx, cx))
        if deterministic:
            action = dist.probs.argmax(-1)
        else:
            action = dist.sample()
        return action, dist.log_prob(action), value, hx, cx

    def _replay_sequence(self, obs, actions, hx, cx, dones=None):
        """Re-run obs through LSTM in order. Returns flat tensors.

        If *dones* is provided, hidden state is zeroed at episode boundaries
        to match the rollout collection behavior.
        """
        is_vec = obs.dim() == 5
        log_probs, entropies, values = [], [], []
        for t in range(obs.shape[0]):
            # Zero hidden state at done boundaries (mirrors collect_rollout)
            if dones is not None and t > 0:
                if is_vec:
                    hx, cx = self._zero_done_hidden(hx, cx, dones[t - 1].bool(), self.device)
                elif dones[t - 1]:
                    hx, cx = self._zero_hidden(1)
            obs_t = obs[t] if is_vec else obs[t].unsqueeze(0)
            dist, val, hx, cx = self.model(obs_t, (hx, cx))
            log_probs.append(dist.log_prob(actions[t]))
            entropies.append(dist.entropy())
            values.append(val)
        return torch.cat(log_probs), torch.cat(entropies), torch.cat(values)

    def _ppo_step(self, chunk, trajectory, start, end, scale=1.0):
        """One gradient step on a sequential chunk."""
        with torch.amp.autocast(self._amp_device_type, enabled=self._amp_enabled):
            new_lp, ent, new_val = self._replay_sequence(
                chunk["obs"],
                chunk["actions"],
                trajectory["saved_hx"][start],
                trajectory["saved_cx"][start],
                dones=chunk.get("dones"),
            )
            flat = {
                k: trajectory[k][start:end].reshape(-1) for k in ("old_log_probs", "advantages", "returns", "values")
            }
            loss, metrics = self._compute_loss(new_lp, ent, new_val, flat)

            # ICM forward/inverse loss
            if self.icm is not None and "next_obs" in trajectory:
                obs_chunk = chunk["obs"].reshape(-1, *chunk["obs"].shape[-3:])
                next_chunk = trajectory["next_obs"][start:end].reshape(-1, *chunk["obs"].shape[-3:])
                act_chunk = chunk["actions"].reshape(-1)
                icm_loss_val = icm_loss(self.icm, obs_chunk, next_chunk, act_chunk)
                loss = loss + icm_loss_val
                metrics["icm_loss"] = icm_loss_val.item()

        self.scaler.scale(loss * scale).backward()
        return metrics

    def _compute_loss(self, new_lp, entropy, new_val, flat):
        """Clipped PPO loss -> (loss_tensor, metrics_dict)."""
        adv = flat["advantages"]
        ratio = (new_lp - flat["old_log_probs"]).exp()

        policy_loss = compute_policy_loss(ratio, adv, self.clip_coef)
        value_loss = compute_value_loss(new_val, flat["values"], flat["returns"], self.clip_coef)
        ent_mean = entropy.mean()
        loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * ent_mean
        metrics = {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": ent_mean.item(),
        }
        return loss, metrics

    def _accumulate_window(self, trajectory, start, end):
        """Split a sequential window into micro-chunks to reduce peak memory."""
        scale = 1.0 / self._grad_accum_steps
        micro = max(1, self.batch_size // self._grad_accum_steps)
        agg: dict[str, float] = {}
        n_micro = 0
        for ms in range(start, end, micro):
            me = min(ms + micro, end)
            chunk = {k: trajectory[k][ms:me] for k in ("obs", "actions", "dones")}
            metrics = self._ppo_step(chunk, trajectory, ms, me, scale=scale)
            accumulate_metrics(agg, metrics)
            n_micro += 1
        return {k: v / n_micro for k, v in agg.items()}

    def _update_epoch(self, trajectory):
        num_steps = trajectory["obs"].shape[0]
        agg, num_updates = {}, 0

        for start in range(0, num_steps, self.batch_size):
            end = min(start + self.batch_size, num_steps)
            self.optimizer.zero_grad()
            metrics = self._accumulate_window(trajectory, start, end)
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            num_updates += 1
            accumulate_metrics(agg, metrics)

        return agg, num_updates

    def learn(self, rollout):
        rollout = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in rollout.items()}
        trajectory = self._process_trajectory(rollout)
        agg, num_updates = {}, 0
        for _ in range(self.update_epochs):
            epoch_agg, epoch_n = self._update_epoch(trajectory)
            accumulate_metrics(agg, epoch_agg)
            num_updates += epoch_n
        if self.scheduler is not None:
            self.scheduler.step()
        return {k: v / max(1, num_updates) for k, v in agg.items()}

    def _process_trajectory(self, rollout):
        next_value = self._bootstrap_value(rollout)
        rewards = rollout["rewards"]
        obs = rollout["obs"]

        # Build next_obs sequence for ICM
        next_obs_seq = None
        if self.icm is not None:
            next_obs_seq = torch.cat([obs[1:], rollout["next_obs"].unsqueeze(0)], dim=0)
            flat_obs = obs.reshape(-1, *obs.shape[-3:])
            flat_next = next_obs_seq.reshape(-1, *obs.shape[-3:])
            flat_actions = rollout["actions"].reshape(-1)
            intrinsic = icm_intrinsic_reward(self.icm, flat_obs, flat_next, flat_actions)
            rewards = rewards + self.icm_coef * intrinsic.reshape(rewards.shape)

        advantages = compute_gae(
            rewards,
            rollout["values"],
            rollout["dones"],
            next_value,
            rollout["next_done"],
            self.gamma,
            self.gae_lambda,
        )
        returns = advantages + rollout["values"]

        # Normalize advantages per-rollout (not per-minibatch)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        trajectory = dict(
            obs=rollout["obs"],
            actions=rollout["actions"],
            dones=rollout["dones"],
            old_log_probs=rollout["log_probs"],
            values=rollout["values"],
            advantages=advantages,
            returns=returns,
            saved_hx=rollout["saved_hx"],
            saved_cx=rollout["saved_cx"],
        )
        if self.icm is not None and next_obs_seq is not None:
            trajectory["next_obs"] = next_obs_seq
        return trajectory

    def _bootstrap_value(self, rollout):
        is_vec = rollout["obs"].dim() == 5
        batch = rollout["obs"].shape[1] if is_vec else 1
        hidden = self.hidden or self._zero_hidden(batch)
        next_obs = rollout["next_obs"]
        if not is_vec:
            next_obs = next_obs.unsqueeze(0)
        with torch.no_grad():
            _, value, _, _ = self.model(next_obs, hidden)
        return value if is_vec else value.squeeze()

    def _state(self, step=0):
        data = {"model": self.model.state_dict(), "step": step}
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.icm is not None:
            data["icm"] = self.icm.state_dict()
        if hasattr(self, "scaler") and self.scaler is not None and self.scaler.is_enabled():
            data["grad_scaler"] = self.scaler.state_dict()
        return data

    def _load_state(self, state):
        if "model" not in state:
            return 0  # incompatible checkpoint (e.g. old PPO format)
        self.model.load_state_dict(strip_wrapper_prefixes(state["model"]))
        if self.optimizer is not None and "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        if self.icm is not None and "icm" in state:
            self.icm.load_state_dict(strip_wrapper_prefixes(state["icm"]))
        if hasattr(self, "scaler") and self.scaler is not None and "grad_scaler" in state:
            self.scaler.load_state_dict(state["grad_scaler"])
        return state.get("step", 0)

    def load_non_model_state(self, state: dict) -> int:
        if self.optimizer is not None and "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        return state.get("step", 0)

    def reset(self) -> None:
        # Obs shape may change on env swap — drop cached pinned buffers.
        self._pin_memory.reset()


class LstmEnvPPO(EnvAgentMixin, LstmPPOBase):
    """Single-env LSTM-PPO."""

    def collect_rollout(self, env, num_steps, obs=None):
        if obs is None:
            obs, _ = env.reset()
            obs = self._pin_memory.to(obs, self.device)
            self.hidden = None

        autoreset = getattr(env, "autoreset", False)
        hx, cx = self.hidden or self._zero_hidden(1)
        all_obs, actions, log_probs, rewards, dones, values = [], [], [], [], [], []
        saved_hx, saved_cx = [], []

        for _ in range(num_steps):
            saved_hx.append(hx.detach())
            saved_cx.append(cx.detach())
            all_obs.append(obs)

            action, log_prob, value, hx, cx = self._forward(obs.unsqueeze(0), hx, cx)
            next_obs, reward, term, trunc, _ = env.step(action.squeeze().item())
            done = term or trunc

            actions.append(action.squeeze())
            log_probs.append(log_prob.squeeze())
            rewards.append(reward)
            dones.append(done)
            values.append(value.squeeze())

            obs = self._pin_memory.to(next_obs, self.device)
            if not done:
                continue
            hx, cx = self._zero_hidden(1)
            if not autoreset:
                obs = self._pin_memory.to(env.reset()[0], self.device)

        self.hidden = (hx, cx)
        rollout = dict(
            obs=torch.stack(all_obs),
            actions=torch.stack(actions),
            log_probs=torch.stack(log_probs),
            rewards=torch.tensor(rewards, dtype=torch.float32, device=self.device),
            dones=torch.tensor(dones, dtype=torch.float32, device=self.device),
            values=torch.stack(values),
            next_obs=obs,
            next_done=float(done),
            saved_hx=torch.stack(saved_hx),
            saved_cx=torch.stack(saved_cx),
        )
        return rollout, obs

    def get_action(self, obs, *, deterministic=False):
        obs_in = obs.unsqueeze(0) if obs.dim() < 4 else obs
        obs_dev = self._pin_memory.to(obs_in, self.device)
        hx, cx = self.hidden or self._zero_hidden(1)
        action, log_prob, value, hx, cx = self._forward(obs_dev, hx, cx, deterministic)
        self.hidden = (hx, cx)
        return action.squeeze(0).cpu(), log_prob.squeeze(0).cpu(), value.squeeze(0).cpu()

    def act(self, obs, *, deterministic=False):
        return self.get_action(obs, deterministic=deterministic)[0].item()

    def evaluate(self, env, num_episodes=5):
        saved = self.hidden
        self.hidden = None
        scores = EnvAgentMixin.evaluate(self, env, num_episodes)
        self.hidden = saved
        return scores


class LstmVecPPO(VecAgentMixin, LstmPPOBase):
    def collect_rollout(self, env, num_steps, obs=None):
        if obs is None:
            obs = torch.as_tensor(env.reset()[0], dtype=torch.float32)
            self.hidden = None
        elif obs.device.type != "cpu":
            obs = obs.cpu()

        hx, cx = self.hidden or self._zero_hidden(env.num_envs)
        all_obs, actions, log_probs, rewards, dones, values = [], [], [], [], [], []
        saved_hx, saved_cx = [], []

        for _ in range(num_steps):
            saved_hx.append(hx.detach())
            saved_cx.append(cx.detach())
            all_obs.append(obs)

            obs_dev = self._pin_memory.to(obs, self.device)
            action, log_prob, value, hx, cx = self._forward(obs_dev, hx, cx)
            next_obs, rews, term, trunc, _ = env.step(action.cpu().numpy())
            dones_arr = term | trunc

            actions.append(action.cpu())
            log_probs.append(log_prob.cpu())
            rewards.append(torch.as_tensor(rews, dtype=torch.float32))
            dones.append(torch.as_tensor(dones_arr, dtype=torch.float32))
            values.append(value.cpu())

            obs = torch.as_tensor(next_obs, dtype=torch.float32)
            if dones_arr.any():
                hx, cx = self._zero_done_hidden(hx, cx, dones_arr, self.device)

        self.hidden = (hx, cx)
        rollout = dict(
            obs=torch.stack(all_obs),
            actions=torch.stack(actions),
            log_probs=torch.stack(log_probs),
            rewards=torch.stack(rewards),
            dones=torch.stack(dones),
            values=torch.stack(values),
            next_obs=obs,
            next_done=torch.as_tensor(dones_arr, dtype=torch.float32),
            saved_hx=torch.stack(saved_hx),
            saved_cx=torch.stack(saved_cx),
        )
        return rollout, obs

    def get_action(self, obs, *, deterministic=False):
        obs_dev = self._pin_memory.to(obs, self.device)
        hx, cx = self.hidden or self._zero_hidden(obs.shape[0])
        action, log_prob, value, hx, cx = self._forward(obs_dev, hx, cx, deterministic)
        self.hidden = (hx, cx)
        return action.cpu(), log_prob.cpu(), value.cpu()

    def act(self, obs, *, deterministic=False):
        return self.get_action(obs, deterministic=deterministic)[0].numpy()

    def evaluate(self, env, num_episodes=5):
        saved = self.hidden
        self.hidden = None
        scores = VecAgentMixin.evaluate(self, env, num_episodes)
        self.hidden = saved
        return scores


def _create_lstm_ppo(config: LstmPPOConfig, env=None):
    if env is None or is_vector_env(env):
        return LstmVecPPO(config)
    return LstmEnvPPO(config)


class LstmPPO(Agent):
    def __init__(self, config: LstmPPOConfig, env=None):
        self.config = config
        self.lstm_ppo = _create_lstm_ppo(config, env)
        super().__init__(self.lstm_ppo)

    def get_action(self, obs, **kwargs):
        return self.lstm_ppo.get_action(obs, **kwargs)

    def collect_rollout(self, env, num_steps, obs=None):
        return self.lstm_ppo.collect_rollout(env, num_steps, obs)

    def learn(self, rollout):
        return self.lstm_ppo.learn(rollout)

    @classmethod
    def restore(cls, config: LstmPPOConfig, f, env=None):
        agent = cls(config, env)
        agent.lstm_ppo.model.eval()
        agent.lstm_ppo.load(f)
        return agent
