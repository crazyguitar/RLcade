"""Base agent interface shared by PPO, DQN, and future algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.distributed.checkpoint._state_dict_stager import StateDictStager

from rlcade.utils import to_tensor


def strip_wrapper_prefixes(sd: dict) -> dict:
    """Drop `_orig_mod.` (torch.compile), `module.` (DDP), and `_module.` (CUDAGraphWrapper) prefixes."""
    prefixes = ("_orig_mod.", "module.", "_module.")
    out = {}
    for k, v in sd.items():
        while isinstance(k, str):
            for p in prefixes:
                if k.startswith(p):
                    k = k[len(p) :]
                    break
            else:
                break
        out[k] = v
    return out


def unwrap_module(module: nn.Module) -> nn.Module:
    """Peel CUDAGraphWrapper, torch.compile (OptimizedModule), and DDP layers
    to reach the underlying nn.Module. Parameter tensors are shared, so an
    optimizer built on the wrapped module sees the same params on the bare
    module.
    """
    from torch.nn.parallel import DistributedDataParallel

    seen: set[int] = set()
    while id(module) not in seen:
        seen.add(id(module))
        # CUDAGraphWrapper stores its inner module on `_module`. Import lazily
        # so this helper works in environments where rlcade.graph is unused.
        from rlcade.graph import CUDAGraphWrapper

        if isinstance(module, CUDAGraphWrapper):
            module = module._module
            continue
        orig = getattr(module, "_orig_mod", None)
        if isinstance(orig, nn.Module) and orig is not module:
            module = orig
            continue
        if isinstance(module, DistributedDataParallel):
            module = module.module
            continue
        break
    return module


class AgentBase(ABC):
    """Base class for single-env and vec-env agent implementations."""

    device: torch.device
    stager: StateDictStager | None = None
    pin_memory: bool = True

    @abstractmethod
    def get_action(self, obs: torch.Tensor, *, deterministic: bool = False):
        """Return action(s) for the given observation(s)."""

    def act(self, obs: torch.Tensor, *, deterministic: bool = False):
        """Return action(s) suitable for env.step(). Override if get_action returns extra values."""
        return self.get_action(obs, deterministic=deterministic)

    def create_optimizers(self):
        """Create optimizers and schedulers. Called by the trainer after wrapping."""
        pass

    def state(self, step: int = 0, *, staging: bool = False) -> dict:
        """Return full checkpoint state as a dict.

        When ``staging=True``, the state is deep-copied onto CPU via PyTorch's
        StateDictStager. The stager caches CPU storage across calls so periodic
        saves reuse the same buffers. Flip pin_memory on the stager to enable
        pinned-memory D2H transfers.
        """
        s = self._state(step)
        if not staging:
            return s
        if self.stager is None:
            self.stager = StateDictStager(pin_memory=self.pin_memory)
        return self.stager.stage(s)

    @abstractmethod
    def _state(self, step: int = 0) -> dict:
        """Return the raw checkpoint state dict. Subclasses implement this."""

    def load(self, f) -> int:
        """Deserialize checkpoint from a readable stream. Returns the step."""
        state = torch.load(f, map_location=self.device, weights_only=True)
        return self._load_state(state)

    @abstractmethod
    def _load_state(self, state: dict) -> int:
        """Restore from checkpoint state dict. Returns the step."""

    def load_non_model_state(self, state: dict) -> int:
        """Load only non-model state (optimizers, scalars). Defaults to full _load_state()."""
        return self._load_state(state)

    def evaluate(self, env, num_episodes: int = 5) -> list[float]:
        raise NotImplementedError

    def models(self) -> list[tuple[str, nn.Module]]:
        """Return list of (attr_name, module) for trainable models. Override per agent."""
        return []

    def target_models(self) -> list[tuple[str, nn.Module]]:
        """Return list of (attr_name, module) for target networks (no gradients). Override per agent."""
        return []

    def compile(self, eager: bool = False, strategy: str | None = None) -> None:
        """Apply torch.compile and CUDA graph wrapping to all models and target models.

        Skipped under FSDP2: compatibility issue
        """
        if eager or self.device.type != "cuda" or strategy == "fsdp2":
            return
        from rlcade.graph import CUDAGraphWrapper

        for attr, module in self.models() + self.target_models():
            compiled = torch.compile(module)
            setattr(self, attr, CUDAGraphWrapper(compiled, name=attr))

    def optimizers(self) -> list[tuple[str, torch.optim.Optimizer, nn.Module]]:
        """Return list of (state_key, optimizer, model) for FSDP2 optimizer state loading."""
        return []

    def reset(self) -> None:
        """Hook invoked by the trainer when the env is swapped (e.g. curriculum stage change)."""
        pass


class EnvAgentMixin:
    """Mixin providing single-env evaluate."""

    def evaluate(self, env, num_episodes: int = 5) -> list[float]:
        scores = []
        for _ in range(num_episodes):
            obs, _ = env.reset()
            total, done = 0.0, False
            while not done:
                action = self.act(to_tensor(obs, self.device), deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total += reward
                done = terminated or truncated
            scores.append(total)
        return scores


class VecAgentMixin:
    """Mixin providing vec-env evaluate."""

    def evaluate(self, env, num_episodes: int = 5) -> list[float]:
        obs, _ = env.reset()
        scores = []
        totals = [0.0] * env.num_envs
        while len(scores) < num_episodes:
            actions = self.act(to_tensor(obs, self.device), deterministic=True)
            obs, rewards, terminated, truncated, _ = env.step(actions)
            dones = terminated | truncated
            for i in range(env.num_envs):
                totals[i] += float(rewards[i])
                if dones[i]:
                    scores.append(totals[i])
                    totals[i] = 0.0
        return scores[:num_episodes]


def is_vector_env(env) -> bool:
    """Check if env is a vectorized environment."""
    return hasattr(env, "num_envs") and env.num_envs > 1


class Agent:
    """Facade base class that delegates to an inner Env*/Vec* implementation."""

    def __init__(self, impl: AgentBase):
        self._impl = impl

    def get_action(self, obs, **kwargs):
        return self._impl.get_action(obs, **kwargs)

    def act(self, obs, **kwargs):
        return self._impl.act(obs, **kwargs)

    def evaluate(self, env, num_episodes: int = 5) -> list[float]:
        return self._impl.evaluate(env, num_episodes)

    def state(self, step: int = 0, *, staging: bool = False) -> dict:
        return self._impl.state(step, staging=staging)

    def load(self, f) -> int:
        """Load checkpoint from a readable stream. Returns the step."""
        return self._impl.load(f)

    @property
    def device(self):
        return self._impl.device

    def create_optimizers(self):
        self._impl.create_optimizers()

    def compile(self, eager: bool = False, strategy: str | None = None):
        self._impl.compile(eager=eager, strategy=strategy)

    def models(self) -> list[tuple[str, nn.Module]]:
        return self._impl.models()

    def target_models(self) -> list[tuple[str, nn.Module]]:
        return self._impl.target_models()

    def optimizers(self):
        return self._impl.optimizers()

    def load_non_model_state(self, state: dict) -> int:
        return self._impl.load_non_model_state(state)

    def reset(self) -> None:
        self._impl.reset()


# Agent wrappers for distributed training


class AgentWrapper:
    """Transparent wrapper — delegates everything to the underlying agent."""

    def __init__(self, agent: Agent):
        self._agent = agent

    def __getattr__(self, name):
        return getattr(self._agent, name)

    def state(self, step: int = 0, *, staging: bool = False) -> dict:
        return self._agent.state(step, staging=staging)

    def load(self, f) -> int:
        return self._agent.load(f)


class DDPAgentWrapper(AgentWrapper):
    """DDP wrapper — wraps models with DistributedDataParallel, rank 0 saves."""

    def __init__(self, agent: Agent):
        super().__init__(agent)
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP

        device = agent.device
        device_ids = [device.index] if device.type == "cuda" else None
        for attr, module in agent.models():
            setattr(agent._impl, attr, DDP(module, device_ids=device_ids))

        # Broadcast target network params from rank 0 so all ranks start in sync
        for _, net in agent.target_models():
            for p in net.parameters():
                dist.broadcast(p.data, src=0)

    def _unwrapped(self):
        """Context manager: temporarily swap wrapped models for their bare nn.Module.

        Peels CUDAGraphWrapper, torch.compile, and DDP so ``state_dict`` and
        ``load_state_dict`` operate on the underlying parameters directly.
        Without full unwrap, ``OptimizedModule(DDP(...))`` keeps the
        ``_orig_mod.module.`` key prefix on its state dict and ``_load_state``
        would mismatch after ``strip_wrapper_prefixes``. Target networks are
        also unwrapped: they aren't DDP-wrapped but ``compile()`` still puts
        them under ``CUDAGraphWrapper(torch.compile(...))``.
        """
        from contextlib import contextmanager

        @contextmanager
        def ctx():
            wrapped = {}
            for attr, module in self._agent.models() + self._agent.target_models():
                wrapped[attr] = module
                setattr(self._agent._impl, attr, unwrap_module(module))
            try:
                yield
            finally:
                for attr, module in wrapped.items():
                    setattr(self._agent._impl, attr, module)

        return ctx()

    def state(self, step: int = 0, *, staging: bool = False) -> dict:
        with self._unwrapped():
            return self._agent.state(step, staging=staging)

    def load(self, f) -> int:
        import torch.distributed as dist

        with self._unwrapped():
            step = self._agent.load(f)
        dist.barrier()
        return step


class FSDP2AgentWrapper(AgentWrapper):
    """FSDP2 wrapper — wraps models with fully_shard, gathers full tensors for checkpoints."""

    def __init__(self, agent: Agent):
        super().__init__(agent)
        from torch.distributed.fsdp import fully_shard

        for _, module in agent.models() + agent.target_models():
            for child in module.children():
                fully_shard(child)
            fully_shard(module)

    def state(self, step: int = 0, *, staging: bool = False) -> dict:
        """Gather full state from sharded models using FSDP2 APIs.

        Output is always CPU-offloaded via ``StateDictOptions(cpu_offload=True)``,
        so ``staging`` is a no-op for FSDP2 -- the staging deep-copy would be
        redundant on already-CPU tensors.
        """
        from torch.distributed.checkpoint.state_dict import (
            get_model_state_dict,
            get_optimizer_state_dict,
            StateDictOptions,
        )

        agent = self._agent
        opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
        state: dict = {"step": step}

        for attr, module in agent.models() + agent.target_models():
            state[attr] = get_model_state_dict(module, options=opts)

        for key, optimizer, model in agent.optimizers():
            state[key] = get_optimizer_state_dict(model, optimizer, options=opts)

        return state

    def load(self, f) -> int:
        """Load full checkpoint into sharded models -- deserializes to CPU."""
        import torch.distributed as dist
        from torch.distributed.checkpoint.state_dict import (
            set_model_state_dict,
            set_optimizer_state_dict,
            StateDictOptions,
        )

        state = torch.load(f, map_location="cpu", weights_only=True)
        agent = self._agent
        opts = StateDictOptions(full_state_dict=True)

        for attr, module in agent.models() + agent.target_models():
            if attr in state:
                set_model_state_dict(module, state[attr], options=opts)

        for key, optimizer, model in agent.optimizers():
            if key in state:
                osd = state[key]
                if "state" in osd and osd["state"]:
                    first_key = next(iter(osd["state"]))
                    if isinstance(first_key, int):
                        osd = _raw_optim_to_fqn(osd, model)
                set_optimizer_state_dict(model, optimizer, osd, options=opts)

        step = state.get("step", 0)
        dist.barrier()
        return step


def _raw_optim_to_fqn(osd: dict, model: nn.Module) -> dict:
    """Convert integer-keyed optimizer state dict to FQN-keyed format.

    Raw ``optimizer.state_dict()`` uses integer param indices as keys.
    ``set_optimizer_state_dict`` expects fully-qualified parameter names.
    """
    fqns = [n for n, _ in model.named_parameters()]
    new_state = {}
    for idx, vals in osd["state"].items():
        if idx < len(fqns):
            new_state[fqns[idx]] = vals
    return {"state": new_state, "param_groups": osd.get("param_groups", [])}


_WRAPPERS: dict[str, type[AgentWrapper]] = {
    "ddp": DDPAgentWrapper,
    "fsdp2": FSDP2AgentWrapper,
}


def wrap_agent(agent: Agent, name: str | None = None, distributed: bool = False) -> Agent | AgentWrapper:
    """Wrap an agent for distributed training. Returns the agent unwrapped if not distributed."""
    if not distributed or not name:
        return agent
    cls = _WRAPPERS.get(name)
    if cls is None:
        return agent
    return cls(agent)
