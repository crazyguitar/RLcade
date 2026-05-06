from rlcade.agent.ppo import PPO, PPOConfig, LstmPPO, LstmPPOConfig
from rlcade.agent.dqn import DQN, DQNConfig, RainbowDQN, RainbowDQNConfig
from rlcade.agent.sac import SAC, SACConfig
from rlcade.agent.base import AgentWrapper, wrap_agent
from rlcade.checkpoint.checkpoint import Checkpoint
from rlcade.checkpoint.safetensors import load_safetensors

_AGENTS = {
    "ppo": (PPO, PPOConfig),
    "lstm_ppo": (LstmPPO, LstmPPOConfig),
    "dqn": (DQN, DQNConfig),
    "rainbow_dqn": (RainbowDQN, RainbowDQNConfig),
    "sac": (SAC, SACConfig),
}


def create_agent(name: str, args, env=None):
    """Create an agent by name from CLI args."""
    cls, config_cls = _AGENTS[name]
    return cls(config_cls.from_args(args), env)


def load_agent(name: str, args, env=None):
    """Load a trained agent for inference. Dispatches on extension."""
    cls, config_cls = _AGENTS[name]
    config = config_cls.from_args(args)
    url = args.checkpoint
    if url.endswith(".safetensors"):
        agent = cls(config, env)
        _load_safetensors_into(agent, url, args.device)
        return agent
    with Checkpoint(url).reader() as f:
        return cls.restore(config, f, env)


def _load_safetensors_into(agent, url: str, device) -> None:
    """Load model weights from a safetensors file into an already-constructed agent."""
    import torch

    if isinstance(device, str):
        device = torch.device(device)
    state, _step = load_safetensors(url, device=device)
    impl = agent._impl if hasattr(agent, "_impl") else agent
    for name, sd in state.items():
        module = _resolve_attr(impl, name)
        module.load_state_dict(sd)


def _resolve_attr(impl, name: str):
    """Resolve an agent's model attr by name. Raises AttributeError if missing.

    Some agents nest the actor/critic on a sub-attr (e.g. PPO.ppo.actor). We
    try the impl directly first, then a single level of nesting.
    """
    if hasattr(impl, name):
        return getattr(impl, name)
    for child_name in dir(impl):
        if child_name.startswith("_"):
            continue
        child = getattr(impl, child_name, None)
        if child is None:
            continue
        if hasattr(child, name) and hasattr(getattr(child, name), "load_state_dict"):
            return getattr(child, name)
    raise AttributeError(f"agent has no model attribute named {name!r}")
