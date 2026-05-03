from rlcade.agent.ppo import PPO, PPOConfig, LstmPPO, LstmPPOConfig
from rlcade.agent.dqn import DQN, DQNConfig, RainbowDQN, RainbowDQNConfig
from rlcade.agent.sac import SAC, SACConfig
from rlcade.agent.base import AgentWrapper, wrap_agent
from rlcade.checkpoint.checkpoint import Checkpoint

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
    """Load a trained agent from checkpoint."""
    cls, config_cls = _AGENTS[name]
    with Checkpoint(args.checkpoint).reader() as f:
        return cls.restore(config_cls.from_args(args), f, env)
