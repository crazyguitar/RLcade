from rlcade.training.trainer import Trainer
from rlcade.training.on_policy_trainer import OnPolicyTrainer
from rlcade.training.off_policy_trainer import OffPolicyTrainer
from rlcade.training.ppo_trainer import PPOTrainer
from rlcade.training.dqn_trainer import DQNTrainer, RainbowDQNTrainer
from rlcade.training.sac_trainer import SACTrainer

_TRAINERS = {
    "ppo": PPOTrainer,
    "lstm_ppo": PPOTrainer,
    "dqn": DQNTrainer,
    "rainbow_dqn": RainbowDQNTrainer,
    "sac": SACTrainer,
}


def create_trainer(name: str, args, *, plugins=None):
    """Create a trainer by name. Raises KeyError if unknown."""
    return _TRAINERS[name](args, plugins=plugins)
