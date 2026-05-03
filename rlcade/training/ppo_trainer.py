"""PPO Trainer — thin subclass of OnPolicyTrainer."""

from rlcade.training.on_policy_trainer import OnPolicyTrainer


class PPOTrainer(OnPolicyTrainer):
    _loss_key = "loss"
