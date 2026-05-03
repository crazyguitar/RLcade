"""SAC-Discrete Trainer — thin subclass of OffPolicyTrainer."""

from rlcade.training.off_policy_trainer import OffPolicyTrainer


class SACTrainer(OffPolicyTrainer):
    _loss_key = "critic_loss"
