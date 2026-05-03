"""DQN Trainers — thin subclasses of OffPolicyTrainer."""

from rlcade.training.off_policy_trainer import OffPolicyTrainer


class DQNTrainer(OffPolicyTrainer):
    _loss_key = "loss"


class RainbowDQNTrainer(OffPolicyTrainer):
    """Off-policy trainer with PER beta annealing."""

    _loss_key = "loss"

    def step(self, iteration: int) -> None:
        frac = min(1.0, iteration / self.max_iterations)
        self.agent.beta = self.agent.beta_start + frac * (self.agent.beta_end - self.agent.beta_start)
        super().step(iteration)
