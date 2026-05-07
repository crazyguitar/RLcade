"""TensorBoard plugin for RL training metrics."""

from torch.utils.tensorboard import SummaryWriter

from rlcade.plugins import TrainerPlugin

# summary key → (tensorboard tag, group)
_SCALAR_MAP = {
    "score": "reward/mean_score",
    "steps": "reward/total_steps",
    "policy_loss": "loss/policy",
    "value_loss": "loss/value",
    "entropy": "loss/entropy",
    "loss": "loss/total",
    "kl": "policy/kl",
    "clip_fraction": "policy/clip_fraction",
    "sps": "time/sps",
    "rollout": "time/rollout",
    "train": "time/train",
}


class TensorBoardPlugin(TrainerPlugin):
    def __init__(self, log_dir: str = "runs"):
        self.writer = SummaryWriter(log_dir)

    def on_step_end(self, trainer, iteration: int, summary: dict[str, float] | None) -> None:
        if not summary:
            return
        for key, tag in _SCALAR_MAP.items():
            if key in summary:
                self.writer.add_scalar(tag, summary[key], iteration)

    def on_eval(self, trainer, iteration: int, scores: list[float]) -> None:
        if not scores:
            return
        self.writer.add_scalar("eval/mean_score", sum(scores) / len(scores), iteration)
        self.writer.add_scalar("eval/max_score", max(scores), iteration)
        self.writer.add_scalar("eval/min_score", min(scores), iteration)

    def on_done(self, trainer) -> None:
        self.writer.close()
