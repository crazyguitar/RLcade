"""Training metrics tracker."""


class Metrics:
    """Tracks and computes training metrics."""

    def __init__(self):
        self.episode_scores: list[float] = []
        self.losses: list[float] = []
        self.eval_scores: list[float] = []
        self.total_steps: int = 0
        self.sps: float = 0.0
        self.extras: dict[str, float] = {}
        self.score: float = 0.0

    def record_episodes(self, scores: list[float]):
        """Record completed episode scores."""
        self.episode_scores.extend(scores)

    def record_loss(self, loss: float):
        """Record a training loss value."""
        self.losses.append(loss)

    def record_eval(self, scores: list[float]):
        """Record evaluation episode scores."""
        self.eval_scores = scores

    def advance(self, steps: int):
        """Increment total step counter."""
        self.total_steps += steps

    def mean_score(self, n: int = 10) -> float:
        """Mean of last n episode scores."""
        if not self.episode_scores:
            return 0.0
        recent = self.episode_scores[-n:]
        return sum(recent) / len(recent)

    def mean_loss(self, n: int = 10) -> float:
        """Mean of last n losses."""
        if not self.losses:
            return 0.0
        recent = self.losses[-n:]
        return sum(recent) / len(recent)

    def eval_summary(self) -> dict[str, float]:
        """Return eval metrics as a dict for display."""
        if not self.eval_scores:
            return {}
        return {
            "eval_mean": sum(self.eval_scores) / len(self.eval_scores),
            "eval_max": max(self.eval_scores),
        }

    def summary(self) -> dict[str, float]:
        """Return current metrics as a dict for display."""
        s = {
            "loss": self.mean_loss(),
            "score": self.score,
            "sps": self.sps,
            "steps": self.total_steps,
        }
        s.update(self.eval_summary())
        s.update(self.extras)
        return s
