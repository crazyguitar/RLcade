"""Checkpoint plugin -- saves and loads model checkpoints."""

from __future__ import annotations

from rlcade.checkpoint.checkpoint import Checkpoint
from rlcade.logger import get_logger, get_log0
from rlcade.plugins import TrainerPlugin
from rlcade.plugins._distributed import save_and_sync

logger = get_logger(__name__)
log0 = get_log0(__name__)


class CheckpointPlugin(TrainerPlugin):
    """Trainer plugin that handles checkpoint save/load.

    Args:
        checkpoint_path: File path or S3 URL for the checkpoint (empty to disable).
        checkpoint_interval: Save every N iterations (0 to disable periodic saves).
        num_steps: Rollout length for on-policy resume (iteration = step // num_steps).
            Pass None for off-policy trainers.
    """

    def __init__(
        self,
        checkpoint_path: str = "",
        checkpoint_interval: int = 100,
        num_steps: int | None = None,
    ):
        self.checkpoint_path = checkpoint_path
        self.checkpoint_interval = checkpoint_interval
        self.num_steps = num_steps
        self._checkpoint = Checkpoint(checkpoint_path) if checkpoint_path else None
        self._last_saved_step: int = -1

    def on_setup(self, trainer) -> None:
        """Load checkpoint if it exists. Sets trainer._start_iteration for resume."""
        if self._checkpoint is None or not self._checkpoint.exists():
            return
        with self._checkpoint.reader() as f:
            step = trainer.agent.load(f)
        trainer.metrics.total_steps = step
        iteration = step // self.num_steps if self.num_steps else 0
        trainer._start_iteration = iteration
        # Prime dedup so on_done skips saving at the resumed step.
        self._last_saved_step = step
        logger.info(
            "Resumed from checkpoint %s (step %d, iteration %d)",
            self.checkpoint_path,
            step,
            iteration,
        )

    def _save(self, trainer) -> None:
        if self._checkpoint is None:
            return
        step = trainer.metrics.total_steps
        state = trainer.agent.state(step)
        ok = save_and_sync(
            trainer,
            lambda: self._checkpoint.save(state),
            what="Checkpoint save",
        )
        if not ok:
            raise RuntimeError(f"Checkpoint save failed at step {step}")
        self._last_saved_step = step
        log0.info("Checkpoint saved to %s", self.checkpoint_path)

    def on_step_end(self, trainer, iteration: int, summary: dict[str, float] | None) -> None:
        if self.checkpoint_interval and iteration > 0 and iteration % self.checkpoint_interval == 0:
            self._save(trainer)

    def on_done(self, trainer) -> None:
        # Skip if we just saved at this exact step (avoid double-write at final interval)
        if self._last_saved_step == trainer.metrics.total_steps:
            return
        self._save(trainer)
