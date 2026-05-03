"""Checkpoint plugin -- saves and loads model checkpoints."""

from __future__ import annotations

import torch
import torch.distributed as dist

from rlcade.checkpoint.checkpoint import Checkpoint
from rlcade.logger import get_logger, get_log0

logger = get_logger(__name__)
log0 = get_log0(__name__)


def _save_rank0(trainer, checkpoint: Checkpoint, state) -> bool:
    """Write the checkpoint on rank 0. Returns False if the write raised."""
    if not trainer.rank0:
        return True
    try:
        checkpoint.save(state)
        return True
    except Exception:
        logger.exception("Checkpoint save failed on rank 0")
        return False


def _all_reduce_ok(trainer, ok: bool) -> bool:
    """All-reduce ``ok`` across ranks (MIN). Returns False if any rank failed.

    Serves as the per-save sync point -- propagates a rank-0 write failure to
    every rank so no peer returns before rank 0 has finished writing, and
    every rank raises consistently when the save fails.
    """
    if not trainer.distributed:
        return ok
    t = torch.tensor([1 if ok else 0], device=trainer.agent.device)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return bool(t.item())


def _save_and_sync(trainer, checkpoint: Checkpoint, state) -> bool:
    """Save on rank 0 and all-reduce the outcome across ranks.

    The all-reduce runs in ``finally`` so every rank enters the collective
    even if the rank-0 save unwinds via BaseException -- peers never hang
    waiting for a rank that has already exited.
    """
    ok = False
    try:
        ok = _save_rank0(trainer, checkpoint, state)
    finally:
        all_ok = _all_reduce_ok(trainer, ok)
    return all_ok


class CheckpointPlugin:
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
        if not _save_and_sync(trainer, self._checkpoint, state):
            raise RuntimeError(f"Checkpoint save failed at step {step}")
        self._last_saved_step = step
        log0.info("Checkpoint saved to %s", self.checkpoint_path)

    def on_step_start(self, trainer, iteration: int) -> None:
        pass

    def on_step_end(self, trainer, iteration: int, summary: dict[str, float] | None) -> None:
        if self.checkpoint_interval and iteration > 0 and iteration % self.checkpoint_interval == 0:
            self._save(trainer)

    def on_eval(self, trainer, iteration: int, scores: list[float]) -> None:
        pass

    def on_done(self, trainer) -> None:
        # Skip if we just saved at this exact step (avoid double-write at final interval)
        if self._last_saved_step == trainer.metrics.total_steps:
            return
        self._save(trainer)
