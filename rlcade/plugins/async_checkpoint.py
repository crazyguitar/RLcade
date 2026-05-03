"""Async checkpoint plugin -- offloads disk writes to a background thread.

All ranks call ``agent.state(step, staging=True)`` so FSDP2's gather collective
runs identically everywhere; only rank 0 holds the executor and submits the
write. A failed rank-0 write is detected when the next save joins the prior
future; the outcome is all-reduced across ranks so every peer raises
consistently rather than silently lying in the log.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor

import torch
import torch.distributed as dist

from rlcade.checkpoint.checkpoint import Checkpoint
from rlcade.logger import get_logger, get_log0

logger = get_logger(__name__)
log0 = get_log0(__name__)


def _join_rank0(trainer, future: Future | None) -> bool:
    """Wait for the pending write on rank 0. Returns False if it raised."""
    if not trainer.rank0 or future is None:
        return True
    try:
        future.result()
        return True
    except Exception:
        logger.exception("Async checkpoint save failed on rank 0")
        return False


def _submit_rank0(trainer, executor: ThreadPoolExecutor | None, checkpoint: Checkpoint, state) -> Future | None:
    """Submit a background save on rank 0. Returns the future (or None)."""
    if not trainer.rank0:
        return None
    assert executor is not None
    return executor.submit(checkpoint.save, state)


def _shutdown_rank0(trainer, executor: ThreadPoolExecutor | None) -> None:
    if trainer.rank0 and executor is not None:
        executor.shutdown(wait=True)


def _all_reduce_ok(trainer, ok: bool) -> bool:
    """All-reduce ``ok`` across ranks (MIN). Returns False if any rank failed."""
    if not trainer.distributed:
        return ok
    t = torch.tensor([1 if ok else 0], device=trainer.agent.device)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return bool(t.item())


def _join_and_sync(trainer, future: Future | None) -> bool:
    """Wait for the pending write on rank 0, then all-reduce the outcome.

    The all-reduce runs in ``finally`` so every rank enters the collective
    even if the rank-0 join unwinds via BaseException -- peers never hang
    waiting at the next collective for a rank that has already exited.
    """
    ok = False
    try:
        ok = _join_rank0(trainer, future)
    finally:
        all_ok = _all_reduce_ok(trainer, ok)
    return all_ok


class AsyncCheckpointPlugin:
    """Trainer plugin that saves checkpoints on a background thread."""

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
        self._executor: ThreadPoolExecutor | None = None
        self._future: Future | None = None

    def on_setup(self, trainer) -> None:
        """Load checkpoint if it exists. Sets trainer._start_iteration for resume."""
        if self._checkpoint is not None and self._checkpoint.exists():
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
        # Create executor only after a successful load so a corrupt checkpoint
        # does not leak a worker thread. Idempotent across repeat on_setup calls.
        if trainer.rank0 and self._checkpoint is not None and self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ckpt")

    def _save(self, trainer) -> None:
        if self._checkpoint is None:
            return

        # Drain the prior write BEFORE staging new state. StateDictStager caches
        # CPU buffers keyed by source storage and copy_()s into the same buffer
        # on every call -- staging while the worker is still reading the buffer
        # would silently corrupt the in-flight checkpoint.
        prior_future = self._future
        self._future = None
        if not _join_and_sync(trainer, prior_future):
            raise RuntimeError("Prior async checkpoint save failed")

        step = trainer.metrics.total_steps
        # FSDP2 gather collective -- all ranks participate; safe now that the
        # prior worker has finished reading the stager buffer.
        state = trainer.agent.state(step, staging=True)
        self._future = _submit_rank0(trainer, self._executor, self._checkpoint, state)
        self._last_saved_step = step
        log0.info("Checkpoint save submitted (step %d)", step)

    def on_step_start(self, trainer, iteration: int) -> None:
        pass

    def on_step_end(self, trainer, iteration: int, summary: dict[str, float] | None) -> None:
        if self.checkpoint_interval and iteration > 0 and iteration % self.checkpoint_interval == 0:
            self._save(trainer)

    def on_eval(self, trainer, iteration: int, scores: list[float]) -> None:
        pass

    def on_done(self, trainer) -> None:
        if self._checkpoint is None:
            return
        try:
            if self._last_saved_step != trainer.metrics.total_steps:
                self._save(trainer)
            if not _join_and_sync(trainer, self._future):
                raise RuntimeError("Final async checkpoint save failed")
        finally:
            _shutdown_rank0(trainer, self._executor)
            self._executor = None
            self._future = None
