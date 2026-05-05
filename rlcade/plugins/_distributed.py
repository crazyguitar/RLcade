"""Shared rank-0 save + all-reduce sync helpers for plugins that write artifacts."""

from __future__ import annotations

from typing import Callable

import torch
import torch.distributed as dist

from rlcade.logger import get_logger

logger = get_logger(__name__)


def save_rank0(trainer, write: Callable[[], None], *, what: str) -> bool:
    """Run *write* on rank 0 only. Returns False if it raised. *what* is used for logging."""
    if not trainer.rank0:
        return True
    try:
        write()
        return True
    except Exception:
        logger.exception("%s failed on rank 0", what)
        return False


def all_reduce_ok(trainer, ok: bool) -> bool:
    """All-reduce *ok* across ranks (MIN). Returns False if any rank failed.

    Per-save sync point: propagates a rank-0 write failure to every rank so no
    peer returns before rank 0 has finished writing, and every rank raises
    consistently when the save fails.
    """
    if not trainer.distributed:
        return ok
    t = torch.tensor([1 if ok else 0], device=trainer.agent.device)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return bool(t.item())


def save_and_sync(trainer, write: Callable[[], None], *, what: str) -> bool:
    """Run *write* on rank 0, then all-reduce the outcome.

    The all-reduce runs in ``finally`` so every rank enters the collective even
    if the rank-0 save unwinds via BaseException -- peers never hang waiting
    for a rank that has already exited.
    """
    ok = False
    try:
        ok = save_rank0(trainer, write, what=what)
    finally:
        all_ok = all_reduce_ok(trainer, ok)
    return all_ok
