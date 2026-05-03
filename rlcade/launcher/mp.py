"""Multiprocessing launcher — ``torch.multiprocessing.spawn`` for single-node multi-GPU."""

import os

import torch.multiprocessing as torchmp

from rlcade.logger import get_logger

logger = get_logger(__name__)


def _worker(local_rank, args, train_fn):
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(args.nproc_per_node)
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)
    train_fn(args)


def launch(args, train_fn):
    nproc = args.nproc_per_node
    logger.info("mp.spawn: nproc_per_node=%d", nproc)
    if nproc == 1:
        _worker(0, args, train_fn)
    else:
        torchmp.spawn(_worker, nprocs=nproc, args=(args, train_fn), join=True)
