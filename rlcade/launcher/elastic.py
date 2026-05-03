"""Elastic launcher — programmatic equivalent of ``torchrun``."""

import uuid

from torch.distributed.launcher.api import LaunchConfig, elastic_launch

from rlcade.logger import get_logger

logger = get_logger(__name__)


def launch(args, train_fn):
    # Single-node: use localhost:0 (auto-pick port) like torchrun --standalone
    if args.nnodes == 1:
        rdzv_endpoint = "localhost:0"
    else:
        rdzv_endpoint = f"{args.master_addr}:{args.master_port}"

    config = LaunchConfig(
        min_nodes=args.nnodes,
        max_nodes=args.nnodes,
        nproc_per_node=args.nproc_per_node,
        run_id=str(uuid.uuid4()),
        rdzv_backend="c10d",
        rdzv_endpoint=rdzv_endpoint,
        max_restarts=0,
        monitor_interval=5,
        local_addr="localhost",
    )
    logger.info("elastic_launch: nnodes=%d nproc_per_node=%d rdzv=%s", args.nnodes, args.nproc_per_node, rdzv_endpoint)
    elastic_launch(config, train_fn)(args)
