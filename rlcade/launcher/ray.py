"""Ray launcher — pure Ray distributed training.

If ``--ray-address`` is provided, connects to an existing Ray cluster.
Otherwise, starts a local single-node cluster via ``ray.init()``.

Each worker is a Ray actor with 1 GPU. Ray sets CUDA_VISIBLE_DEVICES
per actor, so each worker sees device 0. Workers call train_fn directly
with RANK/WORLD_SIZE env vars set for torch.distributed.
"""

import os
import socket

from rlcade.logger import get_logger

logger = get_logger(__name__)


def _find_free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _bin_pack_gpus(node_gpus, num_gpus):
    """Greedily bin-pack ``num_gpus`` onto nodes, filling each before spilling."""
    nodes = sorted(node_gpus, key=lambda x: -x[1])  # largest first
    if not nodes:
        raise RuntimeError("Ray cluster has no GPUs available")

    total_available = sum(g for _, g in nodes)
    if num_gpus is None:
        num_gpus = total_available
    if num_gpus > total_available:
        raise RuntimeError(f"Requested {num_gpus} GPUs but cluster only has {total_available}")

    process_on_nodes = []
    remaining = num_gpus
    for ip, gpus in nodes:
        if remaining <= 0:
            break
        use = min(remaining, gpus)
        process_on_nodes.append(use)
        remaining -= use
        logger.info("Node %s: using %d/%d GPUs", ip, use, gpus)

    return process_on_nodes


def _detect_topology(num_gpus):
    """Query the Ray cluster and return ``process_on_nodes``."""
    import ray

    node_gpus = [
        (n["NodeManagerAddress"], int(n["Resources"].get("GPU", 0)))
        for n in ray.nodes()
        if n["Alive"] and n["Resources"].get("GPU", 0) > 0
    ]
    return _bin_pack_gpus(node_gpus, num_gpus)


def _pg_strategy(pg, bundle_index):
    """Build a PlacementGroupSchedulingStrategy for the given bundle."""
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    return PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=bundle_index)


def _resolve_master_addr(pg):
    """Resolve the IP address of the node hosting the first bundle in ``pg``."""
    import ray

    task = ray.remote(ray.util.get_node_ip_address)
    task = task.options(scheduling_strategy=_pg_strategy(pg, 0), num_cpus=0)
    return ray.get(task.remote())


def _create_placement_groups(process_on_nodes):
    """Create one STRICT_PACK placement group per node and wait until ready."""
    import ray
    from ray.util.placement_group import placement_group

    pgs = [placement_group([{"GPU": 1}] * n, strategy="STRICT_PACK") for n in process_on_nodes]
    ray.get([pg.ready() for pg in pgs])
    return pgs


def _run_worker(rank, world_size, master_addr, master_port, args, train_fn):
    """Runs inside a Ray actor — sets env vars and calls train_fn."""
    import ray

    # Use the real GPU ID so NCCL can see all GPUs for P2P/NVLink
    gpu_ids = ray.get_gpu_ids()
    local_rank = int(gpu_ids[0]) if gpu_ids else 0
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    train_fn(args)


def _make_worker():
    """Create a Ray remote actor class (1 GPU per worker)."""
    import ray

    @ray.remote(num_gpus=1, num_cpus=0)
    class Worker:
        def run(self, rank, world_size, master_addr, master_port, args, train_fn):
            _run_worker(rank, world_size, master_addr, master_port, args, train_fn)

    return Worker


def launch(args, train_fn):
    try:
        import ray
    except ImportError:
        raise ImportError("ray is required for --launcher ray. Install with: pip install 'ray[default]>=2.9'") from None

    ray.init(address=args.ray_address)
    process_on_nodes = _detect_topology(args.num_gpus)
    world_size = sum(process_on_nodes)
    logger.info("Ray topology: %s (world_size=%d)", process_on_nodes, world_size)

    pgs = _create_placement_groups(process_on_nodes)
    master_addr = _resolve_master_addr(pgs[0])
    master_port = _find_free_port()

    Worker = _make_worker()

    # Flatten assignments: (pg, bundle_index, global_rank)
    workers = []
    rank = 0
    for pg, n_gpus in zip(pgs, process_on_nodes):
        for bundle_idx in range(n_gpus):
            w = Worker.options(scheduling_strategy=_pg_strategy(pg, bundle_idx)).remote()
            workers.append((w, rank))
            rank += 1

    logger.info("Launching %d workers (master=%s:%d)", len(workers), master_addr, master_port)
    futures = []
    for w, rank in workers:
        f = w.run.remote(rank, world_size, master_addr, master_port, args, train_fn)
        futures.append(f)
    ray.get(futures)
