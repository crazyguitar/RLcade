"""Launcher dispatch — select backend and spawn training processes."""

import importlib
import os

from rlcade.logger import get_logger

logger = get_logger(__name__)

_BACKENDS = {
    "none": "rlcade.launcher.none",
    "elastic": "rlcade.launcher.elastic",
    "mp": "rlcade.launcher.mp",
    "ray": "rlcade.launcher.ray",
}


def launch(args, train_fn):
    """Dispatch to the appropriate launcher backend."""
    if "OMP_NUM_THREADS" not in os.environ and getattr(args, "nproc_per_node", 1) > 1:
        os.environ["OMP_NUM_THREADS"] = "1"
        logger.info("Setting OMP_NUM_THREADS=1 (nproc_per_node=%d)", args.nproc_per_node)

    name = args.launcher
    logger.info("Launcher: %s", name)
    mod = importlib.import_module(_BACKENDS[name])
    mod.launch(args, train_fn)
