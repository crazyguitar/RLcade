"""Logging configuration for RLcade."""

import logging
import os

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", level=logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """Standard logger — logs on every rank."""
    return logging.getLogger(name)


def get_log0(name: str) -> logging.Logger:
    """Rank-0 logger — only logs when RANK is unset or 0."""
    logger = logging.getLogger(f"{name}.rank0")
    rank = int(os.environ.get("RANK", 0))
    if rank != 0:
        logger.setLevel(logging.CRITICAL + 1)
    return logger
