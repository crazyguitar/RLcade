"""None launcher — direct single-process execution.

Also works when an external scheduler (e.g. Slurm) pre-sets RANK,
WORLD_SIZE, MASTER_ADDR, MASTER_PORT environment variables.
"""


def launch(args, train_fn):
    train_fn(args)
