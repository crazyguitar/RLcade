"""python -m rlcade.training — train an RL agent on NES games."""

import argparse

from rlcade.args import (
    add_env_args,
    add_launcher_args,
    add_model_args,
    add_off_policy_args,
    add_ppo_args,
    add_dqn_args,
    add_rainbow_dqn_args,
    add_sac_args,
    add_smb_args,
    add_training_args,
    add_viztracer_args,
    add_nsys_args,
    add_memory_profiler_args,
)
from rlcade.launcher import launch
from rlcade.entrypoint import train_fn

# fmt: off
def parse_args():
    parser = argparse.ArgumentParser(description="Train an RL agent on NES games")
    add_launcher_args(parser)
    add_env_args(parser)
    add_smb_args(parser)
    add_model_args(parser)
    add_ppo_args(parser)
    add_off_policy_args(parser)
    add_dqn_args(parser)
    add_rainbow_dqn_args(parser)
    add_sac_args(parser)
    add_training_args(parser)
    add_viztracer_args(parser)
    add_nsys_args(parser)
    add_memory_profiler_args(parser)
    parser.add_argument("--log-interval", type=int, default=1, help="DQN log interval (steps)")
    return parser.parse_args()
# fmt: on


def main():
    args = parse_args()
    launch(args, train_fn)


if __name__ == "__main__":
    main()
