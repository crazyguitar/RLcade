"""python -m rlcade.agent — play with a trained checkpoint."""

import argparse

from rlcade.agent import load_agent
from rlcade.args import add_env_args, add_model_args, add_smb_args
from rlcade.utils import resolve_device
from rlcade.envs import create_env
from rlcade.nes import Nes

# fmt: off
def parse_args():
    parser = argparse.ArgumentParser(description="Play NES games with a trained agent")
    add_env_args(parser)
    add_smb_args(parser)
    add_model_args(parser)
    return parser.parse_args()
# fmt: on


def main():
    args = parse_args()
    args.device = resolve_device(args.device)
    env = create_env(args)
    args.obs_shape = env.observation_space.shape
    args.n_actions = env.action_space.n
    agent = load_agent(args.agent, args, env)
    Nes(env, agent, device=args.device).play()


if __name__ == "__main__":
    main()
