"""Entry point: python -m bench --rom <path> [--bench env|ppo|dqn|rainbow_dqn|all]"""

import copy
import os

from bench.utils import parse_args, AGENTS
from bench.env import bench_env_step, bench_vec_env_step
from bench.trainer import bench_trainer


def main():
    args = parse_args()

    if args.bench == "env" or args.bench == "all":
        from bench.utils import agent_defaults

        env_args = copy.copy(args)
        agent_defaults("ppo", env_args)
        bench_env_step(env_args, env_args.num_steps)
        bench_vec_env_step(env_args, env_args.num_steps)

    agents = AGENTS if args.bench == "all" else [args.bench]
    for agent_name in agents:
        if agent_name == "env":
            continue
        agent_plugins = []
        if args.viztracer:
            from rlcade.plugins.viztracer import VizTracerPlugin

            os.makedirs("profile", exist_ok=True)
            output_file = f"profile/{args.viztracer}_{agent_name}.json"
            agent_plugins.append(
                VizTracerPlugin(
                    start_step=1,
                    end_step=args.iterations,
                    output_file=output_file,
                )
            )
        bench_trainer(args, agent_name, plugins=agent_plugins)


if __name__ == "__main__":
    main()
