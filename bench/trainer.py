"""Trainer benchmarks — run agent+trainer for N iterations with plugin support."""

import copy

from rlcade.agent import create_agent
from rlcade.envs import create_vector_env, get_env_info
from rlcade.training import create_trainer
from bench.utils import agent_defaults, summary, timer


def bench_trainer(args, agent_name: str, plugins=None):
    """Benchmark a trainer end-to-end for args.iterations steps."""
    bench_args = copy.copy(args)
    agent_defaults(agent_name, bench_args)

    env = create_vector_env(bench_args)
    num_envs = env.num_envs if hasattr(env, "num_envs") else 1
    bench_args.obs_shape, bench_args.n_actions = get_env_info(env)

    agent = create_agent(agent_name, bench_args, env)

    is_ppo = agent_name == "ppo"
    trainer_kwargs = dict(
        env=env,
        agent=agent,
        checkpoint_interval=0,
        checkpoint_path="",
        eval_interval=0,
        plugins=plugins or [],
    )
    if is_ppo:
        trainer_kwargs["num_steps"] = bench_args.num_steps
        trainer_kwargs["max_iterations"] = bench_args.iterations
    else:
        trainer_kwargs["max_steps"] = bench_args.iterations
        trainer_kwargs["log_interval"] = bench_args.iterations

    with timer() as t:
        trainer = create_trainer(agent_name, **trainer_kwargs)
        trainer.train()

    iters = bench_args.iterations
    total_frames = iters * bench_args.num_steps * num_envs if is_ppo else iters * num_envs
    summary(
        f"{agent_name.upper()} Trainer ({num_envs} envs, {iters} iters, {bench_args.device})",
        t["elapsed"],
        fps_total=total_frames / t["elapsed"],
        iter_avg=t["elapsed"] / iters,
    )
    env.close()
