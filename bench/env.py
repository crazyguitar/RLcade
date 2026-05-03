"""Environment benchmarks — single env and vectorized env step throughput."""

import numpy as np

from rlcade.envs import create_env, create_vector_env
from bench.utils import summary, timer


def bench_env_step(args, steps: int = 2048):
    """Benchmark single-env step throughput."""
    env = create_env(args)
    env.reset()
    n_actions = env.action_space.n

    for _ in range(10):
        env.step(np.random.randint(0, n_actions))

    with timer() as t:
        for _ in range(steps):
            _, _, terminated, truncated, _ = env.step(np.random.randint(0, n_actions))
            if terminated or truncated:
                env.reset()

    sps = steps / t["elapsed"]
    summary("Single Env Step", t["elapsed"], steps=steps, sps=sps)
    env.close()
    return {"sps": sps}


def bench_vec_env_step(args, steps: int = 2048):
    """Benchmark vectorized env step throughput."""
    env = create_vector_env(args)
    env.reset()
    n_actions = env.action_space.n
    num_envs = env.num_envs

    for _ in range(10):
        env.step(np.random.randint(0, n_actions, size=num_envs))

    with timer() as t:
        for _ in range(steps):
            env.step(np.random.randint(0, n_actions, size=num_envs))

    total = steps * num_envs
    summary(
        f"Vec Env Step ({num_envs} envs)",
        t["elapsed"],
        steps=steps,
        sps_per_env=steps / t["elapsed"],
        fps_total=total / t["elapsed"],
    )
    env.close()
    return {"sps_per_env": steps / t["elapsed"], "fps_total": total / t["elapsed"]}
