"""Shared benchmark utilities and argument parsing."""

import argparse
import time
from contextlib import contextmanager

from rlcade.logger import get_logger

logger = get_logger("bench")

AGENTS = ["ppo", "dqn", "rainbow_dqn"]


@contextmanager
def timer():
    """Context manager that records elapsed time."""
    result = {"elapsed": 0.0}
    t0 = time.time()
    yield result
    result["elapsed"] = time.time() - t0


def summary(title: str, elapsed: float, **metrics):
    """Log a benchmark summary with title and key-value metrics."""
    logger.info("=== %s ===", title)
    parts = [f"Time: {elapsed:.2f}s"]
    for k, v in metrics.items():
        label = k.replace("_", " ").title()
        parts.append(f"{label}: {v:.1f}" if isinstance(v, float) else f"{label}: {v}")
    logger.info("  %s", " | ".join(parts))


def agent_defaults(agent: str, parsed: argparse.Namespace) -> argparse.Namespace:
    """Fill in defaults expected by env/agent/trainer creation for a given agent."""
    common = dict(
        env="rlcade/SuperMarioBros-v0",
        render_mode=None,
        custom_reward=True,
        device=getattr(parsed, "device", "cpu"),
        checkpoint=None,
    )
    agent_specific = {
        "ppo": dict(
            agent="ppo",
            actor="actor",
            critic="critic",
            lr=2.5e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_coef=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            update_epochs=4,
            batch_size=256,
            lr_schedule=False,
            max_iterations=10000,
        ),
        "dqn": dict(
            agent="dqn",
            qnet="qnet",
            lr=1e-4,
            gamma=0.99,
            tau=1e-3,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=100_000,
            batch_size=32,
            buffer_size=100_000,
            learn_start=0,
            learn_freq=4,
            double=True,
        ),
        "rainbow_dqn": dict(
            agent="rainbow_dqn",
            qnet="rainbow_qnet",
            lr=6.25e-5,
            gamma=0.99,
            tau=1e-3,
            batch_size=32,
            buffer_size=100_000,
            learn_start=0,
            learn_freq=4,
            alpha=0.6,
            beta_start=0.4,
            beta_end=1.0,
            prior_eps=1e-6,
            num_atoms=51,
            v_min=-200.0,
            v_max=200.0,
            noise_std=0.5,
            n_step=3,
        ),
    }

    defaults = {**common, **agent_specific[agent]}
    for k, v in defaults.items():
        if not hasattr(parsed, k):
            setattr(parsed, k, v)
    parsed.agent = agent
    return parsed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RLcade benchmarks")
    p.add_argument("--rom", type=str, required=True)
    p.add_argument("--world", type=int, default=1)
    p.add_argument("--stage", type=int, default=None)
    p.add_argument("--actions", type=str, default="complex")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--num-steps", type=int, default=128)
    p.add_argument("--iterations", type=int, default=8)
    p.add_argument(
        "--bench",
        type=str,
        default="all",
        choices=["env", "ppo", "dqn", "rainbow_dqn", "all"],
    )
    p.add_argument("--viztracer", type=str, default=None, help="VizTracer output file prefix (enables profiling)")
    return p.parse_args()
