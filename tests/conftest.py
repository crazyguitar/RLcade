import argparse
import os

import pytest

ENV_ID = "rlcade/SuperMarioBros-v0"

from rlcade.envs import register_envs

register_envs()


def pytest_addoption(parser):
    parser.addoption("--rom", default=None, help="Path to .nes ROM file")
    parser.addoption("--s3", default=None, help="S3 URL for filesystem tests (e.g. s3://bucket/prefix)")


@pytest.fixture(scope="session")
def rom(request):
    path = request.config.getoption("--rom")
    if not path or not os.path.isfile(path):
        pytest.skip(f"ROM not found: {path}")
    return path


@pytest.fixture(scope="session")
def s3_url(request):
    url = request.config.getoption("--s3")
    if not url:
        pytest.skip("--s3 not specified")
    return url


def make_args(rom, **overrides):
    defaults = dict(
        agent="ppo",
        env=ENV_ID,
        rom=rom,
        actor="actor",
        critic="critic",
        world=1,
        stage=1,
        actions="complex",
        render_mode=None,
        custom_reward=False,
        device="cpu",
        checkpoint=None,
        lr=2.5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        num_steps=16,
        batch_size=16,
        update_epochs=2,
        max_iterations=2,
        checkpoint_interval=1,
        checkpoint_path="",
        target_score=None,
        eval_interval=0,
        eval_episodes=1,
        lr_schedule=False,
        tensorboard=None,
        # DQN replay buffer — keep small for tests to avoid multi-GB allocations
        # when scaled by num_envs (default 100K × num_envs × obs_shape is huge)
        buffer_size=1000,
        # Distributed (disabled by default in tests)
        distributed=None,
        backend="gloo",
        # Encoder
        encoder="cnn",
        resnet_channels=[16, 32, 32],
        resnet_out_dim=256,
        # AMP & gradient accumulation
        amp=False,
        grad_accum_steps=1,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def make_vec_args(rom, **overrides):
    """Make args that produce a vector env (all stages of world 1)."""
    return make_args(rom, world=1, stage=None, **overrides)


def step_until_terminated(env, rng, max_steps=50000):
    """Step with random actions until the episode terminates. Returns final (obs, info)."""
    for _ in range(max_steps):
        obs, reward, terminated, truncated, info = env.step(rng.randint(env.action_space.n))
        if terminated or truncated:
            return obs, info
    raise RuntimeError(f"Episode did not terminate within {max_steps} steps")


def make_off_policy_trainer(rom, agent_name, *, vec=False, learn_start=16, eval_interval=0, **extra):
    """Create trainer for off-policy tests. Returns (trainer, env, eval_env)."""
    from rlcade.training import create_trainer

    extra_defaults = dict(
        max_iterations=50,
        checkpoint_path="",
        buffer_size=1000,
        log_interval=25,
        eval_interval=eval_interval,
        eval_episodes=1,
        learn_start=learn_start,
        learn_freq=1,
    )
    extra_defaults.update(extra)

    if vec:
        args = make_vec_args(rom, agent=agent_name, **extra_defaults)
    else:
        args = make_args(rom, agent=agent_name, **extra_defaults)

    trainer = create_trainer(agent_name, args)

    # Mirror learn_start onto the agent for tests that inspect agent.can_learn().
    inner_name = {"rainbow_dqn": "rainbow"}.get(agent_name, agent_name)
    inner = getattr(trainer.agent, inner_name)
    inner.learn_start = learn_start

    return trainer, trainer.env, trainer.eval_env


def fill_buffer(env, agent, steps=32):
    """Fill an off-policy agent's replay buffer with random transitions."""
    import torch

    obs, _ = env.reset()
    for _ in range(steps):
        action = agent.get_action(torch.as_tensor(obs, dtype=torch.float32))
        next_obs, reward, terminated, truncated, _ = env.step(action)
        agent.store(obs, action, reward, next_obs, terminated or truncated)
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
