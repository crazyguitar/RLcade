"""Integration tests for AMP and gradient accumulation across all agents."""

import pytest
import torch

from tests.conftest import make_args


def _make_agent(rom, agent_name, *, amp=False, grad_accum_steps=1, **extra):
    """Create an agent with AMP/grad-accum settings. Returns (env, agent)."""
    from rlcade.envs import create_env
    from rlcade.agent import create_agent

    args = make_args(rom, agent=agent_name, amp=amp, grad_accum_steps=grad_accum_steps, **extra)
    env = create_env(args)
    args.obs_shape = env.observation_space.shape
    args.n_actions = env.action_space.n
    agent = create_agent(agent_name, args, env)
    agent.create_optimizers()
    return env, agent


class TestPPOAmp:
    def test_learn_with_amp(self, rom):
        env, agent = _make_agent(rom, "ppo", amp=True)
        try:
            rollout, _ = agent.collect_rollout(env, num_steps=16)
            metrics = agent.learn(rollout)
            assert "loss" in metrics
            assert "policy_loss" in metrics
        finally:
            env.close()

    def test_learn_with_grad_accum(self, rom):
        env, agent = _make_agent(rom, "ppo", grad_accum_steps=2)
        try:
            rollout, _ = agent.collect_rollout(env, num_steps=16)
            metrics = agent.learn(rollout)
            assert "loss" in metrics
        finally:
            env.close()

    def test_learn_with_amp_and_grad_accum(self, rom):
        env, agent = _make_agent(rom, "ppo", amp=True, grad_accum_steps=2)
        try:
            rollout, _ = agent.collect_rollout(env, num_steps=16)
            metrics = agent.learn(rollout)
            assert "loss" in metrics
        finally:
            env.close()


class TestLstmPPOAmp:
    def test_learn_with_amp(self, rom):
        env, agent = _make_agent(rom, "lstm_ppo", amp=True)
        try:
            rollout, _ = agent.collect_rollout(env, num_steps=16)
            metrics = agent.learn(rollout)
            assert "loss" in metrics
        finally:
            env.close()

    def test_learn_with_grad_accum(self, rom):
        env, agent = _make_agent(rom, "lstm_ppo", grad_accum_steps=2)
        try:
            rollout, _ = agent.collect_rollout(env, num_steps=16)
            metrics = agent.learn(rollout)
            assert "loss" in metrics
        finally:
            env.close()

    def test_learn_with_amp_and_grad_accum(self, rom):
        env, agent = _make_agent(rom, "lstm_ppo", amp=True, grad_accum_steps=2)
        try:
            rollout, _ = agent.collect_rollout(env, num_steps=16)
            metrics = agent.learn(rollout)
            assert "loss" in metrics
        finally:
            env.close()


class TestDQNAmp:
    def test_learn_with_amp(self, rom):
        env, agent = _make_agent(rom, "dqn", amp=True, buffer_size=1000)
        try:
            from tests.conftest import fill_buffer

            fill_buffer(env, agent, steps=64)
            agent.dqn.learn_start = 32
            metrics = agent.learn()
            assert "loss" in metrics
        finally:
            env.close()

    def test_learn_with_grad_accum(self, rom):
        env, agent = _make_agent(rom, "dqn", grad_accum_steps=2, buffer_size=1000)
        try:
            from tests.conftest import fill_buffer

            fill_buffer(env, agent, steps=64)
            agent.dqn.learn_start = 32
            metrics = agent.learn()
            assert "loss" in metrics
        finally:
            env.close()

    def test_learn_with_amp_and_grad_accum(self, rom):
        env, agent = _make_agent(rom, "dqn", amp=True, grad_accum_steps=2, buffer_size=1000)
        try:
            from tests.conftest import fill_buffer

            fill_buffer(env, agent, steps=64)
            agent.dqn.learn_start = 32
            metrics = agent.learn()
            assert "loss" in metrics
        finally:
            env.close()


class TestRainbowDQNAmp:
    def test_learn_with_amp(self, rom):
        env, agent = _make_agent(rom, "rainbow_dqn", amp=True, buffer_size=1000)
        try:
            from tests.conftest import fill_buffer

            fill_buffer(env, agent, steps=64)
            agent.rainbow.learn_start = 32
            metrics = agent.learn()
            assert "loss" in metrics
        finally:
            env.close()

    def test_learn_with_grad_accum(self, rom):
        env, agent = _make_agent(rom, "rainbow_dqn", grad_accum_steps=2, buffer_size=1000)
        try:
            from tests.conftest import fill_buffer

            fill_buffer(env, agent, steps=64)
            agent.rainbow.learn_start = 32
            metrics = agent.learn()
            assert "loss" in metrics
        finally:
            env.close()

    def test_learn_with_amp_and_grad_accum(self, rom):
        env, agent = _make_agent(rom, "rainbow_dqn", amp=True, grad_accum_steps=2, buffer_size=1000)
        try:
            from tests.conftest import fill_buffer

            fill_buffer(env, agent, steps=64)
            agent.rainbow.learn_start = 32
            metrics = agent.learn()
            assert "loss" in metrics
        finally:
            env.close()


class TestSACAmp:
    def test_learn_with_amp(self, rom):
        env, agent = _make_agent(rom, "sac", amp=True, buffer_size=1000)
        try:
            from tests.conftest import fill_buffer

            fill_buffer(env, agent, steps=64)
            agent.sac.learn_start = 32
            metrics = agent.learn()
            assert "critic_loss" in metrics
            assert "actor_loss" in metrics
        finally:
            env.close()

    def test_learn_with_grad_accum(self, rom):
        env, agent = _make_agent(rom, "sac", grad_accum_steps=2, buffer_size=1000)
        try:
            from tests.conftest import fill_buffer

            fill_buffer(env, agent, steps=64)
            agent.sac.learn_start = 32
            metrics = agent.learn()
            assert "critic_loss" in metrics
            assert "actor_loss" in metrics
        finally:
            env.close()

    def test_learn_with_amp_and_grad_accum(self, rom):
        env, agent = _make_agent(rom, "sac", amp=True, grad_accum_steps=2, buffer_size=1000)
        try:
            from tests.conftest import fill_buffer

            fill_buffer(env, agent, steps=64)
            agent.sac.learn_start = 32
            metrics = agent.learn()
            assert "critic_loss" in metrics
            assert "actor_loss" in metrics
        finally:
            env.close()
