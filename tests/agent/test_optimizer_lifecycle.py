"""Tests for create_optimizers lifecycle: inference without optimizer, idempotency."""

import os
import tempfile

import pytest
import torch

from tests.conftest import make_args


class TestInferenceWithoutOptimizer:
    """Agents should work for inference (get_action, save, load) without create_optimizers."""

    def test_ppo_get_action_no_optimizer(self, rom):
        from rlcade.agent import create_agent
        from rlcade.envs import create_env

        args = make_args(rom)
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n
        agent = create_agent("ppo", args, env)
        # No create_optimizers — inference only
        obs, _ = env.reset()
        action, log_prob, value = agent.get_action(torch.as_tensor(obs, dtype=torch.float32))
        assert 0 <= action.item() < env.action_space.n
        env.close()

    def test_dqn_get_action_no_optimizer(self, rom):
        from rlcade.agent import create_agent
        from rlcade.envs import create_env

        args = make_args(rom, agent="dqn")
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n
        agent = create_agent("dqn", args, env)
        obs, _ = env.reset()
        action = agent.get_action(torch.as_tensor(obs, dtype=torch.float32), deterministic=True)
        assert 0 <= action < env.action_space.n
        env.close()

    def test_save_without_optimizer_excludes_optimizer_keys(self, rom):
        from rlcade.agent import create_agent
        from rlcade.envs import create_env

        args = make_args(rom)
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n
        agent = create_agent("ppo", args, env)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            from rlcade.checkpoint.checkpoint import Checkpoint

            Checkpoint(path).save(agent.state(step=10))
            state = torch.load(path, weights_only=True)
            assert "actor" in state
            assert "critic" in state
            assert "optimizer" not in state
        finally:
            os.unlink(path)
            env.close()


class TestCreateOptimizersIdempotency:
    """Calling create_optimizers twice should work and produce valid optimizers."""

    def test_ppo_double_create(self, rom):
        from rlcade.agent import create_agent
        from rlcade.envs import create_env

        args = make_args(rom)
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n
        agent = create_agent("ppo", args, env)
        agent.create_optimizers()
        agent.create_optimizers()  # second call — should not crash
        # Optimizer should reference current model params
        opt_params = {id(p) for group in agent._impl.optimizer.param_groups for p in group["params"]}
        model_params = {id(p) for p in agent._impl.parameters}
        assert opt_params == model_params
        env.close()

    def test_sac_double_create(self, rom):
        from rlcade.agent import create_agent
        from rlcade.envs import create_env

        args = make_args(rom, agent="sac")
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n
        agent = create_agent("sac", args, env)
        agent.create_optimizers()
        agent.create_optimizers()
        assert agent._impl.actor_optimizer is not None
        assert agent._impl.critic_optimizer is not None
        assert agent._impl.alpha_optimizer is not None
        env.close()
