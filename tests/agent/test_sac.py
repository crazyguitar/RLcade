import os
import tempfile

import numpy as np
import pytest
import torch
from torch.distributions import Categorical

from tests.conftest import make_args, fill_buffer


class TestSACAgent:
    @pytest.fixture()
    def env_and_agent(self, rom):
        from rlcade.envs import create_env
        from rlcade.agent import create_agent

        args = make_args(rom, agent="sac")
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n
        agent = create_agent("sac", args, env)
        agent.create_optimizers()
        yield env, agent
        env.close()

    def test_create_agent(self, env_and_agent):
        from rlcade.agent.sac import SAC

        _, agent = env_and_agent
        assert isinstance(agent, SAC)

    def test_get_action(self, env_and_agent):
        env, agent = env_and_agent
        obs, _ = env.reset()
        action = agent.get_action(torch.as_tensor(obs, dtype=torch.float32), deterministic=True)
        assert 0 <= action < env.action_space.n

    def test_get_action_deterministic(self, env_and_agent):
        env, agent = env_and_agent
        obs, _ = env.reset()
        t = torch.as_tensor(obs, dtype=torch.float32)
        a1 = agent.get_action(t, deterministic=True)
        a2 = agent.get_action(t, deterministic=True)
        assert a1 == a2

    def test_store_and_learn(self, env_and_agent):
        env, agent = env_and_agent
        agent.sac.learn_start = 16

        obs, _ = env.reset()
        for _ in range(32):
            action = agent.get_action(torch.as_tensor(obs, dtype=torch.float32))
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.store(obs, action, reward, next_obs, terminated or truncated)
            obs = next_obs
            if terminated or truncated:
                obs, _ = env.reset()

        assert agent.can_learn()
        metrics = agent.learn()
        assert "critic_loss" in metrics
        assert "actor_loss" in metrics
        assert "alpha" in metrics

    def test_evaluate(self, env_and_agent):
        env, agent = env_and_agent
        scores = agent.evaluate(env, num_episodes=1)
        assert len(scores) == 1
        assert isinstance(scores[0], float)

    def test_alpha_is_positive(self, env_and_agent):
        _, agent = env_and_agent
        assert agent.sac.alpha.item() > 0


class TestSACCorrectness:
    """Edge case tests verifying SAC-Discrete algorithm correctness."""

    @pytest.fixture()
    def agent(self, rom):
        from rlcade.envs import create_env
        from rlcade.agent.sac import SAC, SACConfig

        args = make_args(rom, agent="sac")
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n
        agent = SAC(SACConfig.from_args(args), env)
        yield agent
        env.close()

    def test_dual_q_networks_are_independent(self, agent):
        """Q1 and Q2 should have different weights after init (random init)."""
        q1_params = list(agent.sac.q1.parameters())
        q2_params = list(agent.sac.q2.parameters())
        # At least one layer should differ (random init)
        any_different = any(not torch.equal(p1, p2) for p1, p2 in zip(q1_params, q2_params))
        assert any_different, "Q1 and Q2 should have different random initializations"

    def test_target_networks_start_equal(self, agent):
        """Target networks should be exact copies of online networks at init."""
        for p, tp in zip(agent.sac.q1.parameters(), agent.sac.q1_target.parameters()):
            assert torch.equal(p, tp)
        for p, tp in zip(agent.sac.q2.parameters(), agent.sac.q2_target.parameters()):
            assert torch.equal(p, tp)

    def test_soft_update_moves_targets(self, agent):
        """After soft update, targets should move toward online but not equal."""
        # Perturb online network so it differs from target
        with torch.no_grad():
            for p in agent.sac.q1.parameters():
                p.add_(torch.randn_like(p))
        old_target = [p.clone() for p in agent.sac.q1_target.parameters()]
        agent.sac._soft_update()
        for old_p, new_p in zip(old_target, agent.sac.q1_target.parameters()):
            assert not torch.equal(old_p, new_p), "Target should have moved"

    def test_alpha_updates_after_learn(self, rom):
        """Alpha should change after a learn step."""
        from rlcade.envs import create_env
        from rlcade.agent.sac import SAC, SACConfig

        args = make_args(rom, agent="sac")
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n
        agent = SAC(SACConfig.from_args(args), env)
        agent.create_optimizers()
        agent.sac.learn_start = 16

        fill_buffer(env, agent)

        alpha_before = agent.sac.alpha.item()
        agent.learn()
        alpha_after = agent.sac.alpha.item()
        assert alpha_before != alpha_after, "Alpha should change after learning"
        assert alpha_after > 0, "Alpha must remain positive"
        env.close()

    def test_actor_probs_sum_to_one(self, agent):
        """Actor output should be a valid probability distribution."""
        obs = torch.randn(1, *agent.config.obs_shape)
        with torch.no_grad():
            dist = Categorical(logits=agent.sac.actor(obs.to(agent.device)))
        probs = dist.probs
        assert probs.shape[-1] == agent.config.n_actions
        assert torch.allclose(probs.sum(dim=-1), torch.ones(1), atol=1e-5)
        assert (probs >= 0).all()

    def test_target_entropy_is_positive(self, agent):
        """Target entropy = log(|A|) should be positive for |A| > 1."""
        assert agent.sac.target_entropy > 0

    def test_learn_returns_finite_values(self, rom):
        """All metrics from learn() should be finite (no NaN/Inf)."""
        from rlcade.envs import create_env
        from rlcade.agent.sac import SAC, SACConfig

        args = make_args(rom, agent="sac")
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n
        agent = SAC(SACConfig.from_args(args), env)
        agent.create_optimizers()
        agent.sac.learn_start = 16

        fill_buffer(env, agent)

        metrics = agent.learn()
        for key, val in metrics.items():
            assert np.isfinite(val), f"{key} is not finite: {val}"
        env.close()

    def test_checkpoint_preserves_alpha(self, rom):
        """Save/load should preserve the exact alpha value."""
        from rlcade.envs import create_env
        from rlcade.agent.sac import SAC, SACConfig

        args = make_args(rom, agent="sac")
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n
        agent = SAC(SACConfig.from_args(args), env)
        agent.create_optimizers()

        # Mutate alpha so it's not the default
        agent.sac.log_alpha.data.fill_(1.5)
        expected_alpha = agent.sac.alpha.item()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            from rlcade.checkpoint.checkpoint import Checkpoint

            Checkpoint(path).save(agent.state())
            agent.sac.log_alpha.data.fill_(0.0)  # clobber
            with open(path, "rb") as ckpt_f:
                agent.load(ckpt_f)
            assert abs(agent.sac.alpha.item() - expected_alpha) < 1e-5
        finally:
            os.unlink(path)
            env.close()


class TestSACCheckpoint:
    def test_save_and_load(self, rom):
        from rlcade.envs import create_env
        from rlcade.agent.sac import SAC, SACConfig

        args = make_args(rom, agent="sac")
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n
        agent = SAC(SACConfig.from_args(args))
        agent.create_optimizers()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            from rlcade.checkpoint.checkpoint import Checkpoint

            Checkpoint(path).save(agent.state(step=500))
            with open(path, "rb") as ckpt_f:
                step = agent.load(ckpt_f)
            assert step == 500
        finally:
            os.unlink(path)
            env.close()

    def test_load_checkpoint_inference(self, rom):
        from rlcade.envs import create_env
        from rlcade.agent.sac import SAC, SACConfig

        args = make_args(rom, agent="sac")
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            from rlcade.checkpoint.checkpoint import Checkpoint

            agent = SAC(SACConfig.from_args(args), env)
            Checkpoint(path).save(agent.state(step=50))

            with Checkpoint(path).reader() as f:
                loaded = SAC.restore(SACConfig.from_args(args), f, env)
            obs, _ = env.reset()
            action = loaded.get_action(torch.as_tensor(obs, dtype=torch.float32), deterministic=True)
            assert 0 <= action < env.action_space.n
        finally:
            os.unlink(path)
            env.close()


class TestSACSafetensors:
    def test_save_and_load(self, rom):
        from rlcade.envs import create_env
        from rlcade.agent import create_agent
        from rlcade.checkpoint.safetensors import save_safetensors, load_safetensors

        args = make_args(rom, agent="sac")
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n
        agent = create_agent("sac", args, env)
        agent.create_optimizers()

        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        try:
            state = agent.state(step=123)
            save_safetensors(state, path, step=123)

            loaded, step = load_safetensors(path, device=torch.device("cpu"))
            assert step == 123
            assert loaded, "no models saved"
            for name, sub in loaded.items():
                for k, v in sub.items():
                    assert torch.equal(state[name][k], v), f"weight mismatch: {name}.{k}"
        finally:
            os.unlink(path)
            env.close()

    def test_load_inference(self, rom):
        from rlcade.envs import create_env
        from rlcade.agent import create_agent, load_agent
        from rlcade.checkpoint.safetensors import save_safetensors

        args = make_args(rom, agent="sac")
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n

        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        try:
            source = create_agent("sac", args, env)
            source.create_optimizers()
            with torch.no_grad():
                for _, m in source._impl.models():
                    for p in m.parameters():
                        p.fill_(0.123)
            save_safetensors(source.state(step=7), path, step=7)

            args.checkpoint = path
            loaded = load_agent("sac", args, env)
            for _, m in loaded._impl.models():
                for p in m.parameters():
                    assert torch.allclose(p, torch.full_like(p, 0.123)), "weights not restored"

            obs, _ = env.reset()
            t = torch.as_tensor(obs, dtype=torch.float32)
            action = loaded.get_action(t, deterministic=True)
            assert 0 <= action < env.action_space.n
        finally:
            os.unlink(path)
            env.close()


class TestVecSACAgent:
    @pytest.fixture()
    def vec_env_and_agent(self, rom):
        from rlcade.envs import create_vector_env
        from rlcade.agent import create_agent
        from tests.conftest import make_vec_args

        args = make_vec_args(rom, agent="sac")
        env = create_vector_env(args)
        args.obs_shape = env.observation_space.shape[1:]
        args.n_actions = env.action_space.n
        agent = create_agent("sac", args, env)
        agent.create_optimizers()
        yield env, agent
        env.close()

    def test_create_vec_agent(self, vec_env_and_agent):
        from rlcade.agent.sac import SAC, VecSAC

        _, agent = vec_env_and_agent
        assert isinstance(agent, SAC)
        assert isinstance(agent.sac, VecSAC)

    def test_get_action_batched(self, vec_env_and_agent):
        env, agent = vec_env_and_agent
        obs, _ = env.reset()
        actions = agent.get_action(torch.as_tensor(obs, dtype=torch.float32), deterministic=True)
        assert len(actions) == env.num_envs
        assert all(0 <= a < env.action_space.n for a in actions)

    def test_store_batched(self, vec_env_and_agent):
        env, agent = vec_env_and_agent
        obs, _ = env.reset()
        actions = agent.get_action(torch.as_tensor(obs, dtype=torch.float32))
        next_obs, rewards, terminated, truncated, _ = env.step(actions)
        agent.store(obs, actions, rewards, next_obs, terminated | truncated)
        assert len(agent.sac.buffer) == env.num_envs

    def test_evaluate_vec(self, vec_env_and_agent):
        env, agent = vec_env_and_agent
        scores = agent.evaluate(env, num_episodes=2)
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)
