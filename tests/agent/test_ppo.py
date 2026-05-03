import os
import tempfile

import pytest
import torch

from tests.conftest import make_args


class TestPPOAgent:
    @pytest.fixture()
    def env_and_agent(self, rom):
        from rlcade.envs import create_env
        from rlcade.agent import create_agent

        args = make_args(rom)
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n
        agent = create_agent("ppo", args, env)
        agent.create_optimizers()
        yield env, agent
        env.close()

    def test_create_agent(self, env_and_agent):
        from rlcade.agent.ppo import PPO

        _, agent = env_and_agent
        assert isinstance(agent, PPO)

    def test_get_action(self, env_and_agent):
        env, agent = env_and_agent
        obs, _ = env.reset()
        t = torch.as_tensor(obs, dtype=torch.float32)
        action, log_prob, value = agent.get_action(t)
        assert 0 <= action.item() < env.action_space.n
        assert log_prob.dim() == 0
        assert value.dim() == 0

    def test_get_action_deterministic(self, env_and_agent):
        env, agent = env_and_agent
        obs, _ = env.reset()
        t = torch.as_tensor(obs, dtype=torch.float32)
        # Deterministic should return the same action every time
        a1, _, _ = agent.get_action(t, deterministic=True)
        a2, _, _ = agent.get_action(t, deterministic=True)
        assert a1.item() == a2.item()
        assert 0 <= a1.item() < env.action_space.n

    def test_collect_rollout(self, env_and_agent):
        env, agent = env_and_agent
        rollout, next_obs = agent.collect_rollout(env, num_steps=8)
        assert rollout["obs"].shape[0] == 8
        assert rollout["actions"].shape[0] == 8
        assert rollout["rewards"].shape[0] == 8
        assert next_obs.shape == (4, 84, 84)

    def test_learn(self, env_and_agent):
        env, agent = env_and_agent
        rollout, _ = agent.collect_rollout(env, num_steps=16)
        metrics = agent.learn(rollout)
        assert "loss" in metrics
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics


class TestCheckpoint:
    def test_save_and_load(self, rom):
        from rlcade.envs import create_env
        from rlcade.agent import create_agent

        args = make_args(rom)
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n
        agent = create_agent("ppo", args, env)
        agent.create_optimizers()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            from rlcade.checkpoint.checkpoint import Checkpoint

            Checkpoint(path).save(agent.state(step=100))
            with open(path, "rb") as ckpt_f:
                step = agent.load(ckpt_f)
            assert step == 100
        finally:
            os.unlink(path)
            env.close()

    def test_load_checkpoint_inference(self, rom):
        from rlcade.envs import create_env
        from rlcade.agent.ppo import PPO, PPOConfig

        args = make_args(rom)
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            from rlcade.checkpoint.checkpoint import Checkpoint

            agent = PPO(PPOConfig.from_args(args), env)
            agent.create_optimizers()
            Checkpoint(path).save(agent.state(step=50))

            with Checkpoint(path).reader() as f:
                loaded = PPO.restore(PPOConfig.from_args(args), f, env)
            obs, _ = env.reset()
            t = torch.as_tensor(obs, dtype=torch.float32)
            action, _, _ = loaded.get_action(t)
            assert 0 <= action.item() < env.action_space.n
        finally:
            os.unlink(path)
            env.close()


class TestVecPPOAgent:
    @pytest.fixture()
    def vec_env_and_agent(self, rom):
        from rlcade.envs import create_vector_env
        from rlcade.agent import create_agent
        from tests.conftest import make_vec_args

        args = make_vec_args(rom)
        env = create_vector_env(args)
        args.obs_shape = env.observation_space.shape[1:]
        args.n_actions = env.action_space.n
        agent = create_agent("ppo", args, env)
        agent.create_optimizers()
        yield env, agent
        env.close()

    def test_create_vec_agent(self, vec_env_and_agent):
        from rlcade.agent.ppo import PPO, VecPPO

        _, agent = vec_env_and_agent
        assert isinstance(agent, PPO)
        assert isinstance(agent.ppo, VecPPO)

    def test_get_action_batched(self, vec_env_and_agent):
        env, agent = vec_env_and_agent
        obs, _ = env.reset()
        t = torch.as_tensor(obs, dtype=torch.float32)
        action, log_prob, value = agent.get_action(t)
        assert action.shape[0] == env.num_envs
        assert log_prob.shape[0] == env.num_envs
        assert value.shape[0] == env.num_envs

    def test_get_action_deterministic_batched(self, vec_env_and_agent):
        env, agent = vec_env_and_agent
        obs, _ = env.reset()
        t = torch.as_tensor(obs, dtype=torch.float32)
        a1, _, _ = agent.get_action(t, deterministic=True)
        a2, _, _ = agent.get_action(t, deterministic=True)
        assert torch.equal(a1, a2)
        assert a1.shape[0] == env.num_envs

    def test_collect_rollout_vec(self, vec_env_and_agent):
        env, agent = vec_env_and_agent
        rollout, next_obs = agent.collect_rollout(env, num_steps=8)
        # Rollout keeps (T, N) shape; flattening happens in process_trajectory
        assert rollout["obs"].shape[0] == 8
        assert rollout["obs"].shape[1] == env.num_envs
        assert rollout["actions"].shape == (8, env.num_envs)
        assert rollout["rewards"].shape == (8, env.num_envs)
        assert next_obs.shape[0] == env.num_envs

    def test_evaluate_vec(self, vec_env_and_agent):
        env, agent = vec_env_and_agent
        scores = agent.evaluate(env, num_episodes=2)
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)

    def test_process_trajectory_flattens(self, vec_env_and_agent):
        env, agent = vec_env_and_agent
        rollout, _ = agent.collect_rollout(env, num_steps=8)
        # Rollout is (T, N), process_trajectory should flatten to (T*N,)
        trajectory = agent.ppo.process_trajectory(rollout)
        total = 8 * env.num_envs
        assert trajectory["obs"].shape[0] == total
        assert trajectory["actions"].shape[0] == total
        assert trajectory["advantages"].shape[0] == total
        assert trajectory["returns"].shape[0] == total

    def test_learn_vec(self, vec_env_and_agent):
        env, agent = vec_env_and_agent
        rollout, _ = agent.collect_rollout(env, num_steps=16)
        metrics = agent.learn(rollout)
        assert "loss" in metrics
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
