import os
import tempfile

import pytest
import torch

from tests.conftest import make_args, make_vec_args


def _lstm_args(rom, **overrides):
    return make_args(rom, agent="lstm_ppo", lstm_hidden=64, **overrides)


def _lstm_vec_args(rom, **overrides):
    return make_vec_args(rom, agent="lstm_ppo", lstm_hidden=64, **overrides)


class TestLstmPPOAgent:
    @pytest.fixture()
    def env_and_agent(self, rom):
        from rlcade.envs import create_env
        from rlcade.agent import create_agent

        args = _lstm_args(rom)
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n
        agent = create_agent("lstm_ppo", args, env)
        agent.create_optimizers()
        yield env, agent
        env.close()

    def test_create_agent(self, env_and_agent):
        from rlcade.agent.ppo import LstmPPO

        _, agent = env_and_agent
        assert isinstance(agent, LstmPPO)

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
        a1, _, _ = agent.get_action(t, deterministic=True)
        a2, _, _ = agent.get_action(t, deterministic=True)
        assert a1.item() == a2.item()

    def test_collect_rollout(self, env_and_agent):
        env, agent = env_and_agent
        rollout, next_obs = agent.collect_rollout(env, num_steps=8)
        assert rollout["obs"].shape[0] == 8
        assert rollout["actions"].shape[0] == 8
        assert rollout["saved_hx"].shape[0] == 8
        assert next_obs.shape == (4, 84, 84)

    def test_learn(self, env_and_agent):
        env, agent = env_and_agent
        rollout, _ = agent.collect_rollout(env, num_steps=16)
        metrics = agent.learn(rollout)
        assert "loss" in metrics
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics


class TestLstmPPOCheckpoint:
    def test_save_and_load(self, rom):
        from rlcade.envs import create_env
        from rlcade.agent import create_agent

        args = _lstm_args(rom)
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n
        agent = create_agent("lstm_ppo", args, env)
        agent.create_optimizers()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            from rlcade.checkpoint.checkpoint import Checkpoint

            Checkpoint(path).save(agent.state(step=42))
            with open(path, "rb") as ckpt_f:
                step = agent.load(ckpt_f)
            assert step == 42
        finally:
            os.unlink(path)
            env.close()


class TestLstmPPOSafetensors:
    def test_save_and_load(self, rom):
        from rlcade.envs import create_env
        from rlcade.agent import create_agent
        from rlcade.checkpoint.safetensors import save_safetensors, load_safetensors

        args = _lstm_args(rom)
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n
        agent = create_agent("lstm_ppo", args, env)
        agent.create_optimizers()

        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        try:
            state = agent.state(step=123)
            save_safetensors(state, path, step=123)

            loaded, step = load_safetensors(path, device=torch.device("cpu"))
            assert step == 123
            for key, value in state.items():
                if (
                    isinstance(value, dict)
                    and value
                    and all(isinstance(v, torch.Tensor) for v in value.values())
                ):
                    assert key in loaded, f"missing model {key} after round-trip"
                    for k, v in value.items():
                        assert torch.equal(loaded[key][k], v), f"weight mismatch: {key}.{k}"
        finally:
            os.unlink(path)
            env.close()

    def test_load_inference(self, rom):
        from rlcade.envs import create_env
        from rlcade.agent import create_agent, load_agent
        from rlcade.checkpoint.safetensors import save_safetensors

        args = _lstm_args(rom)
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n

        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        try:
            source = create_agent("lstm_ppo", args, env)
            source.create_optimizers()
            with torch.no_grad():
                for _, m in source._impl.models():
                    for p in m.parameters():
                        p.fill_(0.123)
            save_safetensors(source.state(step=7), path, step=7)

            args.checkpoint = path
            loaded = load_agent("lstm_ppo", args, env)
            for _, m in loaded._impl.models():
                for p in m.parameters():
                    assert torch.allclose(p, torch.full_like(p, 0.123)), "weights not restored"

            obs, _ = env.reset()
            t = torch.as_tensor(obs, dtype=torch.float32)
            action, _, _ = loaded.get_action(t)
            assert 0 <= action.item() < env.action_space.n
        finally:
            os.unlink(path)
            env.close()


class TestLstmVecPPOAgent:
    @pytest.fixture()
    def vec_env_and_agent(self, rom):
        from rlcade.envs import create_vector_env
        from rlcade.agent import create_agent

        args = _lstm_vec_args(rom)
        env = create_vector_env(args)
        args.obs_shape = env.observation_space.shape[1:]
        args.n_actions = env.action_space.n
        agent = create_agent("lstm_ppo", args, env)
        agent.create_optimizers()
        yield env, agent
        env.close()

    def test_create_vec_agent(self, vec_env_and_agent):
        from rlcade.agent.ppo import LstmPPO, LstmVecPPO

        _, agent = vec_env_and_agent
        assert isinstance(agent, LstmPPO)
        assert isinstance(agent.lstm_ppo, LstmVecPPO)

    def test_get_action_batched(self, vec_env_and_agent):
        env, agent = vec_env_and_agent
        obs, _ = env.reset()
        t = torch.as_tensor(obs, dtype=torch.float32)
        action, log_prob, value = agent.get_action(t)
        assert action.shape[0] == env.num_envs
        assert log_prob.shape[0] == env.num_envs
        assert value.shape[0] == env.num_envs

    def test_collect_rollout_vec(self, vec_env_and_agent):
        env, agent = vec_env_and_agent
        rollout, next_obs = agent.collect_rollout(env, num_steps=8)
        assert rollout["obs"].shape[0] == 8
        assert rollout["obs"].shape[1] == env.num_envs
        assert rollout["saved_hx"].shape[0] == 8
        assert next_obs.shape[0] == env.num_envs

    def test_learn_vec(self, vec_env_and_agent):
        env, agent = vec_env_and_agent
        rollout, _ = agent.collect_rollout(env, num_steps=16)
        metrics = agent.learn(rollout)
        assert "loss" in metrics
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
