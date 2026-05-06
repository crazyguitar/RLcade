import os
import tempfile

import pytest
import torch

from tests.conftest import make_args


class TestRainbowDQNAgent:
    @pytest.fixture()
    def env_and_agent(self, rom):
        from rlcade.envs import create_env
        from rlcade.agent import create_agent

        args = make_args(rom, agent="rainbow_dqn", qnet="rainbow_qnet", buffer_size=1000)
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n
        agent = create_agent("rainbow_dqn", args, env)
        agent.create_optimizers()
        yield env, agent
        env.close()

    def test_create_agent(self, env_and_agent):
        from rlcade.agent.dqn import RainbowDQN

        _, agent = env_and_agent
        assert isinstance(agent, RainbowDQN)

    def test_get_action(self, env_and_agent):
        env, agent = env_and_agent
        obs, _ = env.reset()
        t = torch.as_tensor(obs, dtype=torch.float32)
        action = agent.get_action(t, deterministic=True)
        assert 0 <= action < env.action_space.n

    def test_get_action_deterministic(self, env_and_agent):
        env, agent = env_and_agent
        obs, _ = env.reset()
        t = torch.as_tensor(obs, dtype=torch.float32)
        # In eval mode (deterministic), NoisyNet uses mu only — should be deterministic
        a1 = agent.get_action(t, deterministic=True)
        a2 = agent.get_action(t, deterministic=True)
        assert a1 == a2

    def test_store_and_learn(self, env_and_agent):
        env, agent = env_and_agent
        agent.rainbow.learn_start = 16

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
        assert "loss" in metrics
        assert "q_mean" in metrics

    def test_evaluate(self, env_and_agent):
        env, agent = env_and_agent
        scores = agent.evaluate(env, num_episodes=1)
        assert len(scores) == 1
        assert isinstance(scores[0], float)

    def test_beta_property(self, env_and_agent):
        _, agent = env_and_agent
        assert agent.beta == agent.rainbow.beta_start
        agent.beta = 0.8
        assert agent.rainbow.beta == 0.8


class TestRainbowDQNCheckpoint:
    def test_save_and_load(self, rom):
        from rlcade.envs import create_env
        from rlcade.agent.dqn import RainbowDQN, RainbowDQNConfig

        args = make_args(rom, agent="rainbow_dqn", qnet="rainbow_qnet", buffer_size=1000)
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n
        agent = RainbowDQN(RainbowDQNConfig.from_args(args))
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
        from rlcade.agent.dqn import RainbowDQN, RainbowDQNConfig

        args = make_args(rom, agent="rainbow_dqn", qnet="rainbow_qnet", buffer_size=1000)
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            from rlcade.checkpoint.checkpoint import Checkpoint

            agent = RainbowDQN(RainbowDQNConfig.from_args(args), env)
            agent.create_optimizers()
            Checkpoint(path).save(agent.state(step=50))

            with Checkpoint(path).reader() as f:
                loaded = RainbowDQN.restore(RainbowDQNConfig.from_args(args), f, env)
            obs, _ = env.reset()
            action = loaded.get_action(torch.as_tensor(obs, dtype=torch.float32), deterministic=True)
            assert 0 <= action < env.action_space.n
        finally:
            os.unlink(path)
            env.close()


class TestRainbowDQNSafetensors:
    def test_save_and_load(self, rom):
        from rlcade.envs import create_env
        from rlcade.agent import create_agent
        from rlcade.checkpoint.safetensors import save_safetensors, load_safetensors

        args = make_args(rom, agent="rainbow_dqn", qnet="rainbow_qnet", buffer_size=1000)
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n
        agent = create_agent("rainbow_dqn", args, env)
        agent.create_optimizers()

        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        try:
            state = agent.state(step=123)
            save_safetensors(state, path, step=123)

            loaded, step = load_safetensors(path, device=torch.device("cpu"))
            assert step == 123
            for key, value in state.items():
                if isinstance(value, dict) and value and all(isinstance(v, torch.Tensor) for v in value.values()):
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

        args = make_args(rom, agent="rainbow_dqn", qnet="rainbow_qnet", buffer_size=1000)
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n

        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        try:
            source = create_agent("rainbow_dqn", args, env)
            source.create_optimizers()
            with torch.no_grad():
                for _, m in source._impl.models():
                    for p in m.parameters():
                        p.fill_(0.123)
            save_safetensors(source.state(step=7), path, step=7)

            args.checkpoint = path
            loaded = load_agent("rainbow_dqn", args, env)
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


class TestVecRainbowDQNAgent:
    @pytest.fixture()
    def vec_env_and_agent(self, rom):
        from rlcade.envs import create_vector_env
        from rlcade.agent import create_agent
        from tests.conftest import make_vec_args

        args = make_vec_args(rom, agent="rainbow_dqn", qnet="rainbow_qnet", buffer_size=1000)
        env = create_vector_env(args)
        args.obs_shape = env.observation_space.shape[1:]
        args.n_actions = env.action_space.n
        agent = create_agent("rainbow_dqn", args, env)
        agent.create_optimizers()
        yield env, agent
        env.close()

    def test_create_vec_agent(self, vec_env_and_agent):
        from rlcade.agent.dqn import RainbowDQN, VecRainbowDQN

        _, agent = vec_env_and_agent
        assert isinstance(agent, RainbowDQN)
        assert isinstance(agent.rainbow, VecRainbowDQN)

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
        # N-step buffer may not have flushed yet, so just check step_count
        assert agent.step_count == env.num_envs

    def test_evaluate_vec(self, vec_env_and_agent):
        env, agent = vec_env_and_agent
        scores = agent.evaluate(env, num_episodes=2)
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)


class TestNStepBuffer:
    def test_no_output_until_n_steps(self):
        from rlcade.agent.dqn import NStepBuffer

        buf = NStepBuffer(n_step=3, gamma=0.99)
        assert buf.append((0, 0, 1.0, 1, False)) is None
        assert buf.append((1, 0, 1.0, 2, False)) is None

    def test_returns_after_n_steps(self):
        from rlcade.agent.dqn import NStepBuffer

        buf = NStepBuffer(n_step=3, gamma=0.99)
        buf.append(("obs0", 0, 1.0, "next0", False))
        buf.append(("obs1", 1, 2.0, "next1", False))
        result = buf.append(("obs2", 2, 3.0, "next2", False))
        assert result is not None
        obs, action, reward, next_obs, done = result
        assert obs == "obs0"
        assert action == 0
        assert next_obs == "next2"
        assert not done
        assert abs(reward - (1.0 + 0.99 * 2.0 + 0.99**2 * 3.0)) < 1e-6

    def test_flush_drains_all_on_done(self):
        from rlcade.agent.dqn import NStepBuffer

        buf = NStepBuffer(n_step=3, gamma=0.99)
        buf.append(("obs0", 0, 1.0, "next0", False))
        buf.append(("obs1", 1, 2.0, "next1", False))
        result = buf.append(("obs2", 2, 3.0, "next2", True))
        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0][0] == "obs0"
        assert abs(result[0][2] - (1.0 + 0.99 * 2.0 + 0.99**2 * 3.0)) < 1e-6
        assert result[1][0] == "obs1"
        assert abs(result[1][2] - (2.0 + 0.99 * 3.0)) < 1e-6
        assert result[2][0] == "obs2"
        assert abs(result[2][2] - 3.0) < 1e-6
        assert all(r[3] == "next2" for r in result)
        assert all(r[4] is True for r in result)

    def test_buffer_empty_after_flush(self):
        from rlcade.agent.dqn import NStepBuffer

        buf = NStepBuffer(n_step=3, gamma=0.99)
        buf.append(("a", 0, 1.0, "b", False))
        buf.append(("b", 0, 1.0, "c", True))
        assert len(buf.buffer) == 0

    def test_single_step_done(self):
        from rlcade.agent.dqn import NStepBuffer

        buf = NStepBuffer(n_step=3, gamma=0.99)
        result = buf.append(("obs0", 0, 5.0, "next0", True))
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0][2] == 5.0
