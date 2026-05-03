"""Tests for agent.wrap() distributed model wrapping."""

import torch
import torch.nn as nn

from rlcade.agent.base import Agent, AgentWrapper, wrap_agent
from tests.conftest import make_args


class TestAgentWrapperState:
    def test_agent_wrapper_state_delegates(self):
        """AgentWrapper.state() should delegate to the underlying agent."""

        class FakeImpl:
            device = torch.device("cpu")

            def state(self, step=0, *, staging=False):
                return {"actor": {"w": 1}, "step": step}

            def load(self, state):
                return state.get("step", 0)

            def models(self):
                return []

            def target_models(self):
                return []

            def get_action(self, obs, *, deterministic=False):
                return 0

        agent = Agent(FakeImpl())
        wrapper = AgentWrapper(agent)
        s = wrapper.state(step=42)
        assert s == {"actor": {"w": 1}, "step": 42}


class TestModels:
    """Each agent returns correct trainable models."""

    def test_ppo_models(self, rom):
        from rlcade.agent import create_agent
        from rlcade.envs import create_env

        args = make_args(rom)
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n
        agent = create_agent("ppo", args, env)

        models = agent._impl.models()
        names = [name for name, _ in models]
        assert "actor" in names
        assert "critic" in names
        for _, m in models:
            assert isinstance(m, nn.Module)
        env.close()

    def test_ppo_models_with_icm(self, rom):
        from rlcade.agent import create_agent
        from rlcade.envs import create_env

        args = make_args(rom, icm=True, icm_coef=1.0, icm_feature_dim=256)
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n
        agent = create_agent("ppo", args, env)

        names = [name for name, _ in agent._impl.models()]
        assert "icm" in names
        env.close()

    def test_dqn_models(self, rom):
        from rlcade.agent import create_agent
        from rlcade.envs import create_env

        args = make_args(rom, agent="dqn")
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n
        agent = create_agent("dqn", args, env)

        models = agent._impl.models()
        names = [name for name, _ in models]
        assert names == ["qnet"]
        env.close()

    def test_sac_models(self, rom):
        from rlcade.agent import create_agent
        from rlcade.envs import create_env

        args = make_args(rom, agent="sac")
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n
        agent = create_agent("sac", args, env)

        models = agent._impl.models()
        names = [name for name, _ in models]
        assert "actor" in names
        assert "q1" in names
        assert "q2" in names
        env.close()

    def test_lstm_ppo_models(self, rom):
        from rlcade.agent import create_agent
        from rlcade.envs import create_env

        args = make_args(rom, agent="lstm_ppo")
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n
        agent = create_agent("lstm_ppo", args, env)

        names = [name for name, _ in agent._impl.models()]
        assert "model" in names
        env.close()


class TestWrap:
    """wrap_agent with None or invalid name returns the agent unchanged."""

    def test_wrap_none_is_noop(self, rom):
        from rlcade.agent import create_agent
        from rlcade.envs import create_env

        args = make_args(rom)
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n
        agent = create_agent("ppo", args, env)

        wrapped = wrap_agent(agent, None, False)
        assert wrapped is agent
        env.close()

    def test_wrap_invalid_name_is_noop(self, rom):
        from rlcade.agent import create_agent
        from rlcade.envs import create_env

        args = make_args(rom)
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n
        agent = create_agent("ppo", args, env)

        wrapped = wrap_agent(agent, "nonexistent", True)
        assert wrapped is agent
        env.close()


class TestTargetModels:
    """Each agent returns correct target networks."""

    def test_dqn_target_models(self, rom):
        from rlcade.agent import create_agent
        from rlcade.envs import create_env

        args = make_args(rom, agent="dqn", buffer_size=100)
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n
        agent = create_agent("dqn", args, env)

        targets = agent._impl.target_models()
        names = [name for name, _ in targets]
        assert names == ["target"]
        for _, m in targets:
            assert isinstance(m, nn.Module)
        env.close()

    def test_sac_target_models(self, rom):
        from rlcade.agent import create_agent
        from rlcade.envs import create_env

        args = make_args(rom, agent="sac", buffer_size=100)
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n
        agent = create_agent("sac", args, env)

        targets = agent._impl.target_models()
        names = [name for name, _ in targets]
        assert "q1_target" in names
        assert "q2_target" in names
        env.close()

    def test_ppo_has_no_target_models(self, rom):
        from rlcade.agent import create_agent
        from rlcade.envs import create_env

        args = make_args(rom)
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n
        agent = create_agent("ppo", args, env)

        assert agent._impl.target_models() == []
        env.close()

    def test_rainbow_dqn_target_models(self, rom):
        from rlcade.agent import create_agent
        from rlcade.envs import create_env

        args = make_args(rom, agent="rainbow_dqn", buffer_size=100)
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n
        agent = create_agent("rainbow_dqn", args, env)

        targets = agent._impl.target_models()
        assert [n for n, _ in targets] == ["target"]
        env.close()


class TestAgentFacade:
    """Agent facade delegates models/target_models/load_non_model_state."""

    def test_facade_models(self, rom):
        from rlcade.agent import create_agent
        from rlcade.envs import create_env

        args = make_args(rom, agent="dqn", buffer_size=100)
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n
        agent = create_agent("dqn", args, env)

        assert agent.models() == agent._impl.models()
        env.close()

    def test_facade_target_models(self, rom):
        from rlcade.agent import create_agent
        from rlcade.envs import create_env

        args = make_args(rom, agent="sac", buffer_size=100)
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n
        agent = create_agent("sac", args, env)

        assert agent.target_models() == agent._impl.target_models()
        env.close()


class TestLoadNonModelState:
    """load_non_model_state loads scalars without touching model weights."""

    def test_dqn_load_non_model_state(self, rom):
        import torch
        from rlcade.agent import create_agent
        from rlcade.envs import create_env

        args = make_args(rom, agent="dqn", buffer_size=100)
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n
        agent = create_agent("dqn", args, env)
        agent.create_optimizers()

        # Snapshot model weights before
        qnet_before = {k: v.clone() for k, v in agent._impl.qnet.state_dict().items()}

        state = {"step": 42, "step_count": 100, "qnet": torch.zeros(1), "target": torch.zeros(1)}
        step = agent._impl.load_non_model_state(state)

        assert step == 42
        assert agent._impl.step_count == 100
        # Model weights should be unchanged
        for k, v in agent._impl.qnet.state_dict().items():
            assert torch.equal(v, qnet_before[k])
        env.close()

    def test_ppo_load_non_model_state(self, rom):
        import torch
        from rlcade.agent import create_agent
        from rlcade.envs import create_env

        args = make_args(rom)
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n
        agent = create_agent("ppo", args, env)
        agent.create_optimizers()

        actor_before = {k: v.clone() for k, v in agent._impl.actor.state_dict().items()}

        state = {"step": 10}
        step = agent._impl.load_non_model_state(state)

        assert step == 10
        for k, v in agent._impl.actor.state_dict().items():
            assert torch.equal(v, actor_before[k])
        env.close()

    def test_sac_load_non_model_state(self, rom):
        import torch
        from rlcade.agent import create_agent
        from rlcade.envs import create_env

        args = make_args(rom, agent="sac", buffer_size=100)
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n
        agent = create_agent("sac", args, env)
        agent.create_optimizers()

        actor_before = {k: v.clone() for k, v in agent._impl.actor.state_dict().items()}

        state = agent._impl.state(step=55)
        step = agent._impl.load_non_model_state(state)

        assert step == 55
        # Actor weights should be unchanged (load_non_model_state doesn't touch models)
        for k, v in agent._impl.actor.state_dict().items():
            assert torch.equal(v, actor_before[k])
        env.close()
