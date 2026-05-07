"""Tests for agent.wrap() distributed model wrapping."""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from rlcade.agent.base import Agent, AgentWrapper, strip_wrapper_prefixes, unwrap_module, wrap_agent
from rlcade.graph import CUDAGraphWrapper
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


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))


def _init_single_rank_gloo():
    """Init a world_size=1 gloo group so DistributedDataParallel can wrap on CPU."""
    if dist.is_initialized():
        return False
    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    os.environ.update({"MASTER_ADDR": "localhost", "MASTER_PORT": str(port), "RANK": "0", "WORLD_SIZE": "1"})
    dist.init_process_group("gloo")
    return True


class TestUnwrapModule:
    """unwrap_module peels CUDAGraphWrapper / torch.compile / DDP layers."""

    def test_bare_module_returns_self(self):
        m = _Tiny()
        assert unwrap_module(m) is m

    def test_unwraps_cuda_graph_wrapper(self):
        m = _Tiny()
        assert unwrap_module(CUDAGraphWrapper(m)) is m

    def test_unwraps_torch_compile(self):
        m = _Tiny()
        assert unwrap_module(torch.compile(m)) is m

    def test_unwraps_full_chain_with_ddp(self):
        """Reproduces the resume-train path: CUDAGraphWrapper(compile(DDP(qnet)))."""
        from torch.nn.parallel import DistributedDataParallel as DDP

        owns_pg = _init_single_rank_gloo()
        try:
            m = _Tiny()
            chain = CUDAGraphWrapper(torch.compile(DDP(m)))
            assert unwrap_module(chain) is m
        finally:
            if owns_pg:
                dist.destroy_process_group()


class TestDDPWrapperLoadFullChain:
    """Regression: DDPAgentWrapper must load checkpoints saved with full wrap chain.

    Before the fix, ``_unwrapped`` only peeled one layer (CUDAGraphWrapper),
    leaving ``OptimizedModule(DDP(qnet))`` whose ``state_dict`` keeps the
    ``_orig_mod.module.`` prefix. ``strip_wrapper_prefixes`` would then strip
    the prefix off saved keys, producing a missing-keys error on load.
    """

    @pytest.mark.skipif(dist.is_initialized(), reason="needs to control its own pg")
    def test_load_after_compile_and_ddp_wrap(self):
        from torch.nn.parallel import DistributedDataParallel as DDP

        owns_pg = _init_single_rank_gloo()
        try:
            bare = _Tiny()
            saved = {f"_orig_mod.module.{k}": v.clone() for k, v in bare.state_dict().items()}

            target = _Tiny()
            wrapped = CUDAGraphWrapper(torch.compile(DDP(target)))
            unwrap_module(wrapped).load_state_dict(strip_wrapper_prefixes(saved))

            for k, v in bare.state_dict().items():
                assert torch.equal(target.state_dict()[k], v)
        finally:
            if owns_pg:
                dist.destroy_process_group()

    def test_unwrapped_covers_target_networks(self):
        """DDPAgentWrapper._unwrapped() must also unwrap target_models().

        Target nets aren't DDP-wrapped but compile() still puts them under
        CUDAGraphWrapper(torch.compile(...)). Skipping them leaves saved
        target keys mismatching after strip_wrapper_prefixes.
        """
        from rlcade.agent.base import Agent, DDPAgentWrapper

        bare_q = _Tiny()
        bare_t = _Tiny()

        class Impl(nn.Module):
            device = torch.device("cpu")

            def __init__(self):
                super().__init__()
                self.qnet = bare_q
                self.target = bare_t

            def models(self):
                return [("qnet", self.qnet)]

            def target_models(self):
                return [("target", self.target)]

            def state(self, step=0, *, staging=False):
                return {"qnet": self.qnet.state_dict(), "target": self.target.state_dict(), "step": step}

            def load(self, f):
                return 0

            def get_action(self, obs, *, deterministic=False):
                return 0

        impl = Impl()
        agent = Agent(impl)
        # Skip __init__ (would require process group); construct shell and seat _agent.
        wrapper = DDPAgentWrapper.__new__(DDPAgentWrapper)
        wrapper._agent = agent

        # Apply CUDAGraphWrapper(torch.compile(...)) to both qnet and target,
        # mirroring agent.compile() but without CUDA/DDP requirements.
        impl.qnet = CUDAGraphWrapper(torch.compile(bare_q))
        impl.target = CUDAGraphWrapper(torch.compile(bare_t))

        with wrapper._unwrapped():
            assert impl.qnet is bare_q
            assert impl.target is bare_t

        # Restoration after exit
        assert isinstance(impl.qnet, CUDAGraphWrapper)
        assert isinstance(impl.target, CUDAGraphWrapper)
