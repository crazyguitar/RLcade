"""Tests for torch.compile and CUDA graph support (--eager flag)."""

import pytest
import torch

from tests.conftest import make_args
from rlcade.graph import CUDAGraphWrapper

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="torch.compile requires CUDA")


def _make_agent(rom, agent_name, **extra):
    from rlcade.envs import create_env
    from rlcade.agent import create_agent

    args = make_args(rom, agent=agent_name, device="cuda", **extra)
    env = create_env(args)
    args.obs_shape = env.observation_space.shape
    args.n_actions = env.action_space.n
    agent = create_agent(agent_name, args, env)
    return env, agent


@requires_cuda
class TestCompile:
    def test_ppo_compile(self, rom):
        env, agent = _make_agent(rom, "ppo")
        agent.compile()
        ppo = agent._impl
        assert isinstance(ppo.actor, CUDAGraphWrapper)
        assert isinstance(ppo.critic, CUDAGraphWrapper)
        assert isinstance(ppo.actor.module, torch._dynamo.eval_frame.OptimizedModule)
        assert isinstance(ppo.critic.module, torch._dynamo.eval_frame.OptimizedModule)
        env.close()

    def test_dqn_compile(self, rom):
        env, agent = _make_agent(rom, "dqn")
        agent.compile()
        dqn = agent._impl
        assert isinstance(dqn.qnet, CUDAGraphWrapper)
        assert isinstance(dqn.target, CUDAGraphWrapper)
        assert isinstance(dqn.qnet.module, torch._dynamo.eval_frame.OptimizedModule)
        assert isinstance(dqn.target.module, torch._dynamo.eval_frame.OptimizedModule)
        env.close()

    def test_sac_compile(self, rom):
        env, agent = _make_agent(rom, "sac")
        agent.compile()
        sac = agent._impl
        assert isinstance(sac.actor, CUDAGraphWrapper)
        assert isinstance(sac.q1, CUDAGraphWrapper)
        assert isinstance(sac.q2, CUDAGraphWrapper)
        assert isinstance(sac.q1_target, CUDAGraphWrapper)
        assert isinstance(sac.q2_target, CUDAGraphWrapper)
        assert isinstance(sac.actor.module, torch._dynamo.eval_frame.OptimizedModule)
        assert isinstance(sac.q1.module, torch._dynamo.eval_frame.OptimizedModule)
        assert isinstance(sac.q2.module, torch._dynamo.eval_frame.OptimizedModule)
        assert isinstance(sac.q1_target.module, torch._dynamo.eval_frame.OptimizedModule)
        assert isinstance(sac.q2_target.module, torch._dynamo.eval_frame.OptimizedModule)
        env.close()

    def test_eager_skips_compile(self, rom):
        env, agent = _make_agent(rom, "ppo")
        agent.compile(eager=True)
        ppo = agent._impl
        assert not isinstance(ppo.actor, CUDAGraphWrapper)
        assert not isinstance(ppo.critic, CUDAGraphWrapper)
        assert not isinstance(ppo.actor, torch._dynamo.eval_frame.OptimizedModule)
        assert not isinstance(ppo.critic, torch._dynamo.eval_frame.OptimizedModule)
        env.close()

    def test_cpu_skips_compile(self, rom):
        from rlcade.envs import create_env
        from rlcade.agent import create_agent

        args = make_args(rom, agent="ppo", device="cpu")
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n
        agent = create_agent("ppo", args, env)
        agent.compile()
        ppo = agent._impl
        assert not isinstance(ppo.actor, CUDAGraphWrapper)
        assert not isinstance(ppo.critic, CUDAGraphWrapper)
        assert not isinstance(ppo.actor, torch._dynamo.eval_frame.OptimizedModule)
        assert not isinstance(ppo.critic, torch._dynamo.eval_frame.OptimizedModule)
        env.close()


class TestCUDAGraphWrapper:
    def test_wrapper_initial_state(self):
        model = torch.nn.Linear(4, 2)
        wrapped = CUDAGraphWrapper(model, warmup_steps=2)
        assert wrapped._graph is None
        assert wrapped._warmup_steps == 2
