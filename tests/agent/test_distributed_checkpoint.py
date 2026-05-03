"""Distributed checkpoint tests using real process groups (gloo + mp.spawn)."""

import os
import socket
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rlcade.checkpoint.checkpoint import Checkpoint


def _free_port():
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _init_pg(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


# DDP checkpoint workers (module-level for pickling)


def _ddp_save_load_worker(rank, world_size, port, ckpt_path, rom):
    """Each rank creates a PPO agent, wraps with DDPAgentWrapper, saves/loads."""
    _init_pg(rank, world_size, port)
    try:
        from rlcade.envs import create_env, register_envs
        from rlcade.agent import create_agent
        from rlcade.agent.base import DDPAgentWrapper
        from tests.conftest import make_args

        register_envs()
        args = make_args(rom)
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n
        agent = create_agent("ppo", args, env)
        agent.create_optimizers()

        # Sync model weights across ranks before wrapping
        for _, module in agent._impl.models():
            for p in module.parameters():
                dist.broadcast(p.data, src=0)

        wrapped = DDPAgentWrapper(agent)
        state = wrapped.state(step=42)
        if dist.get_rank() == 0:
            Checkpoint(ckpt_path).save(state)
        dist.barrier()

        # Only rank 0 actually writes the file
        assert os.path.exists(ckpt_path)
        with Checkpoint(ckpt_path).reader() as f:
            step = wrapped.load(f)
        assert step == 42

        env.close()
    finally:
        dist.destroy_process_group()


def _ddp_only_rank0_writes_worker(rank, world_size, port, ckpt_path, rom):
    """Verify that only rank 0 writes the checkpoint file."""
    _init_pg(rank, world_size, port)
    try:
        from rlcade.envs import create_env, register_envs
        from rlcade.agent import create_agent
        from rlcade.agent.base import DDPAgentWrapper
        from tests.conftest import make_args

        register_envs()
        args = make_args(rom)
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n
        agent = create_agent("ppo", args, env)
        agent.create_optimizers()

        for _, module in agent._impl.models():
            for p in module.parameters():
                dist.broadcast(p.data, src=0)

        wrapped = DDPAgentWrapper(agent)

        # Remove file if it exists before save
        if rank == 0 and os.path.exists(ckpt_path):
            os.unlink(ckpt_path)
        dist.barrier()

        state = wrapped.state(step=99)
        if dist.get_rank() == 0:
            Checkpoint(ckpt_path).save(state)
        dist.barrier()

        # After barrier, rank 0 should have written the file
        assert os.path.exists(ckpt_path)

        # Verify the checkpoint has the correct step
        if rank == 0:
            with Checkpoint(ckpt_path).reader() as f:
                loaded = torch.load(f, map_location="cpu", weights_only=True)
            assert loaded["step"] == 99

        env.close()
    finally:
        dist.destroy_process_group()


def _ddp_weights_consistent_worker(rank, world_size, port, ckpt_path, rom):
    """Verify weights are consistent across ranks after load."""
    _init_pg(rank, world_size, port)
    try:
        from rlcade.envs import create_env, register_envs
        from rlcade.agent import create_agent
        from rlcade.agent.base import DDPAgentWrapper
        from tests.conftest import make_args

        register_envs()
        args = make_args(rom)
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n
        agent = create_agent("ppo", args, env)
        agent.create_optimizers()

        for _, module in agent._impl.models():
            for p in module.parameters():
                dist.broadcast(p.data, src=0)

        wrapped = DDPAgentWrapper(agent)
        state = wrapped.state(step=10)
        if dist.get_rank() == 0:
            Checkpoint(ckpt_path).save(state)
        dist.barrier()

        # Create a fresh agent with random weights and load checkpoint
        agent2 = create_agent("ppo", args, env)
        wrapped2 = DDPAgentWrapper(agent2)
        with Checkpoint(ckpt_path).reader() as f:
            wrapped2.load(f)

        # Gather actor params from all ranks and verify they match
        for _, module in agent2._impl.models():
            for p in module.parameters():
                gathered = [torch.zeros_like(p.data) for _ in range(world_size)]
                dist.all_gather(gathered, p.data)
                for g in gathered[1:]:
                    assert torch.equal(gathered[0], g), "Weights differ across ranks"

        env.close()
    finally:
        dist.destroy_process_group()


def _ddp_dqn_save_load_worker(rank, world_size, port, ckpt_path, rom):
    """DDP checkpoint round-trip for DQN agent."""
    _init_pg(rank, world_size, port)
    try:
        from rlcade.envs import create_env, register_envs
        from rlcade.agent import create_agent
        from rlcade.agent.base import DDPAgentWrapper
        from tests.conftest import make_args

        register_envs()
        args = make_args(rom, agent="dqn", buffer_size=100)
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n
        agent = create_agent("dqn", args, env)
        agent.create_optimizers()

        for _, module in agent._impl.models():
            for p in module.parameters():
                dist.broadcast(p.data, src=0)

        wrapped = DDPAgentWrapper(agent)
        state = wrapped.state(step=200)
        if dist.get_rank() == 0:
            Checkpoint(ckpt_path).save(state)
        dist.barrier()

        with Checkpoint(ckpt_path).reader() as f:
            step = wrapped.load(f)
        assert step == 200

        env.close()
    finally:
        dist.destroy_process_group()


def _ddp_sac_save_load_worker(rank, world_size, port, ckpt_path, rom):
    """DDP checkpoint round-trip for SAC agent."""
    _init_pg(rank, world_size, port)
    try:
        from rlcade.envs import create_env, register_envs
        from rlcade.agent import create_agent
        from rlcade.agent.base import DDPAgentWrapper
        from tests.conftest import make_args

        register_envs()
        args = make_args(rom, agent="sac", buffer_size=100)
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n
        agent = create_agent("sac", args, env)
        agent.create_optimizers()

        for _, module in agent._impl.models():
            for p in module.parameters():
                dist.broadcast(p.data, src=0)

        wrapped = DDPAgentWrapper(agent)
        state = wrapped.state(step=300)
        if dist.get_rank() == 0:
            Checkpoint(ckpt_path).save(state)
        dist.barrier()

        with Checkpoint(ckpt_path).reader() as f:
            step = wrapped.load(f)
        assert step == 300

        env.close()
    finally:
        dist.destroy_process_group()


# wrap_agent integration worker


def _wrap_agent_integration_worker(rank, world_size, port, ckpt_path, rom):
    """Test wrap_agent('ddp') produces a working DDPAgentWrapper."""
    _init_pg(rank, world_size, port)
    try:
        from rlcade.envs import create_env, register_envs
        from rlcade.agent import create_agent
        from rlcade.agent.base import wrap_agent, DDPAgentWrapper
        from tests.conftest import make_args

        register_envs()
        args = make_args(rom)
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n
        agent = create_agent("ppo", args, env)

        wrapped = wrap_agent(agent, "ddp", distributed=True)
        assert isinstance(wrapped, DDPAgentWrapper)

        state = wrapped.state(step=7)
        if dist.get_rank() == 0:
            Checkpoint(ckpt_path).save(state)
        dist.barrier()
        with Checkpoint(ckpt_path).reader() as f:
            step = wrapped.load(f)
        assert step == 7

        env.close()
    finally:
        dist.destroy_process_group()


# Test class


class TestDDPCheckpoint:
    """DDP distributed checkpoint save/load with real gloo process group."""

    def _run(self, worker_fn, rom, nprocs=2):
        port = _free_port()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            mp.spawn(worker_fn, args=(nprocs, port, path, rom), nprocs=nprocs, join=True)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_ppo_save_and_load(self, rom):
        self._run(_ddp_save_load_worker, rom)

    def test_only_rank0_writes(self, rom):
        self._run(_ddp_only_rank0_writes_worker, rom)

    def test_weights_consistent_after_load(self, rom):
        self._run(_ddp_weights_consistent_worker, rom)

    def test_dqn_save_and_load(self, rom):
        self._run(_ddp_dqn_save_load_worker, rom)

    def test_sac_save_and_load(self, rom):
        self._run(_ddp_sac_save_load_worker, rom)

    def test_wrap_agent_integration(self, rom):
        self._run(_wrap_agent_integration_worker, rom)


def _ddp_target_broadcast_worker(rank, world_size, port, ckpt_path, rom):
    """Verify DDPAgentWrapper broadcasts target network params from rank 0."""
    _init_pg(rank, world_size, port)
    try:
        from rlcade.envs import create_env, register_envs
        from rlcade.agent import create_agent
        from rlcade.agent.base import DDPAgentWrapper
        from tests.conftest import make_args

        register_envs()
        args = make_args(rom, agent="dqn", buffer_size=100)
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n
        agent = create_agent("dqn", args, env)
        agent.create_optimizers()

        DDPAgentWrapper(agent)

        # Target network params should be identical across ranks
        for _, net in agent.target_models():
            for p in net.parameters():
                gathered = [torch.zeros_like(p.data) for _ in range(world_size)]
                dist.all_gather(gathered, p.data)
                for g in gathered[1:]:
                    assert torch.equal(gathered[0], g), "Target weights differ across ranks"

        env.close()
    finally:
        dist.destroy_process_group()


class TestDDPTargetBroadcast:
    """DDP target network broadcast correctness."""

    def _run(self, worker_fn, rom, nprocs=2):
        port = _free_port()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            mp.spawn(worker_fn, args=(nprocs, port, path, rom), nprocs=nprocs, join=True)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_dqn_target_broadcast(self, rom):
        self._run(_ddp_target_broadcast_worker, rom)


def _init_fsdp2_pg(rank, world_size, port):
    """Init NCCL process group and set CUDA device for FSDP2 workers."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def _cleanup_fsdp2_pg():
    dist.destroy_process_group()
    for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        os.environ.pop(k, None)


def _create_fsdp2_agent(rank, rom):
    """Create a PPO agent on cuda:<rank>, wrap with FSDP2, create optimizers."""
    from rlcade.envs import create_env, register_envs
    from rlcade.agent import create_agent
    from rlcade.agent.base import FSDP2AgentWrapper
    from tests.conftest import make_args

    register_envs()
    args = make_args(rom, device=f"cuda:{rank}")
    env = create_env(args)
    args.obs_shape = env.observation_space.shape
    args.n_actions = env.action_space.n
    agent = create_agent("ppo", args, env)
    wrapped = FSDP2AgentWrapper(agent)
    agent.create_optimizers()
    return agent, wrapped, env


def _run_dummy_optim_step(agent):
    """Run a forward+backward+step so optimizer state is populated."""
    obs = torch.randn(1, 4, 84, 84, device=agent.device)
    for _, module in agent.models():
        out = module(obs)
        # Actor may return a Categorical distribution; use .logits if so
        if hasattr(out, "logits"):
            out = out.logits
        elif isinstance(out, tuple):
            out = out[0]
        out.sum().backward()
    for _, optim, _ in agent._impl.optimizers():
        optim.step()
        optim.zero_grad()


def _snapshot_fsdp2_state(agent):
    """Gather full model + optimizer state dicts (collective — all ranks must call)."""
    from torch.distributed.checkpoint.state_dict import (
        get_model_state_dict,
        get_optimizer_state_dict,
        StateDictOptions,
    )

    opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
    model_state = {
        attr: {k: v.clone() for k, v in get_model_state_dict(m, options=opts).items()} for attr, m in agent.models()
    }
    optim_state = {
        key: get_optimizer_state_dict(model, optim, options=opts) for key, optim, model in agent._impl.optimizers()
    }
    return model_state, optim_state


def _assert_states_equal(pre_model, pre_optim, post_model, post_optim):
    """Compare model params and optimizer state tensors for exact equality."""
    for attr in pre_model:
        for k in pre_model[attr]:
            assert torch.equal(pre_model[attr][k], post_model[attr][k]), f"Model param mismatch: {attr}.{k}"
    for key in pre_optim:
        pre_s = pre_optim[key].get("state", {})
        post_s = post_optim[key].get("state", {})
        assert set(pre_s.keys()) == set(post_s.keys()), f"Optimizer state keys mismatch for {key}"
        for param_name in pre_s:
            for k, v in pre_s[param_name].items():
                if isinstance(v, torch.Tensor):
                    assert torch.equal(v, post_s[param_name][k]), f"Optimizer state mismatch: {key}.{param_name}.{k}"


def _fsdp2_state_roundtrip_worker(rank, world_size, port, ckpt_path, rom):
    """Verify model and optimizer state survives FSDP2 save→load round-trip."""
    _init_fsdp2_pg(rank, world_size, port)
    try:
        agent, wrapped, env = _create_fsdp2_agent(rank, rom)
        _run_dummy_optim_step(agent)
        pre_model, pre_optim = _snapshot_fsdp2_state(agent)

        state = wrapped.state(step=42)
        if dist.get_rank() == 0:
            Checkpoint(ckpt_path).save(state)
        dist.barrier()

        agent2, wrapped2, _ = _create_fsdp2_agent(rank, rom)
        with Checkpoint(ckpt_path).reader() as f:
            step = wrapped2.load(f)
        assert step == 42

        post_model, post_optim = _snapshot_fsdp2_state(agent2)

        if rank == 0:
            _assert_states_equal(pre_model, pre_optim, post_model, post_optim)

        env.close()
    finally:
        _cleanup_fsdp2_pg()


class TestFSDP2Checkpoint:
    """FSDP2 tests — require CUDA."""

    @pytest.fixture(autouse=True)
    def _skip_no_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("FSDP2 requires CUDA")

    def test_state_roundtrip(self, rom):
        port = _free_port()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            mp.spawn(
                _fsdp2_state_roundtrip_worker,
                args=(2, port, path, rom),
                nprocs=2,
                join=True,
            )
        finally:
            if os.path.exists(path):
                os.unlink(path)
