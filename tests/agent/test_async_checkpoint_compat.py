"""Async checkpoint compatibility: any training backend (local/DDP/FSDP2) using
AsyncCheckpointPlugin produces checkpoints loadable by any other backend and by
inference.

Mirrors test_checkpoint_compat.py one-to-one, but the saving side uses
AsyncCheckpointPlugin (ThreadPoolExecutor-backed writes).

Test matrix:
  Trainer (local) save  → Trainer (local) resume
  Trainer (local) save  → Trainer (DDP) resume
  Trainer (DDP) save    → Trainer (local) resume
  Trainer (DDP) save    → Trainer (DDP) resume
  Trainer (local) save  → inference load
  Trainer (DDP) save    → inference load

FSDP2 variants require CUDA and are skipped on CPU-only machines.
"""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rlcade.checkpoint.checkpoint import Checkpoint
from tests.conftest import make_args

# Distributed helpers


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _set_dist_env(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)


def _cleanup_dist_env():
    if dist.is_initialized():
        dist.destroy_process_group()
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        os.environ.pop(key, None)


def _spawn(worker, nprocs, *args):
    port = _free_port()
    mp.spawn(worker, args=(nprocs, port, *args), nprocs=nprocs, join=True)


# Agent / checkpoint helpers


def _create_fsdp2_agent(rank, rom):
    from rlcade.envs import create_env, register_envs
    from rlcade.agent import create_agent
    from rlcade.agent.base import FSDP2AgentWrapper

    register_envs()
    torch.cuda.set_device(rank)
    args = make_args(rom, device=f"cuda:{rank}")
    env = create_env(args)
    args.obs_shape = env.observation_space.shape
    args.n_actions = env.action_space.n
    agent = create_agent("ppo", args, env)
    wrapped = FSDP2AgentWrapper(agent)
    agent.create_optimizers()
    return agent, wrapped, env


def _assert_checkpoint_weights_match(expected_state, agent, *, fsdp2=False):
    if fsdp2:
        from torch.distributed.checkpoint.state_dict import (
            get_model_state_dict,
            StateDictOptions,
        )

        opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
        rank = dist.get_rank()
        for attr, module in agent.models():
            loaded = get_model_state_dict(module, options=opts)
            if rank == 0:
                for k in expected_state[attr]:
                    assert torch.equal(expected_state[attr][k], loaded[k]), f"Weight mismatch: {attr}.{k}"
    else:
        for attr, module in agent._impl.models():
            for k, v in module.state_dict().items():
                assert torch.equal(expected_state[attr][k], v), f"Weight mismatch: {attr}.{k}"


def _assert_clean_checkpoint(ckpt_path):
    with Checkpoint(ckpt_path).reader() as f:
        state = torch.load(f, map_location="cpu", weights_only=True)
    for key, val in state.items():
        if isinstance(val, dict):
            for k in val:
                assert not k.startswith("module."), f"DDP prefix leaked: {key}.{k}"


def _assert_inference_loads(rom, ckpt_path):
    from rlcade.envs import create_env, register_envs
    from rlcade.agent import load_agent

    register_envs()
    args = make_args(rom, checkpoint=ckpt_path)
    env = create_env(args)
    args.obs_shape = env.observation_space.shape
    args.n_actions = env.action_space.n
    agent = load_agent("ppo", args, env)
    obs, _ = env.reset()
    action = agent.get_action(torch.as_tensor(obs, dtype=torch.float32))
    assert action is not None
    env.close()


# Train / resume workers (module-level for pickling)


def _local_train_worker(rank, world_size, port, ckpt_path, rom):
    from rlcade.envs import register_envs
    from rlcade.training import create_trainer
    from rlcade.plugins.async_checkpoint import AsyncCheckpointPlugin

    register_envs()
    args = make_args(rom, checkpoint_path=ckpt_path, max_iterations=2)
    plugins = [AsyncCheckpointPlugin(checkpoint_path=ckpt_path, checkpoint_interval=1, num_steps=args.num_steps)]
    trainer = create_trainer("ppo", args, plugins=plugins)
    trainer.train()
    trainer.env.close()


def _ddp_train_worker(rank, world_size, port, ckpt_path, rom):
    _set_dist_env(rank, world_size, port)
    try:
        from rlcade.envs import register_envs
        from rlcade.training import create_trainer
        from rlcade.plugins.async_checkpoint import AsyncCheckpointPlugin

        register_envs()
        args = make_args(
            rom,
            checkpoint_path=ckpt_path,
            max_iterations=2,
            distributed="ddp",
            backend="gloo",
            stage=None,
        )
        plugins = [AsyncCheckpointPlugin(checkpoint_path=ckpt_path, checkpoint_interval=1, num_steps=args.num_steps)]
        trainer = create_trainer("ppo", args, plugins=plugins)
        trainer.train()
        trainer.env.close()
    finally:
        _cleanup_dist_env()


def _ddp_resume_worker(rank, world_size, port, ckpt_path, rom):
    _set_dist_env(rank, world_size, port)
    try:
        from rlcade.envs import register_envs
        from rlcade.training import create_trainer
        from rlcade.plugins.async_checkpoint import AsyncCheckpointPlugin

        register_envs()
        args = make_args(
            rom,
            checkpoint_path=ckpt_path,
            max_iterations=4,
            distributed="ddp",
            backend="gloo",
            stage=None,
        )
        plugins = [AsyncCheckpointPlugin(checkpoint_path=ckpt_path, checkpoint_interval=1, num_steps=args.num_steps)]
        trainer = create_trainer("ppo", args, plugins=plugins)
        trainer.train()
        assert trainer.start_iteration > 0, f"Expected resume, got start_iteration={trainer.start_iteration}"
        trainer.env.close()
    finally:
        _cleanup_dist_env()


def _fsdp2_train_worker(rank, world_size, port, ckpt_path, rom):
    _set_dist_env(rank, world_size, port)
    try:
        from rlcade.envs import register_envs
        from rlcade.training import create_trainer
        from rlcade.plugins.async_checkpoint import AsyncCheckpointPlugin

        register_envs()
        args = make_args(
            rom,
            checkpoint_path=ckpt_path,
            max_iterations=2,
            distributed="fsdp2",
            backend="nccl",
            stage=None,
            device="auto",
        )
        plugins = [AsyncCheckpointPlugin(checkpoint_path=ckpt_path, checkpoint_interval=1, num_steps=args.num_steps)]
        trainer = create_trainer("ppo", args, plugins=plugins)
        trainer.train()
        trainer.env.close()
    finally:
        _cleanup_dist_env()


def _fsdp2_resume_worker(rank, world_size, port, ckpt_path, rom):
    _set_dist_env(rank, world_size, port)
    try:
        from rlcade.envs import register_envs
        from rlcade.training import create_trainer
        from rlcade.plugins.async_checkpoint import AsyncCheckpointPlugin

        register_envs()
        args = make_args(
            rom,
            checkpoint_path=ckpt_path,
            max_iterations=4,
            distributed="fsdp2",
            backend="nccl",
            stage=None,
            device="auto",
        )
        plugins = [AsyncCheckpointPlugin(checkpoint_path=ckpt_path, checkpoint_interval=1, num_steps=args.num_steps)]
        trainer = create_trainer("ppo", args, plugins=plugins)
        trainer.train()
        assert trainer.start_iteration > 0, f"Expected resume, got start_iteration={trainer.start_iteration}"
        trainer.env.close()
    finally:
        _cleanup_dist_env()


def _fsdp2_load_state_equality_worker(rank, world_size, port, ckpt_path, rom):
    _set_dist_env(rank, world_size, port)
    try:
        from rlcade.checkpoint.checkpoint import Checkpoint

        with Checkpoint(ckpt_path).reader() as f:
            expected = torch.load(f, map_location="cpu", weights_only=True)
        agent, wrapped, env = _create_fsdp2_agent(rank, rom)
        with Checkpoint(ckpt_path).reader() as f:
            step = wrapped.load(f)
        assert step == expected.get("step", 0)
        _assert_checkpoint_weights_match(expected, agent, fsdp2=True)
        env.close()
    finally:
        _cleanup_dist_env()


# Convenience wrappers


def _local_train(rom, ckpt_path):
    _spawn(_local_train_worker, 1, ckpt_path, rom)


def _ddp_train(rom, ckpt_path, nprocs=2):
    _spawn(_ddp_train_worker, nprocs, ckpt_path, rom)


def _ddp_resume(rom, ckpt_path, nprocs=2):
    _spawn(_ddp_resume_worker, nprocs, ckpt_path, rom)


def _fsdp2_train(rom, ckpt_path, nprocs=2):
    _spawn(_fsdp2_train_worker, nprocs, ckpt_path, rom)


def _fsdp2_resume(rom, ckpt_path, nprocs=2):
    _spawn(_fsdp2_resume_worker, nprocs, ckpt_path, rom)


def _fsdp2_assert_weights(ckpt_path, rom, nprocs=2):
    _spawn(_fsdp2_load_state_equality_worker, nprocs, ckpt_path, rom)


# Tests: local ↔ DDP (async save)


class TestLocalDDPAsyncCompat:
    """Async checkpoint compatibility between local and DDP training."""

    def test_local_save_local_resume(self, rom, tmp_path):
        ckpt = str(tmp_path / "ckpt.pt")
        _local_train(rom, ckpt)
        assert os.path.exists(ckpt)
        _assert_clean_checkpoint(ckpt)

        from rlcade.training import create_trainer
        from rlcade.plugins.async_checkpoint import AsyncCheckpointPlugin

        args = make_args(rom, checkpoint_path=ckpt, max_iterations=4)
        plugins = [AsyncCheckpointPlugin(checkpoint_path=ckpt, checkpoint_interval=1, num_steps=args.num_steps)]
        trainer = create_trainer("ppo", args, plugins=plugins)
        trainer.train()
        assert trainer.start_iteration > 0
        trainer.env.close()

    def test_local_save_ddp_resume(self, rom, tmp_path):
        ckpt = str(tmp_path / "ckpt.pt")
        _local_train(rom, ckpt)
        _assert_clean_checkpoint(ckpt)
        _ddp_resume(rom, ckpt)

    def test_ddp_save_local_resume(self, rom, tmp_path):
        ckpt = str(tmp_path / "ckpt.pt")
        _ddp_train(rom, ckpt)
        _assert_clean_checkpoint(ckpt)

        from rlcade.training import create_trainer
        from rlcade.plugins.async_checkpoint import AsyncCheckpointPlugin

        args = make_args(rom, checkpoint_path=ckpt, max_iterations=4)
        plugins = [AsyncCheckpointPlugin(checkpoint_path=ckpt, checkpoint_interval=1, num_steps=args.num_steps)]
        trainer = create_trainer("ppo", args, plugins=plugins)
        trainer.train()
        assert trainer.start_iteration > 0
        trainer.env.close()

    def test_ddp_save_ddp_resume(self, rom, tmp_path):
        ckpt = str(tmp_path / "ckpt.pt")
        _ddp_train(rom, ckpt)
        _assert_clean_checkpoint(ckpt)
        _ddp_resume(rom, ckpt)

    def test_local_save_inference_load(self, rom, tmp_path):
        ckpt = str(tmp_path / "ckpt.pt")
        _local_train(rom, ckpt)
        _assert_inference_loads(rom, ckpt)

    def test_ddp_save_inference_load(self, rom, tmp_path):
        ckpt = str(tmp_path / "ckpt.pt")
        _ddp_train(rom, ckpt)
        _assert_inference_loads(rom, ckpt)


# Tests: FSDP2 ↔ everything (async save)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="FSDP2 requires at least 2 CUDA devices",
)
class TestFSDP2AsyncCompat:
    """Async checkpoint compatibility involving FSDP2 (CUDA only)."""

    def test_fsdp2_save_fsdp2_resume(self, rom, tmp_path):
        ckpt = str(tmp_path / "ckpt.pt")
        _fsdp2_train(rom, ckpt)
        _assert_clean_checkpoint(ckpt)
        _fsdp2_assert_weights(ckpt, rom)
        _fsdp2_resume(rom, ckpt)

    def test_fsdp2_save_local_resume(self, rom, tmp_path):
        ckpt = str(tmp_path / "ckpt.pt")
        _fsdp2_train(rom, ckpt)
        _assert_clean_checkpoint(ckpt)

        from rlcade.envs import create_env, register_envs
        from rlcade.agent import create_agent
        from rlcade.training import create_trainer
        from rlcade.plugins.async_checkpoint import AsyncCheckpointPlugin

        register_envs()
        with Checkpoint(ckpt).reader() as f:
            expected = torch.load(f, map_location="cpu", weights_only=True)
        args = make_args(rom)
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n
        agent = create_agent("ppo", args, env)
        with Checkpoint(ckpt).reader() as f:
            agent.load(f)
        _assert_checkpoint_weights_match(expected, agent)
        env.close()

        args = make_args(rom, checkpoint_path=ckpt, max_iterations=4)
        plugins = [AsyncCheckpointPlugin(checkpoint_path=ckpt, checkpoint_interval=1, num_steps=args.num_steps)]
        trainer = create_trainer("ppo", args, plugins=plugins)
        trainer.train()
        assert trainer.start_iteration > 0
        trainer.env.close()

    def test_fsdp2_save_ddp_resume(self, rom, tmp_path):
        ckpt = str(tmp_path / "ckpt.pt")
        _fsdp2_train(rom, ckpt)
        _assert_clean_checkpoint(ckpt)
        _ddp_resume(rom, ckpt)

    def test_local_save_fsdp2_resume(self, rom, tmp_path):
        ckpt = str(tmp_path / "ckpt.pt")
        _local_train(rom, ckpt)
        _fsdp2_assert_weights(ckpt, rom)
        _fsdp2_resume(rom, ckpt)

    def test_ddp_save_fsdp2_resume(self, rom, tmp_path):
        ckpt = str(tmp_path / "ckpt.pt")
        _ddp_train(rom, ckpt)
        _fsdp2_assert_weights(ckpt, rom)
        _fsdp2_resume(rom, ckpt)

    def test_fsdp2_save_inference_load(self, rom, tmp_path):
        ckpt = str(tmp_path / "ckpt.pt")
        _fsdp2_train(rom, ckpt)
        _assert_inference_loads(rom, ckpt)
