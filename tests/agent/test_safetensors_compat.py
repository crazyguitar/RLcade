"""Safetensors export compatibility across local / DDP / FSDP2 training.

Mirrors test_checkpoint_compat.py for the SafetensorsExportPlugin export path.

Test matrix:
  Trainer (local) export → inference load (weight-equal)
  Trainer (DDP)   export → inference load (weight-equal, no `module.` prefix)
  Trainer (FSDP2) export → inference load (weight-equal)  -- CUDA only
"""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rlcade.checkpoint.safetensors import load_safetensors
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


# Train + export workers (module-level for pickling)


def _local_export_worker(rank, world_size, port, ckpt_path, st_path, rom):
    from rlcade.envs import register_envs
    from rlcade.training import create_trainer
    from rlcade.plugins.checkpoint import CheckpointPlugin
    from rlcade.plugins.safetensors_export import SafetensorsExportPlugin

    register_envs()
    args = make_args(rom, checkpoint_path=ckpt_path, max_iterations=2)
    plugins = [
        CheckpointPlugin(checkpoint_path=ckpt_path, checkpoint_interval=1, num_steps=args.num_steps),
        SafetensorsExportPlugin(safetensors_path=st_path),
    ]
    trainer = create_trainer("ppo", args, plugins=plugins)
    trainer.train()
    trainer.env.close()


def _ddp_export_worker(rank, world_size, port, ckpt_path, st_path, rom):
    _set_dist_env(rank, world_size, port)
    try:
        from rlcade.envs import register_envs
        from rlcade.training import create_trainer
        from rlcade.plugins.checkpoint import CheckpointPlugin
        from rlcade.plugins.safetensors_export import SafetensorsExportPlugin

        register_envs()
        args = make_args(
            rom,
            checkpoint_path=ckpt_path,
            max_iterations=2,
            distributed="ddp",
            backend="gloo",
            stage=None,
        )
        plugins = [
            CheckpointPlugin(checkpoint_path=ckpt_path, checkpoint_interval=1, num_steps=args.num_steps),
            SafetensorsExportPlugin(safetensors_path=st_path),
        ]
        trainer = create_trainer("ppo", args, plugins=plugins)
        trainer.train()
        trainer.env.close()
    finally:
        _cleanup_dist_env()


def _fsdp2_export_worker(rank, world_size, port, ckpt_path, st_path, rom):
    _set_dist_env(rank, world_size, port)
    try:
        from rlcade.envs import register_envs
        from rlcade.training import create_trainer
        from rlcade.plugins.checkpoint import CheckpointPlugin
        from rlcade.plugins.safetensors_export import SafetensorsExportPlugin

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
        plugins = [
            CheckpointPlugin(checkpoint_path=ckpt_path, checkpoint_interval=1, num_steps=args.num_steps),
            SafetensorsExportPlugin(safetensors_path=st_path),
        ]
        trainer = create_trainer("ppo", args, plugins=plugins)
        trainer.train()
        trainer.env.close()
    finally:
        _cleanup_dist_env()


# Convenience wrappers


def _local_export(rom, ckpt_path, st_path):
    _spawn(_local_export_worker, 1, ckpt_path, st_path, rom)


def _ddp_export(rom, ckpt_path, st_path, nprocs=2):
    _spawn(_ddp_export_worker, nprocs, ckpt_path, st_path, rom)


def _fsdp2_export(rom, ckpt_path, st_path, nprocs=2):
    _spawn(_fsdp2_export_worker, nprocs, ckpt_path, st_path, rom)


# Assertions


def _assert_clean_safetensors(st_path):
    """Flattened keys must not start with `module.` -- DDP wrap should be stripped."""
    state, _ = load_safetensors(st_path, device=torch.device("cpu"))
    for model_name, sd in state.items():
        for k in sd:
            assert not k.startswith("module."), f"DDP module. prefix leaked: {model_name}.{k}"


def _assert_inference_weights_match_safetensors(rom, st_path):
    """Load via inference path; verify each model's state_dict matches the safetensors file."""
    from rlcade.envs import create_env, register_envs
    from rlcade.agent import load_agent

    register_envs()
    expected, _ = load_safetensors(st_path, device=torch.device("cpu"))

    args = make_args(rom, checkpoint=st_path)
    env = create_env(args)
    args.obs_shape = env.observation_space.shape
    args.n_actions = env.action_space.n
    agent = load_agent("ppo", args, env)

    for attr, module in agent._impl.models():
        assert attr in expected, f"Model {attr!r} missing from safetensors export"
        actual_sd = module.state_dict()
        for k, v in expected[attr].items():
            assert torch.equal(v, actual_sd[k].cpu()), f"Weight mismatch: {attr}.{k}"

    obs, _ = env.reset()
    action = agent.get_action(torch.as_tensor(obs, dtype=torch.float32))
    assert action is not None
    env.close()


# Tests: local + DDP


class TestLocalDDPSafetensorsCompat:
    def test_local_export_inference_load(self, rom, tmp_path):
        ckpt = str(tmp_path / "ckpt.pt")
        st = str(tmp_path / "model.safetensors")
        _local_export(rom, ckpt, st)
        assert os.path.exists(st)
        _assert_clean_safetensors(st)
        _assert_inference_weights_match_safetensors(rom, st)

    def test_ddp_export_inference_load(self, rom, tmp_path):
        ckpt = str(tmp_path / "ckpt.pt")
        st = str(tmp_path / "model.safetensors")
        _ddp_export(rom, ckpt, st)
        assert os.path.exists(st)
        _assert_clean_safetensors(st)
        _assert_inference_weights_match_safetensors(rom, st)


# Tests: FSDP2 (CUDA only)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FSDP2 requires CUDA")
class TestFSDP2SafetensorsCompat:
    def test_fsdp2_export_inference_load(self, rom, tmp_path):
        ckpt = str(tmp_path / "ckpt.pt")
        st = str(tmp_path / "model.safetensors")
        _fsdp2_export(rom, ckpt, st)
        assert os.path.exists(st)
        _assert_clean_safetensors(st)
        _assert_inference_weights_match_safetensors(rom, st)
