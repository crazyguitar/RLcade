"""SafetensorsExportPlugin -- real trainer integration. No fakes, no mocks."""

from __future__ import annotations

import os
import socket

import torch
import torch.multiprocessing as mp

from rlcade.checkpoint.safetensors import load_safetensors
from tests.conftest import make_args


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _local_train_worker(rank, world_size, port, ckpt_path, st_path, rom):
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


def _local_train_no_safetensors_worker(rank, world_size, port, ckpt_path, rom):
    from rlcade.envs import register_envs
    from rlcade.training import create_trainer
    from rlcade.plugins.checkpoint import CheckpointPlugin
    from rlcade.plugins.safetensors_export import SafetensorsExportPlugin

    register_envs()
    args = make_args(rom, checkpoint_path=ckpt_path, max_iterations=2)
    plugins = [
        CheckpointPlugin(checkpoint_path=ckpt_path, checkpoint_interval=1, num_steps=args.num_steps),
        SafetensorsExportPlugin(safetensors_path=""),  # disabled
    ]
    trainer = create_trainer("ppo", args, plugins=plugins)
    trainer.train()
    trainer.env.close()


def _spawn(worker, *args):
    port = _free_port()
    mp.spawn(worker, args=(1, port, *args), nprocs=1, join=True)


class TestSafetensorsExportPlugin:
    """End-to-end: train a PPO agent and confirm the plugin writes a valid safetensors file."""

    def test_on_done_writes_safetensors(self, rom, tmp_path):
        ckpt = str(tmp_path / "ckpt.pt")
        st = str(tmp_path / "model.safetensors")
        _spawn(_local_train_worker, ckpt, st, rom)

        # File exists.
        assert os.path.exists(st), "Safetensors file not written"

        # File loads with the rlcade format and has the expected models.
        loaded, step = load_safetensors(st, device=torch.device("cpu"))
        assert step >= 0
        assert "actor" in loaded
        assert "critic" in loaded
        # Every value should be a tensor (not arbitrary metadata).
        for name, sd in loaded.items():
            for k, v in sd.items():
                assert isinstance(v, torch.Tensor), f"{name}.{k} not a tensor"

    def test_empty_path_skips_export(self, rom, tmp_path):
        ckpt = str(tmp_path / "ckpt.pt")
        # Use a sentinel path that should NOT be touched.
        sentinel = tmp_path / "should_not_exist.safetensors"
        _spawn(_local_train_no_safetensors_worker, ckpt, rom)

        assert not sentinel.exists(), "SafetensorsExportPlugin wrote despite empty path"

    def test_load_inference_from_trained_safetensors(self, rom, tmp_path):
        from rlcade.agent import load_agent
        from rlcade.envs import create_env

        ckpt = str(tmp_path / "ckpt.pt")
        st = str(tmp_path / "model.safetensors")
        _spawn(_local_train_worker, ckpt, st, rom)

        args = make_args(rom, checkpoint=st)
        env = create_env(args)
        args.obs_shape = env.observation_space.shape
        args.n_actions = env.action_space.n
        agent = load_agent("ppo", args, env)
        obs, _ = env.reset()
        action, _, _ = agent.get_action(torch.as_tensor(obs, dtype=torch.float32))
        assert 0 <= action.item() < env.action_space.n
        env.close()
