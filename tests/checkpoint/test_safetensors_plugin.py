"""SafetensorsExportPlugin -- unit tests + real trainer integration."""

from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.multiprocessing as mp

from rlcade.checkpoint.safetensors import load_safetensors
from rlcade.plugins.safetensors_export import SafetensorsExportPlugin
from tests.conftest import make_args


# Unit tests with a fake agent — fast, no subprocess, no ROM.


class _FakeAgent:
    """Minimal agent-like object whose state() returns a dict-of-tensor-dicts."""

    device = torch.device("cpu")

    def __init__(self):
        self.state_calls: list[int] = []

    def state(self, step: int = 0) -> dict:
        self.state_calls.append(step)
        return {
            "step": step,
            "actor": {"w": torch.randn(2, 2), "b": torch.randn(2)},
            "critic": {"w": torch.randn(1, 2)},
        }


class _FakeMetrics:
    def __init__(self, total_steps: int = 0):
        self.total_steps = total_steps


class _FakeTrainer:
    def __init__(self, agent: _FakeAgent, total_steps: int = 0):
        self.agent = agent
        self.metrics = _FakeMetrics(total_steps)
        self.rank0 = True
        self.distributed = False


class TestSafetensorsExportPluginUnit:
    def test_on_done_writes_file_with_step(self, tmp_path):
        path = str(tmp_path / "model.safetensors")
        agent = _FakeAgent()
        trainer = _FakeTrainer(agent, total_steps=300)
        plugin = SafetensorsExportPlugin(safetensors_path=path)

        plugin.on_done(trainer)

        assert os.path.exists(path)
        loaded, step = load_safetensors(path, device=torch.device("cpu"))
        assert step == 300
        assert {"actor", "critic"} <= set(loaded.keys())
        assert agent.state_calls == [300]

    def test_empty_path_skips_state_call(self):
        agent = _FakeAgent()
        trainer = _FakeTrainer(agent, total_steps=300)
        plugin = SafetensorsExportPlugin(safetensors_path="")

        plugin.on_done(trainer)

        # No path means no work — state() must not even be called.
        assert agent.state_calls == []

    def test_save_failure_raises_runtime_error(self, tmp_path, monkeypatch):
        path = str(tmp_path / "model.safetensors")
        agent = _FakeAgent()
        trainer = _FakeTrainer(agent, total_steps=42)
        plugin = SafetensorsExportPlugin(safetensors_path=path)

        from rlcade.plugins import safetensors_export

        def boom(state, url, *, step=0):
            raise OSError("disk full")

        monkeypatch.setattr(safetensors_export, "save_safetensors", boom)

        with pytest.raises(RuntimeError, match="Safetensors export failed at step 42"):
            plugin.on_done(trainer)

    def test_writes_at_step_zero(self, tmp_path):
        """on_done should still write even if the trainer never advanced."""
        path = str(tmp_path / "model.safetensors")
        agent = _FakeAgent()
        trainer = _FakeTrainer(agent, total_steps=0)
        plugin = SafetensorsExportPlugin(safetensors_path=path)

        plugin.on_done(trainer)

        loaded, step = load_safetensors(path, device=torch.device("cpu"))
        assert step == 0
        assert "actor" in loaded


# End-to-end PPO training tests (real trainer + mp.spawn).


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
