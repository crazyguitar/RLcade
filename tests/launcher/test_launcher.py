"""Tests for rlcade.launcher — launcher × agent × distributed strategy matrix.

Each test calls the launcher API directly (like __main__ does), creating a
real trainer that runs for a few iterations. Tests that require GPUs are
skipped when unavailable.
"""

import os
import socket

import pytest
import torch

from rlcade.launcher import launch
from rlcade.training import create_trainer
from tests.conftest import make_args, make_vec_args


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(autouse=True)
def _cleanup_dist():
    """Clean up distributed state after each test."""
    yield
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()
    for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        os.environ.pop(k, None)


def _launcher_args(launcher, nproc=1, nnodes=1):
    return dict(
        launcher=launcher,
        nproc_per_node=nproc,
        nnodes=nnodes,
        master_addr="127.0.0.1",
        master_port=_free_port(),
        ray_address=None,
        num_gpus=None,
    )


def _train_fn(args):
    """Train function passed to launcher — runs in each worker process."""
    plugins = []
    ckpt_path = getattr(args, "checkpoint_path", "")
    if ckpt_path:
        if getattr(args, "async_checkpoint", False):
            from rlcade.plugins.async_checkpoint import AsyncCheckpointPlugin as _Plugin
        else:
            from rlcade.plugins.checkpoint import CheckpointPlugin as _Plugin

        num_steps = getattr(args, "num_steps", None)
        plugins.append(_Plugin(checkpoint_path=ckpt_path, checkpoint_interval=1, num_steps=num_steps))
    trainer = create_trainer(args.agent, args, plugins=plugins)
    if hasattr(trainer.agent, args.agent):
        getattr(trainer.agent, args.agent).learn_start = 16
    trainer.train()


def _ppo_args(rom, launcher, distributed=None, **kw):
    base = make_args if distributed is None else make_vec_args
    kw.setdefault("checkpoint_path", "")
    kw.setdefault("max_iterations", 2)
    kw.setdefault("backend", "gloo")
    defaults = _launcher_args(launcher)
    defaults.update(kw)
    return base(rom, agent="ppo", eval_interval=0, distributed=distributed, **defaults)


def _offpolicy_args(rom, agent, launcher, distributed=None, **kw):
    base = make_args if distributed is None else make_vec_args
    kw.setdefault("checkpoint_path", "")
    kw.setdefault("max_iterations", 50)
    kw.setdefault("backend", "gloo")
    defaults = _launcher_args(launcher)
    defaults.update(kw)
    return base(
        rom,
        agent=agent,
        eval_interval=0,
        buffer_size=1000,
        log_interval=25,
        distributed=distributed,
        **defaults,
    )


class TestNoneLauncher:
    def test_ppo(self, rom):
        launch(_ppo_args(rom, "none"), _train_fn)

    def test_dqn(self, rom):
        launch(_offpolicy_args(rom, "dqn", "none"), _train_fn)

    def test_sac(self, rom):
        launch(_offpolicy_args(rom, "sac", "none"), _train_fn)

    def test_ppo_ddp(self, rom):
        launch(_ppo_args(rom, "none", distributed="ddp"), _train_fn)

    def test_dqn_ddp(self, rom):
        launch(_offpolicy_args(rom, "dqn", "none", distributed="ddp"), _train_fn)

    def test_sac_ddp(self, rom):
        launch(_offpolicy_args(rom, "sac", "none", distributed="ddp"), _train_fn)

    def test_checkpoint_roundtrip(self, rom, tmp_path):
        from rlcade.plugins.checkpoint import CheckpointPlugin

        ckpt = str(tmp_path / "ckpt.pt")
        launch(_ppo_args(rom, "none", checkpoint_path=ckpt), _train_fn)
        assert os.path.exists(ckpt)

        resume_args = _ppo_args(rom, "none", checkpoint_path=ckpt, max_iterations=4)
        plugins = [CheckpointPlugin(checkpoint_path=ckpt, checkpoint_interval=1, num_steps=resume_args.num_steps)]
        trainer = create_trainer("ppo", resume_args, plugins=plugins)
        trainer.setup()
        assert trainer.start_iteration == 2

    def test_async_checkpoint_roundtrip(self, rom, tmp_path):
        from rlcade.plugins.async_checkpoint import AsyncCheckpointPlugin

        ckpt = str(tmp_path / "ckpt.pt")
        launch(_ppo_args(rom, "none", checkpoint_path=ckpt, async_checkpoint=True), _train_fn)
        assert os.path.exists(ckpt)

        resume_args = _ppo_args(rom, "none", checkpoint_path=ckpt, max_iterations=4, async_checkpoint=True)
        plugins = [AsyncCheckpointPlugin(checkpoint_path=ckpt, checkpoint_interval=1, num_steps=resume_args.num_steps)]
        trainer = create_trainer("ppo", resume_args, plugins=plugins)
        trainer.setup()
        assert trainer.start_iteration == 2
        plugins[0].on_done(trainer)


class TestMpLauncher:
    def test_ppo(self, rom):
        launch(_ppo_args(rom, "mp"), _train_fn)

    def test_dqn(self, rom):
        launch(_offpolicy_args(rom, "dqn", "mp"), _train_fn)

    def test_sac(self, rom):
        launch(_offpolicy_args(rom, "sac", "mp"), _train_fn)

    def test_ppo_ddp(self, rom):
        launch(_ppo_args(rom, "mp", distributed="ddp", nproc_per_node=2), _train_fn)

    def test_dqn_ddp(self, rom):
        launch(_offpolicy_args(rom, "dqn", "mp", distributed="ddp", nproc_per_node=2), _train_fn)

    def test_sac_ddp(self, rom):
        launch(_offpolicy_args(rom, "sac", "mp", distributed="ddp", nproc_per_node=2), _train_fn)

    def test_ddp_checkpoint_roundtrip(self, rom, tmp_path):
        from rlcade.plugins.checkpoint import CheckpointPlugin

        ckpt = str(tmp_path / "ckpt.pt")
        launch(_ppo_args(rom, "mp", distributed="ddp", nproc_per_node=2, checkpoint_path=ckpt), _train_fn)
        assert os.path.exists(ckpt)

        resume_args = _ppo_args(rom, "none", checkpoint_path=ckpt, max_iterations=4)
        plugins = [CheckpointPlugin(checkpoint_path=ckpt, checkpoint_interval=1, num_steps=resume_args.num_steps)]
        trainer = create_trainer("ppo", resume_args, plugins=plugins)
        trainer.setup()
        assert trainer.start_iteration == 2

    def test_ddp_async_checkpoint_roundtrip(self, rom, tmp_path):
        from rlcade.plugins.async_checkpoint import AsyncCheckpointPlugin

        ckpt = str(tmp_path / "ckpt.pt")
        launch(
            _ppo_args(
                rom,
                "mp",
                distributed="ddp",
                nproc_per_node=2,
                checkpoint_path=ckpt,
                async_checkpoint=True,
            ),
            _train_fn,
        )
        assert os.path.exists(ckpt)

        resume_args = _ppo_args(rom, "none", checkpoint_path=ckpt, max_iterations=4, async_checkpoint=True)
        plugins = [AsyncCheckpointPlugin(checkpoint_path=ckpt, checkpoint_interval=1, num_steps=resume_args.num_steps)]
        trainer = create_trainer("ppo", resume_args, plugins=plugins)
        trainer.setup()
        assert trainer.start_iteration == 2
        plugins[0].on_done(trainer)


class TestElasticLauncher:
    def test_ppo(self, rom):
        launch(_ppo_args(rom, "elastic"), _train_fn)

    def test_dqn(self, rom):
        launch(_offpolicy_args(rom, "dqn", "elastic"), _train_fn)

    def test_sac(self, rom):
        launch(_offpolicy_args(rom, "sac", "elastic"), _train_fn)

    def test_ppo_ddp(self, rom):
        launch(_ppo_args(rom, "elastic", distributed="ddp"), _train_fn)

    def test_ppo_ddp_nproc2(self, rom):
        launch(_ppo_args(rom, "elastic", distributed="ddp", nproc_per_node=2), _train_fn)

    def test_dqn_ddp(self, rom):
        launch(_offpolicy_args(rom, "dqn", "elastic", distributed="ddp"), _train_fn)

    def test_sac_ddp(self, rom):
        launch(_offpolicy_args(rom, "sac", "elastic", distributed="ddp"), _train_fn)


class TestMpLauncherFSDP2:
    @pytest.fixture(autouse=True)
    def _skip_no_cuda(self):
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            pytest.skip("FSDP2 requires >= 2 CUDA GPUs")

    def test_ppo(self, rom):
        launch(_ppo_args(rom, "mp", distributed="fsdp2", nproc_per_node=2, backend="nccl", device="cuda"), _train_fn)

    def test_dqn(self, rom):
        launch(
            _offpolicy_args(rom, "dqn", "mp", distributed="fsdp2", nproc_per_node=2, backend="nccl", device="cuda"),
            _train_fn,
        )

    def test_sac(self, rom):
        launch(
            _offpolicy_args(rom, "sac", "mp", distributed="fsdp2", nproc_per_node=2, backend="nccl", device="cuda"),
            _train_fn,
        )

    def test_fsdp2_checkpoint_roundtrip(self, rom, tmp_path):
        from rlcade.plugins.checkpoint import CheckpointPlugin

        ckpt = str(tmp_path / "ckpt.pt")
        launch(
            _ppo_args(
                rom,
                "mp",
                distributed="fsdp2",
                nproc_per_node=2,
                backend="nccl",
                device="cuda",
                checkpoint_path=ckpt,
            ),
            _train_fn,
        )
        assert os.path.exists(ckpt)

        resume_args = _ppo_args(rom, "none", checkpoint_path=ckpt, max_iterations=4)
        plugins = [CheckpointPlugin(checkpoint_path=ckpt, checkpoint_interval=1, num_steps=resume_args.num_steps)]
        trainer = create_trainer("ppo", resume_args, plugins=plugins)
        trainer.setup()
        assert trainer.start_iteration == 2

    def test_fsdp2_async_checkpoint_roundtrip(self, rom, tmp_path):
        from rlcade.plugins.async_checkpoint import AsyncCheckpointPlugin

        ckpt = str(tmp_path / "ckpt.pt")
        launch(
            _ppo_args(
                rom,
                "mp",
                distributed="fsdp2",
                nproc_per_node=2,
                backend="nccl",
                device="cuda",
                checkpoint_path=ckpt,
                async_checkpoint=True,
            ),
            _train_fn,
        )
        assert os.path.exists(ckpt)

        resume_args = _ppo_args(rom, "none", checkpoint_path=ckpt, max_iterations=4, async_checkpoint=True)
        plugins = [AsyncCheckpointPlugin(checkpoint_path=ckpt, checkpoint_interval=1, num_steps=resume_args.num_steps)]
        trainer = create_trainer("ppo", resume_args, plugins=plugins)
        trainer.setup()
        assert trainer.start_iteration == 2
        plugins[0].on_done(trainer)


class TestElasticLauncherFSDP2:
    @pytest.fixture(autouse=True)
    def _skip_no_cuda(self):
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            pytest.skip("FSDP2 requires >= 2 CUDA GPUs")

    def test_ppo(self, rom):
        launch(
            _ppo_args(rom, "elastic", distributed="fsdp2", nproc_per_node=2, backend="nccl", device="cuda"), _train_fn
        )

    def test_dqn(self, rom):
        launch(
            _offpolicy_args(
                rom, "dqn", "elastic", distributed="fsdp2", nproc_per_node=2, backend="nccl", device="cuda"
            ),
            _train_fn,
        )

    def test_sac(self, rom):
        launch(
            _offpolicy_args(
                rom, "sac", "elastic", distributed="fsdp2", nproc_per_node=2, backend="nccl", device="cuda"
            ),
            _train_fn,
        )


class TestRayLauncher:
    @pytest.fixture(autouse=True)
    def _skip_no_gpu_or_ray(self):
        if not torch.cuda.is_available():
            pytest.skip("Ray launcher requires CUDA GPUs")
        try:
            import ray
        except ImportError:
            pytest.skip("ray not installed")
        yield
        import ray

        if ray.is_initialized():
            ray.shutdown()

    def _ppo(self, rom, **kw):
        kw.setdefault("checkpoint_path", "")
        kw.setdefault("max_iterations", 2)
        kw.setdefault("backend", "nccl")
        defaults = _launcher_args("ray")
        defaults.update(kw)
        return make_args(
            rom, agent="ppo", eval_interval=0, distributed="ddp", device="cuda", world=None, stage=None, **defaults
        )

    def _offpolicy(self, rom, agent, **kw):
        kw.setdefault("checkpoint_path", "")
        kw.setdefault("max_iterations", 50)
        kw.setdefault("backend", "nccl")
        defaults = _launcher_args("ray")
        defaults.update(kw)
        return make_args(
            rom,
            agent=agent,
            eval_interval=0,
            distributed="ddp",
            device="cuda",
            world=None,
            stage=None,
            buffer_size=1000,
            log_interval=25,
            **defaults,
        )

    def test_ppo(self, rom):
        launch(self._ppo(rom), _train_fn)

    def test_dqn(self, rom):
        launch(self._offpolicy(rom, "dqn"), _train_fn)

    def test_sac(self, rom):
        launch(self._offpolicy(rom, "sac"), _train_fn)


class TestLauncherDispatch:
    def test_dispatch_none(self):
        import argparse

        called = []
        args = argparse.Namespace(**_launcher_args("none"))
        launch(args, lambda a: called.append(a))
        assert called == [args]

    def test_dispatch_unknown_raises(self):
        import argparse

        args = argparse.Namespace(launcher="bogus")
        with pytest.raises(KeyError):
            launch(args, lambda a: None)

    def test_dispatch_without_nproc_per_node(self):
        import argparse

        called = []
        args = argparse.Namespace(**_launcher_args("none"))
        del args.nproc_per_node
        launch(args, lambda a: called.append(a))
        assert called == [args]


class TestBinPackGpus:
    def test_greedy_fills_largest_first(self):
        from rlcade.launcher.ray import _bin_pack_gpus

        assert _bin_pack_gpus([("10.0.0.1", 8), ("10.0.0.2", 4)], 10) == [8, 2]

    def test_single_node_exact(self):
        from rlcade.launcher.ray import _bin_pack_gpus

        assert _bin_pack_gpus([("10.0.0.1", 8)], 8) == [8]

    def test_auto_uses_all(self):
        from rlcade.launcher.ray import _bin_pack_gpus

        assert sum(_bin_pack_gpus([("10.0.0.1", 4), ("10.0.0.2", 4)], None)) == 8

    def test_request_exceeds_available(self):
        from rlcade.launcher.ray import _bin_pack_gpus

        with pytest.raises(RuntimeError, match="Requested 8 GPUs but cluster only has 2"):
            _bin_pack_gpus([("10.0.0.1", 2)], 8)

    def test_no_gpu_nodes(self):
        from rlcade.launcher.ray import _bin_pack_gpus

        with pytest.raises(RuntimeError, match="no GPUs available"):
            _bin_pack_gpus([], None)

    def test_partial_fill_second_node(self):
        from rlcade.launcher.ray import _bin_pack_gpus

        assert _bin_pack_gpus([("10.0.0.1", 8), ("10.0.0.2", 8)], 12) == [8, 4]

    def test_sorts_by_size_not_input_order(self):
        from rlcade.launcher.ray import _bin_pack_gpus

        assert _bin_pack_gpus([("10.0.0.2", 4), ("10.0.0.1", 8)], 10) == [8, 2]

    def test_single_gpu_request(self):
        from rlcade.launcher.ray import _bin_pack_gpus

        assert _bin_pack_gpus([("10.0.0.1", 8)], 1) == [1]
