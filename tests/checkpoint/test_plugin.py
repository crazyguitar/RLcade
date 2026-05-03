import pytest
import torch

from rlcade.checkpoint.checkpoint import Checkpoint
from rlcade.plugins.checkpoint import CheckpointPlugin


class _FakeAgent:
    """Minimal agent-like object for testing CheckpointPlugin."""

    device = torch.device("cpu")

    def __init__(self):
        self.state_calls = []
        self.load_calls = []

    def state(self, step=0):
        self.state_calls.append(step)
        return {"step": step, "data": "test"}

    def load(self, f) -> int:
        state = torch.load(f, map_location=self.device, weights_only=True)
        self.load_calls.append(state)
        return state.get("step", 0)


class _FakeMetrics:
    def __init__(self):
        self.total_steps = 0


class _FakeTrainer:
    def __init__(self, agent):
        self.agent = agent
        self.metrics = _FakeMetrics()
        self._start_iteration = 0
        self.rank0 = True
        self.distributed = False


class TestCheckpointPlugin:
    def test_on_setup_loads_existing_checkpoint(self, tmp_path):
        path = str(tmp_path / "ckpt.pt")
        Checkpoint(path).save({"step": 500, "data": "test"})

        agent = _FakeAgent()
        trainer = _FakeTrainer(agent)

        plugin = CheckpointPlugin(checkpoint_path=path, checkpoint_interval=10)
        plugin.on_setup(trainer)

        assert trainer._start_iteration == 0
        assert trainer.metrics.total_steps == 500
        assert len(agent.load_calls) == 1
        assert agent.load_calls[0]["step"] == 500

    def test_on_setup_no_change_when_no_checkpoint(self, tmp_path):
        path = str(tmp_path / "ckpt.pt")
        agent = _FakeAgent()
        trainer = _FakeTrainer(agent)

        plugin = CheckpointPlugin(checkpoint_path=path, checkpoint_interval=10)
        plugin.on_setup(trainer)

        assert trainer._start_iteration == 0

    def test_on_setup_no_change_when_no_path(self):
        agent = _FakeAgent()
        trainer = _FakeTrainer(agent)

        plugin = CheckpointPlugin(checkpoint_path="", checkpoint_interval=10)
        plugin.on_setup(trainer)

        assert trainer._start_iteration == 0

    def test_on_step_end_saves_at_interval(self, tmp_path):
        path = str(tmp_path / "ckpt.pt")
        agent = _FakeAgent()
        trainer = _FakeTrainer(agent)
        trainer.metrics.total_steps = 1000

        plugin = CheckpointPlugin(checkpoint_path=path, checkpoint_interval=10)
        plugin.on_setup(trainer)

        plugin.on_step_end(trainer, 5, {"loss": 0.1})
        assert len(agent.state_calls) == 0

        plugin.on_step_end(trainer, 10, {"loss": 0.05})
        assert len(agent.state_calls) == 1
        assert Checkpoint(path).exists()
        with Checkpoint(path).reader() as f:
            loaded = torch.load(f, weights_only=True)
        assert loaded["step"] == 1000

    def test_on_done_saves_final_checkpoint(self, tmp_path):
        path = str(tmp_path / "ckpt.pt")
        agent = _FakeAgent()
        trainer = _FakeTrainer(agent)
        trainer.metrics.total_steps = 2000

        plugin = CheckpointPlugin(checkpoint_path=path, checkpoint_interval=10)
        plugin.on_setup(trainer)

        plugin.on_done(trainer)
        assert len(agent.state_calls) == 1
        with Checkpoint(path).reader() as f:
            loaded = torch.load(f, weights_only=True)
        assert loaded["step"] == 2000

    def test_no_save_when_no_path(self, tmp_path):
        agent = _FakeAgent()
        trainer = _FakeTrainer(agent)

        plugin = CheckpointPlugin(checkpoint_path="", checkpoint_interval=10)
        plugin.on_setup(trainer)

        plugin.on_step_end(trainer, 10, {"loss": 0.1})
        plugin.on_done(trainer)
        assert len(agent.state_calls) == 0

    def test_on_setup_computes_iteration_for_on_policy(self, tmp_path):
        path = str(tmp_path / "ckpt.pt")
        Checkpoint(path).save({"step": 1600, "data": "test"})

        agent = _FakeAgent()
        trainer = _FakeTrainer(agent)

        plugin = CheckpointPlugin(checkpoint_path=path, checkpoint_interval=10, num_steps=128)
        plugin.on_setup(trainer)

        assert trainer._start_iteration == 1600 // 128
        assert trainer.metrics.total_steps == 1600

    def test_checkpoint_interval_zero_disables_periodic_saves(self, tmp_path):
        path = str(tmp_path / "ckpt.pt")
        agent = _FakeAgent()
        trainer = _FakeTrainer(agent)
        trainer.metrics.total_steps = 500

        plugin = CheckpointPlugin(checkpoint_path=path, checkpoint_interval=0)
        plugin.on_setup(trainer)

        plugin.on_step_end(trainer, 1, {})
        plugin.on_step_end(trainer, 10, {})
        plugin.on_step_end(trainer, 100, {})
        assert len(agent.state_calls) == 0

        plugin.on_done(trainer)
        assert len(agent.state_calls) == 1

    def test_on_step_end_ignores_iteration_zero(self, tmp_path):
        path = str(tmp_path / "ckpt.pt")
        agent = _FakeAgent()
        trainer = _FakeTrainer(agent)
        trainer.metrics.total_steps = 100

        plugin = CheckpointPlugin(checkpoint_path=path, checkpoint_interval=5)
        plugin.on_setup(trainer)

        plugin.on_step_end(trainer, 0, {})
        assert len(agent.state_calls) == 0

    def test_on_done_skips_double_save_at_final_interval(self, tmp_path):
        path = str(tmp_path / "ckpt.pt")
        agent = _FakeAgent()
        trainer = _FakeTrainer(agent)
        trainer.metrics.total_steps = 1000

        plugin = CheckpointPlugin(checkpoint_path=path, checkpoint_interval=10)
        plugin.on_setup(trainer)

        plugin.on_step_end(trainer, 10, {})
        assert len(agent.state_calls) == 1

        plugin.on_done(trainer)
        assert len(agent.state_calls) == 1

    def test_on_done_saves_when_step_advanced_after_last_periodic(self, tmp_path):
        path = str(tmp_path / "ckpt.pt")
        agent = _FakeAgent()
        trainer = _FakeTrainer(agent)
        trainer.metrics.total_steps = 1000

        plugin = CheckpointPlugin(checkpoint_path=path, checkpoint_interval=10)
        plugin.on_setup(trainer)

        plugin.on_step_end(trainer, 10, {})
        assert len(agent.state_calls) == 1

        trainer.metrics.total_steps = 1100
        plugin.on_done(trainer)
        assert len(agent.state_calls) == 2

    def test_on_setup_with_corrupt_checkpoint_propagates(self, tmp_path):
        path = str(tmp_path / "ckpt.pt")
        with open(path, "wb") as f:
            f.write(b"not a valid checkpoint")

        agent = _FakeAgent()
        trainer = _FakeTrainer(agent)

        plugin = CheckpointPlugin(checkpoint_path=path, checkpoint_interval=10)
        with pytest.raises(Exception):
            plugin.on_setup(trainer)

    def test_off_policy_checkpoint_roundtrip(self, tmp_path):
        path = str(tmp_path / "ckpt.pt")
        agent = _FakeAgent()
        trainer = _FakeTrainer(agent)
        trainer.metrics.total_steps = 5000

        plugin = CheckpointPlugin(checkpoint_path=path, checkpoint_interval=10)
        plugin.on_setup(trainer)
        plugin.on_done(trainer)

        agent2 = _FakeAgent()
        trainer2 = _FakeTrainer(agent2)

        plugin2 = CheckpointPlugin(checkpoint_path=path, checkpoint_interval=10)
        plugin2.on_setup(trainer2)

        assert trainer2.metrics.total_steps == 5000
        assert trainer2._start_iteration == 0

    def test_on_policy_checkpoint_roundtrip(self, tmp_path):
        path = str(tmp_path / "ckpt.pt")
        agent = _FakeAgent()
        trainer = _FakeTrainer(agent)
        trainer.metrics.total_steps = 2560

        plugin = CheckpointPlugin(checkpoint_path=path, checkpoint_interval=10, num_steps=128)
        plugin.on_setup(trainer)
        plugin.on_done(trainer)

        agent2 = _FakeAgent()
        trainer2 = _FakeTrainer(agent2)

        plugin2 = CheckpointPlugin(checkpoint_path=path, checkpoint_interval=10, num_steps=128)
        plugin2.on_setup(trainer2)

        assert trainer2.metrics.total_steps == 2560
        assert trainer2._start_iteration == 2560 // 128

    def test_plugin_exposes_checkpoint_instance(self, tmp_path):
        path = str(tmp_path / "ckpt.pt")
        plugin = CheckpointPlugin(checkpoint_path=path)
        assert plugin._checkpoint is not None
        assert plugin._checkpoint.url == path

    def test_plugin_no_checkpoint_when_empty_path(self):
        plugin = CheckpointPlugin(checkpoint_path="")
        assert plugin._checkpoint is None

    def test_resume_primes_last_saved_step_so_on_done_skips(self, tmp_path):
        """After resume, on_done must not re-save the same step."""
        path = str(tmp_path / "ckpt.pt")
        Checkpoint(path).save({"step": 500, "data": "test"})

        agent = _FakeAgent()
        trainer = _FakeTrainer(agent)

        plugin = CheckpointPlugin(checkpoint_path=path, checkpoint_interval=10)
        plugin.on_setup(trainer)
        # total_steps == 500 == _last_saved_step, so on_done must be a no-op.
        plugin.on_done(trainer)
        assert agent.state_calls == []  # no save was triggered

    def test_save_failure_raises_runtime_error(self, tmp_path):
        """Rank 0 save failure is caught, propagated via all_reduce, and surfaces as RuntimeError."""
        path = str(tmp_path / "ckpt.pt")
        agent = _FakeAgent()
        trainer = _FakeTrainer(agent)
        trainer.metrics.total_steps = 100

        plugin = CheckpointPlugin(checkpoint_path=path, checkpoint_interval=10)
        plugin.on_setup(trainer)

        def boom(_state):
            raise OSError("disk full")

        plugin._checkpoint.save = boom

        with pytest.raises(RuntimeError, match="Checkpoint save failed at step 100"):
            plugin.on_step_end(trainer, 10, {})

        # _last_saved_step stays at its initial value so on_done knows to retry.
        assert plugin._last_saved_step == -1
