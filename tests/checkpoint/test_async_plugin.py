import pytest
import torch

from rlcade.checkpoint.checkpoint import Checkpoint
from rlcade.plugins.async_checkpoint import AsyncCheckpointPlugin


class _FakeAgent:
    """Minimal agent-like object for testing AsyncCheckpointPlugin."""

    device = torch.device("cpu")

    def __init__(self):
        self.state_calls = []
        self.load_calls = []

    def state(self, step=0, *, staging=False):
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


class TestAsyncCheckpointPlugin:
    def test_on_setup_loads_existing_checkpoint(self, tmp_path):
        path = str(tmp_path / "ckpt.pt")
        Checkpoint(path).save({"step": 500, "data": "test"})

        agent = _FakeAgent()
        trainer = _FakeTrainer(agent)

        plugin = AsyncCheckpointPlugin(checkpoint_path=path, checkpoint_interval=10)
        plugin.on_setup(trainer)
        plugin.on_done(trainer)

        assert trainer._start_iteration == 0
        assert trainer.metrics.total_steps == 500
        assert len(agent.load_calls) == 1
        assert agent.load_calls[0]["step"] == 500

    def test_on_setup_no_change_when_no_checkpoint(self, tmp_path):
        path = str(tmp_path / "ckpt.pt")
        agent = _FakeAgent()
        trainer = _FakeTrainer(agent)

        plugin = AsyncCheckpointPlugin(checkpoint_path=path, checkpoint_interval=10)
        plugin.on_setup(trainer)
        plugin.on_done(trainer)

        assert trainer._start_iteration == 0

    def test_on_setup_no_change_when_no_path(self):
        agent = _FakeAgent()
        trainer = _FakeTrainer(agent)

        plugin = AsyncCheckpointPlugin(checkpoint_path="", checkpoint_interval=10)
        plugin.on_setup(trainer)
        plugin.on_done(trainer)

        assert trainer._start_iteration == 0

    def test_on_step_end_saves_at_interval(self, tmp_path):
        path = str(tmp_path / "ckpt.pt")
        agent = _FakeAgent()
        trainer = _FakeTrainer(agent)
        trainer.metrics.total_steps = 1000

        plugin = AsyncCheckpointPlugin(checkpoint_path=path, checkpoint_interval=10)
        plugin.on_setup(trainer)

        plugin.on_step_end(trainer, 5, {"loss": 0.1})
        assert len(agent.state_calls) == 0

        plugin.on_step_end(trainer, 10, {"loss": 0.05})
        plugin.on_done(trainer)

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

        plugin = AsyncCheckpointPlugin(checkpoint_path=path, checkpoint_interval=10)
        plugin.on_setup(trainer)

        plugin.on_done(trainer)
        assert len(agent.state_calls) == 1
        with Checkpoint(path).reader() as f:
            loaded = torch.load(f, weights_only=True)
        assert loaded["step"] == 2000

    def test_no_save_when_no_path(self, tmp_path):
        agent = _FakeAgent()
        trainer = _FakeTrainer(agent)

        plugin = AsyncCheckpointPlugin(checkpoint_path="", checkpoint_interval=10)
        plugin.on_setup(trainer)

        plugin.on_step_end(trainer, 10, {"loss": 0.1})
        plugin.on_done(trainer)
        assert len(agent.state_calls) == 0

    def test_on_setup_computes_iteration_for_on_policy(self, tmp_path):
        path = str(tmp_path / "ckpt.pt")
        Checkpoint(path).save({"step": 1600, "data": "test"})

        agent = _FakeAgent()
        trainer = _FakeTrainer(agent)

        plugin = AsyncCheckpointPlugin(checkpoint_path=path, checkpoint_interval=10, num_steps=128)
        plugin.on_setup(trainer)
        plugin.on_done(trainer)

        assert trainer._start_iteration == 1600 // 128
        assert trainer.metrics.total_steps == 1600

    def test_checkpoint_interval_zero_disables_periodic_saves(self, tmp_path):
        path = str(tmp_path / "ckpt.pt")
        agent = _FakeAgent()
        trainer = _FakeTrainer(agent)
        trainer.metrics.total_steps = 500

        plugin = AsyncCheckpointPlugin(checkpoint_path=path, checkpoint_interval=0)
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

        plugin = AsyncCheckpointPlugin(checkpoint_path=path, checkpoint_interval=5)
        plugin.on_setup(trainer)

        plugin.on_step_end(trainer, 0, {})
        plugin.on_done(trainer)
        assert len(agent.state_calls) == 1  # on_done always writes the final state

    def test_on_done_skips_double_save_at_final_interval(self, tmp_path):
        path = str(tmp_path / "ckpt.pt")
        agent = _FakeAgent()
        trainer = _FakeTrainer(agent)
        trainer.metrics.total_steps = 1000

        plugin = AsyncCheckpointPlugin(checkpoint_path=path, checkpoint_interval=10)
        plugin.on_setup(trainer)

        plugin.on_step_end(trainer, 10, {})
        plugin.on_done(trainer)
        assert len(agent.state_calls) == 1

    def test_on_done_saves_when_step_advanced_after_last_periodic(self, tmp_path):
        path = str(tmp_path / "ckpt.pt")
        agent = _FakeAgent()
        trainer = _FakeTrainer(agent)
        trainer.metrics.total_steps = 1000

        plugin = AsyncCheckpointPlugin(checkpoint_path=path, checkpoint_interval=10)
        plugin.on_setup(trainer)

        plugin.on_step_end(trainer, 10, {})
        trainer.metrics.total_steps = 1100
        plugin.on_done(trainer)
        assert len(agent.state_calls) == 2

    def test_on_setup_with_corrupt_checkpoint_propagates(self, tmp_path):
        path = str(tmp_path / "ckpt.pt")
        with open(path, "wb") as f:
            f.write(b"not a valid checkpoint")

        agent = _FakeAgent()
        trainer = _FakeTrainer(agent)

        plugin = AsyncCheckpointPlugin(checkpoint_path=path, checkpoint_interval=10)
        with pytest.raises(Exception):
            plugin.on_setup(trainer)
        # Executor must not be created when load fails — otherwise the worker leaks.
        assert plugin._executor is None

    def test_on_setup_is_idempotent(self, tmp_path):
        """Calling on_setup twice must not create a second executor."""
        path = str(tmp_path / "ckpt.pt")
        agent = _FakeAgent()
        trainer = _FakeTrainer(agent)

        plugin = AsyncCheckpointPlugin(checkpoint_path=path, checkpoint_interval=10)
        plugin.on_setup(trainer)
        first_executor = plugin._executor
        plugin.on_setup(trainer)
        assert plugin._executor is first_executor
        plugin.on_done(trainer)

    def test_off_policy_checkpoint_roundtrip(self, tmp_path):
        path = str(tmp_path / "ckpt.pt")
        agent = _FakeAgent()
        trainer = _FakeTrainer(agent)
        trainer.metrics.total_steps = 5000

        plugin = AsyncCheckpointPlugin(checkpoint_path=path, checkpoint_interval=10)
        plugin.on_setup(trainer)
        plugin.on_done(trainer)

        agent2 = _FakeAgent()
        trainer2 = _FakeTrainer(agent2)

        plugin2 = AsyncCheckpointPlugin(checkpoint_path=path, checkpoint_interval=10)
        plugin2.on_setup(trainer2)
        plugin2.on_done(trainer2)

        assert trainer2.metrics.total_steps == 5000
        assert trainer2._start_iteration == 0

    def test_on_policy_checkpoint_roundtrip(self, tmp_path):
        path = str(tmp_path / "ckpt.pt")
        agent = _FakeAgent()
        trainer = _FakeTrainer(agent)
        trainer.metrics.total_steps = 2560

        plugin = AsyncCheckpointPlugin(checkpoint_path=path, checkpoint_interval=10, num_steps=128)
        plugin.on_setup(trainer)
        plugin.on_done(trainer)

        agent2 = _FakeAgent()
        trainer2 = _FakeTrainer(agent2)

        plugin2 = AsyncCheckpointPlugin(checkpoint_path=path, checkpoint_interval=10, num_steps=128)
        plugin2.on_setup(trainer2)
        plugin2.on_done(trainer2)

        assert trainer2.metrics.total_steps == 2560
        assert trainer2._start_iteration == 2560 // 128

    def test_plugin_exposes_checkpoint_instance(self, tmp_path):
        path = str(tmp_path / "ckpt.pt")
        plugin = AsyncCheckpointPlugin(checkpoint_path=path)
        assert plugin._checkpoint is not None
        assert plugin._checkpoint.url == path

    def test_plugin_no_checkpoint_when_empty_path(self):
        plugin = AsyncCheckpointPlugin(checkpoint_path="")
        assert plugin._checkpoint is None

    def test_save_drains_prior_future_before_restaging(self, tmp_path):
        """state(staging=True) must not be called while a prior save's worker is still reading.

        StateDictStager reuses cached CPU buffers keyed by source storage, so a
        second stage() call while the worker is mid-write would silently corrupt
        the in-flight checkpoint.  We verify ordering by recording the state of
        plugin._future at the moment state() is called: if it's pending (not
        done), the plugin violated the "drain before stage" invariant.
        """
        import threading

        path = str(tmp_path / "ckpt.pt")

        class _OrderCheckingAgent(_FakeAgent):
            def __init__(self):
                super().__init__()
                self.plugin = None
                self.violations = []

            def state(self, step=0, *, staging=False):
                pending = self.plugin is not None and self.plugin._future is not None and not self.plugin._future.done()
                if pending:
                    self.violations.append(step)
                return super().state(step, staging=staging)

        agent = _OrderCheckingAgent()
        trainer = _FakeTrainer(agent)
        trainer.metrics.total_steps = 100

        plugin = AsyncCheckpointPlugin(checkpoint_path=path, checkpoint_interval=10)
        plugin.on_setup(trainer)
        agent.plugin = plugin

        gate = threading.Event()
        real_save = plugin._checkpoint.save

        def gated_save(state):
            gate.wait(timeout=5)
            real_save(state)

        plugin._checkpoint.save = gated_save

        # First save: submits; worker blocks on gate, so plugin._future is pending.
        plugin.on_step_end(trainer, 10, {})
        assert plugin._future is not None and not plugin._future.done()

        # Release the worker shortly after we enter the second _save.  With the
        # fix, _save joins the prior future (blocks on the gate until release),
        # then stages step=200 after _future has been cleared.  Without the fix,
        # state(200) would be called immediately while _future is still pending.
        trainer.metrics.total_steps = 200
        releaser = threading.Timer(0.1, gate.set)
        releaser.start()
        try:
            plugin.on_step_end(trainer, 20, {})
        finally:
            releaser.cancel()

        assert (
            agent.violations == []
        ), f"state() called while prior save was still in-flight at step(s) {agent.violations}"
        plugin.on_done(trainer)

    def test_save_failure_surfaces_on_next_save(self, tmp_path):
        """A failed background save is detected on the next _join_and_sync call."""
        path = str(tmp_path / "ckpt.pt")
        agent = _FakeAgent()
        trainer = _FakeTrainer(agent)
        trainer.metrics.total_steps = 100

        plugin = AsyncCheckpointPlugin(checkpoint_path=path, checkpoint_interval=10)
        plugin.on_setup(trainer)

        def boom(_state):
            raise OSError("disk full")

        plugin._checkpoint.save = boom

        # First save: submit succeeds (queued); worker raises asynchronously.
        plugin.on_step_end(trainer, 10, {})
        # Confirm the failure genuinely happened on the worker thread, not inline.
        first_future = plugin._future
        assert first_future is not None
        assert isinstance(first_future.exception(timeout=5), OSError)

        # Second save: join picks up the prior failure and surfaces it.
        trainer.metrics.total_steps = 200
        with pytest.raises(RuntimeError, match="Prior async checkpoint save failed"):
            plugin.on_step_end(trainer, 20, {})

        # on_done cleans up the executor even when recovery save also fails.
        with pytest.raises(RuntimeError):
            plugin.on_done(trainer)
        assert plugin._executor is None
        assert plugin._future is None


class TestStagerPinMemory:
    """Trainer propagates args.pin_memory to the agent's stager."""

    def test_pin_memory_true_by_default(self, rom):
        from rlcade.training import create_trainer
        from tests.conftest import make_vec_args

        args = make_vec_args(rom, agent="ppo", max_iterations=1, eval_interval=0)
        trainer = create_trainer("ppo", args)
        trainer.setup()
        assert trainer.agent._impl.pin_memory is True

    def test_no_pin_memory_flag_disables(self, rom):
        from rlcade.training import create_trainer
        from tests.conftest import make_vec_args

        args = make_vec_args(rom, agent="ppo", max_iterations=1, eval_interval=0, pin_memory=False)
        trainer = create_trainer("ppo", args)
        trainer.setup()
        assert trainer.agent._impl.pin_memory is False
        trainer.agent.state(step=0, staging=True)
        assert trainer.agent._impl.stager.pin_memory is False
