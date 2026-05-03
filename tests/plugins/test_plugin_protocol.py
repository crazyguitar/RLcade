"""Tests for the TrainerPlugin protocol — verify hooks and conformance."""

import pytest

from rlcade.plugins import TrainerPlugin
from rlcade.plugins.checkpoint import CheckpointPlugin


def test_protocol_is_runtime_checkable():
    """TrainerPlugin should be runtime_checkable for isinstance() checks."""
    plugin = CheckpointPlugin()
    assert isinstance(plugin, TrainerPlugin)


def test_concrete_class_satisfies_protocol():
    """A class implementing all hooks satisfies TrainerPlugin."""

    class _DummyPlugin:
        def on_setup(self, trainer) -> None:
            pass

        def on_step_start(self, trainer, iteration: int) -> None:
            pass

        def on_step_end(self, trainer, iteration: int, summary: dict[str, float] | None) -> None:
            pass

        def on_eval(self, trainer, iteration: int, scores: list[float]) -> None:
            pass

        def on_done(self, trainer) -> None:
            pass

    assert isinstance(_DummyPlugin(), TrainerPlugin)


def test_incomplete_class_does_not_satisfy_protocol():
    """A class missing on_setup should not satisfy TrainerPlugin."""

    class _IncompletePlugin:
        def on_step_start(self, trainer, iteration: int) -> None:
            pass

        def on_step_end(self, trainer, iteration: int, summary: dict[str, float] | None) -> None:
            pass

        def on_eval(self, trainer, iteration: int, scores: list[float]) -> None:
            pass

        def on_done(self, trainer) -> None:
            pass

    assert not isinstance(_IncompletePlugin(), TrainerPlugin)


def test_all_builtin_plugins_satisfy_protocol(tmp_path):
    """Every built-in plugin should satisfy the TrainerPlugin protocol."""
    from rlcade.plugins.tensorboard import TensorBoardPlugin
    from rlcade.plugins.nsys import NsysPlugin
    from rlcade.plugins.memory_profiler import MemoryProfilerPlugin
    from rlcade.plugins.viztracer import VizTracerPlugin

    assert isinstance(CheckpointPlugin(), TrainerPlugin)
    # Use tmp_path to avoid SummaryWriter creating runs/ in the repo root
    assert isinstance(TensorBoardPlugin(log_dir=str(tmp_path / "tb")), TrainerPlugin)
    assert isinstance(NsysPlugin(), TrainerPlugin)
    assert isinstance(MemoryProfilerPlugin(), TrainerPlugin)
    assert isinstance(VizTracerPlugin(output_file=""), TrainerPlugin)


@pytest.mark.parametrize(
    "method, args",
    [
        ("on_setup", ("trainer_sentinel",)),
        ("on_step_start", ("trainer_sentinel", 1)),
        ("on_step_end", ("trainer_sentinel", 1, {"loss": 0.1})),
        ("on_eval", ("trainer_sentinel", 1, [100.0])),
        ("on_done", ("trainer_sentinel",)),
    ],
)
def test_notify_dispatches_all_hooks(method, args):
    """Simulating _notify should invoke each hook type on all plugins."""
    calls = []

    class _SpyPlugin:
        def on_setup(self, trainer) -> None:
            calls.append(("on_setup", trainer))

        def on_step_start(self, trainer, iteration: int) -> None:
            calls.append(("on_step_start", trainer, iteration))

        def on_step_end(self, trainer, iteration: int, summary: dict[str, float] | None) -> None:
            calls.append(("on_step_end", trainer, iteration, summary))

        def on_eval(self, trainer, iteration: int, scores: list[float]) -> None:
            calls.append(("on_eval", trainer, iteration, scores))

        def on_done(self, trainer) -> None:
            calls.append(("on_done", trainer))

    spy = _SpyPlugin()
    # Simulate what Trainer._notify does
    for p in [spy]:
        getattr(p, method)(*args)

    assert len(calls) == 1
    assert calls[0][0] == method
