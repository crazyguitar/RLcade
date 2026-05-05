from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from rlcade.training.trainer import Trainer


@runtime_checkable
class TrainerPlugin(Protocol):
    """Trainer plugin hooks.

    Inherit from this for free no-op defaults; override only the hooks you
    care about. Inheritance is optional -- the trainer dispatches via
    ``getattr``, so any object that structurally matches also satisfies
    ``isinstance(obj, TrainerPlugin)`` thanks to ``@runtime_checkable``.
    """

    def on_setup(self, trainer: Trainer) -> None:
        pass

    def on_step_start(self, trainer: Trainer, iteration: int) -> None:
        pass

    def on_step_end(self, trainer: Trainer, iteration: int, summary: dict[str, float] | None) -> None:
        pass

    def on_eval(self, trainer: Trainer, iteration: int, scores: list[float]) -> None:
        pass

    def on_done(self, trainer: Trainer) -> None:
        pass
