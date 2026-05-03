from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from rlcade.training.trainer import Trainer


@runtime_checkable
class TrainerPlugin(Protocol):
    """Plugin interface for trainer hooks."""

    def on_setup(self, trainer: Trainer) -> None: ...

    def on_step_start(self, trainer: Trainer, iteration: int) -> None: ...

    def on_step_end(self, trainer: Trainer, iteration: int, summary: dict[str, float] | None) -> None: ...

    def on_eval(self, trainer: Trainer, iteration: int, scores: list[float]) -> None: ...

    def on_done(self, trainer: Trainer) -> None: ...
