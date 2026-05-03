"""Curriculum learning plugin — progressively unlock harder stages.

Starts training on easy stages and adds harder ones as the agent improves.
Works by replacing the trainer's env with a new vec env containing more stages.
"""

from __future__ import annotations

import argparse

from rlcade.envs import get_world_stage_pairs, _create_inprocess_vector_env
from rlcade.logger import get_logger

logger = get_logger(__name__)

# Default stage ordering: world 1 first, then 2, etc.
ALL_STAGES = get_world_stage_pairs(None, None)


class CurriculumPlugin:
    """Trainer plugin that expands the stage set when performance improves.

    Args:
        args: CLI args (needed to create new envs with same config).
        stages: Ordered list of (world, stage) pairs. Defaults to all 32.
        initial_stages: Number of stages to start with.
        expand_threshold: Mean score (over last N episodes) to trigger expansion.
        expand_count: Number of stages to add per expansion.
        window: Number of recent episodes to average for threshold check.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        stages: list[tuple[int, int]] | None = None,
        initial_stages: int = 4,
        expand_threshold: float = 500.0,
        expand_count: int = 4,
        window: int = 20,
    ):
        self.args = args
        self.all_stages = stages or ALL_STAGES
        self.current_count = min(initial_stages, len(self.all_stages))
        self.expand_threshold = expand_threshold
        self.expand_count = expand_count
        self.window = window

    @property
    def active_stages(self) -> list[tuple[int, int]]:
        return self.all_stages[: self.current_count]

    def on_setup(self, trainer) -> None:
        self._rebuild_env(trainer)

    def on_step_start(self, trainer, iteration: int) -> None:
        pass

    def on_step_end(self, trainer, iteration: int, summary: dict[str, float] | None) -> None:
        if self.current_count >= len(self.all_stages):
            return
        scores = trainer.metrics.episode_scores
        if len(scores) < self.window:
            return
        recent_mean = sum(scores[-self.window :]) / self.window
        if recent_mean >= self.expand_threshold:
            self._expand(trainer)

    def on_eval(self, trainer, iteration: int, scores: list[float]) -> None:
        pass

    def on_done(self, trainer) -> None:
        pass

    def _expand(self, trainer) -> None:
        prev = self.current_count
        self.current_count = min(self.current_count + self.expand_count, len(self.all_stages))
        if self.current_count == prev:
            return
        logger.info(
            "Curriculum: expanding from %d to %d stages (added %s)",
            prev,
            self.current_count,
            self.all_stages[prev : self.current_count],
        )
        self._rebuild_env(trainer)

    def _rebuild_env(self, trainer) -> None:
        """Replace the trainer's env via trainer.swap()."""
        new_env = _create_inprocess_vector_env(self.args, self.active_stages, label="train")
        trainer.swap(new_env)
        logger.info("Curriculum: env rebuilt with %d envs", new_env.num_envs)
