"""Weights & Biases plugin -- logs metrics, run config, and final-checkpoint artifacts."""

from __future__ import annotations

import os
from argparse import Namespace
from typing import Any

import torch.distributed as dist

from rlcade.logger import get_logger
from rlcade.plugins import TrainerPlugin
from rlcade.plugins._metric_keys import SCALAR_MAP

logger = get_logger(__name__)


def _is_rank0() -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


class WandbPlugin(TrainerPlugin):
    """Trainer plugin that mirrors training to a Weights & Biases server."""

    def __init__(
        self,
        base_url: str | None,
        entity: str | None,
        project: str,
        run_name: str | None,
        checkpoint_path: str,
        safetensors_path: str,
        args: Namespace,
    ) -> None:
        self._base_url = base_url
        self._entity = entity
        self._project = project
        self._run_name = run_name
        self._checkpoint_path = checkpoint_path
        self._safetensors_path = safetensors_path
        self._args = args
        self._run: Any = None

    def on_setup(self, trainer) -> None:
        if not _is_rank0():
            return
        if self._base_url:
            os.environ["WANDB_BASE_URL"] = self._base_url

        import wandb

        config = {k: v for k, v in vars(self._args).items() if not k.startswith("_")}
        self._run = wandb.init(
            entity=self._entity,
            project=self._project,
            name=self._run_name,
            config=config,
        )

    def on_step_start(self, trainer, iteration: int) -> None:
        return

    def on_step_end(self, trainer, iteration: int, summary: dict[str, float] | None) -> None:
        if not _is_rank0() or not summary:
            return

        import wandb

        payload = {tag: float(summary[key]) for key, tag in SCALAR_MAP.items() if key in summary}
        if not payload:
            return
        wandb.log(payload, step=iteration)

    def on_eval(self, trainer, iteration: int, scores: list[float]) -> None:
        if not _is_rank0() or not scores:
            return

        import wandb

        wandb.log(
            {
                "eval/mean_score": sum(scores) / len(scores),
                "eval/max_score": float(max(scores)),
                "eval/min_score": float(min(scores)),
            },
            step=iteration,
        )

    def on_done(self, trainer) -> None:
        if not _is_rank0() or self._run is None:
            return

        self._upload_artifact(self._checkpoint_path)
        self._upload_artifact(self._safetensors_path)
        self._finish_run()

    def _upload_artifact(self, path: str) -> None:
        if not path or not os.path.exists(path):
            return

        import wandb

        try:
            artifact = wandb.Artifact(name=os.path.basename(path), type="model")
            artifact.add_file(path)
            self._run.log_artifact(artifact)
        except Exception as exc:
            logger.warning("wandb: failed to upload artifact %s: %s", path, exc)

    def _finish_run(self) -> None:
        import wandb

        try:
            wandb.finish()
        except Exception as exc:
            logger.warning("wandb: finish failed: %s", exc)
