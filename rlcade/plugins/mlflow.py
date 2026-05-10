"""MLflow plugin -- logs metrics, params, and final-checkpoint artifacts."""

from __future__ import annotations

import os
from argparse import Namespace
from typing import Any

import torch.distributed as dist

from rlcade.logger import get_logger
from rlcade.plugins import TrainerPlugin
from rlcade.plugins._metric_keys import SCALAR_MAP

logger = get_logger(__name__)

_MAX_PARAM_LEN = 500


class MLflowPlugin(TrainerPlugin):
    """Trainer plugin that mirrors training to an MLflow tracking server."""

    def __init__(
        self,
        tracking_uri: str | None,
        experiment: str,
        run_name: str | None,
        checkpoint_path: str,
        safetensors_path: str,
        args: Namespace,
    ) -> None:
        self._tracking_uri = tracking_uri
        self._experiment = experiment
        self._run_name = run_name
        self._checkpoint_path = checkpoint_path
        self._safetensors_path = safetensors_path
        self._args = args
        self._run: Any = None  # set by on_setup on rank 0

    @staticmethod
    def _is_rank0() -> bool:
        if not dist.is_available() or not dist.is_initialized():
            return True
        return dist.get_rank() == 0

    def on_setup(self, trainer) -> None:
        if not self._is_rank0():
            return

        import mlflow  # lazy: only rank 0 needs it

        if self._tracking_uri:
            mlflow.set_tracking_uri(self._tracking_uri)
        mlflow.set_experiment(self._experiment)
        self._run = mlflow.start_run(run_name=self._run_name)

        params: dict[str, str] = {}
        for key, value in vars(self._args).items():
            if key.startswith("_"):
                continue
            try:
                params[key] = str(value)[:_MAX_PARAM_LEN]
            except Exception as exc:
                logger.warning("mlflow: skipping param %s (str() failed: %s)", key, exc)
        if params:
            mlflow.log_params(params)

    def on_step_start(self, trainer, iteration: int) -> None:
        return

    def on_step_end(self, trainer, iteration: int, summary: dict[str, float] | None) -> None:
        if not self._is_rank0() or not summary:
            return

        import mlflow

        for key, tag in SCALAR_MAP.items():
            if key in summary:
                mlflow.log_metric(tag, float(summary[key]), step=iteration)

    def on_eval(self, trainer, iteration: int, scores: list[float]) -> None:
        if not self._is_rank0() or not scores:
            return

        import mlflow

        mean_score = sum(scores) / len(scores)
        mlflow.log_metric("eval/mean_score", mean_score, step=iteration)
        mlflow.log_metric("eval/max_score", float(max(scores)), step=iteration)
        mlflow.log_metric("eval/min_score", float(min(scores)), step=iteration)

    def on_done(self, trainer) -> None:
        if not self._is_rank0():
            return

        import mlflow

        for path in (self._checkpoint_path, self._safetensors_path):
            if not path or not os.path.exists(path):
                continue
            try:
                mlflow.log_artifact(path)
            except Exception as exc:
                logger.warning("mlflow: failed to upload artifact %s: %s", path, exc)

        try:
            mlflow.end_run()
        except Exception as exc:
            logger.warning("mlflow: end_run failed: %s", exc)
