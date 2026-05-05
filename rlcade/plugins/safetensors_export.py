"""Safetensors export plugin -- writes model weights once at training end."""

from __future__ import annotations

from rlcade.checkpoint.safetensors import save_safetensors
from rlcade.logger import get_log0
from rlcade.plugins import TrainerPlugin
from rlcade.plugins._distributed import save_and_sync

log0 = get_log0(__name__)


class SafetensorsExportPlugin(TrainerPlugin):
    """Trainer plugin that exports model weights to safetensors at end of training.

    Args:
        safetensors_path: File path or S3 URL for the export (empty to disable).
    """

    def __init__(self, safetensors_path: str = ""):
        self.safetensors_path = safetensors_path

    def on_done(self, trainer) -> None:
        if not self.safetensors_path:
            return
        step = trainer.metrics.total_steps
        state = trainer.agent.state(step)
        path = self.safetensors_path

        def write():
            save_safetensors(state, path, step=step)

        ok = save_and_sync(trainer, write, what="Safetensors export")
        if not ok:
            raise RuntimeError(f"Safetensors export failed at step {step}")
        log0.info("Safetensors exported to %s", path)
