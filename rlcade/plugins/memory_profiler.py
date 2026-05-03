"""PyTorch CUDA memory profiling plugin."""

import torch

from rlcade.plugins import TrainerPlugin
from rlcade.logger import get_logger

logger = get_logger(__name__)


def _cuda_available() -> bool:
    return torch.cuda.is_available()


class MemoryProfilerPlugin(TrainerPlugin):
    """Profile CUDA memory over a window of training steps."""

    def __init__(
        self,
        start_step: int = 1,
        end_step: int = 10,
        output_file: str = "memory_snapshot.pickle",
        max_entries: int = 100000,
    ):
        self.start_step = start_step
        self.end_step = end_step
        self.output_file = output_file
        self.max_entries = max_entries
        self._recording = False

    def _start_recording(self, iteration: int) -> None:
        logger.info("memory_profiler: start recording at step %d", iteration)
        torch.cuda.memory._record_memory_history(max_entries=self.max_entries)
        torch.cuda.reset_peak_memory_stats()
        self._recording = True

    def _stop_recording(self) -> None:
        torch.cuda.memory._dump_snapshot(self.output_file)
        logger.info(
            "memory_profiler: snapshot saved to %s (peak alloc %.2f MB)",
            self.output_file,
            torch.cuda.max_memory_allocated() / (1024 * 1024),
        )
        torch.cuda.memory._record_memory_history(enabled=None)
        self._recording = False

    def on_setup(self, trainer) -> None:
        pass

    def on_step_start(self, trainer, iteration: int) -> None:
        if iteration == self.start_step and not self._recording and _cuda_available():
            self._start_recording(iteration)

    def on_step_end(self, trainer, iteration: int, summary: dict[str, float] | None) -> None:
        if self._recording and iteration >= self.end_step:
            logger.info("memory_profiler: stop recording at step %d", iteration)
            self._stop_recording()

    def on_eval(self, trainer, iteration: int, scores: list[float]) -> None:
        pass

    def on_done(self, trainer) -> None:
        if self._recording and _cuda_available():
            logger.info("memory_profiler: stop recording at shutdown")
            self._stop_recording()
