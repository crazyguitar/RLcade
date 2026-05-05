"""Nsight Systems profiling plugin.

Profiles a window of training steps using the CUDA profiler API and NVTX
annotations, following the NeMo nsight.py pattern. When launched under
``nsys profile``, only the steps in [start_step, end_step] are captured.

Usage:
    nsys profile -o trace python -m rlcade.training --nsys --nsys-start 5 --nsys-end 10 ...
"""

import torch

from rlcade.plugins import TrainerPlugin
from rlcade.logger import get_logger

logger = get_logger(__name__)


def _cuda_available() -> bool:
    return torch.cuda.is_available()


class NsysPlugin(TrainerPlugin):
    """Profile a window of training steps with Nsight Systems.

    Args:
        start_step: Iteration to start profiling (inclusive).
        end_step: Iteration to stop profiling (inclusive).
    """

    def __init__(self, start_step: int = 1, end_step: int = 10):
        self.start_step = start_step
        self.end_step = end_step
        self._profiling = False

    def on_step_start(self, trainer, iteration: int) -> None:
        if iteration == self.start_step and not self._profiling and _cuda_available():
            logger.info("nsys: start profiling at step %d", iteration)
            torch.cuda.cudart().cudaProfilerStart()
            self._profiling = True
        if self._profiling:
            torch.cuda.nvtx.range_push(f"step_{iteration}")

    def on_step_end(self, trainer, iteration: int, summary: dict[str, float] | None) -> None:
        if self._profiling:
            torch.cuda.nvtx.range_pop()
        if self._profiling and iteration >= self.end_step:
            logger.info("nsys: stop profiling at step %d", iteration)
            torch.cuda.cudart().cudaProfilerStop()
            self._profiling = False

    def on_done(self, trainer) -> None:
        if self._profiling and _cuda_available():
            torch.cuda.cudart().cudaProfilerStop()
            self._profiling = False
