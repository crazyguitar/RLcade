"""VizTracer plugin for profiling training runs."""

from viztracer import VizTracer

from rlcade.plugins import TrainerPlugin
from rlcade.logger import get_logger

logger = get_logger(__name__)


class VizTracerPlugin(TrainerPlugin):
    """Profile a window of training steps with VizTracer.

    Args:
        start_step: Iteration to start profiling (inclusive).
        end_step: Iteration to stop profiling (inclusive).
        **kwargs: Forwarded to VizTracer (output_file, max_stack_depth,
                  log_gc, ignore_c_function, etc.).
    """

    def __init__(self, start_step: int = 1, end_step: int = 10, **kwargs):
        self.start_step = start_step
        self.end_step = end_step
        kwargs.setdefault("output_file", "training.json")
        self.tracer = VizTracer(**kwargs)
        self.active = False

    def on_step_start(self, trainer, iteration: int) -> None:
        if iteration == self.start_step and not self.active:
            logger.info("VizTracer: start profiling at step %d", iteration)
            self.tracer.start()
            self.active = True

    def on_step_end(self, trainer, iteration: int, summary: dict[str, float] | None) -> None:
        if self.active and iteration >= self.end_step:
            logger.info("VizTracer: stop profiling at step %d", iteration)
            self.tracer.stop()
            self.tracer.save()
            self.active = False

    def on_done(self, trainer) -> None:
        if self.active:
            self.tracer.stop()
            self.tracer.save()
            self.active = False
