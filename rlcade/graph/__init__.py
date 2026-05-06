import functools
import gc

import torch
import torch.nn as nn

from rlcade.logger import get_logger

logger = get_logger(__name__)


def gc_disabled(fn):
    """Decorator that disables GC for the duration of the call."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        gc.disable()
        try:
            return fn(*args, **kwargs)
        finally:
            gc.enable()

    return wrapper


class CUDAGraphWrapper(nn.Module):
    """Wraps an nn.Module to capture and replay its forward pass as a CUDA graph.

    Falls back to eager execution when:
      * grad is enabled -- replayed outputs are detached from the autograd
        graph, so running under grad mode must bypass the graph for backward
        to propagate to the wrapped module's parameters;
      * the module's output is not a ``torch.Tensor`` (e.g. a ``Distribution``),
        in which case capture is disabled permanently on the first call;
      * the input shapes do not match the shapes used at capture time, in which
        case this specific call runs eager (the captured graph is preserved for
        future matching-shape calls).
    """

    def __init__(self, module: nn.Module, warmup_steps: int = 3, name: str | None = None):
        super().__init__()
        self._module = module
        self._warmup_steps = warmup_steps
        self._graph: torch.cuda.CUDAGraph | None = None
        self._static_inputs: list[torch.Tensor] = []
        self._static_output: torch.Tensor | None = None
        self._disabled = False
        self._name = name or module.__class__.__name__

    def _warmup(self, *args: torch.Tensor) -> None:
        """Run eager iterations to settle allocations."""
        for _ in range(self._warmup_steps):
            self._module(*args)

    @gc_disabled
    def _capture(self, *args: torch.Tensor) -> None:
        """Record the forward pass into a CUDA graph."""
        self._static_inputs = [a.clone() for a in args]
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._static_output = self._module(*self._static_inputs)

    def warmup(self, *args: torch.Tensor) -> None:
        """Warmup then capture on a dedicated side stream."""
        side_stream = torch.cuda.Stream()
        curr_stream = torch.cuda.current_stream()
        side_stream.wait_stream(curr_stream)

        with torch.cuda.stream(side_stream):
            self._warmup(*args)
            torch.cuda.synchronize()
            self._capture(*args)
            torch.cuda.synchronize()

        curr_stream.wait_stream(side_stream)

    def forward(self, *args: torch.Tensor, **kwargs):
        if self._disabled or torch.is_grad_enabled() or kwargs:
            return self._module(*args, **kwargs)

        if self._graph is None:
            in_shapes = [tuple(a.shape) for a in args]
            logger.info("[cuda-graph:%s] capturing for input shapes %s", self._name, in_shapes)
            self.warmup(*args)
            # Graphs replay a fixed output buffer. Non-Tensor outputs (e.g.
            # torch.distributions objects) cannot be cloned/replayed safely,
            # so disable capture permanently and always run eager.
            if not isinstance(self._static_output, torch.Tensor):
                logger.info(
                    "[cuda-graph:%s] disabled: output is %s, not a Tensor",
                    self._name,
                    type(self._static_output).__name__,
                )
                self._disabled = True
                self._graph = None
                self._static_inputs = []
                self._static_output = None
                return self._module(*args)
            logger.info("[cuda-graph:%s] captured, replay active", self._name)

        if any(s.shape != a.shape for s, a in zip(self._static_inputs, args)):
            logger.debug(
                "[cuda-graph:%s] shape mismatch -- eager fallback (captured=%s, got=%s)",
                self._name,
                [tuple(s.shape) for s in self._static_inputs],
                [tuple(a.shape) for a in args],
            )
            return self._module(*args)

        for static, new in zip(self._static_inputs, args):
            static.copy_(new)
        self._graph.replay()
        return self._static_output.clone()

    @property
    def module(self) -> nn.Module:
        """Return the underlying module."""
        return self._module

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        # Peel through torch.compile (`_orig_mod`) and DDP (`.module`) wrappers
        # to find `name` on the underlying module. DDP does not forward
        # attribute access; OptimizedModule does.
        m = super().__getattr__("_module")
        while m is not None:
            try:
                return getattr(m, name)
            except AttributeError:
                inner = getattr(m, "_orig_mod", None) or getattr(m, "module", None)
                m = inner if isinstance(inner, nn.Module) and inner is not m else None
        raise AttributeError(f"'{type(self).__name__}' object has no attribute {name!r}")
