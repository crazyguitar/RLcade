"""Pinned-memory staging buffer for async CPU↔GPU transfers."""

import numpy as np
import torch


class PinMemory:
    """Reusable pinned CPU buffers for async CPU↔GPU transfers."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._bufs: dict[tuple, torch.Tensor] = {}

    def _buf(self, shape, dtype) -> torch.Tensor:
        key = (tuple(shape), dtype)
        if key not in self._bufs:
            self._bufs[key] = torch.empty(shape, dtype=dtype, pin_memory=True)
        return self._bufs[key]

    def reset(self) -> None:
        """Drop cached pinned buffers (call on env swap if obs shape changes)."""
        self._bufs.clear()

    @staticmethod
    def _same_device(a: torch.device, b: torch.device) -> bool:
        # torch.device("cuda") != torch.device("cuda:0"); compare by type + index.
        if a.type != b.type:
            return False
        return a.index is None or b.index is None or a.index == b.index

    def to(self, x, device) -> torch.Tensor:
        """Async transfer x to device via a pinned staging buffer."""
        device = torch.device(device)
        if not isinstance(x, torch.Tensor):
            # Intentional uint8→float32 promotion for typical frame-based obs.
            x = torch.from_numpy(np.ascontiguousarray(x)).float()
        if self._same_device(x.device, device):
            return x
        # Disabled or no CUDA involved → plain copy
        if not self.enabled or (device.type != "cuda" and x.device.type != "cuda"):
            return x.to(device)

        buf = self._buf(x.shape, x.dtype)
        buf.copy_(x, non_blocking=True)
        # D2H: `buf` is aliased with the internal cache — caller must consume or
        # clone before the next pm.to() of the same (shape, dtype).
        return buf if device.type == "cpu" else buf.to(device, non_blocking=True)
