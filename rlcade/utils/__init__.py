import torch
import torch.nn as nn

from rlcade.utils.amp import resolve_amp_device_type, create_grad_scaler
from rlcade.utils.pin_memory import PinMemory
from rlcade.utils.progress import ProgressBar
from rlcade.utils.metrics import Metrics
from rlcade.utils.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


def resolve_device(device: str) -> str:
    """Resolve 'auto' to the best available torch device."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def to_tensor(x, device) -> torch.Tensor:
    """Convert numpy array or tensor to float tensor on device."""
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    """Polyak-average source parameters into target: θ_tgt ← τ·θ_src + (1−τ)·θ_tgt."""
    for p, tp in zip(source.parameters(), target.parameters()):
        tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)


def clip_grad_norm_(parameters, max_norm: float, norm_type: float = 2.0):
    """Clip gradient norm, safe for both regular tensors and FSDP2 DTensors.

    Aligned with VeRL's fsdp2_clip_grad_norm_: computes total_norm on the
    parameter device, then clips via PyTorch internals with foreach=None
    (auto-selects the fast path for CUDA DTensors).
    """
    from torch.nn.utils.clip_grad import _clip_grads_with_norm_, _get_total_norm

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        parameters = list(parameters)
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    # error_if_nonfinite=False, foreach=None (auto-select fast path)
    total_norm = _get_total_norm(grads, norm_type, False, None)
    total_norm = total_norm.to(grads[0].device)
    _clip_grads_with_norm_(parameters, max_norm, total_norm, None)
    return total_norm
