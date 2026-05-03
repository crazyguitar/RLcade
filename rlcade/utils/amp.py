"""Mixed precision training utilities."""

import torch


def resolve_amp_device_type(device: torch.device | str) -> str:
    """Extract device type string for torch.amp.autocast.

    torch.device('cuda:0') -> 'cuda', 'cpu' -> 'cpu'
    """
    if isinstance(device, torch.device):
        return device.type
    return str(device).split(":")[0]


def create_grad_scaler(device_type: str, enabled: bool) -> torch.amp.GradScaler:
    """Create a GradScaler. Only active for float16 on CUDA.

    bfloat16 (CPU/MPS) has sufficient dynamic range and doesn't need scaling.
    When enabled=False, all scaler methods (scale, step, update) are no-ops.
    """
    use_scaler = enabled and device_type == "cuda"
    return torch.amp.GradScaler(enabled=use_scaler)
