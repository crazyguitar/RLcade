"""Safetensors model export -- flatten nested state_dicts and write."""

from __future__ import annotations

import json
import struct

import torch
from safetensors.torch import load as safetensors_load
from safetensors.torch import save as safetensors_save

from rlcade.checkpoint.checkpoint import Checkpoint

_FORMAT = "rlcade-safetensors-v1"


def _is_state_dict(value) -> bool:
    """True if *value* is a non-empty dict whose leaf values are all tensors."""
    if not isinstance(value, dict) or not value:
        return False
    return all(isinstance(v, torch.Tensor) for v in value.values())


def _flatten(state: dict) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Flatten {model_name: state_dict} to {f'{model_name}.{param}': tensor}.

    Skips entries whose values are not state_dicts (e.g. step, optimizer).
    Returns (flat_dict, ordered_model_names).
    """
    flat: dict[str, torch.Tensor] = {}
    model_names: list[str] = []
    for key, value in state.items():
        if not _is_state_dict(value):
            continue
        model_names.append(key)
        for param_name, tensor in value.items():
            flat[f"{key}.{param_name}"] = tensor.detach().contiguous().cpu()
    return flat, model_names


def save_safetensors(state: dict, url: str, *, step: int = 0) -> None:
    """Flatten model state_dicts into safetensors and write to *url*.

    Non-tensor entries (``step``, ``optimizer``, ``grad_scaler``) are dropped
    from the tensor section. ``step`` is preserved in safetensors metadata.
    """
    flat, model_names = _flatten(state)
    metadata = {
        "format": _FORMAT,
        "step": str(step),
        "models": ",".join(model_names),
    }
    blob = safetensors_save(flat, metadata=metadata)
    with Checkpoint(url).writer() as f:
        f.write(blob)


def _unflatten(flat: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    """Reverse of _flatten: 'a.b.c' -> {'a': {'b.c': tensor}}.

    Splits on the FIRST '.' only, so nested module FQNs (e.g. 'conv.0.weight')
    are preserved inside the per-model state_dict.
    """
    nested: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in flat.items():
        model, _, param = key.partition(".")
        if not param:
            raise ValueError(f"safetensors key {key!r} missing model prefix")
        nested.setdefault(model, {})[param] = tensor
    return nested


def _read_metadata(blob: bytes) -> dict[str, str]:
    """Parse the safetensors 8-byte header length + JSON header.

    Returns the ``__metadata__`` dict, or an empty dict if absent.
    """
    if len(blob) < 8:
        raise ValueError("safetensors blob shorter than 8-byte header length")
    (header_len,) = struct.unpack("<Q", blob[:8])
    if 8 + header_len > len(blob):
        raise ValueError("safetensors header length exceeds blob")
    header = json.loads(blob[8 : 8 + header_len].decode("utf-8"))
    return header.get("__metadata__", {}) or {}


def load_safetensors(
    url: str, device: torch.device
) -> tuple[dict[str, dict[str, torch.Tensor]], int]:
    """Read safetensors from *url*, un-flatten into {model_name: state_dict}.

    Returns ``(state, step)``. Raises ``ValueError`` if the file's metadata
    is missing or the format string does not match.
    """
    with Checkpoint(url).reader() as f:
        blob = f.read()

    flat = safetensors_load(blob)
    metadata = _read_metadata(blob)

    fmt = metadata.get("format")
    if fmt != _FORMAT:
        raise ValueError(
            f"safetensors file {url!r} has unrecognized format {fmt!r}; "
            f"expected {_FORMAT!r}"
        )
    step = int(metadata.get("step", "0"))

    nested = _unflatten(flat)
    nested = {name: {k: v.to(device) for k, v in sd.items()} for name, sd in nested.items()}
    return nested, step
