import torch
import torch.nn as nn

from rlcade.utils import clip_grad_norm_


class TestClipGradNorm:
    def _make_params_with_grad(self, grad_values: list[float]) -> list[nn.Parameter]:
        params = []
        for g in grad_values:
            p = nn.Parameter(torch.tensor([1.0]))
            p.grad = torch.tensor([g])
            params.append(p)
        return params

    def test_clips_when_above_max(self):
        params = self._make_params_with_grad([3.0, 4.0])
        # norm = sqrt(9+16) = 5.0, max_norm = 1.0 → should clip
        total_norm = clip_grad_norm_(params, max_norm=1.0)
        assert abs(total_norm.item() - 5.0) < 1e-5
        clipped_norm = torch.sqrt(sum(p.grad.square().sum() for p in params)).item()
        assert abs(clipped_norm - 1.0) < 1e-5

    def test_noop_when_below_max(self):
        params = self._make_params_with_grad([0.3, 0.4])
        # norm = 0.5, max_norm = 1.0 → no clipping
        clip_grad_norm_(params, max_norm=1.0)
        assert abs(params[0].grad.item() - 0.3) < 1e-5
        assert abs(params[1].grad.item() - 0.4) < 1e-5

    def test_empty_grads_returns_zero(self):
        p = nn.Parameter(torch.tensor([1.0]))
        # No grad set
        total_norm = clip_grad_norm_([p], max_norm=1.0)
        assert total_norm.item() == 0.0

    def test_single_tensor_input(self):
        p = nn.Parameter(torch.tensor([6.0]))
        p.grad = torch.tensor([6.0])
        clip_grad_norm_(p, max_norm=3.0)
        assert abs(p.grad.item() - 3.0) < 1e-5
