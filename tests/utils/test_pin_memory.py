import numpy as np
import pytest
import torch

from rlcade.utils import PinMemory

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class TestPinMemoryCPU:
    """Tests that run on CPU-only hosts (no pinning fast path taken)."""

    def test_numpy_to_cpu_returns_float_tensor(self):
        pm = PinMemory()
        x = np.arange(6, dtype=np.float32).reshape(2, 3)
        out = pm.to(x, "cpu")
        assert isinstance(out, torch.Tensor)
        assert out.dtype == torch.float32
        assert out.device.type == "cpu"
        assert torch.equal(out, torch.from_numpy(x))

    def test_numpy_non_float_promoted_to_float32(self):
        pm = PinMemory()
        x = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        out = pm.to(x, "cpu")
        assert out.dtype == torch.float32
        assert torch.equal(out, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    def test_tensor_already_on_target_device_passthrough(self):
        pm = PinMemory()
        x = torch.arange(4, dtype=torch.float32)
        out = pm.to(x, "cpu")
        # CPU→CPU passthrough should return the same tensor object
        assert out is x

    def test_disabled_falls_back_to_plain_to(self):
        pm = PinMemory(enabled=False)
        x = torch.arange(4, dtype=torch.float32)
        out = pm.to(x, "cpu")
        assert out.device.type == "cpu"
        assert torch.equal(out, x)
        # No pinned buffers should ever be allocated
        assert pm._bufs == {}

    def test_disabled_still_converts_numpy_to_tensor(self):
        pm = PinMemory(enabled=False)
        x = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        out = pm.to(x, "cpu")
        assert isinstance(out, torch.Tensor)
        assert out.dtype == torch.float32
        assert torch.equal(out, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        assert pm._bufs == {}

    def test_non_contiguous_numpy_is_copied(self):
        pm = PinMemory()
        # Transposed view is non-contiguous; ascontiguousarray inside must copy
        x = np.arange(12, dtype=np.float32).reshape(3, 4).T
        assert not x.flags["C_CONTIGUOUS"]
        out = pm.to(x, "cpu")
        assert out.is_contiguous()
        assert torch.equal(out, torch.from_numpy(np.ascontiguousarray(x)))

    def test_reset_clears_cached_buffers(self):
        pm = PinMemory()
        # Force a buffer entry via the internal accessor (no CUDA needed).
        pm._buf((2, 3), torch.float32)
        assert len(pm._bufs) == 1
        pm.reset()
        assert pm._bufs == {}


class TestPinMemoryCUDA:
    """Tests that exercise the pinned-buffer H2D/D2H fast path."""

    @requires_cuda
    def test_h2d_returns_fresh_cuda_tensor(self):
        pm = PinMemory()
        x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        out = pm.to(x, "cuda")
        assert out.device.type == "cuda"
        # Value must round-trip correctly
        torch.cuda.synchronize()
        assert torch.equal(out.cpu(), x)

    @requires_cuda
    def test_h2d_numpy_input(self):
        pm = PinMemory()
        x = np.arange(6, dtype=np.float32).reshape(2, 3)
        out = pm.to(x, "cuda")
        torch.cuda.synchronize()
        assert out.device.type == "cuda"
        assert torch.equal(out.cpu(), torch.from_numpy(x))

    @requires_cuda
    def test_h2d_reuses_pinned_buffer_for_same_shape(self):
        pm = PinMemory()
        x1 = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        x2 = torch.arange(6, dtype=torch.float32).reshape(2, 3) + 100
        pm.to(x1, "cuda")
        out2 = pm.to(x2, "cuda")
        # Single (shape, dtype) key → single buffer
        assert len(pm._bufs) == 1
        (buf,) = pm._bufs.values()
        assert buf.is_pinned()
        assert buf.shape == (2, 3)
        # Second call's data must actually land in the buffer / output
        torch.cuda.synchronize()
        assert torch.equal(out2.cpu(), x2)
        assert torch.equal(buf, x2)

    @requires_cuda
    def test_sequential_h2d_returns_distinct_cuda_tensors(self):
        # EnvPPO.collect_rollout relies on this: each step's obs tensor is
        # held in a list and must not alias the next step's obs.
        pm = PinMemory()
        x1 = torch.zeros(2, 3, dtype=torch.float32)
        x2 = torch.ones(2, 3, dtype=torch.float32)
        out1 = pm.to(x1, "cuda")
        out2 = pm.to(x2, "cuda")
        assert out1.data_ptr() != out2.data_ptr()
        torch.cuda.synchronize()
        assert torch.equal(out1.cpu(), x1)
        assert torch.equal(out2.cpu(), x2)

    @requires_cuda
    def test_h2d_allocates_distinct_buffers_per_shape(self):
        pm = PinMemory()
        pm.to(torch.zeros(2, 3), "cuda")
        pm.to(torch.zeros(4, 5), "cuda")
        assert len(pm._bufs) == 2

    @requires_cuda
    def test_d2h_returns_pinned_buffer(self):
        pm = PinMemory()
        x = torch.arange(6, dtype=torch.float32, device="cuda").reshape(2, 3)
        out = pm.to(x, "cpu")
        assert out.device.type == "cpu"
        assert out.is_pinned()
        torch.cuda.synchronize()
        assert torch.equal(out, x.cpu())

    @requires_cuda
    def test_disabled_skips_pinning_even_on_cuda(self):
        pm = PinMemory(enabled=False)
        x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        out = pm.to(x, "cuda")
        torch.cuda.synchronize()
        assert out.device.type == "cuda"
        assert torch.equal(out.cpu(), x)
        # No pinned staging buffer allocated
        assert pm._bufs == {}

    @requires_cuda
    def test_tensor_already_on_cuda_passthrough(self):
        pm = PinMemory()
        x = torch.arange(4, dtype=torch.float32, device="cuda")
        out = pm.to(x, "cuda")
        assert out is x

    @requires_cuda
    def test_cuda_passthrough_ignores_missing_index(self):
        # Tensor on "cuda:0" with target "cuda" (no index) must passthrough,
        # not trigger a D2H→pinned→H2D roundtrip.
        pm = PinMemory()
        x = torch.arange(4, dtype=torch.float32, device="cuda:0")
        out = pm.to(x, "cuda")
        assert out is x
        assert pm._bufs == {}
