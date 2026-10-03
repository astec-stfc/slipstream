"""Optional CUDA/Warp parity tests for collective accelerator primitives."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

# Keep optional Warp/Inductor caches writable on the Windows development
# host.  These are test-process settings only; production code keeps its
# existing cache policy.
_TEST_ROOT = Path(__file__).resolve().parents[1]
os.environ["LOCALAPPDATA"] = str(_TEST_ROOT / ".cache" / "test-localappdata")
os.environ["WARP_CACHE_PATH"] = str(_TEST_ROOT / ".cache" / "test-warp")

from slipstream.tracking.native import collective, warp_kernels


_CUDA_WARP = torch.cuda.is_available() and warp_kernels.available("cuda", torch.float32)
pytestmark = pytest.mark.skipif(not _CUDA_WARP, reason="requires CUDA and warp-lang")


def test_warp_exact_csr_pairwise_matches_torch_reference():
    n = 128
    generator = torch.Generator(device="cuda").manual_seed(8128)
    tau = (torch.randn(n, device="cuda", generator=generator) * 2e-3).contiguous()
    score = (torch.randn(n, device="cuda", generator=generator) * 0.5).contiguous()
    softening = torch.tensor(2.5e-5, device="cuda")

    reference = collective._csr_chunked_pairwise_sum(tau, score, softening, n, compile_chunks=False)
    candidate = warp_kernels.csr_pairwise_sum(tau, score, softening)
    torch.testing.assert_close(candidate, reference, rtol=1e-2, atol=5e-4)


def test_warp_fixed_grid_cic_matches_torch_reference():
    tau = torch.linspace(-2e-3, 2e-3, 257, device="cuda")
    bin_idx, interp_w, valid, _, dtau, n_bins = collective._fixed_grid(tau, 64)
    charge = torch.tensor(1e-12, device="cuda")

    reference = collective._bin_line_density(charge, bin_idx, interp_w, valid, n_bins, dtau, _use_warp=False)
    candidate = warp_kernels.cic_line_density(
        bin_idx.to(torch.int32).contiguous(), interp_w.contiguous(), valid.contiguous(), charge, n_bins, dtau,
    )
    torch.testing.assert_close(candidate, reference, rtol=3e-4, atol=3e-6)


def test_warp_short_range_wake_histogram_matches_torch_reference():
    n_bins = 64
    safe_idx = torch.arange(257, device="cuda", dtype=torch.int32).remainder(n_bins + 1).contiguous()
    reference = torch.zeros(n_bins + 1, device="cuda").scatter_add_(
        0, safe_idx.to(torch.int64), torch.ones(safe_idx.shape[0], device="cuda"),
    )
    candidate = warp_kernels.wake_histogram(safe_idx, n_bins + 1, torch.float32)
    torch.testing.assert_close(candidate, reference)


def test_collective_csr_dispatch_is_float32_only(monkeypatch):
    calls = []

    def fake_csr(tau, score, softening):
        calls.append((tau.dtype, score.dtype, softening.dtype))
        return torch.zeros_like(tau)

    monkeypatch.setattr(collective, "_CSR_WARP_MIN_N", 1)
    monkeypatch.setattr(warp_kernels, "csr_pairwise_sum", fake_csr)

    tau32 = torch.linspace(-1e-3, 1e-3, 32, device="cuda", dtype=torch.float32)
    score32 = torch.ones_like(tau32)
    eps32 = torch.tensor(2.5e-5, device="cuda", dtype=torch.float32)
    collective._csr_chunked_pairwise_sum(tau32, score32, eps32, 32, compile_chunks=False)
    assert calls == [(torch.float32, torch.float32, torch.float32)]

    calls.clear()
    tau64 = tau32.double()
    score64 = score32.double()
    eps64 = eps32.double()
    collective._csr_chunked_pairwise_sum(tau64, score64, eps64, 32, compile_chunks=False)
    assert calls == []


def test_collective_cic_dispatch_converts_indices_to_int32(monkeypatch):
    calls = []

    def fake_cic(bin_idx, interp_w, valid, charge, n_bins, dtau):
        calls.append((bin_idx.dtype, interp_w.dtype, valid.dtype))
        return torch.zeros(n_bins, device=bin_idx.device, dtype=interp_w.dtype)

    monkeypatch.setattr(warp_kernels, "cic_line_density", fake_cic)
    tau = torch.linspace(-2e-3, 2e-3, 32, device="cuda")
    bin_idx, interp_w, valid, _, dtau, n_bins = collective._fixed_grid(tau, 16)
    charge = torch.tensor(1e-12, device="cuda")
    collective._bin_line_density(charge, bin_idx, interp_w, valid, n_bins, dtau)
    assert calls == [(torch.int32, torch.float32, torch.bool)]
