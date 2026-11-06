"""Gradients through the collective kernels and ``LinacTracker`` passes.

The fast CSR and LSC kernels must stay differentiable with a scalar macro-charge on every device, the binomial smoother must
equal its pass-by-pass definition, the exact reference kernels must be backpropagable, and a charged ``LinacTracker`` pass must be
replayable and capturable in a CUDA graph. Finite differences use shifts far below one grid cell, so no particle changes bin.
"""
import math

import pytest
import torch

from slipstream.tracking.native import collective
from slipstream.tracking.native.collective import (
    csr_steady_state_delta_kick, csr_steady_state_delta_kick_fast, lsc_delta_kick_fast,
)
from slipstream.tracking.native.lattice_import import ElementSpec
from slipstream.tracking.native.linac import LinacTracker

ENERGY_EV, CHARGE_C, MC2_EV = 150e6, 100e-12, 0.511e6
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"))]
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _bunch(n, device, seed=3):
    generator = torch.Generator().manual_seed(seed)
    draw = lambda scale: (torch.randn(n, generator=generator, dtype=torch.float64) * scale).to(device)  # noqa: E731
    return draw(30e-6), draw(2e-4), draw(2e-4)


def _directional_error(loss, x, shift_m=1e-11, seed=11):
    """Relative difference between the autograd and central finite-difference derivative of ``loss`` along a random direction."""
    direction = torch.randn(x.shape, generator=torch.Generator().manual_seed(seed), dtype=x.dtype).to(x.device)
    x = x.detach().requires_grad_(True)
    (grad,) = torch.autograd.grad(loss(x), x)
    with torch.no_grad():
        finite_difference = (loss(x + shift_m * direction) - loss(x - shift_m * direction)) / 2.0
    autograd = (grad * direction).sum() * shift_m
    return float((autograd - finite_difference).abs() / finite_difference.abs())


def _weights(n, device):
    return torch.randn(n, generator=torch.Generator().manual_seed(5), dtype=torch.float64).to(device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("slippage_m", [None, 5e-5])
def test_fast_csr_scalar_charge_gradient_matches_finite_difference(device, slippage_m):
    tau, _, _ = _bunch(2048, device)
    energy = torch.tensor(ENERGY_EV, dtype=torch.float64, device=device)
    weights = _weights(tau.numel(), device)

    def loss(t):
        kick = csr_steady_state_delta_kick_fast(t, CHARGE_C / t.numel(), 1.5, energy, 0.1, slippage_m=slippage_m)
        return (kick * weights).sum()

    assert loss(tau.clone().requires_grad_(True)).requires_grad
    assert _directional_error(loss, tau) < 1e-6


@pytest.mark.parametrize("device", DEVICES)
def test_fast_lsc_scalar_charge_gradient_matches_finite_difference(device):
    tau, x, y = _bunch(2048, device)
    energy = torch.tensor(ENERGY_EV, dtype=torch.float64, device=device)
    mc2 = torch.tensor(MC2_EV, dtype=torch.float64, device=device)
    weights = _weights(tau.numel(), device)

    def loss(t):
        kick = lsc_delta_kick_fast(t, CHARGE_C / t.numel(), x, y, energy, mc2, 1.0, pipe_radius_b_m=0.01,
                                   n_bins=256, smooth_passes=64)
        return (kick * weights).sum() * 1e6

    assert loss(tau.clone().requires_grad_(True)).requires_grad
    assert _directional_error(loss, tau) < 1e-6


@pytest.mark.parametrize(("n", "passes"), [(64, 8), (257, 100), (2048, 512)])
def test_dense_smoother_equals_the_pass_loop_and_its_gradient(n, passes):
    lam = torch.rand(n, dtype=torch.float64)
    lam[: n // 8] *= 0.01
    weights = torch.randn(n, dtype=torch.float64)

    def loop(v):
        for _ in range(passes):
            pad = torch.zeros_like(v[:1])
            padded = torch.cat([pad, v, pad])
            v = 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
        return v

    reference_in = lam.clone().requires_grad_(True)
    dense_in = lam.clone().requires_grad_(True)
    reference = loop(reference_in)
    dense = collective._smooth_bins(dense_in, passes)
    torch.testing.assert_close(dense, reference, rtol=0, atol=1e-13)
    (reference * weights).sum().backward()
    (dense * weights).sum().backward()
    torch.testing.assert_close(dense_in.grad, reference_in.grad, rtol=0, atol=1e-13)


def test_dense_smoother_float32_is_closer_to_float64_than_the_pass_loop_budget():
    lam = torch.rand(2048, dtype=torch.float64)
    exact = collective._smooth_bins(lam, 512)
    single = collective._smooth_bins(lam.float(), 512)
    assert single.dtype == torch.float32
    assert float((single.double() - exact).abs().max()) < 1e-6


def test_smoother_falls_back_to_the_loop_above_the_grid_cap(monkeypatch):
    lam = torch.rand(collective._DENSE_SMOOTH_MAX_BINS + 5, dtype=torch.float64)
    built = []
    monkeypatch.setattr(collective, "_dense_smoother", lambda *a: built.append(a))
    out = collective._smooth_bins(lam, 12)
    assert not built and out.shape == lam.shape
    assert float(out.sum()) <= float(lam.sum())  # zero padding only removes weight at the edges


def test_default_lsc_matches_the_loop_it_replaces(monkeypatch):
    tau, x, y = _bunch(4096, "cpu")
    energy = torch.tensor(ENERGY_EV, dtype=torch.float64)
    mc2 = torch.tensor(MC2_EV, dtype=torch.float64)
    dense = lsc_delta_kick_fast(tau, CHARGE_C / tau.numel(), x, y, energy, mc2, 1.0, pipe_radius_b_m=0.01)
    monkeypatch.setattr(collective, "_DENSE_SMOOTH_MIN_PASSES", 10 ** 9)
    loop = lsc_delta_kick_fast(tau, CHARGE_C / tau.numel(), x, y, energy, mc2, 1.0, pipe_radius_b_m=0.01)
    torch.testing.assert_close(dense, loop, rtol=0, atol=1e-12 * float(loop.abs().max()))


def test_chunk_steps_agree_between_the_in_place_and_the_autograd_branch():
    tau, _, _ = _bunch(300, "cpu")
    score = torch.rand(300, dtype=torch.float64)
    softening = torch.tensor(2e-6, dtype=torch.float64)
    with torch.no_grad():
        forward_only = collective._csr_chunk_step(tau[:40], tau, score, softening)
    live = collective._csr_chunk_step(tau[:40].clone().requires_grad_(True), tau, score, softening)
    torch.testing.assert_close(live, forward_only, rtol=1e-12, atol=0)

    window = tau[None, :].expand(40, -1).clone()
    h = torch.tensor(5e-6, dtype=torch.float64)
    with torch.no_grad():
        density, deriv = collective._kde_chunk_step(tau[:40], window, 1e-15, h)
    live_density, live_deriv = collective._kde_chunk_step(tau[:40].clone().requires_grad_(True), window, 1e-15, h)
    torch.testing.assert_close(live_density, density, rtol=1e-12, atol=0)
    torch.testing.assert_close(live_deriv, deriv, rtol=1e-12, atol=0)


def test_exact_pairwise_csr_with_windowed_kde_is_backpropagable_and_matches_finite_difference():
    tau, _, _ = _bunch(1200, "cpu")
    energy = torch.tensor(ENERGY_EV, dtype=torch.float64)
    weights = _weights(tau.numel(), "cpu")

    def loss(t):
        kick = csr_steady_state_delta_kick(t, CHARGE_C / t.numel(), 1.5, energy, 0.1, compile_chunks=False)
        return (kick * weights).sum() * 1e6

    assert _directional_error(loss, tau, shift_m=1e-12) < 1e-6  # below the closest pair separation: the one-sided sum jumps when a pair crosses


_LINE = [ElementSpec("D0", "drift", 0.2), ElementSpec("Q1", "quad", 0.1, k1=2.0), ElementSpec("D1", "drift", 0.4)]
for _i, _angle in enumerate((0.08, -0.08, -0.08, 0.08)):
    _LINE += [ElementSpec(f"B{_i}", "bend", 0.2, angle=_angle, csr=True), ElementSpec(f"DB{_i}", "drift", 0.4)]


def _tracker(device, dtype=torch.float64, lsc=False):
    return LinacTracker(
        _LINE, ENERGY_EV, device=device, dtype=dtype, always_change_p0=False, bunch_charge_C=CHARGE_C, csr=True, lsc=lsc,
        pipe_radius_b_m=0.01 if lsc else None, fast_collective_effects=True, compile=False, compile_collective_chunks=False,
    )


def _beam(n, device, dtype=torch.float64):
    scales = torch.tensor([2e-4, 2e-5, 2e-4, 2e-5, 30e-6, 1e-4], dtype=dtype)
    return (torch.randn(n, 6, generator=torch.Generator().manual_seed(9), dtype=dtype) * scales).to(device)


@pytest.mark.parametrize("device", DEVICES)
def test_two_charged_passes_then_one_backward_sum_their_gradients(device):
    tracker = _tracker(device)
    beam = _beam(256, device).requires_grad_(True)
    first = tracker.track(beam)[0][:, 5].mean()
    second = tracker.track(beam * 1.01)[0][:, 5].mean()
    (g_first,) = torch.autograd.grad(first, beam, retain_graph=True)
    (g_second,) = torch.autograd.grad(second, beam, retain_graph=True)
    (first + second).backward()
    assert torch.isfinite(beam.grad).all() and float(beam.grad.abs().max()) > 0.0
    torch.testing.assert_close(beam.grad, g_first + g_second, rtol=1e-12, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_reference_energy_update_does_not_invalidate_an_earlier_pass(device):
    tracker = _tracker(device)
    beam = _beam(128, device).requires_grad_(True)
    loss = tracker.track(beam)[0][:, 5].mean()
    tracker._set_reference_energy(ENERGY_EV * 1.1)
    loss.backward()
    assert torch.isfinite(beam.grad).all()


@needs_cuda
def test_graph_dispatch_is_used_without_gradients_and_bypassed_with_them():
    n = 1531  # an unusual size so the graph key is new
    tau, _, _ = _bunch(n, "cuda")
    energy = torch.tensor(ENERGY_EV, dtype=torch.float64, device="cuda")
    cache = collective._csr_fast_graph_cache
    before = len(cache)
    with torch.no_grad():
        plain = csr_steady_state_delta_kick_fast(tau, CHARGE_C / n, 1.5, energy, 0.1)
    assert len(cache) == before + 1
    live = csr_steady_state_delta_kick_fast(tau.clone().requires_grad_(True), CHARGE_C / n, 1.5, energy, 0.1)
    assert live.requires_grad and len(cache) == before + 1
    torch.testing.assert_close(live.detach(), plain, rtol=1e-10, atol=0)


def _screen(p, bins=32):
    uv = (torch.stack([p[:, 0], p[:, 2]], dim=1) / 0.008 + 0.5) * (bins - 1)
    ij = uv.floor().clamp(0, bins - 2).long()
    w = uv - ij.to(uv.dtype)
    out = p.new_zeros(bins * bins)
    for a in (0, 1):
        for b in (0, 1):
            wx = w[:, 0] if a else 1 - w[:, 0]
            wy = w[:, 1] if b else 1 - w[:, 1]
            out.scatter_add_(0, (ij[:, 0] + a) * bins + ij[:, 1] + b, wx * wy)
    return out / p.shape[0]


@needs_cuda
def test_whole_iteration_cuda_graph_replays_the_eager_forward_and_backward():
    tracker = _tracker("cuda", lsc=True)
    beam = _beam(1024, "cuda")

    def step(x):
        out, _ = tracker.track(x)
        loss = _screen(out).square().sum()
        loss.backward()
        return loss

    def eager(x0):
        x = x0.clone().requires_grad_(True)
        return float(step(x).detach()), x.grad

    static = beam.clone().requires_grad_(True)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            static.grad = None
            step(static)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    static.grad = None
    with torch.cuda.graph(graph):
        captured_loss = step(static)
    for scale in (1.0, 1.03):
        new = beam * scale
        loss, grad = eager(new)
        static.data.copy_(new)
        graph.replay()
        assert math.isclose(float(captured_loss), loss, rel_tol=1e-8)
        torch.testing.assert_close(static.grad, grad, rtol=1e-6, atol=1e-8 * float(grad.abs().max()))
