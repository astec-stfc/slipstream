"""Fixed wake windows, cloud-in-cell deposit and overflow monitoring for ``LinacTracker`` wakes.

A fixed window makes the wake binning static in shape and free of host synchronisation (so it can be captured in a CUDA graph);
cloud-in-cell (CIC) deposit and readback give the wakes a gradient to the longitudinal position and are second-order accurate where
the default nearest-grid-point (NGP) deposit is first-order.
"""
import math

import numpy as np
import pytest
import torch

from slipstream.constants import C_LIGHT
from slipstream.tracking.native.lattice_import import ElementSpec
from slipstream.tracking.native.linac import LinacTracker
from slipstream.wakefields.shortrange import ShortRangeWake, _bin_nodes, _deposit
from slipstream.wakefields.transverse import TransverseWake

F64 = torch.float64
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
T_WAKE = np.linspace(0.0, 8e-12, 1000)
CHARGE_C = 200e-12


def _z_wake(**kwargs):
    return ShortRangeWake(T_WAKE, -5e14 * np.exp(-T_WAKE / 1e-12), **kwargs)


def _x_wake(**kwargs):
    return TransverseWake(T_WAKE, 2e16 * np.exp(-T_WAKE / 1e-12), **kwargs)


def _line(voltage_V=20e6, transverse=True):
    rf = ElementSpec(
        "L1", "rf", 1.0, voltage_V=voltage_V, phase_rad=0.3, frequency_Hz=2.998e9, n_kicks=4,
        wake=_z_wake(), transverse_wake_x=_x_wake() if transverse else None,
    )
    return [ElementSpec("D0", "drift", 0.3), rf, ElementSpec("D1", "drift", 0.3)]


def _beam(n, device="cpu", seed=1):
    generator = torch.Generator().manual_seed(seed)
    scales = torch.tensor([3e-4, 2e-5, 3e-4, 2e-5, 0.0, 1e-4], dtype=F64)
    beam = torch.randn(n, 6, generator=generator, dtype=F64) * scales
    beam[:, 4] = torch.randn(n, generator=generator, dtype=F64) * 1e-12 * C_LIGHT  # about 1 ps rms
    return beam.to(device)


def _tracker(line, device="cpu"):
    return LinacTracker(line, 100e6, device=device, dtype=F64, always_change_p0=False, bunch_charge_C=CHARGE_C)


def _tau_s(n=500, sigma_s=1e-12, seed=2):
    return torch.randn(n, generator=torch.Generator().manual_seed(seed), dtype=F64) * sigma_s


def test_a_covering_fixed_window_reproduces_the_dynamic_window():
    tau = _tau_s()
    q = CHARGE_C / tau.numel()
    dynamic = _z_wake().voltage_at(tau, q)
    wake = _z_wake()
    wake.set_fixed_bin_window(-6e-12, 6e-12)
    wake.prepare(F64, "cpu")
    torch.testing.assert_close(wake.voltage_at(tau, q), dynamic, rtol=1e-12, atol=1e-9 * float(dynamic.abs().max()))


def test_cic_conserves_charge_and_equals_ngp_at_the_nodes():
    dt = T_WAKE[1] - T_WAKE[0]
    tau = _tau_s() * 0.5
    idx, weight = _bin_nodes(tau, dt, -400, "cic")
    grid, inside = _deposit(idx, weight, 800)
    assert bool(inside.all()) and math.isclose(float(grid.sum()), tau.numel(), rel_tol=1e-12)
    assert torch.all((weight >= 0) & (weight <= 1))
    nodes = torch.arange(-30, 30, dtype=F64) * dt
    wake_cic, wake_ngp = _z_wake(deposit="cic"), _z_wake()
    for wake in (wake_cic, wake_ngp):
        wake.set_fixed_bin_window(-6e-12, 6e-12)
        wake.prepare(F64, "cpu")
    torch.testing.assert_close(wake_cic.voltage_at(nodes, 1e-12), wake_ngp.voltage_at(nodes, 1e-12), rtol=1e-10, atol=0)


def test_cic_is_second_order_and_ngp_first_order_against_the_exact_pairwise_sum():
    sigma, decay, amplitude, n = 1e-12, 2e-12, 1e14, 2000
    tau = torch.randn(n, generator=torch.Generator().manual_seed(4), dtype=F64) * sigma
    q = 1e-12 / n
    kernel = lambda t: np.where(t >= 0, amplitude * t * np.exp(-t / decay), 0.0)  # noqa: E731  continuous, causal
    reference = q * kernel((tau[:, None] - tau[None, :]).numpy()).sum(1)
    errors = {"ngp": [], "cic": []}
    for dt in (0.2 * sigma, 0.1 * sigma, 0.05 * sigma):
        t = np.arange(0.0, 12 * decay, dt)
        for deposit in errors:
            wake = ShortRangeWake(t, kernel(t), deposit=deposit)
            wake.set_fixed_bin_window(-6 * sigma, 6 * sigma)
            wake.prepare(F64, "cpu")
            v = wake.voltage_at(tau, q).numpy()
            errors[deposit].append(np.linalg.norm(v - reference) / np.linalg.norm(reference))
    assert all(c < n_ / 5 for c, n_ in zip(errors["cic"], errors["ngp"]))
    assert errors["cic"][0] / errors["cic"][2] > 12      # about 16 for second order over two halvings
    assert 3 < errors["ngp"][0] / errors["ngp"][2] < 6   # about 4 for first order


def test_ngp_has_no_gradient_to_tau_and_cic_matches_finite_differences():
    wake_ngp, wake_cic = _z_wake(), _z_wake(deposit="cic")
    for wake in (wake_ngp, wake_cic):
        wake.set_fixed_bin_window(-6e-12, 6e-12)
        wake.prepare(F64, "cpu")
    tau = _tau_s(300)
    weights = torch.randn(300, generator=torch.Generator().manual_seed(6), dtype=F64)
    direction = torch.randn(300, generator=torch.Generator().manual_seed(7), dtype=F64)

    def loss(wake, t):
        return (wake.voltage_at(t, 1e-12) * weights).sum() * 1e-6

    leaf = tau.clone().requires_grad_(True)
    value = loss(wake_ngp, leaf)
    grad_ngp = torch.autograd.grad(value, leaf, allow_unused=True)[0] if value.requires_grad else None
    assert grad_ngp is None or float(grad_ngp.abs().max()) == 0.0

    leaf = tau.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(loss(wake_cic, leaf), leaf)
    shift = 1e-18  # far below a bin (8e-15 s), so almost no particle crosses a node
    fd = (loss(wake_cic, tau + shift * direction) - loss(wake_cic, tau - shift * direction)) / (2.0 * shift)
    assert math.isclose(float((grad * direction).sum()), float(fd), rel_tol=1e-6)
    assert float(grad.abs().max()) > 0.0


def test_transverse_cic_gradient_to_tau_and_offset_matches_finite_differences():
    wake = _x_wake(deposit="cic")
    wake.set_fixed_bin_window(-6e-12, 6e-12)
    wake.prepare(F64, "cpu")
    tau = _tau_s(300)
    offset = torch.randn(300, generator=torch.Generator().manual_seed(8), dtype=F64) * 1e-4
    weights = torch.randn(300, generator=torch.Generator().manual_seed(6), dtype=F64)
    d_tau = torch.randn(300, generator=torch.Generator().manual_seed(7), dtype=F64)
    d_off = torch.randn(300, generator=torch.Generator().manual_seed(9), dtype=F64)

    def loss(t, x):
        return (wake.kick_at(t, x, 1e-12) * weights).sum() * 1e-3

    t0, x0 = tau.clone().requires_grad_(True), offset.clone().requires_grad_(True)
    g_tau, g_off = torch.autograd.grad(loss(t0, x0), [t0, x0])
    fd_tau = (loss(tau + 1e-18 * d_tau, offset) - loss(tau - 1e-18 * d_tau, offset)) / 2e-18
    fd_off = (loss(tau, offset + 1e-9 * d_off) - loss(tau, offset - 1e-9 * d_off)) / 2e-9
    assert math.isclose(float((g_tau * d_tau).sum()), float(fd_tau), rel_tol=1e-6)
    assert math.isclose(float((g_off * d_off).sum()), float(fd_off), rel_tol=1e-6)


def test_overflow_is_the_largest_fraction_outside_the_window_and_resets():
    wake = _z_wake(monitor_overflow=True)
    wake.set_fixed_bin_window(-1e-12, 1e-12)
    wake.prepare(F64, "cpu")
    tau = torch.linspace(-3e-12, 3e-12, 1001, dtype=F64)
    wake.voltage_at(tau, 1e-15)
    dt = T_WAKE[1] - T_WAKE[0]
    idx = torch.round(tau / dt).to(torch.int64) - wake._fixed_lo_idx
    expected = float(((idx < 0) | (idx >= wake._fixed_n_bins)).to(F64).mean())
    assert math.isclose(float(wake.overflow), expected, rel_tol=1e-12) and 0.5 < expected < 0.8
    wake.voltage_at(tau * 0.1, 1e-15)  # a narrower bunch must not lower the running maximum
    assert math.isclose(float(wake.overflow), expected, rel_tol=1e-12)
    wake.reset_overflow()
    assert float(wake.overflow) == 0.0
    assert _z_wake().overflow is None


def test_with_wake_windows_reproduces_the_default_tracker_and_leaves_the_original_wakes_alone():
    beam = _beam(1500)
    line = _line()
    expected, expected_energy = _tracker(_line()).track(beam)
    fixed = _tracker(line).with_wake_windows(particles6=beam, margin=1.0, monitor_overflow=True)
    out, energy = fixed.track(beam)
    torch.testing.assert_close(out, expected, rtol=1e-12, atol=1e-17)
    torch.testing.assert_close(energy, expected_energy, rtol=1e-14, atol=0)
    assert float(fixed.wake_overflow()) == 0.0
    wake, transverse = line[1].wake, line[1].transverse_wake_x
    assert wake._fixed_n_bins is None and transverse._fixed_n_bins is None and wake.deposit == "ngp"
    fixed.reset_wake_overflow()
    assert _tracker(line).wake_overflow() is None


def test_with_wake_windows_takes_an_explicit_window_and_validates_its_arguments():
    tracker = _tracker(_line())
    with pytest.raises(ValueError, match="exactly one"):
        tracker.with_wake_windows()
    with pytest.raises(ValueError, match="exactly one"):
        tracker.with_wake_windows(window_s=(-1e-11, 1e-11), particles6=_beam(10))
    with pytest.raises(ValueError, match="deposit"):
        tracker.with_wake_windows(window_s=(-1e-11, 1e-11), deposit="tsc")
    explicit = tracker.with_wake_windows(window_s=(-8e-12, 8e-12))
    beam = _beam(800)
    torch.testing.assert_close(explicit.track(beam)[0], _tracker(_line()).track(beam)[0], rtol=1e-12, atol=1e-17)


def test_wake_gradient_to_tau_through_a_tracker_needs_cic():
    beam = _beam(300)
    weights = torch.randn(300, generator=torch.Generator().manual_seed(6), dtype=F64)
    direction = torch.randn(300, generator=torch.Generator().manual_seed(7), dtype=F64)
    results = {}
    for deposit in ("ngp", "cic"):
        tracker = _tracker(_line(voltage_V=0.0, transverse=False)).with_wake_windows(particles6=beam, margin=1.0, deposit=deposit)

        def loss(b):
            return (tracker.track(b)[0][:, 5] * weights).sum() * 1e3

        leaf = beam.clone().requires_grad_(True)
        (grad,) = torch.autograd.grad(loss(leaf), leaf)
        shift = 1e-11  # metres of tau: about 0.004 of a wake bin
        fd = (loss(beam + shift * direction.unsqueeze(1) * torch.tensor([0, 0, 0, 0, 1.0, 0], dtype=F64))
              - loss(beam - shift * direction.unsqueeze(1) * torch.tensor([0, 0, 0, 0, 1.0, 0], dtype=F64))) / (2.0 * shift)
        results[deposit] = (float(grad[:, 4].abs().max()), float((grad[:, 4] * direction).sum()), float(fd))
    assert results["ngp"][0] == 0.0
    assert results["cic"][0] > 0.0 and math.isclose(results["cic"][1], results["cic"][2], rel_tol=1e-5)


def test_dynamic_window_query_is_safe_on_a_cpu_only_torch_build(monkeypatch):
    def unavailable():
        raise RuntimeError("Tried to instantiate dummy base class _cuda_isCurrentStreamCapturing")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", unavailable)
    assert torch.isfinite(_z_wake().voltage_at(_tau_s(50), 1e-12)).all()
    out, _ = _tracker(_line()).track(_beam(64))
    assert torch.isfinite(out).all()


@needs_cuda
@pytest.mark.parametrize("deposit", ["ngp", "cic"])
def test_a_pass_with_longitudinal_and_transverse_wakes_captures_and_replays_the_eager_result(deposit):
    beam = _beam(1024, "cuda")
    tracker = _tracker(_line(), "cuda").with_wake_windows(particles6=beam, margin=1.0, deposit=deposit, monitor_overflow=True)
    static = beam.clone().requires_grad_(True)
    weights = torch.randn(beam.shape[0], 6, generator=torch.Generator().manual_seed(3), dtype=F64).to("cuda")

    def step(x=static):
        loss = (tracker.track(x)[0] * weights).square().sum() * 1e6
        loss.backward()
        return loss

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            static.grad = None
            step()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    static.grad = None
    tracker.reset_wake_overflow()
    with torch.cuda.graph(graph):
        captured = step()
    for scale in (1.0, 1.05):
        x = (beam * scale).requires_grad_(True)
        expected = step(x)
        static.data.copy_(beam * scale)
        graph.replay()
        assert math.isclose(float(captured.detach()), float(expected.detach()), rel_tol=1e-9)
        torch.testing.assert_close(static.grad, x.grad, rtol=1e-6, atol=1e-9 * float(x.grad.abs().max()))
    assert float(tracker.wake_overflow()) == 0.0


@needs_cuda
def test_dynamic_wake_window_refuses_capture_with_a_clear_error():
    tracker = _tracker(_line(transverse=False), "cuda")
    beam = _beam(256, "cuda")
    with pytest.raises(RuntimeError, match="dynamic wake window"):
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            tracker.track(beam)
    torch.cuda.synchronize()  # the context must still be usable


@needs_cuda
def test_transverse_wake_without_a_window_refuses_capture_with_a_clear_error():
    tracker = _tracker(_line(), "cuda").with_wake_windows(window_s=(-8e-12, 8e-12))
    unwindowed = _tracker(_line(), "cuda")
    beam = _beam(256, "cuda")
    with pytest.raises(RuntimeError, match="no bin window yet"):
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            unwindowed.track(beam)
    torch.cuda.synchronize()
    assert torch.isfinite(tracker.track(beam)[0]).all()
