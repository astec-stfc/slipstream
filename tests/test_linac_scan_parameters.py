"""Tensor-valued scan settings for ``LinacTracker`` and its CUDA-graph capture.

``LinacTracker.with_settings`` must reproduce a tracker rebuilt with the same float values, pass gradients to tensor settings, and
let one captured forward+backward graph replay new settings written into static tensors. Fast CSR and LSC must also capture without
gradients, and the Stupakov drift model must capture; the retarded drift model must fail with a clear message, not a CUDA error.
"""
import math
from dataclasses import replace

import pytest
import torch

from slipstream.tracking.native.errors import FiducialReference
from slipstream.tracking.native.lattice_import import ElementSpec
from slipstream.tracking.native.linac import LinacTracker

ENERGY_EV, CHARGE_C, FREQ_HZ = 150e6, 100e-12, 2.998e9
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
F64 = torch.float64


def _line():
    line = [
        ElementSpec("D0", "drift", 0.3), ElementSpec("Q1", "quad", 0.1, k1=1.5), ElementSpec("S1", "sext", 0.1, k2=4.0),
        ElementSpec("D1", "drift", 0.4), ElementSpec("H1", "hkick", 0.0, kick_x=1e-4),
        ElementSpec("L1", "rf", 1.0, voltage_V=20e6, phase_rad=0.3, frequency_Hz=FREQ_HZ, n_kicks=4),
        ElementSpec("D2", "drift", 0.4),
    ]
    for i, angle in enumerate((0.08, -0.08, -0.08, 0.08)):
        line += [ElementSpec(f"B{i}", "bend", 0.2, angle=angle, csr=True), ElementSpec(f"DB{i}", "drift", 0.4)]
    return line


def _with(line, **values):
    """``line`` with each named element's fields replaced: ``_with(line, Q1=dict(k1=2.0))``."""
    return [replace(spec, **values[spec.name]) if spec.name in values else spec for spec in line]


def _beam(n, device="cpu"):
    scales = torch.tensor([2e-4, 2e-5, 2e-4, 2e-5, 30e-6, 1e-4], dtype=F64)
    return (torch.randn(n, 6, generator=torch.Generator().manual_seed(9), dtype=F64) * scales).to(device)


def _tracker(line, device="cpu", **kwargs):
    options = dict(always_change_p0=False, compile=False, compile_collective_chunks=False)
    options.update(kwargs)
    return LinacTracker(line, ENERGY_EV, device=device, dtype=F64, **options)


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


NEW = dict(Q1=dict(k1=2.2), S1=dict(k2=-3.0), H1=dict(kick_x=-2e-4), L1=dict(voltage_V=25e6, phase_rad=0.1))


@pytest.mark.parametrize("as_tensor", [False, True])
def test_with_settings_reproduces_a_tracker_rebuilt_with_the_same_values(as_tensor):
    line, beam = _line(), _beam(128)
    expected, expected_energy = _tracker(_with(line, **NEW)).track(beam)
    value = (lambda v: torch.tensor(v, dtype=F64)) if as_tensor else (lambda v: v)  # noqa: E731
    tracker = _tracker(line).with_settings(
        k1={"q1": value(2.2)}, k2={"S1": value(-3.0)}, kick_x={"H1": value(-2e-4)},
        voltage_V={"L1": value(25e6)}, phase_rad={"L1": value(0.1)},
    )
    out, energy = tracker.track(beam)
    torch.testing.assert_close(out, expected, rtol=1e-12, atol=1e-18)
    torch.testing.assert_close(energy, expected_energy, rtol=1e-14, atol=0)


def test_with_settings_leaves_the_original_tracker_unchanged_and_is_independent_of_it():
    line, beam = _line(), _beam(64)
    tracker = _tracker(line)
    before = tracker.track(beam)[0]
    copy = tracker.with_settings(k1={"Q1": torch.tensor(3.0, dtype=F64)})
    assert not torch.equal(copy.track(beam)[0], before)
    torch.testing.assert_close(tracker.track(beam)[0], before, rtol=0, atol=0)
    assert tracker._element_params[1].k1 == 1.5 and isinstance(copy._element_params[1].k1, torch.Tensor)


def test_with_settings_applies_the_fiducial_phase_shift_like_a_rebuilt_tracker():
    line, beam = _line(), _beam(64)
    fiducial = FiducialReference(trajectory=torch.zeros(1), t_ref_s=3.1e-9)
    expected = _tracker(_with(line, L1=dict(phase_rad=0.2)), fiducial=fiducial).track(beam)[0]
    out = _tracker(line, fiducial=fiducial).with_settings(phase_rad={"L1": torch.tensor(0.2, dtype=F64)}).track(beam)[0]
    torch.testing.assert_close(out, expected, rtol=1e-12, atol=1e-18)


def test_tensor_settings_are_read_on_every_call_so_a_static_tensor_can_be_rewritten():
    """A setting written into a tensor after ``with_settings`` must take effect, with or without a fiducial phase shift."""
    line, beam = _line(), _beam(64)
    fiducial = FiducialReference(trajectory=torch.zeros(1), t_ref_s=3.1e-9)
    k1, voltage_V, phase_rad = (torch.tensor(v, dtype=F64) for v in (1.5, 20e6, 0.3))
    scanned = _tracker(line, fiducial=fiducial).with_settings(k1={"Q1": k1}, voltage_V={"L1": voltage_V}, phase_rad={"L1": phase_rad})
    for values in [dict(k1=2.2, voltage_V=25e6, phase_rad=0.1), dict(k1=0.7, voltage_V=15e6, phase_rad=0.5)]:
        k1.fill_(values["k1"])
        voltage_V.fill_(values["voltage_V"])
        phase_rad.fill_(values["phase_rad"])
        expected = _tracker(_with(line, Q1=dict(k1=values["k1"]), L1=dict(voltage_V=values["voltage_V"], phase_rad=values["phase_rad"])),
                            fiducial=fiducial).track(beam)[0]
        torch.testing.assert_close(scanned.track(beam)[0], expected, rtol=1e-12, atol=1e-18)


def test_settings_gradients_match_central_finite_differences():
    line, beam = _line(), _beam(64)
    tracker = _tracker(line)
    base = dict(k1=2.2, voltage_V=25e6, phase_rad=0.1)

    def loss(k1, voltage_V, phase_rad):
        out, energy = tracker.with_settings(k1={"Q1": k1}, voltage_V={"L1": voltage_V}, phase_rad={"L1": phase_rad}).track(beam)
        return 1e7 * out[:, 0].square().mean() + 1e6 * out[:, 5].square().mean() + 1e-8 * energy

    leaves = {key: torch.tensor(value, dtype=F64, requires_grad=True) for key, value in base.items()}
    grads = dict(zip(base, torch.autograd.grad(loss(**leaves), list(leaves.values()))))
    for key, step in dict(k1=1e-3, voltage_V=1e2, phase_rad=1e-5).items():
        up = {k: torch.tensor(v + (step if k == key else 0.0), dtype=F64) for k, v in base.items()}
        down = {k: torch.tensor(v - (step if k == key else 0.0), dtype=F64) for k, v in base.items()}
        fd = (loss(**up) - loss(**down)) / (2.0 * step)
        assert math.isclose(float(grads[key]), float(fd), rel_tol=2e-4), key


def test_with_settings_rejects_unknown_names_and_settings_that_do_not_apply():
    tracker = _tracker(_line())
    with pytest.raises(KeyError, match="NOPE"):
        tracker.with_settings(k1={"nope": 1.0})
    with pytest.raises(ValueError, match="k1 does not apply to element L1"):
        tracker.with_settings(k1={"L1": 1.0})
    with pytest.raises(ValueError, match="voltage_V does not apply to element Q1"):
        tracker.with_settings(voltage_V={"Q1": 1.0})
    with pytest.raises(ValueError, match="kick_y does not apply to element H1"):
        tracker.with_settings(kick_y={"H1": 1.0})


def test_reference_energy_update_keeps_tensor_strength_offsets_and_refreshes_float_ones():
    line = [ElementSpec("B0", "bend", 0.3, angle=0.1, k1=0.5), ElementSpec("Q1", "quad", 0.1, k1=1.0)]
    tracker = _tracker(line).with_settings(k1={"B0": torch.tensor(0.5, dtype=F64)})
    offsets = [p.offset.clone() for p in tracker._element_params]
    tracker._set_reference_energy(ENERGY_EV * 1.2)
    assert torch.equal(tracker._element_params[0].offset, offsets[0])
    expected = _tracker(line)
    expected._set_reference_energy(ENERGY_EV * 1.2)
    torch.testing.assert_close(tracker._element_params[1].offset, expected._element_params[1].offset, rtol=0, atol=0)


@pytest.mark.parametrize("drift_model", ["overtaking", "stupakov", "retarded"])
def test_capture_detection_is_not_called_on_a_cpu_only_torch_build(monkeypatch, drift_model):
    """On a CPU-only wheel ``torch.cuda.is_current_stream_capturing`` raises; CPU tracking must never reach it."""
    def unavailable():
        raise RuntimeError("Tried to instantiate dummy base class _cuda_isCurrentStreamCapturing")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", unavailable)
    tracker = _tracker(_drift_line(), bunch_charge_C=CHARGE_C, csr=True, lsc=True, pipe_radius_b_m=0.01,
                       fast_collective_effects=True, csr_drift_model=drift_model)
    out, _ = tracker.track(_beam(128))
    assert torch.isfinite(out).all()


def _capture(step, static):
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
    with torch.cuda.graph(graph):
        loss = step()
    return graph, loss


@needs_cuda
def test_one_captured_forward_and_backward_replays_new_scan_settings():
    line, beam = _line(), _beam(512, "cuda")
    statics = {key: torch.tensor(value, dtype=F64, device="cuda") for key, value in dict(k=2.0, v=20e6, p=0.3).items()}
    scanned = _tracker(line, "cuda", bunch_charge_C=CHARGE_C, csr=True, fast_collective_effects=True).with_settings(
        k1={"Q1": statics["k"]}, voltage_V={"L1": statics["v"]}, phase_rad={"L1": statics["p"]},
    )
    static_beam = beam.clone().requires_grad_(True)

    def step():
        loss = _screen(scanned.track(static_beam)[0]).square().sum()
        loss.backward()
        return loss

    graph, captured = _capture(step, static_beam)
    for k1, voltage_V, phase_rad in [(2.0, 20e6, 0.3), (2.6, 18e6, 0.25), (1.4, 22e6, 0.35)]:
        reference = _tracker(
            _with(line, Q1=dict(k1=k1), L1=dict(voltage_V=voltage_V, phase_rad=phase_rad)), "cuda",
            bunch_charge_C=CHARGE_C, csr=True, fast_collective_effects=True,
        )
        x = beam.clone().requires_grad_(True)
        expected = _screen(reference.track(x)[0]).square().sum()
        expected.backward()
        for key, value in zip("kvp", (k1, voltage_V, phase_rad)):
            statics[key].fill_(value)
        graph.replay()
        assert math.isclose(float(captured.detach()), float(expected.detach()), rel_tol=1e-8)
        torch.testing.assert_close(static_beam.grad, x.grad, rtol=1e-6, atol=1e-8 * float(x.grad.abs().max()))


@needs_cuda
@pytest.mark.parametrize("lsc", [False, True])
def test_forward_only_capture_of_fast_csr_and_lsc_replays_the_eager_result(lsc):
    """A scalar charge normally replays a manually captured graph, which cannot be nested in an outer capture."""
    line, beam = _line(), _beam(512, "cuda")
    tracker = _tracker(
        line, "cuda", bunch_charge_C=CHARGE_C, csr=True, lsc=lsc, pipe_radius_b_m=0.01 if lsc else None, fast_collective_effects=True,
    )
    static = beam.clone()

    def step():
        return _screen(tracker.track(static)[0]).square().sum()

    with torch.no_grad():
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = step()
        for scale in (1.0, 1.04):
            expected = float(_screen(tracker.track(beam * scale)[0]).square().sum())
            static.copy_(beam * scale)
            graph.replay()
            assert math.isclose(float(captured), expected, rel_tol=1e-8)


def _drift_line():
    line = [ElementSpec("D0", "drift", 0.2)]
    for i, angle in enumerate((0.08, -0.08, -0.08, 0.08)):
        line += [
            ElementSpec(f"B{i}", "bend", 0.2, angle=angle, csr=True),
            ElementSpec(f"DB{i}", "drift", 0.4, csr_drift_rho_m=0.2 / abs(angle), csr_drift_phi_m=abs(angle), csr_drift_dz=0.05),
        ]
    return line


@needs_cuda
def test_stupakov_drift_model_captures_and_replays_the_eager_forward_and_backward():
    beam = _beam(512, "cuda")
    tracker = _tracker(_drift_line(), "cuda", bunch_charge_C=CHARGE_C, csr=True, fast_collective_effects=True, csr_drift_model="stupakov")
    static = beam.clone().requires_grad_(True)

    def step(x=static):
        loss = _screen(tracker.track(x)[0]).square().sum()
        loss.backward()
        return loss

    graph, captured = _capture(step, static)
    x = (beam * 1.03).requires_grad_(True)
    expected = step(x)
    static.data.copy_(beam * 1.03)
    graph.replay()
    assert math.isclose(float(captured.detach()), float(expected.detach()), rel_tol=1e-8)
    torch.testing.assert_close(static.grad, x.grad, rtol=1e-6, atol=1e-8 * float(x.grad.abs().max()))


@needs_cuda
def test_retarded_drift_model_refuses_capture_with_a_clear_error():
    tracker = _tracker(_drift_line(), "cuda", bunch_charge_C=CHARGE_C, csr=True, fast_collective_effects=True, csr_drift_model="retarded")
    beam = _beam(256, "cuda")
    with pytest.raises(RuntimeError, match="retarded.*cannot be captured"):
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            tracker.track(beam)
    torch.cuda.synchronize()  # the context must still be usable
