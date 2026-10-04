import math

import pytest
import torch

from slipstream import (
    ElementSpec, Lattice, NativeTracker, Ring6DResult, betatron_tunes6d,
    dynamic_aperture6d, frequency_map6d, momentum_aperture6d, track_ring6d,
)
from slipstream.analysis.nonlinear import LinearOptics
from slipstream.tracking.native.elements import _misalign
from slipstream.tracking.native.ring import RingMap


def test_tensor_alignment_ensemble_matches_independent_machines_and_gradient():
    ring = RingMap([ElementSpec("Q", "quad", .2, k1=.7)], 1e9, device="cpu")
    values = torch.tensor([[1e-3], [-2e-3]], dtype=torch.float64, requires_grad=True)
    state = torch.randn(2, 3, 6, dtype=torch.float64) * 1e-4
    ensemble = ring.with_alignments(dx={"q": values}, dy={"Q": values * .2}, tilt={"Q": values * 3})
    got = ensemble.turn(state)
    expected = torch.stack([ring.with_alignments(dx={"Q": values[i]}, dy={"Q": values[i] * .2},
                                               tilt={"Q": values[i] * 3}).turn(state[i]) for i in range(2)])
    torch.testing.assert_close(got, expected, atol=1e-16, rtol=1e-13)
    torch.testing.assert_close(ensemble.float64_reference().turn(state), got)
    gradient = torch.autograd.grad(got[..., 1].sum(), values)[0]
    eps = 1e-7
    def evaluate(v):
        return ring.with_alignments(dx={"Q": v}, dy={"Q": v * .2}, tilt={"Q": v * 3}).turn(state)[..., 1].sum()
    for i in range(2):
        step = torch.zeros_like(values)
        step[i] = eps
        fd = (evaluate(values.detach() + step) - evaluate(values.detach() - step)) / (2 * eps)
        torch.testing.assert_close(gradient[i, 0], fd, rtol=1e-7, atol=1e-10)
    with pytest.raises(KeyError):
        ring.with_alignments(dx={"missing": values})


def test_tensor_alignment_is_exact_canonical_inverse():
    u = torch.randn(4, 2, dtype=torch.float64)
    dx = torch.tensor([.1, -.2], dtype=torch.float64)
    tilt = torch.tensor([.3, -.7], dtype=torch.float64)
    entered = _misalign(*u, dx, dx * 2, tilt, True)
    restored = torch.stack(_misalign(*entered, dx, dx * 2, tilt, False))
    torch.testing.assert_close(restored, u, atol=5e-16, rtol=1e-15)


def _tracker():
    params = Lattice(1., 1e9, .01, 1e6, 5e8, 0., math.inf, 0.)
    return NativeTracker([ElementSpec("D", "drift", 1.)], params, device="cpu")


def test_rf_recording_matches_normal_tracking_and_delta_scan(monkeypatch):
    state = torch.zeros(3, 6, dtype=torch.float64)
    state[:, 4] = torch.tensor([-.001, 0., .001])
    normal, _ = _tracker().track(state, 4)
    tracker = _tracker()
    def forbidden_collection(*args):
        raise AssertionError("RF analysis must not collect host diagnostics")
    monkeypatch.setattr(tracker, "_collect", forbidden_collection)
    recorded = track_ring6d(tracker, state, 4)
    assert tracker._elapsed_turns == 4
    torch.testing.assert_close(recorded.state, normal, atol=0, rtol=0)
    assert recorded.tbt.shape == (4, 3, 6)
    assert recorded.state[:, 5].abs().max() > 0  # RF genuinely kicks
    # Independent voltage/phase calculation: on-axis zero-delta drift leaves
    # tau unchanged, so the first kick is known in closed form (no damping).
    from slipstream import C_LIGHT
    lattice = _tracker().params
    phase = lattice.phi_s + 2 * math.pi * lattice.f_rf_Hz * state[:, 4] / C_LIGHT
    expected_kick = lattice.Vrf_V / lattice.E0_eV * (torch.sin(phase) - math.sin(lattice.phi_s))
    torch.testing.assert_close(recorded.tbt[0, :, 5], expected_kick, atol=1e-18, rtol=1e-12)
    delta = torch.tensor([-.03, 0., .03], dtype=torch.float64)
    scan = momentum_aperture6d(_tracker(), torch.zeros(6), delta, 3, delta_limit=.02)
    assert scan.survived.tolist() == [False, True, False]
    assert scan.loss_turn.tolist() == [0, -1, 0]


def test_betatron_carrier_selected_despite_stronger_synchrotron_sideband():
    T = 2048
    qx, qy, qs = .31, .27, .025
    n = torch.arange(T, dtype=torch.float64)
    z = torch.exp(2j * torch.pi * qx * n) + 3 * torch.exp(2j * torch.pi * (qx + qs) * n)
    zy = torch.exp(2j * torch.pi * qy * n)
    tbt = torch.zeros(T, 2, 6, dtype=torch.float64)
    tbt[:, :, 0] = z.real[:, None]
    tbt[:, :, 1] = -z.imag[:, None]
    tbt[:, :, 2] = zy.real[:, None]
    tbt[:, :, 3] = -zy.imag[:, None]
    scalar = lambda x: torch.tensor(x, dtype=torch.float64)
    optics = LinearOptics(torch.zeros(6), torch.eye(6), scalar(qx), scalar(qy), scalar(1), scalar(0), scalar(1), scalar(0))
    result = Ring6DResult(tbt[-1], torch.tensor([-1, 10]), tbt)
    tunes = betatron_tunes6d(result, optics, qs)
    torch.testing.assert_close(tunes[0], torch.tensor([qx, qy], dtype=torch.float64), atol=2e-8, rtol=0)
    assert torch.isnan(tunes[1]).all()
    with pytest.raises(ValueError, match="half_width"):
        betatron_tunes6d(result, optics, qs, half_width=qs)
    with pytest.raises(ValueError, match="record_every"):
        betatron_tunes6d(result._replace(record_every=2), optics, qs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_alignment_and_rf_match_cpu_eager():
    specs = [ElementSpec("Q", "quad", .2, k1=.7)]
    state = torch.randn(2, 3, 6, dtype=torch.float64) * 1e-4
    errors = torch.tensor([[1e-3], [-2e-3]], dtype=torch.float64)
    cpu = RingMap(specs, 1e9, device="cpu").with_alignments(dx={"Q": errors}, tilt={"Q": errors})
    gpu = RingMap(specs, 1e9, device="cuda", dtype=torch.float64).with_alignments(dx={"Q": errors.cuda()}, tilt={"Q": errors.cuda()})
    torch.testing.assert_close(gpu.turn(state.cuda()).cpu(), cpu.turn(state), atol=1e-15, rtol=1e-12)
    cpu_rf = _tracker()
    gpu_rf = NativeTracker(cpu_rf.specs, cpu_rf.params, device="cuda", dtype=torch.float64)
    launches = torch.zeros(3, 6, dtype=torch.float64)
    launches[:, 4] = torch.tensor([-.001, 0., .001])
    expected = track_ring6d(cpu_rf, launches, 4)
    got = track_ring6d(gpu_rf, launches.cuda(), 4)
    torch.testing.assert_close(got.tbt.cpu(), expected.tbt, atol=1e-15, rtol=1e-12)


def test_rf_da_and_frequency_map_independent_linear_map():
    # An independent analytic 6D oscillator verifies analysis geometry and
    # two-half diffusion without relying on the element implementation.
    class Oscillator:
        device = torch.device("cpu")
        dtype = torch.float64
        bunch_current_A = 0
        ibs = False
        wake = transverse_wake_x = transverse_wake_y = None

        def track(self, state, n_turns):
            out = state.clone()
            for q, p, tune in ((0, 1, .31), (2, 3, .27), (4, 5, .025)):
                c, s = math.cos(2 * math.pi * tune), math.sin(2 * math.pi * tune)
                out[..., q] = c * state[..., q] + s * state[..., p]
                out[..., p] = -s * state[..., q] + c * state[..., p]
            return out, None

        def advance(self, state, n_turns=1):
            return self.track(state, n_turns)[0]

    scalar = lambda x: torch.tensor(x, dtype=torch.float64)
    opt = LinearOptics(torch.zeros(6), torch.eye(6), scalar(.31), scalar(.27), scalar(1), scalar(0), scalar(1), scalar(0))
    da = dynamic_aperture6d(Oscillator(), opt, .01, 8, n_angles=3, n_r=4, refine=1)
    torch.testing.assert_close(da.radius_m, torch.full((3,), .01, dtype=torch.float64))
    fma = frequency_map6d(Oscillator(), opt, .025, (.001, .002), (.001, .002), nx=2, ny=2, n_turns=512)
    torch.testing.assert_close(fma.Qx, torch.full((2, 2), .31, dtype=torch.float64), atol=1e-9, rtol=0)
    torch.testing.assert_close(fma.Qy, torch.full((2, 2), .27, dtype=torch.float64), atol=1e-9, rtol=0)
    assert (fma.diffusion < -12).all()
