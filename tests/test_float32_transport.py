"""Float32 transport keeps the change of ``tau`` (issue #59), and with it the CSR gradient agrees with float64 (issue #43).

``tau`` changes by the path excess and the velocity slip, both far smaller than the path itself. Formed as the difference of two
path lengths, float32 loses them: errors of 1e-10 to 3e-8 m per element in the cases below.
"""
import math

import pytest
import torch

from slipstream.tracking.native import elements
from slipstream.tracking.native.lattice_import import ElementSpec
from slipstream.tracking.native.linac import LinacTracker

ENERGY_EV, MC2_EV, OMEGA_RF = 150e6, 0.51099895e6, 2 * math.pi * 3e9


def beam(n):
    generator = torch.Generator().manual_seed(5)
    scale = torch.tensor([2e-4, 2e-5, 2e-4, 2e-5, 30e-6, 1e-4], dtype=torch.float64)
    return torch.randn(n, 6, dtype=torch.float64, generator=generator) * scale


def magnet(length=0.2, angle=0.0, k1=0.0, **options):
    return lambda s, e, m: elements.track_element(s, length, angle, k1, 0.0, 0.0, 0.0, e, m, **options)


# transport map and bound on the rms float32 error of the change in tau [m]; storing tau itself costs about 8e-13 m here
CASES = {
    "drift": (magnet(0.5), 3e-12),
    "quad": (magnet(0.1, k1=2.0), 3e-12),
    "quad-order-8": (magnet(0.1, k1=2.0, integration_order=8), 3e-12),
    "bend": (magnet(angle=0.08), 5e-11),
    "reverse-bend": (magnet(angle=-0.08), 5e-11),
    "exact-dipole": (magnet(angle=0.08, k1=1.0, exact_dipole=True), 5e-11),
    "corrector": (lambda s, e, m: elements._corrector_kick(s, 0.2, 2e-5, "x", e, m), 3e-12),
    "deflecting-cavity": (lambda s, e, m: elements._tdc_kick(s, 0.4, 1e5, 0.3, OMEGA_RF, 0.2, e, m), 5e-12),
    "rf": (lambda s, e, m: elements._rf_cavity_kick(s, 1.0, 15e6, 0.3, OMEGA_RF, e, m, n_kicks=4)[0], 3e-11),
    "rf-srs": (lambda s, e, m: elements._rf_cavity_kick(s, 1.0, 15e6, 0.3, OMEGA_RF, e, m, n_kicks=4,
                                                       body_focus_model="srs")[0], 3e-11),
}


@pytest.mark.parametrize("name", CASES)
def test_float32_keeps_the_change_of_tau(name):
    transport, bound_m = CASES[name]
    change = {}
    for dtype in (torch.float32, torch.float64):
        start = beam(4096).to(dtype)
        out = transport(start, torch.tensor(ENERGY_EV, dtype=dtype), torch.tensor(MC2_EV, dtype=dtype))
        assert out.dtype == dtype
        change[dtype] = out[:, 4].double() - start[:, 4].double()
    assert (change[torch.float32] - change[torch.float64]).std().item() < bound_m


def chicane_gradient(dtype):
    """Input gradient of a smooth image-like loss behind a four-bend chicane with CSR, 16384 particles, 128 bins."""
    specs = [ElementSpec("D0", "drift", .5), ElementSpec("Q1", "quad", .1, k1=2.), ElementSpec("D1", "drift", .4),
             ElementSpec("Q2", "quad", .1, k1=-2.), ElementSpec("D2", "drift", .4)]
    for i, angle in enumerate((.08, -.08, -.08, .08)):
        specs += [ElementSpec(f"B{i}", "bend", .2, angle=angle, csr=True, csr_n_slices=4, csr_bins=128),
                  ElementSpec(f"DB{i}", "drift", .4)]
    specs += [ElementSpec("Q3", "quad", .1, k1=1.5), ElementSpec("D3", "drift", 1.0)]
    tracker = LinacTracker(specs, ENERGY_EV, device="cpu", dtype=dtype, always_change_p0=False, bunch_charge_C=100e-12,
                           csr=True, fast_collective_effects=True)
    start = beam(16384).to(dtype).requires_grad_(True)
    out = tracker.track(start)[0]
    generator = torch.Generator().manual_seed(11)
    wave = torch.randn(64, 2, dtype=torch.float64, generator=generator).to(dtype)
    phase = (torch.rand(64, dtype=torch.float64, generator=generator) * 2 * math.pi).to(dtype)
    # random cosine features of the screen position: no pixel edges, whose own float32 noise would hide the tracker's
    features = torch.cos(out[:, [0, 2]] / 1e-3 @ wave.T + phase)
    features.mean(0).square().sum().backward()
    return start.grad.double()


def test_float32_csr_gradient_is_within_one_percent_of_float64():
    single, double = chicane_gradient(torch.float32), chicane_gradient(torch.float64)
    for column in (0, 1, 4, 5):  # x, px, tau, delta
        assert ((single[:, column] - double[:, column]).norm() / double[:, column].norm()).item() < 0.01
