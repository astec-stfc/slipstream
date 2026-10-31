"""Independent photon-spectrum moments and a public single-pass radiation check."""
import math
from dataclasses import replace

import numpy as np
import pytest
from scipy.constants import alpha, c, hbar, m_e
from scipy.integrate import quad
from scipy.special import kv
import torch

from slipstream.constants import ELECTRON_MASS_EV
from slipstream.tracking.native.lattice_import import parse_elegant_lattice
from slipstream.tracking.native.linac import LinacTracker
from slipstream.tracking.native.radiation import bend_radiation_kick
from slipstream.tracking.native.ring import RingMap


def photon_moments(energy, curvature, length):
    gamma = energy / ELECTRON_MASS_EV
    rate = math.sqrt(3) * alpha * gamma * abs(curvature) / (2 * math.pi)
    critical_fraction = 1.5 * hbar / (m_e * c) * gamma**2 * abs(curvature)
    # Swap the photon-spectrum integral order, independent of native coefficients.
    integrals = [quad(lambda t: t**(n+1) * kv(5/3, t) / (n+1), 0, np.inf)[0]
                 for n in (1, 2)]
    return tuple(length * rate * critical_fraction**n * value
                 for n, value in zip((1, 2), integrals))


def test_spectrum_loss_diffusion_recoil_and_scaling():
    energy, curvature, length = 5e11, 1.35e-3 / 7, 7.
    loss, variance = photon_moments(energy, curvature, length)
    state = torch.zeros((200000, 6), dtype=torch.float64)
    energy_t = torch.tensor(energy, dtype=torch.float64)
    deterministic = bend_radiation_kick(state[:1], energy_t, curvature, length, synch_rad=True, isr=False)
    assert -deterministic[0, 5].item() == pytest.approx(loss, rel=2e-8)
    out = bend_radiation_kick(state, energy_t, curvature, length, synch_rad=False, isr=True,
                              generator=torch.Generator().manual_seed(71))
    assert out[:, 5].var().item() == pytest.approx(variance, rel=.012)
    assert abs(out[:, 5].mean()) < 4 * math.sqrt(variance / len(state))
    scaled = bend_radiation_kick(state, 2 * energy_t, -2 * curvature, length, synch_rad=False, isr=True,
                                 generator=torch.Generator().manual_seed(71))
    torch.testing.assert_close(scaled[:, 5], out[:, 5] * 16, atol=2e-15, rtol=1e-9)
    probe = torch.tensor([[0., .001, 0., -.002, 0., .01]], dtype=torch.float64)
    result = bend_radiation_kick(probe, energy_t, curvature, length, synch_rad=True, isr=False)
    torch.testing.assert_close(result[:, [1, 3]] / (1 + result[:, 5:6]),
                               probe[:, [1, 3]] / (1 + probe[:, 5:6]))
    for h, ds in ((0., length), (curvature, 0.)):
        assert bend_radiation_kick(probe, energy_t, h, ds, synch_rad=True, isr=True) is probe
    with pytest.raises(ValueError, match="nonnegative"):
        bend_radiation_kick(probe, energy_t, curvature, -1, synch_rad=True, isr=True)


def test_public_zero_charge_tracking_and_import(tmp_path):
    path = tmp_path / "bend.lte"
    path.write_text("B: CSBEND,L=7,ANGLE=.00135,SYNCH_RAD=1,ISR=1;\nBL: LINE=(B)\n")
    spec, = parse_elegant_lattice(path, "BL", strict=True)
    assert spec.synch_rad and spec.isr
    state = torch.zeros((30000, 6), dtype=torch.float64)
    loss, variance = photon_moments(5e11, spec.angle / spec.length, spec.length)
    def run():
        tracker = LinacTracker([spec], 5e11, device="cpu", n_kicks_bend=16,
                               always_change_p0=False, bunch_charge_C=0.,
                               generator=torch.Generator().manual_seed(5))
        return tracker.track(state)[0]
    result = run()
    torch.testing.assert_close(result, run(), atol=0, rtol=0)
    assert -result[:, 5].mean().item() == pytest.approx(loss, rel=.015)
    assert result[:, 5].var().item() == pytest.approx(variance, rel=.025)
    with pytest.raises(NotImplementedError, match="always_change_p0"):
        LinacTracker([spec], 5e11, device="cpu")
    with pytest.raises(NotImplementedError, match="LinacTracker"):
        RingMap([spec], 5e11, device="cpu")
    with pytest.raises(NotImplementedError, match="pure dipoles"):
        LinacTracker([replace(spec, k1=.1)], 5e11, device="cpu", always_change_p0=False)
    plain = replace(spec, synch_rad=False, isr=False)
    zero_field = replace(spec, angle=0)
    a = LinacTracker([zero_field], 5e11, device="cpu", always_change_p0=False).track(state[:2])[0]
    b = LinacTracker([replace(plain, angle=0)], 5e11, device="cpu", always_change_p0=False).track(state[:2])[0]
    torch.testing.assert_close(a, b, atol=0, rtol=0)
