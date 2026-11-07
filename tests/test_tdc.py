"""Thin transverse deflecting cavity (``ElementSpec(kind="tdc")``): kick, Panofsky-Wenzel energy term, symplecticity, scan settings, import.

The kick is ``V / (p0 c) * sin(phase + omega tau / c)`` along ``(cos(tilt), sin(tilt))`` and the energy change is fixed by the Panofsky-Wenzel
theorem, so the map must preserve ``dx^dpx + dy^dpy - beta0 dtau^ddelta`` (the time and energy pair); the sign of the energy term is checked
that way, not assumed.
"""
import math

import pytest
import torch

from slipstream.constants import C_LIGHT
from slipstream.reconstruction import ScanModel, Screen
from slipstream.tracking.native.elements import _tdc_kick
from slipstream.tracking.native.errors import FiducialReference
from slipstream.tracking.native.lattice_import import ElementSpec, parse_elegant_lattice
from slipstream.tracking.native.linac import LinacTracker

ENERGY_EV, MC2 = 150e6, 0.51099895e6
FREQ_HZ = 2.9985e9
OMEGA = 2.0 * math.pi * FREQ_HZ
F64 = torch.float64
P0C = math.sqrt(ENERGY_EV**2 - MC2**2)
BETA0 = P0C / ENERGY_EV


def _kick(state, voltage=2e6, phase=0.0, tilt=0.0, length=0.0):
    e, m = torch.tensor(ENERGY_EV, dtype=F64), torch.tensor(MC2, dtype=F64)
    return _tdc_kick(state, length, voltage, phase, OMEGA, tilt, e, m)


def _tracker(specs, **kwargs):
    return LinacTracker(specs, ENERGY_EV, device="cpu", dtype=F64, compile=False, always_change_p0=False, **kwargs)


def _bunch(n=200, seed=1):
    scales = torch.tensor([1e-4, 1e-5, 1e-4, 1e-5, 2e-4, 1e-3], dtype=F64)
    return torch.randn(n, 6, generator=torch.Generator().manual_seed(seed), dtype=F64) * scales


def test_kick_and_panofsky_wenzel_energy_change_match_the_formulas():
    s = torch.tensor([[2e-4, 0.0, -1e-4, 0.0, 3e-4, 0.0]], dtype=F64)
    phase, volt = 0.4, 2e6
    out = _kick(s, volt, phase, tilt=0.0)
    arg = phase + OMEGA * 3e-4 / C_LIGHT
    a = volt / P0C
    assert float(out[0, 1]) == pytest.approx(a * math.sin(arg), rel=1e-12) and float(out[0, 3]) == 0.0
    assert float(out[0, 5]) == pytest.approx(-a * OMEGA / (C_LIGHT * BETA0) * 2e-4 * math.cos(arg), rel=1e-12)  # offset along x
    vertical = _kick(s, volt, phase, tilt=math.pi / 2)
    assert float(vertical[0, 1]) == pytest.approx(0.0, abs=1e-18) and float(vertical[0, 3]) == pytest.approx(a * math.sin(arg), rel=1e-12)
    assert float(vertical[0, 5]) == pytest.approx(-a * OMEGA / (C_LIGHT * BETA0) * -1e-4 * math.cos(arg), rel=1e-12)  # offset along y
    diagonal = _kick(s, volt, phase, tilt=math.pi / 4)
    assert float(diagonal[0, 1]) == pytest.approx(float(diagonal[0, 3]), rel=1e-12)


def _defect(jacobian, beta_weight):
    block = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=F64)
    omega = torch.block_diag(block, block, beta_weight * block)
    return float((jacobian.T @ omega @ jacobian - omega).abs().max())


@pytest.mark.parametrize("length", [0.0, 0.5])
@pytest.mark.parametrize("tilt", [0.0, math.pi / 2, 0.3])
def test_tdc_map_is_symplectic_only_with_the_panofsky_wenzel_sign(length, tilt):
    s0 = torch.tensor([1e-4, 2e-5, -2e-4, 1e-5, 1.5e-4, 1e-3], dtype=F64)
    jacobian = torch.autograd.functional.jacobian(lambda s: _kick(s[None], 3e6, 0.7, tilt, length)[0], s0)
    assert _defect(jacobian, -BETA0) < 1e-6
    assert _defect(jacobian, +BETA0) > 1e-2  # the opposite energy-time form is not preserved: the check discriminates


def test_vertical_streak_is_the_kick_times_the_drift():
    p = _bunch()
    p[:, 5] = 0.0
    p[:, 1] = p[:, 3] = 0.0
    tracker = _tracker([ElementSpec("TD", "tdc", 0.0, voltage_V=2e6, frequency_Hz=FREQ_HZ, tilt=math.pi / 2), ElementSpec("D", "drift", 1.2)])
    out, _ = tracker.track(p)
    arg = OMEGA * p[:, 4] / C_LIGHT
    py = (2e6 / P0C) * torch.sin(arg)
    delta = -(2e6 / P0C) * OMEGA / (C_LIGHT * BETA0) * p[:, 2] * torch.cos(arg)  # the Panofsky-Wenzel energy change, before the drift
    y_expected = p[:, 2] + 1.2 * py / torch.sqrt((1.0 + delta) ** 2 - py**2)
    assert torch.allclose(out[:, 2], y_expected, rtol=0.0, atol=1e-12)
    slope = float(((out[:, 2] - p[:, 2]) * p[:, 4]).sum() / (p[:, 4] ** 2).sum())
    assert slope == pytest.approx(1.2 * 2e6 / P0C * OMEGA / C_LIGHT, rel=1e-2)  # linear streak: dy/dtau = d * V k / (p0 c)


def test_scan_settings_float_tensor_gradient_and_validation():
    specs = [ElementSpec("TD", "tdc", 0.0, voltage_V=1e6, phase_rad=0.2, frequency_Hz=FREQ_HZ, tilt=math.pi / 2), ElementSpec("D", "drift", 1.0)]
    p = _bunch()
    base = _tracker(specs)
    rebuilt = _tracker([ElementSpec("TD", "tdc", 0.0, voltage_V=2.5e6, phase_rad=0.5, frequency_Hz=FREQ_HZ, tilt=math.pi / 2), specs[1]])
    scanned = base.with_settings(voltage_V={"TD": 2.5e6}, phase_rad={"TD": 0.5})
    assert torch.equal(scanned.track(p)[0], rebuilt.track(p)[0])
    volt = torch.tensor(2.5e6, dtype=F64, requires_grad=True)
    phase = torch.tensor(0.5, dtype=F64, requires_grad=True)
    weights = torch.randn(200, 6, dtype=F64, generator=torch.Generator().manual_seed(5))

    def loss(v, ph):
        return (base.with_settings(voltage_V={"TD": v}, phase_rad={"TD": ph}).track(p)[0] * weights).sum()

    gv, gp = torch.autograd.grad(loss(volt, phase), (volt, phase))
    hv, hp = 1e3, 1e-6
    fd_v = (loss(volt.detach() + hv, phase.detach()) - loss(volt.detach() - hv, phase.detach())) / (2 * hv)
    fd_p = (loss(volt.detach(), phase.detach() + hp) - loss(volt.detach(), phase.detach() - hp)) / (2 * hp)
    assert float(gv) == pytest.approx(float(fd_v), rel=1e-6) and float(gp) == pytest.approx(float(fd_p), rel=1e-6)
    with pytest.raises(ValueError):
        base.with_settings(k1={"TD": 1.0})


def test_phase_is_relative_to_the_fiducial_arrival_time():
    spec = ElementSpec("TD", "tdc", 0.0, voltage_V=1e6, phase_rad=0.9, frequency_Hz=FREQ_HZ, tilt=math.pi / 2)
    t_ref = 3e-12
    fiducial = FiducialReference(trajectory=torch.zeros(2, 1, 6, dtype=F64), t_ref_s=t_ref)
    p = _bunch()
    shifted = _tracker([ElementSpec("TD", "tdc", 0.0, voltage_V=1e6, phase_rad=0.9 - OMEGA * t_ref, frequency_Hz=FREQ_HZ, tilt=math.pi / 2)])
    with_fiducial = _tracker([spec], fiducial=fiducial)
    assert torch.equal(with_fiducial.track(p)[0], shifted.track(p)[0])
    assert torch.equal(with_fiducial.with_settings(phase_rad={"TD": 0.9}).track(p)[0], shifted.track(p)[0])


def test_elegant_rftm110_and_rfdf_import(tmp_path):
    lte = tmp_path / "tdc.lte"
    lte.write_text(
        "TD: RFTM110, L=0.0, VOLTAGE=2e6, FREQUENCY=2.9985e9, PHASE=30.0, TILT=1.5707963267948966\n"
        "TD2: RFDF, L=0.4, VOLTAGE=1e6, FREQUENCY=2.9985e9, PHASE=0.0\n"
        "D1: DRIFT, L=1.0\n"
        "LN: LINE=(TD,D1,TD2)\n"
    )
    audit: list = []
    specs = parse_elegant_lattice(lte, "LN", audit=audit)
    td, _, td2 = specs
    assert td.kind == "tdc" and td.voltage_V == 2e6 and td.frequency_Hz == 2.9985e9 and td.tilt == pytest.approx(math.pi / 2)
    assert td.phase_rad == pytest.approx(math.radians(30.0)) and td.length == 0.0
    assert td2.kind == "tdc" and td2.length == 0.4 and td2.tilt == 0.0
    assert audit == []


def test_tdc_voltage_is_a_scannable_variable_of_a_streak_image():
    tracker = _tracker([ElementSpec("TD", "tdc", 0.0, voltage_V=0.0, frequency_Hz=FREQ_HZ, tilt=math.pi / 2), ElementSpec("D", "drift", 1.0)])
    screen = Screen(shape=(24, 24), pixel_size=(6e-5, 1.5e-4), axes=(4, 2))  # tau along the columns, y along the rows
    model = ScanModel(tracker, screen, [("TD", "voltage_V")])
    model.set_lattice_parameters(torch.tensor([[0.0], [3e6]], dtype=F64))
    (images,) = model.track_and_observe(_bunch(2000).clamp(-4 * torch.tensor([1e-4, 1e-5, 1e-4, 1e-5, 2e-4, 1e-3], dtype=F64),
                                                          4 * torch.tensor([1e-4, 1e-5, 1e-4, 1e-5, 2e-4, 1e-3], dtype=F64)))
    rows = (torch.arange(24, dtype=F64) - 11.5) * 1.5e-4
    y_var = [float((image.sum(1) * rows**2).sum() / image.sum()) for image in images]
    assert y_var[1] > 4.0 * y_var[0]  # the streak maps tau onto y

    cols = (torch.arange(24, dtype=F64) - 11.5) * 6e-5
    correlation = [float((image * rows[:, None] * cols[None, :]).sum() / image.sum()) for image in images]
    assert correlation[1] > 0 and abs(correlation[0]) < 0.05 * correlation[1]  # a late particle is kicked to +y; no correlation without the cavity
