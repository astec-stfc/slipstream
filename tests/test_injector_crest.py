"""Crest scans retain the full field model and skip empty flight, without saved results."""
import math

import numpy as np
import pytest
import torch
from scipy.integrate import solve_ivp

from slipstream.constants import C_LIGHT, ELECTRON_MASS_EV
from slipstream.tracking.native.fields import RFCavityField, TravelingWaveField
from slipstream.tracking.native import injector as module
from slipstream.tracking.native.injector import Cavity, Injector, _crest_field, _on_axis_energy_MeV


def cavity_field(traveling=False, offset=0.0, phase=0.0, frequency=3e9):
    z = np.linspace(0.0, 0.08, 401)
    profile = np.sin(2 * math.pi * z / 0.04)
    kw = dict(e_max_V_per_m=30e6, frequency_hz=frequency, phase_rad=phase, z_offset=offset,
              device="cpu", dtype=torch.float64)
    if traveling:
        return TravelingWaveField(z, profile, z1=0.02, z2=0.06, n=1, m=3, n_cells=6, **kw)
    return RFCavityField(z, profile, **kw)


@pytest.mark.parametrize("traveling", [False, True])
@pytest.mark.parametrize("offset", [-0.1, 1.3])
def test_crest_field_matches_full_torch_field_at_edges_and_tile_joins(traveling, offset):
    field = cavity_field(traveling, offset)
    end = field.length if traveling else 0.08
    local = np.r_[np.linspace(-0.01, end + 0.01, 100), 0.0, end]
    if traveling:
        local = np.r_[local, field.z1 + np.arange(field.n_tiles + 1) * field.period]
    z = torch.tensor(local + offset, dtype=torch.float64)
    field.phase_rad = torch.linspace(-3.0, 4.0, len(z), dtype=torch.float64)
    fast = _crest_field(field)
    zero = torch.zeros_like(z)
    for t in [-1e-9, 0.0, 2.3e-9]:
        full = field(zero, zero, z, torch.tensor(t, dtype=torch.float64))[2]
        np.testing.assert_allclose(fast(z.numpy(), t), full.numpy(), rtol=2e-12, atol=1e-5)


def test_empty_flight_is_exact_and_does_not_step_through_the_gap(monkeypatch):
    field = cavity_field(offset=100.0, frequency=0.0, phase=math.pi)
    field.scale = 0.0
    calls = 0
    original = module._crest_field

    def counted(f):
        evaluate = original(f)

        def ez(z, t):
            nonlocal calls
            calls += 1
            return evaluate(z, t)

        return ez

    monkeypatch.setattr(module, "_crest_field", counted)
    u = 0.3 * C_LIGHT
    e_in, e_out = _on_axis_energy_MeV([field], 0.0, u, 0.0, 50.0, 90.0, -1.0, ELECTRON_MASS_EV)
    expected = (math.sqrt(1.09) - 1.0) * ELECTRON_MASS_EV / 1e6
    assert float(e_in[0]) == pytest.approx(expected, rel=1e-13)
    assert float(e_out[0]) == pytest.approx(expected, rel=1e-13)
    assert calls == 0


@pytest.mark.parametrize("traveling", [False, True])
def test_long_drift_preserves_rf_arrival_phase(traveling):
    u, dt = 5.0 * C_LIGHT, 0.1e-12
    flight = 1_000_000 * dt
    distance = u / math.sqrt(26.0) * flight
    phases = torch.tensor([1.8, 2.4, 3.2], dtype=torch.float64)
    field = cavity_field(traveling, phase=phases)
    end = field.length if traveling else 0.08
    _, near = _on_axis_energy_MeV([field], 0.0, u, flight, 0.0, end, -1.75882000838e11,
                                  ELECTRON_MASS_EV, dt_fine=dt, dt_coarse=dt)
    field.z_offset = distance
    _, far = _on_axis_energy_MeV([field], 0.0, u, 0.0, distance, distance + end, -1.75882000838e11,
                                 ELECTRON_MASS_EV, dt_fine=dt, dt_coarse=dt)
    torch.testing.assert_close(far, near, rtol=0.0, atol=1e-7)


@pytest.mark.parametrize("traveling", [False, True])
def test_scan_energies_match_independent_adaptive_reference(traveling):
    field = cavity_field(traveling, offset=0.01)
    phases = np.array([1.8, 2.4, 3.2])
    field.phase_rad = torch.from_numpy(phases)
    z_end = 0.01 + (field.length if traveling else 0.08)
    u0, t0 = 5.0 * C_LIGHT, 1.7e-12
    qm = -1.75882000838e11
    _, energy = _on_axis_energy_MeV([field], 0.0, u0, t0, 0.01, z_end, qm, ELECTRON_MASS_EV,
                                   dt_fine=0.01e-12, dt_coarse=0.01e-12)

    for phase, got in zip(phases, energy):
        # Integrate arrival time and proper velocity in z, independently of the time-step scan.
        field.phase_rad = float(phase)

        def rhs(z, state):
            t, u = state
            velocity = u / math.sqrt(1 + (u / C_LIGHT)**2)
            zero = torch.zeros(1, dtype=torch.float64)
            ez = float(field(zero, zero, torch.tensor([z], dtype=torch.float64),
                             torch.tensor(t, dtype=torch.float64))[2][0])
            return [1.0 / velocity, qm * ez / velocity]

        ref = solve_ivp(rhs, (0.0, z_end), [t0, u0], rtol=2e-10, atol=[1e-20, 1e-2], max_step=0.0002)
        assert ref.success
        u = ref.y[1, -1]
        expected = (math.sqrt(1 + (u / C_LIGHT)**2) - 1) * ELECTRON_MASS_EV / 1e6
        assert float(got) == pytest.approx(expected, abs=2e-5)  # MeV, includes first-step exit overshoot


def test_reversed_launch_is_retired():
    z = np.linspace(0.0, 0.1, 11)
    field = RFCavityField(z, np.ones_like(z), 10e6, 0.0, device="cpu", dtype=torch.float64)
    field.phase_rad = torch.tensor([0.0, math.pi], dtype=torch.float64)
    _, out = _on_axis_energy_MeV([field], 0.0, 0.0, 0.0, 0.0, 0.1, -1.75882000838e11, ELECTRON_MASS_EV)
    assert float(out[0]) < 1e-6
    assert float(out[1]) == pytest.approx(1.0, abs=0.001)


def test_cresting_keeps_z_order_upstream_offsets_and_fresh_inputs(tmp_path, monkeypatch):
    z = np.linspace(0.0, 0.04, 101)
    path = tmp_path / "cavity.dat"
    np.savetxt(path, np.c_[z, np.sin(math.pi * z / 0.04)])
    gun = Cavity(str(path), 0.0, 60e6, 3e9, phase_deg=-12.0)
    downstream = Cavity(str(path), 0.3, 30e6, 1.3e9, phase_deg=7.0)
    machine = Injector("", 0.4, cavities=[downstream, gun])
    monkeypatch.setattr(Injector, "_reference_particle", lambda self: (0.0, 0.1 * C_LIGHT, 1e-12))
    gains = machine.auto_phase(verbose=False)
    assert all(np.isfinite(gains))
    # Values from the previous float64 PyTorch integrator, including the gun's -12 degree offset.
    assert [gun.crest_deg, downstream.crest_deg] == pytest.approx([44.0125366407, 263.0324652193], abs=2e-5)
    assert gains == pytest.approx([0.738971191315, 0.512337389469], abs=1e-10)
    previous = downstream.crest_deg
    gun.phase_deg += 3.0
    machine.auto_phase(verbose=False)
    assert abs(downstream.crest_deg - previous) > 0.01
