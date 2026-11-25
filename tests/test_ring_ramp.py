"""Energy-ramped ring tracking (``NativeTracker(ramp=...)``): adiabatic damping, radiation scaling and the RF program.

The reference energy follows the ramp by fiat, so each turn ``delta`` and the angles are rescaled to the new reference:
momentum is conserved (``1 + delta ~ E_prev/E_now``) and the geometric emittance falls as ``1/E``. Radiation follows the
scaling laws of ``scaled_lattice``: ``sigma_delta0 ~ gamma`` and ``tau_z ~ gamma^-3``. With no energy change the ramped
tracker must reproduce the fixed-energy one, using the ramp's RF voltage, phase and frequency instead of the lattice's.
"""
import math

import numpy as np
import pytest
import torch

from slipstream import C_LIGHT, ElementSpec, Lattice, NativeTracker
from slipstream.lattice.ramp import RampProfile
from slipstream.tracking.native.ring_physics import rms_emittance

F64 = torch.float64
E0_EV, SIGMA0, A = 1e9, 1e-3, 0.01  # A = T0 / tau_z: radiation damping per turn at the injection energy

STRAIGHT = [ElementSpec("QF", "quad", .2, k1=1.), ElementSpec("D1", "drift", 1.), ElementSpec("QD", "quad", .2, k1=-1.),
            ElementSpec("D2", "drift", 1.)]


def bending_ring(cells=4):
    """FODO ring of ``cells`` cells with two sector bends each: momentum compaction ~0.4, 16 m, harmonic 20 at 1 GeV."""
    theta = math.pi / cells
    cell = [("QF", "quad", .2, dict(k1=1.)), ("D1", "drift", .3, {}), ("B1", "bend", 1., dict(angle=theta)),
            ("D2", "drift", .3, {}), ("QD", "quad", .2, dict(k1=-1.)), ("D3", "drift", .3, {}),
            ("B2", "bend", 1., dict(angle=theta)), ("D4", "drift", .3, {})]
    return [ElementSpec(f"{name}_{i}", kind, length, **extra) for i in range(cells) for name, kind, length, extra in cell]


RING = bending_ring()
CIRC_M = 16.0
BETA = math.sqrt(1 - 1 / (E0_EV / 0.51099895e6) ** 2)
F_RF_HZ = 20 * BETA * C_LIGHT / CIRC_M


def ramp_of(times_s, energies_eV, voltage_V, f_rf_Hz, phase_rad):
    n = len(times_s)
    return RampProfile(time_s=np.array(times_s, dtype=float), E0_eV=np.array(energies_eV, dtype=float),
                       Vrf_V=np.full(n, float(voltage_V)), f_rf_Hz=np.full(n, float(f_rf_Hz)), phi_rf_rad=np.full(n, float(phase_rad)))


def test_unaccelerated_beam_conserves_momentum_and_its_emittance_falls_as_the_inverse_energy():
    params = Lattice(2.4, E0_EV, .01, 1.0, 5e8, 0., math.inf, 0.)  # no radiation, a 1 V cavity
    turns, final_ratio = 300, 1.5
    ramp = ramp_of([0.0, (turns - 1) * params.T0_s], [E0_EV, final_ratio * E0_EV], 1.0, 5e8, params.phi_s)
    generator = torch.Generator().manual_seed(4)
    state = torch.zeros(4000, 6, dtype=F64)
    state[:, [0, 1, 2, 3]] = torch.randn(4000, 4, generator=generator, dtype=F64) * torch.tensor([1e-4, 1e-5, 1e-4, 1e-5], dtype=F64)
    emittance0 = rms_emittance(state[:, 0], state[:, 1])
    final, result = NativeTracker(STRAIGHT, params, ramp=ramp, device="cpu").track(state, turns, sample_every=turns - 1)
    assert result.E0_eV[-1] == pytest.approx(final_ratio * E0_EV, rel=1e-12)
    assert final[:, 5].mean().item() == pytest.approx(1 / final_ratio - 1, abs=1e-6)  # p E0/E is conserved; 1e-6 covers E vs pc
    assert final[:, 5].std().item() < 1e-12  # a uniform shift: no spread appears from nothing
    assert rms_emittance(final[:, 0], final[:, 1]) == pytest.approx(emittance0 / final_ratio, rel=1e-8)
    assert rms_emittance(final[:, 2], final[:, 3]) == pytest.approx(rms_emittance(state[:, 2], state[:, 3]) / final_ratio, rel=1e-8)


def test_energy_step_rescales_the_radiation_parameters_by_their_laws_and_damps_the_spread_adiabatically():
    ratio, turns = 1.5, 40
    params = Lattice(2.4, E0_EV, .01, 1.0, 5e8, 0., 1.0, SIGMA0)
    params = Lattice(2.4, E0_EV, .01, 1.0, 5e8, 0., params.T0_s / A, SIGMA0)
    ramp = ramp_of([0.0, params.T0_s, 1.0], [E0_EV, ratio * E0_EV, ratio * E0_EV], 1.0, 5e8, params.phi_s)
    generator = torch.Generator().manual_seed(8)
    state = torch.zeros(4000, 6, dtype=F64)
    state[:, 5] = SIGMA0 * torch.randn(4000, generator=generator, dtype=F64)
    variance = state[:, 5].var().item()
    for turn in range(turns):  # rescale to the new reference, then damp and excite with that energy's parameters
        step_in = 1.0 / ratio if turn == 1 else 1.0
        gamma_ratio = ratio if turn >= 1 else 1.0
        decay = 1 - A * gamma_ratio**3  # tau_z ~ gamma^-3
        variance = decay**2 * step_in**2 * variance + 2 * A * gamma_ratio**3 * (SIGMA0 * gamma_ratio) ** 2  # sigma_delta0 ~ gamma
    tracker = NativeTracker(STRAIGHT, params, ramp=ramp, device="cpu", generator=generator)
    final, _ = tracker.track(state, turns, sample_every=turns)
    assert final[:, 5].var().item() == pytest.approx(variance, rel=0.06)


@pytest.mark.parametrize("voltage_V,f_rf_Hz", [(2e5, F_RF_HZ), (8e5, F_RF_HZ), (2e5, 1.01 * F_RF_HZ)])
def test_ramped_rf_uses_the_ramp_program_not_the_lattice_values_and_matches_fixed_tracking_at_constant_energy(voltage_V, f_rf_Hz):
    tau_z = Lattice(CIRC_M, E0_EV, .01, 1.0, F_RF_HZ, 0., 1.0, SIGMA0).T0_s / 0.002

    def lattice(voltage, f_rf):
        return Lattice(CIRC_M, E0_EV, .4, voltage, f_rf, 5e3, tau_z, SIGMA0)  # 5 keV lost per turn, balanced by phi_s

    def start():
        generator = torch.Generator().manual_seed(6)
        state = torch.zeros(100, 6, dtype=F64)
        state[:, 4] = 5e-2 * torch.randn(100, generator=generator, dtype=F64)  # 0.8 rad rms of the 0.4 m RF wavelength
        state[:, 5] = 1e-3 * torch.randn(100, generator=generator, dtype=F64)
        return state

    def track(params, ramp=None):
        generator = torch.Generator().manual_seed(7)
        return NativeTracker(RING, params, ramp=ramp, device="cpu", generator=generator).track(start(), 60)[0]

    fixed = track(lattice(voltage_V, f_rf_Hz))
    ramp = ramp_of([0.0, 1.0], [E0_EV, E0_EV], voltage_V, f_rf_Hz, lattice(voltage_V, f_rf_Hz).phi_s)
    ramped = track(lattice(2e5, F_RF_HZ), ramp)  # the lattice says 2e5 V at the design frequency; the ramp says otherwise
    assert torch.isfinite(ramped).all()
    torch.testing.assert_close(ramped, fixed, rtol=1e-9, atol=1e-12)
    if (voltage_V, f_rf_Hz) != (2e5, F_RF_HZ):
        assert (track(lattice(2e5, F_RF_HZ)) - fixed).abs().max() > 1e-4  # the program genuinely matters for this motion


@pytest.mark.parametrize("gain_eV_per_turn", [0.0, 1000.0])
def test_an_energy_balanced_rf_program_keeps_the_synchronous_particle_in_place(gain_eV_per_turn):
    """The RF supplies the radiation loss U0 and the ramp's energy gain: with V sin(phase) = U0 + dE/turn the particle at the
    centre of the bunch stays at tau = delta = 0.  (The phase is a table value, so it is U0, not V sin(phase), that the kick
    subtracts: subtracting V sin(phase) leaves the particle without the gain and it slides back by dE/E every turn.)"""
    u0_eV, turns = 5e3, 300
    params = Lattice(CIRC_M, E0_EV, .4, 2e5, F_RF_HZ, u0_eV, math.inf, 0.)
    phase = math.pi - math.asin((u0_eV + gain_eV_per_turn) / 2e5)
    ramp = ramp_of([0.0, turns * params.T0_s], [E0_EV, E0_EV + turns * gain_eV_per_turn], 2e5, F_RF_HZ, phase)
    state = torch.zeros(1, 6, dtype=F64)
    final, result = NativeTracker(RING, params, ramp=ramp, device="cpu").track(state, turns, sample_every=turns)
    assert result.E0_eV[-1] == pytest.approx(E0_EV + (turns - 1) * gain_eV_per_turn, rel=1e-12)
    # one turn's gain (the kick comes before the reference rises) and U0 growing as E^4 (5 eV of 1 GeV over the ramp)
    assert abs(final[0, 5].item()) < 1.1 * gain_eV_per_turn / E0_EV + 5e-7
    assert abs(2 * math.pi * F_RF_HZ * final[0, 4].item() / C_LIGHT) < 2e-4  # the phase offset it would take to find the gain: 5e-3 rad


def test_ramped_tracking_is_continuous_across_calls_and_reports_the_program_energy():
    params = Lattice(CIRC_M, E0_EV, .4, 2e5, F_RF_HZ, 0., 1e-4, SIGMA0)
    turns, end = 60, 1.002 * E0_EV
    ramp = ramp_of([0.0, turns * params.T0_s], [E0_EV, end], 2e5, F_RF_HZ, params.phi_s)

    def start():
        state = torch.zeros(50, 6, dtype=F64)
        state[:, 4] = torch.linspace(-2e-2, 2e-2, 50, dtype=F64)
        return state

    whole = NativeTracker(RING, params, ramp=ramp, device="cpu", generator=torch.Generator().manual_seed(9))
    one_go, result = whole.track(start(), turns, sample_every=10)
    split = NativeTracker(RING, params, ramp=ramp, device="cpu", generator=torch.Generator().manual_seed(9))
    midway, _ = split.track(start(), turns // 2)
    two_calls, _ = split.track(midway, turns // 2)
    torch.testing.assert_close(two_calls, one_go, rtol=1e-12, atol=1e-14)
    assert whole._elapsed_turns == split._elapsed_turns == turns
    np.testing.assert_allclose(result.E0_eV, ramp.at_array(result.turn * params.T0_s).E0_eV, rtol=1e-14)
    assert result.turn[0] == 0 and result.E0_eV[0] == E0_EV
    assert np.all(np.diff(result.E0_eV) > 0)  # the program energy rises monotonically
