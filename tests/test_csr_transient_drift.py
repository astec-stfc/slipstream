"""Entrance-transient and drift CSR kernels: limits that follow from the model, not from another code.

The reference numbers come from the closed-form mean loss of a Gaussian bunch in the steady state,
``K Q^2 / (4 pi eps0 rho^(2/3) sigma^(4/3))`` per unit length with ``K = 2^(8/3) Gamma(5/6) / (8 sqrt(pi) 3^(1/3))``, and from the
retarded-field check of ``slipstream_dev/Stories/CLARA_CSRValidation`` (not run here).
"""
import math

import numpy as np
import pytest
import torch

from slipstream.constants import EPS0_F_PER_M
from slipstream.tracking.native.collective import (
    csr_drift_delta_kick_fast, csr_drift_delta_kick_retarded, csr_steady_state_delta_kick_fast,
)
from slipstream.tracking.native.lattice_import import ElementSpec
from slipstream.tracking.native.linac import LinacTracker

SIGMA, CHARGE, RHO, PHI, ENERGY_EV = 20e-6, 100e-12, 1.5, 0.1325, 190e6
K_STEADY = 2 ** (8 / 3) * math.gamma(5 / 6) / (8 * math.sqrt(math.pi) * 3 ** (1 / 3))


def _bunch(n=2 ** 18, seed=0):
    return torch.tensor(np.sort(np.random.default_rng(seed).normal(0.0, SIGMA, n)), dtype=torch.float64)


def _mean_loss_V_per_m(kick, length):
    return float(-kick.mean()) * ENERGY_EV / length


def _steady_loss_V_per_m():
    return K_STEADY * CHARGE / (4 * math.pi * EPS0_F_PER_M) * RHO ** (-2 / 3) * SIGMA ** (-4 / 3)


def test_transient_with_slippage_beyond_the_bunch_is_the_steady_state():
    tau, energy = _bunch(), torch.tensor(ENERGY_EV, dtype=torch.float64)
    steady = csr_steady_state_delta_kick_fast(tau, CHARGE / len(tau), RHO, energy, 0.1, n_bins=400)
    transient = csr_steady_state_delta_kick_fast(tau, CHARGE / len(tau), RHO, energy, 0.1, n_bins=400, slippage_m=1.0)
    torch.testing.assert_close(transient, steady, rtol=0, atol=1e-18)


def test_transient_with_slippage_below_one_bin_is_zero():
    tau, energy = _bunch(), torch.tensor(ENERGY_EV, dtype=torch.float64)
    kick = csr_steady_state_delta_kick_fast(tau, CHARGE / len(tau), RHO, energy, 0.1, n_bins=400, slippage_m=1e-9)
    assert float(kick.abs().max()) < 1e-6 * float(csr_steady_state_delta_kick_fast(tau, CHARGE / len(tau), RHO, energy, 0.1, n_bins=400).abs().max())


def test_transient_loss_starts_small_overshoots_the_steady_state_and_settles_to_it():
    """The retarded-field reference (Stories/CLARA_CSRValidation) gives 0.07 of the steady value at 0.033 rad, 1.14 at 0.066 rad and 1.00 at the
    full bend angle of 0.1325 rad for this bunch: the loss overshoots before it settles."""
    tau, energy = _bunch(), torch.tensor(ENERGY_EV, dtype=torch.float64)
    steady = _steady_loss_V_per_m()
    losses = [_mean_loss_V_per_m(csr_steady_state_delta_kick_fast(tau, CHARGE / len(tau), RHO, energy, 0.01, n_bins=800,
                                                                   slippage_m=RHO * phi ** 3 / 24), 0.01) for phi in (0.03, 0.06, 0.1325)]
    assert losses[0] < 0.2 * steady
    assert losses[1] > 1.05 * steady
    assert losses[2] == pytest.approx(steady, rel=0.03)


def test_steady_state_mean_loss_matches_the_gaussian_closed_form():
    tau, energy = _bunch(2 ** 20), torch.tensor(ENERGY_EV, dtype=torch.float64)
    kick = csr_steady_state_delta_kick_fast(tau, CHARGE / len(tau), RHO, energy, 0.1, n_bins=400)
    assert _mean_loss_V_per_m(kick, 0.1) == pytest.approx(_steady_loss_V_per_m(), rel=0.01)


def test_retarded_drift_kick_continues_the_steady_state_at_the_exit_and_decays():
    tau, energy = _bunch(), torch.tensor(ENERGY_EV, dtype=torch.float64)
    loss = {z: _mean_loss_V_per_m(csr_drift_delta_kick_retarded(tau, CHARGE / len(tau), RHO, PHI, z, 0.01, energy, n_bins=400), 0.01)
            for z in (0.002, 0.05, 0.3, 1.0)}
    assert loss[0.002] == pytest.approx(_steady_loss_V_per_m(), rel=0.05)
    assert loss[0.002] > loss[0.05] > loss[0.3] > loss[1.0] > 0.0
    assert loss[1.0] < 0.1 * loss[0.002]


def test_stupakov_drift_kick_agrees_with_the_retarded_kernel_near_the_exit():
    tau, energy = _bunch(), torch.tensor(ENERGY_EV, dtype=torch.float64)
    q = CHARGE / len(tau)
    for z in (0.005, 0.05):
        a = _mean_loss_V_per_m(csr_drift_delta_kick_fast(tau, q, RHO, PHI, z, 0.01, energy, n_bins=400), 0.01)
        b = _mean_loss_V_per_m(csr_drift_delta_kick_retarded(tau, q, RHO, PHI, z, 0.01, energy, n_bins=400), 0.01)
        assert a == pytest.approx(b, rel=0.08)


def test_drift_kicks_vanish_without_charge_or_bend():
    tau, energy = _bunch(2 ** 12), torch.tensor(ENERGY_EV, dtype=torch.float64)
    assert float(csr_drift_delta_kick_retarded(tau, 0.0, RHO, PHI, 0.1, 0.01, energy, n_bins=64).abs().max()) == 0.0
    assert float(csr_drift_delta_kick_retarded(tau, 1e-15, RHO, 0.0, 0.1, 0.01, energy, n_bins=64).abs().max()) == 0.0
    assert float(csr_drift_delta_kick_fast(tau, 1e-15, RHO, 0.0, 0.1, 0.01, energy, n_bins=64).abs().max()) == 0.0


def test_tracker_options_are_validated():
    specs = [ElementSpec("D", "drift", 0.1)]
    with pytest.raises(ValueError):
        LinacTracker(specs, energy_eV=ENERGY_EV, csr_transient=True)           # needs csr and the fast kernels
    with pytest.raises(ValueError):
        LinacTracker(specs, energy_eV=ENERGY_EV, csr=True, csr_drift_model="stupakov")
    with pytest.raises(ValueError):
        LinacTracker(specs, energy_eV=ENERGY_EV, csr=True, fast_collective_effects=True, csr_drift_model="nonsense")


def _dipole_loss_eV(n_slices, transient=True, n=2 ** 17):
    """Bunch-mean energy loss of a monoenergetic Gaussian bunch through one VBC-like dipole (0.2 m, 0.1325 rad) cut into ``n_slices`` CSR kicks."""
    arc = RHO * PHI
    spec = ElementSpec("B", "bend", arc, angle=PHI, csr=True, csr_n_slices=n_slices, csr_bins=400)
    tracker = LinacTracker([spec], energy_eV=ENERGY_EV, device="cpu", dtype=torch.float64, bunch_charge_C=CHARGE, csr=True,
                           fast_collective_effects=True, csr_transient=transient)
    beam = torch.zeros(n, 6, dtype=torch.float64)
    beam[:, 4] = _bunch(n)
    out, energy = tracker.track(beam)
    return ENERGY_EV - float(energy) * (1.0 + float(out[:, 5].mean()))


def test_transient_loss_is_evaluated_at_the_slice_centre_and_converges_with_the_number_of_slices():
    # With the wake taken at the end of each slice an 8-slice dipole over-counts the rising transient by ~8%; at the centre the
    # loss is already converged at 8 slices and lies far below the steady-state loss of the same arc.
    coarse, fine = _dipole_loss_eV(8), _dipole_loss_eV(32)
    assert coarse == pytest.approx(fine, rel=0.02)
    steady = _dipole_loss_eV(8, transient=False)
    assert 0.6 < coarse / steady < 0.8                     # retarded-field ideal: 61.5 / 88.4 keV = 0.70 on this geometry
