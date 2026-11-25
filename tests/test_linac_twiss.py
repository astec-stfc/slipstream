"""Linear Twiss and dispersion propagation through a non-periodic linac line, against transfer-matrix closed forms and
against statistics of a tracked particle ensemble.

An accelerating section changes the geometric emittance by det R = p0 / p1, so its Twiss parameters (defined against the
emittance after the section) and its dispersion (defined against the local reference momentum) differ from a naive matrix
propagation by the momentum ratio; the ensemble comparison pins that.
"""
import math

import numpy as np
import pytest
import torch

from slipstream.tracking.native.lattice_import import ElementSpec
from slipstream.tracking.native.linac import LinacTracker
from slipstream.tracking.native.linac_twiss import LinacTwissTable, _plane_twiss_from_ensemble, linac_twiss_table, twiss_from_trajectory

MC2_EV = 0.51099895e6
ENERGY_EV = 1e8
BETA0, ALPHA0 = 5.0, -1.2
RF_FREQUENCY_HZ = 2.998e9


def propagate(matrix, beta, alpha):
    """Twiss through a 2x2 matrix, by the textbook formula (``beta gamma - alpha^2 = 1`` is kept)."""
    gamma = (1 + alpha**2) / beta
    (a, b), (c, d) = matrix
    return a * a * beta - 2 * a * b * alpha + b * b * gamma, -a * c * beta + (a * d + b * c) * alpha - b * d * gamma


def line_table(specs, **start):
    start = {"beta_x0": BETA0, "alpha_x0": ALPHA0, "beta_y0": BETA0, "alpha_y0": ALPHA0, **start}
    return linac_twiss_table(specs, ENERGY_EV, MC2_EV, **start)


def test_drift_propagates_beta_alpha_and_dispersion_in_closed_form():
    length, d0, dp0 = 2.0, 0.1, 0.02
    table = line_table([ElementSpec("D", "drift", length)], Dx0=d0, Dpx0=dp0)
    beta, alpha = propagate([[1, length], [0, 1]], BETA0, ALPHA0)
    assert table.beta_x_m[1] == pytest.approx(beta, rel=1e-9) and table.alpha_x[1] == pytest.approx(alpha, rel=1e-9)
    assert table.D_x_m[1] == pytest.approx(d0 + dp0 * length, rel=1e-9) and table.Dp_x[1] == pytest.approx(dp0, rel=1e-9)
    assert list(table.s_m) == [0.0, length] and table.element_name == ["_INJECTION_", "D"]


def test_quadrupole_focuses_one_plane_and_defocuses_the_other():
    k, length = 2.0, 0.3
    phi = math.sqrt(k) * length
    focusing = [[math.cos(phi), math.sin(phi) / math.sqrt(k)], [-math.sqrt(k) * math.sin(phi), math.cos(phi)]]
    defocusing = [[math.cosh(phi), math.sinh(phi) / math.sqrt(k)], [math.sqrt(k) * math.sinh(phi), math.cosh(phi)]]
    table = line_table([ElementSpec("Q", "quad", length, k1=k)])
    for got, matrix, start in ((table.beta_x_m[1], focusing, (BETA0, ALPHA0)), (table.beta_y_m[1], defocusing, (BETA0, ALPHA0))):
        assert got == pytest.approx(propagate(matrix, *start)[0], rel=1e-6)
    assert table.alpha_x[1] == pytest.approx(propagate(focusing, BETA0, ALPHA0)[1], rel=1e-6)
    assert table.alpha_y[1] == pytest.approx(propagate(defocusing, BETA0, ALPHA0)[1], rel=1e-6)


def test_sector_bend_makes_the_dispersion_of_a_circular_arc():
    angle, length = 0.1, 1.0
    rho = length / angle
    table = line_table([ElementSpec("B", "bend", length, angle=angle)])
    assert table.D_x_m[1] == pytest.approx(rho * (1 - math.cos(angle)), rel=1e-8)  # D = rho (1 - cos theta)
    assert table.Dp_x[1] == pytest.approx(math.sin(angle), rel=1e-8)  # D' = sin theta
    arc = [[math.cos(angle), rho * math.sin(angle)], [-math.sin(angle) / rho, math.cos(angle)]]
    assert table.beta_x_m[1] == pytest.approx(propagate(arc, BETA0, ALPHA0)[0], rel=1e-8)
    assert table.D_y_m[1] == 0.0


def test_a_matched_fodo_cell_repeats_its_twiss_parameters_and_energy_is_unchanged():
    k, lq, ld = 1.2, 0.2, 1.8
    cell = [ElementSpec("QF", "quad", lq, k1=k), ElementSpec("D1", "drift", ld),
            ElementSpec("QD", "quad", lq, k1=-k), ElementSpec("D2", "drift", ld)]
    phi = math.sqrt(k) * lq
    qf = np.array([[math.cos(phi), math.sin(phi) / math.sqrt(k)], [-math.sqrt(k) * math.sin(phi), math.cos(phi)]])
    qd = np.array([[math.cosh(phi), math.sinh(phi) / math.sqrt(k)], [math.sqrt(k) * math.sinh(phi), math.cosh(phi)]])
    drift = np.array([[1.0, ld], [0.0, 1.0]])
    one_cell = drift @ qd @ drift @ qf
    mu = math.acos((one_cell[0, 0] + one_cell[1, 1]) / 2)
    beta, alpha = one_cell[0, 1] / math.sin(mu), (one_cell[0, 0] - one_cell[1, 1]) / (2 * math.sin(mu))
    table = line_table(cell * 3, beta_x0=beta, alpha_x0=alpha)  # the periodic solution of the cell, from numpy matrices
    for row in (4, 8, 12):
        assert table.beta_x_m[row] == pytest.approx(beta, rel=1e-6) and table.alpha_x[row] == pytest.approx(alpha, abs=1e-6)
    assert (table.energy_eV == ENERGY_EV).all()


@pytest.mark.parametrize("phase", [math.pi / 2, math.pi / 3, 0.0])
def test_rf_section_gains_voltage_times_sine_of_the_phase(phase):
    voltage = 1e6  # a small gain and a long wavelength keep the phase slip across the cavity negligible
    cavity = ElementSpec("RF", "rf", 1.0, voltage_V=voltage, phase_rad=phase, frequency_Hz=RF_FREQUENCY_HZ / 10, n_kicks=16)
    table = line_table([cavity])
    assert table.energy_eV[1] - ENERGY_EV == pytest.approx(voltage * math.sin(phase), abs=1e-3 * voltage)


def tracked_ensemble(n=20000, dispersion=(0.3, 0.05), seed=3):
    """Gaussian ensemble whose sample covariance is exactly the matched one (whitened), with x = D delta added."""
    generator = torch.Generator().manual_seed(seed)
    z = torch.randn(n, 6, generator=generator, dtype=torch.float64)
    z = z - z.mean(0)
    z = z @ torch.linalg.inv(torch.linalg.cholesky(z.T @ z / n)).T
    eps = 1e-6 / math.sqrt((ENERGY_EV / MC2_EV) ** 2 - 1)
    delta = 1e-4 * z[:, 5]
    x = math.sqrt(eps * BETA0) * z[:, 0] + dispersion[0] * delta
    px = math.sqrt(eps / BETA0) * (z[:, 1] - ALPHA0 * z[:, 0]) + dispersion[1] * delta
    y = math.sqrt(eps * BETA0) * z[:, 2]
    py = math.sqrt(eps / BETA0) * (z[:, 3] - ALPHA0 * z[:, 2])
    return torch.stack([x, px, y, py, 1e-3 / 3e8 * z[:, 4], delta], dim=1)


def test_ensemble_statistics_recover_the_injected_twiss_and_dispersion():
    state = tracked_ensemble()
    x, px, delta = state[:, 0], state[:, 1], state[:, 5]
    beta, alpha, d, dp = _plane_twiss_from_ensemble(x, px / (1 + delta), delta)
    assert (beta, alpha) == pytest.approx((BETA0, ALPHA0), rel=1e-4)
    assert (d, dp) == pytest.approx((0.3, 0.05), rel=1e-4)


@pytest.mark.parametrize("name,specs", [
    ("optics only", [ElementSpec("D", "drift", 1.0), ElementSpec("Q", "quad", 0.2, k1=1.5), ElementSpec("D2", "drift", 1.0)]),
    ("rf then drift", [ElementSpec("D", "drift", 0.5),
                       ElementSpec("RF", "rf", 1.0, voltage_V=2e7, phase_rad=math.pi / 2, frequency_Hz=RF_FREQUENCY_HZ, n_kicks=16),
                       ElementSpec("D2", "drift", 2.0)]),
])
def test_linear_twiss_table_agrees_row_for_row_with_the_tracked_ensemble(name, specs):
    table = line_table(specs, Dx0=0.3, Dpx0=0.05)
    tracker = LinacTracker(specs, energy_eV=ENERGY_EV, mc2_eV=MC2_EV, device="cpu", dtype=torch.float64, compile=False)
    _, _, trajectory, energy = tracker.track(tracked_ensemble(), return_trajectory=True)
    tracked = twiss_from_trajectory(trajectory, energy, specs)
    np.testing.assert_allclose(tracked.energy_eV, table.energy_eV, rtol=1e-6)
    np.testing.assert_allclose(tracked.s_m, table.s_m)
    np.testing.assert_allclose(tracked.beta_x_m, table.beta_x_m, rtol=2e-3)
    np.testing.assert_allclose(tracked.alpha_x, table.alpha_x, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(tracked.D_x_m, table.D_x_m, rtol=2e-3)
    np.testing.assert_allclose(tracked.Dp_x, table.Dp_x, rtol=2e-3, atol=1e-4)
    np.testing.assert_allclose(tracked.beta_y_m, table.beta_y_m, rtol=2e-3)


def test_twiss_table_round_trips_through_hdf5(tmp_path):
    table = line_table([ElementSpec("D", "drift", 1.0), ElementSpec("Q", "quad", 0.2, k1=1.5)], Dx0=0.1)
    path = tmp_path / "twiss.h5"
    table.save(path)
    table.save(path, group="second")
    for loaded in (LinacTwissTable.load(path), LinacTwissTable.load(path, group="second")):
        assert loaded.element_name == table.element_name
        np.testing.assert_array_equal(loaded.beta_x_m, table.beta_x_m)
        np.testing.assert_array_equal(loaded.D_x_m, table.D_x_m)
