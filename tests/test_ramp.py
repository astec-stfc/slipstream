"""Energy ramps: linear interpolation of the RF program and the scaling laws of fixed-geometry radiation parameters.

Energy loss per turn goes as gamma^4, damping time as gamma^-3, natural energy spread as gamma and equilibrium
emittance as gamma^2 (synchrotron radiation in a fixed lattice).
"""
import math

import numpy as np
import pytest

from slipstream.constants import ELECTRON_MASS_EV
from slipstream.lattice.lattice import Lattice
from slipstream.lattice.optics import TransverseOptics
from slipstream.lattice.ramp import RampProfile, scaled_lattice, scaled_optics

LATTICE = Lattice(circumference_m=240.0, E0_eV=1e9, alpha_c=1.7e-4, Vrf_V=2.5e6, f_rf_Hz=499.654e6, U0_eV=2.0e5, tau_z_s=0.02,
                  sigma_delta0=5e-4)
OPTICS = TransverseOptics(beta_x_m=8.0, alpha_x=-0.5, D_x_m=0.1, Q_x=10.2, Qp_x=-12.0, tau_x_s=0.03, epsx_m=1e-9,
                          beta_y_m=6.0, alpha_y=0.2, D_y_m=0.0, Q_y=5.3, Qp_y=-8.0, tau_y_s=0.04, epsy_m=1e-11)


def ramp():
    return RampProfile(time_s=np.array([0.0, 1.0, 3.0]), E0_eV=np.array([1e9, 2e9, 2e9]), Vrf_V=np.array([1e6, 2e6, 4e6]),
                       f_rf_Hz=np.array([5e8, 5e8, 5.001e8]), phi_rf_rad=np.array([0.0, 1.0, 1.0]))


def test_ramp_interpolates_linearly_between_samples_and_holds_the_end_values():
    mid = ramp().at(0.5)
    assert (mid.E0_eV, mid.Vrf_V, mid.f_rf_Hz, mid.phi_rf_rad) == pytest.approx((1.5e9, 1.5e6, 5e8, 0.5))
    assert ramp().at(2.0).Vrf_V == pytest.approx(3e6)
    assert ramp().at(-1.0).E0_eV == 1e9 and ramp().at(10.0).Vrf_V == 4e6  # outside the program the end value holds
    arrays = ramp().at_array(np.array([0.0, 0.5, 2.0, 3.0]))
    np.testing.assert_allclose(arrays.E0_eV, [1e9, 1.5e9, 2e9, 2e9])
    np.testing.assert_allclose(arrays.f_rf_Hz, [5e8, 5e8, 5.0005e8, 5.001e8])


def test_ramp_file_gives_total_energy_from_the_dimensionless_momentum_and_phase_in_radians(tmp_path):
    from slipstream.sdds_io import sdds_object

    path = tmp_path / "ramp.sdds"
    columns = {"time": [0.0, 0.5, 1.0], "momentum": [100.0, 200.0, 300.0], "Voltage": [1e6, 2e6, 3e6],
               "Frequency": [5e8, 5e8, 5e8], "Phase": [0.0, 90.0, 180.0]}
    with sdds_object() as (sdds, x):
        x.mode = sdds.SDDS_ASCII
        for name, values in columns.items():
            x.defineColumn(name, type=sdds.SDDS_DOUBLE)
            x.setColumnValueList(name, values)
        x.save(str(path))
    profile = RampProfile.from_sdds_file(path)
    np.testing.assert_allclose(profile.E0_eV, np.sqrt(np.array(columns["momentum"]) ** 2 + 1) * ELECTRON_MASS_EV)  # E = sqrt(p^2 + 1) mc^2
    np.testing.assert_allclose(profile.phi_rf_rad, [0.0, math.pi / 2, math.pi])
    assert profile.at(0.25).Vrf_V == pytest.approx(1.5e6)


@pytest.mark.parametrize("ratio", [0.5, 1.0, 2.0, 3.0])
def test_lattice_radiation_parameters_scale_with_the_energy_ratio(ratio):
    scaled = scaled_lattice(LATTICE, ratio * LATTICE.gamma)
    assert scaled.E0_eV == pytest.approx(ratio * LATTICE.E0_eV, rel=1e-12)
    assert scaled.U0_eV == pytest.approx(LATTICE.U0_eV * ratio**4, rel=1e-12)
    assert scaled.tau_z_s == pytest.approx(LATTICE.tau_z_s / ratio**3, rel=1e-12)
    assert scaled.sigma_delta0 == pytest.approx(LATTICE.sigma_delta0 * ratio, rel=1e-12)
    assert (scaled.alpha_c, scaled.Vrf_V, scaled.circumference_m) == (LATTICE.alpha_c, LATTICE.Vrf_V, LATTICE.circumference_m)
    assert scaled.gamma == pytest.approx(ratio * LATTICE.gamma, rel=1e-12)


def test_scaling_in_two_steps_equals_scaling_once_and_the_original_is_untouched():
    twice = scaled_lattice(scaled_lattice(LATTICE, 2 * LATTICE.gamma), 3 * LATTICE.gamma)
    once = scaled_lattice(LATTICE, 3 * LATTICE.gamma)
    assert (twice.U0_eV, twice.tau_z_s, twice.sigma_delta0) == pytest.approx((once.U0_eV, once.tau_z_s, once.sigma_delta0), rel=1e-12)
    assert LATTICE.E0_eV == 1e9 and LATTICE.U0_eV == 2.0e5


def test_transverse_damping_times_and_emittances_scale_with_the_energy_ratio():
    scaled = scaled_optics(OPTICS, 2.0)
    assert (scaled.tau_x_s, scaled.tau_y_s) == pytest.approx((OPTICS.tau_x_s / 8, OPTICS.tau_y_s / 8), rel=1e-12)
    assert (scaled.epsx_m, scaled.epsy_m) == pytest.approx((OPTICS.epsx_m * 4, OPTICS.epsy_m * 4), rel=1e-12)
    assert (scaled.beta_x_m, scaled.Q_x, scaled.D_x_m) == (OPTICS.beta_x_m, OPTICS.Q_x, OPTICS.D_x_m)  # the optics itself is fixed
