"""Tests for slipstream.lattice.Lattice."""
from __future__ import annotations

import math

import pytest

from slipstream.lattice.lattice import Lattice

def test_yaml_round_trip(tmp_path):
    lat = Lattice(
        circumference_m=100.0, E0_eV=1e9, alpha_c=1e-4, Vrf_V=5e5,
        f_rf_Hz=3e8, U0_eV=1e5, tau_z_s=1e-2, sigma_delta0=5e-4,
    )
    path = tmp_path / "lattice.yaml"
    lat.to_yaml(path)
    loaded = Lattice.from_yaml(path)
    assert loaded == lat


def test_non_relativistic_rest_mass_gives_correct_gamma_beta_and_T0():
    """rest_mass_eV=PROTON_MASS_EV at a genuinely non-relativistic energy
    (ELENA-like, P0=13.7 MeV/c) must flow through gamma/beta_rel/T0_s
    exactly -- the bug this generalizes away from was T0_s silently using
    circumference_m/C_LIGHT (beta_rel=1) unconditionally."""
    from slipstream.constants import PROTON_MASS_EV

    P0_MeV = 13.7
    m_MeV = PROTON_MASS_EV / 1e6
    E_tot_MeV = math.sqrt(P0_MeV**2 + m_MeV**2)
    expected_gamma = E_tot_MeV / m_MeV
    expected_beta_rel = P0_MeV / E_tot_MeV

    lat = Lattice(
        circumference_m=30.4055, E0_eV=E_tot_MeV * 1e6, alpha_c=0.249, Vrf_V=100.0,
        f_rf_Hz=144_000.0, U0_eV=0.0, tau_z_s=1234.0, sigma_delta0=1e-4,
        rest_mass_eV=PROTON_MASS_EV,
    )
    assert lat.gamma == pytest.approx(expected_gamma)
    assert lat.beta_rel == pytest.approx(expected_beta_rel)
    assert lat.T0_s == pytest.approx(30.4055 / (expected_beta_rel * 299792458.0))


def test_default_rest_mass_is_electron_and_ultra_relativistic():
    """Default (no rest_mass_eV given) stays byte-for-byte the old
    ultra-relativistic-electron behavior -- beta_rel~1, T0_s~circumference/c."""
    lat = Lattice(
        circumference_m=480.0, E0_eV=2.2e9, alpha_c=9.4e-5, Vrf_V=1.8e6,
        f_rf_Hz=499.8e6, U0_eV=4e5, tau_z_s=1e-2, sigma_delta0=1e-3,
    )
    assert lat.beta_rel == pytest.approx(1.0, abs=1e-7)
    assert lat.T0_s == pytest.approx(480.0 / 299792458.0, rel=1e-7)



