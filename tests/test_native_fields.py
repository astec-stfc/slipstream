"""Tests for ``tracking.native.fields`` -- on-axis field-map reading and
paraxial off-axis expansion. See ``plans/NATIVE_GUN_SPACE_CHARGE_PLAN.md``
for the derivation this checks (both formulas verified against a primary
source/first-principles Maxwell derivation before being written down, not
just trusted from memory -- this file is the executable record of that
verification, not just the module docstring's prose claim)."""
from __future__ import annotations

import math

import numpy as np
import torch

from slipstream.constants import C_LIGHT
from slipstream.tracking.native.fields import (
    OnAxisFieldMap,
    RFCavityField,
    SolenoidField,
    TravelingWaveField,
)


def test_on_axis_field_map_matches_analytic_sine():
    z = np.linspace(0.0, 1.0, 101)
    f = np.sin(2 * math.pi * z)
    field_map = OnAxisFieldMap(z, f, n_samples=4001, device="cpu", dtype=torch.float64)

    z_test = torch.linspace(0.05, 0.95, 37, dtype=torch.float64)
    f_val, df_val, d2f_val = field_map(z_test)

    f_exact = torch.sin(2 * math.pi * z_test)
    df_exact = 2 * math.pi * torch.cos(2 * math.pi * z_test)
    d2f_exact = -((2 * math.pi) ** 2) * torch.sin(2 * math.pi * z_test)

    assert torch.allclose(f_val, f_exact, atol=1e-5)
    assert torch.allclose(df_val, df_exact, atol=1e-3)
    assert torch.allclose(d2f_val, d2f_exact, atol=5e-2)


def test_on_axis_field_map_zero_outside_range():
    z = np.linspace(0.0, 1.0, 11)
    f = np.ones_like(z)
    field_map = OnAxisFieldMap(z, f, n_samples=101, device="cpu", dtype=torch.float64)
    f_val, df_val, d2f_val = field_map(torch.tensor([-1.0, 2.0], dtype=torch.float64))
    assert torch.equal(f_val, torch.zeros(2, dtype=torch.float64))
    assert torch.equal(df_val, torch.zeros(2, dtype=torch.float64))
    assert torch.equal(d2f_val, torch.zeros(2, dtype=torch.float64))


def test_solenoid_on_axis_matches_scaled_profile():
    z = np.linspace(-1.0, 1.0, 401)
    bz = np.exp(-(z**2) / 0.1)
    b_max_T = 0.35
    field = SolenoidField(z, bz, b_max_T, device="cpu", dtype=torch.float64)

    z_test = torch.tensor([-0.3, 0.0, 0.4], dtype=torch.float64)
    zero = torch.zeros_like(z_test)
    Bx, By, Bz = field(zero, zero, z_test)
    assert torch.allclose(Bx, zero, atol=1e-10)
    assert torch.allclose(By, zero, atol=1e-10)
    expected = b_max_T * torch.exp(-(z_test**2) / 0.1)
    assert torch.allclose(Bz, expected, atol=1e-4)


def test_solenoid_br_matches_linear_ramp_analytic():
    # Bz0(z) = z over a wide range -> dBz0/dz = 1 exactly (away from the
    # spline's own edges), d2Bz0/dz2 = 0 -> Br(r,z) = -(r/2)*1 analytically.
    z = np.linspace(-1.0, 2.0, 301)
    bz = z.copy()
    peak = np.abs(bz).max()  # at z=2
    field = SolenoidField(z, bz, b_max_T=peak, device="cpu", dtype=torch.float64)

    x = torch.tensor([0.01, -0.02, 0.03], dtype=torch.float64)
    y = torch.tensor([0.0, 0.01, -0.01], dtype=torch.float64)
    z_t = torch.tensor([0.5, 0.5, 1.0], dtype=torch.float64)
    Bx, By, Bz = field(x, y, z_t)
    assert torch.allclose(Bx, -0.5 * x, atol=1e-6)
    assert torch.allclose(By, -0.5 * y, atol=1e-6)
    assert torch.allclose(Bz, z_t, atol=1e-6)


def test_rf_cavity_field_quadrature_matches_analytic_linear_ramp():
    # Ez0(z) = z (scaled so the loaded peak maps to e_max_V_per_m=peak,
    # i.e. the physical on-axis field equals the raw profile exactly) ->
    # dEz0/dz = 1 analytically. At phase=0 the field is purely
    # electric (Er=-(r/2)*1, Bphi=0); at phase=pi/2 it's purely magnetic
    # (Er=0, Bphi=-(r*omega/(2c^2))*Ez0(z)) -- the standard RF standing-wave
    # E/B quadrature, and a check on the sign derived in this module's
    # docstring (an earlier draft had the wrong Bphi sign).
    z = np.linspace(-1.0, 2.0, 301)
    ez = z.copy()
    peak = np.abs(ez).max()  # at z=2
    frequency_hz = 3e9
    field = RFCavityField(
        z, ez, e_max_V_per_m=peak, frequency_hz=frequency_hz, device="cpu", dtype=torch.float64
    )
    omega = 2 * math.pi * frequency_hz

    x = torch.tensor([0.01, -0.02], dtype=torch.float64)
    y = torch.tensor([0.0, 0.01], dtype=torch.float64)
    z_t = torch.tensor([0.5, 1.0], dtype=torch.float64)

    t_zero = torch.zeros_like(x)
    Ex, Ey, Ez, Bx, By, Bz = field(x, y, z_t, t_zero)
    assert torch.allclose(Ex, -0.5 * x, atol=1e-6)
    assert torch.allclose(Ey, -0.5 * y, atol=1e-6)
    assert torch.allclose(Ez, z_t, atol=1e-6)
    assert torch.allclose(Bx, torch.zeros_like(Bx), atol=1e-10)
    assert torch.allclose(By, torch.zeros_like(By), atol=1e-10)

    t_quarter = torch.full_like(x, (math.pi / 2) / omega)
    Ex, Ey, Ez, Bx, By, Bz = field(x, y, z_t, t_quarter)
    assert torch.allclose(Ex, torch.zeros_like(Ex), atol=1e-6)
    assert torch.allclose(Ey, torch.zeros_like(Ey), atol=1e-6)
    bphi_over_r_expected = -(omega / (2.0 * C_LIGHT**2)) * z_t
    assert torch.allclose(Bx, -bphi_over_r_expected * y, atol=1e-12)
    assert torch.allclose(By, bphi_over_r_expected * x, atol=1e-12)


def test_tws_reduces_to_standing_wave_when_quadrature_part_vanishes():
    # With Ei = 0 everywhere the field must collapse onto RFCavityField's
    # own already-verified standing-wave formulas -- a regression against
    # validated code, not a fresh closed form. theta=pi makes sin(theta)=0
    # unusable, so build the degenerate case by zeroing map_imag directly
    # after construction.
    z = np.linspace(0.0, 1.0, 501)
    ez = np.sin(math.pi * z)
    freq = 3e9
    peak = float(np.abs(ez).max())
    tw = TravelingWaveField(
        z, ez, z1=0.25, z2=0.75, n=1, m=3, e_max_V_per_m=peak, frequency_hz=freq,
        n_cells=3, device="cpu", dtype=torch.float64,
    )
    tw.map_imag.f = torch.zeros_like(tw.map_imag.f)
    tw.map_imag.df = torch.zeros_like(tw.map_imag.df)
    sw = RFCavityField(z, ez, e_max_V_per_m=peak, frequency_hz=freq, device="cpu", dtype=torch.float64)

    x = torch.tensor([0.01, -0.02], dtype=torch.float64)
    y = torch.tensor([0.0, 0.01], dtype=torch.float64)
    # Inside the first tile, where the folded lookup is the identity.
    z_t = torch.tensor([0.3, 0.45], dtype=torch.float64)
    t = torch.full_like(x, 0.37e-10)
    for a, b in zip(tw(x, y, z_t, t), sw(x, y, z_t, t)):
        assert torch.allclose(a, b, atol=1e-12)



