"""Resistive-wall impedance of the multilayer round chamber, its Bessel building blocks and the Yokoya factors.

The references are the thick-wall resistive-wall formulas (Chao; Mounet's IW2D note), the Wronskian and recurrence
identities of the modified Bessel functions, and Yokoya's round and flat-plate limits of the elliptic form factors.
"""
import math
import warnings

import numpy as np
import pytest
import torch

from slipstream.wakefields.chambers import FreqScan, RoundChamber
from slipstream.wakefields.multilayer.bessel import (
    bessel_i_ratio, bessel_k_ratio, modified_bessel_i, modified_bessel_i_hat, modified_bessel_k, modified_bessel_k_hat,
)
from slipstream.wakefields.multilayer.round import (
    check_conditioning, estimate_max_bessel_argument, round_chamber_dipole_impedance, round_chamber_impedance,
)
from slipstream.wakefields.multilayer.yokoya import yokoya_elliptic_factors

C = 299792458.0
MU0 = 1.25663706212e-6
RHO_CU = 1.7e-8  # ohm m
GAMMA = 1e4
FREQ = np.array([1e7, 1e8, 1e9, 1e10])  # skin depth well below the 10 mm pipe radius and omega b / (gamma c) << 1
Z_ARGS = torch.tensor([0.7 + 0.3j, 2.0 + 1.5j, 5.0 - 2.0j, 30.0 + 10.0j], dtype=torch.complex128)


def copper_pipe(radius=10e-3, length=1.0, rho=RHO_CU, layers=None):
    return RoundChamber("pipe", radius, length, layers or [(rho, 0.0, 1.0)])  # (rho, tau_ps, thickness_m): thick wall


def as_complex(impedance):
    return impedance.ReZ_ohm + 1j * impedance.ImZ_ohm


@pytest.mark.parametrize("n", [0, 1, 2])
def test_modified_bessel_functions_obey_the_wronskian_and_the_ratio_and_scaled_forms(n):
    z = Z_ARGS
    wronskian = modified_bessel_i(n, z) * modified_bessel_k(n + 1, z) + modified_bessel_i(n + 1, z) * modified_bessel_k(n, z)
    torch.testing.assert_close(wronskian * z, torch.ones_like(z), rtol=0, atol=1e-13)  # I_n K_n+1 + I_n+1 K_n = 1/z
    torch.testing.assert_close(bessel_i_ratio(n + 1, z), modified_bessel_i(n, z) / modified_bessel_i(n + 1, z), rtol=1e-12, atol=0)
    torch.testing.assert_close(bessel_k_ratio(n + 1, z), modified_bessel_k(n, z) / modified_bessel_k(n + 1, z), rtol=1e-12, atol=0)
    torch.testing.assert_close(modified_bessel_i_hat(n, z), modified_bessel_i(n, z) * torch.exp(-z), rtol=1e-12, atol=0)
    torch.testing.assert_close(modified_bessel_k_hat(n, z), modified_bessel_k(n, z) * torch.exp(z), rtol=1e-12, atol=0)


def test_scaled_bessel_functions_follow_the_large_argument_asymptotes_without_overflow():
    z = torch.tensor([1e4 + 3e3j], dtype=torch.complex128)  # exp(z) would overflow I_n and underflow K_n
    assert (modified_bessel_i_hat(1, z) * torch.sqrt(2 * math.pi * z)).item() == pytest.approx(1.0, abs=1e-3)  # I_n ~ e^z / sqrt(2 pi z)
    assert (modified_bessel_k_hat(1, z) / torch.sqrt(math.pi / (2 * z))).item() == pytest.approx(1.0, abs=1e-3)  # K_n ~ sqrt(pi / 2z) e^-z


@pytest.mark.parametrize("function", [
    lambda z: modified_bessel_i(1, z), lambda z: modified_bessel_k(1, z), lambda z: bessel_i_ratio(1, z),
    lambda z: bessel_k_ratio(2, z), lambda z: modified_bessel_i_hat(1, z), lambda z: modified_bessel_k_hat(0, z),
])
def test_bessel_derivatives_match_finite_differences(function):
    z = torch.tensor([1.3 + 0.4j, 0.8 + 1.1j], dtype=torch.complex128, requires_grad=True)
    assert torch.autograd.gradcheck(function, (z,), eps=1e-6, atol=1e-6, rtol=1e-5)


def test_thick_wall_impedance_is_the_resistive_wall_formula_and_inductive():
    radius, length = 10e-3, 1.0
    z = as_complex(round_chamber_impedance(copper_pipe(radius, length), FREQ, GAMMA))
    skin_depth = np.sqrt(2 * RHO_CU / (MU0 * 2 * np.pi * FREQ))
    expected = length * RHO_CU / (2 * np.pi * radius * skin_depth)  # Re Z = L rho / (2 pi b delta), Im Z = Re Z
    np.testing.assert_allclose(z.real, expected, rtol=0.01)
    np.testing.assert_allclose(z.imag, z.real, rtol=0.01)


def test_dipole_impedance_is_the_transverse_resistive_wall_relation_to_the_longitudinal_one():
    radius = 10e-3
    chamber = copper_pipe(radius)
    longitudinal = as_complex(round_chamber_impedance(chamber, FREQ, GAMMA))
    dipole = as_complex(round_chamber_dipole_impedance(chamber, FREQ, GAMMA))
    np.testing.assert_allclose(dipole, 2 * C / (2 * np.pi * FREQ * radius**2) * longitudinal, rtol=0.01)  # Z_dip = 2 c Z_long / (omega b^2)


def test_resistive_wall_scales_as_sqrt_frequency_sqrt_resistivity_length_and_inverse_radius():
    base = as_complex(round_chamber_impedance(copper_pipe(), FREQ, GAMMA))
    np.testing.assert_allclose(base[1:] / base[:-1], np.sqrt(10.0), rtol=0.01)
    bigger = as_complex(round_chamber_impedance(copper_pipe(radius=20e-3, length=3.0, rho=4 * RHO_CU), FREQ, GAMMA))
    np.testing.assert_allclose(bigger / base, 2.0 * 3.0 / 2.0, rtol=0.01)  # sqrt(4) * 3 / 2


def test_a_wall_split_into_identical_layers_equals_the_single_thick_wall():
    single = as_complex(round_chamber_impedance(copper_pipe(), FREQ, GAMMA))
    split = as_complex(round_chamber_impedance(copper_pipe(layers=[(RHO_CU, 0.0, 3e-3), (RHO_CU, 0.0, 1.0)]), FREQ, GAMMA))
    np.testing.assert_allclose(split, single, rtol=1e-9)


def test_the_conditioning_estimate_is_the_thick_wall_argument_and_warns_beyond_the_validated_range():
    radius = 10e-3
    chamber = copper_pipe(radius)
    skin_depth = math.sqrt(2 * RHO_CU / (MU0 * 2 * math.pi * 1e9))
    assert estimate_max_bessel_argument(chamber, np.array([1e9]), GAMMA) == pytest.approx(math.sqrt(2) * radius / skin_depth, rel=1e-3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_conditioning(chamber, np.array([1e9]), GAMMA)  # inside the validated range: silent
    with pytest.warns(RuntimeWarning, match="Bessel argument"):
        check_conditioning(chamber, np.array([1e15]), GAMMA)


def test_yokoya_factors_reduce_to_the_round_pipe_and_to_the_flat_plates():
    long, dip_x, dip_y, quad_x, quad_y = yokoya_elliptic_factors(10e-3, 10e-3)  # round: every dipole factor is one, no quadrupole
    assert (long, dip_x, dip_y) == pytest.approx((1.0, 1.0, 1.0), abs=0.01) and (quad_x, quad_y) == pytest.approx((0.0, 0.0), abs=0.01)
    long, dip_x, dip_y, quad_x, quad_y = yokoya_elliptic_factors(1e-3, 1.0)  # flat plates, narrow gap vertical
    assert long == 1.0
    assert dip_y == pytest.approx(math.pi**2 / 12, rel=0.01) and dip_x == pytest.approx(math.pi**2 / 24, rel=0.01)
    assert quad_y == pytest.approx(math.pi**2 / 24, rel=0.01) and quad_x == pytest.approx(-math.pi**2 / 24, rel=0.02)


def test_yokoya_factors_depend_only_on_the_aspect_ratio_and_need_the_semi_axes_in_order():
    assert yokoya_elliptic_factors(2e-3, 6e-3) == pytest.approx(yokoya_elliptic_factors(1.0, 3.0), rel=1e-12)
    with pytest.raises(ValueError, match="must not exceed"):
        yokoya_elliptic_factors(6e-3, 2e-3)


def test_frequency_scan_grids_have_the_requested_spacing_and_refinement():
    log = FreqScan(start_exp=3, stop_exp=5, scan_type=0, pts_per_decade=10).build_freq_hz()
    assert len(log) == 21 and log[0] == pytest.approx(1e3) and log[-1] == pytest.approx(1e5)
    np.testing.assert_allclose(log[1:] / log[:-1], 10 ** 0.1, rtol=1e-12)
    linear = FreqScan(start_exp=3, stop_exp=5, scan_type=1, lin_exp=3).build_freq_hz()
    np.testing.assert_allclose(np.diff(linear), 1e3, rtol=1e-12)
    refined = FreqScan(start_exp=3, stop_exp=5, scan_type=2, pts_per_decade=10, refine_fmin_THz=1, refine_fmax_THz=2,
                       refine_n=10).build_freq_hz()
    assert len(refined) == 21 + 11 and (np.diff(refined) > 0).all() and refined[-1] == pytest.approx(2e12)
