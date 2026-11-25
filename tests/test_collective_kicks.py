"""Longitudinal space-charge (LSC) and steady-state CSR energy kicks of a Gaussian bunch against their closed forms.

LSC: ``E_z = Z0 c (1 + 2 ln(b/a)) / (4 pi gamma^2) * d(lambda)/d(tau)``, the on-axis field of a long bunch in a round pipe.
CSR: the steady-state mean loss ``K Q / (4 pi eps0) rho^(-2/3) sigma^(-4/3)``, ``K = 2^(8/3) Gamma(5/6) / (8 sqrt(pi) 3^(1/3))``.
The kernel-density versions smooth the line density with a Silverman bandwidth, so they agree to a few percent; the fixed-grid
versions to a fraction of a percent. The block-multipole CSR sum is an approximation of the exact pairwise one.
"""
import math

import numpy as np
import pytest
import torch
from scipy.special import ndtri

from slipstream.constants import C_LIGHT, EPS0_F_PER_M
from slipstream.tracking.native.collective import (
    csr_steady_state_delta_kick, csr_steady_state_delta_kick_fast, csr_steady_state_delta_kick_subquadratic, lsc_delta_kick,
    lsc_delta_kick_fast,
)

F64 = torch.float64
Z0_OHM = 376.730313668
N = 3000
SIGMA_M, CHARGE_C, ENERGY_EV, MC2_EV = 20e-6, 100e-12, 190e6, 0.51099895e6
PIPE_RADIUS_M, LENGTH_M, RHO_M = 5e-3, 0.5, 1.5
K_STEADY = 2 ** (8 / 3) * math.gamma(5 / 6) / (8 * math.sqrt(math.pi) * 3 ** (1 / 3))
ENERGY = torch.tensor(ENERGY_EV, dtype=F64)
MC2 = torch.tensor(MC2_EV, dtype=F64)


def gaussian_quantiles(sigma, seed=None):
    """N points at the Gaussian quantiles: the smoothest sample of the distribution (no sampling noise)."""
    points = sigma * ndtri((np.arange(N) + 0.5) / N)
    return torch.tensor(points if seed is None else points[np.random.default_rng(seed).permutation(N)], dtype=F64)


TAU = gaussian_quantiles(SIGMA_M)
X, Y = gaussian_quantiles(30e-6, seed=1), gaussian_quantiles(30e-6, seed=2)


def lsc_closed_form(pipe_radius=PIPE_RADIUS_M):
    a_m = 1.7 * 0.5 * (X.std() + Y.std()).item()
    form = 1 + 2 * math.log(pipe_radius / a_m)
    gamma = ENERGY_EV / MC2_EV
    line_density = CHARGE_C / (math.sqrt(2 * math.pi) * SIGMA_M) * torch.exp(-TAU**2 / (2 * SIGMA_M**2))
    # kick = -V / E with V = Z0 c L F / (4 pi gamma^2) d(lambda)/d(tau) and d(lambda)/d(tau) = -tau lambda / sigma^2
    return Z0_OHM * C_LIGHT * LENGTH_M * form / (4 * math.pi * gamma**2 * ENERGY_EV) * TAU / SIGMA_M**2 * line_density


@pytest.mark.parametrize("kick,tolerance", [(lsc_delta_kick, 0.06), (lsc_delta_kick_fast, 0.01)])
def test_lsc_kick_of_a_gaussian_bunch_is_the_derivative_of_its_line_density(kick, tolerance):
    got = kick(TAU, CHARGE_C / N, X, Y, ENERGY, MC2, LENGTH_M, pipe_radius_b_m=PIPE_RADIUS_M)
    expected = lsc_closed_form()
    core = TAU.abs() < 1.5 * SIGMA_M
    assert ((got - expected).abs()[core].max() / expected.abs().max()).item() < tolerance
    assert got[TAU > 0].min() > 0 > got[TAU < 0].max()  # the sign of the chirp: positive tau gains, negative tau loses
    assert got.sum().abs().item() < 1e-3 * got.abs().sum().item()  # a conservative force: no net energy change


def test_lsc_kick_scales_with_charge_length_and_inverse_energy_squared_and_the_pipe_form_factor():
    base = lsc_delta_kick_fast(TAU, CHARGE_C / N, X, Y, ENERGY, MC2, LENGTH_M, pipe_radius_b_m=PIPE_RADIUS_M)
    twice = lsc_delta_kick_fast(TAU, 2 * CHARGE_C / N, X, Y, ENERGY, MC2, 3 * LENGTH_M, pipe_radius_b_m=PIPE_RADIUS_M)
    torch.testing.assert_close(twice, 6 * base, rtol=1e-12, atol=0)
    stiff = lsc_delta_kick_fast(TAU, CHARGE_C / N, X, Y, 2 * ENERGY, MC2, LENGTH_M, pipe_radius_b_m=PIPE_RADIUS_M)
    torch.testing.assert_close(stiff, base / 8, rtol=1e-12, atol=0)  # kick = -V/E and V ~ 1/gamma^2
    a_m = 1.7 * 0.5 * (X.std() + Y.std()).item()
    bare = lsc_delta_kick_fast(TAU, CHARGE_C / N, X, Y, ENERGY, MC2, LENGTH_M)  # no pipe: form factor one
    torch.testing.assert_close(base, bare * (1 + 2 * math.log(PIPE_RADIUS_M / a_m)), rtol=1e-12, atol=0)


@pytest.mark.parametrize("kick", [lsc_delta_kick, lsc_delta_kick_fast])
def test_lsc_kick_vanishes_for_a_single_particle_or_no_length(kick):
    assert not kick(TAU[:1], CHARGE_C, X[:1], Y[:1], ENERGY, MC2, LENGTH_M).any()
    assert not kick(TAU, CHARGE_C / N, X, Y, ENERGY, MC2, 0.0).any()


def mean_loss_V_per_m(kick, length):
    return float(-kick.mean()) * ENERGY_EV / length


def steady_loss_V_per_m():
    return K_STEADY * CHARGE_C / (4 * math.pi * EPS0_F_PER_M) * RHO_M ** (-2 / 3) * SIGMA_M ** (-4 / 3)


@pytest.mark.parametrize("kick,tolerance", [
    (csr_steady_state_delta_kick, 0.07), (csr_steady_state_delta_kick_subquadratic, 0.07), (csr_steady_state_delta_kick_fast, 0.01),
])
def test_steady_state_csr_mean_loss_is_the_gaussian_closed_form(kick, tolerance):
    got = kick(TAU, CHARGE_C / N, RHO_M, ENERGY, 0.1)
    assert mean_loss_V_per_m(got, 0.1) == pytest.approx(steady_loss_V_per_m(), rel=tolerance)


def test_block_multipole_csr_sum_agrees_with_the_exact_pairwise_sum():
    exact = csr_steady_state_delta_kick(TAU, CHARGE_C / N, RHO_M, ENERGY, 0.1)
    approximate = csr_steady_state_delta_kick_subquadratic(TAU, CHARGE_C / N, RHO_M, ENERGY, 0.1, tolerance=1e-3)
    assert ((approximate - exact).abs().max() / exact.abs().max()).item() < 5e-3


def test_csr_kick_is_linear_in_charge_and_length_and_zero_without_a_bend_or_charge():
    base = csr_steady_state_delta_kick(TAU, CHARGE_C / N, RHO_M, ENERGY, 0.1)
    torch.testing.assert_close(csr_steady_state_delta_kick(TAU, 2 * CHARGE_C / N, RHO_M, ENERGY, 0.3), 6 * base, rtol=1e-10, atol=0)
    assert not csr_steady_state_delta_kick(TAU, CHARGE_C / N, 0.0, ENERGY, 0.1).any()  # straight line: rho = 0
    assert not csr_steady_state_delta_kick(TAU, 0.0, RHO_M, ENERGY, 0.1).any()
    assert not csr_steady_state_delta_kick_subquadratic(TAU, 0.0, RHO_M, ENERGY, 0.1).any()
