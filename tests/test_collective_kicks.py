"""Longitudinal space-charge (LSC) and steady-state CSR energy kicks of a Gaussian bunch against their closed forms.

LSC with a pipe radius: ``E_z = Z0 c (1 + 2 ln(b/a)) / (4 pi gamma^2) * d(lambda)/d(tau)``, the on-axis field of a long
bunch in a round pipe, positive toward the head (decreasing ``tau``): the head gains energy and the tail loses it.
LSC without one: the free-space on-axis field of a uniform cylinder, integrated directly.
CSR: the steady-state mean loss ``K Q / (4 pi eps0) rho^(-2/3) sigma^(-4/3)``, ``K = 2^(8/3) Gamma(5/6) / (8 sqrt(pi) 3^(1/3))``.
The kernel-density versions smooth the line density with a Silverman bandwidth, so they agree to a few percent; the fixed-grid
versions to a fraction of a percent. The block-multipole CSR sum is an approximation of the exact pairwise one.
"""
import math

import numpy as np
import pytest
import torch
from scipy.integrate import quad
from scipy.special import ndtri

from slipstream.constants import C_LIGHT, EPS0_F_PER_M
from slipstream.tracking.native.collective import (
    csr_steady_state_delta_kick, csr_steady_state_delta_kick_fast, csr_steady_state_delta_kick_subquadratic, lsc_delta_kick,
    lsc_delta_kick_fast,
)
from slipstream.tracking.native.lattice_import import ElementSpec
from slipstream.tracking.native.linac import LinacTracker

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
    # kick = +E_z L / E with E_z = Z0 c F / (4 pi gamma^2) d(lambda)/d(tau) and d(lambda)/d(tau) = -tau lambda / sigma^2
    return -Z0_OHM * C_LIGHT * LENGTH_M * form / (4 * math.pi * gamma**2 * ENERGY_EV) * TAU / SIGMA_M**2 * line_density


@pytest.mark.parametrize("kick,tolerance", [(lsc_delta_kick, 0.06), (lsc_delta_kick_fast, 0.01)])
def test_lsc_kick_of_a_gaussian_bunch_is_the_derivative_of_its_line_density(kick, tolerance):
    got = kick(TAU, CHARGE_C / N, X, Y, ENERGY, MC2, LENGTH_M, pipe_radius_b_m=PIPE_RADIUS_M)
    expected = lsc_closed_form()
    core = TAU.abs() < 1.5 * SIGMA_M
    assert ((got - expected).abs()[core].max() / expected.abs().max()).item() < tolerance
    assert got[TAU < 0].min() > 0 > got[TAU > 0].max()  # the head (negative tau) gains energy, the tail loses it
    assert got.sum().abs().item() < 1e-3 * got.abs().sum().item()  # a conservative force: no net energy change


def test_lsc_kick_scales_with_charge_length_and_inverse_energy_squared_and_the_pipe_form_factor():
    base = lsc_delta_kick_fast(TAU, CHARGE_C / N, X, Y, ENERGY, MC2, LENGTH_M, pipe_radius_b_m=PIPE_RADIUS_M)
    twice = lsc_delta_kick_fast(TAU, 2 * CHARGE_C / N, X, Y, ENERGY, MC2, 3 * LENGTH_M, pipe_radius_b_m=PIPE_RADIUS_M)
    torch.testing.assert_close(twice, 6 * base, rtol=1e-12, atol=0)
    stiff = lsc_delta_kick_fast(TAU, CHARGE_C / N, X, Y, 2 * ENERGY, MC2, LENGTH_M, pipe_radius_b_m=PIPE_RADIUS_M)
    torch.testing.assert_close(stiff, base / 8, rtol=1e-12, atol=0)  # kick = E_z L / E and E_z ~ 1/gamma^2
    a_m = 1.7 * 0.5 * (X.std() + Y.std()).item()
    unit = lsc_delta_kick_fast(TAU, CHARGE_C / N, X, Y, ENERGY, MC2, LENGTH_M, pipe_radius_b_m=a_m)  # b = a: form factor one
    torch.testing.assert_close(base, unit * (1 + 2 * math.log(PIPE_RADIUS_M / a_m)), rtol=1e-12, atol=0)


def free_space_kick(tau_m, radius_m, sigma_m=SIGMA_M, energy_eV=ENERGY_EV):
    """``delta`` kick over ``LENGTH_M`` on the axis of a uniform cylinder of radius ``radius_m`` with a Gaussian line density, as
    a sum of charged-disk fields in the rest frame (no long-wavelength or impedance approximation, no grid)."""
    gamma = energy_eV / MC2_EV
    sigma = gamma * sigma_m

    def density(z):
        return CHARGE_C / (math.sqrt(2 * math.pi) * sigma) * math.exp(-0.5 * (z / sigma) ** 2)

    def at(tau):
        z = -gamma * tau  # toward the head

        def integrand(w):  # u = radius sinh(w) removes the near-cancellation of the disk integral
            return math.exp(-w) * (density(z - radius_m * math.sinh(w)) - density(z + radius_m * math.sinh(w)))

        top = math.asinh(20 * sigma / radius_m)
        value = quad(integrand, 0.0, top, limit=200, points=[math.asinh(max(abs(z), radius_m) / radius_m)])[0]
        return value / (2 * math.pi * EPS0_F_PER_M * radius_m) * LENGTH_M / energy_eV

    return np.array([at(tau) for tau in tau_m])


def against_free_space(tau, got, radius_m, sigma_m=SIGMA_M, energy_eV=ENERGY_EV):
    """Largest difference between a kick (per particle) and the Coulomb integral within 1.5 sigma, relative to its peak."""
    probes = np.array([-1.5, -1.0, -0.5, 0.5, 1.0, 1.5]) * sigma_m
    order = torch.argsort(tau)
    values = np.interp(probes, tau[order].numpy(), got[order].numpy())
    expected = free_space_kick(probes, radius_m, sigma_m, energy_eV)
    assert expected[0] > 0 > expected[-1]  # the reference itself: the head (negative tau) gains, the tail loses
    return np.abs(values - expected).max() / np.abs(expected).max()


def through_tracker(fast, csr, bend=False, **settings):
    """The LSC part of a ``LinacTracker`` pass through one drift, or an almost straight bend (``csr=True`` with the default
    kernels takes the joint exact LSC+CSR path)."""
    bunch = torch.zeros((N, 6), dtype=F64)
    bunch[:, 0], bunch[:, 2], bunch[:, 4] = X, Y, TAU
    kw = dict(energy_eV=ENERGY_EV, device="cpu", bunch_charge_C=CHARGE_C, fast_collective_effects=fast, always_change_p0=False)
    curved = csr or bend
    spec = ElementSpec("D", "bend" if curved else "drift", LENGTH_M, angle=1e-9 if curved else 0.0, csr=csr)
    on = LinacTracker([spec], lsc=True, csr=csr, **kw, **settings).track(bunch)[0][:, 5]
    return on - LinacTracker([spec], lsc=False, csr=csr, **kw, **settings).track(bunch)[0][:, 5]


# The Silverman-bandwidth paths smooth a 3000-particle bunch by 0.21 sigma, which lowers the kick by a few percent.
@pytest.mark.parametrize("kick,tolerance", [
    (lambda: lsc_delta_kick(TAU, CHARGE_C / N, X, Y, ENERGY, MC2, LENGTH_M), 0.06),
    (lambda: lsc_delta_kick_fast(TAU, CHARGE_C / N, X, Y, ENERGY, MC2, LENGTH_M), 0.01),
    (lambda: through_tracker(fast=False, csr=False), 0.06),
    (lambda: through_tracker(fast=False, csr=True), 0.06),
    (lambda: through_tracker(fast=True, csr=False), 0.01),
], ids=["pairwise", "fast", "tracker", "tracker-joint-csr", "tracker-fast"])
def test_lsc_kick_without_a_pipe_is_the_free_space_coulomb_field(kick, tolerance):
    radius = 1.7 * 0.5 * (X.std() + Y.std()).item()
    assert against_free_space(TAU, kick(), radius) < tolerance


@pytest.mark.parametrize("energy_eV,sigma_m,size_m", [(35e6, 300e-6, 500e-6), (190e6, 100e-6, 100e-6), (1e9, 20e-6, 50e-6)])
def test_free_space_lsc_follows_energy_length_and_radius(energy_eV, sigma_m, size_m):
    tau = gaussian_quantiles(sigma_m)
    x, y = gaussian_quantiles(size_m, seed=1), gaussian_quantiles(size_m, seed=2)
    got = lsc_delta_kick_fast(tau, CHARGE_C / N, x, y, torch.tensor(energy_eV, dtype=F64), MC2, LENGTH_M)
    radius = 1.7 * 0.5 * (x.std() + y.std()).item()
    assert against_free_space(tau, got, radius, sigma_m, energy_eV) < 0.01


@pytest.mark.parametrize("kick", [lsc_delta_kick, lsc_delta_kick_fast])
def test_free_space_lsc_is_linear_in_charge_and_length_and_exchanges_no_net_energy(kick):
    base = kick(TAU, CHARGE_C / N, X, Y, ENERGY, MC2, LENGTH_M)
    torch.testing.assert_close(kick(TAU, 2 * CHARGE_C / N, X, Y, ENERGY, MC2, 3 * LENGTH_M), 6 * base, rtol=1e-12, atol=0)
    assert base.sum().abs().item() < 1e-9 * base.abs().sum().item()  # the field of each pair of bins is equal and opposite


@pytest.mark.parametrize("kick", [lsc_delta_kick, lsc_delta_kick_fast])
def test_free_space_lsc_float32_agrees_with_float64(kick):
    double = kick(TAU, CHARGE_C / N, X, Y, ENERGY, MC2, LENGTH_M)
    single = kick(TAU.float(), CHARGE_C / N, X.float(), Y.float(), ENERGY.float(), MC2.float(), LENGTH_M)
    assert single.dtype == torch.float32
    assert ((single.double() - double).abs().max() / double.abs().max()).item() < 2e-4


def test_free_space_lsc_per_particle_charge_and_batches_match_single_bunches():
    single = lsc_delta_kick(TAU, CHARGE_C / N, X, Y, ENERGY, MC2, LENGTH_M)
    vector = lsc_delta_kick(TAU, torch.full((N,), CHARGE_C / N, dtype=F64), X, Y, ENERGY, MC2, LENGTH_M)
    torch.testing.assert_close(vector, single, rtol=1e-12, atol=0)
    batch = lsc_delta_kick(torch.stack([TAU, 2 * TAU]), CHARGE_C / N, torch.stack([X, X]), torch.stack([Y, Y]), ENERGY, MC2, LENGTH_M)
    torch.testing.assert_close(batch[0], single, rtol=1e-12, atol=0)
    torch.testing.assert_close(batch[1], lsc_delta_kick(2 * TAU, CHARGE_C / N, X, Y, ENERGY, MC2, LENGTH_M), rtol=1e-12, atol=0)


def test_silverman_smoothing_is_refused_with_a_pipe_radius():
    with pytest.raises(ValueError, match="pipe_radius_b_m"):
        lsc_delta_kick_fast(TAU, CHARGE_C / N, X, Y, ENERGY, MC2, LENGTH_M, pipe_radius_b_m=PIPE_RADIUS_M, smooth_passes=None)


@pytest.mark.parametrize("fast,csr,settings,kernel", [
    (True, False, {"lsc_bins": 256, "lsc_smooth_passes": 64}, {"n_bins": 256, "smooth_passes": 64}),
    (False, False, {"lsc_bins": 32}, {"n_bins": 32, "smooth_passes": None}),
    (False, True, {"lsc_smooth_passes": 32}, {"smooth_passes": 32}),
    (True, False, {"lsc_bins": 256, "lsc_smooth_passes": 64, "pipe_radius_b_m": PIPE_RADIUS_M},
     {"n_bins": 256, "smooth_passes": 64, "pipe_radius_b_m": PIPE_RADIUS_M}),
    (True, False, {"lsc_bins": 256, "lsc_smooth_passes": 64, "bend": True, "dipole_collective_subcycle": True},
     {"n_bins": 256, "smooth_passes": 64}),
], ids=["fast", "default", "default-joint-csr", "fast-pipe", "subcycled-dipole"])
def test_linac_tracker_lsc_grid_settings_reach_the_kernel(fast, csr, settings, kernel):
    got = through_tracker(fast, csr, **settings)
    expected = lsc_delta_kick_fast(TAU, CHARGE_C / N, X, Y, ENERGY, MC2, LENGTH_M, **kernel)
    scale = expected.abs().max().item()
    torch.testing.assert_close(got, expected, rtol=0, atol=1e-4 * scale)
    default = through_tracker(fast, csr, pipe_radius_b_m=settings.get("pipe_radius_b_m"))
    assert (got - default).abs().max().item() > 1e-3 * scale  # the settings change the result


@pytest.mark.parametrize("settings,match", [
    ({"lsc_bins": 1}, "lsc_bins"),
    ({"lsc_bins": 256.0}, "lsc_bins"),
    ({"lsc_smooth_passes": -1}, "lsc_smooth_passes"),
    ({"lsc_bins": 256, "pipe_radius_b_m": PIPE_RADIUS_M}, "fast_collective_effects"),
])
def test_linac_tracker_refuses_unusable_lsc_grid_settings(settings, match):
    with pytest.raises(ValueError, match=match):
        LinacTracker([ElementSpec("D", "drift", LENGTH_M)], ENERGY_EV, device="cpu", lsc=True, **settings)


def test_pairwise_pipe_formula_refuses_grid_settings():
    with pytest.raises(ValueError, match="no grid"):
        lsc_delta_kick(TAU, CHARGE_C / N, X, Y, ENERGY, MC2, LENGTH_M, pipe_radius_b_m=PIPE_RADIUS_M, n_bins=256)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_free_space_lsc_cuda_graph_survives_many_other_grid_settings():
    """A CUDA graph holds the addresses of the cached grid constants, so the caches must never evict them."""
    tau, x, y, energy, mc2 = (value.cuda() for value in (TAU, X, Y, ENERGY, MC2))

    def kick(bins, passes):
        return lsc_delta_kick_fast(tau, CHARGE_C / N, x, y, energy, mc2, LENGTH_M, n_bins=bins, smooth_passes=passes)

    first = kick(512, 32)
    for i in range(1, 25):
        kick(512 + 8 * i, 32 + i)
    # fill whatever was freed
    junk = [torch.full((size,), 1e30, dtype=F64, device="cuda") for size in (257, 513, 514, 1025, 1026) for _ in range(40)]
    torch.testing.assert_close(kick(512, 32), first, rtol=1e-12, atol=0)
    assert junk


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
