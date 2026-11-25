"""The ring's one-turn kicks: RF, radiation damping and quantum excitation, IBS-modified damping, and the longitudinal
parameters they are built from.

References are the exact moment recursions of the damping/excitation map, ``Var' = (1 - T0/tau)^2 Var + 2 (T0/tau) sigma0^2``
(whose fixed point is ``sigma0^2 / (1 - T0 / 2 tau)``), and the linearised synchrotron map against ``Lattice``.
"""
import math

import pytest
import torch

from slipstream.constants import C_LIGHT
from slipstream.ibs import IBSGrowthRates
from slipstream.lattice.lattice import Lattice
from slipstream.tracking.native.ring_physics import (
    damping_excitation, ibs_effective_taus, rf_kick, rms_emittance, rms_emittance_tensor, transverse_damping_excitation,
)

F64 = torch.float64
LATTICE = Lattice(circumference_m=240.0, E0_eV=3e9, alpha_c=1.7e-4, Vrf_V=2.5e6, f_rf_Hz=499.654e6, U0_eV=1.2e6,
                  tau_z_s=8e-3, sigma_delta0=1e-3)


def test_synchronous_phase_balances_the_radiation_loss_and_the_kick_vanishes_there():
    p = LATTICE
    assert p.Vrf_V * math.sin(p.phi_s) == pytest.approx(p.U0_eV, rel=1e-12)
    assert math.cos(p.phi_s) < 0  # above transition-free stability: the second-quadrant root
    assert rf_kick(torch.zeros(3, dtype=F64), p).abs().max().item() == 0.0
    tau = torch.tensor([-3e-11, 1e-11, 4e-11], dtype=F64)
    expected = p.Vrf_V / p.E0_eV * (torch.sin(p.phi_s + p.omega_rf * tau) - math.sin(p.phi_s))
    torch.testing.assert_close(rf_kick(tau, p), expected, rtol=1e-13, atol=0)
    period = 2 * math.pi / p.omega_rf
    torch.testing.assert_close(rf_kick(tau + period, p), rf_kick(tau, p), rtol=0, atol=1e-13)


def test_rf_kick_accepts_overrides_for_voltage_phase_frequency_and_energy():
    tau = torch.tensor([2e-11], dtype=F64)
    got = rf_kick(tau, LATTICE, Vrf_V=1e6, omega_rf=2 * math.pi * 1e9, phi_s=2.0, E0_eV=2e9)
    assert got.item() == pytest.approx(1e6 / 2e9 * (math.sin(2.0 + 2 * math.pi * 1e9 * 2e-11) - math.sin(2.0)), rel=1e-12)


def test_the_linearised_one_turn_map_oscillates_at_the_synchrotron_tune_of_the_lattice():
    p = LATTICE
    slope = p.Vrf_V * p.omega_rf * math.cos(p.phi_s) / p.E0_eV  # d(delta kick) / d(tau)
    slip = p.alpha_c * p.T0_s  # d(tau) / d(delta) over one turn
    assert -slope * slip > 0  # stable
    tune = math.sqrt(-slope * slip) / (2 * math.pi)  # small-amplitude: cos(2 pi Qs) = 1 + slope * slip / 2
    assert tune == pytest.approx(p.synchrotron_omega / p.omega0, rel=1e-12)
    assert p.natural_bunch_length_s == pytest.approx(p.alpha_c * p.sigma_delta0 / p.synchrotron_omega, rel=1e-12)
    assert p.harmonic_number == pytest.approx(p.f_rf_Hz * p.circumference_m / (p.beta_rel * C_LIGHT), rel=1e-12)


def test_damping_excitation_step_is_exact_for_given_noise_and_decays_without_it():
    delta = torch.linspace(-2e-3, 2e-3, 7, dtype=F64)
    a = LATTICE.T0_s / LATTICE.tau_z_s
    noise = torch.linspace(-1.0, 1.0, 7, dtype=F64)
    got = damping_excitation(delta, LATTICE, noise=noise)
    torch.testing.assert_close(got, (1 - a) * delta + math.sqrt(2 * a) * LATTICE.sigma_delta0 * noise, rtol=1e-14, atol=0)
    torch.testing.assert_close(damping_excitation(delta, LATTICE, noise=torch.zeros_like(delta)), (1 - a) * delta, rtol=1e-14, atol=0)


def test_damping_excitation_variance_follows_the_moment_recursion_and_its_fixed_point():
    n = 1_000_000
    generator = torch.Generator().manual_seed(7)
    a = LATTICE.T0_s / LATTICE.tau_z_s
    sigma0 = LATTICE.sigma_delta0
    start = 3e-3 * torch.randn(n, generator=generator, dtype=F64)  # far from equilibrium
    after = damping_excitation(start, LATTICE, generator=generator)
    assert after.var().item() == pytest.approx((1 - a) ** 2 * start.var().item() + 2 * a * sigma0**2, rel=0.01)
    stationary = sigma0 * torch.randn(n, generator=generator, dtype=F64) / math.sqrt(1 - a / 2)  # variance sigma0^2 / (1 - a/2)
    assert damping_excitation(stationary, LATTICE, generator=generator).var().item() == pytest.approx(stationary.var().item(), rel=0.01)
    assert stationary.var().item() == pytest.approx(sigma0**2, rel=0.01)  # a << 1: the natural energy spread


def test_effective_damping_time_changes_both_the_decay_and_the_noise():
    n = 400_000
    generator = torch.Generator().manual_seed(3)
    zero = torch.zeros(n, dtype=F64)
    tau_eff = torch.tensor(LATTICE.tau_z_s / 4, dtype=F64)
    sigma_eff = torch.tensor(2e-3, dtype=F64)
    kicked = damping_excitation(zero, LATTICE, generator=generator, tau_z_eff_s=tau_eff, sigma_delta0_eff=sigma_eff)
    a = LATTICE.T0_s / tau_eff.item()
    assert kicked.var().item() == pytest.approx(2 * a * sigma_eff.item() ** 2, rel=0.01)
    decoupled = damping_excitation(zero, LATTICE, generator=generator, tau_z_eff_s=tau_eff, sigma_delta0_eff=sigma_eff,
                                   tau_z_noise_s=torch.tensor(LATTICE.tau_z_s, dtype=F64))
    assert decoupled.var().item() == pytest.approx(2 * LATTICE.T0_s / LATTICE.tau_z_s * sigma_eff.item() ** 2, rel=0.01)


def test_transverse_damping_excitation_has_the_natural_divergence_as_its_fixed_point():
    n = 1_000_000
    generator = torch.Generator().manual_seed(11)
    eps, beta, alpha, tau = 2e-9, 8.0, -0.7, 6e-3
    a = LATTICE.T0_s / tau
    sigma_up = math.sqrt(eps * (1 + alpha**2) / beta)
    exact = torch.linspace(-1.0, 1.0, 5, dtype=F64)
    torch.testing.assert_close(transverse_damping_excitation(exact, tau, eps, beta, alpha, LATTICE.T0_s, noise=exact),
                               (1 - a) * exact + math.sqrt(2 * a) * sigma_up * exact, rtol=1e-14, atol=0)
    start = 5 * sigma_up * torch.randn(n, generator=generator, dtype=F64)
    after = transverse_damping_excitation(start, tau, eps, beta, alpha, LATTICE.T0_s, generator=generator)
    assert after.var().item() == pytest.approx((1 - a) ** 2 * start.var().item() + 2 * a * sigma_up**2, rel=0.01)
    stationary = sigma_up * torch.randn(n, generator=generator, dtype=F64) / math.sqrt(1 - a / 2)
    again = transverse_damping_excitation(stationary, tau, eps, beta, alpha, LATTICE.T0_s, generator=generator)
    assert again.var().item() == pytest.approx(stationary.var().item(), rel=0.01)
    eff = transverse_damping_excitation(torch.zeros(n, dtype=F64), tau, eps, beta, alpha, LATTICE.T0_s, generator=generator,
                                        tau_eff_s=torch.tensor(tau / 2, dtype=F64), eps_eff_m=torch.tensor(4 * eps, dtype=F64))
    assert eff.var().item() == pytest.approx(2 * (2 * LATTICE.T0_s / tau) * 4 * sigma_up**2, rel=0.01)  # sigma_up ~ sqrt(eps)


def test_ibs_growth_lengthens_the_damping_times_by_its_rates():
    tau_z, tau_x, tau_y = 8e-3, 12e-3, 15e-3
    rates = IBSGrowthRates(inv_Tp_s=40.0, inv_Tx_s=25.0, inv_Ty_s=5.0)
    got = ibs_effective_taus(tau_z, tau_x, tau_y, rates)
    assert got[0] == pytest.approx(1 / (1 / tau_z - 40.0)) and got[1] == pytest.approx(1 / (1 / tau_x - 2 * 25.0))
    assert got[2] == pytest.approx(1 / (1 / tau_y - 2 * 5.0))
    # growth twice the damping rate: the net damping time is minus the radiation damping time (the beam anti-damps)
    anti = ibs_effective_taus(tau_z, tau_x, tau_y, IBSGrowthRates(inv_Tp_s=2 / tau_z, inv_Tx_s=1 / tau_x, inv_Ty_s=1 / tau_y))
    assert anti == pytest.approx((-tau_z, -tau_x, -tau_y))


def test_rms_emittance_is_the_square_root_of_the_covariance_determinant():
    n = 50_000
    generator = torch.Generator().manual_seed(5)
    z = torch.randn(n, 2, generator=generator, dtype=F64)
    z = z - z.mean(0)
    z = z @ torch.linalg.inv(torch.linalg.cholesky(z.T @ z / n)).T  # sample covariance exactly the identity
    eps, beta, alpha = 3e-9, 4.0, 1.3
    u = 10 + math.sqrt(eps * beta) * z[:, 0]  # a mean offset must not matter
    up = -2 + math.sqrt(eps / beta) * (z[:, 1] - alpha * z[:, 0])
    assert rms_emittance(u, up) == pytest.approx(eps, rel=1e-9)
    assert rms_emittance_tensor(u, up).item() == pytest.approx(eps, rel=1e-9)
    assert rms_emittance(u, 5.0 * u) == pytest.approx(0.0, abs=1e-20)  # perfectly correlated: no area in phase space
