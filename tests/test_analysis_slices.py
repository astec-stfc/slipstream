"""Tests for ``analysis.slices`` -- see
plans/DONE/E1_SLICE_ANALYSIS_PLAN.md's own Validation section for the
checks this file follows: closed-form Gaussian-bunch quantities, chirp
recovery, a directional slice-emittance check, a compile-equivalence
check, and empty-slice NaN handling.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from slipstream.analysis.slices import (
    bunching_factor,
    chirp_fit,
    correlated_emittance,
    slice_indices,
    slice_stats,
)
from slipstream.constants import C_LIGHT


def test_slice_stats_matches_closed_form_gaussian_bunch():
    torch.manual_seed(0)
    N = 400_000
    sigma_tau = 2e-3  # m
    sigma_delta = 1e-3
    eps_x0 = 1e-9  # m*rad, matched (uncorrelated x/xp)
    beta_x = 2.0
    sigma_x = math.sqrt(eps_x0 * beta_x)
    sigma_xp = math.sqrt(eps_x0 / beta_x)
    Q = 50e-12  # C total bunch charge

    tau = torch.randn(N, dtype=torch.float64) * sigma_tau
    delta = torch.randn(N, dtype=torch.float64) * sigma_delta
    x = torch.randn(N, dtype=torch.float64) * sigma_x
    xp = torch.randn(N, dtype=torch.float64) * sigma_xp
    y = torch.zeros(N, dtype=torch.float64)
    yp = torch.zeros(N, dtype=torch.float64)
    q_per_particle = Q / N

    n_slices = 21
    edges = torch.linspace(-4 * sigma_tau, 4 * sigma_tau, n_slices + 1, dtype=torch.float64)
    idx, edges, counts = slice_indices(tau, slice_edges_m=edges)
    stats = slice_stats(tau, delta, x, xp, y, yp, q_per_particle, idx, edges)

    mid = n_slices // 2
    tau_mid = 0.5 * (edges[mid] + edges[mid + 1])
    expected_peak_I = Q * C_LIGHT / (math.sqrt(2 * math.pi) * sigma_tau) * math.exp(-float(tau_mid) ** 2 / (2 * sigma_tau ** 2))
    assert stats["current_A"][mid].item() == pytest.approx(expected_peak_I, rel=0.1)

    occupied = stats["n_particles"] > 500
    eps_vals = stats["eps_x_m"][occupied].numpy()
    assert np.all(np.isfinite(eps_vals))
    np.testing.assert_allclose(eps_vals, eps_x0, rtol=0.15)

    np.testing.assert_allclose(stats["sigma_delta"][occupied].numpy(), sigma_delta, rtol=0.1)

    k = 1.0 / sigma_tau
    b = bunching_factor(tau, k)
    expected_b = math.exp(-0.5 * (k * sigma_tau) ** 2)
    assert b.item() == pytest.approx(expected_b, rel=0.05)

    # vector-k form agrees with the scalar form, elementwise.
    b_vec = bunching_factor(tau, torch.tensor([k, 2 * k], dtype=torch.float64))
    assert b_vec[0].item() == pytest.approx(b.item(), rel=1e-12)


def test_chirp_fit_recovers_seeded_slope_and_correlated_emittance_removes_it():
    torch.manual_seed(1)
    N = 200_000
    sigma_tau = 1e-3
    sigma_delta_uncorr = 2e-4
    slope_true = 5.0  # 1/m

    tau = torch.randn(N, dtype=torch.float64) * sigma_tau
    delta = slope_true * tau + torch.randn(N, dtype=torch.float64) * sigma_delta_uncorr

    slope, intercept, r = chirp_fit(tau, delta)
    assert slope.item() == pytest.approx(slope_true, rel=0.02)
    assert intercept.item() == pytest.approx(0.0, abs=5e-6)
    assert abs(r.item()) > 0.99  # dominated by the chirp

    out = correlated_emittance(tau, delta)
    assert out["sigma_delta_correlated_removed"].item() == pytest.approx(sigma_delta_uncorr, rel=0.02)
    assert out["sigma_delta_projected"].item() > 5 * out["sigma_delta_correlated_removed"].item()


def test_slice_emittance_rises_with_slice_dependent_x_delta_correlation():
    """A dispersion-like x-delta correlation that grows along the bunch
    (tau) inflates the *projected* per-slice x-xp emittance in later
    slices relative to earlier ones -- a real, well-known effect
    (measuring emittance without removing dispersion inflates it), and
    exactly the directional/physical check the plan calls for."""
    torch.manual_seed(2)
    N = 300_000
    sigma_tau = 2e-3
    tau = torch.randn(N, dtype=torch.float64) * sigma_tau
    sigma_delta = 1e-3
    delta = torch.randn(N, dtype=torch.float64) * sigma_delta
    eps_beta = 1e-9
    beta_x = 1.0
    x_beta = torch.randn(N, dtype=torch.float64) * math.sqrt(eps_beta * beta_x)
    xp = torch.randn(N, dtype=torch.float64) * math.sqrt(eps_beta / beta_x)
    # Anchor D at the slicing window's own lower edge (not the sample min,
    # which sits in a near-empty tail) so the occupied bins actually span a
    # wide D range with plenty of counts in every bin.
    anchor = -3.0 * sigma_tau
    dispersion_per_m = 50.0  # m of x-shift per unit delta, per m of tau
    D = dispersion_per_m * (tau - anchor)
    x = x_beta + D * delta
    y = torch.zeros(N, dtype=torch.float64)
    yp = torch.zeros(N, dtype=torch.float64)

    edges = torch.linspace(anchor, 3.0 * sigma_tau, 12, dtype=torch.float64)
    idx, edges, counts = slice_indices(tau, slice_edges_m=edges)
    stats = slice_stats(tau, delta, x, xp, y, yp, 1.0, idx, edges)
    occupied = stats["n_particles"] > 1000
    eps_vals = stats["eps_x_m"][occupied].numpy()
    assert eps_vals[-1] > 3.0 * eps_vals[0]


def test_slice_stats_compiles_without_changing_values():
    torch.manual_seed(3)
    N = 5000
    n_slices = 10
    tau0 = torch.randn(N, dtype=torch.float64) * 1e-3
    delta = torch.randn(N, dtype=torch.float64) * 1e-4
    x = torch.randn(N, dtype=torch.float64) * 1e-4
    xp = torch.randn(N, dtype=torch.float64) * 1e-5
    y = torch.zeros(N, dtype=torch.float64)
    yp = torch.zeros(N, dtype=torch.float64)
    edges = torch.linspace(-5e-3, 5e-3, n_slices + 1, dtype=torch.float64)

    compiled = torch.compile(slice_stats)
    for shift in (0.0, 2e-4):
        tau = tau0 + shift
        idx, edges_i, _ = slice_indices(tau, slice_edges_m=edges)
        eager = slice_stats(tau, delta, x, xp, y, yp, 1e-12, idx, edges_i)
        got = compiled(tau, delta, x, xp, y, yp, 1e-12, idx, edges_i)
        for key in eager:
            np.testing.assert_allclose(
                torch.nan_to_num(got[key]).numpy(), torch.nan_to_num(eager[key]).numpy(), rtol=1e-9, atol=1e-12,
            )


def test_slice_stats_accepts_plain_float_energy_and_mc2():
    """A real regression: energy_eV/mc2_eV as plain Python floats (not 0-d
    tensors) must work too -- caught via BeamlineStudy.track() passing
    float(energy_out)/self.mc2_eV straight through."""
    torch.manual_seed(5)
    tau = torch.randn(1000, dtype=torch.float64) * 1e-3
    delta = torch.randn(1000, dtype=torch.float64) * 1e-4
    x = torch.randn(1000, dtype=torch.float64) * 1e-4
    xp = torch.randn(1000, dtype=torch.float64) * 1e-5
    y = torch.zeros(1000, dtype=torch.float64)
    yp = torch.zeros(1000, dtype=torch.float64)
    idx, edges, _ = slice_indices(tau, n_slices=5)
    stats = slice_stats(tau, delta, x, xp, y, yp, 1e-12, idx, edges, energy_eV=100e6, mc2_eV=0.511e6)
    assert "eps_x_n_m" in stats
    assert torch.all(torch.isfinite(stats["eps_x_n_m"][stats["n_particles"] > 0]))


def test_empty_slice_is_nan_not_zero_or_garbage():
    N = 1000
    torch.manual_seed(4)
    tau = torch.rand(N, dtype=torch.float64) * 1e-3  # confined to [0, 1e-3)
    delta = torch.randn(N, dtype=torch.float64) * 1e-4
    x = torch.randn(N, dtype=torch.float64) * 1e-5
    xp = torch.randn(N, dtype=torch.float64) * 1e-6
    y = torch.zeros(N, dtype=torch.float64)
    yp = torch.zeros(N, dtype=torch.float64)
    edges = torch.linspace(-1e-2, 1e-2, 21, dtype=torch.float64)  # far wider than the data

    idx, edges, counts = slice_indices(tau, slice_edges_m=edges)
    stats = slice_stats(tau, delta, x, xp, y, yp, 1e-12, idx, edges)
    empty_bins = counts == 0
    assert bool(empty_bins.any())
    assert torch.all(torch.isnan(stats["mean_delta"][empty_bins]))
    assert torch.all(torch.isnan(stats["eps_x_m"][empty_bins]))
    assert torch.all(stats["charge_C"][empty_bins] == 0)
    assert torch.all(stats["current_A"][empty_bins] == 0)


def test_slice_indices_out_of_range_particles_go_to_trash_bin():
    tau = torch.tensor([0.0, 0.5, 1.0, 5.0, -5.0], dtype=torch.float64)
    edges = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)  # 4 bins over [0, 1]
    idx, edges, counts = slice_indices(tau, slice_edges_m=edges)
    n_bins = edges.shape[0] - 1
    assert idx[3].item() == n_bins  # 5.0 out of range
    assert idx[4].item() == n_bins  # -5.0 out of range
    assert int(counts.sum().item()) == 3  # only the 3 in-range particles counted


def test_slice_indices_rejects_ambiguous_binning_args():
    tau = torch.randn(10, dtype=torch.float64)
    with pytest.raises(ValueError):
        slice_indices(tau)
    with pytest.raises(ValueError):
        slice_indices(tau, n_slices=5, ds_m=1e-3)

