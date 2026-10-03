"""Tests for ``tracking.distributions`` -- see
plans/DONE/E2_DISTRIBUTION_IO_PLAN.md's Validation section: sigma-
matrix reproduction (statistical, 1/sqrt(N) tolerance), waterbag/KV
moments and the uniform-in-ellipse property, and same-seed determinism.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from slipstream.lattice.optics import TransverseOptics
from slipstream.tracking.distributions import (
    chirped_longitudinal,
    gaussian_matched_6d,
    gaussian_matched_optics,
    kv_4d,
    waterbag_6d,
)


def _sample_sigma_matrix(u: np.ndarray, up: np.ndarray) -> tuple[float, float, float]:
    """(beta, alpha, eps) recovered from a sample -- ``eps=sqrt(det(cov))``,
    ``beta=var(u)/eps``, ``alpha=-cov(u,up)/eps``."""
    cov = np.cov(u, up)
    eps = math.sqrt(max(cov[0, 0] * cov[1, 1] - cov[0, 1] ** 2, 0.0))
    return cov[0, 0] / eps, -cov[0, 1] / eps, eps


def test_gaussian_matched_6d_reproduces_input_twiss():
    torch.manual_seed(0)
    n = 200_000
    beta_x, alpha_x, epsx = 3.0, -0.7, 1.2e-9
    beta_y, alpha_y, epsy = 1.5, 0.3, 0.8e-9
    sigma_tau_m, sigma_delta = 1.5e-3, 4e-4

    p = gaussian_matched_6d(
        n, beta_x, alpha_x, epsx, beta_y, alpha_y, epsy, sigma_tau_m, sigma_delta, seed=1,
    )
    x, xp, y, yp, tau, delta = (p[:, i].numpy() for i in range(6))

    beta_x_s, alpha_x_s, eps_x_s = _sample_sigma_matrix(x, xp)
    beta_y_s, alpha_y_s, eps_y_s = _sample_sigma_matrix(y, yp)
    rtol = 5.0 / math.sqrt(n)  # 1/sqrt(N)-scale statistical tolerance, generous factor
    assert beta_x_s == pytest.approx(beta_x, rel=rtol)
    assert alpha_x_s == pytest.approx(alpha_x, abs=rtol)
    assert eps_x_s == pytest.approx(epsx, rel=rtol)
    assert beta_y_s == pytest.approx(beta_y, rel=rtol)
    assert alpha_y_s == pytest.approx(alpha_y, abs=rtol)
    assert eps_y_s == pytest.approx(epsy, rel=rtol)
    assert tau.std() == pytest.approx(sigma_tau_m, rel=rtol)
    assert delta.std() == pytest.approx(sigma_delta, rel=rtol)


def test_gaussian_matched_6d_dispersion_and_chirp():
    torch.manual_seed(0)
    n = 200_000
    D_x_m, sigma_delta = 0.5, 2e-4
    p = gaussian_matched_6d(
        n, 2.0, 0.0, 1e-10, 2.0, 0.0, 1e-10, 1e-3, sigma_delta,
        D_x_m=D_x_m, chirp_per_m=3.0, seed=2,
    )
    x, delta = p[:, 0], p[:, 5]
    # x picks up a D*delta shift -- correlation coefficient close to 1.
    r = torch.corrcoef(torch.stack([x, delta]))[0, 1].item()
    assert r > 0.9
    tau = p[:, 4]
    r_chirp = torch.corrcoef(torch.stack([tau, delta]))[0, 1].item()
    assert r_chirp > 0.9  # dominated by the seeded chirp


def test_gaussian_matched_optics_matches_direct_call():
    optics = TransverseOptics(
        beta_x_m=2.0, alpha_x=0.1, D_x_m=0.0, Q_x=0.2, Qp_x=0.0, tau_x_s=1.0, epsx_m=1e-9,
        beta_y_m=1.0, alpha_y=-0.2, D_y_m=0.0, Q_y=0.15, Qp_y=0.0, tau_y_s=1.0, epsy_m=1e-9,
    )
    p1 = gaussian_matched_optics(1000, optics, 1e-3, 1e-4, seed=7)
    p2 = gaussian_matched_6d(1000, 2.0, 0.1, 1e-9, 1.0, -0.2, 1e-9, 1e-3, 1e-4, seed=7)
    assert torch.allclose(p1, p2)


def test_kv_4d_uniform_on_ellipsoid_surface_and_right_emittance():
    torch.manual_seed(0)
    n = 100_000
    beta_x, alpha_x, epsx = 2.0, 0.0, 1e-9
    beta_y, alpha_y, epsy = 1.0, 0.0, 1e-9
    p = kv_4d(n, beta_x, alpha_x, epsx, beta_y, alpha_y, epsy, seed=3)
    x, xp, y, yp = (p[:, i].numpy() for i in range(4))

    # Normalized amplitude X1^2/eps_x + P1^2/eps_x + X2^2/eps_y + P2^2/eps_y
    # (with X1=x/sqrt(beta), P1=alpha*X1+xp*sqrt(beta)) is the same constant
    # (4.0) for every particle -- the defining KV property.
    X1 = x / math.sqrt(beta_x)
    P1 = alpha_x * X1 + xp * math.sqrt(beta_x)
    X2 = y / math.sqrt(beta_y)
    P2 = alpha_y * X2 + yp * math.sqrt(beta_y)
    amp = (X1 ** 2 + P1 ** 2) / epsx + (X2 ** 2 + P2 ** 2) / epsy
    np.testing.assert_allclose(amp, 4.0, rtol=1e-6)

    beta_x_s, alpha_x_s, eps_x_s = _sample_sigma_matrix(x, xp)
    assert eps_x_s == pytest.approx(epsx, rel=0.02)
    assert beta_x_s == pytest.approx(beta_x, rel=0.02)


def test_waterbag_6d_moments_and_bounded_amplitude():
    torch.manual_seed(0)
    n = 100_000
    beta_x, alpha_x, epsx = 2.0, 0.0, 1e-9
    beta_y, alpha_y, epsy = 1.0, 0.0, 1e-9
    sigma_tau_m, sigma_delta = 1e-3, 1e-4
    p = waterbag_6d(n, beta_x, alpha_x, epsx, beta_y, alpha_y, epsy, sigma_tau_m, sigma_delta, seed=4)
    x, xp, y, yp, tau, delta = (p[:, i].numpy() for i in range(6))

    X1 = x / math.sqrt(beta_x)
    P1 = alpha_x * X1 + xp * math.sqrt(beta_x)
    X2 = y / math.sqrt(beta_y)
    P2 = alpha_y * X2 + yp * math.sqrt(beta_y)
    amp = (X1 ** 2 + P1 ** 2) / epsx + (X2 ** 2 + P2 ** 2) / epsy
    assert amp.max() <= 4.0 + 1e-6  # never outside the ellipsoid (volume fill, not surface)
    assert amp.min() < 3.0  # actually fills the interior, not stuck at the surface

    # rms emittance of a uniformly-filled 4-ball is (2/3)*eps_x per plane --
    # E[radial**2] = 2/3 for the r=u**(1/4) volume-fill draw (verified
    # directly, not from a textbook lookup: this package's own eps_x
    # parameter is defined so kv_4d's r=1 (surface) case reproduces it
    # exactly, and waterbag_6d reuses that same construction).
    beta_x_s, alpha_x_s, eps_x_s = _sample_sigma_matrix(x, xp)
    assert eps_x_s == pytest.approx(epsx * 2.0 / 3.0, rel=0.05)
    assert tau.std() == pytest.approx(sigma_tau_m, rel=0.02)
    assert delta.std() == pytest.approx(sigma_delta, rel=0.02)


def test_chirped_longitudinal_gaussian_and_flattop():
    torch.manual_seed(0)
    n = 100_000
    tau, delta = chirped_longitudinal(n, 1e-3, 2e-4, chirp_per_m=10.0, seed=5)
    assert tau.std().item() == pytest.approx(1e-3, rel=0.02)
    r = torch.corrcoef(torch.stack([tau, delta]))[0, 1].item()
    assert r > 0.99

    tau_ft, _delta_ft = chirped_longitudinal(n, 1e-3, 1e-5, profile="flattop", flattop_half_width_m=2e-3, seed=6)
    # A flat-top core is closer to uniform than Gaussian: excess kurtosis
    # well below the Gaussian's 0 (a real, distinguishing signature).
    z = (tau_ft - tau_ft.mean()) / tau_ft.std()
    kurtosis = (z ** 4).mean().item() - 3.0
    assert kurtosis < -0.5

    with pytest.raises(ValueError):
        chirped_longitudinal(10, 1e-3, 1e-4, profile="bogus")


def test_seeded_generators_are_deterministic_on_cpu():
    p1 = gaussian_matched_6d(500, 2.0, 0.1, 1e-9, 1.0, -0.1, 1e-9, 1e-3, 1e-4, seed=42)
    p2 = gaussian_matched_6d(500, 2.0, 0.1, 1e-9, 1.0, -0.1, 1e-9, 1e-3, 1e-4, seed=42)
    assert torch.equal(p1, p2)

    k1 = kv_4d(500, 2.0, 0.0, 1e-9, 1.0, 0.0, 1e-9, seed=42)
    k2 = kv_4d(500, 2.0, 0.0, 1e-9, 1.0, 0.0, 1e-9, seed=42)
    assert torch.equal(k1, k2)

