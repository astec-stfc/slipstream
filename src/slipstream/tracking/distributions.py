"""Generates Twiss-matched Gaussian, KV surface and waterbag volume distributions, plus chirped
longitudinal profiles. See docs/modules/tracking/distributions.md."""
from __future__ import annotations

import torch


def _generator(device: torch.device, seed: int | None) -> torch.Generator | None:
    return torch.Generator(device=device).manual_seed(seed) if seed is not None else None


def gaussian_matched_6d(
    n: int,
    beta_x_m: float, alpha_x: float, epsx_m: float,
    beta_y_m: float, alpha_y: float, epsy_m: float,
    sigma_tau_m: float, sigma_delta: float,
    D_x_m: float = 0.0, Dp_x: float = 0.0,
    D_y_m: float = 0.0, Dp_y: float = 0.0,
    chirp_per_m: float = 0.0,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float64,
    seed: int | None = None,
) -> torch.Tensor:
    """6D Gaussian matched to explicit per-plane Twiss (``beta``/``alpha``/ ``eps``) + dispersion, plus
    an independent longitudinal ``sigma_tau_m``/``sigma_delta`` (+ optional linear energy chirp
    ``delta += chirp_per_m*tau``)."""
    device = torch.device(device or "cpu")
    gen = _generator(device, seed)

    def plane(beta: float, alpha: float, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
        gamma = (1.0 + alpha * alpha) / beta
        sigma = torch.tensor([[eps * beta, -eps * alpha], [-eps * alpha, eps * gamma]], dtype=dtype, device=device)
        L = torch.linalg.cholesky(sigma)
        z = torch.randn(2, n, generator=gen, dtype=dtype, device=device)
        uv = L @ z
        return uv[0], uv[1]

    tau = sigma_tau_m * torch.randn(n, generator=gen, dtype=dtype, device=device)
    delta = chirp_per_m * tau + sigma_delta * torch.randn(n, generator=gen, dtype=dtype, device=device)
    x, xp = plane(beta_x_m, alpha_x, epsx_m)
    y, yp = plane(beta_y_m, alpha_y, epsy_m)
    x = x + D_x_m * delta
    xp = xp + Dp_x * delta
    y = y + D_y_m * delta
    yp = yp + Dp_y * delta
    return torch.stack([x, xp, y, yp, tau, delta], dim=-1)


def gaussian_matched_optics(
    n: int, optics, sigma_tau_m: float, sigma_delta: float, chirp_per_m: float = 0.0,
    device: str | torch.device | None = None, dtype: torch.dtype = torch.float64, seed: int | None = None,
) -> torch.Tensor:
    """``gaussian_matched_6d`` fed directly from a ``lattice.optics.
    TransverseOptics`` instance (this package's own ring-side Twiss type
    -- already carries ``beta``/``alpha``/``D``/``eps`` per plane)."""
    return gaussian_matched_6d(
        n, optics.beta_x_m, optics.alpha_x, optics.epsx_m,
        optics.beta_y_m, optics.alpha_y, optics.epsy_m,
        sigma_tau_m, sigma_delta,
        D_x_m=optics.D_x_m, Dp_x=optics.Dp_x_m, D_y_m=optics.D_y_m, Dp_y=optics.Dp_y_m,
        chirp_per_m=chirp_per_m, device=device, dtype=dtype, seed=seed,
    )


def kv_4d(
    n: int,
    beta_x_m: float, alpha_x: float, epsx_m: float,
    beta_y_m: float, alpha_y: float, epsy_m: float,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float64,
    seed: int | None = None,
) -> torch.Tensor:
    """4D Kapchinskij-Vladimirskij distribution: uniform on the *surface* of the real 4D phase-space
    ellipsoid (every particle at the same total normalized amplitude) -- ASTRA's
    ``DISTRIBUTION=2``-comparable."""
    device = torch.device(device or "cpu")
    gen = _generator(device, seed)
    g = torch.randn(4, n, generator=gen, dtype=dtype, device=device)
    u = g / g.norm(dim=0, keepdim=True)

    def plane(beta: float, alpha: float, eps: float, n1: torch.Tensor, n2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r = (4.0 * eps) ** 0.5
        X1, P1 = r * n1, r * n2
        x = X1 * beta ** 0.5
        xp = (P1 - alpha * X1) / beta ** 0.5
        return x, xp

    x, xp = plane(beta_x_m, alpha_x, epsx_m, u[0], u[1])
    y, yp = plane(beta_y_m, alpha_y, epsy_m, u[2], u[3])
    return torch.stack([x, xp, y, yp], dim=-1)


def waterbag_6d(
    n: int,
    beta_x_m: float, alpha_x: float, epsx_m: float,
    beta_y_m: float, alpha_y: float, epsy_m: float,
    sigma_tau_m: float, sigma_delta: float,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float64,
    seed: int | None = None,
) -> torch.Tensor:
    """Compute waterbag 6d; see the module documentation for details."""
    device = torch.device(device or "cpu")
    gen = _generator(device, seed)
    g = torch.randn(4, n, generator=gen, dtype=dtype, device=device)
    unit = g / g.norm(dim=0, keepdim=True)
    radial = torch.rand(n, generator=gen, dtype=dtype, device=device).clamp(min=torch.finfo(dtype).tiny) ** 0.25
    u = unit * radial

    def plane(beta: float, alpha: float, eps: float, n1: torch.Tensor, n2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r = (4.0 * eps) ** 0.5
        X1, P1 = r * n1, r * n2
        x = X1 * beta ** 0.5
        xp = (P1 - alpha * X1) / beta ** 0.5
        return x, xp

    x, xp = plane(beta_x_m, alpha_x, epsx_m, u[0], u[1])
    y, yp = plane(beta_y_m, alpha_y, epsy_m, u[2], u[3])
    tau = sigma_tau_m * torch.randn(n, generator=gen, dtype=dtype, device=device)
    delta = sigma_delta * torch.randn(n, generator=gen, dtype=dtype, device=device)
    return torch.stack([x, xp, y, yp, tau, delta], dim=-1)


def chirped_longitudinal(
    n: int, sigma_tau_m: float, sigma_delta: float,
    chirp_per_m: float = 0.0, quad_chirp_per_m2: float = 0.0,
    profile: str = "gaussian", flattop_half_width_m: float | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float64,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute chirped longitudinal; see the module documentation for details."""
    device = torch.device(device or "cpu")
    gen = _generator(device, seed)
    if profile == "gaussian":
        tau = sigma_tau_m * torch.randn(n, generator=gen, dtype=dtype, device=device)
    elif profile == "flattop":
        half = flattop_half_width_m if flattop_half_width_m is not None else sigma_tau_m
        core = (torch.rand(n, generator=gen, dtype=dtype, device=device) * 2.0 - 1.0) * half
        edge_noise = 0.3 * sigma_tau_m * torch.randn(n, generator=gen, dtype=dtype, device=device)
        tau = core + edge_noise
    else:
        raise ValueError(f"unknown profile {profile!r}; use 'gaussian' or 'flattop'")
    delta = chirp_per_m * tau + quad_chirp_per_m2 * tau * tau + sigma_delta * torch.randn(n, generator=gen, dtype=dtype, device=device)
    return tau, delta
