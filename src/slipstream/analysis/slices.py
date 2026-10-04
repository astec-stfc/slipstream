"""Bins length-like tau in metres and computes charge/current, widths and centered transverse
emittances per slice. See docs/modules/analysis/slices.md."""
from __future__ import annotations

import math

import torch

from ..constants import C_LIGHT
from .emittance import rms_emittance


def slice_indices(
    tau_m: torch.Tensor,
    n_slices: int | None = None,
    ds_m: float | None = None,
    slice_edges_m: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bin ``tau_m`` into longitudinal slices."""
    given = sum(v is not None for v in (n_slices, ds_m, slice_edges_m))
    if given != 1:
        raise ValueError("exactly one of n_slices/ds_m/slice_edges_m must be given")

    if slice_edges_m is not None:
        edges = slice_edges_m
    else:
        tau_min = float(tau_m.min())
        tau_max = float(tau_m.max())
        if n_slices is not None:
            edges = torch.linspace(tau_min, tau_max, n_slices + 1, dtype=tau_m.dtype, device=tau_m.device)
        else:
            span = tau_max - tau_min
            n = max(1, math.ceil(span / ds_m)) if span > 0 else 1
            edges = tau_min + ds_m * torch.arange(n + 1, dtype=tau_m.dtype, device=tau_m.device)

    n_bins = edges.shape[0] - 1
    idx = torch.bucketize(tau_m.contiguous(), edges[1:-1], right=False)
    out_of_range = (tau_m < edges[0]) | (tau_m > edges[-1])
    idx = torch.where(out_of_range, torch.full_like(idx, n_bins), idx)
    counts = torch.zeros(n_bins, dtype=torch.long, device=tau_m.device).scatter_add_(
        0, idx.clamp(max=max(n_bins - 1, 0)), (~out_of_range).long(),
    )
    return idx, edges, counts


def slice_stats(
    tau_m: torch.Tensor, delta: torch.Tensor,
    x_m: torch.Tensor, xp: torch.Tensor, y_m: torch.Tensor, yp: torch.Tensor,
    charge_per_particle_C: torch.Tensor | float,
    idx: torch.Tensor, edges: torch.Tensor,
    energy_eV: torch.Tensor | float | None = None,
    mc2_eV: torch.Tensor | float | None = None,
) -> dict[str, torch.Tensor]:
    """Compute slice stats; see the module documentation for details."""
    n_slices = edges.shape[0] - 1
    dtype, device = tau_m.dtype, tau_m.device
    valid = idx < n_slices
    idx_c = idx.clamp(max=max(n_slices - 1, 0))
    vmask = valid.to(dtype)

    def scatter_sum(v: torch.Tensor) -> torch.Tensor:
        return torch.zeros(n_slices, dtype=dtype, device=device).scatter_add_(0, idx_c, v * vmask)

    count = scatter_sum(torch.ones_like(tau_m))
    n_safe = count.clamp(min=1.0)
    empty = count == 0

    def wmean(v: torch.Tensor) -> torch.Tensor:
        return scatter_sum(v) / n_safe

    mean_tau, mean_delta = wmean(tau_m), wmean(delta)
    mean_x, mean_xp = wmean(x_m), wmean(xp)
    mean_y, mean_yp = wmean(y_m), wmean(yp)

    def wvar(v: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        d = v - m[idx_c]
        return scatter_sum(d * d) / n_safe

    def wcov(v: torch.Tensor, mv: torch.Tensor, w: torch.Tensor, mw: torch.Tensor) -> torch.Tensor:
        return scatter_sum((v - mv[idx_c]) * (w - mw[idx_c])) / n_safe

    var_delta = wvar(delta, mean_delta)
    var_x, var_xp, cov_x_xp = wvar(x_m, mean_x), wvar(xp, mean_xp), wcov(x_m, mean_x, xp, mean_xp)
    var_y, var_yp, cov_y_yp = wvar(y_m, mean_y), wvar(yp, mean_yp), wcov(y_m, mean_y, yp, mean_yp)
    cov_x_delta = wcov(x_m, mean_x, delta, mean_delta)
    cov_y_delta = wcov(y_m, mean_y, delta, mean_delta)

    eps_x = torch.sqrt(torch.clamp(var_x * var_xp - cov_x_xp ** 2, min=0.0))
    eps_y = torch.sqrt(torch.clamp(var_y * var_yp - cov_y_yp ** 2, min=0.0))
    sigma_delta = torch.sqrt(var_delta.clamp(min=0.0))

    tiny = torch.finfo(dtype).tiny
    corr_x_delta = cov_x_delta / torch.sqrt(torch.clamp(var_x * var_delta, min=0.0)).clamp(min=tiny)
    corr_y_delta = cov_y_delta / torch.sqrt(torch.clamp(var_y * var_delta, min=0.0)).clamp(min=tiny)

    q = charge_per_particle_C if isinstance(charge_per_particle_C, torch.Tensor) \
        else torch.tensor(float(charge_per_particle_C), dtype=dtype, device=device)
    charge_C = count * q
    ds_m = (edges[1:] - edges[:-1]).abs().clamp(min=tiny)
    current_A = charge_C.abs() * C_LIGHT / ds_m

    nan = torch.full((), float("nan"), dtype=dtype, device=device)

    def nan_if_empty(v: torch.Tensor) -> torch.Tensor:
        return torch.where(empty, nan, v)

    out = {
        "n_particles": count,
        "charge_C": charge_C,
        "current_A": torch.where(empty, torch.zeros_like(current_A), current_A),
        "mean_tau_m": nan_if_empty(mean_tau),
        "mean_delta": nan_if_empty(mean_delta),
        "sigma_delta": nan_if_empty(sigma_delta),
        "mean_x_m": nan_if_empty(mean_x),
        "mean_y_m": nan_if_empty(mean_y),
        "eps_x_m": nan_if_empty(eps_x),
        "eps_y_m": nan_if_empty(eps_y),
        "corr_x_delta": nan_if_empty(corr_x_delta),
        "corr_y_delta": nan_if_empty(corr_y_delta),
    }
    if energy_eV is not None and mc2_eV is not None:
        gamma = energy_eV / mc2_eV
        if isinstance(gamma, torch.Tensor):
            beta_gamma = torch.sqrt(torch.clamp(gamma * gamma - 1.0, min=0.0))
        else:
            beta_gamma = math.sqrt(max(gamma * gamma - 1.0, 0.0))
        out["eps_x_n_m"] = nan_if_empty(eps_x * beta_gamma)
        out["eps_y_n_m"] = nan_if_empty(eps_y * beta_gamma)
    return out


def bunching_factor(
    tau_m: torch.Tensor, k_per_m: torch.Tensor | float,
    charge_per_particle_C: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """``|<exp(i*k*tau)>|`` -- the bunching factor at wavenumber(s) ``k_per_m`` (rad/m)."""
    k = torch.as_tensor(k_per_m, dtype=tau_m.dtype, device=tau_m.device)
    scalar = k.dim() == 0
    k_flat = k.reshape(-1)
    phase = k_flat[:, None] * tau_m[None, :]
    if charge_per_particle_C is None:
        w = torch.ones_like(tau_m)
    else:
        w = torch.as_tensor(charge_per_particle_C, dtype=tau_m.dtype, device=tau_m.device).expand_as(tau_m)
    w_sum = w.sum().clamp(min=torch.finfo(tau_m.dtype).tiny)
    re = (w * torch.cos(phase)).sum(dim=-1) / w_sum
    im = (w * torch.sin(phase)).sum(dim=-1) / w_sum
    b = torch.sqrt(re * re + im * im)
    return b.squeeze(0) if scalar else b


def chirp_fit(tau_m: torch.Tensor, delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Linear fit ``delta ~= slope*tau_m + intercept`` via closed-form normal equations (fixed-shape,
    ``torch.compile``-safe)."""
    tau_mean, delta_mean = tau_m.mean(), delta.mean()
    dtau, ddelta = tau_m - tau_mean, delta - delta_mean
    var_tau = (dtau * dtau).mean()
    var_delta = (ddelta * ddelta).mean()
    cov = (dtau * ddelta).mean()
    tiny = torch.finfo(tau_m.dtype).tiny
    slope = cov / var_tau.clamp(min=tiny)
    intercept = delta_mean - slope * tau_mean
    r = cov / torch.sqrt(torch.clamp(var_tau * var_delta, min=0.0)).clamp(min=tiny)
    return slope, intercept, r


def correlated_emittance(
    tau_m: torch.Tensor, delta: torch.Tensor,
    energy_eV: torch.Tensor | float | None = None,
    mc2_eV: torch.Tensor | float | None = None,
) -> dict[str, torch.Tensor]:
    """Compute correlated emittance; see the module documentation for details."""
    slope, intercept, r = chirp_fit(tau_m, delta)
    delta_corr = delta - (slope * tau_m + intercept)
    out = {
        "sigma_tau_m": tau_m.std(),
        "sigma_delta_projected": delta.std(),
        "sigma_delta_correlated_removed": delta_corr.std(),
        "eps_z_projected_m": rms_emittance(tau_m, delta),
        "eps_z_correlated_removed_m": rms_emittance(tau_m, delta_corr),
        "chirp_slope_per_m": slope,
        "chirp_r": r,
    }
    if energy_eV is not None and mc2_eV is not None:
        from .emittance import longitudinal_emittance_keV_mm
        ekin_eV = delta * energy_eV
        ekin_corr_eV = delta_corr * energy_eV
        out["eps_z_projected_keV_mm"] = longitudinal_emittance_keV_mm(tau_m, ekin_eV)
        out["eps_z_correlated_removed_keV_mm"] = longitudinal_emittance_keV_mm(tau_m, ekin_corr_eV)
    return out


def bunch_summary(
    particles6: torch.Tensor, energy_eV: float, charge_C: float, n_slices: int = 60, window_sigma: float = 3.0,
    mc2_eV: float = 510998.95,
) -> dict[str, float]:
    """One-line-per-quantity longitudinal/transverse summary of a ``tracking.native`` bunch ``(N, 6)``
    at reference energy ``energy_eV`` -- what a compressor paper tabulates."""
    p = particles6.detach().to("cpu", torch.float64)
    x, px, y, py, tau, delta = (p[:, i] for i in range(6))
    bg = math.sqrt(max((energy_eV / mc2_eV) ** 2 - 1.0, 0.0))
    sig_tau = tau.std()
    lo, hi = tau.mean() - window_sigma * sig_tau, tau.mean() + window_sigma * sig_tau
    idx, edges, _ = slice_indices(tau, slice_edges_m=torch.linspace(float(lo), float(hi), n_slices + 1, dtype=torch.float64))
    st = slice_stats(tau, delta, x, px, y, py, charge_C / p.shape[0], idx, edges, energy_eV=energy_eV, mc2_eV=mc2_eV)
    cur = st["current_A"]
    core = (cur >= 0.5 * cur.max()) & (st["n_particles"] > 20)
    w = st["n_particles"][core]
    return {
        "sigma_t_fs": float(sig_tau / C_LIGHT * 1e15),
        "mean_t_fs": float(tau.mean() / C_LIGHT * 1e15),
        "sigma_delta": float(delta.std()),
        "mean_energy_MeV": float(energy_eV * (1.0 + delta.mean()) / 1e6),
        "eps_x_n_um": float(rms_emittance(x - x.mean(), px - px.mean()) * bg * 1e6),
        "eps_y_n_um": float(rms_emittance(y - y.mean(), py - py.mean()) * bg * 1e6),
        "I_peak_A": float(cur.max()),
        "slice_eps_x_n_um": float((st["eps_x_n_m"][core] * w).sum() / w.sum() * 1e6),
        "slice_sigma_delta": float((st["sigma_delta"][core] * w).sum() / w.sum()),
        "charge_fraction_in_window": float(st["n_particles"].sum() / p.shape[0]),
    }

