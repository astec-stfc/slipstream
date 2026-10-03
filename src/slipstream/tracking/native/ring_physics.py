"""Small ring-physics maps shared by native tracking and dev backends.

This module is intentionally independent of the historical pySTABLE tracker.
The native tracker can therefore be installed and used without importing any
development backend.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ...ibs import IBSGrowthRates
    from ...lattice.lattice import Lattice


def rf_kick(
    tau: torch.Tensor,
    params: "Lattice",
    Vrf_V: torch.Tensor | float | None = None,
    omega_rf: torch.Tensor | float | None = None,
    phi_s: torch.Tensor | float | None = None,
    E0_eV: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Return the fractional-energy kick from the main RF cavity."""
    Vrf_V = params.Vrf_V if Vrf_V is None else Vrf_V
    omega_rf = params.omega_rf if omega_rf is None else omega_rf
    phi_s = params.phi_s if phi_s is None else phi_s
    E0_eV = params.E0_eV if E0_eV is None else E0_eV
    phase = phi_s + omega_rf * tau
    sin_phi_s = torch.sin(phi_s) if torch.is_tensor(phi_s) else math.sin(phi_s)
    return (Vrf_V / E0_eV) * (torch.sin(phase) - sin_phi_s)


def damping_excitation(
    delta: torch.Tensor,
    params: "Lattice",
    generator: torch.Generator | None = None,
    tau_z_eff_s: torch.Tensor | None = None,
    sigma_delta0_eff: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
    tau_z_noise_s: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply synchrotron-radiation damping and quantum excitation."""
    tau_z = tau_z_eff_s if tau_z_eff_s is not None else params.tau_z_s
    T0_over_tau_z = params.T0_s / tau_z
    decay = 1.0 - T0_over_tau_z
    if sigma_delta0_eff is not None:
        T0_over_tau_noise = T0_over_tau_z if tau_z_noise_s is None else params.T0_s / tau_z_noise_s
        noise_scale = math.sqrt(2.0) * sigma_delta0_eff * torch.sqrt(T0_over_tau_noise)
    else:
        noise_scale = math.sqrt(2.0) * params.sigma_delta0 * math.sqrt(params.T0_s / params.tau_z_s)
    if noise is None:
        noise = torch.randn(delta.shape, generator=generator, dtype=delta.dtype, device=delta.device)
    return delta * decay + noise_scale * noise


def transverse_damping_excitation(
    up: torch.Tensor,
    tau_plane_s: float,
    eps_m: float,
    beta_m: float,
    alpha: float,
    T0_s: float,
    generator: torch.Generator | None = None,
    tau_eff_s: torch.Tensor | None = None,
    eps_eff_m: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
    tau_noise_s: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply transverse radiation damping and quantum excitation."""
    tau = tau_eff_s if tau_eff_s is not None else tau_plane_s
    T0_over_tau = T0_s / tau
    decay = 1.0 - T0_over_tau
    if eps_eff_m is not None:
        sigma_up = torch.sqrt(eps_eff_m * (1.0 + alpha**2) / beta_m)
        T0_over_tau_noise = T0_over_tau if tau_noise_s is None else T0_s / tau_noise_s
        noise_scale = math.sqrt(2.0) * sigma_up * torch.sqrt(T0_over_tau_noise)
    else:
        sigma_up = math.sqrt(eps_m * (1.0 + alpha**2) / beta_m)
        noise_scale = math.sqrt(2.0) * sigma_up * math.sqrt(T0_s / tau_plane_s)
    if noise is None:
        noise = torch.randn(up.shape, generator=generator, dtype=up.dtype, device=up.device)
    return up * decay + noise_scale * noise


def ibs_effective_taus(tau_z_sr, tau_x_sr, tau_y_sr, rates: "IBSGrowthRates"):
    """Fold IBS growth rates into the three effective damping times."""
    return (
        1.0 / (1.0 / tau_z_sr - rates.inv_Tp_s),
        1.0 / (1.0 / tau_x_sr - 2.0 * rates.inv_Tx_s),
        1.0 / (1.0 / tau_y_sr - 2.0 * rates.inv_Ty_s),
    )


def rms_emittance_tensor(u: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Return the live geometric RMS emittance as a scalar tensor."""
    du = u - u.mean()
    dup = up - up.mean()
    var_u = (du * du).mean()
    var_up = (dup * dup).mean()
    cov = (du * dup).mean()
    return torch.sqrt(torch.clamp(var_u * var_up - cov * cov, min=0.0))


def rms_emittance(u: torch.Tensor, up: torch.Tensor) -> float:
    """Return the live geometric RMS emittance as a Python float."""
    return rms_emittance_tensor(u, up).item()


__all__ = [
    "damping_excitation",
    "ibs_effective_taus",
    "rf_kick",
    "rms_emittance",
    "rms_emittance_tensor",
    "transverse_damping_excitation",
]
