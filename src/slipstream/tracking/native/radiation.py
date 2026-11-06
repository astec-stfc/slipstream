"""Collinear synchrotron loss and Gaussian diffusion in ultrarelativistic dipoles."""
from __future__ import annotations

import math

import torch

from ...constants import ELECTRON_MASS_EV, R_E_M

# CODATA reduced Compton wavelength; r_e * lambda_bar = r_e**2 / alpha.
_LAMBDA_BAR_M = 3.8615926796e-13


def bend_radiation_rates(energy_eV, curvature_m_inv):
    """On-momentum mean relative loss and quantum variance per metre."""
    gamma = energy_eV / ELECTRON_MASS_EV
    loss = (2 / 3) * R_E_M * gamma**3 * curvature_m_inv**2
    variance = (55 / (24 * math.sqrt(3))) * R_E_M * _LAMBDA_BAR_M * gamma**5 * abs(curvature_m_inv)**3
    return loss, variance


def bend_radiation_kick(
    state: torch.Tensor, energy_eV: torch.Tensor, curvature_m_inv: float,
    length_m: float, *, synch_rad: bool, isr: bool,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Positive-length midpoint kick at fixed reference; electron/positron only.

    Coefficients are the first two raw photon-spectrum moments. At fixed
    magnetic field both relative-momentum kicks scale as (1+delta)**2.
    """
    if length_m < 0:
        raise ValueError("Radiation length must be nonnegative")
    if length_m == 0 or curvature_m_inv == 0 or not (synch_rad or isr):
        return state
    loss_rate, variance_rate = bend_radiation_rates(energy_eV, curvature_m_inv)
    opd = 1 + state[..., 5]
    pz = torch.sqrt(opd**2 - state[..., 1]**2 - state[..., 3]**2)
    ds = length_m * (1 + curvature_m_inv * state[..., 0]) * opd / pz
    kick = torch.zeros_like(opd)
    if synch_rad:
        kick = kick - loss_rate * opd**2 * ds
    if isr:
        variance = variance_rate * ds
        kick = kick + opd**2 * torch.sqrt(variance) * torch.randn(
            opd.shape, dtype=state.dtype, device=state.device, generator=generator,
        )
    new_opd = opd + kick
    scale = new_opd / opd
    return torch.stack((state[..., 0], state[..., 1] * scale, state[..., 2],
                        state[..., 3] * scale, state[..., 4], new_opd - 1), dim=-1)
