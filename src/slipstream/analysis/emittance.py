"""RMS-emittance helpers, factored out of ``tracking.native.injector``
(previously private there, ``_geometric_emittance_m``/
``_longitudinal_emittance_keV_mm``) so ``slices.py`` -- and any other
consumer -- doesn't duplicate the formula. ``injector.py`` re-exports
these under its old private names, unchanged behavior.
"""
from __future__ import annotations

import torch


def rms_emittance(u: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Whole-ensemble RMS geometric emittance: ``sqrt(<u^2><up^2> -
    <u*up>^2)``, via the unbiased (N-1) variance -- same convention as
    ``tracking.native.ring_physics.rms_emittance_tensor``."""
    return torch.sqrt(torch.clamp(u.var() * up.var() - (u * up).mean() ** 2, min=0.0))


def geometric_emittance_xy(
    xp: torch.Tensor, yp: torch.Tensor, x: torch.Tensor, y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(eps_x, eps_y)`` -- moved from ``injector._geometric_emittance_m``
    unchanged (note the historical argument order: slopes first)."""
    return rms_emittance(x, xp), rms_emittance(y, yp)


def longitudinal_emittance_keV_mm(z_or_tau_m: torch.Tensor, ekin_eV: torch.Tensor) -> torch.Tensor:
    """Normalized longitudinal emittance in ASTRA's Zemit convention
    (manual sec. 4.13): the real RMS phase-space area
    ``sqrt(<dz^2><dE^2> - <dz dE>^2)`` over centered ``(z, Ekin)`` -- the
    correlation term removes an RF-accelerated bunch's energy chirp.
    Moved from ``injector._longitudinal_emittance_keV_mm`` unchanged."""
    dz_mm = (z_or_tau_m - z_or_tau_m.mean()) * 1e3
    de_keV = (ekin_eV - ekin_eV.mean()) / 1e3
    return torch.sqrt(torch.clamp(dz_mm.var() * de_keV.var() - (dz_mm * de_keV).mean() ** 2, min=0.0))
