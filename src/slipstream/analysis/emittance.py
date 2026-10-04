"""RMS helpers combine unbiased position/momentum variances with the mean cross product. See
docs/modules/analysis/emittance.md."""
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
    """Compute longitudinal emittance keV mm; see the module documentation for details."""
    dz_mm = (z_or_tau_m - z_or_tau_m.mean()) * 1e3
    de_keV = (ekin_eV - ekin_eV.mean()) / 1e3
    return torch.sqrt(torch.clamp(dz_mm.var() * de_keV.var() - (dz_mm * de_keV).mean() ** 2, min=0.0))
