"""A documented, minimal HDF5 particle-dump schema -- the "modern format"
fallback named in plans/DONE/E2_DISTRIBUTION_IO_PLAN.md (a later
OpenPMD-beamphysics upgrade would replace this, not add a second one).

Schema: one dataset ``particles6`` (``(N, 6)``, float64), plus scalar
attrs ``charge_C`` (total bunch charge, signed), ``energy_eV`` (reference
energy), ``mc2_eV`` (reference rest mass), ``canonical`` (bool -- whether
columns are ``tracking.native``'s canonical ``(x, px, y, py, tau, delta)``
or the slope convention ``(x, xp, y, yp, tau, delta)``, see
``tracking.distributions``' own module docstring), ``tau_units``
(always ``"m"`` -- this schema never stores seconds-valued tau; convert a
pystable ``tau_s`` via ``*C_LIGHT`` before saving)."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch


def save_particles_h5(
    path: str | Path, particles6: torch.Tensor, charge_C: float,
    energy_eV: float, mc2_eV: float, canonical: bool = True,
) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset("particles6", data=particles6.detach().cpu().numpy().astype(np.float64))
        f.attrs["charge_C"] = charge_C
        f.attrs["energy_eV"] = energy_eV
        f.attrs["mc2_eV"] = mc2_eV
        f.attrs["canonical"] = canonical
        f.attrs["tau_units"] = "m"


def load_particles_h5(
    path: str | Path, device: str | torch.device | None = None, dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, dict]:
    """``(particles6, meta)`` -- ``meta`` has ``charge_C``/``energy_eV``/
    ``mc2_eV``/``canonical``/``tau_units``."""
    with h5py.File(path, "r") as f:
        particles6 = torch.as_tensor(f["particles6"][()], dtype=dtype, device=device)
        meta = {
            "charge_C": float(f.attrs["charge_C"]),
            "energy_eV": float(f.attrs["energy_eV"]),
            "mc2_eV": float(f.attrs["mc2_eV"]),
            "canonical": bool(f.attrs["canonical"]),
            "tau_units": str(f.attrs["tau_units"]),
        }
    return particles6, meta
