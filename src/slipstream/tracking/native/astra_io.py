"""Converts ASTRA distribution rows from their reference-relative z/pz/time convention to absolute
particle values. See docs/modules/tracking/native/astra_io.md."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch

from ...constants import C_LIGHT, E_CHARGE_C


def read_astra_distribution(
    path: str | Path,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    """Read an ASTRA particle-distribution file, returning the *real* (status == -1) electron
    macroparticles only, with absolute ``z``/``pz``/``clock`` already added back in."""
    return distribution_from_rows(np.loadtxt(path), device, dtype)


def distribution_from_rows(
    raw: np.ndarray,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    status: tuple[int, ...] = (-1,),
) -> dict[str, torch.Tensor]:
    """``read_astra_distribution``'s row -> particle conversion, on an already-loaded ``(N, 10)``
    ASTRA-layout array (also what ``generator.Beam.particles`` feeds it)."""
    from .fields import _default_device_dtype

    device, dtype = _default_device_dtype(device, dtype)
    x0, y0, z0, px0, py0, pz0, clock0, q0, idx0, status0 = raw[0]

    idx = np.nonzero(np.isin(raw[:, 9], status))[0]
    rows = raw[idx]
    not_ref = (idx != 0)  # only add the reference baseline back for a row that isn't itself the reference

    x = rows[:, 0]
    y = rows[:, 1]
    z = rows[:, 2] + z0 * not_ref
    px = rows[:, 3]
    py = rows[:, 4]
    pz = rows[:, 5] + pz0 * not_ref
    clock = rows[:, 6] + clock0 * not_ref
    q_nC = rows[:, 7]
    species = rows[:, 8]

    q_C = np.where(species == 1, -np.abs(q_nC), np.abs(q_nC)) * 1e-9

    def t(a):
        return torch.as_tensor(a, device=device, dtype=dtype)

    return {
        "x": t(x), "y": t(y), "z": t(z),
        "px": t(px), "py": t(py), "pz": t(pz),
        "clock": t(clock), "q": t(q_C),
    }


def cathode_emission_status(raw: np.ndarray) -> tuple[int, ...]:
    """Compute cathode emission status; see the module documentation for details."""
    return (-1,) if bool(np.any(raw[1:, 9] == -1)) else (5,)


def momentum_eV_to_SI(p_eV_per_c: torch.Tensor) -> torch.Tensor:
    """``eV/c`` -> ``kg*m/s`` (SI momentum)."""
    return p_eV_per_c * E_CHARGE_C / C_LIGHT


def write_astra_distribution(
    path: str | Path,
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
    px: torch.Tensor, py: torch.Tensor, pz: torch.Tensor,
    clock_ns: torch.Tensor, q_C: torch.Tensor,
    species_index: int = 1,
    status_flag: int = 5,
) -> None:
    """The exact inverse of ``read_astra_distribution``: writes an ASTRA particle-distribution file,
    real (electron, by default) macroparticles only."""
    def n(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy().astype(np.float64)

    x_, y_, z_, px_, py_, pz_, clock_, q_ = (n(t) for t in (x, y, z, px, py, pz, clock_ns, q_C))
    z0, pz0, clock0 = z_.mean(), pz_.mean(), clock_.mean()
    n_particles = x_.shape[0]
    q_nC = -q_ * 1e9 if species_index == 1 else np.abs(q_) * 1e9
    row0_status = -3 if status_flag == -1 else status_flag

    rows = np.zeros((n_particles + 1, 10))
    rows[0] = [x_.mean(), y_.mean(), z0, px_.mean(), py_.mean(), pz0, clock0, q_nC.mean(), species_index, row0_status]
    rows[1:, 0] = x_
    rows[1:, 1] = y_
    rows[1:, 2] = z_ - z0
    rows[1:, 3] = px_
    rows[1:, 4] = py_
    rows[1:, 5] = pz_ - pz0
    rows[1:, 6] = clock_ - clock0
    rows[1:, 7] = q_nC
    rows[1:, 8] = species_index
    rows[1:, 9] = status_flag
    np.savetxt(path, rows, fmt=["%.9e"] * 8 + ["%d", "%d"])


def read_astra_emit(stem: str | Path, run: int = 1) -> dict[str, np.ndarray]:
    """Read ASTRA's ``<stem>.Xemit/.Yemit/.Zemit.<run:03d>`` output tables (plain 7-column ASCII,
    manual Table 4) into the named arrays the native trajectory also carries, so the two plot on the
    same keys."""
    stem = Path(stem)
    if stem.suffix == ".in":
        stem = stem.with_suffix("")
    zemit, xemit, yemit = (np.loadtxt(f"{stem}.{p}emit.{run:03d}") for p in "ZXY")
    return {
        "z_m": zemit[:, 0],
        "t_ns": zemit[:, 1],
        "Ekin_MeV": zemit[:, 2],
        "sigma_z_mm": zemit[:, 3],
        "dE_rms_keV": zemit[:, 4],
        "sigma_x_mm": xemit[:, 3],
        "sigma_y_mm": yemit[:, 3],
        "emit_x_mm_mrad": xemit[:, 5],
        "emit_y_mm_mrad": yemit[:, 5],
        "emit_z_kev_mm": zemit[:, 5],
    }


def astra_reference_at_z(astra: dict[str, np.ndarray], z_m: float) -> dict[str, float]:
    """The ``read_astra_emit`` row nearest ``z_m`` -- so an endpoint
    comparison follows wherever a run actually stopped."""
    i = int(np.argmin(np.abs(astra["z_m"] - z_m)))
    return {key: float(val[i]) for key, val in astra.items()}


# One `name = value`, `name(i) = value` or `name(i,j) = value` assignment.
_NML_GROUP = re.compile(r"&(\w+)(.*?)(?:^\s*/|\s*&END\b)", re.S | re.M | re.I)
_NML_ITEM = re.compile(
    r"(\w+)\s*(?:\(\s*(\d+(?:\s*,\s*\d+)*)\s*\))?\s*=\s*('[^']*'|\"[^\"]*\"|\(\s*[^()]*\)|[^,\s]+)")


def _nml_value(tok: str):
    if tok[0] in "'\"":
        return tok[1:-1]
    if tok[0] == "(":  # Fortran complex: (x, z) -> (float, float)
        return tuple(float(v.strip().lower().replace("d", "e")) for v in tok[1:-1].split(","))
    low = tok.lower().strip(".")
    if low in ("t", "true"):
        return True
    if low in ("f", "false"):
        return False
    for conv in (int, lambda s: float(s.lower().replace("d", "e"))):
        try:
            return conv(tok)
        except ValueError:
            pass
    return tok  # a bare word, e.g. `head = trial`, `File_Aperture(1) = Col_X`


def read_astra_namelist_groups(path: str | Path) -> list[tuple[str, dict]]:
    """``read_astra_namelist``'s parse, but one ``(group, {key: value})`` per namelist *in file
    order*, so a repeated group (the generator's ``Add=T`` stacks several ``&INPUT``) stays
    separate instead of being merged."""
    text = re.sub(r"!.*", "", Path(path).read_text())
    out: list[tuple[str, dict]] = []
    for group, body in _NML_GROUP.findall(text):
        grp: dict = {}
        for key, idx, tok in _NML_ITEM.findall(body):
            val = _nml_value(tok)
            if idx:
                parts = tuple(int(i) for i in idx.split(","))
                grp.setdefault(key.lower(), {})[parts[0] if len(parts) == 1 else parts] = val
            else:
                grp[key.lower()] = val
        out.append((group.lower(), grp))
    return out


def read_astra_namelist(path: str | Path) -> dict[str, dict]:
    """Parse an ASTRA input deck (Fortran namelists) into ``{group: {key: value}}``, all names
    lower-cased (ASTRA is case-insensitive: ``FILE_EFieLD``)."""
    out: dict[str, dict] = {}
    for group, grp in read_astra_namelist_groups(path):
        merged = out.setdefault(group, {})
        for key, val in grp.items():
            if isinstance(val, dict) and isinstance(merged.get(key), dict):
                merged[key].update(val)
            else:
                merged[key] = val
    return out
