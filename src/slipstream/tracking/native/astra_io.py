"""ASTRA file I/O for the native gun tracker. Three readers: the particle
distribution (below), the ``.Xemit``/``.Yemit``/``.Zemit`` output tables
(``read_astra_emit``), and the input deck's Fortran namelists
(``read_astra_namelist``) -- the last one is only a *parser*; turning a
deck into trackable elements is
``tracking.native.injector.Injector.from_astra``'s job.

Column layout and conventions per the real ASTRA manual (Floettmann, DESY,
``ASTRA A Space Charge Tracking Algorithm``, v3.0, 2011), table 1
(``Definition of the initial particle distribution``) and table 2 (status
flags):

- Columns: ``x, y, z, px, py, pz, clock, macro_charge, particle_index,
  status_flag``. Units: ``x, y, z`` in m; ``px, py, pz`` in eV/c;
  ``clock`` in ns; ``macro_charge`` in nC.
- **The first row is the reference particle, in absolute coordinates.
  Every other row's `z`, `pz`, `clock` are *relative* to that first
  row** ("Longitudinal particle coordinates, i.e. z, pz and t are given
  relative to the reference particle" -- manual sec. 2); `x, y, px, py`
  are already absolute for every row.
- Status flag `-1` = a standard particle at the cathode, not yet
  started -- the real macroparticles a cathode-emission distribution
  like ``laser.astra`` consists of. `-3` = a trajectory-probe particle
  at the cathode -- the 6 diagnostic-only rows ``generator``'s default
  ``Probe=True`` adds (positions at 0.5/1.0/1.5 sigma_x/sigma_z and
  sigma_y/sigma_z) -- excluded here, they carry no real bunch charge.
- The macro-charge column's *sign* is not authoritative (manual: "The
  sign of the charge specified in column 8 is not relevant" -- ASTRA
  infers sign from the particle-index species column instead); this
  reader applies the correct sign for electrons (index 1) explicitly
  rather than trusting the file's own sign bit.
"""
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
    """Read an ASTRA particle-distribution file, returning the *real*
    (status == -1) electron macroparticles only, with absolute
    ``z``/``pz``/``clock`` already added back in. Returns a dict of
    ``(N,)`` tensors: ``x, y, z`` (m), ``px, py, pz`` (eV/c, absolute),
    ``clock`` (ns, absolute -- each particle's emission time), ``q``
    (C, signed, one per macroparticle)."""
    return distribution_from_rows(np.loadtxt(path), device, dtype)


def distribution_from_rows(
    raw: np.ndarray,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    status: tuple[int, ...] = (-1,),
) -> dict[str, torch.Tensor]:
    """``read_astra_distribution``'s row -> particle conversion, on an already-loaded ``(N, 10)``
    ASTRA-layout array (also what ``generator.Beam.particles`` feeds it). ``status`` selects the
    rows kept: ``(-1,)`` (default) is a cathode-emission file's standard particles; ``generator``
    passes ``(-1, 5)`` so a beam that starts in flight (``Cathode=F``) keeps its particles too.

    Row 0 (the reference particle) is ``x, y, z, px, py, pz, clock`` **already absolute**, unlike
    every other row (relative to it, manual sec. 2) -- both real ASTRA's own ``generator.exe`` and
    this module's own ``generator.py`` clone deliberately reuse it as a genuine degenerate
    macroparticle too (sitting exactly on the reference trajectory, its own status set to the real
    -- not reference-only -- status code), so it must stay counted when its status matches. What
    must NOT happen is adding ``z0``/``pz0``/``clock0`` to it a *second* time (it is not relative
    to itself): confirmed directly on ISIS MEBT's real ``laser.astra`` (status ``5``, matching
    every other real particle there) -- without this, row 0 alone came out at exactly double its
    real momentum, a silently negligible one-in-2.6e5 outlier on CLARA's cathode files but a
    measurable one-in-2000 outlier here."""
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
    """``(-1,)`` if ``raw`` (row 0 excluded -- never itself a real particle, see
    ``distribution_from_rows``) has any real cathode-not-yet-emitted (status ``-1``) row, else
    ``(5,)`` -- a beam already in flight, ASTRA's own ``Cathode=F`` convention (real particle
    status ``5``, confirmed on ISIS MEBT's ``laser.astra``). Only for a genuine tracking *input*:
    an exit/screen-dump file also uses status ``5`` for its real particles (``write_astra_
    distribution``'s default), so this must not become ``read_astra_distribution``'s own default
    -- callers that specifically mean "the beamline's own starting distribution"
    (``Injector.track``) call this explicitly instead."""
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
    """The exact inverse of ``read_astra_distribution``: writes an ASTRA
    particle-distribution file, real (electron, by default) macroparticles
    only. Row 0 (the reference particle) is the mean of every column
    (ASTRA tolerates a synthetic reference row -- it need not be a real
    macroparticle); every other row's ``z``/``pz``/``clock`` are written
    *relative* to it, matching the reader's own "first row absolute, rest
    relative" convention exactly (manual sec. 2 -- see this module's
    docstring). ``status_flag`` defaults to ``5`` (ASTRA's "standard
    particle, active" code) rather than ``-1`` (cathode, not yet started)
    -- this writer is for a *tracked* distribution (an exit/screen dump),
    not a cathode-emission input; pass ``-1`` explicitly to write a
    cathode-ready file instead.

    Row 0's own status is ``status_flag`` unless that's ``-1``, in which
    case row 0 gets ``-3`` instead (ASTRA's own "trajectory-probe particle
    at the cathode, diagnostic only" code -- see ``read_astra_distribution``'s
    docstring) -- row 0 is a *synthetic mean*, not a genuine (n+1)th real
    particle, so it must not satisfy the reader's own ``status==-1`` real-
    particle mask, or reading the file back would silently add one spurious
    particle at the bunch centroid.

    ``q_C`` is signed (this module's own convention); written as ``-q_C``
    in nC for an electron (species 1, matching the reader's ``species==1``
    sign-flip) or plain ``|q_C|`` otherwise (the file's own charge-column
    sign is never authoritative per the ASTRA manual, so either round-trips
    through the reader)."""
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
    """Read ASTRA's ``<stem>.Xemit/.Yemit/.Zemit.<run:03d>`` output tables
    (plain 7-column ASCII, manual Table 4) into the named arrays the
    native trajectory also carries, so the two plot on the same keys.
    ``stem`` may be the input deck itself (``injector400.in``) -- ASTRA
    names its outputs after the deck, so the suffix is dropped.

    Columns: Xemit/Yemit are ``z[m], t[ns], avr[mm], rms[mm],
    slope_rms[mrad], eps_norm[pi mrad mm], corr[mrad]``; Zemit is ``z[m],
    t[ns], Ekin[MeV], zrms[mm], dE_rms[keV], eps_z_norm[pi keV mm],
    corr[keV]``. Note the emittance is column 6 in both (column 5 is
    ``slope_rms``/``dE_rms``), and every ASTRA emittance is the canonical
    RMS phase-space area -- see ``injector._longitudinal_emittance_keV_mm``
    for the longitudinal convention. Files are z-sorted."""
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


# One `name = value`, `name(i) = value` or `name(i,j) = value` assignment. Values never
# contain whitespace or commas unless quoted or a Fortran complex `(x, z)` (ASTRA's
# `D1(i) = (0.1, 2.5)` dipole corners), so a regex over the group body is a complete
# tokenizer for ASTRA's decks (it writes no array-valued lists).
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
    """Parse an ASTRA input deck (Fortran namelists) into
    ``{group: {key: value}}``, all names lower-cased (ASTRA is
    case-insensitive: ``FILE_EFieLD``). An indexed key ``Phi(2) = -16``
    becomes ``{"phi": {2: -16}}``. Logicals (``T``/``True``/``.T.``) map to
    bool, numbers to int/float (Fortran ``D`` exponents included), quoted
    and bare words to str, a Fortran complex ``(x, z)`` to a ``(float, float)``
    tuple. A two-index key ``D_Gap(1,3) = 0.05`` becomes ``{"d_gap": {(1, 3):
    0.05}}``. ``!`` comments are stripped. A repeated group is merged (use
    ``read_astra_namelist_groups`` to keep them apart)."""
    out: dict[str, dict] = {}
    for group, grp in read_astra_namelist_groups(path):
        merged = out.setdefault(group, {})
        for key, val in grp.items():
            if isinstance(val, dict) and isinstance(merged.get(key), dict):
                merged[key].update(val)
            else:
                merged[key] = val
    return out
