"""On-axis field-map readers + paraxial off-axis expansion for the native
gun tracker. Pure PyTorch, no external hybrid backend (which has no field-map/RF/solenoid
element of this kind). Real ASTRA field files
are plain 2-column ``z[m] value[normalized]`` ASCII.

Formulas (both leading-order, ``O(r^2)`` truncation -- adequate near the
axis, not for a beam filling the aperture):

- **Solenoid** (static, on-axis ``Bz0(z)``): ``Br(r,z) = -(r/2) dBz0/dz``,
  ``Bz(r,z) = Bz0(z) - (r^2/4) d2Bz0/dz2 + O(r^4)`` -- eq. (15)-(16) of
  Kirk T. McDonald, "Expansion of an Axially Symmetric, Static Magnetic
  Field in Terms of Its Axial Field" (Princeton, 2011/2015,
  https://kirkmcd.princeton.edu/examples/axial.pdf), from ``div B=0``/
  ``curl B=0`` alone.
- **RF cavity** (on-axis ``Ez0(z,t) = Ez0(z) cos(omega t + phi)``, an
  axisymmetric TM-like near-axis field with no ``E_phi``/``B_r``/``B_z``):
  ``Er(r,z,t) = -(r/2) dEz0/dz * cos(omega t + phi)``,
  ``Bphi(r,z,t) = -(r omega / (2 c^2)) Ez0(z) * sin(omega t + phi)`` --
  derived directly from Maxwell's equations (same even/odd-in-``r``
  power-series ansatz as the McDonald solenoid derivation), cross-checked
  against ASTRA's own manual (Appendix I, K. Floettmann, DESY, v3.0 2011):
  ASTRA's ``sin(omega t)``/``cos(omega t)`` convention matches this
  module's ``cos(omega t + phase_rad)`` once ``phase_rad = phase_ASTRA -
  pi/2``. ASTRA's ``MaxE``/``Nue`` units are MV/m and GHz respectively.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import CubicSpline

from ...constants import C_LIGHT


def _scale_to_peak(profile: torch.Tensor, target: float) -> float:
    """Scale factor so ``profile * scale`` has value exactly ``target`` at
    the profile's own largest-magnitude sample -- **sign preserved**, not
    ``target / profile.abs().max()`` (which always yields a positive scale
    regardless of whether that extremum is the profile's positive or
    negative lobe)."""
    idx = int(profile.abs().argmax())
    return target / float(profile[idx])


def _default_device_dtype(
    device: str | torch.device | None, dtype: torch.dtype | None
) -> tuple[torch.device, torch.dtype]:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if dtype is None:
        dtype = torch.float32 if device.type == "cuda" else torch.float64
    return device, dtype


def read_astra_field_file(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a plain 2-column ASTRA field file (``z[m] value[normalized]``,
    e.g. ``HRRG_1D_RF.astra``, ``HRRG_combined_sols_100mm_onaxis.astra``).

    Multi-cell travelling-wave-structure files (e.g. ``TWS_S-DL.astra``)
    have a different, header-prefixed format -- use ``read_astra_tws_file``
    for those instead; this raises a clear error rather than silently
    misparsing the header line as a data row.
    """
    try:
        z, f = np.loadtxt(path, unpack=True)
    except ValueError as exc:
        raise ValueError(
            f"{path}: not a plain 2-column on-axis field file (got a ragged/"
            "header-prefixed layout -- multi-cell travelling-wave structure "
            "files like TWS_S-DL.astra need read_astra_tws_file instead)"
        ) from exc
    return z, f


def read_astra_tws_file(
    path: str | Path,
) -> tuple[float, float, int, int, np.ndarray, np.ndarray]:
    """Read an ASTRA traveling-wave-structure field file (name starts with
    ``TWS``, e.g. ``TWS_S-DL.astra``) -- manual (v3.2, ch. 4 "Traveling
    wave structures") sec.: a free-format header line ``z1 z2 n m``
    precedes the usual 2-column ``z, on-axis field`` table. ``z1``/``z2``
    are the start/end of the periodic traveling-wave cells (excluding the
    input/output coupler cells, which extend past them and are "treated
    as standing wave cells"); ``n``/``m`` are integers giving the phase
    advance per cell ``theta = 2*pi*n/m`` (``n<0`` for a backward wave).
    This function only decodes the file -- see ``TravelingWaveField`` for
    the field construction it feeds."""
    with open(path) as fh:
        header = fh.readline().split()
    z1, z2, n, m = float(header[0]), float(header[1]), int(float(header[2])), int(float(header[3]))
    z, f = np.loadtxt(path, skiprows=1, unpack=True)
    return z1, z2, n, m, z, f


class OnAxisFieldMap:
    """A 1D on-axis field profile ``f(z)``, resampled onto a uniform grid
    together with its first and second derivatives (via a cubic spline
    fit, evaluated once at construction) for fast GPU linear-interpolation
    lookup at runtime -- avoids a host<->device round trip every tracking
    step, which repeatedly evaluating a CPU spline per step would cost."""

    def __init__(
        self,
        z: np.ndarray,
        f: np.ndarray,
        n_samples: int = 4001,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        device, dtype = _default_device_dtype(device, dtype)
        # Some real ASTRA field files are assembled from several independently-computed
        # sub-segments and carry an exact-duplicate z at each seam. 
        # CubicSpline requires strictly increasing x; keep the
        # first sample at each repeated z rather than erroring.
        z, first_idx = np.unique(z, return_index=True)
        f = f[first_idx]
        spline = CubicSpline(z, f)
        z_uniform = np.linspace(z[0], z[-1], n_samples)
        self.z0 = float(z_uniform[0])
        self.dz = float(z_uniform[1] - z_uniform[0])
        self.n = n_samples
        self.device = device
        self.dtype = dtype
        self.f = torch.as_tensor(spline(z_uniform), device=device, dtype=dtype)
        self.df = torch.as_tensor(spline(z_uniform, 1), device=device, dtype=dtype)
        self.d2f = torch.as_tensor(spline(z_uniform, 2), device=device, dtype=dtype)

    @classmethod
    def from_astra_file(cls, path: str | Path, **kwargs) -> "OnAxisFieldMap":
        z, f = read_astra_field_file(path)
        return cls(z, f, **kwargs)

    def __call__(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate ``(f, df/dz, d2f/dz2)`` at particle z-positions (any
        shape), zero outside the map's z-range."""
        idx_float = (z - self.z0) / self.dz
        idx0 = torch.clamp(idx_float.floor().long(), 0, self.n - 2)
        frac = (idx_float - idx0).clamp(0.0, 1.0)
        in_range = (z >= self.z0) & (z <= self.z0 + self.dz * (self.n - 1))

        def lerp(table: torch.Tensor) -> torch.Tensor:
            v = table[idx0] * (1 - frac) + table[idx0 + 1] * frac
            return torch.where(in_range, v, torch.zeros_like(v))

        return lerp(self.f), lerp(self.df), lerp(self.d2f)


class SolenoidField:
    """Static axisymmetric solenoid field from an on-axis ``Bz(z)``
    profile, scaled so its peak matches ``b_max_T`` (ASTRA's own ``MaxB``
    convention -- the loaded profile is normalized, this rescales it to a
    physical peak field in tesla). ``z_offset`` is the solenoid's position
    along the beamline (ASTRA's ``S_pos``)."""

    def __init__(
        self,
        z: np.ndarray,
        bz_profile: np.ndarray,
        b_max_T: float,
        z_offset: float = 0.0,
        **kwargs,
    ):
        self.map = OnAxisFieldMap(z, bz_profile, **kwargs)
        self.scale = _scale_to_peak(self.map.f, b_max_T)
        self.z_offset = z_offset

    @classmethod
    def from_astra_file(cls, path: str | Path, b_max_T: float, **kwargs) -> "SolenoidField":
        z, f = read_astra_field_file(path)
        return cls(z, f, b_max_T, **kwargs)

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bz0, dbz0, d2bz0 = self.map(z - self.z_offset)
        bz0, dbz0, d2bz0 = bz0 * self.scale, dbz0 * self.scale, d2bz0 * self.scale
        r2 = x * x + y * y
        br_over_r = -0.5 * dbz0
        Bx = br_over_r * x
        By = br_over_r * y
        Bz = bz0 - 0.25 * r2 * d2bz0
        return Bx, By, Bz


class RFCavityField:
    """Axisymmetric RF cavity field from an on-axis ``Ez(z)`` profile,
    scaled so its peak matches ``e_max_V_per_m`` (ASTRA's own ``MaxE``,
    already converted to SI V/m by the caller -- this class takes no
    stance on ASTRA's MV/m file convention). ``frequency_hz``/``phase_rad``
    give the time dependence ``cos(omega t + phase_rad)``; ``z_offset`` is
    the cavity's position along the beamline (ASTRA's ``C_pos``).

    ASTRA's manual (Appendix I) gives its own on-axis field as
    ``Ez0(z)*sin(omega*t + Phi_ASTRA)`` -- the same physical convention as
    this class's ``cos(omega*t + phase_rad)``, offset by which instant is
    called ``t=0``: pass ``phase_rad = Phi_ASTRA - pi/2`` for a direct
    match. This class does not replicate ASTRA's ``Auto_phase`` layer
    (relative to the cavity's own scanned crest, not the absolute wave
    phase) -- that's ``Injector.auto_phase``'s job, not a field-formula
    concern.
    """

    def __init__(
        self,
        z: np.ndarray,
        ez_profile: np.ndarray,
        e_max_V_per_m: float,
        frequency_hz: float,
        phase_rad: float = 0.0,
        z_offset: float = 0.0,
        **kwargs,
    ):
        self.map = OnAxisFieldMap(z, ez_profile, **kwargs)
        self.scale = _scale_to_peak(self.map.f, e_max_V_per_m)
        self.omega = 2.0 * math.pi * frequency_hz
        self.phase_rad = phase_rad
        self.z_offset = z_offset

    @classmethod
    def from_astra_file(
        cls, path: str | Path, e_max_V_per_m: float, frequency_hz: float, **kwargs
    ) -> "RFCavityField":
        z, f = read_astra_field_file(path)
        return cls(z, f, e_max_V_per_m, frequency_hz, **kwargs)

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, t: torch.Tensor
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        ez0, dez0, _ = self.map(z - self.z_offset)
        ez0, dez0 = ez0 * self.scale, dez0 * self.scale
        phase = self.omega * t + self.phase_rad
        cos_p, sin_p = torch.cos(phase), torch.sin(phase)

        er_over_r = -0.5 * dez0 * cos_p
        Ex = er_over_r * x
        Ey = er_over_r * y
        Ez = ez0 * cos_p

        bphi_over_r = -(self.omega / (2.0 * C_LIGHT**2)) * ez0 * sin_p
        Bx = -bphi_over_r * y
        By = bphi_over_r * x
        Bz = torch.zeros_like(Ez)
        return Ex, Ey, Ez, Bx, By, Bz


class TravelingWaveField:
    """ASTRA-style multi-cell **traveling-wave** structure. 

    **The file stores a signed field snapshot, not an amplitude
    envelope** -- ASTRA's manual, Appendix II ("Representation of a
    travelling wave by two standing waves"): the field is carried by a
    real and imaginary part, ``Ez(z,t) = Er(z) cos(wt) - Ei(z) sin(wt)``,
    which is exactly what this class evaluates.

    **Getting ``Ei`` from the one stored table**: Floquet's theorem for a
    periodic structure with phase advance ``theta = 2*pi*n/m`` per cell of
    length ``L`` gives ``E(z+L) = E(z) * exp(-i*theta)``. Taking real
    parts, ``Er(z+L) = Er(z) cos(theta) + Ei(z) sin(theta)``, i.e.

        ``Ei(z) = [Er(z+L) - Er(z)*cos(theta)] / sin(theta)``

    -- an exact, closed-form reconstruction from the table evaluated one
    cell downstream, no second data column, no fitting. Built once at
    construction.

    **Geometry** (from the file's own ``z1 z2 n m`` header plus
    ``n_cells`` = ASTRA's ``C_numb``): the table holds an input coupler
    cell over ``[0, z1)``, one RF period of periodic traveling-wave cells
    over ``[z1, z2]`` (``m/|n|`` cells of length ``L=(z2-z1)*|n|/m``), and
    an output coupler cell over ``[z2, z_table_end]``. The real structure
    repeats the periodic section to ``n_cells`` cells, so this class
    **tiles** ``[z1, z2]`` and brackets it with the two coupler cells,
    total length ``z1 + n_cells*L + (z_table_end - z2)``. One stored
    period's own Floquet factor is ``exp(-i*2*pi*n)=1``, so both ``Er``
    and ``Ei`` tile with no phase bookkeeping.

    The coupler cells are "treated as standing wave cells" (manual):
    ``Ei=0`` there, continuous with the periodic section since ``Ei`` is
    ~0 at both its ends by design.

    **Tile-join continuity is enforced exactly, in both value and
    derivative — see ``_enforce_periodicity``.** The raw table's two ends
    only approximately agree; left uncorrected, ``dEz/dz`` (which sets the
    paraxial transverse focusing) jumps at every tile join. Floquet's
    theorem assumes exact periodicity to begin with, so
    ``_enforce_periodicity`` refits ``Er`` with a periodic-boundary-
    condition spline and re-derives ``Ei`` from that, patching both tables
    in the periodic region after construction.

    **No free wave number.** The phase velocity is fixed by the file's own
    geometry (``omega*L/theta``); ``self.k`` is exposed for diagnostics
    only, not used in the field evaluation.

    **Off-axis expansion**: the same leading-order paraxial result
    ``Er_transverse = -(r/2) dEz/dz``, ``Bphi = (r/2c^2) dEz/dt`` this
    module derives for ``RFCavityField`` -- depends only on the on-axis
    ``Ez(z,t)``, whatever waveform produced it. ``tests/test_native_fields.py``
    checks a structure with ``Ei=0`` everywhere reproduces ``RFCavityField``
    exactly."""

    def __init__(
        self,
        z: np.ndarray,
        ez_profile: np.ndarray,
        z1: float,
        z2: float,
        n: int,
        m: int,
        e_max_V_per_m: float,
        frequency_hz: float,
        n_cells: int,
        phase_rad: float = 0.0,
        z_offset: float = 0.0,
        **kwargs,
    ):
        self.omega = 2.0 * math.pi * frequency_hz
        self.phase_rad = phase_rad
        self.z_offset = z_offset
        self.z1, self.z2 = float(z1), float(z2)
        self.n, self.m = n, m
        self.n_cells = n_cells
        self.period = self.z2 - self.z1
        self.cell_length = self.period * abs(n) / m
        self.theta = 2.0 * math.pi * n / m
        self.n_tiles = round(n_cells * abs(n) / m)
        self.tw_length = n_cells * self.cell_length
        # Total structure length: input coupler + tiled cells + output coupler.
        self.length = self.tw_length + self.z1 + (float(z[-1]) - self.z2)
        self.k = self.theta / self.cell_length  # diagnostics only (see docstring)

        self.map = OnAxisFieldMap(z, ez_profile, **kwargs)
        self.map_imag = OnAxisFieldMap(z, self._imaginary_part(z, ez_profile), **kwargs)
        self.scale = _scale_to_peak(self.map.f, e_max_V_per_m)
        self._enforce_periodicity(z, ez_profile)

    def _enforce_periodicity(self, z: np.ndarray, er: np.ndarray) -> None:
        """Patch ``self.map``/``self.map_imag`` inside ``[z1, z2]`` so a tile
        join is exactly continuous in both value and derivative -- see
        class docstring. Refits ``Er`` with a periodic-boundary-condition
        cubic spline (``bc_type="periodic"``, matching value + first/second
        derivative at both ends), first snapping the raw endpoint values
        together (``scipy`` requires them to agree exactly). ``Ei`` is
        re-derived from this periodic ``Er`` via the same Floquet formula
        ``_imaginary_part`` uses, inheriting the same periodicity. Coupler
        cells (``Ei=0``, raw ``Er``) are untouched -- not periodic, don't
        need to be."""
        in_period = (z >= self.z1) & (z <= self.z2)
        z_p, er_p = z[in_period].copy(), er[in_period].copy()
        er_p[0] = er_p[-1] = 0.5 * (er_p[0] + er_p[-1])
        per_spline = CubicSpline(z_p, er_p, bc_type="periodic")

        zg = self.map.z0 + self.map.dz * np.arange(self.map.n)
        # Widen by one resampled cell on each side: the runtime fold always
        # lands strictly inside [z1, z2] (_envelope_z clamps it there), but a
        # lerp query near z1 or z2 can still pair a patched grid point with
        # its immediate (otherwise unpatched) neighbour one cell outside --
        # patch that neighbour too (a well-defined, continuous cubic
        # extrapolation one cell past the fit interval) so every lerp bin
        # the fold can actually select is fully periodic-consistent.
        mask = (zg >= self.z1 - self.map.dz) & (zg <= self.z2 + self.map.dz)
        er_fix = per_spline(zg[mask])
        der_fix = per_spline(zg[mask], 1)
        shifted = self.z1 + np.mod(zg[mask] + self.cell_length - self.z1, self.period)
        ei_fix = (per_spline(shifted) - er_fix * math.cos(self.theta)) / math.sin(self.theta)
        dei_fix = (per_spline(shifted, 1) - der_fix * math.cos(self.theta)) / math.sin(self.theta)

        mask_t = torch.as_tensor(mask, device=self.map.device)
        self.map.f[mask_t] = torch.as_tensor(er_fix, device=self.map.device, dtype=self.map.dtype)
        self.map.df[mask_t] = torch.as_tensor(der_fix, device=self.map.device, dtype=self.map.dtype)
        self.map_imag.f[mask_t] = torch.as_tensor(ei_fix, device=self.map.device, dtype=self.map.dtype)
        self.map_imag.df[mask_t] = torch.as_tensor(dei_fix, device=self.map.device, dtype=self.map.dtype)

    def _imaginary_part(self, z: np.ndarray, er: np.ndarray) -> np.ndarray:
        """``Ei(z) = [Er(z+L) - Er(z) cos(theta)] / sin(theta)`` inside the
        periodic section, zero in the standing-wave coupler cells (see the
        class docstring for the Floquet derivation). ``Er(z+L)`` is taken
        with the periodic continuation, since ``z+L`` runs past ``z2`` for
        the last cell of the stored period."""
        spline = CubicSpline(z, er)
        in_tw = (z >= self.z1) & (z <= self.z2)
        shifted = self.z1 + np.mod(z[in_tw] + self.cell_length - self.z1, self.period)
        ei = np.zeros_like(er)
        ei[in_tw] = (spline(shifted) - er[in_tw] * math.cos(self.theta)) / math.sin(self.theta)
        return ei

    @classmethod
    def from_astra_file(
        cls,
        path: str | Path,
        e_max_V_per_m: float,
        frequency_hz: float,
        n_cells: int,
        **kwargs,
    ) -> "TravelingWaveField":
        z1, z2, n, m, z, f = read_astra_tws_file(path)
        return cls(z, f, z1, z2, n, m, e_max_V_per_m, frequency_hz, n_cells, **kwargs)

    def _envelope_z(self, z_local: torch.Tensor) -> torch.Tensor:
        """Map a structure-local ``z`` onto the field table's own
        coordinate, by folding the periodic section back onto its single
        stored period. The tile index is clamped to ``[0, n_tiles-1]`` so
        the input coupler (``z_local < z1``) maps to itself and the output
        coupler lands past ``z2`` in the table -- both continuous with the
        periodic section they adjoin."""
        tile = torch.clamp(
            torch.floor((z_local - self.z1) / self.period), min=0.0, max=self.n_tiles - 1.0
        )
        # This subtracts two quantities of order the structure length to
        # leave one of order the period, so in float32 the result inherits
        # ~1% of a field-table cell of rounding -- measured not to matter
        # (the compensated Boris push, not this, was the real float32
        # accuracy driver). Reinstate an exact fmod form if a longer
        # structure or coarser table ever makes that 1% matter.
        return z_local - tile * self.period

    def on_axis(
        self, z: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(Ez, dEz/dz, dEz/dt)`` on the axis -- the three quantities the
        paraxial expansion in ``__call__`` is built from, exposed
        separately so an on-axis-only caller (a phase scan, a fused
        tracking kernel) can skip the transverse algebra. The real and
        imaginary tables share one uniform grid, so index/interpolation
        weights are computed once here rather than per
        ``OnAxisFieldMap.__call__``."""
        z_local = z - self.z_offset
        z_table = self._envelope_z(z_local)
        m, mi = self.map, self.map_imag
        idx = (z_table - m.z0) / m.dz
        i0 = torch.clamp(idx.floor().long(), 0, m.n - 2)
        w1 = (idx - i0).clamp(0.0, 1.0)
        w0 = 1.0 - w1
        i1 = i0 + 1
        # ``_envelope_z``'s clamped fold maps [0, length] onto the table's
        # own z-range by construction, so this physical mask subsumes the
        # per-map range mask it replaces.
        inside = (z_local >= 0.0) & (z_local <= self.length)
        s = inside.to(m.f.dtype) * self.scale
        er = (m.f[i0] * w0 + m.f[i1] * w1) * s
        der = (m.df[i0] * w0 + m.df[i1] * w1) * s
        ei = (mi.f[i0] * w0 + mi.f[i1] * w1) * s
        dei = (mi.df[i0] * w0 + mi.df[i1] * w1) * s

        phase = self.omega * t + self.phase_rad
        cos_p, sin_p = torch.cos(phase), torch.sin(phase)
        Ez = er * cos_p - ei * sin_p
        dEz_dz = der * cos_p - dei * sin_p
        dEz_dt = -self.omega * (er * sin_p + ei * cos_p)
        return Ez, dEz_dz, dEz_dt

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, t: torch.Tensor
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        Ez, dEz_dz, dEz_dt = self.on_axis(z, t)
        er_over_r = -0.5 * dEz_dz
        Ex = er_over_r * x
        Ey = er_over_r * y
        bphi_over_r = dEz_dt / (2.0 * C_LIGHT**2)
        Bx = -bphi_over_r * y
        By = bphi_over_r * x
        Bz = torch.zeros_like(Ez)
        return Ex, Ey, Ez, Bx, By, Bz

