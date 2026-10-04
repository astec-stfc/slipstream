"""Reads on-axis profiles and expands RF/solenoid fields near the axis. See
docs/modules/tracking/native/fields.md."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import CubicSpline

from ...constants import C_LIGHT


def _scale_to_peak(profile: torch.Tensor, target: float) -> float:
    """Compute scale to peak; see the module documentation for details."""
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
    """Compute read astra field file; see the module documentation for details."""
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
    """Compute read astra tws file; see the module documentation for details."""
    with open(path) as fh:
        header = fh.readline().split()
    z1, z2, n, m = float(header[0]), float(header[1]), int(float(header[2])), int(float(header[3]))
    z, f = np.loadtxt(path, skiprows=1, unpack=True)
    return z1, z2, n, m, z, f


class OnAxisFieldMap:
    """Describe OnAxisFieldMap; see the module documentation for details."""

    def __init__(
        self,
        z: np.ndarray,
        f: np.ndarray,
        n_samples: int = 4001,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        device, dtype = _default_device_dtype(device, dtype)
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
    """Describe SolenoidField; see the module documentation for details."""

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
    """Describe RFCavityField; see the module documentation for details."""

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
    """ASTRA-style multi-cell **traveling-wave** structure."""

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
        """Patch ``self.map``/``self.map_imag`` inside ``[z1, z2]`` so a tile join is exactly
        continuous in both value and derivative -- see class docstring."""
        in_period = (z >= self.z1) & (z <= self.z2)
        z_p, er_p = z[in_period].copy(), er[in_period].copy()
        er_p[0] = er_p[-1] = 0.5 * (er_p[0] + er_p[-1])
        per_spline = CubicSpline(z_p, er_p, bc_type="periodic")

        zg = self.map.z0 + self.map.dz * np.arange(self.map.n)
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
        """``Ei(z) = [Er(z+L) - Er(z) cos(theta)] / sin(theta)`` inside the periodic section, zero in
        the standing-wave coupler cells (see the class docstring for the Floquet derivation)."""
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
        """Map a structure-local ``z`` onto the field table's own coordinate, by folding the periodic
        section back onto its single stored period."""
        tile = torch.clamp(
            torch.floor((z_local - self.z1) / self.period), min=0.0, max=self.n_tiles - 1.0
        )
        return z_local - tile * self.period

    def on_axis(
        self, z: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute on axis; see the module documentation for details."""
        z_local = z - self.z_offset
        z_table = self._envelope_z(z_local)
        m, mi = self.map, self.map_imag
        idx = (z_table - m.z0) / m.dz
        i0 = torch.clamp(idx.floor().long(), 0, m.n - 2)
        w1 = (idx - i0).clamp(0.0, 1.0)
        w0 = 1.0 - w1
        i1 = i0 + 1
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

