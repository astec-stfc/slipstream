"""Constructs a Gaussian-probe wake from impedance via inverse Fourier transformation, enforcing
`Z(-omega)=conj(Z(omega))`. See docs/modules/wakefields/shortrange.md."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..constants import C_LIGHT
from ..sdds_io import sdds_object
from .impedance import Impedance


def wake_potential_from_impedance(
    impedance: Impedance,
    sigma_s_m: float,
    dt: float = 2e-15,
    n_points: int = 1 << 15,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (t [s], W [V/C]), t centered on 0, for a Gaussian probe bunch."""
    sigma_t = sigma_s_m / C_LIGHT

    f_fft = np.fft.fftfreq(n_points, d=dt)
    af = np.abs(f_fft)
    re_i = np.interp(af, impedance.freq_Hz, impedance.ReZ_ohm, left=0.0, right=0.0)
    im_i = np.interp(af, impedance.freq_Hz, impedance.ImZ_ohm, left=0.0, right=0.0)
    Z = re_i + 1j * im_i
    Z = np.where(f_fft < 0, np.conj(Z), Z)

    omega = 2 * np.pi * f_fft
    lam = np.exp(-0.5 * (omega * sigma_t) ** 2)

    W = np.fft.ifft(Z * lam) / dt
    W = W.real

    t = np.arange(n_points) * dt
    t = np.where(t >= n_points * dt / 2, t - n_points * dt, t)
    order = np.argsort(t)
    return t[order], W[order]


def next_fast_fft_len(n: int) -> int:
    """Compute next fast fft len; see the module documentation for details."""
    return 1 << (n - 1).bit_length()


def trim_wake(t: np.ndarray, W: np.ndarray, t_min: float, t_max: float) -> tuple[np.ndarray, np.ndarray]:
    """Keep only [t_min, t_max] -- a tracker (or elegant's WAKE element) wants a compact table, not the
    full FFT window."""
    mask = (t >= t_min) & (t <= t_max)
    return t[mask], W[mask]


@dataclass
class ShortRangeWake:
    """A short-range wake potential table W(t) [V/C] on a uniform time grid."""

    t_s: np.ndarray
    W_V_per_C: np.ndarray
    dt_s: float = field(init=False)
    _fixed_lo_idx: int | None = field(init=False, default=None, repr=False)
    _fixed_n_bins: int | None = field(init=False, default=None, repr=False)
    # Precomputed by set_fixed_bin_window/prepare.
    _fixed_o_min_w: int | None = field(init=False, default=None, repr=False)
    _fixed_conv_len: int | None = field(init=False, default=None, repr=False)
    _fixed_W_fft: torch.Tensor | None = field(init=False, default=None, repr=False)
    _W_fft_cache: dict = field(init=False, default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.t_s = np.asarray(self.t_s, dtype=np.float64)
        self.W_V_per_C = np.asarray(self.W_V_per_C, dtype=np.float64)
        dt = np.diff(self.t_s)
        if len(dt) == 0 or not np.allclose(dt, dt[0], rtol=1e-6):
            raise ValueError("wake table must be on a uniform time grid")
        self.dt_s = float(dt[0])

    @classmethod
    def from_impedance(
        cls,
        impedance: Impedance,
        sigma_s_m: float,
        dt: float = 2e-15,
        n_points: int = 1 << 15,
        trim: tuple[float, float] | None = None,
    ) -> "ShortRangeWake":
        t, W = wake_potential_from_impedance(impedance, sigma_s_m, dt=dt, n_points=n_points)
        if trim is not None:
            t, W = trim_wake(t, W, *trim)
        return cls(t, W)

    @classmethod
    def from_sdds_file(cls, path: str | Path, t_column: str = "t", w_column: str = "W") -> "ShortRangeWake":
        """Compute from sdds file; see the module documentation for details."""
        with sdds_object() as (_sdds, x):
            x.load(str(path))
            cols = x.columnName
            t = np.array(x.columnData[cols.index(t_column)][0])
            W = np.array(x.columnData[cols.index(w_column)][0])
        return cls(t, W)

    def to_sdds_file(self, path: str | Path) -> None:
        """Write an ASCII SDDS file with columns t[s], W[V/C], for
        elegant's WAKE element (TCOLUMN="t", WCOLUMN="W")."""
        with sdds_object() as (sdds, x):
            x.mode = sdds.SDDS_ASCII
            x.description = ["Pseudo-Green-function longitudinal wake potential", ""]
            x.defineColumn("t", units="s", type=sdds.SDDS_DOUBLE)
            x.defineColumn("W", units="V/C", type=sdds.SDDS_DOUBLE)
            x.setColumnValueList("t", self.t_s.tolist())
            x.setColumnValueList("W", self.W_V_per_C.tolist())
            x.save(str(path))

    def truncated(self, t_min_s: float, t_max_s: float) -> "ShortRangeWake":
        """A copy restricted to ``[t_min_s, t_max_s]`` -- cuts FFT convolution
        cost when the wake has decayed to negligible levels well inside the
        table's full range."""
        mask = (self.t_s >= t_min_s) & (self.t_s <= t_max_s)
        return ShortRangeWake(self.t_s[mask], self.W_V_per_C[mask])

    def set_fixed_bin_window(self, t_min_s: float, t_max_s: float) -> None:
        """Fix the per-turn charge-binning window instead of resizing it to the bunch's instantaneous
        tau range every turn."""
        self._fixed_lo_idx = int(round(t_min_s / self.dt_s))
        self._fixed_n_bins = int(round((t_max_s - t_min_s) / self.dt_s))
        self._fixed_o_min_w = int(round(self.t_s[0] / self.dt_s))
        self._fixed_conv_len = next_fast_fft_len(self._fixed_n_bins + self.W_V_per_C.shape[0] - 1)
        start = -self._fixed_o_min_w
        if start < 0 or start + self._fixed_n_bins > self._fixed_conv_len:
            raise ValueError("wake table does not cover the fixed bin window")
        self._fixed_W_fft = None
        self._W_fft_cache.clear()

    def prepare(self, dtype: torch.dtype, device: str | torch.device) -> None:
        """Precompute the fixed window's wake FFT for a given dtype/device ahead of time (no-op if no
        fixed window is set)."""
        if self._fixed_n_bins is None:
            return
        W = torch.as_tensor(self.W_V_per_C, dtype=dtype, device=device)
        self._fixed_W_fft = torch.fft.rfft(W, n=self._fixed_conv_len)

    def voltage_at(self, tau: torch.Tensor, charge_per_particle_C: torch.Tensor | float) -> torch.Tensor:
        """Voltage [V] felt by each particle from the present bunch's own short-range wake."""
        idx = torch.round(tau / self.dt_s).to(torch.int64)
        if self._fixed_n_bins is not None:
            lo, n_bins = self._fixed_lo_idx, self._fixed_n_bins
            # Fixed-shape extra bin excludes out-of-range charge.
            in_range = (idx >= lo) & (idx < lo + n_bins)
            safe_idx = torch.where(in_range, idx - lo, n_bins)
            charge_bin_full = torch.zeros(n_bins + 1, dtype=tau.dtype, device=tau.device)
            # Scatter counts, then scale by macro-charge.
            charge_bin_full.scatter_add_(0, safe_idx, torch.ones_like(tau))
            charge_bin = charge_bin_full[:n_bins] * charge_per_particle_C
            idx_readback = idx.clamp(lo, lo + n_bins - 1)

            if self._fixed_W_fft is None:
                raise RuntimeError("call ShortRangeWake.prepare(dtype, device) before voltage_at")
            conv_len = self._fixed_conv_len
            fb = self._fixed_W_fft
            fa = torch.fft.rfft(charge_bin, n=conv_len)
            full = torch.fft.irfft(fa * fb, n=conv_len)
            start = -self._fixed_o_min_w
            V_bin = full[start:start + n_bins]
            return V_bin[idx_readback - lo]

        lo = int(idx.min().item())
        n_bins = int(idx.max().item()) - lo + 1
        hist_idx = idx - lo
        charge_bin = torch.bincount(hist_idx, minlength=n_bins).to(tau.dtype) * charge_per_particle_C

        o_min_w = int(round(self.t_s[0] / self.dt_s))
        conv_len = n_bins + self.W_V_per_C.shape[0] - 1
        cache_key = (conv_len, tau.device, tau.dtype)
        fb = self._W_fft_cache.get(cache_key)
        if fb is None:
            W = torch.as_tensor(self.W_V_per_C, dtype=tau.dtype, device=tau.device)
            fb = torch.fft.rfft(W, n=conv_len)
            self._W_fft_cache[cache_key] = fb
        fa = torch.fft.rfft(charge_bin, n=conv_len)
        full = torch.fft.irfft(fa * fb, n=conv_len)

        # V_bin[j] = sum_i charge_bin[i] * W[j - i] = full[j - o_min_w]
        start = -o_min_w
        if start < 0 or start + n_bins > conv_len:
            raise ValueError("wake table does not cover the bunch's tau range")
        V_bin = full[start:start + n_bins]

        return V_bin[idx - lo]
