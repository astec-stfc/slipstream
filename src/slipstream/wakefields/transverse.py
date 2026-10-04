"""Dipole wake voltage is a convolution against charge-weighted transverse offset rather than charge
alone. See docs/modules/wakefields/transverse.md."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..sdds_io import sdds_object
from .impedance import Impedance
from .shortrange import next_fast_fft_len, trim_wake, wake_potential_from_impedance


@dataclass
class TransverseWake:
    """A transverse dipole wake potential table Wx(t) [V/C/m] (or Wy) on a uniform time grid -- same
    shape/conventions as ``ShortRangeWake`` (``t_s`` may include a small negative acausal precursor,
    t>=0 is the causal part)."""

    t_s: np.ndarray
    W_V_per_C_per_m: np.ndarray
    dt_s: float = field(init=False)
    _fixed_lo_idx: int | None = field(init=False, default=None, repr=False)
    _fixed_n_bins: int | None = field(init=False, default=None, repr=False)
    _fixed_o_min_w: int | None = field(init=False, default=None, repr=False)
    _fixed_conv_len: int | None = field(init=False, default=None, repr=False)
    _fixed_W_fft: torch.Tensor | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.t_s = np.array(self.t_s, dtype=np.float64)
        self.W_V_per_C_per_m = np.array(self.W_V_per_C_per_m, dtype=np.float64)
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
    ) -> "TransverseWake":
        t, W = wake_potential_from_impedance(impedance, sigma_s_m, dt=dt, n_points=n_points)
        if trim is not None:
            t, W = trim_wake(t, W, *trim)
        return cls(t, W)

    def to_csv(self, path: str | Path) -> None:
        pd.DataFrame({"t_s": self.t_s, "W_V_per_C_per_m": self.W_V_per_C_per_m}).to_csv(path, index=False)

    @classmethod
    def from_csv(cls, path: str | Path) -> "TransverseWake":
        df = pd.read_csv(path)
        return cls(df["t_s"].to_numpy(), df["W_V_per_C_per_m"].to_numpy())

    @classmethod
    def from_sdds_file(cls, path: str | Path, t_column: str = "t", w_column: str = "W") -> "TransverseWake":
        """Compute from sdds file; see the module documentation for details."""
        with sdds_object() as (_sdds, x):
            x.load(str(path))
            cols = x.columnName
            t = np.array(x.columnData[cols.index(t_column)][0])
            W = np.array(x.columnData[cols.index(w_column)][0])
        return cls(t, W)

    def truncated(self, t_min_s: float, t_max_s: float) -> "TransverseWake":
        """A copy restricted to ``[t_min_s, t_max_s]`` -- see
        ``ShortRangeWake.truncated``."""
        mask = (self.t_s >= t_min_s) & (self.t_s <= t_max_s)
        return TransverseWake(self.t_s[mask], self.W_V_per_C_per_m[mask])

    def set_fixed_bin_window(self, t_min_s: float, t_max_s: float) -> None:
        """Compute set fixed bin window; see the module documentation for details."""
        self._fixed_lo_idx = int(round(t_min_s / self.dt_s))
        self._fixed_n_bins = int(round((t_max_s - t_min_s) / self.dt_s))
        self._fixed_o_min_w = int(round(self.t_s[0] / self.dt_s))
        self._fixed_conv_len = next_fast_fft_len(self._fixed_n_bins + self.W_V_per_C_per_m.shape[0] - 1)
        start = -self._fixed_o_min_w
        if start < 0 or start + self._fixed_n_bins > self._fixed_conv_len:
            raise ValueError("wake table does not cover the fixed bin window")
        self._fixed_W_fft = None

    def prepare(self, dtype: torch.dtype, device: str | torch.device) -> None:
        """Precompute the fixed window's wake FFT -- see
        ``ShortRangeWake.prepare``. No-op if no fixed window is set."""
        if self._fixed_n_bins is None:
            return
        W = torch.as_tensor(self.W_V_per_C_per_m, dtype=dtype, device=device)
        self._fixed_W_fft = torch.fft.rfft(W, n=self._fixed_conv_len)

    def kick_at(
        self, tau: torch.Tensor, offset: torch.Tensor, charge_per_particle_C: torch.Tensor | float
    ) -> torch.Tensor:
        """Transverse kick voltage [V] felt by each particle from the present bunch's own dipole wake,
        given its own transverse offset (x or y) [m]."""
        if self._fixed_n_bins is None:
            raise RuntimeError("call TransverseWake.set_fixed_bin_window before kick_at")
        if self._fixed_W_fft is None:
            raise RuntimeError("call TransverseWake.prepare(dtype, device) before kick_at")

        lo, n_bins = self._fixed_lo_idx, self._fixed_n_bins
        idx = torch.round(tau / self.dt_s).to(torch.int64)
        in_range = (idx >= lo) & (idx < lo + n_bins)
        safe_idx = torch.where(in_range, idx - lo, n_bins)
        dipole_bin_full = torch.zeros(n_bins + 1, dtype=tau.dtype, device=tau.device)
        # Scatter bare offset, scale by charge_per_particle_C after binning -- same reasoning as
        # Match the longitudinal wake's fixed-window binning.
        dipole_bin_full.scatter_add_(0, safe_idx, offset)
        dipole_bin = dipole_bin_full[:n_bins] * charge_per_particle_C
        idx_readback = idx.clamp(lo, lo + n_bins - 1)

        conv_len = self._fixed_conv_len
        fb = self._fixed_W_fft
        fa = torch.fft.rfft(dipole_bin, n=conv_len)
        full = torch.fft.irfft(fa * fb, n=conv_len)
        start = -self._fixed_o_min_w
        V_bin = full[start:start + n_bins]
        return V_bin[idx_readback - lo]
