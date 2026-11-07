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


def _capturing() -> bool:
    """Whether a CUDA graph capture is underway on the current stream (False on a CPU-only PyTorch build, where the query raises)."""
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _bin_nodes(tau: torch.Tensor, dt_s: float, lo: int, deposit: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Grid nodes and weights of every particle for a fixed window starting at node ``lo``: indices and weights of shape ``(k, N)``,
    ``k = 1`` for the nearest grid point (``"ngp"``, no gradient to ``tau``) or ``k = 2`` for cloud in cell (``"cic"``, linear
    weights that depend on ``tau``). The node ``j`` sits at ``(lo + j) * dt_s``; an index outside ``[0, n_bins)`` is out of the window."""
    u = tau / dt_s
    if deposit == "ngp":
        return (torch.round(u).to(torch.int64) - lo)[None], torch.ones_like(tau)[None]
    i0 = torch.floor(u)
    frac = u - i0
    idx0 = i0.to(torch.int64) - lo
    return torch.stack([idx0, idx0 + 1]), torch.stack([1.0 - frac, frac])


def _deposit(idx: torch.Tensor, weight: torch.Tensor, n_bins: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter ``weight`` (``(k, N)``) onto ``n_bins`` nodes; weight outside the window goes to a discarded extra bin. Returns the
    grid and the boolean ``(N,)`` mask of particles with every node inside the window."""
    inside = (idx >= 0) & (idx < n_bins)
    safe = torch.where(inside, idx, n_bins)
    grid = torch.zeros(n_bins + 1, dtype=weight.dtype, device=weight.device)
    for k in range(idx.shape[0]):
        grid.scatter_add_(0, safe[k], weight[k])
    return grid[:n_bins], inside.all(0)


def _readback(field_bin: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Field at every particle from the node values ``field_bin``, with the weights of ``_bin_nodes``; outside the window the edge node."""
    return (field_bin[idx.clamp(0, field_bin.shape[0] - 1)] * weight).sum(0)


@dataclass
class ShortRangeWake:
    """A short-range wake potential table W(t) [V/C] on a uniform time grid.

    With a fixed bin window (``set_fixed_bin_window``) the binning has a static shape and no host synchronisation, so it can be
    captured in a CUDA graph. ``deposit="cic"`` (fixed window only) spreads each particle over its two neighbouring nodes and reads the
    voltage back by linear interpolation, which gives a gradient to the longitudinal position; the default ``"ngp"`` rounds to the
    nearest node and has none. ``monitor_overflow`` keeps the largest fraction of particles outside the fixed window seen since the
    last ``reset_overflow`` in the tensor ``overflow`` (updated in place, so it works inside a CUDA graph)."""

    t_s: np.ndarray
    W_V_per_C: np.ndarray
    deposit: str = "ngp"
    monitor_overflow: bool = False
    dt_s: float = field(init=False)
    _overflow: torch.Tensor | None = field(init=False, default=None, repr=False)
    _fixed_lo_idx: int | None = field(init=False, default=None, repr=False)
    _fixed_n_bins: int | None = field(init=False, default=None, repr=False)
    # Precomputed by set_fixed_bin_window/prepare.
    _fixed_o_min_w: int | None = field(init=False, default=None, repr=False)
    _fixed_conv_len: int | None = field(init=False, default=None, repr=False)
    _fixed_W_fft: torch.Tensor | None = field(init=False, default=None, repr=False)
    _W_fft_cache: dict = field(init=False, default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.deposit not in ("ngp", "cic"):
            raise ValueError("deposit must be 'ngp' or 'cic'")
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
        if self.monitor_overflow:
            self._overflow = torch.zeros((), dtype=dtype, device=device)

    @property
    def overflow(self) -> torch.Tensor | None:
        """Largest fraction of particles outside the fixed window since ``reset_overflow`` (a 0-d tensor), or ``None`` if not monitored."""
        return self._overflow

    def reset_overflow(self) -> None:
        if self._overflow is not None:
            self._overflow.zero_()

    def voltage_at(self, tau: torch.Tensor, charge_per_particle_C: torch.Tensor | float) -> torch.Tensor:
        """Voltage [V] felt by each particle from the present bunch's own short-range wake."""
        if self._fixed_n_bins is not None:
            lo, n_bins = self._fixed_lo_idx, self._fixed_n_bins
            idx, weight = _bin_nodes(tau, self.dt_s, lo, self.deposit)
            # Fixed-shape extra bin excludes out-of-range charge; scatter weights, then scale by macro-charge.
            counts, in_window = _deposit(idx, weight, n_bins)
            charge_bin = counts * charge_per_particle_C

            if self._fixed_W_fft is None:
                raise RuntimeError("call ShortRangeWake.prepare(dtype, device) before voltage_at")
            if self._overflow is not None:
                self._overflow.copy_(torch.maximum(self._overflow, (1.0 - in_window.to(tau.dtype).mean()).detach().to(self._overflow.dtype)))
            conv_len = self._fixed_conv_len
            fb = self._fixed_W_fft
            fa = torch.fft.rfft(charge_bin, n=conv_len)
            full = torch.fft.irfft(fa * fb, n=conv_len)
            start = -self._fixed_o_min_w
            V_bin = full[start:start + n_bins]
            return _readback(V_bin, idx, weight)

        if self.deposit != "ngp":
            raise RuntimeError("deposit='cic' needs a fixed bin window (set_fixed_bin_window)")
        if _capturing():
            raise RuntimeError(
                "a CUDA graph cannot capture the dynamic wake window (it reads the bunch extent on the host); call set_fixed_bin_window "
                "(or LinacTracker.with_wake_windows) before capture"
            )
        idx = torch.round(tau / self.dt_s).to(torch.int64)
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
