"""Stores round/flat chamber geometry, section length and material layers. See
docs/modules/wakefields/chambers.md."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RoundChamber:
    name: str
    radius_m: float
    length_m: float
    layers: list[tuple]


@dataclass
class FlatChamber:
    name: str
    half_gap_m: float
    length_m: float
    layers: list[tuple]


def layer_fields(layer: tuple) -> tuple[float, float, float, float]:
    """Return ``(rho, tau_ps, thickness_m, epsb)`` for one layer."""
    rho, tau_ps, thickness_m = layer[:3]
    epsb = layer[3] if len(layer) > 3 else 1.0
    return rho, tau_ps, thickness_m, epsb


@dataclass
class FreqScan:
    """Frequency-grid settings shared with the optional IW2D driver."""

    start_exp: float = 8
    stop_exp: float = 14.3
    scan_type: int = 2
    lin_exp: float = 9
    pts_per_decade: float = 20
    refine_fmin_THz: float = 0.05
    refine_fmax_THz: float = 5.0
    refine_n: int = 300

    def build_freq_hz(self) -> np.ndarray:
        freq: list[float] = []
        if self.scan_type == 1:
            step = 10.0**self.lin_exp
            f = 10.0**self.start_exp
            while True:
                freq.append(f)
                if f >= 10.0**self.stop_exp:
                    break
                f += step
        else:
            i = 0
            while True:
                f = 10.0 ** (self.start_exp + i / self.pts_per_decade)
                freq.append(f)
                i += 1
                if f >= 10.0**self.stop_exp:
                    break
            if self.scan_type == 2:
                f_min_hz = self.refine_fmin_THz * 1e12
                f_max_hz = self.refine_fmax_THz * 1e12
                freq += [
                    f_min_hz + i * (f_max_hz - f_min_hz) / self.refine_n
                    for i in range(self.refine_n + 1)
                ]
        return np.unique(freq)


__all__ = ["FlatChamber", "FreqScan", "RoundChamber", "layer_fields"]
