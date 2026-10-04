"""Stores single-point transverse Twiss/dispersion, tunes/chromaticities, equilibrium geometric
emittances and damping times. See docs/modules/lattice/optics.md."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..serialization import dump_yaml, load_yaml


@dataclass
class TransverseOptics:
    """Transverse machine parameters for the single-bunch one-turn map."""

    beta_x_m: float
    alpha_x: float
    D_x_m: float
    Q_x: float
    Qp_x: float  # chromaticity dQ_x/d(delta)
    tau_x_s: float  # transverse damping time (horizontal)
    epsx_m: float  # equilibrium (natural) geometric emittance, x

    beta_y_m: float
    alpha_y: float
    D_y_m: float
    Q_y: float
    Qp_y: float
    tau_y_s: float
    epsy_m: float

    # Dispersion slope (dD/ds) at this matched point -- optional, defaults to flat dispersion (0.0).
    Dp_x_m: float = 0.0
    Dp_y_m: float = 0.0

    def to_yaml(self, path: str | Path) -> None:
        dump_yaml(self, path)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TransverseOptics":
        return load_yaml(cls, path)
