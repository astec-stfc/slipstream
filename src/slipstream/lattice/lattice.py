"""Machine-agnostic longitudinal lattice parameters.

Holds the handful of longitudinal machine parameters needed by both the
pySTABLE tracker and the elegant driver, plus the small set of derived
quantities (revolution period, synchronous phase, synchrotron tune, ...)
both need. No lattice-specific (HALF, ALS-U, ...) information lives here --
concrete machines are YAML files under ``examples/<machine>/lattice.yaml``,
loaded via ``Lattice.from_yaml``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from ..constants import C_LIGHT, ELECTRON_MASS_EV
from ..serialization import dump_yaml, load_yaml


@dataclass
class Lattice:
    """Longitudinal machine parameters for the single-bunch one-turn map."""

    circumference_m: float
    E0_eV: float
    alpha_c: float
    Vrf_V: float
    f_rf_Hz: float
    U0_eV: float  # energy loss per turn
    tau_z_s: float  # longitudinal damping time
    sigma_delta0: float  # natural (equilibrium) energy spread
    rest_mass_eV: float = ELECTRON_MASS_EV  # defaults to electron -- override for e.g. a proton/antiproton ring
    alpha_c2: float = 0.0  # 2nd-order momentum compaction (elegant's ILMATRIX ALPHAC2) -- 0.0 is a pure first-order drift, unaffected

    @property
    def beta_rel(self) -> float:
        """Relativistic beta = v/c."""
        return math.sqrt(1.0 - 1.0 / self.gamma**2)

    @property
    def T0_s(self) -> float:
        """Revolution period."""
        return self.circumference_m / (self.beta_rel * C_LIGHT)

    @property
    def omega0(self) -> float:
        """Angular revolution frequency."""
        return 2 * math.pi / self.T0_s

    @property
    def harmonic_number(self) -> float:
        return self.f_rf_Hz * self.T0_s

    @property
    def omega_rf(self) -> float:
        return 2 * math.pi * self.f_rf_Hz

    @property
    def phi_s(self) -> float:
        """Synchronous phase (rad), second-quadrant root for alpha_c > 0."""
        return math.pi - math.asin(self.U0_eV / self.Vrf_V)

    @property
    def synchrotron_omega(self) -> float:
        """Small-amplitude synchrotron angular frequency (rad/s)."""
        Qs2 = (
            self.alpha_c * self.harmonic_number * self.Vrf_V * abs(math.cos(self.phi_s))
            / (2 * math.pi * self.E0_eV)
        )
        return self.omega0 * math.sqrt(Qs2)

    @property
    def natural_bunch_length_s(self) -> float:
        """Small-amplitude equilibrium rms bunch length (rms tau, seconds)."""
        return self.alpha_c * self.sigma_delta0 / self.synchrotron_omega

    @property
    def gamma(self) -> float:
        """Lorentz factor, ``E0_eV / rest_mass_eV``."""
        return self.E0_eV / self.rest_mass_eV

    def to_yaml(self, path: str | Path) -> None:
        dump_yaml(self, path)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Lattice":
        return load_yaml(cls, path)
