"""Interpolates supplied RF/energy programs and scales fixed-geometry radiation parameters: energy loss
proportional to gamma^4, damping time to gamma^-3, natural spread to gamma and equilibrium emittance
to gamma^2. See docs/modules/lattice/ramp.md."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

from ..constants import ELECTRON_MASS_EV
from ..sdds_io import sdds_object
from .lattice import Lattice
from .optics import TransverseOptics


class RampState(NamedTuple):
    E0_eV: float
    Vrf_V: float
    f_rf_Hz: float
    phi_rf_rad: float


class RampArrayState(NamedTuple):
    """Array-valued twin of ``RampState`` -- one entry per requested time,
    see ``RampProfile.at_array``."""

    E0_eV: np.ndarray
    Vrf_V: np.ndarray
    f_rf_Hz: np.ndarray
    phi_rf_rad: np.ndarray


@dataclass
class RampProfile:
    """A time-sampled RF/momentum program."""

    time_s: np.ndarray
    E0_eV: np.ndarray
    Vrf_V: np.ndarray
    f_rf_Hz: np.ndarray
    phi_rf_rad: np.ndarray

    def at(self, t_s: float) -> RampState:
        return RampState(
            E0_eV=float(np.interp(t_s, self.time_s, self.E0_eV)),
            Vrf_V=float(np.interp(t_s, self.time_s, self.Vrf_V)),
            f_rf_Hz=float(np.interp(t_s, self.time_s, self.f_rf_Hz)),
            phi_rf_rad=float(np.interp(t_s, self.time_s, self.phi_rf_rad)),
        )

    def at_array(self, t_s: np.ndarray) -> RampArrayState:
        """Compute at array; see the module documentation for details."""
        return RampArrayState(
            E0_eV=np.interp(t_s, self.time_s, self.E0_eV),
            Vrf_V=np.interp(t_s, self.time_s, self.Vrf_V),
            f_rf_Hz=np.interp(t_s, self.time_s, self.f_rf_Hz),
            phi_rf_rad=np.interp(t_s, self.time_s, self.phi_rf_rad),
        )

    @classmethod
    def from_sdds_file(cls, path: str | Path, rest_mass_eV: float = ELECTRON_MASS_EV) -> "RampProfile":
        """Compute from sdds file; see the module documentation for details."""
        with sdds_object() as (_sdds, x):
            x.load(str(path))
            cols = x.columnName

            def col(name: str) -> np.ndarray:
                return np.array(x.columnData[cols.index(name)][0])

            columns = {name: col(name) for name in ("time", "momentum", "Voltage", "Frequency", "Phase")}
        E0_eV = np.sqrt(columns["momentum"] ** 2 + 1.0) * rest_mass_eV
        return cls(
            time_s=columns["time"],
            E0_eV=E0_eV,
            Vrf_V=columns["Voltage"],
            f_rf_Hz=columns["Frequency"],
            phi_rf_rad=np.deg2rad(columns["Phase"]),
        )


def scaled_lattice(base: Lattice, gamma_now: float) -> Lattice:
    """Scale reference energy and fixed-geometry radiation parameters to gamma_now."""
    ratio = gamma_now / base.gamma
    return dataclasses.replace(
        base,
        E0_eV=gamma_now * base.rest_mass_eV,
        U0_eV=base.U0_eV * ratio**4,
        tau_z_s=base.tau_z_s / ratio**3,
        sigma_delta0=base.sigma_delta0 * ratio,
    )


def scaled_optics(base: TransverseOptics, ratio: float) -> TransverseOptics:
    """Compute scaled optics; see the module documentation for details."""
    return dataclasses.replace(
        base,
        tau_x_s=base.tau_x_s / ratio**3,
        tau_y_s=base.tau_y_s / ratio**3,
        epsx_m=base.epsx_m * ratio**2,
        epsy_m=base.epsy_m * ratio**2,
    )
