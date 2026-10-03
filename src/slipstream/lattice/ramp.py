"""Time-varying RF/momentum ramp for ramping synchrotrons (booster rings)
-- a sibling to ``Lattice``, not merged into it, mirroring how
``TransverseOptics`` is composed alongside ``Lattice`` rather than bolted
onto it (see ``lattice/optics.py``). Passed to ``SingleBunchTracker`` as
``ramp=``.

Two pieces:

- ``RampProfile`` -- the literal, externally-supplied time program for
  ``E0``/``Vrf``/``f_rf``/``phi_rf`` (e.g. loaded from an elegant
  RAMPP+RAMPRF-style SDDS file such as
  ``examples/diamond2booster/ramp_files/RfRamp<E>D2.sdds``). These are
  *not* derived from anything else -- whoever generated the ramp table
  already solved the energy-balance-plus-acceleration problem.
- ``scaled_lattice``/``scaled_optics`` -- rescale a ``Lattice``'s/
  ``TransverseOptics``'s injection-energy reference values to the
  instantaneous energy via the same power laws elegant's own ``SREFFECT``
  element uses (verified against ``apps/src/elegant/SReffects.c``, not
  guessed): for a fixed-geometry ring, ``U0 ~ gamma^4``, damping times
  ``~ 1/gamma^3``, equilibrium ``sigma_delta ~ gamma``, equilibrium
  emittances ``~ gamma^2``. ``alpha_c`` and the betatron optics
  (``beta``/``alpha``/``D``/``Q``/``Qp``) stay fixed -- elegant's own
  ``ILMATRIX`` doesn't re-Twiss during the ramp either. See
  ``plans/RAMP_PLAN.md`` for the full derivation and source citation.
"""
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
    """A time-sampled RF/momentum program. ``time_s`` must be sorted
    ascending; ``at()`` linearly interpolates each column (and clamps to
    the boundary values outside ``[time_s[0], time_s[-1]]``, via
    ``np.interp``'s own default behavior)."""

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
        """Vectorized twin of ``at()`` -- one ``np.interp`` call per column
        over the whole ``t_s`` array instead of one call per element, same
        numerics (``np.interp`` is deterministic regardless of whether it's
        called on a scalar or a batch). Used by
        ``SingleBunchTracker._track_ramped`` to interpolate a whole
        ``track(n_turns)`` call's worth of per-turn ramp state in one shot
        rather than once per turn -- the per-turn version was a measured
        CPU-side bottleneck (single-core-bound ``np.interp`` search calls
        starving the GPU of work) at particle counts too small for GPU
        kernel time to hide it, see ``plans/RAMP_PLAN.md``."""
        return RampArrayState(
            E0_eV=np.interp(t_s, self.time_s, self.E0_eV),
            Vrf_V=np.interp(t_s, self.time_s, self.Vrf_V),
            f_rf_Hz=np.interp(t_s, self.time_s, self.f_rf_Hz),
            phi_rf_rad=np.interp(t_s, self.time_s, self.phi_rf_rad),
        )

    @classmethod
    def from_sdds_file(cls, path: str | Path, rest_mass_eV: float = ELECTRON_MASS_EV) -> "RampProfile":
        """Load an elegant RAMPP+RAMPRF-style ramp table: columns ``time``
        [s], ``Voltage`` [V], ``Frequency`` [Hz], ``Phase`` [deg],
        ``momentum`` [rest-mass-c units] -- exactly what
        ``examples/diamond2booster/ramp_files/RfRamp<E>D2.sdds`` has.
        ``rest_mass_eV`` converts the momentum column to ``E0_eV`` via the
        exact relativistic relation (not the ultra-relativistic ``E~pc``
        approximation) -- defaults to the electron mass, override for a
        non-electron ring the same way ``Lattice.rest_mass_eV`` does.
        """
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
    """A copy of ``base`` with ``E0_eV``/``U0_eV``/``tau_z_s``/
    ``sigma_delta0`` rescaled from ``base``'s own (injection-energy)
    values to ``gamma_now``, via elegant ``SREFFECT``'s verified power
    laws (module docstring). ``Vrf_V``/``f_rf_Hz`` on the returned object
    are *not* meaningful -- the ramped RF program comes from
    ``RampProfile.at()`` directly, not from scaling a reference value."""
    ratio = gamma_now / base.gamma
    return dataclasses.replace(
        base,
        E0_eV=gamma_now * base.rest_mass_eV,
        U0_eV=base.U0_eV * ratio**4,
        tau_z_s=base.tau_z_s / ratio**3,
        sigma_delta0=base.sigma_delta0 * ratio,
    )


def scaled_optics(base: TransverseOptics, ratio: float) -> TransverseOptics:
    """A copy of ``base`` with ``tau_x_s``/``tau_y_s``/``epsx_m``/``epsy_m``
    rescaled from ``base``'s own (injection-energy) values by ``ratio =
    gamma_now/gamma_ref`` -- the same verified elegant ``SREFFECT`` power
    laws as ``scaled_lattice`` (``Srxp``'s ``exRef*gamma2Ratio`` term):
    transverse damping time ``~ 1/ratio**3`` (same ``Ddelta`` scaling as
    the longitudinal damping time), equilibrium emittance ``~ ratio**2``.
    ``beta``/``alpha``/``D``/``Q``/``Qp`` on the returned object are
    ``base``'s own unchanged values -- elegant's ``ILMATRIX`` doesn't
    re-Twiss during the ramp, see ``scaled_lattice``'s own docstring."""
    return dataclasses.replace(
        base,
        tau_x_s=base.tau_x_s / ratio**3,
        tau_y_s=base.tau_y_s / ratio**3,
        epsx_m=base.epsx_m * ratio**2,
        epsy_m=base.epsy_m * ratio**2,
    )
