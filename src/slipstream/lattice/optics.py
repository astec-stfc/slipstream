"""Transverse (betatron) optics parameters -- a sibling to ``Lattice``, not
merged into it, so every existing longitudinal-only ``lattice.yaml``/tracker
stays valid untouched. Passed alongside ``Lattice`` into a tracker the same
way ``ShortRangeWake``/``LongRangeWake`` already are (see
``tracking/pystable/tracker.py``).

Linear one-turn-map approximation throughout: a single matched point (fixed
beta/alpha in and out), no nonlinear optics -- the same level of realism
``Lattice``'s own longitudinal map, PyHEADTAIL's ``Synchrotron``, and
elegant's ``ILMATRIX`` already use elsewhere in this codebase.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..serialization import dump_yaml, load_yaml


@dataclass
class TransverseOptics:
    """Transverse machine parameters for the single-bunch one-turn map.

    ``epsx_m``/``epsy_m`` are geometric (not normalized) equilibrium
    emittances, and ``tau_x_s``/``tau_y_s`` transverse damping times --
    direct inputs, matching ``Lattice``'s own flat-parameter style for
    ``tau_z_s``/``sigma_delta0`` rather than deriving them from
    synchrotron-radiation integrals.
    """

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

    # Dispersion slope (dD/ds) at this matched point -- optional, defaults to
    # flat dispersion (0.0). Not used by the betatron/damping maps above;
    # only consumed by slipstream.ibs's dispersion invariant H_x/H_y, where
    # a nonzero slope matters (e.g. a point inside a bend). Defaulting to 0
    # keeps every existing constructor call/YAML file valid unchanged.
    Dp_x_m: float = 0.0
    Dp_y_m: float = 0.0

    def to_yaml(self, path: str | Path) -> None:
        dump_yaml(self, path)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TransverseOptics":
        return load_yaml(cls, path)
