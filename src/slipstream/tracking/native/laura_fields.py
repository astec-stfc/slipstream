"""Read RF/magnet field maps and structure wakefields directly from LAURA's
own HDF5 field-map format (``laura.translator.utils.fields.FieldMap``) --
see plans/LAURA_INTERFACE_PLAN.md. Bypasses ASTRA-text/elegant-SDDS
regeneration entirely: reuses LAURA's own HDF5 reader (units, the
point-count-vs-metres ``length`` quirk, wake ``t``/``z`` interconversion are
all handled there, not reimplemented here), then feeds the raw arrays
straight into slipstream's existing format-agnostic field/wake classes.

Optional dependency, same as ``laura_import.py``: only this module needs
``laura`` installed.

**Scope**: 1D on-axis field maps only (``1DElectroDynamic``,
``1DMagnetoStatic``, ``1DElectroStatic``, ``1DQuadrupole``) and 1D wakes
(``LongitudinalWake``, ``TransverseWake``, ``3DWake``). LAURA's 2D/3D
off-axis field maps (``2DElectroDynamic`` etc) need a genuine grid
interpolator slipstream doesn't have yet (every existing field class here is
on-axis-profile + analytic paraxial expansion, see ``fields.py``'s module
docstring) -- ``field_from_laura_hdf5`` raises ``NotImplementedError`` for
those rather than silently mishandling them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ...wakefields.shortrange import ShortRangeWake
from ...wakefields.transverse import TransverseWake
from .fields import RFCavityField, SolenoidField, TravelingWaveField

try:
    from laura.translator.utils.fields import FieldMap
    _HAVE_LAURA = True
except ImportError:
    _HAVE_LAURA = False

_SUPPORTED_FIELD_TYPES = {
    "1DElectroDynamic", "1DElectroStatic", "1DMagnetoStatic", "1DQuadrupole",
}
# V/pC and V/pC/m are the only non-SI unit spellings seen in real CLARA
# wake files (case varies, e.g. "V/c" for "V/C") -- normalize by string,
# not by blindly trusting a particular case.
_PC_TO_C = 1e12


def _unit_scale(units: str | None) -> float:
    u = (units or "").replace(" ", "").lower()
    if "pc" in u:
        return _PC_TO_C
    return 1.0


def field_from_laura_hdf5(
    path: str | Path,
    e_max_V_per_m: float | None = None,
    b_max_T: float | None = None,
    frequency_hz: float | None = None,
    n_cells: int = 0,
    phase_rad: float = 0.0,
    z_offset: float = 0.0,
    **kwargs,
) -> SolenoidField | RFCavityField | TravelingWaveField:
    """One LAURA HDF5 field-map file -> the matching slipstream field object,
    dispatched on the file's own ``type``/``cavity_type`` attrs (no need to
    tell this function which kind of element it is).

    ``e_max_V_per_m``/``b_max_T``/``frequency_hz`` are the *element's* own
    physical amplitude/frequency (``laura``'s ``SlipstreamElement.
    e_max_v_per_m``/``b_max_t``/``frequency_hz``) -- the HDF5 file itself
    stores a normalized profile (peak magnitude ~1), not physical units, same
    convention as the existing ASTRA-file readers in ``fields.py``.
    """
    if not _HAVE_LAURA:
        raise ImportError(
            "field_from_laura_hdf5 needs the optional 'laura' package installed "
            "(pip install -e <path-to-laura>)"
        )
    field_map = FieldMap(filename=str(path))
    field_type = field_map.field_type
    if field_type not in _SUPPORTED_FIELD_TYPES:
        raise NotImplementedError(
            f"{path}: field_type={field_type!r} is a 2D/3D off-axis field map -- "
            "slipstream has no grid-interpolated external-field tracking yet "
            "(only on-axis profile + paraxial expansion, see fields.py)"
        )
    z = np.asarray(field_map.z.value, dtype=np.float64)

    if field_type == "1DMagnetoStatic":
        bz = np.asarray(field_map.Bz.value, dtype=np.float64)
        return SolenoidField(z, bz, b_max_T, z_offset=z_offset, **kwargs)

    if field_type == "1DQuadrupole":
        raise NotImplementedError(
            "1DQuadrupole HDF5 profile reading not wired up yet -- "
            "magnets.profile_quad_element takes (z, g) arrays directly, "
            "read field_map.G.value if adding this."
        )

    # 1DElectroDynamic / 1DElectroStatic: an RF cavity, standing- or
    # travelling-wave (LAURA has no separate field_type for the latter --
    # cavity_type on the same 1DElectroDynamic map distinguishes them).
    ez = np.asarray(field_map.Ez.value, dtype=np.float64)
    if getattr(field_map, "cavity_type", None) == "TravellingWave":
        return TravelingWaveField(
            z, ez,
            z1=float(field_map.start_cell_z), z2=float(field_map.end_cell_z),
            n=int(field_map.mode_numerator), m=int(field_map.mode_denominator),
            e_max_V_per_m=e_max_V_per_m, frequency_hz=frequency_hz, n_cells=n_cells,
            phase_rad=phase_rad, z_offset=z_offset, **kwargs,
        )
    return RFCavityField(
        z, ez, e_max_V_per_m=e_max_V_per_m, frequency_hz=frequency_hz,
        phase_rad=phase_rad, z_offset=z_offset, **kwargs,
    )


def wake_from_laura_hdf5(path: str | Path, component: str) -> ShortRangeWake | TransverseWake:
    """One LAURA HDF5 wake file -> ``ShortRangeWake`` (``component="Wz"``) or
    ``TransverseWake`` (``component="Wx"``/``"Wy"``). ``t_s`` comes from
    LAURA's own ``FieldMap.t_values`` (``z / c``, confirmed exact match
    against LAURA's own elegant-SDDS wake export for a real CLARA cavity,
    see plans/LAURA_INTERFACE_PLAN.md's Validation section) rather than
    recomputed here."""
    if not _HAVE_LAURA:
        raise ImportError(
            "wake_from_laura_hdf5 needs the optional 'laura' package installed "
            "(pip install -e <path-to-laura>)"
        )
    field_map = FieldMap(filename=str(path))
    t_s = np.asarray(field_map.t_values, dtype=np.float64)
    w_param = getattr(field_map, component)
    if w_param.value is None:
        # A LongitudinalWake file has no Wx/Wy -- FieldMap always exposes
        # the FieldParameter itself (never a plain AttributeError), just
        # with .value=None, so this must be checked explicitly rather than
        # relying on attribute lookup to fail.
        raise ValueError(f"{path}: no {component!r} component in this wake file")
    w = np.asarray(w_param.value, dtype=np.float64) * _unit_scale(getattr(w_param.value, "units", None))
    if component == "Wz":
        return ShortRangeWake(t_s, w)
    return TransverseWake(t_s, w)
