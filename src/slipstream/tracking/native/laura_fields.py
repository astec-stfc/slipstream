"""Loads LAURA on-axis field profiles and wake tables into existing native field/wake classes. See
docs/modules/tracking/native/laura_fields.md."""
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
    """One LAURA HDF5 field-map file -> the matching slipstream field object, dispatched on the file's
    own ``type``/``cavity_type`` attrs (no need to tell this function which kind of element it is)."""
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
    """One LAURA HDF5 wake file -> ``ShortRangeWake`` (``component="Wz"``) or ``TransverseWake``
    (``component="Wx"``/``"Wy"``)."""
    if not _HAVE_LAURA:
        raise ImportError(
            "wake_from_laura_hdf5 needs the optional 'laura' package installed "
            "(pip install -e <path-to-laura>)"
        )
    field_map = FieldMap(filename=str(path))
    t_s = np.asarray(field_map.t_values, dtype=np.float64)
    w_param = getattr(field_map, component)
    if w_param.value is None:
        raise ValueError(f"{path}: no {component!r} component in this wake file")
    w = np.asarray(w_param.value, dtype=np.float64) * _unit_scale(getattr(w_param.value, "units", None))
    if component == "Wz":
        return ShortRangeWake(t_s, w)
    return TransverseWake(t_s, w)
