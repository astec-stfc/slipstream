"""Constructs field-map injector elements from chained LAURA sections. See
docs/modules/tracking/native/laura_injector.md."""
from __future__ import annotations

import warnings

from .injector import Aperture, Cavity, Injector, Quadrupole, Solenoid

try:
    from laura.laura import LAURA
    from laura.translator.converters.section import SectionLatticeTranslator
    _HAVE_LAURA = True
except ImportError:
    _HAVE_LAURA = False


def from_laura(
    machine_module,
    section_names: tuple[str, ...],
    z_stop_m: float,
    distribution: str = "",
    **injector_kwargs,
) -> Injector:
    """Compute from laura; see the module documentation for details."""
    if not _HAVE_LAURA:
        raise ImportError(
            "from_laura needs the optional 'laura'/'laura_lattices' packages "
            "installed (pip install -e <path-to-laura> -e <path-to-laura-lattices>)"
        )
    lattice = LAURA(lattice=machine_module)
    records = []
    for name in section_names:
        records += SectionLatticeTranslator.from_section(lattice.sections[name]).to_slipstream()

    cavities: list[Cavity] = []
    solenoids: list[Solenoid] = []
    quadrupoles: list[Quadrupole] = []
    apertures: list[Aperture] = []
    screens: list[float] = []
    ignored: list[str] = []

    for r in records:
        hw = r.hardware_type.lower()
        if hw in ("rfcavity", "rfdeflectingcavity"):
            if not r.field_map_hdf5_path:
                ignored.append(f"{r.name} (RF cavity with no resolved field map)")
                continue
            cavities.append(Cavity(
                file=r.field_map_hdf5_path, z_m=r.z_m,
                e_max_V_per_m=r.e_max_v_per_m, frequency_hz=r.frequency_hz,
                phase_deg=r.injector_phase_deg, n_cells=r.n_cells,
            ))
        elif hw == "solenoid":
            if not r.field_map_hdf5_path or r.b_max_t == 0.0:
                continue  # a real ASTRA deck likewise drops MaxB=0 solenoids
            solenoids.append(Solenoid(
                file=r.field_map_hdf5_path, z_m=r.z_m, b_max_T=r.b_max_t, x_m=r.dx,
            ))
        elif "quadrupole" in hw:
            if r.grad_T_per_m == 0.0:
                if r.k1 != 0.0:
                    ignored.append(f"{r.name} (standard quadrupole -- needs a reference momentum to convert k1 to a real gradient, see module docstring)")
                continue
            quadrupoles.append(Quadrupole(
                grad_T_per_m=r.grad_T_per_m, z_m=r.z_m, length_m=r.length, x_m=r.dx,
            ))
        elif hw == "dipole":
            ignored.append(f"{r.name} (dipole -- needs a reference momentum to convert to a real field strength, see module docstring)")
        elif hw == "aperture":
            apertures += _to_apertures(r)
        elif hw == "screen":
            screens.append(r.z_m)

    if ignored:
        warnings.warn("not modelled natively, ignored: " + "; ".join(ignored), stacklevel=2)

    return Injector(
        distribution=distribution,
        z_stop_m=z_stop_m,
        cavities=cavities,
        solenoids=solenoids,
        quadrupoles=quadrupoles,
        apertures=apertures,
        screens=sorted(screens),
        source=f"{machine_module.__name__}:{','.join(section_names)}",
        **injector_kwargs,
    )


def _to_apertures(r) -> list[Aperture]:
    """Compute to apertures; see the module documentation for details."""
    z1 = r.z_m
    # ASTRA's own degenerate-extent rule (_write_astra_common): a
    # zero-length aperture still needs Ap_Z2 > Ap_Z1, padded by 1mm.
    z2 = r.z_m + r.length if r.length >= 1e-3 else r.z_m + 1e-3
    shape = r.aperture_shape
    if shape in ("circular", "elliptical"):
        radius = r.aperture_radius_m or min(
            (v for v in (r.aperture_horizontal_m, r.aperture_vertical_m) if v), default=0.0,
        )
        if radius <= 0.0:
            return []
        return [Aperture(kind="RAD", z1_m=z1, z2_m=z2, r_m=radius, x_m=r.dx)]
    if shape in ("planar", "rectangular", "scraper"):
        prefix = "SCR" if shape == "scraper" else "COL"
        out = []
        if r.aperture_horizontal_m > 0:
            out.append(Aperture(kind=f"{prefix}_X", z1_m=z1, z2_m=z2, r_m=r.aperture_horizontal_m, x_m=r.dx))
        if r.aperture_vertical_m > 0:
            out.append(Aperture(kind=f"{prefix}_Y", z1_m=z1, z2_m=z2, r_m=r.aperture_vertical_m, x_m=r.dx))
        return out
    return []
