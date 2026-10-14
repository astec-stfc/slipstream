"""Constructs field-map injector elements from chained LAURA sections. See
docs/modules/tracking/native/laura_injector.md."""
from __future__ import annotations

import warnings
from types import SimpleNamespace

from .injector import Aperture, Cavity, Injector, Quadrupole, Solenoid

try:
    from laura.laura import LAURA
    from laura.translator.converters.converter import translate_elements
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
    lattice = machine_module if isinstance(machine_module, LAURA) else LAURA(
        lattice=machine_module,
        layout={"default_layout": "INJECTOR", "layouts": {"INJECTOR": list(section_names)}},
    )
    records = []
    for name in section_names:
        section = lattice.sections[name]
        elements = translate_elements(
            [section.elements[n] for n in section.order], master_lattice=section.master_lattice,
        )
        records += [_element_record(e) for e in elements.values()]

    cavities: list[Cavity] = []
    solenoids: list[Solenoid] = []
    quadrupoles: list[Quadrupole] = []
    apertures: list[Aperture] = []
    screens: list[float] = []
    ignored: list[str] = []

    absolute_phases = injector_kwargs.get("absolute_phases", False)
    for r in records:
        hw = r.hardware_type.lower()
        if hw in ("rfcavity", "rfdeflectingcavity"):
            if not r.field_map_hdf5_path:
                ignored.append(f"{r.name} (RF cavity with no resolved field map)")
                continue
            cavities.append(Cavity(
                file=r.field_map_hdf5_path, z_m=r.z_m,
                e_max_V_per_m=r.e_max_v_per_m, frequency_hz=r.frequency_hz,
                phase_deg=r.crest_deg + r.injector_phase_deg if absolute_phases else r.injector_phase_deg,
                n_cells=r.n_cells, crest_deg=-90.0 if absolute_phases else None,
            ))
        elif hw == "solenoid":
            if not r.field_map_hdf5_path or r.b_max_t == 0.0:
                continue  # a real ASTRA deck likewise drops MaxB=0 solenoids
            solenoids.append(Solenoid(
                file=r.field_map_hdf5_path, z_m=r.z_m, b_max_T=r.b_max_t, x_m=r.dx, y_m=r.dy,
                xrot_rad=r.xrot_rad, yrot_rad=r.yrot_rad,
            ))
        elif "quadrupole" in hw:
            if r.grad_T_per_m == 0.0:
                if r.k1 != 0.0:
                    ignored.append(f"{r.name} (standard quadrupole -- needs a reference momentum to convert k1 to a real gradient, see module docstring)")
                continue
            quadrupoles.append(Quadrupole(
                grad_T_per_m=r.grad_T_per_m, z_m=r.z_m, length_m=r.length, x_m=r.dx, xrot_rad=r.xrot_rad,
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
        source=f"{getattr(machine_module, '__name__', lattice.master_lattice)}:{','.join(section_names)}",
        **injector_kwargs,
    )


def _element_record(e):
    """Use LAURA's typed translator conventions without writing converted field files."""
    hw = e.hardware_type.lower()
    is_quad = "quadrupole" in hw
    ref = e.get_field_reference_position(if_none="middle" if is_quad else "start")
    r = SimpleNamespace(
        name=e.name, hardware_type=hw, z_m=float(ref[2] + e.dz), length=e.length,
        dx=float(ref[0] + e.dx), dy=float(ref[1] + e.dy),
        xrot_rad=e._astra_rotation("x"), yrot_rad=e._astra_rotation("y"),
        field_map_hdf5_path=None, grad_T_per_m=0.0, k1=0.0,
    )
    if hw in ("rfcavity", "rfdeflectingcavity", "solenoid"):
        e.update_field_definition()
        field_map = e.simulation.field_definition
        r.field_map_hdf5_path = field_map.filename if field_map is not None else None
        if hw == "solenoid":
            r.b_max_t = float(e.resolve(e.magnetic.field_amplitude))
        else:
            if r.dx or r.dy or r.xrot_rad or r.yrot_rad:
                raise NotImplementedError(f"{e.name}: placed RF cavities are not supported by Injector")
            r.e_max_v_per_m = float(e.field_amplitude)
            r.frequency_hz = float(e.cavity.frequency)
            r.injector_phase_deg = -float(e.phase)  # LAURA off-crest phase -> ASTRA Phi with Auto_phase.
            r.n_cells = int(e.get_cells())
            r.crest_deg = float(e.cavity.crest)
    elif is_quad:
        if e.simulation.field_definition:
            raise NotImplementedError(f"{e.name}: quadrupole field profiles are not supported by this importer")
        if e.dy or e._astra_rotation("y") or e._astra_rotation("z"):
            raise NotImplementedError(f"{e.name}: this quadrupole placement is not supported by Injector")
        r.grad_T_per_m = float(e.resolve(e.magnetic.gradient)) if e.magnetic.gradient is not None else 0.0
        r.k1 = float(e.resolve(e.k1 + e.dk1))
    elif hw == "aperture":
        if r.dy or r.yrot_rad or e._astra_rotation("z"):
            raise NotImplementedError(f"{e.name}: this aperture placement is not supported by Injector")
        r.z_m = float(e.physical.start.z + e.dz)
        r.aperture_shape = e.aperture.shape
        r.aperture_radius_m = e.aperture.radius or 0.0
        r.aperture_horizontal_m = e.aperture.horizontal_size or 0.0
        r.aperture_vertical_m = e.aperture.vertical_size or 0.0
        if e.aperture.negative_extent is not None or e.aperture.positive_extent is not None:
            raise NotImplementedError(f"{e.name}: relative aperture extents are not supported by this importer")
    elif "corrector" in hw:
        if e.resolve(e.magnetic.horizontal_kick) or e.resolve(e.magnetic.vertical_kick):
            warnings.warn(f"{e.name}: powered corrector is not modelled by Injector", stacklevel=3)
    return r


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
        return [Aperture(kind="RAD", z1_m=z1, z2_m=z2, r_m=radius, x_m=r.dx, xrot_rad=r.xrot_rad)]
    if shape in ("planar", "rectangular", "scraper"):
        prefix = "SCR" if shape == "scraper" else "COL"
        out = []
        if r.aperture_horizontal_m > 0:
            out.append(Aperture(kind=f"{prefix}_X", z1_m=z1, z2_m=z2, r_m=r.aperture_horizontal_m, x_m=r.dx, xrot_rad=r.xrot_rad))
        if r.aperture_vertical_m > 0:
            out.append(Aperture(kind=f"{prefix}_Y", z1_m=z1, z2_m=z2, r_m=r.aperture_vertical_m, x_m=r.dx, xrot_rad=r.xrot_rad))
        return out
    return []
