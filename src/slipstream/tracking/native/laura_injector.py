"""Build an ``Injector`` (the space-charge-dominated gun/booster tracker)
straight from a chain of LAURA sections, instead of an ASTRA input deck --
the LAURA-sourced equivalent of ``Injector.from_astra``. See
``plans/LAURA_INTERFACE_PLAN.md``.

Optional dependency: only needed for this module, same as
``laura_import.py``/``laura_fields.py``.

**Validated against a real machine**: CLARA's real ASTRA injector deck
(``examples/clara_linac/ASTRA/5pC/injector400.in``) covers exactly the
LAURA sections ``HRG1``, ``S01``, ``L01`` (confirmed directly -- every real
``C_pos``/``S_pos``/``Ap_Z1``/``Ap_R`` value in that deck matches this
module's own output for the same elements to the deck's own rounding, and
its ``zstop=3.29693`` is exactly ``CLA-S02-SIM-APER-01``'s own start
z -- S02's own first element, correctly *not* included here, since tracking
already stops at that same z regardless).

**Not supported here** (both real, documented gaps -- CLARA's own real
injector needs neither, so neither is exercised or validated):

- **Dipoles**: ``Injector.Dipole.strength_T`` needs a real field in tesla;
  a LAURA dipole normally only carries ``k1``-equivalent multipole strength,
  needing a reference momentum to convert -- unknown at lattice-construction
  time. ``Injector.from_astra`` has the *identical* limitation (it warns and
  skips ``D_radius``, which needs the same conversion) -- not a new gap this
  module introduces.
- **"Standard" (non-profile) quadrupoles**: same reference-momentum problem
  for turning ``k1`` into a real ``grad_T_per_m`` -- ``Injector.from_astra``
  warns and skips these too (``Q_K given without Q_grad``). A *profile*-type
  quadrupole (LAURA's own real field-gradient value, ``magnetic.gradient``)
  works today.

Both are silently skipped with one combined warning (mirroring
``Injector.from_astra``'s own "everything ignored, one warning" convention)
rather than raising -- a machine with only cavities/solenoids/apertures
(like CLARA's) still builds cleanly.
"""
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
    """Build an ``Injector`` from ``section_names`` (in beamline order,
    e.g. ``("HRG1", "S01", "L01")`` for CLARA), stopping tracking at
    ``z_stop_m``.

    ``distribution``/``generator`` (the latter via ``injector_kwargs``):
    LAURA has no single-file equivalent of an ASTRA deck's own
    ``Distribution=``/a paired ``generator.in`` -- pass an existing ASTRA-
    format distribution file or ``generator.in`` explicitly, exactly as
    ``Injector.from_astra`` itself only ever reads a path, never generates
    one from the deck alone. Any other ``Injector`` field (``dt_s``,
    ``space_charge=``, ...) can be overridden via ``injector_kwargs``.
    """
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
    """One aperture ``SlipstreamElement`` -> 0/1/2 ``Injector.Aperture``
    objects, mirroring ``ApertureTranslator.to_astra()``'s own shape
    dispatch exactly (``aperture.py``): circular/elliptical -> one ``RAD``;
    planar/rectangular/scraper -> up to two (X and Y), whichever sizes are
    actually positive."""
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
