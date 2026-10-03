"""Load a section straight from LAURA (the CLARA lattice source of truth,
the ``laura``/``laura_lattices`` packages) instead of a static ``.lte`` file
-- see plans/LAURA_INTERFACE_PLAN.md for the full design.

Optional dependency: only needed for this module, nothing else in
``slipstream`` requires ``laura``/``laura_lattices`` installed. Install both
editable from their real checkouts to use this (``pip install -e
<path-to-laura> -e <path-to-laura-lattices>``).
"""
from __future__ import annotations

import math
from dataclasses import replace

from .lattice_import import ElementSpec

try:
    from laura.laura import LAURA
    from laura.translator.converters.section import SectionLatticeTranslator
    _HAVE_LAURA = True
except ImportError:
    _HAVE_LAURA = False


def parse_laura_lattice(
    machine_module, section_name: str, load_wakes: bool = True
) -> list[ElementSpec]:
    """Real per-element linac list for ``LinacTracker``, in beamline order --
    the LAURA-sourced equivalent of ``lattice_import.parse_elegant_lattice``.

    Dispatch mirrors ``parse_elegant_lattice`` exactly: quadrupole/sextupole/
    dipole/RF-cavity keep their real physics, anything else with nonzero
    length becomes a plain drift (correctors, apertures, diagnostics --
    real elegant CLARA `.lte` files have no transport model for these
    either), zero-length dropped. Same 1-based ``occurrence`` counters, for
    the same downstream reason (error-matching identity, see ``errors.py``).

    ``load_wakes`` (default ``True``): a cavity's structure wake, if it has
    one, is read directly from LAURA's own HDF5 wakefield format (see
    ``laura_fields.wake_from_laura_hdf5``) rather than via
    ``linac.resolve_wakes``'s SDDS-only path -- set ``False`` to skip this
    (e.g. to inspect geometry only, or if a wake file is temporarily
    unavailable).

    Parameters
    ----------
    machine_module
        e.g. ``laura_lattices.CLARA`` -- passed straight to ``LAURA(lattice=...)``.
    section_name
        e.g. ``"S02"`` -- a key of ``LAURA(...).sections``.
    """
    if not _HAVE_LAURA:
        raise ImportError(
            "parse_laura_lattice needs the optional 'laura'/'laura_lattices' packages "
            "installed (pip install -e <path-to-laura> -e <path-to-laura-lattices>)"
        )
    lattice = LAURA(lattice=machine_module)
    section = lattice.sections[section_name]
    records = SectionLatticeTranslator.from_section(section).to_slipstream()

    occurrence_counts: dict[str, int] = {}
    out: list[ElementSpec] = []
    for record in records:
        spec = _to_element_spec(record, load_wakes=load_wakes)
        if spec is None:
            continue
        occurrence_counts[spec.name] = occurrence_counts.get(spec.name, 0) + 1
        out.append(replace(spec, occurrence=occurrence_counts[spec.name]))
    return out


def _to_element_spec(r, load_wakes: bool) -> ElementSpec | None:
    """One ``SlipstreamElement`` (see ``laura.translator.converters.codes.
    slipstream``) -> one ``ElementSpec``, or ``None`` if it's a zero-length
    no-op (marker/watch/zero-length aperture/etc)."""
    hw = r.hardware_type.lower()
    name = r.name.upper()
    if "quadrupole" in hw:
        return ElementSpec(name, "quad", r.length, k1=r.k1)
    if "sextupole" in hw:
        return ElementSpec(name, "sext", r.length, k2=r.k2)
    if hw == "dipole":
        # LAURA has no CSRCSBEND-vs-plain-CSBEND equivalent (both are
        # ASTRA/elegant-side-only quantities) -- csr=False (the ElementSpec
        # default) until a real convention exists, see the plan's Open items.
        return ElementSpec(name, "bend", r.length, angle=r.angle, e1=r.e1, e2=r.e2)
    if hw in ("rfcavity", "rfdeflectingcavity"):
        if r.length == 0.0:
            # A real zero-length "RFCavity" marker exists in FERMI's lattice (HERFX, a phase reference,
            # not an accelerating structure) -- dropped like any other zero-length no-op, since an
            # unconditional kind="rf" here would divide by zero in LinacTracker's RF-kick formula.
            return None
        body_focus_model = r.body_focus_model.lower()
        if body_focus_model not in ("none", "tw1", "srs"):
            body_focus_model = "none"
        wake, wake_x, wake_y = None, None, None
        if load_wakes and r.wakefield_hdf5_path:
            wake = _try_wake(r.wakefield_hdf5_path, "Wz")
            wake_x = _try_wake(r.wakefield_hdf5_path, "Wx")
            wake_y = _try_wake(r.wakefield_hdf5_path, "Wy")
        return ElementSpec(
            name, "rf", r.length,
            voltage_V=r.voltage_v, phase_rad=r.phase_deg * math.pi / 180.0,
            frequency_Hz=r.frequency_hz, change_p0=r.change_p0,
            n_kicks=r.n_kicks if r.n_kicks > 0 else 1,
            end1_focus=r.end1_focus, end2_focus=r.end2_focus,
            body_focus_model=body_focus_model,
            wake=wake, transverse_wake_x=wake_x, transverse_wake_y=wake_y,
        )
    if r.length != 0.0:
        return ElementSpec(name, "drift", r.length)
    return None


def _try_wake(path: str, component: str):
    """``wake_from_laura_hdf5(path, component)``, or ``None`` if that
    component isn't present in this particular wake file (e.g. a
    ``LongitudinalWake`` file has no ``Wx``/``Wy``)."""
    from .laura_fields import wake_from_laura_hdf5
    try:
        return wake_from_laura_hdf5(path, component)
    except ValueError:
        return None
