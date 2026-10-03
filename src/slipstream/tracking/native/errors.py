"""Error sampling and application for ``tracking.native`` -- see
``plans/PHYSICS.md``'s "Errors and fiducialization" section for the design
rationale and docs/tracking_native.md for usage.

Mirrors real elegant's own ``&error_element`` model (``elegant/src/
error.c``/``error.nl``/``vary.c``): a name/type-matched
error rule (``ErrorSpec``) draws one random value per matching element
instance from a chosen distribution, resolved into either an *absolute*
(additive) or *fractional* (multiplicative, relative to that element's own
current value) change -- ``elegant``'s ``fractional``/``additive`` flags,
not a "TYPE=ABSOLUTE/FRACTIONAL" enum. One draw per realization (matching
this module's own scope -- see ``plans/NATIVE_ERRORS_PLAN.md``'s explicit
"turn-by-turn-varying errors" non-goal): call ``sample_errors`` once before
tracking, not per turn.

**Magnetic field and alignment errors** (``K1``/``K2``/``ANGLE``/``DX``/
``DY``/``TILT``) resolve onto the matching ``ElementSpec``'s own field via
``apply_errors`` -- a pure, one-shot, construction-time transform (returns
a *new* ``list[ElementSpec]``), since a per-element-instance plain-float
perturbation is already exactly how ``tracker._build_params`` treats every
element (no ``torch.compile`` implications -- see ``PHYSICS.md`` Gotcha 14).

**RF errors** (``PHASE``/``VOLT``/``FREQ``) apply through **two**
mechanisms, one per RF model (``plans/NATIVE_RF_PLAN.md``/
``plans/NATIVE_LINAC_PLAN.md``): for the *lumped*, once-per-turn model
(``NativeTracker(rf="lumped")``, the default -- RF applied externally via
``pystable.rf_kick`` using ``Lattice.Vrf_V``/``f_rf_Hz``/``phi_s``, or
``RampProfile.at()``'s per-turn state during a ramp), ``resolve_rf_errors``
sums these into a single ``{"phi_s", "Vrf_V", "omega_rf"}`` delta dict,
which ``NativeTracker`` adds on top of whatever it would otherwise use via
``rf_kick``'s own ``*_eff`` override mechanism -- **not** by mutating
``Lattice``, since ``Lattice.phi_s`` is a derived property (from
``U0_eV``/``Vrf_V``), not a stored field a real RF phase error could shift
directly. For the *per-cavity* model (``rf="per_element"``/
``tracking.native.linac.LinacTracker``, where each real ``RFCA``/``RFCW``
element has its *own* phase/voltage/frequency), ``apply_errors`` resolves
these the same way as ``K1``/``K2``/etc: directly onto the matching
``kind="rf"`` ``ElementSpec``'s own field. Both mechanisms run off the
same ``sampled`` list; each tracking mode reads only the one it needs.
"""
from __future__ import annotations

import fnmatch
import math
from dataclasses import dataclass, replace

import torch

from .lattice_import import ElementSpec

# ElementSpec field each magnetic/alignment error parameter resolves onto.
_FIELD_NAMES = {"K1": "k1", "K2": "k2", "ANGLE": "angle", "DX": "dx", "DY": "dy", "TILT": "tilt"}
_RF_PARAMS = {"PHASE", "VOLT", "FREQ"}
# ElementSpec field each RF error parameter resolves onto, for a
# kind="rf" element -- see apply_errors's own docstring for why this
# coexists with (does not replace) resolve_rf_errors's global-override
# path: only the mode that actually reads a given mechanism uses it.
_RF_FIELD_NAMES = {"PHASE": "phase_rad", "VOLT": "voltage_V", "FREQ": "frequency_Hz"}
_DISTRIBUTIONS = {"gaussian", "uniform", "plus_or_minus"}


@dataclass(frozen=True)
class ErrorSpec:
    """One error rule -- elegant's own ``&error_element`` fields this
    module actually implements (``bind``/``sampled`` are explicit Phase-2
    non-goals, see ``plans/NATIVE_ERRORS_PLAN.md``).

    ``name_pattern``: ``fnmatch``-style wildcard (``"BD"``, ``"B?"``,
    ``"Q*"``) matched case-insensitively against ``ElementSpec.name``.
    ``element_type``: optional additional filter against ``ElementSpec.kind``
    (``"quad"``/``"sext"``/``"bend"``/``"drift"``) -- ``None`` matches any.
    ``parameter``: one of ``"K1"``, ``"K2"``, ``"ANGLE"``, ``"DX"``, ``"DY"``,
    ``"TILT"`` (magnetic/alignment -- see ``apply_errors``) or ``"PHASE"``,
    ``"VOLT"``, ``"FREQ"`` (RF -- see ``resolve_rf_errors``).
    ``amplitude``: absolute units (m for DX/DY, rad for ANGLE/TILT/PHASE, V
    for VOLT, Hz for FREQ, elegant's own K1/K2 units) if ``fractional=False``;
    a dimensionless fraction of the element's own current value if
    ``fractional=True``.
    ``distribution``: ``"gaussian"`` (clipped at +/-``cutoff`` sigma,
    resampled if outside -- elegant's own ``gauss_rn_lim``), ``"uniform"``
    (``[-amplitude, +amplitude]``), or ``"plus_or_minus"`` (exactly
    +/-``amplitude``, coin-flip sign).
    """

    name_pattern: str
    parameter: str
    amplitude: float
    distribution: str = "gaussian"
    cutoff: float = 3.0
    fractional: bool = False
    element_type: str | None = None

    def __post_init__(self) -> None:
        if self.parameter not in _FIELD_NAMES and self.parameter not in _RF_PARAMS:
            raise ValueError(f"unknown error parameter {self.parameter!r} (expected one of "
                              f"{sorted(_FIELD_NAMES) + sorted(_RF_PARAMS)})")
        if self.distribution not in _DISTRIBUTIONS:
            raise ValueError(f"unknown distribution {self.distribution!r} (expected one of {sorted(_DISTRIBUTIONS)})")


@dataclass(frozen=True)
class SampledError:
    """One resolved error draw -- ``delta`` is the actual applied change,
    already resolved from ``ErrorSpec``'s absolute/fractional distinction
    (same convention as elegant's own ``error_log=`` audit SDDS
    ``ParameterError`` column -- see ``tracking.elegant.errors_io``).
    ``(name, occurrence)`` identifies the exact element instance
    (``ElementSpec.occurrence``, matching elegant's own ``ElementOccurence``
    -- e.g. ``br14_simple.lte``'s 38 separate ``BD``s are distinguishable)."""

    name: str
    occurrence: int
    parameter: str
    delta: float


def _draw_unit(distribution: str, cutoff: float, generator: torch.Generator | None) -> float:
    """Raw distribution sample, in units of ``amplitude`` (i.e. not yet
    scaled by it) -- gaussian clipped at +/-``cutoff`` (elegant's own
    ``gauss_rn_lim``: draw, reject and redraw if ``|v| > cutoff``), uniform
    in ``[-1, 1]``, or exactly +/-1 for ``plus_or_minus``."""
    device = generator.device if generator is not None else "cpu"
    if distribution == "gaussian":
        while True:
            v = torch.randn((), generator=generator, device=device, dtype=torch.float64).item()
            if abs(v) <= cutoff:
                return v
    if distribution == "uniform":
        return torch.rand((), generator=generator, device=device, dtype=torch.float64).item() * 2.0 - 1.0
    # "plus_or_minus", enforced by ErrorSpec.__post_init__
    return 1.0 if torch.rand((), generator=generator, device=device, dtype=torch.float64).item() < 0.5 else -1.0


def sample_errors(
    specs: list[ElementSpec], error_specs: list[ErrorSpec], generator: torch.Generator | None = None,
) -> list[SampledError]:
    """Draw one random error per ``(ErrorSpec, matching element)`` pair --
    call once per realization (a whole ramp, or one FEL shot), using the
    tracker's own shared ``generator`` (the exact convention
    ``NativeTracker._draw_noise`` already establishes), outside any
    compiled region. Matching is case-insensitive
    ``fnmatch``/``ElementSpec.kind`` -- mirrors elegant's own wildcard
    element-name matching."""
    out: list[SampledError] = []
    for es in error_specs:
        for spec in specs:
            if es.element_type is not None and spec.kind != es.element_type:
                continue
            if not fnmatch.fnmatchcase(spec.name.upper(), es.name_pattern.upper()):
                continue
            draw = _draw_unit(es.distribution, es.cutoff, generator)
            if es.fractional:
                current = getattr(spec, _FIELD_NAMES[es.parameter]) if es.parameter in _FIELD_NAMES else 0.0
                delta = current * es.amplitude * draw
            else:
                delta = es.amplitude * draw
            out.append(SampledError(spec.name, spec.occurrence, es.parameter, delta))
    return out


def apply_errors(specs: list[ElementSpec], sampled: list[SampledError]) -> list[ElementSpec]:
    """Returns a **new** ``list[ElementSpec]`` with every magnetic/alignment
    ``SampledError`` (``K1``/``K2``/``ANGLE``/``DX``/``DY``/``TILT``) added
    onto the matching ``(name, occurrence)`` element's own field -- multiple
    errors on the same ``(element, parameter)`` sum. Elements with no
    matching error pass through unchanged (same object, not a copy).

    RF errors (``PHASE``/``VOLT``/``FREQ``) **also** resolve here now, onto
    the matching ``kind="rf"`` element's own ``phase_rad``/``voltage_V``/
    ``frequency_Hz`` field (same summing convention) -- this is what a
    per-cavity RF error actually needs for ``NativeTracker(rf=
    "per_element")``/``tracking.native.linac.LinacTracker`` (each real
    cavity has its *own* phase/voltage, unlike the once-per-turn lumped
    model). This coexists with, does not replace, ``resolve_rf_errors``'s
    global-override path -- that path is still what the *lumped* model
    needs (an RF-kind ``ElementSpec``'s own fields are never read there,
    folded into a plain drift instead), so both mechanisms run off the
    same ``sampled`` list and each is read by exactly one tracking mode.
    A RF-parameter error matching a non-``kind="rf"`` element (a
    misconfigured ``ErrorSpec.element_type``, most likely) is silently
    dropped here rather than writing a meaningless field -- the matching
    magnetic/alignment error, if also present on that same element, still
    applies."""
    by_key: dict[tuple[str, int], dict[str, float]] = {}
    for s in sampled:
        if s.parameter in _FIELD_NAMES:
            field = _FIELD_NAMES[s.parameter]
        elif s.parameter in _RF_FIELD_NAMES:
            field = _RF_FIELD_NAMES[s.parameter]
        else:
            continue
        key = (s.name.upper(), s.occurrence)
        by_key.setdefault(key, {})
        by_key[key][field] = by_key[key].get(field, 0.0) + s.delta

    rf_fields = set(_RF_FIELD_NAMES.values())
    out = []
    for spec in specs:
        deltas = by_key.get((spec.name.upper(), spec.occurrence))
        if not deltas:
            out.append(spec)
            continue
        updates = {
            field: getattr(spec, field) + delta for field, delta in deltas.items()
            if field not in rf_fields or spec.kind == "rf"
        }
        out.append(replace(spec, **updates) if updates else spec)
    return out


def resolve_rf_errors(sampled: list[SampledError]) -> dict[str, float]:
    """Sums ``PHASE``/``VOLT``/``FREQ`` ``SampledError``s into
    ``{"phi_s": rad, "Vrf_V": volts, "omega_rf": rad/s}`` -- ``FREQ``
    (elegant's own Hz convention) is converted to the angular-frequency
    units ``rf_kick``'s own ``omega_rf`` argument wants. Applied by
    ``NativeTracker`` as a **fixed** offset added to whatever ``Vrf_V``/
    ``phi_s``/``omega_rf`` it would otherwise use (nominal, or -- during a
    ramp -- that turn's own ramp-scaled values) for the whole realization,
    not re-resolved per turn."""
    out = {"phi_s": 0.0, "Vrf_V": 0.0, "omega_rf": 0.0}
    for s in sampled:
        if s.parameter == "PHASE":
            out["phi_s"] += s.delta
        elif s.parameter == "VOLT":
            out["Vrf_V"] += s.delta
        elif s.parameter == "FREQ":
            out["omega_rf"] += 2.0 * math.pi * s.delta
    return out


@dataclass
class FiducialReference:
    """The nominal reference established by ``NativeTracker.fiducialize()``
    -- see ``PHYSICS.md``'s "Errors and fiducialization" section. Pass to
    a *different*, error-perturbed ``NativeTracker(fiducial=...)`` so its
    RF phase is measured relative to this frozen reference instead of
    being (re-)derived from the perturbed beam's own timing -- elegant's
    own ``t_fiducial``/``phase_fiducial`` idea (``phase_reference.c``),
    generalized from "per RF cavity" to this tracker's single lumped RF
    kick.

    ``trajectory`` is the nominal per-element-instance ``(x,px,y,py,tau,
    delta)`` state, index ``i`` = the state *entering* element ``i``
    (``trajectory[0]`` is the fiducializing particle's own initial state,
    ``trajectory[-1]`` the ring exit) -- available directly for a
    downstream orbit-error diagnostic (subtract this from a perturbed
    run's own per-element trajectory), not just used internally.
    ``t_ref_s`` is this pass's own mean arrival time (seconds) -- the
    frozen timing reference.
    """

    trajectory: torch.Tensor
    t_ref_s: float
