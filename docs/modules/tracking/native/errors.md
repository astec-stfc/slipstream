# `slipstream.tracking.native.errors`

[Source](../../../../src/slipstream/tracking/native/errors.py) · [Documentation index](../../../README.md)

## Physics

Samples one static error per matched element instance. Absolute and fractional perturbations modify transport parameters; RF errors resolve differently for lumped ring RF and per-cavity models. Fiducial references preserve the unperturbed RF timing.

### Limits and conventions

- Error realization is construction-time, not fresh turn-by-turn noise.
- Occurrence-aware matching matters for repeated element names.
- Derived synchronous phase must be overridden explicitly rather than assigned as a stored lattice field.

### `Model and numerical conventions`

An ErrorSpec matches element names/types and draws one value per matched
instance. Absolute changes are additive; fractional changes scale the
element's design value. Sampling occurs once per realization.

apply_errors returns new element specifications with magnetic field and
alignment changes applied. It does not mutate the original list. RF phase,
voltage and frequency errors have separate resolution paths for lumped ring
RF and inline cavities.

### `_draw_unit`

Raw distribution sample, in units of `amplitude` (i.e. not yet
scaled by it) -- gaussian clipped at +/-`cutoff` (elegant's own
`gauss_rn_lim`: draw, reject and redraw if `|v| > cutoff`), uniform
in `[-1, 1]`, or exactly +/-1 for `plus_or_minus`.

## Usage

Create `ErrorSpec` rules, call `sample_errors` once per realization, and pass samples to a tracker or `apply_errors`. `resolve_rf_errors` supplies lumped RF overrides; `FiducialReference` stores calibration.

### API reference

```python
class ErrorSpec:
    ...
class SampledError:
    ...
def sample_errors(specs: list[ElementSpec], error_specs: list[ErrorSpec], generator: torch.Generator | None=None) -> list[SampledError]: ...
def apply_errors(specs: list[ElementSpec], sampled: list[SampledError]) -> list[ElementSpec]: ...
def resolve_rf_errors(sampled: list[SampledError]) -> dict[str, float]: ...
class FiducialReference:
    ...
```

### `ErrorSpec`

An error rule representing supported `&error_element` fields.
Cross-rule `bind` and `sampled` behavior is not implemented.

`name_pattern`: `fnmatch`-style wildcard (`"BD"`, `"B?"`,
`"Q*"`) matched case-insensitively against `ElementSpec.name`.
`element_type`: optional additional filter against `ElementSpec.kind`
(`"quad"`/`"sext"`/`"bend"`/`"drift"`) -- `None` matches any.
`parameter`: one of `"K1"`, `"K2"`, `"ANGLE"`, `"DX"`, `"DY"`,
`"TILT"` (magnetic/alignment -- see `apply_errors`) or `"PHASE"`,
`"VOLT"`, `"FREQ"` (RF -- see `resolve_rf_errors`).
`amplitude`: absolute units (m for DX/DY, rad for ANGLE/TILT/PHASE, V
for VOLT, Hz for FREQ, elegant's own K1/K2 units) if `fractional=False`;
a dimensionless fraction of the element's own current value if
`fractional=True`.
`distribution`: `"gaussian"` (clipped at +/-`cutoff` sigma,
resampled if outside -- elegant's own `gauss_rn_lim`), `"uniform"`
(`[-amplitude, +amplitude]`), or `"plus_or_minus"` (exactly
+/-`amplitude`, coin-flip sign).

### `SampledError`

One resolved error draw -- `delta` is the actual applied change,
already resolved from `ErrorSpec`'s absolute/fractional distinction
(the same convention as elegant's `ParameterError` audit column).
`(name, occurrence)` identifies the exact element instance
(`ElementSpec.occurrence`, matching elegant's own `ElementOccurence`
-- e.g. `br14_simple.lte`'s 38 separate `BD`s are distinguishable).

### `sample_errors`

Draw one random error per `(ErrorSpec, matching element)` pair --
call once per realization (a whole ramp, or one FEL shot), using the
tracker's own shared `generator` (the exact convention
`NativeTracker._draw_noise` already establishes), outside any
compiled region. Matching is case-insensitive
`fnmatch`/`ElementSpec.kind` -- mirrors elegant's own wildcard
element-name matching.

### `apply_errors`

Returns a **new** `list[ElementSpec]` with every magnetic/alignment
`SampledError` (`K1`/`K2`/`ANGLE`/`DX`/`DY`/`TILT`) added
onto the matching `(name, occurrence)` element's own field -- multiple
errors on the same `(element, parameter)` sum. Elements with no
matching error pass through unchanged (same object, not a copy).

RF errors (`PHASE`/`VOLT`/`FREQ`) **also** resolve here now, onto
the matching `kind="rf"` element's own `phase_rad`/`voltage_V`/
`frequency_Hz` field (same summing convention) -- this is what a
per-cavity RF error actually needs for ``NativeTracker(rf=
"per_element")`/`tracking.native.linac.LinacTracker`` (each real
cavity has its *own* phase/voltage, unlike the once-per-turn lumped
model). This coexists with, does not replace, `resolve_rf_errors`'s
global-override path -- that path is still what the *lumped* model
needs (an RF-kind `ElementSpec`'s own fields are never read there,
folded into a plain drift instead), so both mechanisms run off the
same `sampled` list and each is read by exactly one tracking mode.
A RF-parameter error matching a non-`kind="rf"` element (a
misconfigured `ErrorSpec.element_type`, most likely) is silently
dropped here rather than writing a meaningless field -- the matching
magnetic/alignment error, if also present on that same element, still
applies.

### `resolve_rf_errors`

Sums `PHASE`/`VOLT`/`FREQ` `SampledError`s into
`{"phi_s": rad, "Vrf_V": volts, "omega_rf": rad/s}` -- `FREQ`
(elegant's own Hz convention) is converted to the angular-frequency
units `rf_kick`'s own `omega_rf` argument wants. Applied by
`NativeTracker` as a **fixed** offset added to whatever `Vrf_V`/
`phi_s`/`omega_rf` it would otherwise use (nominal, or -- during a
ramp -- that turn's own ramp-scaled values) for the whole realization,
not re-resolved per turn.

### `FiducialReference`

The nominal reference established by `NativeTracker.fiducialize()`
-- see `PHYSICS.md`'s "Errors and fiducialization" section. Pass to
a *different*, error-perturbed `NativeTracker(fiducial=...)` so its
RF phase is measured relative to this frozen reference instead of
being (re-)derived from the perturbed beam's own timing -- elegant's
own `t_fiducial`/`phase_fiducial` idea (`phase_reference.c`),
generalized from "per RF cavity" to this tracker's single lumped RF
kick.

`trajectory` is the nominal per-element-instance ``(x,px,y,py,tau,
delta)` state, index `i` = the state *entering* element `i``
(`trajectory[0]` is the fiducializing particle's own initial state,
`trajectory[-1]` the ring exit) -- available directly for a
downstream orbit-error diagnostic (subtract this from a perturbed
run's own per-element trajectory), not just used internally.
`t_ref_s` is this pass's own mean arrival time (seconds) -- the
frozen timing reference.

## Numerical and implementation notes

### `Module constants`

ElementSpec field each RF error parameter resolves onto, for a
kind="rf" element -- see apply_errors's own docstring for why this
coexists with (does not replace) resolve_rf_errors's global-override
path: only the mode that actually reads a given mechanism uses it.
