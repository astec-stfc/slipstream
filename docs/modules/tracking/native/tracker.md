# `slipstream.tracking.native.tracker`

[Source](../../../../src/slipstream/tracking/native/tracker.py) · [Documentation index](../../../README.md)

## Physics

Integrates actual per-element canonical ring transport on every turn and composes native RF, radiation damping/excitation, wakes, optional IBS and ramps. Lumped RF and per-element cavities have distinct placement/timing behavior.

### Limits and conventions

- State is `(N,6)` with canonical momenta and tau in metres; RF helpers receive converted seconds.
- Native physics helpers provide correctness independently of external executables or development backends.
- Compiled specialization is sensitive to scalar values and shapes.

### `_ElementParams`

One element's plain-float transport parameters -- a fixed
Python-level property per element (never particle data), so looping
over this list and branching per-element inside
`elements.track_element` costs `torch.compile` nothing (resolved
at trace time).

`offset` is `elements.reference_offset`'s own (6,) tensor for this
element's exact params, subtracted from every particle's output in
`_apply_elements`: elegant's own default kick counts leave the
on-axis reference orbit at a real, nonzero fixed point for a bend with
real K1+K2, which compounds into a many-turn instability if
uncorrected.

`dx`/`dy`/`tilt` (misalignment) are deliberately **excluded**
from `offset`'s own computation (always computed at the design axis,
`dx=dy=tilt=0`, even with a real misalignment error present):
`reference_offset` cancels a *discretization* artifact of the ideal
element, not the orbit distortion a real DX/DY/TILT error is supposed
to produce. K1/K2/ANGLE field errors, by contrast, correctly *do* flow
into `offset` (a field error legitimately shifts the closed orbit).

### `_build_params`

`rf_mode="lumped"` (default): an `ElementSpec(kind="rf")` is
folded into a plain drift of the same length, its VOLT/PHASE/FREQ
ignored -- a drift-shaped `_ElementParams` is what every non-bend/
quad/sext `ElementSpec` already produces regardless of `kind`.
`rf_mode="per_element"` instead builds a real RF-kind
`_ElementParams` for every `kind="rf"` spec -- see
`NativeTracker`'s own `rf=` docstring.

`change_p0_override` (default `None`): when not `None`,
overrides every RF-kind element's own `ElementSpec.change_p0` --
used by `LinacTracker(always_change_p0=True)` to force `False` on
every cavity's own per-element recentering, since its per-element loop
already recenters after *every* element (avoiding a redundant double
`do_match_energy` call right after the cavity's own).

### `_apply_one_element`

Dispatch a single element -- RF-kind params go through
`elements._rf_cavity_kick` (which can change `energy_eV`);
everything else through `elements.track_element` (`energy_eV`
passed through unchanged). Shared by `_apply_elements` (the per-turn
hot path) and `fiducialize()`'s own per-element loop, so RF-kind
elements can't be mishandled by one call site diverging from the other.

`charge_per_particle_C` (default `0.0`, a no-op) is only read when
`p` carries a wake (an `RFCW`-equivalent).
`compile_rf_sub_kicks` (default `False`) is threaded through to
`elements._rf_cavity_kick`'s own parameter of the same name -- only
`LinacTracker` sets this.

### `_apply_elements`

Apply every element's transport in sequence -- a plain Python loop
over a fixed-length list, `torch.compile`-friendly (no object
construction, no `.item()` calls). `energy_eV`/`mc2_eV` are 0-d
tensors, not plain floats, since ramped tracking changes `energy_eV`
every turn (a Python float there would force a recompile every turn).
Returns the (possibly cavity-updated) exit `energy_eV` alongside the
transported particles.

### `_rf_and_damping`

`ring_physics.rf_kick` + short-range wake kick(s) +
`damping_excitation` on the delta column, and (if `optics` is
given) transverse dipole-wake kick + `transverse_damping_excitation`
on px/py. The ordering is RF, then wake kicks at the same tau/x/y,
then damping/excitation. `wake` (longitudinal) works with or
without `optics`; `transverse_wake_x`/`_y` need `optics`.

`skip_rf_kick` (fixed per tracker instance) skips the lumped
`rf_kick` call -- used by `NativeTracker(rf="per_element")`, where
every RF-kind element already applied its own kick inline during
`_apply_elements`, so the lumped once-per-turn kick would otherwise
double-count on top of the per-cavity ones.

### `_step`

One turn: transport via every element in sequence (`_apply_elements`,
which may update the reference energy if any element is an RF-kind
cavity with `change_p0=True`), then `_rf_and_damping` -- factored
out so it can be wrapped in `torch.compile` (`NativeTracker`'s
`compile=`). When `skip_rf_kick` (`rf="per_element"`), the
just-updated `energy_eV` is also used as the wake-normalization
reference (`E0_eff_eV`), since that's the tracker's actual current
reference at this point in the turn.

### `NativeTracker._recompute_ibs_rates`

Real per-element-averaged IBS growth rates
(`ibs.element_averaged_growth_rates`, fed this tracker's own
`periodic_twiss_table` -- a real accuracy improvement over the
single-averaged-point approximation native ring-physics IBS
integration uses) from the live beam at the reference (`s=0`)
point. Dispersion is subtracted from the raw `x`/`y` before
computing emittance -- the growth-rate formulas want the pure
betatron emittance, and this tracker's real per-element transport
genuinely disperses the beam.

## Usage

Construct `NativeTracker(specs, lattice, ..., optics=..., rf=...)`, then use `track` for sampled diagnostics or `advance` for state propagation. `fiducialize` establishes unperturbed timing; configure slices/order, device/dtype and optional errors explicitly.

### API reference

```python
class NativeTrackResult:
    ...
class NativeTracker:
    ...
    def __init__(self, specs: list[ElementSpec], params: Lattice, optics: TransverseOptics | None=None, ramp: RampProfile | None=None, generator: torch.Generator | None=None, mc2_eV: float | None=None, device: str | torch.device | None=None, dtype: torch.dtype=torch.float64, n_kicks: int=1, n_kicks_bend: int=4, integration_order: int=4, compile: bool=False, compile_mode: str | None=None, wake: ShortRangeWake | None=None, transverse_wake_x: TransverseWake | None=None, transverse_wake_y: TransverseWake | None=None, bunch_current_A: float=0.0, ibs: bool=False, ibs_recompute_every: int=50, ibs_coulomb_log: float | None=None, ibs_model: str='bane', ibs_particle_radius_m: float=R_E_M, errors: list[SampledError] | None=None, fiducial: FiducialReference | None=None, rf: str='lumped', exact_dipole: bool=False): ...
    def fiducialize(self, particles0: torch.Tensor) -> FiducialReference: ...
    def track(self, particles6: torch.Tensor, n_turns: int, sample_every: int=1, progress: bool=False) -> tuple[torch.Tensor, NativeTrackResult]: ...
    def advance(self, particles6: torch.Tensor, n_turns: int=1) -> torch.Tensor: ...
```

### `NativeTrackResult`

Per-sample diagnostics from `NativeTracker.track`.

### `NativeTracker`

Track canonical `(x,px,y,py,tau,delta)` particles through parsed elements for a requested number of turns. Supply a `Lattice` for ring-average RF/radiation and optional transverse optics, wakes, errors, IBS or ramp. `rf="per_element"` uses physical cavity locations; `rf="lumped"` uses the main cavity kick once per turn. Sampled diagnostics are returned with the final tensor. Tracker RNG and elapsed-turn state persist between calls. See the native tracking guide for configuration and the API reference for defaults.

### `NativeTracker.fiducialize`

Establish the nominal reference this tracker's current
lattice/RF produces, for a different, error-perturbed tracker to
freeze against (pass the result as that tracker's own
`fiducial=`

Tracks `particles0` through one pass via the same per-element
dispatch `_apply_elements` uses (no lumped RF/damping/wake,
matching elegant's own fiducialization pass -- a real per-cavity
kick under `rf="per_element"` is still applied, as part of the
beamline's own transport, not the once-per-turn lumped
mechanism), recording the state entering every element (index 0 is
`particles0` itself, index -1 the ring exit) and this pass's own
mean arrival time (`tau/C_LIGHT`, seconds) as the frozen timing
reference.

### `NativeTracker.advance`

Advance fixed-energy native turns without collecting host diagnostics.

Uses the normal transport, RF, damping, noise and elapsed-turn state.
No turn records or scalar diagnostic reductions are performed. Ramp
tracking must use `track` because its time-dependent reference and
result collection are handled together by the ramp implementation.

## Numerical and implementation notes

### `NativeTracker`

Track canonical `(x,px,y,py,tau,delta)` particles through parsed elements for a requested number of turns. Supply a `Lattice` for ring-average RF/radiation and optional transverse optics, wakes, errors, IBS or ramp. `rf="per_element"` uses physical cavity locations; `rf="lumped"` uses the main cavity kick once per turn. Sampled diagnostics are returned with the final tensor. Tracker RNG and elapsed-turn state persist between calls. See the native tracking guide for configuration and the API reference for defaults.

### `NativeTracker`

Track canonical `(x,px,y,py,tau,delta)` particles through parsed elements for a requested number of turns. Supply a `Lattice` for ring-average RF/radiation and optional transverse optics, wakes, errors, IBS or ramp. `rf="per_element"` uses physical cavity locations; `rf="lumped"` uses the main cavity kick once per turn. Sampled diagnostics are returned with the final tensor. Tracker RNG and elapsed-turn state persist between calls. See the native tracking guide for configuration and the API reference for defaults.

### `NativeTracker`

Track canonical `(x,px,y,py,tau,delta)` particles through parsed elements for a requested number of turns. Supply a `Lattice` for ring-average RF/radiation and optional transverse optics, wakes, errors, IBS or ramp. `rf="per_element"` uses physical cavity locations; `rf="lumped"` uses the main cavity kick once per turn. Sampled diagnostics are returned with the final tensor. Tracker RNG and elapsed-turn state persist between calls. See the native tracking guide for configuration and the API reference for defaults.

### `NativeTracker`

Track canonical `(x,px,y,py,tau,delta)` particles through parsed elements for a requested number of turns. Supply a `Lattice` for ring-average RF/radiation and optional transverse optics, wakes, errors, IBS or ramp. `rf="per_element"` uses physical cavity locations; `rf="lumped"` uses the main cavity kick once per turn. Sampled diagnostics are returned with the final tensor. Tracker RNG and elapsed-turn state persist between calls. See the native tracking guide for configuration and the API reference for defaults.

### `NativeTracker`

Track canonical `(x,px,y,py,tau,delta)` particles through parsed elements for a requested number of turns. Supply a `Lattice` for ring-average RF/radiation and optional transverse optics, wakes, errors, IBS or ramp. `rf="per_element"` uses physical cavity locations; `rf="lumped"` uses the main cavity kick once per turn. Sampled diagnostics are returned with the final tensor. Tracker RNG and elapsed-turn state persist between calls. See the native tracking guide for configuration and the API reference for defaults.

### `NativeTracker`

Track canonical `(x,px,y,py,tau,delta)` particles through parsed elements for a requested number of turns. Supply a `Lattice` for ring-average RF/radiation and optional transverse optics, wakes, errors, IBS or ramp. `rf="per_element"` uses physical cavity locations; `rf="lumped"` uses the main cavity kick once per turn. Sampled diagnostics are returned with the final tensor. Tracker RNG and elapsed-turn state persist between calls. See the native tracking guide for configuration and the API reference for defaults.

### `NativeTracker`

Track canonical `(x,px,y,py,tau,delta)` particles through parsed elements for a requested number of turns. Supply a `Lattice` for ring-average RF/radiation and optional transverse optics, wakes, errors, IBS or ramp. `rf="per_element"` uses physical cavity locations; `rf="lumped"` uses the main cavity kick once per turn. Sampled diagnostics are returned with the final tensor. Tracker RNG and elapsed-turn state persist between calls. See the native tracking guide for configuration and the API reference for defaults.

### `NativeTracker`

Track canonical `(x,px,y,py,tau,delta)` particles through parsed elements for a requested number of turns. Supply a `Lattice` for ring-average RF/radiation and optional transverse optics, wakes, errors, IBS or ramp. `rf="per_element"` uses physical cavity locations; `rf="lumped"` uses the main cavity kick once per turn. Sampled diagnostics are returned with the final tensor. Tracker RNG and elapsed-turn state persist between calls. See the native tracking guide for configuration and the API reference for defaults.

### `NativeTracker`

Track canonical `(x,px,y,py,tau,delta)` particles through parsed elements for a requested number of turns. Supply a `Lattice` for ring-average RF/radiation and optional transverse optics, wakes, errors, IBS or ramp. `rf="per_element"` uses physical cavity locations; `rf="lumped"` uses the main cavity kick once per turn. Sampled diagnostics are returned with the final tensor. Tracker RNG and elapsed-turn state persist between calls. See the native tracking guide for configuration and the API reference for defaults.

### `NativeTracker`

Track canonical `(x,px,y,py,tau,delta)` particles through parsed elements for a requested number of turns. Supply a `Lattice` for ring-average RF/radiation and optional transverse optics, wakes, errors, IBS or ramp. `rf="per_element"` uses physical cavity locations; `rf="lumped"` uses the main cavity kick once per turn. Sampled diagnostics are returned with the final tensor. Tracker RNG and elapsed-turn state persist between calls. See the native tracking guide for configuration and the API reference for defaults.
