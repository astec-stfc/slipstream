# `slipstream.tracking.native.linac`

[Source](../../../../src/slipstream/tracking/native/linac.py) · [Documentation index](../../../README.md)

## Physics

Tracks one canonical single pass through ordered elements, with inline RF and optional structure wakes, local LSC and CSR. By default the reference momentum follows the bunch centroid after elements. Ring equilibrium damping, excitation and IBS are not applied.

Opt-in [local dipole radiation](radiation.md) adds classical loss and Gaussian
ISR for aligned pure dipoles with a fixed reference (`always_change_p0=False`).

### Limits and conventions

- CSR requires CSR-enabled specs; `lsc=True` acts in finite-length elements.
- Fast collective grids and preceding-bend drift-tail approximations require convergence checks.
- Misaligned bends retain a lumped exit collective kick instead of subcycling.
- Changing a source reference energy must refresh energy-dependent bend offsets.

### `Model and numerical conventions`

Single-pass linac/FEL tracking. Distinct from `NativeTracker` (ring:
requires a `Lattice`, loops `n_turns`, applies ring-average radiation
damping/excitation and optional IBS every turn) because a single pass is a
genuinely different physics regime, not a special case of a ring with
`n_turns=1`. See the [native tracking guide](../../../tracking_native.md).

- No periodicity -- one pass through an ordered element list, no turns.
- The reference energy continuously tracks the beam's own centroid
  (`always_change_p0`, elegant's own run-level convention): after every
  non-diagnostic element, not just RF cavities with `CHANGE_P0=1`. This
  generalizes the per-cavity `change_p0` this module already has
  (`elements._rf_cavity_kick`) via the shared
  `elements._match_energy_to_centroid`.
- No equilibrium radiation damping/excitation and no IBS -- both are
  ring-average, asymptotic-equilibrium concepts with no meaning over one
  pass, and forcing a ring-only `Lattice` onto a linac section is a real
  correctness hazard (`Vrf_V=0` makes `Lattice.phi_s` raise
  `ZeroDivisionError`). `LinacTracker` needs no `Lattice` at all.
- Local dipole ISR is available within the limits in [radiation](radiation.md).
  Transverse deflecting cavities remain unsupported.

Section "joining" (CLARA's own multi-`.lte`, bunch-dump-chained
convention) needs no new mechanism: track through one section's `specs`,
then feed its output beam + reference energy as the next section's
injection beam + energy (``beam2, E2 = tracker2.track(*tracker1.track
(beam1))`), or concatenate multiple `parse_elegant_lattice(...)``
results into one `specs` list for a single combined run.

### `_apply_collective_kick`

LSC (if `lsc`, over this element's own `length_m`) and/or CSR
(if `csr`, at bending radius `csr_rho_m` -- caller already resolves
this from either the element's own `ElementSpec.csr` bend or an
inherited `csr_drift_rho_m` carry, see `_maybe_collective_kick`)
applied to this element's `delta` column -- see `collective.py` for
both kernels' derivation. Applied once per element, using that
element's own total length (not sub-cycled at Yoshida-sub-slice
granularity

`csr_drift_start_m` (`None` for a real CSR bend, using its own
static `csr_length_m` unchanged) marks a `CSRDRIFT`-carried drift:
its CSR-active length is capped live, ``clamp(overtaking_length_m -
csr_drift_start_m, 0, csr_length_m)` where `overtaking_length_m =
(24*sigma_z*csr_rho_m^2)^(1/3)` (elegant's own `csbend.c``
`CSRDRIFT_OVERTAKINGLENGTH` formula) uses the beam's *live*
`particles6`-own bunch length -- real elegant's steady-state
treatment shielding-cuts off once the bend's own radiation has slipped
past the bunch. `torch.clamp`, not a Python `if`, over this live
value -- see `lsc_delta_kick`'s own "Clamped, not branched" note for
why (`torch.compile` graph-breaks on a Python branch over a live
tensor).

`fast_collective_effects` (default `False`, zero behavior change)
swaps in the fixed-bin-grid kernels

### `_maybe_collective_kick`

Skip the collective-kick computation entirely when neither is
active for this element -- `lsc`/`csr` are fixed per-instance
Python bools and `p.csr`/`p.csr_drift_rho_m` fixed per-element ones
(all resolved at `torch.compile` trace time, same as
`always_change_p0`). A CSR-flagged bend uses its own `length`/
`angle` for `rho_m` (no live cutoff, `csr_drift_start_m=None`); a
drift carrying a `csr_drift_rho_m` (a real `CSRDRIFT` tail, see
`ElementSpec`'s own docstring) uses that inherited radius with a live
overtaking-length cutoff instead.

### `_track_dipole_subcycled`

A real dipole (`p.angle != 0`), transported and CSR/LSC-kicked in
`p.n_kicks` equal slices instead of one lumped end-of-element kick --
mirrors elegant's own `CSRCSBEND` (``T1[iBin] *= CSRConstant *
csbend->length / csbend->nSlices``, recomputed against the live bunch
at *each* slice, same `nSlices` the geometric integration already
uses). `_apply_collective_kick`'s own single-lumped-kick default
freezes the line-density estimate at the bend's *exit* even though a
chicane can compress the bunch by an order of magnitude across one
bend -- this reuses the live, evolving distribution instead.

Bit-identical geometric transport to the un-subcycled path with
collective effects off: `track_element`'s own `n_kicks` loop is
`for _ in range(n_kicks): drift(ds*df); kick(ds*kf)` with
`ds=length/n_kicks` -- calling it `n` times at
`length/n, angle/n, n_kicks=1` reproduces that *exact* same `ds`/
`h=angle/length` sequence (edges/misalignment untouched, gated to
only the first/last sub-call), so `p.offset` (the whole-element
reference-particle discretization residual, computed once in
`_build_params` at `nk=n_kicks`) is still the right correction to
subtract at the end -- no separate per-slice offset needed.

Only entered for `p.dx == p.dy == p.tilt == 0` (see
`_track_one_element`'s own gating) -- a misaligned dipole falls back
to the single-lumped-kick path instead of this function, since
`track_element`'s misalign transform isn't independently gateable
per side the way `edge{1,2}_effects` already is. A real, narrow,
documented gap, not a silently wrong answer.

### `_track_one_element`

One element's worth of work -- transport, then (if active) the
collective kick, then (if `always_change_p0`) recentering. This is
the single unit `LinacTracker(compile=True)` compiles for a
magnetic/drift element (`self._step_fn`), **not** the whole per-pass
loop: compiling the whole 182-element real-CLARA pass as one graph
OOM-killed WSL/CUDA after ~15 minutes even with LSC/CSR off (

**Never compiled for an RF-kind element** (`LinacTracker.__init__`
always passes the plain, uncompiled version as `rf_step_fn`): a real
CLARA cavity's own `N_KICKS` (e.g. L02's real 360) would let dynamo
unroll `elements._rf_cavity_kick`'s `for ik in range(n_kicks):`
loop into one enormous trace, minutes of pure CPU-side Inductor
codegen. Instead `compile_rf_sub_kicks` compiles **one RF sub-kick**
(`elements._rf_one_sub_kick`) reused across every `ik` and cavity,
while this outer function and the `ik` loop stay eager -- "compile
the small reused inner unit, not the loop around it", recursed one
level down from this function's own existence. See docs/tracking_native.md
for the compile-time/runtime numbers this bought.

`compile_collective_chunks` (default `False`) is the same
principle recursed into `collective.py`'s own LSC/CSR row-chunk
loops. Unlike `compile_rf_sub_kicks`, this applies regardless of
element kind (LSC/CSR act on RF elements too).

`fast_collective_effects` (default `False`) swaps in the
fixed-bin-grid LSC/CSR kernels -- see `_apply_collective_kick`.
Bound into `self._step_fn`/`self._rf_step_fn` via the same
`functools.partial` as `compile_collective_chunks`, not re-passed
on every `.track()` call.

### `_track_single_pass`

One pass through every element -- a plain, always-eager Python loop
calling `step_fn` for a magnetic/drift element or `rf_step_fn` for
an RF-kind one (see `_track_one_element` for why compilation happens
there, per element, not by wrapping this whole loop).
`charge_per_particle_C` is only read by a wake-carrying element (a
no-op for a beamline with no wakes). `lsc`/`csr` (default
`False`) apply the two `collective.py` kernels once per element.

### `LinacTracker._set_reference_energy`

Update entrance energy and cached magnet reference offsets together.

Source handoffs may change energy after construction. Never mutate
`reference_offset`'s shared cached CPU tensor; each refreshed offset
is an owned tensor with the original shape, dtype and device.

### `LinacTracker._track_design_reference`

One pass against a *fixed* reference-energy profile (`(len(specs) + 1,)`, entering
every element, e.g. the `energy_trajectory` of a nominal `track(return_trajectory=True)`)
instead of `always_change_p0`'s beam-mean reference: magnets keep their design strengths and a
beam whose energy errs carries that error in `delta` (dispersion, R56 arrival-time and
path-length effects are then real). What a jitter/error study needs -- `always_change_p0`
would re-scale every magnet to the jittered beam and absorb the error. `tau` is then the
arrival time relative to the *design* particle. Eager (no `torch.compile` variant of this loop).

## Usage

Construct `LinacTracker(specs, energy_eV, ..., bunch_charge_C=...)` and call `track(particles6, return_trajectory=...)`. Set `always_change_p0`, collective toggles, integration slices/order, dtype/device and compilation options explicitly as needed; `resolve_wakes` loads attached paths.

### API reference

```python
def resolve_wakes(specs: list[ElementSpec], base_dir: str | Path='.') -> list[ElementSpec]: ...
class LinacTracker:
    ...
    def __init__(self, specs: list[ElementSpec], energy_eV: float, mc2_eV: float=ELECTRON_MASS_EV, device: str | torch.device | None=None, dtype: torch.dtype=torch.float64, n_kicks: int=1, n_kicks_bend: int=4, integration_order: int=4, compile: bool=False, compile_mode: str | None=None, always_change_p0: bool=True, errors: list[SampledError] | None=None, fiducial: FiducialReference | None=None, bunch_charge_C: float=0.0, lsc: bool=False, csr: bool=False, pipe_radius_b_m: float | None=None, lsc_radius_factor: float=1.7, csr_softening_m: float | None=None, compile_collective_chunks: bool | None=None, fast_collective_effects: bool=False, dipole_collective_subcycle: bool=False, csr_transient: bool=False, csr_drift_model: str='overtaking', generator: torch.Generator | None=None): ...
    def track(self, particles6: torch.Tensor, return_trajectory: bool=False, reference_energy_eV: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...
    def fiducialize(self, particles0: torch.Tensor) -> FiducialReference: ...
```

### Wake conventions

Wake tables are per cell: `ElementSpec.cell_length` (`CELL_LENGTH`) sets how much of the table each sub-kick applies
(`length / cell_length / n_kicks`, see `elements.md`). `LinacTracker.track` gives any `TransverseWake` that has no fixed bin window one from the
time span of the first bunch tracked (plus half that span each side); set `set_fixed_bin_window` yourself for a bunch that changes length a lot.

### `resolve_wakes`

Returns a **new** `list[ElementSpec]` with every `kind="rf"`
element's wake *metadata* (plain strings, see `ElementSpec`'s own
docstring) resolved into real, loaded `wake`/`transverse_wake_x`/
`transverse_wake_y` objects (`ShortRangeWake`/`TransverseWake`,
via their own `from_sdds_file`) -- a one-shot construction-time
transform, mirroring `errors.apply_errors`. `base_dir` resolves a
relative `WAKEFILE=` path (relative to the `.lte`'s own
directory). An element with no `wakefile` set passes through
unchanged. A missing `wake_w{x,y,z}_column` simply skips that plane.

### `LinacTracker`

Tracks a plain `(N, 6)` particle tensor (columns `x, px, y, py, tau, delta` -- see the element model description) through **one
pass** of `specs` (`lattice_import.parse_elegant_lattice`'s
output, in beamline order). No `Lattice`, no turns -- see this
module's own docstring for why a linac single pass is a genuinely
different regime from `NativeTracker(rf="per_element")` at
`n_turns=1`.

Every `kind="rf"` element in `specs` (elegant's `RFCA`/`RFCW`
only) gets its own inline kick at its actual `s`-position
(`elements._rf_cavity_kick`, reused from the ring tracker's own
`rf="per_element"` machinery).

`always_change_p0` (default `True`, matching real CLARA's own
`&run_setup` convention) reproduces elegant's run-level
`always_change_p0`: after every element (not just RF cavities), the
reference momentum recenters on the beam's own current mean momentum
(`elements._match_energy_to_centroid`). When `True`, every RF
element's own `ElementSpec.change_p0` is overridden to `False`
(redundant otherwise, since the always-mode already recenters after
that same element). `always_change_p0=False` instead honors each RF
element's own real `CHANGE_P0` value, identical to
`NativeTracker(rf="per_element")`'s per-cavity model without turns.

`n_kicks`/`n_kicks_bend`/`integration_order`/`compile`/
`compile_mode` mirror `NativeTracker`'s own constructor parameters.

`errors`/`fiducial` reuse `tracking.native.errors` directly --
`errors.apply_errors` resolves `PHASE`/`VOLT`/`FREQ` errors
onto a matching `kind="rf"` element's own field (per-cavity RF
jitter studies), and `fiducialize()` below mirrors
`NativeTracker.fiducialize()`, generalized to this tracker's own
`always_change_p0` setting. `fiducial=` (from a different,
unperturbed sibling `LinacTracker`'s `fiducialize()`) freezes the
RF phase reference this tracker measures against, generalized from
"one lumped phase" to **per cavity**: every `kind="rf"` element's
own `phase_rad` gets an additional `-omega_rf*t_ref_s` term (that
element's own `omega_rf`) before `_build_params` sees it.

Per-bend incoherent radiation is opt-in for the [bounded pure-dipole model](radiation.md). CSR and the supported RF
focusing branches are optional local element effects.

`bunch_charge_C` (default `0.0`) is this pass's single-bunch
charge -- not `NativeTracker`'s own circulating `bunch_current_A`,
since there's no "current" for one pass.
`charge_per_particle_C = bunch_charge_C / n_particles` is computed
lazily in `.track()`, as a new 0-d tensor on every call: autograd saves it,
so refilling one shared tensor in place would invalidate the backward of an
earlier pass (two charged passes summed into one loss). `_set_reference_energy`
likewise replaces the stored reference-energy tensor. Every wake-carrying
element's wake object gets `.prepare(dtype, device)` called once here.

`lsc`/`csr` (both default `False`) turn on the two
`collective.py` kernels. `csr=True` only affects elements with
`ElementSpec.csr=True` (`CSRCSBEND`/`CSRCSBEN`); `lsc=True`
applies to every element with nonzero length, matching real elegant's
"LSC acts continuously along the beamline" physics.
`pipe_radius_b_m`/`lsc_radius_factor`/`csr_softening_m` mirror
`collective.lsc_delta_kick`/`collective.csr_steady_state_delta_kick`'s
own parameters.

`compile_collective_chunks` (default `None`) enables the compiled
exact LSC/CSR chunk bodies on CUDA even when outer transport compilation
is disabled; on CPU it follows `compile`. Pass `True` or `False`
explicitly to override that default. This keeps the pairwise kernels'
particle-count/dtype specialization independent from per-element transport
specializations.

`fast_collective_effects` (default `False`) swaps `lsc`/`csr`
to the fixed-bin-grid approximations

`csr_transient=True` selects the binned entrance-transient bend model;
`csr_drift_model="overtaking" | "stupakov" | "retarded"` selects downstream
drift treatment. Transients and non-default drift models require `csr=True`
and `fast_collective_effects=True`. See [model settings](../../../model_settings.md)
for a selection table and runnable comparisons. Construct a new tracker to
change these choices; element parameters retain their construction-time settings.

`dipole_collective_subcycle` (default `False`) applies the LSC/CSR
kick `n_kicks_bend` times per real dipole (at `length/n_kicks_bend`
each, against the live bunch at that point) instead of once, lumped,
over the whole magnet -- see `_track_dipole_subcycled`. Mirrors real
elegant's own `CSRCSBEND` (same `nSlices` for both the geometric
integration and the CSR wake). A no-op for a misaligned dipole
(`DX`/`DY`/`TILT` != 0), which still uses the lumped kick.

### `LinacTracker.track`

One pass. Returns `(particles6_out, energy_eV_out)` --
`energy_eV_out` is a 0-d tensor, the beamline's own exit
reference energy.

`return_trajectory=True` additionally returns a stacked
`(len(specs)+1, N, 6)` particle-state tensor **and** a stacked
`(len(specs)+1,)` reference-energy tensor, both for the state
entering every element (index 0 is the injection state/energy,
index -1 the beamline exit) -- a stand-in for CLARA-style
`WATCH` diagnostic points, and the direct input to
`linac_twiss.twiss_from_trajectory`. Reuses `self._step_fn`
(the same per-element step `return_trajectory=False` calls) for
each element's own physics -- only the trajectory bookkeeping (a
growing Python list + a final `torch.stack`) stays uncompiled.

### `LinacTracker.fiducialize`

Establish the nominal reference this tracker's current
beamline/RF produces, for a different, error-perturbed
`LinacTracker` to freeze against (pass as its own `fiducial=`)
-- mirrors `NativeTracker.fiducialize()` (

## Numerical and implementation notes

### `LinacTracker`

Compile the PER-ELEMENT step, not the whole per-pass loop -- see
_track_one_element's docstring. self._step_fn is also reused
directly by the return_trajectory=True path and fiducialize()
below, so both get the same per-element compiled speedup.

### `LinacTracker`

Bounded by the number of distinct element configurations in
specs (a loose over-estimate: real lattices repeat magnet
types) -- past dynamo's default recompile_limit, further
distinct elements silently fall back to eager.

### `LinacTracker`

collective_chunks (independent of `compile`) and
fast_collective_effects bound via partial before wrapping in
torch.compile -- both fixed per-instance, never re-passed per.track() call.

### `LinacTracker`

RF-kind elements: this OUTER function is never compiled (avoids
dynamo unrolling elements._rf_cavity_kick's own N_KICKS loop),
but when compile=True the INNER per-sub-kick body still gets
compiled, reused across every ik and cavity.

### `LinacTracker`

0-d tensors, not plain floats -- repeated.track() calls (e.g. a
Monte Carlo jitter study) must not force a recompile per
distinct injection energy.
