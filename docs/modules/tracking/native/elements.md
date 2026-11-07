# `slipstream.tracking.native.elements`

[Source](../../../../src/slipstream/tracking/native/elements.py) · [Documentation index](../../../README.md)

## Physics

Integrates canonical (x,px,y,py,tau,delta) transport using symplectic splitting, sector-bend drifts, normal/skew multipole and corrector/edge kicks, and RF cavities with supported focusing and optional structure wakes. Tau is flight-length surplus: positive means late/tail. reference_offset removes the numerical bend reference displacement.

### Limits and conventions

- Canonical transverse momenta are not slopes; use the conversion helpers at boundaries.
- Per-particle beta enters time of flight and RF phase.
- Preserve cancellation-resistant bend algebra and reference subtraction.
- Alternative edge models and higher curved-field terms are supported only where the actual implementation explicitly handles them.
- Distributed higher/skew multipoles require positive length; integrated_multipoles require zero length. Edge-effect flags support only 0 or 1.

### `Model and numerical conventions`

- `elegant/src/multipole.c`/`csbend.c`: each of the `n_kicks` slices
  is itself a 4th-order symplectic (Yoshida) splitting by default
  (`integration_order=4`, the `BETA=2**(1/3)` drift/kick fraction
  tables in the source). The real default `n_kicks` (`N_SLICES`) is not a
  uniform 4 -- KQUAD/KSEXT/KQUSE default to `N_SLICES=1` (4 kicks total
  from the splitting alone), CSBEND/CCBEND/CSRCSBEND default to
  `N_SLICES=4` (16 kicks total) -- see `tracker.py`'s `n_kicks`/
  `n_kicks_bend` constructor params.
- `csbend.c`'s drift sub-step is an exact closed-form sector-bend drift
  in canonical coordinates -- `_canonical_drift` below, `h=0`
  collapsing to a plain canonical straight-line drift, so one function
  covers DRIFT/KQUAD/KSEXT (`h=0`) and CSBEND (`h=angle/length != 0`).
- The kick step applies a curvature-path-length weight `(1 + h*x)` that
  a bend's K1/K2 content needs -- see `_multipole_kick` below. The
  pure-dipole term (`Fy=1` at the reference orbit) is carried by this
  same kick, not folded into the drift.
- `_edge_kick` implements slope-space linear and Brown second-order
  edge maps (`EDGE_EFFECTS=1`), including `HGAP`/`FINT` focusing.
- `reference_offset` (below): elegant's own default kick counts leave
  the on-axis reference orbit at a real, nonzero fixed point for a bend
  with real K1+K2 -- elegant itself avoids this via an explicit
  reference-trajectory subtraction; this reproduces the fix at this
  module's own per-element granularity.

### `_yoshida_extend`

Yoshida's recursive "triple jump" composition (H. Yoshida,
"Construction of higher order symplectic integrators", Phys. Lett. A
150 (1990) 262-268): given any symmetric drift/kick splitting of
accuracy `order`, composing three copies at different scales,
`S_new(h) = S(x1 h). S(x0 h). S(x1 h)` with ``x1 = 1/(2 -
2**(1/(order+1)))`, `x0 = 1 - 2*x1``, cancels the leading error term
and yields a splitting of accuracy `order+2`. This is exactly how
`_BETA`/`_DRIFT_FRAC4`/`_KICK_FRAC4` are constructed from the
trivial order-2 splitting -- verified by
`tests/test_native_tracker.py`'s
`test_yoshida_extend_reproduces_order4_table`.

Applying this once more to elegant's own order-6 table gives a genuine
8th-order splitting (`_DRIFT_FRAC8`/`_KICK_FRAC8` in the source) with no
external reference needed -- elegant itself has no order-8 option, so
this is a native-only extension, verified by an empirical
convergence-rate test (error ~ ds^8 under step refinement).

Adjacent drift fragments at the seams between the three composed
copies are summed into one (the standard optimization turning the
naive `3*len(drift_frac)` raw fragments into
`3*len(drift_frac)-2`).

### `_canonical_drift`

Return `(x,px,y,py,arc_length)` after a canonical drift over fixed reference length `ds` and curvature `h`. In a straight drift, `pz=sqrt((1+delta)**2-px**2-py**2)`, `x+=ds*px/pz`, `y+=ds*py/pz`, and `arc_length=ds*(1+delta)/pz`. In a sector bend, preserve the cancellation-resistant sine/cosine identities. Its auxiliary factor gives `arc_length=factor*(1+delta)`: using factor alone changes time of flight.

### `_exact_dipole_drift`

Exact map of a pure sector dipole whose field matches the reference
curvature `h` (Forest's exact sector bend, as in PTC/Bmad), over
reference length `ds` -- the `exact_dipole=True` alternative to
`_canonical_drift` + the dipole part of `_multipole_kick`. Same
Hamiltonian, different splitting (the kicks then carry only K1/K2).

Derived from the invariant centre of the particle's horizontal circle,
`C = rho*(1 + h*x - pz, px)` in the entrance frame, rotated by
`phi = h*ds`. Every quantity formed is O(deviation): `pz - 1` is
computed as `(delta*(2+delta) - px^2 - py^2)/(pz + 1)`, and an on-axis
on-momentum particle is an exact fixed point. With elegant's splitting
`px` instead swings by ~`h*ds` every sub-step (0.014 rad on
br14_simple vs a 1e-4 betatron signal) and float32 rounding of that
swing dominated the tracking noise

### `_multipole_kick_increments`

Return the dipole and normal/skew multipole increments. The base terms are `Fy=h+k1*x+k2*(x*x-y*y)/2`, `Fx=k1*y+k2*x*y`, with `dpx=-ds*(1+h*x)*Fy` and `dpy=ds*(1+h*x)*Fx`. Higher terms use `Re/Im[(Kn+i*Ks)*(x+i*y)**n/n!]`. With `dipole=False`, the exact sector-dipole map carries the dipole field. Distributed strengths require a positive element length; zero-length kicks use integrated strengths.

### `_corrector_kick`

Real elegant `HKICK`/`VKICK` transport (`bend_matrix.c`'s
`corrector_matrix`/zero-length branch, transcribed with `B2=0` and
no edge focusing -- this repo's real correctors set neither): a
genuine dipole kick of angle `kick` over path length `length`.

Written entirely in `sin(kick)` (never `1/kick`), so it stays
finite and differentiable exactly at `kick=0` -- LOCO's own nominal
operating point. This is deliberately **not** routed through
`track_element`'s own `angle`/`h=angle/length` path: that path
computes `math.cos(ds*h)`/`math.sin(ds*h)` in plain Python,
built for a fixed, never-differentiated lattice curvature (

`length<=0` (zero-length corrector): elegant's own distinct
zero-length matrix -- the leading term is a plain slope addition
(`kick`, not `sin(kick)`), elegant's own convention, confirmed
against a live `elegant` run (`tests/test_native_loco.py`).
Position is unchanged (no path to drift through).

`length>0`: exact reproduction of elegant's own `corrector_matrix`
at `B2=0` (every higher-order/edge term in that matrix carries a
`B2` factor and vanishes identically) -- not a small-angle
approximation, an exact closed form. `tau` gets the ordinary
(`h=0`) drift arc-length update; the `O(kick^2)` curvature
correction to arc length is dropped (real corrector kicks are
milliradian-scale, and this only feeds RF-phase bookkeeping, never a
LOCO-tapped BPM reading).

Elegant's own `R`/`C` matrix entries act on **slope** coordinates
(`x'`, `y'`, its universal TRANSPORT-style matrix-method
convention -- unlike the kick-method integrator `track_element` uses
for CSBEND/KQUAD/KSEXT, which is canonical throughout), so this
function converts to slopes (`_slopes_from_canonical`), applies the
matrix, and converts back (`_canonical_from_slopes`) -- confirmed
necessary (not a no-op) by a live-elegant check that fails by ~20% on
`xp`/`yp` without it, see `tests/test_native_loco.py`.

### `_edge_kick`

Elegant style slope-space edge map (EDGE_EFFECTS=1).

Includes finite-gap focusing and Brown second-order terms, including
zero-angle faces.

### `_misalign`

Enter (`entering=True`)/exit (`False`) misalignment coordinate
transform, transcribed from `multipole.c`'s call sequence around the
entire element body: ``offsetBeamCoordinatesForMisalignment(dx,dy,
dz=0)` then `rotateBeamCoordinatesForMisalignment(tilt)`` at entry;
the exact inverse, in reverse order, at exit. `dz` (longitudinal
placement error) is not modeled.

The rotation applies the same 2x2 rotation matrix to `(x,y)` and
`(px,py)` independently -- valid for canonical momenta here since a
rigid rotation of the transverse plane commutes with the
slope<->canonical `pz` normalization (depends only on
`px^2+py^2`, rotation-invariant).

**Bounded, documented approximation for CSBEND specifically**: real
elegant gives bends a separate, exact rigid-body misalignment
treatment accounting for the curved reference trajectory
(`malign_mat.c`, cited to M. Venturini, ALSU-AP-TN-2021-001) -- not
implemented here. This function's flat-space sandwich is applied
uniformly to every element type, exact for `h=0` but an
approximation for CSBEND, bounded the same way this module's other
CSBEND simplifications are (`h*x ~ 1e-4` for this repo's real
bends). Revisit if a lattice with much larger bend curvature or
particle amplitude needs the exact treatment.

`dx`/`dy`/`tilt` may each be a 0-d tensor (LOCO's own alignment
fit needs gradients exactly at the nominal zero-misalignment point) --
`_has_offset`/`_has_tilt` (below) gate on `isinstance` first, not
on value, so a differentiable knob is never silently detached by a
plain-float `!= 0.0` branch the way a Python `if` alone would (see
`elements._corrector_kick`'s own docstring for the same trap with
`track_element`'s `ANGLE`). `torch.cos`/`torch.sin` are used
whenever `tilt` is a tensor; `math.cos`/`math.sin` otherwise
(cheaper, and keeps every existing plain-float call site bit-identical).

### `_match_energy_to_centroid`

The new reference momentum is the
ensemble's own momentum-space mean, `P_avg = mean_i[P0*(1+delta_i)]`
-- not an energy-space mean, and not a `Vrf*sin(phase)`-derived
design value. ``dp=(P0-P_avg)/P_avg,
dr=P0/P_avg, delta_new=dp+delta*dr``.
This module's `px`/`py` are *canonical* (P0-
normalized) momenta, so they do need rescaling by `dr` to keep the
physical transverse momentum invariant when `P0` changes.

Extracted from `_rf_cavity_kick`'s own `change_p0=True` branch so
it can be called after any element, not only from inside an RF kick
-- see `LinacTracker`'s `always_change_p0`.

### `_rebase_reference`

Re-express `state6` (relative to a reference at `energy_eV`) relative to a reference at
`energy_new_eV`, keeping every particle's physical momentum: the same `delta`/`px`/`py`
rescaling as `_match_energy_to_centroid`, but to a *prescribed* reference instead of the
ensemble mean (`LinacTracker.track(reference_energy_eV=...)`, a fixed design energy profile).

### `_rf_one_sub_kick`

One RF sub-kick -- `_rf_cavity_kick`'s own inner ``for ik in
range(n_kicks)`` loop body, extracted verbatim so
`LinacTracker(compile=True)` can compile this small, reusable unit
instead of `torch.compile` unrolling the whole loop into one graph
(real CLARA cavities set a genuine physical-cell `N_KICKS` in the
hundreds -- see docs/tracking_native.md).

`phase_slip` is passed in pre-computed (``(omega_rf/C_LIGHT) *
(ik*length_k) * (1/beta0 - 1/beta_wave)``, from the caller's own
`ik`) rather than this function taking `ik`/`beta0`/
`beta_wave` and recomputing it: `ik` is a genuinely
per-iteration-varying Python `int`, and `torch.compile`
specializes (recompiles per distinct value) on a raw int argument by
default -- exactly the problem this split exists to avoid.
`phase_slip` as a plain `float` does not force a recompile in this
codebase's torch version (`specialize_float=False`).
`is_first`/`is_last` (computed by the caller) are the only
remaining structural choices -- at most 3 compiled variants
regardless of `n_kicks`.

### `_tdc_kick`

A thin transverse deflecting cavity (`ElementSpec(kind="tdc")`): `drift(length/2) -> kick -> drift(length/2)`, with `length` possibly zero.
With `a = V / (p0 c)` and `arg = phase_rad + omega_rf * tau / c`, the particle receives `dp = a * sin(arg)` along the deflection direction
`(cos(tilt), sin(tilt))` (`tilt = 0` is `x`, `pi/2` is `y`) and, from the Panofsky-Wenzel theorem, `d(delta) = -a * omega_rf / (beta0 * c) * q * cos(arg)`
with `q` the offset along that direction (a late particle is kicked towards `+q`; one at positive `q` loses energy where the kick is rising).
The pair is symplectic for `dx^dpx + dy^dpy - beta0 dtau^ddelta`, the time and energy form; the tests check this and that the opposite sign of the
energy term is not. Phase zero is the zero crossing of the deflection. Not modelled: transverse focusing and second-order terms of the mode,
the finite length of the kick, wakes and any change of the reference energy.

### `_rf_cavity_kick`

Real elegant's own RFCA transport (`simple_rfca.c`):
`drift(length/2) -> thin energy kick -> drift(length/2)`. The drift
halves reuse `_canonical_drift`'s `h=0` branch.

The local energy kick differs from the lumped ring helper: real per-cavity RFCA
gives every particle (including the one exactly on-crest) the full
`Vrf*sin(phase)` energy gain relative to the unchanged old
reference -- unlike `rf_kick`'s own convention (designed for the
once-per-turn lumped model), which subtracts a `sin(phi_s)`
baseline so the on-crest particle nets zero. 

Follows `simple_rfca.c`'s `add_to_particle_energy` exactly (full
relativistic relation in mc-normalized momentum `P`, not the
ultra-relativistic `delta += dgamma/P0` shortcut): ``volt =
Vrf_eV/mc2_eV`, `dgamma = volt*sin(phase_rad + omega_rf*tau_s)``,
`P0 = sqrt(gamma0^2-1)`, `P = P0*(1+delta)`, ``gamma =
sqrt(P^2+1)`, `gamma1 = gamma+dgamma`, `delta_new =
(sqrt(gamma1^2-1)-P0)/P0``.

Returns `(state6_new, energy_eV_new)`. `change_p0` defaults to
`False`, matching real elegant's own `RFCA` default -- the
reference is returned unchanged. `change_p0=True` reproduces
elegant's own `CHANGE_P0=1` exactly via
`_match_energy_to_centroid`.

`wake`/`transverse_wake_x`/`transverse_wake_y` (all `None` by
default) make this an `RFCW` equivalent: real elegant's own wake
placement is ``drift(L_k/2) -> RF energy kick -> [wake kick(s)] ->
drift(L_k/2)`` per sub-kick -- the wake sits between the energy kick
and the following half-drift, at the same `tau` already computed
for the RF phase, applied at every sub-kick with the same
`charge_per_particle_C` and multiplied by `wake_scale`. The wake table is a
*per-cell* wake (as in elegant's `RFCW`), so `wake_scale` is
`length / cell_length / n_kicks` (`ElementSpec.cell_length`, from `CELL_LENGTH`; unset means the table
is for the whole element, `wake_scale = 1 / n_kicks`): the total over the element is
`length / cell_length` cells whatever `n_kicks` is. Before 1 Nov 2026 the whole table was applied at every sub-kick, which is
`n_kicks` cells instead (2.95 times too strong for CLARA L02-L04, 360 kicks in 122 cells); the two agree only for
`n_kicks = 1, cell_length = length`, the only case the older tests used. Reuses
`ShortRangeWake.voltage_at`/`TransverseWake.kick_at` unmodified
and the same sign/normalization convention
`tracker._rf_and_damping` uses for the once-per-turn case.

`n_kicks` (default `1`) splits the cavity into `n_kicks`
sub-kicks of length/voltage `length/n_kicks`, `voltage_V/n_kicks`
-- elegant's own real per-cell discretization, a real structure's own
physical cell count (one RF kick + wake per cell).

`beta_wave` (default `1.0`, **not** the reference's own `beta0`)
is the traveling wave's own phase velocity, in units of `c`. Real
elegant's `RFCA`/`RFCW` have no phase-velocity parameter at all --
`beta_wave=1` (light speed) is hardcoded. This is the physically
correct default for a real fixed-cell-length structure (most real
electron linacs, including CLARA's, use a constant-phase-velocity
structure regardless of the injected beam's own energy, so a low-beta
injected bunch genuinely phase-slips against the wave). A
beta-matched structure (common in proton linacs) needs `beta_wave`
set explicitly -- nothing in a real elegant `.lte` file signals
this.

**Derivation**: a particle's local phase relative to a wave
`E(z,t)=E0*sin(omega*t - k*z + phase_0)`, `k=omega/(beta_wave*c)`,
evolves along `z` as ``dpsi/dz = (omega/c)*(1/beta(z) -
1/beta_wave)`. Splitting this integral at `beta_wave = beta0`` (the
reference's own fixed entrance velocity) leaves exactly the
reference-relative `omega*tau_s` term this function already
computes, plus a purely reference-position-dependent residual,
`(omega/c)*s*(1/beta0 - 1/beta_wave)`. Anchored to the convention
`phase_rad` already uses (the phase at the center of the first
cell) by subtracting this residual's own `ik=0` value -- so the
correction is exactly zero at `n_kicks=1` regardless of
`beta_wave`. The per-sub-kick correction is
`(omega_rf/C_LIGHT) * (ik*length_k) * (1/beta0 - 1/beta_wave)`,
`beta0` from the entrance `energy_eV` (fixed for the whole
element). Verified against live elegant's real `RFCA` multi-kick
output to ~1e-7 relative

`change_p0` (if set) is applied once, after the last sub-kick's own
energy kick + wake but **before** that sub-kick's own trailing
half-drift, not after -- the trailing half-drift's own reference-beta
term must use the new, post-match reference beta. 

`end1_focus`/`end2_focus`/`body_focus_model` add real elegant's
own RF-focusing (`simple_rfca.c`). `end1_focus`/`end2_focus` are
thin-lens kicks at the cavity's own entrance (`ik==0`, using a
phase/`dgamma` estimate from the entrance `tau`) and exit
(`ik==n_kicks-1`, using the exit `gamma1`/`dgamma` already
computed that iteration), independent of `body_focus_model` and
applied in every mode. This
module's state is **canonical** -- translated via ``d(px_canonical) =
-(Pz/P0)*x*inverseF``.

`body_focus_model="tw1"` (real elegant's own traveling-wave RF
focusing) adds one real per-sub-kick term on top of the ordinary
drift-kick-drift transport: ``dpr = (voltage_k/mc2)/(2*beta_i) *
(omega_rf/C_LIGHT) * (1-beta_i) * cos(phase_arg)``, translating into
`d(px_canonical) = dpr*x/P0` (canonical is linearly proportional to
physical momentum via the fixed reference `P0`, so no approximation
needed here unlike the slope-based END1/END2 kicks). For `x=y=0`
(the on-axis design particle) both END1/END2 and `dpr` kicks vanish
identically, so the reference-energy advance
(`linac_twiss._rf_forward_and_R`) is unaffected.

`body_focus_model="srs"` (real CLARA L02/L03/L04's own actual
setting -- forces elegant's own `STANDING_WAVE=1` internally too, per
`identifyRfcaBodyFocusModel`) **replaces** the drift-kick-drift
transport for x/y entirely with the closed-form Rosenzweig & Serafini
standing-wave R-matrix (elegant's own `useSRSModel` branch): ``alpha
= ln(gamma1/gamma) / (2*sqrt(2)*sin(phase_arg))``
(`dgammaMax/gamma/(2*sqrt(2))` in the `sin(phase_arg)~0` limit
elegant itself guards against), `R11 = cos(alpha)`, ``R22 =
R11*gamma/gamma1`, `R12 = 2*sqrt(2)*gamma*length_k/dgammaMax*
sin(alpha)`, `R21 = -sin(alpha)*dgammaMax/(length_k*gamma1*2*
sqrt(2))`, applied to the sub-kick's own **entrance** `(x, xp)``/
`(y, yp)` (slope convention) over the **full** `length_k`, not
composed with the two half-drifts the way `"tw1"`'s additive term
is. END1/END2_FOCUS still apply as above, independent of
`BODY_FOCUS_MODEL`, acting on the already-SRS-transported state for
END2. 

`compile_sub_kicks` (default `False`) makes the `ik` loop call a
single compiled `_rf_one_sub_kick` instead of the plain eager one --
the loop itself always stays a plain eager Python `for`. Only
`LinacTracker` sets this `True`; `NativeTracker`'s existing
`rf="per_element"` usage never passes it.

## Usage

Use `track_element` with lengths in metres, angles in radians, normalized strengths and reference energy/rest mass in eV. Integration order and slice count control convergence; tracker-level APIs construct these arguments.

### API reference

```python
def track_element(state6: torch.Tensor, length: float, angle: float, k1: float, k2: float, e1: float, e2: float, energy_eV: float | torch.Tensor, mc2_eV: float | torch.Tensor, n_kicks: int=4, integration_order: int=4, dx: float=0.0, dy: float=0.0, tilt: float=0.0, hgap: float=0.0, fint: float=0.5, edge_order: int=1, edge1_effects: int=1, edge2_effects: int=1, exact_dipole: bool=False, k3: float=0.0, multipoles: tuple=(), skew_multipoles: tuple=(), integrated_multipoles: tuple=()) -> torch.Tensor: ...
def reference_offset(length: float, angle: float, k1: float, k2: float, e1: float, e2: float, energy_eV: float, mc2_eV: float, n_kicks: int, integration_order: int=4, hgap: float=0.0, fint: float=0.5, edge_order: int=1, edge1_effects: int=1, edge2_effects: int=1, exact_dipole: bool=False) -> torch.Tensor: ...
```

### `track_element`

Track `state6` (`..., 6` -- any leading batch shape) through one
element: misalignment entry transform (if `dx`/`dy`/`tilt`
given) -> entrance edge kick (if `angle != 0`) -> `n_kicks` slices
of the `integration_order`-th-order (4, 6, or 8) Yoshida drift/kick
splitting -> exit edge kick -> misalignment exit transform. A single
generalized integrator for every element type this repo's real
lattices use (DRIFT: `angle=k1=k2=0`; KQUAD: `angle=k2=0`; KSEXT:
`angle=k1=0`; CSBEND: all four generally nonzero). No precomputed
2nd-order Taylor map anywhere: every element is exactly integrated
every turn.

`integration_order=6` (elegant's `INTEGRATION_ORDER=6`, 8
sub-stages per slice) is opt-in, matching elegant's own default of 4
-- this repo's own real machine lattice
(`BR14_BST_Aligned_Matched.lte`) uses order 6 with fewer slices for
its nonzero-K2 KSEXTs/CSBENDs. `integration_order=8`
(`_DRIFT_FRAC8`/`_KICK_FRAC8`) is a native-only extension elegant
doesn't have.

`dx`/`dy`/`tilt` (misalignment, see `_misalign`) default to
no-op -- see `tracking/native/errors.py`.

`exact_dipole=True` (bends only, native-only, not elegant's splitting):
each drift sub-step is the exact pure-dipole map
(`_exact_dipole_drift`) and the kicks carry only K1/K2 -- same
Hamiltonian, converges to the same result, but float32-robust and a
pure dipole is then exact in one step. See `_exact_dipole_drift`.

### `reference_offset`

`track_element` applied to the exact reference orbit (`state6=0`)
-- ideally exactly zero (an on-axis, on-momentum particle should be an
exact fixed point of any real element), but real elegant's own default
kick counts are tuned for per-particle tracking accuracy, not for
making the reference orbit an exact numerical fixed point. Applied
every turn across many bends in a real ring, this residual compounds
into a multi-hundred-turn instability if uncorrected

Real elegant does not have this problem despite using the same finite
`N_SLICES`: it explicitly records a reference trajectory once per
unique bend configuration and subtracts it from every particle
(`csbend.c`'s `RECORD_TRAJECTORY`/`SUBTRACT_TRAJECTORY`,
per-slice). This reproduces the same fix at this module's own level
of abstraction (per-element, not per-slice) -- see
`tracking.native.tracker._apply_elements`, which subtracts this
from every real particle's own output. `lru_cache`d like
`_element_R`: the same magnet type recurs many times in a real
ring, and this never touches the per-turn tracking loop directly
(computed once, stored in `tracker._ElementParams`, then just a
plain tensor subtraction every turn).

## Numerical and implementation notes

### `Module constants`

The trivial order-2 (Strang/leapfrog) splitting: drift(ds/2), kick(ds),
drift(ds/2) -- the base case _yoshida_extend below recursively builds
every higher even order from.

### `Module constants`

elegant's own defaults: DEFAULT_N_KICKS=4, integration_order=4 (4th-order
Yoshida symplectic splitting) -- BETA=2**(1/3), transcribed verbatim from
multipole.c/csbend.c's driftFrac4/kickFrac4 tables.

### `Module constants`

elegant's optional 6th-order splitting (INTEGRATION_ORDER=6, 8 sub-stages)
-- transcribed verbatim from multipole.c/csbend.c's driftFrac6/kickFrac6
tables, both citing "AOP-TN-2020-064".

### `Module constants`

Extends elegant's own (minimal, 8-stage) order-6 splitting one more
level -- fewer stages and smaller error than a from-scratch
order-4-derived order-6 for the same n_kicks (verified empirically).

### `_canonical_drift`

Return `(x,px,y,py,arc_length)` after a canonical drift over fixed reference length `ds` and curvature `h`. In a straight drift, `pz=sqrt((1+delta)**2-px**2-py**2)`, `x+=ds*px/pz`, `y+=ds*py/pz`, and `arc_length=ds*(1+delta)/pz`. In a sector bend, preserve the cancellation-resistant sine/cosine identities. Its auxiliary factor gives `arc_length=factor*(1+delta)`: using factor alone changes time of flight.

### `_canonical_drift`

Return `(x,px,y,py,arc_length)` after a canonical drift over fixed reference length `ds` and curvature `h`. In a straight drift, `pz=sqrt((1+delta)**2-px**2-py**2)`, `x+=ds*px/pz`, `y+=ds*py/pz`, and `arc_length=ds*(1+delta)/pz`. In a sector bend, preserve the cancellation-resistant sine/cosine identities. Its auxiliary factor gives `arc_length=factor*(1+delta)`: using factor alone changes time of flight.

### `_corrector_kick`

(1 - delta + delta^2): elegant's own order-2 (default_order>=2) Taylor truncation of the exact
1/(1+delta) momentum-dependence of a fixed-field kick's angle -- confirmed by a live-elegant
check that a bare (1-delta) leaves an xp residual growing exactly as sin(kick)*delta^2
(bend_matrix.c's corrector_matrix, T[1][5][5]=T[0][5][5]/(0.5*length)=K*length=sin(kick)).

### `_edge_kick`

Under `torch.compile` these per-element floats may be represented as
SymFloats. Dynamo can trace multiplication of opaque trig results but
currently fails while evaluating `math.sin(symfloat) ** 2` with fake
tensors (seen on the CUDA dipole-subcycle path). Spell the squares as
products; this is algebraically identical and remains compile-safe.

### `_rf_one_sub_kick`

Pre-drift phase/dgamma estimate, entrance tau -- numerically
indistinguishable from elegant's own pre-drift dc4 estimate.
elegant's own *kick-method* branch (this "not is_srs" case)
computes END1's dgamma this way; its *matrix-method* branch
(is_srs) instead reuses the sub-kick's own center-of-cavity
dgamma -- see the is_srs-specific END1 application below.

### `_rf_one_sub_kick`

elegant's own matrix-method branch stays in **slope** space for END1 (unlike the kick-method
branch's own canonical pz_norm-translated version above) -- SRS's R-matrix is exponentially
sensitive to its entrance slope, so the kick-method's canonical approximation isn't good enough
here.

### `_rf_one_sub_kick`

Rosenzweig & Serafini standing-wave R-matrix (elegant's own
simple_rfca.c, useSRSModel branch) -- REPLACES the drift-kick-
drift decomposition above entirely for x/y (the two half-drifts
computed into `x`/`px`/`y`/`py` above are discarded here, kept
only for their own `tau`/phase-at-center side effect, identical
regardless of body_focus_model). Applied to the sub-kick's own
entrance state, over the *full* `length_k` (not `half`) -- see
ElementSpec's own body_focus_model docstring. `xp_entrance`/
`yp_entrance` already include END1's own slope-space kick above
(a no-op, same values, when END1 doesn't apply this sub-kick).

### `_rf_one_sub_kick`

elegant's own matrix-method branch re-derives its "coord[4]"
(path length) via a plain (not beta_ref-relative) arc length,
then rescales the whole accumulated value by the exit/entrance
beta ratio at the very end ("adjust s for the new particle
velocity") -- this module's own tau convention already folds an
equivalent beta_ref-relative correction into `_drift_half`
(validated for "none"/"tw1"), so this multiplicative ratio is
the residual correction matrix-method mode alone still needs.

### `_rf_cavity_kick`

phase_slip computed here (a plain float, not the loop index ik
itself) -- see _rf_one_sub_kick's own docstring for why. Zero at
ik=0 regardless of beta_wave -- see this function's own
"anchored to phase_rad's convention" derivation.

### `track_element`

isinstance-gated (not just != 0.0) so a differentiable tensor
dx/dy/tilt at its nominal zero value still routes through
_misalign -- see _misalign's own docstring.
