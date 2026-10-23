# `slipstream.tracking.native.injector`

[Source](../../../../src/slipstream/tracking/native/injector.py) · [Documentation index](../../../README.md)

## Physics

Tracks cathode emission and Cartesian proper velocities through standing/traveling-wave cavities, solenoids, magnets, apertures and cylindrical or 3D space charge with optional cathode images. Cavity auto-phasing finds maximum energy gain sequentially.

Optional local binary Coulomb collisions are configured by
`Injector.ibs = InjectorIBSSettings(...)`; see [injector IBS](injector_ibs.md)
for the model, physical cutoffs, float64 requirement and validation limits.
IBS defaults off and is explicitly unsupported in `track_differentiable`.

### Limits and conventions

- YAML paths are relative to the machine file and physical fields/settings use SI.
- Mesh shapes/buckets must stay stable for compiled reuse; float32 kicks can become smaller than representable increments.
- Particle endpoint conversion must preserve actual energy and surviving charge.
- Unsupported ASTRA settings must remain surfaced in warnings/errors.

### `_on_axis_energy_MeV`

Kinetic energy (MeV) of a batch of on-axis reference particles (one
per trial phase) on first crossing `z_in` and `z_out`, tracked from
the cathode through `fields` with a 1D relativistic leapfrog (field at
the half step, second order -- a plain kick-drift's first-order error
showed up directly as a real crest-phase shift).

Step size: `dt_fine` while any live candidate has ``beta <
beta_coarse`, else `dt_coarse`` (near the cathode the particle starts
at rest under a strong field and needs the fine step; once `v~c` the
push degenerates to a quadrature of `Ez` along the line). Fields no
candidate is inside are skipped.

The small trial batch runs in CPU float64 NumPy, evaluating only `Ez`
from the same standing/traveling-wave tables used by particle tracking.
Empty drift intervals are advanced analytically by summing whole time
steps, stopping before any live candidate reaches the next field. The
phase grids and integration steps inside fields are unchanged; each scan
computes its results afresh without compilation or saved trajectories.

A candidate whose `uz` reverses (phased to decelerate at the cathode)
is retired at its current energy -- it cannot be the crest, and would
otherwise drift backwards forever.

### `_traj_stats`

One record event as a single `torch.compile` kernel, returned as
one stacked row in `TRAJ_KEYS` order (eagerly this is ~35 reductions
per event; stacking makes the whole trajectory one D2H sync).

`mc2_eV`/`q_C` are the tracked species' rest mass/charge (default:
electron) -- `Injector.track` passes its own `self.mc2_eV`/
`self.q_C` through `stats_frame`, never the module-level
`ELECTRON_MASS_EV`/`ELECTRON_Q_C` constants directly, for a
non-electron beamline (e.g. ISIS's proton MEBT).

The canonical emittances use the canonical transverse momentum of an
axisymmetric solenoid, `p_can = p_kin + q*A` with `A_theta = Bz*r/2`
(paraxial) -- the convention ASTRA's `Xemit`/`Yemit` effectively
report. Inside a solenoid the plain kinetic-slope emittance shows the
reversible Larmor rotation as apparent growth; the canonical one removes
it. `Bz = Bz0 - r^2/4 * d2Bz0/dz2` summed over `sol` (and the framed
`placed_sol`, see `_axial_bz`).

`beam_frame` (a bent beamline): every transverse/longitudinal quantity is
evaluated in the beam's own frame -- coordinates and momenta rotated about
`y` by the mean-momentum direction, in float64 -- except `z_m` (the first
column), which stays the lab `z` of the centroid so it keeps lining up with
ASTRA's `z` axis.

### `_twiss_row_from_ensemble`

One `simba.Modules.Twiss`-compatible row, computed directly from a
real tracked ensemble at a fixed simulation time (this tracker's
equivalent of ASTRA's Xemit/Yemit/Zemit columns -- ``beta_x =
sigma_x^2/ex`, `alpha_x = -sign(cov(x,x'))*sigma_x*sigma_xp/ex``, the
same formulas `simba.Modules.Twiss.astra` uses, but computed once from
SI tensors instead of reverse-engineered from ASTRA's own lossy ASCII
columns). Skips `eta_x`/`eta_xp` (no dispersion source in a straight
injector) and `mux`/`muy` (phase advance isn't meaningful across a
handful of widely-spaced, space-charge-dominated screens).

`beam_frame` (a bent beamline): the ensemble is rotated into the beam's own
frame first, as in `_traj_stats`; `z` in the row stays the lab centroid.

### `_write_twiss_hdf5`

`simba.Modules.Twiss`-compatible HDF5: version-2 layout
(`Parameters/{Source,Version}` + one dataset per field under
`twiss`, each with a `Units` attr), written directly with h5py so
slipstream stays independent of the simba package while still loading
straight into `simba.Modules.Twiss.twiss.read_HDF5_twiss_file`.

### `_dump_beam_openpmd`

Full 6D macroparticle dump in openPMD-beamphysics format
(`beamphysics.ParticleGroup`) -- this tracker's equivalent of an
ASTRA screen/end-of-line particle file. `weight` is the positive-
definite macroparticle charge magnitude in C (openPMD convention: sign
comes from `species`).

### `_quad_tables`

`_fused_step`'s own `quad` tuple: `(g, z0, zend)` triples,
`g` already sign-translated from a `Quadrupole`'s own ASTRA-`Q_grad`-
matching convention into the plain-physics `By=g*x, Bx=g*y` formula
(standard: `g>0` horizontally focuses a *positive* charge; a
`Quadrupole.grad_T_per_m>0` horizontally focuses an *electron*, so
`g = -grad_T_per_m` here -- the injector's own real, signed electron
charge does the rest via the ordinary Lorentz force, no separate
per-species handling needed downstream).

### `_aperture_tables`

Plain `(kind, z1, z2, r)` tuples for `_apply_apertures`,
consumed by `_step_with_apertures`. Kinds and geometry are constants
specialized at trace time, so CUDA checks fuse with the particle push
rather than launching dozens of kernels per step. Every aperture is
checked over the path each particle sweeps in the step (see
`_apply_apertures`).

### `_swept_aperture_hit`

Lost on one aperture anywhere along the straight segment `(xa0, y0, za0) -> (xa, y, za)`.

The step is clipped to the slab `[z1, z2]` (a zero-thickness slab is a plane crossing) and the
acceptance tested at the two clipped ends. That is exact for every cut: the excluded region is
convex along the segment (hole, collimator: the far end is the worst point) or a half-plane
(scraper), and only the solid plug needs the closest approach in between.

### `_apply_apertures`

One AND-accumulation pass over every aperture -- a particle already
lost stays lost (monotonic). Geometry per `Aperture`'s own docstring.

With `prev = (x0, y0, z0)`, the positions before the step, each particle's
straight path over the step is tested against the slab, so the loss count
does not depend on `dt` (a step of a few mm steps clean over a 1 mm slab,
or over a zero-thickness plane, and end-of-step positions alone would never
see it). Without `prev` only the positions `(x, y, z)` are tested.

### `_step_with_apertures`

Push particles, then check the path of the step (previous to updated positions) in the same kernel.

Keep the standalone pusher's return signature for `gun.track_particles`
and physics cross-checks. The empty-aperture case returns `alive` as-is.

### `_rf_cos_sin`

`(cos, sin)` of a 0-d float64 RF phase, as `dtype`. The phase must be
formed in float64 (`omega*t` is ~1e2 rad, far past float32's resolution),
but Inductor fuses a 0-d `cos`/`sin` into the per-particle kernel, where
every thread pays for float64 transcendentals -- 1/64 rate on a consumer GPU,
~30% of the whole step at 1M particles.
So reduce to `[-pi, pi]` in float64 (a few fma) and take `cos`/`sin` in
`dtype`: the same value to ~2 ulp of float32, exact-to-rounding in float64.

### `_rf_phases`

`(cos, sin)` of `omega*t+phase` for every element of an `sw`
or `tw` tuple (its own last two entries), stacked into one
`(len(elements),)` tensor each -- for `warp_kernels.applied_fields`,
which wants these precomputed once per step (one value per element),
not per particle. Empty input returns empty tensors, matching a
zero-cavity/zero-TW-structure machine.

### `_applied_fields`

`Ex, Ey, Ez, Bx, By, Bz` from every configured element (`sw`
standing-wave cavities, `tw` travelling-wave structures, `sol`
solenoids, `quad` quadrupoles, lab-frame `magnets`) -- everything
`_fused_step` applies *except* the cached space-charge field and the
push itself. Shared by the torch push (`_fused_step`) and the
warp-backed push (`_fused_step_warp`): this part stays eager torch
either way (control flow over a handful of elements, not a per-particle
hot kernel

`t` is a 0-d float64 device tensor so `omega*t` stays a tensor op
and one graph/one set of dispatches serves every step regardless of
the wall-clock time reached.

### `_fused_step`

One full tracking step -- every applied field (see
`_applied_fields`), the cached space-charge field, and the
relativistic Boris push -- written to fuse into a single
`torch.compile` kernel (collapses ~200 per-step eager dispatches into
one launch). The element loops unroll at trace time (the tuples are
fixed for a run). Run eagerly on CPU. See `_fused_step_warp` for the
CUDA/warp-backed sibling -- same
field-table computation, warp kernel for the push tail below instead.

`emitted_all` is a Python bool fixed per compiled variant, so the
emitting and post-emission regimes get one graph each.

**Momenta and positions are accumulated with Kahan compensation**
(`gun.kahan_add`): at typical CLARA-gun-exit momenta, a plain ``u +
du`` is an exact no-op for the space-charge kick every step, silently
deleting longitudinal space charge (see docs/tracking_native.md). This
is what makes float32 tracking physically correct.

### `_applied_fields_warp`

Same as `_applied_fields`, but `sw`/`tw`/`sol` run as warp
kernel launches (`warp_kernels.applied_fields`) instead of through
`_applied_fields_compiled`. Correct (parity-tested,
`tests/test_warp_kernels.py::test_applied_fields_warp_matches_torch`)
but **not currently called by anything** -- see docs/tracking_native.md's
"Injector warp backend" for the measured numbers.

`cos`/`sin` of each cavity/TW structure's RF phase are computed
once per element here (`_rf_phases`, plain torch on the 0-d clock
tensor) and handed to the kernels as small per-element arrays -- one
transcendental evaluation per element per step, not one per particle.

### `_fused_step_warp`

Same as `_fused_step`, but the Boris-push tail runs as one warp
kernel launch (`warp_kernels.push`) instead of inline torch ops --
CUDA only, see `warp_kernels.available`. The field-table part stays on
`_applied_fields_compiled`, not `_applied_fields_warp` -- see
docs/tracking_native.md's "Injector warp backend" for why (measured
slower, not just "not yet built"), including the `dt`/`qmdt2`
plain-float vs. 0-d-tensor distinction from `_fused_step`.

### `_step_with_apertures_warp`

`_step_with_apertures`'s warp-backed sibling -- see `_fused_step_warp`.
No-op when there are no apertures to check. Clones `x`/`y`/`z`
before the push when apertures *are* configured, and uses
`warp_kernels.apply_apertures` rather than a compiled
`_apply_apertures` -- see docs/tracking_native.md's "Injector warp
backend" for why.

### `_quantize_bucket_hysteresis`

`_quantize_bucket` with hysteresis: once settled on `current`,
only move to a smaller bucket once `needed` comfortably clears it
(`<= down_factor` of the next bucket down), so a value hovering at a
bucket boundary doesn't rebuild (and recompile) the solver every
recompute -- growth is still immediate. `current` is `None` on the
first call for a given quantity.

### `_fused_sc_fields`

Space-charge deposit -> Poisson solve -> gather -> relativistic
lab-frame boost (`CylindricalSpaceCharge.deposit`/`solve`/`gather`
plus `gun.relativistic_space_charge_field`'s transform, inlined so
`torch.compile` can fuse the chain -- see docs/tracking_native.md).

`with_image` is a Python bool fixed per compiled variant (the image
doubles the deposited set -- a different graph shape). `smooth_r`/
`smooth_z`/`smooth_gamma_ref` are plain Python scalars, also
compile-time constants for the whole run: ASTRA's `&CHARGE` shot-noise
filter, blended in with weight `1/(1+(gamma/smooth_gamma_ref)**2)`
computed from the *tensor* `bunch_gamma_t` (not a synced-out float) so
the fade itself needs no extra recompiles. 0/inf by default (unchanged
behavior) -- see `space_charge.smooth_rho`/`CylindricalSpaceCharge.solve`. `bunch_gamma_t`/`z_ref_t`/`beta_t`/`dr`/`dz`/`z_min`
are 0-d device tensors, not floats, for the same recompile-avoidance
reason as `t` in `_fused_step`.

### `_sc_gather_and_boost`

`_fused_sc_fields`'s tail -- rho normalize/smooth, Poisson FFT
solve, gather, relativistic lab-frame boost -- everything *after* the
deposit, split out so it can be compiled independently of which
backend produced `charge` (see `_fused_sc_fields_warp`, which
calls this via `_sc_gather_and_boost_compiled` below with a
warp-deposited `charge`, vs. `_fused_sc_fields`, which does this
inline with a torch-deposited one under one bigger compiled graph).
`r0r`/`wrr`/`z0r`/`wzr` are the CIC coordinates already sliced
to the real (non-image) particles.

### `_fused_sc_fields_warp`

Same as `_fused_sc_fields` -- identical signature, drop-in at the
call site -- with the CIC scatter-add deposit (the `charge =...`/
`scatter_add_` block there) run as one warp kernel launch
(`warp_kernels.sc_deposit`) instead of four chained
`scatter_add_` calls, and everything after it (`_sc_gather_and_boost`)
`torch.compile`'d separately -- see `_sc_gather_and_boost_compiled`'s
comment for why the tail is still compiled despite warp handling the
deposit. CUDA only, see `warp_kernels.available` and
 - the deposit is the kernel measured warp winning on at every particle count, and the one whose
per-`(nr, nz)`-bucket recompile (when it was part of one bigger
compiled graph, as it still is in `_fused_sc_fields`) was this plan's
original motivation.

`r0`/`wr`/`z0`/`wz` are computed once here in torch (needed
regardless for the gather) and handed to `warp_kernels.sc_deposit`,
which only does the atomic-add-bound part -- see that function's
docstring.

### `_fused_sc_fields_3d`

3D Cartesian analogue of `_fused_sc_fields` (see there and
`SpaceCharge3D`): stretch `z` into the bunch rest frame about
`z_ref_t`, CIC-deposit (with the antisymmetric cathode mirror set
appended when `with_image`), Hockney FFT convolution with the cached
cell-averaged kernel `G_k`, central-difference fields, trilinear
gather at the real particles, then the lab-frame boost back
(`E_perp` x gamma, `B = (beta/c) z x E`) -- one compiled kernel for
the whole chain.

Every varying scalar is a 0-d device tensor (`bunch_gamma_t`,
`z_ref_t`, `beta_t`, `x_lo`, `y_lo`, `z_lo`, `hx`, `hy`,
`hz`): a Python float is a compile-time constant under
`torch.compile(dynamic=False)`, and these change whenever the bucketed
grid is re-fit. `(nx, ny, nz, px, py, pz)` are Python ints fixed per
`SpaceCharge3D` instance (a bucket change is a fresh instance and a
fresh graph, as in the cylindrical path); `G_k` is a same-shape buffer
swapped on bucket changes -- guarded by shape/dtype only, so it never
recompiles. `with_image` is a Python bool, one compiled variant per
value.

### `_fused_sc_fields_3d_warp`

3D space charge with direct Warp CIC deposit and field gather.

PyTorch retains the cuFFT solve; both particle-grid boundaries stay in
Warp so no eager index/weight or field-grid scratch tensors are needed.

### `_warn_if_sc_kick_unresolvable`

One-off check that float32 can still represent the space-charge kick
given Kahan compensation. The compensation term sits at ~ulp(u) and
resolves `2**-23` of it -- this checks where even *that* runs out
(scales with beam energy, inversely with field strength -- reported,
not assumed). See docs/tracking_native.md.

### `_local_sc_omega`

Local transverse space-charge angular frequency (rad/s), measured
directly from the just-computed SC field rather than modeled from total
charge -- `SpaceChargeSettings.dt_adaptive`'s estimator.
`du/dt = (q/m)*E_sc + (q/(gamma*m))*(u x B_sc)` (`u` = proper
velocity, the SAME split `_fused_step`'s Boris push uses) and
`omega = sqrt(<ax^2+ay^2> / <x^2+y^2>)` (RMS force over RMS radius,
not a signed correlation). See docs/tracking_native.md's "Injector
adaptive time-stepping" and "`_local_sc_omega`'s relativistic split" for
why both choices matter and what breaks otherwise. Returns 0.0 (caller
falls back to the ceiling) for an empty/near-empty `counted` set.

### `_compute_space_charge_in_beam_frame`

Space charge for a bent beam: rotate the bunch about `y` by `theta` (its mean
momentum direction, from `+z` toward `+x`) so its axis lies along `z`, solve there with the
unchanged axisymmetric solver, and rotate the fields back. Also returns the local space-charge
frequency `_local_sc_omega` measured *in that frame* (where `x` is transverse) -- `qm` is the
tracked species' charge/mass ratio (default: electron), passed through to it. Post-emission only
(it is the caller's `bent and emitted_all`), and the cathode image is switched off: the beam is
then far from the cathode (and in a rotated frame the mirror plane would be wrong anyway).

### `_snap_h3d`

Quantise a 3D cell size onto the geometric ladder anchored at
`current` (the previously chosen size, `None` on the first fit,
where the ladder anchors at the ideal itself). Growth snaps UP to the
next rung -- the box must keep covering the bunch -- shrink moves down
only when comfortably clear of the rung below (one full step), the same
hysteresis discipline as `_quantize_bucket_hysteresis`: a value
hovering at a boundary must not rebuild the Green's function every
recompute. Cell sizes are `SpaceCharge3D` constructor arguments, so
every rung change is a fresh solver (cached-`G_k` amortisation, see
_compute_space_charge_3d).

### `_compute_space_charge_3d`

`SpaceChargeSettings.geometry == "3d"`: lab-frame `(E, B)` space-charge
fields via the persistent `SpaceCharge3D` solver and (when compiled) the fused
`_fused_sc_fields_3d` kernel. Same rest-frame boost as the cylindrical path:
`z` stretched by the mean gamma about the reference plane (the cathode while
its image is on, else the centroid), electrostatic solve there, `E_perp` x gamma
back in the lab, `B = (beta/c) z x E`.

Grid discipline (the cylindrical branch's, adapted): the grid SHAPE is fixed by
config (`n_xyz`, with the image-on `nz` bucketed against
`SC_IMAGE_NZ_BUCKETS` as the bunch leaves the cathode), while the physical box
is re-fit every recompute -- but the CELL SIZES are snapped to the `_snap_h3d`
ladder, so the expensive Green's function is rebuilt only on a bucket change
(amortised), not per recompute. The requested
box retains the stateless path's full particle extent and 5%-of-span margin so
caching does not also change the discretised physics. Eager before emission
(`sc3d_fn=None`): the active set's shape changes nearly every step there and
would recompile constantly.

### `_differentiable_space_charge`

`Injector.track_differentiable`'s space charge: `_fused_sc_fields` on `solver`'s fixed
`(nr, nz)` shape, with the box sized and placed by tensors, so the field stays a smooth function
of the particles (away from CIC cell crossings) -- no bucket choice, no rebuild, no host sync.

The sizing is `_compute_space_charge`'s: radially the outermost particle plus `box_margin`
sigma_r; longitudinally, in the rest frame, the bunch extent plus `box_margin` sigma_zs and 25 %
slack, about the centroid -- or, with `cathode_z` (the image), symmetric about the cathode out
past the mirror lobe. `fixed_box = (r_max_m, z_half_m)` pins it instead (same centring). The
solver coefficients follow the cell size exactly (`CylindricalSpaceCharge.coefficients`).
`active` (`None` once all are emitted) masks unemitted particles out at fixed shape. Also
returns the fraction of active particles outside the box (no gradient; CIC clamps them onto it).

## Usage

Build `Injector.from_astra`, load/save its YAML form or construct its dataclass elements. Use `auto_phase`, `initial_beam` and `track`; `track_differentiable` exposes the constrained differentiable path. `injector_exit_to_canonical6` forms a common-plane native endpoint.

### API reference

```python
def injector_exit_to_canonical6(result: dict, device: str | torch.device | None=None, dtype: torch.dtype=torch.float64, mc2_eV: float=ELECTRON_MASS_EV) -> tuple[torch.Tensor, float, float]: ...
class Cavity:
    ...
    def is_traveling_wave(self) -> bool: ...
class Solenoid:
    ...
    def placed(self) -> bool: ...
class SpaceChargeSettings:
    ...
class Quadrupole:
    ...
class Corrector:
    ...
class Aperture:
    ...
class Injector:
    ...
    def mass_kg(self) -> float: ...
    def q_C(self) -> float: ...
    def qm_C_per_kg(self) -> float: ...
    def from_astra(cls, path: str | Path, mc2_eV: float=ELECTRON_MASS_EV, charge_sign: float=-1.0) -> 'Injector': ...
    def resolve(self, rel: str) -> Path: ...
    def save(self, path: str | Path) -> None: ...
    def load(cls, path: str | Path) -> 'Injector': ...
    def fields(self, device='cpu', dtype=torch.float64, params: dict | None=None): ...
    def magnet_elements(self, device='cpu', dtype=torch.float64, frame_all: bool | None=None) -> tuple[tuple, tuple]: ...
    def auto_phase(self, verbose: bool=True) -> list[float]: ...
    def initial_beam(self, n_particles: int | None, device, dtype: torch.dtype, verbose: bool=False) -> dict: ...
    def track(self, n_particles: int | None=None, device: str | None=None, dtype: torch.dtype | str=torch.float64, use_compile: bool=True, record_every: int=10, z_stop_m: float | None=None, mesh: str | None=None, verbose: bool=True, dump_dir: str | Path | None=None, return_particles: bool=False, screen_z_m: float | None=None, use_warp: bool=True) -> dict: ...
    def track_differentiable(self, params: dict | None=None, *, n_steps: int | None=None, dt_schedule=None, beam: dict | None=None, n_particles: int | None=None, device: str | None=None, dtype: torch.dtype | str=torch.float64, use_compile: bool=False, record_every: int=0, space_charge: bool | None=None, sc_box_m: tuple[float, float] | None=None, sc_image_steps: int=0, checkpoint_every: int=0, return_particles: bool=False, screen_z_m: float | None=None) -> dict: ...
def parabolic_vertex(degs: torch.Tensor, gains: torch.Tensor, i: int, h: float) -> float: ...
```

### `injector_exit_to_canonical6`

`Injector.track(..., return_particles=True)`'s own raw exit state
(particles at a common lab *time*, real spread in `z`) into
`tracking.native`'s canonical `(x, px, y, py, tau, delta)` at a
common *s* (real spread in arrival time instead) -- the standard
ballistic back/forward-drift every downstream handoff needs (`track`'s
own docstring points here). Same construction as
`Stories/Moustache/track_moustache.py`'s `_injector_to_canonical6`,
generalized into a shared function instead of a one-off script copy.

Returns `(particles6, energy_eV, charge_C)` -- `energy_eV` from the
bunch's own mean celerity magnitude, `charge_C` the summed absolute
macroparticle charge. Done in float64 throughout regardless of
`dtype` (a one-time O(N) coordinate transform, not a many-step
integration -- cheap either way, and avoids the near-cancellation in
`u/u0 - 1` losing precision at float32), cast to `dtype` only at
the very end. `mc2_eV` must match the `Injector` that produced
`result` (its own `mc2_eV`, default electron).

### `Cavity`

One RF cavity: standing wave (a plain 2-column on-axis `Ez` file)
or travelling wave (ASTRA's own `z1 z2 n m`-headed file format --
`fields.TravelingWaveField`). Real ASTRA (and `fields.py`'s own
`read_astra_tws_file`/`read_astra_field_file` docstrings) dispatch
on the file's own name starting with `TWS`, **not** on whether
`C_numb`/`n_cells` is nonzero -- `is_traveling_wave` below is the
single place that check happens. A real, previously-latent bug found on
ISIS MEBT's QWR buncher: a real ASTRA deck can set `C_numb` on a
perfectly ordinary standing-wave file (there it's simply unused by real
ASTRA); dispatching on it here instead crashed trying to parse a
2-column file as `z1 z2 n m`-headed. Every existing CLARA/FERMI
cavity happened to have a real `TWS*`-named file exactly where
`C_numb` was also nonzero, so the two conditions never disagreed
until now. `n_cells` (`C_numb`) itself is still needed, unchanged,
once a file genuinely is a TWS file (the periodic-cell repeat count).
`phase_deg` is relative to the crest; `crest_deg` is the absolute
`phase_rad` convention of `fields.py` (`cos(omega t + phase)`),
filled in by `Injector.auto_phase`. ASTRA's logged crest phases map
onto it as `crest_deg = Phase_ASTRA - 90`.

### `Solenoid`

Static solenoid from an on-axis `Bz` file, peak scaled to
`b_max_T` (sign included -- ASTRA's `MaxB`). `x_m`/`xrot_rad`
(ASTRA's `S_xoff`/`S_xrot`) move it off the `z` axis: rotate about
the field table's centre by `xrot_rad` (from `+z` toward `+x`), then
shift by `x_m` -- see `magnets.placement`. `y_m`/`yrot_rad` do the
same vertically (`yrot_rad` from `+z` toward `+y`, applied after
`xrot_rad` -- `magnets.placement3`); `from_astra` does not import
`S_yoff`/`S_yrot` (ASTRA's sign convention for them is unchecked).

### `SpaceChargeSettings`

Space-charge resolution/cadence knobs. `mesh="adaptive"` (default)
auto-derives the grid shape `(nr, nz)` every recompute from the live
bunch size against a fixed accuracy target. `mesh="fixed"` keeps
fully manual behavior, as a backup/override. See docs/tracking_native.md
and for the design and validation
numbers behind every default below.

Used by **both** modes:
- `box_margin`: half-box = margin x RMS size beyond the bunch extent.
- `sigma_z_floor_m`: minimum bunch length used to size the z box.
- `every_step_until` / `subcycle_steps`: recompute SC every step for
  the first N steps (emission), then every M steps, both in units of
  `Injector.dt_s`. `every_step_until=None` auto-derives N as the
  number of steps needed to finish emission.

Used only when `mesh="fixed"`:
- `nr`, `nz`: grid cells.
- `image_threshold_m`: cathode image switched off once the centroid
  is this far from the cathode.

Used only when `mesh="adaptive"`:
- `cells_per_sigma_r` / `cells_per_sigma_zs`: accuracy targets
  (not cell counts) -- `nr`/`nz` are derived every recompute and
  quantized (with hysteresis) into the module's bucket sets. The
  cathode image auto-disables once resolving it at this target would
  need more cells than `SC_IMAGE_NZ_MAX` allows.
- `min_particles_per_cell`: cap adaptive grids at this average
  macroparticle occupancy; set to zero to disable the noise guard.

`smooth_r` / `smooth_z`: binomial `(1,2,1)/4` smoothing passes on
the deposited charge-density grid before the Poisson solve. Default
0 (off). `Injector.from_astra` imports ASTRA's `&CHARGE`
`smooth_x`/`smooth_y`/`smooth_z` only for `LSPCH3D = T`: ASTRA applies them to its
3-D FFT algorithm only (manual section 6.6), so its cylindrical algorithm -- the one
this solver is compared with -- never smooths. Importing them anyway blurred a density
ASTRA does not blur: at the converged emission step it moved the CLARA 100 pC
emittance by -9 % (see `Stories/CLARA_InjectorConvergence`).

`emission_steps`: the number of steps across the emission window (latest minus earliest
emission time of the beam); the emission-phase step is `min(Injector.dt_s, window /
emission_steps)`, 0 keeps `dt_s`. A macroparticle is pushed from the first step at or
after its emission time, so a step that is coarse against the window quantises the
emission; ASTRA instead starts each particle at its own emission time. For the 2.9 ps
CLARA window the 100 pC emittance is +14.7 % of ASTRA's at a 100 fs step and converged
(-7.7 %, smoothing on; +1.2 %, off) at 50 fs and below, at no cost in run time. After
emission a fixed-step run returns to `dt_s` and an adaptive one to its own rule. A
`photocathode` run keeps `dt_s`: there it is the resolution of the emission law the
caller chose. An in-flight beam has no window.

`smooth_gamma_ref`: smoothing is blended in with weight
`1 / (1 + (gamma/smooth_gamma_ref)**2)`, fading out at high gamma —
full-strength smoothing helps near the cathode but measurably hurts
further downstream. Default 3.0 is a measured compromise across a
5pC/100pC A/B, not a clean derivation from first principles.

`dt_adaptive` (the default; step errors measured in `Stories/CLARA_InjectorConvergence`):
re-derive the tracking step at every space-charge recompute from the
*live* local transverse space-charge frequency instead of tracking the
whole line at one fixed step. See `_local_sc_omega`:
`dt = clamp(dt_safety / omega, dt_floor_s, dt_ceiling_s)`, and
docs/tracking_native.md's "Injector adaptive time-stepping" for the
2026-09-23/25 validation history and why it isn't a speedup yet.

### `Quadrupole`

A hard-edge, real-field quadrupole: a uniform gradient over
`[z_m - length_m/2, z_m + length_m/2]`, zero outside -- no fringe
taper or skew/multipole content in this hard-edge form. Profile fields and
placement are supported separately as described below. Unsupported ASTRA
options are reported by the importer.

`grad_T_per_m` is a genuine field gradient (T/m), applied via the
same real Lorentz-force Boris pusher the solenoid/cavity tables already
use -- not a normalized `k1` linear-optics kick, so it needs no
separate non-relativistic correction near the cathode. Its sign
matches ASTRA's own `Q_grad` convention (positive focuses electrons
horizontally) so a value copied straight out of a deck behaves the
same here; `_quad_tables` does the one-time translation into the
plain-physics `By=g*x, Bx=g*y` field `_fused_step`'s loop applies.
A momentum-normalized strength (ASTRA's `Q_K`) isn't accepted --
resolving it to a gradient needs the reference particle's live
momentum, which this dataclass has no notion of; convert to a gradient
before constructing one.

**Profile form** (ASTRA `Q_type` naming a `...data...` file, two columns
`z - z_m` and gradient): `profile` names the file; `grad_T_per_m` is
then the profile's *peak* gradient (ASTRA rescales the table to it) unless
`noscale` -- the file's values are taken as T/m and `grad_T_per_m` is
ignored. The fringe follows the table with Maxwell's longitudinal component
(`magnets.profile_quad_element`); `length_m` is unused. `multipoles`
(`Q_mult_b`) maps an order `n >= 3` to `Q_grad * Q_mult_b(n)` -- only
`n = 3` (normal sextupole, `B_y = -C_3 (x^2 - y^2)`) has been checked
against ASTRA -- applied uniformly over the table's span. That is how an
ASTRA deck spells a sextupole: a negligible `Q_grad` and `Q_mult_b(3)`.

`x_m`/`xrot_rad` (`Q_xoff`/`Q_xrot`) place the element off-axis, as
for `Solenoid`: the frame is centred on `(x_m, z_m)`.

### `Aperture`

A particle-loss check: a transverse cut, active over
`[z1_m, z2_m]` (a native element, not an ASTRA one; `from_astra`
imports ASTRA's own `&APERTURE` keyword forms into it -- the
file-table/z-radius form and secondary-electron-emission `"CIR"`
grids aren't imported, warned when seen). `kind` is one of
`"RAD"` (circular), `"COL_X"`/`"COL_Y"` (double-sided planar
collimator -- the only form the bundled CLARA decks actually use),
`"SCR_X"`/`"SCR_Y"` (single-sided scraper). `r_m` is signed:
for `RAD`, negative makes a solid plug (particles *inside*
`|r_m|` lost) instead of a hole (particles *outside* lost); for
`SCR_X`/`SCR_Y`, negative scrapes the opposite side;
`COL_X`/`COL_Y` is a symmetric half-gap, sign irrelevant.
`x_m`/`xrot_rad` (`A_xoff`/`A_xrot`) place the slab off-axis: its
centre is `((z1_m + z2_m) / 2, x_m)` and it is rotated about that.
A particle is lost if the straight path of a step, clipped to the slab,
leaves the acceptance -- so the result does not depend on the time step,
however thin the slab (`z1_m == z2_m` is a plane).

### `Injector.from_astra`

Build from an ASTRA input deck (`&NEWRUN`, `&OUTPUT`,
`&CHARGE`, `&CAVITY`, `&SOLENOID`, `&QUADRUPOLE` (standard
and `...data...` profile types), `&DIPOLE` (horizontal,
`D_strength`), `&APERTURE` (keyword forms only)). Unit conversions:
`MaxE` MV/m, `Nue` GHz, `Phi` deg (relative to crest for
`Auto_phase=T`; absolute, in ASTRA's sine convention, for
`Auto_phase=F`), `C_pos`/`S_pos`/`Q_pos`/`zstop` m,
`MaxB`/`Q_grad` T/T-per-m, `Ap_R` mm -> m, `D1..D4` `(x, z)` m,
`*_xrot` rad. Solenoid/quadrupole/aperture `_xoff`/`_xrot` become
placements; dipole `D_xoff`/`D_zoff`/`D_xrot` are applied to the
corner points.
Everything present in the deck that this model ignores is listed in
one warning -- a silently dropped wake or dipole is worse than a
loud one.

`mc2_eV`/`charge_sign` (default: electron) are not in the deck
itself -- ASTRA infers species from the *distribution file*'s own
species-index column (only used there for the per-macroparticle
space-charge charge, see `astra_io.distribution_from_rows`), but
this tracker's single-particle Boris push needs one rest mass/sign
for the whole beam, so pass them explicitly for a non-electron
machine (e.g. `mc2_eV=PROTON_MASS_EV, charge_sign=+1.0` for a
proton distribution).

### `Injector.fields`

The phased cavity field objects (z-sorted) and solenoid field
objects, zero-amplitude elements dropped (ASTRA decks switch magnets
off with `MaxB=0`); solenoids with an `x_m`/`xrot_rad` placement
are left to `magnet_elements`. Needs `crest_deg` set -- see
`auto_phase`.

`params` overlays `track_differentiable`'s knobs (`"cavities[i].phase_deg"`,
`"cavities[i].e_max_V_per_m"`, `"solenoids[i].b_max_T"`, `i` the list index) onto the
built fields as tensors, so gradients reach them; the crest stays the nominal one.

### `Corrector`

A steering magnet: the uniform transverse field `bx_T`, `by_T` over `length_m` centred on `z_m` (`Injector.correctors`). It
kicks a particle of charge `q` by `dp_x = -q B_y L` and `dp_y = +q B_x L` whatever its energy, so settings that centre a
beam at one gun phase leave the slower electrons of another phase off axis. The strengths are tensors in the compiled
step: changing them does not recompile. The CLARA example sets the two S01 correctors to centre the beam on the BPM and
the screen at a reference phase (`examples/photocathode/schottky_scan.py`).

### `Injector.magnet_elements`

The lab-frame magnets `magnets.magnet_fields` evaluates: dipoles, quadrupoles that are
placed, profile-table or carry multipoles, placed solenoids and the steering `Injector.correctors` -- plus `placed_sol`, the
`(place, table)` pairs of the solenoids among them (for the canonical-emittance `Bz`).

`frame_all` (default: whether the machine has any dipole) also routes the *unplaced* solenoids
and quadrupoles through here, each in an identity frame that carries the lateral validity mask
(`magnets.LATERAL_LIMIT_M`): after a bend, a lab-`z`-indexed element must not act on a
particle that merely shares its `z` range on a different leg of the line. Otherwise only
elements that need it are returned and the rest keep `_fused_step`'s own faster tables.

### `Injector.auto_phase`

ASTRA's `Auto_phase` (manual sec. 4.8): scan each cavity in
turn, in z-order, for the phase of maximum on-axis energy gain
(upstream cavities already at their own crest, downstream ones off).
Stores `crest_deg` on each cavity and returns the per-cavity
energy gains (MeV) -- the same numbers ASTRA prints as "Cavity
number / Energy gain [MeV] at Phase [deg]". See
docs/tracking_native.md's Injector Usage section for the scan
algorithm and `parabolic_vertex` for the sub-grid refinement.

### `Injector.initial_beam`

The beam `track` starts from, at the cathode: `x, y, z, q, ux, uy, uz` (`u = gamma*v`,
m/s) and `emission_time_s` -- see `track`'s docstring for `n_particles` (a file subsample
is drawn from torch's global RNG). Build it once and pass it to `track_differentiable(beam=...)`
to hold the beam fixed across gradient evaluations.

### `Injector.track`

Track the distribution from the cathode to `z_stop_m` (default:
the machine's own). Returns endpoint scalars plus `"trajectory"`,
the along-z record (`TRAJ_KEYS` + `t_ns`) every `record_every`
steps -- same keys `astra_io.read_astra_emit` uses. Auto-phases
first unless the cached crests are still valid.

`dtype="float32"` is safe (compensated accumulation, see
`_fused_step`) and faster on a consumer GPU. `n_particles` below
the file's count subsamples at random with charge reweighting;
`None` tracks the file deterministically. With `self.generator` set
(an ASTRA `generator.in`), the beam is instead generated natively at
exactly `n_particles` (any count, up or down; `None` = the deck's
`Ipart`) with total charge `Q_total` -- see `tracking.native.generator`.

`verbose` (default on) shows a `tqdm` bar over the distance
travelled (mm to `z_stop_m`), with the step, time, active count and
dt range in its postfix; dump messages print above it.

`dump_dir` (default `None`: no file I/O) turns on ASTRA-style
beam dumps at every `self.screens` crossing plus the final
`z_stop_m` -- resolved via `self.resolve()`. Each dump writes a
full 6D macroparticle file in openPMD-beamphysics format
(`<dump_dir>/<stem>.<label>.h5`) plus one accumulated
`<stem>.twiss.h5` (simba-Twiss-compatible, written directly with
h5py so simba isn't a dependency here) and a copy of the machine
alongside them for reproducibility.

With `Injector.photocathode` set (see [`photocathode`](photocathode.md)) the macroparticle
charges come from the laser pulse energy and the Schottky-dependent quantum efficiency,
limited by the surface field of the charge already emitted, instead of from the beam file.
Macroparticles that cannot be extracted are dropped (`n_suppressed`), those that fall back
through the cathode plane are absorbed (`n_returned`), and `out["emission"]` reports the
ideal, emitted, returned and surviving charge. `track_differentiable` does not support it.
Trajectory statistics remain macroparticle averages, not charge-weighted.

### `Injector.track_differentiable`

`track` as one autograd graph: the same `_fused_step`, `_fused_sc_fields` and
`_traj_stats`, run for a fixed `n_steps` of `self.dt_s`, returning the endpoint
`TRAJ_KEYS` statistics as 0-d tensors that `backward()` through to `params`.

`params` maps knob names to (0-d tensor) values that replace the machine's own:
`"cavities[i].phase_deg"` (off crest; the crest stays the nominal `auto_phase` one),
`"cavities[i].e_max_V_per_m"`, `"solenoids[i].b_max_T"`, `"quadrupoles[i].grad_T_per_m"`
(`i` the list index; plain unplaced elements only), `"charge_scale"` (multiplies every
macroparticle charge) and `"spot_scale"` (multiplies every initial `x`, `y`: the cathode
spot scaled about the axis -- for the generator's radial/gaussian spots exactly a change of
`sig_x`/`sig_y`, the momenta being independent of it). `beam` (`initial_beam`'s dict) holds the beam fixed between
calls; without it one is built per call.

What makes the derivative well defined, and what `track` does that this does not -- see
docs/tracking_native.md's "Differentiable injector":

- a fixed step schedule: `n_steps` of `dt_s`, or `dt_schedule` (one `dt` per step,
  e.g. a nominal `track()`'s own `"dt_steps_s"` -- its adaptive steps replayed, not
  re-adapted); no stop at `z_stop_m`: the endpoint is a fixed *time* (`t_ns`), so e.g.
  the final `z_m` moves with the knobs;
- no `auto_phase` (a scan + argmax): stale crests raise instead;
- space charge on one cylindrical mesh shape (`space_charge.nr`/`nz`) whose box is sized
  from the bunch at every recompute as a smooth tensor function (`track()`'s sizing rules,
  no bucket or rebuild -- see `_differentiable_space_charge`), centred on the bunch -- or
  on the cathode, with the image charge, for the first `sc_image_steps` steps (a fixed
  schedule; a nominal `track()`'s own `"sc_image_steps"` replays where it dropped the
  image). `sc_box_m = (r_max_m, z_half_m)` (rest
  frame) pins the box instead; `"sc_box_overflow"` then reports the largest fraction of
  particles found outside it (clamped onto its edge). No 3D or bent-beam solve;
- apertures: `"n_lost_on_aperture"` is the exact hard-loss count, computed beside the
  graph; lost particles keep being tracked and stay in the statistics (a 0/1 loss has no
  useful derivative);
- piecewise smooth only: CIC cells and field-table samples are chosen by `floor`, so the
  derivative jumps where a particle crosses a cell edge.

`checkpoint_every` (steps per block, 0 = off) keeps only each block's input state in the graph
and recomputes the block during backward: graph memory ~`n_steps / checkpoint_every` states plus
one block, instead of every step (~255 B per particle-step in float32), for one extra forward.

`use_compile` compiles the per-step kernels (not the outer loop). `record_every`
adds `"trajectory"` (`TRAJ_KEYS` columns as tensors, plus `t_ns`); `return_particles`
the final state (all particles, plus the `alive` mask).

### Screen-plane observation

Both trackers accept `screen_z_m=<finite lab z in metres>`. The optional `out["screen"]`
records each emitted particle's first forward crossing, interpolated on the pusher's straight
step segment. It returns full-length `x`, `y`, `z`, `ux`, `uy`, `uz` (metres and proper
velocities in m/s), per-particle `t_s` (float64 seconds), signed `q` (C) and boolean
`crossed`. Unobserved coordinates/time and charge are zero; always use `crossed` when
selecting observations. The supplied macroparticle charge is retained exactly for observed
particles, including those lost downstream after crossing. A swept aperture check up to
the plane excludes losses before observation. Particles remain in the tracker after
crossing; the monitor does not remove bunch charge from space-charge deposition.

`z_plane_m`, `n_crossed`, `n_unobserved_alive`, `n_lost_before_screen` and
`incomplete_charge_C` distinguish observation, aperture loss and unfinished transport.
Missing observations are reported, never silently interpreted as physical transmission
loss. Require `n_unobserved_alive == 0` before treating a screen image as complete.
Backward, stationary, not-yet-emitted or initially downstream particles may remain
unobserved. No backward projection or new crossing is fabricated for them.

Ordinary tracking with this option stops after its centroid reaches `z_stop_m` **and**
every particle has crossed or been lost, subject to its existing maximum step count.
Default stopping is unchanged. Differentiable tracking retains its fixed schedule; select
or replay a schedule long enough over the complete fit bounds and check completeness.
The crossing state passes through checkpoint blocks, with no additional per-step host
synchronization. `use_compile` also compiles the crossing helper on CUDA.

This is a finite-step observation, not an exact event solver under within-step acceleration.
Converge the step size against the intended image/observable error budget. Derivatives
are piecewise smooth while the crossing and loss masks stay fixed; loss/crossing transitions
and emission-time comparisons have no useful gradient. Screen monitoring does not change
the differentiable tracker's existing treatment of aperture-lost particles in space charge
or endpoint statistics; use a loss-free fit domain or validate ordinary tracking separately.
The common-time endpoint and dump files remain unchanged and should not be confused
with this plane observation.

### `parabolic_vertex`

Sub-grid peak position from the three points bracketing index `i`
-- the gain is locally quadratic in phase at the crest, so this locates
the true peak well under the grid spacing. Returns the grid value
unchanged at a window edge or if the curvature degenerates. Matters
because the crest is exactly where `d(gain)/dphi=0`, so a phase error
there is invisible in the energy gain but first-order in the bunching
(1 deg at 2.9985 GHz = 0.93 ps = 0.28 mm of crest travel).

## Numerical and implementation notes

### `Module constants`

Image branch's own longitudinal cell reference (not SpaceChargeSettings.nz)
and torch.compile shape buckets -- see docs/tracking_native.md ("Grid/mesh
sizing", "torch.compile discipline").

### `Module constants`

Charge-adaptive default time step reference; see docs/tracking_native.md
("Injector time step vs space-charge strength") for the validation numbers
and why this is a separate constant from SpaceChargeSettings.dt_ceiling_s.

### `Module constants`

Handoff to a downstream s-based tracker (E3, BeamlineStudy.track_with_injector)

### `Module constants`

#### Machine description
### `SpaceChargeSettings`

"cylindrical" or "3d" (Cartesian open-boundary FFT, space_charge.SpaceCharge3D) -- see
docs/tracking_native.md's "3D space charge". Mesh/box/smoothing knobs above are cylindrical-only.

### `SpaceChargeSettings`

Deliberately its own field, not _DT_REF_S -- see docs/tracking_native.md's "Injector time step
vs space-charge strength". Defaults to the same value but is free to move independently.

### `Module constants`

conversion below derives from this instead of a hardcoded electron mass (e.g. PROTON_MASS_EV for
a proton beamline like ISIS's MEBT); default preserves this class's original electron-only behavior.

### `Module constants`

distribution file's own species column (ASTRA species index 1=electron -> -1 here, 3=proton -> +1)
-- only the per-macroparticle charge used for space-charge deposition is read from that column
directly, the single-particle Boris push uses this one sign for every particle.

### `Injector.initial_beam`

Status filter is resolved from the file itself, not assumed -- a real ASTRA
Cathode=F beamline (e.g. ISIS MEBT) has no status==-1 rows at all, its real
particles are status 5 instead (see cathode_emission_status's docstring).

### `Injector.initial_beam`

Already-in-flight beam (ASTRA Cathode=F): every real particle is active from the
very first step -- there is no cathode emission process to stagger. Real ASTRA's
own generator.exe still writes a nonzero "clock" column here (confirmed on ISIS
MEBT's laser.astra: clock ~= -z/v_ref, i.e. the same real longitudinal spread as
the z column, just expressed as an equivalent time-of-flight); treating it as an
activation gate anyway silently distorts the fixed-time ensemble statistics from
the very first recorded step -- measured directly: sigma_z was already 3.10mm vs
ASTRA's 1.83mm by the first cavity, before any cavity/space-charge had acted at
all, purely from this kinematic staggering artifact, then compounded to a 38x
final-sigma_z mismatch by the end of the line.

### `Injector.track`

Some macroparticles have emission_time_s < 0 (ASTRA's relative clock convention) -- start the
clock at the earliest real emission, not 0, or those particles are clipped to "already active".

### `Injector.track`

canonical-emittance solenoids, the bent-beam flag, and this beam's own species (mc2_eV/q_C) --
all four are positional-unpacked into _traj_stats/_twiss_row_from_ensemble's trailing params.

### `Injector.track`

Runaway-safety cap only (real stop condition is centroid_z >= z_target); generous vs. the
fixed-dt formula since dt_adaptive can sub-step the cathode region before growing to dt_s.

### `Injector.track`

"counted" = emitted AND not lost on an aperture (every statistics/SC/output site downstream
uses it); step_fn's own `active` stays emission-only. See docs/tracking_native.md's Aperture note.

### `Injector.track`

Time-based `not emitted_all`, not just the step-count `sc.every_step_until`: dt_adaptive can
shrink dt below its setup value during emission, and a step-count threshold alone would then
stop recomputing every step before emission has actually finished.

### `Injector.track`

Ceiling is dt_ceiling_s, not self.dt_s/_DT_REF_S; an unmeasurable omega leaves dt
untouched rather than resetting it -- see docs/tracking_native.md's "Injector
adaptive time-stepping" and "Injector time step vs space-charge strength".

### `Injector.track`

Raw final state at a common lab time t, real z-spread -- see docs/tracking_native.md's
Injector Usage section and injector_exit_to_canonical6 for the downstream-handoff convention.

### `Injector.track_differentiable`

One _fused_step variant per particle count x dtype x grad mode x emitting/emitted x whether the
state requires grad yet: an FD + autograd study at two sizes already passes dynamo's default 8,
past which it runs eagerly. Check the actual compiled path for the intended
shapes and settings; the performance penalty depends on the study.

### `Module constants`

Auto-phase helpers

### `Module constants`

Diagnostics (one compiled kernel per record event)

### `_traj_stats`

rms_emittance assumes zero-mean positions (its cross term is the raw <x x'>), true on a straight
line but not for a bunch far from the lab origin: centre them for the emittance calls only (the
canonical momentum below needs the position relative to the solenoid axis, i.e. uncentred).

### `Module constants`

Screen/endpoint dumps: openPMD-beamphysics beam files + a simba-Twiss-
compatible HDF5 (Injector.track(dump_dir=...))

### `Module constants`

#### The tracking step
### `_applied_fields`

Travelling-wave structures: the periodic section is stored once and
tiled; Ez = Er*cos(phase) - Ei*sin(phase), Ei from the Floquet
relation -- see fields.TravelingWaveField.

### `Module constants`

Compiled once at module scope, not per track() call: shapes depend only on particle/element counts
(fixed for a whole run), so this survives across calls instead of recompiling every time. See
docs/tracking_native.md's "Injector warp backend" for why this stays compiled even on the warp path.

### `Module constants`

#### Space charge
### `_fused_sc_fields`

sqrt's derivative is 0/0 = NaN on the axis -- where every generated beam's reference particle starts --
so take it only where r > 0: the same value everywhere, zero (sub)gradient at r = 0.

### `Module constants`

Compiled once at module scope, same reasoning as _applied_fields_compiled above -- see
docs/tracking_native.md's "Injector warp backend" for why this tail stays compiled even
when warp handles the deposit (_fused_sc_fields_warp, in the source).

### `_fused_sc_fields_3d`

Antisymmetric cathode image: mirror in z about z_ref_t, charge flipped -- z_ref_t sits exactly on
the grid's centre node (see _compute_space_charge_3d), so the deposit antisymmetry is exact.

### `_compute_space_charge_in_beam_frame`

The axisymmetric solver is centred on r = 0: recentre the bunch on its own axis (after a bend it sits
far from the lab origin -- the translation leaves the fields unchanged). z needs none (its box tracks
the bunch), and neither does y in practice, but the same shift keeps a vertically kicked bunch centred.

### `Module constants`

Every scalar the box sizing needs, stacked into ONE tensor and synced
in a single D2H transfer -- six serialized float(t) syncs cost more
per call than the compiled SC kernel itself.

### `Module constants`

r box = (actual max radius) + margin*sigma_r -- additive, not
r_max*margin (multiplicative lets one halo particle set the whole
grid's scale). See docs/tracking_native.md.

### `Module constants`

Image-branch domain: symmetric about the cathode, half-width reaching the mirror lobe at
+/-gamma*(z_centroid-cathode_z) plus the bunch's own margin -- grows with distance from the cathode.

### `Module constants`

The z box lives in the Lorentz-stretched rest frame, because that is
the coordinate the deposit uses -- sizing on the lab-frame extent
overflows the grid by ~gamma once relativistic (see docs/tracking_native.md).

### `Module constants`

Grid symmetric about the cathode plane, plane exactly on the centre node (odd node count):
the real+image deposit is then exactly antisymmetric, so phi(r, cathode)=0 exactly (ASTRA's
own documented treatment) -- mid-cell, a spurious radial field survives near the cathode.

### `Module constants`

Stretched-frame box: bunch extent + margin, plus 25% open-
boundary slack so the Hockney-padded FFT convolution sees
near-zero charge at the edges.
