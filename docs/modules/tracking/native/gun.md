# `slipstream.tracking.native.gun`

[Source](../../../../src/slipstream/tracking/native/gun.py) · [Documentation index](../../../README.md)

## Physics

Uses the relativistic Boris pusher in Cartesian `(x,y,z,ux,uy,uz)`, with proper velocity `u=p/m=gamma*v` in m/s. Quasi-static space charge is solved in the stretched bunch rest frame and boosted back, including the magnetic self-field.

### Limits and conventions

- This Cartesian state is not the canonical six-column ring/linac state.
- Magnetic-only Boris rotation conserves proper-velocity magnitude.
- Apply the boosted electric and self-magnetic fields once, in the caller's physical frame.

### `Model and numerical conventions`

`ux, uy, uz` are the *proper velocity* `u = p/m = gamma*v` (m/s) -- the
standard relativistic-Boris state variable: a pure rotation preserves
`|u|` (hence `gamma`) exactly, so `gamma` is evaluated once per step
rather than solved for implicitly.

**Pusher algorithm**: the non-relativistic base algorithm is Qin, Zhang,
Xiao & Tang, "Why is Boris algorithm so good?", Phys. Plasmas 20, 084503
(2013), Eqs. 3, 5-7 (energy, hence `gamma`, is conserved exactly when
`E=0`, checked directly by this module's tests). The relativistic
extension (`u=p/m` in place of `v`, `gamma` evaluated once from
`u_minus` before the rotation) is the standard relativistic-Boris
extension (Boris, 1970).

## Usage

Use `boris_push` for one step or `track_particles` with a supplied field function. `relativistic_space_charge_field` couples an electrostatic solver; `kahan_add` compensates small updates.

### API reference

```python
def kahan_add(u: torch.Tensor, a: torch.Tensor, c: torch.Tensor): ...
def boris_push(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor, uz: torch.Tensor, q_C: torch.Tensor | float, mass_kg: float, Ex: torch.Tensor, Ey: torch.Tensor, Ez: torch.Tensor, Bx: torch.Tensor, By: torch.Tensor, Bz: torch.Tensor, dt: float, comp: tuple[torch.Tensor, ...] | None=None): ...
def track_particles(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor, uz: torch.Tensor, q_C: torch.Tensor | float, mass_kg: float, field_fn: FieldFn, t0: float, dt: float, n_steps: int, use_compile: bool=True): ...
def relativistic_space_charge_field(solver: CylindricalSpaceCharge, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, q: torch.Tensor, bunch_gamma: float, cathode_z: float | None=None, smooth_r: int=0, smooth_z: int=0, smooth_gamma_ref: float=math.inf): ...
```

### `kahan_add`

One compensated (Kahan) accumulate `u += a`, returning
`(u_new, c_new)`.

`c` carries the part of `a` that `u`'s mantissa could not hold.
The next call adds `a - c` instead of `a`, so increments far below
`ulp(u)` are not discarded -- they accrue in `c` and flush into
`u` once they exceed half an ulp. `c` sits at magnitude ~ulp(u),
where its own resolution is ~1e-7 of that, which is where the extra
~7 decimal digits of effective precision come from.

`c` is state: carry it across steps beside `u`. A freshly zeroed
`c` each step compensates nothing.

### `boris_push`

One relativistic Boris step (SI units throughout: m, m/s, V/m, T,
kg, C, s). `q_C` may be a scalar or an `(N,)` tensor of signed
per-particle charges; every other argument is `(N,)`.

`comp` opts into **compensated (Kahan) momentum accumulation** -- in
float32, the difference between reproducing the physics and silently
deleting it once the per-step increment `a=(q/m)*E*dt` falls below
`u`'s ulp (see docs/tracking_native.md).

Pass a 6-tuple of per-particle compensation tensors (zeros to start,
ordered `(cx, cy, cz, cux, cuy, cuz)`); they are returned updated as
a seventh element, fed back in next step as state. Positions need it
for the same reason: the differential advance between particles (the
longitudinal physics) is far below `z`'s own ulp per step. In this
mode the push is also rearranged to form the step's whole change in
`u` from small terms before adding it once, instead of two half-kicks
around a rotation -- algebraically identical, fewer chances to round a
sub-ulp increment away. `None` (default) keeps the original
uncompensated behaviour and return signature.

### `track_particles`

Advance particles `n_steps`, evaluating
`field_fn(x, y, z, t) -> (Ex,Ey,Ez,Bx,By,Bz)` once per step at the
start-of-step position/time (the primary-source algorithm's own
`E_k = E(x_k)`, `B_k = B(x_k)` convention). Returns the final
`(x, y, z, ux, uy, uz)`.

`use_compile=True` (default) drives the step through the injector's
own compiled `_fused_step` (the same function object, not a copy --
shared rather than duplicated). The injector's applied-field tables are
empty here: every applied field arrives through `field_fn`, evaluated
once per step outside the compiled region (the injector's own split
with its space-charge cache), so the compiled kernel is the
relativistic Boris push alone with Kahan-compensated state -- same
result as `boris_push`, differing only in float rounding order.
`use_compile=False` is the original eager `boris_push` loop.
Unlike `Injector.track` there is no `device == "cuda"` gate: this
path compiles wherever asked (CPU compilation is a parity feature here,
not a speed one -- the fused kernel can be slower per step than eager
on CPU).

### `relativistic_space_charge_field`

Space-charge self-field with the quasi-static relativistic boost
(see model description): stretch `z` by `bunch_gamma` about a
fixed reference (`cathode_z` if given, else the bunch's own z-mean
-- the choice of reference point is immaterial, since it cancels in
every particle-pair separation the field actually depends on),
solve electrostatics in `solver`, then
transform the result back to the lab frame. `solver`'s grid must
be sized to hold the *stretched* z-extent (and, with a cathode, the
stretched image lobe too) -- caller's responsibility, same as
`field_with_cathode_image`'s own grid-sizing note. `smooth_r`/
`smooth_z`/`smooth_gamma_ref` -- see `CylindricalSpaceCharge.solve`;
`bunch_gamma` doubles as that blend's `gamma`, already the exact
quantity the fade needs.

## Numerical and implementation notes

### `boris_push`

Same update, assembled as an increment of small terms and added
once with compensation (see the docstring). `ux_plus` etc. are
only used here to recover the rotation's contribution, which is
`u_plus - u_minus`; forming it directly from the cross products
keeps every term small.

### `track_particles`

Lazy import: injector.py imports this module at module level
(kahan_add, relativistic_space_charge_field), so a top-level import of
_fused_step here would be circular.

### `track_particles`

Mirror of Injector.track's step_fn gate (injector.py:498): compile the
shared step rather than duplicating it. No recompile_limit guard here --
that guard exists because _fused_sc_fields takes (nr, nz) grid shapes
that change between solver rebuilds (injector.py:487-497), and this
driver has no space-charge mesh at all (any SC lives inside the
caller's field_fn, evaluated eagerly): the only tensor shapes in the
call are the fixed (n,) particle arrays, so one graph serves the run.

### `track_particles`

Kahan-compensated state, positions AND momenta -- the same layout
Injector.track carries across steps (injector.py:464); it is what
makes the compiled path float32-safe where the eager loop above needs
the caller to opt in via boris_push(comp=...).

### `track_particles`

Same 0-d float64 device-tensor time convention as Injector.track
(injector.py:479); unused with empty sw/tw/sol (no element phase to
evaluate) but part of the shared signature, so keep the convention.

### `track_particles`

No emission model in this driver -- every particle is live from step
0, so pass emitted_all=True and _fused_step skips the
where(active,...) merge (it is a Python bool, fixed per variant).
