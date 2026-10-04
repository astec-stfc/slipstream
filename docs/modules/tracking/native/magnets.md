# `slipstream.tracking.native.magnets`

[Source](../../../../src/slipstream/tracking/native/magnets.py) · [Documentation index](../../../README.md)

## Physics

Evaluates lab-frame static dipole, solenoid and quadrupole fields, including element offsets/rotations. Dipole fringes use finite-width Fermi profiles with off-midplane Maxwell corrections; positive dipole strength bends electrons toward +x.

### Limits and conventions

- Placement rotations are about the element center; fields must rotate back to the lab frame.
- Overlapping dipole fringes use a smooth combined profile rather than reproducing ASTRA's discontinuous splice.
- Beam-frame diagnostic rotation must not alter the lab transport frame.

### `Model and numerical conventions`

Everything here is a *static* magnetic field evaluated at the particle's lab
position `(x, y, z)` by `magnet_fields` -- the Boris pusher in
`injector._fused_step` is unchanged, it just sums these fields with the
cavity/solenoid/space-charge ones. That is what lets one integrator follow a
beam round a bend: the lab frame never changes, so the beam is not "switched"
between frames the way GPT's CCS chain or a curvilinear tracker does.

**Dipoles** are ASTRA's `&DIPOLE` model (manual sec. 6.13), measured against
`Astra.exe`: a horizontal magnet is the strip between two edge lines (through
`D1-D2` and `D3-D4`), field `B_y = D_strength * P` with a Fermi fringe
`F(d) = 1/(1 + exp(4 d / D_Gap))` around each edge (`d` the normal distance
outside it), truncated exactly as ASTRA does -- `F = 0` beyond ``d = 1.5
D_Gap` outside, `1` beyond `1.5 D_Gap`` inside -- and cut laterally at the
extrapolated side lines `D1-D3` / `D2-D4`. The two edges combine as
`P = F1 + F2 - 1`: for non-overlapping fringes (all ASTRA permits) that is
identical to ASTRA; for overlapping ones ASTRA splices the two edge profiles
with a discontinuity mid-magnet (measured: 3% too much bend at ``D_Gap =
0.08` on a 0.15 m magnet) while `F1 + F2 - 1`` -- also GPT's form -- stays
smooth and conserves the field integral, so this module deliberately does not
copy that artefact. Positive `D_strength` bends electrons toward `+x` (the
sign was measured; the manual only states it for `D_radius`). Off the
mid-plane the field follows Maxwell to second order in `y`: ``B_x = y B_0
dP/dx`, `B_z = y B_0 dP/dz`, `B_y = B_0 (P - y^2/2 laplacian(P))``.

**Quadrupoles from a profile table** (ASTRA `Q_type = '...data...'`): the
on-axis gradient `G(z)` (a peak-normalised table scaled so its maximum is
`Q_grad`, or taken literally with `Q_noscale`) with the Maxwell-consistent
fringe expansion `B = grad V`, `V = g(z) x y - g''(z)(x^3 y + x y^3)/12`,
`g = -Q_grad` (electron-focusing positive, like `injector.Quadrupole`).
ASTRA's `Q_mult_b(n)` multipoles ride on the same element as
`B_y + i B_x = -C_n (x + i y)^(n-1)`, `C_n = Q_grad * Q_mult_b(n)` -- again
measured, the manual's formula is garbled in print -- *uniform* over the table's
whole `z` span (not tapered by the profile; ASTRA ignores the multipoles'
longitudinal field, so does this).

## Usage

Build `Dipole` and packed magnet elements, evaluate `magnet_fields`, and use placement/to-local/to-lab helpers for explicit frame transforms. Profile gradients are tesla per metre; fields are tesla.

### API reference

```python
class Dipole:
    ...
def placement(x_m: float, z_m: float, angle_rad: float, force: bool=False): ...
def placement3(x_m: float, y_m: float, z_m: float, xrot_rad: float, yrot_rad: float, force: bool=False): ...
def to_local3(x, y, z, place): ...
def to_lab3(bx, by, bz, place): ...
def lateral_ok(xl, y): ...
def to_local(x, z, place): ...
def to_lab(bx_local, bz_local, place): ...
def lerp_tables(z, z0, dz, ntab, zend, *tables): ...
def dipole_element(d: Dipole) -> tuple: ...
def solenoid_element(map_f, map_df, map_d2f, z0, dz, ntab, zend, place) -> tuple: ...
def profile_quad_element(z: np.ndarray, profile: np.ndarray, grad_T_per_m: float, noscale: bool, multipoles: dict[int, float], place, device, dtype, n_samples: int=4001) -> tuple: ...
def hard_quad_element(grad_T_per_m: float, z0: float, zend: float, place) -> tuple: ...
def magnet_fields(x, y, z, elements): ...
def beam_angle(ux, uz): ...
def rotate_to_beam(x, z, ux, uz, theta): ...
```

### `Dipole`

One ASTRA horizontal dipole: the four lab-frame `(x, z)` corners
`D1..D4` (metres; `D1-D2` is the entrance edge, `D3-D4` the exit edge,
`D1` and `D3` on the same side), the field `strength_T` (`D_strength`;
positive bends electrons toward `+x`) and the two edge fringe parameters
`D_Gap(1)`/`D_Gap(2)`. A magnet whose gap-scaled fringes would overlap is
fine here (see the model description) -- but superpose dipoles if you need
ASTRA-compatible input, as `examples/RUEDI/ASTRA` does.

### `placement`

`(x_m, z_m, cos, sin)` of an element frame centred at `(x_m, z_m)` in
the lab, its axis rotated `angle_rad` from `+z` toward `+x` -- or
`None` when that is the identity (so unplaced elements cost nothing and
stay bit-identical to before placement existed). `force` returns the
identity frame explicitly instead, which is how an unplaced element opts in
to the lateral validity mask (`lateral_ok`) on a bent line.

### `placement3`

`placement` in 3D, for solenoids: rotate about the element centre `(0, 0, z_m)` by
`R = R_x(-yrot_rad) R_y(xrot_rad)` -- `xrot_rad` from `+z` toward `+x` (as `placement`),
then `yrot_rad` from `+z` toward `+y` -- and move the centre to `(x_m, y_m)`. Returns
`(x_m, y_m, z_m, R)` with `R` row-major, or `None` for the identity (unless `force`).

### `magnet_fields`

Sum of every listed element's magnetic field at the lab points
`(x, y, z)`: `(B_x, B_y, B_z)`. `elements` is a tuple of
`(kind, place, payload)` -- a compile-time constant, so the loop unrolls
inside `torch.compile`.

## Numerical and implementation notes

### `Module constants`

Paraxial element fields (solenoid/quadrupole tables) are only meaningful near their own axis. In a
framed element (one that has a `place`) a particle further than this from the element axis gets no
field: a lab-z-indexed table must not fire on a distant leg of a bent line that shares its z range.

### `Module constants`

#### Frames
### `Module constants`

Element tables: (kind, place, payload) tuples `magnet_fields` consumes

### `profile_quad_element`

g = -Q_grad: electron-focusing positive. Sign-preserving peak scale (see fields._scale_to_peak's
own docstring) -- a bipolar profile whose true extremum sits on the negative lobe would otherwise
silently flip this quadrupole's sign, the same bug already found and fixed for RF cavities.

### `Module constants`

#### Fields
### `Module constants`

Beam frame (bent beamlines)
