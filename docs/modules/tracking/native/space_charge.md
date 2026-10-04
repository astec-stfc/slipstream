# `slipstream.tracking.native.space_charge`

[Source](../../../../src/slipstream/tracking/native/space_charge.py) · [Documentation index](../../../README.md)

## Physics

Cylindrical PIC deposits/gathers with CIC and combines longitudinal open-boundary FFT treatment with radial finite differences. The radial axis uses the conservative `4*(phi1-phi0)/dr^2` stencil and matching control volume. Cartesian 3D PIC uses an integrated Green function; cathode images mirror position and reverse charge.

### Limits and conventions

- Solvers return electrostatic fields; the caller adds relativistic boosting and magnetic self-fields.
- Keep axis stencil and deposit volumes consistent.
- Open-boundary FFT padding must avoid circular-image wraparound.
- Use axis-reflected/outer-zero smoothing and avoid compiled odd-grid concatenation alignment failures.

### `Model and numerical conventions`

Returns pure **electrostatic** `(Er, Ez)`. The beam's own magnetic
self-field (from longitudinal motion, `B = (beta/c) x E`) is applied by
the caller once a per-particle/per-bunch beta is known -- not this
module's concern, so it stays testable purely against closed-form
electrostatics.

Radial discretization (node-centered, `r_i = i*dr` for `i=0..nr-1`,
Dirichlet `phi=0` at the implicit ghost node `r=nr*dr`): the standard
second-order conservative (flux-form) finite difference for
`(1/r) d/dr(r dphi/dr)` (Birdsall & Langdon, *Plasma Physics via
Computer Simulation*), with the `r=0` axis via the L'Hopital limit
`(1/r) d(r dphi/dr)/dr|_{r=0} = 2 d2phi/dr2|_0`, discretized as
`4*(phi_1 - phi_0)/dr^2`. `node_volume[0]` uses the matching half-disk
volume `pi*(dr/2)^2*dz` -- the flux-form stencil and the deposit's
density normalization share the same control volumes.

The cathode's conducting-plane boundary condition needs no new solver
code: `field_with_cathode_image` below is the method of images -- a real
(`z`-reflected, charge-flipped) mirror macroparticle set, deposited
through this same solver (matching ASTRA's own `Lmirror=T`).

### `_radial_laplacian_matrix`

`(nr, nr)` matrix `L` such that `(L @ phi)[i]` approximates
`(1/r) d/dr(r dphi/dr)` at node `r_i = i*dr`. Dirichlet
`phi=0` is assumed at the implicit ghost node `r_nr = nr*dr`, so
there is no explicit `nr`-th column -- its (known-zero) contribution
simply drops out of the last row.

## Usage

Use `CylindricalSpaceCharge` or `SpaceCharge3D` to deposit, solve and gather fields. `field_with_cathode_image`, `space_charge_3d` and coordinate helpers compose boundary handling; `smooth_rho` controls grid noise.

### API reference

```python
def smooth_rho(rho: torch.Tensor, passes_r: int, passes_z: int) -> torch.Tensor: ...
class CylindricalSpaceCharge:
    ...
    def __init__(self, nr: int, nz: int, r_max: float, z_min: float, z_max: float, device: str | torch.device | None=None, dtype: torch.dtype | None=None): ...
    def coefficients(self, dr, dz): ...
    def rescale(self, r_max: float, z_min: float, z_max: float) -> None: ...
    def deposit(self, r: torch.Tensor, z: torch.Tensor, q: torch.Tensor) -> torch.Tensor: ...
    def solve(self, rho: torch.Tensor, smooth_r: int=0, smooth_z: int=0, gamma: float=1.0, smooth_gamma_ref: float=math.inf) -> tuple[torch.Tensor, torch.Tensor]: ...
    def gather(self, grid: torch.Tensor, r: torch.Tensor, z: torch.Tensor) -> torch.Tensor: ...
    def field_at(self, r: torch.Tensor, z: torch.Tensor, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...
def cartesian_field_at(solver: CylindricalSpaceCharge, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, q: torch.Tensor, smooth_r: int=0, smooth_z: int=0, gamma: float=1.0, smooth_gamma_ref: float=math.inf) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
def field_with_cathode_image(solver: CylindricalSpaceCharge, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, q: torch.Tensor, cathode_z: float, smooth_r: int=0, smooth_z: int=0, gamma: float=1.0, smooth_gamma_ref: float=math.inf) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
def integrated_green(ox: torch.Tensor, oy: torch.Tensor, oz: torch.Tensor, hx: float, hy: float, hz: float, far_cells: float=20.0) -> torch.Tensor: ...
class SpaceCharge3D:
    ...
    def __init__(self, nx: int, ny: int, nz: int, hx: float, hy: float, hz: float, x_lo: float, y_lo: float, z_lo: float, device: str | torch.device | None=None, dtype: torch.dtype | None=None, far_cells: float=20.0, max_grid_cells: int=SC3D_MAX_GRID_CELLS): ...
    def rescale(self, x_lo: float, y_lo: float, z_lo: float) -> None: ...
    def x_hi(self) -> float: ...
    def y_hi(self) -> float: ...
    def z_hi(self) -> float: ...
    def deposit(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, q: torch.Tensor) -> torch.Tensor: ...
    def solve(self, charge: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    def gather(self, grid: torch.Tensor, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor: ...
    def field_at(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
def space_charge_3d(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, q: torch.Tensor, n_cells: tuple[int, int, int], box: tuple[float, float, float, float, float, float], mirror_z: float | None=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
```

### `smooth_rho`

Binomial `(1,2,1)/4` smoothing of a deposited `(nr, nz)` charge-
density grid, `passes_r` times along r then `passes_z` times along
z -- the same shot-noise-suppression filter ASTRA applies via its own
`&CHARGE` `smooth_x`/`smooth_y`/`smooth_z` (collapsed to one
radial pass count here since this solver is axisymmetric, not separate
x/y). `passes_r`/`passes_z` are plain ints (0 = off, matching every
pre-existing config's behavior exactly) so this stays a compile-time-
constant Python loop, not traced control flow -- safe to call from
inside a `torch.compile`'d function.

`r=0` is the symmetry axis, not a domain edge: the "ghost" node at
`r=-dr` equals the physical density at `r=+dr` (node 1), not a
replicated `r=0` value -- reflecting there keeps the stencil centered
on a real node like every other pass. Every other boundary (`r_max`,
both `z` ends) is zero-padded, which is accurate here specifically
because the caller sizes the box with margin so density is already
~0 at those edges.

Padding is `F.pad` (+ an explicit axis-ghost row), not `torch.cat`:
under `torch.compile` on CUDA, Inductor writes a cat's pieces into
one buffer at odd float offsets when `nz` is odd (every
`SC_IMAGE_NZ_BUCKETS` entry) but still tells Triton the pointers are
16-byte aligned -- `CUDA error: misaligned address` at the first
call for that mesh shape (traced 2026-09-24, nr=64 nz=513).

### `CylindricalSpaceCharge`

Axisymmetric (r, z) electrostatic space-charge solver on a fixed
grid. Build once per run: the radial-solve matrices depend only on the
grid, not on particle data, so precomputing them once keeps repeated
`solve()` calls cheap.

The grid's *shape* (nr, nz) is fixed at construction, but its physical
box can be moved/resized in place with `rescale()` -- the cheap
operation an adaptive-mesh driver calls to re-fit the grid to the bunch
every subcycle, without reconstructing the whole solver.

### `CylindricalSpaceCharge.coefficients`

`(node_volume, inv_diag)` for cells of `dr` x `dz` on this `(nr, nz)` shape --
`_set_grid`'s formulas, written so `dr`/`dz` may be 0-d tensors carrying gradients (a box
sized from the bunch itself, `injector._differentiable_space_charge`): `node_volume` scales
as `dr^2 dz` and every `k` as `1/dz`. `P_left`/`P_right` do not depend on the box.

### `CylindricalSpaceCharge.rescale`

Move/resize the grid's physical box in place, keeping nr, nz
(hence the deposited-grid shape, the FFT length, and every
`torch.compile` graph that captured them) unchanged. Numerically
identical to constructing a fresh solver at the new box -- the
discretization is re-derived from the same stencil, so this is the
fast path for adaptively re-fitting the grid every solve.

### `CylindricalSpaceCharge.deposit`

Cloud-in-Cell deposit of macroparticle charges (Coulombs,
`r`/`z` in metres) onto the `(nr, nz)` node grid, returned
as a charge *density* (C/m^3) -- raw charge is deposited with
charge-conserving linear weights first, then each node's
accumulated charge is divided by its own geometric dual-cell
volume (`node_volume`). Particles are assumed to stay within
`[0, r_max)` -- one past the last radial node saturates onto
it rather than being lost or erroring, a standard PIC assumption
the caller satisfies by sizing the grid with margin.

### `CylindricalSpaceCharge.solve`

Poisson solve: charge-density grid `(nr, nz)` -> `(Er, Ez)`
field grids `(nr, nz)`. `smooth_r`/`smooth_z` (see
`smooth_rho`) are blended in with weight
`1 / (1 + (gamma/smooth_gamma_ref)**2)` -- the same relativistic
`1/gamma**2` suppression this solver's own lab-frame boost
applies to the net space-charge force, so the smoothing fades out
exactly where the force it would bias is fading out too. Not a
stored/cached setting (unlike the grid itself): `gamma` changes
every call, so this is a per-call blend, not solver state.

### `field_with_cathode_image`

Space-charge field including the cathode's conducting-plane
boundary condition, via the method of images: a mirror macroparticle
for every real one, reflected in `z` about `cathode_z` with charge
sign flipped, deposited into the same grid as the real particles (so
`solver`'s `z`-range must cover both the real and image lobes with
margin -- caller's responsibility, e.g. `z_min < 2*cathode_z - z.max()`).
Superposition on a linear solve makes this exact -- no new
Poisson-solver code. Returns `(Ex, Ey, Ez)` at the *real* particles'
own positions only (the image particles are deposit-only). `smooth_r`/
`smooth_z`/`gamma`/`smooth_gamma_ref` -- see `solve`.

### `integrated_green`

Cell-averaged `1/r` kernel on the (broadcast) offset grid `(ox, oy, oz)`: ``int_cell dV / |r - r'|
/ (hx hy hz)`` for a uniformly charged cell centred at the offset -- exact for any cell aspect ratio
(a point-sampled `1/r` is badly wrong for a pancake bunch's flat cells). Past `far_cells` x the
largest cell size the plain `1/r` is used instead (error `~(h/r)^2`, and the 8-corner sum would
cancel away float64 there). Evaluate in float64.

### `SpaceCharge3D`

Persistent Cartesian `(x, y, z)` electrostatic space-charge solver:
the stateful, cacheable form of `space_charge_3d` (same IGF kernel,
same Hockney FFT convolution), following the `CylindricalSpaceCharge`
discipline -- build the expensive grid-dependent object once, then move
the box with `rescale()`.

What makes the Green's function cacheable where the stateless API
rebuilds it every call: `G` depends only on the cell sizes
`(hx, hy, hz)` and the grid shape, so the cell sizes are FIXED at
construction (chosen by the caller from a quantized bucket ladder, see
`injector._compute_space_charge_3d`) and `rescale()` only moves the
box origin. A change of bucket means a fresh solver (`G` rebuild,
amortised); an in-bucket move is sub-ms.

The cathode image is the antisymmetric deposit, not a shifted Green's
function: the caller deposits mirror macroparticles
(`z_img = 2*z_ref - z`, charge flipped) through the same CIC, which
needs only the ONE direct kernel cached here and gives `phi = 0` at
the mirror plane exactly when the plane sits on the centre node (odd
`nz`, with the caller centring the grid at the cathode). See
`field_with_cathode_image`/`_fused_sc_fields` for the cylindrical
precedent.

Convolves raw deposited CHARGE (Coulombs per node) with the cell-
averaged kernel -- the second-order-correct pairing of a piecewise-
constant charge density with the IGF (see `integrated_green`), so
`deposit()` returns charge, unlike `CylindricalSpaceCharge`'s
density.

### `SpaceCharge3D.deposit`

Cloud-in-Cell deposit of macroparticle charges onto the
`(nx, ny, nz)` node grid, returned as CHARGE per node (see the
class docstring). Particles outside the box saturate onto the edge
cells (clamp), as in `space_charge_3d` -- the caller sizes the
box with margin.

### `SpaceCharge3D.solve`

Charge grid `(nx, ny, nz)` -> `(Ex, Ey, Ez)` field grids
(V/m): Hockney convolution with the cached cell-averaged kernel,
central-difference gradients (one-sided at edges, matching
`torch.gradient(edge_order=1)`).

### `space_charge_3d`

Electrostatic `(Ex, Ey, Ez)` of macroparticles `(x, y, z, q)` at their own positions: CIC
deposit on an `n_cells = (nx, ny, nz)` node grid spanning ``box = (x_lo, x_hi, y_lo, y_hi, z_lo,
z_hi)`` (particles outside saturate onto the edge cells, so size it with margin), open-boundary
Hockney FFT convolution with the integrated Green's function, central-difference fields, trilinear
gather. The box only has to cover the bunch -- the grid is rebuilt every call, so no state.

`mirror_z`: a conducting plane at `z = mirror_z` (the cathode), by the method of images through a
*shifted* Green's function (Qiang et al. 2006, as IMPACT-T): the image charge is the deposited grid
flipped in `z` with sign reversed, sitting on a mirrored grid, so its potential is one more
convolution with `G` evaluated at the displaced offsets -- exact at any bunch-cathode distance
without stretching the grid to cover the image lobe (GPT's `spacecharge3Dmesh("Cathode")` does the
same job).

## Numerical and implementation notes

### `CylindricalSpaceCharge`

Hockney zero-padding: at least double the z-domain so the
periodic FFT convolution reproduces an open boundary in z.
Rounded up to a power of two -- not just free, but required:
2*nz can otherwise land on a length with a large prime factor
(e.g. 514, 16382), which sends cuFFT onto the slow Bluestein
algorithm (

### `CylindricalSpaceCharge`

Eigendecomposition of the radial Laplacian, done once per nr (independent of dr/dz) -- see
docs/tracking_native.md's "CylindricalSpaceCharge's radial Poisson solve" for the derivation.
P_left/P_right are grid-independent; every rescale() rebuilds only the (nr, n_freq) diagonal.

### `CylindricalSpaceCharge._set_grid`

A^-1 = P_left @ diag(inv_diag) @ P_right, per z-frequency (see
__init__). All eigenvalues are negative (Dirichlet at r_max), so
this never divides by zero, even at k=0.

### `CylindricalSpaceCharge.solve`

P_right/P_left act on the radial axis only, so they commute with
the z-FFT: apply them as real matmuls outside it, leaving just an
elementwise complex scale inside.

### `Module constants`

3D Cartesian solver
