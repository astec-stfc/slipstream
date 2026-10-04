# `slipstream.tracking.native.warp_kernels`

[Source](../../../../src/slipstream/tracking/native/warp_kernels.py) · [Documentation index](../../../README.md)

## Physics

Optional CUDA kernels accelerate Boris pushes, selected field evaluations, apertures, charge deposition/gathering, line-density histograms and CSR pairwise sums. PyTorch retains the authoritative physical implementation and CPU path.

### Limits and conventions

- These are forward-only zero-copy kernels; do not imply autograd coverage.
- Keep Warp typed literals compatible with float64 specialization.
- Optional Warp absence must preserve PyTorch correctness.

### `Model and numerical conventions`


Every kernel is written once as a *generic* (`Any`-dtype) warp kernel
and instantiated for float32/float64 via `wp.overload` rather than as a
hand-duplicated f32/f64 pair ('s spike did the latter, flagged in
its own Finding 4 as worth fixing). Measured directly: both overloads of a
generic kernel compile from *one* module, vs. two independent modules for
two hand-written kernels -- fewer compiled artifacts, less first-use
latency, one body to keep in sync with the torch reference instead of two.
Warp's literal typing is still strict inside a generic body (a bare
`1.0` defaults to float32 and cannot mix with a float64 operand --
confirmed empirically, `Any`/`wp.overload` doesn't change this), so
typed `0`/`1`/`2` constants are derived from an already-typed
argument instead (`push`: `ONE = dt / dt`, safe since `dt` is always
physically nonzero; `_apertures_kernel`: passed in explicitly as
`zero`/`one` params instead, since nothing in its own inputs is
guaranteed nonzero the way `dt` is).

### `_push_kernel`

Relativistic Boris push, Kahan-compensated -- transcribed from
`injector._fused_step`'s push tail. See that function's docstring
for why the compensation is physically load-bearing (float32
ulp-cancellation at CLARA-gun-exit momenta).

### `_deposit_kernel`

CIC scatter-add of `q` onto the flat `(nr, nz)` charge grid's
four surrounding corners -- native `wp.atomic_add` in place of
`injector._fused_sc_fields`'s four chained `scatter_add_` calls,
the kernel found warp winning on at every N. Cell indices
and CIC weights are precomputed in torch (`sc_deposit` in the source) and
passed in, not recomputed here: `injector.py` needs the same
`r0`/`wr`/`z0`/`wz` again for the post-solve gather, so
computing them once in vectorized torch is strictly less work than
computing them a second time on-device in the kernel.

### `_deposit3d_kernel`

Direct 3D CIC deposit from particle coordinates.

Computing the eight indices/weights here avoids materialising and
filling two `(8, N)` torch tensors before this launch. The cathode
image, when requested, is deposited in the same launch.

### `_apertures_kernel`

`injector._apply_apertures`'s swept AND-accumulation loop, one
kernel launch instead of one eager torch-op sequence per aperture.
Transcribed from `_aperture_hit`/`_swept_aperture_hit`/
`magnets.to_local` -- see those docstrings for the geometry.
`kind`: 0=RAD 1=COL_X 2=COL_Y 3=SCR_X 4=SCR_Y (`injector.py`'s
own `_aperture_tables` order). A RAD aperture with `r < 0` is a
solid plug (excluded region is the disc, needs the extra
closest-approach check); `r >= 0` is a hole (RAD/COL) or a
one-sided cut (SCR). `zero`/`one` are typed 0/1 constants
(warp's literal typing is strict even in a generic kernel, see
`push_kernel`'s docstring -- passed in from the wrapper rather
than derived from an array element here, since nothing in this
kernel's own inputs is guaranteed nonzero the way `push`'s
`dt` is).

### `_sw_kernel`

Standing-wave cavities: on-axis Ez0(z) + paraxial Er/Bphi
expansion, transcribed from `injector._applied_fields`'s `sw`
loop. `cos_p`/`sin_p` (one value per cavity) are precomputed in
torch (`injector._rf_phases`) and passed in rather than computed
per-particle here: the RF phase is the *same* scalar for every
particle at a given step, so computing it once in torch (as the
existing code already did) and reading it N times is strictly
cheaper than N per-thread transcendental evaluations. Adds onto
whatever `Ex..Bz` already hold (accumulator, not overwrite) --
callers chain `_sw_kernel`/`_tw_kernel`/`_sol_kernel` in
sequence onto the same buffers.

### `_tw_kernel`

Travelling-wave structures: the periodic section is stored once
and tiled; `Ez = Er*cos(phase) - Ei*sin(phase)`, `Ei` from the
Floquet relation -- transcribed from `injector._applied_fields`'s
`tw` loop (see `fields.TravelingWaveField`). `cos_t`/`sin_t`
precomputed per structure, same reasoning as `_sw_kernel`.

### `_sol_kernel`

Solenoids: on-axis Bz0(z) + paraxial Br/Bz expansion --
transcribed from `injector._applied_fields`'s `sol` loop. No
`Ex`/`Ey`/`Ez` params: a solenoid has no E field, so there's
nothing for those to do here (also saves 3 unneeded `wp.from_torch`
wraps per call vs. carrying them through unused).

### `_csr_pairwise_kernel`

Exact one-sided softened CSR sum, one thread per observer.

This is the same operation as `collective._csr_chunk_step` without
materialising a row-block `(rows, N)` temporary. Every observer
still visits every source, so the algorithm remains O(N^2); Warp only
changes the memory traffic and reduction implementation.

### `_aperture_arrays`

`apertures` (`injector._aperture_tables`'s output: a tuple of
`(kind, z1, z2, r[, (x_m, z_m, cos, sin)])` per aperture, all plain
Python values) packed into the fixed-length arrays `apply_apertures`
launches over. Cached on `(apertures, device, dtype)` -- `apertures`
is itself a plain tuple of plain values, so it's hashable and fixed for
a whole run, exactly like `sw`/`tw`/`sol` are for
`_applied_fields_compiled`: build these arrays once, not every step.

### `_cat_tables`

Concatenate one or more per-element 1D tables (`table_indices` into
each `entries[i]` tuple) into flat arrays plus an `int32` offset
array -- the ragged-array packing `_sw_kernel`/`_tw_kernel`/
`_sol_kernel` index with `offset[c] + i0`.

## Usage

Dispatch through `available(device, dtype)` and the native tracker wrappers. Kernels share the active PyTorch CUDA stream and support their documented float32/float64 variants.

### API reference

```python
def available(device: str, dtype: torch.dtype) -> bool: ...
def csr_pairwise_sum(tau: torch.Tensor, score: torch.Tensor, softening: torch.Tensor) -> torch.Tensor: ...
def cic_line_density(bin_idx: torch.Tensor, interp_w: torch.Tensor, valid: torch.Tensor, charge_per_particle: torch.Tensor, n_bins: int, dtau: torch.Tensor) -> torch.Tensor: ...
def wake_histogram(safe_idx: torch.Tensor, n_bins: int, dtype: torch.dtype) -> torch.Tensor: ...
def push(x, y, z, ux, uy, uz, cx, cy, cz, cux, cuy, cuz, Ex, Ey, Ez, Bx, By, Bz, qmdt2: float, dt: float, c2_inv: float): ...
def push_inplace(x, y, z, ux, uy, uz, cx, cy, cz, cux, cuy, cuz, Ex, Ey, Ez, Bx, By, Bz, qmdt2: float, dt: float, c2_inv: float): ...
def sc_deposit(r0: torch.Tensor, z0: torch.Tensor, wr: torch.Tensor, wz: torch.Tensor, q: torch.Tensor, nr: int, nz: int) -> torch.Tensor: ...
def sc_fields_3d(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, q: torch.Tensor, bunch_gamma: torch.Tensor, z_ref: torch.Tensor, beta: torch.Tensor, x_lo: torch.Tensor, y_lo: torch.Tensor, z_lo: torch.Tensor, hx: torch.Tensor, hy: torch.Tensor, hz: torch.Tensor, nx: int, ny: int, nz: int, px: int, py: int, pz: int, G_k: torch.Tensor, with_image: bool, inv_c: float): ...
def apply_apertures(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, x0: torch.Tensor, y0: torch.Tensor, z0: torch.Tensor, alive: torch.Tensor, apertures: tuple) -> torch.Tensor: ...
def applied_fields(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, sw: tuple, tw: tuple, sol: tuple, cos_sw: torch.Tensor, sin_sw: torch.Tensor, cos_tw: torch.Tensor, sin_tw: torch.Tensor, c2_inv: float): ...
```

### `csr_pairwise_sum`

Run the exact one-sided CSR pairwise sum with one Warp thread/observer.

`tau` and `score` are one-dimensional, same-dtype CUDA tensors;
`softening` is a scalar CUDA tensor. The scalar remains device-side so
this wrapper does not introduce a host synchronization into a collective
kick. This is an accelerator primitive, not a replacement for the
authoritative PyTorch/autograd path.

### `cic_line_density`

Deposit fixed-grid CIC line charge and return `lambda = charge/dtau`.

The index/weight arrays are deliberately supplied by the caller: this is
the same split used by the existing fixed-grid implementation, and lets
the benchmark isolate the atomic deposition work from grid construction.

### `wake_histogram`

Deposit unit counts for the fixed-window short-range wake histogram.

`safe_idx` includes the extra trash-bin index used by
`ShortRangeWake.voltage_at`. It must already use the reference path's
rounding and clamping semantics; this kernel only replaces the atomic
histogram update.

### `push`

Boris-push tail of `injector._fused_step` as one warp kernel
launch. `qmdt2`/`dt`/`c2_inv` are plain Python floats, not 0-d
tensors: unlike `torch.compile(dynamic=False)`, a warp kernel never
recompiles on a changed scalar *value* (only on argument *type*), so
`_fused_step`'s tensor-indirection trick to dodge that has no
equivalent cost here and would just add an unneeded host readback.

### `push_inplace`

Same physics as `push`, but writes the twelve state tensors back
into themselves instead of allocating twelve fresh output tensors.
Safe: the kernel only reads a given particle's slot of `x`/`cx`/etc.
*before* it writes that same slot (checked directly against the kernel
body above), so aliasing input and output buffers is race-free and
gives a bit-identical result to `push` -- confirmed empirically, not
just by inspection. This matters because `push` alone measured
surprisingly far from 's isolated-kernel numbers once wired into
a real per-step loop: `wp.from_torch` wrapping and fresh
`torch.empty_like` allocation, both negligible in a benchmark that
hoists them outside the timed loop (`benchmark/warp_port/bench.py`),
are *not* negligible paid fresh every step (measured ~0.27 ms/call at
N=50,000 float32 vs. this function's ~0.15 ms -- allocation plus half
the wrapping gone).

Only safe to use when the caller has no further use for the pre-step
values (`_fused_step_warp` only calls this when `emitted_all`,
where there's no old/new masking blend that needs them -- see there
for the `emitted_all=False` case, which still needs `push`'s
allocating, non-aliased version).

### `sc_deposit`

CIC scatter-add half of `injector._fused_sc_fields`'s deposit, as
one warp kernel launch. `r0`/`z0` (int64, already clamped to
`[0, nr-2]`/`[0, nz-2]`) and `wr`/`wz` (the bilinear weights)
are `injector.py`'s own already-computed CIC coordinates -- see
`_deposit_kernel`'s docstring for why they're passed in rather than
recomputed. Returns the flat `(nr*nz,)` charge array.

### `apply_apertures`

`injector._apply_apertures`'s swept AND-accumulation check as one
warp kernel launch instead of one chained eager torch-op sequence per
aperture -- see `_apertures_kernel`'s docstring for the geometry.
Measured as the single largest eager-dispatch cost once the two hot
kernels moved to warp (the 5pC CLARA deck's 8 apertures dominated the
whole warp-backed step, ~4.3 of ~4.8 ms/step, before this existed).
Mutates and returns `alive` in place (same in-place-is-safe reasoning
as `push_inplace`: each thread only ever reads then writes its own
`alive[i]`, never another thread's).

### `applied_fields`

`injector._applied_fields`'s `sw`/`tw`/`sol` loops (cavities,
TW structures, solenoids -- *not* `quad`/`magnets`, see
`injector._applied_fields_warp`'s docstring for why those stay on the
`torch.compile` path) as up to three warp kernel launches instead of
one `torch.compile` graph. Each kernel adds onto the same `Ex..Bz`
buffers in sequence (a solenoid only ever touches `Bx`/`By`/`Bz`).
`cos_sw`/`sin_sw`/`cos_tw`/`sin_tw` are one value per element,
precomputed in torch by `injector._rf_phases` -- see `_sw_kernel`'s
docstring for why (the RF phase is one scalar per element per step, not
per-particle; computing it in the kernel would redundantly repeat a
transcendental evaluation once per particle instead of once).

## Numerical and implementation notes

### `_csr_pairwise_kernel`

The reference reduction is a tree reduction while this
kernel is source-serial per observer. Compensated
summation keeps float32 parity from degrading with N.

### `Module constants`

Packed-array cache for sw/tw/sol, keyed by id() of the tuple injector.py
builds once per Injector.track() call and never mutates -- (id, packed)
pairs, with a strong ref to the *original* tuple kept alongside the
packed tensors so its id can never be reused by a different object while
the cache entry is alive (the usual safety condition for id()-keyed
memoization). unbounded -- grows by one entry per distinct
sw/tw/sol tuple built in this process; fine for how this is used (one
entry per Injector.track() call), add an eviction policy if that stops
being true.
