# `slipstream.tracking.native.collective`

[Source](../../../../src/slipstream/tracking/native/collective.py) · [Documentation index](../../../README.md)

## Physics

Implements 1D longitudinal space charge (LSC) and steady-state 1D CSR momentum kicks. LSC is the free-space on-axis field of a uniform cylinder or, when a pipe radius is given, the long-wavelength pipe formula. CSR is directional in late-positive tau. KDE, block-multipole and fixed-grid paths trade cost, smoothing and approximation.

### Limits and conventions

- LSC changes `delta` only, with one field per longitudinal position: the on-axis value of a uniform cylinder of radius
  `radius_factor` times the mean rms transverse size, for ultrarelativistic motion. There is no transverse variation. The
  radius factor is a model parameter, as in elegant's `LSCKICK`.
- `pipe_radius_b_m` selects the long-wavelength formula with `F = 1 + 2 ln(b/a)`. It holds while `b` is much smaller than
  `gamma` times the length of the density structure and overestimates the field beyond that. Leave it `None` for free space.
- `LinacTracker` applies one LSC kick per element, at its exit. Split long drifts along which the bunch or the dispersion
  changes.
- The smoothing of the line density limits the resolution: see [`lsc_delta_kick_fast`](#lsc_delta_kick_fast).
- Pairwise CSR contains no entrance transient, shielding or drift propagation. The fixed-grid path adds an entrance transient
  (`slippage_m`), a closed-form drift model (`csr_drift_delta_kick_fast`) and a retarded-field drift kernel
  (`csr_drift_delta_kick_retarded`); shielding by a conducting pipe is not modelled (the shielding parameter
  `sigma_z sqrt(rho) / h^(3/2)` of the CLARA chicane is below 0.1 at full compression, an estimate to verify).
- Sharp profiles require convergence checks on bandwidth, softening and bin/smoothing parameters.
- Preserve the directional sign: reversing sign and source direction can conceal pointwise error in symmetric mean-loss checks.

### `Model and numerical conventions`

Mesh-free kernels: a Gaussian KDE of the line charge density `lambda(tau)`
(and its derivative), evaluated at every particle's own position, shared by
both physics kernels (`_gaussian_kde_density_and_deriv`). Fixed-grid
alternatives (`lsc_delta_kick_fast`, `csr_steady_state_delta_kick_fast`,
bottom of this module) trade accuracy for O(N) cost at large N, opt-in via
`LinacTracker(fast_collective_effects=True)`.

`tau_m` is this package's native `tau` (meters, path-length-like,
`tau>0`=tail/late, `tau<0`=head/early -- see `elements.py`).

### `_kde_window`

Particle-count window for the truncated KDE: negligible contribution
(`exp(-0.5*k^2)`) beyond `+-k*h` of a Gaussian bunch's peak density.
A function of `n` only (static under `torch.compile`), never a live
tensor value. See docs/tracking_native.md for the derivation.

### `_csr_chunked_pairwise_sum`

CSR's all-vs-all sum (no compact support to window against), row-
chunked the same way as the KDE's windowed branch. `compile_chunks`
(default off) routes each chunk through the compiled step instead of
eager. When a differentiated input is live (`_grad_live`), the chunk
steps use out-of-place expressions, with a finite argument where the
one-sided mask is false so the masked gradient is not `0 * inf`; the
in-place, storage-reusing versions remain for forward-only calls.

### `_grad_live`

True when grad mode is on and any tensor argument requires grad. Every
forward-only fast path in this module (the manually captured CUDA graphs,
the in-place chunk steps) is taken only when it is false: a replayed graph
or an in-place chain records no autograd history, and without this test the
result would silently have no gradient (graph) or fail in backward (in
place).

### `_graph_replay_ok`

`not _grad_live(...)` and not inside an outer CUDA graph capture. The four sites that replay a
manually captured forward-only graph (`csr_steady_state_delta_kick_fast`, `lsc_delta_kick_fast`,
the pairwise CSR sum and the windowed KDE) test this, not `_grad_live` alone: replaying a graph
inside another capture is illegal (`cudaErrorStreamCaptureUnsupported`) and, with a scalar bunch
charge, made a forward-only whole-iteration capture fail. In either case the caller takes its
ordinary PyTorch path, which the outer capture records and autograd differentiates.

### `_scalar_on` and `_const`

`_const(value, like)` is a 0-d tensor with `like`'s dtype and device: a tensor is converted, a
Python number comes from `_scalar_on`, an `lru_cache`d constructor. Building a tensor from a float
on CUDA is a host-to-device copy, which a CUDA graph capture rejects; the warm-up call pays it and
later calls (including inside a capture) look it up. Cached tensors are shared: never modify one in
place. Used for the CSR softening, CSR length, transient slippage and the Stupakov drift geometry.
`csr_drift_delta_kick_retarded` raises a `RuntimeError` when called during a capture, because it
reads the reference energy on the host and passes it to its kernels as a constant.

### `_CSRPairwiseGraph`

CUDA graph for the exact CSR chunk loop at one fixed particle count.

The graph changes only the copied `tau`, KDE score, and softening value;
chunking and the compiled row-block body stay fixed. This preserves the
pairwise model exactly while removing Python dispatch between chunks.

### `_KDEWindowedGraph`

CUDA-graph capture of `_kde_windowed_scalar_charge` for one fixed
`(n, w, dtype, device, compile_chunks)` -- collapses the ~15-20 small
eager kernel launches per call into one `graph.replay()`. Safe to
reuse across elements/passes: the op sequence depends only on the fixed
shape params, never on the live tensor values (those are copied into
the graph's own input buffers fresh before every replay). See
docs/tracking_native.md.

### `_gaussian_kde_density_and_deriv`

`(lambda(tau_i), dlambda/dtau(tau_i))` for every particle `i`,
from a Gaussian-kernel density estimate over *every* particle (including
`i`). Below `_kde_window(n)` particles: exact `(N, N)` broadcast.
Above it: truncated to each particle's own window of nearest neighbors
(O(N*w) instead of O(N^2)) -- row-chunked (`_chunk_rows_for`) to bound
peak memory at large N. `charge_per_particle_C` may be scalar (the
real-caller case) or a full per-source tensor (test-only, probes extra
zero-charge points).

`@torch._dynamo.disable()`: this function's own Python loop over a
static range is exactly what dynamo unrolls into the traced graph, and
at large N (hundreds of chunks) that compile never finishes in
reasonable time -- disabled here since every call site already reuses
one compiled *outer* graph across many calls anyway. See
docs/tracking_native.md.

### `_csr_block_multipole_sum`

Candidate subquadratic CSR sum using sorted block monopoles/dipoles.

Sources in the observer's block and a small number of following blocks
are evaluated exactly. More distant blocks use their score moments in a
first-order Taylor expansion of the softened kernel. This is intentionally
an opt-in candidate: the block-neighborhood heuristic still needs the
section-5 parity matrix before it can replace the exact reference.

### `_fixed_grid`

Fixed `n_bins`-bin CIC grid over this call's live `tau_m` window,
padded by `margin`. Returns ``(bin_idx, interp_w, valid, tau_min,
dtau, n_bins)`: `bin_idx`` is the left of each particle's two
bracketing bin centers (clamped into `[0, n_bins-2]`); `interp_w`
is the linear weight on the right bin. `valid` is `False` for
particles outside the window -- deposit/gather zero these out rather
than clamping onto the edge bin (see docs/tracking_native.md, "trash-
bin" semantics). `tau_min`/`dtau` are 0-d tensors so this stays
`torch.compile`-safe across calls with different live windows.

### `_smooth_bins`

Binomial `(1,2,1)/4` smoothing, `passes` times (same convention
as `space_charge.smooth_rho`) -- suppresses the CIC histogram's shot
noise before it gets amplified by `1/dtau` in the derivative step.
`passes` is a plain Python int (static under `torch.compile`). See
docs/tracking_native.md for why the default is in the hundreds, not one.

For `passes >= 8` on a 1D grid of at most 4096 bins the passes are applied
as one matrix-vector product with `_dense_smoother(n, passes, dtype,
device)`: the same operator, not an approximation. The smoother with
zero padding is a Dirichlet tridiagonal Toeplitz matrix, so its power is
`Q diag(cos(pi k / (2 (n + 1)))**(2 passes)) Q` with
`Q[j, k] = sqrt(2 / (n + 1)) sin(pi j k / (n + 1))`, built once in float64
and cached per `(n, passes, dtype, device)` (16 MiB in float32 at 2048
bins). It matches the pass loop to 1e-15 in float64. At 16384 particles
and 2048 bins it replaces ~60-100 ms (210-245 ms with backward) of
sequential launches with ~1.5-2 ms (4-6 ms with backward). The first call
for a key builds the matrix, so do it outside a CUDA-graph capture; any
earlier call, including the warm-up passes, does.

### `_LSCFastGraph`

CUDA-graph capture of `_lsc_fast_field`, for one fixed ``(n, n_bins,
smooth_passes, dtype, device, pipe_radius_b_m, radius_factor)``: its
many small dependent launches become one `graph.replay()`, as in
`_KDEWindowedGraph` above. **Never keyed on anything element- or
lattice-specific** (`length_m` and the energy are applied after `run()`
returns or copied into the graph's inputs), so one graph is reused by
every LSC-active element of every lattice that shares the key. The graph
is forward-only: `lsc_delta_kick_fast` does not use it when a
differentiated input requires grad, and the same tensor operations then
run eagerly and are differentiable. The pipe form factor divides a
Python float by a tensor so that nothing is copied from host to device
inside a capture.

### `_free_space_lsc_field`

The field of the free-space model on the grid. In the rest frame a disk
of radius `a` with unit charge has the on-axis field
`(sign(v) - v / sqrt(v^2 + a^2)) / (2 pi eps0 a^2)` at distance `v`.
The line density is linear between bin centres (the shape of the
cloud-in-cell deposit), so the field at a bin centre from a source bin is
that disk field integrated against a triangle. It is the second
difference, over bin offsets, of the closed form
`asinh(t) + t exp(-|asinh(t)|)` with `t = v / a`, divided by
`4 pi eps0` times the rest-frame bin width. No long-wavelength
approximation or Bessel function is needed, and the result does not
depend on the bin width being smaller than `a / gamma`.

The kernel is built in float64 because the second differences cancel;
the FFT convolution with the density, which also applies the smoothing,
runs in the particle dtype. Float32 agrees with float64 to 3e-5 of the
peak field or better for 2048 and 8192 bins. The cost is O(B log B) for B
bins, in addition to the O(N) deposit and gather.

### `_csr_fast_field`

Histogram -> smoothed derivative -> zero-padded forward CSR convolution.

Integrates a linearly interpolated
derivative analytically over the singular first interval. Its weights
are 0.9 at the observer, 1.1 at the next bin, then j**(-1/3), all
multiplied by dtau**(2/3). This includes the near-cell contribution
omitted by a strictly off-diagonal Riemann sum. Zero padding prevents
a periodic wake from wrapping across the bunch.

## Usage

Use `lsc_delta_kick`, `csr_steady_state_delta_kick`, its subquadratic variant or fast fixed-grid counterparts; ordinary callers enable these through `LinacTracker(lsc=True, csr=True)`. Inputs use tau/length/radius in metres, energy in eV and particle charge in coulombs.

### API reference

```python
def silverman_bandwidth_m(tau_m: torch.Tensor) -> torch.Tensor: ...
def lsc_delta_kick(tau_m: torch.Tensor, charge_per_particle_C: torch.Tensor | float, x_m: torch.Tensor, y_m: torch.Tensor, energy_eV: torch.Tensor, mc2_eV: torch.Tensor, length_m: float, pipe_radius_b_m: float | None=None, radius_factor: float=1.7, compile_chunks: bool=False, n_bins: int | None=None, smooth_passes: int | None=None) -> torch.Tensor: ...
def csr_steady_state_delta_kick(tau_m: torch.Tensor, charge_per_particle_C: torch.Tensor | float, rho_m: float, energy_eV: torch.Tensor, length_m: float, softening_m: float | torch.Tensor | None=None, compile_chunks: bool=False) -> torch.Tensor: ...
def csr_steady_state_delta_kick_subquadratic(tau_m: torch.Tensor, charge_per_particle_C: torch.Tensor | float, rho_m: float, energy_eV: torch.Tensor, length_m: float, softening_m: float | torch.Tensor | None=None, tolerance: float=0.001) -> torch.Tensor: ...
def lsc_delta_kick_fast(tau_m: torch.Tensor, charge_per_particle_C: torch.Tensor | float, x_m: torch.Tensor, y_m: torch.Tensor, energy_eV: torch.Tensor, mc2_eV: torch.Tensor, length_m: float, pipe_radius_b_m: float | None=None, radius_factor: float=1.7, n_bins: int=2048, smooth_passes: int | None=512) -> torch.Tensor: ...
def csr_steady_state_delta_kick_fast(tau_m: torch.Tensor, charge_per_particle_C: torch.Tensor | float, rho_m: float, energy_eV: torch.Tensor, length_m: float, softening_m: float | torch.Tensor | None=None, n_bins: int=256, smooth_passes: int | None=None, sg_halfwidth: int=1) -> torch.Tensor: ...
```

### `lsc_delta_kick`

1D LSC kick, `delta` only: `delta += E_z(tau) * length_m / energy_eV`,
where `E_z` is positive toward the head (decreasing `tau`), so the head
gains energy and the tail loses it. `a` is the live rms beam radius times
`radius_factor` (elegant's `RADIUS_FACTOR=1.7` default).

- `pipe_radius_b_m=None` (default): **free space**. `E_z` is the on-axis
  field of a uniform cylinder of radius `a`, valid at every wavelength
  ([`_free_space_lsc_field`](#_free_space_lsc_field)). It is computed by
  `lsc_delta_kick_fast` on 2048 bins with the Gaussian kernel of
  Silverman's bandwidth, the density estimator of the pipe formula below.
  `n_bins` and `smooth_passes` replace the bin count and, with binomial
  passes, the smoothing; both raise `ValueError` with a pipe radius.
  A batch of bunches `(..., n)` gets one grid and radius per bunch.
- With a pipe radius `b`: the long-wavelength formula
  ``E_z = Z0 * F * c / (4 pi gamma^2) * dlambda/dtau``,
  ``F = 1 + 2 ln(b/a)``, from the pairwise Gaussian kernel density.

`compile_chunks` (default off, pipe formula only, forwarded to
`_gaussian_kde_density_and_deriv`) trades some speed for memory at large
N.

Validation: `tests/test_collective_kicks.py` compares the free-space kick
with the Coulomb integral of a uniform cylinder with a Gaussian line
density, which uses neither the grid nor an impedance formula (CPU,
float64, 3000 particles, 35 MeV to 1 GeV, 20 to 300 um rms length):
within 1 % for the fast defaults and 6 % for the Silverman paths.

### `csr_steady_state_delta_kick`

Steady-state 1D CSR, using a KDE-score pairwise quadrature.

In our tau=c*t convention. The positive prefactor multiplies d(lambda)/d(tau).
Reversing both the direction and sign preserves mean loss for a
symmetric bunch but gives the wrong pointwise wake.

The kernel is Plummer softened, defaulting to half the Silverman KDE
bandwidth. The finite nearest-neighbor KDE window can bias sharp
profiles; this estimator is not an exact continuum reference.
No entrance transient, shielding, or drift propagation is included here.

### `lsc_delta_kick_fast`

Fixed-grid LSC in O(N + B log B): a cloud-in-cell deposit on `n_bins`
bins spanning the live bunch, `smooth_passes` binomial smoothing passes,
then the free-space convolution or, with a pipe radius, the smoothed
central-difference derivative, and a linear gather.
`smooth_passes=None` (free space only) smooths with a Gaussian of
Silverman's bandwidth instead. `LinacTracker` passes its `lsc_bins` and
`lsc_smooth_passes` to these two arguments.

No smoothing suits every bunch. Root-mean-square error of the free-space
kick, relative to its rms, on 2048 bins at 16 384 / 131 072 / 1 048 576
particles (CPU, float64):

| Smoothing | Gaussian, 20 um | Gaussian, 400 um | Compressed bunch with a sharp spike |
| --- | --- | --- | --- |
| 32 passes | 0.68 / 0.21 / 0.07 | 2.0 / 0.57 / 0.17 | 0.20 / 0.22 / 0.24 |
| 512 passes (default) | 0.24 / 0.07 / 0.02 | 0.37 / 0.11 / 0.03 | 0.60 / 0.62 / 0.65 |
| Silverman | 0.11 / 0.06 / 0.03 | 0.13 / 0.08 / 0.04 | 0.93 / 0.87 / 0.78 |

The Gaussian references are the Coulomb integral (190 MeV, rms transverse
size 100 um, three samples). The compressed bunch is a chicane exit at
full compression (rms length 19 um) against 1 048 576 particles on 16 384
bins with 16 passes. Check convergence in smoothing and particle number
where a spike matters.

Against elegant 2026.3.0 on a four-dipole compressor (CSR off, 100 pC,
131 072 particles, CPU float64, LSC in the same drifts, kicks every
0.01 m), the projected horizontal emittance at chicane factors 0.8,
0.925 and 1.0 is 0.483, 0.714 and 0.567 um with the fast defaults and
0.483, 0.710 and 0.537 um with Silverman smoothing; elegant with 100 to
1600 `LSC_BINS` gives 0.483 to 0.485, 0.712 to 0.715 and 0.572 to
0.578 um.

Less smoothing removes most of the gap at factor 1.0. With the default
path and `lsc_smooth_passes` of 32, 128 and 512 on 2048 bins the
emittance is 0.577, 0.574 and 0.567 um (0.573, 0.571 and 0.564 um with
524 288 particles, where Silverman smoothing gives 0.546 um). At factor
0.925 the same settings give 0.714 to 0.716 um.

`@torch._dynamo.disable()` (like the reference kernels). On CUDA with a
scalar charge (every real `LinacTracker` caller) the field computation is
captured once into a CUDA graph ([`_LSCFastGraph`](#_lscfastgraph)) and
replayed. `length_m` is applied after replay, in plain eager Python.
The cached tensors a graph reads (the grid constants and smoothers) are
never evicted, so each distinct bin count and number of passes keeps its
own memory until the process ends.

Gradients: the graph records no autograd history, so it is used only when
no differentiated input requires grad (`_grad_live`). With `tau_m`,
`x_m`, `y_m` or the charge requiring grad the same operations run eagerly
and the kick is differentiable, with a scalar charge, on every device;
the derivative is that of the discretised model, including the grid
origin and spacing, which follow the live minimum and maximum of `tau_m`.
The float32 gradient of a binned density derivative is sensitive to small
errors in `tau`; for CSR keep about 128 particles per bin and see the
[measured limits](../../../tracking_native.md#gradients-through-a-linac-pass).

### `csr_steady_state_delta_kick_fast`

Elegant-style steady-state CSR in O(N + B log B) time and O(N+B) memory.
Differentiable on every device with a scalar charge: the forward-only CUDA
graph is bypassed when `tau_m` (or `slippage_m`) requires grad, see
`lsc_delta_kick_fast`.

Uses CIC deposition, degree-one Savitzky-Golay histogram smoothing,
a central derivative, and near-cell-corrected trapezoidal
integration. The default is unsoftened; the integrable singularity is
handled analytically. An explicit positive softening retains Plummer
regularization. Explicit smooth_passes selects binomial smoothing;
otherwise sg_halfwidth applies. No adaptive host-side tier selection.

LinacTracker supplies the source dipole's BINS/SG_HALFWIDTH, including
for its approximate drift tail. Without `slippage_m` this is the steady-state wake.

With `slippage_m = rho * phi**3 / 24` (`phi` the bend angle accumulated at the *centre* of the slice) it is the entrance transient of
Saldin, Schneidmiller and Yurkov as in elegant's CSRCSBEND without STEADY_STATE: the derivative sum runs over sources within the slippage
length and the density term `(lambda(s + L_s) - lambda(s + 4 L_s)) / L_s^(1/3)` is subtracted. It is checked against the retarded-field
reference of `Stories/CLARA_CSRValidation/lw_reference` (5.6% waveform error at the full 0.1325 rad of a VBC dipole, bunch-mean loss
within 1%; the steady kernel alone is 18% off at half that angle). Below about two histogram bins of slippage the transient is unresolved.
`LinacTracker(csr_transient=True)` sets it for every slice of every CSR dipole (fast kernels only), at the slice centre angle: the wake rises
steeply over the first slippage length, and evaluating it at the end of each slice overestimates the loss of an 8-slice VBC dipole by 8%
(0.991 of the retarded-field ideal with the centre rule against 1.077 with the end rule; elegant's own slicing is 1.087 at 8 slices and 1.010 at 64).

### `csr_drift_delta_kick_fast`, `csr_drift_delta_kick_retarded`

Both drift kernels evaluate their per-step wake (the retarded field, or the root of Stupakov's equation) in one fused Warp launch when Warp is installed (CUDA or CPU, float64);
the PyTorch bisections remain the fallback and the reference (`_use_warp=False`).

CSR in the drift behind a dipole of radius `rho` and angle `phi`, at distance `z` from its exit, for one kick of length `length_m`.
`csr_drift_delta_kick_fast` is the closed-form model of Stupakov used by elegant's CSRDRIFT (USE_STUPAKOV): it is 3-6% above the retarded
field of a Gaussian bunch near the exit and 17-22% above it at 0.5-1 m. `csr_drift_delta_kick_retarded` tabulates the point-charge kernel
from the retarded Lienard-Wiechert field (velocity and acceleration terms, bend minus infinite straight line), bin by bin, and convolves it with
the line density; it reproduces the independent reference to 0.2% in mean loss rate from 2 mm to 1 m. It costs about ten times more per kick.
`LinacTracker(csr_drift_model="stupakov" | "retarded")` tracks every CSR drift in kicks, one wake evaluation per kick at the centre of the step: uniform
`csr_drift_dz` (0.01 m, elegant's `DZ`) for `"stupakov"`, and for `"retarded"` graded steps from 0.005 m close to the exit up to 0.1 m, because that
kernel is smooth in the distance; `"overtaking"`, the default, keeps the earlier lumped steady-state kick cut at the overtaking length.

## Numerical and implementation notes

### `Module constants`

Row-block cap for every chunked pairwise buffer below, in elements
(dtype-agnostic). Bounds peak memory at large N without changing values
(each row's result is independent of chunk grouping) -- see
docs/tracking_native.md.

### `Module constants`

The Warp CSR kernel removes the materialised row-block, but has the same
O(N^2) work. Below this size its launch/setup cost is not consistently
recovered. Float64 CSR deliberately stays on the PyTorch path: on the
supported consumer GPU its Warp kernel is slower, while the Warp wrapper
still exposes float64 for explicit benchmarking and opt-in callers.

### `_gaussian_kde_density_and_deriv`

The real hot path: uniform, 0-d-tensor charge (linac.py's
_charge_per_particle_C) -- captured as one CUDA graph instead of
re-dispatching op-by-op from Python (bit-identical values).

### `_csr_block_multipole_sum`

monopole+dipole far field; adaptive higher moments are the
upgrade if the measured tolerance is insufficient for a profile.

### `Module constants`

Fast, fixed-bin-grid LSC/CSR

### `_bin_line_density`

_fixed_grid currently returns int64 indices for native torch
scatter_add_. Warp's compact index type is sufficient here and
avoids changing the established CPU/reference representation.
