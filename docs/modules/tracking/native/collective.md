# `slipstream.tracking.native.collective`

[Source](../../../../src/slipstream/tracking/native/collective.py) · [Documentation index](../../../README.md)

## Physics

Implements local long-wavelength 1D LSC and steady-state 1D CSR momentum kicks. LSC uses the line-density derivative and the `1/gamma^2` suppression, with optional pipe logarithm. CSR is directional in late-positive tau. KDE, block-multipole and fixed-grid paths trade cost, smoothing and approximation.

### Limits and conventions

- LSC is a local model rather than elegant's full nonlocal LSCDRIFT implementation.
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

CUDA-graph capture of the fixed-bin-grid LSC kernel's tensor math,
for one fixed ``(n, n_bins, smooth_passes, dtype, device,
pipe_radius_b_m, radius_factor)` -- collapses `smooth_passes``
(default 512) sequential dependent tiny kernel launches into one
`graph.replay()`, the same trick `_KDEWindowedGraph` above already
uses for the reference kernel. **Never keyed on anything
element/lattice-specific** (`length_m`/`energy_eV` are applied
after `run()` returns, in plain eager Python -- see
`lsc_delta_kick_fast`) -- so one graph is reused across every
LSC-active element of every lattice sharing this key: a true
compile-once block. The graph is forward-only: `lsc_delta_kick_fast`
does not use it when a differentiated input requires grad, and the same
tensor math then runs eagerly and is differentiable. The pipe form
factor divides a Python float by a tensor so that nothing is copied from
host to device inside a capture.

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
def lsc_delta_kick(tau_m: torch.Tensor, charge_per_particle_C: torch.Tensor | float, x_m: torch.Tensor, y_m: torch.Tensor, energy_eV: torch.Tensor, mc2_eV: torch.Tensor, length_m: float, pipe_radius_b_m: float | None=None, radius_factor: float=1.7, compile_chunks: bool=False) -> torch.Tensor: ...
def csr_steady_state_delta_kick(tau_m: torch.Tensor, charge_per_particle_C: torch.Tensor | float, rho_m: float, energy_eV: torch.Tensor, length_m: float, softening_m: float | torch.Tensor | None=None, compile_chunks: bool=False) -> torch.Tensor: ...
def csr_steady_state_delta_kick_subquadratic(tau_m: torch.Tensor, charge_per_particle_C: torch.Tensor | float, rho_m: float, energy_eV: torch.Tensor, length_m: float, softening_m: float | torch.Tensor | None=None, tolerance: float=0.001) -> torch.Tensor: ...
def lsc_delta_kick_fast(tau_m: torch.Tensor, charge_per_particle_C: torch.Tensor | float, x_m: torch.Tensor, y_m: torch.Tensor, energy_eV: torch.Tensor, mc2_eV: torch.Tensor, length_m: float, pipe_radius_b_m: float | None=None, radius_factor: float=1.7, n_bins: int=2048, smooth_passes: int=512) -> torch.Tensor: ...
def csr_steady_state_delta_kick_fast(tau_m: torch.Tensor, charge_per_particle_C: torch.Tensor | float, rho_m: float, energy_eV: torch.Tensor, length_m: float, softening_m: float | torch.Tensor | None=None, n_bins: int=256, smooth_passes: int | None=None, sg_halfwidth: int=1) -> torch.Tensor: ...
```

### `lsc_delta_kick`

Local (long-wavelength) 1D LSC kick, `delta` only: ``E_z(tau) =
Z0/(4*pi*gamma^2) * F * c * dlambda/dtau(tau)`, `F = 1 + 2*ln(b/a)``
(`a` = live rms beam radius x `radius_factor`, elegant's own
`RADIUS_FACTOR=1.7` default; `b` = pipe radius, `None` -> free
space, `F=1`). Deliberately **not** real elegant's own nonlocal
`LSCDRIFT`/`LSCKICK` formula (histogram+FFT, no pipe radius) -- see
docs/tracking_native.md for the model choice, the frequency-domain
cross-check and why a live-elegant comparison only agrees to order of
magnitude, not tight rtol.

Sign/normalization convention (`delta -= V(tau)/energy_eV`) matches
`ShortRangeWake.voltage_at`'s. `compile_chunks` (default off,
forwarded to `_gaussian_kde_density_and_deriv`) trades some speed for
memory at large N

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

Fixed-bin-grid approximation of `lsc_delta_kick` -- same formula,
conventions and citations; only `dlambda/dtau`'s estimator differs
(CIC histogram + binomial smoothing + central differences in place of
the reference's Gaussian KDE, O(N) instead of O(N*w)). See
docs/tracking_native.md for the fidelity-loss discussion and measured
agreement against the reference. `n_bins`/`smooth_passes` are the
grid-only knobs this approximation adds -- see docs/tracking_native.md
for why `smooth_passes` needs to be in the hundreds, not one, to
match the reference kernel's own implicit smoothing.

`@torch._dynamo.disable()` (like the reference kernels): the
`smooth_passes`-deep sequential dependency chain is exactly what
`LinacTracker`'s per-element `torch.compile` graph would otherwise
have to re-trace/re-cache-lookup for every distinct element signature
in a real lattice -- see `_LSCFastGraph`'s own docstring. On CUDA
with a scalar charge (every real `LinacTracker` caller), this whole
tensor computation is instead captured once into a CUDA graph, keyed
only by ``(n, n_bins, smooth_passes, dtype, device, pipe_radius_b_m,
radius_factor)` -- never by `length_m`` or anything else that varies
per element -- and replayed for every call sharing that key.
`length_m` is applied after replay, in plain eager Python (one
multiply -- not worth graph-capturing).

Gradients: the graph records no autograd history, so it is used only when
no differentiated input requires grad (`_grad_live`). With `tau_m`,
`x_m`, `y_m` or the charge requiring grad the same operations run eagerly
and the kick is differentiable, with a scalar charge, on every device;
the derivative is that of the discretised model, including the grid
origin and spacing, which follow the live minimum and maximum of `tau_m`.
Float32 gradients of a binned density derivative are rough at about 64
particles per bin (8-18 % against float64 for the CSR kernel in the test
chicane at 256 bins), so check them against float64 for the configuration
in use.

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
