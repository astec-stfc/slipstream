# `slipstream.tracking.native.collective`

[Source](../../../../src/slipstream/tracking/native/collective.py) · [Documentation index](../../../README.md)

## Physics

Implements local long-wavelength 1D LSC and steady-state 1D CSR momentum kicks. LSC uses the line-density derivative and the `1/gamma^2` suppression, with optional pipe logarithm. CSR is directional in late-positive tau. KDE, block-multipole and fixed-grid paths trade cost, smoothing and approximation.

### Limits and conventions

- LSC is a local model rather than elegant's full nonlocal LSCDRIFT implementation.
- Pairwise CSR contains no entrance transient, shielding or drift propagation.
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
eager.

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
compile-once block.

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

### `csr_steady_state_delta_kick_fast`

Elegant-style steady-state CSR in O(N + B log B) time and O(N+B) memory.

Uses CIC deposition, degree-one Savitzky-Golay histogram smoothing,
a central derivative, and near-cell-corrected trapezoidal
integration. The default is unsoftened; the integrable singularity is
handled analytically. An explicit positive softening retains Plummer
regularization. Explicit smooth_passes selects binomial smoothing;
otherwise sg_halfwidth applies. No adaptive host-side tier selection.

LinacTracker supplies the source dipole's BINS/SG_HALFWIDTH, including
for its approximate drift tail. This implements the steady-state wake,
not transient or Stupakov the csrdrift model.

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
