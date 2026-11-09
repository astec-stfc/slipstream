# Autograd through Warp kernels: plan

Investigation date: 5 November 2026. This is a development plan with local measurements, not a supported feature. No package source was changed. It is the companion to [gpsr_feasibility.md](gpsr_feasibility.md), which found that the PyTorch path is launch-bound and that several Warp dispatches silently drop gradients.

**Status: deferred (5 November 2026).** The cost is not justified now. The same day the GPSR note measured a cheaper route to a fast linac reconstruction with collective effects: whole-iteration CUDA-graph capture ran 5.7 to 15 times faster than eager and a dense LSC smoothing operator removed the largest single cost, neither needing Warp adjoints. Revisit if reverse mode is still launch-bound at the target scale after capture, or if injector reverse-mode memory becomes the limit. The Warp-version finding below (pin `>=1.19,<1.20` only when differentiable kernels land) stays valid.

## Recommendation

1. **Do not start with a rewrite of the Warp kernels.** Warp's automatic adjoints are correct on the straight-line and additive-loop kernels this package uses, and cost about 1.2 to 1.4 times the forward kernel. The risk is not the adjoint maths; it is the Python wrapper cost, a small set of kernel patterns that Warp differentiates wrongly, and silent gradient loss at the dispatch gates.
2. **Gate the work on a baseline.** Measure PyTorch forward plus backward for a 2,000-step injector run and for the toy linac line first. Continue only if fused Warp forward plus backward reaches at least 2 times the PyTorch wall time (hypothesis, not yet measured). Otherwise keep Warp forward-only.
3. **Order of work:** injector step kernels first (push, applied fields, space-charge deposit and gather), then the 1D collective deposit and gather, then CUDA-graph capture of the whole forward and backward. Leave the ring turn kernel and the CSR pairwise kernel on PyTorch for gradients.
4. **Warp version:** move the `cuda` extra to `warp-lang>=1.19,<1.20` when the first differentiable kernel lands, not before. The gain is correctness fixes, not speed. Details are [below](#warp-version).

## Where Warp is used today and what is blocking gradients

`track_differentiable` on the injector is pure PyTorch (optionally `torch.compile`); the `using_warp` branch exists only in `Injector.track`. The ring Warp turn is gated to states without gradients. The collective Warp dispatches check `requires_grad` and fall back. So **no Warp kernel is on a gradient path today**, and the blocking is deliberate.

| Kernel (all in [warp_kernels.py](../../src/slipstream/tracking/native/warp_kernels.py)) | Role | AD difficulty | Priority |
|---|---|---|---|
| `_push_kernel` (Boris, Kahan-compensated) | injector step | Straight-line. Easy. Kahan carries must be treated as non-differentiable state. | 1 |
| `_sw_kernel`, `_tw_kernel`, `_sol_kernel` | cavity, travelling-wave, solenoid fields | Dynamic loop over cavities, but updates are additive, which Warp differentiates correctly. Table arrays are fixed inputs. | 1 |
| `_deposit_kernel`, `_deposit3d_kernel` | space-charge CIC scatter | Adjoint of a scatter is a gather, which is cheap and race-free. | 1 |
| `_gather3d_kernel` | space-charge gather and boost | Adjoint is an atomic scatter into the potential's gradient (8 corners, 3-point stencil). Contended and non-deterministic. | 1 |
| `_cic_line_density_kernel` | 1D CIC for CSR, LSC, wakes | The weights `interp_w` are computed in PyTorch, so only the kernel's adjoint with respect to `interp_w` and charge is needed. Hand-write it as a gather. | 2 |
| `_histogram_kernel` | integer wake histogram | Not differentiable by construction. Replace with the CIC deposit on the differentiable path (as the GPSR note already requires). | 2 |
| `_apertures_kernel` | hard loss mask | Boolean output, no gradient. Mark non-differentiable. | n/a |
| `_csr_pairwise_kernel` | exact O(N^2) CSR sum | Correct adjoint, but 4 to 6 times the forward and O(N^2). Validation only. | no |
| `_retarded_wake_kernel`, `_stupakov_phi_kernel` | bisection on fixed machine geometry | Bisection assignments have no useful adjoint. Keep forward-only; gradients reach the particles through the wake-table readback, not the root. | no |
| `ring_warp._turn_kernel` | fused one-turn ring map | Triple-nested dynamic loop that multiplies state. **Warp's adjoint would be wrong.** | no (see WP4) |

## What the probes established

Probes ran on Warp 1.17.0 (installed) and 1.19.0 (scratch install), PyTorch 2.14.0+cu132, RTX 3050 8 GiB, Windows. Scripts are in [.cache/warp_ad/](../../.cache/warp_ad/) (ignored by git). Times are synchronized wall-clock medians of 30 runs after warmup.

**Correctness**

| Pattern | Result |
|---|---|
| 1D CIC deposit adjoint vs PyTorch, f32 and f64 | Gradient relative error 0 (identical) |
| Dynamic loop with an **additive** accumulator (the `_csr_pairwise` and `_sw` shape) | Relative error 6e-16 |
| Dynamic loop with a **multiplicative** update, `p = p * (1 + a*b)` | **5.3% error on both 1.17 and 1.19** |
| Same loop with a constant trip count of 8 (unrolled) | Relative error 2e-16 |
| Kahan-compensated update | Sum has gradient exactly 1 to both operands and exactly -1 to the incoming carry; the outgoing carry has exactly zero gradient to the operands |
| Deposit adjoint in f32 against an f64 reference | Relative error 3.6e-8 |
| Real CSR pairwise kernel adjoint at N=2048 | f64 2e-15, f32 1e-6 against a dense PyTorch reference |

**Cost (RTX 3050)**

| Quantity | Measurement |
|---|---|
| Deposit plus gather, kernel only, N=65,536, 257 bins | forward 0.046 ms, backward 0.054 ms |
| Same, N=1,000,000 | forward 0.34 ms, backward 0.47 ms (1.4 times) |
| Deposit only, N=1e6, f32 | forward 0.30 ms, backward 0.12 ms (the adjoint is a gather) |
| `wp.Tape` inside `torch.autograd.Function`, deposit forward plus backward, N=65,536 | 0.64 ms, against 0.47 ms for PyTorch `index_add` |
| Each extra recorded launch under a Tape | about 0.13 ms of Python overhead |
| Stateless adjoint launch inside `autograd.Function`, deposit plus gather, N=8,192 and 65,536 | 1.72 and 1.53 ms, against 1.74 and 1.98 ms for eager PyTorch |
| `tape.backward()` captured in a CUDA graph | Works. 0.046 ms against 0.054 ms uncaptured at N=65,536 |
| CSR pairwise, backward over forward, N=16,384 | f32 47 ms over 8.3 ms (5.7 times); f64 1,070 ms over 498 ms (2.1 times) |
| Deterministic atomics mode (`DeterministicMode.RUN_TO_RUN`), N=1e6 f32 | forward 7.1 ms (24 times), backward 9.1 ms (75 times) |

Reading these together: **at the sizes GPSR uses, kernel time is a few hundredths of a millisecond and the wrapper dominates.** Warp autograd will not beat PyTorch on a single small kernel. It wins by replacing many small PyTorch operations with one fused launch and by capturing the whole sequence in a graph, as the forward path already did for the retarded wake (6.7 ms against 206 ms at 1e6 particles, from [warp_kernels.md](../modules/tracking/native/warp_kernels.md)). Whether the same fusion gain holds in reverse mode is the hypothesis the baseline gate tests.

Gradient reproducibility: position gradients through the deposit were bitwise repeatable across runs. The deposited charge itself was not (floating-point atomics), so any gradient that depends on an atomic scatter (the adjoint of `_gather3d_kernel`) will vary at rounding level run to run.

## Implementation routes

| Route | How | Verdict |
|---|---|---|
| A. `wp.Tape` in `autograd.Function` | Record launches in `forward`, call `tape.backward()` in `backward`. | Correct, but 0.13 ms per recorded launch and the tape holds device arrays. Use only for prototyping. |
| B. Stateless adjoint launch | `forward` saves input tensors with `save_for_backward`; `backward` calls `wp.launch(..., adj_inputs=[...], adjoint=True)` with gradient buffers from PyTorch. Warp recomputes the forward values inside the adjoint kernel. | **Recommended default.** Lowest memory (inputs only), no tape, graph-capturable. Needs explicit `adj_inputs`: passing arrays with `.grad` attached crashed with an access violation in the probe. |
| C. `torch.library.custom_op` with `register_autograd` and `register_fake` | Same kernels, registered so `torch.compile` can trace around them instead of graph-breaking. | Add after B works and only if compiled paths need it. Untested here. |
| D. Hand-written adjoint kernel | For obvious adjoints (CIC deposit with respect to `interp_w` is `q*(g[b+1]-g[b])`) write the gather directly. | Use where the adjoint is simpler than Warp's generated one. |

## Work packages

Estimates assume one developer who knows this code. They are planning figures, not measurements.

**WP0, 0.5 day. Baseline and version.**
Add a benchmark under `.cache` that times PyTorch forward plus backward for (a) `Injector.track_differentiable` at 2,000 steps, 1e4 and 1e5 particles, f32, with and without `checkpoint_every`, and (b) the toy linac line from the GPSR note. Record peak memory. Install Warp 1.19 and rerun the 52 Warp tests (already done, below). Write down the kill criterion before building.

**WP1, 2 to 3 days. Differentiable-op plumbing in `warp_kernels.py`.**
- A small helper that builds the `adj_inputs` list from PyTorch gradient buffers with low overhead (reuse `return_ctype=True` arguments; cache stream objects as `_launch_stream` already does).
- Replace the blanket `requires_grad -> False` in `_warp_collective_ready` and `_warp_f64_wake_ready` with a per-kernel `differentiable` capability. A kernel without an adjoint must still force the PyTorch fallback, and must do so loudly if the user asked for Warp.
- Mark outputs that must not carry gradient with `ctx.mark_non_differentiable` (Kahan carries, `alive`, integer indices).
- Support the Warp CPU backend in the new wrappers so the gradient tests run in CPU CI. `_launch_target` already does this for the wake kernels.

**WP2, 2 days. 1D collective kernels (optional, depends on WP0).**
- Hand-written adjoint for `cic_line_density` with respect to `interp_w` and charge.
- A new differentiable CIC gather kernel for the observer readback (the GPSR note requires gradients through both source and observer).
- Wire both into `_bin_line_density` and the wake readback. Keep PyTorch `index_add` as the reference.
- This work pays off only if fusing deposit, FFT and gather cuts launches enough; PyTorch autograd already gives correct gradients here.

**WP3, 4 to 6 days. Injector step.**
- Differentiable `push`: Boris kernel with Kahan carries detached from the graph.
- Differentiable `applied_fields` (`sw`, `tw`, `sol`). Make the cavity loop trip count compile-time constant per lattice where possible so Warp unrolls it, or keep the additive form that the probe showed is safe.
- Differentiable `sc_deposit` and `sc_fields_3d`. The FFT between them stays in `torch.fft`, so autograd handles it. Reduce contention in the gather adjoint (see issues).
- Add a `use_warp` option to `track_differentiable` with the same fallback rules as `track`. Keep `checkpoint_every`; saved inputs per step are about 12 state and field arrays.
- Aperture mask stays non-differentiable and is documented as such.

**WP4, deferred. Ring turn and CSR pairwise.**
The fused ring turn cannot use Warp's automatic adjoint because its loops over elements, kicks and stages multiply state and are dynamic. Options if ring gradients become a bottleneck: per-element launches under a CUDA graph with straight-line element functions; or a hand-written reverse sweep with state checkpoints. Until a benchmark justifies this, ring gradients stay on the PyTorch path. The CSR pairwise kernel stays validation-only; training uses the grid CSR.

**WP5, 2 days. CUDA graph capture of forward plus backward.**
Existing Warp launches already use PyTorch's current stream, so `torch.cuda.graph` capture is the first thing to try. `wp.ScopedCapture` around `tape.backward()` worked in the probe. Capture only fixed-shape, fixed-schedule scan evaluations. Never capture a graph whose inputs require gradient unless the graph is the autograd-aware one (`make_graphed_callables`); the GPSR note found a forward-only captured graph silently disconnected the field from particle positions.

**WP6, 2 days. Tests, docs, CI skip rules.** Listed under validation.

Total for WP0, WP1, WP3, WP5, WP6 (the injector path): about 11 to 15 working days. WP2 adds 2 days. WP4 is unscoped.

## Expected issues and mitigations

| Issue | Evidence | Mitigation |
|---|---|---|
| **Dynamic loops with non-additive updates give wrong gradients.** | 5.3% error, unchanged on 1.19 | Keep such loops out of differentiable kernels, or give them a constant trip count (unrolled up to `max_unroll`, default 16). Add a test that compares every exposed kernel's gradient to a PyTorch reference so a new loop cannot slip through. |
| **Unrolling limit.** Warp 1.19 counts the exact iteration count of constant `range()` loops, so `range(0, 33, 2)` now exceeds 16 and silently becomes dynamic. | 1.19 changelog | Set `max_unroll` explicitly in the module options of differentiable kernels. |
| **Wrapper overhead dominates small launches.** | 0.13 ms per tape launch; 1.5 ms for a deposit-gather pair | Route B, ctype arguments, fewer and fatter kernels, graph capture. Do not wrap single small kernels. |
| **Atomic scatter in the gather adjoint** is contended (many particles per cell) and non-deterministic. | Atomics non-bitwise in probe; deterministic mode costs 24 to 75 times | Accept rounding-level variation for training; use deterministic mode only for regression tests. Sort or bin particles by cell if contention measures badly. |
| **In-place updates and shared output buffers.** `push_inplace` and the field kernels add onto shared `Ex..Bz` buffers. | Version-counter failures already seen in `LinacTracker.track` | Differentiable path uses fresh outputs, never in-place writes. Warp 1.19 fixes some in-place gradient ordering bugs (GH-1817) but do not depend on it. |
| **Kahan carries** are state that need no gradient. | Probe: carry output has zero gradient to the operands, so the -1 on the carry input never propagates further | Detach carry inputs and mark carry outputs non-differentiable; this also saves memory bandwidth. |
| **Generic-dtype constants** such as `one = hx[0] / hx[0]` appear in the kernels. If the divisor array ever requires gradient these produce cancelling terms, and 0/0 if a value is zero. | Code inspection, not probed | Do not request gradients for grid origin, spacing or scalar parameters (pass no `adj_inputs` for them). Replace the trick with typed overloads in differentiable variants. |
| **Discontinuities.** `floor`, clamps, apertures and window overflow give zero or undefined gradient. | By construction | Fixed physical windows with overflow reporting, as the GPSR note proposes. Document apertures as non-differentiable. |
| **Bisection kernels** (retarded wake, Stupakov root). | Assignments `lo = mid` have no adjoint | Keep forward-only for fixed geometry. If a geometry parameter must be fitted, use the implicit-function step: evaluate the residual once at the detached root in PyTorch and divide by its detached derivative. |
| **Precision.** f32 gradients differ from f64 by about 4e-8 on a deposit and 1e-6 on the CSR sum; RTX 3050 f64 is slow (CSR f64 is 60 times slower than f32 at 16k). | Probes | Train in f32, validate gradients in f64 on a small case. `fast_math` or reciprocal tricks in the forward path must be checked against the adjoint separately. |
| **`torch.compile` interaction.** Function calls into Warp inside a compiled region cause graph breaks. The existing guards use `torch._dynamo.is_compiling()`. | Not tested | Keep the guards; add Route C only if profiling shows breaks matter. |
| **Memory.** Saved inputs for 12 f32 arrays at 1e5 particles are about 4.8 MB per step, so 2,000 steps need about 9.6 GB (19 GB in f64) against 8 GiB on this GPU. | Arithmetic | Keep `checkpoint_every`. PyTorch autograd saves more per step, but that ratio is a hypothesis for WP0 to measure. |
| **Compile time and cache.** Adjoint kernels roughly double generated code; the IBS module already sets `enable_backward=False`. | Not timed on the full module set | Time a cold build of the whole module before and after. Put forward-only kernels in a module with `enable_backward=False`. |
| **Contiguity and device gates.** Inputs must be contiguous CUDA f32 or f64. | Existing `available()` checks | Reuse the same gates; fall back otherwise. |

## Warp version

Measured on 1.17.0 and 1.19.0 with the same scripts:

- Kernel forward and backward times, gradient values and the 5.3% loop error are **identical** across versions (differences within run noise).
- The 52 tests in `test_warp_collective_kernels.py`, `test_injector_ibs_warp.py`, `test_injector_screen.py` and `test_placement_tensors.py` pass on both (42.7 s and 43.1 s).
- Deterministic mode behaves the same on both.

So there is **no speed benefit**. What 1.19 adds that matters for this plan: fixes for gradients when arrays are modified in place (GH-1817), for separate inner dynamic loops assigning one variable (GH-2048), for a spurious iteration in adjoints of dynamic ranges (GH-2046), and for conditional reductions in static loops containing `continue` (GH-1552); and `WarpDeterminismError` messages that name the cause. 1.18 added differentiable assignment to components of array elements (for example `out[i].x = v`), useful if kernels move to vector-typed state.

Costs and risks of moving:
- 1.18 and later wheels are built with CUDA 13.4 and require an R580 or newer driver and an sm_75 or newer GPU. The local driver is 616.56 and the GPU is sm_86. CI and other users' machines are unchecked; CUDA 12 wheels exist as `+cu12`.
- 1.19 deprecates Python 3.10 with a warning and removes it in 1.20. The package declares `>=3.10`, so an upper bound `<1.20` is needed until that is decided.
- 1.19 makes the unrolling and array-subclass rules stricter (breaking), as listed above.

**Recommendation:** keep the `cuda` extra as is until WP1 starts. Then set `warp-lang>=1.19,<1.20` in `pyproject.toml` and note the pin in [compilation.md](../compilation.md). The case rests on the in-place and loop gradient fixes, not on performance. Do not use an exact `==` pin in the extra; keep exact pins in a checkout environment file if wanted.

## Validation and acceptance

1. **Per-kernel gradient tests**, for each differentiable wrapper: compare with the PyTorch implementation at f64 (relative error below 1e-10) and f32 (below 1e-5), compare with centered finite differences along random directions, and run `torch.autograd.gradcheck` on a small f64 case.
2. **Eager, Warp and compiled agreement** on identical inputs for any path that claims all three.
3. **Repeat-use tests**: two forwards then one backward, and backward after `checkpoint`, on the same tracker object (the failure the GPSR note found).
4. **Dispatch tests**: requesting Warp with gradients on an unsupported kernel falls back to PyTorch (or raises if Warp was forced); the result keeps its `grad_fn`.
5. **Loop-hazard test**: every kernel exposed to autograd has a PyTorch reference gradient test, so a new non-additive dynamic loop fails in CI.
6. **Determinism test**: position gradients bitwise repeatable; atomic-scatter gradients within a stated tolerance.
7. **Skip rules**: no CUDA, no Warp or no compiler skips only these tests. The Warp CPU backend runs the gradient tests where CUDA is absent.
8. **Performance acceptance**: at least 2 times reverse-mode wall time and at least 2 times lower peak memory than PyTorch on the WP0 benchmarks, or the feature stays off by default. Report cold build time and first-backward cost separately from warm throughput.

## Not established

- The injector and linac speedups are hypotheses. Only the micro-benchmarks above were measured; no full `track_differentiable` run with Warp exists.
- `torch.compile`, `torch.cuda.graph` capture of an `autograd.Function` that launches Warp, and `torch.library.custom_op` registration were not tried.
- Deterministic mode in f64 returned a non-bitwise deposited charge in the probe; the cause was not investigated.
- Only an RTX 3050 was used. Consumer-GPU f64 throughput and atomic contention differ on other cards.
- Warp 1.19 was installed to a scratch directory and the shared conda environment was not changed. To reproduce, run `pip install --no-deps --target <dir> warp-lang==1.19.0` and set `PYTHONPATH=<dir>`.
