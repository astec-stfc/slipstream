# Slipstream for generative phase-space reconstruction

Investigation date: 5 November 2026. This is a feasibility assessment and development proposal, not a supported GPSR API. No package implementation was changed by the investigation.

A second pass the same day re-ran the original probes (every failure below reproduced; see [Second probe set](#second-probe-set-collective-cost-gradient-fidelity-and-graph-capture)), checked the line references and literature claims against the source and the web, and measured LSC, whole-line cost, float32 gradient fidelity and CUDA-graph capture of a complete forward and backward pass. Differentiating the Warp kernels was assessed separately and deferred: see [warp_autograd_plan.md](warp_autograd_plan.md).

## Recommendation

Use `LinacTracker` as the first GPSR forward model. Add a verified differentiable path for its existing particle-mesh collective kernels, differentiable structure wakes, and a screen-image adapter. Start with a short relativistic diagnostic line. Add transport-map acceleration only after the reconstruction and its gradients are validated.

The opportunity is strongest when CSR or wakefields act **between the reconstruction plane and the measured screens**. Collective effects upstream of that plane are already encoded in the unknown incoming distribution. They need not be simulated during reconstruction unless the goal is to infer an earlier source distribution. Choosing the reconstruction plane carefully can reduce the work substantially.

There is enough existing physics to avoid a tracker rewrite. There are also demonstrated gradient failures that prevent treating the present forward tracker as a training backend without changes.

Three measured results set the priorities for a fast reconstruction with collective effects (details in the second probe set):

1. **The step is launch-bound, and bend transport rather than CSR dominates it.** On a 15-element chicane the bends and drifts account for roughly 70 to 80 percent of forward and forward-plus-backward time; sixteen CSR kicks add 20 to 30 percent. Capturing the whole forward and backward pass in one CUDA graph ran 5.7 to 15 times faster than eager, and under `torch.use_deterministic_algorithms(True)` the replayed loss and gradients matched eager bit for bit.
2. **The LSC fast kernel is far more expensive than CSR, and it acts on every element.** Its 512 smoothing passes cost 63 to 96 ms per call (210 to 245 ms with backward) at 16,384 particles. A precomputed dense operator gives the same result (1e-15 in float64) and is 40 to 55 times faster.
3. **Float32 gradients through CSR are not trustworthy at the starting resolution previously suggested.** At 16,384 particles and 256 bins the float32 gradient differs from float64 by 8 to 18 percent in the x, tau and delta components. This is not a precision problem in the CSR kernel; it is the roughness of a binned density derivative (see below).

Graph capture, dense LSC smoothing and gradient-stable binning are therefore a shorter path to a fast-enough reconstruction than a new CSR algorithm or Warp autograd.

## Research context

The core GPSR loop is a trainable generator of particle coordinates, differentiable transport at several diagnostic settings, differentiable screen images, and an image loss. Gradients propagate through transport to the generator. The 2024 six-dimensional demonstration used Bmad-X and KDE screen rendering. Thus GPSR is not inherently tied to Cheetah. [Roussel et al., PRAB 27, 094601 (2024)](https://doi.org/10.1103/PhysRevAccelBeams.27.094601).

The current public GPSR repository does use Cheetah, but its `GPSRLattice` interface separates `set_lattice_parameters` and `track_and_observe` from the generator. A Slipstream implementation of this interface is a plausible integration point. Beam classes, device transfers, deepcopy behavior, and output shapes still require an integration test. Keep a dependency on GPSR/Cheetah optional for Slipstream. [GPSR modeling source](https://github.com/roussel-ryan/gpsr/blob/main/gpsr/modeling.py).

The PAL-XFEL work demonstrates reconstruction using RF, quadrupoles, a compressor and screens, without requiring a transverse deflector in the reconstruction line. Its preprint (v1) reports approximately 30 minutes for 1,000 iterations with 100,000 particles and 80 measurement samples (5 RF phases) on an A100 at NERSC; the revised version (v3) keeps the 30 minutes for 1,000 iterations but does not state the particle count, and states that collective effects such as space charge and CSR were neglected in the training simulation, with the final elegant propagation including them. The journal page could not be fetched, so these checks used the arXiv versions. The published article explicitly identifies collective effects during training as future work. This is a useful runtime reference and motivation, not a directly comparable benchmark. [Preprint](https://arxiv.org/html/2502.20654v1), [published article](https://doi.org/10.1038/s41598-025-26954-6).

A related PAL study reconstructs a distribution and then uses non-differentiable simulations to study collective effects. That establishes a useful validation workflow, but does not establish reconstruction through CSR within the training loop. [Kim et al., ICALEPCS 2025](https://proceedings.jacow.org/icalepcs2025/pdf/MOCG004.pdf).

Space-charge differentiation is also developing elsewhere: a 2026 Cheetah study (accepted at PASC26) implements grid-based space charge and reports reverse-mode memory growing linearly with the number of macroparticles and cells and proportional to the number of kicks. Differentiable collective tracking alone is therefore not a defensible novelty claim. The relevant opportunity is a validated, efficient CSR/wake-aware inverse workflow. [Dhamrait et al., PASC 2026](https://arxiv.org/abs/2605.27282).

## What the checkout already provides

| Capability | Existing implementation | GPSR consequence |
|---|---|---|
| Native one-pass transport | `LinacTracker`, canonical `(N, 6)` particles, RF, magnets and reference-energy profiles | Appropriate entry point for relativistic diagnostic lines. |
| CSR | Pairwise KDE/score model; fast CIC/FFT steady-state and entrance-transient model; downstream drift models | Start from the fast grid algorithm, with numerical convergence checks. |
| Structure wakes | Longitudinal and transverse tables, FFT convolution, per-cell RF scaling | Preserve the existing physics and replace the non-differentiable deposition/readback. |
| Low-energy space charge | `Injector.track_differentiable`, fixed-step/schedule replay, cylindrical PIC, checkpointing and screen crossings | A second route for straight injector studies, with explicit restrictions. |
| Native map linearization | `optics._element_R` and RF-aware linearization in `linac_twiss` | Reuse the existing machinery when building accelerated transport sections. |

Source entry points: [linac.py](../../src/slipstream/tracking/native/linac.py), [collective.py](../../src/slipstream/tracking/native/collective.py), [shortrange.py](../../src/slipstream/wakefields/shortrange.py), [transverse.py](../../src/slipstream/wakefields/transverse.py), [injector.py](../../src/slipstream/tracking/native/injector.py), [optics.py](../../src/slipstream/tracking/native/optics.py).

## Confirmed gradient blockers

Local probes exercised the checkout using PyTorch 2.14.1+cu132, Windows, four CPU threads and an RTX 3050 with 8 GiB. The probes did not modify package source. The second pass re-ran the same script with PyTorch 2.14.0+cu132 in the project environment: every failure and every gradient comparison reproduced (relative errors 1.31e-10 and 1.36e-10 unchanged), and the line references below are still correct. Repeat timings differed by up to about 30 percent from the first run, so treat timings in this note as order-of-magnitude on this machine.

| Location | Finding | Necessary change |
|---|---|---|
| `collective._csr_chunk_step`, around line 70 | Eager pairwise CSR backward raises an autograd version-counter error. Intermediate arrays are repeatedly modified in place. `_kde_chunk_step` also reuses intermediate storage. | Use functional expressions on the differentiable path; retain forward-only storage reuse where appropriate. Audit both chunk kernels. |
| `collective.csr_steady_state_delta_kick_fast`, around line 1055 | With ordinary scalar macro-charge on CUDA, the public fast kernel returns a field without a gradient connection to particle positions. A finite-difference derivative is nonzero. | Bypass the manually captured forward-only CUDA graph whenever relevant inputs require gradients. Audit analogous LSC, KDE and pairwise graph dispatch. `compile=False` alone does not disable these graphs. |
| `ShortRangeWake.voltage_at`, around line 143 | Rounded integer bin deposition and integer readback provide no gradient with respect to longitudinal positions. | Differentiate both charge deposition and field interpolation, using CIC initially and optional smoother assignment if needed. |
| `TransverseWake.kick_at`, around line 97 | Offset gradients exist, but longitudinal-position gradients are absent. | Differentiate deposition of charge times offset and longitudinal readback. Interpolating only the readback would still omit the distribution's effect on its own field. |
| `LinacTracker.track`, around line 484 | Two charged forward calls on one tracker followed by one backward raise an autograd version-counter error. Every call refills the shared charge tensor in place. | Make per-call physical inputs immutable for the lifetime of the graph. This is required for accumulated scan losses and checkpoint replay. |

For the fast CSR kernel, the CPU derivative of mean kick with respect to bunch-length scale agreed with centered finite differences to relative error `1.31e-10`. On CUDA, passing equal per-particle charges as a vector selected the existing PyTorch fallback and gave relative error `1.31e-10`. Independent directional perturbations of particle positions gave errors `1.26e-9` on CPU and `4.21e-10` on CUDA. This demonstrates a useful derivative of the current discretized model; it does not establish physical convergence for every distribution.

The equal-charge vector was an experimental dispatch bypass, not a proposed public workaround. Production selection should inspect differentiation requirements, preserve scalar-charge efficiency, and avoid changing physical behavior. Existing Warp readiness checks already reject several gradient-requiring inputs, but that does not protect the separate manual CUDA graph dispatches.

The second pass found further problems on the same path ("probed" means demonstrated by running it; "by inspection" means read from the source and not run):

| Location | Finding | Necessary change |
|---|---|---|
| `collective.lsc_delta_kick_fast`, around line 761 | Probed: on CUDA with a scalar charge the result has `requires_grad=False`. With an equal-charge vector the gradient is connected and agrees with a finite difference to 6.9e-10 (float64). The same forward-only graph pattern is used by `_CSRPairwiseGraph` and `_KDEWindowedGraph` (by inspection). | Skip every graph dispatch when a differentiated input is live and grad mode is on. |
| `collective._smooth_bins` via `lsc_delta_kick_fast` | Probed: 512 sequential passes cost 63 to 96 ms forward and 210 to 245 ms forward plus backward per call at 2,048 bins. LSC is applied to every element of positive length, so a 15-element line costs 1.3 s forward and 3.4 s with backward. | Replace the passes by one precomputed dense operator (exactly equal; see the second probe set). |
| `collective._kde_chunk_step` | By inspection: the in-place chain that breaks eager CSR backward also exists in the KDE chunk step (`div_`, `neg_`, `mul_` on a tensor autograd saved). | Functional expressions when gradients are needed. |
| `LinacTracker._ensure_transverse_wake_windows`, `linac.py` line 471 | By inspection: the transverse wake window is fixed from the first bunch tracked, plus 50 percent margin, and never updated. In a reconstruction the first bunch is the untrained generator output. Probed for the longitudinal wake: a particle outside a fixed window is excluded from the deposit but still reads back the edge-bin voltage rather than zero. | Set wake windows explicitly from the design bunch and report overflow. |
| `ShortRangeWake.voltage_at`, dynamic branch | Probed: with no fixed window, `LinacTracker` takes a branch with two host synchronisations (`.item()`) and a data-dependent length; the wake FFT cache gained one entry per distinct convolution length (40 entries after 40 slightly different bunches). `LinacTracker` only fixes the transverse windows. | Call `set_fixed_bin_window` for the longitudinal wake as well; required for graph capture. |
| `ElementSpec`, `_ElementParams` | By inspection: strengths, RF voltage and phase are Python floats. No gradient reaches a machine parameter, so fitting nuisance calibration constants (see GPSR integration below) needs tensor-valued parameters. A new scan setting means a new tracker; building one measured 0.6 to 10 ms, which is not a bottleneck. | Optional: tensor-valued RF phase, voltage and quadrupole strength for calibration fits. |
| `collective.csr_drift_delta_kick_retarded`, line 1012 | By inspection: `float(energy_eV)` forces a host synchronisation, and Gauss-Legendre nodes are built from NumPy on every call. Both block graph capture. | Precompute per element or hoist out of the step. |


## Proposed operating modes

### 1. Differentiable collective tracking: first deliverable

Provide an explicit `track_differentiable`-style contract for `LinacTracker`, following the injector precedent, or an equivalently explicit differentiation mode. It should select existing physical models rather than introduce a second set of equations.

1. Use CIC deposition, FFT collective convolution and interpolated field gathering, with gradients through **both the source density and the observer position**. Cost is approximately `O(N + B log B)` per CSR/wake kick, rather than the pairwise CSR sum's `O(N^2)`. Chunking the pairwise sum does not remove its quadratic work or guarantee low backward memory.
2. Use fixed bin counts and explicit physical windows per kick location. The existing `_fixed_grid` keeps the bin count fixed but derives physical extent from live minima/maxima. Outliers and extrema changes therefore affect the grid and gradient. Probed: detaching the grid origin and spacing changed the float64 gradient by 1.5 percent (x) to 4.4 percent (delta) on the toy chicane, so the extremum term is real but small. Start with generous design-derived windows and report overflow; refine windows between training stages if necessary. The wakes differ from CSR and LSC: their bin width is the wake-table spacing (889 bins for a 2 fs table over a 1.8 ps window in the probe), not 256 or 512, and a differentiable deposit has to work on that grid or on a resampled coarser one.
3. Use an externally defined reference-energy profile where the diagnostic magnets and RF calibration are fixed. Recentring each candidate distribution must not unintentionally change the represented machine. Known scan settings can be prebuilt; source reconstruction does not require making every lattice parameter differentiable.
4. Make the step function safe to replay, checkpoint blocks of elements/slices, and return particles only at requested screens. Full recorded trajectories are unnecessary, although disabling trajectory output alone does not remove autograd's saved intermediates. For whole-iteration CUDA-graph capture the step must also contain no host synchronisation (`float(tensor)`, `.item()`, `bool(tensor)`) and no host-to-device scalar construction such as `torch.as_tensor(float, device="cuda")`; the probe found the second in the LSC form-factor line and in an indexing expression of its own image function.
5. Reject unsupported training combinations explicitly. Stochastic emission/collisions, hard loss boundaries and unsupported CSR approximations need defined handling. Enable additional effects only when the actual diagnostic section requires them.

The simplest implementation uses ordinary PyTorch autograd. Only after that path passes gradient tests should compilation or autograd-aware CUDA graph capture be introduced. PyTorch provides an autograd-aware graph API; the existence of raw CUDA graphs does not supply backward support automatically. [PyTorch graphed callables](https://docs.pytorch.org/docs/2.15/generated/torch.cuda.make_graphed_callables.html), [checkpointing](https://docs.pytorch.org/docs/2.15/checkpoint.html).

### 2. Transport maps interleaved with collective kicks: next speed improvement

Collapse appropriate external-field sections into cached affine or higher-order maps, while retaining explicit density-dependent kicks at physically resolved locations. Linear maps act on every generated particle and therefore retain arbitrary incoming distributions. Tracking only covariance matrices would not be sufficient for GPSR.

Reuse native linearization to construct maps, but retain RF curvature, chromatic terms and longitudinal compression terms when the diagnostic is sensitive to them. A first-order map is insufficient where those nonlinearities carry the information being reconstructed. Compare maps against native particle tracking over the full candidate beam envelope, not just on the reference orbit.

Do not merge across a collective kick, and do not freeze one wake or CSR field for the whole reconstruction. Recompute fields from the current generated bunch on every forward pass. A deliberately lagged-field approximation could be investigated later, but would require separate bias tests.

Known scan settings permit caching maps and static wake spectra across optimizer steps. Beam distributions and their self-fields cannot be cached across changes to generator parameters. A shared upstream beam can be reused within an iteration when the settings genuinely share that section and the autograd lifetime is handled correctly.

### 3. Validation tracking: required companion mode

Track the final distribution with more particles, finer grids, more slices and appropriate CSR entrance/drift models. Include wakefields and space charge where physically needed. Check the 1D unshielded CSR assumption for the actual bunch and chamber: the current CSR implementation is not a general shielded or fully three-dimensional model.

The validation mode can use existing forward accelerators, including eligible Warp paths, without needing gradients. Convergence against the same code is necessary but insufficient: use independent physics references and withheld experimental measurements to expose model bias.

If entrance/drift transients, shielding or space charge materially change the diagnostic images, they must also be represented during training. Post-fit validation can expose that bias but cannot correct it. The transient and drift kernels need their own derivative audit before being advertised as training modes; for example, the retarded drift implementation converts energy and grid spacing to Python floats in parts of the kernel construction. Fixed machine parameters and physical grid windows can make some of that work legitimately cacheable, while fitted parameters require differentiable treatment.

## GPSR integration and diagnostics

A small adapter can reuse GPSR's generator and training infrastructure while supplying Slipstream transport and image predictions. Avoid implementing a new inference framework inside the core tracking package.

1. Convert beam coordinates explicitly. Cheetah uses a seventh affine coordinate and an energy-like sixth coordinate, `(E-E0)/(p0*c)`; Slipstream uses relative momentum `(p-p0)/p0`. These agree only approximately at high energy. Use the exact energy-momentum relation at the boundary. Both document late-positive, metre-valued tau. [Cheetah coordinate definitions](https://cheetah-accelerator.readthedocs.io/en/stable/coordinate_system.html) (`px = P_x / p0`, `delta = (E - E0) / (p0 c)`, `tau < 0` for particles arriving earlier than the reference).
2. Add differentiable image rendering on calibrated screen pixels: charge splatting plus a measured point-spread function is an efficient starting point; KDE is another option. Include screen orientation, resolution, gain/background and field of view. Charge-normalized image shape alone cannot determine bunch charge, which controls collective strength.
3. Initially iterate over scan settings and accumulate gradients safely. Each setting needs a full representative bunch with total physical charge preserved. Independent particle minibatches experience a different noisy self-field and are not equivalent to minibatching independent images. A later `(settings, particles, 6)` API must keep reductions and charge deposition separate for every setting; the present 1D scatter kernels do not support that automatically.
4. Use a fixed set of latent samples during a deterministic fit, then validate with fresh samples and independent initializations. Choose particle count, bins and smoothing for **gradient stability as well as forward convergence**. Probed on the toy chicane (float32 against float64, relative L2 error of the gradient with respect to the initial coordinates, x / px / tau / delta): 16,384 particles with 256 bins and the default 3-point smoothing give 8.1 / 4.7 / 16 / 18 percent; 64 bins give 2.8 / 2.5 / 2.6 / 2.0 percent; 256 bins with 9-point smoothing give 3.2 / 2.6 / 6.1 / 6.0 percent; 65,536 particles with 256 bins give 3.3 / 2.5 / 8.2 / 9.0 percent. Coarser grids are more stable but must still resolve the physical structure, so the choice is a trade to be validated per experiment. The earlier suggestion of 8k to 16k particles with 256 to 512 bins is in the unstable regime for float32 gradients.
5. Choose scans that constrain the requested dimensions. A basic quadrupole scan cannot establish an arbitrary full 6D distribution. Slipstream currently lacks a native transverse deflecting cavity. Add one, including its consistent longitudinal companion kick, if the chosen experiment needs it; the demonstrated RF/compressor approach offers another measurement design. Fit only a small, constrained set of nuisance calibration parameters to avoid absorbing machine-model errors into the inferred beam.

## Local computational measurements

The timing probe used an **11-element toy line**: one quadrupole, four bends, six drifts, four lumped steady-state CSR kicks, fourth-order magnetic integration with four slices per bend, 100 pC, 150 MeV, 256 CSR bins, and a differentiable 64 by 64 bilinear image. It used the PyTorch CSR fallback described above, without compilation, transient/drift CSR, LSC or structure wakes. The objective's backward pass ended at the input particles; no neural generator or optimizer was timed.

Times are synchronized wall-clock medians of five warmed evaluations, after two warmups per path. They are throughput observations on this machine, not precision validation of float32 or a completed GPSR reconstruction.

| Device / precision | Particles | Forward | Forward plus image loss and backward |
|---|---:|---:|---:|
| CPU, float64, 4 threads | 8,192 | 19.5 ms | 74.5 ms |
| RTX 3050, float64 | 8,192 | 49.4 ms | 146.3 ms |
| RTX 3050, float32 | 8,192 | 45.8 ms | 153.0 ms |
| RTX 3050, float32 | 65,536 | 60.8 ms | 173.0 ms |

An isolated 256-bin CSR kick plus backward took 1.57 ms on CPU at 8,192 particles, 6.98 ms on GPU at 8,192, and 6.91 ms on GPU at 65,536. Additional peak PyTorch allocated memory for that single CUDA kick was approximately 0.69 MiB and 5.35 MiB respectively; these are not complete training-memory measurements.

The weak timing dependence on an eightfold particle-count increase, together with the GPU losing to the CPU at small N, points to launch/dispatch overhead in this eager workload. This is evidence to prioritize map compression, stable kernels and batching before a new CSR algorithm. It does not prove the bottleneck on a longer beamline or another GPU.

For budgeting, if `U` optimizer updates each evaluate `S` settings, then `T ≈ U*S*t_setting` plus generator, optimizer, data and startup costs. At 1,000 updates and 20 settings per update, the measured transport/image/backward contribution is about 25 minutes on CPU at 8k particles or 58 minutes on GPU at 65k particles. Four settings per update reduce those arithmetic totals to about 5 and 12 minutes, but may require more updates to converge. These are workload extrapolations, not reconstruction-runtime promises.

A practical objective is a validated short-line reconstruction in tens of minutes. Long injector simulations, many CSR slices, detailed transients, or large simultaneous scan batches can cost substantially more. Benchmark compilation startup and the first backward separately from warm throughput. Validate float32 observables and gradients against float64; use mixed precision only where justified.

These timings cannot be extended directly to LSC: its fast kernel currently defaults to 2,048 bins and 512 binomial smoothing passes. That smoothing loop needs separate profiling and convergence-based settings, particularly for GPU launch cost. The measured CSR results do not establish a runtime for combined CSR, LSC and structure wakes.

## Second probe set: collective cost, gradient fidelity and graph capture

Same machine and method as above (float32, RTX 3050, synchronized medians; PyTorch 2.14.0+cu132). Scripts are in [.cache/gpsr_linac/](../../.cache/gpsr_linac/) (ignored by git). The line is the 11-element toy line extended to 15 elements: a chicane of four bends with CSR (`csr_n_slices=4`, the value elegant import defaults to, so 16 kicks), quadrupoles and drifts, charge 100 pC, 150 MeV, with the earlier 64 by 64 bilinear screen loss. Gradient-connected CSR and LSC used the equal-charge-vector bypass, which stands in for the dispatch fix.

**Per-call cost, 16,384 particles**

| Kernel | Forward | Forward plus backward |
|---|---:|---:|
| CSR steady state, 256 bins (PyTorch path) | 1.4 to 1.8 ms | 5.4 to 5.5 ms |
| CSR with entrance transient | 2.2 to 2.7 ms | 5.5 to 6.9 ms |
| CSR steady state, existing forward-only CUDA graph | 0.34 ms | no gradient |
| LSC fast, 2,048 bins, 512 passes (loop) | 63 to 96 ms | 210 to 245 ms |
| LSC fast, existing forward-only CUDA graph | 7.5 to 9.8 ms | no gradient |
| LSC fast, dense smoothing operator | 1.4 to 2.0 ms | 3.9 to 5.9 ms |
| Longitudinal wake, float64, dynamic window | 1.0 to 1.1 ms | no gradient to tau |
| Transverse wake, float64, fixed window | 0.6 to 0.8 ms | 1.2 to 2.0 ms (offset only) |

The dense operator is the 512th power of the zero-padded `(1,2,1)/4` smoothing matrix, built once per bin count in float64 and applied as one matrix-vector product (16 MiB in float32 at 2,048 bins). It matches the loop to 1e-15 in float64 and to 2e-5 (kick) and 5e-5 (gradient) in float32, where the 512-step loop accumulates its own rounding.

**Whole line, 16,384 particles**

| Configuration | Forward | Forward plus backward |
|---|---:|---:|
| 7 elements, magnets only, no charge | 10.6 ms | 27.9 ms |
| 15 elements, CSR on 4 bends, 4 kicks | 69 ms | 203 ms |
| 15 elements, CSR 16 kicks | 90 ms | 265 ms |
| same, plus LSC with the 512-pass loop | 1,263 ms | 3,427 ms |
| same, plus LSC with the dense operator | 126 ms | 366 ms |

Absolute values varied up to 1.7 times between runs (a separate run gave 54 and 160 ms for the 16-kick line), so rely on ratios within a run. In that run the same chicane without CSR took 43 and 113 ms: bends and drifts are about 80 percent of the forward time and 70 percent of forward plus backward. Building a `LinacTracker` took 0.6 to 10 ms.

**Whole-iteration CUDA graph** (the complete `track`, screen loss and `backward`, captured once with `torch.cuda.graph`, then replayed with new input values):

| Configuration | Eager | Replay | Speed-up |
|---|---:|---:|---:|
| 15 elements, CSR 16 kicks | 191 to 314 ms | 21 to 23 ms | 8 to 15 times |
| plus dense LSC | 242 ms | 31 ms | 7.8 times |
| CSR 16 kicks, deterministic algorithms | 198 ms | 35 ms | 5.7 times |

Under `torch.use_deterministic_algorithms(True)` the replayed loss, gradient, and the gradient for a new input were **identical** to eager (relative difference 0.0). Without it the graph and eager gradients differ by 1.8 to 3.9 percent, which is the same size as the run-to-run noise of eager float32 itself (below), so deterministic mode is the way to validate a captured step. Deterministic scatter was somewhat slower (35 ms against 23 ms). Capture needed the equal-charge bypass, an image function free of indexing with Python lists, and an LSC with a Python-float pipe radius. Not tried: RF elements with wakes, the CSR drift models, `always_change_p0=True`, gradients with respect to scan settings, and compiled (`torch.compile`) steps.

**Float32 gradient fidelity** (relative L2 difference of the gradient with respect to the initial particle coordinates)

| Line | f32 against f32, same input | f32 against f64 (x / px / tau / delta) |
|---|---|---|
| magnets only | below 1e-6 | 2e-6 (tau carries no signal) |
| chicane, no CSR | 1e-6 | 1.4 / 1.4 / n.a. / 1.4 percent |
| chicane, CSR 256 bins | 4 to 6.5 percent | 8.1 / 4.7 / 16 / 18 percent |
| same, deterministic algorithms | 0 (bitwise) | 8.1 / 4.7 / 16 / 18 percent |

Forward quantities agree much better: the screen image differs by 6.7e-5 and the final particle array by 1.2e-3 (relative L2). Evaluating the CSR kernel in float64 inside float32 transport did not help (x 8.6, delta 22 percent), so the discrepancy is not CSR-kernel rounding. The consistent reading is that a histogram-derivative kick built from about 64 particles per bin is a rough function of position at the scale of one bin, so the small differences between float32 and float64 transport change which bin each particle falls in and hence the gradient. Run-to-run float32 differences come from non-deterministic atomic scatter and vanish in deterministic mode. This also means the gradient of the discretised model is a high-variance estimate of the gradient of the physical kick; check it against float64 and across seeds for every configuration used.

## Low-energy injector scope

`Injector.track_differentiable` already provides fixed schedules, block checkpointing, screen-plane crossing, and cylindrical space charge. Existing tests include finite-difference checks with space charge. Reuse it for straight, approximately cylindrical injector sections rather than introducing that capability again.

Its present contract rejects injector wake kicks, photocathode emission, stochastic IBS, 3D space charge and space charge after bends. General low-energy 6D reconstruction can require asymmetric 3D self-fields; a cylindrical model may not preserve the correlations one is trying to infer. Supporting arbitrary 3D injector GPSR is a separate project, not part of the short-linac estimate below. The derivative of a hard aperture or emission threshold also needs special treatment if the fit depends on losses or emission timing.

## Smallest complete development sequence

Estimates assume one developer familiar with this code, a calibrated short beamline, available images or synthetic truth, and a working PyTorch toolchain. They are planning estimates, not measured task durations.

1. **2-4 working days:** implement gradient-safe collective dispatch, remove autograd-invalid storage reuse, make charged forwards replay-safe, and add CIC wake derivatives. Gate with finite differences for particle positions, offsets and charge, plus multiple-forward/backward checks on CPU and CUDA. Split into two parts. Part A, tracked as a GitHub feature: skip graph dispatch when gradients are live (CSR fast, LSC fast, pairwise CSR, KDE), dense LSC smoothing, an immutable per-call charge tensor, capture-safe LSC, functional chunk steps under autograd, and a documented whole-iteration capture recipe with a CUDA test. Part B, not started: CIC deposit and readback for the longitudinal and transverse wakes (opt-in, with convergence against the rounded-bin version), explicit wake windows, tensor-valued RF and quadrupole parameters.
2. **3-5 working days:** implement a thin GPSR adapter and calibrated differentiable screen; demonstrate a synthetic short-line reconstruction with collective effects on. Keep the chosen existing transport physics and physical charge fixed across comparisons.
3. **5-10 working days:** establish particle/bin/slice convergence, held-out scan predictions and model-bias comparisons; add the measured performance improvements, including selected-screen recording and checkpointing. Whole-iteration graph capture (one graph per scan setting) is the first speed lever because it was measured at 5.7 to 15 times; extend the capture audit to RF with wakes and the CSR drift models. Add transport-map compression if profiling after capture still justifies it. Warp autograd is deferred (see [warp_autograd_plan.md](warp_autograd_plan.md)).

This gives roughly **2-4 working weeks for a validated short-line demonstration**, with existing diagnostic elements. A new deflector, 3D injector differentiation, extensive real-machine calibration or a production deployment adds scope.

Acceptance should require recovery of known synthetic observables and correlations at the experiment's resolution, successful prediction of withheld settings, stable results under numerical refinement, and a measured time/memory budget. Compare training with and without collectives using truth generated at higher fidelity; fitting images alone can conceal a biased inferred source. Report seed-to-seed spread and poorly constrained dimensions instead of implying a unique 6D solution.

## Investigation artifacts and validation boundary

Local ignored artifacts contain the probes and raw timing output: [main probe](../../.cache/gpsr_probe.py), [main results](../../.cache/gpsr_probe.json), [directional/timing probe](../../.cache/gpsr_followup.py), [isolated timing results](../../.cache/gpsr_followup.json). The main probe records expected failures instead of changing the production code. Optional dependencies for this investigation were placed under `.cache/gpsr-deps`; existing environments were not modified.

The second pass added probes in [.cache/gpsr_linac/](../../.cache/gpsr_linac/) (LSC, wakes, whole-line cost, graph capture, float32 gradient fidelity, binning dependence). CPU eager and CUDA eager PyTorch paths were exercised. No full GPSR optimization, real-beam reconstruction, compiled-path benchmark, Warp validation or complete test suite was run. The measured finite-difference agreement establishes local numerical derivatives, not the accuracy or uniqueness of an inferred physical distribution.
