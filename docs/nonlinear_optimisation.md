# Nonlinear optimisation

`slipstream.analysis.nonlinear_opt` supplies differentiable tracking
surrogates and bounded, independently budgeted derivative-free baselines.
Use float64 for tune diffusion and converge element slicing first.

`Footprint` combines amplitude/chromatic tune shifts and optional
`log1p((D/diff_scale)**2)` diffusion. `action_growth_loss(tbt, optics)`
adds a smooth normalized-action growth penalty. Constant action has a
`log(2)**2` baseline; decay lowers the loss and growth raises it. These are surrogates for
survival, not replacements for long-turn dynamic aperture. Use launches
inside the DA: lost launches have constant penalties and no restoring
gradient. A preliminary no-gradient pass prevents NaN backward paths.

`joint_optics_loss(ring, target)` adds tune and observation-point beta
constraints to a nonlinear objective. Set tensor-valued K1 and K2 strengths
with `RingMap.with_strengths`; choose tune/beta scales as tolerances.

`naff_harmonics(z, n_harmonics)` extracts successive frequencies and complex
coefficients. `spectral_lines(z, frequencies)` jointly fits specified lines,
including combinations of betatron tunes for resonance studies. These
spectral amplitudes preserve phase but are not calibrated Hamiltonian RDTs:
conversion needs a declared normal-form and action convention. Lines should
be separated by several Fourier bins. Peak selection is discrete; local
frequency/coefficient derivatives apply only while line identities persist.

`gradient_free(objective, theta, bounds, method=..., max_evaluations=...)`
provides Powell, Nelder-Mead and seeded differential evolution. Every call
counts toward the strict objective budget and the returned candidate is the
best finite point actually evaluated. Bounds are one `(lower, upper)` pair
per knob; objectives are minimised. For DA use **negative area**.

`multi_fidelity(smooth_objective, aperture_objective, theta, bounds)` first
uses bounded Adam on the differentiable surrogate, then measures the candidate
and performs derivative-free refinement against the high-fidelity objective.
The latter must perform its own longer DA tracking. Differential evolution
is the default; Powell and Nelder-Mead are alternatives. `CMA-ES` is available by installing the optional `cma` package; requesting it
without that dependency raises an explicit ImportError.

CPU tests use an independent two-tone spectrum, analytic action-growth and
quadratic minima, plus a synthetic native FODO ring whose joint-constraint
strength gradient is checked against central finite differences. They do
not establish improved DA on a real machine, GPU performance, or elegant
agreement. Historical br14 studies referenced by the external plan are not
part of this checkout.

Run `python examples/nonlinear_optimisation.py --budget 6 --turns 32` for an
MLS-inspired storage-ring demonstration with joint K1/K2 constraints,
surrogate screening, longer DA retracking and equally budgeted baselines.
The model uses 629 MeV, 48 m circumference, eight 45-degree bends, 24
quadrupoles, 24 sextupoles and four inactive octupole slots, with alternating
6 m and 2.5 m straights. Five representative quadrupole families distinguish
the long and short straight optics. Independent hard-edge matrix matching
sets full tunes near (3.18, 2.23) and roughly matches the beta/dispersion
envelopes in the user-provided [MLS Twiss plot](../examples/MLS/Twiss.png).
The native beta maxima are about 15.20 m horizontally and 9.03 m vertically,
and the maximum horizontal dispersion is about 1.67 m. The reference scales
are approximately 15 m, 9 m and 1.5 m, respectively. The native one-turn
analysis reports fractional tunes near (.18, .23). See the
[matching notes and family settings](../examples/MLS/README.md).

Geometry is based on the [PTB ring description](https://www.ptb.de/cms/en/ptb/fachabteilungen/abt7/ptb-sr/mls/storage-ring.html)
and [EPAC04 MLS design paper](https://epaper.kek.jp/e04/PAPERS/THPKF016.PDF).
The energy/tune targets come from Table 2 of the
[IPAC21 lattice study](https://proceedings.jacow.org/ipac2021/papers/mopab262.pdf).
Individual magnet lengths, drift allocation, gradients, sextupole settings,
and the 20 mm entrance aperture are illustrative choices. This is an
MLS-inspired DBA approximation, not an operational MLS lattice or a
reproduction of its measured chromaticities, momentum compaction or emittance.
It excludes RF, radiation, insertion devices, fringe fields and alignment errors.
This is an envelope match: the inferred magnet positions do not reproduce
the reference's peak locations or pointwise Twiss curves. Straights are
dispersive; the earlier achromatic constraint has been removed.

The optimisation varies the central Q1 quadrupole family and all three
sextupole families; the other quadrupoles remain fixed. JSON output identifies
the approximation, nominal optics, SI strengths and objective units (negative
upper-half-plane DA polygon area in mm²). Default grids/budgets are deliberately
small. Surrogate launches use 50/100 µm horizontal and 30/50 µm vertical
offsets to stay within this approximation's stable region. The default DA
scan extends to 2 mm (`--r-max` in metres), with one refinement pass.
For a finer study, increase `--turns`, `--budget`, `--angles`, `--radii`
and `--refine`; vary `--r-max` to establish that the boundary is bracketed,
and check convergence before interpreting acceptance. The bent ring is
substantially more expensive in CPU eager mode than the earlier FODO demo.


Install CMA-ES with `pip install 'slipstream-accel[optimise]'` and add
`--cma` to the example. With workspace-local cma 4.5.0, all 13 focused
CPU regression tests pass, including seeded CMA-ES quadratic convergence with anisotropic knob bounds.

The example now writes reference optics, lattice and derived-parameter tables,
plus Twiss/layout and nominal DA plots. See [ring reference outputs](ring_outputs.md)
for the reusable package APIs. Run `python examples/nonlinear_optimisation.py
--reference-only` for the lattice/parameter outputs without optimisation;
`--no-plots` needs no Matplotlib, and `--output-dir` chooses the destination.
