# `slipstream.analysis.nonlinear_opt`

[Source](../../../src/slipstream/analysis/nonlinear_opt.py) · [Documentation index](../../README.md)

## Physics

Tensor magnet strengths support differentiable chromaticity and tune-footprint objectives. Null-space knobs preserve chromaticity to first order; action-growth and footprint losses are smooth surrogates for survival aperture.

### Limits and conventions

- A surrogate improvement requires independent long-turn DA/FMA validation.
- Loss bookkeeping itself is not differentiable; launches for action-growth loss must survive inside the aperture.

## Usage

Create `KnobSet` groups, form a `chromaticity_response` and `null_basis`, and build a small launch set for `Footprint`. `gradient_free` offers bounded SciPy baselines; `multi_fidelity` combines screening and explicit high-fidelity evaluation.

### API reference

```python
def periodic_specs(specs: list[ElementSpec], names: tuple[str, ...]=SEXTUPOLES, n_periods: int=2) -> list[ElementSpec]: ...
def unique_names(specs: list[ElementSpec], kinds: tuple[str, ...]=('quad', 'bend')) -> list[ElementSpec]: ...
class KnobSet:
    ...
    def __init__(self, specs: list[ElementSpec], mode: str, combined: tuple[str, ...]=('BD', 'BF')): ...
    def k2(self, theta: torch.Tensor) -> dict[str, torch.Tensor]: ...
    def ring(self, base: RingMap, theta: torch.Tensor, extra_k1: dict | None=None, extra_k2: dict | None=None) -> RingMap: ...
    def elegant_k2(self, theta: torch.Tensor, specs: list[ElementSpec]) -> dict: ...
def chromaticity_response(ring: RingMap, knobs: KnobSet, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...
def null_basis(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...
class Launches:
    ...
def build_launches(opt_by_delta: dict[float, nl.LinearOptics], amps_mm, chrom_deltas, a_small_mm: float=0.3, device='cpu', diff_points: tuple=()) -> Launches: ...
class Footprint:
    ...
    def __init__(self, launches: Launches, opt_norm: nl.LinearOptics, xi_target: tuple[float, float], n_turns: int=256, checkpoint: bool=False, weights: dict[str, float] | None=None, n_machines: int | None=None, std_weight: float=0.0, diff_weight: float=0.0, diff_scale: float=1e-05, lost_penalty: float=1000.0, loss_aperture: float=0.1): ...
    def analyse(self, ring: RingMap) -> tuple[torch.Tensor, ...]: ...
    def tunes(self, ring: RingMap) -> tuple[torch.Tensor, torch.Tensor]: ...
    def shifts(self, ring: RingMap) -> torch.Tensor: ...
    def per_machine(self, ring: RingMap) -> torch.Tensor: ...
    def loss(self, ring: RingMap) -> torch.Tensor: ...
def sample_field_errors(specs: list[ElementSpec], rules: list[tuple[str, str, float]], n_machines: int, seed: int=0, dtype=torch.float64, device='cpu', cutoff: float=3.0) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]: ...
def action_growth_loss(tbt: torch.Tensor, opt: nl.LinearOptics, scale: float=1.0) -> torch.Tensor: ...
def joint_optics_loss(ring: RingMap, target: nl.LinearOptics, tune_scale: float=0.001, beta_scale: float=0.01) -> torch.Tensor: ...
class SearchResult:
    ...
def gradient_free(objective, theta: torch.Tensor, bounds, method: str='Powell', max_evaluations: int=100, seed: int=0) -> SearchResult: ...
def multi_fidelity(smooth_objective, aperture_objective, theta: torch.Tensor, bounds, gradient_steps: int=20, learning_rate: float=0.01, max_evaluations: int=100, seed: int=0, method: str='differential_evolution') -> tuple[SearchResult, SearchResult]: ...
```

### `build_launches`

A small-amplitude reference (delta = 0), amplitude scans in x, y and on the diagonal at delta = 0, and
small-amplitude launches at `chrom_deltas` (each on that delta's closed orbit from `opt_by_delta`).
`diff_points` = `(x_mm, y_mm, delta)` triples of *large-amplitude* launches (each `delta` must be a key of
`opt_by_delta`) whose tune diffusion between the two halves of the run is penalised -- the smooth stand-in for
frequency-map chaos near the dynamic-aperture edge.

### `Footprint`

`loss(ring)`: a smooth surrogate for dynamic aperture, differentiable in every tensor strength of `ring`
(NAFF and the tracking both are). Two parts over one small tracked launch set of `n_turns` turns:

- *footprint*: mean squared tune shift `[Q(x, y, delta) - Q(ref) - xi_target delta]` over the `amp` and
  `chrom` launches, in units of 1e-3 -- the amplitude detuning plus the higher-order chromaticity the sextupole
  scheme leaves behind, at fixed linear chromaticity (a compact footprint crosses fewer resonances);
- *diffusion* (`diff_weight > 0`, `diff` launches): `mean log(1 + (D / diff_scale)^2)` with `D` the tune
  change between the two halves of the run -- the frequency-map diffusion, which stays ~1e-6..1e-5 on regular
  motion and jumps toward 1e-3 as the particle nears chaos. It keeps the optimiser from buying a small footprint
  with large-amplitude stability. A `diff` launch lost inside `n_turns` counts as `D = 0.1`; a lost
  `amp`/`chrom` launch contributes the constant `lost_penalty` (squared tune shift, units of 1e-6) to the footprint
  term -- neither has a gradient, so the optimiser is pushed back out of that region rather than crashing.

`n_machines=M` tracks the launch set on `M` machines at once (`ring` strengths `(M, 1)`); `loss` is then
the mean over machines plus `std_weight` times their standard deviation (a robust objective).

### `sample_field_errors`

Per-element Gaussian field errors for `n_machines` machines, as `(M, 1)`
`(k1 dict, k2 dict)` ready for `RingMap.with_strengths` (specs must have per-element names,
`unique_names`). `rules` are `(kind, parameter, rms)` with `kind` an `ElementSpec.kind`
(`"quad"`, `"bend"`), `parameter` `"K1"` or `"K2"` and `rms` *relative* to the element's own
value (K1 for quads and bends; K2 of a bend). Gaussians are truncated at `cutoff` sigma.

### `action_growth_loss`

Smooth soft-survival surrogate: mean squared soft positive action growth.

Normalized action is `|z|**2/2`; reference is the first recorded turn.
Keep every launch inside the aperture: this cannot differentiate losses.
`scale` sets the acceptable fractional action growth.

### `joint_optics_loss`

Differentiable tune and fractional beta constraints for K1/K2 optimisation.

Add this dimensionless penalty to a footprint/detuning objective. Beta
constraints refer to the observation point, not an entire-ring envelope.

### `gradient_free`

Bounded Powell, Nelder-Mead or differential-evolution baseline.

Objective accepts a tensor with theta's dtype/device. Each evaluation is
detached; hardware synchronization here is intentional optimiser control.
SciPy is a core package dependency. Budgets count objective calls.

### `multi_fidelity`

Gradient screening followed by bounded gradient-free DA refinement.

Both objectives are minimised (use negative DA area for the second).
The high-fidelity objective must retrack with its own long survival run;
it receives only strengths, never a surrogate survival estimate. Returns
independently measured best candidates from both stages. This uses DE,
Powell, Nelder-Mead or optional CMA-ES (install `cma`).

## Numerical and implementation notes

### `Footprint.analyse`

A launch that is lost inside the run blows up (inf/NaN) in the turn it goes, and although the loss mask
zeroes its state, the backward pass multiplies a zero gradient by the NaN local derivative and that NaN
is summed into every strength's gradient. So find the lost launches with a cheap no-gradient pass
(CUDA-graph path) and start them from a harmless launch in the tracked-with-gradient pass.
