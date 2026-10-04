# `slipstream.analysis.nonlinear`

[Source](../../../src/slipstream/analysis/nonlinear.py) · [Documentation index](../../README.md)

## Physics

Computes fixed-energy ring closed orbits, Jacobian optics, chromaticity, NAFF frequencies, amplitude detuning, dynamic aperture and frequency-map diffusion. Tunes are fractional in `[0,1)`; launch offsets are physical metres at the ring entrance.

### Limits and conventions

- Converge integration slicing and observation length before trusting nonlinear results.
- Uncoupled optics rejects coupling; coupled eigenvector derivatives become ill-defined at degenerate tunes.
- Spectral lines are observables, not an automatic normal-form resonance-driving-term calculation.

### `Model and numerical conventions`

Tunes are fractional, in `[0, 1)`. Amplitudes are physical offsets at the
ring start (`s=0`, the first element's entrance), launched from the closed
orbit with zero momentum offset, like elegant's `find_aperture`/
`frequency_map`.

### `_closed_orbit_and_matrix`

Newton fixed point of the one-turn map, differentiable in any
tensor strength `ring` carries (`create_graph=True`).

The convergence check runs *after* the Newton update, not before --
deliberately, not merely for the usual "check what you just computed"
reason. At the LOCO operating point (correctors and K1 errors exactly
at their nominal value), the starting guess `co=0` already satisfies
`ring.turn(0)[:4] == 0` to machine precision (F(0)=0 for any purely
multipole ring with correctors at zero), so a check-then-update loop
would `break` on iteration 1 *before* ever computing an update --
returning the original `torch.zeros(...)` tensor completely
disconnected from the autograd graph (no `grad_fn` at all), silently
dropping the closed-orbit self-consistency (wraparound) term from
`d(co)/d(any strength)` -- exactly the derivative LOCO's ORM needs.
Running the update first, using the *symbolic* (graph-connected)
`resid` even when its *value* is exactly zero, reproduces the
correct implicit-function-theorem gradient
(`d(co)/d(theta) = -(J-I)^-1 @ d(resid)/d(theta)`) from the very
first iteration -- found via a real-booster story computing an exactly
zero d(ORM)/dK1 for a K1 knob upstream of every tapped corrector/BPM
pair,

## Usage

Construct a `RingMap`, then use `closed_orbit`, `linear_optics`/`coupled_optics`, `chromaticity`, `tunes`, `detuning`, `dynamic_aperture` or `frequency_map`. `naff_harmonics` extracts successive lines; `spectral_lines` fits specified frequencies.

### API reference

```python
class LinearOptics:
    ...
def closed_orbit(ring: RingMap, delta: float=0.0, iters: int=8, create_graph: bool=False) -> torch.Tensor: ...
def linear_optics(ring: RingMap, delta: float=0.0, create_graph: bool=False) -> LinearOptics: ...
class CoupledOptics:
    ...
def coupled_optics(ring: RingMap, delta: float=0.0, create_graph: bool=False) -> CoupledOptics: ...
def coupled_signals(tbt: torch.Tensor, opt: CoupledOptics) -> tuple[torch.Tensor, torch.Tensor]: ...
def chromaticity(ring: RingMap, ddelta: float=0.0001, create_graph: bool=False) -> tuple[torch.Tensor, torch.Tensor]: ...
def naff(z: torch.Tensor, window_order: int=1, newton_iters: int=5, pad: int=4, center: torch.Tensor | float | None=None, half_width: float=0.1) -> tuple[torch.Tensor, torch.Tensor]: ...
def normalized_signals(tbt: torch.Tensor, opt: LinearOptics) -> tuple[torch.Tensor, torch.Tensor]: ...
def launch(opt: LinearOptics, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: ...
def tunes(ring: RingMap, opt: LinearOptics, state: torch.Tensor, n_turns: int, checkpoint: bool=False, band: float | None=None, centers: tuple | None=None, **naff_kw) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
class Detuning:
    ...
def detuning(ring: RingMap, x_max: float, y_max: float, n_amp: int=6, n_turns: int=512, delta: float=0.0, opt: LinearOptics | None=None, band: float | None=0.05) -> Detuning: ...
class DynamicAperture:
    ...
def dynamic_aperture(ring: RingMap, r_max: float, n_turns: int, n_angles: int=19, n_r: int=32, refine: int=2, delta: float=0.0, aperture: tuple[float, float]=(0.05, 0.05), batch_shape: tuple[int, ...]=(), opt: LinearOptics | None=None) -> DynamicAperture: ...
class FrequencyMap:
    ...
def frequency_map(ring: RingMap, x_range: tuple[float, float], y_range: tuple[float, float], nx: int=41, ny: int=21, n_turns: int=1024, delta: float=0.0, aperture: tuple[float, float]=(0.05, 0.05), opt: LinearOptics | None=None, band: float | None=0.1) -> FrequencyMap: ...
def naff_harmonics(z: torch.Tensor, n_harmonics: int=3, **naff_kw) -> tuple[torch.Tensor, torch.Tensor]: ...
def spectral_lines(z: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor: ...
```

### `coupled_optics`

Stable 4D normal modes from symplectically normalized eigenvectors.

Modes are sorted by fractional tune. Eigenvectors/gradients are ill-defined
at degenerate tunes; avoid crossings in differentiated objectives. This
frozen-energy API excludes longitudinal coupling and radiation damping.

### `naff`

Dominant frequency (turns^-1, in `[0, 1)`) and amplitude of each
complex signal `z` `(..., T)` -- Laskar's NAFF first harmonic:
Hann^`window_order` window, zero-padded FFT peak as the seed, then
Newton refinement of `argmax |sum_n w_n z_n exp(-2 pi i nu n)|`.
Error on a pure tone ~1e-10 at T=1024 (float64); see
tests/test_ring_nonlinear.py.

`center` (broadcastable to `z.shape[:-1]`) restricts the seed to
`|nu - center| < half_width`: with a large amplitude in one plane the
other plane's signal carries driven lines (`2 Q_y`... ) that can
out-weigh its own small betatron line, and an unrestricted peak search
then returns the wrong line (a ~0.2 "tune shift").

### `tunes`

Track `state` `n_turns` (recording every turn) and NAFF both
planes. Returns `(Qx, Qy, lost)`; lost particles' tunes are NaN.
`state` may sit at any `delta` (NAFF removes the mean, so the launch
need not be exactly on that `delta`'s closed orbit); `checkpoint`
recomputes each turn in the backward pass (memory O(1 turn)).
`band` (half-width, tune units) restricts each plane's NAFF seed to
`|nu - Q| < band` around `centers = (Qx, Qy)` (default the linear
tunes of `opt`) -- see `naff`.

### `detuning`

First-order amplitude-detuning coefficients (per metre of action,
`J = gamma*u^2/2` for a pure-offset launch) from a linear+quadratic
least-squares fit of NAFF tunes over `n_amp` amplitudes up to
`x_max` (y fixed at `y_max/(2 n_amp)`) and up to `y_max` (x fixed
at `x_max/(2 n_amp)`), all in one batch. The fixed other-plane
amplitude only offsets the fitted line; it must stay well above the
working-precision noise floor (a 1 um amplitude in float32 gave a
cross-term 100x off).
`band`: NAFF seed window around the linear tunes (see `naff`) -- keeps
the fit on the betatron line when the other plane's amplitude drives lines
that out-weigh it.
Differentiable through NAFF and the fit (keep amplitudes inside the DA).

### `dynamic_aperture`

Upper-half-plane DA along `n_angles` rays in `[0, pi]` (elegant
`find_aperture` `mode="n-lines"` geometry): `n_r` radii per ray in
one batched run; the boundary on each ray is the last radius before the
*first* loss (islands beyond it don't count). `refine` further batched
passes subdivide each ray's bracketing interval by `n_r`, so the
final resolution is `r_max / n_r**(refine+1)`. `batch_shape` prefixes
the particle batch -- e.g. `(M,)` with a `with_strengths` error
ensemble of shape `(M, 1)`, launched from the nominal closed orbit.

### `frequency_map`

Laskar FMA on an `nx x ny` grid of initial offsets: one run of
`n_turns`, NAFF on each half, diffusion `log10 sqrt(dQx^2+dQy^2)`.
Keep `y_range[0] > 0` (zero vertical amplitude has no vertical tune).
`band`: NAFF seed window `|nu - Q| < band` around the linear tunes (see `naff`) --
without it a small-`x`, large-`y` launch returns the driven `2 Q_y` line as its
horizontal "tune"; `None` restores the unrestricted peak search.

### `naff_harmonics`

Successive NAFF lines and complex coefficients, shaped `(..., H)`.

Each line is removed before finding the next. Coefficients are jointly
refitted by complex least squares, preserving phase and autograd. Lines
must be separated by several Fourier bins; nearly degenerate lines are
ill conditioned. Peak selection is discrete, so derivatives are local.

### `spectral_lines`

Complex amplitudes at specified lines by a joint least-squares fit.

Frequencies broadcast as `(..., H)`. For RDT studies supply known
combinations of the fundamental tunes; these amplitudes are spectral
observables, not convention-independent Hamiltonian RDT coefficients.

## Numerical and implementation notes

### `Module constants`

vectorize=True: ~4x faster than 6 separate backward passes; forward
mode would be faster still but can't create_graph (torch 2.14). On the
real booster, however, vectorized reverse-mode retains a multi-GB graph
even when the result is not differentiable; keep ordinary closed-orbit
probes bounded and retain vectorization for the higher-order path.
