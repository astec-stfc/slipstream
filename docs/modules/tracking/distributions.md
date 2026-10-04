# `slipstream.tracking.distributions`

[Source](../../../src/slipstream/tracking/distributions.py) · [Documentation index](../../README.md)

## Physics

Generates Twiss-matched Gaussian, KV surface and waterbag volume distributions, plus chirped longitudinal profiles. Generated coordinates use mechanical slopes, not normalized canonical momenta; tau is metres.

### Limits and conventions

- Convert slopes to canonical momenta before native ring/linac tracking.
- A seed reproduces results on the same device; CPU and CUDA random sequences need not agree.

## Usage

Use `gaussian_matched_6d`/`gaussian_matched_optics`, `kv_4d`, `waterbag_6d` and `chirped_longitudinal`. Explicit seed, dtype and device control reproducibility and numerical representation.

### API reference

```python
def gaussian_matched_6d(n: int, beta_x_m: float, alpha_x: float, epsx_m: float, beta_y_m: float, alpha_y: float, epsy_m: float, sigma_tau_m: float, sigma_delta: float, D_x_m: float=0.0, Dp_x: float=0.0, D_y_m: float=0.0, Dp_y: float=0.0, chirp_per_m: float=0.0, device: str | torch.device | None=None, dtype: torch.dtype=torch.float64, seed: int | None=None) -> torch.Tensor: ...
def gaussian_matched_optics(n: int, optics, sigma_tau_m: float, sigma_delta: float, chirp_per_m: float=0.0, device: str | torch.device | None=None, dtype: torch.dtype=torch.float64, seed: int | None=None) -> torch.Tensor: ...
def kv_4d(n: int, beta_x_m: float, alpha_x: float, epsx_m: float, beta_y_m: float, alpha_y: float, epsy_m: float, device: str | torch.device | None=None, dtype: torch.dtype=torch.float64, seed: int | None=None) -> torch.Tensor: ...
def waterbag_6d(n: int, beta_x_m: float, alpha_x: float, epsx_m: float, beta_y_m: float, alpha_y: float, epsy_m: float, sigma_tau_m: float, sigma_delta: float, device: str | torch.device | None=None, dtype: torch.dtype=torch.float64, seed: int | None=None) -> torch.Tensor: ...
def chirped_longitudinal(n: int, sigma_tau_m: float, sigma_delta: float, chirp_per_m: float=0.0, quad_chirp_per_m2: float=0.0, profile: str='gaussian', flattop_half_width_m: float | None=None, device: str | torch.device | None=None, dtype: torch.dtype=torch.float64, seed: int | None=None) -> tuple[torch.Tensor, torch.Tensor]: ...
```

### `gaussian_matched_6d`

6D Gaussian matched to explicit per-plane Twiss (`beta`/`alpha`/
`eps`) + dispersion, plus an independent longitudinal
`sigma_tau_m`/`sigma_delta` (+ optional linear energy chirp
`delta += chirp_per_m*tau`). `(N, 6)`, module's own slope
convention -- see model description.

Each transverse plane is drawn from its own 2x2 sigma matrix
`eps*[[beta, -alpha], [-alpha, gamma]]` (`gamma=(1+alpha^2)/beta`)
via a Cholesky factor applied to iid unit Gaussians -- exact by
construction (see `test_distributions.py`'s sigma-matrix check), not
a large-N-only approximation. Dispersion is added afterward as a
deterministic `D*delta`/`Dp*delta` shift, uncorrelated with the
betatron part (the standard "dispersive orbit + betatron oscillation"
decomposition).

### `kv_4d`

4D Kapchinskij-Vladimirskij distribution: uniform on the *surface*
of the real 4D phase-space ellipsoid (every particle at the same total
normalized amplitude) -- ASTRA's `DISTRIBUTION=2`-comparable.
`(N, 4)`, columns `(x, xp, y, yp)`, slope convention.

Construction (Reiser, *Theory and Design of Charged Particle Beams*):
draw a uniform point `n` on the unit 3-sphere in R^4 (four iid unit
Gaussians, normalized), scale plane `i`'s pair by `sqrt(4*eps_i)`,
then un-normalize through each plane's own Floquet transform. No x-y
coupling (the same uniform-on-sphere `n` couples the planes only
through the projection, not through the physical optics) -- axisymmetric
optics blocks only, document this at the call site for a machine with
real x-y coupling.

### `waterbag_6d`

4D waterbag transverse distribution (uniform *volume* fill of the
same phase-space ellipsoid `kv_4d` puts on the surface of -- scale
the KV construction's unit-sphere point by `r = u**(1/4)` for
`u ~ Uniform(0,1)`, since a 4-ball's volume grows as `r**4`) packed
with an independent Gaussian longitudinal `(tau, delta)` into ``(N,
6)``. No standard 6D waterbag exists (this package's own choice,
matching `kv_4d`'s "4D + independent longitudinal Gaussian" Gotcha,
not a literature convention) -- ASTRA's uniform-ellipsoid
`DISTRIBUTION=4`-comparable transversely.

### `chirped_longitudinal`

`(tau_m, delta)` -- longitudinal profile generator for bunch-
compressor/CSR studies, taking correlation coefficients
(`chirp_per_m`/`quad_chirp_per_m2`), not a raw covariance, to keep
the call site at the physicist level.

`profile="gaussian"` (default): `tau ~ N(0, sigma_tau_m)`.
`profile="flattop"`: a uniform core of half-width
`flattop_half_width_m` (default `sigma_tau_m`) plus independent
Gaussian smoothing noise (`0.3*sigma_tau_m` rms) added on top --
 a real flat-top-with-erf-edges profile needs solving for
the edge shape or an actual convolution; this composition (uniform +
Gaussian noise) gives a flat core with softened edges at a fraction of
the code, good enough for "not a pure Gaussian" bunch-compressor input.
Add a real erf-edge solver if a study needs the edge shape itself to be
accurate, not just present.

``delta = chirp_per_m*tau + quad_chirp_per_m2*tau**2 +
Gaussian(sigma_delta)``.
