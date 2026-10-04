# `slipstream.ibs`

[Source](../../src/slipstream/ibs.py) · [Documentation index](../README.md)

## Physics

Computes amplitude-normalized intrabeam-scattering growth rates in inverse seconds. Bane is a high-energy approximation; Bjorken-Mtingwa retains the general relativistic factors and evaluates an integral. Element averaging weights local optics by physical segment lengths.

### Limits and conventions

- Bane is unsuitable for nonrelativistic rings.
- General-theory rates can be negative in a plane because IBS redistributes phase-space widths; do not clamp them solely for that reason.

### `Model and numerical conventions`

- `bane_growth_rates`: K. Bane's closed-form high-energy approximation
  to Bjorken-Mtingwa/Piwinski (SLAC-PUB-9226 / arXiv:physics/0205058
  (2002), Eqs. 8-14). Fast (no numerical integration), but assumes
  ultra-relativistic beam_rel~1 and gamma>>1 throughout its derivation --
  unsuitable for nonrelativistic rings.
- `bjorken_mtingwa_growth_rates`: the general theory these approximate,
  transcribed from Kubo, Mtingwa & Wolski, "Intrabeam scattering formulas
  for high energy beams," Phys. Rev. ST Accel. Beams 8, 081001 (2005)
  (KMW's own Eqs. 4-9 restate Bjorken & Mtingwa, Part. Accel. 13, 115
  (1983) directly, not just KMW's own high-energy CIMP approximation --
  see that paper's Eq. 4 vs Eq. 35 for the distinction). No high-energy
  assumption anywhere -- keeps the full beta_rel^3*gamma^4 dependence Bane
  drops -- at the cost of a numerical integral (`scipy.integrate.quad`)
  per plane per call. S. Nagaitsev's "Intrabeam scattering formulas for
  fast numerical evaluation," Phys. Rev. ST Accel. Beams 8, 064403 (2005)
  gives a faster closed-form reformulation of this same integral (not
  implemented here).

### `_g`

Bane's fitting approximation to the g(a/b) auxiliary integral (Eqs.
11-12): g(alpha) = g(1/alpha) (stated explicitly in the paper), so only
alpha in (0, 1] is ever evaluated by the fit; the fit's stated validity
floor is alpha > 0.01, clamped rather than extrapolated below it.

### `_bm_matrices`

The three auxiliary 3x3 matrices L^(p), L^(h), L^(v) (KMW Eqs. 7-9),
index order (horizontal-momentum-like, momentum-spread-like,
vertical-momentum-like) so index 1 (0-based, the middle one) is the
one every plane couples through. `phi = D' + alpha*D/beta` (KMW's
`eta' - 0.5*beta'*eta/beta`, using beta'=-2*alpha, same identity
`dispersion_invariant` already relies on).

### `_bm_integral`

KMW Eq. 4's lambda-integral for one plane's numerator matrix Li
(Lp, Lh, or Lv), i.e. everything inside the outer 4*pi*A*(log) --
dimensionless (L's entries are all dimensionless, see ibs.py's module
docstring's dimensional-analysis note in the git history/.

L's diagonal entries span many orders of magnitude for a real beam
(`gamma^2/sigma_p^2` alone is ~1e13 for a GeV-scale, 1e-4-level
sigma_p beam) -- raw quadrature over that range is numerically
unreliable (this exact difficulty is why Nagaitsev's reformulation
exists). Substitute lambda = kappa*u with kappa = |det(L)|^(1/3) (the
geometric-mean eigenvalue) first, so the rescaled matrix L~ = L/kappa
has an O(1)-ish eigenvalue spread and quad's adaptive subdivision can
actually resolve it. Li itself is *not* rescaled (only the
L~+u*I matrix being inverted/determinant'd is) -- this only rescales
the integration variable, it doesn't change what's being integrated.

## Usage

Use `bane_growth_rates` or `bjorken_mtingwa_growth_rates` with lattice, optics and bunch parameters; `element_averaged_growth_rates` consumes a `TwissTable`. Helpers include `dispersion_invariant` and `coulomb_log`.

### API reference

```python
class IBSGrowthRates:
    ...
def dispersion_invariant(D_m: float, Dp: float, beta_m: float, alpha: float) -> float: ...
def coulomb_log(sigma_x_m: float, sigma_y_m: float, sigma_z_m: float, gamma: float, beta_rel: float, sigma_p: float, n_particles_bunch: float) -> float: ...
def bane_growth_rates(*, gamma: float, n_particles_bunch: float, sigma_z_m: float, sigma_p: float, epsx_m: float, epsy_m: float, beta_x_m: float, alpha_x: float, D_x_m: float, Dp_x: float, beta_y_m: float, alpha_y: float, D_y_m: float, Dp_y: float, coulomb_log_value: float | None=None) -> IBSGrowthRates: ...
def bjorken_mtingwa_growth_rates(*, gamma: float, beta_rel: float, n_particles_bunch: float, sigma_z_m: float, sigma_p: float, epsx_m: float, epsy_m: float, beta_x_m: float, alpha_x: float, D_x_m: float, Dp_x: float, beta_y_m: float, alpha_y: float, D_y_m: float, Dp_y: float, coulomb_log_value: float, particle_radius_m: float=R_E_M) -> IBSGrowthRates: ...
def element_averaged_growth_rates(twiss: TwissTable, *, model: str='bjorken-mtingwa', gamma: float, n_particles_bunch: float, sigma_z_m: float, sigma_p: float, epsx_m: float, epsy_m: float, coulomb_log_value: float, beta_rel: float | None=None, particle_radius_m: float=R_E_M) -> IBSGrowthRates: ...
```

### `coulomb_log`

Piwinski's Coulomb logarithm ln(b_max/b_min) (see e.g. the CERN
Accelerator School's IBS lectures for the derivation). `b_max` is
capped by the beam's own size or mean interparticle spacing (a
collision can't have an unperturbed-trajectory impact parameter wider
than the beam itself); `b_min` is the classical distance of closest
approach for a collision at the beam's characteristic relative angle
sigma_p (the quantum/de Broglie floor is neglected -- always negligible
at these energies/densities). Enters the growth rate only
logarithmically, so this level of approximation is standard practice.

### `bane_growth_rates`

IBS growth rates (Bane Eqs. 8-14) at a single smooth-lattice optics
point. `epsx_m`/`epsy_m` are geometric rms *betatron* emittances
(matching `TransverseOptics.epsx_m`'s convention) -- the dispersive
contribution to the physical beam size is added internally via
`D_x_m`/`D_y_m`, not folded into the emittance passed in.
`coulomb_log_value`, if not given, is computed from the beam's own
density via `coulomb_log`.

### `bjorken_mtingwa_growth_rates`

Compute general Bjorken-Mtingwa growth rates without the high-energy
approximation. Supply beta_rel and coulomb_log_value explicitly. Use the
appropriate classical particle radius; the default is the electron radius.

Input dispersions use the raw canonical-momentum (pt) convention of
MAD-X/xsuite Twiss tables. The implementation scales them by beta_rel to
the physical relative-momentum (deltap) convention required by the L matrices.
At ultra-relativistic energy this conversion is nearly unity; it matters
at low beta. Betatron emittances exclude the dispersive size contribution.

### `element_averaged_growth_rates`

Evaluate local rates along a Twiss table and weight them by physical segment
length. The default method is the general Bjorken-Mtingwa integral. Select
Bane only in its high-energy regime. Smooth-lattice optics can differ
substantially from this element average.

This helper loops over CPU rows; the general calculation requires adaptive
quadrature for each plane. It is intended for parameter calculations rather
than a call at every tracking turn.

