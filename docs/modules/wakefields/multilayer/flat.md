# `slipstream.wakefields.multilayer.flat`

[Source](../../../../src/slipstream/wakefields/multilayer/flat.py) · [Documentation index](../../../README.md)

## Physics

Contains the symmetric flat multilayer field-matching formulation, but raw matrix inversion can overflow float64 for realistic chamber/frequency parameters.

### Limits and conventions

- PARKED: not numerically reliable for production use.

### `Model and numerical conventions`

Flat-chamber multilayer resistive-wall impedance, ported directly from IW2D's `flat.cc`/`flatchamber.cc`
(N. Mounet & E. Metral, "Electromagnetic fields and beam coupling
impedances in a multilayer flat chamber", CERN note, 2010).

Numerically, flat chamber is a different problem from round: the
transverse dependence is Cartesian (exponentials in a Fourier-type
horizontal wavenumber `kx`), not a Bessel expansion, but `alpha_mn`
itself is a semi-infinite integral over `kx` (via `u` with
`kx=(k/gamma)*sinh(u)`) with no closed form -- `flat.cc` evaluates it
with GSL's adaptive Gauss-Kronrod quadrature to infinity (`qagiu`); this
port uses `scipy.integrate.quad`'s equivalent (QUADPACK's own infinite-
range transform), the direct Python counterpart, not a re-derivation.
Each integrand evaluation needs a 4x4 complex boundary-matching matrix
chain (`_field_matching_matrix`, ported from `multilayerm_flat`) and a
4x4 complex inverse (`_etachi`, ported from `etachi`) -- cheap per
call, but this means (unlike `round.py`'s closed-form, frequency-batched
`_alpha_tm`) evaluation is inherently a Python-level loop over frequency
and, within each frequency, an adaptive number of quadrature points -- not
batched, not GPU-vectorized, and (via `scipy.integrate.quad`) not
autograd-differentiable. This flat API provides NumPy output, not a tensor
gradient interface.

### `_csqrt`

Principal-branch complex sqrt, matching IW2D's `csqrtMP` exactly
(including its -0j -> +0j fixup right at the branch cut) -- same fixup
as `round.py`'s own `_csqrt`, reimplemented here in plain
numpy/complex rather than torch since this module's per-frequency
quadrature loop (model description) evaluates it scalar-at-a-time, far
too often to pay a torch round-trip on each call.

### `_field_matching_matrix`

4x4 field-matching matrix for one (top==bottom, by symmetry) stack
at a single frequency and horizontal wavenumber `kx` -- ported from
`flat.cc`'s `multilayerm_flat`. `eps1_inv` must be
`1/eps1_physical` (`flat.cc`'s own in-place `eps1(p)=1/eps1(p)`
reassignment before this matrix build -- see the model description's
"NB" note in the original source); mu=1 throughout (model description).

### `_integrand`

Ported from `flat.cc`'s `integrand` (`kx=(k/gamma)*sinh(u)`);
a NaN real or imaginary part (from an overflowing `kx` at large
`u`, same as the source) is replaced by zero, real/imag
independently, matching `flat.cc` exactly.

### `_u_max`

Upper bound replacing IW2D's literal semi-infinite `alphamn`
integral (model description's numerical-conditioning note): IW2D's own
160-bit arithmetic tolerates evaluating the integrand at arbitrarily
large `u` (GSL's infinite-range quadrature rule samples near-infinite
`u` on its very first panel, before any adaptive refinement) --
double precision does not, because `kx=(k/gamma)*sinh(u)` grows
exponentially with `u`, and so does `kyp~|kx|` and hence
`exp((kyp+kyp1)*b[p])` (`_field_matching_matrix`'s `exp3`/`exp4`).
`b[-1]` (the outermost material boundary) is the least forgiving --
`kyp`/`kyp1` are the same order for every boundary at large `u`,
so the largest `b` overflows first. Truncating where that product
would first reach `_MAX_SAFE_EXPONENT` is negligible, not
approximate: the true integrand there is suppressed by the reciprocal
of that same (squared) factor -- order `exp(-1200)` or smaller, many
orders below double precision's own ~1e-16 resolution -- because the
physical (decaying, not growing) branch is what a correctly-conditioned
evaluation would keep (the same "only the decaying solution survives"
principle behind `round.py`'s own overflow fix, worked out there
algebraically; here enforced by never evaluating far enough out for the
growing branch to be sampled in the first place).

## Usage

`flat_chamber_impedances` and `flat_chamber_impedance` remain experimental; ordinary aperture calculations use round matching plus Yokoya factors.

### API reference

```python
class FlatImpedances:
    ...
def flat_chamber_impedances(chamber: FlatChamber, freq_Hz: np.ndarray, gamma: float) -> FlatImpedances: ...
def flat_chamber_impedance(chamber: FlatChamber, freq_Hz: np.ndarray, gamma: float) -> Impedance: ...
```

### `flat_chamber_impedances`

All six standard outputs for `chamber` -- `flatchamber.cc`'s
`order<=0` branch (model description), drop-in (same physical
definitions and units) for what IW2D's `flatchamber.x` itself
writes. Top-bottom symmetric, non-magnetic, no differentiability (see
model description) -- unlike `round.py`'s functions, this loops over
frequency in Python (each point needs its own adaptive quadrature), so
it is much slower per point. Its conditioning limitation also applies to
offline generation; the raw solver is not a production model.

### `flat_chamber_impedance`

Return the longitudinal component of `flat_chamber_impedances` as an
`Impedance`. This experimental wrapper inherits the full flat solver's
numerical limitations. The aperture aggregation API uses the round solver
with Yokoya factors instead.
