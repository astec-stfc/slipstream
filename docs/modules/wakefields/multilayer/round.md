# `slipstream.wakefields.multilayer.round`

[Source](../../../../src/slipstream/wakefields/multilayer/round.py) · [Documentation index](../../../README.md)

## Physics

Matches nonmagnetic multilayer cylindrical fields using Drude-relaxation resistivity and background permittivity. Factored complex Bessel matrices avoid exponentiating huge intermediate values. Longitudinal mode is m=0 and equal x/y dipole mode is m=1.

### Limits and conventions

- Only mu=1 layers are supported.
- Layer ordering and accumulated boundary radii matter; the final layer is semi-infinite.
- Conditioning warnings indicate a validation-range boundary, not a proof that the result is wrong or correct.

### `Model and numerical conventions`

Round-chamber multilayer resistive-wall impedance, ported directly from IW2D's `round.cc`/`roundchamber.cc`
(N. Mounet & E. Metral, "Electromagnetic fields created by a macroparticle
in an infinitely long and axisymmetric multilayer beam pipe",
CERN-BE-2009-039, 2009), batched over the frequency array as one set of
vectorized tensor ops rather than IW2D's own per-frequency C++ loop.

Numerical conditioning: a real boundary-matching matrix chain can span
Bessel arguments from ~1e-5 (vacuum side) to ~9000 (conductor side),
overflowing raw `I_m`/`K_m` and producing `inf*0=nan` once
multiplied together. `round.cc` survives this via 160-bit arithmetic;
this port instead carries every 4x4 layer-matching matrix in a
holomorphic-safe factored form (``value = norm * exp(row_exp[i] +
col_exp[j])`, built from `modified_bessel_i_hat``/
`modified_bessel_k_hat`) and only exponentiates the *difference* of two
`col_exp` entries at the very end, in `_alpha_tm` -- provably tied to
the innermost (vacuum-side) boundary alone regardless of layer count.

### `_layer_eps`

eps1, shape (n_layers+1, n_freq) complex128 -- index 0 is vacuum
(eps=1); index p is `chamber.layers[p-1]`'s Drude-relaxation
permittivity on top of its background dielectric constant `epsb`
(`roundchamber.cc`'s ``eps1(p) = epsb + 1/(j*eps0*rho*omega*
(1+j*omega*tau))`; `epsb`` is the layer tuple's optional 4th
element, default 1 -- see `iw2d.layer_fields`).

### `_boundary_radii`

b, shape (n_layers,) float64 -- b[0] is the beam-pipe inner radius
(vacuum/layer-1 boundary); b[p] is the outer boundary of
`chamber.layers[p-1]` (= inner boundary of `chamber.layers[p]`).
The last layer's own (semi-infinite) outer boundary is never
referenced, matching `roundchamber.cc`.

### `_boundary_matrix_norm`

The O(1) "norm" part of one boundary's 4x4 field-matching matrix --
same formula as round.cc's mp1p, with every I_m(x)*K_m(y)-type raw
product replaced by the corresponding product of O(1) hat-values
(model description).

### `_scaled_matmul`

(A@B) in the (norm, row_exp, col_exp) factored representation
(value[i,j] = norm[i,j]*exp(row_exp[i]+col_exp[j])) -- safe even when
A_col+B_row would individually overflow if exponentiated directly, by
factoring out the per-frequency max real part before exponentiating
(model description).

## Usage

Pass a `RoundChamber`, Hz grid and beam gamma to `round_chamber_impedance` or `round_chamber_dipole_impedance`. `check_conditioning`/`estimate_max_bessel_argument` report difficult parameter ranges.

### API reference

```python
def estimate_max_bessel_argument(chamber: RoundChamber, freq_Hz: np.ndarray, gamma: float) -> float: ...
def check_conditioning(chamber: RoundChamber, freq_Hz: np.ndarray, gamma: float, warn_factor: float=1.0) -> None: ...
def round_chamber_impedance(chamber: RoundChamber, freq_Hz: np.ndarray, gamma: float, check: bool=True) -> Impedance: ...
def round_chamber_dipole_impedance(chamber: RoundChamber, freq_Hz: np.ndarray, gamma: float, check: bool=True) -> Impedance: ...
```

### `estimate_max_bessel_argument`

Max |nu_layer * boundary_radius| this geometry/frequency range would
evaluate Bessel functions at, across every layer boundary -- cheap
(just the layer wavenumbers, no Bessel calls or matrix chain needed),
a direct proxy for how far outside `VALIDATED_MAX_ARGUMENT` a new
geometry pushes.

### `check_conditioning`

Warns (does not raise or block) if this geometry/frequency range
pushes `estimate_max_bessel_argument` beyond the historical live
IW2D comparison envelope. Not a proof of correctness beyond
the validated range -- a concrete, actionable "you're outside the
tested envelope, consider a live IW2D cross-check for this specific
case before trusting it" signal, rather than silently returning a
number either way.

### `round_chamber_impedance`

Return longitudinal impedance in the NumPy-backed `Impedance` container.
The internal field-matching algebra uses PyTorch, but the public chamber
path is not a gradient-preserving layer-parameter API: scalar/NumPy setup,
boundary tensor construction and NumPy output conversion impose limits.
See `round_chamber_dipole_impedance` for the transverse (m=1) counterpart.

`check=True` (default) runs `check_conditioning` first -- cheap
(no Bessel evaluation), and warns rather than raising, so it never
blocks a run, just flags an untested regime.

### `round_chamber_dipole_impedance`

Zxdip(f) = Zydip(f) for `chamber` -- the m=1 transverse-dipole
mode (`roundchamber.cc`'s `alphaTM1` branch: same `alphaTM`
field-matching chain as `round_chamber_impedance`'s Zlong, called
with m=1 instead of m=0, times the extra `k/(2*gamma^2)` dipole
prefactor `roundchamber.cc` applies on top of the longitudinal
`cst`). Round (axisymmetric) chambers have Yokoya factors
xdip=ydip=1, so a single result serves both planes -- same
`Impedance` contract as `round_chamber_impedance`, but in Ohm/m
(transverse), not Ohm; drop-in replacement for
`wakefields.iw2d.total_cached_component`'s cached Zxdip/Zydip
output.

`check=True` (default) runs `check_conditioning` first, same as
`round_chamber_impedance` -- the conditioning bound was derived from
the m=0 case, but the same boundary-matrix chain (only the Bessel
order and an m>0 TM-TE coupling term differ) is expected to share it.

## Numerical and implementation notes

### `_boundary_matrix_norm`

The four products are reused throughout both polarisation blocks.
Keep them named after their Bessel factors to make the matrix below
easier to compare with the corresponding analytic expression.
