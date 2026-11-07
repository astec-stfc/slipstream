# `slipstream.tracking.native.optics`

[Source](../../../../src/slipstream/tracking/native/optics.py) · [Documentation index](../../../README.md)

## Physics

Linearizes actual native per-element transport using autograd, solves periodic per-plane Twiss and dispersion and propagates element-local values. Periodic dispersion solves `(I-R)*D=r`.

### Limits and conventions

- Periodic optics requires stable supported per-plane transport.
- Use the same slicing and edge conventions as the tracker for consistent optics.

### `_element_R`

6x6 linear transfer matrix for one element, via autograd Jacobian of
`elements.track_element` at the reference orbit (`state6=0`) --
`lru_cache`d on the plain-float params since a real ring reuses the
same few magnet *types* many times.

### `_propagate_twiss`

Standard sigma-matrix Twiss propagation, ``sigma1 = R @ sigma0 @
R.T` for `sigma = [[beta, -alpha], [-alpha, gamma]]``, expanded to
the usual closed-form beta/alpha update -- `R` is one plane's 2x2
transfer-matrix block.

### `_periodic_plane`

Periodic (matched) beta/alpha/dispersion/tune for one plane's 2x2
transfer-matrix block and delta-column coupling -- same formulas as
## Usage

Call `periodic_twiss_table` with element specs, reference energy and matching integration settings to obtain a `TwissTable`, including IBS segment weights.

### API reference

```python
def periodic_twiss_table(specs: list[ElementSpec], energy_eV: float, mc2_eV: float, n_kicks: int=16, n_kicks_bend: int=64, integration_order: int=4, n_kicks_quad: int | None=None, n_kicks_sext: int | None=None) -> tuple[TwissTable, float, float]: ...
```

### `periodic_twiss_table`

Real per-element `TwissTable` (beta(s)/alpha(s)/dispersion(s) at
every element's *entrance*, `s_m[i]` the cumulative path length up to
that point) from the periodic solution of the ring's own linear
one-turn matrix -- feed this directly to
`ibs.element_averaged_growth_rates` for a genuinely per-element IBS
calculation using this tracker's own real lattice (no MAD-X round trip
needed). Returns `(table, Qx_frac, Qy_frac)` -- the fractional tunes
aren't part of `TwissTable` itself (a MAD-X-derived one has no single
natural "one-turn matrix" reference point the same way), but are cheap
byproducts of the same one-turn-matrix computation, so returned
alongside rather than silently discarded.

`n_kicks`/`n_kicks_bend` default to 16/64 here -- **deliberately
much higher than `NativeTracker`'s own per-turn tracking defaults of
1/4**: a per-element linearization error that's negligible for per-
particle tracking compounds multiplicatively across a 290-fold matrix
product for tune/beta extraction, and `beta=M12/sin(mu)` is highly
sensitive to the resulting phase error. This is a one-time,
`lru_cache`d cost with no per-turn tracking-speed impact, so there's
no reason to reuse the tracker's cheaper default here.
`n_kicks_quad`/`n_kicks_sext` (default `None`, meaning `n_kicks`)
override the count for `quad`/`sext` elements, matching
`RingMap`'s per-kind kick counts. See
docs/tracking_native.md for the convergence numbers.
