# `slipstream.lattice.optics`

[Source](../../../src/slipstream/lattice/optics.py) · [Documentation index](../../README.md)

## Physics

Stores single-point transverse Twiss/dispersion, tunes/chromaticities, equilibrium geometric emittances and damping times. These are reference machine values rather than a nonlinear transport integrator.

### Limits and conventions

- Emittances are geometric, not normalized.
- A per-element optics table is represented by `TwissTable`.

## Usage

Construct `TransverseOptics` alongside `Lattice`, or use its YAML round trip. Distribution matching and ring damping/IBS consume these values.

### API reference

```python
class TransverseOptics:
    ...
    def to_yaml(self, path: str | Path) -> None: ...
    def from_yaml(cls, path: str | Path) -> 'TransverseOptics': ...
```

### `TransverseOptics`

Transverse machine parameters for the single-bunch one-turn map.

`epsx_m`/`epsy_m` are geometric (not normalized) equilibrium
emittances, and `tau_x_s`/`tau_y_s` transverse damping times --
direct inputs, matching `Lattice`'s own flat-parameter style for
`tau_z_s`/`sigma_delta0` rather than deriving them from
synchrotron-radiation integrals.

## Numerical and implementation notes

### `TransverseOptics`

Dispersion slope (dD/ds) at this matched point -- optional, defaults to
flat dispersion (0.0). Not used by the betatron/damping maps above;
only consumed by slipstream.ibs's dispersion invariant H_x/H_y, where
a nonzero slope matters (e.g. a point inside a bend). Defaulting to 0
keeps every existing constructor call/YAML file valid unchanged.
