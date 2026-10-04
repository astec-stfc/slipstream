# `slipstream.lattice.apertures`

[Source](../../../src/slipstream/lattice/apertures.py) · [Documentation index](../../README.md)

## Physics

Builds longitudinal resistive-wall impedance from shared named materials and chamber sections. Round sections use multilayer matching; flat sections use a Yokoya correction to the equivalent round chamber at the smaller half-aperture.

### Limits and conventions

- The aperture aggregation API is longitudinal-only.
- Yokoya flat geometry is an approximation; the standalone full flat solver is parked.

### `Model and numerical conventions`

Round apertures use multilayer field matching. Flat apertures use a Yokoya
elliptic form factor on the equivalent round chamber with radius equal to
the smaller half-aperture. The full flat solver is numerically unreliable
and is not used by this aggregation API.

Materials are resolved by name when impedance is computed. Editing
`Apertures.materials[name]` updates every section sharing that material.
Use `Aperture.material_overrides` for a section-specific change.

## Usage

Load `Apertures.from_yaml`, inspect `resolved_layers`, compute `aperture_impedance` or summed `impedance`, and save with `to_yaml`. Edit shared `materials[name]` or use `material_overrides` for a section.

### API reference

```python
class Material:
    ...
class Aperture:
    ...
class Apertures:
    ...
    def resolved_layers(self, aperture_name: str) -> list[tuple[float, float, float]]: ...
    def default_length_m(self, circumference_m: float) -> float: ...
    def total_length_m(self, aperture_name: str, circumference_m: float) -> float: ...
    def aperture_impedance(self, aperture_name: str, freq_Hz: np.ndarray, gamma: float, circumference_m: float) -> Impedance: ...
    def impedance(self, freq_Hz: np.ndarray, gamma: float, circumference_m: float) -> Impedance: ...
    def from_yaml(cls, path: str | Path) -> 'Apertures': ...
    def to_yaml(self, path: str | Path) -> None: ...
```

### `Material`

Either a leaf material (`rho`/`tau_ps`/`thickness_m`, the same
Drude-relaxation layer `RoundChamber.layers` already uses) or a
composite (`layers`: an ordered list of other material names,
innermost first -- e.g. a NEG coating over a stainless substrate).

### `Apertures`

The whole Materials + Apertures set from one YAML file. Not derived
from `Lattice` and doesn't duplicate any of its fields (circumference
is passed into `impedance`/`default_length_m` instead) -- mirrors
how `resonators.yaml` sits alongside, not inside, `lattice.yaml`.

### `Apertures.aperture_impedance`

Zlong(f) for one named aperture (its full `count`-repeated
length, or the Default aperture's residual length) -- round
chambers exactly, flat chambers via the Yokoya approximation
(model description).
