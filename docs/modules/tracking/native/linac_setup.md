# `slipstream.tracking.native.linac_setup`

[Source](../../../../src/slipstream/tracking/native/linac_setup.py) · [Documentation index](../../../README.md)

## Physics

Crest calibration maximizes reference-particle energy gain through the upstream line and selected cavity using a phase scan and parabolic refinement.

### Limits and conventions

- Calibration depends on upstream transport and the supplied reference particle, not just the isolated cavity.

## Usage

Use `element_index` to locate a named element, `with_element` for an immutable parameter change and `crest_phase_rad(specs, cavity_idx, ref6, energy_eV)` to obtain the absolute phase in radians.

### API reference

```python
def element_index(specs: list[ElementSpec], name: str) -> int: ...
def with_element(specs: list[ElementSpec], name: str, **fields) -> list[ElementSpec]: ...
def crest_phase_rad(specs: list[ElementSpec], cavity_idx: int, ref6: torch.Tensor, energy_eV: float, coarse_deg: float=10.0, mc2_eV: float=ELECTRON_MASS_EV) -> float: ...
```
