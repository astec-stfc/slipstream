# `slipstream.wakefields.chambers`

[Source](../../../src/slipstream/wakefields/chambers.py) · [Documentation index](../../README.md)

## Physics

Stores round/flat chamber geometry, section length and material layers. Layers encode resistivity, relaxation time in ps, thickness in metres and optional background permittivity.

### Limits and conventions

- These are model inputs; no external IW2D executable is launched.

## Usage

Construct `RoundChamber`/`FlatChamber` and a `FreqScan`; `build_freq_hz` generates the frequency grid. `layer_fields` normalizes three/four-value layer tuples.

### API reference

```python
class RoundChamber:
    ...
class FlatChamber:
    ...
def layer_fields(layer: tuple) -> tuple[float, float, float, float]: ...
class FreqScan:
    ...
    def build_freq_hz(self) -> np.ndarray: ...
```
