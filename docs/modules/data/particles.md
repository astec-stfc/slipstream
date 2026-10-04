# `slipstream.data.particles`

[Source](../../../src/slipstream/data/particles.py) · [Documentation index](../../README.md)

## Physics

The HDF5 particle schema distinguishes canonical `(x,px,y,py,tau,delta)` from slope `(x,xp,y,yp,tau,delta)` coordinates. Tau is always metres, and metadata stores signed total charge, reference total energy and rest energy.

### Limits and conventions

- Readers must inspect `canonical` and `tau_units` rather than assuming coordinate conventions.

## Usage

Use `save_particles_h5(path, particles6, charge_C, energy_eV, mc2_eV, canonical=True)` and `load_particles_h5(path, device, dtype)`. On disk, `particles6` is float64 and the loader returns tensor plus metadata.

### API reference

```python
def save_particles_h5(path: str | Path, particles6: torch.Tensor, charge_C: float, energy_eV: float, mc2_eV: float, canonical: bool=True) -> None: ...
def load_particles_h5(path: str | Path, device: str | torch.device | None=None, dtype: torch.dtype=torch.float64) -> tuple[torch.Tensor, dict]: ...
```
