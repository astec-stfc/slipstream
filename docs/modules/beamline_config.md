# `slipstream.beamline_config`

[Source](../../src/slipstream/beamline_config.py) · [Documentation index](../README.md)

## Physics

Normalizes total energy and rest mass in eV, bunch charge in coulombs and canonical native coordinates. SDDS particle momentum is dimensionless beta*gamma; times become length-like tau through the speed of light.

### Limits and conventions

- Reference total energy must exceed rest energy; HDF5 sources carry their own energy/charge metadata.

## Usage

Use `read_config(path)` to validate a YAML study; unknown keys and invalid numeric values raise errors. Paths resolve relative to the configuration, and wake paths default to the lattice directory. `load_sdds_particles` loads one elegant-style particle page.

`overrides={"tracker.csr_transient": True, ...}` applies dotted-key values
before validation. `csr_transient` and non-default `csr_drift_model` require
`csr` and `fast_collective_effects`; unsupported devices and unavailable CUDA
are rejected with actionable errors. `lattice.strict_import` controls the
separate lattice import policy. See [the schema](../beamline_studies.md).

### API reference

```python
def read_config(path, *, overrides: dict[str, object] | None=None): ...
def load_sdds_particles(path, energy_eV, mc2_eV, device, dtype, time_origin_s=0.0): ...
```
