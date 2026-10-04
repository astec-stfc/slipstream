# `slipstream.tracking.native.laura_import`

[Source](../../../../src/slipstream/tracking/native/laura_import.py) · [Documentation index](../../../README.md)

## Physics

Maps LAURA section records into the native element specification used for linac transport. Magnetic/RF parameters and optional structure wakes retain the same physical interpretation as the lattice importer.

### Limits and conventions

- Unsupported elements follow the current mapper's fallback rules; a drift is not a substitute for the original element's missing physics.

## Usage

Use `parse_laura_lattice(machine_module, section_name, load_wakes=True)`. LAURA and its lattice packages are optional dependencies imported by this module.

### API reference

```python
def parse_laura_lattice(machine_module, section_name: str, load_wakes: bool=True) -> list[ElementSpec]: ...
```

### `parse_laura_lattice`

Real per-element linac list for `LinacTracker`, in beamline order --
the LAURA-sourced equivalent of `lattice_import.parse_elegant_lattice`.

Dispatch mirrors `parse_elegant_lattice` exactly: quadrupole/sextupole/
dipole/RF-cavity keep their real physics, anything else with nonzero
length becomes a plain drift (correctors, apertures, diagnostics --
real elegant CLARA `.lte` files have no transport model for these
either), zero-length dropped. Same 1-based `occurrence` counters, for
the same downstream reason (error-matching identity, see `errors.py`).

`load_wakes` (default `True`): a cavity's structure wake, if it has
one, is read directly from LAURA's own HDF5 wakefield format (see
`laura_fields.wake_from_laura_hdf5`) rather than via
`linac.resolve_wakes`'s SDDS-only path -- set `False` to skip this
(e.g. to inspect geometry only, or if a wake file is temporarily
unavailable).

#### Parameters
machine_module
    e.g. `laura_lattices.CLARA` -- passed straight to `LAURA(lattice=...)`.
section_name
    e.g. `"S02"` -- a key of `LAURA(...).sections`.

## Numerical and implementation notes

### `_to_element_spec`

LAURA has no CSRCSBEND-vs-plain-CSBEND equivalent (both are
ASTRA/elegant-side-only quantities) -- csr=False (the ElementSpec
default). Enable CSR explicitly in the resulting native specifications
when the study requires it.

### `_to_element_spec`

A real zero-length "RFCavity" marker exists in FERMI's lattice (HERFX, a phase reference,
not an accelerating structure) -- dropped like any other zero-length no-op, since an
unconditional kind="rf" here would divide by zero in LinacTracker's RF-kick formula.
