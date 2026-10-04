# `slipstream.tracking.native.laura_injector`

[Source](../../../../src/slipstream/tracking/native/laura_injector.py) · [Documentation index](../../../README.md)

## Physics

Constructs field-map injector elements from chained LAURA sections. Cavities, solenoids, apertures and profile gradients use their physical values.

### Limits and conventions

- Dipoles or normalized quadrupole strengths needing an unavailable reference-momentum conversion are skipped with a warning; inspect that warning before trusting transport.

### `Model and numerical conventions`

**Validated against a real machine**: CLARA's real ASTRA injector deck
(`examples/clara_linac/ASTRA/5pC/injector400.in`) covers exactly the
LAURA sections `HRG1`, `S01`, `L01` (confirmed directly -- every real
`C_pos`/`S_pos`/`Ap_Z1`/`Ap_R` value in that deck matches this
module's own output for the same elements to the deck's own rounding, and
its `zstop=3.29693` is exactly `CLA-S02-SIM-APER-01`'s own start
z -- S02's own first element, correctly *not* included here, since tracking
already stops at that same z regardless).

### `_to_apertures`

One aperture `SlipstreamElement` -> 0/1/2 `Injector.Aperture`
objects, mirroring `ApertureTranslator.to_astra()`'s own shape
dispatch exactly (`aperture.py`): circular/elliptical -> one `RAD`;
planar/rectangular/scraper -> up to two (X and Y), whichever sizes are
actually positive.

## Usage

Call `from_laura` with the machine/section chain and the supported source/settings arguments; this requires optional LAURA packages.

### API reference

```python
def from_laura(machine_module, section_names: tuple[str, ...], z_stop_m: float, distribution: str='', **injector_kwargs) -> Injector: ...
```

### `from_laura`

Build an `Injector` from `section_names` (in beamline order,
e.g. `("HRG1", "S01", "L01")` for CLARA), stopping tracking at
`z_stop_m`.

`distribution`/`generator` (the latter via `injector_kwargs`):
LAURA has no single-file equivalent of an ASTRA deck's own
`Distribution=`/a paired `generator.in` -- pass an existing ASTRA-
format distribution file or `generator.in` explicitly, exactly as
`Injector.from_astra` itself only ever reads a path, never generates
one from the deck alone. Any other `Injector` field (`dt_s`,
`space_charge=`,...) can be overridden via `injector_kwargs`.
