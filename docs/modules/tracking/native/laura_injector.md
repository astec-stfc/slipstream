# `slipstream.tracking.native.laura_injector`

[Source](../../../../src/slipstream/tracking/native/laura_injector.py) · [Documentation index](../../../README.md)

## Physics

Constructs field-map injector elements from chained LAURA sections. Cavities, solenoids, apertures and profile gradients use their physical values.

### Limits and conventions

- Dipoles or normalized quadrupole strengths needing an unavailable reference-momentum conversion are skipped with a warning; inspect that warning before trusting transport.

### `Model and numerical conventions`

**Compared with a machine-specific ASTRA deck**: CLARA's real ASTRA injector deck
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

Call `from_laura` with a machine module or a prebuilt `LAURA` instance and the section chain and the supported source/settings arguments; this requires optional LAURA packages.

### API reference

```python
def from_laura(machine_module, section_names: tuple[str, ...], z_stop_m: float, distribution: str='', **injector_kwargs) -> Injector: ...
```

### `from_laura`

Build an `Injector` from `section_names` (in beamline order,
e.g. `("HRG1", "S01", "L01")` for CLARA), stopping tracking at
`z_stop_m`. A module is opened with a scoped layout containing only this section
chain, so unrelated full-machine layout validation cannot block an injector import.
A prebuilt `LAURA` instance retains caller-applied control/readback updates.

The importer uses LAURA's supported typed `translate_elements` API. It does not require
`SectionLatticeTranslator.to_slipstream` (absent in some installed LAURA versions), parse
raw machine YAML again, or write temporary ASTRA field maps. Original resolved HDF5
map filenames remain attached to the injector. Importing does not mutate the caller's
LAURA elements. Physical positions use LAURA's field reference plus `dz`; solenoid
x/y offsets and two tilts use the same values/signs as LAURA's ASTRA translator, in
metres/radians. Aperture x placement and quad x tilt are retained. LAURA off-crest RF
phase maps to negative ASTRA `Phi`; with `absolute_phases=True`, `phase_deg` is the
absolute ASTRA phase (stored LAURA crest minus off-crest phase). Native `crest_deg=-90`
converts ASTRA sine to the native cosine convention without recalculating the crest.

Unsupported placed RF cavities, out-of-plane aperture/quad placement, quadrupole
field profiles and relative aperture extents raise rather than being approximated
silently. Powered correctors warn and are omitted; off-mode studies must use the
appropriate readbacks. Screen planes are retained as `Injector.screens`; screen names,
camera calibration and element identities remain available in the caller's LAURA model.
Use that model for diagnostic provenance, rather than inventing names from the numeric list.

`distribution`/`generator` (the latter via `injector_kwargs`):
LAURA has no single-file equivalent of an ASTRA deck's own
`Distribution=`/a paired `generator.in` -- pass an existing ASTRA-
format distribution file or `generator.in` explicitly, exactly as
`Injector.from_astra` itself only ever reads a path, never generates
one from the deck alone. Any other `Injector` field (`dt_s`,
`space_charge=`,...) can be overridden via `injector_kwargs`.
