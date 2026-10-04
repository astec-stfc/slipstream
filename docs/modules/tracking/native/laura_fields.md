# `slipstream.tracking.native.laura_fields`

[Source](../../../../src/slipstream/tracking/native/laura_fields.py) · [Documentation index](../../../README.md)

## Physics

Loads LAURA on-axis field profiles and wake tables into existing native field/wake classes. Unit and t/z handling use LAURA's reader.

### Limits and conventions

- Supported fields are 1D on-axis profiles.
- 2D/3D field maps raise `NotImplementedError`; do not treat them as paraxial 1D data.

## Usage

Call `field_from_laura_hdf5` or `wake_from_laura_hdf5(path, component)` with LAURA installed.

### API reference

```python
def field_from_laura_hdf5(path: str | Path, e_max_V_per_m: float | None=None, b_max_T: float | None=None, frequency_hz: float | None=None, n_cells: int=0, phase_rad: float=0.0, z_offset: float=0.0, **kwargs) -> SolenoidField | RFCavityField | TravelingWaveField: ...
def wake_from_laura_hdf5(path: str | Path, component: str) -> ShortRangeWake | TransverseWake: ...
```

### `field_from_laura_hdf5`

One LAURA HDF5 field-map file -> the matching slipstream field object,
dispatched on the file's own `type`/`cavity_type` attrs (no need to
tell this function which kind of element it is).

`e_max_V_per_m`/`b_max_T`/`frequency_hz` are the *element's* own
physical amplitude/frequency (`laura`'s ``SlipstreamElement.
e_max_v_per_m`/`b_max_t`/`frequency_hz``) -- the HDF5 file itself
stores a normalized profile (peak magnitude ~1), not physical units, same
convention as the existing ASTRA-file readers in `fields.py`.

### `wake_from_laura_hdf5`

One LAURA HDF5 wake file -> `ShortRangeWake` (`component="Wz"`) or
`TransverseWake` (`component="Wx"`/`"Wy"`). `t_s` comes from
LAURA's own `FieldMap.t_values` (`z / c`, confirmed exact match
against LAURA's own elegant-SDDS wake export for a real CLARA cavity,

## Numerical and implementation notes

### `Module constants`

V/pC and V/pC/m are the only non-SI unit spellings seen in real CLARA
wake files (case varies, e.g. "V/c" for "V/C") -- normalize by string,
not by blindly trusting a particular case.

### `field_from_laura_hdf5`

1DElectroDynamic / 1DElectroStatic: an RF cavity, standing- or
travelling-wave (LAURA has no separate field_type for the latter --
cavity_type on the same 1DElectroDynamic map distinguishes them).

### `wake_from_laura_hdf5`

A LongitudinalWake file has no Wx/Wy -- FieldMap always exposes
the FieldParameter itself (never a plain AttributeError), just
with.value=None, so this must be checked explicitly rather than
relying on attribute lookup to fail.
