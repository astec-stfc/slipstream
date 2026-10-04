# `slipstream.tracking.native.fields`

[Source](../../../../src/slipstream/tracking/native/fields.py) · [Documentation index](../../../README.md)

## Physics

Reads on-axis profiles and expands RF/solenoid fields near the axis. Solenoid radial field is `-(r/2)*dBz/dz`; longitudinal correction is `-(r^2/4)*d2Bz/dz2`. RF uses consistent electric/magnetic transverse terms from Maxwell's equations.

### Limits and conventions

- These are paraxial expansions, not arbitrary off-axis 2D/3D field interpolators.
- ASTRA field profile tables are normalized z/value arrays; field amplitude must be supplied with the appropriate units.
- Keep the RF electric/magnetic phase sign convention consistent.

### `_scale_to_peak`

Scale factor so `profile * scale` has value exactly `target` at
the profile's own largest-magnitude sample -- **sign preserved**, not
`target / profile.abs().max()` (which always yields a positive scale
regardless of whether that extremum is the profile's positive or
negative lobe).

### `TravelingWaveField._enforce_periodicity`

Patch `self.map`/`self.map_imag` inside `[z1, z2]` so a tile
join is exactly continuous in both value and derivative -- see
class docstring. Refits `Er` with a periodic-boundary-condition
cubic spline (`bc_type="periodic"`, matching value + first/second
derivative at both ends), first snapping the raw endpoint values
together (`scipy` requires them to agree exactly). `Ei` is
re-derived from this periodic `Er` via the same Floquet formula
`_imaginary_part` uses, inheriting the same periodicity. Coupler
cells (`Ei=0`, raw `Er`) are untouched -- not periodic, don't
need to be.

### `TravelingWaveField._imaginary_part`

`Ei(z) = [Er(z+L) - Er(z) cos(theta)] / sin(theta)` inside the
periodic section, zero in the standing-wave coupler cells (see the
class docstring for the Floquet derivation). `Er(z+L)` is taken
with the periodic continuation, since `z+L` runs past `z2` for
the last cell of the stored period.

### `TravelingWaveField._envelope_z`

Map a structure-local `z` onto the field table's own
coordinate, by folding the periodic section back onto its single
stored period. The tile index is clamped to `[0, n_tiles-1]` so
the input coupler (`z_local < z1`) maps to itself and the output
coupler lands past `z2` in the table -- both continuous with the
periodic section they adjoin.

## Usage

Construct `OnAxisFieldMap`, `SolenoidField`, `RFCavityField` or `TravelingWaveField` from arrays or ASTRA field files. Supply physical amplitude/frequency and phase in the class's SI convention.

### API reference

```python
def read_astra_field_file(path: str | Path) -> tuple[np.ndarray, np.ndarray]: ...
def read_astra_tws_file(path: str | Path) -> tuple[float, float, int, int, np.ndarray, np.ndarray]: ...
class OnAxisFieldMap:
    ...
    def __init__(self, z: np.ndarray, f: np.ndarray, n_samples: int=4001, device: str | torch.device | None=None, dtype: torch.dtype | None=None): ...
    def from_astra_file(cls, path: str | Path, **kwargs) -> 'OnAxisFieldMap': ...
class SolenoidField:
    ...
    def __init__(self, z: np.ndarray, bz_profile: np.ndarray, b_max_T: float, z_offset: float=0.0, **kwargs): ...
    def from_astra_file(cls, path: str | Path, b_max_T: float, **kwargs) -> 'SolenoidField': ...
class RFCavityField:
    ...
    def __init__(self, z: np.ndarray, ez_profile: np.ndarray, e_max_V_per_m: float, frequency_hz: float, phase_rad: float=0.0, z_offset: float=0.0, **kwargs): ...
    def from_astra_file(cls, path: str | Path, e_max_V_per_m: float, frequency_hz: float, **kwargs) -> 'RFCavityField': ...
class TravelingWaveField:
    ...
    def __init__(self, z: np.ndarray, ez_profile: np.ndarray, z1: float, z2: float, n: int, m: int, e_max_V_per_m: float, frequency_hz: float, n_cells: int, phase_rad: float=0.0, z_offset: float=0.0, **kwargs): ...
    def from_astra_file(cls, path: str | Path, e_max_V_per_m: float, frequency_hz: float, n_cells: int, **kwargs) -> 'TravelingWaveField': ...
    def on_axis(self, z: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
```

### `read_astra_field_file`

Read a plain 2-column ASTRA field file (`z[m] value[normalized]`,
e.g. `HRRG_1D_RF.astra`, `HRRG_combined_sols_100mm_onaxis.astra`).

Multi-cell travelling-wave-structure files (e.g. `TWS_S-DL.astra`)
have a different, header-prefixed format -- use `read_astra_tws_file`
for those instead; this raises a clear error rather than silently
misparsing the header line as a data row.

### `read_astra_tws_file`

Read an ASTRA traveling-wave-structure field file (name starts with
`TWS`, e.g. `TWS_S-DL.astra`) -- manual (v3.2, ch. 4 "Traveling
wave structures") sec.: a free-format header line `z1 z2 n m`
precedes the usual 2-column `z, on-axis field` table. `z1`/`z2`
are the start/end of the periodic traveling-wave cells (excluding the
input/output coupler cells, which extend past them and are "treated
as standing wave cells"); `n`/`m` are integers giving the phase
advance per cell `theta = 2*pi*n/m` (`n<0` for a backward wave).
This function only decodes the file -- see `TravelingWaveField` for
the field construction it feeds.

### `OnAxisFieldMap`

A 1D on-axis field profile `f(z)`, resampled onto a uniform grid
together with its first and second derivatives (via a cubic spline
fit, evaluated once at construction) for fast GPU linear-interpolation
lookup at runtime -- avoids a host<->device round trip every tracking
step, which repeatedly evaluating a CPU spline per step would cost.

### `SolenoidField`

Static axisymmetric solenoid field from an on-axis `Bz(z)`
profile, scaled so its peak matches `b_max_T` (ASTRA's own `MaxB`
convention -- the loaded profile is normalized, this rescales it to a
physical peak field in tesla). `z_offset` is the solenoid's position
along the beamline (ASTRA's `S_pos`).

### `RFCavityField`

Axisymmetric RF cavity field from an on-axis `Ez(z)` profile,
scaled so its peak matches `e_max_V_per_m` (ASTRA's own `MaxE`,
already converted to SI V/m by the caller -- this class takes no
stance on ASTRA's MV/m file convention). `frequency_hz`/`phase_rad`
give the time dependence `cos(omega t + phase_rad)`; `z_offset` is
the cavity's position along the beamline (ASTRA's `C_pos`).

ASTRA's manual (Appendix I) gives its own on-axis field as
`Ez0(z)*sin(omega*t + Phi_ASTRA)` -- the same physical convention as
this class's `cos(omega*t + phase_rad)`, offset by which instant is
called `t=0`: pass `phase_rad = Phi_ASTRA - pi/2` for a direct
match. This class does not replicate ASTRA's `Auto_phase` layer
(relative to the cavity's own scanned crest, not the absolute wave
phase) -- that's `Injector.auto_phase`'s job, not a field-formula
concern.

### `TravelingWaveField`

ASTRA-style multi-cell **traveling-wave** structure. 

**The file stores a signed field snapshot, not an amplitude
envelope** -- ASTRA's manual, Appendix II ("Representation of a
travelling wave by two standing waves"): the field is carried by a
real and imaginary part, `Ez(z,t) = Er(z) cos(wt) - Ei(z) sin(wt)`,
which is exactly what this class evaluates.

**Getting `Ei` from the one stored table**: Floquet's theorem for a
periodic structure with phase advance `theta = 2*pi*n/m` per cell of
length `L` gives `E(z+L) = E(z) * exp(-i*theta)`. Taking real
parts, `Er(z+L) = Er(z) cos(theta) + Ei(z) sin(theta)`, i.e.

    `Ei(z) = [Er(z+L) - Er(z)*cos(theta)] / sin(theta)`

-- an exact, closed-form reconstruction from the table evaluated one
cell downstream, no second data column, no fitting. Built once at
construction.

**Geometry** (from the file's own `z1 z2 n m` header plus
`n_cells` = ASTRA's `C_numb`): the table holds an input coupler
cell over `[0, z1)`, one RF period of periodic traveling-wave cells
over `[z1, z2]` (`m/|n|` cells of length `L=(z2-z1)*|n|/m`), and
an output coupler cell over `[z2, z_table_end]`. The real structure
repeats the periodic section to `n_cells` cells, so this class
**tiles** `[z1, z2]` and brackets it with the two coupler cells,
total length `z1 + n_cells*L + (z_table_end - z2)`. One stored
period's own Floquet factor is `exp(-i*2*pi*n)=1`, so both `Er`
and `Ei` tile with no phase bookkeeping.

The coupler cells are "treated as standing wave cells" (manual):
`Ei=0` there, continuous with the periodic section since `Ei` is
~0 at both its ends by design.

**Tile-join continuity is enforced exactly, in both value and
derivative — see `_enforce_periodicity`.** The raw table's two ends
only approximately agree; left uncorrected, `dEz/dz` (which sets the
paraxial transverse focusing) jumps at every tile join. Floquet's
theorem assumes exact periodicity to begin with, so
`_enforce_periodicity` refits `Er` with a periodic-boundary-
condition spline and re-derives `Ei` from that, patching both tables
in the periodic region after construction.

**No free wave number.** The phase velocity is fixed by the file's own
geometry (`omega*L/theta`); `self.k` is exposed for diagnostics
only, not used in the field evaluation.

**Off-axis expansion**: the same leading-order paraxial result
`Er_transverse = -(r/2) dEz/dz`, `Bphi = (r/2c^2) dEz/dt` this
module derives for `RFCavityField` -- depends only on the on-axis
`Ez(z,t)`, whatever waveform produced it. `tests/test_native_fields.py`
checks a structure with `Ei=0` everywhere reproduces `RFCavityField`
exactly.

### `TravelingWaveField.on_axis`

`(Ez, dEz/dz, dEz/dt)` on the axis -- the three quantities the
paraxial expansion in `__call__` is built from, exposed
separately so an on-axis-only caller (a phase scan, a fused
tracking kernel) can skip the transverse algebra. The real and
imaginary tables share one uniform grid, so index/interpolation
weights are computed once here rather than per
`OnAxisFieldMap.__call__`.

## Numerical and implementation notes

### `OnAxisFieldMap`

Some real ASTRA field files are assembled from several independently-computed
sub-segments and carry an exact-duplicate z at each seam.
CubicSpline requires strictly increasing x; keep the
first sample at each repeated z rather than erroring.

### `TravelingWaveField._enforce_periodicity`

Widen by one resampled cell on each side: the runtime fold always
lands strictly inside [z1, z2] (_envelope_z clamps it there), but a
lerp query near z1 or z2 can still pair a patched grid point with
its immediate (otherwise unpatched) neighbour one cell outside --
patch that neighbour too (a well-defined, continuous cubic
extrapolation one cell past the fit interval) so every lerp bin
the fold can actually select is fully periodic-consistent.

### `TravelingWaveField._envelope_z`

This subtracts two quantities of order the structure length to
leave one of order the period, so in float32 the result inherits
~1% of a field-table cell of rounding -- measured not to matter
(the compensated Boris push, not this, was the real float32
accuracy driver). Reinstate an exact fmod form if a longer
structure or coarser table ever makes that 1% matter.

### `TravelingWaveField.on_axis`

`_envelope_z`'s clamped fold maps [0, length] onto the table's
own z-range by construction, so this physical mask subsumes the
per-map range mask it replaces.
