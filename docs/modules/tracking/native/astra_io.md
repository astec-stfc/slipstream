# `slipstream.tracking.native.astra_io`

[Source](../../../../src/slipstream/tracking/native/astra_io.py) · [Documentation index](../../../README.md)

## Physics

Converts ASTRA distribution rows from their reference-relative z/pz/time convention to absolute particle values. ASTRA positions are metres, momenta eV/c, clocks ns and macrocharges nC; electron charge sign follows species.

### Limits and conventions

- The first row is an absolute reference; other rows' z, pz and clock are relative to it.
- Cathode probes are diagnostic rows, not real bunch charge.
- Namelist parsing does not execute ASTRA or construct all injector physics by itself.

### `Model and numerical conventions`

- Columns: ``x, y, z, px, py, pz, clock, macro_charge, particle_index,
  status_flag`. Units: `x, y, z` in m; `px, py, pz`` in eV/c;
  `clock` in ns; `macro_charge` in nC.
- **The first row is the reference particle, in absolute coordinates.
  Every other row's `z`, `pz`, `clock` are *relative* to that first
  row** ("Longitudinal particle coordinates, i.e. z, pz and t are given
  relative to the reference particle" -- manual sec. 2); `x, y, px, py`
  are already absolute for every row.
- Status flag `-1` = a standard particle at the cathode, not yet
  started -- the real macroparticles a cathode-emission distribution
  like `laser.astra` consists of. `-3` = a trajectory-probe particle
  at the cathode -- the 6 diagnostic-only rows `generator`'s default
  `Probe=True` adds (positions at 0.5/1.0/1.5 sigma_x/sigma_z and
  sigma_y/sigma_z) -- excluded here, they carry no real bunch charge.
- The macro-charge column's *sign* is not authoritative (manual: "The
  sign of the charge specified in column 8 is not relevant" -- ASTRA
  infers sign from the particle-index species column instead); this
  reader applies the correct sign for electrons (index 1) explicitly
  rather than trusting the file's own sign bit.

## Usage

Use `read_astra_distribution` or `distribution_from_rows`, `write_astra_distribution`, `read_astra_emit`, `astra_reference_at_z` and namelist readers. `momentum_eV_to_SI` converts eV/c to kg*m/s.

### API reference

```python
def read_astra_distribution(path: str | Path, device: str | torch.device | None=None, dtype: torch.dtype | None=None) -> dict[str, torch.Tensor]: ...
def distribution_from_rows(raw: np.ndarray, device: str | torch.device | None=None, dtype: torch.dtype | None=None, status: tuple[int, ...]=(-1,)) -> dict[str, torch.Tensor]: ...
def cathode_emission_status(raw: np.ndarray) -> tuple[int, ...]: ...
def momentum_eV_to_SI(p_eV_per_c: torch.Tensor) -> torch.Tensor: ...
def write_astra_distribution(path: str | Path, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, px: torch.Tensor, py: torch.Tensor, pz: torch.Tensor, clock_ns: torch.Tensor, q_C: torch.Tensor, species_index: int=1, status_flag: int=5) -> None: ...
def read_astra_emit(stem: str | Path, run: int=1) -> dict[str, np.ndarray]: ...
def astra_reference_at_z(astra: dict[str, np.ndarray], z_m: float) -> dict[str, float]: ...
def read_astra_namelist_groups(path: str | Path) -> list[tuple[str, dict]]: ...
def read_astra_namelist(path: str | Path) -> dict[str, dict]: ...
```

### `read_astra_distribution`

Read an ASTRA particle-distribution file, returning the *real*
(status == -1) electron macroparticles only, with absolute
`z`/`pz`/`clock` already added back in. Returns a dict of
`(N,)` tensors: `x, y, z` (m), `px, py, pz` (eV/c, absolute),
`clock` (ns, absolute -- each particle's emission time), `q`
(C, signed, one per macroparticle).

### `distribution_from_rows`

`read_astra_distribution`'s row -> particle conversion, on an already-loaded `(N, 10)`
ASTRA-layout array (also what `generator.Beam.particles` feeds it). `status` selects the
rows kept: `(-1,)` (default) is a cathode-emission file's standard particles; `generator`
passes `(-1, 5)` so a beam that starts in flight (`Cathode=F`) keeps its particles too.

Row 0 (the reference particle) is `x, y, z, px, py, pz, clock` **already absolute**, unlike
every other row (relative to it, manual sec. 2) -- both real ASTRA's own `generator.exe` and
this module's own `generator.py` clone deliberately reuse it as a genuine degenerate
macroparticle too (sitting exactly on the reference trajectory, its own status set to the real
-- not reference-only -- status code), so it must stay counted when its status matches. What
must NOT happen is adding `z0`/`pz0`/`clock0` to it a *second* time (it is not relative
to itself): confirmed directly on ISIS MEBT's real `laser.astra` (status `5`, matching
every other real particle there) -- without this, row 0 alone came out at exactly double its
real momentum, a silently negligible one-in-2.6e5 outlier on CLARA's cathode files but a
measurable one-in-2000 outlier here.

### `cathode_emission_status`

`(-1,)` if `raw` (row 0 excluded -- never itself a real particle, see
`distribution_from_rows`) has any real cathode-not-yet-emitted (status `-1`) row, else
`(5,)` -- a beam already in flight, ASTRA's own `Cathode=F` convention (real particle
status `5`, confirmed on ISIS MEBT's `laser.astra`). Only for a genuine tracking *input*:
an exit/screen-dump file also uses status `5` for its real particles (``write_astra_
distribution`'s default), so this must not become `read_astra_distribution``'s own default
-- callers that specifically mean "the beamline's own starting distribution"
(`Injector.track`) call this explicitly instead.

### `write_astra_distribution`

The exact inverse of `read_astra_distribution`: writes an ASTRA
particle-distribution file, real (electron, by default) macroparticles
only. Row 0 (the reference particle) is the mean of every column
(ASTRA tolerates a synthetic reference row -- it need not be a real
macroparticle); every other row's `z`/`pz`/`clock` are written
*relative* to it, matching the reader's own "first row absolute, rest
relative" convention exactly (manual sec. 2 -- see this module's
docstring). `status_flag` defaults to `5` (ASTRA's "standard
particle, active" code) rather than `-1` (cathode, not yet started)
-- this writer is for a *tracked* distribution (an exit/screen dump),
not a cathode-emission input; pass `-1` explicitly to write a
cathode-ready file instead.

Row 0's own status is `status_flag` unless that's `-1`, in which
case row 0 gets `-3` instead (ASTRA's own "trajectory-probe particle
at the cathode, diagnostic only" code -- see `read_astra_distribution`'s
docstring) -- row 0 is a *synthetic mean*, not a genuine (n+1)th real
particle, so it must not satisfy the reader's own `status==-1` real-
particle mask, or reading the file back would silently add one spurious
particle at the bunch centroid.

`q_C` is signed (this module's own convention); written as `-q_C`
in nC for an electron (species 1, matching the reader's `species==1`
sign-flip) or plain `|q_C|` otherwise (the file's own charge-column
sign is never authoritative per the ASTRA manual, so either round-trips
through the reader).

### `read_astra_emit`

Read ASTRA's `<stem>.Xemit/.Yemit/.Zemit.<run:03d>` output tables
(plain 7-column ASCII, manual Table 4) into the named arrays the
native trajectory also carries, so the two plot on the same keys.
`stem` may be the input deck itself (`injector400.in`) -- ASTRA
names its outputs after the deck, so the suffix is dropped.

Columns: Xemit/Yemit are ``z[m], t[ns], avr[mm], rms[mm],
slope_rms[mrad], eps_norm[pi mrad mm], corr[mrad]`; Zemit is `z[m],
t[ns], Ekin[MeV], zrms[mm], dE_rms[keV], eps_z_norm[pi keV mm],
corr[keV]``. Note the emittance is column 6 in both (column 5 is
`slope_rms`/`dE_rms`), and every ASTRA emittance is the canonical
RMS phase-space area -- see `injector._longitudinal_emittance_keV_mm`
for the longitudinal convention. Files are z-sorted.

### `read_astra_namelist`

Parse an ASTRA input deck (Fortran namelists) into
`{group: {key: value}}`, all names lower-cased (ASTRA is
case-insensitive: `FILE_EFieLD`). An indexed key `Phi(2) = -16`
becomes `{"phi": {2: -16}}`. Logicals (`T`/`True`/`.T.`) map to
bool, numbers to int/float (Fortran `D` exponents included), quoted
and bare words to str, a Fortran complex `(x, z)` to a `(float, float)`
tuple. A two-index key `D_Gap(1,3) = 0.05` becomes ``{"d_gap": {(1, 3):
0.05}}`. `!`` comments are stripped. A repeated group is merged (use
`read_astra_namelist_groups` to keep them apart).

## Numerical and implementation notes

### `Module constants`

One `name = value`, `name(i) = value` or `name(i,j) = value` assignment. Values never
contain whitespace or commas unless quoted or a Fortran complex `(x, z)` (ASTRA's
`D1(i) = (0.1, 2.5)` dipole corners), so a regex over the group body is a complete
tokenizer for ASTRA's decks (it writes no array-valued lists).
