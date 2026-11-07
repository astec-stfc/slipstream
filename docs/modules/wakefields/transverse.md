# `slipstream.wakefields.transverse`

[Source](../../../src/slipstream/wakefields/transverse.py) · [Documentation index](../../README.md)

## Physics

Dipole wake voltage is a convolution against charge-weighted transverse offset rather than charge alone. Impedance is ohms/metre; wake is V/C/metre.

### Limits and conventions

- Sign validation includes the analytic head-tail model; broader reference-code comparisons remain a separate validation boundary.
- Convert tau from metres to seconds at the native boundary.

### `Model and numerical conventions`

Transverse (dipole) short-range wake: same construction as
`ShortRangeWake` (an impedance -> Gaussian-probe wake potential via FFT,
`wake_potential_from_impedance`, reused verbatim -- it's unit-agnostic)
applied to a transverse dipole impedance (Zxdip/Zydip [Ohm/m]) instead of
the longitudinal one (Zlong [Ohm]), consumed by `kick_at` instead of
`voltage_at`.

Physics: a dipole wake kicks a trailing test particle in proportion to the
source bunch's *charge-weighted mean transverse offset* (dipole moment) at
each tau, not just its charge -- see PyHEADTAIL's `DipoleWakeKickX` for
the reference formula this mirrors. Concretely: `W_x(t)` [V/C/m]
convolved against a per-bin `charge_per_particle_C * offset_m` histogram
gives an effective transverse kick voltage [V], read back per particle
exactly like `ShortRangeWake.voltage_at` -- same fixed-bin-window
FFT-caching trick (`set_fixed_bin_window`/`prepare`), same "trash bin"
handling for particles outside the window, just with a dipole-moment
numerator instead of a charge one.

## Usage

Construct `TransverseWake` from impedance, CSV or SDDS, prepare/cache it and evaluate `kick_at`. Plane selection and transverse offsets must match the intended dipole component.

### API reference

```python
class TransverseWake:
    ...
    def from_impedance(cls, impedance: Impedance, sigma_s_m: float, dt: float=2e-15, n_points: int=1 << 15, trim: tuple[float, float] | None=None) -> 'TransverseWake': ...
    def to_csv(self, path: str | Path) -> None: ...
    def from_csv(cls, path: str | Path) -> 'TransverseWake': ...
    def from_sdds_file(cls, path: str | Path, t_column: str='t', w_column: str='W') -> 'TransverseWake': ...
    def truncated(self, t_min_s: float, t_max_s: float) -> 'TransverseWake': ...
    def set_fixed_bin_window(self, t_min_s: float, t_max_s: float) -> None: ...
    def prepare(self, dtype: torch.dtype, device: str | torch.device) -> None: ...
    def kick_at(self, tau: torch.Tensor, offset: torch.Tensor, charge_per_particle_C: torch.Tensor | float) -> torch.Tensor: ...
    overflow: torch.Tensor | None  # property
    def reset_overflow(self) -> None: ...
```

### `TransverseWake`

A transverse dipole wake potential table Wx(t) [V/C/m] (or Wy) on a
uniform time grid -- same shape/conventions as `ShortRangeWake`
(`t_s` may include a small negative acausal precursor, t>=0 is the
causal part). Plane-agnostic: the same class instance is used for
either x or y, the tracker decides which offset to pass into
`kick_at`.

### `TransverseWake.from_sdds_file`

Load a wake table from an SDDS file by column name -- mirrors
`ShortRangeWake.from_sdds_file` exactly (see its own docstring
for why the column names must be given explicitly, not assumed --
a real elegant `RFCW` wake file has one combined table with
separate `WXCOLUMN=`/`WYCOLUMN=`/`WZCOLUMN=` names, so
loading the transverse planes from that same file needs this.

### `TransverseWake.set_fixed_bin_window`

Fix the per-turn dipole-moment-binning window -- see
`ShortRangeWake.set_fixed_bin_window` for the full rationale
(constant convolution length so the wake's own FFT is cached once,
not recomputed every turn; particles outside the window are
dropped from the histogram rather than clamped in, so a rare
transient outlier can't pile artificial dipole moment onto an edge
bin).

### `TransverseWake.kick_at`

Transverse kick voltage [V] felt by each particle from the
present bunch's own dipole wake, given its own transverse offset
(x or y) [m]. Same fixed-bin-window convolution as
`ShortRangeWake.voltage_at`, but the per-bin histogram is
charge-weighted *offset* (the source bunch's dipole moment) rather
than charge alone -- `PyHEADTAIL`'s `DipoleWakeKickX` is the
reference for this weighting.

`deposit="cic"` and `monitor_overflow` behave as for `ShortRangeWake` (see there): the charge-weighted
offset is spread over two bins and the kick is read back by interpolation, so `kick_at` has a gradient
to `tau` as well as to `offset`; the default `"ngp"` has the gradient to `offset` only. The window
is always fixed (`kick_at` raises without one).

## Numerical and implementation notes

### `TransverseWake`

np.array (not asarray): from_csv's pandas-CoW arrays come back
read-only, and asarray is a no-op on an already-float64 array --
torch.as_tensor on that in prepare() would then wrap a read-only
buffer, undefined behavior if ever written to.
