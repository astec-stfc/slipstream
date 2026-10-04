# `slipstream.wakefields.shortrange`

[Source](../../../src/slipstream/wakefields/shortrange.py) · [Documentation index](../../README.md)

## Physics

Constructs a Gaussian-probe wake from impedance via inverse Fourier transformation, enforcing `Z(-omega)=conj(Z(omega))`. Intrabunch voltage comes from charge-histogram convolution; wake units are V/C and time is seconds.

### Limits and conventions

- Finite Gaussian probes can produce a small negative-time precursor; it is not automatically an invalid wake.
- FFT resolution/window and tracking bins are independent numerical choices.
- Out-of-window particles require explicit masking rather than edge-bin contamination.

### `Model and numerical conventions`

Physics for `wake_potential_from_impedance`: for a very short Gaussian
"probe" bunch of rms length sigma_s, the wake potential is the inverse
Fourier transform of Z_parallel(omega) multiplied by the bunch's
frequency-domain line density lambda~(omega) = exp(-omega^2 sigma_t^2 / 2),
sigma_t = sigma_s / c. Z_parallel is only known on the positive-frequency
grid; the negative-frequency side is filled in by the reality condition
Z(-omega) = conj(Z(omega)) before the inverse FFT.

## Usage

Use `ShortRangeWake.from_impedance` or SDDS I/O, optionally `truncated`, and call `prepare` plus `set_fixed_bin_window` for stable cached tracking. `voltage_at` evaluates particle voltages; native tracker boundaries convert metre-valued tau.

### API reference

```python
def wake_potential_from_impedance(impedance: Impedance, sigma_s_m: float, dt: float=2e-15, n_points: int=1 << 15) -> tuple[np.ndarray, np.ndarray]: ...
def next_fast_fft_len(n: int) -> int: ...
def trim_wake(t: np.ndarray, W: np.ndarray, t_min: float, t_max: float) -> tuple[np.ndarray, np.ndarray]: ...
class ShortRangeWake:
    ...
    def from_impedance(cls, impedance: Impedance, sigma_s_m: float, dt: float=2e-15, n_points: int=1 << 15, trim: tuple[float, float] | None=None) -> 'ShortRangeWake': ...
    def from_sdds_file(cls, path: str | Path, t_column: str='t', w_column: str='W') -> 'ShortRangeWake': ...
    def to_sdds_file(self, path: str | Path) -> None: ...
    def truncated(self, t_min_s: float, t_max_s: float) -> 'ShortRangeWake': ...
    def set_fixed_bin_window(self, t_min_s: float, t_max_s: float) -> None: ...
    def prepare(self, dtype: torch.dtype, device: str | torch.device) -> None: ...
    def voltage_at(self, tau: torch.Tensor, charge_per_particle_C: torch.Tensor | float) -> torch.Tensor: ...
```

### `wake_potential_from_impedance`

Return (t [s], W [V/C]), t centered on 0, for a Gaussian probe bunch.

dt/n_points set the FFT's time resolution and window
(T = n_points*dt, Nyquist = 1/(2*dt)) -- independent of the bin size a
tracker later uses (resample/trim separately, see `trim_wake` and
`ShortRangeWake.set_fixed_bin_window`).

### `next_fast_fft_len`

Next power of 2 >= n. cuFFT (like every major FFT backend) only has
a fast mixed-radix decomposition for small-prime lengths; an unlucky
length with a large prime factor falls back to the much slower,
multi-kernel Bluestein algorithm. A wake's fixed convolution length
(n_bins + wake_table_length - 1) has no reason to land on a good
length by construction -- measured 122499 = 3^3*13*349 for one wake
table vs 122500 = 2^2*5^4*7^2 for another, one element longer, a ~4x
slowdown from that single-sample difference (profiled on the
transverse HALF dipole wake). Padding up is always safe for an
FFT-based linear convolution (only adds trailing zeros the convolution
never reads back, see set_fixed_bin_window's callers), so there's no
reason not to round up unconditionally.

### `trim_wake`

Keep only [t_min, t_max] -- a tracker (or elegant's WAKE element)
wants a compact table, not the full FFT window. A small negative
t_min is often wanted deliberately: a Gaussian-probe wake potential can
have a small acausal precursor before t=0 (a real feature of a
finite-length probe bunch, not a bug) -- elegant's WAKE element needs
ACAUSAL_ALLOWED=1 to accept it, `ShortRangeWake.voltage_at` handles it
natively.

### `ShortRangeWake`

A short-range wake potential table W(t) [V/C] on a uniform time grid.

`t_s` may include a small negative "acausal precursor" -- expected for
a pseudo-Green-function wake from a finite-length probe bunch, not a bug.

### `ShortRangeWake.from_sdds_file`

Load a wake table written by `to_sdds_file` (default column
names) or a real elegant wake file with **free-form column
names** -- elegant's own `WAKE`/`RFCW` elements take the
column names as separate string parameters (`TCOLUMN=`,
`WCOLUMN=`/`WZCOLUMN=`, confirmed from `track_data.c`'s own
`wake_param`/`rfcw_param` tables -- neither has a built-in
default of `"t"`/`"W"`, that's only this module's own
`to_sdds_file` convention), so a real file (e.g. an `RFCW`'s
combined wake file with `t`/`Wx`/`Wy`/`Wz` columns) needs
the actual column name passed explicitly.

### `ShortRangeWake.set_fixed_bin_window`

Fix the per-turn charge-binning window instead of resizing it to
the bunch's instantaneous tau range every turn.

Tracking calls the wake convolution once per turn; with a dynamic
window the FFT size changes every turn, so the wake table's own FFT
(which never changes) can't be cached and gets recomputed every
single turn -- the dominant cost of a multi-thousand-turn run. A
fixed window makes the convolution length constant, so the wake's
FFT is computed once here and reused for every subsequent
`voltage_at` call. Particles landing outside the window are
dropped from the shared charge histogram (so a rare transient
outlier can't pile artificial charge onto the edge bin and kick off
a numerical runaway -- clamping them into the histogram instead of
dropping them did exactly that in an earlier version); they still
get a voltage back, read from the nearest edge bin. Choose the
window generously (a handful of expected rms bunch lengths) so
drops stay rare.

### `ShortRangeWake.prepare`

Precompute the fixed window's wake FFT for a given dtype/device
ahead of time (no-op if no fixed window is set). Call this once
before tracking, especially before wrapping the per-turn step in
`torch.compile` -- `voltage_at`'s fixed-window path assumes
this has already been done and never computes the FFT lazily
itself, since a first-call-computes-and-caches pattern is exactly
the kind of thing that graph-breaks under dynamo.

### `ShortRangeWake.voltage_at`

Voltage [V] felt by each particle from the present bunch's own
short-range wake.

Bins `tau` onto the same absolute dt-grid as the wake table (bin k
and wake sample k both sit at time k*dt from a common t=0),
convolves, and reads the result back at each particle's own bin.
