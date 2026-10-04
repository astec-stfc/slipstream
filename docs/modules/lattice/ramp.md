# `slipstream.lattice.ramp`

[Source](../../../src/slipstream/lattice/ramp.py) · [Documentation index](../../README.md)

## Physics

Interpolates supplied RF/energy programs and scales fixed-geometry radiation parameters: energy loss proportional to gamma^4, damping time to gamma^-3, natural spread to gamma and equilibrium emittance to gamma^2.

### Limits and conventions

- The supplied phase/voltage program already embodies energy-balance choices.
- Scaling keeps geometric optics and momentum compaction fixed; it does not rematch the lattice.

### `Model and numerical conventions`

`RampProfile` supplies an RF and reference-energy program to
`NativeTracker(ramp=...)`, alongside the base `Lattice` and optional
`TransverseOptics`. It does not solve for an energy-balanced RF phase.

## Usage

Build `RampProfile` from arrays or `from_sdds_file`, query `at`/`at_array`, and use `scaled_lattice`/`scaled_optics` for instantaneous parameters.

### API reference

```python
class RampState:
    ...
class RampArrayState:
    ...
class RampProfile:
    ...
    def at(self, t_s: float) -> RampState: ...
    def at_array(self, t_s: np.ndarray) -> RampArrayState: ...
    def from_sdds_file(cls, path: str | Path, rest_mass_eV: float=ELECTRON_MASS_EV) -> 'RampProfile': ...
def scaled_lattice(base: Lattice, gamma_now: float) -> Lattice: ...
def scaled_optics(base: TransverseOptics, ratio: float) -> TransverseOptics: ...
```

### `RampProfile`

A time-sampled RF/momentum program. `time_s` must be sorted
ascending; `at()` linearly interpolates each column (and clamps to
the boundary values outside `[time_s[0], time_s[-1]]`, via
`np.interp`'s own default behavior).

### `RampProfile.at_array`

Interpolate all four program columns over an array of times. Each column
uses one `np.interp` call; clamping and interpolation semantics match scalar
`at()`. Batching avoids repeated host interpolation calls.

### `RampProfile.from_sdds_file`

Load an elegant RAMPP+RAMPRF-style ramp table: columns `time`
[s], `Voltage` [V], `Frequency` [Hz], `Phase` [deg],
`momentum` [rest-mass-c units].
`rest_mass_eV` converts the momentum column to `E0_eV` via the
exact relativistic relation (not the ultra-relativistic `E~pc`
approximation) -- defaults to the electron mass, override for a
non-electron ring the same way `Lattice.rest_mass_eV` does.

### `scaled_lattice`

A copy of `base` with `E0_eV`/`U0_eV`/`tau_z_s`/
`sigma_delta0` rescaled from `base`'s own (injection-energy)
values to `gamma_now`, via elegant `SREFFECT`'s verified power
laws (model description). `Vrf_V`/`f_rf_Hz` on the returned object
are *not* meaningful -- the ramped RF program comes from
`RampProfile.at()` directly, not from scaling a reference value.

### `scaled_optics`

Return a copy with damping times divided by ratio**3 and equilibrium
emittances multiplied by ratio**2, where ratio=gamma_now/gamma_ref.
Beta, alpha, dispersion, tunes and chromaticities remain fixed; this helper
does not rematch the optics.

