# Diamond-II booster ramp against elegant

Checks `NativeTracker(ramp=...)` against elegant on the Diamond-II booster, which accelerates from 100 or 150 MeV/c to
3.5 GeV/c in 100 ms while the RF voltage follows E^4 from 0.2 to 2 MV. The magnet ramp is read from
[MagRamp100.sdds](MagRamp100.sdds) (default) or [MagRamp150.sdds](MagRamp150.sdds): time in s and momentum in m_e c.

elegant tracks the real lattice ([lattice/booster.lte](lattice/booster.lte): 78 bends, 20 quadrupoles, 44 sextupoles) with
`RAMPP` (reference momentum), `RAMPRF` (voltage and phase) and `SREFFECT` (radiation damping, quantum excitation and the
energy loss per turn). slipstream tracks the same lattice from the same injected bunch, with the optics, damping times,
energy spread and `U0` taken from elegant's own Twiss and radiation integrals.

```bash
python examples/diamond2booster/validate_ramp_vs_elegant.py                    # 50 to 60 ms, the machine's phase table
python examples/diamond2booster/validate_ramp_vs_elegant.py --balanced-phase   # RF phase with V sin(phase) = U0 + dE/turn
python examples/diamond2booster/validate_ramp_vs_elegant.py --ramp examples/diamond2booster/MagRamp150.sdds
python examples/diamond2booster/validate_ramp_vs_elegant.py --integration-order 6   # sixth-order bends in both codes
python examples/diamond2booster/validate_ramp_vs_elegant.py --help
```

It needs elegant (set `ELEGANT_HOME`, default `C:/Program Files/APS/Elegant x64`) and writes everything to `output/`
(not tracked). elegant runs once and is cached; `--tag` names a slipstream run and `--overlay` shows earlier runs beside
it, so two versions of the code can be compared with one elegant result. The script prints, every `--interval` turns, the
centroid and rms of energy and time and the horizontal size, then the rms difference to elegant over the window, and saves
a plot.

## The window

The default window is 50 to 60 ms of the 100 MeV/c ramp (1.8 to 2.3 GeV, 18 300 turns). There the radiation loss grows as
E^4 from 66 to 185 keV per turn, the RF voltage from 0.2 to 0.39 MV, and the ramp takes about 28 keV per turn. The first
tens of ms are nearly adiabatic. The whole ramp (`--start-ms 0 --duration-ms 100`) is 183 000 turns, about 7 hours in
slipstream on one core with fourth-order bends (0.14 s per turn).

The machine's table keeps the RF phase at 180 degrees, so `V sin(phase)` is zero and the injected bunch starts 0.5 rad from
the synchronous phase and oscillates through the whole window. `--balanced-phase` sets the phase at which the RF supplies
exactly the loss and the gain, `V sin(phase) = U0 + dE/turn`, which leaves the bunch quiet and shows the energy balance
directly. Both are worth running.

## Three things in elegant that the comparison handles

* **RF phase reference.** elegant measures the RF phase from a nominal arrival time (the sum of the revolution times
  `C / (beta c)`), not from the tracked particle, and ignores the frequency column of the ramp file. slipstream measures the
  phase from the reference particle, an RF locked to the revolution. The two agree when the particle takes exactly
  `C / (beta c)` per turn.
* **Integration error in BD.** elegant's default 4-kick CSBEND integration of BD (K1 and K2 non-zero) takes 2.4 ps per turn
  longer than `C / (beta c)`. That is a numerical error, not path length: it falls to 0.14 ps at 8 kicks, 0.006 ps at 16, and
  0.007 ps with `INTEGRATION_ORDER=6` and `N_KICKS=4`. Left in, it is a frequency error of 4.4e-6 that moves elegant's
  synchronous particle to an energy offset of -8e-4. The script measures it on the bare ring (`BARE` line, no RF,
  radiation or ramp) and takes it off elegant's phase column.
* **SREFFECT during a ramp.** An element that configures itself from the radiation integrals scales its numbers with one
  power of `p / p0` too few. Measured with elegant, momentum raised to `1.3 p0`:

  | per turn | physics | self-configured | explicit, with `PREF` |
  |---|---|---|---|
  | energy loss, in delta | p^3 | p^2.00 | p^2.98 |
  | damping rate | p^3 | p^2.00 | p^3.00 |
  | quantum excitation | p^5 | p^3.99 | p^4.98 |

  The script therefore passes `PREF`, `JX`, `JY`, `JDELTA`, `EXREF`, `SDELTAREF` and `DDELTAREF` from the Twiss run
  explicitly. Left self-configured, the loss in the default window ends 22% low and elegant's bunch arrives 38 ps late;
  over the whole ramp from 100 MeV/c the factor reaches `p / p0` = 35.

## Results

The default window (50 to 60 ms of `MagRamp100`, 18 300 turns), 500 particles, fourth-order bends with 4 kicks in both
codes. rms difference to elegant over the window, before and after the correction of the ramped RF kick:

| phase table | slipstream | centroid delta | centroid t | rms delta | rms t | rms x |
|---|---|---|---|---|---|---|
| machine (180 degrees) | before | 5.0e-4 | 173 ps | 39% | 40% | 4.6% |
| machine (180 degrees) | after | 4.6e-5 | 4.7 ps | 4.1% | 4.3% | 4.6% |
| balanced | before | bunch lost | | | | |
| balanced | after | 3.1e-5 | 2.7 ps | 3.3% | 3.6% | 5.7% |

* After the correction the energy spread and bunch length agree at the level of the sampling noise: with 500 particles,
  two independent estimates of an rms value differ by about 4.5%.
* Before it, with the balanced table, the bunch blows up near turn 6000 and its statistics are not finite after turn 6500.

## Files

| File | Purpose |
|---|---|
| `validate_ramp_vs_elegant.py` | builds the ramp, runs elegant, tracks with slipstream, compares |
| `MagRamp100.sdds`, `MagRamp150.sdds` | the magnet ramp from 100 or 150 MeV/c to 3.5 GeV/c, momentum against time |
| `lattice/booster.lte` | the booster for the comparison, with the `RAMPP`, `RAMPRF` and `SREFFECT` elements |
| `lattice/br14_simple.lte` | the static lattice (no ramp), used by `tests/test_native_errors.py` |
