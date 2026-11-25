# Diamond-II booster ramp against elegant

Checks `NativeTracker(ramp=...)` against elegant on the Diamond-II booster, which accelerates from 100 MeV to 3.5 GeV in
100 ms while the RF voltage follows E^4 from 0.2 to 2 MV.

elegant tracks the real lattice ([lattice/booster.lte](lattice/booster.lte): 78 bends, 20 quadrupoles, 44 sextupoles) with
`RAMPP` (reference momentum), `RAMPRF` (voltage and phase) and `SREFFECT` (radiation damping, quantum excitation and the
energy loss per turn). slipstream tracks the same lattice from the same injected bunch, with the optics, damping times,
energy spread and `U0` taken from elegant's own Twiss and radiation integrals.

```bash
python examples/diamond2booster/validate_ramp_vs_elegant.py                    # 50 to 60 ms, the machine's phase table
python examples/diamond2booster/validate_ramp_vs_elegant.py --balanced-phase   # RF phase with V sin(phase) = U0 + dE/turn
python examples/diamond2booster/validate_ramp_vs_elegant.py --help
```

It needs elegant (set `ELEGANT_HOME`, default `C:/Program Files/APS/Elegant x64`) and writes everything to `output/`
(not tracked). elegant runs once and is cached; `--tag` names a slipstream run and `--overlay` shows earlier runs beside
it, so two versions of the code can be compared with one elegant result. The script prints, every `--interval` turns, the
centroid and rms of energy and time and the horizontal size, then the rms difference to elegant over the window, and saves
a plot.

## The window

The default window, 50 to 60 ms (1.8 to 2.6 GeV, 18 300 turns), is where the RF energy balance matters most: the radiation
loss grows as E^4 (66 to 288 keV per turn, against 0.2 to 0.6 MV of RF) and the ramp takes 29 keV per turn. The first tens
of ms are nearly adiabatic. The whole ramp (`--start-ms 0 --duration-ms 100`) is 183 000 turns, about 7 hours in slipstream
on one core (0.14 s per turn).

The machine's table keeps the RF phase at 180 degrees, so `V sin(phase)` is zero and the injected bunch starts 0.5 rad from
the synchronous phase and oscillates through the whole window. `--balanced-phase` sets the phase at which the RF supplies
exactly the loss and the gain, `V sin(phase) = U0 + dE/turn`, which leaves the bunch quiet and shows the energy balance
directly. Both are worth running.

## Two things in elegant that the comparison handles

* **RF phase reference.** elegant measures the RF phase from a nominal arrival time (the sum of the revolution times
  `C / (beta c)`), not from the tracked particle, and ignores the frequency column of the ramp file. slipstream measures the
  phase from the reference particle, an RF locked to the revolution. The two agree when the particle takes exactly
  `C / (beta c)` per turn.
* **Integration error in BD.** elegant's default 4-kick CSBEND integration of BD (K1 and K2 non-zero) takes 2.4 ps per turn
  longer than `C / (beta c)`. That is a numerical error, not path length: it falls to 0.14 ps at 8 kicks, 0.006 ps at 16, and
  0.007 ps with `INTEGRATION_ORDER=6` and `N_KICKS=4`. Left in, it is a frequency error of 4.4e-6 that moves elegant's
  synchronous particle to an energy offset of -8e-4. The script measures it on the bare ring (`BARE` line, no RF,
  radiation or ramp) and takes it off elegant's phase column.

## Files

| File | Purpose |
|---|---|
| `validate_ramp_vs_elegant.py` | builds the ramp, runs elegant, tracks with slipstream, compares |
| `lattice/booster.lte` | the booster for the comparison, with the `RAMPP`, `RAMPRF` and `SREFFECT` elements |
| `lattice/br14_simple.lte` | the static lattice (no ramp), used by `tests/test_native_errors.py` |
