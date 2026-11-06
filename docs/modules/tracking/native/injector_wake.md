# `slipstream.tracking.native.injector_wake`

[Source](../../../../src/slipstream/tracking/native/injector_wake.py) · [Documentation index](../../../README.md)

## Physics

Localized longitudinal (monopole) wake kicks for the Cartesian injector, the model of ASTRA's `&WAKE` namelist
(ASTRA manual 3.2, Sec. 6.8, pp. 68–69; Dohlus et al., “Fast Particle Tracking with Wake Fields”, DESY 12-012,
[arXiv:1201.5270](https://arxiv.org/abs/1201.5270)). When the bunch centroid reaches `z_m`, the particle at `z` gets the
energy change

```text
dE(z) = - q * scaling * V(z),      V(z) = sum_j q_j w(z_j - z)   over the particles ahead of it (z_j > z)
```

with `w(s)` the monopole table in V/C, `q_j` the signed macroparticle charges and `q` the signed charge of the particle
the equations of motion track. A positive `w` retards like charges; for `w(s)` = const = `w0` the bunch loses
`Q^2 w0 / 2` in total and `V(zeta) = w0 Q/2 erfc(zeta/(sqrt2 sigma))` for a Gaussian bunch. A purely longitudinal force
changes the proper velocity by `du_z = gamma dE / (m u_z)`; the transverse velocities are untouched.

One `&WAKE` entry is one kick of the whole table. A travelling-wave structure of `n` cells gets `n` kicks, one at the
centre of each cell (`Wk_z = start + (i + 1/2) * cell_length`), because the CLARA tables are per-cell wake functions
(LAURA's ASTRA writer does the same).

### Table formats

`monopole_f` and `monopole_p` files are `N 0` followed by `N` rows `s w`. A `taylor_f` file is `Nt 0` followed by `Nt`
sub-tables `[Nq Np]`, `[R L]`, `[C~ 10a+b]`, `Nq` rows `(s, q(s))` and `Np` rows `(s, p(s))`; the sub-table with code
`10a+b = 0` is the monopole and its `q(s)` is `w(s)`. For the CLARA table `SzSx5um10mm.dat` it equals elegant's
`Wz(t = s/c)` of `SzSx5um10mm.sdds` exactly (5 µm spacing to 10 mm, `w(0) = 9.224e12 V/C`). The table is zero beyond its
last point.

## Usage

```python
inj = Injector.from_astra("injector400.in")      # reads &WAKE: LWAKE, Wk_Type, Wk_filename, Wk_z, Wk_scaling, Wk_N_bin
inj.wakes.append(InjectorWake(z_m=1.21, file="SzSx5um10mm.dat", method="taylor_f", n_bin=128))
out = inj.track(...)
out["wakes"]   # per kick: nominal z_m, the centroid z it was applied at, the mean energy change [eV]
```

Wakes are applied by `Injector.track` after the cavities have been passed (all particles emitted) and are saved with
`Injector.save`. `track_differentiable` raises `NotImplementedError` when wakes are present. Dipole tables
(`Dipole_Method_*`), a kick direction other than +z and transverse origins are not modelled; `from_astra` warns about
them. `Wk_N_bin` keeps ASTRA's default of 10 when a deck gives none, which is coarse: set it explicitly.

## Implementation notes

* The potential is evaluated on `n_bin` equidistant bins between the extreme particles. Charge goes to the bin nodes
  with linear weights, the node potential is the trapezoid sum `q_k w(0)/2 + sum_{m>k} q_m w((m-k) h)` and is
  interpolated back with the same weights, so the result converges to the continuum limit with `n_bin` (the closed forms
  above are reproduced to about 1% of the peak at 256 bins). ASTRA's `Wk_ip_method`, `Wk_smooth`, `Wk_sigma_min`,
  `Wk_equi_grid` and `Wk_sub` interpolation options are not reproduced: compare with ASTRA at converged `Wk_N_bin`.
* The kick time needs the exact centroid, a host sync; it is read only within 12 steps of the next kick, otherwise the
  loop's 10-step centroid is used.
* The kick is computed in float64 whatever the tracking dtype and added to `u_z` in the particle dtype. A 250 pC kick of
  about 650 eV per cell is well above the float32 resolution of `u_z` at 35 MeV, but a much weaker wake is not.
* Tests: `tests/test_injector_wake.py` (closed forms, energy balance through `Injector.track`, deck and YAML round
  trips). The ASTRA comparison at 250 pC is in the workbench story `CLARA_InjectorConvergence` (`wake_runs.py`).
