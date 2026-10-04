# Native beamline studies

`BeamlineStudy` composes a single `LinacTracker` with a bunch source and
optional injector stage. A strict YAML file can run a complete one-pass
study without external tracking executables:

```console
python -m slipstream.beamline_cli examples/beamline/study.yaml -o result.h5
# After installing the package:
slipstream-beamline examples/beamline/study.yaml -o result.h5
```

The example is synthetic. It needs CPU PyTorch and the package's ordinary
dependencies. Results contain native canonical particles, exit total energy,
charge, slice diagnostics, optional element trajectories, and scalar reference
differences. The CLI prints scalar diagnostics as JSON. `-o` is relative to the
current directory; every path inside YAML is relative to the YAML file.

```python
from slipstream import BeamlineStudy

study = BeamlineStudy.from_yaml("examples/beamline/study.yaml")
result = study.run()
result.save("result.h5")
restored = study.load_result("result.h5")
```

The configuration is paths and scalars only: no includes, expressions, Python
objects, or external commands. Unknown keys and invalid scalar types raise
clear errors. YAML parsing uses `safe_load`.

## Configuration keys

Top-level keys are `lattice`, `energy_eV`, `mc2_eV`, `bunch`, `tracker`, `run`,
and `reference`. The first, second, and fourth are required. Energy is total
energy, including rest energy; it must exceed `mc2_eV` (electron rest energy by
default). Use SI units throughout. Construct a new study to change the lattice
or physics switches; repeated runs reuse the same tracker.

`lattice` requires `path` (an elegant `.lte` file) and `root` (beamline name).
The native parser's documented subset applies. `wakes` defaults to `true`;
`false` skips lattice wake loading. By default `WAKEFILE` paths are relative to
the lattice directory. An optional `wake_base_dir` overrides that base and is
itself relative to the YAML file. Wake-table readers need the optional `[sdds]`
extra. There is no independent YAML wake model.

`tracker` accepts these existing constructor options:

| Options | Supported values |
| --- | --- |
| `device`, `dtype` | `cpu` (default), `cuda`, `cuda:N`; `float64` (default), `float32` |
| `compile`, `lsc`, `csr`, `fast_collective_effects`, `always_change_p0`, `dipole_collective_subcycle` | booleans, with native tracker defaults |
| `compile_mode` | null, `default`, `reduce-overhead`, `max-autotune`, `max-autotune-no-cudagraphs` |
| `compile_collective_chunks` | boolean or null (native default) |
| `bunch_charge_C` | signed total bunch charge; default 0 |
| `pipe_radius_b_m`, `csr_softening_m` | positive number or null |
| `lsc_radius_factor` | positive number |
| `n_kicks`, `n_kicks_bend`, `n_slices` | positive integers; `n_slices` controls output slice diagnostics |
| `integration_order` | 4, 6, or 8 |

CUDA and compilation require the corresponding hardware/toolchain described
in [compilation.md](compilation.md). YAML does not accept Python objects such
as sampled errors or fiducial references; supply those through the Python API.

`run` accepts `return_trajectory` (boolean, default false) and `output` (HDF5
path, optional). `run()` saves when `output` is configured; the CLI's `-o` or
Python's `run(output_path=...)` overrides that path.

## Bunch sources

`bunch.source: generated` requires `n` (at least two), `beta_x0`, `alpha_x0`,
`epsx_m`, `beta_y0`, `alpha_y0`, `epsy_m`, `sigma_tau_m`, and `sigma_delta`.
Optional keys are `Dx0`, `Dpx0`, `Dy0`, `Dpy0`, `chirp_per_m`, and `seed`.
Emittances are geometric; matching is at the beamline entrance. Seed defaults
to 0, so repeated runs reproduce on the same device. CPU and CUDA random
streams differ. Generated slopes are converted to normalized canonical
momenta before tracking.

`bunch.source: hdf5` requires `path` in the package's
`save_particles_h5` schema (`particles6`, energy/rest-mass/charge/canonical
attributes and `tau_units=m`). The file supplies its own reference energy and
charge; rest mass must match the study. Slope files are converted to canonical
momenta. This is not a generic openPMD or injector dump reader.

`bunch.source: sdds` requires `path` and the optional `[sdds]` extra. This
reader supports exactly one elegant-style particle page with columns
`x`, `xp`, `y`, `yp`, `t`, `p`. Positions are metres, slopes dimensionless,
`t` seconds, and `p` is positive momentum in rest-mass-c units (`beta*gamma`).
The configured total energy defines `p0 = sqrt((energy_eV/mc2_eV)^2 - 1)`;
`delta=p/p0-1` and canonical momenta include the exact slope normalization.
`tau=c*(t-time_origin_s)`: positive tau means later arrival.
`time_origin_s` is optional and defaults to 0; timestamps are not automatically
centered. Set it explicitly to match the lattice's RF timing convention.
Charge comes from `tracker.bunch_charge_C`. Multi-page files, alternative
column/unit schemas, and SDDS scalar reference tables are unsupported.

`bunch.source: astra` requires `path` to an ASTRA deck, read by the native
`Injector.from_astra` parser. Deck-local distribution and field paths retain
the injector's normal path resolution. Optional `n_particles` selects the
particle count (at least two). `injector` accepts `use_compile`, `record_every`,
`z_stop_m`, `mesh` (`adaptive` or `fixed`), and `verbose`. Device/dtype come
from `tracker`; compilation defaults to the tracker setting, verbosity to
false. The injector's exit particles undergo the documented ballistic
common-position conversion; its reference energy and summed absolute charge
override the initial beamline settings. ASTRA subsampling can be random;
generated-source repeatability does not promise injector repeatability.

Fast CSR bend substeps are prepared independently of initial charge, allowing
an injector or HDF5 source to supply charge after construction. Every pass
updates the charge tensor, including zero, so a previous charged run cannot
leave a stale collective kick. Zero-charge passes use the ordinary bend
transport. Source energy changes also refresh the cached magnet reference
offsets without rebuilding the tracker or changing tensor shapes.

## References

`reference` is either a scalar mapping or a path to a YAML file containing
that mapping. Supported quantities are `energy_eV`, `sigma_x_m`, `sigma_y_m`,
`sigma_tau_m`, and `sigma_delta`. They are compared using
`BeamlineResult.compare_reference`: relative difference for nonzero reference
values, absolute difference for zero. Differences are saved in
`metadata["reference_diff"]` and survive an HDF5 roundtrip. References are
diagnostics, not pass/fail tolerances. `.twi`, `.sig`, and ASTRA emit-table
reference files require explicit preprocessing into these scalar quantities.
