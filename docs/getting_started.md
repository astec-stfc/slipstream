# Getting started

## Install

Use Python 3.10 or newer. Create an isolated environment:

```console
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in PowerShell or
`source .venv/bin/activate` on Linux/macOS. Then install:

```console
python -m pip install slipstream-accel
python -c "from slipstream import BeamlineStudy; print('Slipstream is ready')"
```

For a source checkout, run `python -m pip install -e .` from the repository
root instead. This makes local source changes available immediately. The
package name is **slipstream-accel**, and the import name is **slipstream**.
The default package runs on CPU without a compiler or external tracking code.
`requirements.txt` is specifically for CUDA development.

Optional extras add SDDS I/O (`[sdds]`), figures (`[plot]`), CMA-ES optimization
(`[optimise]`) and Warp acceleration (`[cuda]`). Install only what the example
needs, for example `python -m pip install "slipstream-accel[plot]"` or
`python -m pip install -e ".[plot]"` in a checkout. For CUDA wheel selection
and compilation, follow [compilation.md](compilation.md).

## Run a first bunch

The [README quick start](../README.md#quick-start) is a complete Python
example that needs no input files. From a checkout you can also run:

```console
python docs/examples/quickstart.py
```

It generates 1,024 particles, transports them through a one-metre drift, and
prints output shape `(1024, 6)`, total energy `100.000 MeV`, and rms horizontal
size in millimetres. It uses a fixed reference energy (`always_change_p0=False`),
a fixed random seed, and writes no files.

Choose the interface matching your task:

| Task | Interface | Starting point |
| --- | --- | --- |
| Generate, track and analyze one pass | `BeamlineStudy` | [YAML studies](beamline_studies.md) |
| Work directly with canonical tensors | `LinacTracker` | [Native tracking](tracking_native.md#linac-tracking) |
| Track a storage ring for many turns | `NativeTracker` | [Ring tracking](tracking_native.md#ring-tracking) |
| Track a gun/injector in fields | `Injector` | [Injector usage](tracking_native.md#injector-usage) |
| Compute closed orbit, optics or aperture | `RingMap`, analysis helpers | [Nonlinear dynamics](nonlinear_dynamics.md) |

Native ring/linac tensors have columns `(x, px, y, py, tau, delta)`.
Positions and `tau` are metres; `px`, `py` and `delta` are dimensionless.
Total energy includes rest energy. The standalone distribution generators
return slopes; `BeamlineStudy.generate_bunch()` converts them to canonical
momenta. Injector Cartesian states require the documented conversion before
being passed to a beamline. See [coordinates](native_reference.md).

## Run YAML and read results

From the checkout root:

```console
python -m slipstream.beamline_cli examples/beamline/study.yaml --check
python -m slipstream.beamline_cli examples/beamline/study.yaml --set bunch.n=128 -o result.h5
```

The check validates configuration and constructs the lattice/tracker; it
does not load particle-source files or track particles. A normal run prints
JSON diagnostics. The synthetic RF example exits near 32 MeV total energy.
`-o result.h5` writes in the current directory and replaces an existing file.
Paths inside YAML, including overridden paths, are relative to that YAML file.

Read a saved result in Python:

```python
from slipstream import BeamlineResult

result = BeamlineResult.load("result.h5")
print(result.scalars())
print(result.particles_out.shape)  # (128, 6) for the command above
print(result.trajectory.shape)    # (5, 128, 6), including the entrance
```

`trajectory` is present only when `run.return_trajectory: true`. Slice arrays
are in `result.slice_summary`. `result.metadata["config_yaml"]` stores the
configuration with overrides applied; `config_path` records its path base.
`BeamlineResult` files and input particle HDF5 files have different schemas;
see [file formats](formats.md) before using a saved file as a bunch source.

## Resolve common problems

| Symptom | Next action |
| --- | --- |
| `No module named slipstream` | Run `python -m pip install -e .` from the checkout using the same Python interpreter as the script |
| CUDA unavailable | Use `device="cpu"` in Python or `--set tracker.device=cpu` in the CLI |
| Missing lattice or wake table | Check paths relative to the YAML/lattice file; inspect [path rules](beamline_studies.md#configuration-keys) |
| CSR/LSC appears inactive | Set nonzero bunch charge and enable the effect; CSR also requires a CSR-enabled bend; see [model settings](model_settings.md) |
| Compilation/toolchain error | Start with `compile=False` (`use_compile=False` for the injector), then follow [compilation.md](compilation.md) |

Continue with the [examples index](examples/README.md) or run
`python -m slipstream.beamline_cli --help` to see command-line options.
