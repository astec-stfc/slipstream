# Slipstream

[![CI](https://github.com/astec-stfc/slipstream/actions/workflows/ci.yml/badge.svg)](https://github.com/astec-stfc/slipstream/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Native accelerator beam-dynamics and tracking tools built on PyTorch.

Slipstream tracks particles through rings, linacs and photoinjectors,
including the collective effects that matter for bright beams. It reads
elegant `.lte` lattice files and ASTRA input/distribution files directly; it
does not launch elegant or ASTRA. It runs on CPU, uses CUDA when available,
and its trackers are differentiable, so beam parameters can be fitted or
optimised with gradients.

## What it does

- **Rings.** `NativeTracker` integrates per-element transport every turn,
  with RF, radiation damping and excitation, wakes, intrabeam scattering,
  energy ramps and static errors. Frozen-energy analysis covers closed orbits,
  chromaticity, tune footprints, amplitude detuning, dynamic aperture and
  frequency maps. See [nonlinear dynamics](docs/nonlinear_dynamics.md) and
  [RF-on ring studies](docs/ring6d.md).
- **Linacs and beamlines.** `LinacTracker` follows one pass element by element
  with inline RF, structure wakes, longitudinal space charge, coherent
  synchrotron radiation (CSR), local bend radiation for electrons and a
  transverse deflecting cavity. `BeamlineStudy` runs a complete study from a
  YAML file or the command line.
- **Injectors.** `Injector` follows ASTRA-style cathode emission through
  cavities, solenoids, magnets and apertures, with cylindrical or 3D space
  charge, cathode images, structure-wake kicks and optional binary-collision
  intrabeam scattering. An optional `Photocathode` derives its emission from
  laser energy, quantum efficiency (with the Schottky effect) and the
  space-charge-limited surface field.
- **Gradients.** Linac passes (including fast CSR and longitudinal space
  charge), injector tracking (with checkpointing to bound memory) and ring
  optics are differentiable. A differentiable screen and a CUDA-graph step
  support gradient-based beam reconstruction; see
  [`slipstream.reconstruction`](docs/modules/reconstruction.md).
- **Inputs and outputs.** elegant `.lte`, ASTRA decks and distributions,
  YAML machine files and HDF5 particle files, with optional SDDS and LAURA
  support. See [file formats](docs/formats.md).

## Installation

Requires Python 3.10 or newer. The default install can run on CPU; no GPU,
compiler, elegant or ASTRA executable is needed for a first run.

Install the released package from PyPI:

```bash
python -m pip install slipstream-accel
```

From a checkout, install the package for local development with:

```bash
python -m pip install -e .
```

`requirements.txt` is the CUDA-oriented development setup, including Warp
and pytest. See [getting started](docs/getting_started.md) for virtual
environments and common installation problems.

Optional features are installed as extras. For example, SDDS support and the
Matplotlib figures for [ring outputs](docs/ring_outputs.md):

```bash
python -m pip install "slipstream-accel[sdds,plot]"
```

## Quick start

The Python import name is `slipstream`. This complete example generates its
own bunch and runs on CPU:

```python
from slipstream import BeamlineStudy, ElementSpec

study = BeamlineStudy(
    [ElementSpec("D", "drift", length=1.0)], energy_eV=100e6, device="cpu", always_change_p0=False,
)
particles = study.generate_bunch(
    n=1024, beta_x0=2.0, alpha_x0=0.0, epsx_m=1e-8,
    beta_y0=2.0, alpha_y0=0.0, epsy_m=1e-8,
    sigma_tau_m=1e-4, sigma_delta=1e-3, seed=42,
)
result = study.track(particles)
print(result.particles_out.shape)  # (1024, 6)
print(result.energy_eV_out / 1e6)  # 100.0 MeV total energy
```

Particle states use the canonical `(x, px, y, py, tau, delta)` layout;
`tau` is in metres and energies include rest energy. `generate_bunch()`
returns canonical momenta ready for tracking. See
[`docs/tracking_native.md`](docs/tracking_native.md) for ring, linac, and
injector usage, and [`docs/formats.md`](docs/formats.md) for supported input
formats.

Run a complete synthetic beamline study from a portable YAML configuration:

```bash
python -m slipstream.beamline_cli examples/beamline/study.yaml -o result.h5
```

See [beamline studies](docs/beamline_studies.md) for generated, HDF5, SDDS,
and native ASTRA sources. Change settings without editing YAML, for example:

```bash
python -m slipstream.beamline_cli docs/examples/csr.yaml --set tracker.csr_transient=true --set tracker.csr_drift_model=retarded
```

The [model settings guide](docs/model_settings.md) explains CSR choices and
other switches. The [examples index](docs/examples/README.md) lists runnable
starting points. The [documentation index](docs/README.md) links
physics explanations, usage guides and the reference for every module.

## Documentation map

| I want to... | Start here |
| --- | --- |
| Install and run a first study | [getting started](docs/getting_started.md) |
| Track a ring, linac or injector | [native tracking](docs/tracking_native.md) |
| Choose CSR and other model switches | [model settings](docs/model_settings.md) |
| Study ring nonlinear dynamics | [nonlinear dynamics](docs/nonlinear_dynamics.md), [RF-on ring studies](docs/ring6d.md) |
| Produce ring tables and plots | [ring outputs](docs/ring_outputs.md) |
| Fit or optimise with gradients | [beam reconstruction](docs/tracking_native.md#beam-reconstruction-with-a-differentiable-screen), [nonlinear optimisation](docs/nonlinear_optimisation.md) |
| Add injector intrabeam scattering | [local binary collisions](docs/modules/tracking/native/injector_ibs.md) |
| Check a convention or formula | [native physics reference](docs/native_reference.md) |

## Compilation and CUDA

Start with eager CPU tracking, as above. When enabling `torch.compile`, see
[compilation.md](docs/compilation.md) for toolchains, caches and optional
Warp acceleration. On Windows, set `$env:CXX = "clang-cl"` in PowerShell
before starting Python; `clang-cl` and the Windows SDK must be available.

For the CUDA-accelerated backend, install the `cuda` extra and select the
CUDA 13.2 PyTorch index. The extra adds NVIDIA Warp; if Warp is unavailable,
Slipstream falls back to its PyTorch implementation:

```bash
python -m pip install \
  --index-url https://download.pytorch.org/whl/cu132 \
  --extra-index-url https://pypi.org/simple \
  "slipstream-accel[cuda]"
```

`warp-lang` accelerates supported CUDA paths; it is not required for CPU
execution.

## Status and roadmap

Slipstream is at version 1.1. Known gaps, planned physics, validation work
and performance items are tracked as
[GitHub issues](https://github.com/astec-stfc/slipstream/issues). Items
labelled `deferred` have no committed schedule and wait for a concrete use
case; comment on one if you have it. Each physics model records the reference
it was checked against in its [module page](docs/modules/README.md); a model
described there as validated has been compared with an analytic limit or a
reference code under the stated conditions only.

## Contributing

Bug reports, physics requests, documentation and code are welcome, including
work written with AI assistance. Read [CONTRIBUTING.md](CONTRIBUTING.md) for
the workflow, checks and code standards before opening a pull request.

## Licence

Slipstream is released under the [Apache License 2.0](LICENSE).
