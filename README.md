# Slipstream

Native accelerator beam-dynamics and tracking tools built on PyTorch.

Slipstream provides native ring, linac, and injector tracking. It reads
elegant `.lte` lattice files and ASTRA input/distribution files directly; it
does not launch elegant or ASTRA.

## Installation

Requires Python 3.10 or newer. The default install can run on CPU; no GPU,
compiler, elegant or ASTRA executable is needed for a first run:

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

Optional SDDS support is available with:

```bash
python -m pip install "slipstream-accel[sdds]"
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

For frozen-energy ring maps, normal modes, loss locations, and spectral
analysis, see [`docs/nonlinear_dynamics.md`](docs/nonlinear_dynamics.md).
RF-on dynamic aperture, frequency maps, and sampled momentum acceptance are
documented in [`docs/ring6d.md`](docs/ring6d.md).

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

## Scope

For opt-in photocathode/injector intrabeam scattering, see
[local binary collisions](docs/modules/tracking/native/injector_ibs.md).
`InjectorIBSSettings` configures the collision cells, angular model, cutoff
and seed independently of storage-ring IBS growth-rate models.

- `NativeTracker` models ring transport and turn-by-turn effects.
- `LinacTracker` models one-pass element-by-element transport.
- `Injector` models ASTRA-style gun, cavity, solenoid, space-charge, and
  aperture transport. An optional `Photocathode` derives its emission from
  laser energy, quantum efficiency (with the Schottky effect) and the
  space-charge-limited surface field.
- `warp-lang` accelerates supported CUDA paths; it is not required for CPU
  execution.

Ring reference tables and plots are described in
[`docs/ring_outputs.md`](docs/ring_outputs.md). Install the optional
`slipstream-accel[plot]` extra to render the figures.
