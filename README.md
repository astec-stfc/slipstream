# Slipstream

Native accelerator beam-dynamics and tracking tools built on PyTorch.

Slipstream provides native ring, linac, and injector tracking. It reads
elegant `.lte` lattice files and ASTRA input/distribution files directly; it
does not launch elegant or ASTRA.

## Installation

The default install works on CPU and CUDA through PyTorch's normal package
selection:

```bash
python -m pip install slipstream-accel
```

For the CUDA-accelerated backend, install the `cuda` extra and select the
CUDA 13.2 PyTorch index. The extra adds NVIDIA Warp; if Warp is unavailable,
Slipstream falls back to its PyTorch implementation.

```bash
python -m pip install \
  --index-url https://download.pytorch.org/whl/cu132 \
  --extra-index-url https://pypi.org/simple \
  "slipstream-accel[cuda]"
```

From a checkout, `requirements.txt` installs the editable package with the
same CUDA extra plus the test runner:

```bash
python -m pip install -r requirements.txt
```

Optional SDDS support is available with:

```bash
python -m pip install "slipstream-accel[sdds]"
```

## Windows compilation

On Windows, use `clang-cl` as the C++ compiler for `torch.compile`. It compiles
Slipstream's CUDA paths better than MSVC. Set it before starting Python, and
make sure `clang-cl` and the Windows SDK are available:

```powershell
$env:CXX = "clang-cl"
```

The equivalent setting is `set CXX=clang-cl` in `cmd.exe` or
`export CXX=clang-cl` in a POSIX shell.

See [`docs/compilation.md`](docs/compilation.md) for CPU and CUDA
`torch.compile` setup, compilation-cache guidance, and optional NVIDIA Warp
acceleration.

## Quick start

The Python import name is `slipstream`:

```python
import torch

from slipstream import LinacTracker, parse_elegant_lattice

specs = parse_elegant_lattice("beamline.lte", "BEAMLINE")
tracker = LinacTracker(specs, energy_eV=100e6, device="cuda")
particles = torch.zeros((1024, 6), device="cuda", dtype=torch.float64)
particles_out, energy_out = tracker.track(particles)
```

Particle states use the canonical `(x, px, y, py, tau, delta)` layout. See
[`docs/tracking_native.md`](docs/tracking_native.md) for ring, linac, and
injector usage, and [`docs/formats.md`](docs/formats.md) for supported input
formats.

## Scope

- `NativeTracker` models ring transport and turn-by-turn effects.
- `LinacTracker` models one-pass element-by-element transport.
- `Injector` models ASTRA-style gun, cavity, solenoid, space-charge, and
  aperture transport.
- `warp-lang` accelerates supported CUDA paths; it is not required for CPU
  execution.
