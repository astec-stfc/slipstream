# Slipstream agent guide

This file is the repository-local onboarding guide for coding agents. It
describes the package in this checkout only; do not assume that sibling
workbenches, machine examples, external tracking executables, or planning
documents exist here.

## What this repository is

`slipstream-accel` is a Python package for native accelerator beam dynamics.
The import name is `slipstream`. Its implementation is PyTorch-based and can
run on CPU or CUDA. Optional NVIDIA Warp kernels accelerate selected CUDA
paths, but Warp is not required for correctness or CPU use.

The package reads relevant input formats directly. It does not launch elegant
or ASTRA executables and those programs are not package dependencies.

## Repository map

- `src/slipstream/` — installable package, using the `src` layout.
- `src/slipstream/tracking/native/` — native ring, linac, injector, ASTRA,
  elegant-lattice, field-map, magnet, space-charge, and Warp support.
- `src/slipstream/lattice/` — machine parameters, transverse optics, ramps,
  Twiss tables, and aperture/material descriptions.
- `src/slipstream/wakefields/` — impedance, short-range, long-range,
  transverse, chamber, and multilayer models.
- `src/slipstream/analysis/` — slice diagnostics, emittance, nonlinear ring
  analysis, and optimization helpers.
- `src/slipstream/data/` — particle and tracking-result HDF5 schemas.
- `docs/` — physics explanations, usage guides and file-format documentation;
  start with `docs/README.md` and the per-module index in `docs/modules/README.md`.
- `tests/` — the package's regression and numerical tests.
- `.github/workflows/` — CI, package building, and CodeQL workflows.

The public top-level API is defined in `src/slipstream/__init__.py`. When a
new public object is intended for top-level use, update that module and its
`__all__` list, then add a smoke or behavior test.

## Main tracking boundaries

Choose the tracker that matches the physical problem:

- `NativeTracker` — ring transport and turn-by-turn effects. It consumes an
  `ElementSpec` list, normally produced by
  `parse_elegant_lattice(...)`, and a `Lattice`.
- `LinacTracker` — one-pass, element-by-element transport. It does not need a
  ring `Lattice` and does not apply ring-average radiation damping, excitation,
  or IBS.
- `Injector` — ASTRA-style gun/injector transport with field maps, cavities,
  solenoids, magnets, apertures, and space charge. It can be built from an
  ASTRA deck or its saved YAML form.
- `RingMap` and `analysis.nonlinear` — native one-turn maps and nonlinear
  diagnostics such as closed orbit, optics, tunes, detuning, dynamic aperture,
  and frequency maps.

Supporting objects include `Lattice`, `TransverseOptics`, `RampProfile`,
`Impedance`, `ShortRangeWake`, `LongRangeWake`, `Resonator`, and
`TransverseWake`.

## Coordinate and unit conventions

Do not infer coordinate conventions from variable names alone.

- Native ring and linac tracker particle tensors are `(N, 6)` with columns
  `(x, px, y, py, tau, delta)`.
- In that canonical native state, `px` and `py` are normalized canonical
  transverse momenta, `tau` is a length-like longitudinal coordinate in
  metres, and `delta` is relative momentum deviation.
- Distribution generators use the slope convention
  `(x, xp, y, yp, tau, delta)`, where `xp`/`yp` are slopes. Convert that state
  before passing it to a native tracker.
- The injector's internal particle state is Cartesian
  `(x, y, z, ux, uy, uz)`; use its documented endpoint and trajectory outputs
  rather than treating that state as a native `(N, 6)` tensor.
- Keep SI units in new APIs unless the surrounding module explicitly defines a
  different unit. Preserve the existing dtype and device of tensor results.

## Development setup

The package requires Python 3.10 or newer. A generic CPU development install
is:

```bash
python -m pip install -e .
python -m pip install pytest ruff build twine
```

The repository's `requirements.txt` is the CUDA-oriented checkout setup. It
uses the PyTorch CUDA 13.2 index, installs the editable `[cuda]` extra, and
adds pytest. Use it when a CUDA/Warp environment is intentionally required;
do not use it as the default hosted CI install.

Optional extras:

```bash
python -m pip install "slipstream-accel[sdds]"
python -m pip install "slipstream-accel[cuda]"
```

## Verification commands

Run the focused test first, then the full suite when practical:

```bash
python -m pytest tests/test_package_smoke.py -q
python -m pytest -q
ruff check src tests
python -m build
python -m twine check dist/*
```

The CI workflow uses CPU PyTorch wheels and tests Python 3.10, 3.12, and
3.13. CUDA-specific behavior needs a CUDA-capable environment and should not
be silently replaced with a CPU assertion.

See [`docs/compilation.md`](docs/compilation.md) for the full CPU/CUDA
`torch.compile` and optional Warp setup.

On Windows, use `clang-cl` as the C++ compiler for `torch.compile`; it compiles
this codebase's CUDA paths more reliably than MSVC. Set it before starting
Python, for example in PowerShell:

```powershell
$env:CXX = "clang-cl"
```

`clang-cl` and the Windows SDK must be available in the environment. The
compiled-parity test skips itself when the required toolchain is unavailable.
Do not make the whole test suite depend on a local compiler that is not needed
by the eager path.

## Engineering rules

- Treat physics formulas and coordinate conversions as correctness-critical.
  Prefer a quantitative regression against a closed form, independent
  calculation, or known reference over a test that only checks a value moved
  in the expected direction.
- Preserve CPU fallback behavior when adding CUDA or Warp paths. A Warp path
  must remain an optimization of the PyTorch implementation, not a second
  source of physical behavior.
- Keep tensor operations batched. Avoid per-particle Python loops and avoid
  host/device synchronization in hot paths unless the API explicitly requires
  a scalar result.
- `torch.compile` is shape- and value-sensitive. Keep fixed-shape kernels and
  mesh buckets stable within a run, and be deliberate about Python scalar
  arguments that could trigger recompilation.
- Use float64 when a small physical increment, cancellation, or long
  accumulation requires it; use float32 only when its precision is adequate
  for the requested physics and performance target.
- Add or update tests with every behavior change, especially for unit
  conversions, RF phase/sign conventions, field-map interpolation,
  space-charge kicks, wake convolution, and file formats.
- Keep file readers deterministic and local. Never make a parser execute an
  external program or depend on a machine-specific absolute path.
- Keep generated caches, `dist/`, `build/`, virtual environments, and large
  simulation outputs out of commits; `.gitignore` already covers the common
  cases.
- Make the smallest change that proves the requested behavior. Do not add
  unrelated backend abstractions or copy functionality from a separate
  workbench into this package.
- Keep comments and docstrings concise, retaining essential units, contracts
  and local correctness cautions. Put longer explanations in the matching
  `docs/modules/` page, with separate Physics and Usage sections. Keep active
  work and development lessons out of API documentation.

## File-format boundaries

- The elegant reader supports the documented subset of `.lte` syntax and
  returns native `ElementSpec` objects. It does not evaluate elegant macros or
  execute elegant.
- ASTRA namelists, distributions, emit tables, and field files are parsed by
  `tracking.native.astra_io` and the injector helpers. Unsupported options
  should be surfaced clearly, preferably as warnings or explicit errors.
- YAML serialization is for the package's scalar/dataclass machine objects;
  large array-backed objects have their own HDF5, CSV, or SDDS I/O.
- SDDS support is optional. Tests that require it should make the dependency
  requirement explicit instead of making core imports fail.

## Before handing work back

Check `git diff`, confirm that no unrelated files changed, run the narrowest
useful tests plus the full suite when available, and report any unavailable
optional dependency or hardware. Do not claim CUDA, Warp, external-program,
or compiled-path validation when only the CPU eager path was exercised.
