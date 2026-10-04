# Slipstream Claude notes

Read `AGENTS.md` first. This file is the shorter operational reference for
working in this checkout with Claude or another coding agent.

## Quick facts

- Package: `slipstream-accel`; import name: `slipstream`.
- Python: `>=3.10`.
- Build: setuptools from `pyproject.toml`, with a `src/` layout.
- Tests: pytest under `tests/`.
- Main runtime dependencies: NumPy, SciPy, pandas, h5py, PyYAML, and PyTorch.
- Optional dependencies: `sdds` and `warp-lang` through the `sdds` and `cuda`
  extras.
- CI is CPU-oriented; the checkout `requirements.txt` is CUDA 13.2/Warp
  oriented and should not be copied into CPU CI jobs.

## Fast workflow

```bash
python -m pip install -e .
python -m pip install pytest ruff build twine
python -m pytest tests/test_package_smoke.py -q
python -m pytest -q
ruff check src tests
python -m build
python -m twine check dist/*
```

See [`docs/compilation.md`](docs/compilation.md) for CPU/CUDA
`torch.compile`, cache, and Warp setup details.

Use `python -m pytest <path-or-test> -q` for a focused iteration. If Python,
PyTorch, CUDA, Warp, SDDS, or a compiler is unavailable, say so explicitly
instead of changing the test to hide the missing capability.

## Where to work

- Native ring transport: `src/slipstream/tracking/native/tracker.py` and
  `ring.py`.
- One-pass linac transport: `src/slipstream/tracking/native/linac.py`.
- Injector and space charge: `injector.py`, `space_charge.py`, `fields.py`,
  and `magnets.py`.
- Element definitions and elegant parsing: `elements.py` and
  `lattice_import.py`.
- Beam distributions and ASTRA I/O: `tracking/distributions.py` and
  `tracking/native/astra_io.py`.
- Ring/linac results and particle files: `data/results.py` and
  `data/particles.py`.
- Wake and impedance models: `wakefields/`.
- Slice and nonlinear diagnostics: `analysis/`.
- Stable PyTorch Inductor/Triton cache setup: `_compile_env.py`.
- User-facing behavior and examples: `README.md` and `docs/`.

## Important conventions

Native tracker states use `(x, px, y, py, tau, delta)`, with `tau` in metres
and `delta` as relative momentum deviation. Distribution generators instead
return `(x, xp, y, yp, tau, delta)` in the slope convention. Keep these
interfaces distinct and test conversions explicitly.

`NativeTracker` is a ring tracker and expects a `Lattice`; `LinacTracker` is a
single-pass tracker and does not; `Injector` is an ASTRA-style space-charge
transport model. Do not turn a linac or injector change into a ring special
case just to reuse a constructor.

The native `.lte` parser is intentionally small: it handles the supported
element and line syntax documented in `docs/formats.md`, not the full elegant
language. It must remain a local parser and must not invoke elegant.

## PyTorch and compilation

`slipstream` imports PyTorch at package import time. `_compile_env.py` sets
stable, OS-specific cache directories for Inductor and Triton when explicitly
imported; do not share a compile-cache root between Windows and WSL/Linux.

Compiled paths are shape-sensitive and can recompile when changing Python
values are passed into a traced function. Keep fixed shapes, particle counts,
mesh buckets, and static loop structure where the surrounding code expects
them. Preserve an eager PyTorch path for unsupported devices or dtypes.

On Windows, use `clang-cl` as the C++ compiler for `torch.compile`; it compiles
this codebase's CUDA paths better than MSVC. Set it before launching Python:

```powershell
$env:CXX = "clang-cl"
```

`clang-cl` and the Windows SDK must be available. The test suite skips compiled
parity coverage if the required toolchain is not present. This is separate
from CUDA/Warp availability.

When changing a compiled kernel, compare eager and compiled results on the
same inputs. For numerical code, test the precision and device combinations
that the change claims to support; do not assume CPU float64 behavior proves
CUDA float32 behavior.

## Testing expectations

Tests should assert exact or appropriately bounded numerical behavior when a
reference is available. For physics changes, include the units and convention
in the test setup, and test edge cases such as empty/truncated bins, particles
outside a fixed window, zero or degenerate fields, and missing optional input
files when relevant.

Keep optional or hardware-dependent coverage isolated. A missing CUDA device,
Warp install, SDDS install, or Windows compiler must not break core CPU tests
unless the test is specifically for that capability.

## Documentation and scope

Update `README.md` or `docs/` when changing a public workflow, input format,
coordinate convention, or user-visible option. Keep new documentation tied to
files that exist in this repository and avoid machine-specific absolute paths.

Keep `slipstream/__init__.py` and `__all__` aligned with the intended public
API. Prefer module-local imports for implementation details and avoid adding a
new import-time side effect to the top-level package.

Before finishing, inspect the diff, run relevant tests, and distinguish tested
CPU/eager behavior from untested CUDA, Warp, external-format, or compiled
behavior.
