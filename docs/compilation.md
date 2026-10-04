# PyTorch compilation and CUDA acceleration

This guide explains how Slipstream uses an installed PyTorch build. It covers
`torch.compile` (PyTorch's runtime kernel compilation) and the optional NVIDIA
Warp kernels.

## CPU installation

From a checkout, install the package and the development tools into the active
environment:

```bash
python -m pip install -e .
python -m pip install pytest ruff build twine
```

The default package dependencies install a CPU-capable PyTorch build through
the normal Python package index. Verify the installation before trying a
compiled path:

```bash
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

On Linux, make sure a supported C++ compiler is available before using
`torch.compile` on CPU. On Windows, use LLVM's `clang-cl` as the compiler for
this repository. It compiles Slipstream's CUDA paths better than MSVC and must
be selected before Python starts:

```powershell
$env:CXX = "clang-cl"
```

`clang-cl` and the Windows SDK must be available in the environment. The
equivalents in other shells are `set CXX=clang-cl` in `cmd.exe` and
`export CXX=clang-cl` in a POSIX shell.

## Enabling `torch.compile`

Compilation is opt-in for the ring and linac trackers:

```python
from slipstream import LinacTracker, NativeTracker

ring_tracker = NativeTracker(specs, lattice, device="cpu", compile=True)
linac_tracker = LinacTracker(specs, energy_eV=100e6, device="cpu", compile=True)
```

`NativeTracker` and `LinacTracker` compile their reusable transport step, not
the entire outer tracking loop. The first call is slower because PyTorch is
tracing and generating kernels; later calls reuse the compiled functions.

`Injector.track()` defaults to `use_compile=True`, but its regular tracking
path only enables compiled kernels when `device="cuda"`. CPU injector tracking
is intentionally eager. `Injector.track_differentiable(use_compile=True)` can
compile its fixed-step autograd path on CPU or CUDA when a compiler is
available.

Compilation specializes on tensor shape and some fixed values. For useful
performance and predictable compile time:

- keep particle count and tensor shapes stable within a run;
- keep injector mesh buckets stable, or use a fixed mesh for benchmarking;
- avoid repeatedly changing Python scalar arguments to a compiled function;
- compare compiled and eager results on identical inputs before trusting a
  new compiled kernel.

For numerical work, validate the compiled result against the eager PyTorch
path at the dtype and device used by the study. CPU float64 agreement does not
by itself validate CUDA float32 behavior.

## CUDA installation

The repository's CUDA extra uses the CUDA 13.2 PyTorch wheel index and adds
`warp-lang`:

```bash
python -m pip install \
  --index-url https://download.pytorch.org/whl/cu132 \
  --extra-index-url https://pypi.org/simple \
  -e ".[cuda]"
```

For the same editable checkout setup used by the repository, including pytest:

```bash
python -m pip install -r requirements.txt
```

Check that the installed PyTorch can see the GPU:

```bash
python -c "import torch; print(torch.__version__); print('PyTorch CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

The CUDA wheel, driver, GPU, and installed Warp version must be mutually
compatible. If `torch.cuda.is_available()` is false, fix the PyTorch/driver
installation first; installing Warp cannot make an unavailable PyTorch CUDA
device usable.

Use `device="cuda"` and an explicitly chosen dtype when constructing a
tracker. For example:

```python
import torch

from slipstream import LinacTracker

tracker = LinacTracker(
    specs,
    energy_eV=100e6,
    device="cuda",
    dtype=torch.float32,
    compile=True,
)
particles_out, energy_out = tracker.track(particles6)
```

Float32 is usually the performance-oriented CUDA choice. Use float64 when
the physics, cancellation, or long accumulation requires it, and benchmark
the memory and throughput cost on the target GPU.

## Using NVIDIA Warp

Warp is an optional CUDA accelerator, not a separate physical model. The
PyTorch implementation remains the correctness fallback when Warp is absent.
No `backend="warp"` argument is needed: supported paths select Warp
automatically when all of the following are true:

- PyTorch is using CUDA;
- the dtype is float32 or float64;
- `warp-lang` is importable; and
- the tracker path has Warp support enabled.

The main user-facing automatic path is:

```python
result = injector.track(
    device="cuda",
    dtype=torch.float32,
    use_compile=True,
)
```

For this injector path, Warp handles selected hot kernels such as particle
push, aperture checks, and space-charge deposition. Other parts remain in
PyTorch, including field evaluation and operations where PyTorch is faster or
needed for the surrounding algorithm. `torch.compile` may therefore still be
used alongside Warp.

Some native ring-map forward paths also select Warp automatically for CUDA
tracking when the state does not require gradients and the map is eligible.
Gradient-based or unsupported paths stay in PyTorch. This applies with
`RingMap(compile=True)` too. The packed Warp turn was faster than compiled
per-element kernels at 1e2-1e5 particles; see
[ring_warp](modules/tracking/native/ring_warp.md).

To inspect whether the optional Warp module is present and applicable:

```bash
python -c "import torch; from slipstream.tracking.native import warp_kernels; print('Warp installed:', warp_kernels._HAVE_WARP); print('Warp float32 path:', warp_kernels.available('cuda', torch.float32) if torch.cuda.is_available() else False)"
```

The `available()` result is a capability check, not a benchmark. Measure the
complete tracking workload with representative particle counts, mesh sizes,
dtype, and recording settings before deciding whether Warp is faster.

## Compilation caches

Slipstream provides `slipstream._compile_env` to place PyTorch Inductor and
Triton caches in stable, OS-specific directories. Opt in at process startup:

```python
import slipstream._compile_env  # noqa: F401
```

Do not share one compilation-cache directory between Windows and WSL/Linux;
compiled artifacts are platform- and toolchain-specific. The first run after a
new PyTorch version, compiler change, dtype, or graph shape can still compile
again.

## Troubleshooting checklist

1. Confirm the Python environment is the one where Slipstream is installed.
2. Check `torch.__version__`, `torch.version.cuda`, and
   `torch.cuda.is_available()`.
3. On Windows, set `$env:CXX = "clang-cl"` before launching Python.
4. Run the same case with `compile=False` or `use_compile=False` to separate a
   compiler problem from a physics or input problem.
5. If Warp is unavailable, verify the PyTorch CUDA path first; Slipstream will
   fall back to compiled PyTorch kernels where supported.
6. If compilation repeats, keep shapes, particle counts, and mesh buckets
   fixed and inspect whether a changing Python value is being guarded.
