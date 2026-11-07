# Slipstream documentation

Start with a tracker matching the physical problem. `NativeTracker` follows
ring turns, `LinacTracker` follows one beamline pass, and `Injector` follows
Cartesian particles through injector fields and space charge.

New users: [getting started](getting_started.md) walks from installation to
a saved result. The [examples index](examples/README.md) lists runnable
entry points; [model settings](model_settings.md) explains switches and CSR choices.

## Physics

- [Native physics reference](native_reference.md): coordinates, transport,
  RF, radiation, IBS and optics conventions.
- [Module reference](modules/README.md): physics, approximations and usage
  for every package module.
- [Nonlinear dynamics](nonlinear_dynamics.md), [RF-on ring studies](ring6d.md)
  and [multipoles](ring_multipoles.md).

## Usage

- [Native tracking](tracking_native.md) and [beamline YAML studies](beamline_studies.md).
- [Beam reconstruction](tracking_native.md#beam-reconstruction-with-a-differentiable-screen): screen images, scan models and CUDA-graph steps for GPSR-style fits.
- [File formats](formats.md), [ring tables and plots](ring_outputs.md) and
  [nonlinear optimization](nonlinear_optimisation.md).
- [Compilation and CUDA](compilation.md): toolchains, precision, caches and
  optional Warp acceleration.

## Maintaining documentation

Each module page separates **Physics** (model, units, conventions and limits)
from **Usage** (inputs, outputs, examples and API contracts). Numerical notes
explain choices that matter for correctness or performance. Short docstrings
and local comments keep essential contracts and cautions next to the code.

Keep current behavior here. Remaining work belongs in the workbench plans;
reusable development lessons belong in its `plans/NOTES.md`. Git history
preserves retired design narratives. Record the tested model, inputs, dtype,
device and reference before describing an algorithm as validated.

This structure adapts [Diátaxis](https://diataxis.fr/)'s separation of
explanation, practical guides and reference. The [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings)
informs the distinction between API docstrings and comments explaining
non-obvious local choices. Existing repository conventions take precedence
over either guide's unrelated tooling or formatting preferences.
