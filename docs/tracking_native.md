# Native tracking

Slipstream's native backend is PyTorch code and has no dependency on an
external tracking executable. The public trackers are:

- `NativeTracker` for ring transport and turn-by-turn effects;
- `LinacTracker` for one-pass element-by-element transport;
- `Injector` for ASTRA-style gun and injector simulations.

All canonical beamline trackers use a particle tensor shaped `(N, 6)` with
columns `(x, px, y, py, tau, delta)`. Coordinates are SI-based; `tau` is a
length-like longitudinal coordinate and `delta` is the relative momentum
deviation. The injector works in Cartesian `(x, y, z, ux, uy, uz)` state
internally and returns its documented endpoint/trajectory dictionaries.

See the [native physics reference](native_reference.md) for normalization,
transport and RF conventions, and the [module reference](modules/README.md)
for individual model limits and API contracts.

## Choosing a backend

The default install contains the PyTorch implementation and runs on CPU or
CUDA. The optional `cuda` extra installs NVIDIA Warp for supported CUDA hot
kernels:

```bash
python -m pip install \
  --index-url https://download.pytorch.org/whl/cu132 \
  --extra-index-url https://pypi.org/simple \
  "slipstream-accel[cuda]"
```

Warp is an accelerator, not a correctness dependency. If `warp-lang` is not
installed, or the tracker is on CPU, the PyTorch path remains active.

## Ring tracking

Parse an elegant lattice and provide its machine parameters to
`NativeTracker`:

```python
import torch

from slipstream import Lattice, NativeTracker, parse_elegant_lattice

specs = parse_elegant_lattice("ring.lte", "RING")
params = Lattice.from_yaml("ring.yaml")
tracker = NativeTracker(specs, params, device="cuda", dtype=torch.float64)
particles6 = torch.zeros((1024, 6), device="cuda", dtype=torch.float64)
result = tracker.track(particles6, n_turns=100)
```

`NativeTracker` applies exact per-element transport, with optional RF,
radiation, wakefield, IBS, ramp, and error models configured through its
constructor. Use `rf="per_element"` when RFCA elements must act at their
physical lattice positions; the default `rf="lumped"` applies one ring RF
kick per turn.

## Linac tracking

For a complete run from YAML, including bunch generation or an injector
stage, use [BeamlineStudy](beamline_studies.md). The package includes a
portable synthetic example and `python -m slipstream.beamline_cli` runner.

`LinacTracker` takes the same parsed element list and an injection energy:

```python
from slipstream import LinacTracker

tracker = LinacTracker(specs, energy_eV=100e6, device="cuda")
particles_out, energy_out = tracker.track(particles6)
```

Set `return_trajectory=True` to also receive the state and reference-energy
profiles at every element boundary. LSC and CSR are opt-in through the
tracker constructor.

## Injector usage

Build an injector from an ASTRA deck, then track its distribution:

```python
from slipstream import Injector

injector = Injector.from_astra("injector.in")
injector.save("injector.yaml")
result = injector.track(device="cuda", dtype="float32")
```

`track()` auto-phases cavities when needed, returns endpoint values and a
trajectory, and accepts `n_particles` for a reweighted sample. Use
`return_particles=True` when a downstream tracker needs the final particle
state. `track_differentiable()` exposes the fixed-step autograd path for
parameter studies.

## Photocathode emission

By default the distribution file or `generator.in` fixes the bunch charge. Set
`injector.photocathode = Photocathode(pulse_energy_J=..., wavelength_nm=...,
qe=..., ...)` (`from slipstream import Photocathode`) to derive it from the laser energy and a quantum efficiency that
follows the Schottky-lowered work function, limited by the net field at the
cathode surface (space charge) and by electrons falling back onto the cathode.
Read `result["emission"]` and track with `screen_z_m` at the charge monitor to
reproduce a Schottky scan, or scan laser energy and spot size for the
space-charge limit. See the [`photocathode`](modules/tracking/native/photocathode.md)
page and `examples/photocathode/`. Only `track()` supports it.

## Space charge and mesh sizing

Injector space charge is enabled by default through `SpaceChargeSettings`.
The default `mesh="adaptive"` derives cylindrical `(nr, nz)` resolution from
the live bunch size and quantizes it into stable grid buckets. Use
`mesh="fixed"` with explicit `nr` and `nz` for reproducible manual sizing.

`geometry="cylindrical"` is the default. `geometry="3d"` selects the
Cartesian open-boundary FFT solver and uses `n_xyz`; cylindrical smoothing,
cell targets, and cathode-image settings do not control that 3-D grid.

`smooth_r` and `smooth_z` add binomial smoothing to the deposited charge
grid. Leave them at zero unless deposition noise is part of the problem being
studied. `every_step_until` and `subcycle_steps` control how often the space-
charge field is recomputed.

## Time stepping and compilation

`dt_adaptive=True` derives the injector step from the local space-charge
frequency and clamps it between `dt_floor_s` and `dt_ceiling_s`. Set it to
`False` for a fixed-step comparison. The `dt_safety` setting is the main
accuracy/performance knob.

`use_compile=True` is the default for `Injector.track()` and can reduce the
cost of repeated fixed-shape CUDA steps. Compilation is shape-sensitive:
keep particle count and mesh buckets stable within a run. Use
`track_differentiable()` with a fixed `n_steps` when gradients are required.

### Scans and step cost

Up to a few hundred thousand macroparticles a CUDA step is launch-bound: the
host spends about 1 ms per step issuing kernels while the GPU is mostly idle.
Measured on an RTX 3050 laptop GPU (Windows, PyTorch 2.14, Warp 1.17, 1500
macroparticles, float32, one gun phase to z = 0.45 m, screen and apertures on):
eager 3.1 ms/step (13 before idle checks were skipped), Warp-backed
0.65-0.72 ms/step (1.1-1.6 before), plain `torch.compile` 0.33-0.54 ms/step,
with identical charges. `track(use_warp=False)` keeps the step and the
space-charge deposit on `torch.compile` where Warp's Python argument handling
costs more than it saves (it did at 1.5k to 1M macroparticles here: 25-35 %
slower per step); the default is unchanged.

Scanning an RF phase must not recompile: the cavity phases enter the compiled
step as 0-d tensors. As Python floats they were compile-time constants, so each
new phase recompiled the step (about 10 s) until `torch._dynamo`'s recompile
limit silently dropped it to eager. The screen plane, the aperture slabs and
the cathode wall are only checked on steps where a live particle can reach
them (the live z-range at the last 10-step sync, widened by `c*dt` per step);
this skips idle work without changing any result.

For a scan that needs only the charge, `dt_adaptive=True` with a larger
`dt_safety` (5e7 against the 5e5 default) and `dt_ceiling_s=2e-12` keeps the
fine step while space charge is strong and cuts the step count about fourfold:
the CLARA Schottky scan changes by 1.2 pC rms (at most 6.7 pC in the sparse tail
of the cut-off) against a fixed 0.4 ps step. Over six phases of the aperture-limited
ramp it deviates from a fixed 0.1 ps step by 1.7 pC rms, as the fixed 0.4 ps step
does (1.6 pC); a 4 ps ceiling is visibly worse (2.7 pC).

## Bends, apertures, and external programs

Injector bends and placed elements are evaluated in their local beam frame;
apertures remove particles whose step path leaves the configured acceptance.
The native models warn when an imported input option is not represented.

Elegant and ASTRA execution, executable-driven comparisons, and their result
readers remain parent research-workbench responsibilities.
