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

## Bends, apertures, and external programs

Injector bends and placed elements are evaluated in their local beam frame;
apertures remove particles whose step path leaves the configured acceptance.
The native models warn when an imported input option is not represented.

Elegant and ASTRA execution, executable-driven comparisons, and their result
readers remain parent research-workbench responsibilities.
