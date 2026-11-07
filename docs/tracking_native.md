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

For a file-free CPU introduction, start with [getting started](getting_started.md).
For CSR choices, dependencies and changing settings, see [model settings](model_settings.md).

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

Aligned pure-dipole synchrotron loss and ISR are opt-in through bend spec
flags (including imported `SYNCH_RAD`/`ISR`). This bounded model requires
eager electron tracking with `always_change_p0=False`; see the
[local radiation reference](modules/tracking/native/radiation.md).

### Gradients through a linac pass

`LinacTracker.track` is differentiable with respect to the input particles,
including through LSC and CSR with `fast_collective_effects=True`, on CPU and
CUDA. The fast kernels skip their forward-only CUDA graphs whenever an input
requires grad, so no flag is needed. Each call builds its own charge tensor, so
several charged passes can be accumulated into one loss before a single
`backward()`.

```python
tracker = LinacTracker(specs, 150e6, device="cuda", bunch_charge_C=100e-12,
                       csr=True, lsc=True, pipe_radius_b_m=0.01, fast_collective_effects=True)
# image_loss and beam_for stand for your own screen model and scan settings
loss = sum(image_loss(tracker.track(beam_for(setting))[0]) for setting in settings)
loss.backward()
```

Limits to know before relying on a gradient:

- Wakes: the longitudinal `ShortRangeWake` rounds particles to integer bins and passes no
  gradient to `tau`; the transverse wake passes gradient to the offsets only. RF phase,
  voltage and magnet strengths are Python floats unless set as tensors with `with_settings` (below), so no gradient reaches them otherwise.
- The grid origin and spacing of the binned kernels follow the live extremes of `tau`.
- Float32 gradients of a binned density derivative are rough: in the test chicane
  (16384 particles, 256 bins) they differ from float64 by 8-18 % in `x`, `tau` and `delta`,
  against 1e-6 without CSR. Compare with float64 for the bin count, smoothing and particle
  count in use; `torch.use_deterministic_algorithms(True)` removes the run-to-run noise of
  float32 scatter but not that difference.

The step is launch-bound, so a fixed-shape forward and backward can be captured in one CUDA
graph (measured 5.7 to 15 times faster than eager on a 15-element chicane with CSR). Warm up
first on a side stream, as the PyTorch CUDA-graph documentation describes, so that the dense
smoothing operator, cuFFT plans and cached scalar constants exist before capture, then capture
`track`, the loss and `backward()` together and replay it with new values copied into the static
input. The step must contain no host synchronisation and no host-to-device scalar construction.
`tests/test_collective_gradients.py` and `tests/test_linac_scan_parameters.py` hold working
examples. Under deterministic algorithms a CSR-only replay matched eager bit for bit in float32.

Capture was checked, with and without gradients, on CUDA in float64 for: drifts, quadrupoles,
sextupoles, correctors, bends, RF cavities without wakes (with `always_change_p0` on or off and
with per-element `change_p0`), fast CSR (steady state, `csr_transient`, `csr_n_slices`) and
fast LSC (including `dipole_collective_subcycle`), and `csr_drift_model="stupakov"`. A scalar
bunch charge no longer replays the manual forward-only graphs of the fast kernels inside an
outer capture (that failed before). Not capturable yet: the longitudinal and transverse wakes
(their bin window is read from the data), and `csr_drift_model="retarded"`, which reads the
reference energy on the host and raises a `RuntimeError` during capture instead of invalidating
it. The Warp kernels were not exercised under capture.

### Scan settings as tensors

Quadrupole and sextupole strengths, corrector kicks and RF voltage and phase are fixed when a
`LinacTracker` is built. `tracker.with_settings(k1={"Q1": k}, voltage_V={"L1": v},
phase_rad={"L1": phi})` returns a copy in which the named elements (case-insensitive, every
occurrence) take those values. Floats give the same result as rebuilding the tracker; tensors
keep their autograd history and are read on every call. In a captured graph that means a scan
needs one capture: create the settings as static tensors, capture, then `copy_` each new value
into them before `graph.replay()`.

```python
k = torch.tensor(2.0, device="cuda", dtype=torch.float64)
scan = tracker.with_settings(k1={"Q1": k})
# ... warm up, capture `scan.track(static_beam)` + loss + backward ...
for value in settings:
    k.fill_(value)
    graph.replay()
```

An override replaces the value including any sampled error. The phase is the element's phase as
in `ElementSpec`; the tracker's fiducial shift is applied on top. The reference offset of a bend
with a tensor `k1` or `k2` stays at its build-time value, which is zero for a straight magnet.

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
The Cartesian box is fitted to the live particle extent with a fractional
margin `box_padding_3d` (default 0.05 on each side). At fixed `n_xyz`, a larger
margin also increases the cell sizes; a 1 nm absolute pad is added on each side.
This setting is independent of the cylindrical
`box_margin` setting.

`smooth_r` and `smooth_z` add binomial smoothing to the deposited charge
grid. Leave them at zero unless deposition noise is part of the problem being
studied. `every_step_until` and `subcycle_steps` control how often the space-
charge field is recomputed.

## Time stepping and compilation

`dt_adaptive=True` derives the injector step after emission from the local
space-charge frequency and clamps it between `dt_floor_s` and `dt_ceiling_s`. Set
it to `False` for a fixed-step comparison. The `dt_safety` setting is the main
accuracy/performance knob. The emission phase has its own step,
`min(Injector.dt_s, emission window / emission_steps)`, because the first push of a
macroparticle is quantised to it; see the `SpaceChargeSettings` page.

The defaults (`dt_safety=3.5e5`, `dt_floor_s=15 fs`, `dt_ceiling_s=2 ps`,
`subcycle_steps=20`) are the measured accuracy/time optimum on the CLARA decks at
20, 100 and 250 pC (`Stories/CLARA_InjectorConvergence`): the final emittance is
within 2.4 % of the converged value at all three charges, in 0.72-0.83 of the
tracking-loop time of the previous settings (the process wall time, which also
holds start-up, compile-cache load and beam generation, falls by 12-21 %). Where the step matters is the late gun and the drift to the
linac (mean Lorentz factor 4 to 10.5); halving the step elsewhere changes nothing.
Raising `dt_safety` to 1e6 puts the 250 pC emittance 13.5 % above the converged value.

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

Offsets and tilts of placed solenoids and quadrupoles (`magnets.place_as_tensors`) and the screen plane are 0-d tensors
too, for the same reason: a scan or a fit over a solenoid offset or tilt reuses the compiled step. As floats each new
value recompiled (10 to 25 s) and, after the recompile limit, ran eager. A 24-phase trial of the CLARA gun with a
displaced and tilted solenoid takes 38 s (65 to 150 s before), and changing the solenoid or gun field, the flatness, the
offset or the tilt compiles nothing new; only switching between a placed and an unplaced solenoid does, once.
Placed *apertures* still take their offset and rotation as floats.

For a scan that needs only the charge, `dt_adaptive=True` with a larger
`dt_safety` (5e7 against the 3.5e5 default) and `dt_ceiling_s=2e-12` keeps the
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
