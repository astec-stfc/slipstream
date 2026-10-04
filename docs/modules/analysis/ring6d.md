# `slipstream.analysis.ring6d`

[Source](../../../src/slipstream/analysis/ring6d.py) · [Documentation index](../../README.md)

## Physics

RF-on survival and tune diagnostics advance the real `NativeTracker`, including configured longitudinal motion, damping and excitation. Betatron carrier extraction subtracts linear dispersion; recorded state retains all six canonical coordinates.

### Limits and conventions

- The tracker elapsed-turn and RNG states advance normally; use independent trackers for independent scans.
- Each FMA half needs adequate synchrotron periods; configured excitation contributes to measured diffusion.
- These are single-particle survival diagnostics rather than a self-consistent collective aperture calculation.

## Usage

Use `track_ring6d`, `betatron_tunes6d`, `momentum_aperture6d`, `dynamic_aperture6d` and `frequency_map6d`. Provide a synchronous six-coordinate launch for acceptance studies and suitable sideband-separated tune bands.

### API reference

```python
class Ring6DResult:
    ...
def track_ring6d(tracker: NativeTracker, state: torch.Tensor, n_turns: int, aperture: tuple[float, float]=(0.05, 0.05), delta_limit: float | None=None, record_every: int=1) -> Ring6DResult: ...
def betatron_tunes6d(result: Ring6DResult, optics: LinearOptics, synchrotron_tune: float, half_width: float | None=None, dispersion: tuple[float, float, float, float]=(0.0, 0.0, 0.0, 0.0)) -> torch.Tensor: ...
class MomentumScan:
    ...
def momentum_aperture6d(tracker: NativeTracker, launch: torch.Tensor, delta: torch.Tensor, n_turns: int, aperture: tuple[float, float]=(0.05, 0.05), delta_limit: float | None=None) -> MomentumScan: ...
def dynamic_aperture6d(tracker: NativeTracker, optics: LinearOptics, r_max: float, n_turns: int, n_angles: int=19, n_r: int=32, refine: int=2, aperture: tuple[float, float]=(0.05, 0.05), delta_limit: float | None=None) -> DynamicAperture: ...
def frequency_map6d(tracker: NativeTracker, optics: LinearOptics, synchrotron_tune: float, x_range: tuple[float, float], y_range: tuple[float, float], nx: int=41, ny: int=21, n_turns: int=2048, aperture: tuple[float, float]=(0.05, 0.05), half_width: float | None=None, dispersion: tuple[float, float, float, float]=(0.0, 0.0, 0.0, 0.0)) -> FrequencyMap: ...
```

### `track_ring6d`

Track RF-on turns, recording all six coordinates and first losses.

Apertures are checked at the ring entrance after each turn, with optional
`|delta|` limit. Lost slots are frozen at zero. This is a single-particle
survival calculation: collective wakes/current and IBS are unsupported
because frozen losses must not contribute to collective moments.

### `betatron_tunes6d`

Return `(..., 2)` carrier tunes, NaN for lost launches.

Subtract linear dispersion times the instantaneous momentum deviation.
Restrict each NAFF seed to a band narrower than half the synchrotron
sideband spacing. The supplied optics must estimate the carrier within
that band; a large detuning requires updated centres. This does not
identify overlapping or unresolved sidebands automatically. At least
four synchrotron periods must be recorded to resolve the carrier.

### `momentum_aperture6d`

Batched sampled momentum acceptance at a supplied RF synchronous orbit.

`launch` is a single six-coordinate orbit or `(n_delta, 6)` matched
launches. `delta` gives absolute initial momentum deviations. Return
the survival of every sample rather than assuming acceptance is connected
or interpolating across unstable islands. A finite scan bounds acceptance
only over its tested range and turn count.

### `dynamic_aperture6d`

RF-on DA using the same batched ray refinement as frozen-energy DA.

Supply optics whose `closed_orbit` is the six-dimensional synchronous
launch. Refinement advances the tracker RNG; deterministic comparison
requires zero excitation. Damping can change the measured survival.

### `frequency_map6d`

RF-on carrier tunes and two-half Laskar diffusion on a launch grid.

Each half needs four synchrotron periods. Diffusion includes configured
radiation damping/excitation; use a deterministic tracker for a purely
Hamiltonian study. `optics.closed_orbit` is a supplied synchronous orbit.
