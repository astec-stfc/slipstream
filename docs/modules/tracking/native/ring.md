# `slipstream.tracking.native.ring`

[Source](../../../../src/slipstream/tracking/native/ring.py) · [Documentation index](../../../README.md)

## Physics

Provides frozen-energy one-turn nonlinear transport with fixed delta and no RF, radiation damping/excitation or wakes. Tensor strengths permit gradients and batched machine ensembles. Exact-dipole splitting improves float32 bend conditioning.

### Limits and conventions

- Converge slices/order before interpreting DA or FMA.
- Ordinary `track` aperture checks occur at the ring entrance per turn; local loss-map checks provide a different sampling boundary.
- Lost particles keep batch slots and are zeroed; loss turns identify them.
- Use `NativeTracker` for RF-on dynamics.

### `Model and numerical conventions`

**Bends use the exact-dipole splitting by default** (`exact_dipole=True`,
see `elements._exact_dipole_drift`): same Hamiltonian as elegant's
splitting, converges to the same map, but float32-robust -- float32 DA on
br14 is within 1% of elegant with it, 8% off without. Pass
`exact_dipole=False` to integrate elegant's own map (live-elegant
comparisons at equal slicing).

### `RingMap.__init__`

`double_single=True`: every coordinate is a float32 `hi + lo`
pair (`double_single.py`, ~44-bit mantissa) -- float64-grade FMA
on a consumer GPU. Needs `exact_dipole=True`, no misalignments,
Python-float (or 0-d tensor) strengths; no autograd, `tau` not
tracked. Optics (`analysis.nonlinear.linear_optics`) are then
computed on `float64_reference()`. `double_single="hybrid"`:
only the linear, constant-coefficient part of each update is
double-single; small nonlinear remainders are plain float32
(`double_single._hy_*`) -- cheaper, ~1e-12..1e-10 per element
instead of ~1e-15.

## Per-kind kick counts

`n_kicks` applies to every non-bend element and `n_kicks_bend` to bends.
`n_kicks_quad` and `n_kicks_sext` (default `None`, meaning `n_kicks`)
override the count for `kind="quad"` and `kind="sext"` elements, e.g.
`RingMap(specs, E, n_kicks=2, n_kicks_quad=8, n_kicks_sext=1)`. Quadrupole
kicks set the tune accuracy; extra sextupole kicks mostly add cost, so
many quadrupole kicks with a single sextupole kick gave the same tune error
1.4x faster in float64 and 2.2x faster in float32 (1.5x / 2.2x at 50 000
particles) in a separate study. The split is kept by `float64_reference()`;
`periodic_twiss_table` takes the same two keywords.

## Usage

Construct `RingMap(specs, energy_eV, ...)`, use `with_strengths`/`with_alignments`, `turn`, tapped turns and `track`. `track_loss_map` localizes losses; `float64_reference` creates an accuracy reference.

### API reference

```python
class RingTrackResult:
    ...
class RingLossMap:
    ...
class RingMap:
    ...
    def __init__(self, specs: list[ElementSpec], energy_eV: float, mc2_eV: float=ELECTRON_MASS_EV, n_kicks: int=2, n_kicks_bend: int=8, integration_order: int=6, device: str | torch.device | None=None, dtype: torch.dtype | None=None, compile: bool=False, exact_dipole: bool=True, double_single: bool | str=False, n_kicks_quad: int | None=None, n_kicks_sext: int | None=None): ...
    def circumference_m(self) -> float: ...
    def with_strengths(self, k1: dict[str, torch.Tensor] | None=None, k2: dict[str, torch.Tensor] | None=None, kick_x: dict[str, torch.Tensor] | None=None, kick_y: dict[str, torch.Tensor] | None=None, dx: dict[str, torch.Tensor] | None=None, dy: dict[str, torch.Tensor] | None=None, tilt: dict[str, torch.Tensor] | None=None) -> 'RingMap': ...
    def float64_reference(self) -> 'RingMap': ...
    def with_alignments(self, dx: dict[str, torch.Tensor] | None=None, dy: dict[str, torch.Tensor] | None=None, tilt: dict[str, torch.Tensor] | None=None) -> 'RingMap': ...
    def turn(self, state: torch.Tensor, eager: bool=False) -> torch.Tensor: ...
    def turn_with_taps(self, state: torch.Tensor, tap_indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]: ...
    def track_loss_map(self, state: torch.Tensor, n_turns: int, aperture: tuple[float, float]=(0.05, 0.05), element_apertures: dict[str, tuple[float, float]] | None=None, record_every: int=0) -> 'RingLossMap': ...
    def track(self, state: torch.Tensor, n_turns: int, aperture: tuple[float, float]=(0.05, 0.05), record_every: int=0, checkpoint: bool=False) -> RingTrackResult: ...
```

### `RingMap.with_strengths`

Copy of this map with every occurrence of each named element
family (case-insensitive) using the given tensor `K1`/`K2`/
`kick_x`/`kick_y`/`dx`/`dy`/`tilt`. The per-element
reference-orbit offset stays at its nominal value (a
discretization correction; its strength dependence is ~1e-12,
and it is deliberately computed at the design axis regardless of
misalignment -- see `tracker._ElementParams`'s own docstring).
`kick_x`/`kick_y` only apply to `kind="hkick"`/`"vkick"`
elements respectively -- naming an H-plane corrector in
`kick_y` (or vice versa) is a no-op, not an error (mirrors
`k1`/`k2` naming a non-quad/sext family). `dx`/`dy`/
`tilt` apply to any element (LOCO's own alignment fit, ).

### `RingMap.with_alignments`

Copy with absolute family offsets (metres) and rolls (radians).

Values broadcast over particle batch dimensions: `(M, 1)` errors
with `(M, N, 6)` states represent M machines. Tensor values retain
autograd. Alignment ensembles use the PyTorch path. Launch around
each machine's own closed orbit when estimating its dynamic aperture.

### `RingMap.turn`

One turn, no loss check -- differentiable. `eager=True`
bypasses `compile` (Jacobians via `vmap` go this way). A
`double_single` map takes and returns the `(..., 12)` hi/lo
state.

### `RingMap.turn_with_taps`

Eager per-element loop (same `_apply_one_element` as
`turn(eager=True)`, always -- no `compile`/Warp/CUDA-graph/
`double_single` path) that also returns the state at each of
`tap_indices` (0-based positions into `self.params`/`self.specs`,
resolved by name in the LOCO layer -- see ``analysis.loco.
orbit_response_matrix``), stacked in the given order (shape
`(..., n_taps, 6)`).

A tap is the state **exiting** that element (after its own
transport) -- for a BPM (a finite-length `MONI`, parsed as a
plain `kind="drift"` spec, 

Duplicate indices are allowed (unused here, but harmless); every
entry of `tap_indices` must be a valid position. `self.turn()`
pays no overhead from this method's existence -- it is a wholly
separate loop.

### `RingMap.track_loss_map`

Check losses at every element exit; first loss location is an element index.

Apertures are rectangular lab-frame half widths in metres. Named overrides
apply to every occurrence. This path uses PyTorch element transport, with
fixed particle slots; recording decimates observations at the ring start.
Double-single transport is currently unsupported on this optional path.

### `RingMap.track`

`state` `(..., 6)`. `record_every=k>0` stores ``(x, px, y,
py)` after turns `k-1, 2k-1,...` (memory: `n_turns/k * N * 4``
numbers -- decimate or track a subset for 1e6-particle runs).
`checkpoint=True` recomputes each turn in the backward pass
(`torch.utils.checkpoint`) -- memory O(1 turn) for autograd
through many turns.

On CUDA, when nothing needs a gradient, one turn (every element
plus the loss bookkeeping) is captured once into a CUDA graph and
replayed -- the per-turn cost is then pure GPU time, no Python/
dynamo-guard overhead per element.

`double_single`: `state` is converted to hi/lo from float64;
`tbt` and the returned state are float64 (`hi + lo`).

## Numerical and implementation notes

### `RingMap.__init__`

Per-element compile, not whole-turn: an unrolled ~5000-stage turn
graph was still compiling after 15 min / 10 GB (same failure as
LinacTracker's whole-pass compile). One graph per distinct element
config fuses each element into a few kernels -- further reduced to
one graph per element *family* below (K1/K2/length regionalization).

### `RingMap.__init__`

Triton contracts a*b+c into FMA by default, which breaks the
Dekker split inside double-single products (measured: 1e-6
instead of 1e-13); emulate_precision_casts turns FP fusion off
for these kernels only.

### `RingMap.__init__`

Regional compilation -- promote the numeric fields that
vary per *instance* of a family (K1/K2, a drift's length)
to 0-d tensors, so dynamo guards on shape/dtype/device
instead of value and every same-family element reuses one
graph (27 -> 7 graphs measured on br14_simple; full
rationale + numbers in docs/tracking_native.md's
"Compilation-time discipline"). Left as plain floats
wherever a *structural* check downstream reads the
original value -- K1/K2 exactly zero (keeps
track_element's `isinstance`-gated fast paths, elements.py)
and length for angle!=0 or non-positive length (the `h =
angle/length` divide and the `length <= 0.0` early-out
must resolve in plain Python, never on a live tensor).

### `RingMap.track`

Flat batch marked dynamic: the compiled slices otherwise
specialise on particle count, and one double-single recompile
of the whole ring costs ~18 min.

### `RingMap.track`

loss checked once per turn at s=0, not per element --
a particle past the DA diverges within a few turns anyway, so
the loss *turn* can be late by O(1) turns; add a per-element
check if loss maps (where, not when) are ever needed.
