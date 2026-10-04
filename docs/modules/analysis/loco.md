# `slipstream.analysis.loco`

[Source](../../../src/slipstream/analysis/loco.py) · [Documentation index](../../README.md)

## Physics

Differentiates closed-orbit BPM readings with respect to corrector kick angles. The ORM has units metres per radian; BPM gain/roll and corrector gains transform its rows/columns.

### Limits and conventions

- Named corrector channels are evaluated at zero kick; their existing nonzero strengths are overridden.
- BPM offsets cancel from response derivatives.
- Overall BPM and corrector gain scales are degenerate; a fit needs an anchored gain or prior.

## Usage

Call `orbit_response_matrix(ring, corrector_names, bpm_names, delta=0, create_graph=False)` for shape `(n_bpm,2,n_corrector)`. Use occurrence-safe names and `apply_calibration` for device calibration.

### API reference

```python
def orbit_response_matrix(ring: RingMap, corrector_names: list[str], bpm_names: list[str], delta: float=0.0, create_graph: bool=False) -> torch.Tensor: ...
def apply_calibration(J: torch.Tensor, bpm_gain_x: torch.Tensor, bpm_gain_y: torch.Tensor, bpm_roll: torch.Tensor, corrector_gain: torch.Tensor) -> torch.Tensor: ...
```

### `orbit_response_matrix`

`(n_bpm, 2, n_corrector)` exact ORM: the Jacobian of tapped
`(x, y)` closed-orbit readings at every named BPM with respect to
one scalar kick-angle variable per named corrector channel, evaluated
at the zero-kick operating point (`ring`'s own corrector strengths
are overridden to zero-valued *differentiable* tensors here -- any
nonzero kick already baked into `ring` before calling this is
discarded for the corrector columns, not added to).

Each `corrector_names` entry must name a single H- or V-plane
channel (an `ElementSpec(kind="hkick")`/`"vkick"` element) --
the plane is read from the lattice itself, so there is no separate
plane argument. Both `corrector_names` and `bpm_names` should
already be occurrence-safe (`analysis.nonlinear_opt.unique_names`
or a pre-renamed `name#occurrence`) -- a bare name repeated in the
real lattice would silently collapse every occurrence onto one
column/row.

`create_graph=True` keeps the returned ORM differentiable in any
other tensor strength `ring` already carries (e.g. K1, via
`ring.with_strengths(k1=...)` before calling this) -- the nested
Jacobian a LOCO fit needs. `delta` is a fixed off-momentum point
This API returns orbit response, not a separate dispersion observable.

### `apply_calibration`

The measured-orbit-response-matrix model Safranek (1997) fits
alongside the lattice: `J` (`(n_bpm, 2, n_corrector)`, the model
ORM from `orbit_response_matrix`) with a per-BPM 2x2 gain/rotation
block applied to its rows and a per-corrector scalar gain applied to
its columns -- ``measured = diag(gx_i, gy_i) @ Rot(roll_i) @ true *
corrector_gain_j`` (a BPM roll rotates the true orbit into the
tilted device frame *before* the device's own x/y gains scale it;
a corrector gain scales the whole physical kick for one channel, so
it multiplies that channel's entire ORM column).

`bpm_gain_x`/`bpm_gain_y`/`bpm_roll` are `(n_bpm,)`,
`corrector_gain` is `(n_corrector,)` -- all plain multiplicative
tensors (broadcast/autograd-friendly, no solver logic here).

**Gauge**: an overall BPM-gain scale and an overall corrector-gain
scale are degenerate (multiplying every `bpm_gain_*` by `s` and
dividing every `corrector_gain` by `s` leaves `J` unchanged) --
a fit must fix at least one gain (e.g. hold one `corrector_gain`
entry at exactly `1.0`, not a free parameter) or supply a measured
prior; this function does not enforce that itself.

BPM *offsets* (a constant, metres-valued shift) do not belong here --
an ORM is a derivative/difference (metres per radian) and a BPM
offset cancels from it identically;

## Numerical and implementation notes

### `orbit_response_matrix`

At the zero-kick operating point, differentiate the one-turn and
tapped maps separately and apply the implicit-function theorem. This is
algebraically the same derivative as differentiating `closed_orbit`
inside `observe`, but avoids a Jacobian-of-a-Jacobian for every ORM
call on a real lattice.

### `orbit_response_matrix`

Forward-mode keeps the full tapped tape out of a reverse-mode graph.
On the real booster, vectorized reverse-mode retains multiple GB for
this otherwise small dense Jacobian.
