# `slipstream.tracking.native.double_single`

[Source](../../../../src/slipstream/tracking/native/double_single.py) · [Documentation index](../../../README.md)

## Physics

Represents selected frozen-energy ring coordinates as float32 high/low pairs for increased precision. Error-free TwoSum/TwoProd transforms underpin drift, exact-sector-dipole, K1/K2 and first-order-edge transport.

### Limits and conventions

- This path excludes misalignments, tensor strengths and evolving tau.
- IEEE rounding and absence of harmful reassociation/FMA contraction are required.
- Keep the `|a| >= |b|` precondition adjacent to FastTwoSum.

### `Model and numerical conventions`

Scope: the `exact_dipole` integrator of `elements.track_element` --
straight drifts, exact sector dipoles, K1/K2 kicks, linear (`EDGE_ORDER=1`)
edges with optional `HGAP`/`FINT`. No misalignments, no tensor
strengths, no `tau` (frozen-energy tracking never reads it). Transcribed
from `elements.py` operation for operation; `tests/test_ring_nonlinear.py`
checks it against float64 `track_element` to ~1e-13.

## Usage

Use `DS`, `track_element_ds`, `ds_edge` and `ds_slice` through the corresponding `RingMap` precision path. `check_error_free` verifies the actual execution backend.

### API reference

```python
class DS:
    ...
    def of(x: torch.Tensor) -> 'DS': ...
    def double(self) -> torch.Tensor: ...
def ds_sqrt(x: DS) -> DS: ...
def ds_asin_small(u: DS) -> DS: ...
def ds_tan_small(u: DS) -> DS: ...
def ds_edge(s12: torch.Tensor, h: float, e: float, hgap: float, fint: float) -> torch.Tensor: ...
def ds_slice(s12: torch.Tensor, dl: float, h: float, k1: float, k2: float, drift_frac: tuple, kick_frac: tuple, hybrid: bool=False) -> torch.Tensor: ...
def track_element_ds(s12: torch.Tensor, length: float, angle: float, k1: float, k2: float, e1: float, e2: float, n_kicks: int, drift_frac: tuple, kick_frac: tuple, hgap: float=0.0, fint: float=0.5, edge_order: int=1, edge1_effects: int=1, edge2_effects: int=1, slice_fn=ds_slice, edge_fn=ds_edge) -> torch.Tensor: ...
def check_error_free(fn=None, device: str | torch.device='cpu') -> float: ...
```

### `ds_slice`

One Yoshida slice of length `dl` (exact dipole / straight drift +
K1/K2 kicks) on a `(..., 12)` hi/lo state. The unit
`RingMap(double_single=True, compile=True)` compiles: a whole
8-slice bend as one graph was still compiling after 20+ min.

### `check_error_free`

Max relative error of `DS` add/mul/div/sqrt vs float64 on random
data, through `fn` (e.g. `torch.compile(_ds_ops,...)`) if given -- ~1e-13 when
the backend preserves the error-free transforms, ~1e-7 when it does
not (reassociation / FMA contraction inside a split).

## Numerical and implementation notes

### `Module constants`

hybrid: double-single linear map + float32 nonlinear remainder
Every update is split into its constant-coefficient linear part (DS x
Python-float constant, full ~44-bit precision) and a nonlinear remainder
(p_perp^2, K2*x^2, delta*p terms) that is 1e-3..1e-8 of the coordinate, so
evaluating it in float32 from the hi parts costs only eps*(its size).
pz - 1 - delta = -p_perp^2/(pz + 1 + delta) keeps that remainder
cancellation-free.
