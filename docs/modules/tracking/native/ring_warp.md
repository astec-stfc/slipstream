# `slipstream.tracking.native.ring_warp`

[Source](../../../../src/slipstream/tracking/native/ring_warp.py) · [Documentation index](../../../README.md)

## Physics

Implements selected fixed-lattice one-turn ring transport using a packed Warp kernel. It mirrors canonical/exact-dipole drifts, supported multipoles, edge maps and alignment transforms.

### Limits and conventions

- This is a CUDA forward-only optimization; tensor-gradient strengths and unsupported physics use PyTorch fallback.
- Packing returns `None` for unsupported maps, including RF.
- Field-free elements (drifts, zero-strength multipoles and exact sector bends
  without K1/K2) take one exact step, as `elements.track_element` does; only
  elements with kicks are sliced into `n_kicks` integrator stages.
- Edge maps use the chromatic curvature `h/(1+delta)`, matching
  `elements._edge_kick`. Earlier versions used `h`, which put off-momentum
  particles at bends with nonzero edge angles about 1e-6 off.
- One particle's turn is a serial dependency chain, so the stage loop uses
  precomputed reciprocals rather than IEEE divisions. Warp `fast_math` was
  measured only about 15% faster again and is not enabled.
- RTX 3050, MLS example, float32 `RingMap.track`: 0.41, 0.57 and 5.4 ms per
  turn at 1e2, 1e4 and 1e5 particles. Compiled per-element kernels replayed in a
  CUDA graph took 0.60, 1.54 and 14.4 ms. Before the two changes above, this
  kernel took 3.6 ms at 1e2 particles and 25 ms at 1e5 particles.

### `Model and numerical conventions`

This first slice accepts canonical/exact-dipole drift and multipole kicks,
first- and second-order edge maps, and the existing flat-space misalignment
transform. RF remains on the PyTorch fallback. One launch advances every
particle through every packed element; it never crosses the PyTorch/Warp
boundary per element.

## Usage

`pack` prepares scalar lattice parameters; `available`, `prepare` and `turn` support the `RingMap` dispatch. Ordinary users select the tracker backend rather than calling kernels directly.

### API reference

```python
class PackedRing:
    ...
class PreparedTurn:
    ...
def pack(params, device: torch.device, dtype: torch.dtype) -> PackedRing | None: ...
def available(device: torch.device, dtype: torch.dtype, packed: PackedRing | None) -> bool: ...
def prepare(state: torch.Tensor, packed: PackedRing) -> PreparedTurn: ...
def turn(state: torch.Tensor, packed: PackedRing, energy: float, mc2: float, *, out: torch.Tensor | None=None, prepared: PreparedTurn | None=None) -> torch.Tensor: ...
```

## Numerical and implementation notes

### `Module constants`

Match elements.py's grouped reference-beta correction;
left-to-right evaluation accumulates a visible float32
tau residual over long lattices such as BR14.
