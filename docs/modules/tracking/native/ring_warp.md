# `slipstream.tracking.native.ring_warp`

[Source](../../../../src/slipstream/tracking/native/ring_warp.py) · [Documentation index](../../../README.md)

## Physics

Implements selected fixed-lattice one-turn ring transport using a packed Warp kernel. It mirrors canonical/exact-dipole drifts, supported multipoles, edge maps and alignment transforms.

### Limits and conventions

- This is a CUDA forward-only optimization; tensor-gradient strengths and unsupported physics use PyTorch fallback.
- Packing returns `None` for unsupported maps, including RF.
- `RingMap` uses this kernel only for eager (`compile=False`) CUDA maps. With
  `compile=True`, compiled per-element kernels replayed in a CUDA graph were
  faster at 1e2-1e5 particles on an RTX 3050 (MLS example: 0.8 vs 3.6 ms per
  turn at 100 particles, 15 vs 25 ms at 1e5).

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
