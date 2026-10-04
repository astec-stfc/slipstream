# `slipstream.analysis.emittance`

[Source](../../../src/slipstream/analysis/emittance.py) · [Documentation index](../../README.md)

## Physics

RMS helpers combine unbiased position/momentum variances with the mean cross product. For centered transverse arrays this is the RMS phase-space-area estimator. The longitudinal helper explicitly centers position and kinetic energy before evaluating its area in keV*mm.

### Limits and conventions

- `geometric_emittance_xy` retains its historical slopes-first argument order.
- Canonical momentum yields canonical emittance; slopes yield geometric emittance.
- Center transverse position/slope arrays before rms_emittance or geometric_emittance_xy: their cross product is not centered internally.

## Usage

Call rms_emittance(u, up) for one plane, geometric_emittance_xy for centered transverse position/slopes and longitudinal_emittance_keV_mm for longitudinal position/tau in metres and kinetic energy in eV.

### API reference

```python
def rms_emittance(u: torch.Tensor, up: torch.Tensor) -> torch.Tensor: ...
def geometric_emittance_xy(xp: torch.Tensor, yp: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...
def longitudinal_emittance_keV_mm(z_or_tau_m: torch.Tensor, ekin_eV: torch.Tensor) -> torch.Tensor: ...
```

### `longitudinal_emittance_keV_mm`

Normalized longitudinal emittance in ASTRA's Zemit convention
(manual sec. 4.13): the real RMS phase-space area
`sqrt(<dz^2><dE^2> - <dz dE>^2)` over centered `(z, Ekin)` -- the
correlation term removes an RF-accelerated bunch's energy chirp.
Moved from `injector._longitudinal_emittance_keV_mm` unchanged.
