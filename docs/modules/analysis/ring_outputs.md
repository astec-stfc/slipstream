# `slipstream.analysis.ring_outputs`

[Source](../../../src/slipstream/analysis/ring_outputs.py) · [Documentation index](../../README.md)

## Physics

Reports linear optics along the actual closed orbit and derives machine parameters from configured transport. Dispersion momenta follow canonical px/py; energy is total energy.

### Limits and conventions

- Reporting uses CPU float64 and supports one machine rather than a strength ensemble.
- Phase reconstruction assumes element phase advance below 2*pi; subdivide long focusing elements when the integer tune is ambiguous.
- Joined plot samples do not establish unsampled extrema.

## Usage

Use `ring_optics_table`, `ring_parameters` and `ring_lattice_table` for detached pandas tables. Optional `plot_ring_twiss`, `plot_ring_layout` and `plot_dynamic_aperture` return Matplotlib figures.

### API reference

```python
def ring_optics_table(ring: RingMap, drift_samples: int=8) -> pd.DataFrame: ...
def ring_parameters(ring: RingMap, optics: pd.DataFrame | None=None, charge_e: float=1.0, include_chromaticity: bool=True) -> pd.DataFrame: ...
def ring_lattice_table(ring: RingMap) -> pd.DataFrame: ...
def plot_ring_twiss(optics: pd.DataFrame, lattice: pd.DataFrame | None=None): ...
def plot_ring_layout(lattice: pd.DataFrame): ...
def plot_dynamic_aperture(result): ...
```

### `ring_optics_table`

Entrance optics, drift interiors and an END row for an uncoupled ring.

Uses CPU float64 with the original slicing, offsets and strengths.
Rejects coupling through `linear_optics`. Dispersion is d(co)/d(delta).
Phase accumulation assumes each element's phase advance is below 2*pi;
subdivide very long focusing elements if an integer tune is ambiguous.
`drift_samples` adds equally spaced interior points in plain drifts,
showing straight-section waists; zero gives element boundaries only.

### `ring_parameters`

Derived machine parameters as columns `parameter, value, unit`.

Energy is total energy; rigidity uses the magnitude of charge in units of e.
Slip factor includes the closed-orbit dispersion, and momentum compaction
follows eta = alpha_c - 1/gamma**2 for relative momentum delta. No RF,
damping or equilibrium-emittance values are inferred from a frozen map.
Supply a table from the same ring to avoid recomputing its optics.

### `ring_lattice_table`

Configured element names, positions, geometry and K1/K2/K3 strengths.

For one machine only; batch strength/alignment ensembles raise an error.
Higher/skew tuples are retained as tuple columns. All values are SI.

### `plot_ring_twiss`

Return a Matplotlib Figure with beta, alpha, dispersion and optional layout.

Matplotlib is imported only here; install `slipstream-accel[plot]`.
Samples are joined by lines; extrema between samples are not inferred.
The caller owns saving, display and closing the figure.
