# Ring reference outputs

`ring_optics_table`, `ring_parameters` and `ring_lattice_table` are available
from `slipstream`. They return detached pandas tables for reporting. Optics
uses CPU float64 and the actual `RingMap` element maps, slicing, closed orbit
and strength/alignment overrides. It does not substitute an ideal matrix
lattice for the configured transport.

```python
from slipstream import ring_optics_table, ring_parameters, ring_lattice_table
from slipstream import plot_ring_twiss, plot_ring_layout

optics = ring_optics_table(ring)
parameters = ring_parameters(ring, optics)
lattice = ring_lattice_table(ring)
print(parameters.to_string(index=False))
optics.to_csv("optics.csv", index=False)
fig = plot_ring_twiss(optics, lattice)
fig.savefig("twiss.png", dpi=160)
```

Install `slipstream-accel[plot]` for Matplotlib; table generation requires only
the core dependencies. Plotting helpers return Figures without showing,
saving or changing the backend. The caller closes the figures when done.
`plot_ring_layout(lattice)` makes a standalone magnet layout;
`plot_dynamic_aperture(result)` plots one machine's sampled launch boundary
in millimetres, without reflecting an upper-half-plane scan into a full ring.

The optics table has entrance and END rows, beta, alpha, dispersion, closed
orbit and accumulated phase. By default eight interior points per plain
drift show straight-section waists. `drift_samples=0` yields boundaries only.
Magnet interiors are not sampled; maxima are sampled maxima. Dispersion
columns `D_px`/`D_py` differentiate normalized canonical momenta with respect
to relative momentum deviation; they are not generally slope dispersions.
Phase-derived integer tunes assume less than 2*pi advance per element.
Unstable or coupled optics, including locally coupled elements whose net
one-turn coupling cancels, are rejected rather than represented by misleading
per-plane Twiss functions. Report strength ensembles one machine at a time.

The machine table includes total/kinetic energy, momentum, relativistic
factors, circumference, revolution frequency/period, magnetic rigidity,
fractional and accumulated tunes, chromaticity, slip factor, momentum
compaction, bending angle, element count and sampled optics maxima.
Rigidity assumes unit charge magnitude by default; pass `charge_e` for other
charge states. Slip factor includes the closed-orbit dispersion contribution
to time of flight: eta = beta_rel * d(tau)/d(delta) / circumference.
For one superperiod of a symmetric ring, pass `periods=N`; whole-machine
values are then reported, with an explicit `periods` row.
Momentum compaction adds 1/gamma². A supplied optics table must come from
the same ring and retain its DataFrame attributes. CSV export is for display
and downstream analysis, not reconstruction of the cached one-turn matrix.
RF, damping, energy loss and equilibrium emittance are not inferred from
this frozen-energy map.

The MLS-inspired nonlinear example writes `optics.csv`, `lattice.csv`,
`machine_parameters.csv`/`.txt`, `twiss.png` and `layout.png` to
`.cache/nonlinear_optimisation` by default. Use `--output-dir` to choose a
location, `--no-plots` for tables alone, or `--reference-only` to generate
these outputs without optimisation. A normal optimisation run also writes
the nominal finite-turn DA table and plot; this reference scan is separate
from the optimisation evaluation budgets. All outputs describe the example's
illustrative model, not measured MLS machine parameters.
