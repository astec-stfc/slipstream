# Higher multipoles, coupled optics and ring loss maps

`RingMap` is available from `slipstream`. Its frozen-energy transport uses
canonical `(x, px, y, py, tau, delta)` coordinates. All strengths below use
SI units and preserve the selected tensor device and dtype.

`ElementSpec.k3` specifies a distributed normal octupole strength in m⁻⁴.
`multipoles` and `skew_multipoles` are tuples of `(order, strength)` pairs,
where order 0 is a dipole, 1 a quadrupole, 2 a sextupole and 3 an octupole.
The field convention is

```
Fy + i Fx = sum((Kn + i Ks) * (x + i y)**n / n!)
Delta px = -ds * Fy; Delta py = ds * Fx
```

The existing bend factor `(1 + h*x)` also multiplies these kicks. Tuple
terms add to `k1`, `k2` and `k3`; avoid specifying the same physical field
twice. Strengths may be tensors, including tensors requiring gradients.
The loop runs over multipole orders, while particle operations stay batched.

The elegant reader supports `KOCT,L=...,K3=...` and `MULT,L=...,ORDER=...,KNL=...`.
For positive-length MULT elements, `KNL/L` is the distributed strength;
zero-length MULT elements produce a thin integrated kick. `FACTOR`, `DX`,
`DY` and `TILT` are supported. Pole-tip (`BORE`/`BTIPL`), `DZ` and
`SYNCH_RAD` options raise explicit errors. See the
[official MULT definition](https://ops.aps.anl.gov/manuals/elegant_latest/elegantsu213.html)
for its integrated-strength convention. Radiation and fringe physics for
higher multipoles are not modeled. Double-single transport rejects higher
and skew multipoles; the Warp ring path falls back to PyTorch for them.

For coupling, use `analysis.nonlinear.coupled_optics(ring)` and
`coupled_signals(result.tbt, optics)`. The optics result contains two
fractional normal-mode tunes and a symplectic normalizer mapping mode
coordinates to laboratory coordinates. Modes are sorted by tune; mode
labels can switch at tune crossings. `create_graph=True` retains strength
gradients, but degenerate eigenvalues make eigenvector derivatives
ill-defined. The ordinary `linear_optics` API rejects coupled maps instead
of returning misleading per-plane Twiss parameters. Normal-mode optics
here require a stable symplectic 4D map at fixed momentum; they do not
describe RF-on longitudinal coupling or radiation damping.

`ring.track_loss_map(state, n_turns, element_apertures={"COL": (0.01, 0.005)})`
checks rectangular laboratory-frame apertures and nonfinite coordinates
at every element exit. Its `RingLossMap` reports the first loss turn,
zero-based element index and exit position `loss_s_m`; survivors have
indices `-1` and position NaN. Named limits apply to every occurrence;
the global `aperture` supplies the default. Lost particle slots stay in
the fixed batch and their state is zeroed. These are exit checks rather
than continuous checks inside an element. Loss decisions are discrete;
this API is intended for diagnostics, not gradients through lost tracks.

Recordings use `floor(n_turns/record_every) * N * 4 * itemsize` bytes.
One million float64 six-coordinate particles need 48 MB for the state
alone; choose `record_every=0` to omit the turn history, or decimate it.
Workspace, temporary tensors and autograd graphs add to that number.
The package tests exercise one million particles over two drift turns
without a history; this demonstrates fixed batch shape, not a large-ring
performance benchmark.

Validation uses independent complex-polynomial kick references, analytic
quadrupole matrices, symplectic Jacobians and similarity transforms,
normal-mode tracked tunes, finite-difference strength gradients, aperture
geometry, and CPU/CUDA eager multipole parity. No live elegant executable
is needed or exercised by these tests.
