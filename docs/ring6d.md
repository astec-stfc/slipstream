# Alignment ensembles and RF-on ring analysis

`RingMap.with_alignments(dx=..., dy=..., tilt=...)` replaces the absolute
offsets in metres and rolls in radians of named families (case insensitive).
As with `with_strengths`, tensors shaped `(M, 1)` broadcast over native
particle states shaped `(M, N, 6)`. The transforms use tensor trigonometry,
preserve gradients and use the PyTorch transport path. Double-single
tracking does not support alignment errors. Bend alignment retains the
documented flat-space approximation in `elements._misalign`.

Alignment errors move the closed orbit. Dynamic aperture launched from the
nominal orbit measures acceptance of that particular injection, rather than
acceptance around each machine's fixed point. Solve each machine's closed
orbit separately before constructing its launch ensemble; the current
uncoupled `linear_optics` helper is not a coupled normal-mode solver for
rolled magnets; use the coupled transverse optics helper when appropriate.

`track_ring6d(tracker, state, n_turns, aperture=(ax, ay))` runs the same
`NativeTracker` turn used by ordinary tracking, recording all six canonical
coordinates `(x, px, y, py, tau, delta)`. `tau` is in metres. The tracker may
use lumped RF or `rf="per_element"` to place real RF cavities at their
lattice positions. Longitudinal kick and damping functions already live in
`tracking/native/ring_physics.py`; no external pystable installation is used.
The record-free `NativeTracker.advance` path shares normal turn physics but
avoids per-turn host diagnostic reductions. RF analysis requires fixed-energy
tracking; ramped trackers must use the ordinary `track` interface.
The normal per-turn damping remains a ring-average operation, while real
cavity kicks are first-class elements. Per-element cavity RF requires the
correct synchronous phase/reference orbit and the same configured energy
loss balance as normal `NativeTracker` operation.

The helper checks rectangular transverse apertures and optionally
`abs(delta) > delta_limit` after each turn at the ring entrance. Lost slots
keep their indices and are zeroed; `loss_turn` is zero based, with `-1` for
survival. `record_every` decimates records, and zero disables recording;
momentum and DA scans disable records internally to bound memory. Tune
analysis requires undecimated records. Collective wake/current and IBS calculations are rejected because
frozen losses would bias their moments. Damping and quantum excitation retain
the tracker's configured behavior, and calls advance its random-generator and
elapsed-turn state. Use fresh trackers to compare independent scans.

```python
from slipstream import track_ring6d, betatron_tunes6d, momentum_aperture6d

record = track_ring6d(tracker, initial_states, 2048)
# Optics provides an independently estimated betatron carrier near each tune.
Q = betatron_tunes6d(record, optics, synchrotron_tune=0.025,
                    dispersion=(Dx, Dpx, Dy, Dpy))
scan = momentum_aperture6d(fresh_tracker, synchronous_orbit,
                          delta_grid, 2048)
```

`betatron_tunes6d` removes supplied linear dispersion and searches in a
NAFF band narrower than half the synchrotron sideband spacing, selecting the
carrier even when an outside-band sideband is stronger. It requires at least
four synchrotron periods of records. Carrier detuning must fit inside the
specified band; unresolved sidebands and strongly coupled modes require
additional analysis. Lost launches return NaN tunes. This restricted-band
method does not infer the carrier without a prior tune estimate.

`dynamic_aperture6d` applies the frozen-energy helper's batched ray refinement
to real RF-on turns; `frequency_map6d` computes carrier tunes on each half of
a single RF-on record and returns their Laskar diffusion. Each half must
contain at least four synchrotron periods. Configured damping and excitation
contribute to the diffusion; disable excitation for Hamiltonian comparisons.
These helpers require a supplied six-dimensional synchronous launch orbit.
They do not yet solve a six-dimensional RF fixed point automatically.

`momentum_aperture6d` batches absolute initial momentum deviations, returning
every tested value's survival and loss turn. It accepts either one synchronous
launch or individually matched launches. Its result describes sampled
acceptance at that launch position and turn count; it does not interpolate
through unstable islands or promise an untested boundary beyond the scan.
An optional delta limit is a user-supplied cut, not an automatically computed
RF bucket separatrix.

Validation in `tests/test_ring_phase3.py` compares tensor ensembles with
individual machines, alignment derivatives with central differences,
coordinate transforms with their inverse, RF records with normal native
tracking, momentum survival with explicit cuts, and carrier selection with
a synthetic sideband three times stronger than the carrier. CUDA tests
compare eager float64 alignment and RF tracking against CPU eager output;
these do not establish compiled or double-single RF parity.
