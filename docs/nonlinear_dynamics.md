# Native nonlinear ring dynamics

`RingMap` transports canonical states `(x, px, y, py, tau, delta)` through
every element at fixed reference energy. Its longitudinal coordinate `tau`
is in metres and `delta` is relative momentum deviation. RF, radiation
damping and excitation are excluded from this map. Distribution-generator
slopes must be converted to canonical transverse momenta before tracking.

This small CPU example constructs a stable synthetic ring and tracks two
launches in both transverse planes:

```python
import torch
from slipstream import ElementSpec, RingMap
from slipstream.analysis import nonlinear as nl

cell = [
    ElementSpec("QF", "quad", 0.2, k1=1.5),
    ElementSpec("D1", "drift", 1.0),
    ElementSpec("QD", "quad", 0.2, k1=-1.35),
    ElementSpec("S", "sext", 0.1, k2=10.0),
    ElementSpec("D2", "drift", 1.0),
]
ring = RingMap(cell * 3, energy_eV=3e9, device="cpu",
               dtype=torch.float64, n_kicks=4)
optics = nl.linear_optics(ring)
x = torch.tensor([1e-5, 2e-5], dtype=ring.dtype, device=ring.device)
state = nl.launch(optics, x, x)
result = ring.track(state, 512, aperture=(0.03, 0.02), record_every=1)
zx, zy = nl.normalized_signals(result.tbt, optics)
qx, amplitude_x = nl.naff(zx)
```

`closed_orbit(ring, delta=...)` solves the transverse fixed point at a
specified momentum deviation. `linear_optics` returns that orbit, its
one-turn Jacobian, fractional tunes and observation-point beta/alpha.
It rejects transverse coupling; use `coupled_optics` and `coupled_signals`
for stable symplectic normal modes. Mode labels are sorted by tune and can
change at crossings. These frozen-energy optics do not describe RF coupling.

`RingTrackResult.loss_turn` is zero based, with `-1` for survivors. Lost
slots remain present and are set to zero. Records have shape `(T, ..., 4)`
and contain the four transverse coordinates after recorded turns.
`track_loss_map` additionally checks each element exit and returns the
first loss element and its longitudinal position. A boundary check does
not locate a loss inside an element; converge slicing when this matters.
Quadrupole kicks set the tune accuracy, so converge them separately:
`RingMap(..., n_kicks_quad=8, n_kicks_sext=1)` is cheaper than raising
`n_kicks` for every element.

`naff` extracts a fractional frequency and amplitude magnitude from complex signals
whose sample axis is last. Use consecutive turn records for physical tunes.
`tunes` combines tracking and extraction and reports NaN for lost particles.
Converge turn count and use a seed band around the expected betatron tune
when another spectral line dominates. Finite records limit precision.

`chromaticity` differences nearby momentum tunes. `detuning` fits tune
slopes against Courant–Snyder actions in metres; keep all launches inside
the aperture. Both can retain autograd through tensor strengths.
`dynamic_aperture` returns surviving radii along sampled launch rays and
their polygon area. `frequency_map` returns transverse tunes and the
logarithm of tune change between two half-runs. DA is a finite-turn,
finite-grid survival measurement, and diffusion depends on record length.
Use float64 for small tune changes and converge integration slices, grid
resolution and turn count before interpreting either result.

`slipstream.analysis.loco.orbit_response_matrix(ring, corrector_names,
bpm_names)` returns a native closed-orbit response matrix with shape
`(n_bpm, 2, n_corrector)`, in metres per radian. Correctors are evaluated
at zero kick; BPM readings are taken at the named element exits. Use unique
element names for individual devices. `create_graph=True` retains strength
and alignment derivatives for fitting. `apply_calibration` applies BPM
gains/rolls and corrector gains; fix a gain scale or supply a prior to remove
the shared BPM/corrector scale ambiguity. These functions use native transport
and do not require an elegant backend. Optional live-elegant comparison tests
skip when development helpers or the executable are unavailable; native LOCO
regressions still run.

See [multipoles and coupled modes](ring_multipoles.md),
[nonlinear optimisation](nonlinear_optimisation.md), and
[RF-on analysis](ring6d.md) for extended APIs. CUDA and optional Warp
accelerate supported transport paths; CPU PyTorch remains the reference.
