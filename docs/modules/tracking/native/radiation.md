# Local dipole synchrotron radiation

## Physics

`LinacTracker` supports opt-in `ElementSpec.synch_rad` (classical loss) and
`ElementSpec.isr` (Gaussian quantum excitation). The elegant reader imports
the corresponding `SYNCH_RAD` and `ISR` bend attributes. Defaults are off.
The present contract is ultrarelativistic electrons/positrons (gamma >= 1000),
aligned, finite-length pure dipoles, eager tracking with a fixed reference.

For design curvature h = angle/length, reference gamma = E0/(m_e c^2), and
u = 1+delta, a positive physical path increment dl has

```
mean(d delta) = -(2/3) r_e gamma^3 h^2 u^2 dl
var(d delta)  = (55/(24 sqrt(3))) r_e lambda_bar gamma^5 |h|^3 u^4 dl
```

These are the first two *raw* photon-spectrum moments, not the variance
of one photon multiplied by its mean count. The Poisson number fluctuation
also contributes. Starting from the [Geant4 synchrotron spectrum](https://geant4.web.cern.ch/documentation/pipelines/master/prm_html/PhysicsReferenceManual/electromagnetic/xray_production/InvSynFracInt.html),
equations 209/211/214, exchange the order in the integral over photon energy:
integral x^n integral_x^infinity K_5/3(t) dt dx equals
integral t^(n+1) K_5/3(t) dt / (n+1). The n=1,2 moments give the coefficients
above. Package tests evaluate that integral numerically, independently of
the implemented constants. In the ultrarelativistic fixed-field model,
the local curvature is h/u and local gamma is gamma*u; this gives the u^2
factor in both the mean and RMS kicks.

The kick preserves slopes by scaling canonical px,py with u_new/u.
The physical path is dl = ds (1+h*x) u/sqrt(u^2-px^2-py^2).
Each bend has `n_kicks_bend` positive radiation slices. Each slice uses half
a magnetic map, a radiation kick, then half a magnetic map. Entrance/exit
faces occur only at the original magnet boundaries. Negative Yoshida
weights are confined to magnetic transport; they never create negative
emission intervals. Refine slices for convergence.

For weak excitation at constant energy, define
I3 = integral |h|^3 ds and I5 = integral H |h|^3 ds, where
H = gamma_x D^2 + 2 alpha_x D D' + beta_x D'^2.
Then sigma_delta,ISR^2 = C I3 and delta(epsilon_x) = C I5/2,
with C = 55 r_e lambda_bar gamma^5/(24 sqrt(3)). Finite excitation can be
computed by transporting the covariance and adding local delta variance.
Projected and dispersion-regressed emittances need not have equal growth.

## Usage

```python
tracker = LinacTracker(
    specs, energy_eV=500e9, device="cpu", dtype=torch.float64,
    always_change_p0=False, n_kicks=32, n_kicks_bend=32,
    generator=torch.Generator().manual_seed(1729),
)
particles_out, unchanged_reference_energy = tracker.track(particles_in)
```

Set the flags on bend specs or import them from an elegant lattice. A zero
bunch charge does **not** disable this single-particle effect. Reusing a
generator advances its state. Supply a newly seeded generator to reproduce
a run. RNG streams need not agree with elegant or between devices.

## Limits

The fixed reference remains at entrance energy; the beam loses mean energy
through delta. Report final RMS energy spread divided by final mean energy,
and normalised emittance using final mean momentum. For comparison with
constant-energy formulas also report fixed-entrance-normalisation results.
Centroid-following and prescribed changing references are rejected for now.
Ring trackers reject local radiation flags to prevent silent omission or
double counting with ring-average radiation. Combined-function/misaligned
bends, non-electron masses and compiled execution are rejected.

Photon opening angles, discrete photon spectra/tails, quadrupole radiation,
strong-field quantum corrections and ring equilibrium are outside this
model. Use slices small enough that each fractional mean loss is small.
The demonstrated contract is CPU float64 eager; CUDA, float32, gradients
and combined collective-effect accuracy require separate validation.
