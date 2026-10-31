# Model contract

Implemented: local Gaussian quantum diffusion and classical mean loss in
aligned pure dipoles, ultrarelativistic electrons/positrons, fixed reference,
CPU float64 eager single pass. Other radiation configurations raise explicit
errors. Default flags remain off. There is no ring equilibrium claim.

The package documentation `docs/modules/tracking/native/radiation.md`
derives the coefficients from [Geant4's standard synchrotron photon spectrum](https://geant4.web.cern.ch/documentation/pipelines/master/prm_html/PhysicsReferenceManual/electromagnetic/xray_production/InvSynFracInt.html),
equations 209/211/214. Mean relative loss per metre is
(2/3) r_e gamma^3 h^2; diffusion per metre is
55 r_e lambda_bar gamma^5 |h|^3/(24 sqrt(3)). For a fixed magnetic field
both kicks scale with (1+delta)^2. Quantum diffusion uses the second raw
photon moment, including count fluctuations. Canonical px,py scale with
momentum loss so photon recoil is collinear.

`radiation.py` owns the kick, `linac.py` owns positive midpoint slices and
generator wiring, and `lattice_import.py` imports explicit flags.
`tracker._build_params` rejects radiation flags outside LinacTracker.
`n_kicks_bend` sets emission slices; `generator` controls seeded sampling.
Zero bunch charge does not disable radiation. Changing references,
combined-function and misaligned bends, compilation and other masses are
unsupported. Photon tails/opening angles, quadrupole radiation, CUDA,
float32, gradients and collective combinations have no validated claim.

The separate `analysis.beamline_radiation` API implements CPU float64
expected linear covariance with differentiable magnet lengths, curvature,
strengths, faces and input covariance. It reuses the on-momentum rates,
integrates quantum diffusion by Van Loan's block exponential, and composes
maps/noise by a batched associative prefix scan. See
`docs/modules/analysis/beamline_radiation.md`. It fixes reference energy
and omits classical damping and nonlinear/chromatic transport. Its AD
gradient is an expected-moment gradient, not a sampled particle gradient.
Periodic-cell entrance optics may be differentiated through the cell map;
only stable solutions are physical. Validate resulting designs with native
particles including mean loss. [Van Loan (1978)](https://doi.org/10.1109/TAC.1978.1101743)
is the independent covariance-integral reference.
