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
