# Injector IBS model contract

Implemented in `tracking/native/injector_ibs.py` and wired through
`Injector.track`. Read `docs/modules/tracking/native/injector_ibs.md` from the
package root for the full Physics/Usage derivation and primary sources.
The inspected base revision is `7a03349f39c5aff4e6e0ace81c3fba7e68d571f5`;
the implementation/tests/docs are additional uncommitted changes. Verify
current source rather than inferring that the base commit contains this feature.

## Model and source traceability

| Contract | Verified primary source or independent derivation | Oracle |
| --- | --- | --- |
| Equal-mass SI collision strength | Gjonaj, De Gersem, Limberg, FEL2022 WEP14 p. 401 Eq. (1); Caflisch et al., UCRL-JRNL-235460 Section 2.2 Eq. (5) | Rutherford impact-integral calculation with mu=m/2, density from real macro charge, and independent bulk-gamma scaling |
| TA/Nanbu angular laws | WEP14 Eqs. (1), (4), (5); original methods DOI 10.1016/0021-9991(77)90099-7 and 10.1103/PhysRevE.55.4642 | Gaussian quadrature for TA; Nanbu mean cos(theta)=exp(-2 variance) |
| Conservative lab-frame pair update | Explicit equal-mass Lorentz derivation in the package module page | Known-angle rotation and before/after summed relativistic energy and momentum |
| Initial longitudinal diffusion | Landau small-step pair moment: d<uz^2>=K*n*lnLambda*dt/2*E[(w^2-3*wz^2)/w^3] | Independent lattice-pair multiplicities for initially transverse thermal velocities; particle/time refinement |

The published source is available at
https://epaper.kek.jp/fel2022/pdf/WEP14.pdf. The original Indico attachment in
the workbench plan returned a login page rather than the paper. The independent
normalization report is https://www.osti.gov/servlets/purl/941391.
The source page was visually inspected to verify the reduced-mass square and
time normalization. Bibliographic references alone are not equation verification.

## Coordinate and integration constraints

Inputs are Cartesian positions in metres and proper velocity u=p/m in m/s,
not native canonical state or slopes. Collision-cell lengths are fixed in the
laboratory frame. Equal physical macro weights determine real number density;
zero-charge probes, future emissions and aperture losses are excluded.
Both density and time transform by inverse local gamma. Pair rotations use
exact Lorentz kinematics and leave positions unchanged during the collision
substep. Grid shifts/pairing/angles use a private, seeded device RNG.

The automatic Coulomb logarithm uses a classical Landau lower cutoff and a
cell-scale transverse upper cutoff, optionally capped explicitly. The collision
grid is not automatically coupled to the adaptive/cylindrical space-charge
mesh. Match the unresolved physical scale and establish numerical convergence;
a fixed `coulomb_log` overrides this policy and must be reported.

IBS defaults off. It is independent of the storage-ring `ibs.py` formulas.
Settings round-trip through injector machine YAML. Scattering runs after each
field/space-charge push. Float64 CPU/CUDA PyTorch collisions and an optional
CUDA Warp pair kernel are implemented. `backend="auto"` selects Warp when
available; `torch` selects the reference and explicit `warp` requires CUDA/Warp.
Grouping, sorted reductions and RNG stay in PyTorch. The pair kernel handles
rate, angular sampling and the conservative update on the active Torch stream.
Existing field pushing may still use compile/Warp. Unequal weights, relativistic
thermal collision rates, quantum/strong-coupling/DIH physics and gradients are
not supported. `track_differentiable` raises when IBS is enabled. Synthetic
tests do not establish agreement with a particular real photoinjector.
