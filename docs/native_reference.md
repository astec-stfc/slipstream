# Native physics reference

This is the shared convention reference for the native package. Detailed
models and API contracts live in the [module reference](modules/README.md).
Development histories and benchmark lessons live in the workbench's
`plans/NOTES.md`; they are not a claim that every comparison was rerun here.

## Physics

### Coordinates and relativistic normalization

Ring and linac tensors contain `(x, px, y, py, tau, delta)`. Positions and
`tau` are metres; `px=Px/p0` and `py=Py/p0` are normalized canonical momenta.
`delta=(p-p0)/p0`. Reference energy `E0` is total energy in eV, and the rest
energy `mc2` uses the same unit. Momentum expressed in energy units is
`p0c=sqrt(E0**2-mc2**2)`.

```text
pz/p0 = sqrt((1+delta)**2 - px**2 - py**2)
xp = px / (pz/p0)
yp = py / (pz/p0)
px = (1+delta)*xp / sqrt(1+xp**2+yp**2)
py = (1+delta)*yp / sqrt(1+xp**2+yp**2)
```

Positive `tau/c` is later arrival relative to the reference. Native
distribution generators return slopes, and must be converted before canonical
tracking. Injector state is `(x,y,z,ux,uy,uz)` with proper velocities in m/s;
its endpoint conversion is a separate Cartesian-to-common-position handoff.
See [elements](modules/tracking/native/elements.md),
[distributions](modules/tracking/distributions.md) and
[injector](modules/tracking/native/injector.md).

### Magnetic transport

A straight canonical drift over reference length `ds` preserves momenta and
updates `x += ds*px/(pz/p0)`, `y += ds*py/(pz/p0)`.
Its actual path length is `ds*(1+delta)/(pz/p0)`.
Sector-bend drift uses fixed curvature `h=angle/length`; the implementation
rewrites trigonometric differences to avoid cancellation at small `ds*h`.
The bend's auxiliary `factor` is momentum-normalized: physical arc length
is `factor*(1+delta)`, not `factor` alone.

For the dipole, normal quadrupole and sextupole terms:

```text
Fy = h + k1*x + k2*(x*x-y*y)/2
Fx = k1*y + k2*x*y
dpx = -ds*(1+h*x)*Fy
dpy =  ds*(1+h*x)*Fx
```

The current implementation also supports normal/skew higher multipoles and
zero-length integrated multipoles. It uses the factorial-normalized complex
polynomial convention documented in [multipoles](ring_multipoles.md).
The exact-dipole alternative transports the dipole field in the drift map
and excludes it from the remaining kicks.

Magnetic time of flight uses the live relativistic particle velocity:
`tau += arc_length/beta - ds/beta0`. Magnetic transport alone has no
per-bend radiation loss or excitation. Opt-in [local dipole radiation](modules/tracking/native/radiation.md), RF and collective effects update
momentum through their separate element/tracker paths.

Entrance/exit edges support enabled/disabled faces and first-order or Brown
second-order maps, including `HGAP`/`FINT`. In the first-order slope map,
`rho=(1+delta)/h`, `psi0=2*HGAP*FINT*h*(1+sin(e)**2)/cos(e)`,
`xp += tan(e)*x/rho`, and `yp -= tan(e-psi0/(1+delta))*y/rho`.
The second-order map has a different momentum dependence for its fringe
correction; use the implemented map rather than extending this first-order
formula by analogy. Alternative edge models and full curved-reference
rigid-body misalignments are outside this implementation.

### Integration and reference subtraction

Magnetic slices use symmetric drift/kick compositions of order 4, 6 or 8.
Yoshida's triple-jump extends a symmetric order-`n` map by composing it at
`x1*ds`, `x0*ds`, `x1*ds`, where
`x1=1/(2-2**(1/(n+1)))` and `x0=1-2*x1`.
The order-six coefficient table is a separately selected solution; repeatedly
extending the order-four table is not expected to reproduce it.

`reference_offset` subtracts the **ideal** element's numerical reference
displacement. Alignment errors must retain their physical orbit effect.
Offsets depend on reference energy; source handoffs refresh them without
mutating shared cached tensors. Converge slice counts for the observable:
optics and long-turn tune studies can require finer slicing than a short
transport demonstration.

### RF, reference energy and focusing

`NativeTracker(rf="lumped")` applies the main RF kick once per turn.
`rf="per_element"` uses the actual cavity positions. `LinacTracker` always
uses inline cavities and has no ring-average radiation or IBS.

The ring RF helper returns fractional energy increment
`(Vrf/E0)*(sin(phi_s+omega_rf*tau_s)-sin(phi_s))`, with `tau_s=tau/c`.
The lumped tracker adds this directly to momentum delta, an ultrarelativistic
small-kick approximation; it does not perform an exact low-beta conversion.
Per-cavity RF instead adds the local energy gain to particle gamma and
reconstructs momentum; a small `dE/E0` substitution is inadequate at low beta.

Cavities can be split into cells. The supported traveling-wave phase
velocity is `beta_wave*c`, and the continuum phase-slip relation is
`dpsi/dz=(omega_rf/c)*(1/beta(z)-1/beta_wave)`.
Reference matching uses mean **momentum**, not mean particle energy.
For the kick path, final-cell reference matching precedes the trailing
half-drift. Entrance/exit focus kicks convert slopes back to canonical
momenta with the live `pz/p0` factor. TW1 body focusing uses local kicks;
SRS uses the standing-wave transverse matrix. These are distinct models.
See [element RF formulas](modules/tracking/native/elements.md).

Linear RF optics differentiates a slope-in/slope-out wrapper. Canonical
momentum and slope Jacobians differ after acceleration. Reference-code
Twiss tables and reference-code particle tracking may use different RF
matrix models, so select the comparison matching the requested model.

### Ring radiation, ramps and IBS

`ring_physics` implements radiation as once-per-turn damping/excitation.
`tau_z_s` is an amplitude e-folding time. The implemented small-step rule
has `decay=1-T0/tau_z` and noise scale
`sigma_delta0*sqrt(2*T0/tau_z)`. It is a first-order discretization, not the
exact finite-step Ornstein-Uhlenbeck equilibrium rule. Transverse damping
acts on momentum and uses the matched spread
`sqrt(eps*(1+alpha**2)/beta)` for its noise normalization.

IBS growth-rate conventions enter the effective damping times as
`1/tau_z_eff=1/tau_z_SR-inv_Tp`,
`1/tau_x_eff=1/tau_x_SR-2*inv_Tx`, and the corresponding y expression.
Recompute from dispersion-subtracted beam statistics and combine with the
current SR baseline, rather than repeatedly folding into an already modified
time constant. Bane's approximation is high-energy; the general
Bjorken-Mtingwa integral is available for other regimes.

Ramp scaling uses `gamma_now/gamma_ref`: energy loss scales with the fourth
power, radiation damping times with the inverse cube, natural momentum spread
linearly, and equilibrium transverse emittances with the square.
The configured ramp supplies the literal RF program. The tracker also
rescales normalized coordinates at the reference-energy boundary; the
implemented energy-ratio convention is documented in
[tracker](modules/tracking/native/tracker.md) and
[ramp](modules/lattice/ramp.md).

### Collective effects, optics and errors

Longitudinal wakes act on trailing particles; CSR has its own causal/sign
convention because radiation can overtake the bunch. The linac provides
unshielded steady-state CSR, local LSC, explicit fast-grid approximations,
and a single-upstream-bend drift carry. Shielding and complete overlapping
history/transients remain outside that model. See
[collective kernels](modules/tracking/native/collective.md).

Uncoupled periodic optics uses the one-turn Jacobian:
`cos(mu)=(M11+M22)/2`, `beta=M12/sin(mu)`,
`alpha=(M11-M22)/(2*sin(mu))`, with the stable phase/sign branch.
Periodic dispersion solves `(I-R)*D=r`.
Coupled optics needs the eigenmode construction rather than independently
applying those two-dimensional expressions.

Transverse offsets and rolls sandwich local element transport with coordinate
translations/rotations. They are not a full pitch/yaw rigid-body bend model.
Fiducialization preserves the reference-arrival convention across RF models;
its live-reference acceptance is distinct from testing its internal phase
algebra. See [errors](modules/tracking/native/errors.md) and
[optics](modules/tracking/native/optics.md).

### Sources

- H. Yoshida, *Construction of higher order symplectic integrators*,
  Physics Letters A 150 (1990), 262–268.
- elegant implementation: `multipole.c`, `csbend.c`, `track_data.c`,
  `simple_rfca.c`, `do_tracking.c`, `phase_reference.c`, `error.c`,
  `load_parameters.c`, `malign_mat.c` and `tilt_matrices.c`.
- K. Bane, SLAC-PUB-9226; K. Kubo, S. Mtingwa and A. Wolski,
  Physical Review ST Accelerators and Beams 8, 081001 (2005), for IBS.
- K. Floettmann's ASTRA manual for field/distribution semantics. Model-specific
  SRS, collective-effect and multilayer citations are retained in their module
  pages alongside the relevant formulas.

## Usage

Choose the tracker by physical regime, then supply the correct coordinate
convention and total reference energy. Keep the study's dtype/device stable.
Use `return_trajectory=True` when comparing element-boundary diagnostics;
use a fixed nominal reference-energy profile for linac jitter studies that
must preserve the injected error instead of recentering it.

Start with [native tracking](tracking_native.md) or the runnable
[beamline YAML example](beamline_studies.md). For numerical validation,
compare identical particles, charge, reference timing, element settings and
diagnostic definitions. Qualify CPU/CUDA, eager/compiled, statistical noise
and optional reference dependencies independently.
