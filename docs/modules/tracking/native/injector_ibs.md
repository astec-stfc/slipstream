# `slipstream.tracking.native.injector_ibs`

[Source](../../../../src/slipstream/tracking/native/injector_ibs.py) · [Documentation index](../../../README.md)

## Physics

An opt-in local binary Coulomb collision operator for the Cartesian injector.
It supplements the existing smooth space-charge field solve and uses physical
particle density inferred from equal macro charges. Storage-ring Bane and
Bjorken–Mtingwa growth rates are separate models and are not applied here.

The implemented angular laws are Takizuka–Abe and Nanbu. Their normalization
and cell-based use follow E. Gjonaj, H. De Gersem and T. Limberg,
“Intrabeam Scattering Effects in the Electron Injector of the European XFEL”,
[FEL2022 WEP14, pp. 400–404](https://epaper.kek.jp/fel2022/pdf/WEP14.pdf),
DOI [10.18429/JACoW-FEL2022-WEP14](https://doi.org/10.18429/JACoW-FEL2022-WEP14),
especially p. 401, Eqs. (1), (2), (4), (5). The SI reduced-mass normalization
is independently confirmed in Caflisch et al., “A Hybrid Method for Accelerated
Simulation of Coulomb Collisions in a Plasma”, UCRL-JRNL-235460 (2007),
[Section 2.2, Eq. (5)](https://www.osti.gov/servlets/purl/941391).

### Rate and angular sampling

With relative speed w, equal physical masses m, reduced mass mu=m/2, physical
charge magnitude q, rest-frame density n and collision interval dt:

```text
variance = q^4 n lnLambda dt / (8 pi eps0^2 mu^2 w^3)
Takizuka–Abe: delta ~ Normal(0, variance); theta = 2 atan(delta)
Nanbu: coth(A) - 1/A = exp(-2 variance)
       cos(theta) = log(exp(-A) + 2 U sinh(A))/A; U ~ Uniform(0,1)
```

Azimuth is uniform. Nanbu uses a stable inverse-Langevin solve with analytic
limits; small deflections are stored as `1-cos(theta)` to retain tiny kicks.
TA requires time-step convergence in the small-variance regime. Nanbu allows
cumulative deflection, but still requires time, cell and particle convergence
for an evolving distribution.

Cells have explicit fixed **laboratory-frame** side lengths. Eligible charged,
emitted, surviving particles are randomly paired within cells; a random grid
shift prevents persistent boundary isolation. No cross-cell collision is
introduced. A randomly unpaired particle in an odd cell is left unchanged;
the paired rate is multiplied by Ncell/(Ncell-1) to compensate its sampling
probability. Sparse cells are noisy: inspect the unpaired-particle diagnostic
and increase particle count/refine the study before interpreting predictions.

Density is Ncell times macro weight divided by lab cell volume and local cell
Lorentz factor. The interval is also divided by that Lorentz factor. Thus a
lab-density/lab-time rate has the required inverse-gamma-squared factor.
The local frame is obtained from the summed particle four-momenta.

### Conservative pair kinematics

The code deliberately uses exact equal-mass center-of-momentum kinematics,
instead of copying the approximate transverse transformation printed in WEP14
Eq. (3). Write proper velocity as u=p/m. For a pair:

```text
b = (u1+u2)/(c (gamma1+gamma2)); G = 1/sqrt(1-|b|^2)
r = (u1-u2)/2
k = r - G/(G+1) dot(r,b) b
```

In the pair frame the momenta are ±m*k. Rotate k without changing its norm.
If dk is its increment, transform only the increment back:

```text
du = dk + G^2/(G+1) dot(dk,b) b
u1_out = u1 + du; u2_out = u2 - du
```

These identities follow from the Lorentz transformation of equal-mass
four-momenta; `gamma1-gamma2 = dot(u1-u2,b)/c`. They avoid subtracting two large
boosted longitudinal momenta. Pair total momentum and relativistic energy are
conserved to roundoff. Individual particle energies and energy spread can
change. IBS does not create total kinetic energy in an elastic pair.

The **rate** is still nonrelativistic in the local thermal motion. The default
rejects pairs with speed above 0.1c in their center-of-momentum frame, even
when the bulk beam is highly relativistic. This is a domain guard, not a
one-percent accuracy guarantee. Gradients, relativistic thermal collision
rates, unequal macro weights and multiple species are not supported.

### Impact cutoff and limits

By default compute the classical SI Landau length
`bmin = q^2/(4*pi*eps0*mu*w^2)` and `lnLambda=max(0, ln(bmax/bmin))`, with
`bmax=max(dx,dy)`, optionally capped by `max_impact_parameter_m`. Zero-relative
velocity and bmax<=bmin produce no angular kick. A fixed nonnegative
`coulomb_log` can instead be supplied for controlled benchmarks.

The collision grid is independent of the space-charge mesh. Choose its
transverse cutoff no larger than the unresolved scale of the field solver;
the code cannot automatically infer that scale for every cylindrical/adaptive
mesh. A fixed Coulomb log overrides this cutoff policy. Report it explicitly.
Changing these settings changes the collision model as well as discretization.
There is no modeled cathode-image collision, quantum/degenerate correction,
strong-coupling/disorder-induced heating, spin or radiation. A very cold
photocathode cannot be claimed accurate merely because the operator runs.

## Usage

```python
from slipstream import Injector, InjectorIBSSettings

machine = Injector.load("injector.yaml")  # or Injector.from_astra("injector.in")
machine.ibs = InjectorIBSSettings(
    enabled=True,
    cell_size_m=(50e-6, 50e-6, 100e-6),  # study choice; establish convergence
    model="nanbu",  # or "takizuka-abe"
    max_impact_parameter_m=50e-6,
    seed=17,
    backend="auto",  # Warp on supported CUDA; PyTorch otherwise
)
result = machine.track(device="cpu", dtype="float64", return_particles=True)
print(result["ibs"])
machine.save("injector_with_ibs.yaml")
```

IBS defaults off; old machine files remain valid and disabled tracking retains
its previous particle results. The settings round-trip through machine YAML.
Scattering is applied after each field/space-charge push, only to particles
active in that step and still alive after its aperture check. Zero-charge
probes are not collision partners. Equal nonzero macro charges are required.
The source distribution's physical macro charge, not the number of simulated
particles alone, sets the rate.

Only float64 is supported. `backend="auto"` uses the optional Warp pair kernel
on CUDA when installed, and the PyTorch reference otherwise. Set
`backend="torch"` to select the reference explicitly; `backend="warp"` requires
CUDA and Warp and fails explicitly when unavailable. Backend selection is
independent of the field pusher's `use_compile` option. Cell grouping, sorted
segment reductions and random draws remain in PyTorch. A Warp thread handles
one disjoint pair: rate, stable angular sampling, Lorentz rotation and updates,
without momentum atomics. The CPU path needs no Warp installation. Cell reductions
avoid CUDA scatter-atomic nondeterminism. Repeating a
run with the same inputs, device and seed reproduces the tested sequence;
CPU and CUDA use different RNG streams and are not expected to match particle
by particle. Torch and Warp on the same device consume identical private RNG
draws and agree to float64 tolerance; they need not be bitwise identical.
The collision RNG is private. `track_differentiable` raises if
IBS is enabled, rather than silently omitting it.

Diagnostics report calls, selected pairs, accumulated eligible/unpaired
particle calls, maximum angular variance, maximum pair thermal beta and the
actual collision backend.
Large unpaired fractions indicate inadequate local statistics; maximum
variance zero means no collision kick was active. The implementation is
intended for physics studies, not an established performance optimization.

Run `python -m pytest tests/test_injector_ibs.py tests/test_injector_ibs_warp.py -q` for rate normalization,
angular-law moments, CPU/CUDA conservation and seeded replay, limits,
homogeneous initial diffusion under particle/time refinement, and actual
injector/YAML/default-off integration. These synthetic checks do not reproduce
the European XFEL or SwissFEL measurements. A real machine study needs matched
field maps, source, cutoff and grid/particle/time convergence.

The optional parent workbench's CLARA runner supports `--ibs`, `--no-ibs`
and `--ibs-comparison --ibs-cell-size-m DX DY DZ`. The comparison uses float64
for both runs, fixes source/subsampling seeds, labels off/on curves and saves
separate trajectories plus settings/diagnostic JSON. `--ibs-seed` changes only
collisions; `--ibs-backend torch` selects the reference. The cached ASTRA curve
keeps its original physics. The runner's RMS-energy-spread plot is projected;
use the separate slice-study runner when slice spread is the observable.

The original method references are T. Takizuka and H. Abe,
[J. Comput. Phys. 25, 205–219 (1977)](https://doi.org/10.1016/0021-9991(77)90099-7),
and K. Nanbu, [Phys. Rev. E 55, 4642 (1997)](https://doi.org/10.1103/PhysRevE.55.4642).
