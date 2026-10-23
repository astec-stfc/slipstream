# Native straight drift model

Status: existing native physics, checked on CPU float64 eager execution.
Source revision: `7a03349f39c5aff4e6e0ace81c3fba7e68d571f5`, plus the
new drift reference test in this working tree. Re-inspect after API changes.

## Regime and current behavior

Forward, field-free straight transport with constant momentum, no collective
effects, no aperture losses and fixed reference energy. Canonical state is
`(x, px, y, py, tau, delta)`, where transverse momenta are normalized by p0,
delta is relative momentum deviation and tau is in metres, positive for late
arrival. Require total reference energy greater than rest energy, delta > -1
and positive longitudinal momentum. No near-grazing or zero-speed support is
claimed by this pilot.

## Independent derivation

This oracle uses elementary free-flight kinematics, independently of the
candidate's drift helper. No external paper or reference executable is needed.

Let E0 be total reference energy in eV, M the rest energy in eV, and c the speed
of light. Use momentum-times-c in eV for the dimensional calculation:

```text
P0 = sqrt(E0^2 - M^2)
P = P0 (1 + delta)
Px = P0 px; Py = P0 py
Pz = sqrt(P^2 - Px^2 - Py^2)
E = sqrt(P^2 + M^2)
vi = c Pi / E; v0 = c P0 / E0
dt = L / vz
x_out = x + vx dt; y_out = y + vy dt
tau_out = tau + c (dt - L/v0)
```

Position increments and tau have dimensions of length; px, py and delta stay
constant. Positive extra path length gives positive delay. The dimensional
velocity/time route is implemented using Python scalars in the independent
test oracle rather than calling `_canonical_drift`, `_beta_of` or slope
conversion helpers. The package implementation instead computes canonical
path length and beta. Agreement tests both normalization and time of flight.

| Claim | Source/derivation location | Conversion | Independent check |
| --- | --- | --- | --- |
| Straight positions | Free-flight derivation above; `velocity_reference` in the test file | Normalized momenta to dimensional velocities | Endpoint against velocity times travel time |
| Longitudinal delay | Arrival-time subtraction above | Multiply time difference by c | Late/tail tau, including off-momentum particles |
| Momentum constancy | No force in this model | px, py and delta retain their normalization | Exact equality before/after |
| Path composition | Constant velocity over each segment | Fixed reference in both segments | Two half lengths equal the full length |

## Integration points and limits

`tracking.native.elements.track_element` with angle, k1, k2, e1 and e2 zero
uses the existing exact straight drift. The public integration check uses
`LinacTracker` and one `ElementSpec(kind="drift")`, explicitly setting
`always_change_p0=False`. The normal default recenters reference momentum;
comparing its delta against a fixed-reference oracle would be incorrect.

Do not interpret slope distributions as canonical momenta. Do not apply this
formula to injector Cartesian state, a bend, RF, radiation or wakes. This
pilot does not validate CUDA, compiled/Warp execution, gradients, extreme
conditioning, performance or live external-code agreement.
