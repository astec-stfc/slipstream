# Differentiable linear radiation transport

## Physics

`linear_radiation_transport` transports expected second moments in slope
coordinates `(x, x', y, y', delta)` through aligned hard-edge drifts,
quadrupoles and pure sector dipoles. Energy is constant. Each body obeys
`d Sigma/ds = A Sigma + Sigma A.T + Q`. The horizontal/energy block has
`x' = xp`, `xp' = -(k1+h²)x + h delta`, and `delta' = 0`. The vertical
block has `y' = yp`, `yp' = k1 y`. Zero-gap faces give `dxp = h tan(e) x`
and `dyp = -h tan(e) y`.

Only `Q[delta,delta]` is nonzero: the quantum variance per metre from
[`bend_radiation_rates`](../tracking/native/radiation.md). If
`exp([[A,Q],[0,-A.T]] L) = [[T,B],[0,T^-T]]`, the integrated noise is
`B T.T`. This [block-exponential integral](https://doi.org/10.1109/TAC.1978.1101743)
avoids emission-point quadrature error in the linear model. Composition
gives `Sigma_out = R Sigma_in R.T + Sigma_diffusion`.

Lengths, bend and face angles, strengths, and the supplied covariance keep
their PyTorch AD graphs. Differentiate smooth objectives built from the
result using normal Torch autograd. Periodic entrance optics can be derived
from a cell's returned transverse map and included in that graph. This
does not add parameter gradients to the stochastic particle tracker.

The survey integrates sector chords with `sinc(angle/2)` and drift headings.
`i2_m_inv`, `i3_m_inv2` and `loss_parameter = (2/3) r_e gamma³ I2` are
returned. Classical loss, energy-dependent optics, transverse damping,
nonlinear/chromatic transport, collective effects and photon tails are
absent from the propagated covariance. Verify designs with `LinacTracker`
with its classical-loss and ISR flags enabled. A loss parameter is an
estimate, not an assertion that classical damping was propagated.

## Usage

```python
from slipstream.analysis.beamline_radiation import linear_radiation_transport

result = linear_radiation_transport(
    length_m, angle_rad, k1_m2, covariance_5x5,
    energy_eV=500e9, e1_rad=face1_rad, e2_rad=face2_rad,
    samples_per_element=8,
)
gradient, = torch.autograd.grad(result.covariance[0, 0], length_m)
```

Supply equal-length 1D CPU float64 vectors with positive lengths, finite
values, and a symmetric physical covariance. Input and output covariance
units follow their coordinates (metres and dimensionless slopes/momentum).
The electron/positron energy must have gamma >= 1000. Bends with nonzero
quadrupole strength are rejected. Profile sampling subdivides body optics;
it does not change the exact endpoint noise integral. Faces are sampled
at their original boundaries. There are no alignment, fringe-gap, skew,
RF or aperture terms. CUDA, float32 and compilation are not validated.

Portable tests use independent 80-point Gauss-Legendre/SciPy matrix
quadrature, exact drift and sector survey formulas, subdivision invariance,
radiation scaling, and complete-API central finite-difference gradients.
