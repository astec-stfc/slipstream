# MLS Twiss reference

`Twiss.png` is the user-supplied reference for the illustrative ring in
`../nonlinear_optimisation.py`. Its source, working point and detailed lattice
are not specified. Approximate scales read from the axes are maximum
horizontal beta 15 m, vertical beta 9 m, and horizontal dispersion 1.5 m.

The five example quadrupole families use these normalized strengths:

| Family | K1 [m^-2] |
| --- | ---: |
| Q1 | -0.7678415218043021 |
| Q2L | 3.29122194477513 |
| Q2S | 5.0602022474686 |
| Q3L | -2.847681806285773 |
| Q3S | -6.515853025684272 |

L/S distinguish the long/short straight ends of an arc. These signs follow
Slipstream's convention: positive K1 focuses horizontally. The strengths
are representative fits, not measured or operational MLS settings.

The fit used independent thick quadrupole, drift and sector-bend matrices,
periodic uncoupled Twiss and dispersion, and SciPy least squares. Full tunes
(3.18, 2.23) remain the published standard-operation targets used by the
example; they are not inferred from the image. The final residual vector was
`[10000*(Qx-3.18), 10000*(Qy-2.23), max(beta_x)-15,
max(beta_y)-9, 3*(max(D_x)-1.5)]`, with K1 bounded to +/-10 m^-2.
Maxima were evaluated at element boundaries. Native float64 transport with
the example's slicing independently gives beta maxima 15.20/9.03 m and
dispersion maximum 1.67 m, with fractional tunes 0.18000009/0.23000008.

This matches the overall envelope, with nonzero straight-section dispersion.
The original estimated magnet lengths/locations are retained, so the curves,
waists and peak positions still differ. The reference appears to start at
the centre of a long straight; the example starts at its entrance, a 3 m
origin difference. Matching the complete curve needs the actual lattice or
a fit of magnet positions and bend edges as well as gradients.

Generate the current reference tables and plots with:

```bash
python examples/nonlinear_optimisation.py --reference-only
```

Outputs default to `.cache/nonlinear_optimisation`. Matplotlib is optional:
install `slipstream-accel[plot]`, or use `--no-plots` for tables only.
