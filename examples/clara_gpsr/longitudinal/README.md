# Longitudinal GPSR reconstruction of the CLARA S02 bunch, with CSR

A reconstruction of the bunch at the entrance of CLARA S02 from images on `CLA-SP3-DIA-SCR-01`, the first screen after the SP3 spectrometer dipole,
with the VBC on so that CSR matters. The model, the four cases (A to D), the requirements and the modelling choices are in the [overview](../README.md);
this page is the setup and the results of the longitudinal study. The [transverse study](../transverse/README.md) uses the straight-ahead screen
`CLA-S07-DIA-SCR-04` instead and constrains the other half of the bunch.

## The line and the screen

The beam follows S07 to `CLA-S07-MAG-HVCOR-05`, where SP3 branches off, and then the 30 degree SP3 dipole (`CLA-SP3-MAG-DIP-01`) to the screen. The dipole
disperses the beam horizontally (eta = 0.50 m at `CLA-SP3-DIA-SCR-01`, so eta x sigma_delta = 3.5 mm) and the transverse deflecting cavity in S07 streaks
it vertically, so the image is the longitudinal phase space: energy on x, time on y. `CLA-SP3-DIA-SCR-02` was not used: with the default SP3 quadrupole
strengths its dispersion is -0.08 m. The SP3 quadrupoles are set to zero.

The line is built from LAURA's elegant export of the S07 and SP3 sections with three additions, all in `clara_model.py` and `run_longitudinal.py`:

- LAURA exports the SP3 dipole with angle 0 (the angle comes from a magnet current the lattice does not hold). The example sets 0.5236 rad (30 degrees), the
  survey rotation of the elements after the dipole, and keeps the exported sector edges (e1 = e2 = 0). The edge model is an assumption.
- The dipole sits 0.13 m after `CLA-S07-MAG-HVCOR-05`; a drift of that surveyed length joins them, because LAURA's SP3 section starts at the dipole.
- The section export of S07 stops at the corrector, since the quadrupoles and screens after it are on the straight line, not on the SP3 branch.

## Scan

Two S07 quadrupoles, `CLA-S07-MAG-QUAD-06` and `-07` (k1 in {-3, -1.5, 0, 1.5} and {-4, -3, -1.5} per m^2, 12 pairs), with the cavity off and on at 3 MV: 24
images of 96 x 96 pixels, 220 um (x) by 50 um (y) per pixel, point-spread width of one pixel, 1% noise. The pairs move the vertical waist onto the screen:
the vertical size without the cavity runs from 0.01 to 0.7 mm across the scan, so the streak is resolved in some rows and the rows without a waist give the
transverse size. Five further pairs, with the cavity off and on, are held out. About 1% of the charge at most falls off the frame in any scan image, and none in the
held-out images.

## Run

```bash
python run_longitudinal.py  --bunch /path/CLA-S02-SIM-APER-01.openpmd.hdf5 --workdir _cache --out results --cases ACBD --iterations 1200
python plot_longitudinal.py --bunch /path/CLA-S02-SIM-APER-01.openpmd.hdf5 --workdir _cache --out results
python run_longitudinal.py  --bunch /path/CLA-S02-SIM-APER-01.openpmd.hdf5 --smoke    # CPU check of the whole chain, a few minutes
```

The results in `results/` come from `--iterations 1200` and the other defaults. Cost: 18 to 23 minutes per case on an RTX 3050 (8 GB); the whole set takes
about 82 minutes.

## Results

CSR changes these images less than it changes the transverse ones: the measured images with and without CSR differ by 24% (relative RMS, including the
independent noise of the two sets), against 113% on the straight screen. Most of the effect of CSR in this line is on x and px, and here the image
in x is the energy distribution.

Held-out images, relative RMS error (the fit's own model, and the reconstructed bunch tracked through the model that made the data):

| | A | B | C | D |
|---|---|---|---|---|
| fit's own model | 0.057 | 0.055 | 0.059 | 0.059 |
| true machine | 0.057 | **0.317** | 0.059 | **0.153** |

Bunch at the entrance of S02 (`results/bunch_s02.png`), and at the entrance of S07 tracked by each fit's own model (`results/bunch_s07.png`); units um, 1e-3:

| | truth | A | B | C | D |
|---|---|---|---|---|---|
| S02 sigma_tau | 510 | 507 | 487 | 505 | 515 |
| S02 sigma_delta | 4.61 | 4.62 | 4.25 | 4.59 | 4.94 |
| S02 chirp (1/m) | 9.01 | 9.02 | 8.64 | 8.95 | 9.35 |
| S07 sigma_tau | 64.2 (CSR), 63.2 (none) | 64.8 | 64.5 | 64.1 | 64.0 |
| S07 sigma_delta | 6.63 (CSR), 6.91 (none) | 6.69 | 6.69 | 6.94 | 6.90 |
| S02 eps_x | 1.62 | 0.66 | 0.20 | 0.09 | 0.69 |
| S02 eps_y | 1.62 | 1.20 | 0.95 | 1.24 | 1.42 |

What this supports:

- The longitudinal phase space is recovered. In all four cases the S02 bunch length is within 5% of the truth, the energy spread within 8% and the chirp within
  4%, and the energy spread at the S07 entrance within 5%. The straight screen gave 1.6e-3 to 3.9e-3 for the last one, against 6.6e-3 to 6.9e-3 in the truth.
- With the right model (A, C) the reconstructed bunch predicts held-out images to 6%.
- A wrong CSR assumption shows up in the check in the true machine (B 0.317, D 0.153) but much less than on the straight screen (0.877 and 1.205), as the smaller
  effect of CSR on the images (24% against 113%) suggests. The S02 moments of B and D differ from those of A and C by 2 to 8%, which this single seed cannot separate from run to run
  variation.

What it does not support:

- **The transverse emittances are not recovered.** The horizontal size on the screen is eta x delta, so the betatron size, and with it eps_x, is barely seen: the
  fits give 0.09 to 0.69 um against 1.62 (C is the smallest). eps_y comes out at 0.95 to 1.42 um against 1.62. The transverse study constrains these.
- **Local energy structure is not resolved.** The truth is close to a single curve in time and energy, with an energy modulation of 1 to 2e-3 from CSR
  once the chirp is removed. The reconstructed bunches have a visibly larger local energy spread (last column of `results/bunch_s07.png`), which hides that
  modulation; B, with no CSR in its model, shows the same large-scale trend as A.
- One seed, one noise realisation and one set of hyperparameters.
- The CSR of the SP3 dipole itself is left out of truth and fit (see the overview); it has about the same size as the CSR of the VBC on the particle energies.
