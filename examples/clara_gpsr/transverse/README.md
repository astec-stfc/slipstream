# Transverse GPSR reconstruction of the CLARA S02 bunch, with CSR

A reconstruction of the bunch at the entrance of CLARA S02 from images on `CLA-S07-DIA-SCR-04`, the straight-ahead screen in S07, with the VBC on so that
CSR matters. The model, the four cases (A to D), the requirements and the modelling choices are in the [overview](../README.md); this page is the setup
and the results of the transverse study.

The scan varies two S07 quadrupoles (`CLA-S07-MAG-QUAD-08`, `-09`; 12 pairs) with the transverse deflecting cavity (`CLA-S07-DIA-TDC-01`) off and on at
3 MV, which gives 24 images of 96 x 96 pixels (30 um pixels) with 1% noise. Five further quadrupole pairs, with the TDC off and on, are held out. The
screen has no dispersion, so the images constrain the transverse phase space and the time profile (through the TDC streak), not the energy.

## Run

```bash
python run_transverse.py  --bunch /path/CLA-S02-SIM-APER-01.openpmd.hdf5 --workdir _cache --out results --cases ACBD --iterations 1200
python plot_transverse.py --bunch /path/CLA-S02-SIM-APER-01.openpmd.hdf5 --workdir _cache --out results
python run_transverse.py  --bunch /path/CLA-S02-SIM-APER-01.openpmd.hdf5 --smoke    # CPU check of the whole chain, a few minutes
```

The results in `results/` come from `--iterations 1200` and the other defaults. Cost: 17 to 22 minutes per case on an RTX 3050 (8 GB); the whole set takes
about 80 minutes.

## Results

CSR is a strong effect in this line. At the entrance of S07 the truth projected emittance in the bending plane grows from 1.63 to 2.72 um
with CSR, the vertical plane is untouched, and the measured images with and without CSR differ by 113% (relative RMS).

Held-out images, relative RMS error (the fit's own model, and the reconstructed bunch tracked through the model that made the data):

| | A | B | C | D |
|---|---|---|---|---|
| fit's own model | 0.069 | 0.063 | 0.063 | 0.074 |
| true machine | 0.069 | **0.877** | 0.063 | **1.205** |

All four fits reach the same image error in their own model (`results/loss.png`). Only a wrong CSR assumption shows up when the reconstructed bunch is used
to predict the real machine (`results/images.png`).

Bunch at the entrance of S02 (`results/bunch_s02.png`), and at the entrance of S07 tracked by each fit's own model (`results/bunch_s07.png`); units um, 1e-3:

| | truth | A | B | C | D |
|---|---|---|---|---|---|
| S02 eps_x | 1.62 | 0.78 | 2.38 | 1.21 | 1.52 |
| S02 eps_y | 1.62 | 1.53 | 1.53 | 1.57 | 1.64 |
| S02 sigma_tau | 510 | 285 | 130 | 119 | 298 |
| S02 sigma_delta | 4.61 | 2.21 | 0.65 | 1.09 | 2.69 |
| S07 eps_x | 2.72 (CSR), 1.63 (none) | 2.36 | 2.53 | 1.20 | 1.24 |
| S07 sigma_tau | 64.2 (CSR), 63.2 (none) | 67.6 | 66.5 | 64.1 | 70.4 |
| S07 sigma_delta | 6.63 (CSR), 6.91 (none) | 3.68 | 1.74 | 1.59 | 3.87 |

What this supports:

- A model without CSR can match CSR data in its own model but gives the wrong bunch. B raises the S02 emittance to 2.38 um (truth 1.62) to reach the
  emittance the data show at S07, and it fails on the machine.
- With the right model (A, C) the reconstructed bunch predicts held-out images to about 6 to 7%.
- The vertical plane (no bends, no CSR) is recovered well in every case: eps_y within 6% at S02.

What it does not support:

- **Energy spread and chirp are not recovered.** The screen has no dispersion, and sigma_delta at the entrance of S07 comes out at 1.6e-3 to 3.9e-3
  against 6.6e-3 to 6.9e-3 in the truth, with large differences between cases (C is lower than A by a factor of two). The S02 sigma_tau and sigma_delta
  follow, because the compression is many to one. Only sigma_tau at the S07 entrance (64 to 70 um) is constrained. The
  [longitudinal study](../longitudinal/README.md) uses a dispersive screen for this.
- **The S02 emittance in the bending plane is weakly constrained.** The truth rms eps_x is dominated by a low-density halo: dropping the 5% of particles
  with the largest (x, px) amplitude lowers it from 1.62 to 0.99 um. The fits (0.78 for A, 1.21 for C) sit nearer that core value. I did not test whether the halo is
  below the image noise.
- One seed, one noise realisation and one set of hyperparameters. An earlier run of case C with a smaller generator network gave sigma_delta at S07 of 4.2e-3 instead of
  1.6e-3, so the longitudinal moments vary a lot between runs. The comparison of image errors (A, C against B, D) is the robust result.
