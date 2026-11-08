# GPSR reconstruction of the CLARA S02 bunch, with CSR

A phase-space reconstruction (GPSR) of the bunch at the entrance of CLARA S02 from screen images at the end of S07 (`CLA-S07-DIA-SCR-04`),
with the VBC on so that CSR matters. Slipstream's `LinacTracker` is the forward model: it tracks S02 to S07 with RF, wakes and CSR, and it is
differentiable, so GPSR's neural-network beam generator is trained by gradient descent on the image loss.

The true bunch is a simulated, non-Gaussian SIMBA bunch (250 pC, 35.2 MeV, 510 um rms at the entrance of S02, 64 um rms after the VBC).
The scan varies two S07 quadrupoles (`CLA-S07-MAG-QUAD-08`, `-09`; 12 pairs) with the transverse deflecting cavity (`CLA-S07-DIA-TDC-01`) off and on
at 3 MV, which gives 24 images of 96 x 96 pixels with 1% noise. Five further quadrupole pairs, with the TDC off and on, are held out.

Four cases share one machine model and differ in whether CSR is in the data and in the fit:

| Case | Measured images | Model in the fit | Purpose |
|---|---|---|---|
| A | with CSR | with CSR | the full model |
| B | with CSR | without CSR | what ignoring CSR does |
| C | without CSR | without CSR | baseline: GPSR with a correct model, no CSR anywhere |
| D | without CSR | with CSR | a CSR model on a beam that has none |

## Requirements

- Slipstream and PyTorch; a CUDA GPU is strongly advised (see Cost).
- `gpsr` (0.3.0 was used) and its dependency `cheetah-accelerator` (0.8.4), `h5py`, `matplotlib`.
- LAURA and `laura-lattices` (the CLARA lattice), which turn the lattice into Slipstream elements.
- A SIMBA openPMD bunch at the entrance of S02 (`CLA-S02-SIM-APER-01.openpmd.hdf5`). Nothing from the machine is stored in this repository;
  you pass the path.

## Run

```bash
python run_clara_gpsr.py --bunch /path/CLA-S02-SIM-APER-01.openpmd.hdf5 --workdir _cache --out results --cases ACBD --iterations 1200
python plot_results.py   --bunch /path/CLA-S02-SIM-APER-01.openpmd.hdf5 --workdir _cache --out results
python run_clara_gpsr.py --bunch /path/CLA-S02-SIM-APER-01.openpmd.hdf5 --smoke    # CPU check of the whole chain, a few minutes
```

`--workdir` is where LAURA writes the lattice and the wake tables. `plot_results.py` needs the same `--n-particles`, `--rf-kicks` and
`--csr-drift-dz` as the run (the defaults match). The results in `results/` come from `--iterations 1200` and the other defaults: 8192 particles in
float64, Adam at 5e-3 with a cosine schedule, one seed.

Cost: 17 to 22 minutes per case on an RTX 3050 (8 GB) with each step captured as one CUDA graph; the whole set takes about 80 minutes.

## Results

CSR is a strong effect in this line. At the entrance of S07 the truth projected emittance in the bending plane grows from 1.63 to 2.72 um
with CSR, the vertical plane is untouched, and the measured images with and without CSR differ by 113% (relative RMS).

Held-out images, relative RMS error (the fit's own model, and the reconstructed bunch tracked through the model that made the data):

| | A | B | C | D |
|---|---|---|---|---|
| fit's own model | 0.069 | 0.063 | 0.063 | 0.074 |
| true machine | 0.069 | **0.877** | 0.063 | **1.205** |

All four fits reach the same image error in their own model (`loss.png`). Only a wrong CSR assumption shows up when the reconstructed bunch is used to predict
the real machine (`images.png`).

Bunch at the entrance of S02 (`bunch_s02.png`), and at the entrance of S07 tracked by each fit's own model (`bunch_s07.png`); units um, 1e-3:

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

- **Energy spread and chirp are not recovered.** The scan has no dispersive screen, and sigma_delta at the entrance of S07 comes out at 1.6e-3 to 3.9e-3
  against 6.6e-3 to 6.9e-3 in the truth, with large differences between cases (C is lower than A by a factor of two). The S02 sigma_tau and sigma_delta
  follow, because the compression is many to one. Only sigma_tau at the S07 entrance (64 to 70 um) is constrained. A second diagnostic with dispersion
  would be the natural addition; it is not in this example.
- **The S02 emittance in the bending plane is weakly constrained.** The truth rms eps_x is dominated by a low-density halo: dropping the 5% of particles
  with the largest (x, px) amplitude lowers it from 1.62 to 0.99 um. The fits (0.78 for A, 1.21 for C) sit nearer that core value. I did not test whether the halo is
  below the image noise.
- One seed, one noise realisation and one set of hyperparameters. An earlier run of case C with a smaller generator network gave sigma_delta at S07 of 4.2e-3 instead of
  1.6e-3, so the longitudinal moments vary a lot between runs. The comparison of image errors (A, C against B, D) is the robust result.

## Modelling choices

- The lattice comes from LAURA's elegant export of S02, L02, S03, L03, S04, L4H, S05, VBC, S06, L04 and S07 (travelling-wave cavities with body focusing and
  longitudinal and transverse wakes; CSR bends). S02 to L04 is tracked once per step and only S07 up to the screen is repeated for each scan row.
- The VBC dipole angle is set to 0.1254 rad, calibrated so that the bunch length after the VBC matches the SIMBA bunch (64.7 um rms) without CSR.
- The elegant edge model 3 is mapped to the first-order edge (1); synchrotron radiation and incoherent synchrotron radiation are off.
- The exported TDC has no voltage and no deflection plane: the example deflects vertically and sets the phase to the zero crossing at the centre of the bunch.
- The quadrupoles of S06 and S07 are not re-matched for this bunch; the scan ranges were chosen to keep the beam on the screen.
- Cavity sub-kicks are cut from 261 to 360 down to 45 and the CSR kick length in the drifts after the dipoles is 0.05 m (0.01 m as exported) to fit in 8 GB.
  Finer settings were not run in the full fit.
- The fiducial time, the wake window and the TDC phase are fixed once from the true bunch, as an operator would set them; wake-window overflow is monitored and was zero.
- The fit does not know the true bunch: only the images, the lattice and the machine settings.

## Files

- `clara_model.py`: LAURA lattice to Slipstream elements, bunch loader, `Machine` and the forward module `ClaraForward`.
- `run_clara_gpsr.py`: scan table, measured images, GPSR training of the four cases, tables.
- `plot_results.py`: figures and the held-out check in the true machine.
- `results/`: figures and `results.json` / `analysis.json` of the run described above.
