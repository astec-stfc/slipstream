# GPSR reconstruction of the CLARA S02 bunch, with CSR

Phase-space reconstructions (GPSR) of the bunch at the entrance of CLARA S02 from screen images further down the line, with the VBC on so that CSR
matters. Slipstream's `LinacTracker` is the forward model: it tracks S02 to the screen with RF, wakes and CSR, and it is differentiable, so
GPSR's neural-network beam generator is trained by gradient descent on the image loss. The true bunch is a simulated, non-Gaussian SIMBA bunch
(250 pC, 35.2 MeV, 510 um rms at the entrance of S02, 64 um rms after the VBC).

Two studies use the same machine model and the same fit but different screens, so each recovers different parts of the bunch:

| Study | Screen | Scan | Constrains |
|---|---|---|---|
| [`transverse/`](transverse/README.md) | `CLA-S07-DIA-SCR-04`, straight line | S07 quadrupoles 08 and 09, TDC off and on | the transverse phase space (the vertical plane well, the horizontal only its core); the time profile at S07 |
| [`longitudinal/`](longitudinal/README.md) | `CLA-SP3-DIA-SCR-01`, after the SP3 spectrometer dipole | S07 quadrupoles 06 and 07, TDC off and on | the longitudinal phase space (energy on x, time on y) |

Each study folder holds its entry points (`run_*.py`, `plot_*.py`), its README and its `results/`. The shared code stays here:
`clara_model.py` (lattice, bunch, tracker, screen, `Line` and `Study`), `run_clara_gpsr.py` (the fit) and `plot_results.py` (figures and the held-out
check). A study is a `Study` (the `Line` to the screen, the quadrupoles, the scan and held-out settings, the screen) passed to `main`.

## Cases

Four cases share one machine model and differ in whether CSR is in the data and in the fit:

| Case | Measured images | Model in the fit | Purpose |
|---|---|---|---|
| A | with CSR | with CSR | the full model |
| B | with CSR | without CSR | what ignoring CSR does |
| C | without CSR | without CSR | baseline: GPSR with a correct model, no CSR anywhere |
| D | without CSR | with CSR | a CSR model on a beam that has none |

Besides the held-out images in the fit's own model, `plot_results.py` tracks each reconstructed bunch through the model that made the data. A fit made
with the wrong model can agree with its own data while the bunch is wrong, and this is the number that shows it.

## Requirements

- Slipstream and PyTorch; a CUDA GPU is strongly advised.
- `gpsr` (0.3.0 was used) and its dependency `cheetah-accelerator` (0.8.4), `h5py`, `matplotlib`.
- LAURA and `laura-lattices` (the CLARA lattice), which turn the lattice into Slipstream elements.
- A SIMBA openPMD bunch at the entrance of S02 (`CLA-S02-SIM-APER-01.openpmd.hdf5`). Nothing from the machine is stored in this repository; you pass
  the path.

Run a study from its folder, for example `python run_transverse.py --bunch /path/CLA-S02-SIM-APER-01.openpmd.hdf5 --workdir _cache --out results
--cases ACBD --iterations 1200`, then `python plot_transverse.py` with the same `--bunch`, `--workdir` and `--out`. `--smoke` runs the whole chain
on the CPU in a few minutes. `--workdir` is where LAURA writes the lattice and the wake tables. The reported results use 1200 iterations and the
other defaults: 8192 particles in float64, Adam at 5e-3 with a cosine schedule, 1% image noise, one seed. A case takes 17 to 23 minutes on an RTX 3050
(8 GB) with each step captured as one CUDA graph.

## Modelling choices (both studies)

- The lattice comes from LAURA's elegant export of S02, L02, S03, L03, S04, L4H, S05, VBC, S06, L04 and S07 (travelling-wave cavities with body focusing
  and longitudinal and transverse wakes; CSR bends). S02 to L04 is tracked once per step and only the line from the start of S07 to the screen is repeated
  for each scan row.
- The VBC dipole angle is set to 0.1254 rad, calibrated so that the bunch length after the VBC matches the SIMBA bunch (64.7 um rms) without CSR.
- The elegant edge model 3 is mapped to the first-order edge (1); synchrotron radiation and incoherent synchrotron radiation are off.
- The exported TDC has no voltage and no deflection plane: the examples deflect vertically and set the phase to the zero crossing at the centre of the bunch.
- The quadrupoles of S06 and S07 keep the lattice values except the scanned ones; nothing is re-matched for this bunch.
- Cavity sub-kicks are cut from 261 to 360 down to 45 and the CSR kick length in the drifts after the dipoles is 0.05 m (0.01 m as exported) to fit in 8 GB.
  Finer settings were not run in the full fit.
- The fiducial time, the wake window and the TDC phase are fixed once from the true bunch, as an operator would set them; wake-window overflow is monitored
  and was zero.
- The part of the line repeated for each scan row (S07, and SP3 in the longitudinal study) is tracked without CSR; only S02 to L04, which holds the VBC, has
  it. In the longitudinal study this leaves out the CSR of the SP3 dipole: in a test with CSR in that part (4096 particles) it shifted the mean energy by
  -5e-4 and changed the energy of individual particles by 4.5% of the rms; the CSR of the VBC does 4.3%. Truth and fit both leave it out.
- The fit does not know the true bunch: only the images, the lattice and the machine settings.
