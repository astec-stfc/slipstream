# CLARA HRRG photocathode: Schottky scan and space-charge-limited emission

`schottky_scan.py` compares slipstream's opt-in photocathode emission model
([`Photocathode`](../../docs/modules/tracking/native/photocathode.md)) with a measured Schottky scan of the CLARA high
repetition rate gun, then scans the extracted charge against laser energy for several spot sizes.

| File | Content |
| --- | --- |
| `104459_Schottky_approx_crestingData.h5` | measured scan: `xData` gun phase [deg], `yData` charge [pC], `yStd` [pC] |
| `Example_Shottky_Scan.jpg` | a plot of this scan with the phase axis shifted by +167 deg (0 at the start of the plateau) |
| `CLA-VCA-DIA-CAM-01_2022-04-29_10-14-01UTC_1_images.hdf5` | camera frame of the laser at the virtual cathode (`Capture000001`, 188.5 px/mm, x along the columns, y along the rows); the attributes hold the machine settings at the capture: laser energy readback, gun solenoid current, charge monitor, gun phase |
| `generator.in` | the laser profile: the transverse gaussian cut on an ellipse fitted to that frame, an assumed 2 ps flat-top and 0.62 eV emission |
| `fit_laser_image.py` | fits the frame and prints the `generator.in` lines |
| `schottky_scan.py` | the fit, the tracked scan and the saturation scan |
| `reference/` | outputs of both scripts at their defaults (CUDA, float32, 131072 macroparticles) |

```bash
python examples/photocathode/fit_laser_image.py                    # about 10 seconds, no GPU
python examples/photocathode/schottky_scan.py                      # about 9 minutes on a laptop GPU (RTX 3050)
python examples/photocathode/schottky_scan.py --dt-ceiling-ps 2    # about 3 minutes, charges within 1.2 pC rms
python examples/photocathode/schottky_scan.py --quick              # coarser, 600 macroparticles
python examples/photocathode/schottky_scan.py --only saturation --no-plots   # about 6 minutes, needs a Schottky run first
python examples/photocathode/schottky_scan.py --gun-field-MV 120 --solenoid-T 0.345   # LAURA's design values
python examples/photocathode/schottky_scan.py --qe 1e-4            # fix the QE and fit the laser energy instead
```

The tracking is `torch.compile`d on CUDA (`--no-compile` for the eager path, about 10 times slower per step; `--warp` adds the
Warp kernels, which cost more than they save on this GPU). `--dt-ceiling-ps 2` lets the step grow with the space-charge
frequency after emission, about 4 times fewer steps; against the fixed 0.4 ps step the monitor charge changes by 1.2 pC rms
(at most 6.7 pC in the sparse tail of the cut-off, +1 pC in the ramp).

It needs the optional `laura` and `laura_lattices` packages (CLARA gun, solenoid and apertures) and, for plots,
`slipstream-accel[plot]`. Without CUDA use `--device cpu` (much slower). `--generator`, `--wavelength-nm`, `--laser-energy-uJ`,
`--qe`, `--beta`, `--gun-field-MV`, `--solenoid-T`, `--solenoid-T-per-A` and `--monitor-z` replace the assumed laser,
photocathode, gun, solenoid and monitor plane.

## Laser spot

`fit_laser_image.py` fits the virtual-cathode frame (captured 2022-04-29 10:14 UTC, the day of the Schottky scan) to a 2D
gaussian of rms `sig_x`, `sig_y` cut on the ellipse of semi-axes `C_sig_x sig_x` and `C_sig_y sig_y`, which is `Dist_x = Dist_y
= 'g'` with the generator's native `Cut_shape = 'ellipse'` (`C_sig` is the n-sigma cut). The fit includes an edge blur that the
generator's sharp cut leaves out. The frame is a clipped gaussian: an uncut gaussian leaves a 42 % larger residual (40.0
against 28.1 counts rms on a 967 count peak).

| | x | y |
| --- | --- | --- |
| `sig` [mm] | 0.564 +- 0.036 | 0.350 +- 0.014 |
| `C_sig`, the n-sigma cut | 1.60 +- 0.11 | 2.37 +- 0.10 |
| cut at [mm] | +-0.905 | +-0.832 |
| rms size of the cut distribution [mm] | 0.414 | 0.310 |

The uncertainties are a block jackknife, which includes the hot spots in the laser profile. With the right edge shape the
cuts are well constrained: `C_sig_x` of 1.2 or 2.0 leaves the residual 6 % or 14 % larger, `C_sig_y` of 2.0, 3.0 or no cut
5 %, 5 % or 13 % larger. The edge blur is 0.057 mm rms (imaging and diffraction). The rms sizes of the cut distribution match
the second moments of the image (0.418 x 0.312 mm) to 1 %. The spot centre is at (-1.09, +0.44) mm from the sensor centre;
that is not applied (`x_off`, `y_off` stay 0) because the position of the cathode axis on the sensor is not known.

ASTRA's `'g'` cuts x and y independently, so its cut is a rectangle, and the image edge is round. The best rectangle
(`sig` 0.565 and 0.344 mm, `C_sig` 1.55 and 2.32) leaves 31.6 counts rms, has rms sizes 3 % above the image's and puts about
5 % of its particles in corners outside the ellipse. Delete the `Cut_shape` line of `generator.in` and use those values for an
ASTRA run. In the scan the shape matters little: against the rectangle the ellipse changes the monitor charge by less than
0.01 pC through the plateau and the field-limited flank and by 1.3 pC rms (2.5 pC at most) in the aperture-limited low-phase end.

![Laser spot fit](reference/laser_fit.png)

## What the scan contains

The laser is fixed and the gun phase is scanned, so the charge combines three things.

- **Emission.** The cathode field is `E0 sin` of the RF phase. Copper at 266 nm sits just above its Schottky-lowered
  threshold, so the quantum efficiency rises steeply with field: charge follows the field through the plateau at the
  field maximum and falls to zero at the field's zero crossing (scan phase -76 deg).
- **Space charge.** At 120 pC the field of the emitted cloud cuts the emission at the field maximum by 2.9 % in the
  measured spot at 105 MV/m.
- **Transport.** Electrons emitted late in the RF half-cycle leave the gun with low energy and a large divergence. They
  fall back onto the cathode or are clipped by the 10 mm L01 aperture at z = 1.0577 m, which gives the measured ramp,
  shoulder, dip and satellite peak at the low-phase end (-170 to -200 deg).

The scan phase runs opposite to slipstream's gun phase offset `d` from the max-energy-gain crest: `phase = x0 - d`.

## Procedure

1. The field-limited flank (-170 to -70 deg) involves no transport, so the Dowell-Schmerge law with the Schottky-lowered
   work function is fitted there directly (work function, laser energy x QE, phase origin `x0`, monitor baseline).
2. One tracked run at the field maximum corrects the laser energy x QE for the space-charge-limited emission, so that the
   tracked plateau matches the fitted one.
3. The laser energy is the energy-meter readback stored with the camera frame (`--laser-energy-uJ` replaces it) and the QE
   is what remains of the product. `--qe` fixes the QE instead and the energy is what remains.
4. The whole scan is tracked: emission with space charge, fall-back onto the cathode and apertures, charge counted at a
   monitor plane `--monitor-z` (default 1.1 m, past the L01 aperture).

## Results

Machine settings are the current best fit, 105 MV/m at the cathode and a 0.310 T gun solenoid (LAURA's design values are
120 MV/m and 0.345 T).

Field-limited flank (emission law only, `beta = 1`): work function **4.61 eV**, phase origin `x0 = -129.2 deg` (the scan
phase of the gun's max-energy-gain crest, which moves with the gun field), monitor baseline 2.4 pC, chi2 per point 1.37.
With the 0.539 uJ laser energy of the camera frame the quantum efficiency is **1.05e-3** at the 105 MV/m cathode field
(laser energy x QE = 5.6e-10 J once the 2.9 % space-charge reduction is included). The 4.61 eV is the effective work function
at this photon energy and field; the textbook value for polycrystalline copper is 4.6-4.65 eV (Dowell and Schmerge tabulate
4.31 eV for atomically clean copper). The fit constrains only `hw - phi_w` together with `beta E`: a field enhancement
`beta = 2` gives 4.59 eV and `beta = 4` 4.55 eV. The sharp zero-crossing edge limits the laser pulse: chi2 per point is 1.37
for flat-top pulses up to 2 ps, 1.40 at 3 ps, 1.49 at 5 ps, 1.74 at 8 ps and 2.20 at 12 ps.

Tracked scan (`reference/schottky_scan.png`): over the plateau, the field-limited flank and the zero crossing (-164 to
-72 deg, 23 tracked phases) the model differs from the measurement by 1.4 pC rms (3.2 pC at most, mean -0.1 pC). The
aperture-limited low-phase end (-168 to -204 deg) reproduces the ramp, the shoulder, the dip and the satellite peak
qualitatively (24 pC rms, 2 deg steps). The ramp is aligned: its 100 pC point is at -170.0 deg against -171.8 measured and
its 40 pC point at -180.5 against -180.1. The dip and the satellite peak are not: 26 pC at -187.2 deg against 6.5 pC at
-191.1 deg, and 78 pC at -191.2 deg against 34 pC at -196.1 deg. Those depend on the alignment of the beam on the 10 mm
aperture, the transverse emittance and the pulse length, none of which is fitted here (the laser spot shape hardly matters, see
above).

Gun field and solenoid are degenerate in the ramp. Its 100 pC point moves by +1.9 deg per +0.01 T of solenoid field (scans at
100 MV/m with 0.305 to 0.345 T) and by -1.0 deg per +1 MV/m of gun field (scans at 0.310 T with 100, 105 and 110 MV/m), so a
gun field 1 MV/m higher needs a solenoid 0.0053 T stronger to keep the ramp. The camera frame stores the gun solenoid
current, 139.85 A (`CLA-GUN-MAG-SOL-02`), which the 2022 calibration of about 0.0022 T/A turns into 0.308 T; that is the
fitted 0.310 T to within what the ramp resolves (about 0.01 T), and the fitted field implies 0.00222 T/A. The calibration
changed later and is not yet narrowed down: `--solenoid-T-per-A` takes the field from the stored current with a calibration of
your choice instead of `--solenoid-T`.

![Schottky scan](reference/schottky_scan.png)

Saturation (`reference/saturation_scan.png`; round gaussian spots cut at 3 sigma on a circle, not the measured spot, with the
2 ps pulse of `generator.in`): at the field maximum a pulse extracts the unscreened photoemission only while the laser's
areal density stays well below the Gauss density `eps0 E`. At 157 pC unscreened (1.3 times the measured plateau) a 0.5 mm rms
spot extracts 99 %, 0.25 mm 92 %, 0.15 mm 79 %, 0.1 mm 62 % and 0.05 mm 31 %; at 2 nC unscreened they extract 76, 43, 22, 12
and 4 %. The dotted lines are the Gauss-law cap of an instantaneous sheet. The tracked charge lies below it at moderate
overdrive (the screened surface field also lowers the Schottky yield) and, for the smallest spots, above it at strong
overdrive, where the cloud leaves the surface during the pulse.

![Saturation](reference/saturation_scan.png)

## Assumptions to replace

- Laser: the transverse profile is measured (see Laser spot); the 2 ps flat-top and the 0.62 eV emission energy
  (`generator.in`) and 266 nm are assumed. The emission flank limits the pulse length to a few ps; the spot size only
  weakly (the laser energy absorbs the few-percent space-charge reduction).
- Laser energy and quantum efficiency: the energy is the readback `CLA-LAS-DIA-EM-06:E_RB` = 0.539 uJ of the camera frame,
  taken as the energy at the cathode. If the meter reads before losses in the transport, or the laser drifted between the
  capture and the scan, the QE scales inversely. The scan holds 94.2 +- 1.0 pC at the capture's gun phase (-125.09 deg,
  if the scan's axis is that same LLRF phase) against the 99.7 pC of the capture's charge monitor, so the energy did not
  change by more than about 6 %.
- Gun: 105 MV/m peak field and 0.310 T solenoid (best fit, see Results), the LAURA CLARA field map and apertures. The charge
  monitor is taken to lie downstream of the 10 mm L01 aperture (the WCM at 0.676 m would not see it).
