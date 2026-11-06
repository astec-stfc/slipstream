# CLARA HRRG photocathode: Schottky scan and space-charge-limited emission

`schottky_scan.py` compares slipstream's opt-in photocathode emission model
([`Photocathode`](../../docs/modules/tracking/native/photocathode.md)) with a measured Schottky scan of the CLARA high
repetition rate gun, then scans the extracted charge against laser energy for several spot sizes. The gun defaults to
the best point of `fit_structure.py`: 109 MV/m at the cathode, a 0.3053 T solenoid, an RF flatness of 0.942 and a
solenoid offset by 0.87 mm and tilted by -0.77 mrad, with the S01 correctors steering the beam as the operators do.

| File | Content |
| --- | --- |
| `104459_Schottky_approx_crestingData.h5` | measured scan: `xData` gun phase [deg], `yData` charge [pC], `yStd` [pC] |
| `Example_Shottky_Scan.jpg` | a plot of this scan with the phase axis shifted by +167 deg (0 at the start of the plateau) |
| `CLA-VCA-DIA-CAM-01_2022-04-29_10-14-01UTC_1_images.hdf5` | camera frame of the laser at the virtual cathode (`Capture000001`, 188.5 px/mm, x along the columns, y along the rows); the attributes hold the machine settings at the capture: laser energy readback, gun solenoid current, charge monitor, gun phase |
| `generator.in` | the laser profile: the transverse gaussian cut on an ellipse fitted to that frame, an assumed 2 ps flat-top and 0.62 eV emission |
| `fit_laser_image.py` | fits the frame and prints the `generator.in` lines |
| `schottky_scan.py` | the fit, the tracked scan, the saturation scan and the transmission function |
| `fit_structure.py` | fits the gun field, solenoid field, RF flatness and solenoid misalignment to the ramp, dip and satellite |
| `reference/` | outputs of the scripts at their defaults (CUDA, float32, 131072 macroparticles), including `transmission.csv` and `structure_fit.json`; `schottky_scan_aligned.csv` is the scan of the aligned machine (below) |

```bash
python examples/photocathode/fit_laser_image.py                    # about 10 seconds, no GPU
python examples/photocathode/schottky_scan.py                      # about 9 minutes on a laptop GPU (RTX 3050)
python examples/photocathode/schottky_scan.py --dt-ceiling-ps 2    # about 3 minutes, charges within 1.2 pC rms
python examples/photocathode/schottky_scan.py --quick              # coarser, 600 macroparticles
python examples/photocathode/schottky_scan.py --only saturation --no-plots   # about 6 minutes, needs a Schottky run first
python examples/photocathode/schottky_scan.py --only transmission  # about 3 minutes: T(phase) from tracers emitted uniformly in phase
python examples/photocathode/schottky_scan.py --gun-field-MV 120 --solenoid-T 0.345 --flatness 0.941 \
    --solenoid-offset-mm 0 --solenoid-tilt-mrad 0       # LAURA's design machine, aligned
python examples/photocathode/schottky_scan.py --no-steering          # the misaligned solenoid with no correctors
python examples/photocathode/schottky_scan.py --qe 1e-4            # fix the QE and fit the laser energy instead
```

The tracking is `torch.compile`d on CUDA (`--no-compile` for the eager path, about 10 times slower per step; `--warp` adds the
Warp kernels, which cost more than they save on this GPU). `--dt-ceiling-ps 2` lets the step grow with the space-charge
frequency after emission, about 4 times fewer steps; against the fixed 0.4 ps step the monitor charge changes by 1.2 pC rms
(at most 6.7 pC in the sparse tail of the cut-off, +1 pC in the ramp).

It needs the optional `laura` and `laura_lattices` packages (CLARA gun, solenoid and apertures) and, for plots,
`slipstream-accel[plot]`. Without CUDA use `--device cpu` (much slower). `--generator`, `--wavelength-nm`, `--laser-energy-uJ`,
`--qe`, `--beta`, `--gun-field-MV`, `--solenoid-T`, `--solenoid-T-per-A`, `--flatness` (the full cell's peak field over the
cathode's; LAURA's map has 0.941), `--solenoid-offset-mm` (along x), `--solenoid-tilt-mrad` (in the x-z plane),
`--steering`/`--no-steering`, `--reference-offset-deg` and `--monitor-z` replace the assumed laser, photocathode, gun,
solenoid, steering and monitor plane.

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
- **Space charge.** At 120 pC the field of the emitted cloud cuts the emission at the field maximum by 2.8 % in the
  measured spot at 109 MV/m.
- **Transport.** Electrons emitted late in the RF half-cycle leave the gun with low energy and a large divergence. They
  fall back onto the cathode or are clipped by the L01 aperture at z = 1.0577 m (LAURA's simulation aperture: a square of
  +-10 mm half-width, so a 10 mm radius circle would clip more), which gives the measured ramp,
  shoulder, dip and satellite peak at the low-phase end (-170 to -200 deg). A solenoid that is not on the gun's axis
  kicks the slow electrons of those phases harder than the correctors, set at one phase, can cancel.

The scan phase runs opposite to slipstream's gun phase offset `d` from the max-energy-gain crest: `phase = x0 - d`.

## Procedure

1. The field-limited flank (-170 to -70 deg) involves no transport, so the Dowell-Schmerge law with the Schottky-lowered
   work function is fitted there directly (work function, laser energy x QE, phase origin `x0`, monitor baseline).
2. One tracked run at the field maximum corrects the laser energy x QE for the space-charge-limited emission, so that the
   tracked plateau matches the fitted one.
3. The laser energy is the energy-meter readback stored with the camera frame (`--laser-energy-uJ` replaces it) and the QE
   is what remains of the product. `--qe` fixes the QE instead and the energy is what remains.
4. With a misaligned solenoid the two S01 correctors (z = 0.335 and 0.814 m) are set to centre the beam on the BPM and the
   screen at `--reference-offset-deg` (a few tracking runs) and then stay fixed for the whole scan.
5. The whole scan is tracked: emission with space charge, fall-back onto the cathode and apertures, charge counted at a
   monitor plane `--monitor-z` (default 1.1 m, past the L01 aperture).

## Results

Machine settings are the best point of `fit_structure.py` (see the fit below): 109 MV/m at the cathode, a 0.3053 T gun
solenoid, flatness 0.942, and the solenoid offset by 0.87 mm and tilted by -0.77 mrad with the S01 correctors set at the
capture's phase. LAURA's design machine is 120 MV/m, 0.345 T, flatness 0.941 and aligned.

Field-limited flank (emission law only, `beta = 1`): work function **4.61 eV**, phase origin `x0 = -129.9 deg` (the scan
phase of the gun's max-energy-gain crest, which moves with the gun field), monitor baseline 2.4 pC, chi2 per point 1.37.
With the 0.539 uJ laser energy of the camera frame the quantum efficiency is **1.05e-3** at the 109 MV/m cathode field
(laser energy x QE = 5.6e-10 J once the 2.8 % space-charge reduction is included). The 4.61 eV is the effective work function
at this photon energy and field; the textbook value for polycrystalline copper is 4.6-4.65 eV (Dowell and Schmerge tabulate
4.31 eV for atomically clean copper). The fit constrains only `hw - phi_w` together with `beta E`: a field enhancement
`beta = 2` gives 4.58 eV and `beta = 4` 4.55 eV. The sharp zero-crossing edge limits the laser pulse: chi2 per point is 1.37
for flat-top pulses up to 2 ps, 1.40 at 3 ps, 1.49 at 5 ps, 1.73 at 8 ps and 2.20 at 12 ps.

Tracked scan (`reference/schottky_scan.png`): over the plateau, the field-limited flank and the zero crossing (-164 to
-72 deg, 23 tracked phases) the model differs from the measurement by 1.3 pC rms (3.1 pC at most, mean -0.2 pC). The ramp
and the shoulder behind it are reproduced: the 100 pC point is at -171.1 deg against -171.8 measured and the 40 pC point at
-180.2 against -180.1, and over -204 to -168 deg the difference is 9.7 pC rms (24 pC for the aligned machine below). The
satellite is not: the model falls to 3 pC by -196 deg where the measurement has a satellite peak of 31 pC (28 pC is the
largest difference), after matching the measured dip (9 pC against 6.5 pC at -191 deg). That needs something the
model does not have, see the fit below.

The steering the model needs for the misaligned solenoid is Bx = 3.76 mT and By = -2.00 mT over the 3 cm of the first
corrector (z = 0.335 m) and Bx = -0.24 mT and By = +0.44 mT over the 7.5 cm of the second (z = 0.814 m). The camera frame
stores the S01 corrector currents (`CLA-S01-MAG-HCOR-01/02` -3.69 and -1.07 A, `VCOR-01/02` 0.01 and 0.70 A), and their
pattern is different: the vertical corrector at z = 0.335 m (Bx) is idle where the model drives it hardest, and the horizontal
ones (By) have the same sign where the model's have opposite signs (if their polarities are alike). So the direction of the
misalignment assumed here, an offset along x with a tilt in the x-z plane, is probably not the real one. With the
correctors' calibration in T m/A the stored currents would constrain it directly.

Gun field and solenoid are degenerate in the ramp. Its 100 pC point moves by +1.9 deg per +0.01 T of solenoid field (scans
of the aligned machine at 100 MV/m with 0.305 to 0.345 T) and by -1.0 deg per +1 MV/m of gun field (at 0.310 T with 100, 105 and
110 MV/m), so a gun field 1 MV/m higher needs a solenoid 0.0053 T stronger to keep the ramp. The camera frame stores the gun
solenoid current, 139.85 A (`CLA-GUN-MAG-SOL-02`), which the 2022 calibration of about 0.0022 T/A turns into 0.308 T; the
fitted 0.3053 T implies 0.00218 T/A. The calibration changed later and is not yet narrowed down: `--solenoid-T-per-A`
takes the field from the stored current with a calibration of your choice instead of `--solenoid-T`.

![Schottky scan](reference/schottky_scan.png)

Saturation (`reference/saturation_scan.png`; round gaussian spots cut at 3 sigma on a circle, not the measured spot, with the
2 ps pulse of `generator.in`): at the field maximum a pulse extracts the unscreened photoemission only while the laser's
areal density stays well below the Gauss density `eps0 E`. At 157 pC unscreened (1.3 times the measured plateau) a 0.5 mm rms
spot extracts 99 %, 0.25 mm 92 %, 0.15 mm 80 %, 0.1 mm 63 % and 0.05 mm 32 %; at 2 nC unscreened they extract 77, 44, 23, 13
and 4 %. The dotted lines are the Gauss-law cap of an instantaneous sheet. The tracked charge lies below it at moderate
overdrive (the screened surface field also lowers the Schottky yield) and, for the smallest spots, above it at strong
overdrive, where the cloud leaves the surface during the pulse.

![Saturation](reference/saturation_scan.png)

## Transmission function

`--only transmission` tracks 131072 tracer electrons emitted uniformly in RF phase over -60 to 85 deg from the crest (one
run, about 2 minutes, no photocathode weighting and no space charge) and bins the ones that reach the monitor by their
emission phase: `reference/transmission.csv` and `.png`. For a short pulse the scan is the emitted charge (smooth: 97 to
121 pC from 36 to 72 deg) times this transmission averaged over the pulse, so all the structure of the aperture-limited end
is in it. `scan_from_transmission` makes that product, and `transmission_function` is what `fit_structure.py` and any
other study of the gun's acceptance can call.

It is the transmission of a bunch of vanishing charge, and how much the 100 pC bunch differs depends on the machine. No
uniform-phase run can hold the space charge of a 2 ps bunch (its neighbours in the run would interact). For the aligned
machine (`reference/schottky_scan_aligned.csv`: 105 MV/m, 0.310 T, flatness 0.941, solenoid on axis) the space charge
costs 13 to 40 pC at the ramp, the dip and the satellite. Monitor charge in pC at gun phase offsets, tracked with 100 pC
against the same tracking without space charge (which the transmission function reproduces to 1 to 3 pC):

| offset [deg] | 40 | 44 | 50 | 56 | 58 | 62 | 66 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| aligned, tracked, space charge | 102 | 77 | 42 | 24 | 24 | 82 | 41 |
| aligned, no space charge | 115 | 103 | 75 | 61 | 67 | 107 | 69 |
| default machine, tracked, space charge | 106 | 76 | 39 | 20 | 18 | 6 | 2 |
| default machine, no space charge | 113 | 92 | 51 | 22 | 15 | 0 | 0 |

For the aligned machine space charge also moves the 100 pC point of the ramp 4 deg earlier (44.5 to 40.4 deg) and barely
moves the dip and the satellite (55.8 and 61.4 deg without it, 58 and 62 with it, on the 2 deg grid), so those positions can
be studied with the transmission function and the heights need the tracked scan. For the default machine (the plot below)
the transmission is a smooth fall with no dip or satellite, the scan without space charge is 4.9 pC above the tracked one on
average (17 pC at most) and the ramp's 100 pC point moves 1.7 deg (42.6 to 40.9 deg): the steering walk-off, not space
charge, shapes it. The spike at 72 deg is a handful of electrons in a single bin. A scan at low laser energy (1 to 10 pC)
should approach the transmission function directly, which would separate the optics from the space charge.

![Transmission](reference/transmission.png)

## Fit of the gun to the ramp, dip and satellite

`fit_structure.py` fits the cathode field, the solenoid field, the RF flatness (the full cell's peak field over the
cathode's, 0.941 in LAURA's map) and a solenoid offset along x and tilt to the measured scan between -204 and -160 deg. Each
trial tracks the scan phases (2 deg steps) to a screen just before the L01 aperture, counts the charge inside the aperture's
square, and refits the phase origin and the baseline from the field-limited flank, which move with the gun field and the
flatness; the laser energy is the camera frame's readback. A misaligned solenoid is corrected as the operators do: two
steering kicks at the S01 correctors (z = 0.335 and 0.814 m) centre the beam on the BPM and the screen at the capture's gun
phase and are then held at every phase (thin kicks applied to the screen data here; `schottky_scan.py` tracks the same
correctors as magnets and gets the same scan). A trial takes one to two minutes on a laptop GPU, so a fit takes hours, and the
landscape is rough: the satellite is a sharp feature that jumps between neighbouring trials (a flatness of 0.980 against
0.942 raised the residual from 15 to 98 with the rest unchanged), so the formal errors mean little and the valley is better
read from the points the optimiser visited near the minimum.

| | aligned (`reference/schottky_scan_aligned.csv`) | best point found (the defaults of `schottky_scan.py`) |
| --- | --- | --- |
| cathode field [MV/m] | 105 | 109.0 |
| solenoid peak field [T] | 0.310 | 0.3053 |
| flatness | 0.941 | 0.942 |
| solenoid offset [mm] | 0 | 0.87 |
| solenoid tilt [mrad] | 0 | -0.77 |
| residual chi2 per point | 112 | 14.6 |

The visited points within 20 % of the best residual had a cathode field of 107.8 to 109.7 MV/m, a solenoid field of 0.304 to
0.307 T, an offset of 0.74 to 0.9 mm and a flatness of 0.936 to 0.973. The ramp and the shoulder behind it are reproduced:
the 100 pC point is at -171.5 deg (measured -171.8, the aligned run -170.0) and the 40 pC point at -180.2 deg (measured
-180.1); the difference over -190 to -165 deg is 3.1 pC rms against 8.5 pC. The dip and the satellite are not: the best
point removes the satellite instead of reproducing it, and keeps 9 to 10 pC at -193 deg where the measurement has a dip of
6.5 pC at -191 deg and a satellite of 31 pC at -196 deg. Over -205 to -190 deg the difference is 13.6 pC rms against 33.2 pC for the aligned run,
and that gain comes from the missing satellite. The solenoid field of 0.3053 T is the stored current of 139.85 A times
0.00218 T/A, which is the 2022 calibration of 0.0022 T/A to within 1 %.

![Structure fit](reference/structure_fit.png)

What would still have to come from outside the model: RF coupler kicks (dipole and quadrupole, which depend on the RF
phase at which a slow electron crosses the coupler cell), the true geometry of the L01 aperture and pipe, and the laser's
time structure.

## Assumptions to replace

- Laser: the transverse profile is measured (see Laser spot); the 2 ps flat-top and the 0.62 eV emission energy
  (`generator.in`) and 266 nm are assumed. The emission flank limits the pulse length to a few ps; the spot size only
  weakly (the laser energy absorbs the few-percent space-charge reduction).
- Laser energy and quantum efficiency: the energy is the readback `CLA-LAS-DIA-EM-06:E_RB` = 0.539 uJ of the camera frame,
  taken as the energy at the cathode. If the meter reads before losses in the transport, or the laser drifted between the
  capture and the scan, the QE scales inversely. The scan holds 94.2 +- 1.0 pC at the capture's gun phase (-125.09 deg,
  if the scan's axis is that same LLRF phase) against the 99.7 pC of the capture's charge monitor, so the energy did not
  change by more than about 6 %.
- Gun: 109 MV/m peak field, 0.3053 T solenoid, flatness 0.942 and a solenoid offset of 0.87 mm and tilt of -0.77 mrad (the
  best point of `fit_structure.py`, not measurements: the misalignment in particular compensates something the model lacks,
  see the fit), the LAURA CLARA field map and apertures. The charge monitor is taken to lie downstream of the L01
  aperture (the WCM at 0.676 m would not see it).
