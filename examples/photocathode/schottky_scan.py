"""CLARA HRRG Schottky scan and laser-energy/spot-size saturation with the native photocathode emission model.

Needs the optional LAURA packages (``laura``, ``laura_lattices``) for the CLARA gun, solenoid and apertures.

1. Schottky scan: charge reaching a monitor plane against gun phase, compared with the measured scan
   ``104459_Schottky_approx_crestingData.h5``. The field-limited flank of the scan (emission late in the laser's RF
   half-cycle) fixes the effective work function, the quantum-efficiency x laser-energy scale, the phase origin and the
   monitor baseline through the same quantum-efficiency law the tracker uses. Tracking then adds space-charge-limited
   emission, electrons falling back onto the cathode and aperture losses.
2. Saturation: extracted charge against laser energy for several rms spot sizes at the gun phase of maximum field. The
   spots are round gaussians cut at 3 sigma on a circle whatever ``generator.in`` holds; its pulse shape is kept.
3. Transmission (``--only transmission``): the fraction of electrons emitted at each gun phase that reach the monitor,
   from one run of tracers emitted uniformly in RF phase without space charge. It is the scan's structure at the
   aperture-limited end for a bunch of vanishing charge; ``fit_structure.py`` fits the gun to that end of the scan.

The laser profile comes from ``generator.in``: the transverse gaussian cut on an ellipse is fitted to a virtual-cathode
image by ``fit_laser_image.py``, the pulse length and emission energy are assumed, not measured. The laser energy is the
energy-meter readback stored with that image (``--laser-energy-uJ`` replaces it, ``--qe`` fixes the quantum efficiency
instead and fits the energy), so the fit gives the quantum efficiency.

The gun defaults to the best point of ``fit_structure.py``: the cathode field, the solenoid field, the flatness of the
RF field and the offset and tilt of the gun solenoid (``--gun-field-MV``, ``--solenoid-T``, ``--flatness``,
``--solenoid-offset-mm``, ``--solenoid-tilt-mrad``). A misaligned solenoid is steered as the operators steer it: the two
S01 correctors are set to centre the beam on the BPM and the screen at ``--reference-offset-deg`` and then stay
fixed (``--no-steering`` leaves the misalignment uncorrected). Pass your own laser profile with ``--generator``. Install
``slipstream-accel[plot]`` for figures or use ``--no-plots``.
"""

import argparse
import csv
import dataclasses
import json
import math
import os
import tempfile
import time
import warnings
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.optimize import least_squares

from slipstream.constants import E_CHARGE_C, EPS0_F_PER_M
from slipstream.tracking.native.generator import read_generator_input
from slipstream.tracking.native.injector import Corrector, SpaceChargeSettings
from slipstream.tracking.native.laura_injector import from_laura
from slipstream.tracking.native.photocathode import Photocathode

try:
    import laura_lattices.CLARA as CLARA
except ImportError:
    CLARA = None

HERE = Path(__file__).resolve().parent
DATA = HERE / "104459_Schottky_approx_crestingData.h5"
IMAGE = HERE / "CLA-VCA-DIA-CAM-01_2022-04-29_10-14-01UTC_1_images.hdf5"
SECTIONS = ("HRG1", "S01", "L01")
GUN_FIELD_MV = (
    109.0  # peak gun field of the current best fit (LAURA's design value: 120)
)
SOLENOID_T = 0.3053  # gun solenoid peak field of the current best fit (LAURA's: 0.345)
FLATNESS = 0.942  # full cell's peak field over the cathode's, current best fit (LAURA's map: 0.941)
SOLENOID_OFFSET_MM = 0.87  # gun solenoid axis offset along x of the current best fit
SOLENOID_TILT_MRAD = -0.77  # gun solenoid axis tilt in the x-z plane of the current best fit
REFERENCE_OFFSET_DEG = -4.0  # gun phase offset from the crest at which the correctors centre the beam
CORRECTORS = ((0.335, 0.03), (0.8136, 0.075))  # centre and length [m] of CLA-S01-MAG-HVCOR-01 and -02
OBSERVATION_Z = (0.894, 0.9803)  # the BPM and the screen the operators centre the beam on
SOLENOID_T_PER_A = (
    0.0022  # experimental calibration of the gun solenoid in 2022 (it changed later)
)
QE_START = 1e-4  # quantum efficiency while fitting: only its product with the laser energy is fixed
FIELD_FLANK = (
    -170.0,
    -70.0,
)  # measured phases used to fit the emission law: transport is complete there
APERTURE_FLANK = (
    36.0,
    72.0,
)  # gun phase offsets [deg] where the L01 aperture and the cut-off shape the scan
SPOT_CUT = 3.0  # the saturation scan's round gaussian spots are cut at this many sigma
TRANSMISSION_WINDOW = (
    -60.0,
    85.0,
)  # gun phase offsets [deg] emitted by the uniform-phase run


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--only",
        choices=["schottky", "saturation", "both", "transmission"],
        default="schottky",
    )
    p.add_argument(
        "--output-dir", default=str(HERE.parents[1] / ".cache" / "photocathode")
    )
    p.add_argument(
        "--generator",
        default=str(HERE / "generator.in"),
        help="ASTRA generator.in of the laser profile",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--particles", type=int, default=131072)
    p.add_argument(
        "--dtype", default="float32" if torch.cuda.is_available() else "float64"
    )
    p.add_argument(
        "--dt-ps",
        type=float,
        default=0.4,
        help="Schottky-scan step (0.1, 0.2 and 0.4 ps agree to 3%%)",
    )
    p.add_argument(
        "--saturation-dt-ps",
        type=float,
        default=0.1,
        help="emission step of the saturation scan",
    )
    p.add_argument(
        "--monitor-z",
        type=float,
        default=1.1,
        help="charge monitor plane [m], past the 10 mm L01 aperture",
    )
    p.add_argument("--wavelength-nm", type=float, default=266.0)
    energy = p.add_mutually_exclusive_group()
    energy.add_argument(
        "--laser-energy-uJ",
        type=float,
        default=None,
        help="laser pulse energy [uJ]: the fit then gives the QE at the gun's peak field "
        "(default: the energy-meter readback stored with the laser camera frame)",
    )
    energy.add_argument(
        "--qe",
        type=float,
        default=None,
        help="QE at the gun's peak field: the fit then gives the laser energy",
    )
    p.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="field enhancement; degenerate with the work function",
    )
    p.add_argument(
        "--gun-field-MV",
        type=float,
        default=GUN_FIELD_MV,
        help="gun peak field [MV/m] (default: the current best fit; LAURA's is 120)",
    )
    p.add_argument(
        "--solenoid-T",
        type=float,
        default=SOLENOID_T,
        help="gun solenoid peak field [T] (default: the current best fit; LAURA's is 0.345)",
    )
    p.add_argument(
        "--flatness",
        type=float,
        default=FLATNESS,
        help="flatness of the gun's RF field, the full cell's peak over the cathode's "
        "(default: the current best fit; LAURA's map has 0.941)",
    )
    p.add_argument(
        "--solenoid-offset-mm",
        type=float,
        default=SOLENOID_OFFSET_MM,
        help="offset of the gun solenoid's axis along x [mm] (default: the current best fit)",
    )
    p.add_argument(
        "--solenoid-tilt-mrad",
        type=float,
        default=SOLENOID_TILT_MRAD,
        help="tilt of the gun solenoid's axis in the x-z plane [mrad] (default: the current best fit)",
    )
    p.add_argument(
        "--steering",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="with a misaligned solenoid, centre the beam on the S01 BPM and screen at --reference-offset-deg "
        "with the two S01 correctors and keep them fixed (default); --no-steering leaves it uncorrected",
    )
    p.add_argument(
        "--reference-offset-deg",
        type=float,
        default=REFERENCE_OFFSET_DEG,
        help="gun phase offset from the crest at which the correctors centre the beam",
    )
    p.add_argument(
        "--solenoid-T-per-A",
        type=float,
        default=None,
        help="take the solenoid field from the current stored with the laser camera frame times "
        "this calibration (0.0022 was measured in 2022) instead of --solenoid-T",
    )
    p.add_argument(
        "--step",
        type=float,
        default=4.0,
        help="gun-phase step [deg]; half of it in the aperture-limited flank",
    )
    p.add_argument(
        "--quick", action="store_true", help="fewer particles and phase points"
    )
    p.add_argument(
        "--transmission-bin-deg",
        type=float,
        default=0.25,
        help="phase bin of the uniform-phase transmission function [deg]",
    )
    p.add_argument("--no-plots", action="store_true")
    p.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=torch.cuda.is_available(),
        help="torch.compile the CUDA step: 10-40x faster than eager, same charges (default with CUDA)",
    )
    p.add_argument(
        "--warp",
        action="store_true",
        help="with --compile, use the Warp CUDA kernels (slower than plain compile on an RTX 3050)",
    )
    p.add_argument(
        "--dt-ceiling-ps",
        type=float,
        default=0.0,
        help="after emission let the step grow with the space-charge frequency up to this [ps] "
        "(0: fixed --dt-ps); 2 ps cuts the steps about 4x",
    )
    return p.parse_args()


def flat_map(machine, ratio, directory):
    """The gun's Ez map with the full cell scaled so that its peak is ``ratio`` of the cathode field, cached on disk.
    Returns its path and the factor that ``MaxE`` needs for the cathode field to be the one asked for: the map is
    normalised to its largest value, which is the full cell's (and negative) when ``ratio`` > 1."""
    if abs(ratio - 1.0) < 2e-3:  # a tie between the cells would let the loader pick either as the largest
        ratio = 1.0 + (2e-3 if ratio >= 1.0 else -2e-3)
    source = Path(machine.cavities[0].file)
    path = Path(directory) / f"{source.stem}_flat{ratio:.4f}.hdf5"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(source, "r") as f:
            z, ez = f["z"][:], f["Ez"][:]
            attrs = {k: dict(f[k].attrs) for k in f}
            root = dict(f.attrs)
        zero = z[np.argmax(np.sign(ez) != np.sign(ez[0]))]  # the half cell ends at the first zero crossing
        full = np.abs(ez[z > zero]).max() / abs(ez[0])
        partial = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
        with h5py.File(partial, "w") as f:
            f.attrs.update(root)
            f["z"] = z
            f["Ez"] = np.where(z > zero, ez * ratio / full, ez)
            for k, a in attrs.items():
                f[k].attrs.update(a)
        os.replace(partial, path)  # parallel runs share the cache
    with h5py.File(path, "r") as f:
        ez = f["Ez"][:]
    return str(path).replace("\\", "/"), float(ez[np.argmax(np.abs(ez))] / ez[0])


def steer_correctors(machine, args):
    """Add the two S01 correctors, set as the operators set them: so that the beam is centred on the BPM and the screen at
    the gun phase ``--reference-offset-deg``. Their fields then stay fixed at every other phase, where the slower beam
    is kicked harder by the misaligned solenoid and the correctors no longer cancel it.

    A corrector is a thin momentum kick, ``dp = q B L``, so the centroid at a plane is linear in the kicks and a
    least-squares solve (repeated to remove what the finite length leaves) gives the fields."""
    cavity = machine.cavities[0]
    phase, machine.correctors = cavity.phase_deg, []
    cavity.phase_deg = args.reference_offset_deg
    kick = np.zeros((2, len(CORRECTORS)))  # du = dp / m [m/s] per plane (x, y) and corrector
    residual = 0.0
    try:
        for _ in range(3):
            mean, weight = [], []
            for z in OBSERVATION_Z:
                out = machine.track(
                    n_particles=min(args.particles, 8000),
                    device=args.device,
                    dtype=getattr(torch, args.dtype),
                    verbose=False,
                    record_every=0,
                    screen_z_m=z,
                    z_stop_m=z + 0.005,
                    use_compile=args.compile,
                    use_warp=args.warp,
                )
                screen = out["screen"]
                q = screen["q"].abs().double().cpu().numpy()
                u = np.sqrt(sum(screen[k].double().cpu().numpy() ** 2 for k in ("ux", "uy", "uz")))
                mean.append([float((screen[k].double().cpu().numpy() * q).sum() / q.sum()) for k in ("x", "y")])
                weight.append(float((q / u).sum() / q.sum()))
            residual = float(np.abs(mean).max())
            matrix = np.array([[(z - zc) * w for zc, _ in CORRECTORS] for z, w in zip(OBSERVATION_Z, weight)])
            for plane in range(2):
                kick[plane] += np.linalg.solve(matrix, -np.array([m[plane] for m in mean]))
            machine.correctors = [
                Corrector(  # dp_x = -q B_y L and dp_y = +q B_x L for a charge q moving along z
                    zc, length, bx_T=machine.mass_kg * kick[1, j] / (machine.q_C * length),
                    by_T=-machine.mass_kg * kick[0, j] / (machine.q_C * length),
                )
                for j, (zc, length) in enumerate(CORRECTORS)
            ]  # fmt: skip
    finally:
        cavity.phase_deg = phase
    print(
        "steering: S01 correctors "
        + ", ".join(f"{c.bx_T * 1e3:+.3f}/{c.by_T * 1e3:+.3f} mT (Bx/By)" for c in machine.correctors)
        + f"; the beam was {residual * 1e3:.2f} mm off centre at the BPM and screen before the last step",
        flush=True,
    )


def build_machine(args, monitor_z, photo, dt_ps):
    """Gun, S01 and the L01 entrance aperture; the L01 cavity and solenoids lie beyond the monitor and are dropped.
    The gun's flatness, the solenoid's offset and tilt and the S01 steering follow ``args``."""
    machine = from_laura(
        CLARA,
        SECTIONS,
        monitor_z + 0.02,
        generator=str(Path(args.generator).resolve()),
        photocathode=photo,
    )
    machine.cavities = [c for c in machine.cavities if c.z_m < monitor_z]
    machine.solenoids = [s for s in machine.solenoids if s.z_m < monitor_z]
    cavity = machine.cavities[0]
    if args.gun_field_MV is not None:
        cavity.e_max_V_per_m = args.gun_field_MV * 1e6
    if args.flatness is not None:
        maps = Path(getattr(args, "output_dir", tempfile.gettempdir())) / "flat_maps"
        cavity.file, factor = flat_map(machine, args.flatness, maps)
        cavity.e_max_V_per_m *= factor
    if args.solenoid_T is not None:
        machine.solenoids[0].b_max_T = args.solenoid_T
    machine.space_charge = (
        SpaceChargeSettings(
            dt_adaptive=True, dt_safety=5e7, dt_ceiling_s=args.dt_ceiling_ps * 1e-12
        )
        if args.dt_ceiling_ps > 0
        else SpaceChargeSettings(dt_adaptive=False)
    )
    machine.dt_s = dt_ps * 1e-12
    machine.auto_phase(
        verbose=False
    )  # the gun's max-energy-gain crest; scan phases are offsets from it
    machine.absolute_phases = True  # keep that crest while the offset changes
    if args.solenoid_offset_mm or args.solenoid_tilt_mrad:  # placed after the phasing: the crest is the aligned gun's
        machine.solenoids[0].x_m = args.solenoid_offset_mm * 1e-3
        machine.solenoids[0].yrot_rad = args.solenoid_tilt_mrad * 1e-3
        if args.steering and monitor_z > OBSERVATION_Z[-1]:  # a shorter machine ends before the correctors
            steer_correctors(machine, args)
    field = machine.fields()[0][0]
    return (
        machine,
        float(field.map.f[0] * field.scale),
        field.omega,
    )  # cathode field amplitude [V/m], angular frequency


class Emission:
    """Emitted charge against gun phase from the applied field alone: no space charge, transport or losses."""

    def __init__(self, machine, peak_field, omega, args):
        self.t = machine.initial_beam(args.particles, "cpu", torch.float64)[
            "emission_time_s"
        ]
        self.crest, self.peak, self.omega = (
            machine.cavities[0].crest_deg,
            peak_field,
            omega,
        )

    def charge_pC(self, photo, deltas):
        theta = (
            torch.deg2rad(
                torch.as_tensor(self.crest + np.asarray(deltas, dtype=float))
            )[:, None]
            + self.omega * self.t
        )
        qe = photo.quantum_efficiency(
            -self.peak * torch.cos(theta)
        )  # extraction field of an electron
        return (photo.photons * E_CHARGE_C * qe.mean(dim=1)).numpy() * 1e12


def fit_emission(emission, photo, x, y, s):
    """Work function, laser energy, phase origin and baseline from the field-limited flank of the measured scan."""
    sel = (x > FIELD_FLANK[0]) & (x < FIELD_FLANK[1])

    def residual(p):
        wf, log_energy, x0, base = p
        trial = dataclasses.replace(
            photo, work_function_eV=wf, pulse_energy_J=math.exp(log_energy)
        )
        return (base + emission.charge_pC(trial, x0 - x[sel]) - y[sel]) / s[sel]

    start = [4.5, math.log(photo.pulse_energy_J), -131.7, 2.5]
    fit = least_squares(
        residual,
        start,
        bounds=([3.9, math.log(1e-8), -150.0, 0.0], [4.9, math.log(1e-3), -110.0, 6.0]),
    )
    wf, log_energy, x0, base = fit.x
    return dict(
        work_function_eV=wf,
        pulse_energy_J=math.exp(log_energy),
        x0=x0,
        baseline_pC=base,
        chi2_per_point=float((fit.fun**2).sum() / sel.sum()),
    )


def track_charge(machine, delta_deg, args, monitor_z, z_stop_m=None):
    machine.cavities[0].phase_deg = float(delta_deg)
    out = machine.track(
        n_particles=args.particles,
        device=args.device,
        dtype=getattr(torch, args.dtype),
        verbose=False,
        record_every=0,
        screen_z_m=monitor_z,
        z_stop_m=z_stop_m,
        use_compile=args.compile,
        use_warp=args.warp,
    )
    return out["emission"], out["screen"]


def capture_settings(path=IMAGE):
    """Machine settings stored as attributes of the laser camera frame: laser energy-meter readback [uJ], gun solenoid
    current [A], gun phase [deg] and charge monitor [pC]. Empty if the frame is not there.
    """
    if not Path(path).is_file():
        return {}
    with h5py.File(path, "r") as f:
        attrs = f["Capture000001"].attrs
        return dict(
            laser_energy_uJ=float(attrs["CLA-LAS-DIA-EM-06:E_RB"]),
            gun_solenoid_A=float(attrs["CLA-GUN-MAG-SOL-02:READI"]),
            gun_phase_deg=float(attrs["CLA-GUN-LRF-CTRL-01:phase"]),
            charge_pC=float(attrs["CLA-S01-DIA-WCM-01:Q"]),
        )


def apply_capture(args, capture):
    """Fill in from the camera frame's settings what the command line left open: the laser energy (unless ``--qe`` fixes
    the quantum efficiency) and, with ``--solenoid-T-per-A``, the solenoid field."""
    if args.solenoid_T_per_A is not None:
        if not capture:
            raise SystemExit(f"--solenoid-T-per-A needs the laser camera frame {IMAGE}")
        args.solenoid_T = capture["gun_solenoid_A"] * args.solenoid_T_per_A
    if capture and args.qe is None and args.laser_energy_uJ is None:
        args.laser_energy_uJ = capture["laser_energy_uJ"]


def split_energy_and_qe(photo, args):
    """The fit fixes only the product of laser energy and QE. Take the energy from ``--laser-energy-uJ`` and give the rest
    to the QE, or take ``--qe`` (``QE_START`` if neither is given) and give the rest to the energy.
    """
    product = photo.pulse_energy_J * photo.qe
    if args.laser_energy_uJ is not None:
        energy = args.laser_energy_uJ * 1e-6
        return dataclasses.replace(photo, pulse_energy_J=energy, qe=product / energy)
    qe = args.qe or QE_START
    return dataclasses.replace(photo, pulse_energy_J=product / qe, qe=qe)


def schottky_scan(args, x, y, s, out_dir):
    photo = Photocathode(
        pulse_energy_J=3e-6,
        wavelength_nm=args.wavelength_nm,
        qe=QE_START,
        work_function_eV=4.5,
        qe_field_V_per_m=120e6,
        field_enhancement=args.beta,
    )
    machine, peak, omega = build_machine(args, args.monitor_z, photo, args.dt_ps)
    photo = dataclasses.replace(photo, qe_field_V_per_m=peak)
    emission = Emission(machine, peak, omega, args)
    crest = machine.cavities[0].crest_deg
    print(
        f"gun crest {crest:.2f} deg, cathode field {peak / 1e6:.1f} MV/m, "
        f"main solenoid {machine.solenoids[0].b_max_T:.3f} T, "
        f"photon {1239.841984 / args.wavelength_nm:.3f} eV",
        flush=True,
    )

    fit = fit_emission(emission, photo, x, y, s)
    print(
        "emission-law fit on the field-limited flank (no transport, no space charge; the energy is for "
        f"QE {QE_START:g}):\n  " + ", ".join(f"{k} = {v:.4g}" for k, v in fit.items()),
        flush=True,
    )
    photo = dataclasses.replace(
        photo,
        work_function_eV=fit["work_function_eV"],
        pulse_energy_J=fit["pulse_energy_J"],
    )

    # one calibration step: the laser energy that gives the fitted plateau once space-charge-limited emission is included
    machine.photocathode = photo
    delta_peak = (
        180.0 - crest + 180.0
    ) % 360.0 - 180.0  # gun phase offset where the cathode field peaks
    target = emission.charge_pC(photo, [delta_peak])[0]
    em, _ = track_charge(
        machine, delta_peak, args, 0.19, z_stop_m=0.22
    )  # emission is complete just past the gun
    ratio = em["emitted_charge_C"] * 1e12 / target
    photo = split_energy_and_qe(
        dataclasses.replace(photo, pulse_energy_J=photo.pulse_energy_J / ratio), args
    )
    machine.photocathode = photo
    print(
        f"space-charge-limited emission keeps {ratio:.3f} of the ideal charge at the field maximum; "
        f"laser energy {photo.pulse_energy_J * 1e6:.3f} uJ and QE {photo.qe:.3g} at the {peak / 1e6:.0f} MV/m "
        "cathode field",
        flush=True,
    )

    deltas = np.unique(
        np.concatenate(
            [
                np.arange(-60.0, APERTURE_FLANK[0], args.step),
                np.arange(APERTURE_FLANK[0], APERTURE_FLANK[1] + 1e-9, args.step / 2),
            ]
        )
    )
    if args.quick:
        deltas = deltas[::2]
    ideal = emission.charge_pC(photo, deltas)
    rows, t0 = [], time.time()
    for delta, ideal_pC in zip(deltas, ideal):
        em, scr = track_charge(machine, delta, args, args.monitor_z)
        rows.append(
            dict(
                offset_from_crest_deg=delta,
                scan_phase_deg=fit["x0"] - delta,
                ideal_pC=ideal_pC,
                emitted_pC=em["emitted_charge_C"] * 1e12,
                returned_pC=em["returned_charge_C"] * 1e12,
                monitor_pC=float(scr["q"].abs().sum()) * 1e12,
            )
        )
        print(
            f"  offset {delta:6.1f} deg (scan phase {rows[-1]['scan_phase_deg']:8.1f}): ideal {ideal_pC:6.1f} "
            f"emitted {rows[-1]['emitted_pC']:6.1f} at the monitor {rows[-1]['monitor_pC']:6.1f} pC  [{time.time() - t0:4.0f} s]",
            flush=True,
        )
    write_csv(out_dir / "schottky_scan.csv", rows)
    fit |= dict(
        pulse_energy_J=photo.pulse_energy_J,
        qe=photo.qe,
        qe_field_V_per_m=peak,
        wavelength_nm=args.wavelength_nm,
        field_enhancement=args.beta,
        crest_deg=crest,
    )
    (out_dir / "schottky_fit.json").write_text(json.dumps(fit, indent=2) + "\n")
    return rows, fit


def saturation_scan(args, out_dir, fit):
    """Extracted charge against laser energy for several spot sizes, tracked just past the gun."""
    monitor_z = 0.2
    photo = Photocathode(
        pulse_energy_J=fit["pulse_energy_J"],
        wavelength_nm=fit["wavelength_nm"],
        qe=fit["qe"],
        work_function_eV=fit["work_function_eV"],
        qe_field_V_per_m=fit["qe_field_V_per_m"],
        field_enhancement=fit["field_enhancement"],
    )
    machine, peak, _ = build_machine(args, monitor_z, photo, args.saturation_dt_ps)
    delta_peak = (180.0 - machine.cavities[0].crest_deg + 180.0) % 360.0 - 180.0
    scales = np.geomspace(0.1, 30.0, 6 if args.quick else 10)
    rows = []
    for sigma in [0.05, 0.1, 0.15, 0.25, 0.5]:
        machine.generator_overrides = {
            "sig_x": sigma,
            "sig_y": sigma,
            "c_sig_x": SPOT_CUT,
            "c_sig_y": SPOT_CUT,
            "cut_shape": "ellipse",  # a circle here: what gauss_cap_pC assumes
        }
        for scale in scales:
            machine.photocathode = dataclasses.replace(
                photo, pulse_energy_J=photo.pulse_energy_J * scale
            )
            em, _ = track_charge(machine, delta_peak, args, monitor_z - 0.01)
            rows.append(
                dict(
                    sigma_mm=sigma,
                    pulse_energy_uJ=machine.photocathode.pulse_energy_J * 1e6,
                    ideal_pC=em["ideal_charge_C"] * 1e12,
                    emitted_pC=em["emitted_charge_C"] * 1e12,
                    extracted_pC=em["charge_C"] * 1e12,
                )
            )
            print(
                f"  rms spot {sigma:5.2f} mm  laser {rows[-1]['pulse_energy_uJ']:8.3f} uJ  unscreened {rows[-1]['ideal_pC']:9.2f}"
                f"  extracted {rows[-1]['extracted_pC']:9.2f} pC",
                flush=True,
            )
    write_csv(out_dir / "saturation_scan.csv", rows)
    return rows, peak


def transmission_function(machine, args, window=TRANSMISSION_WINDOW, bin_deg=0.25):
    """Fraction of the electrons emitted at each gun phase that reach the monitor, from one run with tracer particles
    emitted uniformly in RF phase over ``window`` (gun phase offsets from the crest, [deg]): the single-particle acceptance
    of the gun, solenoid and apertures. There is no photocathode weighting and no space charge, so it is the transmission
    of a bunch of vanishing charge; the tracked scan's 100 pC bunch is transmitted less (see the README).
    Returns the bin centres [deg] and the transmission."""
    lo, hi = window
    cavity = machine.cavities[0]
    deg_per_s = 360.0 * cavity.frequency_hz
    saved = (
        machine.photocathode,
        machine.space_charge,
        machine.generator_overrides,
        cavity.phase_deg,
    )
    machine.photocathode = None  # unit-weight tracers: the emission physics is not part of the transmission
    machine.space_charge = SpaceChargeSettings(
        enabled=False, dt_adaptive=True, dt_safety=5e7, dt_ceiling_s=2e-12
    )
    machine.generator_overrides = {
        **saved[2],
        "lt": (hi - lo) / deg_per_s * 1e9,  # ns
        "rt": 2e-4,
        "q_total": 1e-3,
    }
    centre = 0.5 * (lo + hi)
    cavity.phase_deg = centre
    try:
        out = machine.track(
            n_particles=args.particles,
            device=args.device,
            dtype=getattr(torch, args.dtype),
            verbose=False,
            record_every=0,
            screen_z_m=args.monitor_z,
            z_stop_m=args.monitor_z + 0.02,
            use_compile=args.compile,
            use_warp=args.warp,
        )
        beam = machine.initial_beam(args.particles, "cpu", torch.float64)
    finally:
        (
            machine.photocathode,
            machine.space_charge,
            machine.generator_overrides,
            cavity.phase_deg,
        ) = saved
    offsets = (
        centre + deg_per_s * beam["emission_time_s"].numpy()
    )  # the RF phase of each electron's emission
    arrived = out["screen"]["q"].abs().double().cpu().numpy() / beam["q"].abs().numpy()
    edges = np.arange(lo, hi + 0.5 * bin_deg, bin_deg)
    emitted, _ = np.histogram(offsets, edges)
    reached, _ = np.histogram(offsets, edges, weights=arrived)
    return 0.5 * (edges[1:] + edges[:-1]), reached / np.maximum(emitted, 1)


def scan_from_transmission(centres, transmission, deltas, emitted_pC, pulse_deg=2.0):
    """The scan without space charge: the emitted charge at each gun phase ``deltas`` times the transmission averaged over
    the laser pulse (``pulse_deg`` wide, centred on it). NaN where the pulse lies outside ``centres``.
    """
    half = max(0.5 * pulse_deg, 0.5 * float(np.min(np.diff(centres))))
    return np.array(
        [
            (
                e * transmission[sel].mean()
                if (sel := np.abs(centres - d) <= half).any()
                else np.nan
            )
            for d, e in zip(deltas, emitted_pC)
        ]
    )


def transmission_scan(args, out_dir):
    """``transmission_function`` of the best-fit machine, written to ``transmission.csv``; if a tracked scan is there,
    its monitor charge / emitted charge (space charge included) is the comparison."""
    machine, _, _ = build_machine(args, args.monitor_z, None, args.dt_ps)
    t0 = time.time()
    centres, transmission = transmission_function(
        machine, args, bin_deg=args.transmission_bin_deg
    )
    print(
        f"transmission function of {args.particles} electrons emitted over "
        f"{TRANSMISSION_WINDOW[0]:g} to {TRANSMISSION_WINDOW[1]:g} deg [{time.time() - t0:.0f} s]",
        flush=True,
    )
    rows = [
        dict(offset_from_crest_deg=c, transmission=t)
        for c, t in zip(centres, transmission)
    ]
    write_csv(out_dir / "transmission.csv", rows)
    tracked = None
    if (out_dir / "schottky_scan.csv").is_file():
        with open(out_dir / "schottky_scan.csv") as fh:
            tracked = list(csv.DictReader(fh))
        deltas = np.array([float(r["offset_from_crest_deg"]) for r in tracked])
        emitted = np.array([float(r["emitted_pC"]) for r in tracked])
        monitor = np.array([float(r["monitor_pC"]) for r in tracked])
        pulse_deg = (
            read_generator_input(args.generator)[0].lt
            * 1e-9
            * 360.0
            * machine.cavities[0].frequency_hz
        )
        predicted = scan_from_transmission(
            centres, transmission, deltas, emitted, pulse_deg
        )
        late = deltas >= APERTURE_FLANK[0]
        diff = predicted - monitor
        print(
            f"without space charge the scan is {np.nanmean(diff[late]):+.1f} pC on average above the tracked one over the "
            f"aperture-limited phases ({np.nanmax(np.abs(diff[late])):.0f} pC at most)",
            flush=True,
        )
    if not args.no_plots:
        plot_transmission(centres, transmission, tracked, out_dir)
    return rows


def plot_transmission(centres, transmission, tracked, out_dir):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        centres,
        transmission,
        color="C0",
        label="uniform-phase tracers, no space charge",
    )
    if tracked:
        ax.plot(
            [float(r["offset_from_crest_deg"]) for r in tracked],
            [
                float(r["monitor_pC"]) / max(float(r["emitted_pC"]), 1e-9)
                for r in tracked
            ],
            "o-",
            color="C3",
            ms=4,
            label="tracked scan: monitor / emitted charge (space charge)",
        )
    ax.set_xlabel("gun phase offset from the crest at emission [deg]")
    ax.set_ylabel("transmission to the monitor")
    ax.set_xlim(20.0, TRANSMISSION_WINDOW[1])
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "transmission.png", dpi=150)
    plt.close(fig)


def write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_schottky(rows, fit, x, y, s, out_dir):
    import matplotlib.pyplot as plt

    phase = np.array([r["scan_phase_deg"] for r in rows])
    order = np.argsort(phase)
    model = fit["baseline_pC"] + np.array([r["monitor_pC"] for r in rows])[order]
    ideal = fit["baseline_pC"] + np.array([r["ideal_pC"] for r in rows])[order]
    phase = phase[order]
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    top.errorbar(x, y, s, fmt=".", color="0.5", ms=3, lw=0.6, label="measured (104459)")
    top.plot(
        phase,
        model,
        "o-",
        color="C3",
        ms=4,
        label="slipstream: emission + tracking to the monitor",
    )
    top.plot(
        phase,
        ideal,
        "--",
        color="C0",
        label="emission only: applied field, no transport",
    )
    top.set_ylabel("charge [pC]")
    top.legend()
    top.grid(alpha=0.3)
    top.set_title(
        f"CLARA HRRG Schottky scan: fitted work function {fit['work_function_eV']:.3f} eV "
        f"(beta {fit['field_enhancement']:g})\n"
        f"chi2/N {fit['chi2_per_point']:.2f} on the field-limited flank (emission only); "
        "tracking adds transport",
        fontsize=10,
    )
    bottom.plot(phase, model - np.interp(phase, x, y), "o-", color="C3", ms=3)
    bottom.axhline(0.0, color="k", lw=0.5)
    bottom.set_ylabel("model - measured [pC]")
    bottom.set_xlabel("gun phase [deg]")
    bottom.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "schottky_scan.png", dpi=150)
    plt.close(fig)


def gauss_cap_pC(ideal_pC, sigma_mm, field, cut=SPOT_CUT):
    """Charge of a gaussian spot (truncated at ``cut`` sigma) clipped at the Gauss-law density eps0*E of a thin sheet."""
    sigma, trunc = sigma_mm * 1e-3, 1.0 - math.exp(-0.5 * cut**2)
    density = (
        ideal_pC * 1e-12 / (2.0 * math.pi * sigma**2 * trunc)
    )  # areal density at the centre
    if density <= EPS0_F_PER_M * field:
        return ideal_pC
    r2 = min(
        2.0 * sigma**2 * math.log(density / (EPS0_F_PER_M * field)), (cut * sigma) ** 2
    )  # laser density = limit
    outside = (math.exp(-0.5 * r2 / sigma**2) - math.exp(-0.5 * cut**2)) / trunc
    return (EPS0_F_PER_M * field * math.pi * r2) * 1e12 + ideal_pC * outside


def plot_saturation(rows, peak_field, out_dir):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for i, sigma in enumerate(sorted({r["sigma_mm"] for r in rows})):
        sel = [r for r in rows if r["sigma_mm"] == sigma]
        ideal = [r["ideal_pC"] for r in sel]
        ax.loglog(
            ideal,
            [r["extracted_pC"] for r in sel],
            "o-",
            color=f"C{i}",
            ms=4,
            label=f"rms spot {sigma:g} mm",
        )
        ax.loglog(
            ideal,
            [gauss_cap_pC(q, sigma, peak_field) for q in ideal],
            ":",
            color=f"C{i}",
            lw=1,
        )
    lo, hi = ax.get_xlim()
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.6, label="no limit")
    ax.set_xlim(lo, hi)
    ax.set_xlabel(
        "unscreened photoemission [pC] (laser energy x quantum efficiency at the applied field)"
    )
    ax.set_ylabel("charge extracted [pC]")
    ax.set_title(
        "Space-charge-limited emission at the field maximum\n"
        "dotted: Gauss-law cap of an instantaneous sheet (eps0 E), without Schottky screening or cloud motion",
        fontsize=9,
    )
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / "saturation_scan.png", dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    if CLARA is None:
        raise SystemExit(
            "this example needs the optional laura and laura_lattices packages"
        )
    if args.quick:
        args.particles = min(args.particles, 600)
    warnings.filterwarnings("ignore")
    capture = capture_settings()
    apply_capture(args, capture)
    if capture:
        print(
            f"camera frame: laser energy readback {capture['laser_energy_uJ']:.3f} uJ, gun solenoid "
            f"{capture['gun_solenoid_A']:.2f} A = {capture['gun_solenoid_A'] * SOLENOID_T_PER_A:.3f} T at "
            f"{SOLENOID_T_PER_A:g} T/A (this run uses {args.solenoid_T:.3f} T), charge monitor "
            f"{capture['charge_pC']:.1f} pC at gun phase {capture['gun_phase_deg']:.2f} deg",
            flush=True,
        )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(DATA, "r") as f:
        x, y, s = f["xData"][:], f["yData"][:], f["yStd"][:]
    if args.only == "transmission":
        transmission_scan(args, out_dir)
        print(f"outputs in {out_dir}")
        return
    fit_file = out_dir / "schottky_fit.json"
    if args.only in ("schottky", "both"):
        rows, fit = schottky_scan(args, x, y, s, out_dir)
        if not args.no_plots:
            plot_schottky(rows, fit, x, y, s, out_dir)
    if args.only in ("saturation", "both"):
        if args.only == "saturation":
            if not fit_file.is_file():
                raise SystemExit(
                    f"--only saturation reuses the calibration of an earlier Schottky run: {fit_file} is missing"
                )
            fit = json.loads(fit_file.read_text())
        rows, peak = saturation_scan(args, out_dir, fit)
        if not args.no_plots:
            plot_saturation(rows, peak, out_dir)
    print(f"outputs in {out_dir}")


if __name__ == "__main__":
    main()
