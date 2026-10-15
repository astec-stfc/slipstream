"""CLARA HRRG Schottky scan and laser-energy/spot-size saturation with the native photocathode emission model.

Needs the optional LAURA packages (``laura``, ``laura_lattices``) for the CLARA gun, solenoid and apertures.

1. Schottky scan: charge reaching a monitor plane against gun phase, compared with the measured scan
   ``104459_Schottky_approx_crestingData.h5``. The field-limited flank of the scan (emission late in the laser's RF
   half-cycle) fixes the effective work function, the quantum-efficiency x laser-energy scale, the phase origin and the
   monitor baseline through the same quantum-efficiency law the tracker uses. Tracking then adds space-charge-limited
   emission, electrons falling back onto the cathode and aperture losses.
2. Saturation: extracted charge against laser energy for several rms spot sizes at the gun phase of maximum field. The
   spots are round gaussians cut at 3 sigma on a circle whatever ``generator.in`` holds; its pulse shape is kept.

The laser profile comes from ``generator.in``: the transverse gaussian cut on an ellipse is fitted to a virtual-cathode
image by ``fit_laser_image.py``, the pulse length and emission energy are assumed, not measured. The laser energy is the
energy-meter readback stored with that image (``--laser-energy-uJ`` replaces it, ``--qe`` fixes the quantum efficiency
instead and fits the energy), so the fit gives the quantum efficiency. The gun field and solenoid default to the current
best fit. Pass your own with ``--generator``. Install ``slipstream-accel[plot]`` for figures or use ``--no-plots``.
"""

import argparse
import csv
import dataclasses
import json
import math
import time
import warnings
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.optimize import least_squares

from slipstream.constants import E_CHARGE_C, EPS0_F_PER_M
from slipstream.tracking.native.injector import SpaceChargeSettings
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
GUN_FIELD_MV = 105.0  # peak gun field of the current best fit (LAURA's design value: 120)
SOLENOID_T = 0.310  # gun solenoid peak field of the current best fit (LAURA's: 0.345)
SOLENOID_T_PER_A = 0.0022  # experimental calibration of the gun solenoid in 2022 (it changed later)
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


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--only", choices=["schottky", "saturation", "both"], default="schottky"
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


def build_machine(args, monitor_z, photo, dt_ps):
    """Gun, S01 and the L01 entrance aperture; the L01 cavity and solenoids lie beyond the monitor and are dropped."""
    machine = from_laura(
        CLARA,
        SECTIONS,
        monitor_z + 0.02,
        generator=str(Path(args.generator).resolve()),
        photocathode=photo,
    )
    machine.cavities = [c for c in machine.cavities if c.z_m < monitor_z]
    machine.solenoids = [s for s in machine.solenoids if s.z_m < monitor_z]
    if args.gun_field_MV is not None:
        machine.cavities[0].e_max_V_per_m = args.gun_field_MV * 1e6
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
    current [A], gun phase [deg] and charge monitor [pC]. Empty if the frame is not there."""
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
    to the QE, or take ``--qe`` (``QE_START`` if neither is given) and give the rest to the energy."""
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
