"""Fit the gun and solenoid settings to the ramp, dip and satellite of the measured CLARA Schottky scan.

The aperture-limited end of the scan (gun phase -165 to -204 deg) is the transmission of the 10 mm L01 aperture for slow
electrons (1-2 MeV), so it depends on the cathode field, the solenoid field, the flatness of the gun's RF field and the
misalignment of the solenoid, which the S01 correctors hide at one gun phase only. Each trial machine is tracked to a screen
just before the aperture (z = 1.057 m) at a grid of gun phases; the charge inside the aperture's square is the monitor charge.
A solenoid offset or tilt is corrected as the operators do: two thin steering kicks at the S01 correctors are solved to centre
the beam on the BPM and the screen at the reference phase, then held at every phase.

Free parameters: cathode field, solenoid peak field, flatness (the full cell's peak field over the cathode's) and the
solenoid offset along x and tilt in the x-z plane (signed). The phase origin and the monitor baseline are refitted from the
field-limited flank for every trial (they depend on the gun field and flatness) and the laser energy is the camera frame's
readback. The residual is the model minus the measured charge over the aperture-limited phases.

A trial takes one to two minutes on a laptop GPU (the tracked step is compiled again for every new solenoid placement
and field scale), so a fit takes hours. The landscape is rough: the satellite is a sharp feature that jumps between
neighbouring trials, so the optimiser settles in a broad valley and its formal errors mean little. Read the
result against the measured curve, which ``structure_<tag>.png`` shows.

    python examples/photocathode/fit_structure.py --tag a  # from schottky_scan.py's defaults
    python examples/photocathode/fit_structure.py --start 105 0.31 0.941 0.5 0.0 --tag b
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import schottky_scan as scan  # noqa: E402

NAMES = ("gun_field_MV", "solenoid_T", "flatness", "solenoid_offset_mm", "solenoid_tilt_mrad")
LOWER = (90.0, 0.26, 0.70, -1.5, -4.0)
UPPER = (125.0, 0.36, 1.10, 1.5, 4.0)
STEP = (1.5, 0.004, 0.04, 0.15, 0.6)  # finite-difference steps of the Jacobian, in the parameters' own units
OFFSETS = np.arange(30.0, 77.0, 2.0)  # gun phase offsets from the crest [deg] tracked per trial
X_WINDOW = (-204.0, -160.0)  # measured scan phases in the residual
REFERENCE_OFFSET = scan.REFERENCE_OFFSET_DEG  # gun phase offset at which the correctors centre the beam
APERTURE_Z, APERTURE_HALF_M = 1.0570, 0.010  # screen just before the L01 aperture; its half-width
CORRECTOR_Z = tuple(z for z, _ in scan.CORRECTORS)  # centres of CLA-S01-MAG-HVCOR-01 and -02
OBSERVATION_Z = scan.OBSERVATION_Z  # the BPM and the screen the operators centre the beam on
SC_KEPT = 0.971  # share of the ideal charge that space-charge-limited emission keeps (reference run at the best fit)


flat_map = scan.flat_map  # the Ez map with a scaled full cell, shared with schottky_scan.py


def make_machine(args, p, maps_dir):
    """The CLARA gun, S01 and L01 aperture at the trial parameters ``p`` (``NAMES`` order), auto-phased. The misaligned
    solenoid is not steered here: ``solve_correctors`` and ``aperture_charge_pC`` apply the steering to the screen data."""
    args.gun_field_MV, args.solenoid_T, args.flatness, args.solenoid_offset_mm, args.solenoid_tilt_mrad = p
    args.output_dir = str(Path(maps_dir).parent)
    return scan.build_machine(args, args.monitor_z, None, 0.4)[0]


def screen_at(machine, args, offset, z):
    machine.cavities[0].phase_deg = float(offset)
    out = machine.track(
        n_particles=args.particles, device=args.device, dtype=getattr(torch, args.dtype), verbose=False, record_every=0,
        screen_z_m=z, z_stop_m=z + 0.005, use_compile=args.compile, use_warp=args.warp,
    )  # fmt: skip
    keep = {k: v.double().cpu().numpy() for k, v in out["screen"].items() if hasattr(v, "cpu")}
    keep["q"] = np.abs(keep["q"]) * (keep["crossed"] > 0)
    return keep, out["emission"]


def momentum(s):
    """|u| = gamma beta c of every particle: a steering kick of momentum dp gives the angle du / |u|."""
    return np.sqrt(s["ux"] ** 2 + s["uy"] ** 2 + s["uz"] ** 2)


def solve_correctors(machine, args):
    """Steering kicks (du = dp/m, per plane) at ``CORRECTOR_Z`` that centre the beam at ``OBSERVATION_Z`` at the reference phase."""
    mean, weight = {}, {}
    for z in OBSERVATION_Z:
        s, _ = screen_at(machine, args, REFERENCE_OFFSET, z)
        mean[z] = {k: float((s[k] * s["q"]).sum() / s["q"].sum()) for k in "xy"}
        weight[z] = float((s["q"] / momentum(s)).sum() / s["q"].sum())
    matrix = np.array([[(z - zc) * weight[z] for zc in CORRECTOR_Z] for z in OBSERVATION_Z])
    return {k: np.linalg.solve(matrix, -np.array([mean[z][k] for z in OBSERVATION_Z])) for k in "xy"}


def aperture_charge_pC(s, kicks=None):
    """Charge inside the aperture's square at the screen, after the steering kicks (their drift to the screen included)."""
    x, y = s["x"].copy(), s["y"].copy()
    if kicks is not None:
        u = momentum(s)
        x += sum(kicks["x"][i] * (APERTURE_Z - z) for i, z in enumerate(CORRECTOR_Z)) / u
        y += sum(kicks["y"][i] * (APERTURE_Z - z) for i, z in enumerate(CORRECTOR_Z)) / u
    inside = (np.abs(x) < APERTURE_HALF_M) & (np.abs(y) < APERTURE_HALF_M)
    return float(s["q"][inside].sum()) * 1e12


def forward(args, p, data, maps_dir):
    """Monitor charge [pC] at the measured scan phases of the window, and the trial's phase origin and baseline."""
    machine = make_machine(args, p, maps_dir)
    field = machine.fields()[0][0]
    peak, omega = float(field.map.f[0] * field.scale), field.omega
    if abs(peak - p[0] * 1e6) > 1e-3 * peak:
        raise RuntimeError(
            f"cathode field {peak / 1e6:.2f} MV/m, not {p[0]:.2f}: the flat map is not normalised as assumed"
        )
    emission = scan.Emission(machine, peak, omega, args)
    start = scan.Photocathode(pulse_energy_J=3e-6, wavelength_nm=266.0, qe=scan.QE_START, work_function_eV=4.5,
                              qe_field_V_per_m=peak)  # fmt: skip
    fit = scan.fit_emission(emission, start, data["x"], data["y"], data["s"])  # on the whole scan: it selects the flank
    energy = data["laser_energy_J"]
    machine.photocathode = scan.Photocathode(
        pulse_energy_J=energy, wavelength_nm=266.0, qe=scan.QE_START * fit["pulse_energy_J"] / energy / SC_KEPT,
        work_function_eV=fit["work_function_eV"], qe_field_V_per_m=peak,
    )  # fmt: skip
    kicks = solve_correctors(machine, args) if (p[3] or p[4]) else None
    charge = np.array([aperture_charge_pC(screen_at(machine, args, o, APERTURE_Z)[0], kicks) for o in OFFSETS])
    curve = PchipInterpolator(OFFSETS, charge, extrapolate=False)
    model = np.nan_to_num(curve(fit["x0"] - data["x"][data["window"]]), nan=0.0) + fit["baseline_pC"]
    return model, fit, dict(zip(OFFSETS.tolist(), charge.tolist()))


def load_data(path=scan.DATA):
    with h5py.File(path, "r") as f:
        x, y, s = f["xData"][:], f["yData"][:], f["yStd"][:]
    return dict(x=x, y=y, s=s, window=(x >= X_WINDOW[0]) & (x <= X_WINDOW[1]))


def plot_fit(result, data, path):
    """The measured scan, the aligned run in ``reference/`` (no solenoid misalignment: ``schottky_scan.py
    --gun-field-MV 105 --solenoid-T 0.310 --flatness 0.941 --solenoid-offset-mm 0 --solenoid-tilt-mrad 0``) and the model."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.errorbar(data["x"], data["y"], data["s"], fmt=".", color="0.4", lw=0.6, label="measured")
    reference = HERE / "reference"
    if (reference / "schottky_scan_aligned.csv").is_file():
        rows = np.genfromtxt(reference / "schottky_scan_aligned.csv", delimiter=",", names=True)
        base = json.loads((reference / "schottky_fit_aligned.json").read_text())["baseline_pC"]
        label = "aligned: 105 MV/m, 0.310 T, no misalignment"
        ax.plot(rows["scan_phase_deg"], rows["monitor_pC"] + base, "-", color="C0", label=label)
    offsets = np.array([float(k) for k in result["curve"]])
    charge = np.array(list(result["curve"].values()))
    label = ", ".join(f"{k.replace('solenoid_', '')} {v:.4g}" for k, v in result["parameters"].items())
    ax.plot(result["x0"] - offsets, charge + result["baseline_pC"], "-", color="C3", label="fit: " + label)
    ax.set_xlim(-207, -158)
    ax.set_ylim(-3, 125)
    ax.set_xlabel("gun phase [deg]")
    ax.set_ylabel("charge [pC]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", type=float, nargs=5, default=[scan.GUN_FIELD_MV, scan.SOLENOID_T, scan.FLATNESS, scan.SOLENOID_OFFSET_MM, scan.SOLENOID_TILT_MRAD],
                    metavar=NAMES)  # fmt: skip
    ap.add_argument("--fixed", nargs="*", default=[], choices=NAMES, help="parameters held at their start value")
    ap.add_argument("--tag", default="fit")
    ap.add_argument("--particles", type=int, default=6000)
    ap.add_argument("--max-nfev", type=int, default=40)
    ap.add_argument("--evaluate", action="store_true", help="evaluate the start point only")
    ap.add_argument("--step", type=float, default=2.0, help="gun phase step of the tracked scan [deg]")
    ap.add_argument("--output-dir", default=str(HERE.parents[1] / ".cache" / "photocathode"))
    a = ap.parse_args()

    warnings.filterwarnings("ignore")
    global OFFSETS
    OFFSETS = np.arange(OFFSETS[0], OFFSETS[-1] + 1e-9, a.step)
    for name in ("cache_size_limit", "recompile_limit", "accumulated_recompile_limit"):
        if hasattr(torch._dynamo.config, name):  # every trial recompiles; do not fall back to eager after 8 or 56
            setattr(torch._dynamo.config, name, 100000)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    maps_dir = out / "flat_maps"
    cuda = torch.cuda.is_available()
    args = argparse.Namespace(
        generator=str(HERE / "generator.in"), gun_field_MV=None, solenoid_T=None, particles=a.particles,
        device="cuda" if cuda else "cpu", dtype="float32" if cuda else "float64", compile=cuda, warp=False,
        dt_ceiling_ps=2.0, monitor_z=1.1, steering=False, reference_offset_deg=REFERENCE_OFFSET,
    )  # fmt: skip
    capture = scan.capture_settings()
    data = load_data()
    data["laser_energy_J"] = capture["laser_energy_uJ"] * 1e-6
    window = data["window"]
    weight = 1.0 / np.maximum(data["s"][window], 2.0)
    start = np.array(a.start, dtype=float)
    free = [i for i, n in enumerate(NAMES) if n not in a.fixed]
    log, t0 = [], time.time()

    def trial(q):
        p = start.copy()
        p[free] = q
        t1 = time.time()
        model, fit, curve = forward(args, p, data, maps_dir)
        resid = (model - data["y"][window]) * weight
        log.append(dict(p=p.tolist(), chi2=float((resid**2).sum() / len(resid)), x0=fit["x0"], baseline=fit["baseline_pC"],
                        work_function_eV=fit["work_function_eV"], curve=curve))  # fmt: skip
        print(f"[{time.time() - t0:6.0f} s, {time.time() - t1:3.0f} s] " + " ".join(f"{n}={v:.4g}" for n, v in zip(NAMES, p))
              + f"  chi2/N {log[-1]['chi2']:.2f}  x0 {fit['x0']:.2f}", flush=True)  # fmt: skip
        return resid

    def finish(best, sigma):
        final = log[-1]
        result = dict(
            parameters=dict(zip(NAMES, np.asarray(best).tolist())), sigma=sigma, chi2_per_point=final["chi2"],
            start=start.tolist(), fixed=a.fixed, evaluations=len(log), curve=final["curve"], x0=final["x0"],
            baseline_pC=final["baseline"], work_function_eV=final["work_function_eV"],
        )  # fmt: skip
        (out / f"structure_{a.tag}.json").write_text(json.dumps(result, indent=2) + "\n")
        plot_fit(result, data, out / f"structure_{a.tag}.png")
        return result

    if a.evaluate:
        trial(start[free])
        finish(start, {})
        return
    lower, upper = np.array(LOWER)[free], np.array(UPPER)[free]
    fit = least_squares(
        trial, start[free], bounds=(lower, upper), diff_step=np.array(STEP)[free] / np.maximum(1.0, np.abs(start[free])),
        x_scale=np.array(STEP)[free] * 3.0, loss="soft_l1", f_scale=2.0, max_nfev=a.max_nfev,
    )  # fmt: skip
    best = start.copy()
    best[free] = fit.x
    trial(fit.x)  # the optimum itself, last in the log
    cov = np.linalg.pinv(fit.jac.T @ fit.jac) * float((fit.fun**2).sum() / max(len(fit.fun) - len(free), 1))
    sigma = dict(zip([NAMES[i] for i in free], np.sqrt(np.diag(cov)).tolist()))
    result = finish(best, sigma)
    print("best:", " ".join(f"{n}={v:.4g}" for n, v in zip(NAMES, best)), f" chi2/N {result['chi2_per_point']:.2f}")
    print("sigma:", {k: round(v, 4) for k, v in sigma.items()})


if __name__ == "__main__":
    main()
