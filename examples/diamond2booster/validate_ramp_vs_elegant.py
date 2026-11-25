"""Check slipstream's ramped ring tracking against elegant on the Diamond-II booster ramp.

The booster accelerates from 100 MeV to 3.5 GeV in 100 ms (half of a 5 Hz cycle: a cosine energy ramp) while the RF voltage
follows E^4 from 0.2 to 2 MV.  elegant tracks the real lattice (``lattice/booster.lte``) with RAMPP (reference momentum),
RAMPRF (voltage and phase) and SREFFECT (radiation damping, quantum excitation and the energy loss per turn).
``NativeTracker(ramp=...)`` tracks the same lattice from the same injected bunch.  Every ``--interval`` turns both report the
central energy, the energy and time centroids and the rms energy spread, bunch length and horizontal size.

    python examples/diamond2booster/validate_ramp_vs_elegant.py                         # 50 to 60 ms, the machine's RF phase table
    python examples/diamond2booster/validate_ramp_vs_elegant.py --balanced-phase        # the same, RF phase V sin(phase) = U0 + dE/turn
    python examples/diamond2booster/validate_ramp_vs_elegant.py --start-ms 0 --duration-ms 100   # the whole ramp, ~183 000 turns

``--tag`` names a run of slipstream (several can be kept and shown together with ``--overlay``), so that two versions of the
code can be compared with the same elegant result: run the script once per version.  elegant is run once and cached in
``output/`` (``--rerun-elegant`` forces it; ``--cores N`` runs Pelegant).  It needs elegant: ``ELEGANT_HOME``, default
``C:/Program Files/APS/Elegant x64``.

The default window, 50 to 60 ms, is where the RF energy balance matters most: the radiation loss grows as E^4 (66 to 288
keV per turn against 0.2 to 0.6 MV of RF) and the ramp takes 29 keV per turn.  The first tens of ms are nearly adiabatic
and quiet.  A window of 10 ms is 18 300 turns, about 45 minutes in slipstream on one core (0.14 s per turn: 290 elements
and 78 bends tracked in 4 kicks) and a few minutes in elegant.

Two conventions of elegant that the comparison has to respect (``write_ramp`` and ``measure_time_excess``):

* elegant's RF phase is measured from a nominal arrival time (the sum of the revolution times C / (beta c)), not from the
  particle that is tracked, and its bend BD takes 2.4 ps per turn longer than C / (beta c) (0.028 mm of extra path each).
  In elegant that is a frequency error of 4.4e-6, which moves the synchronous particle to an energy offset of
  2.4 ps / (alpha_c T0) = -8e-4.  slipstream measures the phase from the reference particle, an RF locked to the revolution,
  so the slip is measured (on the bare ring) and taken off elegant's phase column.  elegant ignores the frequency column.
* The machine's table keeps the phase at 180 degrees, so V sin(phase) is zero: the beam is injected 0.5 rad from the
  synchronous phase and oscillates through the whole window.  ``--balanced-phase`` uses the phase at which the RF supplies
  exactly the loss and the gain, which leaves the bunch quiet and makes the energy balance visible directly.
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import sdds
import torch

from slipstream.constants import C_LIGHT, ELECTRON_MASS_EV
from slipstream.lattice.lattice import Lattice
from slipstream.lattice.optics import TransverseOptics
from slipstream.lattice.ramp import RampProfile
from slipstream.tracking.native import NativeTracker, parse_elegant_lattice
from slipstream.tracking.native.elements import _canonical_from_slopes

HERE = Path(__file__).resolve().parent
LATTICE = HERE / "lattice" / "booster.lte"
OUTPUT = HERE / "output"
PREFIX = ""  # file name prefix of the helper files of one run (set in main), so that runs can go in parallel
ELEGANT_HOME = Path(os.environ.get("ELEGANT_HOME", r"C:\Program Files\APS\Elegant x64"))

E_INJECTION_EV, E_EXTRACTION_EV, RAMP_S = 100e6, 3.5e9, 0.1  # the machine: 100 MeV to 3.5 GeV in half a 5 Hz cycle
V_MIN_V, V_MAX_V, PHASE_DEG = 0.2e6, 2.0e6, 180.0
HARMONIC, CIRCUMFERENCE_M = 273, 163.84704351098517
COUPLING = 0.1  # SR1's vertical to horizontal emittance ratio
EMIT_N_M, SIGMA_DP, SIGMA_S_M = 30e-6, 1e-3, 1e-3  # injected normalized emittance (both planes), energy and length spread


def momentum_mc(energy_eV: np.ndarray | float) -> np.ndarray | float:
    """Momentum in units of m c: beta gamma."""
    return np.sqrt((np.asarray(energy_eV) / ELECTRON_MASS_EV) ** 2 - 1.0)


def write_ramp(path: Path, start_s: float, duration_s: float, u0_start_eV: float | None = None, excess_s: float = 0.0,
               compensate_slip: bool = False, rows: int = 1001) -> dict:
    """ASCII SDDS ramp file read by both codes: the window [start_s, start_s + duration_s] of the booster ramp, a cosine
    energy ramp with V ~ E^4 clamped at V_MIN_V.  Time starts at zero in the file.

    slipstream measures the RF phase from the reference particle, which is the same as an RF locked to the revolution of the
    reference.  elegant measures it from a nominal arrival time (C / (beta c) per turn), while its bend BD takes ``excess_s``
    longer per turn (see ``measure_time_excess``); that is a frequency error of excess_s / T, which pushes the synchronous
    particle to an energy offset of excess_s / (alpha_c T).  ``compensate_slip`` subtracts the slip from the phase column, so
    that elegant's RF is locked too.  (The frequency column has no effect in elegant: only the arrival time difference counts.)

    The phase is the machine's fixed 180 degrees or, given the energy loss per turn at the start of the window
    ``u0_start_eV`` (it scales as E^4), the phase at which the RF supplies exactly the loss plus the energy gain,
    V sin(phase) = U0 + dE/turn, which leaves the synchronous particle at the centre of the bunch."""
    time_s = np.linspace(0.0, duration_s, rows)
    alpha = (E_INJECTION_EV + E_EXTRACTION_EV) / (E_EXTRACTION_EV - E_INJECTION_EV)
    amplitude = E_EXTRACTION_EV / (1 + alpha)
    energy = amplitude * (alpha - np.cos(math.pi * (start_s + time_s) / RAMP_S))  # a half cosine: injection to extraction
    voltage = np.clip(V_MAX_V * (energy / E_EXTRACTION_EV) ** 4, V_MIN_V, None)
    momentum = momentum_mc(energy)
    beta = momentum / np.sqrt(1 + momentum**2)
    revolution = CIRCUMFERENCE_M / (beta * C_LIGHT) + excess_s
    frequency = HARMONIC / revolution
    phase_deg = np.full(rows, PHASE_DEG)
    if u0_start_eV is not None:
        per_turn = np.gradient(energy, time_s) * revolution + u0_start_eV * (energy / energy[0]) ** 4
        phase_deg = 180.0 - np.degrees(np.arcsin(per_turn / voltage))
    if compensate_slip:
        phase_deg = phase_deg - np.degrees(2 * math.pi * frequency * excess_s * time_s / revolution)
    n_passes = int(duration_s / revolution[-1])
    lines = [
        "SDDS1",
        '&parameter name=pCentral, units="m$be$nc", description="Central momentum", type=double, &end',
        "&parameter name=nPasses, type=long, &end",
        "&column name=time, units=s, type=double, &end",
        "&column name=Voltage, units=V, type=double, &end",
        "&column name=Frequency, units=Hz, type=double, &end",
        "&column name=Phase, units=deg, type=double, &end",
        "&column name=MomentumRatio, type=double, &end",
        "&column name=momentum, type=double, &end",
        "&data mode=ascii, &end",
        "! page number 1",
        f"{momentum[0]:.17g}",
        str(n_passes),
        f"                {rows}",
    ]
    lines += [f"{t:.17g}\t{v:.17g}\t{f:.17g}\t{ph:.17g}\t{r:.17g}\t{p:.17g}"
              for t, v, f, ph, r, p in zip(time_s, voltage, frequency, phase_deg, momentum / momentum[0], momentum)]
    path.write_text("\n".join(lines) + "\n")
    return dict(p0=float(momentum[0]), n_passes=n_passes, f_rf_Hz=float(frequency[0]))


def tmp(name: str) -> str:
    """Name of a helper file of this run in ``output/``."""
    return f"{PREFIX}{name}"


def run_elegant(args_ele: Path, cores: int) -> None:
    """Run elegant (``cores`` > 1: Pelegant under MPI) in ``output/`` and raise with its output if it fails."""
    exe = ELEGANT_HOME / ("Pelegant.exe" if cores > 1 else "elegant.exe")
    command = ["mpiexec", "-n", str(cores), str(exe), args_ele.name] if cores > 1 else [str(exe), args_ele.name]
    done = subprocess.run(command, cwd=OUTPUT, capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError(f"elegant failed:\n{done.stdout[-1500:]}\n{done.stderr[-1500:]}")


def read_sdds(path: Path) -> tuple[dict, dict]:
    """``(parameters, columns)`` of the first page of an SDDS file."""
    file = sdds.SDDS(0)
    file.load(str(path))
    parameters = {n: v[0] if isinstance(v, (list, tuple)) and len(v) == 1 else v for n, v in zip(file.parameterName, file.parameterData)}
    return parameters, {n: np.array(file.columnData[i][0]) for i, n in enumerate(file.columnName)}


def alter(name: str, item: str, value) -> str:
    """An elegant ``&alter_elements`` command."""
    text = f'"{value}"' if isinstance(value, str) else repr(value)
    return f'&alter_elements name = {name}, item = {item}, {"string_value" if isinstance(value, str) else "value"} = {text} &end\n'


def ramp_waveforms(ramp_file: str) -> str:
    """Point RAMPP and RAMPRF at the waveforms of the ramp file."""
    text = alter("RAMP", "WAVEFORM", f"{ramp_file}=time+MomentumRatio")
    for item, column in (("VOLT_WAVEFORM", "Voltage"), ("PHASE_WAVEFORM", "Phase"), ("FREQ_WAVEFORM", "Frequency")):
        text += alter("CAVRAMP", item, f"{ramp_file}=time+{column}")
    return text


def injection_optics(p0: float, f_rf_Hz: float, ramp_file: str) -> tuple[Lattice, TransverseOptics, dict]:
    """Radiation parameters and optics of the lattice at injection, from elegant's own Twiss and radiation integrals."""
    (OUTPUT / tmp("twiss.ele")).write_text(f"""&run_setup lattice = "../lattice/booster.lte", use_beamline = "boost", p_central = {p0!r} &end
{ramp_waveforms(ramp_file)}{alter("W1", "FILENAME", tmp("twiss_unused.w1"))}&run_control &end
&twiss_output filename = "{tmp('twiss.twi')}", matched = 1, radiation_integrals = 1 &end
&bunched_beam n_particles_per_bunch = 10, emit_nx = {EMIT_N_M}, emit_ny = {EMIT_N_M}, sigma_dp = {SIGMA_DP}, sigma_s = {SIGMA_S_M}, use_twiss_command_values = 1 &end
&track &end
&stop &end
""")
    run_elegant(OUTPUT / tmp("twiss.ele"), 1)
    par, col = read_sdds(OUTPUT / tmp("twiss.twi"))
    energy_eV = math.sqrt(1 + p0**2) * ELECTRON_MASS_EV
    ex = par["ex0"] / (1 + COUPLING)
    # the native damping times are those of the angle and the energy deviation: half of elegant's amplitude damping times
    lattice = Lattice(
        circumference_m=CIRCUMFERENCE_M, E0_eV=energy_eV, alpha_c=par["alphac"], Vrf_V=V_MIN_V, f_rf_Hz=f_rf_Hz, U0_eV=par["U0"] * 1e6,
        tau_z_s=par["taudelta"] / 2, sigma_delta0=par["Sdelta0"],
    )
    optics = TransverseOptics(
        beta_x_m=col["betax"][0], alpha_x=col["alphax"][0], D_x_m=col["etax"][0], Q_x=par["nux"], Qp_x=par["dnux/dp"],
        tau_x_s=par["taux"] / 2, epsx_m=ex,
        beta_y_m=col["betay"][0], alpha_y=col["alphay"][0], D_y_m=col["etay"][0], Q_y=par["nuy"], Qp_y=par["dnuy/dp"],
        tau_y_s=par["tauy"] / 2, epsy_m=COUPLING * ex,
    )
    return lattice, optics, par


def elegant_ramp_input(ramp_file: str, par: dict, n_passes: int, particles: int, interval: int, watch: str, bunch: str) -> str:
    """The tracking input: lattice at injection momentum, SREFFECT from the Twiss run, the ramp waveforms and a watch point."""
    text = f'&run_setup lattice = "../lattice/booster.lte", use_beamline = "boost", p_central = {par["pCentral"]!r}, random_number_seed = 1 &end\n'
    text += alter("SR1", "COUPLING", COUPLING)  # SREFFECT takes the rest from the radiation integrals of the Twiss run
    text += ramp_waveforms(ramp_file)
    text += alter("W1", "FILENAME", watch) + alter("W1", "INTERVAL", interval) + alter("W1", "FLUSH_INTERVAL", interval)
    text += f"""&run_control n_steps = 1, n_passes = {n_passes} &end
&twiss_output matched = 1, radiation_integrals = 1, filename = "{tmp('ramp_twiss.twi')}" &end
&bunched_beam bunch = "{bunch}", n_particles_per_bunch = {particles}, use_twiss_command_values = 1,
  emit_nx = {EMIT_N_M}, emit_ny = {EMIT_N_M}, sigma_dp = {SIGMA_DP}, sigma_s = {SIGMA_S_M} &end
&track &end
&stop &end
"""
    return text


def load_elegant_bunch(path: Path) -> torch.Tensor:
    """The injected bunch elegant generated, as native ``(x, px, y, py, tau, delta)``."""
    par, col = read_sdds(path)
    delta = torch.tensor(col["p"] / par["pCentral"] - 1.0)
    px, py = _canonical_from_slopes(torch.tensor(col["xp"]), torch.tensor(col["yp"]), delta)
    return torch.stack([torch.tensor(col["x"]), px, torch.tensor(col["y"]), py, torch.tensor(col["t"] * C_LIGHT), delta], dim=-1)


def track_native(lattice, optics, ramp, bunch, n_passes, interval, seed, n_kicks_bend) -> dict:
    """NativeTracker over the whole ramp, recording the same statistics as the elegant watch point every ``interval`` turns."""
    specs = parse_elegant_lattice(LATTICE, root_name="boost")
    torch.set_num_threads(1)  # a few hundred small tensor operations per turn: threads only add overhead
    tracker = NativeTracker(specs, lattice, optics=optics, ramp=ramp, device="cpu", dtype=torch.float64, n_kicks_bend=n_kicks_bend,
                            generator=torch.Generator().manual_seed(seed))
    state, rows = bunch.clone(), []
    start = time.time()
    for turn in range(0, n_passes, interval):
        state, result = tracker.track(state, min(interval, n_passes - turn), sample_every=interval)
        rows.append((turn + interval, result.E0_eV[-1], *statistics(state)))
    print(f"  native: {n_passes} turns, {state.shape[0]} particles in {time.time() - start:.0f} s")
    return as_table(rows)


def statistics(state: torch.Tensor):
    """(Cdelta, Ct, Sdelta, St, Sx, Sy) of an ensemble: centroids and rms values, time in seconds."""
    x, y, tau, delta = state[:, 0], state[:, 2], state[:, 4] / C_LIGHT, state[:, 5]
    return delta.mean().item(), tau.mean().item(), delta.std().item(), tau.std().item(), x.std().item(), y.std().item()


NAMES = ("Pass", "E0", "Cdelta", "Ct", "Sdelta", "St", "Sx", "Sy")


def as_table(rows) -> dict:
    return dict(zip(NAMES, np.array(rows, dtype=float).T))


def measure_time_excess(ramp_name: str, par: dict) -> float:
    """The time per turn elegant adds to C / (beta c) for a particle on the reference orbit (the BD bends: about 0.03 mm of
    path each): the slope of the arrival time of a tiny bunch in the bare ring, with no RF, radiation or ramp."""
    (OUTPUT / tmp("excess.ele")).write_text(f"""&run_setup lattice = "../lattice/booster.lte", use_beamline = "bare", p_central = {par["pCentral"]!r} &end
{alter("W1", "FILENAME", tmp("excess.w1"))}{alter("W1", "INTERVAL", 1)}&run_control n_steps = 1, n_passes = 24 &end
&twiss_output matched = 1, filename = "{tmp('excess.twi')}" &end
&bunched_beam bunch = "{tmp('excess.bun')}", n_particles_per_bunch = 10, use_twiss_command_values = 1, emit_nx = 1e-12, emit_ny = 1e-12,
  sigma_dp = 1e-9, sigma_s = 1e-9 &end
&track &end
&stop &end
""")
    run_elegant(OUTPUT / tmp("excess.ele"), 1)
    table = elegant_table(OUTPUT / tmp("excess.w1"))
    return float(np.polyfit(table["Pass"], table["Ct"], 1)[0])


def elegant_table(watch: Path, excess_s: float = 0.0) -> dict:
    """The watch point of the elegant run.  ``Ct`` is elegant's absolute time, so the arrival time of the reference particle
    (the sum of the revolution times at the central momentum, plus ``excess_s`` per turn) is taken off to leave the time offset the bunch has from it."""
    _, col = read_sdds(watch)
    momentum = col["pCentral"]
    energy = np.sqrt(1 + momentum**2) * ELECTRON_MASS_EV
    revolution = CIRCUMFERENCE_M / C_LIGHT * np.sqrt(1 + momentum**2) / momentum + excess_s  # C / (beta c) + excess
    passes = col["Pass"]
    reference = passes[0] * revolution[0] + np.concatenate([[0.0], np.cumsum(np.diff(passes) * 0.5 * (revolution[1:] + revolution[:-1]))])
    table = {name: col[name] for name in ("Pass", "Cdelta", "Sdelta", "St", "Sx", "Sy")}
    return dict(table, E0=energy, Ct=col["Ct"] - reference, Transmission=col["Transmission"])


def report(ele: dict, variants: dict[str, dict], title: str, png: Path) -> None:
    """Plot and tabulate the slipstream results (``variants``: label to table) against elegant."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(6, 1, figsize=(8, 14), sharex=True)
    labels = (("E0", 1e-9, "central energy [GeV]"), ("Cdelta", 1.0, "centroid delta"), ("Ct", 1e12, "centroid t - reference [ps]"),
              ("Sdelta", 1.0, "rms delta"), ("St", 1e12, "rms t [ps]"), ("Sx", 1e6, "rms x [um]"))
    for ax, (key, scale, label) in zip(axes, labels):
        ax.plot(ele["Pass"], ele[key] * scale, "k", lw=1.5, label="elegant")
        for name, table in variants.items():
            ax.plot(table["Pass"], table[key] * scale, "--", label=f"slipstream {name}".strip())
        ax.set_ylabel(label)
    axes[0].legend()
    axes[-1].set_xlabel("turn")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(png, dpi=130)
    print(f"saved {png}")
    names = list(variants)
    print(f"\n{'quantity':>10s} {'at':>5s} {'elegant':>12s}" + "".join(f"  {('slipstream ' + n).strip():>22s}" for n in names))
    for fraction in (0.25, 0.5, 0.75, 1.0):
        turn = fraction * ele["Pass"][-1]
        for key, scale, unit in (("Cdelta", 1.0, ""), ("Ct", 1e12, " ps"), ("Sdelta", 1.0, ""), ("St", 1e12, " ps"), ("Sx", 1e6, " um")):
            e = np.interp(turn, ele["Pass"], ele[key]) * scale
            cells = ""
            for n in names:
                v = np.interp(turn, variants[n]["Pass"], variants[n][key]) * scale
                diff = f"{(v / e - 1) * 100:+.1f} %" if key not in ("Cdelta", "Ct") else f"{v - e:+.3g}"
                cells += f"  {v:12.5g} ({diff:>7s})"
            print(f"{key + unit:>10s} {f'{fraction:.0%}':>5s} {e:12.5g}{cells}")
    print("\nrms difference to elegant over the window (elegant interpolated at the slipstream turns; the energy spread, bunch length"
          "\nand size are relative differences):")
    print(f"{'':>18s} {'centroid delta':>15s} {'centroid t [ps]':>16s} {'rms delta':>10s} {'rms t':>8s} {'rms x':>8s}")
    for name, table in variants.items():
        d = {key: np.interp(table["Pass"], ele["Pass"], ele[key]) for key in ("Cdelta", "Ct", "Sdelta", "St", "Sx")}
        rms = lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)))
        print(f"{('slipstream ' + name).strip():>18s} {rms(table['Cdelta'] - d['Cdelta']):15.2e} {rms(table['Ct'] - d['Ct']) * 1e12:16.1f} "
              + "".join(f"{rms(table[k] / d[k] - 1):10.1%}" for k in ("Sdelta", "St", "Sx")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start-ms", type=float, default=50.0, help="start of the window within the 100 ms booster ramp")
    parser.add_argument("--duration-ms", type=float, default=10.0, help="length of the window (the whole ramp: 100 from 0)")
    parser.add_argument("--balanced-phase", action="store_true",
                        help="RF phase table with V sin(phase) = U0 + dE/turn (a quiet bunch), not the machine's fixed 180 degrees")
    parser.add_argument("--n-kicks-bend", type=int, default=4, help="native: kicks per bend (elegant's default; 1 gives NaN)")
    parser.add_argument("--particles", type=int, default=500)
    parser.add_argument("--interval", type=int, default=200, help="turns between recorded points")
    parser.add_argument("--cores", type=int, default=1, help="elegant: MPI ranks for Pelegant (1: serial elegant)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rerun-elegant", action="store_true")
    parser.add_argument("--tag", default="", help="label of this slipstream run: its result is saved as slipstream_<window><tag>.npz")
    parser.add_argument("--overlay", nargs="*", default=[], metavar="TAG", help="also show these earlier slipstream runs")
    parser.add_argument("--elegant-only", action="store_true", help="run (or reuse) elegant and stop")
    parser.add_argument("--report-only", action="store_true", help="skip tracking; show --tag and --overlay from their saved results")
    args = parser.parse_args()
    OUTPUT.mkdir(exist_ok=True)

    tag = f"{args.start_ms:g}+{args.duration_ms:g}ms_{args.particles}p{'_bal' if args.balanced_phase else ''}"
    ramp_path = OUTPUT / f"ramp_{args.start_ms:g}+{args.duration_ms:g}ms{'_bal' if args.balanced_phase else ''}.sdds"
    global PREFIX
    PREFIX = f"{tag}_"
    window = (args.start_ms * 1e-3, args.duration_ms * 1e-3)
    info = write_ramp(ramp_path, *window)
    lattice, optics, par = injection_optics(info["p0"], info["f_rf_Hz"], ramp_path.name)
    excess_s = measure_time_excess(ramp_path.name, par)
    u0 = lattice.U0_eV if args.balanced_phase else None
    info = write_ramp(ramp_path, *window, u0_start_eV=u0, excess_s=excess_s)  # slipstream's ramp file
    elegant_ramp = ramp_path.with_name(ramp_path.stem + "_elegant.sdds")
    write_ramp(elegant_ramp, *window, u0_start_eV=u0, excess_s=excess_s, compensate_slip=True)  # the same, for elegant's phase reference
    n_passes = info["n_passes"]
    print(f"injection {lattice.E0_eV / 1e6:.1f} MeV, T0 {lattice.T0_s * 1e9:.2f} ns, {n_passes} turns, "
          f"excess revolution time {excess_s * 1e12:.2f} ps, U0 {lattice.U0_eV:.3f} eV, amplitude damping times (x, y, delta) = {par['taux']:.3g}, {par['tauy']:.3g}, {par['taudelta']:.3g} s")

    watch = OUTPUT / f"elegant_{tag}.w1"
    bunch_file = OUTPUT / f"elegant_{tag}.bun"
    if args.rerun_elegant or not (watch.exists() and bunch_file.exists()):
        (OUTPUT / tmp("ramp.ele")).write_text(elegant_ramp_input(elegant_ramp.name, par, n_passes, args.particles, args.interval, watch.name, bunch_file.name))
        start = time.time()
        run_elegant(OUTPUT / tmp("ramp.ele"), args.cores)
        print(f"  elegant: {n_passes} turns, {args.particles} particles in {time.time() - start:.0f} s")
    if args.elegant_only:
        return
    if not args.report_only:
        bunch = load_elegant_bunch(bunch_file)
        nat = track_native(lattice, optics, RampProfile.from_sdds_file(ramp_path), bunch, n_passes, args.interval, args.seed, args.n_kicks_bend)
        np.savez(OUTPUT / f"slipstream_{tag}{args.tag}.npz", **nat)
    ele = elegant_table(watch, excess_s)  # after the tracking, so that an elegant run still in progress can be waited for
    print(f"elegant transmission {ele['Transmission'].min():.4f} - {ele['Transmission'].max():.4f}")
    variants = {t: dict(np.load(OUTPUT / f"slipstream_{tag}{t}.npz")) for t in [*args.overlay, args.tag]}
    report(ele, variants, f"Diamond-II booster ramp {args.start_ms:g} to {args.start_ms + args.duration_ms:g} ms, {args.particles} particles",
           OUTPUT / f"compare_{tag}.png")


if __name__ == "__main__":
    main()
