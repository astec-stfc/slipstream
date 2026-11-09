"""CLARA S02 to screen forward model for the GPSR examples: LAURA lattice, SIMBA S02 bunch, tracker, scan and screen.

Shared by the transverse and longitudinal studies, which differ in the screen they image and the part of the line they repeat for each
scan row (see ``Line`` and ``Study``). Needs the optional packages LAURA and ``laura-lattices`` (the CLARA lattice) and a SIMBA openPMD bunch
at the entrance of S02 (``CLA-S02-SIM-APER-01.openpmd.hdf5``). Nothing of the machine is stored in this repository.

The line is split at the entrance of S07: everything upstream (S02 to L04, with the VBC on) is tracked once per setting-independent pass,
and only S07 (and the SP3 spectrometer, if the study uses it) up to the screen is repeated for each row of the scan table.
"""

from __future__ import annotations

import math
import os
import re
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import h5py
import numpy as np
import torch

from slipstream import ELECTRON_MASS_EV, C_LIGHT, LinacTracker, ScanModel, Screen
from slipstream.reconstruction import from_cheetah
from slipstream.tracking.native.lattice_import import parse_elegant_lattice
from slipstream.tracking.native.linac import resolve_wakes

SECTIONS = ["S02", "L02", "S03", "L03", "S04", "L4H", "S05", "VBC", "S06", "L04", "S07"]
VBC_ANGLE_RAD = 0.1254  # reproduces the SIMBA bunch length after the VBC (64.7 um rms) without CSR
TDC_TILT_RAD = math.pi / 2  # the LAURA export carries no deflection plane; the examples streak vertically
TDC_NAME = "CLA-S07-DIA-TDC-01"


@dataclass(frozen=True)
class Line:
    """The part of CLARA repeated for each scan row: S07 from its start up to ``s07_last``, then (if ``branch`` is given) a spectrometer
    section up to ``screen``. ``dipole`` of the branch gets ``dipole_angle_rad`` (LAURA exports the angle from the magnet current, which
    the lattice does not hold); a drift of the surveyed length joins ``s07_last`` to the dipole."""

    screen: str
    s07_last: str
    branch: str | None = None
    dipole: str | None = None
    dipole_angle_rad: float = 0.0
    quads_off: tuple[str, ...] = ()  # quadrupoles whose strength is set to zero


@dataclass(frozen=True)
class Study:
    """One reconstruction study: the line and screen, the quadrupoles scanned and the settings (pairs of ``k1``, 1/m^2) of the scan and of
    the held-out check, the TDC voltage of the 'on' rows and the screen."""

    name: str
    line: Line
    scan_quads: tuple[str, str]
    quad_pairs: tuple[tuple[float, float], ...]
    heldout: tuple[tuple[float, float], ...]
    tdc_voltage_V: float = 3.0e6
    pixels: int = 96
    pixel_size_m: float | tuple[float, float] = 3e-5  # one size, or (horizontal, vertical)
    psf_sigma_px: float = 1.0


def build_specs(workdir: str | Path, line: Line, vbc_angle_rad: float = VBC_ANGLE_RAD, rf_kicks: int | None = None,
                csr_drift_dz: float | None = None):
    """``(upstream, scan)`` element lists from LAURA's elegant export: S02 to L04, and ``line`` (S07, and the branch if any) up to and
    including the screen.

    LAURA writes its wake tables into the working directory, so this runs inside ``workdir``. Dipole radiation is switched off, the
    elegant edge model 3 is mapped to the supported first-order edge (1), the TDC deflects vertically and ``rf_kicks`` (if given) replaces
    the 261 to 360 sub-kicks of each travelling-wave cavity and ``csr_drift_dz`` the length of one CSR kick in the drifts after the dipoles
    (0.01 m as exported)."""
    import laura_lattices.CLARA as clara
    from laura.laura import LAURA
    from laura.translator.converters.section import SectionLatticeTranslator

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        sections = SECTIONS + ([line.branch] if line.branch else [])
        lattice = LAURA(lattice=clara, layout={"default_layout": "X", "layouts": {"X": sections}})
        vbc = lattice.sections["VBC"]
        for i, sign in zip((1, 2, 3, 4), (1, -1, -1, 1)):
            vbc.elements[f"CLA-VBC-MAG-DIP-0{i}"].magnetic.angle = sign * vbc_angle_rad
        feed = ""
        if line.branch:
            dipole = lattice.sections[line.branch].elements[line.dipole]
            dipole.magnetic.angle = line.dipole_angle_rad
            last = lattice.sections["S07"].elements[line.s07_last].physical
            gap = (dipole.physical.middle.z - dipole.magnetic.length / 2) - (last.middle.z + last.length / 2)
            feed = f"{line.branch}_feed: drift, l = {gap:.6f}\n"
        text = ""
        for name in sections:
            text += SectionLatticeTranslator.from_section(lattice.sections[name]).to_elegant() + "\n"
        text += feed

        def tokens(section, last):
            body = re.search(rf"^{section}: LINE = \((.*?)\)\s*$", text.replace("&\n", ""), re.MULTILINE | re.DOTALL).group(1)
            names = [t.strip() for t in body.split(",")]
            return names[: names.index(last) + 1]

        s07 = tokens("S07", line.s07_last)
        scan = s07 + ([f"{line.branch}_feed"] + tokens(line.branch, line.screen) if line.branch else [])
        text += f"UP: LINE=({','.join(SECTIONS[:-1])},{s07[0]})\nSCAN: LINE=({','.join(scan)})\n"
        lte = workdir / "clara_lattice.lte"
        lte.write_text(text)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            up, down = (resolve_wakes(parse_elegant_lattice(lte, root), workdir) for root in ("UP", "SCAN"))
    finally:
        os.chdir(cwd)

    def adjust(spec):
        if spec.kind == "bend":
            return replace(spec, synch_rad=False, isr=False, edge1_effects=1, edge2_effects=1)
        if spec.kind == "tdc":
            return replace(spec, tilt=TDC_TILT_RAD)
        if spec.kind == "quad" and spec.name in line.quads_off:
            return replace(spec, k1=0.0)
        if spec.kind == "rf" and rf_kicks:
            return replace(spec, n_kicks=rf_kicks)
        if spec.kind == "drift" and csr_drift_dz and spec.csr_drift_rho_m is not None:
            return replace(spec, csr_drift_dz=csr_drift_dz)
        return spec

    return [adjust(s) for s in up], [adjust(s) for s in down]


def load_bunch(path: str | Path, n: int | None = None, seed: int = 0, dtype=torch.float64):
    """SIMBA openPMD bunch as Slipstream ``(x, px, y, py, tau, delta)`` (``tau = c (t - mean t)``, late positive), the total
    reference energy of the bunch in eV and its charge in C; ``n`` particles are drawn at random if given."""
    with h5py.File(path) as f:
        e = f["particles/electron"]
        mom = np.stack([e[f"momentum/{k}"][:] for k in "xyz"], 1)  # eV/c
        pos = np.stack([e[f"position/{k}"][:] for k in "xyz"], 1)
        t = e["time"][:]
        charge = float(e["weight"].attrs["value"]) * len(t)
    p = np.linalg.norm(mom, axis=1)
    p0 = p.mean()
    state = np.stack([pos[:, 0], mom[:, 0] / p0, pos[:, 1], mom[:, 1] / p0, C_LIGHT * (t - t.mean()), (p - p0) / p0], 1)
    if n is not None and n < len(t):
        state = state[np.random.default_rng(seed).choice(len(t), n, replace=False)]
    return torch.tensor(state, dtype=dtype), float(math.sqrt(p0**2 + ELECTRON_MASS_EV**2)), charge


@dataclass
class Machine:
    """Everything the fit and the truth share: study, lattice, S02 energy and charge, fiducial arrival time, TDC phase and screen."""

    study: Study
    up_specs: list
    scan_specs: list
    energy_eV: float
    charge_C: float
    fiducial: object
    window: tuple[float, float]
    s07_energy_eV: float
    tdc_phase_rad: float
    screen: Screen
    device: str
    dtype: torch.dtype

    def trackers(self, csr: bool):
        kw = dict(device=self.device, dtype=self.dtype, compile=False, always_change_p0=False, bunch_charge_C=self.charge_C,
                  csr=csr, fast_collective_effects=csr, csr_drift_model="stupakov" if csr else "overtaking")
        up = LinacTracker(self.up_specs, self.energy_eV, fiducial=self.fiducial, **kw).with_wake_windows(
            window_s=self.window, deposit="cic", monitor_overflow=True)
        scan_specs = [replace(s, phase_rad=self.tdc_phase_rad) if s.kind == "tdc" else s for s in self.scan_specs]
        down = LinacTracker(scan_specs, self.s07_energy_eV, device=self.device, dtype=self.dtype, compile=False,
                            always_change_p0=False)
        return up, down


def make_machine(bunch_path, workdir, study: Study, device: str, dtype=torch.float64, n_reference: int = 4096,
                 rf_kicks: int | None = None, vbc_angle_rad: float = VBC_ANGLE_RAD, csr_drift_dz: float | None = None):
    """Lattice, fiducial reference and screen for a bunch file. The fiducial time (where the RF phases are referred to), the wake window
    and the TDC zero crossing are fixed once from ``n_reference`` particles of the true bunch, as an operator would set them."""
    up_specs, scan_specs = build_specs(workdir, study.line, vbc_angle_rad, rf_kicks, csr_drift_dz)
    ref, energy, charge = load_bunch(bunch_path, n_reference, seed=1, dtype=dtype)
    ref = ref.to(device)
    kw = dict(device=device, dtype=dtype, compile=False, always_change_p0=False)
    no_wake = [replace(s, include_z_wake=False, include_tr_wake=False, wake=None, transverse_wake_x=None, transverse_wake_y=None)
               if s.kind == "rf" else s for s in up_specs]
    fiducial = LinacTracker(no_wake, energy, **kw).fiducialize(ref)
    tau_s = ref[:, 4] / C_LIGHT
    span = float(tau_s.max() - tau_s.min())
    window = (float(tau_s.min()) - span, float(tau_s.max()) + span)
    machine = Machine(study, up_specs, scan_specs, energy, charge, fiducial, window, 0.0, 0.0, None, device, dtype)
    up, _ = machine.trackers(csr=False)
    out, e_out = up.track(ref)
    machine.s07_energy_eV = float(e_out)
    omega = 2.0 * math.pi * next(s.frequency_Hz for s in scan_specs if s.kind == "tdc")
    machine.tdc_phase_rad = -omega * float(out[:, 4].mean()) / C_LIGHT  # zero crossing at the centre of the bunch
    machine.screen = Screen(shape=(study.pixels, study.pixels), pixel_size=study.pixel_size_m, psf_sigma_px=study.psf_sigma_px,
                            axes=(0, 2))
    return machine


class ClaraForward(torch.nn.Module):
    """GPSR lattice for the whole line: S02 bunch to images on the study's screen at each row of ``(k1 quad 1, k1 quad 2, V_TDC)``.

    The upstream part (with or without CSR, as ``csr`` says) is tracked once, whatever the number of rows; the rest is a `ScanModel`."""

    def __init__(self, machine: Machine, csr: bool):
        super().__init__()
        self.machine = machine
        self.up, down = machine.trackers(csr)
        quads = machine.study.scan_quads
        self.scan = ScanModel(down, machine.screen, [(quads[0], "k1"), (quads[1], "k1"), (TDC_NAME, "voltage_V")])

    def set_lattice_parameters(self, x: torch.Tensor) -> None:
        self.scan.set_lattice_parameters(x)

    def bunch(self, beam) -> torch.Tensor:
        """Slipstream ``(n, 6)`` coordinates at the entrance of S02 from a tensor or a Cheetah-like beam."""
        if isinstance(beam, torch.Tensor):
            return beam
        return from_cheetah(beam.particles, self.machine.energy_eV, ELECTRON_MASS_EV)

    def track_and_observe(self, beam) -> tuple[torch.Tensor]:
        out, _ = self.up.track(self.bunch(beam))
        return self.scan.track_and_observe(out)
