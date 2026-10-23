"""Tracks cathode emission and Cartesian proper velocities through standing/traveling-wave cavities,
solenoids, magnets, apertures and cylindrical or 3D space charge with optional cathode images. See
docs/modules/tracking/native/injector.md."""
from __future__ import annotations

import dataclasses
import hashlib
import logging
import math
import os
import re
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import slipstream._compile_env  # noqa: F401

import numpy as np
import torch
import torch._dynamo
import yaml
from torch.utils.checkpoint import checkpoint

from ...analysis.emittance import geometric_emittance_xy as _geometric_emittance_m
from ...analysis.emittance import longitudinal_emittance_keV_mm as _longitudinal_emittance_keV_mm
from ...constants import C_LIGHT, E_CHARGE_C, ELECTRON_MASS_EV, EPS0_F_PER_M
from .astra_io import (
    cathode_emission_status, distribution_from_rows, momentum_eV_to_SI, read_astra_namelist,
)
from .generator import generate, read_generator_input
from .fields import RFCavityField, SolenoidField, TravelingWaveField, read_astra_field_file
from .gun import kahan_add, relativistic_space_charge_field
from .injector_ibs import BinaryCollisionIBS, InjectorIBSSettings
from .magnets import (
    Dipole, beam_angle, corrector_element, dipole_element, hard_quad_element, lateral_ok, lerp_tables, magnet_fields, placement,
    place_as_tensors, placement3, profile_quad_element, rotate_to_beam, solenoid_element, to_local, to_local3,
)
from .photocathode import PhotoEmitter, Photocathode
from .space_charge import SC3D_MAX_GRID_CELLS, CylindricalSpaceCharge, SpaceCharge3D, smooth_rho
from . import warp_kernels

MASS_KG = ELECTRON_MASS_EV * E_CHARGE_C / C_LIGHT**2
ELECTRON_Q_C = -E_CHARGE_C
_TWO_PI = 2.0 * math.pi
logger = logging.getLogger(__name__)

SC_IMAGE_NZ_REF = 64
SC_IMAGE_NZ_MAX = 1026  # memory-safety backstop, not an accuracy limit
SC_IMAGE_NZ_BUCKETS = (33, 65, 129, 257, 513, 1025)
SC_NR_BUCKETS = (64,)
SC_NZ_BUCKETS = (64, 128, 192, 256, 384, 512, 768)


def injector_exit_to_canonical6(
    result: dict, device: str | torch.device | None = None, dtype: torch.dtype = torch.float64,
    mc2_eV: float = ELECTRON_MASS_EV,
) -> tuple[torch.Tensor, float, float]:
    """Compute injector exit to canonical6; see the module documentation for details."""
    x, y, z = result["x"].double(), result["y"].double(), result["z"].double()
    ux, uy, uz = result["ux"].double(), result["uy"].double(), result["uz"].double()
    q, t_s = result["q"].double(), float(result["t_s"])

    gamma = torch.sqrt(1.0 + (ux**2 + uy**2 + uz**2) / C_LIGHT**2)
    vx, vy, vz = ux / gamma, uy / gamma, uz / gamma
    z_ref = z.mean()
    dt = (z_ref - z) / vz
    x2 = x + vx * dt
    y2 = y + vy * dt
    tau = C_LIGHT * (t_s + dt)

    u_mag = torch.sqrt(ux**2 + uy**2 + uz**2)
    u0 = u_mag.mean()
    delta = u_mag / u0 - 1.0
    px = ux / u0
    py = uy / u0

    particles6 = torch.stack([x2, px, y2, py, tau, delta], dim=-1).to(device=device, dtype=dtype)
    gamma0 = math.sqrt(1.0 + (float(u0) / C_LIGHT) ** 2)
    energy_eV = gamma0 * mc2_eV
    charge_C = float(q.abs().sum())
    return particles6, energy_eV, charge_C



@dataclass
class Cavity:
    """One RF cavity: standing wave (a plain 2-column on-axis ``Ez`` file) or travelling wave (ASTRA's
    own ``z1 z2 n m``-headed file format -- ``fields.TravelingWaveField``)."""
    file: str
    z_m: float
    e_max_V_per_m: float
    frequency_hz: float
    phase_deg: float = 0.0
    n_cells: int = 0
    crest_deg: float | None = None

    @property
    def is_traveling_wave(self) -> bool:
        return Path(self.file).name.upper().startswith("TWS")


@dataclass
class Solenoid:
    """Static solenoid from an on-axis ``Bz`` file, peak scaled to ``b_max_T`` (sign included --
    ASTRA's ``MaxB``)."""
    file: str
    z_m: float
    b_max_T: float
    x_m: float = 0.0
    xrot_rad: float = 0.0
    y_m: float = 0.0
    yrot_rad: float = 0.0

    @property
    def placed(self) -> bool:
        return bool(self.x_m or self.xrot_rad or self.y_m or self.yrot_rad)


@dataclass
class SpaceChargeSettings:
    """Space-charge resolution/cadence knobs."""
    enabled: bool = True
    cathode_image: bool = True
    nr: int = 64
    nz: int = 256
    box_margin: float = 6.0
    sigma_z_floor_m: float = 5e-5
    image_threshold_m: float = 0.05
    cells_per_sigma_r: float = 6.0
    cells_per_sigma_zs: float = 12.0
    min_particles_per_cell: float = 6.0
    every_step_until: int | None = None
    subcycle_steps: int = 20
    mesh: str = "adaptive"
    smooth_r: int = 0.0
    smooth_z: int = 0.0
    smooth_gamma_ref: float = 3.0
    emission_steps: int = 100  # steps across the emission window, ``Injector.dt_s`` capping it; 0: dt_s throughout
    dt_adaptive: bool = True
    geometry: str = "cylindrical"
    n_xyz: list[int] = field(default_factory=lambda: [32, 32, 32])
    max_grid_cells_3d: int = SC3D_MAX_GRID_CELLS  # padded Hockney volume cap, deliberately not nx*ny*nz
    # The step after emission is clamp(dt_safety / omega, dt_floor_s, dt_ceiling_s). The values (and subcycle_steps)
    # are the measured accuracy/time optimum on the CLARA decks at 20-250 pC: Stories/CLARA_InjectorConvergence.
    dt_safety: float = 0.35e6
    dt_floor_s: float = 1.5e-14  # a machine with a bunch of tens of fs needs a smaller floor
    dt_ceiling_s: float = 2e-12


@dataclass
class Quadrupole:
    """Describe Quadrupole; see the module documentation for details."""
    grad_T_per_m: float
    z_m: float  # the magnet's own center, not its start
    length_m: float
    profile: str | None = None
    noscale: bool = False
    multipoles: dict[int, float] = field(default_factory=dict)
    x_m: float = 0.0
    xrot_rad: float = 0.0


@dataclass
class Corrector:
    """A steering magnet: the uniform transverse field ``bx_T``, ``by_T`` over ``length_m`` centred on ``z_m``. It deflects
    an electron by ``q B L / p`` whatever its energy, so settings that centre a beam at one gun phase do not at another."""
    z_m: float
    length_m: float
    bx_T: float = 0.0
    by_T: float = 0.0


@dataclass
class Aperture:
    """Describe Aperture; see the module documentation for details."""
    kind: str
    z1_m: float
    z2_m: float
    r_m: float
    x_m: float = 0.0
    xrot_rad: float = 0.0


@dataclass
class Injector:
    distribution: str
    z_stop_m: float
    cavities: list[Cavity] = field(default_factory=list)
    solenoids: list[Solenoid] = field(default_factory=list)
    quadrupoles: list[Quadrupole] = field(default_factory=list)
    apertures: list[Aperture] = field(default_factory=list)
    dipoles: list[Dipole] = field(default_factory=list)
    absolute_phases: bool = False  # ASTRA Auto_phase=F: cavity ``phase_deg`` is the absolute ASTRA ``Phi``
    mc2_eV: float = ELECTRON_MASS_EV  # tracked species' rest mass -- every energy/gamma/Boris-push
    charge_sign: float = -1.0  # +1/-1; magnitude is always the elementary charge. Must match the
    screens: list[float] = field(default_factory=list)  # ASTRA deck's Screen(i), absolute z (m)
    space_charge: SpaceChargeSettings = field(default_factory=SpaceChargeSettings)
    dt_s: float = 1.0e-13  # resolves the emission spread and early acceleration;
    # ``from_astra`` overrides this with a charge-adaptive default (scales as
    # Q^-1/2 above 5 pC -- see ``_charge_adaptive_dt_s`` for the measured basis)
    source: str | None = None  # the ASTRA deck imported from (a machine-file path)
    generator: str | None = None  # an ASTRA ``generator.in`` (machine-file path): when set, the beam is generated
    # natively at track time (``tracking.native.generator``, any particle count) instead of read from ``distribution``
    generator_seed: int = 0
    generator_sequence: str | None = None  # None: the deck's own Noise_reduc; else 'halton' | 'sobol' | 'random'
    generator_overrides: dict = field(default_factory=dict)  # generator.in fields (ASTRA names/units, e.g.
    # ``{"sig_x": 0.3, "sig_y": 0.3, "q_total": 0.05}``) replacing the file's own values -- for scans
    phasing_key: str | None = None  # see auto_phase
    photocathode: Photocathode | None = None  # laser-driven emission (``track`` only); None: charges from the beam
    correctors: list[Corrector] = field(default_factory=list)  # steering magnets (lab-frame uniform fields)
    base_dir: Path = field(default=Path("."), repr=False, compare=False)
    ibs: InjectorIBSSettings = field(default_factory=InjectorIBSSettings)

    @property
    def mass_kg(self) -> float:
        return self.mc2_eV * E_CHARGE_C / C_LIGHT**2

    @property
    def q_C(self) -> float:
        """Single-particle charge (SI, signed) the Boris push uses --
        ``ELECTRON_Q_C`` when ``charge_sign=-1.0`` (the default)."""
        return self.charge_sign * E_CHARGE_C

    @property
    def qm_C_per_kg(self) -> float:
        return self.q_C / self.mass_kg

    # -- I/O ---------------------------------------------------------------

    @classmethod
    def from_astra(cls, path: str | Path, mc2_eV: float = ELECTRON_MASS_EV, charge_sign: float = -1.0) -> "Injector":
        """Compute from astra; see the module documentation for details."""
        path = Path(path)
        nml = read_astra_namelist(path)
        newrun, output = nml.get("newrun", {}), nml.get("output", {})
        charge, cav, sol = nml.get("charge", {}), nml.get("cavity", {}), nml.get("solenoid", {})
        quad, apert = nml.get("quadrupole", {}), nml.get("aperture", {})
        absolute_phases = not newrun.get("auto_phase", True)

        ignored = []
        if nml.get("wake", {}).get("lwake") and nml["wake"].get("wk_type"):
            ignored.append(f"{len(nml['wake']['wk_type'])} wakefield(s) (&WAKE)")
        dip = nml.get("dipole", {})
        if newrun.get("qbunch"):
            ignored.append("Qbunch rescaling (charge taken from the distribution file)")
        # Solenoid/quadrupole x-plane offsets and rotations are placements (magnets.placement); everything
        # else -- vertical offsets/tilts, and any cavity misalignment -- is not modelled.
        for grp, prefix, placeable in ((cav, "c_", False), (sol, "s_", True), (quad, "q_", True)):
            for suffix in ("xoff", "yoff", "xrot", "yrot", "zrot"):
                if placeable and suffix in ("xoff", "xrot"):
                    continue
                if any(grp.get(prefix + suffix, {}).values()):
                    ignored.append(f"{prefix.upper()}{suffix} (element misalignment)")
            if prefix != "q_" and any(grp.get(prefix + "noscale", {}).values()):
                ignored.append(f"{prefix.upper()}noscale (fields are always scaled to their peak)")
            if any(grp.get(prefix + "smooth", {}).values()):
                ignored.append(f"{prefix.upper()}smooth (field smoothing)")

        quadrupoles: list[Quadrupole] = []
        if quad.get("lquad") and quad.get("q_pos"):
            for i in sorted(quad["q_pos"]):
                qtype_raw = str(quad.get("q_type", {}).get(i, "standard"))
                qtype = qtype_raw.lower()
                grad = quad.get("q_grad", {}).get(i)
                noscale = bool(quad.get("q_noscale", {}).get(i, False))
                mult_b = {n: float(v) for (n, j), v in quad.get("q_mult_b", {}).items() if j == i and v}
                if any(j == i and v for (n, j), v in quad.get("q_mult_a", {}).items()):
                    ignored.append(f"QUADRUPOLE({i}): skew multipoles Q_mult_a")
                is_data = "data" in qtype
                if qtype != "standard" and not is_data:
                    ignored.append(f"QUADRUPOLE({i}): Q_type={qtype!r} (only 'standard' and '...data...' supported)")
                elif grad is None:
                    ignored.append(f"QUADRUPOLE({i}): Q_K given without Q_grad (needs a resolved reference momentum, unsupported)")
                else:
                    if mult_b and not is_data:
                        ignored.append(f"QUADRUPOLE({i}): Q_mult_b on a standard quadrupole (only data-profile ones)")
                        mult_b = {}
                    if noscale and not is_data:
                        ignored.append(f"QUADRUPOLE({i}): Q_noscale on a standard quadrupole")
                    unsupported = sorted(n for n in mult_b if n != 3)
                    if unsupported:
                        ignored.append(f"QUADRUPOLE({i}): Q_mult_b({unsupported}) (only the sextupole, n=3, is checked against ASTRA)")
                    quadrupoles.append(Quadrupole(
                        grad_T_per_m=float(grad), z_m=float(quad["q_pos"][i]),
                        length_m=float(quad.get("q_length", {}).get(i, 0.0)),
                        profile=qtype_raw if is_data else None, noscale=noscale and is_data,
                        multipoles={n: float(grad) * v for n, v in mult_b.items() if n == 3},
                        x_m=float(quad.get("q_xoff", {}).get(i, 0.0)),
                        xrot_rad=float(quad.get("q_xrot", {}).get(i, 0.0)),
                    ))

        _APERTURE_KINDS = {"RAD", "COL_X", "COL_Y", "SCR_X", "SCR_Y"}
        apertures: list[Aperture] = []
        if apert.get("lapert") and apert.get("file_aperture"):
            for i in sorted(apert["file_aperture"]):
                kind = str(apert["file_aperture"][i]).upper()
                if kind not in _APERTURE_KINDS:
                    ignored.append(f"APERTURE({i}): {kind!r} (only RAD/COL_X/COL_Y/SCR_X/SCR_Y keyword forms supported, not CIR or a file table)")
                    continue
                for key in ("a_yrot", "a_zrot", "a_yoff"):
                    if apert.get(key, {}).get(i, 0.0):
                        ignored.append(f"APERTURE({i}).{key} (out-of-plane misalignment)")
                a_pos = float(apert.get("a_pos", {}).get(i, 0.0))
                apertures.append(Aperture(
                    kind=kind,
                    z1_m=float(apert.get("ap_z1", {}).get(i, 0.0)) + a_pos,
                    z2_m=float(apert.get("ap_z2", {}).get(i, 0.0)) + a_pos,
                    r_m=float(apert.get("ap_r", {}).get(i, 0.0)) * 1e-3,
                    x_m=float(apert.get("a_xoff", {}).get(i, 0.0)),
                    xrot_rad=float(apert.get("a_xrot", {}).get(i, 0.0)),
                ))

        dipoles: list[Dipole] = []
        if dip.get("ldipole") and dip.get("d_type"):
            for i in sorted(dip["d_type"]):
                dtype = str(dip["d_type"][i]).lower()
                strength = dip.get("d_strength", {}).get(i, 0.0)
                if not dtype.startswith("h"):
                    ignored.append(f"DIPOLE({i}): D_Type={dtype!r} (only horizontal dipoles)")
                    continue
                if dip.get("d_radius", {}).get(i):
                    ignored.append(f"DIPOLE({i}): D_radius (field set from the reference momentum -- use D_strength)")
                    continue
                for key in ("d_yoff", "d_yrot", "d_zrot"):
                    if dip.get(key, {}).get(i, 0.0):
                        ignored.append(f"DIPOLE({i}).{key} (out-of-plane misalignment)")
                if not strength:
                    continue  # switched off
                pts = np.array([[float(v) for v in dip[f"d{k}"][i]] for k in (1, 2, 3, 4)])  # (x, z) rows
                th = float(dip.get("d_xrot", {}).get(i, 0.0))
                if th:  # ASTRA: rotate about the mean of the corners, from +z toward +x, then offset
                    c0 = pts.mean(axis=0)
                    dx, dz = pts[:, 0] - c0[0], pts[:, 1] - c0[1]
                    pts = np.column_stack([c0[0] + dx * math.cos(th) + dz * math.sin(th),
                                           c0[1] - dx * math.sin(th) + dz * math.cos(th)])
                pts = pts + np.array([float(dip.get("d_xoff", {}).get(i, 0.0)), float(dip.get("d_zoff", {}).get(i, 0.0))])
                gaps = dip.get("d_gap", {})
                dipoles.append(Dipole(
                    corners=pts.tolist(), strength_T=float(strength),
                    gap1_m=float(gaps.get((1, i), 0.05)), gap2_m=float(gaps.get((2, i), 0.05)),
                ))

        if ignored:
            warnings.warn(f"{path.name}: not modelled natively, ignored: " + "; ".join(ignored), stacklevel=2)

        cavities = [
            Cavity(
                file=cav["file_efield"][i],
                z_m=float(cav.get("c_pos", {}).get(i, 0.0)),
                e_max_V_per_m=float(cav["maxe"][i]) * 1e6,
                frequency_hz=float(cav["nue"][i]) * 1e9,
                phase_deg=float(cav.get("phi", {}).get(i, 0.0)),
                n_cells=int(cav.get("c_numb", {}).get(i, 0)),
                # absolute ASTRA phase Phi: sin(w t + Phi) = cos(w t + Phi - 90 deg) -> crest_deg + phase_deg
                crest_deg=-90.0 if absolute_phases else None,
            )
            for i in sorted(cav.get("file_efield", {}))
        ] if cav.get("lefield", True) else []
        solenoids = [
            Solenoid(
                file=sol["file_bfield"][i],
                z_m=float(sol.get("s_pos", {}).get(i, 0.0)),
                b_max_T=float(sol["maxb"][i]),
                x_m=float(sol.get("s_xoff", {}).get(i, 0.0)),
                xrot_rad=float(sol.get("s_xrot", {}).get(i, 0.0)),
            )
            for i in sorted(sol.get("file_bfield", {}))
        ] if sol.get("lbfield", True) else []
        return cls(
            distribution=newrun["distribution"],
            z_stop_m=float(output["zstop"]),
            cavities=cavities,
            solenoids=solenoids,
            quadrupoles=quadrupoles,
            apertures=apertures,
            dipoles=dipoles,
            absolute_phases=absolute_phases,
            mc2_eV=mc2_eV,
            charge_sign=charge_sign,
            screens=sorted(float(v) for v in output.get("screen", {}).values()),
            # dt_s=_charge_adaptive_dt_s(path.parent / newrun["distribution"]),
            space_charge=SpaceChargeSettings(
                enabled=bool(charge.get("lspch", True)),
                cathode_image=bool(charge.get("lmirror", True)),
                # ASTRA applies Smooth_x/y/z to its 3-D FFT algorithm only (manual 6.6); the cylindrical one never
                # smooths, so importing them there made the native solver blur a density ASTRA does not.
                smooth_r=int(max(charge.get("smooth_x", 0) or 0, charge.get("smooth_y", 0) or 0))
                if charge.get("lspch3d", False) else 0,
                smooth_z=int(charge.get("smooth_z", 0) or 0) if charge.get("lspch3d", False) else 0,
                geometry="3d" if charge.get("lspch3d", False) else "cylindrical",
                n_xyz=[int(charge.get(k, 32)) for k in ("nxf", "nyf", "nzf")],
            ),
            source=path.name,
            base_dir=path.parent,
        )

    def _generator_input(self) -> list:
        return [dataclasses.replace(g, **self.generator_overrides) for g in read_generator_input(self.resolve(self.generator))]

    def resolve(self, rel: str) -> Path:
        """A machine-file path (relative to the file it was loaded from) as a real path."""
        return self.base_dir / rel

    def save(self, path: str | Path) -> None:
        """Write the machine (incl. any cached crest phases) as YAML, with
        every file path rewritten relative to ``path``'s directory."""
        path = Path(path)

        def rel(p: str) -> str:
            try:
                return Path(os.path.relpath(self.resolve(p), path.parent)).as_posix()
            except ValueError:  # on another Windows drive: no relative path exists
                return self.resolve(p).resolve().as_posix()

        data = dataclasses.asdict(self)
        del data["base_dir"]
        data["distribution"] = rel(self.distribution)
        if self.generator:
            data["generator"] = rel(self.generator)
        if self.source:
            data["source"] = rel(self.source)
        for elem in data["cavities"] + data["solenoids"]:
            elem["file"] = rel(elem["file"])
        for elem in data["quadrupoles"]:
            if elem["profile"]:
                elem["profile"] = rel(elem["profile"])
        path.write_text(yaml.safe_dump(data, sort_keys=False))

    @classmethod
    def load(cls, path: str | Path) -> "Injector":
        path = Path(path)
        data = yaml.safe_load(path.read_text())
        data["cavities"] = [Cavity(**c) for c in data.get("cavities", [])]
        data["solenoids"] = [Solenoid(**s) for s in data.get("solenoids", [])]
        data["quadrupoles"] = [Quadrupole(**q) for q in data.get("quadrupoles", [])]
        data["apertures"] = [Aperture(**a) for a in data.get("apertures", [])]
        data["dipoles"] = [Dipole(**d) for d in data.get("dipoles", [])]
        data["correctors"] = [Corrector(**c) for c in data.get("correctors", [])]
        data["space_charge"] = SpaceChargeSettings(**data.get("space_charge", {}))
        if data.get("photocathode"):
            data["photocathode"] = Photocathode(**data["photocathode"])
        data["ibs"] = InjectorIBSSettings(**data.get("ibs", {}))
        return cls(**data, base_dir=path.parent)

    # -- fields --------------------------------------------------------------

    def _cavity_field(self, c: Cavity, phase_rad=0.0, device="cpu", dtype=torch.float64):
        path = self.resolve(c.file)
        if str(path).lower().endswith(".hdf5"):
            # Injector.from_laura's own field maps (LAURA's HDF5 format, not ASTRA text); lazily
            # imported since 'laura' is laura_fields' own optional dependency, not injector.py's.
            from .laura_fields import field_from_laura_hdf5
            return field_from_laura_hdf5(
                path, e_max_V_per_m=c.e_max_V_per_m, frequency_hz=c.frequency_hz,
                n_cells=c.n_cells, phase_rad=phase_rad, z_offset=c.z_m,
                device=device, dtype=dtype,
            )
        kw = dict(frequency_hz=c.frequency_hz, phase_rad=phase_rad, z_offset=c.z_m, device=device, dtype=dtype)
        if c.is_traveling_wave:
            return TravelingWaveField.from_astra_file(path, c.e_max_V_per_m, n_cells=c.n_cells, **kw)
        return RFCavityField.from_astra_file(path, c.e_max_V_per_m, **kw)

    def _solenoid_field(self, s: Solenoid, device="cpu", dtype=torch.float64):
        path = self.resolve(s.file)
        if str(path).lower().endswith(".hdf5"):
            from .laura_fields import field_from_laura_hdf5
            return field_from_laura_hdf5(path, b_max_T=s.b_max_T, z_offset=s.z_m, device=device, dtype=dtype)
        return SolenoidField.from_astra_file(path, b_max_T=s.b_max_T, z_offset=s.z_m, device=device, dtype=dtype)

    def fields(self, device="cpu", dtype=torch.float64, params: dict | None = None):
        """Compute fields; see the module documentation for details."""
        p = params or {}
        cavs = []
        for i, c in sorted(enumerate(self.cavities), key=lambda ic: ic[1].z_m):
            if c.e_max_V_per_m == 0.0:
                continue
            f = self._cavity_field(c, math.radians(c.crest_deg + c.phase_deg), device, dtype)
            if f"cavities[{i}].phase_deg" in p:
                f.phase_rad = (c.crest_deg + p[f"cavities[{i}].phase_deg"]) * (math.pi / 180.0)
            if f"cavities[{i}].e_max_V_per_m" in p:
                f.scale = f.scale * (p[f"cavities[{i}].e_max_V_per_m"] / c.e_max_V_per_m)
            cavs.append(f)
        sols = []
        for i, s in enumerate(self.solenoids):
            if s.b_max_T == 0.0 or s.placed:  # placed ones: magnet_elements
                continue
            f = self._solenoid_field(s, device, dtype)
            if f"solenoids[{i}].b_max_T" in p:
                f.scale = f.scale * (p[f"solenoids[{i}].b_max_T"] / s.b_max_T)
            sols.append(f)
        return cavs, sols

    def magnet_elements(self, device="cpu", dtype=torch.float64, frame_all: bool | None = None) -> tuple[tuple, tuple]:
        """Compute magnet elements; see the module documentation for details."""
        if frame_all is None:
            frame_all = bool(self.dipoles)
        elements, placed_sol = [], []
        for d in self.dipoles:
            elements.append(dipole_element(d))
        for s in self.solenoids:
            if s.b_max_T == 0.0 or not (frame_all or s.placed):
                continue
            f = self._solenoid_field(s, device, dtype)
            z0 = f.map.z0 + f.z_offset
            zend = z0 + f.map.dz * (f.map.n - 1)
            place = place_as_tensors(
                placement3(s.x_m, s.y_m, 0.5 * (z0 + zend), s.xrot_rad, s.yrot_rad, force=frame_all), device, dtype)
            el = solenoid_element(f.map.f * f.scale, f.map.df * f.scale, f.map.d2f * f.scale,
                                  z0, f.map.dz, f.map.n, zend, place)
            elements.append(el)
            placed_sol.append((place, el[2]))
        for q in self.quadrupoles:
            if q.grad_T_per_m == 0.0 and not q.multipoles:
                continue
            plain = q.profile is None and not q.multipoles
            if plain and not (frame_all or q.x_m or q.xrot_rad):
                continue  # _fused_step's own hard-edge tables
            if q.profile is None:
                zc = q.z_m
                place = place_as_tensors(placement(q.x_m, zc, q.xrot_rad, force=frame_all), device, dtype)
                elements.append(hard_quad_element(q.grad_T_per_m, zc - q.length_m / 2.0, zc + q.length_m / 2.0, place))
            else:
                z_file, prof = read_astra_field_file(self.resolve(q.profile))
                z_abs = q.z_m + z_file
                place = place_as_tensors(
                    placement(q.x_m, 0.5 * (z_abs[0] + z_abs[-1]), q.xrot_rad, force=frame_all), device, dtype)
                elements.append(profile_quad_element(z_abs, prof, q.grad_T_per_m, q.noscale, dict(q.multipoles),
                                                     place, device, dtype))
        for c in self.correctors:
            elements.append(corrector_element(
                torch.as_tensor(c.bx_T, device=device, dtype=dtype), torch.as_tensor(c.by_T, device=device, dtype=dtype),
                c.z_m - 0.5 * c.length_m, c.z_m + 0.5 * c.length_m))
        return tuple(elements), tuple(placed_sol)

    def _reference_particle(self) -> tuple[float, float, float]:
        """ASTRA's reference particle -- row 0 of the distribution file (or of the generated beam),
        absolute: ``(z_m, uz_m_per_s, t_s)``. Its ``z`` is the cathode."""
        if self.generator:
            row = self._generator_input()[0].reference_row()
        else:
            row = np.loadtxt(self.resolve(self.distribution), max_rows=1)
        uz = float(momentum_eV_to_SI(torch.tensor(row[5], dtype=torch.float64))) / self.mass_kg
        return float(row[2]), uz, float(row[6]) * 1e-9

    # -- auto-phasing ----------------------------------------------------------

    def _phasing_inputs_key(self) -> str:
        inputs = [(Path(c.file).name, c.z_m, c.e_max_V_per_m, c.frequency_hz, c.phase_deg, c.n_cells)
                  for c in self.cavities]
        return hashlib.sha1(repr((inputs, self._reference_particle())).encode()).hexdigest()[:16]

    def auto_phase(self, verbose: bool = True) -> list[float]:
        """ASTRA's ``Auto_phase`` (manual sec. 4.8): scan each cavity in turn, in z-order, for the
        phase of maximum on-axis energy gain (upstream cavities already at their own crest,
        downstream ones off)."""
        z0, uz0, t0 = self._reference_particle()
        order = sorted(range(len(self.cavities)), key=lambda i: self.cavities[i].z_m)
        fields, gains = [], [0.0] * len(self.cavities)
        for i in order:
            c = self.cavities[i]
            f = self._cavity_field(c)
            t_start = time.time()
            z_in = c.z_m + (0.0 if c.is_traveling_wave else f.map.z0)
            z_out = c.z_m + (f.length if c.is_traveling_wave else f.map.z0 + f.map.dz * (f.map.n - 1))

            def energies(deg):
                f.phase_rad = torch.deg2rad(deg)
                return _on_axis_energy_MeV(fields + [f], z0, uz0, t0, z_in, z_out,
                                           self.qm_C_per_kg, self.mc2_eV)

            coarse_deg = torch.arange(0.0, 360.0, 10.0, dtype=torch.float64)
            coarse = energies(coarse_deg)[1]
            peak = parabolic_vertex(coarse_deg, coarse, int(coarse.argmax()), 10.0)
            fine_deg = torch.arange(peak - 5.0, peak + 5.01, 0.2, dtype=torch.float64)
            e_in, e_out = energies(fine_deg)
            j = int(e_out.argmax())
            c.crest_deg = parabolic_vertex(fine_deg, e_out, j, 0.2) % 360.0
            gains[i] = float(e_out[j] - e_in[j])
            f.phase_rad = math.radians(c.crest_deg + c.phase_deg)
            fields.append(f)
            if verbose:
                print(f"auto-phase cavity {i + 1} ({Path(c.file).name}): crest {c.crest_deg:.2f} deg, "
                      f"gain {gains[i]:.3f} MeV, set to {c.phase_deg:+.1f} deg off crest "
                      f"[{time.time() - t_start:.1f}s]")
        self.phasing_key = self._phasing_inputs_key()
        return gains

    # -- tracking ----------------------------------------------------------------

    def initial_beam(self, n_particles: int | None, device, dtype: torch.dtype, verbose: bool = False) -> dict:
        """Compute initial beam; see the module documentation for details."""
        if self.generator:  # generated at exactly the requested count: nothing to subsample
            gen_in = self._generator_input()
            ipart = None if n_particles is None else n_particles + sum(i.n_probe for i in gen_in)
            dist = generate(gen_in, ipart=ipart, seed=self.generator_seed,
                            sequence=self.generator_sequence).particles(device, dtype)
            file_is_cathode = True  # generator.py's own non-cathode branch already writes clock=0
        else:
            raw = np.loadtxt(self.resolve(self.distribution))
            status = cathode_emission_status(raw)
            dist = distribution_from_rows(raw, device=device, dtype=dtype, status=status)
            file_is_cathode = status == (-1,)
        n_total = dist["x"].shape[0]
        if n_total == 0:
            raise ValueError(f"{self.generator or self.distribution}: no cathode (status -1) particles to emit")
        if n_particles is not None and n_particles < n_total and not self.generator:
            idx = torch.randperm(n_total)[:n_particles]
            weight = n_total / n_particles
        else:
            idx = torch.arange(n_total)
            weight = 1.0
        x = dist["x"][idx].clone()
        y = dist["y"][idx].clone()
        z = dist["z"][idx].clone()
        q = dist["q"][idx].clone() * weight
        ux = momentum_eV_to_SI(dist["px"][idx]) / self.mass_kg
        uy = momentum_eV_to_SI(dist["py"][idx]) / self.mass_kg
        uz = momentum_eV_to_SI(dist["pz"][idx]) / self.mass_kg
        emission_time_s = dist["clock"][idx].clone() * 1e-9
        if not file_is_cathode:
            emission_time_s = torch.zeros_like(emission_time_s)
        if verbose:
            print(f"{n_total} real macroparticles from {self.generator or self.distribution}; "
                  f"tracking {x.shape[0]} (weight {weight:.3f})")
        return dict(x=x, y=y, z=z, q=q, ux=ux, uy=uy, uz=uz, emission_time_s=emission_time_s)

    def track(
        self,
        n_particles: int | None = None,
        device: str | None = None,
        dtype: torch.dtype | str = torch.float64,
        use_compile: bool = True,
        record_every: int = 10,
        z_stop_m: float | None = None,
        mesh: str | None = None,
        verbose: bool = True,
        dump_dir: str | Path | None = None,
        return_particles: bool = False,
        screen_z_m: float | None = None,
        use_warp: bool = True,
    ) -> dict:
        """Track the distribution from the cathode to ``z_stop_m`` (default: the machine's own).

        With ``self.photocathode`` set, the laser model assigns each macroparticle's charge as it is emitted and
        ``"emission"`` reports the charges. ``use_warp=False`` keeps the CUDA step and space-charge deposit on
        ``torch.compile`` even where Warp is installed (faster where Python launch overhead dominates)."""
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.ibs.__post_init__()
        if self.ibs.enabled and dtype != torch.float64:
            raise ValueError("Injector IBS requires float64 tracking to resolve thermal collision impulses")
        _check_screen_plane(screen_z_m)
        sc = dataclasses.replace(self.space_charge, mesh=mesh or self.space_charge.mesh)
        z_target = self.z_stop_m if z_stop_m is None else z_stop_m
        dt = self.dt_s
        if not self.absolute_phases and (
                self.phasing_key != self._phasing_inputs_key() or any(c.crest_deg is None for c in self.cavities)):
            self.auto_phase(verbose=verbose)

        beam = self.initial_beam(n_particles, device, dtype, verbose=verbose)
        x, y, z, q, ux, uy, uz, emission_time_s = (
            beam[k] for k in ("x", "y", "z", "q", "ux", "uy", "uz", "emission_time_s"))
        n = x.shape[0]
        collisions = BinaryCollisionIBS(self.ibs, q, self.mass_kg, self.q_C) if self.ibs.enabled else None
        cathode_z = self._reference_particle()[0]
        emitter = None
        if self.photocathode is not None:  # the beam's charges only weight the laser pulse; emission assigns them
            if float(z.max()) - cathode_z > 1e-4:
                raise ValueError("photocathode emission needs a cathode distribution (macroparticles on the cathode)")
            emitter = PhotoEmitter(self.photocathode, torch.hypot(x, y), q)
            q = torch.zeros_like(q)
            z_wall = min(cathode_z, float(z.min()))  # a particle falling back through it is absorbed
            prev_active = torch.zeros(n, device=device, dtype=torch.bool)
            suppressed, returned = torch.zeros_like(prev_active), torch.zeros_like(prev_active)
            emission_pending = True

        bent = bool(self.dipoles)
        cav_fields, sol_fields = self.fields(device, dtype)
        magnets, placed_sol = self.magnet_elements(device, dtype)
        if bent:  # every solenoid/quadrupole is in `magnets` (framed); _fused_step's own tables stay empty
            sol_fields = []
        sw, tw, sol = _field_tables(cav_fields, sol_fields)
        # RF phases as 0-d tensors: a Python float is a compile-time constant, so every new phase of a scan would
        # recompile the step (~10 s each, then torch._dynamo's recompile limit drops it to eager).
        sw, tw = (tuple((*e[:-1], torch.as_tensor(e[-1], dtype=torch.float64, device=device)) for e in tab)
                  for tab in (sw, tw))
        quad = _quad_tables([q for q in self.quadrupoles
                             if not (bent or q.profile or q.multipoles or q.x_m or q.xrot_rad)])
        apertures = _aperture_tables(self.apertures)
        first_cavity_end = _field_span(cav_fields[0])[1] if cav_fields else None

        t = min(0.0, float(emission_time_s.min()))
        zero_n = torch.zeros(n, device=device, dtype=dtype)
        Ex_sc, Ey_sc, Ez_sc = zero_n.clone(), zero_n.clone(), zero_n.clone()
        Bx_sc, By_sc, Bz_sc = zero_n.clone(), zero_n.clone(), zero_n.clone()
        cx, cy, cz, cux, cuy, cuz = (zero_n.clone() for _ in range(6))  # per-particle Kahan terms

        emission_end_s = float(emission_time_s.max() * 2)
        # A macroparticle is pushed from the first step at or after its emission time, so a step that is coarse against
        # the emission window quantises the emission (measured: the CLARA 100 pC emittance moves by 20 % between a
        # 100 fs and a 25 fs step). Resolve the window with `emission_steps` steps; an in-flight beam has no window.
        # A photocathode run keeps `dt_s`: there it is the resolution of the emission law the user chose.
        dt_run = dt
        emission_window_s = float(emission_time_s.max() - emission_time_s.min())
        if sc.emission_steps and emission_window_s > 0.0 and self.photocathode is None:
            dt = min(dt, emission_window_s / sc.emission_steps)
        if sc.every_step_until is None:
            sc.every_step_until = max(0, int(math.ceil((emission_end_s - t) / dt)))
        all_active = torch.ones(n, device=device, dtype=torch.bool)
        alive = torch.ones(n, device=device, dtype=torch.bool)  # not yet lost on an aperture
        track_alive = bool(apertures) or emitter is not None  # losses are possible: mask the live particles
        screen_state = _empty_screen_state(x) if screen_z_m is not None else None
        # the plane is a tensor input of the compiled recorder: as a float every new plane would recompile it
        screen_plane = torch.as_tensor(screen_z_m, device=x.device, dtype=x.dtype) if screen_z_m is not None else None
        screen_fn = (_record_screen_crossing if not (use_compile and device == "cuda" and screen_z_m is not None)
                     else torch.compile(_record_screen_crossing, dynamic=False))
        sc_cache: dict = {}
        c2_inv = 1.0 / C_LIGHT**2
        # 0-d device tensors, not Python floats/scalars -- see docs/tracking_native.md's
        # "torch.compile discipline" and "dtype/precision" (t_ten stays float64 regardless of dtype).
        t_ten = torch.full((), t, device=device, dtype=torch.float64)
        dt_ten = torch.full((), dt, device=device, dtype=dtype)
        qmdt2 = self.qm_C_per_kg * (dt / 2.0)  # plain-float twin of qmdt2_ten, for the warp path below
        qmdt2_ten = torch.full((), qmdt2, device=device, dtype=dtype)
        dt_floor = sc.dt_floor_s
        dt_min_seen, dt_max_seen = dt, dt  # widest dt since the last (coarser) progress-bar update

        # See docs/tracking_native.md's "Injector warp backend" -- compile/warp only pay off on CUDA.
        step_fn, sc_fn, sc3d_fn, rec_fn = _step_with_apertures, None, None, _traj_stats
        stats_frame = (placed_sol, bent, self.mc2_eV, self.q_C)
        if use_compile and device == "cuda":
            # Recompile limit must clear the bucket-shape combinations _traj_stats sees -- see
            # docs/tracking_native.md's "torch.compile discipline".
            torch._dynamo.config.recompile_limit = max(
                torch._dynamo.config.recompile_limit,
                4 * (len(SC_NR_BUCKETS) + len(SC_NZ_BUCKETS) + len(SC_IMAGE_NZ_BUCKETS)),
            )
            rec_fn = torch.compile(_traj_stats, dynamic=False)
            using_warp = use_warp and warp_kernels.available(device, dtype)
            if using_warp:
                step_fn = _step_with_apertures_warp
                sc_fn = _fused_sc_fields_warp
                sc3d_fn = _fused_sc_fields_3d_warp
            else:
                step_fn = torch.compile(_step_with_apertures, dynamic=False)
                sc_fn = torch.compile(_fused_sc_fields, dynamic=False)
                sc3d_fn = torch.compile(_fused_sc_fields_3d, dynamic=False)
        else:
            using_warp = False

        t_run0 = time.time()
        step = 0
        centroid_z = 0.0
        max_steps = 10 * (int(1.5 * (z_target - cathode_z) / (C_LIGHT * dt)) + 10000)
        rec_rows: list = []  # one (11,) device row per record event, stacked and synced once at the end
        rec_t_ns: list = []
        dt_steps: list = []  # the step actually taken, every step -- track_differentiable(dt_schedule=) replays it
        image_steps = 0  # steps up to the last space-charge solve that used the cathode image (ditto, sc_image_steps)
        sc_precision_checked = False
        # Per-step observers (screen plane, aperture slabs, cathode wall) only act near their plane: after emission the
        # live z-range seen at the last host sync, widened by c*dt per step since, tells which can be skipped.
        z_extent, dt_since = (-math.inf, math.inf), 0.0
        slabs = None if any(len(ap) > 4 for ap in apertures) else [(ap[1], ap[2]) for ap in apertures]

        dump_dir_path = self.resolve(dump_dir) if dump_dir is not None else None
        stem = Path(self.source).stem if self.source else "injector"
        twiss_rows: list = []
        if dump_dir_path is not None:
            dump_dir_path.mkdir(parents=True, exist_ok=True)
            self.save(dump_dir_path / "machine.yaml")
            screen_zs = sorted(z for z in self.screens if z < z_target - 1e-6)
            dump_plan = [(z, f"Screen{i + 1:02d}") for i, z in enumerate(screen_zs)]
            dump_plan.append((z_target, "Endpoint"))
        else:
            dump_plan = []
        if verbose:
            # Total is distance, not a step count: dt_adaptive makes the step count unknowable up front.
            from tqdm import tqdm
            bar = tqdm(total=(z_target - cathode_z) * 1e3, unit="mm", desc="tracking",
                       bar_format="{l_bar}{bar}| {n:.0f}/{total:.0f} mm [{elapsed}<{remaining}] {postfix}")
        for step in range(max_steps):
            # Once every particle is emitted (~40 steps for CLARA), the active mask/where-merges become
            # identities -- skip them.
            emitted_all = t >= emission_end_s
            if emitted_all and dt != dt_run and not (sc.dt_adaptive and sc.enabled):
                # only the adaptive rule (which needs the space-charge solve) replaces the fine emission step otherwise
                dt = dt_run
                qmdt2 = self.qm_C_per_kg * (dt / 2.0)
                dt_ten.fill_(dt)
                qmdt2_ten.fill_(qmdt2)
            active = all_active if emitted_all else emission_time_s <= t
            if emitter is not None and emission_pending:
                new_idx = (active & ~prev_active).nonzero().squeeze(1)
                if new_idx.numel():  # the laser frees these macroparticles this step
                    xn, yn, zn = x[new_idx], y[new_idx], z[new_idx]
                    # the field at each macroparticle's own emission time, not at the step that releases it
                    t_emit = emission_time_s[new_idx].to(torch.float64)
                    e_ext = self.charge_sign * _applied_fields(xn, yn, zn, sw, tw, sol, quad, t_emit, c2_inv, magnets)[2]
                    src = prev_active & alive
                    cloud = (torch.hypot(x[src], y[src]).double(), (z[src] - cathode_z).clamp_min(0.0).double(),
                             q[src].abs().double())
                    q_new, _ = emitter.emit(new_idx, torch.hypot(xn, yn), e_ext, cloud)
                    q[new_idx] = (self.charge_sign * q_new).to(dtype)
                    gone = new_idx[q_new <= 0.0]  # no extracting field left: its photoelectrons return
                    alive[gone], suppressed[gone] = False, True
                prev_active = active
                emission_pending = not bool(active.all())
            counted = (alive if emitted_all else active & alive) if track_alive else active
            if sc.enabled and (not emitted_all or step < sc.every_step_until or step % sc.subcycle_steps == 0):
                # A bent beam is solved in its own frame -- see docs/tracking_native.md's "Bends and
                # off-axis elements".
                frame_angle = 0.0
                if bent and emitted_all and int(alive.sum()) > 1:
                    frame_angle = float(beam_angle(ux[alive], uz[alive]))
                # Excludes aperture-lost particles from SC deposition once any have been lost -- see
                # docs/tracking_native.md's Aperture note (ghost charge otherwise).
                sc_emitted_all = emitted_all and not (track_alive and not bool(alive.all()))
                if abs(frame_angle) > 1e-6:
                    (Ex_sc, Ey_sc, Ez_sc, Bx_sc, By_sc, Bz_sc), omega = _compute_space_charge_in_beam_frame(
                        frame_angle, x, y, z, q, ux, uy, uz, counted, device, dtype, sc_cache, sc, cathode_z,
                        sc_fn if sc_emitted_all else None,
                        sc3d_fn=sc3d_fn if sc_emitted_all else None,
                        emitted_all=sc_emitted_all, qm=self.qm_C_per_kg)
                else:
                    Ex_sc, Ey_sc, Ez_sc, Bx_sc, By_sc, Bz_sc = _compute_space_charge(
                        x, y, z, q, ux, uy, uz, counted, device, dtype, sc_cache, sc, cathode_z,
                        # Compiled only post-emission: before that x[counted] changes shape nearly every
                        # step and would recompile, blowing the limit and falling back to eager entirely.
                        sc_fn=sc_fn if sc_emitted_all else None,
                        sc3d_fn=sc3d_fn if sc_emitted_all else None,
                        emitted_all=sc_emitted_all,
                    )
                    omega = None
                if sc_cache.get("solver", (None,) * 5)[4]:  # the cylindrical solve just used the image
                    image_steps = step + 1
                # Checked once past the first cavity, where space charge is the only longitudinal field left.
                if (emitted_all and not sc_precision_checked and first_cavity_end is not None
                        and centroid_z > first_cavity_end):
                    sc_precision_checked = True
                    _warn_if_sc_kick_unresolvable(Ez_sc, uz, dt, dtype, self.qm_C_per_kg)

                if omega is None:  # only the adaptive step and the progress bar read it; it costs ~10 host syncs
                    omega = (_local_sc_omega(x, y, ux, uy, uz, Ex_sc, Ey_sc, Bx_sc, By_sc, counted, self.qm_C_per_kg)
                             if sc.dt_adaptive or verbose else 0.0)
                # dt stays at its pre-loop starting value through emission regardless of dt_adaptive -- see
                # docs/tracking_native.md's "Injector adaptive time-stepping".
                if sc.dt_adaptive and emitted_all:
                    dt = max(dt_floor, min(sc.dt_ceiling_s, sc.dt_safety / omega)) if omega > 0.0 else dt
                qmdt2 = self.qm_C_per_kg * (dt / 2.0)
                dt_ten.fill_(dt)
                qmdt2_ten.fill_(qmdt2)
                dt_min_seen, dt_max_seen = min(dt_min_seen, dt), max(dt_max_seen, dt)

            # Warp wants plain floats, not the recompile-avoidance tensors -- see
            # docs/tracking_native.md's "Injector warp backend".
            dt_arg, qmdt2_arg = (dt, qmdt2) if using_warp else (dt_ten, qmdt2_ten)
            if emitted_all:
                check_screen, check_aper, check_wall = _needed_checks(
                    z_extent, C_LIGHT * (dt_since + dt), screen_z_m if screen_state is not None else None, slabs,
                    z_wall if emitter is not None else -math.inf)
            else:
                check_screen = check_aper = check_wall = True
            if check_screen and screen_state is not None:
                screen_prev = (x, y, z, ux, uy, uz)
                if using_warp and emitted_all:
                    screen_prev = tuple(v.clone() for v in screen_prev)  # Warp pushes state in place.
                alive_prev = alive.clone() if using_warp and check_aper and apertures else alive
            (x, y, z, ux, uy, uz, cx, cy, cz, cux, cuy, cuz, alive) = step_fn(
                alive, apertures if check_aper else (),
                x, y, z, ux, uy, uz, cx, cy, cz, cux, cuy, cuz,
                Ex_sc, Ey_sc, Ez_sc, Bx_sc, By_sc, Bz_sc,
                sw, tw, sol, quad, t_ten, dt_arg, qmdt2_arg, emitted_all, active, c2_inv, magnets,
            )
            if collisions is not None:
                momentum = collisions.apply(torch.stack((x, y, z), -1), torch.stack((ux, uy, uz), -1),
                                             dt, active & alive)
                ux, uy, uz = momentum.unbind(-1)
            if check_screen and screen_state is not None:
                screen_state = screen_fn(
                    screen_state, screen_prev, (x, y, z, ux, uy, uz), active, alive_prev,
                    apertures, screen_plane, t_ten, dt_ten)
            if check_wall and emitter is not None:
                fell_back = active & alive & (z < z_wall)  # absorbed by the cathode
                alive, returned = alive & ~fell_back, returned | fell_back
            t += dt
            t_ten.add_(dt)
            dt_steps.append(dt)
            dt_since += dt

            # Host-synced centroid only every 10 steps -- enough for the
            # stop check, and drops a per-step GPU->CPU sync.
            if step % 10 == 0:
                if emitter is not None:  # the centroid of the particles still in the machine
                    live = alive if emitted_all else active & alive
                    if emitted_all and not bool(live.any()):
                        break  # everything returned to the cathode or was lost
                    if live.any():
                        centroid_z = float(z[live].mean())
                elif emitted_all:
                    centroid_z = float(z.mean())
                elif active.any():
                    centroid_z = float(z[active].mean())
                if emitted_all and (apertures or screen_state is not None or emitter is not None):
                    z_extent = tuple(torch.stack([torch.where(alive, z, math.inf).min(),
                                                  torch.where(alive, z, -math.inf).max()]).tolist())
                    dt_since = 0.0
                while dump_plan and centroid_z >= dump_plan[0][0]:
                    _, label = dump_plan.pop(0)
                    if int(counted.sum()) < 4:
                        if verbose:
                            tqdm.write(f"  skip dump {label}: only {int(counted.sum())} particles "
                                       f"emitted by z={centroid_z:.4f} m")
                        continue
                    xa, ya, za = x[counted], y[counted], z[counted]
                    uxa, uya, uza, qa = ux[counted], uy[counted], uz[counted], q[counted]
                    twiss_rows.append(_twiss_row_from_ensemble(xa, ya, za, uxa, uya, uza, sol, t, label, stem,
                                                               *stats_frame))
                    _dump_beam_openpmd(dump_dir_path / f"{stem}.{label}.h5", xa, ya, za, uxa, uya, uza, qa, t,
                                       mc2_eV=self.mc2_eV, species="proton" if self.charge_sign > 0 else "electron")
                    if verbose:
                        tqdm.write(f"  dumped {label} at z={float(za.mean()):.4f} m "
                                   f"(t={t * 1e9:.3f} ns, n={int(counted.sum())})")
                if record_every and step % record_every == 0 and (emitted_all or int(counted.sum()) >= 4):
                    if emitted_all:
                        # Unfiltered (fixed-shape, compiled rec_fn) until the first aperture loss, then
                        # alive-masked eagerly -- see docs/tracking_native.md's Aperture note.
                        if track_alive and not bool(alive.all()):
                            if bool(alive.any()):
                                rec_rows.append(_traj_stats(x[alive], y[alive], z[alive], ux[alive], uy[alive],
                                                            uz[alive], sol, *stats_frame))
                        else:
                            rec_rows.append(rec_fn(x, y, z, ux, uy, uz, sol, *stats_frame))
                    else:
                        rec_rows.append(_traj_stats(x[counted], y[counted], z[counted],
                                                    ux[counted], uy[counted], uz[counted], sol, *stats_frame))
                    rec_t_ns.append(float(t) * 1e9)
            if verbose and step % 100 == 0:
                safety = sc.dt_safety / omega if omega > 0 else 0
                bar.update(max(0.0, min(centroid_z, z_target) - cathode_z) * 1e3 - bar.n)
                bar.set_postfix(step=step, t_ps=f"{t * 1e12:.1f}", n=int(counted.sum()),
                                dt_ps=f"{dt_min_seen * 1e12:.3f}-{dt_max_seen * 1e12:.3f}",
                                safety=f"{safety * 1e12:.3f}")
                dt_min_seen, dt_max_seen = dt, dt
            if step > 50 and centroid_z >= z_target:
                if screen_state is None or bool((screen_state[-1] | ~alive).all()):
                    break

        if verbose:
            bar.update(bar.total - bar.n)
            bar.close()
            print(f"tracking loop: {step + 1} steps, {time.time() - t_run0:.1f}s")
        loop_time_s = time.time() - t_run0

        if dump_dir_path is not None and twiss_rows:
            _write_twiss_hdf5(dump_dir_path / f"{stem}.twiss.h5", twiss_rows, source=self.source)

        # Final snapshot at the actual stop point (the loop breaks between record events).
        n_alive = int(alive.sum())
        if n_alive and (not rec_rows or abs(float(z.mean()) - float(rec_rows[-1][0])) > 1e-6):
            rec_rows.append(_traj_stats(x[alive], y[alive], z[alive], ux[alive], uy[alive], uz[alive], sol,
                                        *stats_frame))
            rec_t_ns.append(float(t) * 1e9)
        rows = torch.stack(rec_rows).tolist() if rec_rows else []
        traj: dict[str, list] = {k: [r[i] for r in rows] for i, k in enumerate(TRAJ_KEYS)}
        traj["t_ns"] = rec_t_ns

        gamma = torch.sqrt(1.0 + (ux[alive] ** 2 + uy[alive] ** 2 + uz[alive] ** 2) / C_LIGHT**2)
        x_b, z_b = x[alive], z[alive]
        if bent:  # sizes in the beam's own frame (the lab x of a bent beam is not a transverse coordinate)
            x_b, z_b, _, _ = rotate_to_beam(x_b.double(), z_b.double(), ux[alive].double(), uz[alive].double(),
                                            beam_angle(ux[alive].double(), uz[alive].double()))
        out = {
            "z_centroid_m": centroid_z,
            "t_ns": t * 1e9,
            "loop_time_s": loop_time_s,
            "n_steps": step + 1,
            "dt_steps_s": np.asarray(dt_steps),
            "sc_image_steps": image_steps,
            "n_particles_alive": n_alive,
            "n_lost_on_aperture": n - n_alive,
            "Ekin_MeV": float((gamma - 1.0).mean()) * self.mc2_eV / 1e6,
            "sigma_x_mm": _std_mm(x_b),
            "sigma_y_mm": _std_mm(y[alive]),
            "sigma_z_mm": _std_mm(z_b),
            "trajectory": traj,
            "dump_dir": str(dump_dir_path) if dump_dir_path is not None else None,
        }
        if emitter is not None:
            charge = q.abs()
            n_suppressed, n_returned = int(suppressed.sum()), int(returned.sum())
            out["n_lost_on_aperture"] = n - n_alive - n_suppressed - n_returned
            out["emission"] = {
                "photons": self.photocathode.photons,
                "ideal_charge_C": float(emitter.ideal_C),  # applied-field quantum efficiency, no screening
                "emitted_charge_C": float(charge.sum()),
                "returned_charge_C": float(charge[returned].sum()),
                "charge_C": float(charge[alive].sum()),  # still in the machine at the end
                "n_suppressed": n_suppressed,
                "n_returned": n_returned,
                "q": q,  # signed charge of every macroparticle (0: never emitted)
            }
        if screen_state is not None:
            out["screen"] = _screen_result(screen_state, q, alive, screen_z_m)
        if return_particles:
            out.update(
                x=x[alive], y=y[alive], z=z[alive], ux=ux[alive], uy=uy[alive], uz=uz[alive], q=q[alive], t_s=t,
            )
        if collisions is not None:
            out["ibs"] = dict(settings=dataclasses.asdict(self.ibs), **collisions.diagnostics())
        return out

    # -- differentiable tracking -------------------------------------------------------------

    def _check_knobs(self, params: dict) -> None:
        bent = bool(self.dipoles)
        for key in params:
            if key in ("charge_scale", "spot_scale"):
                continue
            m = re.fullmatch(r"(\w+)\[(\d+)\]\.(\w+)", key)
            if not m or m[3] not in _DIFF_KNOBS.get(m[1], ()) or int(m[2]) >= len(getattr(self, m[1])):
                known = ", ".join(f"{k}[i].{a}" for k, v in _DIFF_KNOBS.items() for a in v)
                raise KeyError(f"{key!r} is not a differentiable knob of this machine "
                               f"(known: charge_scale, spot_scale, {known})")
            kind, el = m[1], getattr(self, m[1])[int(m[2])]
            if (kind == "cavities" and el.e_max_V_per_m == 0.0) or (kind == "solenoids" and el.b_max_T == 0.0):
                raise ValueError(f"{key}: the element is switched off (zero amplitude), so its field is never built")
            if (kind == "solenoids" and (bent or el.placed)) or (
                    kind == "quadrupoles" and (bent or el.profile or el.multipoles or el.x_m or el.xrot_rad)):
                raise NotImplementedError(f"{key}: placed/framed elements (magnets.py) have no differentiable knobs yet")

    def track_differentiable(
        self,
        params: dict | None = None,
        *,
        n_steps: int | None = None,
        dt_schedule=None,
        beam: dict | None = None,
        n_particles: int | None = None,
        device: str | None = None,
        dtype: torch.dtype | str = torch.float64,
        use_compile: bool = False,
        record_every: int = 0,
        space_charge: bool | None = None,
        sc_box_m: tuple[float, float] | None = None,
        sc_image_steps: int = 0,
        checkpoint_every: int = 0,
        return_particles: bool = False,
        screen_z_m: float | None = None,
    ) -> dict:
        """Compute track differentiable; see the module documentation for details."""
        if self.ibs.enabled:
            raise NotImplementedError("Injector IBS is stochastic and not supported by track_differentiable; use Injector.track or disable ibs")
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        _check_screen_plane(screen_z_m)
        if self.photocathode is not None:
            raise NotImplementedError("photocathode emission is only supported by track()")
        self._check_knobs(params or {})
        # 0-d float64 on the tracking device: an RF phase needs float64 (see _rf_cos_sin).
        params = {k: torch.as_tensor(v, dtype=torch.float64, device=device) for k, v in (params or {}).items()}
        if not self.absolute_phases and (
                self.phasing_key != self._phasing_inputs_key() or any(c.crest_deg is None for c in self.cavities)):
            raise ValueError("crest phases missing or stale: call auto_phase() first (a scan + argmax is not "
                             "differentiable, so track_differentiable never runs it)")
        sc = self.space_charge
        sc_on = sc.enabled if space_charge is None else space_charge
        if sc_on and (sc.geometry != "cylindrical" or self.dipoles):
            raise NotImplementedError("differentiable space charge is cylindrical and straight-line only "
                                      "(no 3D, no beam-frame solve after a bend): pass space_charge=False")

        if beam is None:
            beam = self.initial_beam(n_particles, device, dtype)
        x, y, z, ux, uy, uz = (beam[k] for k in ("x", "y", "z", "ux", "uy", "uz"))
        if "spot_scale" in params:
            x, y = x * params["spot_scale"], y * params["spot_scale"]
        q = beam["q"] * params["charge_scale"] if "charge_scale" in params else beam["q"]
        emission_time_s = beam["emission_time_s"]
        n = x.shape[0]

        bent = bool(self.dipoles)
        cav_fields, sol_fields = self.fields(device, dtype, params)
        magnets, placed_sol = self.magnet_elements(device, dtype)
        if bent:
            sol_fields = []
        sw, tw, sol = _field_tables(cav_fields, sol_fields)
        quad = _quad_tables([
            dataclasses.replace(qd, grad_T_per_m=params.get(f"quadrupoles[{i}].grad_T_per_m", qd.grad_T_per_m))
            for i, qd in enumerate(self.quadrupoles)
            if not (bent or qd.profile or qd.multipoles or qd.x_m or qd.xrot_rad)])
        apertures = _aperture_tables(self.apertures)
        stats_frame = (placed_sol, bent, self.mc2_eV, self.q_C)
        if use_compile:
            torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)
        step_fn = torch.compile(_fused_step, dynamic=False) if use_compile else _fused_step
        ap_fn = torch.compile(_apply_apertures, dynamic=False) if use_compile else _apply_apertures
        screen_fn = (torch.compile(_record_screen_crossing, dynamic=False)
                     if use_compile and screen_z_m is not None else _record_screen_crossing)
        screen_plane = torch.as_tensor(screen_z_m, device=x.device, dtype=x.dtype) if screen_z_m is not None else None

        if sc_on:
            nz = sc.nz + 1 - sc.nz % 2 if sc_image_steps else sc.nz  # image: the cathode on the centre node (odd nz)
            # Only the shape is used (P_left/P_right/coefficients); the box comes per call.
            solver = CylindricalSpaceCharge(nr=sc.nr, nz=nz, r_max=1.0, z_min=-1.0, z_max=1.0, device=device, dtype=dtype)
            cathode_z = self._reference_particle()[0]
            sc_fn = torch.compile(_fused_sc_fields, dynamic=False) if use_compile else _fused_sc_fields

        # Same clock/schedule as track() (dt fixed, so the emission schedule is too).
        t = min(0.0, float(emission_time_s.min()))
        if dt_schedule is not None:
            dt_schedule = [float(v) for v in dt_schedule]
            n_steps = len(dt_schedule) if n_steps is None else n_steps
            if len(dt_schedule) < n_steps:
                raise ValueError(f"dt_schedule has {len(dt_schedule)} steps, n_steps is {n_steps}")
        elif n_steps is None:
            raise TypeError("track_differentiable needs n_steps (or a dt_schedule to take it from)")
        dt = dt_schedule[0] if dt_schedule else self.dt_s
        emission_end_s = float(emission_time_s.max() * 2)
        every_step_until = sc.every_step_until
        if every_step_until is None:
            every_step_until = max(0, int(math.ceil((emission_end_s - t) / dt)))
        zero_n = torch.zeros(n, device=device, dtype=dtype)
        all_active = torch.ones(n, device=device, dtype=torch.bool)
        c2_inv = 1.0 / C_LIGHT**2

        def step_tensors(dt):  # fresh 0-d tensors, never refilled in place: the graph keeps the old ones
            return (torch.full((), dt, device=device, dtype=dtype),
                    torch.full((), self.qm_C_per_kg * (dt / 2.0), device=device, dtype=dtype))

        def advance(step0, step1, t, alive, box_overflow, *state):
            """Steps ``[step0, step1)``: a pure function of its arguments (trajectory rows are returned, not
            appended), so ``checkpoint`` can rerun it during backward."""
            x, y, z, ux, uy, uz, cx, cy, cz, cux, cuy, cuz = state[:12]
            E_sc, t_ten = state[12:18], state[18]
            screen_state = state[19:] if screen_z_m is not None else None
            rows, rec_t_ns = [], []
            dt = dt_schedule[step0] if dt_schedule else self.dt_s
            dt_ten, qmdt2_ten = step_tensors(dt)
            for step in range(step0, step1):
                if dt_schedule and dt_schedule[step] != dt:
                    dt = dt_schedule[step]
                    dt_ten, qmdt2_ten = step_tensors(dt)
                emitted_all = t >= emission_end_s
                active = all_active if emitted_all else emission_time_s <= t
                if sc_on and (not emitted_all or step < every_step_until or step % sc.subcycle_steps == 0) and (
                        emitted_all or int(active.sum()) >= 4):
                    E_sc, overflow = _differentiable_space_charge(
                        x, y, z, q, ux, uy, uz, None if emitted_all else active, solver, sc_box_m,
                        cathode_z if step < sc_image_steps else None, sc, sc_fn)
                    box_overflow = torch.maximum(box_overflow, overflow)
                prev = (x, y, z)
                if screen_state is not None:
                    screen_prev = (x, y, z, ux, uy, uz)
                (x, y, z, ux, uy, uz, cx, cy, cz, cux, cuy, cuz) = step_fn(
                    x, y, z, ux, uy, uz, cx, cy, cz, cux, cuy, cuz, *E_sc,
                    sw, tw, sol, quad, t_ten, dt_ten, qmdt2_ten, emitted_all, active, c2_inv, magnets,
                )
                if screen_state is not None:
                    screen_state = screen_fn(
                        screen_state, screen_prev, (x, y, z, ux, uy, uz), active, alive,
                        apertures, screen_plane, t_ten, dt_ten)
                if apertures:
                    with torch.no_grad():
                        alive = ap_fn(x, y, z, alive, apertures, prev=prev)
                t += dt
                t_ten = t_ten + dt  # not add_(): an in-place clock bump would invalidate saved graph inputs
                if record_every and step % record_every == 0 and (emitted_all or int(active.sum()) >= 4):
                    sel = (x, y, z, ux, uy, uz) if emitted_all else (a[active] for a in (x, y, z, ux, uy, uz))
                    rows.append(_traj_stats(*sel, sol, *stats_frame))
                    rec_t_ns.append(t * 1e9)
            return t, alive, box_overflow, rows, rec_t_ns, (x, y, z, ux, uy, uz, cx, cy, cz, cux, cuy, cuz,
                                                            *E_sc, t_ten, *(screen_state or ()))

        # Particles, their Kahan terms, the held space-charge field, the clock.
        state = (x, y, z, ux, uy, uz, *(zero_n.clone() for _ in range(12)),
                 torch.full((), t, device=device, dtype=torch.float64))
        if screen_z_m is not None:
            state += _empty_screen_state(x)
        alive = all_active.clone()
        box_overflow = torch.zeros((), device=device, dtype=dtype)
        rows, rec_t_ns = [], []
        block = checkpoint_every or max(n_steps, 1)
        for step0 in range(0, n_steps, block):
            args = (step0, min(step0 + block, n_steps), t, alive, box_overflow, *state)
            t, alive, box_overflow, r, tn, state = (
                checkpoint(advance, *args, use_reentrant=False) if checkpoint_every else advance(*args))
            rows += r
            rec_t_ns += tn
        x, y, z, ux, uy, uz = state[:6]

        emitted = emission_time_s <= t
        final = _traj_stats(*(a[emitted] for a in (x, y, z, ux, uy, uz)), sol, *stats_frame)
        out = {k: final[i] for i, k in enumerate(TRAJ_KEYS)}
        out.update(t_ns=t * 1e9, n_steps=n_steps, n_lost_on_aperture=n - int(alive.sum()))
        if sc_on:
            out["sc_box_overflow"] = float(box_overflow)
            if out["sc_box_overflow"] > 0.0:
                warnings.warn(f"{out['sc_box_overflow']:.2%} of the particles left the space-charge box "
                              f"(sc_box_m={sc_box_m}) at some step and were deposited on its edge")
        if rows:
            traj = torch.stack(rows)
            out["trajectory"] = {k: traj[:, i] for i, k in enumerate(TRAJ_KEYS)} | {"t_ns": rec_t_ns}
        if screen_z_m is not None:
            out["screen"] = _screen_result(state[19:], q, alive, screen_z_m)
        if return_particles:
            out.update(x=x, y=y, z=z, ux=ux, uy=uy, uz=uz, q=q, alive=alive, t_s=t)
        return out


def _check_screen_plane(screen_z_m):
    if screen_z_m is not None and not math.isfinite(screen_z_m):
        raise ValueError("screen_z_m must be a finite lab-frame plane in metres")


def _needed_checks(extent, travel, screen_z_m, slabs, z_wall, margin: float = 1e-5):
    """``(screen, apertures, wall)``: which per-step observers can act. ``extent`` is the ``(z_lo, z_hi)`` of the live
    particles at the last host sync and ``travel`` the farthest any can have moved since (``c*dt`` per step), so a
    plane, aperture slab or cathode wall out of reach cannot be crossed and its check is skipped without changing any
    result. ``slabs`` is ``None`` for placed apertures (their lab extent is not tracked): always checked."""
    z_lo, z_hi = extent[0] - travel - margin, extent[1] + travel + margin
    return (screen_z_m is not None and z_hi >= screen_z_m,
            slabs is None or any(z_hi >= a and z_lo <= b for a, b in slabs),
            z_lo < z_wall)


def _std_mm(values: torch.Tensor) -> float:
    """RMS size in mm, NaN (not a warning) when fewer than two particles remain."""
    return float(values.std()) * 1e3 if values.numel() > 1 else float("nan")


def _empty_screen_state(x):
    return (*(torch.zeros_like(x) for _ in range(6)),
            torch.zeros_like(x, dtype=torch.float64), torch.zeros_like(x, dtype=torch.bool))


def _record_screen_crossing(state, prev, current, active, alive, apertures, z_plane, t, dt):
    """Record the first forward crossing on the pusher's straight step segment."""
    dz = current[2] - prev[2]
    forward = dz > 0.0
    fraction = ((z_plane - prev[2]) / torch.where(forward, dz, torch.ones_like(dz))).clamp(0.0, 1.0)
    at_plane = tuple(a + fraction * (b - a) for a, b in zip(prev, current))
    with torch.no_grad():
        hit = ~state[-1] & active & alive & forward & (prev[2] <= z_plane) & (current[2] >= z_plane)
        if apertures:
            # Check only the segment up to the plane: later losses do not erase a screen observation.
            hit = hit & _apply_apertures(*at_plane[:3], alive, apertures, prev=prev[:3])
    values = (*at_plane[:2], torch.zeros_like(prev[2]) + z_plane, *at_plane[3:], t + fraction.double() * dt)
    return (*(torch.where(hit, new, old) for old, new in zip(state[:-1], values)), state[-1] | hit)


def _screen_result(state, q, alive, z_plane):
    crossed = state[-1]
    missing = ~crossed & alive
    return dict(zip(("x", "y", "z", "ux", "uy", "uz", "t_s"), state[:-1])) | {
        "z_plane_m": z_plane, "crossed": crossed, "q": torch.where(crossed, q, 0.0),
        "n_crossed": int(crossed.sum()), "n_unobserved_alive": int(missing.sum()),
        "n_lost_before_screen": int((~crossed & ~alive).sum()),
        "incomplete_charge_C": float(q[missing].detach().abs().sum()),
    }


# Knobs track_differentiable accepts, per element list ("cavities[i].phase_deg" ...) -- plus "charge_scale".
_DIFF_KNOBS = {"cavities": ("phase_deg", "e_max_V_per_m"), "solenoids": ("b_max_T",),
               "quadrupoles": ("grad_T_per_m",)}



def parabolic_vertex(degs: torch.Tensor, gains: torch.Tensor, i: int, h: float) -> float:
    """Sub-grid peak position from the three points bracketing index ``i`` -- the gain is locally
    quadratic in phase at the crest, so this locates the true peak well under the grid spacing."""
    if not (0 < i < degs.numel() - 1):
        return float(degs[i])
    e_m, e_0, e_p = float(gains[i - 1]), float(gains[i]), float(gains[i + 1])
    denom = e_m - 2.0 * e_0 + e_p
    if denom >= 0.0:
        return float(degs[i])
    offset = 0.5 * (e_m - e_p) / denom
    return float(degs[i]) + (offset * h if abs(offset) <= 1.0 else 0.0)


def _field_span(f) -> tuple[float, float]:
    """Lab-frame z-range over which a cavity field is nonzero."""
    if isinstance(f, TravelingWaveField):
        return f.z_offset, f.z_offset + f.length
    z0 = f.map.z0 + f.z_offset
    return z0, z0 + f.map.dz * (f.map.n - 1)


def _ez_on_axis(f, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """On-axis ``Ez`` of either cavity type (paraxial ``Er``/``Bphi`` vanish
    on-axis identically). ``f.phase_rad`` may be a tensor of trial phases,
    broadcasting to one candidate per particle."""
    if isinstance(f, TravelingWaveField):
        return f.on_axis(z, t)[0]
    z0, zend = _field_span(f)
    (ez0,) = _lerp_tables(z, z0, f.map.dz, f.map.n, zend, f.map.f)
    return ez0 * f.scale * torch.cos(f.omega * t + f.phase_rad)


def _on_axis_energy_MeV(fields, z0, uz0, t0, z_in, z_out, qm, mc2_eV, dt_fine=1.0e-13, dt_coarse=2.0e-12,
                        beta_coarse=0.9):
    """Compute on axis energy MeV; see the module documentation for details."""
    n = max((torch.as_tensor(f.phase_rad).numel() for f in fields), default=1)
    spans = [_field_span(f) for f in fields]
    z = torch.full((n,), z0, dtype=torch.float64)
    uz = torch.full((n,), uz0, dtype=torch.float64)
    e_in = torch.full((n,), math.nan, dtype=torch.float64)
    e_out = torch.zeros(n, dtype=torch.float64)
    done = torch.zeros(n, dtype=torch.bool)
    u_coarse = beta_coarse / math.sqrt(1.0 - beta_coarse**2) * C_LIGHT
    t = t0
    t_max = t0 + 10.0 * (z_out - z0) / C_LIGHT + 1e-9
    while True:
        z_lo, z_hi, u_min = torch.stack([z.min(), z.max(), torch.where(done, math.inf, uz).min()]).tolist()
        dt = dt_coarse if u_min > u_coarse else dt_fine
        z = z + uz / torch.sqrt(1.0 + (uz / C_LIGHT) ** 2) * (0.5 * dt)
        t_half = torch.tensor(t + 0.5 * dt, dtype=torch.float64)
        for f, (lo, hi) in zip(fields, spans):
            if z_hi >= lo and z_lo <= hi:
                uz = uz + (qm * dt) * _ez_on_axis(f, z, t_half)
        gamma = torch.sqrt(1.0 + (uz / C_LIGHT) ** 2)
        z = z + uz / gamma * (0.5 * dt)
        t += dt
        ekin = (gamma - 1.0) * mc2_eV / 1e6
        e_in = torch.where(torch.isnan(e_in) & (z >= z_in), ekin, e_in)
        just = ~done & ((z >= z_out) | (uz < 0))
        e_out = torch.where(just, ekin, e_out)
        done |= just
        if bool(done.all()) or t > t_max:
            return e_in, torch.where(done, e_out, ekin)



# Column order of the row _traj_stats returns. t_ns is not here: it is
# already a Python float and never touches the device.
TRAJ_KEYS = (
    "z_m", "Ekin_MeV", "sigma_x_mm", "sigma_y_mm", "sigma_z_mm",
    "emit_x_mm_mrad", "emit_y_mm_mrad", "emit_z_kev_mm", "dE_rms_keV",
    "emit_can_x_mm_mrad", "emit_can_y_mm_mrad",
)


_lerp_tables = lerp_tables  # linear on-axis table lookup; lives in magnets.py so both modules share it


# RMS geometric/longitudinal emittance formulas live in slipstream.analysis.emittance
# (imported at module top), re-exported under the old private names above.


def _axial_bz(x, y, z, sol, placed_sol=()):
    """``Bz = Bz0 - r^2/4 * d2Bz0/dz2`` summed over the solenoids: the unplaced ones (``sol``, lab-``z``
    tables) and the placed/framed ones (``placed_sol``: ``(place, table)`` -- evaluated in their own frame,
    with the lateral validity mask). What the canonical transverse momentum ``p + q*A`` needs."""
    r2 = x * x + y * y
    Bz = torch.zeros_like(x)
    for f, _df, d2f, z0, dz, ntab, zend in sol:
        bz0, d2bz0 = _lerp_tables(z, z0, dz, ntab, zend, f, d2f)
        Bz = Bz + (bz0 - 0.25 * r2 * d2bz0)
    for place, (f, _df, d2f, z0, dz, ntab, zend) in placed_sol:
        xl, yl, zl = to_local3(x, y, z, place)
        bz0, d2bz0 = _lerp_tables(zl, z0, dz, ntab, zend, f, d2f)
        Bz = Bz + lateral_ok(xl, yl) * (bz0 - 0.25 * (xl * xl + yl * yl) * d2bz0)
    return Bz


def _traj_stats(x, y, z, ux, uy, uz, sol, placed_sol=(), beam_frame=False,
                mc2_eV=ELECTRON_MASS_EV, q_C=ELECTRON_Q_C):
    """One record event as a single ``torch.compile`` kernel, returned as one stacked row in
    ``TRAJ_KEYS`` order (eagerly this is ~35 reductions per event; stacking makes the whole
    trajectory one D2H sync)."""
    Bz = _axial_bz(x, y, z, sol, placed_sol)
    z_lab = z.mean()
    x_em, y_em = x, y  # what the emittance formulas see (see below)
    if beam_frame:
        x, y, z, ux, uy, uz, Bz = (a.double() for a in (x, y, z, ux, uy, uz, Bz))
        z_lab = z_lab.double()
        x, z, ux, uz = rotate_to_beam(x, z, ux, uz, beam_angle(ux, uz))
        x_em, y_em = x - x.mean(), y - y.mean()
    mass_kg = mc2_eV * E_CHARGE_C / C_LIGHT**2
    ga = torch.sqrt(1.0 + (ux**2 + uy**2 + uz**2) / C_LIGHT**2)
    ex_geo, ey_geo = _geometric_emittance_m(ux / uz, uy / uz, x_em, y_em)
    px_can = ux * mass_kg - q_C * Bz * y / 2.0
    py_can = uy * mass_kg + q_C * Bz * x / 2.0
    pz = uz * mass_kg
    ecx_geo, ecy_geo = _geometric_emittance_m(px_can / pz, py_can / pz, x_em, y_em)
    beta_gamma = (ga**2 - 1.0).sqrt().mean()
    ekin_ev = (ga - 1.0) * mc2_eV
    return torch.stack([
        z_lab,
        ekin_ev.mean() / 1e6,
        x.std() * 1e3,
        y.std() * 1e3,
        z.std() * 1e3,
        ex_geo * beta_gamma * 1e6,
        ey_geo * beta_gamma * 1e6,
        _longitudinal_emittance_keV_mm(z, ekin_ev),
        ekin_ev.std() / 1e3,
        ecx_geo * beta_gamma * 1e6,
        ecy_geo * beta_gamma * 1e6,
    ])



_TWISS_UNITS = {
    "z": "m", "t": "s", "kinetic_energy": "eV", "gamma": "", "cp": "eV/c", "mean_cp": "eV/c",
    "mean_x": "m", "mean_y": "m",
    "sigma_x": "m", "sigma_xp": "rad", "sigma_y": "m", "sigma_yp": "rad",
    "sigma_z": "m", "sigma_t": "s", "sigma_p": "", "sigma_cp": "eV/c",
    "enx": "m-rad", "ex": "m-rad", "eny": "m-rad", "ey": "m-rad", "ecnx": "m-rad", "ecny": "m-rad",
    "beta_x": "m", "alpha_x": "", "gamma_x": "",
    "beta_y": "m", "alpha_y": "", "gamma_y": "",
    "element_name": "", "lattice_name": "",
}


def _twiss_row_from_ensemble(x, y, z, ux, uy, uz, sol, t_s, label, lattice_name, placed_sol=(), beam_frame=False,
                             mc2_eV=ELECTRON_MASS_EV, q_C=ELECTRON_Q_C):
    """Compute twiss row from ensemble; see the module documentation for details."""
    Bz = _axial_bz(x, y, z, sol, placed_sol)
    z_lab = z.mean()
    x_em, y_em = x, y  # zero-mean positions for the canonical-emittance formula (see _traj_stats)
    if beam_frame:
        x, y, z, ux, uy, uz, Bz = (a.double() for a in (x, y, z, ux, uy, uz, Bz))
        x, z, ux, uz = rotate_to_beam(x, z, ux, uz, beam_angle(ux, uz))
        x_em, y_em = x - x.mean(), y - y.mean()
    mass_kg = mc2_eV * E_CHARGE_C / C_LIGHT**2
    ga = torch.sqrt(1.0 + (ux**2 + uy**2 + uz**2) / C_LIGHT**2)
    ekin_eV = (ga - 1.0) * mc2_eV
    xp, yp = ux / uz, uy / uz

    def beta_alpha_gamma(u, up):
        cov = ((u - u.mean()) * (up - up.mean())).mean()
        emit = torch.sqrt(torch.clamp(u.var() * up.var() - cov * cov, min=0.0))
        return (u.var() / emit).item(), (-cov / emit).item(), (up.var() / emit).item(), emit.item()

    beta_x, alpha_x, gamma_x, ex = beta_alpha_gamma(x, xp)
    beta_y, alpha_y, gamma_y, ey = beta_alpha_gamma(y, yp)

    # Canonical transverse momentum inside a solenoid -- same convention as _traj_stats.
    px_can = ux * mass_kg - q_C * Bz * y / 2.0
    py_can = uy * mass_kg + q_C * Bz * x / 2.0
    pz = uz * mass_kg
    ecx_geo, ecy_geo = _geometric_emittance_m(px_can / pz, py_can / pz, x_em, y_em)

    bg_mean = float(torch.clamp(ga**2 - 1.0, min=0.0).sqrt().mean())
    gamma_mean = float(ga.mean())
    rms_e_eV = float(ekin_eV.std())
    cp_eV = math.sqrt(max(gamma_mean**2 - 1.0, 0.0)) * mc2_eV
    beta_rel = math.sqrt(max(1.0 - 1.0 / gamma_mean**2, 0.0)) if gamma_mean > 0 else 0.0
    # dE = beta*c*dp (relativistic), so d(pc)[eV] = dE[eV]/beta -- exact,
    # not ASTRA's own keV-scaled approximation.
    sigma_cp_eV = rms_e_eV / beta_rel if beta_rel > 0 else 0.0

    return {
        "z": float(z_lab), "t": t_s,
        "kinetic_energy": float(ekin_eV.mean()), "gamma": gamma_mean, "cp": cp_eV, "mean_cp": cp_eV,
        "mean_x": float(x.mean()), "mean_y": float(y.mean()),
        "sigma_x": float(x.std()), "sigma_xp": float(xp.std()),
        "sigma_y": float(y.std()), "sigma_yp": float(yp.std()),
        "sigma_z": float(z.std()), "sigma_t": float(z.std()) / (beta_rel * C_LIGHT) if beta_rel > 0 else 0.0,
        "sigma_p": sigma_cp_eV / cp_eV if cp_eV > 0 else 0.0, "sigma_cp": sigma_cp_eV,
        "enx": ex * bg_mean, "ex": ex, "eny": ey * bg_mean, "ey": ey,
        "ecnx": float(ecx_geo) * bg_mean, "ecny": float(ecy_geo) * bg_mean,
        "beta_x": beta_x, "alpha_x": alpha_x, "gamma_x": gamma_x,
        "beta_y": beta_y, "alpha_y": alpha_y, "gamma_y": gamma_y,
        "element_name": label, "lattice_name": lattice_name,
    }


def _write_twiss_hdf5(path, rows: list[dict], source: str | None = None) -> None:
    """Compute write twiss hdf5; see the module documentation for details."""
    import h5py
    with h5py.File(path, "w") as f:
        p = f.create_group("Parameters")
        if source:
            p["Source"] = str(source)
        p["Version"] = "2"
        g = f.create_group("twiss")
        for key in rows[0]:
            vals = [r[key] for r in rows]
            data = np.array(vals, dtype="S") if key in ("element_name", "lattice_name") else np.array(vals, dtype=float)
            g.create_dataset(key, data=data).attrs.create("Units", _TWISS_UNITS.get(key, ""))


def _dump_beam_openpmd(path, x, y, z, ux, uy, uz, q, t_s: float,
                       mc2_eV=ELECTRON_MASS_EV, species: str = "electron") -> None:
    """Full 6D macroparticle dump in openPMD-beamphysics format (``beamphysics.ParticleGroup``) -- this
    tracker's equivalent of an ASTRA screen/end-of-line particle file."""
    from beamphysics import ParticleGroup

    mass_kg = mc2_eV * E_CHARGE_C / C_LIGHT**2

    def pc_eV(u):
        return (u * mass_kg * C_LIGHT / E_CHARGE_C).cpu().numpy()

    pg = ParticleGroup(data={
        "x": x.cpu().numpy(), "y": y.cpu().numpy(), "z": z.cpu().numpy(),
        "px": pc_eV(ux), "py": pc_eV(uy), "pz": pc_eV(uz),
        "t": t_s, "status": 1, "species": species,
        "weight": q.abs().cpu().numpy(),
    })
    pg.write(str(path))



def _field_tables(cav_fields, sol_fields):
    """``_fused_step``'s ``(sw, tw, sol)``: one tuple per element of
    pre-scaled tables plus the geometry/phase scalars its lookup needs, with
    ``z0``/``zend`` already in lab coordinates. Built once per run."""
    sw, tw = [], []
    for f in cav_fields:
        if isinstance(f, TravelingWaveField):
            tw.append((f.map.f * f.scale, f.map.df * f.scale, f.map_imag.f * f.scale, f.map_imag.df * f.scale,
                       f.map.z0, f.map.dz, f.map.n, f.z_offset, f.z1, f.period, float(f.n_tiles - 1),
                       f.length, f.omega, f.phase_rad))
        else:
            z0, zend = _field_span(f)
            sw.append((f.map.f * f.scale, f.map.df * f.scale, z0, f.map.dz, f.map.n, zend, f.omega, f.phase_rad))
    sol = []
    for f in sol_fields:
        z0 = f.map.z0 + f.z_offset
        sol.append((f.map.f * f.scale, f.map.df * f.scale, f.map.d2f * f.scale,
                    z0, f.map.dz, f.map.n, z0 + f.map.dz * (f.map.n - 1)))
    return tuple(sw), tuple(tw), tuple(sol)


def _quad_tables(quads: "list[Quadrupole]") -> tuple:
    """Compute quad tables; see the module documentation for details."""
    return tuple((-q.grad_T_per_m, q.z_m - q.length_m / 2.0, q.z_m + q.length_m / 2.0) for q in quads)


def _aperture_tables(apertures: "list[Aperture]") -> tuple:
    """Plain ``(kind, z1, z2, r)`` tuples for ``_apply_apertures``, consumed by
    ``_step_with_apertures``."""
    return tuple((a.kind, a.z1_m, a.z2_m, a.r_m)
                 + ((placement(a.x_m, 0.5 * (a.z1_m + a.z2_m), a.xrot_rad),) if (a.x_m or a.xrot_rad) else ())
                 for a in apertures)


def _aperture_hit(kind: str, r: float, xa: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Outside the acceptance of one aperture, at points ``(xa, y)`` in its own frame."""
    if kind == "RAD":
        r2 = xa * xa + y * y
        return (r2 > r * r) if r >= 0.0 else (r2 < r * r)
    if kind == "COL_X":
        return xa.abs() > abs(r)
    if kind == "COL_Y":
        return y.abs() > abs(r)
    if kind == "SCR_X":
        return (xa > r) if r >= 0.0 else (xa < r)
    return (y > r) if r >= 0.0 else (y < r)  # SCR_Y


def _swept_aperture_hit(kind, r, z1, z2, xa0, y0, za0, xa, y, za):
    """Lost on one aperture anywhere along the straight segment ``(xa0, y0, za0) -> (xa, y, za)``."""
    dz = za - za0
    moving = dz != 0.0
    dz_safe = torch.where(moving, dz, torch.ones_like(dz))
    t1, t2 = (z1 - za0) / dz_safe, (z2 - za0) / dz_safe
    t_in, t_out = torch.minimum(t1, t2), torch.maximum(t1, t2)
    # no z motion: the whole step is inside the slab or none of it
    overlaps = torch.where(moving, (t_out >= 0.0) & (t_in <= 1.0), (za0 >= z1) & (za0 <= z2))
    lo = torch.where(moving, t_in.clamp(0.0, 1.0), torch.zeros_like(dz))
    hi = torch.where(moving, t_out.clamp(0.0, 1.0), torch.ones_like(dz))
    dx, dy = xa - xa0, y - y0
    xl, yl, xh, yh = xa0 + lo * dx, y0 + lo * dy, xa0 + hi * dx, y0 + hi * dy
    hit = _aperture_hit(kind, r, xl, yl) | _aperture_hit(kind, r, xh, yh)
    if kind == "RAD" and r < 0.0:  # plug: the closest approach can lie between the two ends
        sx, sy = xh - xl, yh - yl
        s2 = sx * sx + sy * sy
        s = torch.where(s2 > 0.0, -(xl * sx + yl * sy) / torch.where(s2 > 0.0, s2, torch.ones_like(s2)),
                        torch.zeros_like(s2)).clamp(0.0, 1.0)
        hit = hit | ((xl + s * sx) ** 2 + (yl + s * sy) ** 2 < r * r)
    return overlaps & hit


def _apply_apertures(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, alive: torch.Tensor, apertures: tuple,
                     prev: tuple | None = None) -> torch.Tensor:
    """One AND-accumulation pass over every aperture -- a particle already lost stays lost (monotonic)."""
    for ap in apertures:
        kind, z1, z2, r = ap[:4]
        # A placed aperture (ap[4], see Aperture.x_m/xrot_rad) is tested in its own frame.
        xa, za = to_local(x, z, ap[4]) if len(ap) > 4 else (x, z)
        if prev is None:
            hit = (za >= z1) & (za <= z2) & _aperture_hit(kind, r, xa, y)
        else:
            xa0, za0 = to_local(prev[0], prev[2], ap[4]) if len(ap) > 4 else (prev[0], prev[2])
            hit = _swept_aperture_hit(kind, r, z1, z2, xa0, prev[1], za0, xa, y, za)
        alive = alive & ~hit
    return alive


def _step_with_apertures(alive, apertures, *args):
    """Push particles, then check the path of the step (previous to updated positions) in the same
    kernel."""
    state = _fused_step(*args)
    return (*state, _apply_apertures(*state[:3], alive, apertures, prev=args[:3]))


def _rf_cos_sin(ph, dtype):
    """``(cos, sin)`` of a 0-d float64 RF phase, as ``dtype``."""
    ph = ph - _TWO_PI * torch.round(ph * (1.0 / _TWO_PI))
    ph = ph.to(dtype)
    return torch.cos(ph), torch.sin(ph)


def _rf_phases(elements: tuple, t, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute rf phases; see the module documentation for details."""
    if not elements:
        return torch.zeros(0, dtype=dtype, device=t.device), torch.zeros(0, dtype=dtype, device=t.device)
    cos_list, sin_list = [], []
    for e in elements:
        omega, phase = e[-2], e[-1]
        c, s = _rf_cos_sin(omega * t + phase, dtype)
        cos_list.append(c)
        sin_list.append(s)
    return torch.stack(cos_list), torch.stack(sin_list)


def _applied_fields(x, y, z, sw, tw, sol, quad, t, c2_inv, magnets=()):
    """Compute applied fields; see the module documentation for details."""
    Ex = Ey = Ez = Bx = By = Bz = torch.zeros_like(x)

    # Standing-wave cavities: on-axis Ez0(z) + paraxial Er/Bphi expansion.
    for f, df, z0, dz, ntab, zend, omega, phase in sw:
        ez0, dez0 = _lerp_tables(z, z0, dz, ntab, zend, f, df)
        ph = omega * t + phase
        cos_p, sin_p = _rf_cos_sin(ph, x.dtype)
        er_over_r = -0.5 * dez0 * cos_p
        bphi_over_r = -(omega * 0.5 * c2_inv) * ez0 * sin_p
        Ex = Ex + er_over_r * x
        Ey = Ey + er_over_r * y
        Ez = Ez + ez0 * cos_p
        Bx = Bx - bphi_over_r * y
        By = By + bphi_over_r * x

    for er, der, ei, dei, z0, dz, ntab, zoff, z1, period, tile_max, length, omega, phase in tw:
        zl = z - zoff
        tile = torch.clamp(torch.floor((zl - z1) / period), min=0.0, max=tile_max)
        zt = zl - tile * period
        zi = (zt - z0) / dz
        i0 = torch.clamp(zi.floor().long(), 0, ntab - 2)
        fr = (zi - i0).clamp(0.0, 1.0)
        inside = (zl >= 0.0) & (zl <= length)
        t_er, t_der, t_ei, t_dei = [
            torch.where(inside, tab[i0] * (1 - fr) + tab[i0 + 1] * fr, 0.0) for tab in (er, der, ei, dei)
        ]
        ph = omega * t + phase
        cos_t, sin_t = _rf_cos_sin(ph, x.dtype)
        dEz_dz = t_der * cos_t - t_dei * sin_t
        dEz_dt = -omega * (t_er * sin_t + t_ei * cos_t)
        er_over_r = -0.5 * dEz_dz
        bphi_over_r = dEz_dt * (0.5 * c2_inv)
        Ex = Ex + er_over_r * x
        Ey = Ey + er_over_r * y
        Ez = Ez + (t_er * cos_t - t_ei * sin_t)
        Bx = Bx - bphi_over_r * y
        By = By + bphi_over_r * x

    # Solenoids: on-axis Bz0(z) + paraxial Br/Bz expansion.
    r2 = x * x + y * y
    for f, df, d2f, z0, dz, ntab, zend in sol:
        bz0, dbz0, d2bz0 = _lerp_tables(z, z0, dz, ntab, zend, f, df, d2f)
        br_over_r = -0.5 * dbz0
        Bx = Bx + br_over_r * x
        By = By + br_over_r * y
        Bz = Bz + (bz0 - 0.25 * r2 * d2bz0)

    # Quadrupoles: hard-edge real field, zero outside [z0, zend] -- see
    # _quad_tables for the sign convention.
    for g, z0, zend in quad:
        inside = (z >= z0) & (z <= zend)
        Bx = Bx + torch.where(inside, g * y, torch.zeros_like(y))
        By = By + torch.where(inside, g * x, torch.zeros_like(x))

    # Lab-frame magnets: dipoles, placed/framed solenoids, profile quadrupoles -- see magnets.py.
    if magnets:
        m_bx, m_by, m_bz = magnet_fields(x, y, z, magnets)
        Bx, By, Bz = Bx + m_bx, By + m_by, Bz + m_bz

    return Ex, Ey, Ez, Bx, By, Bz


def _fused_step(
    x, y, z, ux, uy, uz, cx, cy, cz, cux, cuy, cuz,
    Ex_sc, Ey_sc, Ez_sc, Bx_sc, By_sc, Bz_sc,
    sw, tw, sol, quad, t, dt, qmdt2, emitted_all, active, c2_inv, magnets=(),
):
    """Compute fused step; see the module documentation for details."""
    Ex, Ey, Ez, Bx, By, Bz = _applied_fields(x, y, z, sw, tw, sol, quad, t, c2_inv, magnets)

    Ex, Ey, Ez = Ex + Ex_sc, Ey + Ey_sc, Ez + Ez_sc
    Bx, By, Bz = Bx + Bx_sc, By + By_sc, Bz + Bz_sc

    # Relativistic Boris push (field at start-of-step position/time), built
    # as an increment of small terms plus one compensated add.
    uxm = ux + qmdt2 * Ex
    uym = uy + qmdt2 * Ey
    uzm = uz + qmdt2 * Ez
    g = torch.sqrt(1.0 + (uxm * uxm + uym * uym + uzm * uzm) * c2_inv)
    tx = qmdt2 / g * Bx
    ty = qmdt2 / g * By
    tz = qmdt2 / g * Bz
    t2 = tx * tx + ty * ty + tz * tz
    upx = uxm + (uym * tz - uzm * ty)
    upy = uym + (uzm * tx - uxm * tz)
    upz = uzm + (uxm * ty - uym * tx)
    denom = 1.0 + t2
    sx = 2.0 * tx / denom
    sy = 2.0 * ty / denom
    sz = 2.0 * tz / denom
    ax = 2.0 * qmdt2 * Ex + (upy * sz - upz * sy)
    ay = 2.0 * qmdt2 * Ey + (upz * sx - upx * sz)
    az = 2.0 * qmdt2 * Ez + (upx * sy - upy * sx)
    ux_n, cux_n = kahan_add(ux, ax, cux)
    uy_n, cuy_n = kahan_add(uy, ay, cuy)
    uz_n, cuz_n = kahan_add(uz, az, cuz)
    g_n = torch.sqrt(1.0 + (ux_n * ux_n + uy_n * uy_n + uz_n * uz_n) * c2_inv)
    x_n, cx_n = kahan_add(x, ux_n / g_n * dt, cx)
    y_n, cy_n = kahan_add(y, uy_n / g_n * dt, cy)
    z_n, cz_n = kahan_add(z, uz_n / g_n * dt, cz)

    out = (x_n, y_n, z_n, ux_n, uy_n, uz_n, cx_n, cy_n, cz_n, cux_n, cuy_n, cuz_n)
    if emitted_all:
        return out
    old = (x, y, z, ux, uy, uz, cx, cy, cz, cux, cuy, cuz)
    return tuple([torch.where(active, a, b) for a, b in zip(out, old)])


_applied_fields_compiled = torch.compile(_applied_fields, dynamic=False)


def _total_fields(x, y, z, sw, tw, sol, quad, t, c2_inv, magnets, Ex_sc, Ey_sc, Ez_sc, Bx_sc, By_sc, Bz_sc):
    """``_applied_fields`` plus the space-charge fields; compiled, the six adds fuse into the field kernels
    instead of costing six eager launches per step."""
    Ex, Ey, Ez, Bx, By, Bz = _applied_fields(x, y, z, sw, tw, sol, quad, t, c2_inv, magnets)
    return Ex + Ex_sc, Ey + Ey_sc, Ez + Ez_sc, Bx + Bx_sc, By + By_sc, Bz + Bz_sc


_total_fields_compiled = torch.compile(_total_fields, dynamic=False)


def _applied_fields_warp(x, y, z, sw, tw, sol, quad, t, c2_inv, magnets=()):
    """Same as ``_applied_fields``, but ``sw``/``tw``/``sol`` run as warp kernel launches
    (``warp_kernels.applied_fields``) instead of through ``_applied_fields_compiled``."""
    cos_sw, sin_sw = _rf_phases(sw, t, x.dtype)
    cos_tw, sin_tw = _rf_phases(tw, t, x.dtype)
    return warp_kernels.applied_fields(x, y, z, sw, tw, sol, cos_sw, sin_sw, cos_tw, sin_tw, c2_inv)


def _fused_step_warp(
    x, y, z, ux, uy, uz, cx, cy, cz, cux, cuy, cuz,
    Ex_sc, Ey_sc, Ez_sc, Bx_sc, By_sc, Bz_sc,
    sw, tw, sol, quad, t, dt, qmdt2, emitted_all, active, c2_inv, magnets=(),
):
    """Same as ``_fused_step``, but the Boris-push tail runs as one warp kernel launch
    (``warp_kernels.push``) instead of inline torch ops -- CUDA only, see
    ``warp_kernels.available``."""
    # See docs/tracking_native.md's "Injector warp backend" for the measured numbers behind this choice.
    Ex, Ey, Ez, Bx, By, Bz = _total_fields_compiled(x, y, z, sw, tw, sol, quad, t, c2_inv, magnets,
                                                     Ex_sc, Ey_sc, Ez_sc, Bx_sc, By_sc, Bz_sc)

    if emitted_all:
        # push_inplace, not push: no old/new blend follows -- see docs/tracking_native.md's
        # "Injector warp backend".
        return warp_kernels.push_inplace(x, y, z, ux, uy, uz, cx, cy, cz, cux, cuy, cuz,
                                          Ex, Ey, Ez, Bx, By, Bz, qmdt2, dt, c2_inv)
    out = warp_kernels.push(x, y, z, ux, uy, uz, cx, cy, cz, cux, cuy, cuz,
                             Ex, Ey, Ez, Bx, By, Bz, qmdt2, dt, c2_inv)
    old = (x, y, z, ux, uy, uz, cx, cy, cz, cux, cuy, cuz)
    return tuple([torch.where(active, a, b) for a, b in zip(out, old)])


def _step_with_apertures_warp(alive, apertures, *args):
    """``_step_with_apertures``'s warp-backed sibling -- see ``_fused_step_warp``."""
    if not apertures:
        return (*_fused_step_warp(*args), alive)
    prev = (args[0].clone(), args[1].clone(), args[2].clone())
    state = _fused_step_warp(*args)
    return (*state, warp_kernels.apply_apertures(*state[:3], *prev, alive, apertures))



def _quantize_bucket(needed: int, buckets: tuple[int, ...]) -> int:
    """Smallest ``buckets`` entry >= ``needed`` (last entry past the end)."""
    for bucket in buckets:
        if bucket >= needed:
            return bucket
    return buckets[-1]


def _quantize_bucket_hysteresis(needed: int, current: int | None, buckets: tuple[int, ...],
                                down_factor: float = 0.7) -> int:
    """Compute quantize bucket hysteresis; see the module documentation for details."""
    if current is not None and current in buckets and needed <= current:
        idx = buckets.index(current)
        if idx == 0 or needed > buckets[idx - 1] * down_factor:
            return current
    return _quantize_bucket(needed, buckets)


def _limit_mesh_by_particle_occupancy(
    n_particles: int,
    nr: int,
    nz: int,
    nr_buckets: tuple[int, ...],
    nz_buckets: tuple[int, ...],
    min_particles_per_cell: float,
) -> tuple[int, int]:
    """Limit an adaptive mesh so PIC cells do not outnumber particle statistics."""
    if min_particles_per_cell <= 0 or n_particles <= 0:
        return nr, nz

    max_cells = math.floor(n_particles / min_particles_per_cell)
    candidates = [
        (candidate_nr, candidate_nz)
        for candidate_nr in nr_buckets
        for candidate_nz in nz_buckets
        if candidate_nr <= nr
        and candidate_nz <= nz
        and candidate_nr * candidate_nz <= max_cells
    ]
    if not candidates:
        return nr_buckets[0], nz_buckets[0]

    return max(
        candidates,
        key=lambda candidate: (
            min(candidate[0] / nr, candidate[1] / nz),
            candidate[0] / nr + candidate[1] / nz,
            candidate[0] * candidate[1],
        ),
    )


def _fused_sc_fields(
    x, y, z, q, bunch_gamma_t, z_ref_t, beta_t,
    dr, dz, z_min, nr, nz, nz_pad, node_volume, P_left, inv_diag, P_right, with_image,
    smooth_r: int = 0, smooth_z: int = 0, smooth_gamma_ref: float = math.inf,
):
    """Compute fused sc fields; see the module documentation for details."""
    z_stretched = z_ref_t + bunch_gamma_t * (z - z_ref_t)
    r2 = x * x + y * y
    on_axis = r2 == 0.0
    r = torch.where(on_axis, 0.0, torch.sqrt(torch.where(on_axis, 1.0, r2)))

    if with_image:
        z_image = 2 * z_ref_t - z_stretched
        r_all = torch.cat([r, r])
        z_all = torch.cat([z_stretched, z_image])
        q_all = torch.cat([q, -q])
    else:
        r_all, z_all, q_all = r, z_stretched, q

    r_f = r_all / dr
    r0 = torch.clamp(r_f.floor().long(), 0, nr - 2)
    wr = torch.clamp(r_f - r0, 0.0, 1.0)
    z_f = (z_all - z_min) / dz
    z0 = torch.clamp(z_f.floor().long(), 0, nz - 2)
    wz = torch.clamp(z_f - z0, 0.0, 1.0)

    charge = torch.zeros(nr * nz, device=x.device, dtype=x.dtype)
    for dr_i, wr_i in ((0, 1 - wr), (1, wr)):
        for dz_i, wz_i in ((0, 1 - wz), (1, wz)):
            idx = (r0 + dr_i) * nz + (z0 + dz_i)
            charge.scatter_add_(0, idx, q_all * wr_i * wz_i)
    rho = charge.view(nr, nz) / node_volume.unsqueeze(1)
    if smooth_r or smooth_z:
        w = 1.0 / (1.0 + (bunch_gamma_t / smooth_gamma_ref) ** 2)
        rho = rho + w * (smooth_rho(rho, smooth_r, smooth_z) - rho)

    rho_padded = torch.zeros(nr, nz_pad, device=x.device, dtype=x.dtype)
    rho_padded[:, :nz] = P_right @ rho
    rho_k = torch.fft.rfft(rho_padded, dim=1)
    phi_k = rho_k * (-inv_diag / EPS0_F_PER_M)
    phi_padded = torch.fft.irfft(phi_k, n=nz_pad, dim=1)
    phi = P_left @ phi_padded[:, :nz]

    Ez_g = torch.zeros_like(phi)
    Ez_g[:, 1:-1] = -(phi[:, 2:] - phi[:, :-2]) / (2 * dz)
    Ez_g[:, 0] = -(phi[:, 1] - phi[:, 0]) / dz
    Ez_g[:, -1] = -(phi[:, -1] - phi[:, -2]) / dz
    Er_g = torch.zeros_like(phi)
    Er_g[1:-1, :] = -(phi[2:, :] - phi[:-2, :]) / (2 * dr)
    Er_g[0, :] = 0.0
    Er_g[-1, :] = phi[-2, :] / (2 * dr)

    # Gather at the real particles only (the image lobe is deposit-only).
    n_real = x.shape[0]
    r0r, wrr, z0r, wzr = r0[:n_real], wr[:n_real], z0[:n_real], wz[:n_real]
    Er = (Er_g[r0r, z0r] * (1 - wrr) * (1 - wzr) + Er_g[r0r + 1, z0r] * wrr * (1 - wzr)
          + Er_g[r0r, z0r + 1] * (1 - wrr) * wzr + Er_g[r0r + 1, z0r + 1] * wrr * wzr)
    Ez_p = (Ez_g[r0r, z0r] * (1 - wrr) * (1 - wzr) + Ez_g[r0r + 1, z0r] * wrr * (1 - wzr)
            + Ez_g[r0r, z0r + 1] * (1 - wrr) * wzr + Ez_g[r0r + 1, z0r + 1] * wrr * wzr)

    r_safe = torch.clamp(r, min=1e-12)
    Ex_lab = bunch_gamma_t * (Er * x / r_safe)
    Ey_lab = bunch_gamma_t * (Er * y / r_safe)
    Ez_lab = Ez_p
    Bx_lab = -(beta_t / C_LIGHT) * Ey_lab
    By_lab = (beta_t / C_LIGHT) * Ex_lab
    Bz_lab = torch.zeros_like(Ez_lab)
    return Ex_lab, Ey_lab, Ez_lab, Bx_lab, By_lab, Bz_lab


def _sc_gather_and_boost(
    charge, r0r, wrr, z0r, wzr, x, y, bunch_gamma_t, beta_t, dr, dz, nr, nz, nz_pad,
    node_volume, P_left, inv_diag, P_right,
    smooth_r: int = 0, smooth_z: int = 0, smooth_gamma_ref: float = math.inf,
):
    """Compute sc gather and boost; see the module documentation for details."""
    rho = charge.view(nr, nz) / node_volume.unsqueeze(1)
    if smooth_r or smooth_z:
        w = 1.0 / (1.0 + (bunch_gamma_t / smooth_gamma_ref) ** 2)
        rho = rho + w * (smooth_rho(rho, smooth_r, smooth_z) - rho)

    rho_padded = torch.zeros(nr, nz_pad, device=x.device, dtype=x.dtype)
    rho_padded[:, :nz] = P_right @ rho
    rho_k = torch.fft.rfft(rho_padded, dim=1)
    phi_k = rho_k * (-inv_diag / EPS0_F_PER_M)
    phi_padded = torch.fft.irfft(phi_k, n=nz_pad, dim=1)
    phi = P_left @ phi_padded[:, :nz]

    Ez_g = torch.zeros_like(phi)
    Ez_g[:, 1:-1] = -(phi[:, 2:] - phi[:, :-2]) / (2 * dz)
    Ez_g[:, 0] = -(phi[:, 1] - phi[:, 0]) / dz
    Ez_g[:, -1] = -(phi[:, -1] - phi[:, -2]) / dz
    Er_g = torch.zeros_like(phi)
    Er_g[1:-1, :] = -(phi[2:, :] - phi[:-2, :]) / (2 * dr)
    Er_g[0, :] = 0.0
    Er_g[-1, :] = phi[-2, :] / (2 * dr)

    Er = (Er_g[r0r, z0r] * (1 - wrr) * (1 - wzr) + Er_g[r0r + 1, z0r] * wrr * (1 - wzr)
          + Er_g[r0r, z0r + 1] * (1 - wrr) * wzr + Er_g[r0r + 1, z0r + 1] * wrr * wzr)
    Ez_p = (Ez_g[r0r, z0r] * (1 - wrr) * (1 - wzr) + Ez_g[r0r + 1, z0r] * wrr * (1 - wzr)
            + Ez_g[r0r, z0r + 1] * (1 - wrr) * wzr + Ez_g[r0r + 1, z0r + 1] * wrr * wzr)

    r_safe = torch.clamp(torch.sqrt(x * x + y * y), min=1e-12)
    Ex_lab = bunch_gamma_t * (Er * x / r_safe)
    Ey_lab = bunch_gamma_t * (Er * y / r_safe)
    Ez_lab = Ez_p
    Bx_lab = -(beta_t / C_LIGHT) * Ey_lab
    By_lab = (beta_t / C_LIGHT) * Ex_lab
    Bz_lab = torch.zeros_like(Ez_lab)
    return Ex_lab, Ey_lab, Ez_lab, Bx_lab, By_lab, Bz_lab


_sc_gather_and_boost_compiled = torch.compile(_sc_gather_and_boost, dynamic=False)


def _fused_sc_fields_warp(
    x, y, z, q, bunch_gamma_t, z_ref_t, beta_t,
    dr, dz, z_min, nr, nz, nz_pad, node_volume, P_left, inv_diag, P_right, with_image,
    smooth_r: int = 0, smooth_z: int = 0, smooth_gamma_ref: float = math.inf,
):
    """Compute fused sc fields warp; see the module documentation for details."""
    z_stretched = z_ref_t + bunch_gamma_t * (z - z_ref_t)

    if with_image:
        z_image = 2 * z_ref_t - z_stretched
        r_all = torch.cat([torch.sqrt(x * x + y * y)] * 2)
        z_all = torch.cat([z_stretched, z_image])
        q_all = torch.cat([q, -q])
    else:
        r_all, z_all, q_all = torch.sqrt(x * x + y * y), z_stretched, q

    r_f = r_all / dr
    r0 = torch.clamp(r_f.floor().long(), 0, nr - 2)
    wr = torch.clamp(r_f - r0, 0.0, 1.0)
    z_f = (z_all - z_min) / dz
    z0 = torch.clamp(z_f.floor().long(), 0, nz - 2)
    wz = torch.clamp(z_f - z0, 0.0, 1.0)

    charge = warp_kernels.sc_deposit(r0, z0, wr, wz, q_all, nr, nz)

    # Gather at the real particles only (the image lobe is deposit-only).
    n_real = x.shape[0]
    r0r, wrr, z0r, wzr = r0[:n_real], wr[:n_real], z0[:n_real], wz[:n_real]
    return _sc_gather_and_boost_compiled(
        charge, r0r, wrr, z0r, wzr, x, y, bunch_gamma_t, beta_t, dr, dz, nr, nz, nz_pad,
        node_volume, P_left, inv_diag, P_right, smooth_r, smooth_z, smooth_gamma_ref,
    )


def _fused_sc_fields_3d(
    x, y, z, q, bunch_gamma_t, z_ref_t, beta_t,
    x_lo, y_lo, z_lo, hx, hy, hz,
    nx, ny, nz, px, py, pz, G_k, with_image,
):
    """Compute fused sc fields 3d; see the module documentation for details."""
    z_s = z_ref_t + bunch_gamma_t * (z - z_ref_t)
    xs, ys, zs = x, y, z_s
    if with_image:
        xs = torch.cat([xs, xs])
        ys = torch.cat([ys, ys])
        zs = torch.cat([zs, 2.0 * z_ref_t - zs])
        q_dep = torch.cat([q, -q])
    else:
        q_dep = q

    def cic(u, lo, h, n):
        f = (u - lo) / h
        i0 = torch.clamp(f.floor().long(), 0, n - 2)
        return i0, torch.clamp(f - i0, 0.0, 1.0)

    ix, wx = cic(xs, x_lo, hx, nx)
    iy, wy = cic(ys, y_lo, hy, ny)
    iz, wz = cic(zs, z_lo, hz, nz)
    charge = torch.zeros(nx * ny * nz, device=x.device, dtype=x.dtype)
    for a, wa in ((0, 1 - wx), (1, wx)):
        for b, wb in ((0, 1 - wy), (1, wy)):
            for c, wc in ((0, 1 - wz), (1, wz)):
                idx = ((ix + a) * ny + (iy + b)) * nz + (iz + c)
                charge.scatter_add_(0, idx, q_dep * wa * wb * wc)
    charge = charge.view(nx, ny, nz)

    phi_k = torch.fft.rfftn(charge, s=(px, py, pz)) * G_k
    phi = torch.fft.irfftn(phi_k, s=(px, py, pz))[:nx, :ny, :nz]

    def neg_grad(axis, h):
        E = torch.zeros_like(phi)
        if axis == 0:
            E[1:-1, :, :] = -(phi[2:, :, :] - phi[:-2, :, :]) / (2 * h)
            E[0, :, :] = -(phi[1, :, :] - phi[0, :, :]) / h
            E[-1, :, :] = -(phi[-1, :, :] - phi[-2, :, :]) / h
        elif axis == 1:
            E[:, 1:-1, :] = -(phi[:, 2:, :] - phi[:, :-2, :]) / (2 * h)
            E[:, 0, :] = -(phi[:, 1, :] - phi[:, 0, :]) / h
            E[:, -1, :] = -(phi[:, -1, :] - phi[:, -2, :]) / h
        else:
            E[:, :, 1:-1] = -(phi[:, :, 2:] - phi[:, :, :-2]) / (2 * h)
            E[:, :, 0] = -(phi[:, :, 1] - phi[:, :, 0]) / h
            E[:, :, -1] = -(phi[:, :, -1] - phi[:, :, -2]) / h
        return E

    Ex_g, Ey_g, Ez_g = neg_grad(0, hx), neg_grad(1, hy), neg_grad(2, hz)

    # Gather at the real particles only (the mirror set is deposit-only).
    n_real = x.shape[0]
    ixr, wxr, iyr, wyr, izr, wzr = ix[:n_real], wx[:n_real], iy[:n_real], wy[:n_real], iz[:n_real], wz[:n_real]

    def tri_lerp(grid):
        acc = torch.zeros_like(x)
        for a, wa in ((0, 1 - wxr), (1, wxr)):
            for b, wb in ((0, 1 - wyr), (1, wyr)):
                for c, wc in ((0, 1 - wzr), (1, wzr)):
                    acc = acc + grid[ixr + a, iyr + b, izr + c] * wa * wb * wc
        return acc

    Ex_lab = bunch_gamma_t * tri_lerp(Ex_g)
    Ey_lab = bunch_gamma_t * tri_lerp(Ey_g)
    Ez_lab = tri_lerp(Ez_g)
    Bx_lab = -(beta_t / C_LIGHT) * Ey_lab
    By_lab = (beta_t / C_LIGHT) * Ex_lab
    Bz_lab = torch.zeros_like(Ez_lab)
    return Ex_lab, Ey_lab, Ez_lab, Bx_lab, By_lab, Bz_lab


def _fused_sc_fields_3d_warp(
    x, y, z, q, bunch_gamma_t, z_ref_t, beta_t,
    x_lo, y_lo, z_lo, hx, hy, hz,
    nx, ny, nz, px, py, pz, G_k, with_image,
):
    """3D space charge with direct Warp CIC deposit and field gather."""
    return warp_kernels.sc_fields_3d(
        x, y, z, q, bunch_gamma_t, z_ref_t, beta_t,
        x_lo, y_lo, z_lo, hx, hy, hz,
        nx, ny, nz, px, py, pz, G_k, with_image, 1.0 / C_LIGHT,
    )


def _warn_if_sc_kick_unresolvable(Ez_sc, uz, dt, dtype, qm) -> None:
    """One-off check that float32 can still represent the space-charge kick given Kahan compensation."""
    if dtype != torch.float32:
        return
    du = abs(qm * float(Ez_sc.std()) * dt)
    u = float(uz.abs().mean())
    half_ulp = math.ldexp(1.0, math.frexp(u)[1] - 1 - 24)
    floor = half_ulp * 2.0**-23
    if du < 16.0 * floor:
        warnings.warn(
            f"float32 cannot resolve the space-charge kick even with compensated accumulation: "
            f"per step du = {du:.3g} m/s against a compensated floor of {floor:.3g} m/s; "
            f"longitudinal space charge will be partly rounded away. Use float64."
        )


def _local_sc_omega(x, y, ux, uy, uz, Ex_sc, Ey_sc, Bx_sc, By_sc, counted, qm) -> float:
    """Local transverse space-charge angular frequency (rad/s), measured directly from the
    just-computed SC field rather than modeled from total charge --
    ``SpaceChargeSettings.dt_adaptive``'s estimator."""
    xa, ya = x[counted], y[counted]
    uxa, uya, uza = ux[counted], uy[counted], uz[counted]
    if xa.numel() == 0:
        return 0.0
    gamma = torch.sqrt(1.0 + (uxa * uxa + uya * uya + uza * uza) / C_LIGHT**2)
    ax = qm * Ex_sc[counted] - (qm / gamma) * uza * By_sc[counted]
    ay = qm * Ey_sc[counted] + (qm / gamma) * uza * Bx_sc[counted]
    a2 = ax * ax + ay * ay
    r2 = xa * xa + ya * ya
    # Safe mean (divide by max before summing), not a wider dtype -- see docs/tracking_native.md's
    # "Float32 mean-overflow trick".
    a2_max = float(a2.amax())
    if a2_max <= 0.0:
        return 0.0
    a_rms2 = float((a2 / a2_max).mean()) * a2_max
    r_rms2 = float(r2.mean())
    if r_rms2 <= 0.0:
        return 0.0
    omega = math.sqrt(a_rms2 / r_rms2)
    return omega if math.isfinite(omega) else 0.0


def _compute_space_charge_in_beam_frame(theta, x, y, z, q, ux, uy, uz, active, device, dtype, cache,
                                        sc: SpaceChargeSettings, cathode_z: float, sc_fn=None, sc3d_fn=None,
                                        emitted_all=True, qm: float = ELECTRON_Q_C / MASS_KG):
    """Compute compute space charge in beam frame; see the module documentation for details."""
    c, s = math.cos(theta), math.sin(theta)
    xr, zr = x * c - z * s, x * s + z * c
    uxr, uzr = ux * c - uz * s, ux * s + uz * c
    xr = xr - xr[active].mean()
    y = y - y[active].mean()
    sc_free = dataclasses.replace(sc, cathode_image=False)
    Ex, Ey, Ez, Bx, By, Bz = _compute_space_charge(
        xr, y, zr, q, uxr, uy, uzr, active, device, dtype, cache, sc_free, cathode_z, sc_fn=sc_fn,
        sc3d_fn=sc3d_fn, emitted_all=emitted_all)
    omega = _local_sc_omega(xr, y, uxr, uy, uzr, Ex, Ey, Bx, By, active, qm)
    lab = (Ex * c + Ez * s, Ey, -Ex * s + Ez * c, Bx * c + Bz * s, By, -Bx * s + Bz * c)
    return lab, omega


_SC3D_H_STEP = 2.0 ** 0.0625  # cell-size bucket ladder ratio (~4% rungs)


def _snap_h3d(h_ideal: float, current: float | None) -> float:
    """Quantise a 3D cell size onto the geometric ladder anchored at ``current`` (the previously chosen
    size, ``None`` on the first fit, where the ladder anchors at the ideal itself)."""
    if not math.isfinite(h_ideal) or h_ideal <= 0:
        h_ideal = 1e-12
    if current is None:
        return h_ideal
    log_q = math.log(_SC3D_H_STEP)
    k = math.ceil(math.log(h_ideal / current) / log_q - 1e-9)
    if k >= 1 or k <= -2:
        return current * _SC3D_H_STEP ** k
    return current


def _compute_space_charge_3d(x, y, z, q, ux, uy, uz, device, dtype, cache,
                             sc: SpaceChargeSettings, cathode_z: float, sc3d_fn=None):
    """``SpaceChargeSettings.geometry == "3d"``: lab-frame ``(E, B)`` space-charge fields via the
    persistent ``SpaceCharge3D`` solver and (when compiled) the fused ``_fused_sc_fields_3d``
    kernel."""
    nx, ny, nz_config = map(int, sc.n_xyz)
    if min(nx, ny, nz_config) < 3:
        raise ValueError("SpaceChargeSettings.n_xyz entries must each be at least 3")

    gamma_t = torch.clamp(torch.sqrt(1.0 + (ux * ux + uy * uy + uz * uz) / C_LIGHT**2).mean(), min=1.0 + 1e-9)
    (x_lo_f, x_hi_f, y_lo_f, y_hi_f, z_lo_f, z_hi_f, z_mean, gamma) = torch.stack([
        x.min(), x.max(), y.min(), y.max(), z.min(), z.max(),
        z.mean(), gamma_t,
    ]).tolist()

    def axis_box(lo_f, hi_f):
        # Preserve the stateless 3D solver's fitted box exactly.
        pad = 0.05 * (hi_f - lo_f) + 1e-9
        return lo_f - pad, hi_f + pad

    h_prev = cache.get("h3d")
    previous_hx, previous_hy, previous_hz = h_prev if h_prev is not None else (None, None, None)

    # Match the stateless solver's nominal cell size; bucket growth only adds
    # coverage.
    def fit_h(lo, hi, n, current):
        return _snap_h3d((hi - lo) / (n - 1), current)

    x_lo_n, x_hi_n = axis_box(x_lo_f, x_hi_f)
    y_lo_n, y_hi_n = axis_box(y_lo_f, y_hi_f)
    hx = fit_h(x_lo_n, x_hi_n, nx, previous_hx)
    hy = fit_h(y_lo_n, y_hi_n, ny, previous_hy)

    use_image = sc.cathode_image and (z_mean - cathode_z) < sc.image_threshold_m
    z_ref = cathode_z if use_image else z_mean
    zs_lo = z_ref + gamma * (z_lo_f - z_ref)
    zs_hi = z_ref + gamma * (z_hi_f - z_ref)
    z_lo_n, z_hi_n = axis_box(zs_lo, zs_hi)
    hz = fit_h(z_lo_n, z_hi_n, nz_config, previous_hz)

    if use_image:
        domain_half = max(cathode_z - z_lo_n, z_hi_n - cathode_z)
        nz_needed = int(math.ceil(2.0 * domain_half / hz)) + 1
        image_nz_cap = min(SC_IMAGE_NZ_MAX, SC_IMAGE_NZ_BUCKETS[-1])
        if nz_needed > image_nz_cap:
            use_image = False
        else:
            nz = _quantize_bucket_hysteresis(nz_needed, cache.get("nz3d_bucket"), SC_IMAGE_NZ_BUCKETS)
            # Odd node count: the cathode plane sits exactly on the centre node, so the real+image
            # deposit is exactly antisymmetric.
            z_lo_n = cathode_z - (nz // 2) * hz
    if not use_image:
        nz = nz_config
        z_ref = z_mean  # image rejected by its size cap: recentre on the bunch instead of the cathode
        zs_lo = z_ref + gamma * (z_lo_f - z_ref)
        zs_hi = z_ref + gamma * (z_hi_f - z_ref)
        z_lo_n, z_hi_n = axis_box(zs_lo, zs_hi)
        hz = fit_h(z_lo_n, z_hi_n, nz, previous_hz)
    cached = cache.get("solver3d")
    if cached is not None:
        c_solver, c_key = cached
        solver = c_solver if c_key == (nx, ny, nz, hx, hy, hz, use_image) else None
    else:
        solver = None
    if solver is None:
        solver = SpaceCharge3D(nx, ny, nz, hx, hy, hz, x_lo_n, y_lo_n, z_lo_n,
                               device=device, dtype=dtype, max_grid_cells=sc.max_grid_cells_3d)
        cache["solver3d"] = (solver, (nx, ny, nz, hx, hy, hz, use_image))
        cache["h3d"] = (hx, hy, hz)
        cache["nz3d_bucket"] = nz
        cache.pop("grid3d_key", None)
        logger.info("3D space-charge grid rebuilt: %dx%dx%d, h=(%.3g, %.3g, %.3g) m, image=%s",
                    nx, ny, nz, hx, hy, hz, use_image)
    else:
        solver.rescale(x_lo_n, y_lo_n, z_lo_n)

    grid_key = (solver.x_lo, solver.y_lo, solver.z_lo, hx, hy, hz)
    grid = cache.get("grid3d_t")
    if cache.get("grid3d_key") != grid_key:
        grid = tuple(torch.tensor(v, device=device, dtype=dtype)
                     for v in grid_key)
        cache["grid3d_t"] = grid
        cache["grid3d_key"] = grid_key
    x_lo_t, y_lo_t, z_lo_t, hx_t, hy_t, hz_t = grid

    z_ref_t = torch.full((), cathode_z if use_image else z_mean, device=device, dtype=dtype)
    beta_t = torch.sqrt(1.0 - 1.0 / gamma_t**2)
    if sc3d_fn is not None:
        return sc3d_fn(
            x, y, z, q, gamma_t, z_ref_t, beta_t,
            x_lo_t, y_lo_t, z_lo_t, hx_t, hy_t, hz_t,
            solver.nx, solver.ny, solver.nz, solver.px, solver.py, solver.pz,
            solver.G_k, use_image,
        )

    # CPU/pre-emission fallback stays on the class's eager public contract;
    # this is also the direct reference for the fused kernel.
    z_s = z_ref_t + gamma_t * (z - z_ref_t)
    if use_image:
        charge = solver.deposit(
            torch.cat([x, x]), torch.cat([y, y]), torch.cat([z_s, 2 * z_ref_t - z_s]),
            torch.cat([q, -q]),
        )
    else:
        charge = solver.deposit(x, y, z_s, q)
    Ex_g, Ey_g, Ez_g = solver.solve(charge)
    Ex = gamma_t * solver.gather(Ex_g, x, y, z_s)
    Ey = gamma_t * solver.gather(Ey_g, x, y, z_s)
    Ez = solver.gather(Ez_g, x, y, z_s)
    return Ex, Ey, Ez, -(beta_t / C_LIGHT) * Ey, (beta_t / C_LIGHT) * Ex, torch.zeros_like(Ez)


def _compute_space_charge(x, y, z, q, ux, uy, uz, active, device, dtype, cache,
                          sc: SpaceChargeSettings, cathode_z: float, sc_fn=None, sc3d_fn=None,
                          emitted_all=False):
    n = x.shape[0]
    if emitted_all:
        xa, ya, za, qa, uxa, uya, uza = x, y, z, q, ux, uy, uz  # active is all-True: mask would be a no-op copy
    else:
        if int(active.sum()) < 4:
            zero = torch.zeros(n, device=device, dtype=dtype)
            return zero, zero.clone(), zero.clone(), zero.clone(), zero.clone(), zero.clone()
        xa, ya, za, qa = x[active], y[active], z[active], q[active]
        uxa, uya, uza = ux[active], uy[active], uz[active]

    if sc.geometry == "3d":
        fa = _compute_space_charge_3d(xa, ya, za, qa, uxa, uya, uza, device, dtype, cache,
                                      sc, cathode_z, sc3d_fn=sc3d_fn)
        if emitted_all:
            return fa
        return tuple(torch.zeros(n, device=device, dtype=dtype).index_put_((active,), f) for f in fa)

    r_max_a = torch.sqrt((xa * xa + ya * ya).max())
    sigma_r_t = torch.sqrt(xa.var() + ya.var()) if xa.numel() > 1 else torch.zeros((), device=device, dtype=dtype)
    z_centroid_t = za.mean()
    sigma_z_t = za.std() if za.numel() > 1 else torch.zeros((), device=device, dtype=dtype)
    gamma_a = torch.sqrt(1.0 + (uxa ** 2 + uya ** 2 + uza ** 2) / C_LIGHT**2)
    gamma_t = torch.clamp(gamma_a.mean(), min=1.0 + 1e-9)
    (r_max_f, sigma_r_f, z_centroid, z_min_f, z_max_f, sigma_z, gamma_est) = torch.stack(
        [r_max_a, sigma_r_t, z_centroid_t, za.min(), za.max(), sigma_z_t, gamma_t]
    ).tolist()

    margin = sc.box_margin
    sigma_z_floored = max(sigma_z, sc.sigma_z_floor_m)
    sigma_zs = gamma_est * sigma_z_floored
    r_needed = r_max_f + margin * sigma_r_f

    z_extent = gamma_est * ((z_centroid - cathode_z) + margin * sigma_z_floored)
    domain_half = (z_extent + 1e-4) * 1.25
    cells_per_sigma_zs_img = sc.cells_per_sigma_zs if sc.mesh == "adaptive" else (SC_IMAGE_NZ_REF / margin)
    nz_needed_img = int(math.ceil(cells_per_sigma_zs_img * 2.0 * domain_half / sigma_zs)) + 1

    if sc.mesh == "adaptive":
        # Auto image cutoff: once reaching the mirror lobe needs more cells than the memory-safety cap
        # allows, turn the image off rather than silently degrading resolution to keep it.
        use_image = sc.cathode_image and nz_needed_img <= SC_IMAGE_NZ_MAX
    else:
        use_image = sc.cathode_image and (z_centroid - cathode_z) < sc.image_threshold_m
    z_ref_t = torch.full((), cathode_z, device=device, dtype=dtype) if use_image else z_centroid_t
    z_ref = cathode_z if use_image else z_centroid
    zs_min = z_ref + gamma_est * (z_min_f - z_ref)
    zs_max = z_ref + gamma_est * (z_max_f - z_ref)

    solver = None
    cached = cache.get("solver")
    if cached is not None:
        c_solver, c_rmax, c_zlo, c_zhi, c_img = cached
        if (
            r_needed <= c_rmax and zs_min >= c_zlo and zs_max <= c_zhi and use_image == c_img
            and sc.mesh != "adaptive"  # "adaptive" re-fits shape+box every recompute; "fixed" reuses
        ):
            solver = c_solver
    if solver is None:
        if sc.mesh == "adaptive":
            nr_needed = int(math.ceil(sc.cells_per_sigma_r * r_needed / max(sigma_r_f, 1e-9)))
            nr_box = _quantize_bucket_hysteresis(nr_needed, cache.get("nr_bucket"), SC_NR_BUCKETS)
        else:
            nr_box = sc.nr
        if use_image:
            nz_needed_img_capped = min(nz_needed_img, SC_IMAGE_NZ_MAX)
            if sc.mesh == "adaptive":
                nz_box = _quantize_bucket_hysteresis(nz_needed_img_capped, cache.get("nz_bucket"), SC_IMAGE_NZ_BUCKETS)
            else:
                nz_box = _quantize_bucket(nz_needed_img_capped, SC_IMAGE_NZ_BUCKETS)
            dz = 2.0 * domain_half / (nz_box - 1)
            half = dz * (nz_box - 1) / 2.0
            z_lo, z_hi = cathode_z - half, cathode_z + half
        else:
            z_lo = zs_min - margin * sigma_zs - 1e-5
            z_hi = zs_max + margin * sigma_zs + 1e-5
            z_pad = 0.25 * (z_hi - z_lo)
            z_lo, z_hi = z_lo - z_pad, z_hi + z_pad
            if sc.mesh == "adaptive":
                nz_needed = int(math.ceil(sc.cells_per_sigma_zs * (z_hi - z_lo) / sigma_zs))
                nz_box = _quantize_bucket_hysteresis(nz_needed, cache.get("nz_bucket"), SC_NZ_BUCKETS)
            else:
                nz_box = sc.nz
        if sc.mesh == "adaptive":
            nr_box, nz_box = _limit_mesh_by_particle_occupancy(
                xa.numel(),
                nr_box,
                nz_box,
                SC_NR_BUCKETS,
                SC_IMAGE_NZ_BUCKETS if use_image else SC_NZ_BUCKETS,
                sc.min_particles_per_cell,
            )
        r_max = r_needed + 1e-6
        if cached is not None and cached[0].nr == nr_box and cached[0].nz == nz_box:
            # Same grid shape: re-fit in place (sub-ms, no matrix inverse).
            solver = cached[0]
            solver.rescale(r_max, z_lo, z_hi)
        else:
            solver = CylindricalSpaceCharge(nr=nr_box, nz=nz_box, r_max=r_max, z_min=z_lo, z_max=z_hi,
                                            device=device, dtype=dtype)
        cache["solver"] = (solver, r_max, z_lo, z_hi, use_image)
        cache["nr_bucket"], cache["nz_bucket"] = nr_box, nz_box
        # 0-d device tensors, rebuilt only on a box change -- see docs/tracking_native.md's
        # "torch.compile discipline".
        cache["grid_t"] = tuple(torch.tensor(v, device=device, dtype=dtype)
                                for v in (solver.dr, solver.dz, solver.z_min))

    if sc_fn is not None:
        beta_t = torch.sqrt(1.0 - 1.0 / gamma_t**2)
        dr_t, dz_t, zmin_t = cache["grid_t"]
        Exa, Eya, Eza, Bxa, Bya, Bza = sc_fn(
            xa, ya, za, qa, gamma_t, z_ref_t, beta_t,
            dr_t, dz_t, zmin_t, solver.nr, solver.nz, solver.nz_pad,
            solver.node_volume, solver.P_left, solver.inv_diag, solver.P_right, use_image,
            sc.smooth_r, sc.smooth_z, sc.smooth_gamma_ref,
        )
    else:
        Exa, Eya, Eza, Bxa, Bya, Bza = relativistic_space_charge_field(
            solver, xa, ya, za, qa, gamma_est, cathode_z=cathode_z if use_image else None,
            smooth_r=sc.smooth_r, smooth_z=sc.smooth_z, smooth_gamma_ref=sc.smooth_gamma_ref,
        )
    if emitted_all:
        return Exa, Eya, Eza, Bxa, Bya, Bza
    out = []
    for fa in (Exa, Eya, Eza, Bxa, Bya, Bza):
        full = torch.zeros(n, device=device, dtype=dtype)
        full[active] = fa
        out.append(full)
    return tuple(out)


def _differentiable_space_charge(x, y, z, q, ux, uy, uz, active, solver, fixed_box, cathode_z, sc, sc_fn):
    """Compute differentiable space charge; see the module documentation for details."""
    w = None if active is None else active.to(x.dtype)

    def mean(a):
        return a.mean() if w is None else (a * w).sum() / w.sum()

    def var(a):  # unbiased, as _compute_space_charge's .var()/.std()
        n = a.numel() if w is None else w.sum()
        return mean((a - mean(a)) ** 2) * n / (n - 1)

    gamma = torch.sqrt(1.0 + (ux * ux + uy * uy + uz * uz) / C_LIGHT**2)
    gamma_t = torch.clamp(mean(gamma), min=1.0 + 1e-9)
    beta_t = torch.sqrt(1.0 - 1.0 / gamma_t**2)
    z_c = mean(z)
    if w is not None:
        q = q * w
    z_ref = z_c if cathode_z is None else torch.full_like(z_c, cathode_z)
    nr, nz, m = solver.nr, solver.nz, sc.box_margin
    if fixed_box is None:
        r2 = x * x + y * y
        sigma_r = torch.sqrt(var(x) + var(y))
        r_max = torch.sqrt(r2.max() if w is None else torch.where(active, r2, 0.0).max()) + m * sigma_r + 1e-6
        sigma_z = torch.clamp(torch.sqrt(var(z)), min=sc.sigma_z_floor_m)
        if cathode_z is None:
            z_hi = z.max() if w is None else torch.where(active, z, -math.inf).max()
            z_lo = z.min() if w is None else torch.where(active, z, math.inf).min()
            lo = gamma_t * (z_lo - z_c - m * sigma_z) - 1e-5
            hi = gamma_t * (z_hi - z_c + m * sigma_z) + 1e-5
            lo, hi = lo - 0.25 * (hi - lo), hi + 0.25 * (hi - lo)
        else:
            hi = (gamma_t * ((z_c - z_ref) + m * sigma_z) + 1e-4) * 1.25
            lo = -hi
        dr, dz, z_min = r_max / nr, (hi - lo) / (nz - 1), z_ref + lo
    else:
        r_max, z_half = fixed_box
        dr, dz, z_min = r_max / nr, 2.0 * z_half / (nz - 1), z_ref - z_half
    node_volume, inv_diag = solver.coefficients(dr, dz)
    fields = sc_fn(x, y, z, q, gamma_t, z_ref, beta_t, dr, dz, z_min, nr, nz, solver.nz_pad,
                   node_volume, solver.P_left, inv_diag, solver.P_right, cathode_z is not None,
                   sc.smooth_r, sc.smooth_z, sc.smooth_gamma_ref)
    with torch.no_grad():
        zs = z_ref + gamma_t * (z - z_ref)
        outside = (zs < z_min) | (zs > z_min + dz * (nz - 1)) | (x * x + y * y > (dr * nr) ** 2)
        overflow = outside.to(x.dtype).mean() if active is None else (outside & active).to(x.dtype).sum() / active.sum()
    return fields, overflow
