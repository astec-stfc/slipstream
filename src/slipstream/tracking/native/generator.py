"""Samples ASTRA-style initial beams with configurable distributions, correlations, offsets, species
and emission times. See docs/modules/tracking/native/generator.md."""
from __future__ import annotations

import math
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .astra_io import distribution_from_rows, read_astra_namelist_groups

# --- constants measured from the ASTRA generator (see module docstring) -------------------------
ASTRA_MASS_EV = {"electrons": 511003.0, "positrons": 511003.0, "protons": 938.28e6, "hydrogen": 939.30e6}
_SPECIES_INDEX = {"electrons": 1, "positrons": 2, "protons": 3, "hydrogen": 4}
_SPECIES_SIGN = {"electrons": -1.0, "positrons": 1.0, "protons": 1.0, "hydrogen": -1.0}  # hydrogen: an H- ion
# Halton base of each dimension, as the oracle uses them (the particle index n = 1..N is the digit source).
_HALTON_BASE = {"E": 2, "t": 3, "z": 5, "py": 7, "px": 11, "y": 13, "x": 17}
_PROBE_SIGMAS = (0.5, 1.0, 1.5)  # the 6 probes sit at these multiples of the rms size (manual 7.1, ``Probe``)


@dataclass
class GeneratorInput:
    """One ``&INPUT`` namelist."""
    fname: str = "rfgun.ini"
    add: bool = False
    n_add: int = 0
    ipart: int = 100
    species: str = "electrons"
    ion_mass: dict = field(default_factory=dict)
    probe: bool = True
    passive: bool = False
    noise_reduc: bool = True
    cathode: bool = True
    r_cathode: float = 0.0
    high_res: bool = False
    binary: bool = False
    q_total: float = 1.0
    type: str = "standard"
    rad: float = 0.0
    tau: float = 0.0
    ref_zpos: float = 0.0
    ref_clock: float = 0.0
    ref_ekin: float = 0.0
    dist_z: str = "uniform"
    sig_z: float = 0.0
    c_sig_z: float = 0.0
    lz: float = 0.0
    rz: float = 0.0
    sig_clock: float = 1.0e-3
    c_sig_clock: float = 0.0
    lt: float = 0.0
    rt: float = 0.0
    dist_pz: str = "uniform"
    sig_ekin: float = 0.0
    c_sig_ekin: float = 0.0
    le: float = 0.0
    re: float = 0.0
    emit_z: float = 0.0
    cor_ekin: float = 0.0
    e_photon: float = 0.0
    phi_eff: float = 0.0
    dist_x: str = "gaussian"
    sig_x: float = 1.0
    c_sig_x: float = 0.0
    lx: float = 0.0
    rx: float = 0.0
    x_off: float = 0.0
    disp_x: float = 0.0
    dist_px: str = "gaussian"
    nemit_x: float = 0.0
    sig_px: float = 0.0
    c_sig_px: float = 0.0
    lpx: float = 0.0
    rpx: float = 0.0
    cor_px: float = 0.0
    dist_y: str = "gaussian"
    sig_y: float = 1.0
    c_sig_y: float = 0.0
    ly: float = 0.0
    ry: float = 0.0
    y_off: float = 0.0
    disp_y: float = 0.0
    dist_py: str = "gaussian"
    nemit_y: float = 0.0
    sig_py: float = 0.0
    c_sig_py: float = 0.0
    lpy: float = 0.0
    rpy: float = 0.0
    cor_py: float = 0.0

    @property
    def n_probe(self) -> int:
        return 6 if self.probe else 0

    def species_info(self) -> tuple[int, float, float]:
        """``(particle index, charge sign, mass eV/c^2)``."""
        name = self.species.strip().lower()
        if name in _SPECIES_INDEX:
            return _SPECIES_INDEX[name], _SPECIES_SIGN[name], ASTRA_MASS_EV[name]
        if name.startswith("ion"):  # ``ion 3``: user particle 3, mass/charge = |ion_mass(3)|, charge sign = its sign
            n = int(name.split()[1]) if len(name.split()) > 1 else 1
            m = float(self.ion_mass.get(n, 0.0))
            if m == 0.0:
                raise ValueError(f"species {self.species!r}: ion_mass({n}) must be set")
            return 4 + n, math.copysign(1.0, m), abs(m)
        raise ValueError(f"unknown species {self.species!r}")

    @classmethod
    def from_namelist(cls, group: dict) -> "GeneratorInput":
        """Build from one parsed ``&INPUT`` group (``astra_io.read_astra_namelist_groups``)."""
        defaults = cls()
        ignored = {"lprompt"}  # generator's console-prompt switch: irrelevant here
        bad = sorted(k for k in group if not hasattr(defaults, k) and k not in ignored)
        if bad:
            raise ValueError(f"unknown generator namelist key(s) {bad}")
        kw = {}
        for key, val in group.items():
            if key not in ignored:
                typ = type(getattr(defaults, key))
                kw[key] = val if typ is dict else typ(val)
        return cls(**kw)

    def reference_row(self) -> np.ndarray:
        """Row 0 of the file (absolute ``z``, ``pz``, ``clock``). It depends on no sampling, so a caller
        that only needs the reference particle (auto-phasing) does not have to generate a beam."""
        index, sign, mass = self.species_info()
        return np.array([self.x_off * 1e-3, self.y_off * 1e-3, self.ref_zpos, 0.0, 0.0,
                         float(_p_of_E(self._emission_energy_eV(), mass)), self.ref_clock,
                         sign * abs(self.q_total) / self.ipart, index, self._status()[0]])

    def _emission_energy_eV(self) -> float:
        """Kinetic energy of the reference particle (eV): ``Ref_Ekin`` plus, for ``isotropic`` emission,
        ``LE`` (measured: ``Ref_Ekin = 1 keV, LE = 1 eV`` gives ``p(1001 eV)``)."""
        return self.ref_ekin * 1e6 + (self.le * 1e3 if _kind(self.dist_pz, "pz") == "isotropic" else 0.0)

    def _status(self) -> tuple[int, int]:
        """``(particle, probe)`` status flags (manual table 2)."""
        if self.cathode:
            return (-5, -6) if self.passive else (-1, -3)
        return (1, 0) if self.passive else (5, 3)


def read_generator_input(path: str | Path) -> list[GeneratorInput]:
    """Every ``&INPUT`` namelist of a ``generator.in`` (more than one only with ``Add=T``)."""
    groups = [g for name, g in read_astra_namelist_groups(path) if name == "input"]
    if not groups:
        raise ValueError(f"{path}: no &INPUT namelist")
    return [GeneratorInput.from_namelist(g) for g in groups]


# ------------------------------------------------------------------ low-level samplers
def _radical_inverse(n: torch.Tensor, base: int) -> torch.Tensor:
    """van der Corput radical inverse of the integers ``n`` (>= 1) in ``base``."""
    n = n.clone()
    f = torch.ones(n.shape, dtype=torch.float64)
    out = torch.zeros(n.shape, dtype=torch.float64)
    while bool((n > 0).any()):
        f = f / base
        out = out + (n % base).to(torch.float64) * f
        n = n // base
    return out


def _uniforms(n: int, sequence: str, seed: int) -> dict[str, torch.Tensor]:
    """``n`` points of the 7-dimensional unit cube, one named column per phase-space dimension."""
    if sequence == "halton":
        idx = torch.arange(1, n + 1, dtype=torch.int64)
        return {k: _radical_inverse(idx, b) for k, b in _HALTON_BASE.items()}
    if sequence == "sobol":
        u = torch.quasirandom.SobolEngine(len(_HALTON_BASE), scramble=True, seed=seed).draw(n, dtype=torch.float64)
    elif sequence == "random":
        u = torch.rand(n, len(_HALTON_BASE), generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    else:
        raise ValueError(f"sequence must be 'halton', 'sobol' or 'random', not {sequence!r}")
    u = u.clamp(2.0 ** -53, 1.0 - 2.0 ** -53)
    return {k: u[:, i] for i, k in enumerate(_HALTON_BASE)}


def _kind(name: str, role: str) -> str:
    """ASTRA keyword (``'p'``, ``'radial'``, ``'uniform ellipsoid'``, ...) -> canonical kind. ASTRA matches
    on the first letters; ``'i'`` is *inverted* for a position/time and *isotropic* for ``pz``."""
    s = name.strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    if s.startswith(("ue", "ellips", "uniformellips")):
        return "ellipsoid"
    if s.startswith("fd"):
        return "fd300"
    if s.startswith("2d"):
        return "gauss2d"
    for prefix, kind in (("u", "uniform"), ("p", "plateau"), ("g", "gauss"), ("r", "radial")):
        if s.startswith(prefix):
            return kind
    if s.startswith("i"):
        return "isotropic" if role == "pz" else "inverted"
    raise ValueError(f"unknown distribution {name!r} for {role}")


def _ndtri_cut(v: torch.Tensor, cut: float) -> torch.Tensor:
    """``-ndtri(v)`` (ASTRA's sign: the value falls as the sequence rises), mapped onto ``|value| <= cut``."""
    if cut > 0:
        p = 0.5 * math.erfc(cut / math.sqrt(2.0))
        v = p + v * (1.0 - 2.0 * p)
    return -torch.special.ndtri(v)


def _plateau_ppf(u: torch.Tensor, length: float, rise: float) -> torch.Tensor:
    """Inverse CDF of ASTRA's plateau ``f(x) = 1/(1 + exp(4 (|x| - L/2)/rt))`` (manual 7.2.2; measured:
    the ``4/rt`` is what makes the tangent at ``L/2`` climb from 0 to 1 in ``rt``)."""
    if rise <= 0.0:
        return length * (u - 0.5)
    zero = u <= 0.0  # base 2 hits u = 0 exactly (n = 1, v = 1/2); measured: ASTRA puts that one particle at the centre
    s, mu = rise / 4.0, length / 2.0
    sp = torch.nn.functional.softplus
    c0 = s * float(sp(torch.tensor(-mu / s, dtype=torch.float64)))
    half = mu + c0  # int_0^inf f

    def cdf(x):
        a = x.abs()
        return 0.5 + torch.sign(x) * (a - s * sp((a - mu) / s) + c0) / (2.0 * half)

    lo = torch.full_like(u, -(mu + 40.0 * s))
    hi = -lo
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        below = cdf(mid) < u
        lo = torch.where(below, mid, lo)
        hi = torch.where(below, hi, mid)
    return torch.where(zero, torch.zeros_like(lo), 0.5 * (lo + hi))


def _inverted_ppf(u: torch.Tensor, a: float) -> torch.Tensor:
    """Inverse CDF of the inverted parabola ``3/(4a) (1 - z^2/a^2)``: ``F = 1/2 + (3 s - s^3)/4`` with
    ``s = z/a`` is a depressed cubic, solved by its trigonometric root in ``[-1, 1]``."""
    return a * 2.0 * torch.cos(torch.acos(1.0 - 2.0 * u) / 3.0 + 4.0 * math.pi / 3.0)


def _rms(kind: str, sig: float, length: float, rise: float) -> float:
    """Nominal rms of a 1D distribution -- what the rescale targets and where the probes sit. A length
    parameter (``L``, ``rt``) overrides ``sig`` for ``uniform``/``plateau``/``ellipsoid``."""
    if length > 0 and kind == "uniform":
        return length / math.sqrt(12.0)
    if length > 0 and kind == "plateau":
        return math.sqrt(length ** 2 / 12.0 + (math.pi * rise) ** 2 / 48.0)  # Sommerfeld: exact for a Fermi edge
    if length > 0 and kind == "ellipsoid":
        return length / math.sqrt(5.0)
    return sig


def _sample1d(kind: str, v: torch.Tensor, sig: float, cut: float, length: float, rise: float) -> torch.Tensor:
    """Zero-mean 1D samples from the unit-interval points ``v`` (ASTRA's mappings, see module doc)."""
    if kind == "gauss":
        return sig * _ndtri_cut(v, cut)
    if kind == "uniform":
        return (length if length > 0 else math.sqrt(12.0) * sig) * (v - 0.5)
    if kind == "plateau":
        if length <= 0:
            length, rise = math.sqrt(12.0) * sig, 0.0
        return _plateau_ppf(torch.remainder(0.5 - v, 1.0), length, rise)
    if kind == "inverted":
        return _inverted_ppf(v, math.sqrt(5.0) * sig)
    raise ValueError(f"{kind!r} is not a 1D distribution here")


def _conform(real: torch.Tensor, fixed: torch.Tensor, target: float, normalise: bool) -> torch.Tensor:
    """ASTRA's post-sampling fix-up: zero the mean and, if ``normalise``, set the rms to ``target`` --
    both taken over **all** rows (``fixed`` = the reference/probe rows' values), so the particles
    end at ``mean = -sum(fixed)/N_real`` and the whole file at exactly ``(0, target)``."""
    n_real, n_all = real.numel(), real.numel() + fixed.numel()
    c = -float(fixed.sum()) / n_real
    d = real - real.mean()
    d2 = float((d ** 2).sum())
    if not normalise or d2 == 0.0:  # nothing to rescale (e.g. sig_px = 0)
        return d + c
    a2 = (n_all * target ** 2 - float((fixed ** 2).sum()) - n_real * c * c) / d2
    return d * math.sqrt(a2) + c


def _decorrelate(p: torch.Tensor, pos: torch.Tensor, fixed_pos: torch.Tensor) -> torch.Tensor:
    """Remove the linear ``pos``-``p`` correlation, measured over **all** rows (measured: the generator
    prints it as "correlated divergence" and leaves ~0 unless ``cor_px`` asks for one; a 1031-row
    Halton isotropic beam starts at ``<x px>/<x^2> = -9754 eV/c/m`` and ends at ~0)."""
    pa = torch.cat([fixed_pos, pos])
    d = pa - pa.mean()
    pp = torch.cat([torch.zeros_like(fixed_pos), p])
    return p - float((d * (pp - pp.mean())).sum() / (d * d).sum()) * (pos - pa.mean())


def _p_of_E(E_eV, m):
    """Momentum (eV/c) of kinetic energy ``E`` (eV); floats or tensors."""
    return (E_eV * (E_eV + 2.0 * m)) ** 0.5


def _sigma_p(nemit: float, sig_p: float, sigma_pos_mm: float, mass: float) -> float:
    """rms momentum (eV/c): ``Nemit`` [mm mrad] wins over ``sig_p`` (manual 7.1), via eps_n = sigma_x sigma_px/(m c)."""
    return nemit * mass / (1e3 * sigma_pos_mm) if nemit > 0 and sigma_pos_mm > 0 else sig_p


# ------------------------------------------------------------------ one namelist -> rows
def _block(inp: GeneratorInput, ipart: int, sequence: str, seed: int, normalise: bool) -> np.ndarray:
    """The ``(ipart, 10)`` ASTRA-layout array of one ``&INPUT``: reference row, probes, particles."""
    if inp.r_cathode:
        raise NotImplementedError("R_cathode (curved cathode) is not supported")
    if inp.binary:
        raise NotImplementedError("Binary output is not supported")
    index, sign, mass = inp.species_info()
    n_probe = inp.n_probe
    n = ipart - 1 - n_probe
    if n < 1:
        raise ValueError(f"Ipart={ipart} leaves no particles after the reference row and {n_probe} probes")
    U = _uniforms(n, sequence, seed)
    g = torch.Generator().manual_seed(seed + 1)  # the dimensions ASTRA keeps random even with Noise_reduc=T

    def rnd():
        return torch.rand(n, generator=g, dtype=torch.float64).clamp(2.0 ** -53, 1.0 - 2.0 ** -53)

    cathode = inp.cathode
    k_x, k_y, k_z = _kind(inp.dist_x, "x"), _kind(inp.dist_y, "y"), _kind(inp.dist_z, "z")
    k_px, k_py, k_pz = _kind(inp.dist_px, "px"), _kind(inp.dist_py, "py"), _kind(inp.dist_pz, "pz")
    two_d = k_x in ("radial", "gauss2d")  # x and y are then one 2D distribution (sig_y unused)
    # the longitudinal axis is the emission time [ns] from a cathode, else the position [mm]
    z_sig, z_cut, z_L, z_r = ((inp.sig_clock, inp.c_sig_clock, inp.lt, inp.rt) if cathode
                              else (inp.sig_z, inp.c_sig_z, inp.lz, inp.rz))
    z_dim = "t" if cathode else "z"
    sx = _rms(k_x, inp.sig_x, inp.lx, inp.rx)
    sy = sx if two_d else _rms(k_y, inp.sig_y, inp.ly, inp.ry)
    sz = _rms(k_z, z_sig, z_L, z_r)
    cut_x = inp.c_sig_x > 0 and k_x in ("gauss", "gauss2d")
    cut_y = cut_x if two_d else inp.c_sig_y > 0 and k_y == "gauss"
    cut_z = z_cut > 0 and k_z == "gauss"

    # ---- positions [mm] and the longitudinal coordinate [mm | ns]
    if k_z == "ellipsoid":  # uniform filled ellipsoid: x, y, z/t from one ball (rho: x dim, phi: y dim, cos theta: z/t dim)
        ax, ay, az = (math.sqrt(5.0) * s for s in (sx, sy, sz))
        rho, phi, ct = U["x"] ** (1.0 / 3.0), 2.0 * math.pi * U["y"], 2.0 * U[z_dim] - 1.0
        st = torch.sqrt(1.0 - ct * ct)
        x, y, z = ax * rho * st * torch.cos(phi), ay * rho * st * torch.sin(phi), az * rho * ct
    else:
        if two_d:  # r from the x dimension, phi from the y one
            phi = 2.0 * math.pi * U["y"]
            if k_x == "radial":
                r = (inp.lx if inp.lx > 0 else 2.0 * sx) * torch.sqrt(U["x"])
            else:  # circularly cut 2D Gaussian: Rayleigh r
                keep = -math.expm1(-0.5 * inp.c_sig_x ** 2) if inp.c_sig_x > 0 else 1.0
                r = sx * torch.sqrt(-2.0 * torch.log1p(-U["x"] * keep))
            x, y = r * torch.cos(phi), r * torch.sin(phi)
        else:
            x = _sample1d(k_x, U["x"], inp.sig_x, inp.c_sig_x, inp.lx, inp.rx)
            y = _sample1d(k_y, U["y"], inp.sig_y, inp.c_sig_y, inp.ly, inp.ry)
        z = _sample1d(k_z, U[z_dim], z_sig, z_cut, z_L, z_r)
    ring = inp.type.strip().lower().startswith("r")
    if ring:  # every particle is offset by ``Rad`` at a random angle (measured: x, y are then left unconformed)
        psi = 2.0 * math.pi * rnd()
        x, y = x + inp.rad * torch.cos(psi), y + inp.rad * torch.sin(psi)

    # the reference row and the probes, in the same variables: an x probe at +k sx (and +k sz), a y probe at +k sy (-k sz)
    fx, fy, fz = (torch.zeros(1 + n_probe, dtype=torch.float64) for _ in range(3))
    if n_probe:
        k = torch.tensor(_PROBE_SIGMAS, dtype=torch.float64)
        fx[1::2], fy[2::2] = k * sx, k * sy  # file rows 1, 3, 5 are the x probes, 2, 4, 6 the y probes
        fz[1::2], fz[2::2] = k * sz, -k * sz
    if not ring:
        x = _conform(x, fx, sx, normalise and not cut_x)
        y = _conform(y, fy, sy, normalise and not cut_y)
    z = _conform(z, fz, sz, normalise and not cut_z)
    if inp.tau > 0 and cathode:  # exponential emission delay, random even with Noise_reduc=T, mean re-zeroed. Measured: the
        z = _conform(z + inp.tau * torch.log(rnd()), fz, sz, False)  # tail runs to *negative* clock values (skew < 0)

    # ---- momentum: the energy / isotropic emission, then the transverse planes [eV/c]
    E0 = inp._emission_energy_eV()
    p_ref = float(_p_of_E(E0, mass))
    zero_f = torch.zeros(1 + n_probe, dtype=torch.float64)
    if k_pz == "isotropic":
        p_le = float(_p_of_E(inp.le * 1e3, mass))
        c, phi_p = U["py"], 2.0 * math.pi * U["px"]
        pt = p_le * torch.sqrt(1.0 - c * c)
        px, py = pt * torch.cos(phi_p), pt * torch.sin(phi_p)
        pz_abs = torch.sqrt(torch.clamp(p_ref ** 2 - pt * pt, min=0.0))  # energy conservation, before px, py are decorrelated
        px, py = _decorrelate(px, x, fx), _decorrelate(py, y, fy)
        px, py = _conform(px, zero_f, 0.0, False), _conform(py, zero_f, 0.0, False)  # measured: centred, not rescaled
    elif k_pz == "fd300":
        e_m = inp.e_photon - inp.phi_eff
        if e_m <= 0:
            raise ValueError("FD_300 needs E_photon > phi_eff")
        E = e_m * torch.sqrt(rnd())  # energy pdf ~ E on [0, E_m], Lambertian polar angle: reproduces the manual's
        c = torch.sqrt(rnd())        # <E> = 2/3 E_m, sigma_E = E_m/(3 sqrt 2), sigma_px = sqrt(m E_m/3)
        p = _p_of_E(E, mass)
        phi_p = 2.0 * math.pi * rnd()
        px, py = p * torch.sqrt(1.0 - c * c) * torch.cos(phi_p), p * torch.sqrt(1.0 - c * c) * torch.sin(phi_p)
        pz_abs = p_ref + p * c
    else:
        # ``emit_z`` [keV mm] gives the energy spread only when ``sig_Ekin`` is unset (manual), and needs a bunch length
        if inp.sig_ekin <= 0 and inp.emit_z > 0 and (cathode or sz <= 0):
            raise ValueError("emit_z needs a bunch length: use Cathode=F with sig_z/Lz > 0, or give sig_Ekin")
        sig_e = inp.sig_ekin if inp.sig_ekin > 0 or inp.emit_z <= 0 else inp.emit_z / sz
        dE = 1e3 * _sample1d(k_pz, U["E"], sig_e, inp.c_sig_ekin, inp.le, inp.re)
        pz_abs = _p_of_E(torch.clamp(E0 + dE, min=0.0), mass)
        sig_px = _sigma_p(inp.nemit_x, inp.sig_px, sx, mass)
        sig_py = _sigma_p(inp.nemit_y, inp.sig_py, sy, mass)
        if k_px == "radial":  # px, py together: r from the px dimension, phi from the py one
            r = (inp.lpx if inp.lpx > 0 else 2.0 * sig_px) * torch.sqrt(U["px"])
            phi_p = 2.0 * math.pi * U["py"]
            px, py = r * torch.cos(phi_p), r * torch.sin(phi_p)
            t_px = t_py = sig_px
            cut_px = cut_py = False
        else:
            px = _sample1d(k_px, U["px"], sig_px, inp.c_sig_px, inp.lpx, inp.rpx)
            py = _sample1d(k_py, U["py"], sig_py, inp.c_sig_py, inp.lpy, inp.rpy)
            cut_px, cut_py = inp.c_sig_px > 0 and k_px == "gauss", inp.c_sig_py > 0 and k_py == "gauss"
            t_px, t_py = _rms(k_px, sig_px, inp.lpx, inp.rpx), _rms(k_py, sig_py, inp.lpy, inp.rpy)
        px, py = _decorrelate(px, x, fx), _decorrelate(py, y, fy)
        px = _conform(px, zero_f, t_px, normalise and not cut_px)
        py = _conform(py, zero_f, t_py, normalise and not cut_py)

    # ---- correlations and dispersion.
    P = 1 + n_probe
    pz_probe = float(_p_of_E(inp.le * 1e3, mass)) if k_pz == "isotropic" else p_ref  # measured: isotropic probes carry p(LE)
    X, Y, Z = torch.cat([fx[1:], x]), torch.cat([fy[1:], y]), torch.cat([fz[1:], z])
    PX, PY = torch.cat([torch.zeros(n_probe, dtype=torch.float64), px]), torch.cat([torch.zeros(n_probe, dtype=torch.float64), py])
    PZ = torch.cat([torch.full((n_probe,), pz_probe, dtype=torch.float64), pz_abs])
    if inp.cor_ekin and k_pz not in ("isotropic", "fd300"):  # energy change cor_Ekin [keV] at z = sigma_z
        PZ = _p_of_E(torch.sqrt(PZ ** 2 + mass ** 2) - mass + 1e3 * inp.cor_ekin * Z / sz, mass)
    px0, py0 = PX.clone(), PY.clone()
    if inp.cor_px:  # divergence x' = cor_px [mrad] at x = sigma_x (the oracle's slope is within ~0.6 % of this)
        PX = PX + p_ref * 1e-3 * inp.cor_px * X / sx
    if inp.cor_py:
        PY = PY + p_ref * 1e-3 * inp.cor_py * Y / sy
    if inp.cor_px or inp.cor_py:
        PZ = torch.sqrt(torch.clamp(PZ ** 2 + px0 ** 2 + py0 ** 2 - PX ** 2 - PY ** 2, min=0.0))
    if inp.disp_x or inp.disp_y:
        delta = (PZ - p_ref) / p_ref
        delta = delta - delta.mean()
        X, Y = X + inp.disp_x * 1e3 * delta, Y + inp.disp_y * 1e3 * delta

    # ---- the file rows: positions carry the offsets; z / clock / pz are relative to the reference row
    status_p, status_probe = inp._status()
    rows = np.zeros((ipart, 10))
    rows[:, 7], rows[:, 8] = sign * abs(inp.q_total) / ipart, index
    rows[0] = dataclasses.replace(inp, ipart=ipart).reference_row()
    zc, zs = (6, 1.0) if cathode else (2, 1e-3)
    rows[1:, 0], rows[1:, 1] = (inp.x_off + X.numpy()) * 1e-3, (inp.y_off + Y.numpy()) * 1e-3
    rows[1:, zc] = Z.numpy() * zs
    rows[1:, 3], rows[1:, 4] = PX.numpy(), PY.numpy()
    rows[1:, 5] = PZ.numpy() - p_ref
    rows[1:P, 9], rows[P:, 9] = status_probe, status_p
    return rows


# ------------------------------------------------------------------ the beam object
@dataclass
class Beam:
    """A generated distribution in ASTRA's file layout: ``rows`` is ``(N, 10)`` --
    ``x y z px py pz clock macro_charge index status`` (m, m, m, eV/c, eV/c, eV/c, ns, nC), row 0 the
    reference particle (absolute), the rest with ``z``/``pz``/``clock`` relative to it (manual table 1)."""
    rows: np.ndarray
    q_total_nC: float
    high_res: bool = True

    def particles(self, device=None, dtype=None, exact_charge: bool = True) -> dict[str, torch.Tensor]:
        """The trackable particles, as ``astra_io.read_astra_distribution`` returns a file's: the
        standard rows (probes and passive particles excluded), absolute ``z``/``pz``/``clock``."""
        d = distribution_from_rows(self.rows, device, dtype, status=(-1, 5))
        if exact_charge and d["q"].numel():
            d["q"] = d["q"] * (abs(self.q_total_nC) * 1e-9 / abs(float(d["q"].sum())))
        return d

    def write(self, path: str | Path, high_res: bool | None = None) -> None:
        """ASTRA's text format: ``1P,8E12.4,2I4`` (default) or ``1P,8E20.12,2I4`` (``High_res``)."""
        w = "20.12" if (self.high_res if high_res is None else high_res) else "12.4"
        np.savetxt(path, self.rows, fmt=[f"%{w}E"] * 8 + ["%4d", "%4d"], delimiter="")

    def moments(self) -> dict[str, float]:
        """The quantities ASTRA's generator prints in its "Final check", over the standard particles (not the reference row or probes): rms sizes
        (mm, ns), kinetic energy (MeV) and spread (keV), normalised rms emittances (mm mrad,
        ``sqrt(<x^2><px^2> - <x px>^2)/(m c)``) -- for eyeballing a deck against ASTRA."""
        r = self.rows
        sel = r[1:][np.isin(r[1:, 9], (-1, 5))]  # row 0 (the reference particle) holds absolute z, pz, clock: left out
        m = {1: 511003.0, 2: 511003.0, 3: 938.28e6, 4: 939.30e6}.get(int(r[0, 8]), 511003.0)
        E = np.sqrt((sel[:, 5] + r[0, 5]) ** 2 + m ** 2) - m

        def emit(a, pa):
            a, pa = a - a.mean(), pa - pa.mean()
            return 1e6 * math.sqrt(max((a * a).mean() * (pa * pa).mean() - (a * pa).mean() ** 2, 0.0)) / m

        return {"n": len(sel), "sig_x_mm": 1e3 * sel[:, 0].std(), "sig_y_mm": 1e3 * sel[:, 1].std(),
                "sig_z_mm": 1e3 * sel[:, 2].std(), "sig_t_ns": sel[:, 6].std(), "Ekin_MeV": E.mean() * 1e-6,
                "dE_keV": E.std() * 1e-3, "eps_x_mm_mrad": emit(sel[:, 0], sel[:, 3]),
                "eps_y_mm_mrad": emit(sel[:, 1], sel[:, 4])}


def generate(
    source: "str | Path | GeneratorInput | list[GeneratorInput]",
    ipart: int | None = None,
    seed: int = 0,
    sequence: str | None = None,
    noise_reduc: bool | None = None,
    normalise: bool = True,
) -> Beam:
    """Generate a beam from a ``generator.in`` path or already-built ``GeneratorInput`` (a list =
    ``Add=T``)."""
    inputs = [source] if isinstance(source, GeneratorInput) else (
        list(source) if isinstance(source, (list, tuple)) else read_generator_input(source))
    counts = [i.ipart for i in inputs]
    if ipart is not None:
        counts = [max(i.n_probe + 2, round(ipart * c / sum(counts))) for i, c in zip(inputs, counts)]
    blocks = []
    for k, (inp, cnt) in enumerate(zip(inputs, counts)):
        nr = inp.noise_reduc if noise_reduc is None else noise_reduc
        blk = _block(inp, cnt, sequence or ("halton" if nr else "random"), seed + 1000 * k, normalise)
        blocks.append(blk if k == 0 else blk[1:])  # the combined file keeps the first reference particle
    return Beam(np.concatenate(blocks), sum(abs(i.q_total) for i in inputs), inputs[0].high_res)


def main(argv: list[str] | None = None) -> Beam:
    import argparse

    p = argparse.ArgumentParser(description="Native ASTRA generator: generator.in -> particle-distribution file")
    p.add_argument("input", help="generator.in")
    p.add_argument("-n", "--ipart", type=int, default=None, help="override Ipart (counts the reference + probe rows)")
    p.add_argument("-o", "--output", default=None, help="output file (default: the deck's FName, next to it)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sequence", choices=["halton", "sobol", "random"], default=None)
    p.add_argument("--no-normalise", dest="normalise", action="store_false")
    a = p.parse_args(argv)
    beam = generate(a.input, ipart=a.ipart, seed=a.seed, sequence=a.sequence, normalise=a.normalise)
    out = Path(a.output) if a.output else Path(a.input).with_name(read_generator_input(a.input)[0].fname)
    beam.write(out)
    print(f"wrote {out}: {len(beam.rows)} rows, " + ", ".join(f"{k}={v:.5g}" for k, v in beam.moments().items()))
    return beam


if __name__ == "__main__":
    main()

