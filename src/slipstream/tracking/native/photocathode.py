"""Opt-in laser-driven photocathode emission for ``Injector``: pulse energy times a Schottky-dependent quantum
efficiency, limited by the net field at the cathode surface. See docs/modules/tracking/native/photocathode.md."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ...constants import E_CHARGE_C, EPS0_F_PER_M

_HC_EV_NM = 1239.841984  # h c / e in eV nm
# (work function, Fermi energy) in eV. Dowell & Schmerge, PRSTAB 12, 074201 (2009), Table I.
_MATERIALS = {"Cu": (4.31, 7.0)}
_MAX_BINS = 64  # equal-count radial bins of the macroparticle cloud
_SUB_RINGS = 8  # sub-rings per bin in the surface-field sum
_AZIMUTH = 64  # azimuth nodes of the ring kernel, clustered at 0 where it peaks
_BISECT = 32


def schottky_lowering_eV(field_V_per_m, enhancement: float = 1.0):
    """Work-function reduction ``sqrt(e*beta*E/(4*pi*eps0))`` in eV for a surface field ``E`` in V/m."""
    return (E_CHARGE_C * enhancement * field_V_per_m / (4.0 * math.pi * EPS0_F_PER_M)) ** 0.5


@dataclass
class Photocathode:
    """Laser-driven emission for ``Injector.photocathode``; ``None`` there keeps the distribution's own charge.

    ``pulse_energy_J`` is the laser energy on the cathode. ``qe`` (electrons per photon) holds at the surface
    field ``qe_field_V_per_m``. With ``work_function_eV`` (or ``material``) set, the quantum efficiency follows
    the Schottky-lowered work function through the Dowell-Schmerge energy factor; otherwise it is constant.
    """
    pulse_energy_J: float
    wavelength_nm: float
    qe: float
    material: str | None = None
    work_function_eV: float | None = None
    fermi_energy_eV: float | None = None
    qe_field_V_per_m: float | None = None
    field_enhancement: float = 1.0

    def __post_init__(self):
        if self.material is not None:
            if self.material not in _MATERIALS:
                raise KeyError(f"unknown material {self.material!r}; known: {sorted(_MATERIALS)}")
            wf, ef = _MATERIALS[self.material]
            self.work_function_eV = wf if self.work_function_eV is None else self.work_function_eV
            self.fermi_energy_eV = ef if self.fermi_energy_eV is None else self.fermi_energy_eV
        if self.fermi_energy_eV is None:
            self.fermi_energy_eV = 7.0
        if not (self.pulse_energy_J > 0.0 and self.wavelength_nm > 0.0 and 0.0 <= self.qe <= 1.0
                and self.field_enhancement > 0.0):
            raise ValueError("need pulse_energy_J > 0, wavelength_nm > 0, 0 <= qe <= 1 and field_enhancement > 0")
        if self.work_function_eV is not None:
            if self.qe_field_V_per_m is None or self.qe_field_V_per_m < 0.0:
                raise ValueError("qe_field_V_per_m (the field qe was quoted at, >= 0) is required with a work function")
            if self._reference_shape() <= 0.0:
                raise ValueError("qe is quoted below the photoemission threshold: photon energy <= effective "
                                 "work function at qe_field_V_per_m")

    @classmethod
    def from_average_power(cls, power_W: float, rep_rate_Hz: float, **kwargs) -> "Photocathode":
        """The pulse energy of a laser with ``power_W`` average power on the cathode at ``rep_rate_Hz``."""
        return cls(pulse_energy_J=power_W / rep_rate_Hz, **kwargs)

    @property
    def photon_energy_eV(self) -> float:
        return _HC_EV_NM / self.wavelength_nm

    @property
    def photons(self) -> float:
        return self.pulse_energy_J / (self.photon_energy_eV * E_CHARGE_C)

    def _shape(self, phi_eff):
        """Dowell-Schmerge energy factor ``(EF+hw)/(2hw) * (1 - sqrt((EF+phi)/(EF+hw)))^2``, zero at threshold."""
        hw, ef = self.photon_energy_eV, self.fermi_energy_eV
        s = ((ef + phi_eff) / (ef + hw)).clamp(0.0, 1.0) ** 0.5
        return (ef + hw) / (2.0 * hw) * (1.0 - s) ** 2

    def _reference_shape(self) -> float:
        phi = self.work_function_eV - schottky_lowering_eV(self.qe_field_V_per_m, self.field_enhancement)
        return float(self._shape(torch.tensor(phi, dtype=torch.float64)))

    def quantum_efficiency(self, field_V_per_m: torch.Tensor) -> torch.Tensor:
        """Electrons per photon at extraction field ``field_V_per_m``; zero where the field does not extract."""
        on = (field_V_per_m > 0.0).to(field_V_per_m.dtype)
        if self.work_function_eV is None:
            return on * self.qe
        phi = self.work_function_eV - schottky_lowering_eV(field_V_per_m.clamp_min(0.0), self.field_enhancement)
        return on * self.qe * self._shape(phi) / self._reference_shape()


class PhotoEmitter:
    """Run state of ``Photocathode`` emission: the cloud's radial bins and the photon charge of each macroparticle.

    The laser pulse is the macroparticle distribution itself (positions and emission times); ``q_nominal`` only
    weights it. Everything is axisymmetric (radial bins), as the cylindrical space-charge solver is.
    """

    def __init__(self, photo: Photocathode, r: torch.Tensor, q_nominal: torch.Tensor):
        dev, f64 = r.device, torch.float64
        r, w = r.to(f64), q_nominal.abs().to(f64)
        self.photo, self.n = photo, r.numel()
        self.q_phot = photo.photons * E_CHARGE_C * w / w.sum()  # C if every photon freed an electron
        nb = max(4, min(_MAX_BINS, self.n // 32))
        edges = torch.quantile(r, torch.linspace(0.0, 1.0, nb + 1, dtype=f64, device=dev))
        edges[0] = 0.0
        self.edges = edges + torch.arange(nb + 1, dtype=f64, device=dev) * 1e-12  # strictly increasing
        e2 = self.edges ** 2
        area = math.pi * (e2[1:] - e2[:-1])
        self.nb = nb
        self.photon_density = torch.zeros(nb, dtype=f64, device=dev).index_add_(0, self._bin(r), self.q_phot) / area
        # equal-area sub-rings of every bin; the ring sum is smoothed at the widest sub-ring gap
        frac = (torch.arange(_SUB_RINGS, dtype=f64, device=dev) + 0.5) / _SUB_RINGS
        self.sub_r = torch.sqrt(e2[:-1, None] + frac[None, :] * (e2[1:] - e2[:-1])[:, None])
        bounds = torch.cat([self.edges[:-1, None], self.sub_r, self.edges[1:, None]], dim=1)
        self.floor = torch.diff(bounds, dim=1).amax(dim=1)
        self.r_mid = torch.sqrt(0.5 * (e2[:-1] + e2[1:]))
        u = (torch.arange(_AZIMUTH, dtype=f64, device=dev) + 0.5) / _AZIMUTH
        self.cos_phi, self.w_phi = torch.cos(math.pi * u ** 3), 3.0 * u ** 2 / _AZIMUTH
        self.ideal_C = torch.zeros((), dtype=f64, device=dev)

    def _bin(self, r: torch.Tensor) -> torch.Tensor:
        return torch.bucketize(r, self.edges[1:-1])

    def screening(self, r: torch.Tensor, height: torch.Tensor, charge: torch.Tensor) -> torch.Tensor:
        """Reduction (V/m) of the surface extraction field by ``charge`` (C, magnitudes) at radii ``r`` and
        ``height`` above the cathode, with its image; evaluated at the bin-centre radii."""
        b = self._bin(r)
        q = torch.zeros(self.nb, dtype=r.dtype, device=r.device).index_add_(0, b, charge)
        qh = torch.zeros_like(q).index_add_(0, b, charge * height)
        h = torch.maximum(qh / q.clamp_min(1e-300), self.floor)
        rt, a = self.r_mid[:, None, None, None], self.sub_r[None, :, :, None]
        d2 = rt * rt + a * a - 2.0 * rt * a * self.cos_phi + (h * h)[None, :, None, None]
        kernel = (h[None, :, None, None] * d2.pow(-1.5) * self.w_phi).sum(-1)  # (target, bin, sub-ring)
        return (kernel * (q / _SUB_RINGS)[None, :, None]).sum((1, 2)) / (2.0 * math.pi * EPS0_F_PER_M)

    def _at(self, values: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        i = torch.searchsorted(self.r_mid, r).clamp(1, self.nb - 1)
        t = ((r - self.r_mid[i - 1]) / (self.r_mid[i] - self.r_mid[i - 1])).clamp(0.0, 1.0)
        return values[i - 1] * (1.0 - t) + values[i] * t

    def emit(self, idx: torch.Tensor, r: torch.Tensor, field_applied: torch.Tensor, cloud) -> tuple:
        """Charge magnitudes (C) macroparticles ``idx`` emit this step, and the surface field (V/m) they see.

        ``field_applied`` is the applied extraction field at their positions; ``cloud`` is ``(r, height, |q|)`` of
        the charge already emitted and still alive. The new layer may at most cancel the surface field that is
        left (Gauss law), and its own mean screening enters the field-dependent quantum efficiency.
        """
        f64 = torch.float64
        r, e_app = r.to(f64), field_applied.to(f64)
        e0 = e_app - self._at(self.screening(*cloud), r) if cloud[0].numel() else e_app
        qp = self.q_phot[idx]
        layer = qp.sum() / self.q_phot.sum()
        c = layer * self.photon_density[self._bin(r)] / (qp.clamp_min(1e-300) * EPS0_F_PER_M)  # V/m per C
        qe = self.photo.quantum_efficiency
        self.ideal_C = self.ideal_C + (qp * qe(e_app)).sum()
        lo, hi = torch.zeros_like(qp), qp * qe(e0)
        for _ in range(_BISECT):  # q = qp*QE(e0 - c*q/2) falls with q: bisect the unique root
            mid = 0.5 * (lo + hi)
            over = mid > qp * qe(e0 - 0.5 * c * mid)
            lo, hi = torch.where(over, lo, mid), torch.where(over, mid, hi)
        return torch.minimum(0.5 * (lo + hi), e0.clamp_min(0.0) / c), e0
