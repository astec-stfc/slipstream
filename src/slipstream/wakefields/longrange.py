"""High-Q resonators accumulate decaying/rotating interbunch voltage phasors with
`lambda=(i-1/(2Q))*omega_r`. See docs/modules/wakefields/longrange.md."""
from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ..serialization import dump_yaml, load_yaml


@dataclass
class Resonator:
    """A single resonant mode (main cavity or HHC), high-Q approximation."""

    R_ohm: float  # shunt impedance R = (R/Q)*Q
    Q: float
    omega_r: float  # angular resonant frequency (rad/s)

    @classmethod
    def from_detuning(
        cls, R_over_Q_ohm: float, Q: float, harmonic_n: int, detuning_Hz: float, h: int, f_rev_Hz: float
    ) -> "Resonator":
        """Compute from detuning; see the module documentation for details."""
        omega_target = 2 * math.pi * harmonic_n * h * f_rev_Hz
        omega_r = omega_target + 2 * math.pi * detuning_Hz
        return cls(R_over_Q_ohm * Q, Q, omega_r)

    @property
    def lam(self) -> complex:
        """Complex natural frequency: rotates at omega_r, decays at omega_r/(2Q)."""
        return (1j - 1.0 / (2.0 * self.Q)) * self.omega_r

    def to_yaml(self, path: str | Path) -> None:
        dump_yaml(self, path)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Resonator":
        return load_yaml(cls, path)


@dataclass
class LongRangeWake:
    """Per-bucket resonator phasor state (Eqs. 4-5), carried turn to turn."""

    resonator: Resonator
    h: int
    Tb_s: float
    is_main_cavity: bool = False
    V_c0: complex = 0j  # V_c,0 for the upcoming turn (= V_c,h of the previous one)
    chunk: int = field(default=256, repr=False)

    @property
    def a(self) -> complex:
        return cmath.exp(self.resonator.lam * self.Tb_s)

    def init_uniform_fill(self, I0_avg_A: float, F: float = 1.0) -> None:
        """Compute init uniform fill; see the module documentation for details."""
        res = self.resonator
        q_bunch = I0_avg_A * self.Tb_s  # average charge per bucket for uniform fill
        Veq = q_bunch * res.omega_r * res.R_ohm / res.Q * F
        self.V_c0 = Veq / (1 - self.a)

    def step(self, Veq_bucket: torch.Tensor) -> torch.Tensor:
        """Compute step; see the module documentation for details."""
        h = Veq_bucket.shape[0]
        device, real_dtype = Veq_bucket.device, torch.float64
        log_a = self.resonator.lam * self.Tb_s  # a = exp(log_a); a^k = exp(k*log_a), avoids complex-tensor pow
        a = cmath.exp(log_a)
        V_c = torch.empty(h, dtype=torch.complex128, device=device)
        carry = self.V_c0
        for start in range(0, h, self.chunk):
            end = min(start + self.chunk, h)
            n = end - start
            k = torch.arange(1, n + 1, dtype=real_dtype, device=device).to(torch.complex128)
            a_pow_k = torch.exp(k * log_a)
            u = Veq_bucket[start:end] / a_pow_k
            s = torch.cumsum(u, dim=0) + carry
            V_c[start:end] = a_pow_k * s
            carry = V_c[end - 1]
        V_b = a * torch.cat([torch.full((1,), self.V_c0, dtype=torch.complex128, device=device), V_c[:-1]])
        self.V_c0 = complex(V_c[-1])
        return V_b


def equivalent_voltage_phasor(
    tau: torch.Tensor, charge_per_particle_C: torch.Tensor | float, resonator: Resonator, bucket_index, h: int
) -> torch.Tensor:
    """Eq. 3: per-bunch equivalent voltage phasor, scattered to a length-h
    (complex) array indexed by rf bucket -- 0 at any bucket with no bunch.
    `tau` is (N, M); `bucket_index` is length M (this bunch's rf bucket).
    """
    lam = resonator.lam
    Veq_bunch = charge_per_particle_C * resonator.omega_r * resonator.R_ohm / resonator.Q * torch.sum(
        torch.exp(-lam * tau.to(torch.complex128)), dim=0
    )
    Veq_bucket = torch.zeros(h, dtype=torch.complex128, device=tau.device)
    Veq_bucket[bucket_index] = Veq_bunch
    return Veq_bucket


def phasor_at_particles(V_b_bunch: torch.Tensor, tau: torch.Tensor, resonator: Resonator) -> torch.Tensor:
    """Eq. 8: the interbunch voltage phasor felt by each particle, given
    its own bunch's V_b (length M, complex) and its own offset tau (N, M)."""
    return V_b_bunch.unsqueeze(0) * torch.exp(resonator.lam * tau.to(torch.complex128))
