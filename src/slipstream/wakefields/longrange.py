"""Long-range (interbunch) wake: resonant cavity modes (main cavity and/or
higher-harmonic cavities) represented as a decaying, rotating voltage
phasor rather than a stored wake history (He & Bai, PRAB 24, 104401 (2021),
Eqs. 1-9, high-Q approximation). Consumed by the pytorch multi-bunch
tracker (``tracking.pystable.multibunch``); elegant driving isn't
implemented in this repo's elegant module.

Physics, in the paper's own eq. numbers:

- Eq 1/3: a bunch leaves behind an "equivalent voltage phasor" at its own
  bucket, `Veq_i = (q_i*wr*R/Q) * sum_k exp(-lam*tau_k)`, where
  `lam = (1j - 1/(2Q))*wr` is the resonator's complex natural frequency
  (rotation at wr, decay at rate wr/(2Q)). `tau_k` is the particle's own
  offset from its bucket's reference phase.
- Eq 4/5: that phasor is carried bucket-to-bucket around the ring by a
  *fixed* complex factor `a = exp(lam*Tb)`, `Tb = T0/h`:
  `V_b,i = a*V_c,i-1`, `V_c,i = V_b,i + Veq,i`, wrapping `V_c,0 := V_c,h`
  turn to turn. This is the one genuinely sequential piece (bucket i needs
  i-1) -- done here via a chunked weighted-cumsum closed form rather than
  an h-long Python loop, see `LongRangeWake.step`.
- Eq 8: the phasor felt by a particle with offset tau_k in bucket i is
  `V_b,i * exp(lam*tau_k)` -- note the *same* lam as Eq 3, opposite sign in
  the exponent (Eq 3 looks backward from bucket-tau=0 across the source
  bunch; Eq 8 looks forward from bucket-tau=0 to the test particle).
"""
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
        """Build from Table-style parameters (Eq. 7's inverse): the cavity
        targets the harmonic_n'th multiple of the rf frequency
        (harmonic_n=1 for the main cavity, 3 for a third-harmonic HHC), and
        is detuned from it by detuning_Hz."""
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
    """Per-bucket resonator phasor state (Eqs. 4-5), carried turn to turn.

    `is_main_cavity=True` marks the resonator whose generator voltage is
    adjusted by Eq. 9's feedback each turn (only one should be marked --
    HHCs are passive, driven purely by the beam).
    """

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
        """Seed V_c,0 assuming a uniform fill pattern (every bucket filled
        with an identical bunch), so the recursion starts at its steady
        state instead of needing several (possibly many, for a high-Q HHC)
        turns of transient to reach it.

        The paper gives an approximate formula for this (Eq. 6, via the
        detuning angle of Eq. 7). Used here instead is the *exact* closed
        form for the same steady state, derived directly from Eqs. 4-5's
        own recursion plus the periodicity condition V_c,0 = V_c,h: for a
        uniform fill, Veq,i = Veq is the same every bucket, so
        V_c,i = a^i*V_c,0 + Veq*(a^i-1)/(a-1); requiring V_c,h = V_c,0
        gives V_c,0 = Veq/(1-a). Cross-checked against Eq. 6/7 in the
        tests (they agree, as they must).
        """
        res = self.resonator
        q_bunch = I0_avg_A * self.Tb_s  # average charge per bucket for uniform fill
        Veq = q_bunch * res.omega_r * res.R_ohm / res.Q * F
        self.V_c0 = Veq / (1 - self.a)

    def step(self, Veq_bucket: torch.Tensor) -> torch.Tensor:
        """Advance one turn: solve V_c,i = a*V_c,i-1 + Veq,i for i=1..h
        given self.V_c0, update self.V_c0 to V_c,h, and return V_b (length
        h, complex) = a*V_c,i-1 for i=1..h -- the phasor felt by bucket i's
        own bunch this turn (Eq. 4).

        Closed form: with u_i = a^-i*Veq,i (u_0 := V_c,0), V_c,i = a^i *
        cumsum(u)_i. Done in fixed-size chunks (default 256 buckets),
        carrying V_c across chunk boundaries, because a^-i grows without
        bound over h~1e3 buckets for a high-Q resonator (|a|<1) -- a single
        h-long cumsum would overflow long before reaching realistic h.
        """
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
