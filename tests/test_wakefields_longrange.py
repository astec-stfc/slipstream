"""Tests for slipstream.wakefields.longrange (resonator phasor formalism,
Eqs. 3-8 of He & Bai, PRAB 24, 104401 (2021))."""
from __future__ import annotations

import cmath
import math

import pytest
import torch

from slipstream.wakefields.longrange import (
    LongRangeWake,
    Resonator,
    equivalent_voltage_phasor,
    phasor_at_particles,
)


def test_resonator_recursion_matches_closed_form_uniform_fill():
    h = 20
    res = Resonator(R_ohm=1000.0, Q=50.0, omega_r=2 * math.pi * 500e6)
    wake = LongRangeWake(res, h, Tb_s=2e-9)
    Veq = 3.0 + 2.0j
    Veq_bucket = torch.full((h,), Veq, dtype=torch.complex128)

    for _ in range(200):
        wake.step(Veq_bucket)

    a = wake.a
    closed_form_V_c0 = Veq / (1 - a)
    assert wake.V_c0 == pytest.approx(closed_form_V_c0, rel=1e-9)


def test_resonator_recursion_chunking_prevents_overflow():
    """A deliberately extreme (very low Q) case: an unchunked (chunk=h)
    cumsum overflows to NaN; the default chunking doesn't."""
    h = 800
    res = Resonator(R_ohm=1000.0, Q=3.0, omega_r=2 * math.pi * 500e6)
    Veq_bucket = torch.full((h,), 3.0 + 2.0j, dtype=torch.complex128)

    chunked = LongRangeWake(res, h, Tb_s=2e-9, chunk=64)
    unchunked = LongRangeWake(res, h, Tb_s=2e-9, chunk=h)
    for _ in range(5):
        V_b_chunked = chunked.step(Veq_bucket)
        V_b_unchunked = unchunked.step(Veq_bucket)

    assert torch.isfinite(V_b_chunked.real).all() and torch.isfinite(V_b_chunked.imag).all()
    assert not torch.isfinite(V_b_unchunked.real).all()


def test_uniform_fill_closed_form_matches_paper_eq6():
    """Cross-check the exact closed form (used by init_uniform_fill)
    against the paper's own approximate Eq. 6/7 formula for the same
    steady state -- they should agree closely for a realistic, high-Q,
    finely-bucketed case (they're two derivations of the same limit)."""
    h = 800
    Tb = 2e-9
    I0 = 0.35  # A
    F = 1.0
    n_harmonic, f_rev = 1, 1 / (h * Tb)
    res = Resonator.from_detuning(R_over_Q_ohm=90.0, Q=4.035e4, harmonic_n=n_harmonic, detuning_Hz=-16e3, h=h, f_rev_Hz=f_rev)
    wake = LongRangeWake(res, h, Tb)
    wake.init_uniform_fill(I0, F)

    # Eq. 7: tan(psi) = 2Q(wr - n h w0)/(n h w0)
    omega0 = 2 * math.pi * f_rev
    psi = math.atan2(2 * res.Q * (res.omega_r - n_harmonic * h * omega0), n_harmonic * h * omega0)
    eq6_V_c0 = 2 * F * I0 * res.R_ohm * math.cos(psi) * cmath.exp(1j * psi)

    rel_diff = abs(wake.V_c0 - eq6_V_c0) / abs(eq6_V_c0)
    assert rel_diff < 0.05, f"exact closed form vs paper Eq. 6 differ by {rel_diff:.1%}"


def _naive_equivalent_voltage_phasor(tau, charge_per_particle, resonator, bucket_index, h):
    lam = resonator.lam
    Veq_bucket = [0j] * h
    N, M = tau.shape
    for m in range(M):
        total = 0j
        for k in range(N):
            total += cmath.exp(-lam * tau[k, m].item())
        Veq_bucket[bucket_index[m]] = charge_per_particle * resonator.omega_r * resonator.R_ohm / resonator.Q * total
    return torch.tensor(Veq_bucket, dtype=torch.complex128)


def test_equivalent_voltage_phasor_matches_naive_reference():
    torch.manual_seed(0)
    N, M, h = 5, 3, 10
    tau = (torch.rand(N, M, dtype=torch.float64) - 0.5) * 1e-11
    bucket_index = [0, 4, 7]
    res = Resonator(R_ohm=500.0, Q=1000.0, omega_r=2 * math.pi * 1e9)

    got = equivalent_voltage_phasor(tau, 1e-12, res, bucket_index, h)
    want = _naive_equivalent_voltage_phasor(tau, 1e-12, res, bucket_index, h)
    assert torch.allclose(got, want, rtol=1e-10)


def _naive_phasor_at_particles(V_b_bunch, tau, resonator):
    lam = resonator.lam
    N, M = tau.shape
    out = torch.zeros(N, M, dtype=torch.complex128)
    for m in range(M):
        for k in range(N):
            out[k, m] = V_b_bunch[m] * cmath.exp(lam * tau[k, m].item())
    return out


def test_phasor_at_particles_matches_naive_reference():
    torch.manual_seed(1)
    N, M = 4, 3
    tau = (torch.rand(N, M, dtype=torch.float64) - 0.5) * 1e-11
    V_b_bunch = torch.tensor([1.0 + 2.0j, -3.0 + 0.5j, 0.2 - 1.5j], dtype=torch.complex128)
    res = Resonator(R_ohm=500.0, Q=1000.0, omega_r=2 * math.pi * 1e9)

    got = phasor_at_particles(V_b_bunch, tau, res)
    want = _naive_phasor_at_particles(V_b_bunch, tau, res)
    assert torch.allclose(got, want, rtol=1e-10)


def test_resonator_yaml_round_trip(tmp_path):
    res = Resonator(R_ohm=3631500.0, Q=40350.0, omega_r=3139318747.883174)
    path = tmp_path / "resonator.yaml"
    res.to_yaml(path)
    loaded = Resonator.from_yaml(path)
    assert loaded == res

