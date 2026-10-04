"""Experimental flat-chamber impedance; numerically unreliable and not for production use. See
docs/modules/wakefields/multilayer/flat.md."""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy import integrate

from ..impedance import Impedance
from ..chambers import FlatChamber, layer_fields

C_LIGHT = 299792458.0
MU0 = 1.25663706212e-6
EPS0 = 1.0 / (MU0 * C_LIGHT**2)
Z0 = MU0 * C_LIGHT


def _csqrt(z: complex) -> complex:
    """Compute csqrt; see the module documentation for details."""
    imag = z.imag
    if imag == 0.0 and np.signbit(imag):
        imag = -imag
    return complex(np.sqrt(complex(z.real, imag)))


def _layer_eps(chamber: FlatChamber, omega: np.ndarray) -> np.ndarray:
    """Physical (non-inverted) eps1, shape (n_layers+1, n_freq) complex --
    identical Drude-relaxation formula to ``round.py``'s ``_layer_eps``,
    index 0 = vacuum."""
    n_freq = omega.shape[0]
    n_layers = len(chamber.layers)
    eps1 = np.ones((n_layers + 1, n_freq), dtype=complex)
    for p, layer in enumerate(chamber.layers, start=1):
        rho, tau_ps, _thickness_m, epsb = layer_fields(layer)
        if np.isfinite(rho):
            tau = tau_ps * 1e-12
            eps1[p] = epsb + 1 / (1j * EPS0 * rho * omega * (1 + 1j * omega * tau))
        else:
            eps1[p] = epsb
    return eps1


def _boundary_positions(chamber: FlatChamber) -> np.ndarray:
    """b, shape (n_layers,) float -- b[0] is the half-gap (vacuum/layer-1
    boundary); b[p] is the outer boundary of ``chamber.layers[p-1]``, same
    accumulation as ``round.py``'s ``_boundary_radii``."""
    b = [chamber.half_gap_m]
    for layer in chamber.layers[:-1]:
        b.append(b[-1] + layer_fields(layer)[2])
    return np.asarray(b, dtype=float)


def _field_matching_matrix(
    eps1_ratio: np.ndarray, nu2: np.ndarray, nu2_ratio: np.ndarray, eps1_inv: np.ndarray,
    b: np.ndarray, kx: complex, beta: float,
) -> np.ndarray:
    """4x4 field-matching matrix for one (top==bottom, by symmetry) stack at a single frequency and
    horizontal wavenumber ``kx`` -- ported from ``flat.cc``'s ``multilayerm_flat``."""
    n_layers = len(b)
    kyp = _csqrt(nu2[0] + kx**2)
    m = np.eye(4, dtype=complex)
    fac3 = kx / (2 * beta)

    for p in range(n_layers):
        kyp1 = _csqrt(nu2[p + 1] + kx**2)
        tmp1 = kyp * nu2_ratio[p] / kyp1
        exp1 = np.exp((kyp - kyp1) * b[p])
        exp2 = np.exp(-(kyp + kyp1) * b[p])
        exp3 = 1 / exp2
        exp4 = 1 / exp1

        tmp2 = tmp1 * eps1_ratio[p]
        tmpplus, tmpminus = (1 + tmp2) / 2, (1 - tmp2) / 2
        fac = fac3 * (nu2_ratio[p] - 1) / kyp1
        fac2_eps = fac * eps1_inv[p + 1]

        # mu1=1 everywhere (module docstring) => mu1ratio=1, 1/mu1=1.
        tmpplus_mu, tmpminus_mu = (1 + tmp1) / 2, (1 - tmp1) / 2
        fac2_mu = fac

        mp1p = np.array([
            [tmpplus * exp1, tmpminus * exp2, -fac2_eps * exp1, -fac2_eps * exp2],
            [tmpminus * exp3, tmpplus * exp4, fac2_eps * exp3, fac2_eps * exp4],
            [-fac2_mu * exp1, -fac2_mu * exp2, tmpplus_mu * exp1, tmpminus_mu * exp2],
            [fac2_mu * exp3, fac2_mu * exp4, tmpminus_mu * exp3, tmpplus_mu * exp4],
        ], dtype=complex)

        m = mp1p @ m
        kyp = kyp1

    return m


def _etachi(
    eps1_ratio: np.ndarray, nu2: np.ndarray, nu2_ratio: np.ndarray, eps1_inv: np.ndarray,
    b: np.ndarray, kx: complex, beta: float,
) -> tuple[complex, complex, complex, complex]:
    """(chi1, chi2, eta1, eta2) at a given ``kx`` -- ported from
    ``flat.cc``'s ``etachi``. Top-bottom symmetric (module docstring): the
    lower stack's matrix equals the upper stack's, so it's computed once."""
    mat = _field_matching_matrix(eps1_ratio, nu2, nu2_ratio, eps1_inv, b, kx, beta)
    matprime = mat

    pcal = np.array([mat[0, :], mat[2, :], matprime[1, :], matprime[3, :]], dtype=complex)
    pcal_inv = np.linalg.inv(pcal)

    chi1 = pcal_inv[0, 0] * mat[0, 1] + pcal_inv[0, 1] * mat[2, 1]
    chi2 = pcal_inv[1, 0] * mat[0, 1] + pcal_inv[1, 1] * mat[2, 1]
    eta1 = pcal_inv[0, 2] * matprime[1, 0] + pcal_inv[0, 3] * matprime[3, 0]
    eta2 = pcal_inv[1, 2] * matprime[1, 0] + pcal_inv[1, 3] * matprime[3, 0]
    return chi1, chi2, eta1, eta2


def _integrand(
    m: int, n: int,
    eps1_ratio: np.ndarray, nu2: np.ndarray, nu2_ratio: np.ndarray, eps1_inv: np.ndarray,
    b: np.ndarray, beta: float, k_over_gamma: float, u: float,
) -> complex:
    """Compute integrand; see the module documentation for details."""
    kx = k_over_gamma * np.sinh(u)
    chi1, chi2, eta1, eta2 = _etachi(eps1_ratio, nu2, nu2_ratio, eps1_inv, b, kx, beta)
    m1powm = 1 if m % 2 == 0 else -1
    m1pown = 1 if n % 2 == 0 else -1
    m1powmn = m1powm * m1pown
    inte = np.cosh(m * u) * np.cosh(n * u) * (chi1 + m1powm * eta1 + m1pown * chi2 + m1powmn * eta2)
    re = inte.real if np.isfinite(inte.real) else 0.0
    im = inte.imag if np.isfinite(inte.imag) else 0.0
    return complex(re, im)


_MAX_SAFE_EXPONENT = 600.0  # double overflows above ~709; this leaves comfortable margin


def _u_max(b: np.ndarray, k_over_gamma: float) -> float:
    """Compute u max; see the module documentation for details."""
    kx_max = _MAX_SAFE_EXPONENT / b[-1]
    return float(np.arcsinh(kx_max / k_over_gamma))


def _alphamn(
    m: int, n: int,
    eps1_ratio: np.ndarray, nu2: np.ndarray, nu2_ratio: np.ndarray, eps1_inv: np.ndarray,
    b: np.ndarray, beta: float, k_over_gamma: float,
) -> complex:
    """alpha_mn, ported from ``flat.cc``'s ``alphamn`` -- top-bottom
    symmetric (module docstring), so any odd ``m+n`` is identically zero
    without integrating (``flat.cc``'s own ``flag_topbotsym`` short-circuit)."""
    if (m + n) % 2 == 1:
        return 0j

    u_max = _u_max(b, k_over_gamma)

    def real_part(u: float) -> float:
        return _integrand(m, n, eps1_ratio, nu2, nu2_ratio, eps1_inv, b, beta, k_over_gamma, u).real

    def imag_part(u: float) -> float:
        return _integrand(m, n, eps1_ratio, nu2, nu2_ratio, eps1_inv, b, beta, k_over_gamma, u).imag

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=integrate.IntegrationWarning)
        re, _ = integrate.quad(real_part, 0.0, u_max, limit=200)
        im, _ = integrate.quad(imag_part, 0.0, u_max, limit=200)
    return complex(re, im)


@dataclass
class FlatImpedances:
    """The six standard flat-chamber outputs ``flatchamber.cc`` writes
    (Zlong in Ohm, the rest in Ohm/m), all on the same ``freq_Hz`` grid."""

    Zlong: Impedance
    Zxdip: Impedance
    Zydip: Impedance
    Zxquad: Impedance
    Zyquad: Impedance
    Zycst: Impedance


def flat_chamber_impedances(chamber: FlatChamber, freq_Hz: np.ndarray, gamma: float) -> FlatImpedances:
    """All six standard outputs for ``chamber`` -- ``flatchamber.cc``'s ``order<=0`` branch (module
    docstring), drop-in (same physical definitions and units) for what IW2D's ``flatchamber.x``
    itself writes."""
    beta = float(np.sqrt(1 - 1 / gamma**2))
    freq = np.asarray(freq_Hz, dtype=float)
    omega = 2 * np.pi * freq
    k = omega / (beta * C_LIGHT)
    k_over_gamma = k / gamma

    eps1 = _layer_eps(chamber, omega)  # (n_layers+1, n_freq), physical (non-inverted)
    b = _boundary_positions(chamber)
    nu2 = k**2 * (1 - beta**2 * eps1)  # (n_layers+1, n_freq)
    eps1_ratio = eps1[:-1] / eps1[1:]  # (n_layers, n_freq)
    nu2_ratio = nu2[1:] / nu2[:-1]
    eps1_inv = 1.0 / eps1

    n_freq = freq.shape[0]
    Zlong = np.empty(n_freq, dtype=complex)
    Zxdip = np.empty(n_freq, dtype=complex)
    Zydip = np.empty(n_freq, dtype=complex)
    Zyquad = np.empty(n_freq, dtype=complex)
    Zycst = np.zeros(n_freq, dtype=complex)  # top-bottom symmetric -> identically 0 (module docstring)

    for i in range(n_freq):
        args = (eps1_ratio[:, i], nu2[:, i], nu2_ratio[:, i], eps1_inv[:, i], b, beta, k_over_gamma[i])
        alpha00 = _alphamn(0, 0, *args)
        alpha02 = _alphamn(0, 2, *args)
        alpha11 = _alphamn(1, 1, *args)

        cst = 1j * chamber.length_m * (k[i] * Z0 / (beta * gamma**2 * 2 * np.pi))
        cst_dip = cst * (k[i] / (2 * gamma**2))
        Zlong[i] = cst * alpha00
        Zxdip[i] = cst_dip * (alpha02 - alpha00)
        Zydip[i] = 2.0 * cst_dip * alpha11
        Zyquad[i] = cst_dip * (alpha02 + alpha00)

    Zxquad = -Zxdip

    return FlatImpedances(
        Zlong=Impedance(freq_Hz, Zlong.real, Zlong.imag),
        Zxdip=Impedance(freq_Hz, Zxdip.real, Zxdip.imag),
        Zydip=Impedance(freq_Hz, Zydip.real, Zydip.imag),
        Zxquad=Impedance(freq_Hz, Zxquad.real, Zxquad.imag),
        Zyquad=Impedance(freq_Hz, Zyquad.real, Zyquad.imag),
        Zycst=Impedance(freq_Hz, Zycst.real, Zycst.imag),
    )


def flat_chamber_impedance(chamber: FlatChamber, freq_Hz: np.ndarray, gamma: float) -> Impedance:
    """Experimental flat-chamber solver; numerically unreliable and not for production use."""
    return flat_chamber_impedances(chamber, freq_Hz, gamma).Zlong
