"""Flat-chamber multilayer resistive-wall impedance (IMPEDANCE_PLAN.md
Phase 3), ported directly from IW2D's ``flat.cc``/``flatchamber.cc``
(N. Mounet & E. Metral, "Electromagnetic fields and beam coupling
impedances in a multilayer flat chamber", CERN note, 2010).

**PARKED, not currently usable -- do not wire this into anything.** The
formula translation below is structurally complete and faithful to
``flat.cc`` (checked line by line), but ``_etachi``'s 4x4 matrix inversion
overflows double precision even for unremarkable geometries (e.g. a 13mm
half-gap, stainless-steel-like wall, 10 GHz: the field-matching matrix
already reaches ~exp(2e4)) -- the same class of problem ``round.py`` hit
and fixed for its Bessel functions, but round chamber's fix worked because
its final answer is a *determinant ratio* with a provable algebraic
cancellation (see ``round.py``'s docstring); flat chamber's answer comes
from a *matrix inverse* mixing bounded and unbounded entries in a way that
doesn't reduce to a single scale factor the same way. A real fix exists
(redo the matrix arithmetic in arbitrary precision via ``mpmath``,
mirroring what IW2D's own 160-bit arithmetic does -- straightforward but
slow, since this module already loops over frequency and adaptive
quadrature points in plain Python; or derive a genuinely stable
double-precision reformulation, open-ended numerical-methods work) but
wasn't pursued: ``lattice.apertures`` approximates flat apertures via
Yokoya elliptic form factors applied to the (already validated)
round-chamber result instead (``wakefields/multilayer/yokoya.py``, ported
from IW2D's own official Python library) -- the standard, much cheaper
approach, and accurate enough for what the Apertures feature needs. Revisit
this module only if that approximation turns out to be insufficient for a
real case.

Scope, deliberately narrower than ``flat.cc`` itself:

- **Top-bottom symmetric only** (``flat.cc``'s own default, ``flag_topbotsym=1``):
  one material stack, mirrored top and bottom. Nothing in this repo needs
  independently-specified upper/lower stacks (a flat aperture has one
  coating), and symmetry means the lower stack's field-matching matrix is
  identical to the upper one (no separate computation) and the "constant"
  term ``Zycst`` (only nonzero for an up-down *asymmetric* chamber) is
  identically zero -- both proven directly from ``flatchamber.cc``'s own
  ``alphamn``, which zeroes any ``alpha_mn`` with odd ``m+n`` whenever
  ``flag_topbotsym`` is set, not approximated here.
- **Non-magnetic layers only** (mu=1), same assumption ``round.py`` makes
  and for the same reason: nothing in this repo's materials has ever
  needed permeability.
- **"Maximum order of non-linear terms" = 0** (``flat.cc``'s own default):
  the standard six outputs (Zlong, Zxdip, Zydip, Zxquad, Zyquad, Zycst)
  come from exactly four ``alpha_mn`` integrals (``alpha00``, ``alpha01``,
  ``alpha02``, ``alpha11``) -- confirmed directly from
  ``flatchamber.cc``'s ``order<=0`` branch. The general higher-order
  multipole path (``coefmn``, arbitrary ``alpha_mn``) is a separate,
  unbounded-order feature nothing here calls for; not ported.

Numerically, flat chamber is a different problem from round: the
transverse dependence is Cartesian (exponentials in a Fourier-type
horizontal wavenumber ``kx``), not a Bessel expansion, but ``alpha_mn``
itself is a semi-infinite integral over ``kx`` (via ``u`` with
``kx=(k/gamma)*sinh(u)``) with no closed form -- ``flat.cc`` evaluates it
with GSL's adaptive Gauss-Kronrod quadrature to infinity (``qagiu``); this
port uses ``scipy.integrate.quad``'s equivalent (QUADPACK's own infinite-
range transform), the direct Python counterpart, not a re-derivation.
Each integrand evaluation needs a 4x4 complex boundary-matching matrix
chain (``_field_matching_matrix``, ported from ``multilayerm_flat``) and a
4x4 complex inverse (``_etachi``, ported from ``etachi``) -- cheap per
call, but this means (unlike ``round.py``'s closed-form, frequency-batched
``_alpha_tm``) evaluation is inherently a Python-level loop over frequency
and, within each frequency, an adaptive number of quadrature points -- not
batched, not GPU-vectorized, and (via ``scipy.integrate.quad``) not
autograd-differentiable. This matches where the round-chamber code
actually stands too: per IMPEDANCE_PLAN.md, gradients don't flow through
``RoundChamber``'s plain-float inputs yet either, so this isn't a
regression against a capability anything currently uses.
"""
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
    """Principal-branch complex sqrt, matching IW2D's ``csqrtMP`` exactly
    (including its -0j -> +0j fixup right at the branch cut) -- same fixup
    as ``round.py``'s own ``_csqrt``, reimplemented here in plain
    numpy/complex rather than torch since this module's per-frequency
    quadrature loop (module docstring) evaluates it scalar-at-a-time, far
    too often to pay a torch round-trip on each call."""
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
    """4x4 field-matching matrix for one (top==bottom, by symmetry) stack
    at a single frequency and horizontal wavenumber ``kx`` -- ported from
    ``flat.cc``'s ``multilayerm_flat``. ``eps1_inv`` must be
    ``1/eps1_physical`` (``flat.cc``'s own in-place ``eps1(p)=1/eps1(p)``
    reassignment before this matrix build -- see the module docstring's
    "NB" note in the original source); mu=1 throughout (module docstring).
    """
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
    """Ported from ``flat.cc``'s ``integrand`` (``kx=(k/gamma)*sinh(u)``);
    a NaN real or imaginary part (from an overflowing ``kx`` at large
    ``u``, same as the source) is replaced by zero, real/imag
    independently, matching ``flat.cc`` exactly."""
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
    """Upper bound replacing IW2D's literal semi-infinite ``alphamn``
    integral (module docstring's numerical-conditioning note): IW2D's own
    160-bit arithmetic tolerates evaluating the integrand at arbitrarily
    large ``u`` (GSL's infinite-range quadrature rule samples near-infinite
    ``u`` on its very first panel, before any adaptive refinement) --
    double precision does not, because ``kx=(k/gamma)*sinh(u)`` grows
    exponentially with ``u``, and so does ``kyp~|kx|`` and hence
    ``exp((kyp+kyp1)*b[p])`` (``_field_matching_matrix``'s ``exp3``/``exp4``).
    ``b[-1]`` (the outermost material boundary) is the least forgiving --
    ``kyp``/``kyp1`` are the same order for every boundary at large ``u``,
    so the largest ``b`` overflows first. Truncating where that product
    would first reach ``_MAX_SAFE_EXPONENT`` is negligible, not
    approximate: the true integrand there is suppressed by the reciprocal
    of that same (squared) factor -- order ``exp(-1200)`` or smaller, many
    orders below double precision's own ~1e-16 resolution -- because the
    physical (decaying, not growing) branch is what a correctly-conditioned
    evaluation would keep (the same "only the decaying solution survives"
    principle behind ``round.py``'s own overflow fix, worked out there
    algebraically; here enforced by never evaluating far enough out for the
    growing branch to be sampled in the first place)."""
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
    """All six standard outputs for ``chamber`` -- ``flatchamber.cc``'s
    ``order<=0`` branch (module docstring), drop-in (same physical
    definitions and units) for what IW2D's ``flatchamber.x`` itself
    writes. Top-bottom symmetric, non-magnetic, no differentiability (see
    module docstring) -- unlike ``round.py``'s functions, this loops over
    frequency in Python (each point needs its own adaptive quadrature), so
    it is much slower per point; fine for offline impedance generation,
    not for anything called per-turn.
    """
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
    """Zlong(f) only -- the piece ``lattice.apertures`` actually consumes
    today (its longitudinal-only scope, transverse aperture-summing is
    explicitly future work per APERTURES.md). Thin wrapper over
    ``flat_chamber_impedances`` for callers that don't need the other five."""
    return flat_chamber_impedances(chamber, freq_Hz, gamma).Zlong
