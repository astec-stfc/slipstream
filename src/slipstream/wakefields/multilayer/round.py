"""Round-chamber multilayer resistive-wall impedance (IMPEDANCE_PLAN.md
Phases 1-2), ported directly from IW2D's ``round.cc``/``roundchamber.cc``
(N. Mounet & E. Metral, "Electromagnetic fields created by a macroparticle
in an infinitely long and axisymmetric multilayer beam pipe",
CERN-BE-2009-039, 2009), batched over the frequency array as one set of
vectorized tensor ops rather than IW2D's own per-frequency C++ loop.

Only non-magnetic layers (mu=1) are supported; each layer is Drude-
relaxation resistivity on top of a real background dielectric constant
``epsb`` (optional 4th layer-tuple element, default 1 -- added for the
Stories/DielectricNEG lossy-dielectric study, validated there against
live IW2D). A magnetic layer would still need ``mu1`` extended to read
``chi``/``fmu``, not done since nothing in this repo needs it.

Numerical conditioning: a real boundary-matching matrix chain can span
Bessel arguments from ~1e-5 (vacuum side) to ~9000 (conductor side),
overflowing raw ``I_m``/``K_m`` and producing ``inf*0=nan`` once
multiplied together. ``round.cc`` survives this via 160-bit arithmetic;
this port instead carries every 4x4 layer-matching matrix in a
holomorphic-safe factored form (``value = norm * exp(row_exp[i] +
col_exp[j])``, built from ``modified_bessel_i_hat``/
``modified_bessel_k_hat``) and only exponentiates the *difference* of two
``col_exp`` entries at the very end, in ``_alpha_tm`` -- provably tied to
the innermost (vacuum-side) boundary alone regardless of layer count. See
``plans/NOTES.md``'s "holomorphic-safe factored-matrix representation"
entry for the full derivation.
"""
from __future__ import annotations

import warnings

import numpy as np
import torch

from ..impedance import Impedance
from ..chambers import RoundChamber, layer_fields
from .bessel import bessel_i_ratio, bessel_k_ratio, modified_bessel_i_hat, modified_bessel_k_hat

C_LIGHT = 299792458.0
MU0 = 1.25663706212e-6
EPS0 = 1.0 / (MU0 * C_LIGHT**2)
Z0 = MU0 * C_LIGHT


def _csqrt(z: torch.Tensor) -> torch.Tensor:
    """Principal-branch complex sqrt, matching IW2D's ``csqrtMP`` exactly
    (including its -0j -> +0j fixup right at the branch cut) -- guarantees
    Re(result)>=0, which everything else in this module relies on."""
    imag = z.imag
    imag = torch.where((imag == 0) & torch.signbit(imag), -imag, imag)
    return torch.sqrt(torch.complex(z.real, imag))


def _layer_eps(chamber: RoundChamber, omega: torch.Tensor) -> torch.Tensor:
    """eps1, shape (n_layers+1, n_freq) complex128 -- index 0 is vacuum
    (eps=1); index p is ``chamber.layers[p-1]``'s Drude-relaxation
    permittivity on top of its background dielectric constant ``epsb``
    (``roundchamber.cc``'s ``eps1(p) = epsb + 1/(j*eps0*rho*omega*
    (1+j*omega*tau))``; ``epsb`` is the layer tuple's optional 4th
    element, default 1 -- see ``iw2d.layer_fields``)."""
    n_freq = omega.shape[0]
    n_layers = len(chamber.layers)
    eps1 = torch.ones(n_layers + 1, n_freq, dtype=torch.complex128)
    for p, layer in enumerate(chamber.layers, start=1):
        rho, tau_ps, _thickness_m, epsb = layer_fields(layer)
        if np.isfinite(rho):
            tau = tau_ps * 1e-12
            eps1[p] = epsb + 1 / (1j * EPS0 * rho * omega * (1 + 1j * omega * tau))
        else:
            eps1[p] = epsb
    return eps1


def _boundary_radii(chamber: RoundChamber) -> torch.Tensor:
    """b, shape (n_layers,) float64 -- b[0] is the beam-pipe inner radius
    (vacuum/layer-1 boundary); b[p] is the outer boundary of
    ``chamber.layers[p-1]`` (= inner boundary of ``chamber.layers[p]``).
    The last layer's own (semi-infinite) outer boundary is never
    referenced, matching ``roundchamber.cc``."""
    b = [chamber.radius_m]
    for layer in chamber.layers[:-1]:
        b.append(b[-1] + layer_fields(layer)[2])
    return torch.tensor(b, dtype=torch.float64)


def _bessel_ratios(m: int, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(I-ratio, K-ratio) with ``round.cc``'s order correction folded in --
    ``Imppratio``/``Kmppratio``. m=0 uses the order-1 ratio directly (no
    -m/x term, since m=0); m>0 uses the general order-m ratio minus m/x."""
    if m == 0:
        i_ratio = 1.0 / bessel_i_ratio(1, x)  # I_1(x)/I_0(x)
        k_ratio = -1.0 / bessel_k_ratio(1, x)  # -K_1(x)/K_0(x)
    else:
        i_ratio = bessel_i_ratio(m, x) - m / x
        k_ratio = -bessel_k_ratio(m, x) - m / x
    return i_ratio, k_ratio


def _boundary_matrix_norm(
    m: int,
    Ihat_pp: torch.Tensor, Khat_pp: torch.Tensor, Ihat_p1p: torch.Tensor, Khat_p1p: torch.Tensor,
    i_ratio_p: torch.Tensor, k_ratio_p: torch.Tensor, i_ratio_p1: torch.Tensor, k_ratio_p1: torch.Tensor,
    eps_over_nu: torch.Tensor, eps_over_nu_next: torch.Tensor,
    mu_over_nu: torch.Tensor, mu_over_nu_next: torch.Tensor,
    eps1_next: torch.Tensor, nu2: torch.Tensor, nu2_next: torch.Tensor, beta: float, xp1p: torch.Tensor,
) -> torch.Tensor:
    """The O(1) "norm" part of one boundary's 4x4 field-matching matrix --
    same formula as round.cc's mp1p, with every I_m(x)*K_m(y)-type raw
    product replaced by the corresponding product of O(1) hat-values
    (module docstring)."""
    # The four products are reused throughout both polarisation blocks.
    # Keep them named after their Bessel factors to make the matrix below
    # easier to compare with the corresponding analytic expression.
    products = {
        "II": Ihat_pp * Ihat_p1p,
        "IK": Ihat_pp * Khat_p1p,
        "KI": Khat_pp * Ihat_p1p,
        "KK": Khat_pp * Khat_p1p,
    }
    II, IK, KI, KK = (products[name] for name in ("II", "IK", "KI", "KK"))

    tmp1 = -xp1p / eps_over_nu_next
    p11 = tmp1 * IK * (eps_over_nu_next * k_ratio_p1 - eps_over_nu * i_ratio_p)
    p12 = tmp1 * KK * (eps_over_nu_next * k_ratio_p1 - eps_over_nu * k_ratio_p)
    p21 = tmp1 * II * (-eps_over_nu_next * i_ratio_p1 + eps_over_nu * i_ratio_p)
    p22 = tmp1 * KI * (-eps_over_nu_next * i_ratio_p1 + eps_over_nu * k_ratio_p)

    if m > 0:
        tmp1q = -(nu2_next / nu2 - 1) * m / (beta * eps1_next)
        q13, q14, q23, q24 = -tmp1q * IK, -tmp1q * KK, tmp1q * II, tmp1q * KI
    else:
        z = torch.zeros_like(p11)
        q13 = q14 = q23 = q24 = z

    tmp1r = -xp1p / mu_over_nu_next
    r33 = tmp1r * IK * (mu_over_nu_next * k_ratio_p1 - mu_over_nu * i_ratio_p)
    r34 = tmp1r * KK * (mu_over_nu_next * k_ratio_p1 - mu_over_nu * k_ratio_p)
    r43 = tmp1r * II * (-mu_over_nu_next * i_ratio_p1 + mu_over_nu * i_ratio_p)
    r44 = tmp1r * KI * (-mu_over_nu_next * i_ratio_p1 + mu_over_nu * k_ratio_p)

    if m > 0:
        scale = eps1_next
        q31, q32, q41, q42 = scale * q13, scale * q14, scale * q23, scale * q24
    else:
        q31 = q32 = q41 = q42 = z

    rows = [
        [p11, p12, q13, q14],
        [p21, p22, q23, q24],
        [q31, q32, r33, r34],
        [q41, q42, r43, r44],
    ]
    return torch.stack([torch.stack(row, dim=-1) for row in rows], dim=-2)


def _scaled_matmul(
    A_norm: torch.Tensor, A_row: torch.Tensor, A_col: torch.Tensor,
    B_norm: torch.Tensor, B_row: torch.Tensor, B_col: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(A@B) in the (norm, row_exp, col_exp) factored representation
    (value[i,j] = norm[i,j]*exp(row_exp[i]+col_exp[j])) -- safe even when
    A_col+B_row would individually overflow if exponentiated directly, by
    factoring out the per-frequency max real part before exponentiating
    (module docstring)."""
    # w[...,k] = exp(A_col[...,k] + B_row[...,k]), shifted by its own max
    # real part per frequency so the shifted exponent's real part is <=0.
    exponent = A_col + B_row  # (n_freq, 4)
    wmax = exponent.real.amax(dim=-1, keepdim=True)  # (n_freq, 1), real
    w = torch.exp(exponent - wmax)  # (n_freq, 4), |w| <= 1

    inner = torch.einsum("fik,fkj,fk->fij", A_norm, B_norm, w)
    return inner, A_row + wmax, B_col


def _alpha_tm(m: int, eps1: torch.Tensor, b: torch.Tensor, k: torch.Tensor, beta: float) -> torch.Tensor:
    """alphaTM, batched over frequency -- round.cc's ``alphaTM``, built
    from the factored-matrix chain (module docstring)."""
    n_freq = k.shape[0]
    n_boundaries = eps1.shape[0] - 1

    mat_norm = torch.eye(4, dtype=torch.complex128).expand(n_freq, 4, 4).clone()
    mat_row = torch.zeros(n_freq, 4, dtype=torch.complex128)
    mat_col = torch.zeros(n_freq, 4, dtype=torch.complex128)

    nu2 = k**2 * (1 - beta**2 * eps1[0])
    nu = _csqrt(nu2)
    eps_over_nu = eps1[0] / nu

    for p in range(n_boundaries):
        nu2_next = k**2 * (1 - beta**2 * eps1[p + 1])
        nu_next = _csqrt(nu2_next)
        eps_over_nu_next = eps1[p + 1] / nu_next
        mu_over_nu = 1.0 / nu
        mu_over_nu_next = 1.0 / nu_next

        xpp = nu * b[p]
        xp1p = nu_next * b[p]

        i_ratio_p, k_ratio_p = _bessel_ratios(m, xpp)
        i_ratio_p1, k_ratio_p1 = _bessel_ratios(m, xp1p)
        Ihat_pp, Khat_pp = modified_bessel_i_hat(m, xpp), modified_bessel_k_hat(m, xpp)
        Ihat_p1p, Khat_p1p = modified_bessel_i_hat(m, xp1p), modified_bessel_k_hat(m, xp1p)

        mp1p_norm = _boundary_matrix_norm(
            m, Ihat_pp, Khat_pp, Ihat_p1p, Khat_p1p,
            i_ratio_p, k_ratio_p, i_ratio_p1, k_ratio_p1,
            eps_over_nu, eps_over_nu_next, mu_over_nu, mu_over_nu_next,
            eps1[p + 1], nu2, nu2_next, beta, xp1p,
        )
        row_sign = torch.tensor([-1.0, 1.0, -1.0, 1.0], dtype=torch.complex128)
        col_sign = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.complex128)
        mp1p_row = xp1p.unsqueeze(-1) * row_sign
        mp1p_col = xpp.unsqueeze(-1) * col_sign

        mat_norm, mat_row, mat_col = _scaled_matmul(mp1p_norm, mp1p_row, mp1p_col, mat_norm, mat_row, mat_col)

        nu2, nu, eps_over_nu = nu2_next, nu_next, eps_over_nu_next

    num = mat_norm[:, 0, 1] * mat_norm[:, 2, 2] - mat_norm[:, 2, 1] * mat_norm[:, 0, 2]
    den = mat_norm[:, 0, 0] * mat_norm[:, 2, 2] - mat_norm[:, 0, 2] * mat_norm[:, 2, 0]
    correction = torch.exp(mat_col[:, 1] - mat_col[:, 0])
    return correction * num / den


# Measured (not assumed) bound on the live-IW2D-validated test case, rounded down for margin --
# see docs/wakefields.md's check_conditioning note.
VALIDATED_MAX_ARGUMENT = 7e5


def estimate_max_bessel_argument(chamber: RoundChamber, freq_Hz: np.ndarray, gamma: float) -> float:
    """Max |nu_layer * boundary_radius| this geometry/frequency range would
    evaluate Bessel functions at, across every layer boundary -- cheap
    (just the layer wavenumbers, no Bessel calls or matrix chain needed),
    a direct proxy for how far outside ``VALIDATED_MAX_ARGUMENT`` a new
    geometry pushes."""
    beta = float(np.sqrt(1 - 1 / gamma**2))
    freq = torch.as_tensor(np.asarray(freq_Hz, dtype=np.float64).copy())
    omega = 2 * np.pi * freq
    k = omega / (beta * C_LIGHT)
    eps1 = _layer_eps(chamber, omega)
    b = _boundary_radii(chamber)

    nu = _csqrt(k**2 * (1 - beta**2 * eps1[0]))
    max_x = 0.0
    for p in range(len(b)):
        nu_next = _csqrt(k**2 * (1 - beta**2 * eps1[p + 1]))
        max_x = max(max_x, (nu * b[p]).abs().max().item(), (nu_next * b[p]).abs().max().item())
        nu = nu_next
    return max_x


def check_conditioning(chamber: RoundChamber, freq_Hz: np.ndarray, gamma: float, warn_factor: float = 1.0) -> None:
    """Warns (does not raise or block) if this geometry/frequency range
    pushes ``estimate_max_bessel_argument`` beyond what Phase 1-2's live
    IW2D cross-check actually covered. Not a proof of correctness beyond
    the validated range -- a concrete, actionable "you're outside the
    tested envelope, consider a live IW2D cross-check for this specific
    case before trusting it" signal, rather than silently returning a
    number either way."""
    max_arg = estimate_max_bessel_argument(chamber, freq_Hz, gamma)
    if max_arg > warn_factor * VALIDATED_MAX_ARGUMENT:
        warnings.warn(
            f"round_chamber_impedance: Bessel argument reaches |x|~{max_arg:.2e} for "
            f"this geometry/frequency range, beyond the ~{VALIDATED_MAX_ARGUMENT:.0e} "
            "IMPEDANCE_PLAN.md's Phase 1-2 validation against live IW2D actually "
            "covered. Consider a live IW2D cross-check for this specific case "
            "before trusting it (see slipstream/wakefields/iw2d.py).",
            RuntimeWarning, stacklevel=2,
        )


def _setup(
    chamber: RoundChamber, freq_Hz: np.ndarray, gamma: float
) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(beta, k, eps1, b) shared by every ``alphaTM``-based output below."""
    beta = float(np.sqrt(1 - 1 / gamma**2))
    freq = torch.as_tensor(np.asarray(freq_Hz, dtype=np.float64).copy())
    omega = 2 * np.pi * freq
    k = omega / (beta * C_LIGHT)
    eps1 = _layer_eps(chamber, omega)
    b = _boundary_radii(chamber)
    return beta, k, eps1, b


def round_chamber_impedance(
    chamber: RoundChamber, freq_Hz: np.ndarray, gamma: float, check: bool = True
) -> Impedance:
    """Zlong(f) for ``chamber`` -- differentiable (w.r.t. any of
    ``chamber``'s layer parameters that were passed in as tensors) drop-in
    replacement for ``wakefields.iw2d.run_iw2d_round``'s longitudinal
    output, same ``Impedance`` contract. See ``round_chamber_dipole_impedance``
    for the transverse (m=1) counterpart.

    ``check=True`` (default) runs ``check_conditioning`` first -- cheap
    (no Bessel evaluation), and warns rather than raising, so it never
    blocks a run, just flags an untested regime.
    """
    if check:
        check_conditioning(chamber, freq_Hz, gamma)

    beta, k, eps1, b = _setup(chamber, freq_Hz, gamma)

    alpha_tm0 = _alpha_tm(0, eps1, b, k, beta)
    cst = 1j * chamber.length_m * (k * Z0 / (beta * gamma**2 * 2 * np.pi))
    Zlong = cst * alpha_tm0

    return Impedance(freq_Hz, Zlong.real.numpy(), Zlong.imag.numpy())


def round_chamber_dipole_impedance(
    chamber: RoundChamber, freq_Hz: np.ndarray, gamma: float, check: bool = True
) -> Impedance:
    """Zxdip(f) = Zydip(f) for ``chamber`` -- the m=1 transverse-dipole
    mode (``roundchamber.cc``'s ``alphaTM1`` branch: same ``alphaTM``
    field-matching chain as ``round_chamber_impedance``'s Zlong, called
    with m=1 instead of m=0, times the extra ``k/(2*gamma^2)`` dipole
    prefactor ``roundchamber.cc`` applies on top of the longitudinal
    ``cst``). Round (axisymmetric) chambers have Yokoya factors
    xdip=ydip=1, so a single result serves both planes -- same
    ``Impedance`` contract as ``round_chamber_impedance``, but in Ohm/m
    (transverse), not Ohm; drop-in replacement for
    ``wakefields.iw2d.total_cached_component``'s cached Zxdip/Zydip
    output.

    ``check=True`` (default) runs ``check_conditioning`` first, same as
    ``round_chamber_impedance`` -- the conditioning bound was derived from
    the m=0 case, but the same boundary-matrix chain (only the Bessel
    order and an m>0 TM-TE coupling term differ) is expected to share it.
    """
    if check:
        check_conditioning(chamber, freq_Hz, gamma)

    beta, k, eps1, b = _setup(chamber, freq_Hz, gamma)

    alpha_tm1 = _alpha_tm(1, eps1, b, k, beta)
    cst_long = 1j * chamber.length_m * (k * Z0 / (beta * gamma**2 * 2 * np.pi))
    Zdip = cst_long * (k / (2 * gamma**2)) * alpha_tm1

    return Impedance(freq_Hz, Zdip.real.numpy(), Zdip.imag.numpy())
