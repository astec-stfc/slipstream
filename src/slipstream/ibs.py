"""Intra-beam scattering (IBS) growth rates -- two formalisms:

- ``bane_growth_rates``: K. Bane's closed-form high-energy approximation
  to Bjorken-Mtingwa/Piwinski (SLAC-PUB-9226 / arXiv:physics/0205058
  (2002), Eqs. 8-14). Fast (no numerical integration), but assumes
  ultra-relativistic beam_rel~1 and gamma>>1 throughout its derivation --
  wrong tool below a few hundred MeV (see IBS_PLAN.md's ELENA section for
  where this bit).
- ``bjorken_mtingwa_growth_rates``: the general theory these approximate,
  transcribed from Kubo, Mtingwa & Wolski, "Intrabeam scattering formulas
  for high energy beams," Phys. Rev. ST Accel. Beams 8, 081001 (2005)
  (KMW's own Eqs. 4-9 restate Bjorken & Mtingwa, Part. Accel. 13, 115
  (1983) directly, not just KMW's own high-energy CIMP approximation --
  see that paper's Eq. 4 vs Eq. 35 for the distinction). No high-energy
  assumption anywhere -- keeps the full beta_rel^3*gamma^4 dependence Bane
  drops -- at the cost of a numerical integral (``scipy.integrate.quad``)
  per plane per call. S. Nagaitsev's "Intrabeam scattering formulas for
  fast numerical evaluation," Phys. Rev. ST Accel. Beams 8, 064403 (2005)
  gives a faster closed-form reformulation of this same integral (not
  implemented here).

Both are evaluated at ``TransverseOptics``'s single matched point -- the
same smooth-ring simplification already used for betatron motion
elsewhere in this codebase (see ``lattice/optics.py``'s docstring), not an
element-by-element lattice integration.

Both return growth *rates* 1/T_p, 1/T_x, 1/T_y [1/s] for the rms momentum
spread and (horizontal/vertical) emittances -- these are
*amplitude*-normalized rates (Bane Eq. 1 / KMW Eqs. 1-3), e.g.
``1/T_p = (1/sigma_p) * d(sigma_p)/dt``, not ``d(sigma_p^2)/dt`` rates.
Unlike Bane, the general Bjorken-Mtingwa integral *can* come out negative
in a plane (a real effect -- IBS redistributing momentum spread can
shrink an emittance, not just grow it; Bane's formula is structurally
incapable of this, being a sum of non-negative terms throughout). How
these fold into the tracker's per-turn damping/excitation map (as
additional anti- or extra-damping alongside radiation damping) is
``tracking.pystable.tracker``'s concern, not this module's.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import integrate

from .constants import C_LIGHT, R_E_M
from .lattice.twiss_table import TwissTable


@dataclass
class IBSGrowthRates:
    inv_Tp_s: float  # 1/T_p -- rms momentum-spread growth rate [1/s]
    inv_Tx_s: float  # 1/T_x -- horizontal emittance growth rate [1/s]
    inv_Ty_s: float  # 1/T_y -- vertical emittance growth rate [1/s]


def dispersion_invariant(D_m: float, Dp: float, beta_m: float, alpha: float) -> float:
    """Curly-H dispersion invariant, H = [D^2 + (beta*D' + alpha*D)^2] / beta
    (Bane Eq. 9's H, using beta' = -2*alpha at a matched point so that
    beta*D' - 0.5*beta'*D == beta*D' + alpha*D -- the usual Twiss identity).
    """
    return (D_m**2 + (beta_m * Dp + alpha * D_m) ** 2) / beta_m


def _g(alpha: float) -> float:
    """Bane's fitting approximation to the g(a/b) auxiliary integral (Eqs.
    11-12): g(alpha) = g(1/alpha) (stated explicitly in the paper), so only
    alpha in (0, 1] is ever evaluated by the fit; the fit's stated validity
    floor is alpha > 0.01, clamped rather than extrapolated below it."""
    if alpha > 1.0:
        alpha = 1.0 / alpha
    alpha = max(alpha, 0.01)
    return 2.0 * alpha ** (0.021 - 0.044 * math.log(alpha))


def coulomb_log(
    sigma_x_m: float, sigma_y_m: float, sigma_z_m: float,
    gamma: float, beta_rel: float, sigma_p: float, n_particles_bunch: float,
) -> float:
    """Piwinski's Coulomb logarithm ln(b_max/b_min) (see e.g. the CERN
    Accelerator School's IBS lectures for the derivation). ``b_max`` is
    capped by the beam's own size or mean interparticle spacing (a
    collision can't have an unperturbed-trajectory impact parameter wider
    than the beam itself); ``b_min`` is the classical distance of closest
    approach for a collision at the beam's characteristic relative angle
    sigma_p (the quantum/de Broglie floor is neglected -- always negligible
    at these energies/densities). Enters the growth rate only
    logarithmically, so this level of approximation is standard practice.
    """
    n0 = n_particles_bunch / ((2 * math.pi) ** 1.5 * sigma_x_m * sigma_y_m * sigma_z_m)
    b_max = min(sigma_x_m, sigma_y_m, gamma * sigma_z_m, n0 ** (-1.0 / 3.0))
    b_min = R_E_M / (gamma * beta_rel**2 * sigma_p**2)
    return math.log(b_max / b_min)


def bane_growth_rates(
    *,
    gamma: float,
    n_particles_bunch: float,
    sigma_z_m: float,
    sigma_p: float,
    epsx_m: float,
    epsy_m: float,
    beta_x_m: float,
    alpha_x: float,
    D_x_m: float,
    Dp_x: float,
    beta_y_m: float,
    alpha_y: float,
    D_y_m: float,
    Dp_y: float,
    coulomb_log_value: float | None = None,
) -> IBSGrowthRates:
    """IBS growth rates (Bane Eqs. 8-14) at a single smooth-lattice optics
    point. ``epsx_m``/``epsy_m`` are geometric rms *betatron* emittances
    (matching ``TransverseOptics.epsx_m``'s convention) -- the dispersive
    contribution to the physical beam size is added internally via
    ``D_x_m``/``D_y_m``, not folded into the emittance passed in.
    ``coulomb_log_value``, if not given, is computed from the beam's own
    density via ``coulomb_log``.
    """
    beta_rel = math.sqrt(1.0 - 1.0 / gamma**2)

    Hx = dispersion_invariant(D_x_m, Dp_x, beta_x_m, alpha_x)
    Hy = dispersion_invariant(D_y_m, Dp_y, beta_y_m, alpha_y)

    inv_sigma_H2 = 1.0 / sigma_p**2 + Hx / epsx_m + Hy / epsy_m
    sigma_H = math.sqrt(1.0 / inv_sigma_H2)

    a = (sigma_H / gamma) * math.sqrt(beta_x_m / epsx_m)
    b = (sigma_H / gamma) * math.sqrt(beta_y_m / epsy_m)

    if coulomb_log_value is None:
        sigma_x_m = math.sqrt(epsx_m * beta_x_m + (D_x_m * sigma_p) ** 2)
        sigma_y_m = math.sqrt(epsy_m * beta_y_m + (D_y_m * sigma_p) ** 2)
        coulomb_log_value = coulomb_log(
            sigma_x_m, sigma_y_m, sigma_z_m, gamma, beta_rel, sigma_p, n_particles_bunch
        )

    prefactor = (R_E_M**2 * C_LIGHT * n_particles_bunch * coulomb_log_value) / (
        32.0 * gamma**3 * epsx_m**0.75 * epsy_m**0.75 * sigma_z_m * sigma_p**3
    )
    inv_Tp = prefactor * sigma_H * _g(a / b) * (beta_x_m * beta_y_m) ** (-0.25)
    inv_Tx = (sigma_p**2 * Hx / epsx_m) * inv_Tp
    inv_Ty = (sigma_p**2 * Hy / epsy_m) * inv_Tp

    return IBSGrowthRates(inv_Tp_s=inv_Tp, inv_Tx_s=inv_Tx, inv_Ty_s=inv_Ty)


def _bm_matrices(
    gamma: float, sigma_p: float,
    epsx_m: float, beta_x_m: float, alpha_x: float, D_x_m: float, Dp_x: float,
    epsy_m: float, beta_y_m: float, alpha_y: float, D_y_m: float, Dp_y: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three auxiliary 3x3 matrices L^(p), L^(h), L^(v) (KMW Eqs. 7-9),
    index order (horizontal-momentum-like, momentum-spread-like,
    vertical-momentum-like) so index 1 (0-based, the middle one) is the
    one every plane couples through. ``phi = D' + alpha*D/beta`` (KMW's
    ``eta' - 0.5*beta'*eta/beta``, using beta'=-2*alpha, same identity
    ``dispersion_invariant`` already relies on).
    """
    Hx = dispersion_invariant(D_x_m, Dp_x, beta_x_m, alpha_x)
    Hy = dispersion_invariant(D_y_m, Dp_y, beta_y_m, alpha_y)
    phi_x = Dp_x + alpha_x * D_x_m / beta_x_m
    phi_y = Dp_y + alpha_y * D_y_m / beta_y_m

    Lp = np.zeros((3, 3))
    Lp[1, 1] = gamma**2 / sigma_p**2

    Lh = (beta_x_m / epsx_m) * np.array([
        [1.0, -gamma * phi_x, 0.0],
        [-gamma * phi_x, gamma**2 * Hx / beta_x_m, 0.0],
        [0.0, 0.0, 0.0],
    ])

    Lv = (beta_y_m / epsy_m) * np.array([
        [0.0, 0.0, 0.0],
        [0.0, gamma**2 * Hy / beta_y_m, -gamma * phi_y],
        [0.0, -gamma * phi_y, 1.0],
    ])

    return Lp, Lh, Lv


def _bm_integral(L: np.ndarray, Li: np.ndarray) -> float:
    """KMW Eq. 4's lambda-integral for one plane's numerator matrix Li
    (Lp, Lh, or Lv), i.e. everything inside the outer 4*pi*A*(log) --
    dimensionless (L's entries are all dimensionless, see ibs.py's module
    docstring's dimensional-analysis note in the git history/IBS_PLAN.md).

    L's diagonal entries span many orders of magnitude for a real beam
    (``gamma^2/sigma_p^2`` alone is ~1e13 for a GeV-scale, 1e-4-level
    sigma_p beam) -- raw quadrature over that range is numerically
    unreliable (this exact difficulty is why Nagaitsev's reformulation
    exists). Substitute lambda = kappa*u with kappa = |det(L)|^(1/3) (the
    geometric-mean eigenvalue) first, so the rescaled matrix L~ = L/kappa
    has an O(1)-ish eigenvalue spread and quad's adaptive subdivision can
    actually resolve it. Li itself is *not* rescaled (only the
    L~+u*I matrix being inverted/determinant'd is) -- this only rescales
    the integration variable, it doesn't change what's being integrated.
    """
    kappa = abs(np.linalg.det(L)) ** (1.0 / 3.0)
    L_scaled = L / kappa

    def integrand(u: float) -> float:
        M = L_scaled + u * np.eye(3)
        det_M = np.linalg.det(M)
        M_inv = np.linalg.inv(M)
        bracket = np.trace(Li) * np.trace(M_inv) - 3.0 * np.trace(Li @ M_inv)
        return math.sqrt(u) / math.sqrt(det_M) * bracket

    integral, _ = integrate.quad(integrand, 0.0, np.inf, limit=200)
    return integral / kappa


def bjorken_mtingwa_growth_rates(
    *,
    gamma: float,
    beta_rel: float,
    n_particles_bunch: float,
    sigma_z_m: float,
    sigma_p: float,
    epsx_m: float,
    epsy_m: float,
    beta_x_m: float,
    alpha_x: float,
    D_x_m: float,
    Dp_x: float,
    beta_y_m: float,
    alpha_y: float,
    D_y_m: float,
    Dp_y: float,
    coulomb_log_value: float,
    particle_radius_m: float = R_E_M,
) -> IBSGrowthRates:
    """General Bjorken-Mtingwa IBS growth rates (KMW Eqs. 4-9), no
    high-energy assumption -- the right tool below a few hundred MeV
    (Bane's formula assumes beta_rel~1 throughout its own derivation, so
    it silently drops the beta_rel^3 dependence this keeps; see
    IBS_PLAN.md). ``beta_rel`` and ``coulomb_log_value`` are explicit
    (not derived internally like ``bane_growth_rates``'s defaults) --
    ``bane_growth_rates``'s own Coulomb-log/beam-size approximation
    (implicitly beta_rel~1) isn't appropriate to reuse silently in a
    formula whose whole point is not assuming that; pass a value from
    ``coulomb_log`` explicitly (with the real ``beta_rel``) or from an
    independent reference if comparing against one, as
    ``examples/elena/`` does. ``particle_radius_m`` defaults to the
    classical electron radius; pass the classical proton radius
    (``constants.R_P_M``) for a proton/antiproton beam like ELENA's.

    ``D_x_m``/``Dp_x``/``D_y_m``/``Dp_y`` are taken in the raw ("pt"
    canonical-momentum-frame) convention MAD-X/xsuite Twiss tables report
    by default, and scaled by ``beta_rel`` internally to the physical
    ("deltap"-frame, ``d(x)/d(Δp/p)``) dispersion this formula's own L
    matrices need -- a no-op for an ultra-relativistic beam
    (``beta_rel``~1) but a real, previously-missing correction for one
    that isn't (confirmed against two independent reference
    implementations: CERN's xsuite/xfields ``BjorkenMtingwaIBS`` -- same
    comment, "Adjust dispersion... to be in the deltap and not the pt
    frame... Necessary for non-relativistic beams" -- and the CERN-note-
    -sourced tomerten/IBS library's own ``BjorkenMtingwa``, both scaling
    exactly this way and *not* doing so in their Piwinski/Bane-style
    functions, matching ``bane_growth_rates`` here also not needing it).
    Missing this made ELENA's element-averaged rates land 10-100x off
    real per-element codes (see IBS_PLAN.md's "Phase 3") until
    cross-checked directly against a live xsuite/xfields calculation on
    the same lattice, which now agrees to a few percent.
    """
    Lp, Lh, Lv = _bm_matrices(
        gamma, sigma_p,
        epsx_m, beta_x_m, alpha_x, D_x_m * beta_rel, Dp_x * beta_rel,
        epsy_m, beta_y_m, alpha_y, D_y_m * beta_rel, Dp_y * beta_rel,
    )
    L = Lp + Lh + Lv

    A = (particle_radius_m**2 * C_LIGHT * n_particles_bunch) / (
        64.0 * math.pi**2 * beta_rel**3 * gamma**4 * epsx_m * epsy_m * sigma_z_m * sigma_p
    )

    rates = [4.0 * math.pi * A * coulomb_log_value * _bm_integral(L, Li) for Li in (Lp, Lh, Lv)]

    return IBSGrowthRates(inv_Tp_s=rates[0], inv_Tx_s=rates[1], inv_Ty_s=rates[2])


def element_averaged_growth_rates(
    twiss: TwissTable,
    *,
    model: str = "bjorken-mtingwa",
    gamma: float,
    n_particles_bunch: float,
    sigma_z_m: float,
    sigma_p: float,
    epsx_m: float,
    epsy_m: float,
    coulomb_log_value: float,
    beta_rel: float | None = None,
    particle_radius_m: float = R_E_M,
) -> IBSGrowthRates:
    """KMW's own Eq. 4 defines ``1/T_i`` as ``<...>`` -- an average of the
    growth-rate integral evaluated locally at every point around the ring,
    not the integral of a single averaged point. This evaluates the
    existing single-point ``bane_growth_rates``/``bjorken_mtingwa_growth_
    rates`` (unchanged) at every row of a real (e.g. MAD-X-derived)
    ``TwissTable`` and returns the ``length_m``-weighted average -- the
    thing this repo's single-point approximation, used everywhere else in
    this file, can be off from by 2-10x (see IBS_PLAN.md, examples/elena/,
    where ``dispersion_sensitivity_scan`` shows just how nonlinear the
    integral is in the optics).

    The beam-level inputs (``gamma``, ``n_particles_bunch``, ``sigma_z_m``,
    ``sigma_p``, ``epsx_m``, ``epsy_m``, ``coulomb_log_value``, ...) are
    held fixed across the ring -- only the optics (beta/alpha/D/D') vary
    by row, matching how this module already treats the beam state as not
    itself s-dependent (``coulomb_log``'s own docstring notes this is
    always a weak/logarithmic approximation regardless).

    A plain Python loop over rows, not vectorized -- deliberately simple,
    reusing already-tested code verbatim rather than refactoring
    ``_bm_matrices``/``_bm_integral`` to operate on arrays. Fine for a
    one-off calculation against a cached Twiss table; too slow (each
    Bjorken-Mtingwa row costs a ``scipy.integrate.quad`` call per plane)
    to call every turn from a live tracker -- see IBS_PLAN.md's "what
    wasn't built" for what that would need (vectorization, or Nagaitsev's
    faster closed-form reformulation of the same integral).
    """
    if model not in ("bane", "bjorken-mtingwa"):
        raise ValueError(f"model must be 'bane' or 'bjorken-mtingwa', got {model!r}")
    if model == "bjorken-mtingwa" and beta_rel is None:
        raise ValueError("model='bjorken-mtingwa' needs an explicit beta_rel=")

    sum_p = sum_x = sum_y = 0.0
    total_length = 0.0
    for i in range(len(twiss.s_m)):
        common = dict(
            gamma=gamma, n_particles_bunch=n_particles_bunch,
            sigma_z_m=sigma_z_m, sigma_p=sigma_p, epsx_m=epsx_m, epsy_m=epsy_m,
            beta_x_m=twiss.beta_x_m[i], alpha_x=twiss.alpha_x[i],
            D_x_m=twiss.D_x_m[i], Dp_x=twiss.Dp_x[i],
            beta_y_m=twiss.beta_y_m[i], alpha_y=twiss.alpha_y[i],
            D_y_m=twiss.D_y_m[i], Dp_y=twiss.Dp_y[i],
            coulomb_log_value=coulomb_log_value,
        )
        if model == "bane":
            r = bane_growth_rates(**common)
        else:
            r = bjorken_mtingwa_growth_rates(**common, beta_rel=beta_rel, particle_radius_m=particle_radius_m)
        w = float(twiss.length_m[i])
        sum_p += r.inv_Tp_s * w
        sum_x += r.inv_Tx_s * w
        sum_y += r.inv_Ty_s * w
        total_length += w

    return IBSGrowthRates(
        inv_Tp_s=sum_p / total_length, inv_Tx_s=sum_x / total_length, inv_Ty_s=sum_y / total_length,
    )
