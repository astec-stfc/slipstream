"""Computes amplitude-normalized intrabeam-scattering growth rates in inverse seconds. See
docs/modules/ibs.md."""
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
    """Compute g; see the module documentation for details."""
    if alpha > 1.0:
        alpha = 1.0 / alpha
    alpha = max(alpha, 0.01)
    return 2.0 * alpha ** (0.021 - 0.044 * math.log(alpha))


def coulomb_log(
    sigma_x_m: float, sigma_y_m: float, sigma_z_m: float,
    gamma: float, beta_rel: float, sigma_p: float, n_particles_bunch: float,
) -> float:
    """Piwinski's Coulomb logarithm ln(b_max/b_min) (see e.g. the CERN Accelerator School's IBS
    lectures for the derivation)."""
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
    """IBS growth rates (Bane Eqs. 8-14) at a single smooth-lattice optics point."""
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
    """Compute bm matrices; see the module documentation for details."""
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
    """Compute bm integral; see the module documentation for details."""
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
    """Compute bjorken mtingwa growth rates; see the module documentation for details."""
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
    """KMW's own Eq. 4 defines ``1/T_i`` as ``<...>`` -- an average of the growth-rate integral
    evaluated locally at every point around the ring, not the integral of a single averaged point."""
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
