"""Integrates canonical (x,px,y,py,tau,delta) transport using symplectic splitting, sector-bend drifts,
normal/skew multipole and corrector/edge kicks, and RF cavities with supported focusing and optional
structure wakes. See docs/modules/tracking/native/elements.md."""
from __future__ import annotations

import functools
import math

import torch

from ...constants import C_LIGHT
from ...wakefields.shortrange import ShortRangeWake
from ...wakefields.transverse import TransverseWake

_DRIFT_FRAC2 = (0.5, 0.5)
_KICK_FRAC2 = (1.0, 0.0)

_BETA = 2.0 ** (1.0 / 3.0)
_DRIFT_FRAC4 = (0.5 / (2 - _BETA), (1 - _BETA) / (2 - _BETA) / 2, (1 - _BETA) / (2 - _BETA) / 2, 0.5 / (2 - _BETA))
_KICK_FRAC4 = (1.0 / (2 - _BETA), -_BETA / (2 - _BETA), 1.0 / (2 - _BETA), 0.0)

_DRIFT_FRAC6 = (
    0.39225680523878, 0.5100434119184585, -0.47105338540975655, 0.0687531682525181,
    0.0687531682525181, -0.47105338540975655, 0.5100434119184585, 0.39225680523878,
)
_KICK_FRAC6 = (
    0.784513610477560, 0.235573213359357, -1.17767998417887, 1.3151863206839063,
    -1.17767998417887, 0.235573213359357, 0.784513610477560, 0.0,
)


def _yoshida_extend(
    drift_frac: tuple[float, ...], kick_frac: tuple[float, ...], order: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Extend a symmetric splitting by two orders using Yoshida triple-jump composition."""
    x1 = 1.0 / (2.0 - 2.0 ** (1.0 / (order + 1)))
    x0 = 1.0 - 2.0 * x1
    pairs = list(zip(drift_frac, kick_frac))
    result: list[tuple[float, float]] = []
    for i, scale in enumerate((x1, x0, x1)):
        scaled = [(scale * d, scale * k) for d, k in pairs]
        if i == 0:
            result.extend(scaled)
        else:
            prev_d, prev_k = result[-1]
            assert prev_k == 0.0, "kick_frac's last entry must be 0.0 (a symmetric splitting ends in a bare drift)"
            first_d, first_k = scaled[0]
            result[-1] = (prev_d + first_d, first_k)
            result.extend(scaled[1:])
    new_drift, new_kick = zip(*result)
    return new_drift, new_kick


_DRIFT_FRAC8, _KICK_FRAC8 = _yoshida_extend(_DRIFT_FRAC6, _KICK_FRAC6, order=6)


def _canonical_drift(
    x: torch.Tensor, px: torch.Tensor, y: torch.Tensor, py: torch.Tensor, delta: torch.Tensor,
    ds: float, h: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return canonical drift coordinates and true 3D path length for fixed curvature."""
    if h == 0.0:
        pz = torch.sqrt((1 + delta) ** 2 - px * px - py * py)
        x_new = x + ds * px / pz
        y_new = y + ds * py / pz
        arc_length = ds * (1 + delta) / pz
        return x_new, px, y_new, py, arc_length

    c, s = math.cos(ds * h), math.sin(ds * h)
    c2, s2 = math.cos(ds * h / 2.0), math.sin(ds * h / 2.0)
    pz = torch.sqrt((1 + delta) ** 2 - px * px - py * py)
    px_new = px * c + pz * s  # f*sin(theta)
    pz_theta = pz * c - px * s  # f*cos(theta)
    rho0 = 1.0 / h
    # tan(theta) - tan(phi) = sin(ds*h) / (cos(theta)*cos(phi)) -- ds*h a
    # fixed Python float (exact), not a subtraction of two near-equal values.
    factor = (rho0 + x) * s / pz_theta
    y_new = y + py * factor
    g_minus_1 = 2.0 * s2 * (px * c2 + pz * s2) / pz_theta
    x_new = x + (rho0 + x) * g_minus_1
    arc_length = factor * (1 + delta)  # physical path length, not the momentum-normalized factor
    return x_new, px_new, y_new, py, arc_length


def _exact_dipole_drift(
    x: torch.Tensor, px: torch.Tensor, y: torch.Tensor, py: torch.Tensor, delta: torch.Tensor,
    ds: float, h: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return cancellation-resistant coordinate increments for an exact sector dipole."""
    c, s = math.cos(ds * h), math.sin(ds * h)
    rho = 1.0 / h
    opd2 = (1 + delta) ** 2
    dd = delta * (2 + delta)
    pz = torch.sqrt(opd2 - px * px - py * py)
    w = (dd - px * px - py * py) / (pz + 1)  # pz - 1
    cm1 = -2.0 * math.sin(ds * h / 2.0) ** 2  # cos(phi) - 1, without cancellation
    dpx = px * cm1 + (w - h * x) * s
    px_new = px + dpx
    pz_new = torch.sqrt(opd2 - px_new * px_new - py * py)
    w_new = (dd - px_new * px_new - py * py) / (pz_new + 1)
    dx = x * cm1 + rho * (w_new - w * c + px * s)
    # angle turned about the particle's own centre: phi + asin(px/ptt) - asin(px_new/ptt)
    alpha = ds * h + torch.asin((px * pz_new - px_new * pz) / (opd2 - py * py))
    return dx, dpx, py * rho * alpha, (1 + delta) * rho * alpha


def _multipole_kick(
    x: torch.Tensor, px: torch.Tensor, y: torch.Tensor, py: torch.Tensor,
    k1: float, k2: float, ds: float, h: float, dipole: bool = True,
    k3: float = 0.0, multipoles: tuple = (), skew_multipoles: tuple = (),
) -> tuple[torch.Tensor, torch.Tensor]:
    """New ``(px, py)`` after the kick -- see ``_multipole_kick_increments``."""
    dpx, dpy = _multipole_kick_increments(x, y, k1, k2, ds, h, dipole, k3, multipoles, skew_multipoles)
    return px + dpx, py + dpy


def _multipole_kick_increments(
    x: torch.Tensor, y: torch.Tensor, k1: float, k2: float, ds: float, h: float, dipole: bool = True,
    k3: float = 0.0, multipoles: tuple = (), skew_multipoles: tuple = (),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return canonical dipole and normal/skew multipole momentum increments."""
    fy = k1 * x + 0.5 * k2 * (x * x - y * y)
    if dipole:  # False: the dipole term lives in _exact_dipole_drift instead
        fy = fy + h
    fx = k1 * y + k2 * x * y
    if isinstance(k3, torch.Tensor) or k3 != 0.0:
        fy = fy + k3 * (x**3 - 3 * x * y**2) / 6
        fx = fx + k3 * (3 * x**2 * y - y**3) / 6
    # Re/Im[(Kn+i*Ks)*(x+i*y)^n/n!]; scalar order loops, batched particles.
    for terms, skew in ((multipoles, False), (skew_multipoles, True)):
        for order, strength in terms:
            real, imag = torch.ones_like(x), torch.zeros_like(y)
            for j in range(1, order + 1):
                real, imag = (real * x - imag * y) / j, (real * y + imag * x) / j
            fy = fy + strength * (-imag if skew else real)
            fx = fx + strength * (real if skew else imag)
    weight = ds * (1.0 + h * x)
    return -weight * fy, weight * fx


def _corrector_kick(
    state6: torch.Tensor, length: float, kick: float | torch.Tensor, plane: str,
    energy_eV: float | torch.Tensor, mc2_eV: float | torch.Tensor,
) -> torch.Tensor:
    """Compute corrector kick; see the module documentation for details."""
    x, px, y, py, tau, delta = state6.unbind(-1)
    xp, yp, _ = _slopes_from_canonical(px, py, delta)
    sin_k = torch.sin(kick) if isinstance(kick, torch.Tensor) else math.sin(kick)
    if length <= 0.0:
        dv = kick - sin_k * delta
        if plane == "x":
            xp = xp + dv
        else:
            yp = yp + dv
        px, py = _canonical_from_slopes(xp, yp, delta)
        return torch.stack([x, px, y, py, tau, delta], dim=-1)

    xp0, yp0 = xp, yp  # entrance slopes -- this map is linear-optics-exact (elegant's matrix method)
    delta_weight = 1.0 - delta + delta * delta
    dv = sin_k * delta_weight
    du = 0.5 * sin_k * length * delta_weight
    if plane == "x":
        x = x + xp0 * length + du
        xp = xp + dv
        y = y + yp0 * length  # the other (un-kicked) plane still drifts over `length`
    else:
        y = y + yp0 * length + du
        yp = yp + dv
        x = x + xp0 * length

    beta0 = (energy_eV * energy_eV - mc2_eV * mc2_eV) ** 0.5 / energy_eV
    beta = _beta_of(delta, energy_eV, mc2_eV)
    pz_slope = torch.sqrt(1 + xp0 * xp0 + yp0 * yp0)  # 1/cos(entrance angle), for a slope-space arc length
    arc_length = length * pz_slope
    tau = tau + (arc_length / beta - length / beta0)
    px, py = _canonical_from_slopes(xp, yp, delta)
    return torch.stack([x, px, y, py, tau, delta], dim=-1)


def _slopes_from_canonical(
    px: torch.Tensor, py: torch.Tensor, delta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pz = torch.sqrt((1 + delta) ** 2 - px * px - py * py)
    return px / pz, py / pz, pz


def _canonical_from_slopes(
    xp: torch.Tensor, yp: torch.Tensor, delta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pz = (1 + delta) / torch.sqrt(1 + xp * xp + yp * yp)
    return xp * pz, yp * pz


def _edge_kick(
    x: torch.Tensor, px: torch.Tensor, y: torch.Tensor, py: torch.Tensor, delta: torch.Tensor,
    h: float, edge_angle: float, hgap: float = 0.0, fint: float = 0.5,
    edge_order: int = 1, which_edge: int = -1, k1: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Elegant style slope-space edge map (EDGE_EFFECTS=1)."""
    if edge_order < 2 and edge_angle == 0.0 and hgap * fint == 0.0:
        return x, px, y, py
    xp, yp, _ = _slopes_from_canonical(px, py, delta)
    he = h / (1 + delta)
    # Under ``torch.compile`` these per-element floats may be represented as SymFloats.
    t = math.tan(edge_angle)
    sin_e = math.sin(edge_angle)
    cos_e = math.cos(edge_angle)
    sec2 = 1 / (cos_e * cos_e)
    psi = 2 * hgap * fint * h * (1 + sin_e * sin_e) / cos_e
    psi = psi * (1 + delta) if edge_order >= 2 else psi / (1 + delta)
    xp_new = xp + he * t * x
    yp_new = yp - he * torch.tan(edge_angle - psi) * y
    if edge_order >= 2:
        n = -k1 / (h*h)
        t2 = t*t
        h2 = he*he
        t211 = -n*h2*t if which_edge == -1 else -h2*(n+t2/2)*t
        t233 = h2*(n+0.5+t2)*t if which_edge == -1 else h2*(n-t2/2)*t
        xp_new = (xp_new + t211*x*x - which_edge*he*t2*x*xp
                  + t233*y*y + which_edge*he*t2*y*yp)
        yp_new = (yp_new + which_edge*he*t2*yp*x
                  + h2*(2*n+(sec2 if which_edge == 1 else 0))*t*x*y
                  + which_edge*he*sec2*xp*y)
        x_new = x + which_edge*he/2*(t2*x*x-sec2*y*y)
        y = y - which_edge*he*t2*x*y
        x = x_new
    px_new, py_new = _canonical_from_slopes(xp_new, yp_new, delta)
    return x, px_new, y, py_new


def _misalign(
    x: torch.Tensor, px: torch.Tensor, y: torch.Tensor, py: torch.Tensor,
    dx: float | torch.Tensor, dy: float | torch.Tensor, tilt: float | torch.Tensor, entering: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute misalign; see the module documentation for details."""
    offset = isinstance(dx, torch.Tensor) or isinstance(dy, torch.Tensor) or dx != 0.0 or dy != 0.0
    rotate = isinstance(tilt, torch.Tensor) or tilt != 0.0
    if rotate:
        cos_t = torch.cos(tilt) if isinstance(tilt, torch.Tensor) else math.cos(tilt)
        sin_t = torch.sin(tilt) if isinstance(tilt, torch.Tensor) else math.sin(tilt)
    if entering:
        if offset:
            x, y = x - dx, y - dy
        if rotate:
            x, y = x * cos_t + y * sin_t, -x * sin_t + y * cos_t
            px, py = px * cos_t + py * sin_t, -px * sin_t + py * cos_t
    else:
        if rotate:
            sin_t = -sin_t
            x, y = x * cos_t + y * sin_t, -x * sin_t + y * cos_t
            px, py = px * cos_t + py * sin_t, -px * sin_t + py * cos_t
        if offset:
            x, y = x + dx, y + dy
    return x, px, y, py


def _beta_of(delta: torch.Tensor, energy_eV: float | torch.Tensor, mc2_eV: float | torch.Tensor) -> torch.Tensor:
    """Actual velocity fraction (beta = pc/E) of a particle with fractional
    momentum deviation ``delta`` from the reference (``energy_eV`` = total
    reference energy, ``mc2_eV`` = rest energy)."""
    p0 = (energy_eV * energy_eV - mc2_eV * mc2_eV) ** 0.5
    p = p0 * (1 + delta)
    E = torch.sqrt(p * p + mc2_eV * mc2_eV)
    return p / E


def _match_energy_to_centroid(
    state6: torch.Tensor, energy_eV: torch.Tensor, mc2_eV: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The new reference momentum is the ensemble's own momentum-space mean, ``P_avg =
    mean_i[P0*(1+delta_i)]`` -- not an energy-space mean, and not a ``Vrf*sin(phase)``-derived
    design value."""
    x, px, y, py, tau, delta = state6.unbind(-1)
    gamma0 = energy_eV / mc2_eV
    P0 = (gamma0 * gamma0 - 1.0) ** 0.5
    P_avg = (P0 * (1.0 + delta)).mean()
    dp = (P0 - P_avg) / P_avg
    dr = P0 / P_avg  # old/new
    delta = dp + delta * dr
    px = px * dr
    py = py * dr
    gamma0_new = (P_avg * P_avg + 1.0) ** 0.5
    energy_eV_new = gamma0_new * mc2_eV
    return torch.stack([x, px, y, py, tau, delta], dim=-1), energy_eV_new


def _rebase_reference(
    state6: torch.Tensor, energy_eV: torch.Tensor, energy_new_eV: torch.Tensor, mc2_eV: torch.Tensor,
) -> torch.Tensor:
    """Compute rebase reference; see the module documentation for details."""
    x, px, y, py, tau, delta = state6.unbind(-1)
    dr = (((energy_eV / mc2_eV) ** 2 - 1.0) / ((energy_new_eV / mc2_eV) ** 2 - 1.0)) ** 0.5  # P_old / P_new
    return torch.stack([x, px * dr, y, py * dr, tau, (1.0 + delta) * dr - 1.0], dim=-1)


def _rf_one_sub_kick(
    x: torch.Tensor, px: torch.Tensor, y: torch.Tensor, py: torch.Tensor, tau: torch.Tensor, delta: torch.Tensor,
    energy_eV: torch.Tensor, mc2_eV: torch.Tensor, P0: torch.Tensor, half: float, length_k: float,
    voltage_k: float, phase_rad: float, omega_rf: float, phase_slip: float, change_p0: bool,
    wake: ShortRangeWake | None, transverse_wake_x: TransverseWake | None, transverse_wake_y: TransverseWake | None,
    charge_per_particle_C: torch.Tensor | float, is_first: bool, is_last: bool,
    end1_focus: bool, end2_focus: bool, body_focus_model: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute rf one sub kick; see the module documentation for details."""
    def _drift_half(x, px, y, py, tau, delta, e_eV):
        beta_ref = (e_eV * e_eV - mc2_eV * mc2_eV) ** 0.5 / e_eV
        beta = _beta_of(delta, e_eV, mc2_eV)
        x, px, y, py, arc_length = _canonical_drift(x, px, y, py, delta, half, 0.0)
        tau = tau + (arc_length / beta - half / beta_ref)
        return x, px, y, py, tau

    is_srs = body_focus_model == "srs"

    if end1_focus and is_first and not is_srs:
        dgamma_end1 = (voltage_k / mc2_eV) * torch.sin(phase_rad + omega_rf * (tau / C_LIGHT))
        gamma_end1 = (P0 * P0 * (1.0 + delta) * (1.0 + delta) + 1.0) ** 0.5
        pz_norm_end1 = ((1.0 + delta) * (1.0 + delta) - px * px - py * py) ** 0.5
        inverseF1 = dgamma_end1 / (2.0 * gamma_end1 * length_k)
        px = px - pz_norm_end1 * x * inverseF1
        py = py - pz_norm_end1 * y * inverseF1

    x_entrance, px_entrance, y_entrance, py_entrance, delta_entrance = x, px, y, py, delta

    if half > 0.0:
        x, px, y, py, tau = _drift_half(x, px, y, py, tau, delta, energy_eV)

    tau_s = tau / C_LIGHT
    phase_arg = phase_rad + phase_slip + omega_rf * tau_s
    dgamma = (voltage_k / mc2_eV) * torch.sin(phase_arg)

    if is_srs:
        xp_entrance, yp_entrance, _ = _slopes_from_canonical(px_entrance, py_entrance, delta_entrance)
        if end1_focus and is_first:
            gamma_entrance = (P0 * P0 * (1.0 + delta_entrance) * (1.0 + delta_entrance) + 1.0) ** 0.5
            inverse_f1 = dgamma / (2.0 * gamma_entrance * length_k)
            xp_entrance = xp_entrance - x_entrance * inverse_f1
            yp_entrance = yp_entrance - y_entrance * inverse_f1

    if body_focus_model == "tw1":
        beta_i = _beta_of(delta, energy_eV, mc2_eV)
        dpr = (
            (voltage_k / mc2_eV) / (2.0 * beta_i) * (omega_rf / C_LIGHT) * (1.0 - beta_i) * torch.cos(phase_arg)
        )
        px = px + dpr * x / P0
        py = py + dpr * y / P0

    P = P0 * (1.0 + delta)
    gamma = (P * P + 1.0) ** 0.5
    gamma1 = gamma + dgamma
    P1 = (gamma1 * gamma1 - 1.0) ** 0.5
    delta = (P1 - P0) / P0

    if is_srs:
        xp0, yp0 = xp_entrance, yp_entrance
        sin_phase = torch.sin(phase_arg)
        dgamma_max = voltage_k / mc2_eV
        small = sin_phase.abs() <= 1e-6
        sin_phase_safe = torch.where(small, torch.ones_like(sin_phase), sin_phase)
        alpha = torch.where(
            small,
            dgamma_max / gamma / (2.0 * math.sqrt(2.0)),
            torch.log(gamma1 / gamma) / (2.0 * math.sqrt(2.0) * sin_phase_safe),
        )
        sin_alpha, cos_alpha = torch.sin(alpha), torch.cos(alpha)
        r11 = cos_alpha
        r22 = r11 * gamma / gamma1
        r12 = 2.0 * math.sqrt(2.0) * gamma * length_k / dgamma_max * sin_alpha
        r21 = -sin_alpha * dgamma_max / (length_k * gamma1 * 2.0 * math.sqrt(2.0))
        x_new = x_entrance * r11 + xp0 * r12
        xp_new = x_entrance * r21 + xp0 * r22
        y_new = y_entrance * r11 + yp0 * r12
        yp_new = y_entrance * r21 + yp0 * r22
        px, py = _canonical_from_slopes(xp_new, yp_new, delta)
        x, y = x_new, y_new
        beta_entrance = _beta_of(delta_entrance, energy_eV, mc2_eV)
        beta_exit = _beta_of(delta, energy_eV, mc2_eV)
        tau = tau * (beta_exit / beta_entrance)

    if wake is not None:
        delta = delta - wake.voltage_at(tau_s, charge_per_particle_C) / energy_eV
    if transverse_wake_x is not None:
        px = px + transverse_wake_x.kick_at(tau_s, x, charge_per_particle_C) / energy_eV
    if transverse_wake_y is not None:
        py = py + transverse_wake_y.kick_at(tau_s, y, charge_per_particle_C) / energy_eV

    if change_p0 and is_last:
        state6, energy_eV = _match_energy_to_centroid(
            torch.stack([x, px, y, py, tau, delta], dim=-1), energy_eV, mc2_eV,
        )
        x, px, y, py, tau, delta = state6.unbind(-1)

    if half > 0.0 and not is_srs:
        x, px, y, py, tau = _drift_half(x, px, y, py, tau, delta, energy_eV)

    if end2_focus and is_last:
        pz_norm_end2 = ((1.0 + delta) * (1.0 + delta) - px * px - py * py) ** 0.5
        inverseF2 = -dgamma / (2.0 * gamma1 * length_k)
        px = px - pz_norm_end2 * x * inverseF2
        py = py - pz_norm_end2 * y * inverseF2

    return x, px, y, py, tau, delta, energy_eV


_compiled_rf_one_sub_kick = None


def _get_compiled_rf_one_sub_kick():
    """Lazy singleton, built once on first use -- a plain module-level
    global (not ``functools.lru_cache``, since ``torch.compile`` objects
    aren't hashable-argument-friendly for it)."""
    global _compiled_rf_one_sub_kick
    if _compiled_rf_one_sub_kick is None:
        _compiled_rf_one_sub_kick = torch.compile(_rf_one_sub_kick)
    return _compiled_rf_one_sub_kick


def _rf_cavity_kick(
    state6: torch.Tensor, length: float, voltage_V: float, phase_rad: float, omega_rf: float,
    energy_eV: torch.Tensor, mc2_eV: torch.Tensor, change_p0: bool = False,
    wake: ShortRangeWake | None = None, transverse_wake_x: TransverseWake | None = None,
    transverse_wake_y: TransverseWake | None = None, charge_per_particle_C: torch.Tensor | float = 0.0,
    n_kicks: int = 1, beta_wave: float = 1.0,
    end1_focus: bool = False, end2_focus: bool = False, body_focus_model: str = "none",
    compile_sub_kicks: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a local RF energy kick and optional reference-momentum change."""
    n_kicks = max(int(n_kicks), 1)
    length_k = length / n_kicks
    voltage_k = voltage_V / n_kicks
    half = length_k / 2.0

    gamma0 = energy_eV / mc2_eV
    P0 = (gamma0 * gamma0 - 1.0) ** 0.5
    beta0 = P0 / gamma0

    x, px, y, py, tau, delta = state6.unbind(-1)

    sub_kick = _get_compiled_rf_one_sub_kick() if compile_sub_kicks else _rf_one_sub_kick
    for ik in range(n_kicks):
        # phase_slip computed here (a plain float, not the loop index ik itself) -- see _rf_one_sub_kick's own docstring for why.
        phase_slip = (omega_rf / C_LIGHT) * (ik * length_k) * (1.0 / beta0 - 1.0 / beta_wave)
        x, px, y, py, tau, delta, energy_eV = sub_kick(
            x, px, y, py, tau, delta, energy_eV, mc2_eV, P0, half, length_k, voltage_k, phase_rad, omega_rf,
            phase_slip, change_p0, wake, transverse_wake_x, transverse_wake_y, charge_per_particle_C,
            ik == 0, ik == n_kicks - 1, end1_focus, end2_focus, body_focus_model,
        )

    return torch.stack([x, px, y, py, tau, delta], dim=-1), energy_eV


def track_element(
    state6: torch.Tensor, length: float, angle: float, k1: float, k2: float, e1: float, e2: float,
    energy_eV: float | torch.Tensor, mc2_eV: float | torch.Tensor, n_kicks: int = 4,
    integration_order: int = 4, dx: float = 0.0, dy: float = 0.0, tilt: float = 0.0,
    hgap: float = 0.0, fint: float = 0.5, edge_order: int = 1,
    edge1_effects: int = 1, edge2_effects: int = 1, exact_dipole: bool = False,
    k3: float = 0.0, multipoles: tuple = (), skew_multipoles: tuple = (),
    integrated_multipoles: tuple = (),
) -> torch.Tensor:
    """Transport batched canonical particles through one configured element."""
    if angle != 0.0 and (edge1_effects not in (0, 1) or edge2_effects not in (0, 1)):
        raise NotImplementedError("Only EDGE1/2_EFFECTS=0 or 1 are supported")
    for terms in (multipoles, skew_multipoles, integrated_multipoles):
        if any(not isinstance(order, int) or isinstance(order, bool) or order < 0 for order, _ in terms):
            raise ValueError("multipole orders must be nonnegative integers")
    if integrated_multipoles and (isinstance(length, torch.Tensor) or length != 0.0):
        raise ValueError("integrated_multipoles require zero length")
    if not isinstance(length, torch.Tensor) and length <= 0.0 and (isinstance(k3, torch.Tensor) or k3 != 0 or multipoles or skew_multipoles):
        raise ValueError("distributed higher/skew multipoles require positive length")
    # A tensor length (RingMap's regional-compile promotion) is positive by
    # construction; testing it would sync the device and break CUDA-graph capture.
    if not isinstance(length, torch.Tensor) and length <= 0.0 and integrated_multipoles:
        x, px, y, py, tau, delta = state6.unbind(-1)
        x, px, y, py = _misalign(x, px, y, py, dx, dy, tilt, entering=True)
        px, py = _multipole_kick(x, px, y, py, 0.0, 0.0, 1.0, 0.0, False, multipoles=integrated_multipoles)
        x, px, y, py = _misalign(x, px, y, py, dx, dy, tilt, entering=False)
        return torch.stack((x, px, y, py, tau, delta), -1)
    if not isinstance(length, torch.Tensor) and length <= 0.0:
        return state6
    x, px, y, py, tau, delta = state6.unbind(-1)
    h = angle / length if angle != 0.0 else 0.0
    beta0 = (energy_eV * energy_eV - mc2_eV * mc2_eV) ** 0.5 / energy_eV
    beta = _beta_of(delta, energy_eV, mc2_eV)  # delta is constant through an element (no energy loss modeled here)
    drift_frac, kick_frac_table = {
        4: (_DRIFT_FRAC4, _KICK_FRAC4), 6: (_DRIFT_FRAC6, _KICK_FRAC6), 8: (_DRIFT_FRAC8, _KICK_FRAC8),
    }[integration_order]

    misaligned = (
        isinstance(dx, torch.Tensor) or isinstance(dy, torch.Tensor) or isinstance(tilt, torch.Tensor)
        or dx != 0.0 or dy != 0.0 or tilt != 0.0
    )
    if misaligned:
        x, px, y, py = _misalign(x, px, y, py, dx, dy, tilt, entering=True)

    if h != 0.0 and edge1_effects:
        x, px, y, py = _edge_kick(x, px, y, py, delta, h, e1, hgap, fint, edge_order, -1, k1)

    # A tensor k1/k2 (autograd knob or a per-machine error-ensemble batch,
    # see tracking/native/ring.py) always takes the kick path.
    exact_dipole = exact_dipole and h != 0.0
    no_multipole = not any(isinstance(v, torch.Tensor) for v in (k1, k2, k3)) and k1 == 0.0 and k2 == 0.0 and k3 == 0.0 and not multipoles and not skew_multipoles
    if exact_dipole:
        # Exact dipole + K1/K2-only kicks.
        steps = [(length, 0.0)] if no_multipole else [
            (length / n_kicks * df, length / n_kicks * kf) for _ in range(n_kicks)
            for df, kf in zip(drift_frac, kick_frac_table)
        ]
        for dl, kl in steps:
            ix, ipx, iy, arc_length = _exact_dipole_drift(x, px, y, py, delta, dl, h)
            x, y, px = x + ix, y + iy, px + ipx
            tau = tau + (arc_length / beta - dl / beta0)
            if kl != 0.0:
                dpx, dpy = _multipole_kick_increments(x, y, k1, k2, kl, h, dipole=not exact_dipole, k3=k3, multipoles=multipoles, skew_multipoles=skew_multipoles)
                px, py = px + dpx, py + dpy
    elif no_multipole and h == 0.0:
        # Pure drift -- exact at any slice count, so skip the kick loop.
        x, px, y, py, arc_length = _canonical_drift(x, px, y, py, delta, length, h)
        tau = tau + (arc_length / beta - length / beta0)
    else:
        ds = length / n_kicks
        for _ in range(n_kicks):
            for df, kf in zip(drift_frac, kick_frac_table):
                x, px, y, py, arc_length = _canonical_drift(x, px, y, py, delta, ds * df, h)
                tau = tau + (arc_length / beta - ds * df / beta0)
                if kf != 0.0:
                    px, py = _multipole_kick(x, px, y, py, k1, k2, ds * kf, h, k3=k3, multipoles=multipoles, skew_multipoles=skew_multipoles)

    if h != 0.0 and edge2_effects:
        x, px, y, py = _edge_kick(x, px, y, py, delta, h, e2, hgap, fint, edge_order, 1, k1)

    if misaligned:
        x, px, y, py = _misalign(x, px, y, py, dx, dy, tilt, entering=False)

    return torch.stack([x, px, y, py, tau, delta], dim=-1)


@functools.lru_cache(maxsize=None)
def reference_offset(
    length: float, angle: float, k1: float, k2: float, e1: float, e2: float,
    energy_eV: float, mc2_eV: float, n_kicks: int, integration_order: int = 4,
    hgap: float = 0.0, fint: float = 0.5, edge_order: int = 1,
    edge1_effects: int = 1, edge2_effects: int = 1, exact_dipole: bool = False,
) -> torch.Tensor:
    """Return the cached ideal-element numerical reference displacement."""
    dtype = torch.float64
    zero = torch.zeros((1, 6), dtype=dtype)
    energy_t = torch.tensor(energy_eV, dtype=dtype)
    mc2_t = torch.tensor(mc2_eV, dtype=dtype)
    return track_element(
        zero, length, angle, k1, k2, e1, e2, energy_t, mc2_t,
        n_kicks=n_kicks, integration_order=integration_order,
        hgap=hgap, fint=fint, edge_order=edge_order,
        edge1_effects=edge1_effects, edge2_effects=edge2_effects, exact_dipole=exact_dipole,
    ).squeeze(0)

