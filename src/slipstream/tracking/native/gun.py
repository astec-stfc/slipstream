"""Uses the relativistic Boris pusher in Cartesian `(x,y,z,ux,uy,uz)`, with proper velocity
`u=p/m=gamma*v` in m/s. See docs/modules/tracking/native/gun.md."""
from __future__ import annotations

import math
from typing import Callable

import torch

from ...constants import C_LIGHT
from .space_charge import CylindricalSpaceCharge, cartesian_field_at, field_with_cathode_image

FieldFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]


def kahan_add(u: torch.Tensor, a: torch.Tensor, c: torch.Tensor):
    """One compensated (Kahan) accumulate ``u += a``, returning ``(u_new, c_new)``."""
    y_ = a - c
    t = u + y_
    return t, (t - u) - y_


def boris_push(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
    ux: torch.Tensor, uy: torch.Tensor, uz: torch.Tensor,
    q_C: torch.Tensor | float, mass_kg: float,
    Ex: torch.Tensor, Ey: torch.Tensor, Ez: torch.Tensor,
    Bx: torch.Tensor, By: torch.Tensor, Bz: torch.Tensor,
    dt: float,
    comp: tuple[torch.Tensor, ...] | None = None,
):
    """One relativistic Boris step (SI units throughout: m, m/s, V/m, T, kg, C, s)."""
    qmdt2 = q_C / mass_kg * (dt / 2.0)

    ux_minus = ux + qmdt2 * Ex
    uy_minus = uy + qmdt2 * Ey
    uz_minus = uz + qmdt2 * Ez

    gamma_minus = torch.sqrt(1.0 + (ux_minus**2 + uy_minus**2 + uz_minus**2) / C_LIGHT**2)

    tx = qmdt2 / gamma_minus * Bx
    ty = qmdt2 / gamma_minus * By
    tz = qmdt2 / gamma_minus * Bz
    t2 = tx * tx + ty * ty + tz * tz

    upx = ux_minus + (uy_minus * tz - uz_minus * ty)
    upy = uy_minus + (uz_minus * tx - ux_minus * tz)
    upz = uz_minus + (ux_minus * ty - uy_minus * tx)

    denom = 1.0 + t2
    sx = 2.0 * tx / denom
    sy = 2.0 * ty / denom
    sz = 2.0 * tz / denom

    ux_plus = ux_minus + (upy * sz - upz * sy)
    uy_plus = uy_minus + (upz * sx - upx * sz)
    uz_plus = uz_minus + (upx * sy - upy * sx)

    if comp is None:
        ux_new = ux_plus + qmdt2 * Ex
        uy_new = uy_plus + qmdt2 * Ey
        uz_new = uz_plus + qmdt2 * Ez
    else:
        # Same update, assembled as an increment of small terms and added once with compensation (see the docstring).
        cx, cy, cz, cux, cuy, cuz = comp
        ax = 2.0 * qmdt2 * Ex + (upy * sz - upz * sy)
        ay = 2.0 * qmdt2 * Ey + (upz * sx - upx * sz)
        az = 2.0 * qmdt2 * Ez + (upx * sy - upy * sx)
        ux_new, cux = kahan_add(ux, ax, cux)
        uy_new, cuy = kahan_add(uy, ay, cuy)
        uz_new, cuz = kahan_add(uz, az, cuz)

    gamma_new = torch.sqrt(1.0 + (ux_new**2 + uy_new**2 + uz_new**2) / C_LIGHT**2)
    vx, vy, vz = ux_new / gamma_new, uy_new / gamma_new, uz_new / gamma_new

    if comp is None:
        return x + vx * dt, y + vy * dt, z + vz * dt, ux_new, uy_new, uz_new
    x_new, cx = kahan_add(x, vx * dt, cx)
    y_new, cy = kahan_add(y, vy * dt, cy)
    z_new, cz = kahan_add(z, vz * dt, cz)
    return (x_new, y_new, z_new, ux_new, uy_new, uz_new,
            (cx, cy, cz, cux, cuy, cuz))


def track_particles(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
    ux: torch.Tensor, uy: torch.Tensor, uz: torch.Tensor,
    q_C: torch.Tensor | float, mass_kg: float,
    field_fn: FieldFn,
    t0: float, dt: float, n_steps: int,
    use_compile: bool = True,
):
    """Compute track particles; see the module documentation for details."""
    t = t0
    if not use_compile:
        for _ in range(n_steps):
            t_vec = torch.full_like(x, t)
            Ex, Ey, Ez, Bx, By, Bz = field_fn(x, y, z, t_vec)
            x, y, z, ux, uy, uz = boris_push(
                x, y, z, ux, uy, uz, q_C, mass_kg, Ex, Ey, Ez, Bx, By, Bz, dt
            )
            t += dt
        return x, y, z, ux, uy, uz

    from .injector import _fused_step

    # Mirror of Injector.track's step_fn gate (injector.py:498): compile the shared step rather than duplicating it.
    step_fn = torch.compile(_fused_step, dynamic=False)

    n = x.shape[0]
    zeros_n = torch.zeros_like(x)
    cx, cy, cz, cux, cuy, cuz = (zeros_n.clone() for _ in range(6))
    sw, tw, sol, quad = (), (), (), ()  # no injector elements -- field_fn is the field
    t_ten = torch.full((), t0, device=x.device, dtype=torch.float64)
    qmdt2 = q_C / mass_kg * (dt / 2.0)
    c2_inv = 1.0 / C_LIGHT**2
    emitted_all = True
    all_active = torch.ones(n, device=x.device, dtype=torch.bool)
    for _ in range(n_steps):
        t_vec = torch.full_like(x, t)
        Ex, Ey, Ez, Bx, By, Bz = field_fn(x, y, z, t_vec)
        (x, y, z, ux, uy, uz,
         cx, cy, cz, cux, cuy, cuz) = step_fn(
            x, y, z, ux, uy, uz, cx, cy, cz, cux, cuy, cuz,
            Ex, Ey, Ez, Bx, By, Bz,
            sw, tw, sol, quad, t_ten, dt, qmdt2, emitted_all, all_active, c2_inv,
        )
        t += dt
        t_ten.add_(dt)
    return x, y, z, ux, uy, uz


def relativistic_space_charge_field(
    solver: CylindricalSpaceCharge,
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, q: torch.Tensor,
    bunch_gamma: float,
    cathode_z: float | None = None,
    smooth_r: int = 0, smooth_z: int = 0, smooth_gamma_ref: float = math.inf,
):
    """Compute relativistic space charge field; see the module documentation for details."""
    beta = math.sqrt(max(0.0, 1.0 - 1.0 / bunch_gamma**2))
    z_ref = cathode_z if cathode_z is not None else float(z.mean())
    z_stretched = z_ref + bunch_gamma * (z - z_ref)

    if cathode_z is None:
        Ex_p, Ey_p, Ez_lab = cartesian_field_at(solver, x, y, z_stretched, q,
                                                 smooth_r, smooth_z, bunch_gamma, smooth_gamma_ref)
    else:
        Ex_p, Ey_p, Ez_lab = field_with_cathode_image(solver, x, y, z_stretched, q, z_ref,
                                                       smooth_r, smooth_z, bunch_gamma, smooth_gamma_ref)

    Ex_lab = bunch_gamma * Ex_p
    Ey_lab = bunch_gamma * Ey_p
    Bx_lab = -(beta / C_LIGHT) * Ey_lab
    By_lab = (beta / C_LIGHT) * Ex_lab
    Bz_lab = torch.zeros_like(Ez_lab)
    return Ex_lab, Ey_lab, Ez_lab, Bx_lab, By_lab, Bz_lab

