"""Boris-pusher gun tracker -- couples ``tracking.native.fields`` (applied
RF-cavity + solenoid fields) and ``tracking.native.space_charge``
(self-field, including the cathode's conducting-plane image-charge
boundary) in real time ``t``. Works in raw ``(x, y, z, ux, uy, uz)``
physical phase space integrated in time, not the ring/linac
``(x, px, y, py, tau, delta)`` reference-particle convention, which has no
meaning for particles born at rest at the cathode. See docs/tracking_native.md.

``ux, uy, uz`` are the *proper velocity* ``u = p/m = gamma*v`` (m/s) -- the
standard relativistic-Boris state variable: a pure rotation preserves
``|u|`` (hence ``gamma``) exactly, so ``gamma`` is evaluated once per step
rather than solved for implicitly.

**Pusher algorithm**: the non-relativistic base algorithm is Qin, Zhang,
Xiao & Tang, "Why is Boris algorithm so good?", Phys. Plasmas 20, 084503
(2013), Eqs. 3, 5-7 (energy, hence ``gamma``, is conserved exactly when
``E=0``, checked directly by this module's tests). The relativistic
extension (``u=p/m`` in place of ``v``, ``gamma`` evaluated once from
``u_minus`` before the rotation) is the standard relativistic-Boris
extension (Boris, 1970).

**Space-charge relativistic boost**: ``space_charge.py``'s solver works in
the lab frame with uncontracted macroparticle positions, correct only once
``gamma~1``. For real longitudinal motion, the standard quasi-static
treatment (ASTRA, RF-Track, external hybrid backend's ``SpaceChargeKick`` alike): transform
to the bunch's rest frame by stretching the longitudinal coordinate by
``bunch_gamma``, solve electrostatics there unmodified, transform back --
transverse ``E`` enhanced by ``bunch_gamma``, longitudinal ``E`` unchanged,
lab-frame self-``B = (beta/c) x E``. ``relativistic_space_charge_field``
below implements this; the net transverse force on a co-moving test
particle works out to ``q*Er_lab*(1-beta^2) = q*Er_lab/gamma^2``, the
textbook space-charge defocusing suppression factor. At ``bunch_gamma=1``
this reduces exactly to the pure-electrostatic kernel.

Where the RF-cavity, solenoid, and space-charge fields actually get summed
into one ``field_fn(x, y, z, t)`` for ``track_particles`` is the caller's
job (a small closure), so this module stays testable against closed-form
single-particle cases independent of any specific gun geometry.
"""
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
    """One compensated (Kahan) accumulate ``u += a``, returning
    ``(u_new, c_new)``.

    ``c`` carries the part of ``a`` that ``u``'s mantissa could not hold.
    The next call adds ``a - c`` instead of ``a``, so increments far below
    ``ulp(u)`` are not discarded -- they accrue in ``c`` and flush into
    ``u`` once they exceed half an ulp. ``c`` sits at magnitude ~ulp(u),
    where its own resolution is ~1e-7 of that, which is where the extra
    ~7 decimal digits of effective precision come from.

    ``c`` is state: carry it across steps beside ``u``. A freshly zeroed
    ``c`` each step compensates nothing."""
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
    """One relativistic Boris step (SI units throughout: m, m/s, V/m, T,
    kg, C, s). ``q_C`` may be a scalar or an ``(N,)`` tensor of signed
    per-particle charges; every other argument is ``(N,)``.

    ``comp`` opts into **compensated (Kahan) momentum accumulation** -- in
    float32, the difference between reproducing the physics and silently
    deleting it once the per-step increment ``a=(q/m)*E*dt`` falls below
    ``u``'s ulp (see docs/tracking_native.md).

    Pass a 6-tuple of per-particle compensation tensors (zeros to start,
    ordered ``(cx, cy, cz, cux, cuy, cuz)``); they are returned updated as
    a seventh element, fed back in next step as state. Positions need it
    for the same reason: the differential advance between particles (the
    longitudinal physics) is far below ``z``'s own ulp per step. In this
    mode the push is also rearranged to form the step's whole change in
    ``u`` from small terms before adding it once, instead of two half-kicks
    around a rotation -- algebraically identical, fewer chances to round a
    sub-ulp increment away. ``None`` (default) keeps the original
    uncompensated behaviour and return signature.
    """
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
        # Same update, assembled as an increment of small terms and added
        # once with compensation (see the docstring). `ux_plus` etc. are
        # only used here to recover the rotation's contribution, which is
        # `u_plus - u_minus`; forming it directly from the cross products
        # keeps every term small.
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
    """Advance particles ``n_steps``, evaluating
    ``field_fn(x, y, z, t) -> (Ex,Ey,Ez,Bx,By,Bz)`` once per step at the
    start-of-step position/time (the primary-source algorithm's own
    ``E_k = E(x_k)``, ``B_k = B(x_k)`` convention). Returns the final
    ``(x, y, z, ux, uy, uz)``.

    ``use_compile=True`` (default) drives the step through the injector's
    own compiled ``_fused_step`` (the same function object, not a copy --
    shared rather than duplicated). The injector's applied-field tables are
    empty here: every applied field arrives through ``field_fn``, evaluated
    once per step outside the compiled region (the injector's own split
    with its space-charge cache), so the compiled kernel is the
    relativistic Boris push alone with Kahan-compensated state -- same
    result as ``boris_push``, differing only in float rounding order.
    ``use_compile=False`` is the original eager ``boris_push`` loop.
    Unlike ``Injector.track`` there is no ``device == "cuda"`` gate: this
    path compiles wherever asked (CPU compilation is a parity feature here,
    not a speed one -- the fused kernel can be slower per step than eager
    on CPU)."""
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

    # Lazy import: injector.py imports this module at module level
    # (kahan_add, relativistic_space_charge_field), so a top-level import of
    # _fused_step here would be circular.
    from .injector import _fused_step

    # Mirror of Injector.track's step_fn gate (injector.py:498): compile the
    # shared step rather than duplicating it. No recompile_limit guard here --
    # that guard exists because _fused_sc_fields takes (nr, nz) grid shapes
    # that change between solver rebuilds (injector.py:487-497), and this
    # driver has no space-charge mesh at all (any SC lives inside the
    # caller's field_fn, evaluated eagerly): the only tensor shapes in the
    # call are the fixed (n,) particle arrays, so one graph serves the run.
    step_fn = torch.compile(_fused_step, dynamic=False)

    n = x.shape[0]
    zeros_n = torch.zeros_like(x)
    # Kahan-compensated state, positions AND momenta -- the same layout
    # Injector.track carries across steps (injector.py:464); it is what
    # makes the compiled path float32-safe where the eager loop above needs
    # the caller to opt in via boris_push(comp=...).
    cx, cy, cz, cux, cuy, cuz = (zeros_n.clone() for _ in range(6))
    sw, tw, sol, quad = (), (), (), ()  # no injector elements -- field_fn is the field
    # Same 0-d float64 device-tensor time convention as Injector.track
    # (injector.py:479); unused with empty sw/tw/sol (no element phase to
    # evaluate) but part of the shared signature, so keep the convention.
    t_ten = torch.full((), t0, device=x.device, dtype=torch.float64)
    qmdt2 = q_C / mass_kg * (dt / 2.0)
    c2_inv = 1.0 / C_LIGHT**2
    # No emission model in this driver -- every particle is live from step
    # 0, so pass emitted_all=True and _fused_step skips the
    # where(active, ...) merge (it is a Python bool, fixed per variant).
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
    """Space-charge self-field with the quasi-static relativistic boost
    (see module docstring): stretch ``z`` by ``bunch_gamma`` about a
    fixed reference (``cathode_z`` if given, else the bunch's own z-mean
    -- the choice of reference point is immaterial, since it cancels in
    every particle-pair separation the field actually depends on),
    solve electrostatics in ``solver`` (unmodified Phase-2 kernel), then
    transform the result back to the lab frame. ``solver``'s grid must
    be sized to hold the *stretched* z-extent (and, with a cathode, the
    stretched image lobe too) -- caller's responsibility, same as
    ``field_with_cathode_image``'s own grid-sizing note. ``smooth_r``/
    ``smooth_z``/``smooth_gamma_ref`` -- see ``CylindricalSpaceCharge.solve``;
    ``bunch_gamma`` doubles as that blend's ``gamma``, already the exact
    quantity the fade needs.
    """
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

