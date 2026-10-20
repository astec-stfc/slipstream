"""Evaluates lab-frame static dipole, solenoid and quadrupole fields, including element
offsets/rotations. See docs/modules/tracking/native/magnets.md."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.interpolate import CubicSpline

from .fields import _scale_to_peak

FRINGE_ZONE = 1.5  # ASTRA: the Fermi fringe is cut off at +-1.5 D_Gap around each dipole edge
# Paraxial element fields (solenoid/quadrupole tables) are only meaningful near their own axis.
LATERAL_LIMIT_M = 0.25


@dataclass
class Dipole:
    """Describe Dipole; see the module documentation for details."""
    corners: list[list[float]]
    strength_T: float
    gap1_m: float = 0.05
    gap2_m: float = 0.05



def placement(x_m: float, z_m: float, angle_rad: float, force: bool = False):
    """Compute placement; see the module documentation for details."""
    if x_m == 0.0 and angle_rad == 0.0 and not force:
        return None
    return (x_m, z_m, math.cos(angle_rad), math.sin(angle_rad))


def placement3(x_m: float, y_m: float, z_m: float, xrot_rad: float, yrot_rad: float, force: bool = False):
    """Compute placement3; see the module documentation for details."""
    if not (x_m or y_m or xrot_rad or yrot_rad or force):
        return None
    ca, sa, cb, sb = math.cos(yrot_rad), math.sin(yrot_rad), math.cos(xrot_rad), math.sin(xrot_rad)
    return (x_m, y_m, z_m, (cb, 0.0, sb, -sa * sb, ca, sa * cb, -ca * sb, -sa, ca * cb))


def place_as_tensors(place, device, dtype):
    """``placement``/``placement3`` with every number a 0-d tensor (``None`` stays ``None``). A Python float handed to a
    compiled step is a compile-time constant, so every new offset or tilt of a scan or fit would recompile it; a tensor
    is an input of the compiled graph and any value reuses it."""
    if place is None:
        return None
    if isinstance(place, tuple):
        return tuple(place_as_tensors(v, device, dtype) for v in place)
    return torch.as_tensor(place, device=device, dtype=dtype)


def to_local3(x, y, z, place):
    """``to_local`` for a ``placement3`` frame: lab ``(x, y, z)`` -> ``(x_l, y_l, z_equivalent)``."""
    x_m, y_m, z_m, (r00, r01, r02, r10, r11, r12, r20, r21, r22) = place
    dx, dy, dz = x - x_m, y - y_m, z - z_m
    return (r00 * dx + r10 * dy + r20 * dz, r01 * dx + r11 * dy + r21 * dz, z_m + r02 * dx + r12 * dy + r22 * dz)


def to_lab3(bx, by, bz, place):
    """Element-frame field -> lab frame for a ``placement3`` frame."""
    r00, r01, r02, r10, r11, r12, r20, r21, r22 = place[3]
    return r00 * bx + r01 * by + r02 * bz, r10 * bx + r11 * by + r12 * bz, r20 * bx + r21 * by + r22 * bz


def lateral_ok(xl, y):
    """1 within ``LATERAL_LIMIT_M`` of an element's own axis, else 0 (multiplies its field)."""
    return ((xl * xl + y * y) < LATERAL_LIMIT_M**2).to(xl.dtype)


def to_local(x, z, place):
    """Lab ``(x, z)`` -> ``(x_local, z_equivalent)``: the point's transverse
    coordinate in the element frame and its axial coordinate re-expressed on the
    *unrotated* element's own ``z`` scale (centre stays at ``z_m``)."""
    if place is None:
        return x, z
    x_m, z_m, c, s = place
    dx, dz = x - x_m, z - z_m
    return dx * c - dz * s, z_m + dx * s + dz * c


def to_lab(bx_local, bz_local, place):
    """Element-frame ``(B_x, B_z)`` -> lab frame (the inverse rotation)."""
    if place is None:
        return bx_local, bz_local
    _, _, c, s = place
    return bx_local * c + bz_local * s, -bx_local * s + bz_local * c


def lerp_tables(z, z0, dz, ntab, zend, *tables):
    """Linear lookup of uniform on-axis tables at ``z``, zero outside
    ``[z0, zend]`` (``OnAxisFieldMap.__call__`` without the object, so it fuses
    into the compiled kernels)."""
    zi = (z - z0) / dz
    i0 = torch.clamp(zi.floor().long(), 0, ntab - 2)
    fr = (zi - i0).clamp(0.0, 1.0)
    inside = (z >= z0) & (z <= zend)
    return [torch.where(inside, tab[i0] * (1 - fr) + tab[i0 + 1] * fr, 0.0) for tab in tables]



def dipole_element(d: Dipole) -> tuple:
    p = np.asarray(d.corners, dtype=float)
    if p.shape != (4, 2):
        raise ValueError(f"a dipole needs four (x, z) corners, got shape {p.shape}")
    centre = p.mean(axis=0)

    def inward(a, b):
        t = b - a
        n = np.array([-t[1], t[0]]) / np.hypot(*t)
        return tuple(n if (centre - a) @ n > 0.0 else -n)

    edge1 = (tuple(p[0]), inward(p[0], p[1]), 4.0 / d.gap1_m, FRINGE_ZONE * d.gap1_m)
    edge2 = (tuple(p[2]), inward(p[2], p[3]), 4.0 / d.gap2_m, FRINGE_ZONE * d.gap2_m)
    sides = ((tuple(p[0]), inward(p[0], p[2])), (tuple(p[1]), inward(p[1], p[3])))
    return ("dipole", None, (d.strength_T, edge1, edge2, sides))


def solenoid_element(map_f, map_df, map_d2f, z0, dz, ntab, zend, place) -> tuple:
    """A *placed* solenoid (any of ``x_m``/``y_m``/``xrot_rad``/``yrot_rad`` non-zero): the same on-axis
    tables ``injector._field_tables`` builds, plus the ``placement3`` element frame."""
    return ("solenoid", place, (map_f, map_df, map_d2f, z0, dz, ntab, zend))


def profile_quad_element(z: np.ndarray, profile: np.ndarray, grad_T_per_m: float, noscale: bool,
                         multipoles: dict[int, float], place, device, dtype, n_samples: int = 4001) -> tuple:
    """A quadrupole with an on-axis gradient table (ASTRA ``Q_type`` data file).
    ``multipoles`` maps order ``n >= 3`` to ``C_n`` (``Q_grad * Q_mult_b(n)``,
    T/m^(n-1)); they are uniform over ``[z[0], z[-1]]``."""
    spline = CubicSpline(z, profile)
    zg = np.linspace(z[0], z[-1], n_samples)
    # g = -Q_grad: electron-focusing positive.
    scale = -(1.0 if noscale else _scale_to_peak(torch.as_tensor(profile), grad_T_per_m))

    def table(a):
        return torch.as_tensor(scale * a, device=device, dtype=dtype)

    payload = (table(spline(zg)), table(spline(zg, 1)), table(spline(zg, 2)),
               float(zg[0]), float(zg[1] - zg[0]), n_samples, float(zg[-1]), tuple(sorted(multipoles.items())))
    return ("quad_table", place, payload)


def hard_quad_element(grad_T_per_m: float, z0: float, zend: float, place) -> tuple:
    """A hard-edge quadrupole in a placed frame (an unplaced one uses ``injector``'s
    own ``quad`` tuples)."""
    return ("quad_hard", place, (-grad_T_per_m, z0, zend))



def _fringe(d, k, zone):
    """ASTRA's dipole edge function ``F(d) = 1/(1+exp(k d))`` (``k = 4/D_Gap``,
    ``d`` the distance outside the edge), its first and second derivative in
    ``d``, cut off as ASTRA does beyond ``|d| > zone``."""
    f = torch.sigmoid(-k * d)
    inside = d.abs() <= zone
    step = (d < 0.0).to(d.dtype)
    g = f * (1.0 - f)
    return (torch.where(inside, f, step),
            torch.where(inside, -k * g, torch.zeros_like(d)),
            torch.where(inside, k * k * g * (1.0 - 2.0 * f), torch.zeros_like(d)))


def _dipole_field(x, y, z, payload):
    b0, edge1, edge2, sides = payload
    (p1, n1, k1, zone1), (p2, n2, k2, zone2) = edge1, edge2
    d1 = -((x - p1[0]) * n1[0] + (z - p1[1]) * n1[1])
    d2 = -((x - p2[0]) * n2[0] + (z - p2[1]) * n2[1])
    f1, df1, ddf1 = _fringe(d1, k1, zone1)
    f2, df2, ddf2 = _fringe(d2, k2, zone2)
    profile = f1 + f2 - 1.0
    dp_dx = -(df1 * n1[0] + df2 * n2[0])
    dp_dz = -(df1 * n1[1] + df2 * n2[1])
    laplacian = ddf1 + ddf2
    inside = torch.ones_like(x, dtype=torch.bool)
    for (q, m) in sides:
        inside = inside & (((x - q[0]) * m[0] + (z - q[1]) * m[1]) >= 0.0)
    scale = b0 * inside.to(x.dtype)
    return scale * y * dp_dx, scale * (profile - 0.5 * y * y * laplacian), scale * y * dp_dz


def _solenoid_field(x, y, z, place, payload):
    f, df, d2f, z0, dz, ntab, zend = payload
    xl, yl, zl = (x, y, z) if place is None else to_local3(x, y, z, place)
    bz0, dbz0, d2bz0 = lerp_tables(zl, z0, dz, ntab, zend, f, df, d2f)
    br_over_r = -0.5 * dbz0
    ok = 1.0 if place is None else lateral_ok(xl, yl)
    bx_l, by_l, bz_l = ok * br_over_r * xl, ok * br_over_r * yl, ok * (bz0 - 0.25 * (xl * xl + yl * yl) * d2bz0)
    return (bx_l, by_l, bz_l) if place is None else to_lab3(bx_l, by_l, bz_l, place)


def _quad_table_field(x, y, z, place, payload):
    g, dg, d2g, z0, dz, ntab, zend, multipoles = payload
    xl, zl = to_local(x, z, place)
    g0, g1, g2 = lerp_tables(zl, z0, dz, ntab, zend, g, dg, d2g)
    bx_l = g0 * y - g2 * (3.0 * xl * xl * y + y * y * y) / 12.0
    by = g0 * xl - g2 * (xl * xl * xl + 3.0 * xl * y * y) / 12.0
    bz_l = g1 * xl * y
    ok = 1.0 if place is None else lateral_ok(xl, y)
    bx_l, by, bz_l = ok * bx_l, ok * by, ok * bz_l
    if multipoles:
        inside = ((zl >= z0) & (zl <= zend)).to(x.dtype)
        for order, coeff in multipoles:
            re, im = xl, y  # (x + iy)^(n-1)
            for _ in range(order - 2):
                re, im = re * xl - im * y, re * y + im * xl
            by = by - ok * coeff * inside * re
            bx_l = bx_l - ok * coeff * inside * im
    bx, bz = to_lab(bx_l, bz_l, place)
    return bx, by, bz


def corrector_element(bx_T, by_T, z0, zend) -> tuple:
    """A steering magnet: a uniform transverse field ``(bx_T, by_T)`` over ``z0 <= z <= zend``, in the lab frame. The
    field strengths are 0-d tensors (inputs of the compiled step: changing them does not recompile)."""
    return ("corrector", None, (bx_T, by_T, z0, zend))


def _corrector_field(x, z, payload):
    bx, by, z0, zend = payload
    inside = ((z >= z0) & (z <= zend)).to(x.dtype)
    return bx * inside, by * inside, torch.zeros_like(x)


def _hard_quad_field(x, y, z, place, payload):
    g, z0, zend = payload
    xl, zl = to_local(x, z, place)
    inside = ((zl >= z0) & (zl <= zend)).to(x.dtype)
    if place is not None:
        inside = inside * lateral_ok(xl, y)
    bx, bz = to_lab(g * y * inside, torch.zeros_like(x), place)
    return bx, g * xl * inside, bz


def magnet_fields(x, y, z, elements):
    """Sum of every listed element's magnetic field at the lab points ``(x, y, z)``: ``(B_x, B_y,
    B_z)``."""
    bx = by = bz = torch.zeros_like(x)
    for kind, place, payload in elements:
        if kind == "dipole":
            fx, fy, fz = _dipole_field(x, y, z, payload)
        elif kind == "solenoid":
            fx, fy, fz = _solenoid_field(x, y, z, place, payload)
        elif kind == "quad_table":
            fx, fy, fz = _quad_table_field(x, y, z, place, payload)
        elif kind == "quad_hard":
            fx, fy, fz = _hard_quad_field(x, y, z, place, payload)
        elif kind == "corrector":
            fx, fy, fz = _corrector_field(x, z, payload)
        else:  # pragma: no cover - programmer error
            raise ValueError(f"unknown magnet kind {kind!r}")
        bx, by, bz = bx + fx, by + fy, bz + fz
    return bx, by, bz



def beam_angle(ux, uz):
    """Direction of the bunch's mean momentum in the x-z plane, measured from
    ``+z`` toward ``+x`` (radians, 0-d tensor)."""
    return torch.atan2(ux.mean(), uz.mean())


def rotate_to_beam(x, z, ux, uz, theta):
    """Rotate lab ``(x, z)`` and ``(u_x, u_z)`` into the frame whose ``z'`` axis
    points along the beam direction ``theta`` -- the frame ``ASTRA``/``GPT``
    users mean by "the beam's own x, y, z" after a bend."""
    c, s = torch.cos(theta), torch.sin(theta)
    return x * c - z * s, x * s + z * c, ux * c - uz * s, ux * s + uz * c

