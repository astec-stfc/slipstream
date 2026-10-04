"""Represents selected frozen-energy ring coordinates as float32 high/low pairs for increased
precision. See docs/modules/tracking/native/double_single.md."""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import torch

_SPLIT = 4097.0  # 2**12 + 1: Dekker split for a 24-bit mantissa


def _f32(c: float) -> float:
    return float(np.float32(c))


def _two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def _fast_two_sum(a, b):  # |a| >= |b|
    s = a + b
    return s, b - (s - a)


def _split(a):
    t = a * _SPLIT
    hi = t - (t - a)
    return hi, a - hi


def _split_const(c: float) -> tuple[float, float]:
    hi = _f32(c)
    lo = c - hi
    t = _f32(hi * _SPLIT)
    hh = _f32(t - _f32(t - hi))
    return hh, _f32(hi - hh), _f32(lo)


def _two_prod(a, b):
    p = a * b
    ah, al = _split(a)
    bh, bl = _split(b)
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


class DS(NamedTuple):
    """``hi + lo``; operators accept ``DS`` or a Python float (split into
    its own double-single constant, so constants keep ~44 bits too)."""

    hi: torch.Tensor
    lo: torch.Tensor

    @staticmethod
    def of(x: torch.Tensor) -> "DS":
        """From a float64 tensor."""
        hi = x.to(torch.float32)
        return DS(hi, (x - hi.to(x.dtype)).to(torch.float32))

    def double(self) -> torch.Tensor:
        return self.hi.double() + self.lo.double()

    def __neg__(self) -> "DS":
        return DS(-self.hi, -self.lo)

    def __add__(self, o) -> "DS":
        if not isinstance(o, DS):
            ch = _f32(o)
            s, e = _two_sum(self.hi, ch)
            e = e + (self.lo + _f32(o - ch))
            return DS(*_fast_two_sum(s, e))
        s, e = _two_sum(self.hi, o.hi)
        t, f = _two_sum(self.lo, o.lo)
        s, e = _fast_two_sum(s, e + t)
        return DS(*_fast_two_sum(s, e + f))

    __radd__ = __add__

    def __sub__(self, o) -> "DS":
        return self + (-o)

    def __rsub__(self, o) -> "DS":
        return (-self) + o

    def __mul__(self, o) -> "DS":
        if not isinstance(o, DS):
            chh, chl, cl = _split_const(o)
            ch = chh + chl  # exact in float32
            p = self.hi * ch
            ah, al = _split(self.hi)
            e = ((ah * chh - p) + ah * chl + al * chh) + al * chl
            e = e + (self.hi * cl + self.lo * ch)
            return DS(*_fast_two_sum(p, e))
        p, e = _two_prod(self.hi, o.hi)
        e = e + (self.hi * o.lo + self.lo * o.hi)
        return DS(*_fast_two_sum(p, e))

    __rmul__ = __mul__

    def __truediv__(self, o) -> "DS":
        if not isinstance(o, DS):
            return self * (1.0 / o)
        q1 = self.hi / o.hi
        r = self - o * DS(q1, torch.zeros_like(q1))
        q2 = r.hi / o.hi
        r = r - o * DS(q2, torch.zeros_like(q2))
        q3 = r.hi / o.hi
        q1, q2 = _fast_two_sum(q1, q2)
        return DS(q1, q2) + DS(q3, torch.zeros_like(q3))

    def __rtruediv__(self, o: float) -> "DS":
        one = DS(torch.full_like(self.hi, _f32(o)), torch.full_like(self.hi, _f32(o - _f32(o))))
        return one / self


def ds_sqrt(x: DS) -> DS:
    s = torch.sqrt(x.hi)
    p, e = _two_prod(s, s)
    r = (x.hi - p - e + x.lo) / (2 * s)
    return DS(*_fast_two_sum(s, r))


def ds_asin_small(u: DS) -> DS:
    """``asin(u)`` for ``|u| < ~0.05`` by its Taylor series (to ``u**9``;
    truncation < 1e-13 relative there). The exact dipole's argument is
    ``~ |px - px_new| ~ 1e-4``."""
    u2 = u * u
    return u * (1.0 + u2 * (1.0 / 6 + u2 * (3.0 / 40 + u2 * (5.0 / 112 + u2 * (35.0 / 1152)))))


def ds_tan_small(u: DS) -> DS:
    """``tan(u)`` for small ``|u|`` (edge ``psi``, ~1e-2) -- series to ``u**9``."""
    u2 = u * u
    return u * (1.0 + u2 * (1.0 / 3 + u2 * (2.0 / 15 + u2 * (17.0 / 315 + u2 * (62.0 / 2835)))))


def _exact_dipole_drift(x: DS, px: DS, y: DS, py: DS, delta: DS, ds: float, h: float):
    """``elements._exact_dipole_drift`` in double-single -- see its docstring."""
    c, s = math.cos(ds * h), math.sin(ds * h)
    cm1 = -2.0 * math.sin(ds * h / 2.0) ** 2
    rho = 1.0 / h
    opd2 = (delta + 1.0) * (delta + 1.0)
    dd = delta * (delta + 2.0)
    pz = ds_sqrt(opd2 - px * px - py * py)
    w = (dd - px * px - py * py) / (pz + 1.0)
    px_new = px + (px * cm1 + (w - x * h) * s)
    pz_new = ds_sqrt(opd2 - px_new * px_new - py * py)
    w_new = (dd - px_new * px_new - py * py) / (pz_new + 1.0)
    x_new = x + (x * cm1 + (w_new - w * c + px * s) * rho)
    alpha = ds_asin_small((px * pz_new - px_new * pz) / (opd2 - py * py)) + ds * h
    return x_new, px_new, y + py * alpha * rho


def _straight_drift(x: DS, px: DS, y: DS, py: DS, delta: DS, ds: float):
    inv_pz = 1.0 / ds_sqrt((delta + 1.0) * (delta + 1.0) - px * px - py * py)
    return x + px * inv_pz * ds, y + py * inv_pz * ds


def _kick(x: DS, px: DS, y: DS, py: DS, k1: float, k2: float, ds: float, h: float):
    """K1/K2-only part of ``elements._multipole_kick`` (dipole term lives in
    the exact dipole drift)."""
    fy = x * k1
    fx = y * k1
    if k2 != 0.0:
        fy = fy + (x * x - y * y) * (0.5 * k2)
        fx = fx + x * y * k2
    weight = (x * h + 1.0) * ds if h != 0.0 else None
    if weight is None:
        return px - fy * ds, py + fx * ds
    return px - weight * fy, py + weight * fx


def _edge(x: DS, px: DS, y: DS, py: DS, delta: DS, h: float, e: float, hgap: float, fint: float):
    """``elements._edge_kick`` with ``edge_order=1`` in double-single."""
    if e == 0.0 and hgap * fint == 0.0:
        return px, py
    opd = delta + 1.0
    pz = ds_sqrt(opd * opd - px * px - py * py)
    xp, yp = px / pz, py / pz
    he = h / opd
    t = math.tan(e)
    xp_new = xp + he * x * t
    if hgap * fint == 0.0:
        yp_new = yp - he * y * t
    else:
        sin_e, cos_e = math.sin(e), math.cos(e)
        tan_psi = ds_tan_small((2 * hgap * fint * h * (1 + sin_e * sin_e) / cos_e) / opd)
        yp_new = yp - he * y * ((t - tan_psi) / (tan_psi * t + 1.0))
    pz_new = opd / ds_sqrt(xp_new * xp_new + yp_new * yp_new + 1.0)
    return xp_new * pz_new, yp_new * pz_new


def _add_small(v: DS, t: torch.Tensor) -> DS:
    """``v + t`` for a plain float32 ``t`` (a small nonlinear remainder)."""
    s, e = _two_sum(v.hi, t)
    return DS(*_fast_two_sum(s, e + v.lo))



def _hy_exact_dipole_drift(x: DS, px: DS, y: DS, py: DS, delta: DS, ds: float, h: float):
    c, s = math.cos(ds * h), math.sin(ds * h)
    cm1 = -2.0 * math.sin(ds * h / 2.0) ** 2
    rho = 1.0 / h
    d = delta.hi
    opd = 1 + d
    pxh, pyh = px.hi, py.hi
    p2 = pxh * pxh + pyh * pyh
    pz = torch.sqrt(opd * opd - p2)
    r = -p2 / (pz + opd)  # w - delta, w = pz - 1
    px_new = _add_small(px + (px * cm1 + delta * s - x * (h * s)), r * s)
    pxnh = px_new.hi
    p2n = pxnh * pxnh + pyh * pyh
    pz_new = torch.sqrt(opd * opd - p2n)
    r_new = -p2n / (pz_new + opd)
    x_new = _add_small(x + (x * cm1 + delta * (rho * (1 - c)) + px * (rho * s)), rho * (r_new - r * c))
    u = (pxh * pz_new - pxnh * pz) / (opd * opd - pyh * pyh)
    y_new = _add_small(y + py * ds, pyh * rho * torch.asin(u))
    return x_new, px_new, y_new


def _hy_straight_drift(x: DS, px: DS, y: DS, py: DS, delta: DS, ds: float):
    d = delta.hi
    p2 = px.hi * px.hi + py.hi * py.hi
    pz = torch.sqrt((1 + d) * (1 + d) - p2)
    g = ds * (p2 - d * (2 + d)) / ((1 + pz) * pz)  # ds*(1/pz - 1)
    return _add_small(x + px * ds, px.hi * g), _add_small(y + py * ds, py.hi * g)


def _hy_kick(x: DS, px: DS, y: DS, py: DS, k1: float, k2: float, ds: float, h: float):
    xh, yh = x.hi, y.hi
    fy = 0.5 * k2 * (xh * xh - yh * yh)
    fx = k2 * xh * yh
    if h != 0.0:
        fy = fy + h * xh * (k1 * xh + fy)
        fx = fx + h * xh * (k1 * yh + fx)
    if k1 != 0.0:
        return _add_small(px - x * (ds * k1), -ds * fy), _add_small(py + y * (ds * k1), ds * fx)
    return _add_small(px, -ds * fy), _add_small(py, ds * fx)


def _unpack(s12: torch.Tensor):
    return tuple(DS(s12[..., i], s12[..., 6 + i]) for i in (0, 1, 2, 3, 5))


def _pack(s12: torch.Tensor, x: DS, px: DS, y: DS, py: DS) -> torch.Tensor:
    tau, delta_hi, delta_lo = s12[..., 4], s12[..., 5], s12[..., 11]
    return torch.stack(
        [x.hi, px.hi, y.hi, py.hi, tau, delta_hi, x.lo, px.lo, y.lo, py.lo, torch.zeros_like(tau), delta_lo], -1,
    )


def ds_edge(s12: torch.Tensor, h: float, e: float, hgap: float, fint: float) -> torch.Tensor:
    """Linear bend edge on a ``(..., 12)`` hi/lo state (6 hi then 6 lo
    columns; ``tau`` is not tracked)."""
    x, px, y, py, delta = _unpack(s12)
    px, py = _edge(x, px, y, py, delta, h, e, hgap, fint)
    return _pack(s12, x, px, y, py)


def ds_slice(
    s12: torch.Tensor, dl: float, h: float, k1: float, k2: float, drift_frac: tuple, kick_frac: tuple,
    hybrid: bool = False,
) -> torch.Tensor:
    """One Yoshida slice of length ``dl`` (exact dipole / straight drift + K1/K2 kicks) on a ``(...,
    12)`` hi/lo state."""
    x, px, y, py, delta = _unpack(s12)
    dipole, straight, kick = (
        (_hy_exact_dipole_drift, _hy_straight_drift, _hy_kick) if hybrid
        else (_exact_dipole_drift, _straight_drift, _kick)
    )
    for df, kf in zip(drift_frac, kick_frac):
        if h != 0.0:
            x, px, y = dipole(x, px, y, py, delta, dl * df, h)
        else:
            x, y = straight(x, px, y, py, delta, dl * df)
        if kf != 0.0:
            px, py = kick(x, px, y, py, k1, k2, dl * kf, h)
    return _pack(s12, x, px, y, py)


def track_element_ds(
    s12: torch.Tensor, length: float, angle: float, k1: float, k2: float, e1: float, e2: float,
    n_kicks: int, drift_frac: tuple, kick_frac: tuple, hgap: float = 0.0, fint: float = 0.5,
    edge_order: int = 1, edge1_effects: int = 1, edge2_effects: int = 1,
    slice_fn=ds_slice, edge_fn=ds_edge,
) -> torch.Tensor:
    """``track_element(..., exact_dipole=True)`` on a ``(..., 12)`` hi/lo
    state. ``slice_fn``/``edge_fn``: compiled ``ds_slice``/``ds_edge``."""
    if length <= 0.0:
        return s12
    if angle != 0.0 and edge_order != 1 and (edge1_effects or edge2_effects):
        raise NotImplementedError("double-single edges: EDGE_ORDER=1 only")
    h = angle / length if angle != 0.0 else 0.0
    if h != 0.0 and edge1_effects:
        s12 = edge_fn(s12, h, e1, hgap, fint)
    if k1 == 0.0 and k2 == 0.0:  # pure drift / pure dipole: exact in one step
        s12 = slice_fn(s12, length, h, 0.0, 0.0, (1.0,), (0.0,))
    else:
        for _ in range(n_kicks):
            s12 = slice_fn(s12, length / n_kicks, h, k1, k2, drift_frac, kick_frac)
    if h != 0.0 and edge2_effects:
        s12 = edge_fn(s12, h, e2, hgap, fint)
    return s12


def _ds_ops(ah, al, bh, bl):
    A, B = DS(ah, al), DS(bh, bl)
    r = [A + B, A * B, A / B, ds_sqrt(A), A * 0.1234567890123, A - B * 3.0]
    return torch.stack([v.hi for v in r]), torch.stack([v.lo for v in r])


def check_error_free(fn=None, device: str | torch.device = "cpu") -> float:
    """Compute check error free; see the module documentation for details."""
    g = torch.Generator().manual_seed(0)
    a = (torch.rand(4096, generator=g, dtype=torch.float64) + 0.5).to(device)
    b = (torch.rand(4096, generator=g, dtype=torch.float64) + 0.5).to(device)

    A, B = DS.of(a), DS.of(b)
    hi, lo = (fn or _ds_ops)(A.hi, A.lo, B.hi, B.lo)
    got = hi.double() + lo.double()
    ref = torch.stack([a + b, a * b, a / b, torch.sqrt(a), a * 0.1234567890123, a - b * 3.0])
    return ((got - ref).abs() / ref.abs().clamp_min(1e-300)).max().item()
