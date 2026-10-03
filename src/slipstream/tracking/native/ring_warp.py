"""Whole-turn Warp backend for the forward-only ``RingMap`` fast path.

This first slice accepts canonical/exact-dipole drift and multipole kicks,
first- and second-order edge maps, and the existing flat-space misalignment
transform. RF remains on the PyTorch fallback. One launch advances every
particle through every packed element; it never crosses the PyTorch/Warp
boundary per element.
"""
from __future__ import annotations

import math
from typing import Any, NamedTuple

import torch

from . import warp_kernels as wk
from .elements import _DRIFT_FRAC4, _DRIFT_FRAC6, _DRIFT_FRAC8, _KICK_FRAC4, _KICK_FRAC6, _KICK_FRAC8

_MAX_STAGES = 22  # order-8 Yoshida: 3*8-2 after the order-6 triple jump


class PackedRing(NamedTuple):
    length: torch.Tensor
    angle: torch.Tensor
    k1: torch.Tensor
    k2: torch.Tensor
    e1: torch.Tensor
    e2: torch.Tensor
    dx: torch.Tensor
    dy: torch.Tensor
    tilt: torch.Tensor
    hgap: torch.Tensor
    fint: torch.Tensor
    n_kicks: torch.Tensor
    n_stages: torch.Tensor
    edge_order: torch.Tensor
    edge1_effects: torch.Tensor
    edge2_effects: torch.Tensor
    drift_ds: torch.Tensor
    kick_ds: torch.Tensor
    h: torch.Tensor
    rho: torch.Tensor
    cos_step: torch.Tensor
    sin_step: torch.Tensor
    cos_half: torch.Tensor
    sin_half: torch.Tensor
    offset: torch.Tensor
    exact_dipole: torch.Tensor


class PreparedTurn(NamedTuple):
    state: torch.Tensor
    arrays: tuple[Any, ...]


def _plain(value) -> float:
    if isinstance(value, torch.Tensor):
        if value.ndim != 0 or value.requires_grad:
            raise TypeError("RingMap Warp path requires scalar, non-gradient strengths")
        return float(value.detach().cpu())
    return float(value)


def pack(params, device: torch.device, dtype: torch.dtype) -> PackedRing | None:
    """Pack a fixed lattice once; return ``None`` for unsupported physics."""
    if not params:
        return None
    for p in params:
        if p.is_rf or p.edge_order not in (1, 2) or p.edge1_effects not in (0, 1) or p.edge2_effects not in (0, 1):
            return None
        for strength in (p.k1, p.k2):
            if isinstance(strength, torch.Tensor) and (strength.ndim != 0 or strength.requires_grad):
                return None
        if p.integration_order not in (4, 6, 8) or p.n_kicks < 1 or p.length <= 0.0:
            return None

    order_tables = {
        4: (_DRIFT_FRAC4, _KICK_FRAC4),
        6: (_DRIFT_FRAC6, _KICK_FRAC6),
        8: (_DRIFT_FRAC8, _KICK_FRAC8),
    }
    rows = []
    for p in params:
        drift_frac, kick_frac = order_tables[p.integration_order]
        length = _plain(p.length)
        angle = _plain(p.angle)
        h = angle / length if angle != 0.0 else 0.0
        ds0 = length / p.n_kicks
        drift = [0.0] * _MAX_STAGES
        kick = [0.0] * _MAX_STAGES
        c = [1.0] * _MAX_STAGES
        s = [0.0] * _MAX_STAGES
        c2 = [1.0] * _MAX_STAGES
        s2 = [0.0] * _MAX_STAGES
        for i, (df, kf) in enumerate(zip(drift_frac, kick_frac)):
            drift[i] = ds0 * df
            kick[i] = ds0 * kf
            c[i], s[i] = math.cos(ds0 * df * h), math.sin(ds0 * df * h)
            c2[i], s2[i] = math.cos(ds0 * df * h / 2.0), math.sin(ds0 * df * h / 2.0)
        rows.append((
            length, angle, _plain(p.k1), _plain(p.k2), _plain(p.e1), _plain(p.e2),
            _plain(p.dx), _plain(p.dy), _plain(p.tilt), _plain(p.hgap), _plain(p.fint),
            p.n_kicks, len(drift_frac), p.edge_order, p.edge1_effects, p.edge2_effects,
            drift, kick, h, 1.0 / h if h else 0.0, c, s, c2, s2,
            p.offset.detach().to(device=device, dtype=dtype), int(bool(p.exact_dipole)),
        ))

    def col(index, dt=dtype):
        return torch.tensor([r[index] for r in rows], device=device, dtype=dt)

    return PackedRing(
        col(0), col(1), col(2), col(3), col(4), col(5), col(6), col(7), col(8), col(9), col(10),
        col(11, torch.int32), col(12, torch.int32), col(13, torch.int32), col(14, torch.int32),
        col(15, torch.int32),
        torch.tensor([r[16] for r in rows], device=device, dtype=dtype),
        torch.tensor([r[17] for r in rows], device=device, dtype=dtype),
        col(18), col(19),
        torch.tensor([r[20] for r in rows], device=device, dtype=dtype),
        torch.tensor([r[21] for r in rows], device=device, dtype=dtype),
        torch.tensor([r[22] for r in rows], device=device, dtype=dtype),
        torch.tensor([r[23] for r in rows], device=device, dtype=dtype),
        torch.stack([r[24] for r in rows]), col(25, torch.int32),
    )


if wk._HAVE_WARP:

    @wk.wp.kernel
    def _turn_kernel(
        state: wk.wp.array(dtype=Any), out: wk.wp.array(dtype=Any),
        length: wk.wp.array(dtype=Any), angle: wk.wp.array(dtype=Any),
        k1: wk.wp.array(dtype=Any), k2: wk.wp.array(dtype=Any),
        e1: wk.wp.array(dtype=Any), e2: wk.wp.array(dtype=Any),
        dx: wk.wp.array(dtype=Any), dy: wk.wp.array(dtype=Any), tilt: wk.wp.array(dtype=Any),
        hgap: wk.wp.array(dtype=Any), fint: wk.wp.array(dtype=Any),
        n_kicks: wk.wp.array(dtype=wk.wp.int32), n_stages: wk.wp.array(dtype=wk.wp.int32),
        edge_order: wk.wp.array(dtype=wk.wp.int32),
        edge1_effects: wk.wp.array(dtype=wk.wp.int32), edge2_effects: wk.wp.array(dtype=wk.wp.int32),
        drift_ds: wk.wp.array(dtype=Any, ndim=2), kick_ds: wk.wp.array(dtype=Any, ndim=2),
        h: wk.wp.array(dtype=Any), rho: wk.wp.array(dtype=Any),
        cos_step: wk.wp.array(dtype=Any, ndim=2), sin_step: wk.wp.array(dtype=Any, ndim=2),
        cos_half: wk.wp.array(dtype=Any, ndim=2), sin_half: wk.wp.array(dtype=Any, ndim=2),
        offset: wk.wp.array(dtype=Any, ndim=2), exact_dipole: wk.wp.array(dtype=wk.wp.int32),
        energy: Any, mc2: Any,
    ):
        i = wk.wp.tid()
        six = 6
        zero = energy - energy
        one = energy / energy
        two = one + one
        x = state[six * i] + zero
        px = state[six * i + 1] + zero
        y = state[six * i + 2] + zero
        py = state[six * i + 3] + zero
        tau = state[six * i + 4] + zero
        delta = state[six * i + 5] + zero
        gamma0 = energy / mc2
        p0 = wk.wp.sqrt(gamma0 * gamma0 - one)
        beta0 = p0 / gamma0

        for e in range(length.shape[0]):
            he = h[e]
            bend = he != zero
            beta = p0 * (one + delta) / wk.wp.sqrt(p0 * p0 * (one + delta) * (one + delta) + one)

            if dx[e] != zero or dy[e] != zero or tilt[e] != zero:
                x = x - dx[e]
                y = y - dy[e]
                ct = wk.wp.cos(tilt[e])
                st = wk.wp.sin(tilt[e])
                xr = x * ct + y * st
                yr = -x * st + y * ct
                pxr = px * ct + py * st
                pyr = -px * st + py * ct
                x, y, px, py = xr, yr, pxr, pyr

            if bend and edge1_effects[e] != 0:
                pz = wk.wp.sqrt((one + delta) * (one + delta) - px * px - py * py)
                xp = px / pz
                yp = py / pz
                se = wk.wp.sin(e1[e])
                ce = wk.wp.cos(e1[e])
                psi = two * hgap[e] * fint[e] * h[e] * (one + se * se) / ce
                if edge_order[e] >= 2:
                    psi = psi * (one + delta)
                else:
                    psi = psi / (one + delta)
                x0 = x
                y0 = y
                xp0 = xp
                yp0 = yp
                t = wk.wp.tan(e1[e])
                xp = xp + he * t * x0
                yp = yp - he * wk.wp.tan(e1[e] - psi) * y0
                if edge_order[e] >= 2:
                    sec2 = one / (ce * ce)
                    t2 = t * t
                    h2 = he * he
                    n = -k1[e] / (h[e] * h[e])
                    t211 = -n * h2 * t
                    t233 = h2 * (n + one / two + t2) * t
                    xp = xp + t211 * x0 * x0 + he * t2 * x0 * xp0 + t233 * y0 * y0 - he * t2 * y0 * yp0
                    yp = yp - he * t2 * yp0 * x0 + h2 * (two * n) * t * x0 * y0 - he * sec2 * xp0 * y0
                    x = x0 - he / two * (t2 * x0 * x0 - sec2 * y0 * y0)
                    y = y0 + he * t2 * x0 * y0
                pz_new = (one + delta) / wk.wp.sqrt(one + xp * xp + yp * yp)
                px = xp * pz_new
                py = yp * pz_new

            nk = n_kicks[e]
            for kick_index in range(nk):
                for stage in range(_MAX_STAGES):
                    if stage >= n_stages[e]:
                        continue
                    ds = drift_ds[e, stage]
                    kd = kick_ds[e, stage]
                    if bend and exact_dipole[e] != 0:
                        c = cos_step[e, stage]
                        ss = sin_step[e, stage]
                        pz = wk.wp.sqrt((one + delta) * (one + delta) - px * px - py * py)
                        dd = delta * (two + delta)
                        w = (dd - px * px - py * py) / (pz + one)
                        cm1 = c - one
                        dpx = px * cm1 + (w - he * x) * ss
                        px_new = px + dpx
                        pz_new = wk.wp.sqrt((one + delta) * (one + delta) - px_new * px_new - py * py)
                        w_new = (dd - px_new * px_new - py * py) / (pz_new + one)
                        alpha = ds * he + wk.wp.asin((px * pz_new - px_new * pz) / ((one + delta) * (one + delta) - py * py))
                        x = x + x * cm1 + rho[e] * (w_new - w * c + px * ss)
                        y = y + py / he * alpha
                        px = px_new
                        arc = (one + delta) * rho[e] * alpha
                    elif bend:
                        c = cos_step[e, stage]
                        ss = sin_step[e, stage]
                        c2 = cos_half[e, stage]
                        s2 = sin_half[e, stage]
                        pz = wk.wp.sqrt((one + delta) * (one + delta) - px * px - py * py)
                        pz_theta = pz * c - px * ss
                        factor = (rho[e] + x) * ss / pz_theta
                        g_minus_1 = two * s2 * (px * c2 + pz * s2) / pz_theta
                        x = x + (rho[e] + x) * g_minus_1
                        px = px * c + pz * ss
                        y = y + py * factor
                        arc = factor * (one + delta)
                    else:
                        pz = wk.wp.sqrt((one + delta) * (one + delta) - px * px - py * py)
                        x = x + ds * px / pz
                        y = y + ds * py / pz
                        arc = ds * (one + delta) / pz
                    # Match elements.py's grouped reference-beta correction;
                    # left-to-right evaluation accumulates a visible float32
                    # tau residual over long lattices such as BR14.
                    tau = tau + (arc / beta - ds / beta0)

                    if kd != zero:
                        fy = k1[e] * x + (one / two) * k2[e] * (x * x - y * y)
                        fx = k1[e] * y + k2[e] * x * y
                        if exact_dipole[e] == 0:
                            fy = fy + he
                        weight = kd * (one + he * x)
                        px = px - weight * fy
                        py = py + weight * fx

            if bend and edge2_effects[e] != 0:
                pz = wk.wp.sqrt((one + delta) * (one + delta) - px * px - py * py)
                xp = px / pz
                yp = py / pz
                se = wk.wp.sin(e2[e])
                ce = wk.wp.cos(e2[e])
                psi = two * hgap[e] * fint[e] * h[e] * (one + se * se) / ce
                if edge_order[e] >= 2:
                    psi = psi * (one + delta)
                else:
                    psi = psi / (one + delta)
                x0 = x
                y0 = y
                xp0 = xp
                yp0 = yp
                t = wk.wp.tan(e2[e])
                xp = xp + he * t * x0
                yp = yp - he * wk.wp.tan(e2[e] - psi) * y0
                if edge_order[e] >= 2:
                    sec2 = one / (ce * ce)
                    t2 = t * t
                    h2 = he * he
                    n = -k1[e] / (h[e] * h[e])
                    t211 = -h2 * (n + t2 / two) * t
                    t233 = h2 * (n - t2 / two) * t
                    xp = xp + t211 * x0 * x0 - he * t2 * x0 * xp0 + t233 * y0 * y0 + he * t2 * y0 * yp0
                    yp = yp + he * t2 * yp0 * x0 + h2 * (two * n + sec2) * t * x0 * y0 + he * sec2 * xp0 * y0
                    x = x0 + he / two * (t2 * x0 * x0 - sec2 * y0 * y0)
                    y = y0 - he * t2 * x0 * y0
                pz_new = (one + delta) / wk.wp.sqrt(one + xp * xp + yp * yp)
                px = xp * pz_new
                py = yp * pz_new

            if dx[e] != zero or dy[e] != zero or tilt[e] != zero:
                ct = wk.wp.cos(tilt[e])
                st = wk.wp.sin(tilt[e])
                xr = x * ct - y * st
                yr = x * st + y * ct
                pxr = px * ct - py * st
                pyr = px * st + py * ct
                x, y, px, py = xr + dx[e], yr + dy[e], pxr, pyr

            x = x - offset[e, 0]
            px = px - offset[e, 1]
            y = y - offset[e, 2]
            py = py - offset[e, 3]
            tau = tau - offset[e, 4]
            delta = delta - offset[e, 5]

        out[six * i] = x
        out[six * i + 1] = px
        out[six * i + 2] = y
        out[six * i + 3] = py
        out[six * i + 4] = tau
        out[six * i + 5] = delta

    _turn_f32 = wk.wp.overload(
        _turn_kernel,
        [wk.wp.array(dtype=wk.wp.float32)] * 13
        + [wk.wp.array(dtype=wk.wp.int32)] * 5
        + [wk.wp.array(dtype=wk.wp.float32, ndim=2)] * 2
        + [wk.wp.array(dtype=wk.wp.float32)] * 2
        + [wk.wp.array(dtype=wk.wp.float32, ndim=2)] * 4
        + [wk.wp.array(dtype=wk.wp.float32, ndim=2)]
        + [wk.wp.array(dtype=wk.wp.int32)] + [wk.wp.float32] * 2,
    )
    _turn_f64 = wk.wp.overload(
        _turn_kernel,
        [wk.wp.array(dtype=wk.wp.float64)] * 13
        + [wk.wp.array(dtype=wk.wp.int32)] * 5
        + [wk.wp.array(dtype=wk.wp.float64, ndim=2)] * 2
        + [wk.wp.array(dtype=wk.wp.float64)] * 2
        + [wk.wp.array(dtype=wk.wp.float64, ndim=2)] * 4
        + [wk.wp.array(dtype=wk.wp.float64, ndim=2)]
        + [wk.wp.array(dtype=wk.wp.int32)] + [wk.wp.float64] * 2,
    )


def available(device: torch.device, dtype: torch.dtype, packed: PackedRing | None) -> bool:
    return packed is not None and wk.available(str(device), dtype)


def prepare(state: torch.Tensor, packed: PackedRing) -> PreparedTurn:
    """Create stable zero-copy Warp views before CUDA-graph capture."""
    flat = state.reshape(-1, 6)
    if not flat.is_contiguous():
        raise ValueError("prepared RingMap Warp state must be contiguous")
    wk._ensure_init()
    arrays = tuple(wk._wp_arg(t) for t in (
        flat.reshape(-1), flat.reshape(-1), packed.length, packed.angle, packed.k1, packed.k2,
        packed.e1, packed.e2, packed.dx, packed.dy, packed.tilt, packed.hgap, packed.fint,
        packed.n_kicks, packed.n_stages, packed.edge_order, packed.edge1_effects, packed.edge2_effects,
        packed.drift_ds, packed.kick_ds, packed.h, packed.rho,
        packed.cos_step, packed.sin_step, packed.cos_half, packed.sin_half, packed.offset,
        packed.exact_dipole,
    ))
    return PreparedTurn(flat, arrays)


def turn(
    state: torch.Tensor, packed: PackedRing, energy: float, mc2: float,
    *, out: torch.Tensor | None = None, prepared: PreparedTurn | None = None,
) -> torch.Tensor:
    """Advance a flat particle batch through the complete packed lattice."""
    wk._ensure_init()
    dtype = state.dtype
    kernel = _turn_f64 if dtype == torch.float64 else _turn_f32
    flat = state.reshape(-1, 6).contiguous()
    if out is None:
        out = flat if prepared is not None else torch.empty_like(flat)
    out_flat = out.reshape(-1, 6)
    if out_flat.shape != flat.shape or not out_flat.is_contiguous():
        raise ValueError("RingMap Warp output must be a contiguous state-shaped tensor")
    if prepared is None:
        arrays = tuple(wk._wp_arg(t) for t in (
            flat.reshape(-1), out_flat.reshape(-1), packed.length, packed.angle, packed.k1, packed.k2,
            packed.e1, packed.e2, packed.dx, packed.dy, packed.tilt, packed.hgap, packed.fint,
            packed.n_kicks, packed.n_stages, packed.edge_order, packed.edge1_effects, packed.edge2_effects,
            packed.drift_ds, packed.kick_ds, packed.h, packed.rho,
            packed.cos_step, packed.sin_step, packed.cos_half, packed.sin_half, packed.offset,
            packed.exact_dipole,
        ))
    else:
        if prepared.state.data_ptr() != flat.data_ptr() or out_flat.data_ptr() != flat.data_ptr():
            raise ValueError("prepared RingMap Warp views do not match the state tensor")
        arrays = prepared.arrays
    scalar = wk.wp.float64 if dtype == torch.float64 else wk.wp.float32
    wk.wp.launch(
        kernel, dim=flat.shape[0], inputs=list(arrays) + [scalar(energy), scalar(mc2)],
        stream=wk._launch_stream(flat),
    )
    return out.reshape(state.shape)
