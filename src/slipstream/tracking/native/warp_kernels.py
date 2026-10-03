"""Warp-backed hot kernels for ``tracking.native.injector`` -- CUDA only.
See ``plans/WARP_PORT_PLAN.md`` (Phase 1 spike measurements, Phase 2 this
file). Optional dependency, mirrors ``laura_import.py``'s own
``try: import ...; except ImportError:`` pattern: everything here is a
drop-in accelerator for hot sub-computations of ``injector._fused_step``/
``_fused_sc_fields``/``_apply_apertures``, never load-bearing for
correctness -- the existing torch code in ``injector.py`` is the permanent
fallback and the only CPU implementation. **CPU-float64 pytorch stays the
accuracy baseline** (CLAUDE.md, user decision 2026-09-28): this module is
never used on CPU, ``available()`` always returns ``False`` there.

Every kernel is written once as a *generic* (``Any``-dtype) warp kernel
and instantiated for float32/float64 via ``wp.overload`` rather than as a
hand-duplicated f32/f64 pair (Phase 1's spike did the latter, flagged in
its own Finding 4 as worth fixing). Measured directly: both overloads of a
generic kernel compile from *one* module, vs. two independent modules for
two hand-written kernels -- fewer compiled artifacts, less first-use
latency, one body to keep in sync with the torch reference instead of two.
Warp's literal typing is still strict inside a generic body (a bare
``1.0`` defaults to float32 and cannot mix with a float64 operand --
confirmed empirically, ``Any``/``wp.overload`` doesn't change this), so
typed ``0``/``1``/``2`` constants are derived from an already-typed
argument instead (``push``: ``ONE = dt / dt``, safe since ``dt`` is always
physically nonzero; ``_apertures_kernel``: passed in explicitly as
``zero``/``one`` params instead, since nothing in its own inputs is
guaranteed nonzero the way ``dt`` is)."""
from __future__ import annotations

import functools
from typing import Any

import torch

try:
    import warp as wp
    _HAVE_WARP = True
except ImportError:
    _HAVE_WARP = False

_initialized = False
_stream_cache: dict[tuple[int, int], Any] = {}


def _ensure_init() -> None:
    global _initialized
    if not _initialized:
        wp.init()
        _initialized = True


def _launch_stream(tensor: torch.Tensor):
    """Warp view of PyTorch's active CUDA stream, cached by stream handle."""
    stream = torch.cuda.current_stream(tensor.device)
    key = (stream.device.index or 0, stream.cuda_stream)
    if key not in _stream_cache:
        _stream_cache[key] = wp.stream_from_torch(stream)
    return _stream_cache[key]


def _wp_arg(tensor: torch.Tensor):
    """Low-overhead, forward-only zero-copy kernel argument."""
    return wp.from_torch(tensor, requires_grad=False, return_ctype=True)


def available(device: str, dtype: torch.dtype) -> bool:
    """Whether the warp backend applies: CUDA only, float32/float64 only
    (the two dtypes these kernels are instantiated for)."""
    return _HAVE_WARP and device == "cuda" and dtype in (torch.float32, torch.float64)


if _HAVE_WARP:

    @wp.kernel
    def _push_kernel(
        x: wp.array(dtype=Any), y: wp.array(dtype=Any), z: wp.array(dtype=Any),
        ux: wp.array(dtype=Any), uy: wp.array(dtype=Any), uz: wp.array(dtype=Any),
        cx: wp.array(dtype=Any), cy: wp.array(dtype=Any), cz: wp.array(dtype=Any),
        cux: wp.array(dtype=Any), cuy: wp.array(dtype=Any), cuz: wp.array(dtype=Any),
        Ex: wp.array(dtype=Any), Ey: wp.array(dtype=Any), Ez: wp.array(dtype=Any),
        Bx: wp.array(dtype=Any), By: wp.array(dtype=Any), Bz: wp.array(dtype=Any),
        qmdt2: Any, dt: Any, c2_inv: Any,
        x_n: wp.array(dtype=Any), y_n: wp.array(dtype=Any), z_n: wp.array(dtype=Any),
        ux_n: wp.array(dtype=Any), uy_n: wp.array(dtype=Any), uz_n: wp.array(dtype=Any),
        cx_n: wp.array(dtype=Any), cy_n: wp.array(dtype=Any), cz_n: wp.array(dtype=Any),
        cux_n: wp.array(dtype=Any), cuy_n: wp.array(dtype=Any), cuz_n: wp.array(dtype=Any),
    ):
        """Relativistic Boris push, Kahan-compensated -- transcribed from
        ``injector._fused_step``'s push tail. See that function's docstring
        for why the compensation is physically load-bearing (float32
        ulp-cancellation at CLARA-gun-exit momenta)."""
        ONE = dt / dt
        TWO = ONE + ONE
        i = wp.tid()
        uxm = ux[i] + qmdt2 * Ex[i]
        uym = uy[i] + qmdt2 * Ey[i]
        uzm = uz[i] + qmdt2 * Ez[i]
        g = wp.sqrt(ONE + (uxm * uxm + uym * uym + uzm * uzm) * c2_inv)
        tx = qmdt2 / g * Bx[i]
        ty = qmdt2 / g * By[i]
        tz = qmdt2 / g * Bz[i]
        t2 = tx * tx + ty * ty + tz * tz
        upx = uxm + (uym * tz - uzm * ty)
        upy = uym + (uzm * tx - uxm * tz)
        upz = uzm + (uxm * ty - uym * tx)
        denom = ONE + t2
        sx = TWO * tx / denom
        sy = TWO * ty / denom
        sz = TWO * tz / denom
        ax = TWO * qmdt2 * Ex[i] + (upy * sz - upz * sy)
        ay = TWO * qmdt2 * Ey[i] + (upz * sx - upx * sz)
        az = TWO * qmdt2 * Ez[i] + (upx * sy - upy * sx)

        yux = ax - cux[i]
        tux = ux[i] + yux
        cux_n[i] = (tux - ux[i]) - yux
        ux_n[i] = tux

        yuy = ay - cuy[i]
        tuy = uy[i] + yuy
        cuy_n[i] = (tuy - uy[i]) - yuy
        uy_n[i] = tuy

        yuz = az - cuz[i]
        tuz = uz[i] + yuz
        cuz_n[i] = (tuz - uz[i]) - yuz
        uz_n[i] = tuz

        g_n = wp.sqrt(ONE + (ux_n[i] * ux_n[i] + uy_n[i] * uy_n[i] + uz_n[i] * uz_n[i]) * c2_inv)

        dxp = ux_n[i] / g_n * dt
        yx = dxp - cx[i]
        txp = x[i] + yx
        cx_n[i] = (txp - x[i]) - yx
        x_n[i] = txp

        dyp = uy_n[i] / g_n * dt
        yy = dyp - cy[i]
        typ = y[i] + yy
        cy_n[i] = (typ - y[i]) - yy
        y_n[i] = typ

        dzp = uz_n[i] / g_n * dt
        yz = dzp - cz[i]
        tzp = z[i] + yz
        cz_n[i] = (tzp - z[i]) - yz
        z_n[i] = tzp

    _push_f32 = wp.overload(_push_kernel, [wp.array(dtype=wp.float32)] * 18 + [wp.float32] * 3
                             + [wp.array(dtype=wp.float32)] * 12)
    _push_f64 = wp.overload(_push_kernel, [wp.array(dtype=wp.float64)] * 18 + [wp.float64] * 3
                             + [wp.array(dtype=wp.float64)] * 12)

    @wp.kernel
    def _deposit_kernel(
        idx00: wp.array(dtype=wp.int32), idx01: wp.array(dtype=wp.int32),
        idx10: wp.array(dtype=wp.int32), idx11: wp.array(dtype=wp.int32),
        w00: wp.array(dtype=Any), w01: wp.array(dtype=Any),
        w10: wp.array(dtype=Any), w11: wp.array(dtype=Any),
        q: wp.array(dtype=Any), charge: wp.array(dtype=Any),
    ):
        """CIC scatter-add of ``q`` onto the flat ``(nr, nz)`` charge grid's
        four surrounding corners -- native ``wp.atomic_add`` in place of
        ``injector._fused_sc_fields``'s four chained ``scatter_add_`` calls,
        the kernel Phase 1 found warp winning on at every N. Cell indices
        and CIC weights are precomputed in torch (``sc_deposit`` below) and
        passed in, not recomputed here: ``injector.py`` needs the same
        ``r0``/``wr``/``z0``/``wz`` again for the post-solve gather, so
        computing them once in vectorized torch is strictly less work than
        computing them a second time on-device in the kernel."""
        i = wp.tid()
        qi = q[i]
        wp.atomic_add(charge, idx00[i], qi * w00[i])
        wp.atomic_add(charge, idx01[i], qi * w01[i])
        wp.atomic_add(charge, idx10[i], qi * w10[i])
        wp.atomic_add(charge, idx11[i], qi * w11[i])

    _deposit_f32 = wp.overload(_deposit_kernel, [wp.array(dtype=wp.int32)] * 4
                                + [wp.array(dtype=wp.float32)] * 6)
    _deposit_f64 = wp.overload(_deposit_kernel, [wp.array(dtype=wp.int32)] * 4
                                + [wp.array(dtype=wp.float64)] * 6)

    @wp.kernel
    def _deposit3d_kernel(
        x: wp.array(dtype=Any), y: wp.array(dtype=Any), z: wp.array(dtype=Any), q: wp.array(dtype=Any),
        bunch_gamma: wp.array(dtype=Any), z_ref: wp.array(dtype=Any),
        x_lo: wp.array(dtype=Any), y_lo: wp.array(dtype=Any), z_lo: wp.array(dtype=Any),
        hx: wp.array(dtype=Any), hy: wp.array(dtype=Any), hz: wp.array(dtype=Any),
        ny: int, nz: int, ix_max: Any, iy_max: Any, iz_max: Any,
        charge: wp.array(dtype=Any),
    ):
        """Direct 3D CIC deposit from particle coordinates.

        Computing the eight indices/weights here avoids materialising and
        filling two ``(8, N)`` torch tensors before this launch. The cathode
        image, when requested, is deposited in the same launch.
        """
        particle = wp.tid()
        i = particle
        image = particle >= q.shape[0]
        if image:
            i = particle - q.shape[0]
        zero = hx[0] - hx[0]
        one = hx[0] / hx[0]
        two = one + one
        xi, yi = x[i], y[i]
        zi = z_ref[0] + bunch_gamma[0] * (z[i] - z_ref[0])
        qi = q[i]
        if image:
            zi = two * z_ref[0] - zi
            qi = -qi

        fx, fy, fz = (xi - x_lo[0]) / hx[0], (yi - y_lo[0]) / hy[0], (zi - z_lo[0]) / hz[0]
        ixf, iyf, izf = wp.floor(fx), wp.floor(fy), wp.floor(fz)
        ixf = wp.clamp(ixf, zero, ix_max)
        iyf = wp.clamp(iyf, zero, iy_max)
        izf = wp.clamp(izf, zero, iz_max)
        ix, iy, iz = wp.int32(ixf), wp.int32(iyf), wp.int32(izf)
        wx, wy, wz = wp.clamp(fx - ixf, zero, one), wp.clamp(fy - iyf, zero, one), wp.clamp(fz - izf, zero, one)

        for a in range(2):
            wa = one - wx
            if a == 1:
                wa = wx
            for b in range(2):
                wb = one - wy
                if b == 1:
                    wb = wy
                for c in range(2):
                    wc = one - wz
                    if c == 1:
                        wc = wz
                    idx = ((ix + a) * ny + (iy + b)) * nz + iz + c
                    wp.atomic_add(charge, idx, qi * wa * wb * wc)

    _deposit3d_f32 = wp.overload(
        _deposit3d_kernel,
        [wp.array(dtype=wp.float32)] * 12 + [wp.int32] * 2 + [wp.float32] * 3
        + [wp.array(dtype=wp.float32)],
    )
    _deposit3d_f64 = wp.overload(
        _deposit3d_kernel,
        [wp.array(dtype=wp.float64)] * 12 + [wp.int32] * 2 + [wp.float64] * 3
        + [wp.array(dtype=wp.float64)],
    )

    @wp.kernel
    def _gather3d_kernel(
        x: wp.array(dtype=Any), y: wp.array(dtype=Any), z: wp.array(dtype=Any),
        phi: wp.array(dtype=Any),
        bunch_gamma: wp.array(dtype=Any), z_ref: wp.array(dtype=Any), beta: wp.array(dtype=Any),
        x_lo: wp.array(dtype=Any), y_lo: wp.array(dtype=Any), z_lo: wp.array(dtype=Any),
        hx: wp.array(dtype=Any), hy: wp.array(dtype=Any), hz: wp.array(dtype=Any),
        nx: int, ny: int, nz: int, py: int, pz: int,
        ix_max: Any, iy_max: Any, iz_max: Any, inv_c: Any,
        Ex: wp.array(dtype=Any), Ey: wp.array(dtype=Any), Ez: wp.array(dtype=Any),
        Bx: wp.array(dtype=Any), By: wp.array(dtype=Any), Bz: wp.array(dtype=Any),
    ):
        """Gather ``-grad(phi)`` and boost it without field-grid scratch arrays."""
        i = wp.tid()
        zero = hx[0] - hx[0]
        one = hx[0] / hx[0]
        two = one + one
        zs = z_ref[0] + bunch_gamma[0] * (z[i] - z_ref[0])
        fx = (x[i] - x_lo[0]) / hx[0]
        fy = (y[i] - y_lo[0]) / hy[0]
        fz = (zs - z_lo[0]) / hz[0]
        ixf = wp.clamp(wp.floor(fx), zero, ix_max)
        iyf = wp.clamp(wp.floor(fy), zero, iy_max)
        izf = wp.clamp(wp.floor(fz), zero, iz_max)
        ix, iy, iz = wp.int32(ixf), wp.int32(iyf), wp.int32(izf)
        wx = wp.clamp(fx - ixf, zero, one)
        wy = wp.clamp(fy - iyf, zero, one)
        wz = wp.clamp(fz - izf, zero, one)
        ex, ey, ez = zero, zero, zero

        for a in range(2):
            ii = ix + a
            wa = one - wx
            if a == 1:
                wa = wx
            for b in range(2):
                jj = iy + b
                wb = one - wy
                if b == 1:
                    wb = wy
                for c in range(2):
                    kk = iz + c
                    wc = one - wz
                    if c == 1:
                        wc = wz
                    weight = wa * wb * wc

                    gx = zero
                    if ii == 0:
                        gx = -(phi[(py + jj) * pz + kk] - phi[jj * pz + kk]) / hx[0]
                    elif ii == nx - 1:
                        gx = -(phi[((nx - 1) * py + jj) * pz + kk]
                               - phi[((nx - 2) * py + jj) * pz + kk]) / hx[0]
                    else:
                        gx = -(phi[((ii + 1) * py + jj) * pz + kk]
                               - phi[((ii - 1) * py + jj) * pz + kk]) / (two * hx[0])

                    gy = zero
                    if jj == 0:
                        gy = -(phi[(ii * py + 1) * pz + kk] - phi[(ii * py) * pz + kk]) / hy[0]
                    elif jj == ny - 1:
                        gy = -(phi[(ii * py + ny - 1) * pz + kk]
                               - phi[(ii * py + ny - 2) * pz + kk]) / hy[0]
                    else:
                        gy = -(phi[(ii * py + jj + 1) * pz + kk]
                               - phi[(ii * py + jj - 1) * pz + kk]) / (two * hy[0])

                    gz = zero
                    if kk == 0:
                        gz = -(phi[(ii * py + jj) * pz + 1] - phi[(ii * py + jj) * pz]) / hz[0]
                    elif kk == nz - 1:
                        gz = -(phi[(ii * py + jj) * pz + nz - 1]
                               - phi[(ii * py + jj) * pz + nz - 2]) / hz[0]
                    else:
                        gz = -(phi[(ii * py + jj) * pz + kk + 1]
                               - phi[(ii * py + jj) * pz + kk - 1]) / (two * hz[0])

                    ex += weight * gx
                    ey += weight * gy
                    ez += weight * gz

        ex *= bunch_gamma[0]
        ey *= bunch_gamma[0]
        Ex[i] = ex
        Ey[i] = ey
        Ez[i] = ez
        Bx[i] = -beta[0] * inv_c * ey
        By[i] = beta[0] * inv_c * ex
        Bz[i] = zero

    _gather3d_f32 = wp.overload(
        _gather3d_kernel,
        [wp.array(dtype=wp.float32)] * 13 + [wp.int32] * 5 + [wp.float32] * 4
        + [wp.array(dtype=wp.float32)] * 6,
    )
    _gather3d_f64 = wp.overload(
        _gather3d_kernel,
        [wp.array(dtype=wp.float64)] * 13 + [wp.int32] * 5 + [wp.float64] * 4
        + [wp.array(dtype=wp.float64)] * 6,
    )


if _HAVE_WARP:

    @wp.kernel
    def _apertures_kernel(
        x: wp.array(dtype=Any), y: wp.array(dtype=Any), z: wp.array(dtype=Any),
        x0: wp.array(dtype=Any), y0: wp.array(dtype=Any), z0: wp.array(dtype=Any),
        kind: wp.array(dtype=wp.int32), z1: wp.array(dtype=Any), z2: wp.array(dtype=Any), r: wp.array(dtype=Any),
        has_place: wp.array(dtype=wp.int32), px: wp.array(dtype=Any), pz: wp.array(dtype=Any),
        pc: wp.array(dtype=Any), ps: wp.array(dtype=Any),
        zero: Any, one: Any,
        alive: wp.array(dtype=wp.bool),
    ):
        """``injector._apply_apertures``'s swept AND-accumulation loop, one
        kernel launch instead of one eager torch-op sequence per aperture.
        Transcribed from ``_aperture_hit``/``_swept_aperture_hit``/
        ``magnets.to_local`` -- see those docstrings for the geometry.
        ``kind``: 0=RAD 1=COL_X 2=COL_Y 3=SCR_X 4=SCR_Y (``injector.py``'s
        own ``_aperture_tables`` order). A RAD aperture with ``r < 0`` is a
        solid plug (excluded region is the disc, needs the extra
        closest-approach check); ``r >= 0`` is a hole (RAD/COL) or a
        one-sided cut (SCR). ``zero``/``one`` are typed 0/1 constants
        (warp's literal typing is strict even in a generic kernel, see
        ``push_kernel``'s docstring -- passed in from the wrapper rather
        than derived from an array element here, since nothing in this
        kernel's own inputs is guaranteed nonzero the way ``push``'s
        ``dt`` is)."""
        i = wp.tid()
        if not alive[i]:
            return
        n_ap = kind.shape[0]
        lost = wp.bool(False)
        for a in range(n_ap):
            if has_place[a] != 0:
                cx_, cz_, cc, cs = px[a], pz[a], pc[a], ps[a]
                dx0, dz0 = x0[i] - cx_, z0[i] - cz_
                lx0, lz0 = dx0 * cc - dz0 * cs, cz_ + dx0 * cs + dz0 * cc
                dx1, dz1 = x[i] - cx_, z[i] - cz_
                lx1, lz1 = dx1 * cc - dz1 * cs, cz_ + dx1 * cs + dz1 * cc
            else:
                lx0, lz0, lx1, lz1 = x0[i], z0[i], x[i], z[i]
            ly0, ly1 = y0[i], y[i]
            k, z1a, z2a, ra = kind[a], z1[a], z2[a], r[a]

            dz = lz1 - lz0
            moving = dz != zero
            if moving:
                t1, t2 = (z1a - lz0) / dz, (z2a - lz0) / dz
                t_in, t_out = wp.min(t1, t2), wp.max(t1, t2)
                overlaps = (t_out >= zero) and (t_in <= one)
                lo, hi = wp.clamp(t_in, zero, one), wp.clamp(t_out, zero, one)
            else:
                overlaps = (lz0 >= z1a) and (lz0 <= z2a)
                lo, hi = zero, one
            if not overlaps:
                continue

            dx, dy = lx1 - lx0, ly1 - ly0
            xl, yl = lx0 + lo * dx, ly0 + lo * dy
            xh, yh = lx0 + hi * dx, ly0 + hi * dy

            if k == 0:  # RAD
                r2l, r2h = xl * xl + yl * yl, xh * xh + yh * yh
                if ra >= zero:
                    hit = (r2l > ra * ra) or (r2h > ra * ra)
                else:
                    hit = (r2l < ra * ra) or (r2h < ra * ra)
            elif k == 1:  # COL_X
                hit = (wp.abs(xl) > wp.abs(ra)) or (wp.abs(xh) > wp.abs(ra))
            elif k == 2:  # COL_Y
                hit = (wp.abs(yl) > wp.abs(ra)) or (wp.abs(yh) > wp.abs(ra))
            elif k == 3:  # SCR_X
                if ra >= zero:
                    hit = (xl > ra) or (xh > ra)
                else:
                    hit = (xl < ra) or (xh < ra)
            else:  # SCR_Y
                if ra >= zero:
                    hit = (yl > ra) or (yh > ra)
                else:
                    hit = (yl < ra) or (yh < ra)

            if k == 0 and ra < zero:  # RAD plug: closest approach can lie between the two ends
                sx, sy = xh - xl, yh - yl
                s2 = sx * sx + sy * sy
                if s2 > zero:
                    sparam = wp.clamp(-(xl * sx + yl * sy) / s2, zero, one)
                else:
                    sparam = zero
                qx, qy = xl + sparam * sx, yl + sparam * sy
                if (qx * qx + qy * qy) < ra * ra:
                    hit = True

            if hit:
                lost = True
                break
        if lost:
            alive[i] = False

    _apertures_f32 = wp.overload(_apertures_kernel, [
        wp.array(dtype=wp.float32), wp.array(dtype=wp.float32), wp.array(dtype=wp.float32),
        wp.array(dtype=wp.float32), wp.array(dtype=wp.float32), wp.array(dtype=wp.float32),
        wp.array(dtype=wp.int32), wp.array(dtype=wp.float32), wp.array(dtype=wp.float32), wp.array(dtype=wp.float32),
        wp.array(dtype=wp.int32), wp.array(dtype=wp.float32), wp.array(dtype=wp.float32),
        wp.array(dtype=wp.float32), wp.array(dtype=wp.float32),
        wp.float32, wp.float32,
        wp.array(dtype=wp.bool),
    ])
    _apertures_f64 = wp.overload(_apertures_kernel, [
        wp.array(dtype=wp.float64), wp.array(dtype=wp.float64), wp.array(dtype=wp.float64),
        wp.array(dtype=wp.float64), wp.array(dtype=wp.float64), wp.array(dtype=wp.float64),
        wp.array(dtype=wp.int32), wp.array(dtype=wp.float64), wp.array(dtype=wp.float64), wp.array(dtype=wp.float64),
        wp.array(dtype=wp.int32), wp.array(dtype=wp.float64), wp.array(dtype=wp.float64),
        wp.array(dtype=wp.float64), wp.array(dtype=wp.float64),
        wp.float64, wp.float64,
        wp.array(dtype=wp.bool),
    ])


if _HAVE_WARP:

    @wp.kernel
    def _sw_kernel(
        x: wp.array(dtype=Any), y: wp.array(dtype=Any), z: wp.array(dtype=Any),
        f_flat: wp.array(dtype=Any), df_flat: wp.array(dtype=Any), offset: wp.array(dtype=wp.int32),
        z0: wp.array(dtype=Any), dz: wp.array(dtype=Any), ntab_f: wp.array(dtype=Any), zend: wp.array(dtype=Any),
        omega: wp.array(dtype=Any), cos_p: wp.array(dtype=Any), sin_p: wp.array(dtype=Any),
        c2_inv: Any, zero: Any, one: Any, half: Any,
        Ex: wp.array(dtype=Any), Ey: wp.array(dtype=Any), Ez: wp.array(dtype=Any),
        Bx: wp.array(dtype=Any), By: wp.array(dtype=Any), Bz: wp.array(dtype=Any),
    ):
        """Standing-wave cavities: on-axis Ez0(z) + paraxial Er/Bphi
        expansion, transcribed from ``injector._applied_fields``'s ``sw``
        loop. ``cos_p``/``sin_p`` (one value per cavity) are precomputed in
        torch (``injector._rf_phases``) and passed in rather than computed
        per-particle here: the RF phase is the *same* scalar for every
        particle at a given step, so computing it once in torch (as the
        existing code already did) and reading it N times is strictly
        cheaper than N per-thread transcendental evaluations. Adds onto
        whatever ``Ex..Bz`` already hold (accumulator, not overwrite) --
        callers chain ``_sw_kernel``/``_tw_kernel``/``_sol_kernel`` in
        sequence onto the same buffers."""
        i = wp.tid()
        two = one + one
        n_c = z0.shape[0]
        xi, yi, zi = x[i], y[i], z[i]
        ex, ey, ez = Ex[i] + zero, Ey[i] + zero, Ez[i] + zero
        bx, by, bz = Bx[i] + zero, By[i] + zero, Bz[i] + zero
        for c in range(n_c):
            z0c, dzc, ntabc, zendc, omegac = z0[c], dz[c], ntab_f[c], zend[c], omega[c]
            if zi >= z0c and zi <= zendc:
                zf = (zi - z0c) / dzc
                i0f = wp.floor(zf)
                if i0f > ntabc - two:
                    i0f = ntabc - two
                if i0f < zero:
                    i0f = zero
                fr = zf - i0f
                if fr < zero:
                    fr = zero
                if fr > one:
                    fr = one
                i0 = wp.int32(i0f)
                idx = offset[c] + i0
                ez0 = f_flat[idx] * (one - fr) + f_flat[idx + 1] * fr
                dez0 = df_flat[idx] * (one - fr) + df_flat[idx + 1] * fr
                cp, sp = cos_p[c], sin_p[c]
                er_over_r = -half * dez0 * cp
                bphi_over_r = -(omegac * half * c2_inv) * ez0 * sp
                ex = ex + er_over_r * xi
                ey = ey + er_over_r * yi
                ez = ez + ez0 * cp
                bx = bx - bphi_over_r * yi
                by = by + bphi_over_r * xi
        Ex[i] = ex
        Ey[i] = ey
        Ez[i] = ez
        Bx[i] = bx
        By[i] = by
        Bz[i] = bz

    _sw_f32 = wp.overload(_sw_kernel, [wp.array(dtype=wp.float32)] * 3 + [wp.array(dtype=wp.float32)] * 2
                           + [wp.array(dtype=wp.int32)] + [wp.array(dtype=wp.float32)] * 7
                           + [wp.float32] * 4 + [wp.array(dtype=wp.float32)] * 6)
    _sw_f64 = wp.overload(_sw_kernel, [wp.array(dtype=wp.float64)] * 3 + [wp.array(dtype=wp.float64)] * 2
                           + [wp.array(dtype=wp.int32)] + [wp.array(dtype=wp.float64)] * 7
                           + [wp.float64] * 4 + [wp.array(dtype=wp.float64)] * 6)

    @wp.kernel
    def _tw_kernel(
        x: wp.array(dtype=Any), y: wp.array(dtype=Any), z: wp.array(dtype=Any),
        er_flat: wp.array(dtype=Any), der_flat: wp.array(dtype=Any),
        ei_flat: wp.array(dtype=Any), dei_flat: wp.array(dtype=Any), offset: wp.array(dtype=wp.int32),
        z0: wp.array(dtype=Any), dz: wp.array(dtype=Any), ntab_f: wp.array(dtype=Any),
        zoff: wp.array(dtype=Any), z1: wp.array(dtype=Any), period: wp.array(dtype=Any),
        tile_max: wp.array(dtype=Any), length: wp.array(dtype=Any), omega: wp.array(dtype=Any),
        cos_t: wp.array(dtype=Any), sin_t: wp.array(dtype=Any),
        c2_inv: Any, zero: Any, one: Any, half: Any,
        Ex: wp.array(dtype=Any), Ey: wp.array(dtype=Any), Ez: wp.array(dtype=Any),
        Bx: wp.array(dtype=Any), By: wp.array(dtype=Any), Bz: wp.array(dtype=Any),
    ):
        """Travelling-wave structures: the periodic section is stored once
        and tiled; ``Ez = Er*cos(phase) - Ei*sin(phase)``, ``Ei`` from the
        Floquet relation -- transcribed from ``injector._applied_fields``'s
        ``tw`` loop (see ``fields.TravelingWaveField``). ``cos_t``/``sin_t``
        precomputed per structure, same reasoning as ``_sw_kernel``."""
        i = wp.tid()
        two = one + one
        n_c = z0.shape[0]
        xi, yi, zi = x[i], y[i], z[i]
        ex, ey, ez = Ex[i] + zero, Ey[i] + zero, Ez[i] + zero
        bx, by, bz = Bx[i] + zero, By[i] + zero, Bz[i] + zero
        for c in range(n_c):
            zoffc, z1c, periodc, tile_maxc, lengthc = zoff[c], z1[c], period[c], tile_max[c], length[c]
            zl = zi - zoffc
            if zl >= zero and zl <= lengthc:
                tile = wp.floor((zl - z1c) / periodc)
                if tile < zero:
                    tile = zero
                if tile > tile_maxc:
                    tile = tile_maxc
                zt = zl - tile * periodc
                z0c, dzc, ntabc = z0[c], dz[c], ntab_f[c]
                zf = (zt - z0c) / dzc
                i0f = wp.floor(zf)
                if i0f > ntabc - two:
                    i0f = ntabc - two
                if i0f < zero:
                    i0f = zero
                fr = zf - i0f
                if fr < zero:
                    fr = zero
                if fr > one:
                    fr = one
                i0 = wp.int32(i0f)
                idx = offset[c] + i0
                t_er = er_flat[idx] * (one - fr) + er_flat[idx + 1] * fr
                t_der = der_flat[idx] * (one - fr) + der_flat[idx + 1] * fr
                t_ei = ei_flat[idx] * (one - fr) + ei_flat[idx + 1] * fr
                t_dei = dei_flat[idx] * (one - fr) + dei_flat[idx + 1] * fr
                ct, st, omegac = cos_t[c], sin_t[c], omega[c]
                dEz_dz = t_der * ct - t_dei * st
                dEz_dt = -omegac * (t_er * st + t_ei * ct)
                er_over_r = -half * dEz_dz
                bphi_over_r = dEz_dt * (half * c2_inv)
                ex = ex + er_over_r * xi
                ey = ey + er_over_r * yi
                ez = ez + (t_er * ct - t_ei * st)
                bx = bx - bphi_over_r * yi
                by = by + bphi_over_r * xi
        Ex[i] = ex
        Ey[i] = ey
        Ez[i] = ez
        Bx[i] = bx
        By[i] = by
        Bz[i] = bz

    _tw_f32 = wp.overload(_tw_kernel, [wp.array(dtype=wp.float32)] * 3 + [wp.array(dtype=wp.float32)] * 4
                           + [wp.array(dtype=wp.int32)] + [wp.array(dtype=wp.float32)] * 11
                           + [wp.float32] * 4 + [wp.array(dtype=wp.float32)] * 6)
    _tw_f64 = wp.overload(_tw_kernel, [wp.array(dtype=wp.float64)] * 3 + [wp.array(dtype=wp.float64)] * 4
                           + [wp.array(dtype=wp.int32)] + [wp.array(dtype=wp.float64)] * 11
                           + [wp.float64] * 4 + [wp.array(dtype=wp.float64)] * 6)

    @wp.kernel
    def _sol_kernel(
        x: wp.array(dtype=Any), y: wp.array(dtype=Any), z: wp.array(dtype=Any),
        f_flat: wp.array(dtype=Any), df_flat: wp.array(dtype=Any), d2f_flat: wp.array(dtype=Any),
        offset: wp.array(dtype=wp.int32),
        z0: wp.array(dtype=Any), dz: wp.array(dtype=Any), ntab_f: wp.array(dtype=Any), zend: wp.array(dtype=Any),
        zero: Any, one: Any, half: Any, quarter: Any,
        Bx: wp.array(dtype=Any), By: wp.array(dtype=Any), Bz: wp.array(dtype=Any),
    ):
        """Solenoids: on-axis Bz0(z) + paraxial Br/Bz expansion --
        transcribed from ``injector._applied_fields``'s ``sol`` loop. No
        ``Ex``/``Ey``/``Ez`` params: a solenoid has no E field, so there's
        nothing for those to do here (also saves 3 unneeded ``wp.from_torch``
        wraps per call vs. carrying them through unused)."""
        i = wp.tid()
        two = one + one
        n_c = z0.shape[0]
        xi, yi, zi = x[i], y[i], z[i]
        bx, by, bz = Bx[i] + zero, By[i] + zero, Bz[i] + zero
        r2 = xi * xi + yi * yi
        for c in range(n_c):
            z0c, dzc, ntabc, zendc = z0[c], dz[c], ntab_f[c], zend[c]
            if zi >= z0c and zi <= zendc:
                zf = (zi - z0c) / dzc
                i0f = wp.floor(zf)
                if i0f > ntabc - two:
                    i0f = ntabc - two
                if i0f < zero:
                    i0f = zero
                fr = zf - i0f
                if fr < zero:
                    fr = zero
                if fr > one:
                    fr = one
                i0 = wp.int32(i0f)
                idx = offset[c] + i0
                bz0 = f_flat[idx] * (one - fr) + f_flat[idx + 1] * fr
                dbz0 = df_flat[idx] * (one - fr) + df_flat[idx + 1] * fr
                d2bz0 = d2f_flat[idx] * (one - fr) + d2f_flat[idx + 1] * fr
                br_over_r = -half * dbz0
                bx = bx + br_over_r * xi
                by = by + br_over_r * yi
                bz = bz + bz0 - quarter * r2 * d2bz0
        Bx[i] = bx
        By[i] = by
        Bz[i] = bz

    _sol_f32 = wp.overload(_sol_kernel, [wp.array(dtype=wp.float32)] * 3 + [wp.array(dtype=wp.float32)] * 3
                            + [wp.array(dtype=wp.int32)] + [wp.array(dtype=wp.float32)] * 4
                            + [wp.float32] * 4 + [wp.array(dtype=wp.float32)] * 3)
    _sol_f64 = wp.overload(_sol_kernel, [wp.array(dtype=wp.float64)] * 3 + [wp.array(dtype=wp.float64)] * 3
                            + [wp.array(dtype=wp.int32)] + [wp.array(dtype=wp.float64)] * 4
                            + [wp.float64] * 4 + [wp.array(dtype=wp.float64)] * 3)


def push(x, y, z, ux, uy, uz, cx, cy, cz, cux, cuy, cuz,
         Ex, Ey, Ez, Bx, By, Bz, qmdt2: float, dt: float, c2_inv: float):
    """Boris-push tail of ``injector._fused_step`` as one warp kernel
    launch. ``qmdt2``/``dt``/``c2_inv`` are plain Python floats, not 0-d
    tensors: unlike ``torch.compile(dynamic=False)``, a warp kernel never
    recompiles on a changed scalar *value* (only on argument *type*), so
    ``_fused_step``'s tensor-indirection trick to dodge that has no
    equivalent cost here and would just add an unneeded host readback."""
    _ensure_init()
    dtype = x.dtype
    kern = _push_f64 if dtype == torch.float64 else _push_f32
    wp_scalar = wp.float64 if dtype == torch.float64 else wp.float32
    ins = (x, y, z, ux, uy, uz, cx, cy, cz, cux, cuy, cuz, Ex, Ey, Ez, Bx, By, Bz)
    outs = tuple(torch.empty_like(x) for _ in range(12))
    wp.launch(
        kern, dim=x.shape[0],
        inputs=[_wp_arg(t) for t in ins] + [wp_scalar(qmdt2), wp_scalar(dt), wp_scalar(c2_inv)]
        + [_wp_arg(t) for t in outs],
        stream=_launch_stream(x),
    )
    return outs


def push_inplace(x, y, z, ux, uy, uz, cx, cy, cz, cux, cuy, cuz,
                  Ex, Ey, Ez, Bx, By, Bz, qmdt2: float, dt: float, c2_inv: float):
    """Same physics as ``push``, but writes the twelve state tensors back
    into themselves instead of allocating twelve fresh output tensors.
    Safe: the kernel only reads a given particle's slot of ``x``/``cx``/etc.
    *before* it writes that same slot (checked directly against the kernel
    body above), so aliasing input and output buffers is race-free and
    gives a bit-identical result to ``push`` -- confirmed empirically, not
    just by inspection. This matters because ``push`` alone measured
    surprisingly far from Phase 1's isolated-kernel numbers once wired into
    a real per-step loop: ``wp.from_torch`` wrapping and fresh
    ``torch.empty_like`` allocation, both negligible in a benchmark that
    hoists them outside the timed loop (``benchmark/warp_port/bench.py``),
    are *not* negligible paid fresh every step (measured ~0.27 ms/call at
    N=50,000 float32 vs. this function's ~0.15 ms -- allocation plus half
    the wrapping gone).

    Only safe to use when the caller has no further use for the pre-step
    values (``_fused_step_warp`` only calls this when ``emitted_all``,
    where there's no old/new masking blend that needs them -- see there
    for the ``emitted_all=False`` case, which still needs ``push``'s
    allocating, non-aliased version)."""
    _ensure_init()
    dtype = x.dtype
    kern = _push_f64 if dtype == torch.float64 else _push_f32
    wp_scalar = wp.float64 if dtype == torch.float64 else wp.float32
    state = (x, y, z, ux, uy, uz, cx, cy, cz, cux, cuy, cuz)
    state_w = [_wp_arg(t) for t in state]
    field_w = [_wp_arg(t) for t in (Ex, Ey, Ez, Bx, By, Bz)]
    wp.launch(
        kern, dim=x.shape[0],
        inputs=state_w + field_w + [wp_scalar(qmdt2), wp_scalar(dt), wp_scalar(c2_inv)] + state_w,
        stream=_launch_stream(x),
    )
    return state


def sc_deposit(r0: torch.Tensor, z0: torch.Tensor, wr: torch.Tensor, wz: torch.Tensor,
               q: torch.Tensor, nr: int, nz: int) -> torch.Tensor:
    """CIC scatter-add half of ``injector._fused_sc_fields``'s deposit, as
    one warp kernel launch. ``r0``/``z0`` (int64, already clamped to
    ``[0, nr-2]``/``[0, nz-2]``) and ``wr``/``wz`` (the bilinear weights)
    are ``injector.py``'s own already-computed CIC coordinates -- see
    ``_deposit_kernel``'s docstring for why they're passed in rather than
    recomputed. Returns the flat ``(nr*nz,)`` charge array."""
    _ensure_init()
    dtype = q.dtype
    kern = _deposit_f64 if dtype == torch.float64 else _deposit_f32
    idx00 = (r0 * nz + z0).to(torch.int32)
    idx01 = idx00 + 1
    idx10 = idx00 + nz
    idx11 = idx10 + 1
    w00 = (1 - wr) * (1 - wz)
    w01 = (1 - wr) * wz
    w10 = wr * (1 - wz)
    w11 = wr * wz
    charge = torch.zeros(nr * nz, device=q.device, dtype=dtype)
    wp.launch(
        kern, dim=q.shape[0],
        inputs=[_wp_arg(t) for t in (idx00, idx01, idx10, idx11, w00, w01, w10, w11, q, charge)],
        stream=_launch_stream(q),
    )
    return charge


def sc_fields_3d(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, q: torch.Tensor,
    bunch_gamma: torch.Tensor, z_ref: torch.Tensor, beta: torch.Tensor,
    x_lo: torch.Tensor, y_lo: torch.Tensor, z_lo: torch.Tensor,
    hx: torch.Tensor, hy: torch.Tensor, hz: torch.Tensor,
    nx: int, ny: int, nz: int, px: int, py: int, pz: int,
    G_k: torch.Tensor, with_image: bool, inv_c: float,
):
    """3D CIC deposit, PyTorch cuFFT solve, and fused Warp gather/boost."""
    _ensure_init()
    dtype = q.dtype
    wp_scalar = wp.float64 if dtype == torch.float64 else wp.float32
    deposit_kern = _deposit3d_f64 if dtype == torch.float64 else _deposit3d_f32
    gather_kern = _gather3d_f64 if dtype == torch.float64 else _gather3d_f32
    stream = _launch_stream(q)

    particle_w = [_wp_arg(t) for t in (x, y, z, q)]
    scalar_tensors = (bunch_gamma, z_ref, beta, x_lo, y_lo, z_lo, hx, hy, hz)
    scalar_w = [_wp_arg(t.reshape(1)) for t in scalar_tensors]
    maxima = [wp_scalar(nx - 2), wp_scalar(ny - 2), wp_scalar(nz - 2)]
    charge = torch.zeros(nx * ny * nz, device=q.device, dtype=dtype)
    wp.launch(
        deposit_kern,
        dim=q.shape[0] * (2 if with_image else 1),
        inputs=particle_w + [scalar_w[0], scalar_w[1], *scalar_w[3:]]
        + [wp.int32(ny), wp.int32(nz), *maxima, _wp_arg(charge)],
        stream=stream,
    )

    phi = torch.fft.irfftn(
        torch.fft.rfftn(charge.view(nx, ny, nz), s=(px, py, pz)) * G_k,
        s=(px, py, pz),
    )
    fields = tuple(torch.empty_like(x) for _ in range(6))
    wp.launch(
        gather_kern,
        dim=x.shape[0],
        inputs=particle_w[:3] + [_wp_arg(phi.flatten())] + scalar_w
        + [wp.int32(nx), wp.int32(ny), wp.int32(nz), wp.int32(py), wp.int32(pz),
           *maxima, wp_scalar(inv_c)]
        + [_wp_arg(t) for t in fields],
        stream=stream,
    )
    return fields


_APERTURE_KIND_CODE = {"RAD": 0, "COL_X": 1, "COL_Y": 2, "SCR_X": 3, "SCR_Y": 4}


@functools.lru_cache(maxsize=64)
def _aperture_arrays(apertures: tuple, device: str, dtype: torch.dtype):
    """``apertures`` (``injector._aperture_tables``'s output: a tuple of
    ``(kind, z1, z2, r[, (x_m, z_m, cos, sin)])`` per aperture, all plain
    Python values) packed into the fixed-length arrays ``apply_apertures``
    launches over. Cached on ``(apertures, device, dtype)`` -- ``apertures``
    is itself a plain tuple of plain values, so it's hashable and fixed for
    a whole run, exactly like ``sw``/``tw``/``sol`` are for
    ``_applied_fields_compiled``: build these arrays once, not every step."""
    kinds, z1s, z2s, rs, has_place, px, pz, pc, ps = [], [], [], [], [], [], [], [], []
    for ap in apertures:
        kind, z1, z2, r = ap[:4]
        kinds.append(_APERTURE_KIND_CODE[kind])
        z1s.append(z1)
        z2s.append(z2)
        rs.append(r)
        if len(ap) > 4:
            x_m, z_m, c, s = ap[4]
            has_place.append(1)
            px.append(x_m)
            pz.append(z_m)
            pc.append(c)
            ps.append(s)
        else:
            has_place.append(0)
            px.append(0.0)
            pz.append(0.0)
            pc.append(1.0)
            ps.append(0.0)

    def t(seq, dt):
        return torch.tensor(seq, device=device, dtype=dt)

    return (t(kinds, torch.int32), t(z1s, dtype), t(z2s, dtype), t(rs, dtype),
            t(has_place, torch.int32), t(px, dtype), t(pz, dtype), t(pc, dtype), t(ps, dtype))


def apply_apertures(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
                     x0: torch.Tensor, y0: torch.Tensor, z0: torch.Tensor,
                     alive: torch.Tensor, apertures: tuple) -> torch.Tensor:
    """``injector._apply_apertures``'s swept AND-accumulation check as one
    warp kernel launch instead of one chained eager torch-op sequence per
    aperture -- see ``_apertures_kernel``'s docstring for the geometry.
    Measured as the single largest eager-dispatch cost once the two hot
    kernels moved to warp (the 5pC CLARA deck's 8 apertures dominated the
    whole warp-backed step, ~4.3 of ~4.8 ms/step, before this existed).
    Mutates and returns ``alive`` in place (same in-place-is-safe reasoning
    as ``push_inplace``: each thread only ever reads then writes its own
    ``alive[i]``, never another thread's)."""
    if not apertures:
        return alive
    _ensure_init()
    dtype = x.dtype
    kern = _apertures_f64 if dtype == torch.float64 else _apertures_f32
    wp_scalar = wp.float64 if dtype == torch.float64 else wp.float32
    kind, z1, z2, r, has_place, px, pz, pc, ps = _aperture_arrays(apertures, str(x.device), dtype)
    wp.launch(
        kern, dim=x.shape[0],
        inputs=[_wp_arg(t) for t in (x, y, z, x0, y0, z0, kind, z1, z2, r, has_place, px, pz, pc, ps)]
        + [wp_scalar(0.0), wp_scalar(1.0), _wp_arg(alive)],
        stream=_launch_stream(x),
    )
    return alive


# Packed-array cache for sw/tw/sol, keyed by id() of the tuple injector.py
# builds once per Injector.track() call and never mutates -- (id, packed)
# pairs, with a strong ref to the *original* tuple kept alongside the
# packed tensors so its id can never be reused by a different object while
# the cache entry is alive (the usual safety condition for id()-keyed
# memoization). ponytail: unbounded -- grows by one entry per distinct
# sw/tw/sol tuple built in this process; fine for how this is used (one
# entry per Injector.track() call), add an eviction policy if that stops
# being true.
_ELEMENT_CACHE: dict[int, tuple] = {}


def _cat_tables(entries: tuple, table_indices: tuple, device: str, dtype: torch.dtype):
    """Concatenate one or more per-element 1D tables (``table_indices`` into
    each ``entries[i]`` tuple) into flat arrays plus an ``int32`` offset
    array -- the ragged-array packing ``_sw_kernel``/``_tw_kernel``/
    ``_sol_kernel`` index with ``offset[c] + i0``."""
    flats = [[] for _ in table_indices]
    offsets = []
    off = 0
    for e in entries:
        offsets.append(off)
        off += e[table_indices[0]].shape[0]
        for j, ti in enumerate(table_indices):
            flats[j].append(e[ti])
    flat_tensors = [torch.cat(f).to(device=device, dtype=dtype) for f in flats]
    offset_t = torch.tensor(offsets, device=device, dtype=torch.int32)
    return flat_tensors, offset_t


def _pack_sw(sw: tuple, device: str, dtype: torch.dtype):
    if not sw:
        return None
    key = id(sw)
    cached = _ELEMENT_CACHE.get(key)
    if cached is not None and cached[0] is sw:
        return cached[1]
    (f_flat, df_flat), offset = _cat_tables(sw, (0, 1), device, dtype)
    t = lambda idx, dt=dtype: torch.tensor([e[idx] for e in sw], device=device, dtype=dt)
    packed = (f_flat, df_flat, offset, t(2), t(3), t(4, torch.float64).to(dtype), t(5), t(6))
    _ELEMENT_CACHE[key] = (sw, packed)
    return packed


def _pack_tw(tw: tuple, device: str, dtype: torch.dtype):
    if not tw:
        return None
    key = id(tw)
    cached = _ELEMENT_CACHE.get(key)
    if cached is not None and cached[0] is tw:
        return cached[1]
    (er_flat, der_flat, ei_flat, dei_flat), offset = _cat_tables(tw, (0, 1, 2, 3), device, dtype)
    t = lambda idx, dt=dtype: torch.tensor([e[idx] for e in tw], device=device, dtype=dt)
    packed = (er_flat, der_flat, ei_flat, dei_flat, offset, t(4), t(5), t(6, torch.float64).to(dtype),
              t(7), t(8), t(9), t(10), t(11), t(12))
    _ELEMENT_CACHE[key] = (tw, packed)
    return packed


def _pack_sol(sol: tuple, device: str, dtype: torch.dtype):
    if not sol:
        return None
    key = id(sol)
    cached = _ELEMENT_CACHE.get(key)
    if cached is not None and cached[0] is sol:
        return cached[1]
    (f_flat, df_flat, d2f_flat), offset = _cat_tables(sol, (0, 1, 2), device, dtype)
    t = lambda idx, dt=dtype: torch.tensor([e[idx] for e in sol], device=device, dtype=dt)
    packed = (f_flat, df_flat, d2f_flat, offset, t(3), t(4), t(5, torch.float64).to(dtype), t(6))
    _ELEMENT_CACHE[key] = (sol, packed)
    return packed


def applied_fields(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, sw: tuple, tw: tuple, sol: tuple,
                    cos_sw: torch.Tensor, sin_sw: torch.Tensor, cos_tw: torch.Tensor, sin_tw: torch.Tensor,
                    c2_inv: float):
    """``injector._applied_fields``'s ``sw``/``tw``/``sol`` loops (cavities,
    TW structures, solenoids -- *not* ``quad``/``magnets``, see
    ``injector._applied_fields_warp``'s docstring for why those stay on the
    ``torch.compile`` path) as up to three warp kernel launches instead of
    one ``torch.compile`` graph. Each kernel adds onto the same ``Ex..Bz``
    buffers in sequence (a solenoid only ever touches ``Bx``/``By``/``Bz``).
    ``cos_sw``/``sin_sw``/``cos_tw``/``sin_tw`` are one value per element,
    precomputed in torch by ``injector._rf_phases`` -- see ``_sw_kernel``'s
    docstring for why (the RF phase is one scalar per element per step, not
    per-particle; computing it in the kernel would redundantly repeat a
    transcendental evaluation once per particle instead of once)."""
    _ensure_init()
    dtype = x.dtype
    device = str(x.device)
    wp_scalar = wp.float64 if dtype == torch.float64 else wp.float32
    zero, one, half, quarter = wp_scalar(0.0), wp_scalar(1.0), wp_scalar(0.5), wp_scalar(0.25)
    Ex, Ey, Ez, Bx, By, Bz = (torch.zeros_like(x) for _ in range(6))
    stream = _launch_stream(x)

    sw_p = _pack_sw(sw, device, dtype)
    if sw_p is not None:
        f_flat, df_flat, offset, z0, dz, ntab_f, zend, omega = sw_p
        kern = _sw_f64 if dtype == torch.float64 else _sw_f32
        wp.launch(kern, dim=x.shape[0], inputs=[_wp_arg(t) for t in
                  (x, y, z, f_flat, df_flat, offset, z0, dz, ntab_f, zend, omega, cos_sw, sin_sw)]
                  + [wp_scalar(c2_inv), zero, one, half]
                  + [_wp_arg(t) for t in (Ex, Ey, Ez, Bx, By, Bz)], stream=stream)

    tw_p = _pack_tw(tw, device, dtype)
    if tw_p is not None:
        er_flat, der_flat, ei_flat, dei_flat, offset, z0, dz, ntab_f, zoff, z1, period, tile_max, length, omega = tw_p
        kern = _tw_f64 if dtype == torch.float64 else _tw_f32
        wp.launch(kern, dim=x.shape[0], inputs=[_wp_arg(t) for t in
                  (x, y, z, er_flat, der_flat, ei_flat, dei_flat, offset, z0, dz, ntab_f,
                   zoff, z1, period, tile_max, length, omega, cos_tw, sin_tw)]
                  + [wp_scalar(c2_inv), zero, one, half]
                  + [_wp_arg(t) for t in (Ex, Ey, Ez, Bx, By, Bz)], stream=stream)

    sol_p = _pack_sol(sol, device, dtype)
    if sol_p is not None:
        f_flat, df_flat, d2f_flat, offset, z0, dz, ntab_f, zend = sol_p
        kern = _sol_f64 if dtype == torch.float64 else _sol_f32
        wp.launch(kern, dim=x.shape[0], inputs=[_wp_arg(t) for t in
                  (x, y, z, f_flat, df_flat, d2f_flat, offset, z0, dz, ntab_f, zend)]
                  + [zero, one, half, quarter]
                  + [_wp_arg(t) for t in (Bx, By, Bz)], stream=stream)

    return Ex, Ey, Ez, Bx, By, Bz
