"""Nonlinear ring dynamics on top of ``tracking.native.ring.RingMap``:
closed orbit / linear optics / chromaticity from the autograd Jacobian of the
real one-turn map, batched NAFF tunes, tune shift with amplitude (TSA),
dynamic aperture (DA) and frequency-map analysis (FMA). See
plans/RING_NONLINEAR_DYNAMICS_PLAN.md.

Everything is batched over particles (no per-particle Python loops) and --
except the survival bookkeeping of DA -- differentiable in any tensor
strength set via ``RingMap.with_strengths``: ``linear_optics``/
``chromaticity`` via ``create_graph`` Jacobians, ``naff`` via unrolled Newton
refinement (the FFT peak search only seeds it).

Tunes are fractional, in ``[0, 1)``. Amplitudes are physical offsets at the
ring start (``s=0``, the first element's entrance), launched from the closed
orbit with zero momentum offset, like elegant's ``find_aperture``/
``frequency_map``.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch

from ..tracking.native.ring import RingMap


class LinearOptics(NamedTuple):
    closed_orbit: torch.Tensor  # (6,)
    M: torch.Tensor  # (6, 6) one-turn Jacobian at the closed orbit
    Qx: torch.Tensor
    Qy: torch.Tensor
    beta_x: torch.Tensor
    alpha_x: torch.Tensor
    beta_y: torch.Tensor
    alpha_y: torch.Tensor


def _jacobian(ring: RingMap, x0: torch.Tensor, create_graph: bool) -> torch.Tensor:
    # vectorize=True: ~4x faster than 6 separate backward passes; forward
    # mode would be faster still but can't create_graph (torch 2.14). On the
    # real booster, however, vectorized reverse-mode retains a multi-GB graph
    # even when the result is not differentiable; keep ordinary closed-orbit
    # probes bounded and retain vectorization for the higher-order path.
    return torch.autograd.functional.jacobian(
        lambda u: ring.turn(u, eager=True), x0, create_graph=create_graph, vectorize=create_graph,
    )


def _closed_orbit_and_matrix(
    ring: RingMap, delta: float, iters: int, create_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Newton fixed point of the one-turn map, differentiable in any
    tensor strength ``ring`` carries (``create_graph=True``).

    The convergence check runs *after* the Newton update, not before --
    deliberately, not merely for the usual "check what you just computed"
    reason. At the LOCO operating point (correctors and K1 errors exactly
    at their nominal value), the starting guess ``co=0`` already satisfies
    ``ring.turn(0)[:4] == 0`` to machine precision (F(0)=0 for any purely
    multipole ring with correctors at zero), so a check-then-update loop
    would ``break`` on iteration 1 *before* ever computing an update --
    returning the original ``torch.zeros(...)`` tensor completely
    disconnected from the autograd graph (no ``grad_fn`` at all), silently
    dropping the closed-orbit self-consistency (wraparound) term from
    ``d(co)/d(any strength)`` -- exactly the derivative LOCO's ORM needs.
    Running the update first, using the *symbolic* (graph-connected)
    ``resid`` even when its *value* is exactly zero, reproduces the
    correct implicit-function-theorem gradient
    (``d(co)/d(theta) = -(J-I)^-1 @ d(resid)/d(theta)``) from the very
    first iteration -- found via a real-booster story computing an exactly
    zero d(ORM)/dK1 for a K1 knob upstream of every tapped corrector/BPM
    pair, see plans/LOCO_PLAN.md."""
    tol = 1e-13 if ring.dtype == torch.float64 else 1e-6
    co = torch.zeros(6, dtype=ring.dtype, device=ring.device)
    co[5] = delta
    eye = torch.eye(4, dtype=ring.dtype, device=ring.device)
    for _ in range(iters):
        J = _jacobian(ring, co, create_graph)
        resid = ring.turn(co, eager=True)[:4] - co[:4]
        converged = resid.abs().max().item() < tol
        co = torch.cat([co[:4] - torch.linalg.solve(J[:4, :4] - eye, resid), co[4:]])
        if converged:
            break
    else:
        raise RuntimeError(f"closed orbit not converged at delta={delta}")
    return co, J


def closed_orbit(ring: RingMap, delta: float = 0.0, iters: int = 8, create_graph: bool = False) -> torch.Tensor:
    """Transverse fixed point ``(x, px, y, py)`` of the one-turn map at fixed
    ``delta`` by Newton iteration (converges to round-off in ~3 steps for
    a stable ring). Returns the full ``(6,)`` state (``tau=0``)."""
    if not create_graph:
        tol = 1e-13 if ring.dtype == torch.float64 else 1e-6
        co = torch.zeros(6, dtype=ring.dtype, device=ring.device)
        co[5] = delta
        eye = torch.eye(4, dtype=ring.dtype, device=ring.device)
        for _ in range(iters):
            resid = ring.turn(co, eager=True)[:4] - co[:4]
            if resid.abs().max().item() < tol:
                return co
            J = _jacobian(ring, co, create_graph=False)
            co = torch.cat([co[:4] - torch.linalg.solve(J[:4, :4] - eye, resid), co[4:]])
        raise RuntimeError(f"closed orbit not converged at delta={delta}")
    return _closed_orbit_and_matrix(ring, delta, iters, create_graph)[0]


def _plane(B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cos_mu = (B[0, 0] + B[1, 1]) / 2
    if abs(cos_mu.item()) >= 1:
        raise ValueError(f"linearly unstable plane, cos(mu)={cos_mu.item():.6f}")
    sin_mu = torch.sign(B[0, 1]) * torch.sqrt(1 - cos_mu * cos_mu)
    mu = torch.atan2(sin_mu, cos_mu)
    return torch.remainder(mu / (2 * math.pi), 1.0), B[0, 1] / sin_mu, (B[0, 0] - B[1, 1]) / (2 * sin_mu)


def linear_optics(ring: RingMap, delta: float = 0.0, create_graph: bool = False) -> LinearOptics:
    """Uncoupled per-plane tunes/Twiss from the one-turn Jacobian at the
    closed orbit. ``create_graph=True`` keeps everything differentiable in
    the ring's tensor strengths."""
    if ring.double_single:
        ring = ring.float64_reference()
    co, M = _closed_orbit_and_matrix(ring, delta, 8, create_graph)
    Qx, bx, ax = _plane(M[0:2, 0:2])
    Qy, by, ay = _plane(M[2:4, 2:4])
    return LinearOptics(co, M, Qx, Qy, bx, ax, by, ay)


def chromaticity(ring: RingMap, ddelta: float = 1e-4, create_graph: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """``(xi_x, xi_y) = dQ/ddelta`` by central difference of
    ``linear_optics`` tunes (``O(ddelta^2)`` error, second-order
    chromaticity contaminates only at that order)."""
    p = linear_optics(ring, ddelta, create_graph)
    m = linear_optics(ring, -ddelta, create_graph)

    def d(a, b):  # fractional tunes may wrap across 0/1
        return torch.remainder(a - b + 0.5, 1.0) - 0.5

    return d(p.Qx, m.Qx) / (2 * ddelta), d(p.Qy, m.Qy) / (2 * ddelta)


def naff(
    z: torch.Tensor, window_order: int = 1, newton_iters: int = 5, pad: int = 4,
    center: torch.Tensor | float | None = None, half_width: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dominant frequency (turns^-1, in ``[0, 1)``) and amplitude of each
    complex signal ``z`` ``(..., T)`` -- Laskar's NAFF first harmonic:
    Hann^``window_order`` window, zero-padded FFT peak as the seed, then
    Newton refinement of ``argmax |sum_n w_n z_n exp(-2 pi i nu n)|``.
    Error on a pure tone ~1e-10 at T=1024 (float64); see
    tests/test_ring_nonlinear.py.

    ``center`` (broadcastable to ``z.shape[:-1]``) restricts the seed to
    ``|nu - center| < half_width``: with a large amplitude in one plane the
    other plane's signal carries driven lines (``2 Q_y`` ... ) that can
    out-weigh its own small betatron line, and an unrestricted peak search
    then returns the wrong line (a ~0.2 "tune shift")."""
    T = z.shape[-1]
    n = torch.arange(T, dtype=z.real.dtype, device=z.device)
    w = (1 - torch.cos(2 * math.pi * n / T)) ** window_order
    zw = (z - z.mean(-1, keepdim=True)) * (w / w.sum())
    amp = torch.fft.fft(zw, n=pad * T).abs()
    if center is not None:
        f = torch.arange(pad * T, dtype=n.dtype, device=z.device) / (pad * T)
        c = torch.as_tensor(center, dtype=n.dtype, device=z.device).unsqueeze(-1)
        dist = torch.remainder(f - c + 0.5, 1.0) - 0.5
        amp = torch.where(dist.abs() < half_width, amp, torch.zeros_like(amp))
    k = amp.argmax(-1)
    nu = (k.to(n.dtype) / (pad * T)).detach()
    two_pi_n = 2 * math.pi * n
    for _ in range(newton_iters):
        e = torch.exp(-1j * two_pi_n * nu.unsqueeze(-1))
        F0 = (zw * e).sum(-1)
        F1 = (zw * e * (-1j * two_pi_n)).sum(-1)
        F2 = (zw * e * (-(two_pi_n**2))).sum(-1)
        g = (F1 * F0.conj()).real
        gp = (F1.abs() ** 2) + (F2 * F0.conj()).real
        nu = nu - g / gp
    e = torch.exp(-1j * two_pi_n * nu.unsqueeze(-1))
    return torch.remainder(nu, 1.0), (zw * e).sum(-1).abs()


def normalized_signals(tbt: torch.Tensor, opt: LinearOptics) -> tuple[torch.Tensor, torch.Tensor]:
    """Complex normalized betatron signals ``x~ - i p~`` (positive frequency
    = tune) from ``RingMap.track``'s ``(T, ..., 4)`` record; returns
    ``(zx, zy)`` shaped ``(..., T)``."""
    # Always float64/complex128, whatever the tracking dtype: NAFF sums
    # ~1e3 terms, and complex64 accumulation adds its own tune noise.
    u = (tbt.double() - opt.closed_orbit[:4].to(tbt.device, torch.float64)).movedim(0, -1)

    def z(q, p, beta, alpha):
        beta, alpha = beta.to(tbt.device, torch.float64), alpha.to(tbt.device, torch.float64)
        return torch.complex(q, -(alpha * q + beta * p)) / torch.sqrt(beta)

    return z(u[..., 0, :], u[..., 1, :], opt.beta_x, opt.alpha_x), z(u[..., 2, :], u[..., 3, :], opt.beta_y, opt.alpha_y)


def launch(opt: LinearOptics, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Initial ``(..., 6)`` states at the closed orbit plus physical
    offsets ``x``/``y`` (zero momentum offset), broadcast together."""
    x, y = torch.broadcast_tensors(torch.as_tensor(x), torch.as_tensor(y))
    state = opt.closed_orbit.expand(*x.shape, 6).clone()
    state[..., 0] += x.to(state)
    state[..., 2] += y.to(state)
    return state


def tunes(
    ring: RingMap, opt: LinearOptics, state: torch.Tensor, n_turns: int, checkpoint: bool = False,
    band: float | None = None, centers: tuple | None = None, **naff_kw,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Track ``state`` ``n_turns`` (recording every turn) and NAFF both
    planes. Returns ``(Qx, Qy, lost)``; lost particles' tunes are NaN.
    ``state`` may sit at any ``delta`` (NAFF removes the mean, so the launch
    need not be exactly on that ``delta``'s closed orbit); ``checkpoint``
    recomputes each turn in the backward pass (memory O(1 turn)).
    ``band`` (half-width, tune units) restricts each plane's NAFF seed to
    ``|nu - Q| < band`` around ``centers = (Qx, Qy)`` (default the linear
    tunes of ``opt``) -- see ``naff``."""
    res = ring.track(state, n_turns, record_every=1, aperture=(math.inf, math.inf), checkpoint=checkpoint)
    zx, zy = normalized_signals(res.tbt, opt)
    lost = res.loss_turn >= 0
    nan = torch.tensor(math.nan, dtype=ring.dtype, device=ring.device)
    if band is not None:
        cx, cy = centers if centers is not None else (opt.Qx, opt.Qy)
        naff_x = dict(naff_kw, center=cx, half_width=band)
        naff_y = dict(naff_kw, center=cy, half_width=band)
    else:
        naff_x = naff_y = naff_kw
    return (torch.where(lost, nan, naff(zx, **naff_x)[0]), torch.where(lost, nan, naff(zy, **naff_y)[0]), lost)


class Detuning(NamedTuple):
    dQx_dJx: torch.Tensor
    dQy_dJx: torch.Tensor
    dQx_dJy: torch.Tensor
    dQy_dJy: torch.Tensor


def detuning(
    ring: RingMap, x_max: float, y_max: float, n_amp: int = 6, n_turns: int = 512, delta: float = 0.0,
    opt: LinearOptics | None = None, band: float | None = 0.05,
) -> Detuning:
    """First-order amplitude-detuning coefficients (per metre of action,
    ``J = gamma*u^2/2`` for a pure-offset launch) from a linear+quadratic
    least-squares fit of NAFF tunes over ``n_amp`` amplitudes up to
    ``x_max`` (y fixed at ``y_max/(2 n_amp)``) and up to ``y_max`` (x fixed
    at ``x_max/(2 n_amp)``), all in one batch. The fixed other-plane
    amplitude only offsets the fitted line; it must stay well above the
    working-precision noise floor (a 1 um amplitude in float32 gave a
    cross-term 100x off).
    ``band``: NAFF seed window around the linear tunes (see ``naff``) -- keeps
    the fit on the betatron line when the other plane's amplitude drives lines
    that out-weigh it.
    Differentiable through NAFF and the fit (keep amplitudes inside the DA)."""
    opt = opt or linear_optics(ring, delta)
    a = torch.linspace(0, 1, n_amp + 1, dtype=ring.dtype, device=ring.device)[1:]
    xs = torch.cat([a * x_max, torch.full_like(a, x_max / (2 * n_amp))])
    ys = torch.cat([torch.full_like(a, y_max / (2 * n_amp)), a * y_max])
    Qx, Qy, lost = tunes(ring, opt, launch(opt, xs, ys), n_turns, band=band)
    if lost.any():
        raise ValueError("particles lost inside the detuning scan -- reduce x_max/y_max")
    gx = (1 + opt.alpha_x**2) / opt.beta_x
    gy = (1 + opt.alpha_y**2) / opt.beta_y
    Jx, Jy = gx * xs**2 / 2, gy * ys**2 / 2

    def slope(J, Q):
        A = torch.stack([torch.ones_like(J), J, J * J], -1)
        return torch.linalg.lstsq(A, Q.unsqueeze(-1)).solution[1, 0]

    h, v = slice(0, n_amp), slice(n_amp, 2 * n_amp)
    return Detuning(slope(Jx[h], Qx[h]), slope(Jx[h], Qy[h]), slope(Jy[v], Qx[v]), slope(Jy[v], Qy[v]))


class DynamicAperture(NamedTuple):
    angle_rad: torch.Tensor  # (n_angles,)
    radius_m: torch.Tensor  # (..., n_angles) -- last surviving radius along each ray
    x_m: torch.Tensor
    y_m: torch.Tensor
    area_m2: torch.Tensor  # (...,)


def dynamic_aperture(
    ring: RingMap, r_max: float, n_turns: int, n_angles: int = 19, n_r: int = 32, refine: int = 2,
    delta: float = 0.0, aperture: tuple[float, float] = (0.05, 0.05), batch_shape: tuple[int, ...] = (),
    opt: LinearOptics | None = None,
) -> DynamicAperture:
    """Upper-half-plane DA along ``n_angles`` rays in ``[0, pi]`` (elegant
    ``find_aperture`` ``mode="n-lines"`` geometry): ``n_r`` radii per ray in
    one batched run; the boundary on each ray is the last radius before the
    *first* loss (islands beyond it don't count). ``refine`` further batched
    passes subdivide each ray's bracketing interval by ``n_r``, so the
    final resolution is ``r_max / n_r**(refine+1)``. ``batch_shape`` prefixes
    the particle batch -- e.g. ``(M,)`` with a ``with_strengths`` error
    ensemble of shape ``(M, 1)``, launched from the nominal closed orbit."""
    opt = opt or linear_optics(ring, delta)
    dev, dt = ring.device, ring.dtype
    theta = torch.linspace(0, math.pi, n_angles, dtype=dt, device=dev)
    lo = torch.zeros(*batch_shape, n_angles, dtype=dt, device=dev)
    width = torch.full_like(lo, r_max)
    frac = torch.arange(1, n_r + 1, dtype=dt, device=dev) / n_r
    for _ in range(refine + 1):
        r = lo.unsqueeze(-1) + width.unsqueeze(-1) * frac  # (..., n_angles, n_r)
        state = launch(opt, r * torch.cos(theta).unsqueeze(-1), r * torch.sin(theta).unsqueeze(-1))
        with torch.no_grad():
            alive = ring.track(state, n_turns, aperture=aperture).loss_turn < 0
        # number of leading survivors along each ray = index of first loss
        n_ok = torch.cumprod(alive.to(torch.int64), -1).sum(-1)
        lo = lo + width * n_ok.to(dt) / n_r
        width = torch.where(n_ok == n_r, torch.zeros_like(width), width / n_r)
    x, y = lo * torch.cos(theta), lo * torch.sin(theta)
    area = 0.5 * (x * y.roll(-1, -1) - x.roll(-1, -1) * y).sum(-1).abs()
    return DynamicAperture(theta, lo, x, y, area)


class FrequencyMap(NamedTuple):
    x0_m: torch.Tensor
    y0_m: torch.Tensor
    Qx: torch.Tensor
    Qy: torch.Tensor
    diffusion: torch.Tensor  # log10 |dQ| between the two halves; NaN if lost


def frequency_map(
    ring: RingMap, x_range: tuple[float, float], y_range: tuple[float, float], nx: int = 41, ny: int = 21,
    n_turns: int = 1024, delta: float = 0.0, aperture: tuple[float, float] = (0.05, 0.05),
    opt: LinearOptics | None = None, band: float | None = 0.1,
) -> FrequencyMap:
    """Laskar FMA on an ``nx x ny`` grid of initial offsets: one run of
    ``n_turns``, NAFF on each half, diffusion ``log10 sqrt(dQx^2+dQy^2)``.
    Keep ``y_range[0] > 0`` (zero vertical amplitude has no vertical tune).
    ``band``: NAFF seed window ``|nu - Q| < band`` around the linear tunes (see ``naff``) --
    without it a small-``x``, large-``y`` launch returns the driven ``2 Q_y`` line as its
    horizontal "tune"; ``None`` restores the unrestricted peak search."""
    opt = opt or linear_optics(ring, delta)
    dev, dt = ring.device, ring.dtype
    X, Y = torch.meshgrid(
        torch.linspace(*x_range, nx, dtype=dt, device=dev), torch.linspace(*y_range, ny, dtype=dt, device=dev), indexing="ij",
    )
    res = ring.track(launch(opt, X, Y), n_turns, aperture=aperture, record_every=1)
    zx, zy = normalized_signals(res.tbt, opt)
    h = n_turns // 2
    kw = ({"center": opt.Qx, "half_width": band}, {"center": opt.Qy, "half_width": band}) if band is not None else ({}, {})
    q = [naff(s[..., sl], **k)[0] for sl in (slice(0, h), slice(h, 2 * h)) for s, k in zip((zx, zy), kw)]
    dqx = torch.remainder(q[2] - q[0] + 0.5, 1.0) - 0.5
    dqy = torch.remainder(q[3] - q[1] + 0.5, 1.0) - 0.5
    lost = res.loss_turn >= 0
    nan = torch.tensor(math.nan, dtype=dt, device=dev)
    D = torch.where(lost, nan, 0.5 * torch.log10(dqx**2 + dqy**2 + 1e-300))
    return FrequencyMap(X, Y, torch.where(lost, nan, q[0]), torch.where(lost, nan, q[1]), D)
