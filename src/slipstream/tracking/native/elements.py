"""Native (zero-external hybrid backend) per-element transport -- pure PyTorch, transcribed
directly from real elegant's own C source. See plans/PHYSICS.md for the
full formula reference and plans/DONE/NATIVE_TRACKER_PLAN.md for
derivation/validation history.

- ``elegant/src/multipole.c``/``csbend.c``: each of the ``n_kicks`` slices
  is itself a 4th-order symplectic (Yoshida) splitting by default
  (``integration_order=4``, the ``BETA=2**(1/3)`` drift/kick fraction
  tables below). The real default ``n_kicks`` (``N_SLICES``) is not a
  uniform 4 -- KQUAD/KSEXT/KQUSE default to ``N_SLICES=1`` (4 kicks total
  from the splitting alone), CSBEND/CCBEND/CSRCSBEND default to
  ``N_SLICES=4`` (16 kicks total) -- see ``tracker.py``'s ``n_kicks``/
  ``n_kicks_bend`` constructor params.
- ``csbend.c``'s drift sub-step is an exact closed-form sector-bend drift
  in canonical coordinates -- ``_canonical_drift`` below, ``h=0``
  collapsing to a plain canonical straight-line drift, so one function
  covers DRIFT/KQUAD/KSEXT (``h=0``) and CSBEND (``h=angle/length != 0``).
- The kick step applies a curvature-path-length weight ``(1 + h*x)`` that
  a bend's K1/K2 content needs -- see ``_multipole_kick`` below. The
  pure-dipole term (``Fy=1`` at the reference orbit) is carried by this
  same kick, not folded into the drift.
- ``_edge_kick`` implements slope-space linear and Brown second-order
  edge maps (``EDGE_EFFECTS=1``), including ``HGAP``/``FINT`` focusing.
- ``reference_offset`` (below): elegant's own default kick counts leave
  the on-axis reference orbit at a real, nonzero fixed point for a bend
  with real K1+K2 -- elegant itself avoids this via an explicit
  reference-trajectory subtraction; this reproduces the fix at this
  module's own per-element granularity.

State convention: ``(x, px, y, py, tau, delta)``. ``px``/``py`` are
*canonical* transverse momenta (``Px/p0``, elegant/bmad convention), not
slope ``dx/ds`` -- so ``tracker.py``'s boundary with ``pystable.rf_kick``/
``damping_excitation`` (which only touch ``tau``/``delta``) is unchanged.
``tau`` is path-length-like (``tau/C_LIGHT`` = time-of-flight surplus
relative to the reference particle, "arriving late" convention, matching
``tracking.pystable.tracker.rf_kick``'s own ``tau_s``).

Deliberately out of scope (see plans/PHYSICS.md): CSBEND's higher-order
(``h^2``, ``h^3``, ...) field-expansion corrections beyond dipole+K1+K2;
alternative edge models (``EDGE_EFFECTS>1``) and pole-face-curvature
(``H1``/``H2``) terms.
"""
from __future__ import annotations

import functools
import math

import torch

from ...constants import C_LIGHT
from ...wakefields.shortrange import ShortRangeWake
from ...wakefields.transverse import TransverseWake

# The trivial order-2 (Strang/leapfrog) splitting: drift(ds/2), kick(ds),
# drift(ds/2) -- the base case _yoshida_extend below recursively builds
# every higher even order from.
_DRIFT_FRAC2 = (0.5, 0.5)
_KICK_FRAC2 = (1.0, 0.0)

# elegant's own defaults: DEFAULT_N_KICKS=4, integration_order=4 (4th-order
# Yoshida symplectic splitting) -- BETA=2**(1/3), transcribed verbatim from
# multipole.c/csbend.c's driftFrac4/kickFrac4 tables.
_BETA = 2.0 ** (1.0 / 3.0)
_DRIFT_FRAC4 = (0.5 / (2 - _BETA), (1 - _BETA) / (2 - _BETA) / 2, (1 - _BETA) / (2 - _BETA) / 2, 0.5 / (2 - _BETA))
_KICK_FRAC4 = (1.0 / (2 - _BETA), -_BETA / (2 - _BETA), 1.0 / (2 - _BETA), 0.0)

# elegant's optional 6th-order splitting (INTEGRATION_ORDER=6, 8 sub-stages)
# -- transcribed verbatim from multipole.c/csbend.c's driftFrac6/kickFrac6
# tables, both citing "AOP-TN-2020-064".
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
    """Yoshida's recursive "triple jump" composition (H. Yoshida,
    "Construction of higher order symplectic integrators", Phys. Lett. A
    150 (1990) 262-268): given any symmetric drift/kick splitting of
    accuracy ``order``, composing three copies at different scales,
    ``S_new(h) = S(x1 h) . S(x0 h) . S(x1 h)`` with ``x1 = 1/(2 -
    2**(1/(order+1)))``, ``x0 = 1 - 2*x1``, cancels the leading error term
    and yields a splitting of accuracy ``order+2``. This is exactly how
    ``_BETA``/``_DRIFT_FRAC4``/``_KICK_FRAC4`` are constructed from the
    trivial order-2 splitting -- verified by
    ``tests/test_native_tracker.py``'s
    ``test_yoshida_extend_reproduces_order4_table``.

    Applying this once more to elegant's own order-6 table gives a genuine
    8th-order splitting (``_DRIFT_FRAC8``/``_KICK_FRAC8`` below) with no
    external reference needed -- elegant itself has no order-8 option, so
    this is a native-only extension, verified by an empirical
    convergence-rate test (error ~ ds^8 under step refinement).

    Adjacent drift fragments at the seams between the three composed
    copies are summed into one (the standard optimization turning the
    naive ``3*len(drift_frac)`` raw fragments into
    ``3*len(drift_frac)-2``)."""
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


# Extends elegant's own (minimal, 8-stage) order-6 splitting one more
# level -- fewer stages and smaller error than a from-scratch
# order-4-derived order-6 for the same n_kicks (verified empirically).
_DRIFT_FRAC8, _KICK_FRAC8 = _yoshida_extend(_DRIFT_FRAC6, _KICK_FRAC6, order=6)


def _canonical_drift(
    x: torch.Tensor, px: torch.Tensor, y: torch.Tensor, py: torch.Tensor, delta: torch.Tensor,
    ds: float, h: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact canonical drift over reference path-length ``ds`` through
    curvature ``h`` (a fixed lattice parameter, never particle data --
    branching on it in plain Python costs nothing under
    ``torch.compile``). Returns ``(x, px, y, py, arc_length)`` where
    ``arc_length`` is the true physical 3D path length traveled this
    sub-step (needed for the longitudinal update) -- differs from ``ds``
    for anything but a perfectly on-axis, on-momentum particle.

    For ``h != 0``, ``arc_length`` is **not** ``csbend.c``'s own ``factor``
    variable directly: ``factor`` is a canonical-momentum-normalized
    quantity, and the true physical arc length is ``factor*(1+delta)``
    (confirmed algebraically and matching elegant's own longitudinal
    bookkeeping, ``csbend.c``'s ``dist += factor*(1+DPoP)``). Using
    ``factor`` alone is correct for a pure drift (``h=0``) but wrong for a
    genuine bend -- see plans/PHYSICS.md's "arc_length vs factor" gotcha
    for the real ramped-tracking instability this caused before being fixed."""
    if h == 0.0:
        pz = torch.sqrt((1 + delta) ** 2 - px * px - py * py)
        x_new = x + ds * px / pz
        y_new = y + ds * py / pz
        arc_length = ds * (1 + delta) / pz
        return x_new, px, y_new, py, arc_length

    # Exact sector-bend geometry, transcribed from csbend.c's
    # integrate_csbend_ordn drift sub-step (asin/sin/cos construction).
    #
    # Both tang-tan_phi and g-1 below are rewritten via trig identities
    # instead of the direct textbook formulas -- necessary for float32:
    # theta-phi = ds*h is tiny per sub-step, so the direct formulas
    # subtract two nearly-equal values (catastrophic cancellation that
    # eats most of float32's precision, invisible in float64). See
    # plans/PHYSICS.md's bend-geometry float32-cancellation gotcha.
    #
    # Written without per-particle transcendentals: with sin(phi)=px/f and
    # f*cos(phi)=pz (phi in [-pi/2, pi/2], so cos(phi)>=0), the angle-sum
    # identities turn f*sin(theta) and f*cos(theta) into px/pz rotated by
    # the fixed angle ds*h (Python-float cos/sin constants). Algebraically
    # identical to csbend.c's asin/sin/cos form; ~3x cheaper per bend
    # sub-step on GPU, and only + - * / sqrt remain.
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
    # g - 1 = cos(phi)/cos(theta) - 1 = 2*sin(phi+ds*h/2)*sin(ds*h/2)/cos(theta)
    # (cos-difference product-to-sum identity) -- again avoids forming
    # g=cos_phi/cosi (~1) and subtracting 1.0 from it directly.
    g_minus_1 = 2.0 * s2 * (px * c2 + pz * s2) / pz_theta
    x_new = x + (rho0 + x) * g_minus_1
    arc_length = factor * (1 + delta)  # true physical arc length -- see docstring, NOT "factor" alone
    return x_new, px_new, y_new, py, arc_length


def _exact_dipole_drift(
    x: torch.Tensor, px: torch.Tensor, y: torch.Tensor, py: torch.Tensor, delta: torch.Tensor,
    ds: float, h: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact map of a pure sector dipole whose field matches the reference
    curvature ``h`` (Forest's exact sector bend, as in PTC/Bmad), over
    reference length ``ds`` -- the ``exact_dipole=True`` alternative to
    ``_canonical_drift`` + the dipole part of ``_multipole_kick``. Same
    Hamiltonian, different splitting (the kicks then carry only K1/K2).

    Derived from the invariant centre of the particle's horizontal circle,
    ``C = rho*(1 + h*x - pz, px)`` in the entrance frame, rotated by
    ``phi = h*ds``. Every quantity formed is O(deviation): ``pz - 1`` is
    computed as ``(delta*(2+delta) - px^2 - py^2)/(pz + 1)``, and an on-axis
    on-momentum particle is an exact fixed point. With elegant's splitting
    ``px`` instead swings by ~``h*ds`` every sub-step (0.014 rad on
    br14_simple vs a 1e-4 betatron signal) and float32 rounding of that
    swing dominated the tracking noise -- see
    plans/RING_NONLINEAR_DYNAMICS_PLAN.md (float32 precision)."""
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
    """``(dpx, dpy)`` of the canonical dipole+normal-quad+normal-sextupole kick over path-length
    ``ds``, with the curvature-path-length weight ``(1+h*x)`` real elegant
    applies (``QX += -ds*(1+X/rho0)*Fy/rho_actual`` in csbend.c) -- ``h=0``
    (KQUAD/KSEXT/DRIFT) reduces to the standard flat-space thin kick
    (``Delta_px = -(k1*x+0.5*k2*(x^2-y^2))*ds``).

    ``Fy``/``Fx`` are the standard combined-function-magnet field
    expansion, leading order (dipole+K1+K2) only -- ``b1=K1*rho0``,
    ``b2=K2*rho0`` in elegant's own normalization, so ``Fy = 1 + b1*x +
    0.5*b2*(x^2-y^2)`` un-normalizes to ``h + k1*x + 0.5*k2*(x^2-y^2)``."""
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
    """Real elegant ``HKICK``/``VKICK`` transport (``bend_matrix.c``'s
    ``corrector_matrix``/zero-length branch, transcribed with ``B2=0`` and
    no edge focusing -- this repo's real correctors set neither): a
    genuine dipole kick of angle ``kick`` over path length ``length``.

    Written entirely in ``sin(kick)`` (never ``1/kick``), so it stays
    finite and differentiable exactly at ``kick=0`` -- LOCO's own nominal
    operating point. This is deliberately **not** routed through
    ``track_element``'s own ``angle``/``h=angle/length`` path: that path
    computes ``math.cos(ds*h)``/``math.sin(ds*h)`` in plain Python,
    built for a fixed, never-differentiated lattice curvature (see
    plans/LOCO_PLAN.md's Phase-0/1 audit) -- a tensor ``h`` there would
    either error or silently detach.

    ``length<=0`` (zero-length corrector): elegant's own distinct
    zero-length matrix -- the leading term is a plain slope addition
    (``kick``, not ``sin(kick)``), elegant's own convention, confirmed
    against a live ``elegant`` run (``tests/test_native_loco.py``).
    Position is unchanged (no path to drift through).

    ``length>0``: exact reproduction of elegant's own ``corrector_matrix``
    at ``B2=0`` (every higher-order/edge term in that matrix carries a
    ``B2`` factor and vanishes identically) -- not a small-angle
    approximation, an exact closed form. ``tau`` gets the ordinary
    (``h=0``) drift arc-length update; the ``O(kick^2)`` curvature
    correction to arc length is dropped (real corrector kicks are
    milliradian-scale, and this only feeds RF-phase bookkeeping, never a
    LOCO-tapped BPM reading).

    Elegant's own ``R``/``C`` matrix entries act on **slope** coordinates
    (``x'``, ``y'``, its universal TRANSPORT-style matrix-method
    convention -- unlike the kick-method integrator ``track_element`` uses
    for CSBEND/KQUAD/KSEXT, which is canonical throughout), so this
    function converts to slopes (``_slopes_from_canonical``), applies the
    matrix, and converts back (``_canonical_from_slopes``) -- confirmed
    necessary (not a no-op) by a live-elegant check that fails by ~20% on
    ``xp``/``yp`` without it, see ``tests/test_native_loco.py``."""
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
    # (1 - delta + delta^2): elegant's own order-2 (default_order>=2) Taylor truncation of the exact
    # 1/(1+delta) momentum-dependence of a fixed-field kick's angle -- confirmed by a live-elegant
    # check that a bare (1-delta) leaves an xp residual growing exactly as sin(kick)*delta^2
    # (bend_matrix.c's corrector_matrix, T[1][5][5]=T[0][5][5]/(0.5*length)=K*length=sin(kick)).
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
    """Elegant style slope-space edge map (EDGE_EFFECTS=1).

    Includes finite-gap focusing and Brown second-order terms, including
    zero-angle faces.
    """
    if edge_order < 2 and edge_angle == 0.0 and hgap * fint == 0.0:
        return x, px, y, py
    xp, yp, _ = _slopes_from_canonical(px, py, delta)
    he = h / (1 + delta)
    # Under ``torch.compile`` these per-element floats may be represented as
    # SymFloats.  Dynamo can trace multiplication of opaque trig results but
    # currently fails while evaluating ``math.sin(symfloat) ** 2`` with fake
    # tensors (seen on the CUDA dipole-subcycle path).  Spell the squares as
    # products; this is algebraically identical and remains compile-safe.
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
    """Enter (``entering=True``)/exit (``False``) misalignment coordinate
    transform, transcribed from ``multipole.c``'s call sequence around the
    entire element body: ``offsetBeamCoordinatesForMisalignment(dx,dy,
    dz=0)`` then ``rotateBeamCoordinatesForMisalignment(tilt)`` at entry;
    the exact inverse, in reverse order, at exit. ``dz`` (longitudinal
    placement error) is not modeled.

    The rotation applies the same 2x2 rotation matrix to ``(x,y)`` and
    ``(px,py)`` independently -- valid for canonical momenta here since a
    rigid rotation of the transverse plane commutes with the
    slope<->canonical ``pz`` normalization (depends only on
    ``px^2+py^2``, rotation-invariant).

    **Bounded, documented approximation for CSBEND specifically**: real
    elegant gives bends a separate, exact rigid-body misalignment
    treatment accounting for the curved reference trajectory
    (``malign_mat.c``, cited to M. Venturini, ALSU-AP-TN-2021-001) -- not
    implemented here. This function's flat-space sandwich is applied
    uniformly to every element type, exact for ``h=0`` but an
    approximation for CSBEND, bounded the same way this module's other
    CSBEND simplifications are (``h*x ~ 1e-4`` for this repo's real
    bends). Revisit if a lattice with much larger bend curvature or
    particle amplitude needs the exact treatment.

    ``dx``/``dy``/``tilt`` may each be a 0-d tensor (LOCO's own alignment
    fit needs gradients exactly at the nominal zero-misalignment point) --
    ``_has_offset``/``_has_tilt`` (below) gate on ``isinstance`` first, not
    on value, so a differentiable knob is never silently detached by a
    plain-float ``!= 0.0`` branch the way a Python ``if`` alone would (see
    ``elements._corrector_kick``'s own docstring for the same trap with
    ``track_element``'s ``ANGLE``). ``torch.cos``/``torch.sin`` are used
    whenever ``tilt`` is a tensor; ``math.cos``/``math.sin`` otherwise
    (cheaper, and keeps every existing plain-float call site bit-identical)."""
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
    """The new reference momentum is the
    ensemble's own momentum-space mean, ``P_avg = mean_i[P0*(1+delta_i)]``
    -- not an energy-space mean, and not a ``Vrf*sin(phase)``-derived
    design value . ``dp=(P0-P_avg)/P_avg,
    dr=P0/P_avg, delta_new=dp+delta*dr``.
    This module's ``px``/``py`` are *canonical* (P0-
    normalized) momenta, so they do need rescaling by ``dr`` to keep the
    physical transverse momentum invariant when ``P0`` changes.

    Extracted from ``_rf_cavity_kick``'s own ``change_p0=True`` branch so
    it can be called after any element, not only from inside an RF kick
    -- see ``LinacTracker``'s ``always_change_p0``."""
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
    """Re-express ``state6`` (relative to a reference at ``energy_eV``) relative to a reference at
    ``energy_new_eV``, keeping every particle's physical momentum: the same ``delta``/``px``/``py``
    rescaling as ``_match_energy_to_centroid``, but to a *prescribed* reference instead of the
    ensemble mean (``LinacTracker.track(reference_energy_eV=...)``, a fixed design energy profile)."""
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
    """One RF sub-kick -- ``_rf_cavity_kick``'s own inner ``for ik in
    range(n_kicks)`` loop body, extracted verbatim so
    ``LinacTracker(compile=True)`` can compile this small, reusable unit
    instead of ``torch.compile`` unrolling the whole loop into one graph
    (real CLARA cavities set a genuine physical-cell ``N_KICKS`` in the
    hundreds -- see docs/tracking_native.md).

    ``phase_slip`` is passed in pre-computed (``(omega_rf/C_LIGHT) *
    (ik*length_k) * (1/beta0 - 1/beta_wave)``, from the caller's own
    ``ik``) rather than this function taking ``ik``/``beta0``/
    ``beta_wave`` and recomputing it: ``ik`` is a genuinely
    per-iteration-varying Python ``int``, and ``torch.compile``
    specializes (recompiles per distinct value) on a raw int argument by
    default -- exactly the problem this split exists to avoid.
    ``phase_slip`` as a plain ``float`` does not force a recompile in this
    codebase's torch version (``specialize_float=False``).
    ``is_first``/``is_last`` (computed by the caller) are the only
    remaining structural choices -- at most 3 compiled variants
    regardless of ``n_kicks``."""
    def _drift_half(x, px, y, py, tau, delta, e_eV):
        beta_ref = (e_eV * e_eV - mc2_eV * mc2_eV) ** 0.5 / e_eV
        beta = _beta_of(delta, e_eV, mc2_eV)
        x, px, y, py, arc_length = _canonical_drift(x, px, y, py, delta, half, 0.0)
        tau = tau + (arc_length / beta - half / beta_ref)
        return x, px, y, py, tau

    is_srs = body_focus_model == "srs"

    if end1_focus and is_first and not is_srs:
        # Pre-drift phase/dgamma estimate, entrance tau -- numerically
        # indistinguishable from elegant's own pre-drift dc4 estimate.
        # elegant's own *kick-method* branch (this "not is_srs" case)
        # computes END1's dgamma this way; its *matrix-method* branch
        # (is_srs) instead reuses the sub-kick's own center-of-cavity
        # dgamma -- see the is_srs-specific END1 application below.
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
        # elegant's own matrix-method branch stays in **slope** space for END1 (unlike the kick-method
        # branch's own canonical pz_norm-translated version above) -- SRS's R-matrix is exponentially
        # sensitive to its entrance slope, so the kick-method's canonical approximation isn't good enough
        # here. See plans/DONE/CLARA_COLLECTIVE_GAP_PLAN.md for the SRS closure numbers.
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
        # Rosenzweig & Serafini standing-wave R-matrix (elegant's own
        # simple_rfca.c, useSRSModel branch) -- REPLACES the drift-kick-
        # drift decomposition above entirely for x/y (the two half-drifts
        # computed into `x`/`px`/`y`/`py` above are discarded here, kept
        # only for their own `tau`/phase-at-center side effect, identical
        # regardless of body_focus_model). Applied to the sub-kick's own
        # entrance state, over the *full* `length_k` (not `half`) -- see
        # ElementSpec's own body_focus_model docstring. `xp_entrance`/
        # `yp_entrance` already include END1's own slope-space kick above
        # (a no-op, same values, when END1 doesn't apply this sub-kick).
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
        # elegant's own matrix-method branch re-derives its "coord[4]"
        # (path length) via a plain (not beta_ref-relative) arc length,
        # then rescales the whole accumulated value by the exit/entrance
        # beta ratio at the very end ("adjust s for the new particle
        # velocity") -- this module's own tau convention already folds an
        # equivalent beta_ref-relative correction into `_drift_half`
        # (validated for "none"/"tw1"), so this multiplicative ratio is
        # the residual correction matrix-method mode alone still needs.
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
    """Real elegant's own RFCA transport (``simple_rfca.c``):
    ``drift(length/2) -> thin energy kick -> drift(length/2)``. The drift
    halves reuse ``_canonical_drift``'s ``h=0`` branch.

    **The kick does NOT reuse ``pystable.rf_kick``**: real per-cavity RFCA
    gives every particle (including the one exactly on-crest) the full
    ``Vrf*sin(phase)`` energy gain relative to the unchanged old
    reference -- unlike ``rf_kick``'s own convention (designed for the
    once-per-turn lumped model), which subtracts a ``sin(phi_s)``
    baseline so the on-crest particle nets zero. See plans/PHYSICS.md
    Gotcha 20.

    Follows ``simple_rfca.c``'s ``add_to_particle_energy`` exactly (full
    relativistic relation in mc-normalized momentum ``P``, not the
    ultra-relativistic ``delta += dgamma/P0`` shortcut): ``volt =
    Vrf_eV/mc2_eV``, ``dgamma = volt*sin(phase_rad + omega_rf*tau_s)``,
    ``P0 = sqrt(gamma0^2-1)``, ``P = P0*(1+delta)``, ``gamma =
    sqrt(P^2+1)``, ``gamma1 = gamma+dgamma``, ``delta_new =
    (sqrt(gamma1^2-1)-P0)/P0``.

    Returns ``(state6_new, energy_eV_new)``. ``change_p0`` defaults to
    ``False``, matching real elegant's own ``RFCA`` default -- the
    reference is returned unchanged. ``change_p0=True`` reproduces
    elegant's own ``CHANGE_P0=1`` exactly via
    ``_match_energy_to_centroid``.

    ``wake``/``transverse_wake_x``/``transverse_wake_y`` (all ``None`` by
    default) make this an ``RFCW`` equivalent: real elegant's own wake
    placement is ``drift(L_k/2) -> RF energy kick -> [wake kick(s)] ->
    drift(L_k/2)`` per sub-kick -- the wake sits between the energy kick
    and the following half-drift, at the same ``tau`` already computed
    for the RF phase, applied once per sub-kick using the same, undivided
    ``charge_per_particle_C`` each time. Reuses
    ``ShortRangeWake.voltage_at``/``TransverseWake.kick_at`` unmodified
    and the same sign/normalization convention
    ``tracker._rf_and_damping`` uses for the once-per-turn case.

    ``n_kicks`` (default ``1``) splits the cavity into ``n_kicks``
    sub-kicks of length/voltage ``length/n_kicks``, ``voltage_V/n_kicks``
    -- elegant's own real per-cell discretization, a real structure's own
    physical cell count (one RF kick + wake per cell).

    ``beta_wave`` (default ``1.0``, **not** the reference's own ``beta0``)
    is the traveling wave's own phase velocity, in units of ``c``. Real
    elegant's ``RFCA``/``RFCW`` have no phase-velocity parameter at all --
    ``beta_wave=1`` (light speed) is hardcoded. This is the physically
    correct default for a real fixed-cell-length structure (most real
    electron linacs, including CLARA's, use a constant-phase-velocity
    structure regardless of the injected beam's own energy, so a low-beta
    injected bunch genuinely phase-slips against the wave). A
    beta-matched structure (common in proton linacs) needs ``beta_wave``
    set explicitly -- nothing in a real elegant ``.lte`` file signals
    this.

    **Derivation**: a particle's local phase relative to a wave
    ``E(z,t)=E0*sin(omega*t - k*z + phase_0)``, ``k=omega/(beta_wave*c)``,
    evolves along ``z`` as ``dpsi/dz = (omega/c)*(1/beta(z) -
    1/beta_wave)``. Splitting this integral at ``beta_wave = beta0`` (the
    reference's own fixed entrance velocity) leaves exactly the
    reference-relative ``omega*tau_s`` term this function already
    computes, plus a purely reference-position-dependent residual,
    ``(omega/c)*s*(1/beta0 - 1/beta_wave)``. Anchored to the convention
    ``phase_rad`` already uses (the phase at the center of the first
    cell) by subtracting this residual's own ``ik=0`` value -- so the
    correction is exactly zero at ``n_kicks=1`` regardless of
    ``beta_wave``. The per-sub-kick correction is
    ``(omega_rf/C_LIGHT) * (ik*length_k) * (1/beta0 - 1/beta_wave)``,
    ``beta0`` from the entrance ``energy_eV`` (fixed for the whole
    element). Verified against live elegant's real ``RFCA`` multi-kick
    output to ~1e-7 relative -- see plans/PHYSICS.md's multi-kick section.

    ``change_p0`` (if set) is applied once, after the last sub-kick's own
    energy kick + wake but **before** that sub-kick's own trailing
    half-drift, not after -- the trailing half-drift's own reference-beta
    term must use the new, post-match reference beta. See
    plans/PHYSICS.md's multi-kick section for the live-elegant
    chained-cavity check that caught getting this ordering wrong.

    ``end1_focus``/``end2_focus``/``body_focus_model`` add real elegant's
    own RF-focusing (``simple_rfca.c``). ``end1_focus``/``end2_focus`` are
    thin-lens kicks at the cavity's own entrance (``ik==0``, using a
    phase/``dgamma`` estimate from the entrance ``tau``) and exit
    (``ik==n_kicks-1``, using the exit ``gamma1``/``dgamma`` already
    computed that iteration), independent of ``body_focus_model`` and
    applied in every mode. This
    module's state is **canonical** -- translated via ``d(px_canonical) =
    -(Pz/P0)*x*inverseF``.

    ``body_focus_model="tw1"`` (real elegant's own traveling-wave RF
    focusing) adds one real per-sub-kick term on top of the ordinary
    drift-kick-drift transport: ``dpr = (voltage_k/mc2)/(2*beta_i) *
    (omega_rf/C_LIGHT) * (1-beta_i) * cos(phase_arg)``, translating into
    ``d(px_canonical) = dpr*x/P0`` (canonical is linearly proportional to
    physical momentum via the fixed reference ``P0``, so no approximation
    needed here unlike the slope-based END1/END2 kicks). For ``x=y=0``
    (the on-axis design particle) both END1/END2 and ``dpr`` kicks vanish
    identically, so the reference-energy advance
    (``linac_twiss._rf_forward_and_R``) is unaffected.

    ``body_focus_model="srs"`` (real CLARA L02/L03/L04's own actual
    setting -- forces elegant's own ``STANDING_WAVE=1`` internally too, per
    ``identifyRfcaBodyFocusModel``) **replaces** the drift-kick-drift
    transport for x/y entirely with the closed-form Rosenzweig & Serafini
    standing-wave R-matrix (elegant's own ``useSRSModel`` branch): ``alpha
    = ln(gamma1/gamma) / (2*sqrt(2)*sin(phase_arg))``
    (``dgammaMax/gamma/(2*sqrt(2))`` in the ``sin(phase_arg)~0`` limit
    elegant itself guards against), ``R11 = cos(alpha)``, ``R22 =
    R11*gamma/gamma1``, ``R12 = 2*sqrt(2)*gamma*length_k/dgammaMax*
    sin(alpha)``, ``R21 = -sin(alpha)*dgammaMax/(length_k*gamma1*2*
    sqrt(2))``, applied to the sub-kick's own **entrance** ``(x, xp)``/
    ``(y, yp)`` (slope convention) over the **full** ``length_k``, not
    composed with the two half-drifts the way ``"tw1"``'s additive term
    is. END1/END2_FOCUS still apply as above, independent of
    ``BODY_FOCUS_MODEL``, acting on the already-SRS-transported state for
    END2. See ``plans/DONE/CLARA_COLLECTIVE_GAP_PLAN.md`` for the
    live-elegant validation that identified SRS as CLARA's real body-focus
    model.

    ``compile_sub_kicks`` (default ``False``) makes the ``ik`` loop call a
    single compiled ``_rf_one_sub_kick`` instead of the plain eager one --
    the loop itself always stays a plain eager Python ``for``. Only
    ``LinacTracker`` sets this ``True``; ``NativeTracker``'s existing
    ``rf="per_element"`` usage never passes it.
    """
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
        # phase_slip computed here (a plain float, not the loop index ik
        # itself) -- see _rf_one_sub_kick's own docstring for why. Zero at
        # ik=0 regardless of beta_wave -- see this function's own
        # "anchored to phase_rad's convention" derivation.
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
    """Track ``state6`` (``..., 6`` -- any leading batch shape) through one
    element: misalignment entry transform (if ``dx``/``dy``/``tilt``
    given) -> entrance edge kick (if ``angle != 0``) -> ``n_kicks`` slices
    of the ``integration_order``-th-order (4, 6, or 8) Yoshida drift/kick
    splitting -> exit edge kick -> misalignment exit transform. A single
    generalized integrator for every element type this repo's real
    lattices use (DRIFT: ``angle=k1=k2=0``; KQUAD: ``angle=k2=0``; KSEXT:
    ``angle=k1=0``; CSBEND: all four generally nonzero). No precomputed
    2nd-order Taylor map anywhere: every element is exactly integrated
    every turn.

    ``integration_order=6`` (elegant's ``INTEGRATION_ORDER=6``, 8
    sub-stages per slice) is opt-in, matching elegant's own default of 4
    -- this repo's own real machine lattice
    (``BR14_BST_Aligned_Matched.lte``) uses order 6 with fewer slices for
    its nonzero-K2 KSEXTs/CSBENDs. ``integration_order=8``
    (``_DRIFT_FRAC8``/``_KICK_FRAC8``) is a native-only extension elegant
    doesn't have.

    ``dx``/``dy``/``tilt`` (misalignment, see ``_misalign``) default to
    no-op -- see ``tracking/native/errors.py``.

    ``exact_dipole=True`` (bends only, native-only, not elegant's splitting):
    each drift sub-step is the exact pure-dipole map
    (``_exact_dipole_drift``) and the kicks carry only K1/K2 -- same
    Hamiltonian, converges to the same result, but float32-robust and a
    pure dipole is then exact in one step. See ``_exact_dipole_drift``.
    """
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

    # isinstance-gated (not just != 0.0) so a differentiable tensor
    # dx/dy/tilt at its nominal zero value still routes through
    # _misalign -- see _misalign's own docstring.
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
    """``track_element`` applied to the exact reference orbit (``state6=0``)
    -- ideally exactly zero (an on-axis, on-momentum particle should be an
    exact fixed point of any real element), but real elegant's own default
    kick counts are tuned for per-particle tracking accuracy, not for
    making the reference orbit an exact numerical fixed point. Applied
    every turn across many bends in a real ring, this residual compounds
    into a multi-hundred-turn instability if uncorrected -- see
    plans/PHYSICS.md's "non-fixed-point reference orbit" gotcha for the
    measured numbers.

    Real elegant does not have this problem despite using the same finite
    ``N_SLICES``: it explicitly records a reference trajectory once per
    unique bend configuration and subtracts it from every particle
    (``csbend.c``'s ``RECORD_TRAJECTORY``/``SUBTRACT_TRAJECTORY``,
    per-slice). This reproduces the same fix at this module's own level
    of abstraction (per-element, not per-slice) -- see
    ``tracking.native.tracker._apply_elements``, which subtracts this
    from every real particle's own output. ``lru_cache``d like
    ``_element_R``: the same magnet type recurs many times in a real
    ring, and this never touches the per-turn tracking loop directly
    (computed once, stored in ``tracker._ElementParams``, then just a
    plain tensor subtraction every turn)."""
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

