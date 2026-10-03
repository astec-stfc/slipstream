"""Real per-element Twiss functions for the native (zero-external hybrid backend) tracker --
feeds ``ibs.element_averaged_growth_rates`` directly from
``NativeTracker``'s own per-element ``(length, angle, k1, k2)`` list and
exact transport, generalizing ``tracking.external hybrid backend.optics.periodic_twiss``'s
closed-form method (``beta=M12/sin(mu)``, periodic dispersion from
``(I-R)^-1 @ r``) to a full per-element array instead of one matched point.
See docs/tracking_native.md.
"""
from __future__ import annotations

import functools
import math

import numpy as np
import torch

from ...lattice.twiss_table import TwissTable
from .elements import track_element
from .lattice_import import ElementSpec


@functools.lru_cache(maxsize=None)
def _element_R(
    length: float, angle: float, k1: float, k2: float, e1: float, e2: float,
    energy_eV: float, mc2_eV: float, n_kicks: int, integration_order: int = 4,
    hgap: float = 0.0, fint: float = 0.5, edge_order: int = 1,
    edge1_effects: int = 1, edge2_effects: int = 1,
) -> torch.Tensor:
    """6x6 linear transfer matrix for one element, via autograd Jacobian of
    ``elements.track_element`` at the reference orbit (``state6=0``) --
    ``lru_cache``d on the plain-float params since a real ring reuses the
    same few magnet *types* many times."""
    dtype = torch.float64

    def f(state6: torch.Tensor) -> torch.Tensor:
        return track_element(
            state6.unsqueeze(0), length, angle, k1, k2, e1, e2, energy_eV, mc2_eV,
            n_kicks=n_kicks, integration_order=integration_order,
            hgap=hgap, fint=fint, edge_order=edge_order,
            edge1_effects=edge1_effects, edge2_effects=edge2_effects,
        ).squeeze(0)

    x0 = torch.zeros(6, dtype=dtype)
    return torch.autograd.functional.jacobian(f, x0)


def _propagate_twiss(beta0: float, alpha0: float, R: torch.Tensor) -> tuple[float, float]:
    """Standard sigma-matrix Twiss propagation, ``sigma1 = R @ sigma0 @
    R.T`` for ``sigma = [[beta, -alpha], [-alpha, gamma]]``, expanded to
    the usual closed-form beta/alpha update -- ``R`` is one plane's 2x2
    transfer-matrix block."""
    gamma0 = (1.0 + alpha0 * alpha0) / beta0
    R11, R12, R21, R22 = (R[0, 0].item(), R[0, 1].item(), R[1, 0].item(), R[1, 1].item())
    beta1 = R11 * R11 * beta0 - 2.0 * R11 * R12 * alpha0 + R12 * R12 * gamma0
    alpha1 = -R11 * R21 * beta0 + (R11 * R22 + R12 * R21) * alpha0 - R12 * R22 * gamma0
    return beta1, alpha1


def _periodic_plane(Rblock: torch.Tensor, rcol: torch.Tensor) -> tuple[float, float, torch.Tensor, float]:
    """Periodic (matched) beta/alpha/dispersion/tune for one plane's 2x2
    transfer-matrix block and delta-column coupling -- same formulas as
    ``tracking.external hybrid backend.optics.periodic_twiss``'s own ``_plane`` helper."""
    cos_mu = ((Rblock[0, 0] + Rblock[1, 1]) / 2).clamp(-1, 1)
    mu = torch.arccos(cos_mu)
    sin_mu = torch.sin(mu)
    beta = (Rblock[0, 1] / sin_mu).item()
    alpha = ((Rblock[0, 0] - Rblock[1, 1]) / (2 * sin_mu)).item()
    D = torch.linalg.solve(torch.eye(2, dtype=torch.float64) - Rblock, rcol)
    Q_frac = (mu / (2 * math.pi)).item()
    return beta, alpha, D, Q_frac


def periodic_twiss_table(
    specs: list[ElementSpec], energy_eV: float, mc2_eV: float, n_kicks: int = 16, n_kicks_bend: int = 64,
    integration_order: int = 4,
) -> tuple[TwissTable, float, float]:
    """Real per-element ``TwissTable`` (beta(s)/alpha(s)/dispersion(s) at
    every element's *entrance*, ``s_m[i]`` the cumulative path length up to
    that point) from the periodic solution of the ring's own linear
    one-turn matrix -- feed this directly to
    ``ibs.element_averaged_growth_rates`` for a genuinely per-element IBS
    calculation using this tracker's own real lattice (no MAD-X round trip
    needed). Returns ``(table, Qx_frac, Qy_frac)`` -- the fractional tunes
    aren't part of ``TwissTable`` itself (a MAD-X-derived one has no single
    natural "one-turn matrix" reference point the same way), but are cheap
    byproducts of the same one-turn-matrix computation, so returned
    alongside rather than silently discarded.

    ``n_kicks``/``n_kicks_bend`` default to 16/64 here -- **deliberately
    much higher than ``NativeTracker``'s own per-turn tracking defaults of
    1/4**: a per-element linearization error that's negligible for per-
    particle tracking compounds multiplicatively across a 290-fold matrix
    product for tune/beta extraction, and ``beta=M12/sin(mu)`` is highly
    sensitive to the resulting phase error. This is a one-time,
    ``lru_cache``d cost with no per-turn tracking-speed impact, so there's
    no reason to reuse the tracker's cheaper default here. See
    docs/tracking_native.md for the convergence numbers.
    """
    Rs = [
        _element_R(
            s.length, s.angle, s.k1, s.k2, s.e1, s.e2, energy_eV, mc2_eV,
            n_kicks_bend if s.angle != 0.0 else n_kicks, integration_order,
            s.hgap, s.fint, s.edge_order, s.edge1_effects, s.edge2_effects,
        )
        for s in specs
    ]
    M = torch.eye(6, dtype=torch.float64)
    for R in Rs:
        M = R @ M

    beta_x0, alpha_x0, D_x0, Qx_frac = _periodic_plane(M[0:2, 0:2], M[0:2, 5])
    beta_y0, alpha_y0, D_y0, Qy_frac = _periodic_plane(M[2:4, 2:4], M[2:4, 5])

    n = len(specs)
    s_m = np.zeros(n)
    length_m = np.array([s.length for s in specs])
    beta_x = np.zeros(n)
    alpha_x = np.zeros(n)
    D_x = np.zeros(n)
    Dp_x = np.zeros(n)
    beta_y = np.zeros(n)
    alpha_y = np.zeros(n)
    D_y = np.zeros(n)
    Dp_y = np.zeros(n)

    bx, ax = beta_x0, alpha_x0
    by, ay = beta_y0, alpha_y0
    dx, dy = D_x0.clone(), D_y0.clone()
    s = 0.0
    for i, (spec, R) in enumerate(zip(specs, Rs)):
        s_m[i] = s
        beta_x[i], alpha_x[i], D_x[i], Dp_x[i] = bx, ax, dx[0].item(), dx[1].item()
        beta_y[i], alpha_y[i], D_y[i], Dp_y[i] = by, ay, dy[0].item(), dy[1].item()

        bx, ax = _propagate_twiss(bx, ax, R[0:2, 0:2])
        by, ay = _propagate_twiss(by, ay, R[2:4, 2:4])
        dx = R[0:2, 0:2] @ dx + R[0:2, 5]
        dy = R[2:4, 2:4] @ dy + R[2:4, 5]
        s += spec.length

    table = TwissTable(
        s_m=s_m, length_m=length_m, beta_x_m=beta_x, alpha_x=alpha_x, D_x_m=D_x, Dp_x=Dp_x,
        beta_y_m=beta_y, alpha_y=alpha_y, D_y_m=D_y, Dp_y=Dp_y,
    )
    return table, Qx_frac, Qy_frac

