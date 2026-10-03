"""Per-element Twiss for a single-pass linac beamline
(``tracking.native.linac.LinacTracker``) -- two independent sources, both
returned in the same ``LinacTwissTable`` shape for direct, row-for-row
comparison: ``linac_twiss_table`` (linear-optics matrix propagation from
an explicit, non-periodic initial condition -- comparable to real
elegant's own ``&twiss_output matched=0`` ``.twi``) and
``twiss_from_trajectory`` (statistics extracted from the actual
multi-kick particle tracking -- comparable to real elegant's own tracked
``.sig`` file). A new, linac-specific table rather than a reuse of the
ring-only ``lattice.optics.TransverseOptics``/``lattice.twiss_table.TwissTable``
-- a linac's own reference energy varies element-to-element, and there is
no periodicity/tune to speak of.

Two non-obvious correctness points (see docs/tracking_native.md for the
validation numbers):

(1) The canonical-px/py vs. mechanical-slope Twiss convention is
equivalent for a magnetic element's linear R matrix (a magnetic element
never changes ``delta``, so the conversion Jacobian is the identity at
the linearization point). **It is not equivalent for an RF element**: the
on-axis design particle's own output ``delta`` is generally nonzero for
an accelerating cavity, so the canonical<->slope Jacobian at the output is
not the identity there (scales by ~``P0_entrance/P_exit``).
``_rf_forward_and_R`` differentiates a slopes-in/slopes-out *wrapper*
around ``elements._rf_cavity_kick`` for exactly this reason, matching
elegant's own mechanical-slope convention.

(2) An RF element's linear R matrix must never differentiate
``elements._match_energy_to_centroid``'s ensemble-mean ``change_p0``
recentering (a single-particle Jacobian's own "mean" is degenerate) -- the
reference energy instead advances via a single on-axis/design-particle
forward evaluation, matching elegant's own ``matched=0`` ``pCentral0``
convention.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path

import h5py
import numpy as np
import torch

from .elements import _canonical_from_slopes, _rf_cavity_kick, _slopes_from_canonical
from .lattice_import import ElementSpec
from .optics import _element_R, _propagate_twiss


@dataclass
class LinacTwissTable:
    """One row per injection point + element exit (``len(specs)+1`` rows,
    matching ``LinacTracker.track(..., return_trajectory=True)``'s own
    trajectory convention exactly -- row 0 is the injection condition,
    row ``i+1`` is the state after ``specs[i]``). Field names deliberately
    mirror ``lattice.twiss_table.TwissTable``'s own (``beta_x_m``,
    ``alpha_x``, ``D_x_m``, ``Dp_x``, ...) for familiarity, but this is a
    genuinely different type: a linac's own reference energy varies
    element-to-element (``energy_eV``, absent from the ring-only
    ``TwissTable``) and there is no periodicity/tune to speak of."""

    s_m: np.ndarray
    element_name: list[str]
    energy_eV: np.ndarray
    beta_x_m: np.ndarray
    alpha_x: np.ndarray
    D_x_m: np.ndarray
    Dp_x: np.ndarray
    beta_y_m: np.ndarray
    alpha_y: np.ndarray
    D_y_m: np.ndarray
    Dp_y: np.ndarray

    def save(self, path: str | Path, group: str = "/") -> None:
        """Append-mode HDF5 write (``data.results.TrackingResult``'s own
        one-dataset-per-field idiom) -- ``group=`` lets a linear-optics
        table and a tracking-derived table share one file
        (``table.save(path, group="linear")``, a second
        ``table.save(path, group="tracking")``) without one call
        clobbering the other. Existing datasets under the same
        name/group are replaced (safe to call twice against the same
        path while iterating), not merged."""
        with h5py.File(path, "a") as f:
            g = f if group in ("/", "") else f.require_group(group)
            for f_ in fields(self):
                name = f_.name
                if name in g:
                    del g[name]
                value = getattr(self, name)
                if name == "element_name":
                    g.create_dataset(name, data=np.array(value, dtype=h5py.string_dtype()))
                else:
                    g.create_dataset(name, data=np.asarray(value))

    @classmethod
    def load(cls, path: str | Path, group: str = "/") -> "LinacTwissTable":
        with h5py.File(path, "r") as f:
            g = f if group in ("/", "") else f[group]
            kwargs: dict = {}
            for f_ in fields(cls):
                name = f_.name
                raw = g[name][()]
                if name == "element_name":
                    kwargs[name] = [v.decode() if isinstance(v, bytes) else str(v) for v in raw]
                else:
                    kwargs[name] = raw
            return cls(**kwargs)


def _rf_forward_and_R(
    spec: ElementSpec, energy_eV: torch.Tensor, mc2_eV: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(R, energy_eV_new)`` for one RF-kind ``spec`` at its own entrance
    ``energy_eV`` -- ``R`` is the Jacobian of ``elements._rf_cavity_kick``
    at ``state6=0``, always with ``change_p0=False`` (must never
    differentiate the ensemble-mean recentering, see module docstring
    point 2). ``energy_eV_new`` is derived from that same forward call's
    output ``delta`` at ``state6=0`` (``P1=P0*(1+delta)``,
    ``energy_eV_new=mc2*sqrt(P1^2+1)``) -- the single reference particle's
    new energy, matching elegant's own ``matched=0`` ``pCentral0`` advance.
    No wake args -- elegant's own linear ``.twi`` calc ignores wakes/charge.

    Uses ``spec.n_kicks`` (the real per-sub-kick count), not a hardcoded
    ``1``: elegant's own ``matched=0`` linear-optics calculation resolves
    the element's real ``N_KICKS`` for the reference-energy advance (see
    docs/tracking_native.md for the live-elegant cross-check numbers).

    ``R`` is the Jacobian of a **slopes-in/slopes-out** wrapper around
    ``_rf_cavity_kick`` (converting through canonical only internally),
    not a bare canonical-in/canonical-out Jacobian -- see module docstring
    point 1 for why. Confirmed against a live-elegant finite-difference
    R-matrix to ~1e-9 relative for CLARA's own L02 parameters."""
    omega_rf = 2.0 * math.pi * spec.frequency_Hz

    def f_canonical(state6: torch.Tensor) -> torch.Tensor:
        out, _ = _rf_cavity_kick(
            state6.unsqueeze(0), spec.length, spec.voltage_V, spec.phase_rad, omega_rf,
            energy_eV, mc2_eV, change_p0=False, n_kicks=spec.n_kicks, beta_wave=spec.beta_wave,
            end1_focus=spec.end1_focus, end2_focus=spec.end2_focus, body_focus_model=spec.body_focus_model,
        )
        return out.squeeze(0)

    def f_slopes(state6_slopes: torch.Tensor) -> torch.Tensor:
        x, xp, y, yp, tau, delta = state6_slopes.unbind(-1)
        px, py = _canonical_from_slopes(xp, yp, delta)
        out = f_canonical(torch.stack([x, px, y, py, tau, delta], dim=-1))
        xo, pxo, yo, pyo, tauo, deltao = out.unbind(-1)
        xpo, ypo, _ = _slopes_from_canonical(pxo, pyo, deltao)
        return torch.stack([xo, xpo, yo, ypo, tauo, deltao], dim=-1)

    x0 = torch.zeros(6, dtype=torch.float64)
    R = torch.autograd.functional.jacobian(f_slopes, x0)
    out0 = f_canonical(x0)
    delta_out = out0[5]
    gamma0 = energy_eV / mc2_eV
    P0 = torch.sqrt(gamma0 * gamma0 - 1.0)
    P1 = P0 * (1.0 + delta_out)
    energy_eV_new = torch.sqrt(P1 * P1 + 1.0) * mc2_eV
    return R, energy_eV_new


def linac_twiss_table(
    specs: list[ElementSpec], energy_eV: float, mc2_eV: float,
    beta_x0: float, alpha_x0: float, beta_y0: float, alpha_y0: float,
    Dx0: float = 0.0, Dpx0: float = 0.0, Dy0: float = 0.0, Dpy0: float = 0.0,
    n_kicks: int = 16, n_kicks_bend: int = 64, integration_order: int = 4,
) -> LinacTwissTable:
    """Linear-optics Twiss propagated from an explicit, non-periodic
    initial condition through ``specs`` in order -- the linac analogue of
    ``optics.periodic_twiss_table``, minus the periodic-matching solve and
    generalized to a running reference energy (an RF-kind element changes
    it). Reuses ``optics._element_R``/``optics._propagate_twiss`` for
    every magnetic-kind element; RF-kind elements go through
    ``_rf_forward_and_R`` instead. ``n_kicks``/``n_kicks_bend`` default to
    the same high slice counts ``periodic_twiss_table`` uses -- a one-time
    cost that only affects this table's accuracy, never any tracking loop.

    Returns ``len(specs)+1`` rows: row 0 is the given injection condition
    unchanged, row ``i+1`` the state after ``specs[i]``'s transport --
    matching ``LinacTracker.track(..., return_trajectory=True)``'s own
    convention, and elegant's per-element ``.twi`` row convention."""
    n = len(specs)
    s_m = np.zeros(n + 1)
    element_name = ["_INJECTION_"] + [s.name for s in specs]
    energy_eV_arr = np.zeros(n + 1)
    beta_x = np.zeros(n + 1)
    alpha_x = np.zeros(n + 1)
    D_x = np.zeros(n + 1)
    Dp_x = np.zeros(n + 1)
    beta_y = np.zeros(n + 1)
    alpha_y = np.zeros(n + 1)
    D_y = np.zeros(n + 1)
    Dp_y = np.zeros(n + 1)

    e_t = torch.tensor(energy_eV, dtype=torch.float64)
    mc2_t = torch.tensor(mc2_eV, dtype=torch.float64)
    bx, ax = beta_x0, alpha_x0
    by, ay = beta_y0, alpha_y0
    dx = torch.tensor([Dx0, Dpx0], dtype=torch.float64)
    dy = torch.tensor([Dy0, Dpy0], dtype=torch.float64)
    s = 0.0

    for i in range(n + 1):
        s_m[i] = s
        energy_eV_arr[i] = e_t.item()
        beta_x[i], alpha_x[i] = bx, ax
        D_x[i], Dp_x[i] = dx[0].item(), dx[1].item()
        beta_y[i], alpha_y[i] = by, ay
        D_y[i], Dp_y[i] = dy[0].item(), dy[1].item()
        if i == n:
            break

        spec = specs[i]
        if spec.kind == "rf":
            R, e_t = _rf_forward_and_R(spec, e_t, mc2_t)
        else:
            nk = n_kicks_bend if spec.angle != 0.0 else n_kicks
            R = _element_R(
                spec.length, spec.angle, spec.k1, spec.k2, spec.e1, spec.e2,
                e_t.item(), mc2_eV, nk, integration_order,
                spec.hgap, spec.fint, spec.edge_order, spec.edge1_effects, spec.edge2_effects,
            )
        bx, ax = _propagate_twiss(bx, ax, R[0:2, 0:2])
        by, ay = _propagate_twiss(by, ay, R[2:4, 2:4])
        dx = R[0:2, 0:2] @ dx + R[0:2, 5]
        dy = R[2:4, 2:4] @ dy + R[2:4, 5]
        s += spec.length

    return LinacTwissTable(
        s_m, element_name, energy_eV_arr, beta_x, alpha_x, D_x, Dp_x, beta_y, alpha_y, D_y, Dp_y,
    )


def _plane_twiss_from_ensemble(
    u: torch.Tensor, up: torch.Tensor, delta: torch.Tensor,
) -> tuple[float, float, float, float]:
    """Dispersion-corrected ``(beta, alpha, D, Dp)`` from one plane's real
    particle ensemble -- ``D = cov(u,delta)/var(delta)``, removed from
    ``u``/``up`` before computing the geometric emittance/beta/alpha.
    Matches elegant's own dispersion-corrected ``ecx``/``ecny``/
    ``betaxBeam``/``alphaxBeam`` ``.sig`` columns, not its uncorrected
    tracking-derived ones -- the standard treatment for a beam with a real
    energy-position correlation (present after any RF cavity, substantial
    after a bunch compressor)."""
    du = u - u.mean()
    dup = up - up.mean()
    dd = delta - delta.mean()
    var_dd = (dd * dd).mean()
    D = (du * dd).mean() / var_dd
    Dp = (dup * dd).mean() / var_dd
    du_c = du - D * dd
    dup_c = dup - Dp * dd
    var_u = (du_c * du_c).mean()
    var_up = (dup_c * dup_c).mean()
    cov = (du_c * dup_c).mean()
    emittance = torch.sqrt(torch.clamp(var_u * var_up - cov * cov, min=0.0))
    beta = var_u / emittance
    alpha = -cov / emittance
    return beta.item(), alpha.item(), D.item(), Dp.item()


def twiss_from_trajectory(
    trajectory: torch.Tensor, energy_trajectory: torch.Tensor, specs: list[ElementSpec],
) -> LinacTwissTable:
    """Dispersion-corrected Twiss extracted from an actual tracked particle
    ensemble, at each step of ``LinacTracker.track(particles6,
    return_trajectory=True)``'s own ``(len(specs)+1, N, 6)`` state
    trajectory (plus its matching energy trajectory) -- comparable to real
    elegant's tracked ``.sig`` file, not its linear ``matched=0`` ``.twi``
    (see ``linac_twiss_table`` for that).

    Runs entirely outside ``torch.compile`` -- a plain Python loop over the
    already-materialized trajectory (matches this repo's "diagnostic path
    bypasses compile" convention elsewhere); the compiled tracking that
    produced this trajectory already ran with zero graph breaks."""
    n_steps = trajectory.shape[0]
    element_name = ["_INJECTION_"] + [s.name for s in specs]
    s_m = np.zeros(n_steps)
    for i, spec in enumerate(specs):
        s_m[i + 1] = s_m[i] + spec.length

    energy_eV_arr = energy_trajectory.detach().cpu().numpy()
    beta_x = np.zeros(n_steps)
    alpha_x = np.zeros(n_steps)
    D_x = np.zeros(n_steps)
    Dp_x = np.zeros(n_steps)
    beta_y = np.zeros(n_steps)
    alpha_y = np.zeros(n_steps)
    D_y = np.zeros(n_steps)
    Dp_y = np.zeros(n_steps)

    for i in range(n_steps):
        x, px, y, py, tau, delta = trajectory[i].unbind(-1)
        xp, yp, _ = _slopes_from_canonical(px, py, delta)
        beta_x[i], alpha_x[i], D_x[i], Dp_x[i] = _plane_twiss_from_ensemble(x, xp, delta)
        beta_y[i], alpha_y[i], D_y[i], Dp_y[i] = _plane_twiss_from_ensemble(y, yp, delta)

    return LinacTwissTable(
        s_m, element_name, energy_eV_arr, beta_x, alpha_x, D_x, Dp_x, beta_y, alpha_y, D_y, Dp_y,
    )
