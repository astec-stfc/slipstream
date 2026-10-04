"""Propagates nonperiodic linear Twiss/dispersion or extracts ensemble statistics from tracked
trajectories. See docs/modules/tracking/native/linac_twiss.md."""
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
    """Describe LinacTwissTable; see the module documentation for details."""

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
        """Compute save; see the module documentation for details."""
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
    """Compute rf forward and R; see the module documentation for details."""
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
    """Compute linac twiss table; see the module documentation for details."""
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
    """Dispersion-corrected ``(beta, alpha, D, Dp)`` from one plane's real particle ensemble -- ``D =
    cov(u,delta)/var(delta)``, removed from ``u``/``up`` before computing the geometric
    emittance/beta/alpha."""
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
    """Compute twiss from trajectory; see the module documentation for details."""
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
