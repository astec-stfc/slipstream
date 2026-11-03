"""Localized longitudinal wake kicks for the native ``Injector`` (ASTRA ``&WAKE``, monopole only).
See docs/modules/tracking/native/injector_wake.md."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ...constants import C_LIGHT, E_CHARGE_C

METHODS = ("taylor_f", "monopole_f", "monopole_p")  # ASTRA Wk_Type, lower case without the trailing 'Method'


@dataclass
class InjectorWake:
    """One kick: when the bunch centroid reaches ``z_m`` every particle gets the energy change ``-q V(z)`` with the wake
    potential ``V(z) = scaling * sum_j q_j w(z_j - z)`` of the particles ahead of it (``z_j > z``), ``w`` the monopole
    table [V/C] of ``file`` (positive = retarding for like charges)."""
    z_m: float
    file: str
    method: str = "taylor_f"
    scaling: float = 1.0
    n_bin: int = 128  # equidistant bins over the bunch; ASTRA's own default of 10 (Wk_N_bin) is coarse

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"wake method {self.method!r} not in {METHODS}")
        if self.n_bin < 2:
            raise ValueError("a wake needs at least 2 bins")


def read_wake_table(path: str | Path, method: str = "taylor_f") -> tuple[np.ndarray, np.ndarray]:
    """The monopole table ``(s [m], w [V/C])`` of an ASTRA wake file.

    ``monopole_f``/``monopole_p`` files are ``N 0`` then ``N`` rows ``s w``. A ``taylor_f`` file is ``Nt 0`` then ``Nt``
    sub-tables ``[Nq Np]``, ``[R L]``, ``[C 10a+b]`` and ``Nq`` rows ``(s, q(s))`` and ``Np`` rows ``(s, p(s))``; the
    monopole is the one with code ``10a+b = 0`` and its ``q(s)`` is the wake (DESY 12-012)."""
    tok = np.array(Path(path).read_text().split(), dtype=float)
    if method != "taylor_f":
        n = int(tok[0])
        rows = tok[2:2 + 2 * n].reshape(n, 2)
        return rows[:, 0].copy(), rows[:, 1].copy()
    i = 2
    for _ in range(int(tok[0])):
        nq, npp = int(tok[i]), int(tok[i + 1])
        code = int(tok[i + 5])
        start = i + 6
        if code == 0:
            rows = tok[start:start + 2 * nq].reshape(nq, 2)
            return rows[:, 0].copy(), rows[:, 1].copy()
        i = start + 2 * (nq + npp)
    raise ValueError(f"{path}: no monopole (code 0) sub-table")


def wake_potential(z: torch.Tensor, q: torch.Tensor, s_tab: np.ndarray, w_tab: np.ndarray, n_bin: int) -> torch.Tensor:
    """``V_i = sum_j q_j w(z_j - z_i)`` over the particles ahead of particle ``i``, on ``n_bin`` equidistant bins.

    Charge goes to the bin nodes with linear weights, the potential at node ``k`` is the trapezoid sum
    ``q_k w(0)/2 + sum_{m>k} q_m w((m-k) h)`` and is interpolated back with the same weights. ``w`` is zero beyond the
    table. The result carries the units of ``q * w`` and is independent of the particle count's order."""
    z, q = z.double(), q.double()
    z0, z1 = z.min(), z.max()
    h = (z1 - z0) / n_bin
    if not bool(h > 0):  # all particles at one z: only the self term, half of w(0)
        return 0.5 * float(np.interp(0.0, s_tab, w_tab)) * q.sum().expand_as(z).clone()
    pos = (z - z0) / h
    k = pos.floor().clamp(0, n_bin - 1).long()
    f = pos - k
    nodes = torch.zeros(n_bin + 1, dtype=torch.float64, device=z.device)
    nodes.index_add_(0, k, q * (1.0 - f))
    nodes.index_add_(0, k + 1, q * f)
    lag = float(h) * np.arange(n_bin + 1)
    w_lag = torch.as_tensor(np.interp(lag, s_tab, w_tab, right=0.0), dtype=torch.float64, device=z.device)
    m = torch.arange(n_bin + 1, device=z.device)
    d = m[None, :] - m[:, None]  # d[k, m] = m - k
    kernel = torch.where(d > 0, w_lag[d.clamp(min=0)], torch.zeros((), dtype=torch.float64, device=z.device))
    kernel = kernel + 0.5 * w_lag[0] * (d == 0)
    v_nodes = kernel @ nodes
    return v_nodes[k] * (1.0 - f) + v_nodes[k + 1] * f


def apply_wake(wake: InjectorWake, table: tuple[np.ndarray, np.ndarray], z, q, ux, uy, uz, q_single_C: float,
               mass_kg: float, mask: torch.Tensor | None = None):
    """The longitudinal proper velocity after one kick and the mean energy change per particle [eV], over the
    particles selected by ``mask`` (the others are untouched and do not source the wake).

    ``q`` is the signed macroparticle charge (it sources the wake); ``q_single_C`` the signed charge of the particle the
    equations of motion track, so the energy change is ``-q_single_C * V`` per tracked particle. A pure longitudinal
    force changes ``u_z`` by ``gamma dE / (m u_z)``."""
    sel = slice(None) if mask is None else mask
    v = wake_potential(z[sel], q[sel], *table, wake.n_bin) * wake.scaling
    d_energy_J = -q_single_C * v
    u = uz[sel].double()
    gamma = torch.sqrt(1.0 + (ux[sel].double() ** 2 + uy[sel].double() ** 2 + u**2) / C_LIGHT**2)
    new = uz.clone()
    new[sel] = (u + gamma * d_energy_J / (mass_kg * u)).to(uz.dtype)
    return new, float(d_energy_J.mean()) / E_CHARGE_C
