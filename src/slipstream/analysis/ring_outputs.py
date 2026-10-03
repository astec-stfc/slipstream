"""CPU reporting and optional Matplotlib plots for frozen-energy rings.

Optics are linearized along the actual closed orbit using the configured
RingMap transport. Tables are detached pandas outputs, not autograd APIs.
Dispersion momenta use the native canonical px/py convention.
"""
from __future__ import annotations

import math

import pandas as pd
import torch

from ..constants import C_LIGHT
from ..tracking.native.ring import RingMap
from ..tracking.native.tracker import _apply_one_element
from ..tracking.native.optics import _propagate_twiss
from .nonlinear import chromaticity, linear_optics


def ring_optics_table(ring: RingMap, drift_samples: int = 8) -> pd.DataFrame:
    """Entrance optics, drift interiors and an END row for an uncoupled ring.

    Uses CPU float64 with the original slicing, offsets and strengths.
    Rejects coupling through ``linear_optics``. Dispersion is d(co)/d(delta).
    Phase accumulation assumes each element's phase advance is below 2*pi;
    subdivide very long focusing elements if an integer tune is ambiguous.
    ``drift_samples`` adds equally spaced interior points in plain drifts,
    showing straight-section waists; zero gives element boundaries only.
    """
    if not isinstance(drift_samples, int) or drift_samples < 0:
        raise ValueError('drift_samples must be a nonnegative integer')
    ref = ring.float64_reference()
    opt = linear_optics(ref)
    M = opt.M.detach()
    D = torch.linalg.solve(torch.eye(4, dtype=ref.dtype) - M[:4, :4], M[:4, 5])
    state = opt.closed_orbit.detach()
    bx, ax, by, ay = map(float, (opt.beta_x, opt.alpha_x, opt.beta_y, opt.alpha_y))
    s, mux, muy = 0., 0., 0.
    rows = []

    def row(name, kind, length):
        return dict(element=name, kind=kind, s_m=s, length_m=length,
                    beta_x_m=bx, alpha_x=ax, beta_y_m=by, alpha_y=ay,
                    D_x_m=float(D[0]), D_px=float(D[1]), D_y_m=float(D[2]), D_py=float(D[3]),
                    closed_x_m=float(state[0]), closed_px=float(state[1]),
                    closed_y_m=float(state[2]), closed_py=float(state[3]),
                    mu_x_rad=mux, mu_y_rad=muy)

    for spec, param in zip(ref.specs, ref.params):
        rows.append(row(spec.name, spec.kind, spec.length))
        if spec.kind == 'drift' and spec.length > 0 and not (param.multipoles or param.skew_multipoles or param.integrated_multipoles) and all(float(v) == 0 for v in (param.k1, param.k2, param.k3)):
            for j in range(1, drift_samples + 1):
                distance = spec.length * j / (drift_samples + 1)
                prefix = param._replace(length=distance)
                def partial(u):
                    return _apply_one_element(u, prefix, ref._energy_t, ref._mc2_t)[0]
                Rpart = torch.autograd.functional.jacobian(partial, state, vectorize=True).detach()
                sample = row(spec.name, spec.kind, 0.)
                sample['s_m'] = s + distance
                sample['beta_x_m'], sample['alpha_x'] = _propagate_twiss(bx, ax, Rpart[:2, :2])
                sample['beta_y_m'], sample['alpha_y'] = _propagate_twiss(by, ay, Rpart[2:4, 2:4])
                sample.update(zip(('D_x_m', 'D_px', 'D_y_m', 'D_py'), (Rpart[:4, :4] @ D + Rpart[:4, 5]).tolist()))
                sample.update(zip(('closed_x_m', 'closed_px', 'closed_y_m', 'closed_py'), partial(state)[:4].detach().tolist()))
                sample['mu_x_rad'] += math.atan2(float(Rpart[0, 1]), float(Rpart[0, 0]) * bx - float(Rpart[0, 1]) * ax) % (2 * math.pi)
                sample['mu_y_rad'] += math.atan2(float(Rpart[2, 3]), float(Rpart[2, 2]) * by - float(Rpart[2, 3]) * ay) % (2 * math.pi)
                rows.append(sample)
        def transport(u):
            return _apply_one_element(u, param, ref._energy_t, ref._mc2_t)[0]
        R = torch.autograd.functional.jacobian(transport, state, vectorize=True).detach()
        if max(float(R[:2, 2:4].abs().max()), float(R[2:4, :2].abs().max())) > 1e-10:
            raise ValueError('locally coupled elements require normal-mode optics, not per-plane Twiss reporting')
        mux += math.atan2(float(R[0, 1]), float(R[0, 0]) * bx - float(R[0, 1]) * ax) % (2 * math.pi)
        muy += math.atan2(float(R[2, 3]), float(R[2, 2]) * by - float(R[2, 3]) * ay) % (2 * math.pi)
        bx, ax = _propagate_twiss(bx, ax, R[:2, :2])
        by, ay = _propagate_twiss(by, ay, R[2:4, 2:4])
        D = R[:4, :4] @ D + R[:4, 5]
        state = transport(state).detach()
        s += spec.length
    rows.append(row('END', 'end', 0.))
    table = pd.DataFrame(rows)
    table.attrs.update(Qx=float(opt.Qx), Qy=float(opt.Qy),
                       one_turn_matrix=M.numpy(), dispersion0=table.iloc[0][['D_x_m', 'D_px', 'D_y_m', 'D_py']].to_numpy(dtype=float))
    return table


def ring_parameters(ring: RingMap, optics: pd.DataFrame | None = None,
                    charge_e: float = 1., include_chromaticity: bool = True) -> pd.DataFrame:
    """Derived machine parameters as columns ``parameter, value, unit``.

    Energy is total energy; rigidity uses the magnitude of charge in units of e.
    Slip factor includes the closed-orbit dispersion, and momentum compaction
    follows eta = alpha_c - 1/gamma**2 for relative momentum delta. No RF,
    damping or equilibrium-emittance values are inferred from a frozen map.
    Supply a table from the same ring to avoid recomputing its optics.
    """
    C = ring.circumference_m
    if not math.isfinite(C) or C <= 0 or not math.isfinite(charge_e) or charge_e == 0:
        raise ValueError('positive finite circumference and finite nonzero charge required')
    E, mass = float(ring.energy_eV), float(ring.mc2_eV)
    if not 0 < mass < E:
        raise ValueError('total energy must exceed positive rest energy')
    table = ring_optics_table(ring) if optics is None else optics
    gamma = E / mass
    beta = math.sqrt(1 - gamma**-2)
    M, D = table.attrs['one_turn_matrix'], table.attrs['dispersion0']
    eta = beta * float(M[4, 5] + M[4, :4] @ D) / C
    rows = [('energy', E, 'eV'), ('kinetic_energy', E - mass, 'eV'),
            ('momentum', E * beta, 'eV/c'), ('gamma', gamma, '1'), ('beta_rel', beta, '1'),
            ('circumference', C, 'm'), ('revolution_period', C / (beta * C_LIGHT), 's'),
            ('revolution_frequency', beta * C_LIGHT / C, 'Hz'),
            ('magnetic_rigidity', E * beta / (C_LIGHT * abs(charge_e)), 'T m'),
            ('Qx_fractional', table.attrs['Qx'], '1'), ('Qy_fractional', table.attrs['Qy'], '1'),
            ('Qx_accumulated', table.mu_x_rad.iloc[-1] / (2 * math.pi), '1'),
            ('Qy_accumulated', table.mu_y_rad.iloc[-1] / (2 * math.pi), '1'),
            ('slip_factor', eta, '1'), ('momentum_compaction', eta + gamma**-2, '1'),
            ('beta_x_max_sampled', table.beta_x_m.max(), 'm'),
            ('beta_y_max_sampled', table.beta_y_m.max(), 'm'),
            ('abs_Dx_max_sampled', table.D_x_m.abs().max(), 'm'),
            ('net_bending_angle', sum(spec.angle for spec in ring.specs), 'rad'),
            ('element_count', len(ring.specs), '1')]
    if include_chromaticity:
        xi = chromaticity(ring.float64_reference())
        rows += [('chromaticity_x', float(xi[0]), '1'), ('chromaticity_y', float(xi[1]), '1')]
    return pd.DataFrame(rows, columns=['parameter', 'value', 'unit'])


def ring_lattice_table(ring: RingMap) -> pd.DataFrame:
    """Configured element names, positions, geometry and K1/K2/K3 strengths.

    For one machine only; batch strength/alignment ensembles raise an error.
    Higher/skew tuples are retained as tuple columns. All values are SI.
    """
    def scalar(value):
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError('report one machine at a time, not an ensemble')
            return float(value.detach().cpu())
        return float(value)
    rows, s = [], 0.
    for spec, p in zip(ring.specs, ring.params):
        rows.append(dict(element=spec.name, kind=spec.kind, s_m=s, length_m=scalar(p.length),
                         angle_rad=p.angle, k1_m2=scalar(p.k1), k2_m3=scalar(p.k2), k3_m4=scalar(p.k3),
                         dx_m=scalar(p.dx), dy_m=scalar(p.dy), tilt_rad=scalar(p.tilt),
                         multipoles=tuple((n, scalar(v)) for n, v in p.multipoles),
                         skew_multipoles=tuple((n, scalar(v)) for n, v in p.skew_multipoles),
                         integrated_multipoles=tuple((n, scalar(v)) for n, v in p.integrated_multipoles)))
        s += spec.length
    return pd.DataFrame(rows)


def plot_ring_twiss(optics: pd.DataFrame, lattice: pd.DataFrame | None = None):
    """Return a Matplotlib Figure with beta, alpha, dispersion and optional layout.

    Matplotlib is imported only here; install ``slipstream-accel[plot]``.
    Samples are joined by lines; extrema between samples are not inferred.
    The caller owns saving, display and closing the figure.
    """
    import matplotlib.pyplot as plt
    count = 4 if lattice is not None else 3
    fig, axes = plt.subplots(count, 1, sharex=True, figsize=(10, 8),
                             gridspec_kw={'height_ratios': [3, 2, 2, 1] if count == 4 else [3, 2, 2]},
                             layout='constrained')
    for ax, columns, labels, ylabel in zip(axes,
            [('beta_x_m', 'beta_y_m'), ('alpha_x', 'alpha_y'), ('D_x_m', 'D_y_m')],
            [(r'$\beta_x$', r'$\beta_y$'), (r'$\alpha_x$', r'$\alpha_y$'), (r'$D_x$', r'$D_y$')],
            ['Beta [m]', 'Alpha', 'Dispersion [m]']):
        for column, label in zip(columns, labels):
            ax.plot(optics.s_m, optics[column], label=label)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.25)
        ax.legend(loc='upper right', ncol=2)
    if lattice is not None:
        _draw_layout(axes[-1], lattice)
    axes[-1].set_xlabel('s [m]')
    return fig


def _draw_layout(ax, lattice):
    from matplotlib.patches import Patch, Rectangle
    colors = {'bend': '#d69c22', 'quad': '#3973b9', 'sext': '#39a56e', 'oct': '#9762ac', 'multipole': '#9762ac', 'rf': '#d76863'}
    for row in lattice.itertuples():
        if row.kind not in colors:
            continue
        sign = -1 if row.kind == 'quad' and row.k1_m2 < 0 else 1
        ax.add_patch(Rectangle((row.s_m, min(0, sign)), max(row.length_m, .02), 1,
                               color=colors[row.kind], alpha=.85))
    ax.axhline(0, color='0.4', lw=.7)
    ax.set_ylim(-1.2, 2.3)
    ax.set_xlim(0, float((lattice.s_m + lattice.length_m).max()))
    ax.set_yticks([])
    present = set(lattice.kind)
    ax.legend(handles=[Patch(color=color, label=kind) for kind, color in colors.items() if kind in present],
              loc='upper right', ncol=5, fontsize=8)
    ax.set_ylabel('Lattice')


def plot_ring_layout(lattice: pd.DataFrame):
    """Return a standalone magnet-layout Figure from ``ring_lattice_table``."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 2), layout='constrained')
    _draw_layout(ax, lattice)
    ax.set_xlabel('s [m]')
    return fig


def plot_dynamic_aperture(result):
    """Return a DA boundary Figure; result may describe only the upper half-plane."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5), layout='constrained')
    x, y = result.x_m.detach().cpu().numpy(), result.y_m.detach().cpu().numpy()
    if x.ndim != 1:
        raise ValueError('plot one machine at a time')
    ax.plot(x * 1e3, y * 1e3, 'o-')
    ax.set(xlabel='Initial x [mm]', ylabel='Initial y [mm]', title='Sampled dynamic aperture')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(alpha=.25)
    return fig
