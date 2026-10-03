"""MLS-inspired native-ring B2/B4/B5 demonstration.

Published geometry, representative hard-edge magnet settings, not an
operational MLS lattice. Short runs do not establish machine acceptance.
Install slipstream-accel[plot] for figures, or pass --no-plots for tables only.
"""
import argparse
import json
import math
from pathlib import Path

import torch
import pandas as pd

from slipstream import (
    plot_dynamic_aperture, plot_ring_layout, plot_ring_twiss,
    ring_lattice_table, ring_optics_table, ring_parameters,
)
from slipstream.analysis import nonlinear as nl
from slipstream.analysis.nonlinear_opt import (
    action_growth_loss, gradient_free, joint_optics_loss, multi_fidelity,
)
from slipstream.tracking.native.lattice_import import ElementSpec
from slipstream.tracking.native.ring import RingMap


# Rough envelope match to examples/MLS/Twiss.png with independent hard-edge
# matrices: Q=(3.18, 2.23), beta maxima ~15/9 m, dispersion maximum ~1.5 m.
# Order: Q1, Q2L, Q2S, Q3L, Q3S; SI m^-2. See examples/MLS/README.md.
MLS_K1 = (-0.7678415218043021, 3.29122194477513, 5.0602022474686,
          -2.847681806285773, -6.515853025684272)


def mls_specs(k1=MLS_K1):
    """48 m, twofold-symmetric DBA model with 8 bends, 24 quads/sextupoles.

    Geometry/counts: PTB and EPAC04 THPKF016; standard-operation energy
    and tunes: IPAC21 MOPAB262, Table 2. Individual lengths/strengths are
    illustrative, not surveyed MLS data. Four octupole slots have K3=0.
    https://www.ptb.de/cms/en/ptb/fachabteilungen/abt7/ptb-sr/mls/storage-ring.html
    https://epaper.kek.jp/e04/PAPERS/THPKF016.PDF
    https://proceedings.jacow.org/ipac2021/papers/mopab262.pdf
    """
    def half(long):
        suffix = 'L' if long else 'S'
        return [ElementSpec('DA', 'drift', .3),
            ElementSpec('Q3' + suffix, 'quad', .2, k1=k1[3 if long else 4]),
            ElementSpec('S1', 'sext', .1, k2=20.),
            ElementSpec('DB', 'drift', .3),
            ElementSpec('Q2' + suffix, 'quad', .2, k1=k1[1 if long else 2]),
            ElementSpec('DC', 'drift', .325),
            ElementSpec('B', 'bend', 1.3, angle=math.pi/4),
            ElementSpec('DD', 'drift', .3),
            ElementSpec('S2', 'sext', .1, k2=-20.),
            ElementSpec('Q1', 'quad', .2, k1=k1[0]),
            ElementSpec('S3', 'sext', .1, k2=0.),
            ElementSpec('DE', 'drift', .4)]
    specs = []
    for length in (6., 2.5, 6., 2.5):
        arc = half(length == 6.) + [ElementSpec('O', 'oct', .1)] + list(reversed(half(length != 6.)))
        specs += [ElementSpec('STRAIGHT', 'drift', length)] + arc
    return specs


def reference_outputs(ring, output_dir, plots=True):
    """Save nominal optics/layout/parameter tables and optional reference plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    optics = ring_optics_table(ring)
    lattice = ring_lattice_table(ring)
    parameters = ring_parameters(ring, optics)
    files = {}
    for name, table in [('optics', optics), ('lattice', lattice), ('machine_parameters', parameters)]:
        path = output_dir / (name + '.csv')
        table.to_csv(path, index=False)
        files[name] = str(path)
    (output_dir / 'machine_parameters.txt').write_text(parameters.to_string(index=False) + '\n', encoding='utf-8')
    if plots:
        try:
            import matplotlib
        except ImportError as exc:
            raise ImportError('Install slipstream-accel[plot] or pass --no-plots') from exc
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for name, fig in [('twiss', plot_ring_twiss(optics, lattice)), ('layout', plot_ring_layout(lattice))]:
            fig.suptitle('Nominal MLS-inspired model (illustrative)')
            path = output_dir / (name + '.png')
            fig.savefig(path, dpi=160)
            plt.close(fig)
            files[name] = str(path)
    return parameters, files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budget', type=int, default=6)
    parser.add_argument('--turns', type=int, default=32)
    parser.add_argument('--angles', type=int, default=5, help='DA launch rays in the upper half-plane')
    parser.add_argument('--radii', type=int, default=4, help='radial samples per refinement pass')
    parser.add_argument('--refine', type=int, default=1, help='additional DA refinement passes')
    parser.add_argument('--r-max', type=float, default=.002, help='largest DA launch radius [m]')
    parser.add_argument('--cma', action='store_true', help='include optional CMA-ES baseline')
    parser.add_argument('--output-dir', type=Path, default=Path('.cache/nonlinear_optimisation'),
                        help='directory for nominal reference tables and plots')
    parser.add_argument('--no-plots', action='store_true', help='write tables without requiring Matplotlib')
    parser.add_argument('--reference-only', action='store_true', help='write nominal reference outputs and exit')
    args = parser.parse_args()
    if min(args.budget, args.turns, args.radii) < 1 or args.angles < 3 or args.refine < 0 or not 0 < args.r_max <= .02:
        parser.error('positive budget/turns/radii, at least three angles and nonnegative refine required')
    base = RingMap(mls_specs(), energy_eV=629e6, device='cuda', dtype=torch.float32,
                   n_kicks=4, compile=True)
    parameters, reference_files = reference_outputs(base, args.output_dir, plots=not args.no_plots)
    if args.reference_only:
        print(parameters.to_string(index=False))
        print(json.dumps({'reference_files': reference_files}, indent=2))
        return
    target = nl.linear_optics(base)
    theta = torch.tensor([MLS_K1[0], 20., -20., 0.], dtype=torch.float64)
    bounds = [(MLS_K1[0] - .005, MLS_K1[0] + .005), (-150, 150), (-150, 150), (-150, 150)]
    state = nl.launch(target, torch.tensor([0.00005, 0.0001], dtype=base.dtype),
                      torch.tensor([0.00003, 0.00005], dtype=base.dtype))

    def machine(x):
        return base.with_strengths(k1={'Q1': x[0]}, k2={name: value for name, value in zip(('S1', 'S2', 'S3'), x[1:])})

    def smooth(x):
        ring = machine(x)
        result = ring.track(state, args.turns, record_every=1)
        if bool((result.loss_turn >= 0).any()):
            raise ValueError('reduce surrogate launches')
        return action_growth_loss(result.tbt, target).mean() + joint_optics_loss(ring, target)

    def scan(x):
        return nl.dynamic_aperture(machine(x), r_max=args.r_max, n_turns=2*args.turns,
                                   n_angles=args.angles, n_r=args.radii, refine=args.refine,
                                   aperture=(0.02, 0.02))

    def aperture(x):
        return -scan(x).area_m2 * 1e6  # minimise negative area in mm^2

    # A separately labelled nominal scan is outside the optimiser call budgets.
    nominal_da = scan(theta)
    path = args.output_dir / 'nominal_dynamic_aperture.csv'
    pd.DataFrame({name: value.detach().cpu().numpy() for name, value in
                  [('angle_rad', nominal_da.angle_rad), ('radius_m', nominal_da.radius_m),
                   ('x_m', nominal_da.x_m), ('y_m', nominal_da.y_m)]}).to_csv(path, index=False)
    reference_files['dynamic_aperture'] = str(path)
    if not args.no_plots:
        import matplotlib.pyplot as plt
        fig = plot_dynamic_aperture(nominal_da)
        fig.axes[0].set_title('Nominal sampled dynamic aperture')
        path = args.output_dir / 'nominal_dynamic_aperture.png'
        fig.savefig(path, dpi=160)
        plt.close(fig)
        reference_files['dynamic_aperture_plot'] = str(path)

    low, high = multi_fidelity(smooth, aperture, theta, bounds, gradient_steps=2,
                              learning_rate=1e-4, max_evaluations=args.budget, method='CMA-ES' if args.cma else 'Powell')
    rows = [{'method': 'gradient_then_' + high.method, 'loss': high.loss,
             'evaluations': high.evaluations, 'surrogate': low.loss,
             'strengths': high.theta.tolist()}]
    methods = ['Powell', 'Nelder-Mead', 'differential_evolution']
    if args.cma:
        methods.append('CMA-ES')
    for method in methods:
        result = gradient_free(aperture, theta, bounds, method=method,
                               max_evaluations=args.budget, seed=4)
        rows.append({'method': method, 'loss': result.loss, 'evaluations': result.evaluations,
                     'strengths': result.theta.tolist()})
    print(json.dumps({'model': 'MLS-inspired hard-edge approximation; not an operational lattice',
                      'energy_MeV': base.energy_eV / 1e6,
                      'circumference_m': base.circumference_m,
                      'fractional_tunes': [float(target.Qx), float(target.Qy)],
                      'chromaticity': [float(x) for x in nl.chromaticity(base)],
                      'knobs': ['Q1 K1 [m^-2]', 'S1 K2 [m^-3]', 'S2 K2 [m^-3]', 'S3 K2 [m^-3]'],
                      'loss_units': 'negative upper-half-plane DA area [mm^2]',
                      'reference_parameters': parameters.to_dict(orient='records'),
                      'reference_files': reference_files,
                      'nominal_DA_area_mm2': float(nominal_da.area_m2) * 1e6,
                      'results': rows}, indent=2))


if __name__ == '__main__':
    main()
