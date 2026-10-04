"""Diamond storage-ring sextupole dynamic-aperture demonstration on one superperiod.

Tracks SPERIOD from examples/diamond/diamond.lte (a MADInput conversion) at 3 GeV. SPERIOD is
one of six identical superperiods: six period passes make one ring turn, and per-period tunes
and chromaticities scale by six. HALFS1A/S2A (the dispersive SF/SD families) hold the ring
chromaticity at --xi through a float64 response matrix; the other six families are the DA
knobs. Hard-edge model without RF, radiation, insertion devices, errors or physical apertures
beyond one box check per period, so short runs do not establish machine acceptance.
Install slipstream-accel[plot] for figures, or pass --no-plots for tables only.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

from slipstream import (
    plot_dynamic_aperture, plot_ring_layout, plot_ring_twiss,
    ring_lattice_table, ring_optics_table, ring_parameters,
)
from slipstream.analysis import nonlinear as nl
from slipstream.analysis.nonlinear_opt import gradient_free
from slipstream.tracking.native.lattice_import import parse_elegant_lattice
from slipstream.tracking.native.ring import RingMap

HERE = Path(__file__).resolve().parent
PERIODS = 6
# Largest, well-conditioned chromatic response of the eight families (chromaticity_response.csv).
CORRECTORS = ('HALFS1A', 'S2A')
K2_LIMIT = 60.  # illustrative |K2| limit for every family [m^-3]; the lattice file peaks at 35.8


def reference_outputs(ring, output_dir, plots=True):
    """Save SPERIOD optics/layout tables, whole-ring parameters and optional reference plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    optics = ring_optics_table(ring)
    lattice = ring_lattice_table(ring)
    parameters = ring_parameters(ring, optics, periods=PERIODS)
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
            fig.suptitle('Diamond SPERIOD (one of six superperiods), 3 GeV')
            path = output_dir / (name + '.png')
            fig.savefig(path, dpi=160)
            plt.close(fig)
            files[name] = str(path)
    return parameters, files


def chromaticity_response(base, families):
    """Lattice-file K2, ring chromaticity and ring d(xi)/dK2 ``(2, n)`` by autograd on the float64 reference."""
    k2 = {s.name: s.k2 for s in base.specs}
    k0 = torch.tensor([k2[n] for n in families], dtype=torch.float64, requires_grad=True)
    ref = base.float64_reference().with_strengths(k2=dict(zip(families, k0)))
    xi = PERIODS * torch.stack(nl.chromaticity(ref, create_graph=True))
    A = torch.stack([torch.autograd.grad(xi[i], k0, retain_graph=True)[0] for i in range(2)])
    return k0.detach(), xi.detach(), A


def da_table(da):
    return pd.DataFrame({name: value.detach().cpu().numpy() for name, value in
                         [('angle_rad', da.angle_rad), ('radius_m', da.radius_m), ('x_m', da.x_m), ('y_m', da.y_m)]})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--budget', type=int, default=50, help='objective evaluations per method')
    parser.add_argument('--turns', type=int, default=4096, help='DA ring turns (six SPERIOD passes each)')
    parser.add_argument('--angles', type=int, default=36, help='DA launch rays in the upper half-plane')
    parser.add_argument('--radii', type=int, default=16, help='radial samples per refinement pass')
    parser.add_argument('--refine', type=int, default=2, help='additional DA refinement passes')
    parser.add_argument('--r-max', type=float, default=0.04, help='largest DA launch radius and box half-aperture [m]')
    parser.add_argument('--xi', type=float, nargs=2, default=(2., 2.), metavar=('XI_X', 'XI_Y'),
                        help='ring linear chromaticity held by HALFS1A/S2A')
    parser.add_argument('--window', type=float, default=8., help='K2 search half-width around the lattice file [m^-3]')
    parser.add_argument('--cma', action='store_true', help='include optional CMA-ES baseline')
    parser.add_argument('--output-dir', type=Path, default=HERE, help='directory for tables and plots')
    parser.add_argument('--no-plots', action='store_true', help='write tables without requiring Matplotlib')
    parser.add_argument('--reference-only', action='store_true', help='write nominal reference outputs and exit')
    args = parser.parse_args()
    if min(args.budget, args.turns, args.radii) < 1 or args.angles < 3 or args.refine < 0 or not args.r_max > 0 < args.window:
        parser.error('positive budget/turns/radii/r-max/window, at least three angles and nonnegative refine required')
    # Warp prints its init/module banners on stdout; keep that stream for the JSON report.
    stdout, sys.stdout = sys.stdout, sys.stderr
    base = RingMap(parse_elegant_lattice(HERE / 'diamond.lte', 'SPERIOD'), energy_eV=3e9,
                   device='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float32, n_kicks=4)
    parameters, files = reference_outputs(base, args.output_dir, plots=not args.no_plots)
    if args.reference_only:
        sys.stdout = stdout
        print(parameters.to_string(index=False))
        print(json.dumps({'reference_files': files}, indent=2))
        return

    # Chromaticity is affine in K2 with the quadrupoles fixed, so one response matrix makes
    # CORRECTORS hold --xi exactly (to float64 linear optics) for any free-family setting.
    families = sorted({s.name for s in base.specs if s.kind == 'sext'})
    free = [n for n in families if n not in CORRECTORS]
    k0, xi0, A = chromaticity_response(base, families)
    path = args.output_dir / 'chromaticity_response.csv'
    pd.DataFrame({'family': families, 'k2_m3': k0.numpy(), 'dxi_x_dk2': A[0].numpy(),
                  'dxi_y_dk2': A[1].numpy()}).to_csv(path, index=False)
    files['chromaticity_response'] = str(path)
    ic, jf = [families.index(n) for n in CORRECTORS], [families.index(n) for n in free]
    solve = torch.linalg.inv(A[:, ic])
    k0, xi0, A, solve = (t.to(base.device, base.dtype) for t in (k0, xi0, A, solve))
    xi_target = torch.tensor(args.xi, dtype=base.dtype, device=base.device)

    def strengths(x):
        """Free-family K2 -> every family's K2, CORRECTORS on the linear ring-chromaticity target."""
        k = k0.clone()
        k[jf] = x
        k[ic] = k0[ic] + solve @ (xi_target - xi0 - A[:, jf] @ (x - k0[jf]))
        return k

    def scan(x):
        ring = base.with_strengths(k2=dict(zip(families, strengths(x))))
        return nl.dynamic_aperture(ring, r_max=args.r_max, n_turns=PERIODS * args.turns, n_angles=args.angles,
                                   n_r=args.radii, refine=args.refine, aperture=(args.r_max, args.r_max))

    def aperture(x):
        k = strengths(x)
        excess = float((k.abs() - K2_LIMIT).clamp_min(0).max())
        # A derived corrector beyond the limit skips the scan; any positive value is worse than a DA.
        value = 10. * excess if excess > 0 else -float(scan(x).area_m2) * 1e6
        print(f"  Aperture Value {[round(float(v), 3) for v in k]} = {value}", file=sys.stderr)
        return value

    theta = k0[jf].clone()
    # Bounds set the optimisers' step scales (simplex, CMA sigma, DE population): the full +-K2_LIMIT
    # box makes steps of tens of m^-3 that collapse the DA, so search a window around the lattice file.
    bounds = [(max(-K2_LIMIT, v - args.window), min(K2_LIMIT, v + args.window)) for v in theta.tolist()]
    nominal_da = scan(theta)
    path = args.output_dir / 'nominal_dynamic_aperture.csv'
    da_table(nominal_da).to_csv(path, index=False)
    files['dynamic_aperture'] = str(path)

    methods = ['Powell', 'Nelder-Mead', 'differential_evolution'] + (['CMA-ES'] if args.cma else [])
    results = []
    for method in methods:
        print(f"===  Using Method = {method}  ===", file=sys.stderr)
        results.append(gradient_free(aperture, theta, bounds, method=method, max_evaluations=args.budget, seed=4))
    best = min(results, key=lambda r: r.loss)
    best_da = scan(best.theta)
    path = args.output_dir / 'optimised_dynamic_aperture.csv'
    da_table(best_da).to_csv(path, index=False)
    files['optimised_dynamic_aperture'] = str(path)
    if not args.no_plots:
        import matplotlib.pyplot as plt
        fig = plot_dynamic_aperture(nominal_da)
        fig.axes[0].set_title('Nominal sampled dynamic aperture')
        path = args.output_dir / 'nominal_dynamic_aperture.png'
        fig.savefig(path, dpi=160)
        files['dynamic_aperture_plot'] = str(path)
        ax = fig.axes[0]
        ax.lines[0].set_label(f'nominal, {float(nominal_da.area_m2) * 1e6:.0f} mm²')
        ax.plot(best_da.x_m.cpu().numpy() * 1e3, best_da.y_m.cpu().numpy() * 1e3, 's-',
                label=f'best optimised ({best.method}), {float(best_da.area_m2) * 1e6:.0f} mm²')
        ax.set_title(f'Sampled dynamic aperture, {args.turns} turns')
        ax.legend()
        path = args.output_dir / 'dynamic_aperture_comparison.png'
        fig.savefig(path, dpi=160)
        plt.close(fig)
        files['dynamic_aperture_comparison_plot'] = str(path)

    value = parameters.set_index('parameter').value
    sys.stdout = stdout
    print(json.dumps({'model': 'Diamond SPERIOD hard-edge model from diamond.lte; one of six superperiods',
                      'energy_GeV': base.energy_eV / 1e9,
                      'periods': PERIODS,
                      'circumference_m': PERIODS * base.circumference_m,
                      'tunes': [float(value['Qx_accumulated']), float(value['Qy_accumulated'])],
                      'chromaticity_lattice_file': xi0.tolist(),
                      'chromaticity_target': list(args.xi),
                      'chromaticity_correctors': list(CORRECTORS),
                      'free_knobs': [n + ' K2 [m^-3]' for n in free],
                      'k2_limit_m3': K2_LIMIT,
                      'search_window_m3': args.window,
                      'loss_units': 'negative upper-half-plane DA area [mm^2]; positive = corrector beyond limit',
                      'reference_parameters': parameters.to_dict(orient='records'),
                      'reference_files': files,
                      'nominal_DA_area_mm2': float(nominal_da.area_m2) * 1e6,
                      'best_method': best.method,
                      'best_DA_area_mm2': float(best_da.area_m2) * 1e6,
                      'results': [{'method': r.method, 'loss': r.loss, 'evaluations': r.evaluations,
                                   'k2_m3': dict(zip(families, strengths(r.theta).tolist()))} for r in results]},
                     indent=2))


if __name__ == '__main__':
    main()
