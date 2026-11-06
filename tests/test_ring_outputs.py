"""Independent linear-map and relativistic references for ring reports."""
import math

import pytest
import torch

from slipstream import (
    ElementSpec, RingMap, plot_dynamic_aperture, plot_ring_layout, plot_ring_twiss,
    ring_lattice_table, ring_optics_table, ring_parameters,
)
from slipstream.analysis.nonlinear import DynamicAperture, linear_optics
from slipstream.constants import C_LIGHT


def ring():
    return RingMap([ElementSpec('D', 'drift', 1.), ElementSpec('QF', 'quad', .2, k1=1.5),
                    ElementSpec('D', 'drift', 1.), ElementSpec('QD', 'quad', .2, k1=-1.35)] * 3,
                   3e9, device='cpu', dtype=torch.float64, n_kicks=4)


def test_periodicity_drift_samples_and_configured_strengths():
    r = ring().with_strengths(k1={'QF': torch.tensor(1.51, dtype=torch.float64)})
    frame = ring_optics_table(r, drift_samples=1)
    assert len(frame) == len(r.specs) + 1 + 6
    for col in ('beta_x_m', 'beta_y_m', 'alpha_x', 'alpha_y', 'D_x_m', 'D_y_m'):
        assert frame[col].iloc[-1] == pytest.approx(frame[col].iloc[0], rel=1e-10, abs=1e-10)
    first, middle = frame.iloc[0], frame.iloc[1]
    for plane in ('x', 'y'):
        beta, alpha = first['beta_' + plane + '_m'], first['alpha_' + plane]
        expected = beta - 2 * .5 * alpha + .5**2 * (1 + alpha**2) / beta
        assert middle['beta_' + plane + '_m'] == pytest.approx(expected, rel=1e-13)
    opt = linear_optics(r)
    assert frame.attrs['Qx'] == pytest.approx(float(opt.Qx), abs=1e-12)
    assert frame.mu_x_rad.iloc[-1] / (2 * math.pi) % 1 == pytest.approx(float(opt.Qx), abs=1e-12)
    lattice = ring_lattice_table(r)
    assert lattice.loc[lattice.element == 'QF', 'k1_m2'].tolist() == [1.51] * 3
    with pytest.raises(ValueError, match='ensemble'):
        ring_lattice_table(r.with_strengths(k1={'QF': torch.ones(2, 1)}))


def test_derived_parameters_against_relativistic_closed_forms():
    r = ring()
    frame = ring_optics_table(r, drift_samples=0)
    params = ring_parameters(r, frame, include_chromaticity=False).set_index('parameter')
    gamma = r.energy_eV / r.mc2_eV
    beta = math.sqrt(1 - 1 / gamma**2)
    assert params.loc['revolution_frequency', 'value'] == pytest.approx(beta * C_LIGHT / 7.2, rel=1e-14)
    assert params.loc['magnetic_rigidity', 'value'] == pytest.approx(r.energy_eV * beta / C_LIGHT, rel=1e-14)
    assert params.loc['slip_factor', 'value'] == pytest.approx(-gamma**-2, abs=1e-16)
    assert params.loc['momentum_compaction', 'value'] == pytest.approx(0., abs=1e-16)
    doubled = ring_parameters(r, frame, charge_e=-2, include_chromaticity=False).set_index('parameter')
    assert doubled.loc['magnetic_rigidity', 'value'] == pytest.approx(params.loc['magnetic_rigidity', 'value'] / 2)
    with pytest.raises(ValueError, match='charge'):
        ring_parameters(r, frame, charge_e=0)


def test_superperiod_parameters_match_whole_ring():
    whole = ring_parameters(ring()).set_index('parameter').value
    cell = RingMap(ring().specs[:4], 3e9, device='cpu', dtype=torch.float64, n_kicks=4)
    scaled = ring_parameters(cell, periods=3).set_index('parameter').value
    assert scaled['periods'] == 3 and whole['periods'] == 1
    for name in ('circumference', 'revolution_frequency', 'Qx_fractional', 'Qy_fractional', 'Qx_accumulated',
                 'Qy_accumulated', 'chromaticity_x', 'chromaticity_y', 'slip_factor', 'element_count'):
        assert scaled[name] == pytest.approx(whole[name], rel=1e-9, abs=1e-12), name
    with pytest.raises(ValueError, match='periods'):
        ring_parameters(cell, periods=0)


def test_local_coupling_is_rejected_even_if_one_turn_coupling_cancels():
    specs = [ElementSpec('SK1', 'multipole', 0., integrated_multipoles=((1, .1),), tilt=math.pi/4),
             ElementSpec('SK2', 'multipole', 0., integrated_multipoles=((1, -.1),), tilt=math.pi/4)] + ring().specs
    r = RingMap(specs, 3e9, device='cpu')
    linear_optics(r)  # Adjacent opposite skew kicks cancel in the one-turn map.
    with pytest.raises(ValueError, match='locally coupled'):
        ring_optics_table(r)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA hardware required')
def test_gpu_strength_override_reports_use_cpu_reference():
    specs = ring().specs
    gpu = RingMap(specs, 3e9, device='cuda', dtype=torch.float64, n_kicks=4)
    gpu = gpu.with_strengths(k1={'QF': torch.tensor(1.51, device='cuda', dtype=torch.float64)})
    ref = gpu.float64_reference()
    assert next(p.k1 for s, p in zip(ref.specs, ref.params) if s.name == 'QF').device.type == 'cpu'
    actual = ring_optics_table(gpu, drift_samples=0)
    expected = ring_optics_table(ring().with_strengths(k1={'QF': torch.tensor(1.51, dtype=torch.float64)}), drift_samples=0)
    torch.testing.assert_close(torch.tensor(actual.beta_x_m.to_numpy()), torch.tensor(expected.beta_x_m.to_numpy()))


def test_optional_plots_save_and_label_units(tmp_path):
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r = ring()
    lattice = ring_lattice_table(r)
    optics = ring_optics_table(r, drift_samples=1)
    theta = torch.linspace(0, math.pi, 5)
    radius = torch.ones(5) * .002
    da = DynamicAperture(theta, radius, radius * theta.cos(), radius * theta.sin(), torch.tensor(0.))
    figures = [plot_ring_twiss(optics, lattice), plot_ring_layout(lattice), plot_dynamic_aperture(da)]
    assert figures[0].axes[0].get_ylabel() == 'Beta [m]'
    assert figures[2].axes[0].get_xlabel() == 'Initial x [mm]'
    for i, fig in enumerate(figures):
        path = tmp_path / f'plot{i}.png'
        fig.savefig(path)
        assert path.stat().st_size > 1000
        plt.close(fig)
