"""Geometry and optics checks for the illustrative MLS optimisation ring."""
from collections import Counter
import json
import math
from pathlib import Path
import runpy
import sys

import pytest
import torch

from slipstream import RingMap, ring_optics_table, ring_parameters
from slipstream.analysis import nonlinear as nl
from slipstream.tracking.native import ring_warp


def test_mls_example_geometry_and_matched_optics():
    example = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'examples/nonlinear_optimisation.py'))
    specs = example['mls_specs']()
    counts = Counter(s.kind for s in specs)
    assert counts['bend'] == 8 and counts['quad'] == 24
    assert counts['sext'] == 24 and counts['oct'] == 4
    assert math.isclose(sum(s.length for s in specs), 48., abs_tol=1e-12)
    assert math.isclose(sum(s.angle for s in specs), 2 * math.pi, abs_tol=1e-12)
    assert [s.length for s in specs if s.name == 'STRAIGHT'] == [6., 2.5, 6., 2.5]
    ring = RingMap(specs, 629e6, device='cpu', dtype=torch.float64, n_kicks=4)
    opt = nl.linear_optics(ring)
    torch.testing.assert_close(torch.stack((opt.Qx, opt.Qy)),
                               torch.tensor([.18, .23], dtype=torch.float64), rtol=0, atol=2e-5)
    profile = ring_optics_table(ring, drift_samples=8)
    # Approximate scales read from examples/MLS/Twiss.png, not a precise
    # reproduction of its curve: beta_x ~15 m, beta_y ~9 m, D_x ~1.5 m.
    assert abs(profile.beta_x_m.max() - 15.) < 1.
    assert abs(profile.beta_y_m.max() - 9.) < 1.
    assert abs(profile.D_x_m.max() - 1.5) < .3
    assert profile.beta_x_m.min() < 3. and profile.beta_y_m.min() < 3.
    # The reference has dispersive straights. Independently check the
    # reported entrance dispersion against a central closed-orbit difference.
    h = 1e-6
    dispersion = (nl.closed_orbit(ring, delta=h)[0] - nl.closed_orbit(ring, delta=-h)[0]) / (2 * h)
    assert math.isclose(float(dispersion), profile.iloc[0].D_x_m, abs_tol=1e-7)
    launches = nl.launch(opt, torch.tensor([.00005, .0001], dtype=torch.float64),
                         torch.tensor([.00003, .00005], dtype=torch.float64))
    result = ring.track(launches, 32, aperture=(.02, .02))
    assert (result.loss_turn < 0).all()
    parameters = ring_parameters(ring, profile, include_chromaticity=False).set_index('parameter')
    # Closed-orbit time-of-flight finite difference includes the dispersion
    # contribution, giving an independent check on the reported compaction.
    h = 1e-5
    tau = [float(ring.turn(nl.closed_orbit(ring, delta=d))[4]) for d in (-h, h)]
    gamma = ring.energy_eV / ring.mc2_eV
    beta = math.sqrt(1 - gamma**-2)
    expected = beta * (tau[1] - tau[0]) / (2 * h * ring.circumference_m) + gamma**-2
    assert math.isclose(parameters.loc['momentum_compaction', 'value'], expected, abs_tol=1e-8)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_mls_example_float32_optimisation(monkeypatch, tmp_path, capsys, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    monkeypatch.setattr(ring_warp, 'available', lambda *args: False)
    example = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'examples/nonlinear_optimisation.py'))
    spec = example['ElementSpec']
    small_ring = [spec('D', 'drift', .4), spec('QF', 'quad', .2, k1=3.),
                  spec('D', 'drift', .4), spec('QD', 'quad', .2, k1=-3.),
                  spec('Q1', 'quad', .01, k1=example['MLS_K1'][0]),
                  spec('S1', 'sext', .02, k2=20.), spec('S2', 'sext', .02, k2=-20.),
                  spec('S3', 'sext', .02)]

    def eager_ring(*args, **kwargs):
        # Run the actual workflow on a small stable ring: this tests the
        # float32 knob/device boundary without expensive MLS Jacobians.
        kwargs.update(device=device, dtype=torch.float32, compile=False)
        return RingMap(small_ring, *args[1:], **kwargs)

    example['main'].__globals__['RingMap'] = eager_ring
    monkeypatch.setattr(sys, 'argv', ['nonlinear_optimisation.py', '--budget', '1', '--turns', '1',
                                    '--angles', '3', '--radii', '1', '--refine', '0', '--no-plots',
                                    '--output-dir', str(tmp_path)])
    example['main']()
    assert (tmp_path / 'nominal_dynamic_aperture.csv').is_file()
    results = json.loads(capsys.readouterr().out)['results']
    assert len(results) == 4
    assert all(math.isfinite(result['loss']) and result['evaluations'] == 1 for result in results)
