"""Smoke run of the Diamond superperiod DA example on a tiny grid."""
import json
import math
from pathlib import Path
import runpy
import sys

import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples/diamond/nonlinear_optimisation.py'


@pytest.mark.skipif(not EXAMPLE.with_name('diamond.lte').is_file(), reason='Diamond lattice file not present')
def test_diamond_example_superperiod_workflow(monkeypatch, tmp_path, capsys):
    example = runpy.run_path(str(EXAMPLE))
    monkeypatch.setattr(sys, 'argv', ['nonlinear_optimisation.py', '--budget', '1', '--turns', '1', '--angles', '3',
                                      '--radii', '1', '--no-plots', '--output-dir', str(tmp_path)])
    example['main']()
    report = json.loads(capsys.readouterr().out)
    assert report['periods'] == 6
    assert report['circumference_m'] == pytest.approx(561.6, rel=1e-12)
    assert report['tunes'][0] == pytest.approx(27.2, abs=0.1) and report['tunes'][1] == pytest.approx(12.4, abs=0.1)
    # Lattice-file sextupoles leave the ring close to, but not at, the 0/0 target the correctors restore.
    assert report['chromaticity_lattice_file'] == pytest.approx([0.07, 0.85], abs=0.02)
    assert [r['method'] for r in report['results']] == ['Powell', 'Nelder-Mead', 'differential_evolution']
    assert all(r['evaluations'] == 1 and math.isfinite(r['loss']) for r in report['results'])
    k2 = report['results'][0]['k2_m3']
    assert k2['S1B'] == pytest.approx(20.0434, rel=1e-6) and k2['HALFS1A'] != pytest.approx(35.7507, abs=1e-3)
    for name in ('optics', 'lattice', 'machine_parameters', 'chromaticity_response',
                 'dynamic_aperture', 'optimised_dynamic_aperture'):
        assert Path(report['reference_files'][name]).is_file()
