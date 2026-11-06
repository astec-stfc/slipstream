"""Keep the public first-run examples runnable and model selection faithful."""
from pathlib import Path
import runpy

import numpy as np
import pytest
import torch

from slipstream import BeamlineStudy, LinacTracker
from slipstream.tracking.native import collective

ROOT = Path(__file__).resolve().parents[1]


def test_readme_quickstart_needs_no_input_files(tmp_path, monkeypatch):
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    code = readme.split("## Quick start", 1)[1].split("```python\n", 1)[1].split("```", 1)[0]
    monkeypatch.chdir(tmp_path)
    namespace = {}
    exec(compile(code, "README.md", "exec"), namespace)
    result = namespace["result"]
    assert result.particles_out.shape == (1024, 6)
    assert result.energy_eV_out == pytest.approx(100e6)
    assert np.isfinite(result.particles_out).all()
    assert not list(tmp_path.iterdir())


def test_python_quickstart_reports_a_complete_drift(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    example = runpy.run_path(str(ROOT / "docs/examples/quickstart.py"))
    result = example["main"]()
    assert result.trajectory.shape == (2, 1024, 6)
    assert result.energy_eV_out == pytest.approx(100e6)
    # The canonical transverse momenta are unchanged in a field-free drift.
    np.testing.assert_array_equal(result.trajectory[0][:, [1, 3, 5]], result.particles_out[:, [1, 3, 5]])
    assert "Exit total energy: 100.000 MeV" in capsys.readouterr().out
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("fast,transient,drift", [
    (False, False, "overtaking"),
    (True, False, "overtaking"),
    (True, True, "overtaking"),
    (True, True, "stupakov"),
    (True, True, "retarded"),
    (True, False, "retarded"),
])
def test_csr_yaml_choices_match_existing_tracker(fast, transient, drift, tmp_path, monkeypatch):
    path = ROOT / "docs/examples/csr.yaml"
    monkeypatch.chdir(tmp_path)
    # Exercise the default PyTorch installation even where optional Warp is installed.
    monkeypatch.setattr(collective, "_warp_f64_wake_ready", lambda tensor: False)
    options = dict(csr=True, fast_collective_effects=fast, csr_transient=transient, csr_drift_model=drift)
    study = BeamlineStudy.from_yaml(path, overrides={
        **{f"tracker.{key}": value for key, value in options.items()}, "bunch.n": 64,
    })
    particles = study.generate_bunch(64, 2., 0., 1e-8, 2., 0., 1e-8, 2e-5, 1e-4, seed=42)
    direct = LinacTracker(study.specs, 190e6, bunch_charge_C=1e-10, device="cpu", **options)
    expected, energy = direct.track(particles)
    result = study.run()
    np.testing.assert_allclose(result.particles_out, expected.numpy(), rtol=1e-12, atol=1e-15)
    assert result.energy_eV_out == float(energy)
    assert np.isfinite(result.particles_out).all()
    without_csr = LinacTracker(study.specs, 190e6, device="cpu")
    off, off_energy = without_csr.track(particles)
    assert not torch.equal(expected, off)
    assert float(energy) != float(off_energy)
