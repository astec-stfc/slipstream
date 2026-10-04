"""Quantitative tests for portable beamline run configuration and source handoffs."""
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

from slipstream import BeamlineResult, BeamlineStudy
from slipstream.constants import C_LIGHT, ELECTRON_MASS_EV
from slipstream.data.particles import save_particles_h5
from slipstream.tracking.distributions import gaussian_matched_6d
from slipstream.tracking.native.injector import Injector, injector_exit_to_canonical6
from slipstream.tracking.native.lattice_import import ElementSpec, parse_elegant_lattice
from slipstream.tracking.native.linac import LinacTracker


def config(tmp_path, **changes):
    machine = tmp_path / "machine"
    machine.mkdir(exist_ok=True)
    (machine / "line.lte").write_text(
        "D: DRIFT, L=0.2\nQ: KQUAD, L=0.1, K1=0.8\n"
        "R: RFCA, L=0.1, VOLT=2000000, PHASE=90, FREQ=3000000000\n"
        "LINE: LINE=(D,Q,R,D)\n", encoding="utf-8",
    )
    cfg = dict(lattice={"path": "machine/line.lte", "root": "LINE"}, energy_eV=30e6,
               bunch=dict(source="generated", n=64, beta_x0=2.0, alpha_x0=0.3, epsx_m=1e-8,
                          beta_y0=1.5, alpha_y0=-0.1, epsy_m=2e-8, sigma_tau_m=1e-4,
                          sigma_delta=2e-3, seed=123),
               tracker={"device": "cpu", "dtype": "float64", "bunch_charge_C": 2e-10},
               run={"return_trajectory": True})
    cfg.update(changes)
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path, cfg


def test_generated_direct_identity_repeatability_and_conversion(tmp_path, monkeypatch):
    path, cfg = config(tmp_path)
    monkeypatch.chdir(tmp_path.parent)
    study = BeamlineStudy.from_yaml(path)
    tracker_id = id(study._tracker)
    result = study.run()
    second = study.run()
    assert id(study._tracker) == tracker_id
    np.testing.assert_array_equal(result.particles_out, second.particles_out)
    args = dict(cfg["bunch"])
    args.pop("source")
    particles = study.generate_bunch(**args)
    direct = LinacTracker(parse_elegant_lattice(tmp_path / "machine/line.lte", "LINE"),
                          cfg["energy_eV"], device="cpu", dtype=torch.float64, bunch_charge_C=2e-10)
    out, energy, trajectory, energies = direct.track(particles, return_trajectory=True)
    np.testing.assert_allclose(result.particles_out, out.numpy(), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(result.trajectory, trajectory.numpy(), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(result.energy_trajectory, energies.numpy(), rtol=1e-12)
    assert result.energy_eV_out == float(energy)
    slopes = gaussian_matched_6d(64, 2.0, 0.3, 1e-8, 1.5, -0.1, 2e-8, 1e-4, 2e-3, seed=123)
    norm = torch.sqrt(1 + slopes[:, 1] ** 2 + slopes[:, 3] ** 2)
    expected_px = (1 + slopes[:, 5]) * slopes[:, 1] / norm
    torch.testing.assert_close(particles[:, 1], expected_px, rtol=1e-14, atol=1e-16)
    assert not torch.equal(particles[:, 1], slopes[:, 1])


def test_default_seed_and_reference_roundtrip(tmp_path):
    path, cfg = config(tmp_path)
    cfg["bunch"].pop("seed")
    cfg["reference"] = "reference.yaml"
    cfg["run"]["output"] = "result.h5"
    (tmp_path / "reference.yaml").write_text("energy_eV: 32000000\nsigma_delta: 0.0\n", encoding="utf-8")
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    study = BeamlineStudy.from_yaml(path)
    result = study.run()
    again = study.run()
    np.testing.assert_array_equal(result.particles_out, again.particles_out)
    loaded = BeamlineResult.load(tmp_path / "result.h5")
    assert loaded.metadata == result.metadata
    assert loaded.metadata["reference_diff"]["energy_eV"] == pytest.approx((result.energy_eV_out - 32e6) / 32e6)
    assert loaded.metadata["reference_diff"]["sigma_delta"] == result.scalars()["sigma_delta"]
    np.testing.assert_array_equal(loaded.trajectory, result.trajectory)


@pytest.mark.parametrize("change,match", [
    ({"unknown": 1}, "Unknown study"),
    ({"tracker": {"errors": []}}, "Unknown tracker"),
    ({"tracker": {"compile": "false"}}, "boolean"),
    ({"tracker": {"dtype": "float16"}}, "float32 or float64"),
    ({"tracker": {"device": "mps"}}, "device"),
    ({"tracker": {"integration_order": 2}}, "4, 6 or 8"),
    ({"reference": {"made_up": 2}}, "Unknown reference"),
    ({"bunch": {"source": "astra", "path": "deck.in", "injector": {"mesh": "beam"}}}, "adaptive or fixed"),
    ({"bunch": {"source": "astra", "path": "deck.in", "injector": {"mystery": 1}}}, "Unknown bunch.injector"),
])
def test_invalid_configuration_is_clear(tmp_path, change, match):
    path, _ = config(tmp_path, **change)
    with pytest.raises(ValueError, match=match):
        BeamlineStudy.from_yaml(path)


def test_yaml_uses_safe_load(tmp_path):
    path = tmp_path / "unsafe.yaml"
    path.write_text("!!python/object/apply:os.system ['echo should-not-run']", encoding="utf-8")
    with pytest.raises(yaml.constructor.ConstructorError):
        BeamlineStudy.from_yaml(path)


def test_hdf5_slope_source_energy_charge_and_relative_path(tmp_path):
    path, _ = config(tmp_path, bunch={"source": "hdf5", "path": "input.h5"})
    slopes = torch.tensor([[1e-3, 0.02, 2e-3, -0.01, 1e-4, 0.02],
                           [-1e-3, -0.01, -2e-3, 0.02, -1e-4, -0.02]], dtype=torch.float64)
    save_particles_h5(tmp_path / "input.h5", slopes, 3e-10, 40e6, ELECTRON_MASS_EV, canonical=False)
    study = BeamlineStudy.from_yaml(path)
    result = study.run()
    expected = slopes.clone()
    norm = (1 + slopes[:, 1] ** 2 + slopes[:, 3] ** 2).sqrt()
    expected[:, 1] = slopes[:, 1] * (1 + slopes[:, 5]) / norm
    expected[:, 3] = slopes[:, 3] * (1 + slopes[:, 5]) / norm
    np.testing.assert_allclose(result.trajectory[0], expected.numpy(), rtol=1e-14, atol=1e-16)
    assert result.energy_trajectory[0] == 40e6
    assert result.charge_C == 3e-10
    assert study._tracker.bunch_charge_C == 3e-10


@pytest.mark.parametrize("energy,charge,match", [(float("inf"), 0., "energy_eV"),
                                               (30e6, float("nan"), "charge_C")])
def test_hdf5_nonfinite_source_metadata_rejected(tmp_path, energy, charge, match):
    path, _ = config(tmp_path, bunch={"source": "hdf5", "path": "input.h5"})
    save_particles_h5(tmp_path / "input.h5", torch.zeros(2, 6), charge, energy, ELECTRON_MASS_EV)
    with pytest.raises(ValueError, match=match):
        BeamlineStudy.from_yaml(path).run()


def test_sdds_exact_relativistic_coordinates_without_optional_dependency(tmp_path, monkeypatch):
    from slipstream import sdds_io

    path, _ = config(tmp_path, bunch={"source": "sdds", "path": "input.sdds", "time_origin_s": 1e-9})
    columns = [[1e-3, -2e-3], [0.1, -0.2], [2e-3, 3e-3], [-0.2, 0.05],
               [1.1e-9, 0.8e-9], [58.0, 60.0]]

    class Data:
        columnName = ["x", "xp", "y", "yp", "t", "p"]
        columnData = [[c] for c in columns]

        def load(self, filename):
            assert Path(filename) == tmp_path / "input.sdds"

    @contextmanager
    def fake_sdds():
        yield None, Data()

    monkeypatch.setattr(sdds_io, "sdds_object", fake_sdds)
    study = BeamlineStudy.from_yaml(path)
    got = study.run().trajectory[0]
    p0 = math.sqrt((30e6 / ELECTRON_MASS_EV) ** 2 - 1)
    momentum = np.array(columns[5])
    xp, yp = np.array(columns[1]), np.array(columns[3])
    pz_over_p0 = momentum / p0 / np.sqrt(1 + xp ** 2 + yp ** 2)
    np.testing.assert_allclose(got[:, 1], xp * pz_over_p0, rtol=1e-14)
    np.testing.assert_allclose(got[:, 3], yp * pz_over_p0, rtol=1e-14)
    np.testing.assert_allclose(got[:, 4], C_LIGHT * (np.array(columns[4]) - 1e-9), rtol=1e-14)
    np.testing.assert_allclose(got[:, 5], momentum / p0 - 1, rtol=1e-14)


@pytest.mark.parametrize("mesh", ["adaptive", "fixed"])
def test_astra_composition_reference_charge_mass_and_options(tmp_path, monkeypatch, mesh):
    path, _ = config(tmp_path, bunch={"source": "astra", "path": "deck.in", "injector": {"mesh": mesh}})
    out = dict(x=torch.tensor([1e-3, -1e-3], dtype=torch.float64),
               y=torch.tensor([2e-3, -2e-3], dtype=torch.float64),
               z=torch.tensor([1.0, 1.001], dtype=torch.float64),
               ux=torch.tensor([2e6, -3e6], dtype=torch.float64),
               uy=torch.tensor([-1e6, 4e6], dtype=torch.float64),
               uz=torch.tensor([20 * C_LIGHT, 21 * C_LIGHT], dtype=torch.float64),
               q=torch.tensor([-1e-10, -2e-10], dtype=torch.float64), t_s=4e-9)
    injector = object.__new__(Injector)
    injector.mc2_eV = ELECTRON_MASS_EV

    def from_astra(cls, filename, **kwargs):
        assert filename == tmp_path / "deck.in"
        assert kwargs == {"mc2_eV": ELECTRON_MASS_EV}
        return injector

    def track(self, **kwargs):
        assert kwargs["mesh"] == mesh
        assert kwargs["device"] == "cpu"
        assert kwargs["dtype"] == torch.float64
        assert kwargs["use_compile"] is False
        assert kwargs["return_particles"] is True
        return out

    monkeypatch.setattr(Injector, "from_astra", classmethod(from_astra))
    monkeypatch.setattr(Injector, "track", track)
    study = BeamlineStudy.from_yaml(path)
    result = study.run()
    expected, energy, charge = injector_exit_to_canonical6(out)
    np.testing.assert_allclose(result.trajectory[0], expected.numpy(), rtol=1e-14)
    assert result.energy_trajectory[0] == energy
    assert result.charge_C == charge
    direct = LinacTracker(study.specs, energy, device="cpu", bunch_charge_C=charge)
    particles, direct_energy = direct.track(expected)
    np.testing.assert_allclose(result.particles_out, particles.numpy(), rtol=1e-12, atol=1e-15)
    assert result.energy_eV_out == float(direct_energy)


def test_wake_paths_default_to_lattice_directory_and_can_be_disabled(tmp_path, monkeypatch):
    import slipstream.beamline as beamline

    path, cfg = config(tmp_path)
    calls = []

    def resolve(specs, base_dir):
        calls.append(base_dir)
        return specs

    monkeypatch.setattr(beamline, "resolve_wakes", resolve)
    BeamlineStudy.from_yaml(path)
    assert calls == [tmp_path / "machine"]
    cfg["lattice"]["wake_base_dir"] = "wake_tables"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    BeamlineStudy.from_yaml(path)
    assert calls[-1] == tmp_path / "wake_tables"
    cfg["lattice"]["wakes"] = False
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    BeamlineStudy.from_yaml(path)
    assert len(calls) == 2


@pytest.mark.parametrize("particles", [torch.zeros(0, 6), torch.zeros(2, 5),
                                       torch.full((2, 6), float("nan"))])
def test_invalid_bunch_rejected_before_tracking(particles):
    study = BeamlineStudy([], 30e6, device="cpu")
    with pytest.raises(ValueError, match="Bunch"):
        study.track(particles)


def test_direct_facade_keeps_one_particle_support():
    study = BeamlineStudy([ElementSpec("D", "drift", 0.1)], 30e6, device="cpu")
    result = study.track(torch.zeros(1, 6, dtype=torch.float64))
    assert result.particles_out.shape == (1, 6)


def test_cli_hdf5_roundtrip_from_other_directory(tmp_path):
    path, _ = config(tmp_path, reference={"energy_eV": 32e6})
    output = tmp_path / "cli.h5"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    proc = subprocess.run([sys.executable, "-m", "slipstream.beamline_cli", str(path), "-o", str(output)],
                          cwd=tmp_path.parent, env=env, capture_output=True, text=True, check=True)
    summary = json.loads(proc.stdout)
    loaded = BeamlineResult.load(output)
    assert summary["energy_eV"] == loaded.energy_eV_out
    assert summary["reference_diff"] == loaded.metadata["reference_diff"]
    assert loaded.particles_out.shape == (64, 6)


def test_fast_csr_late_charge_matches_direct_and_clears_zero_charge():
    specs = [ElementSpec("B", "bend", length=0.1, angle=0.05, csr=True, csr_n_slices=3,
                         csr_bins=32, csr_sg_halfwidth=1)]
    study = BeamlineStudy(specs, 30e6, device="cpu", csr=True, fast_collective_effects=True,
                          n_kicks_bend=6, always_change_p0=False)
    particles = study.generate_bunch(64, 2., 0., 1e-8, 2., 0., 1e-8, 1e-4, 1e-3, seed=21)
    study._tracker.bunch_charge_C = 1e-10
    late = study.track(particles, return_trajectory=True)
    direct = LinacTracker(specs, 30e6, device="cpu", csr=True, fast_collective_effects=True,
                          n_kicks_bend=6, always_change_p0=False, bunch_charge_C=1e-10)
    out, energy, trajectory, energies = direct.track(particles, return_trajectory=True)
    np.testing.assert_allclose(late.particles_out, out.numpy(), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(late.trajectory, trajectory.numpy(), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(late.energy_trajectory, energies.numpy(), rtol=1e-12)
    assert late.energy_eV_out == float(energy)
    study._tracker.bunch_charge_C = 0.0
    zero = study.track(particles, return_trajectory=True)
    fresh = LinacTracker(specs, 30e6, device="cpu", csr=True, fast_collective_effects=True,
                         n_kicks_bend=6, always_change_p0=False)
    expected, _, zero_traj, zero_energy = fresh.track(particles, return_trajectory=True)
    np.testing.assert_allclose(zero.particles_out, expected.numpy(), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(zero.trajectory, zero_traj.numpy(), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(zero.energy_trajectory, zero_energy.numpy(), rtol=1e-12)
    csr_off = LinacTracker(specs, 30e6, device="cpu", csr=False, fast_collective_effects=True,
                           n_kicks_bend=6, always_change_p0=False)
    off_out, _ = csr_off.track(particles)
    np.testing.assert_allclose(zero.particles_out, off_out.numpy(), rtol=1e-12, atol=1e-15)
    assert np.max(np.abs(late.particles_out[:, 5] - zero.particles_out[:, 5])) > 1e-8


@pytest.mark.parametrize("source", ["hdf5", "astra"])
def test_source_energy_refreshes_bend_offsets_without_corrupting_cached_trackers(tmp_path, monkeypatch, source):
    path, cfg = config(tmp_path, energy_eV=100e6,
                       bunch={"source": source, "path": "input.h5" if source == "hdf5" else "deck.in"},
                       tracker={"device": "cpu", "n_kicks_bend": 1, "always_change_p0": False})
    (tmp_path / "machine/line.lte").write_text("B: CSBEND, L=1.0, ANGLE=0.5\nLINE: LINE=(B)\n", encoding="utf-8")
    particles = torch.tensor([[1e-3, 2e-3, -1e-3, -1e-3, 1e-4, 1e-3],
                              [-1e-3, -2e-3, 1e-3, 2e-3, -1e-4, -1e-3]], dtype=torch.float64)
    if source == "hdf5":
        energy, charge = 2e6, 3e-10
        save_particles_h5(tmp_path / "input.h5", particles, charge, energy, ELECTRON_MASS_EV)
    else:
        injector = object.__new__(Injector)
        injector.mc2_eV = ELECTRON_MASS_EV
        raw = dict(x=particles[:, 0], y=particles[:, 2], z=torch.tensor([1., 1.001], dtype=torch.float64),
                   ux=torch.tensor([1e6, -1e6], dtype=torch.float64),
                   uy=torch.tensor([-1e6, 1e6], dtype=torch.float64),
                   uz=torch.tensor([3 * C_LIGHT, 4 * C_LIGHT], dtype=torch.float64),
                   q=torch.tensor([-1e-10, -2e-10], dtype=torch.float64), t_s=4e-9)
        monkeypatch.setattr(Injector, "from_astra", classmethod(lambda cls, *args, **kwargs: injector))
        monkeypatch.setattr(Injector, "track", lambda *args, **kwargs: raw)
        particles, energy, charge = injector_exit_to_canonical6(raw)
    specs = parse_elegant_lattice(tmp_path / "machine/line.lte", "LINE")
    high = LinacTracker(specs, cfg["energy_eV"], device="cpu", n_kicks_bend=1, always_change_p0=False)
    initial_offset = high._element_params[0].offset.clone()
    study = BeamlineStudy.from_yaml(path)
    tracker_id = id(study._tracker)
    result = study.run()
    again = study.run()
    assert id(study._tracker) == tracker_id
    direct = LinacTracker(specs, energy, device="cpu", n_kicks_bend=1,
                          always_change_p0=False, bunch_charge_C=charge)
    expected, exit_energy, trajectory, energies = direct.track(particles, return_trajectory=True)
    np.testing.assert_allclose(result.particles_out, expected.numpy(), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(result.trajectory, trajectory.numpy(), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(result.energy_trajectory, energies.numpy(), rtol=1e-12)
    np.testing.assert_array_equal(result.particles_out, again.particles_out)
    assert result.energy_eV_out == float(exit_energy)
    torch.testing.assert_close(high._element_params[0].offset, initial_offset, rtol=0, atol=0)
