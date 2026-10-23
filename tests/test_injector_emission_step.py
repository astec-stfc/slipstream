"""The emission-phase step and the ASTRA smoothing import of ``Injector``."""
import pytest
import torch

from slipstream.tracking.native.injector import Injector, SpaceChargeSettings

WINDOW_S = 4e-12


def cathode_beam(n=64):
    """A cold disc emitted over ``WINDOW_S`` -- the smallest beam with an emission window."""
    a = torch.arange(n, dtype=torch.float64)
    zero = torch.zeros(n, dtype=torch.float64)
    return dict(x=(a - n / 2) * 1e-6, y=zero.clone(), z=zero.clone(), ux=zero.clone(), uy=zero.clone(),
                uz=torch.full((n,), 1e6, dtype=torch.float64), q=torch.full((n,), -1e-15, dtype=torch.float64),
                emission_time_s=a / (n - 1) * WINDOW_S)


def machine(monkeypatch, **space_charge):
    monkeypatch.setattr(Injector, "initial_beam", lambda self, n, device, dtype, **k: cathode_beam())
    monkeypatch.setattr(Injector, "_reference_particle", lambda self: (0.0, 0.0, 0.0))
    return Injector("", 1e-3, absolute_phases=True, dt_s=1e-12,
                    space_charge=SpaceChargeSettings(enabled=False, **space_charge))


def steps(inj):
    return inj.track(device="cpu", dtype=torch.float64, use_compile=False, verbose=False, record_every=0)["dt_steps_s"]


def test_emission_window_is_resolved_then_a_fixed_step_run_returns_to_dt_s(monkeypatch):
    dt = steps(machine(monkeypatch, dt_adaptive=False, emission_steps=100))
    assert dt[0] == pytest.approx(WINDOW_S / 100, rel=1e-12)
    assert dt[-1] == 1e-12
    # the fine step lasts exactly until the emission is over (t = 2 x the last emission time), then never returns
    fine = int((dt < 1e-12).sum())
    assert (dt[:fine] < 1e-12).all() and (dt[fine:] == 1e-12).all()
    assert fine * WINDOW_S / 100 == pytest.approx(2 * WINDOW_S, rel=0.02)


def test_adaptive_default_without_space_charge_also_returns_to_dt_s(monkeypatch):
    # the adaptive rule lives in the space-charge solve: with it off nothing else would end the fine emission step
    dt = steps(machine(monkeypatch))
    assert dt[0] < 1e-12 and dt[-1] == 1e-12


def test_emission_steps_zero_keeps_dt_s_throughout(monkeypatch):
    dt = steps(machine(monkeypatch, dt_adaptive=False, emission_steps=0))
    assert (dt == 1e-12).all()


def test_a_step_already_finer_than_the_window_is_kept(monkeypatch):
    inj = machine(monkeypatch, dt_adaptive=False, emission_steps=100)
    inj.dt_s = 1e-14
    assert (steps(inj) == 1e-14).all()


def test_in_flight_beam_has_no_emission_window(monkeypatch):
    beam = cathode_beam()
    beam["emission_time_s"] = torch.zeros_like(beam["emission_time_s"])
    monkeypatch.setattr(Injector, "initial_beam", lambda self, n, device, dtype, **k: beam)
    monkeypatch.setattr(Injector, "_reference_particle", lambda self: (0.0, 0.0, 0.0))
    inj = Injector("", 1e-3, absolute_phases=True, dt_s=1e-12,
                   space_charge=SpaceChargeSettings(enabled=False, dt_adaptive=False))
    assert (steps(inj) == 1e-12).all()


def deck(tmp_path, lspch3d):
    path = tmp_path / "injector.in"
    path.write_text(f"&NEWRUN\nDistribution = 'beam.astra'\n/\n&OUTPUT\nzstop = 1.0\n/\n"
                    f"&CHARGE\nLSPCH = T\nLSPCH3D = {'T' if lspch3d else 'F'}\nsmooth_x = 2, smooth_y = 3, smooth_z = 4\n/\n")
    return path


@pytest.mark.parametrize("lspch3d, expected", [(False, (0, 0)), (True, (3, 4))])
def test_from_astra_imports_smoothing_only_for_the_algorithm_that_applies_it(tmp_path, lspch3d, expected):
    sc = Injector.from_astra(deck(tmp_path, lspch3d)).space_charge
    assert (sc.smooth_r, sc.smooth_z) == expected


def test_new_field_survives_the_machine_file_round_trip(tmp_path):
    inj = Injector("", 1.0, space_charge=SpaceChargeSettings(emission_steps=37))
    inj.save(tmp_path / "m.yaml")
    assert Injector.load(tmp_path / "m.yaml").space_charge.emission_steps == 37
