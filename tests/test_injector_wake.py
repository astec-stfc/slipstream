"""Localized wake kicks of ``Injector`` (ASTRA ``&WAKE``, monopole): table parsing, the wake potential against closed
forms for a Gaussian bunch, the energy balance through ``Injector.track``, and the deck/YAML round trips."""
import math

import numpy as np
import pytest
import torch

from slipstream.constants import C_LIGHT, ELECTRON_MASS_EV
from slipstream.tracking.native.injector import Injector, SpaceChargeSettings
from slipstream.tracking.native.injector_wake import InjectorWake, read_wake_table, wake_potential

Q_C, SIGMA_M = 100e-12, 1e-3


def gaussian_bunch(n=400_000, seed=3):
    g = torch.Generator().manual_seed(seed)
    z = SIGMA_M * torch.randn(n, generator=g, dtype=torch.float64)
    return z, torch.full((n,), Q_C / n, dtype=torch.float64)


def erfc(x):
    return torch.special.erfc(torch.as_tensor(x, dtype=torch.float64))


def binned_mean(z, v, centres, half_width):
    return [float(v[(z - c).abs() < half_width].mean()) for c in centres]


def test_taylor_table_picks_the_monopole_subtable(tmp_path):
    lines = ["2 0",
             "3 0", "0 0", "0 13", "0 0", "1e-3 5", "2e-3 7",
             "3 0", "0 0", "0 0", "0 0", "1e-3 5", "2e-3 4"]
    f = tmp_path / "w.dat"
    f.write_text("\n".join(lines))
    s, w = read_wake_table(f, "taylor_f")
    np.testing.assert_allclose(s, [0.0, 1e-3, 2e-3])
    np.testing.assert_allclose(w, [0.0, 5.0, 4.0])


def test_monopole_table_format(tmp_path):
    f = tmp_path / "w.dat"
    f.write_text("3 0\n0 9\n1e-3 6\n2e-3 3\n")
    s, w = read_wake_table(f, "monopole_f")
    np.testing.assert_allclose(w, [9.0, 6.0, 3.0])


def test_step_wake_of_a_gaussian_bunch_matches_the_closed_form():
    # w(s) = w0 for s > 0: V(zeta) = w0 Q/2 erfc(zeta/(sqrt2 sigma)); the bunch loses Q^2 w0/2 in total
    w0 = 1e12
    z, q = gaussian_bunch()
    v = wake_potential(z, q, np.array([0.0, 1.0]), np.array([w0, w0]), n_bin=256)
    centres = np.array([-2.0, -1.0, 0.0, 1.0, 2.0]) * SIGMA_M
    got = np.array(binned_mean(z, v, centres, 0.1 * SIGMA_M))
    expect = w0 * Q_C / 2 * erfc(centres / (math.sqrt(2) * SIGMA_M)).numpy()
    np.testing.assert_allclose(got, expect, atol=0.01 * w0 * Q_C)
    assert float((q * v).sum()) == pytest.approx(0.5 * Q_C**2 * w0, rel=2e-3)


def test_linear_wake_of_a_gaussian_bunch_matches_the_closed_form():
    # w(s) = a s: V = a Q [sigma/sqrt(2 pi) exp(-zeta^2/2 sigma^2) - zeta/2 erfc(zeta/(sqrt2 sigma))]; total a Q^2 sigma/sqrt(pi)
    a = 5e14
    z, q = gaussian_bunch()
    v = wake_potential(z, q, np.array([0.0, 1.0]), np.array([0.0, a]), n_bin=256)
    centres = np.array([-2.0, -1.0, 0.0, 1.0, 2.0]) * SIGMA_M
    got = np.array(binned_mean(z, v, centres, 0.1 * SIGMA_M))
    x = torch.as_tensor(centres / SIGMA_M)
    expect = (a * Q_C * SIGMA_M * (torch.exp(-x**2 / 2) / math.sqrt(2 * math.pi) - x / 2 * erfc(x / math.sqrt(2)))).numpy()
    np.testing.assert_allclose(got, expect, atol=0.01 * a * Q_C * SIGMA_M)
    assert float((q * v).sum()) == pytest.approx(a * Q_C**2 * SIGMA_M / math.sqrt(math.pi), rel=5e-3)


def test_potential_converges_with_bins_and_is_linear_in_charge():
    z, q = gaussian_bunch(100_000)
    table = (np.array([0.0, 1.0]), np.array([1e12, 1e12]))
    ref = wake_potential(z, q, *table, n_bin=1024)
    errs = [float((wake_potential(z, q, *table, n_bin=n) - ref).abs().max()) / float(ref.max()) for n in (32, 128, 512)]
    assert errs[0] > errs[1] > errs[2] and errs[2] < 2e-3
    torch.testing.assert_close(wake_potential(z, 3 * q, *table, n_bin=128), 3 * wake_potential(z, q, *table, n_bin=128))


def drift_machine(monkeypatch, wakes, n=20_000, gamma=20.0):
    g = torch.Generator().manual_seed(5)
    z = SIGMA_M * torch.randn(n, generator=g, dtype=torch.float64)
    zero = torch.zeros(n, dtype=torch.float64)
    beam = dict(x=zero.clone(), y=zero.clone(), z=z, ux=zero.clone(), uy=zero.clone(),
                uz=torch.full((n,), C_LIGHT * math.sqrt(gamma**2 - 1), dtype=torch.float64),
                q=torch.full((n,), -Q_C / n, dtype=torch.float64), emission_time_s=zero.clone())
    monkeypatch.setattr(Injector, "initial_beam", lambda self, n_, device, dtype, **k: beam)
    monkeypatch.setattr(Injector, "_reference_particle", lambda self: (0.0, 0.0, 0.0))
    return Injector("", 0.6, absolute_phases=True, dt_s=2e-12, space_charge=SpaceChargeSettings(enabled=False),
                    wakes=wakes), gamma


def test_track_loses_the_closed_form_energy_at_the_wake_position(monkeypatch, tmp_path):
    w0 = 1e12
    f = tmp_path / "step.dat"
    f.write_text(f"2 0\n0 {w0}\n1.0 {w0}\n")
    inj, gamma = drift_machine(monkeypatch, [InjectorWake(0.3, str(f), "monopole_f", n_bin=256)])
    inj.base_dir = tmp_path
    out = inj.track(device="cpu", dtype=torch.float64, use_compile=False, verbose=False, record_every=0)
    (kick,) = out["wakes"]
    # the mean over the tracked electrons of -q_e V is -e Q w0/2 for a step wake
    assert kick["mean_dE_eV"] == pytest.approx(-0.5 * Q_C * w0, rel=5e-3)
    assert 0.3 <= kick["applied_at_z_m"] < 0.3 + 12 * C_LIGHT * 2e-12
    e_kin_start = (gamma - 1.0) * ELECTRON_MASS_EV / 1e6
    assert out["Ekin_MeV"] - e_kin_start == pytest.approx(kick["mean_dE_eV"] / 1e6, rel=1e-3)


def test_no_wake_scaling_zero_and_unreached_position_change_nothing(monkeypatch, tmp_path):
    f = tmp_path / "step.dat"
    f.write_text("2 0\n0 1e12\n1.0 1e12\n")
    plain = drift_machine(monkeypatch, [])[0].track(device="cpu", dtype=torch.float64, use_compile=False, verbose=False,
                                                   record_every=0)
    for wk in (InjectorWake(0.3, str(f), "monopole_f", scaling=0.0), InjectorWake(5.0, str(f), "monopole_f")):
        inj = drift_machine(monkeypatch, [wk])[0]
        inj.base_dir = tmp_path
        out = inj.track(device="cpu", dtype=torch.float64, use_compile=False, verbose=False, record_every=0)
        assert out["Ekin_MeV"] == pytest.approx(plain["Ekin_MeV"], rel=1e-12)


def test_wake_survives_the_yaml_round_trip(tmp_path):
    inj = Injector("beam.txt", 1.0, wakes=[InjectorWake(0.5, "w.dat", "taylor_f", scaling=2.0, n_bin=64)], base_dir=tmp_path)
    inj.save(tmp_path / "m.yaml")
    assert Injector.load(tmp_path / "m.yaml").wakes == inj.wakes


def test_deck_import_reads_the_monopole_wakes_and_warns_about_dipoles(tmp_path):
    (tmp_path / "d.in").write_text(
        "&NEWRUN\n Distribution='b.ini'\n/\n&OUTPUT\n ZStop=1.0\n/\n&WAKE\n LWAKE=T\n"
        " Wk_Type(1)='Taylor_Method_F'\n Wk_filename(1)='w.dat'\n Wk_z(1)=0.5\n Wk_N_bin(1)=64\n Wk_scaling=2.0\n"
        " Wk_Type(2)='Dipole_Method_F'\n Wk_filename(2)='d.dat'\n Wk_z(2)=0.6\n/\n")
    with pytest.warns(UserWarning, match="Dipole_Method_F"):
        inj = Injector.from_astra(tmp_path / "d.in")
    assert inj.wakes == [InjectorWake(0.5, "w.dat", "taylor_f", scaling=2.0, n_bin=64)]


def test_wakes_are_rejected_by_the_differentiable_path():
    with pytest.raises(NotImplementedError, match="wake"):
        Injector("", 1.0, wakes=[InjectorWake(0.5, "w.dat")]).track_differentiable()
