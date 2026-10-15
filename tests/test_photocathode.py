"""Photocathode emission checked against closed forms: Schottky lowering, Dowell-Schmerge energy factor, the
Gauss-law space-charge limit of a short pulse, and a disc's analytic surface field."""
import dataclasses
import math

import numpy as np
import pytest
import torch
import yaml

from slipstream.constants import E_CHARGE_C, EPS0_F_PER_M
import slipstream.tracking.native.injector as injector_module
from slipstream.tracking.native.injector import Aperture, Cavity, Injector, SpaceChargeSettings
from slipstream.tracking.native.photocathode import Photocathode, PhotoEmitter, schottky_lowering_eV

HW_266 = 1239.841984 / 266.0  # photon energy, eV


def disc(n, radius):
    """Equal-area sunflower sample of a uniform disc: radii and azimuths."""
    i = torch.arange(n, dtype=torch.float64)
    return radius * torch.sqrt((i + 0.5) / n), i * math.pi * (3.0 - math.sqrt(5.0))


def gaussian_disc(n, sigma, cut=3.0):
    """Radii of a deterministic Gaussian sample truncated at ``cut`` sigma (inverse CDF of the Rayleigh law)."""
    i = torch.arange(n, dtype=torch.float64)
    u = (i + 0.5) / n
    r = sigma * torch.sqrt(-2.0 * torch.log(1.0 - u * (1.0 - math.exp(-0.5 * cut * cut))))
    return r, i * math.pi * (3.0 - math.sqrt(5.0))


def dc_gap(tmp_path, monkeypatch, field, photo, r, theta, tau, dt=0.02e-12, sc=True, frequency=0.0, crest=180.0,
           phase=0.0, z_stop=0.0005):
    """A uniform field over 5 cm, ``field`` V/m extracting at t=0 (or an RF cosine), a beam emitted over ``tau``."""
    path = tmp_path / "flat.astra"
    np.savetxt(path, np.column_stack([np.linspace(0.0, 0.05, 51), np.ones(51)]))
    n = r.numel()
    zero = torch.zeros(n, dtype=torch.float64)
    t = tau * ((torch.arange(n, dtype=torch.float64) * 0.6180339887498949) % 1.0)  # low-discrepancy emission times
    beam = dict(x=r * torch.cos(theta), y=r * torch.sin(theta), z=zero, ux=zero.clone(), uy=zero.clone(),
                uz=torch.full((n,), 1e5, dtype=torch.float64), q=torch.full((n,), -1e-15, dtype=torch.float64),
                emission_time_s=t)
    monkeypatch.setattr(Injector, "initial_beam", lambda self, n, device, dtype, **k: {
        key: v.to(device=device, dtype=dtype) for key, v in beam.items()})
    monkeypatch.setattr(Injector, "_reference_particle", lambda self: (0.0, 0.0, 0.0))
    return Injector("", z_stop, cavities=[Cavity(str(path), 0.0, field, frequency, phase, 0, crest)],
                    absolute_phases=True, dt_s=dt, photocathode=photo,
                    space_charge=SpaceChargeSettings(enabled=sc, dt_adaptive=False, nr=32, nz=64))


def track(machine, device="cpu", dtype=torch.float64, use_compile=False, **kwargs):
    return machine.track(device=device, dtype=dtype, use_compile=use_compile, verbose=False, record_every=0, **kwargs)


def constant_qe(charge_over_limit, field, radius, qe=1e-4):
    """A constant-QE laser whose unscreened charge is ``charge_over_limit`` times eps0*E*pi*R^2."""
    energy = charge_over_limit * EPS0_F_PER_M * field * math.pi * radius ** 2 * HW_266 / qe
    return Photocathode(pulse_energy_J=energy, wavelength_nm=266.0, qe=qe)


def test_schottky_lowering_matches_dowell_schmerge_table():
    # Dowell & Schmerge, PRSTAB 12, 074201, Table I: 0.268 eV at 50 MV/m; Eq. (2): 0.037947 sqrt(E[MV/m]).
    field = torch.tensor([50e6, 200e6], dtype=torch.float64)
    lowering = schottky_lowering_eV(field)
    assert lowering[0].item() == pytest.approx(0.268, abs=5e-4)
    assert lowering.tolist() == pytest.approx([0.037947 * math.sqrt(50.0), 0.037947 * math.sqrt(200.0)], rel=2e-5)
    assert schottky_lowering_eV(field, 4.0)[0].item() == pytest.approx(2.0 * lowering[0].item())


def test_quantum_efficiency_follows_the_dowell_schmerge_energy_factor():
    ef, wf, ref = 7.0, 4.31, 100e6
    photo = Photocathode(1e-6, 266.0, 2e-4, work_function_eV=wf, fermi_energy_eV=ef, qe_field_V_per_m=ref)

    def factor(field):  # (EF+hw)/(2hw) * (1 - sqrt((EF+phi_eff)/(EF+hw)))^2, written out independently
        phi = wf - 0.037947 * math.sqrt(field / 1e6)
        return (ef + HW_266) / (2 * HW_266) * (1 - math.sqrt((ef + phi) / (ef + HW_266))) ** 2

    fields = torch.tensor([-5e6, 0.0, 10e6, 60e6, 100e6, 140e6], dtype=torch.float64)
    qe = photo.quantum_efficiency(fields)
    assert qe[:2].tolist() == [0.0, 0.0]  # a field that does not extract emits nothing
    assert qe[4].item() == pytest.approx(2e-4, rel=1e-12)  # quoted at the reference field
    for f, value in zip(fields[2:].tolist(), qe[2:].tolist()):
        assert value == pytest.approx(2e-4 * factor(f) / factor(ref), rel=1e-4)  # the paper's rounded 0.037947
    assert qe[2] < qe[3] < qe[4] < qe[5]  # the Schottky effect raises the yield with field


def test_quantum_efficiency_is_constant_without_a_work_function_and_validates_inputs():
    photo = Photocathode(1e-6, 266.0, 3e-5)
    qe = photo.quantum_efficiency(torch.tensor([-1.0, 1.0, 1e8], dtype=torch.float64))
    assert qe.tolist() == [0.0, 3e-5, 3e-5]
    cu = Photocathode(1e-6, 266.0, 1e-4, material="Cu", qe_field_V_per_m=50e6)
    assert (cu.work_function_eV, cu.fermi_energy_eV) == (4.31, 7.0)
    with pytest.raises(KeyError, match="Cu"):
        Photocathode(1e-6, 266.0, 1e-4, material="Unobtainium")
    with pytest.raises(ValueError, match="qe_field_V_per_m"):
        Photocathode(1e-6, 266.0, 1e-4, work_function_eV=4.3)
    with pytest.raises(ValueError, match="threshold"):  # 266 nm photons cannot cross a 4.9 eV barrier at 10 MV/m
        Photocathode(1e-6, 266.0, 1e-4, work_function_eV=4.9, qe_field_V_per_m=10e6)
    with pytest.raises(ValueError):
        Photocathode(1e-6, 266.0, 1.5)
    with pytest.raises(ValueError):
        Photocathode(0.0, 266.0, 1e-4)


def test_average_power_and_repetition_rate_give_the_pulse_energy():
    photo = Photocathode.from_average_power(2.0e-3, 400.0, wavelength_nm=266.0, qe=1e-4)  # 2 mW at 400 Hz
    assert photo.pulse_energy_J == pytest.approx(5e-6, rel=1e-12)


def test_photons_and_photocathode_round_trip_through_yaml():
    photo = Photocathode(2e-6, 266.0, 1e-4, material="Cu", qe_field_V_per_m=100e6, field_enhancement=1.5)
    assert photo.photons == pytest.approx(2e-6 / (HW_266 * E_CHARGE_C), rel=1e-12)
    again = Photocathode(**yaml.safe_load(yaml.safe_dump(dataclasses.asdict(photo))))
    assert again == photo


def test_surface_screening_of_a_charged_disc_matches_the_analytic_on_axis_field():
    # A disc of radius R and areal density sigma, a height h above a grounded plane (image included):
    # E(0) = sigma/eps0 * (1 - h/sqrt(h^2 + R^2)).
    n, radius, charge = 4000, 1e-3, 100e-12
    r, _ = disc(n, radius)
    emitter = PhotoEmitter(Photocathode(1e-6, 266.0, 1e-4), r, torch.ones(n, dtype=torch.float64))
    sigma = charge / (math.pi * radius ** 2)
    for height in (50e-6, 200e-6, 1e-3):
        field = emitter.screening(r, torch.full_like(r, height), torch.full_like(r, charge / n))
        exact = sigma / EPS0_F_PER_M * (1.0 - height / math.hypot(height, radius))
        assert field[:3].mean().item() == pytest.approx(exact, rel=0.02)  # bin centres sit a little off the axis
    # a thin sheet screens the whole local sheet field sigma/eps0 away from the edge
    field = emitter.screening(r, torch.full_like(r, 1e-6), torch.full_like(r, charge / n))
    assert emitter._at(field, torch.tensor([0.5 * radius], dtype=torch.float64)).item() == pytest.approx(
        sigma / EPS0_F_PER_M, rel=0.01)


def test_one_layer_emits_at_most_the_gauss_charge_and_solves_the_self_screened_yield():
    n, radius, field = 4000, 1e-3, 50e6
    r, _ = disc(n, radius)
    weights = torch.ones(n, dtype=torch.float64)
    limit = EPS0_F_PER_M * field * math.pi * radius ** 2
    empty = (torch.zeros(0, dtype=torch.float64),) * 3
    idx = torch.arange(n)
    for over in (0.5, 4.0):  # below and above the limit
        photo = constant_qe(over, field, radius)
        emitter = PhotoEmitter(photo, r, weights)
        q, e0 = emitter.emit(idx, r, torch.full((n,), field, dtype=torch.float64), empty)
        assert q.sum().item() == pytest.approx(min(over, 1.0) * limit, rel=0.02)
        assert torch.allclose(e0, torch.full_like(e0, field))
    # the soft (field-dependent) yield: q = qp * QE(E0 - c q / 2) holds at the root the bisection returns
    photo = Photocathode(1e-5, 266.0, 1e-4, work_function_eV=4.6, qe_field_V_per_m=field)
    emitter = PhotoEmitter(photo, r, weights)
    q, _ = emitter.emit(idx, r, torch.full((n,), field, dtype=torch.float64), empty)
    c = emitter.photon_density[emitter._bin(r)] / (emitter.q_phot * EPS0_F_PER_M)
    expected = emitter.q_phot * photo.quantum_efficiency(field - 0.5 * c * q)
    assert torch.allclose(q, expected, rtol=1e-6)
    assert (q < emitter.q_phot * photo.quantum_efficiency(torch.tensor(field, dtype=torch.float64))).all()


def test_small_laser_energy_emits_exactly_qe_times_photons(tmp_path, monkeypatch):
    r, theta = disc(800, 1e-3)
    photo = constant_qe(0.05, 80e6, 1e-3, qe=2e-5)  # 5 % of the Gauss limit: no screening to speak of
    out = track(dc_gap(tmp_path, monkeypatch, 80e6, photo, r, theta, 0.1e-12))
    emission = out["emission"]
    assert emission["photons"] == pytest.approx(photo.pulse_energy_J / (HW_266 * E_CHARGE_C), rel=1e-12)
    assert emission["ideal_charge_C"] == pytest.approx(2e-5 * photo.photons * E_CHARGE_C, rel=1e-12)
    assert emission["emitted_charge_C"] == pytest.approx(emission["ideal_charge_C"], rel=2e-3)
    assert emission["charge_C"] == pytest.approx(emission["emitted_charge_C"], rel=1e-12)
    assert emission["n_suppressed"] == emission["n_returned"] == 0
    assert (emission["q"] < 0).all()  # electrons
    assert out["n_lost_on_aperture"] == 0


@pytest.mark.parametrize("over", [3.0, 10.0])
def test_short_pulse_saturates_at_the_gauss_law_limit(tmp_path, monkeypatch, over):
    # A pulse far shorter than the transit time over a disc much wider than the cloud's height: the surface field
    # cancels once sigma = eps0*E, so the charge is eps0*E*pi*R^2 whatever the laser energy.
    field, radius = 50e6, 1e-3
    r, theta = disc(3000, radius)
    machine = dc_gap(tmp_path, monkeypatch, field, constant_qe(over, field, radius), r, theta, 0.2e-12,
                     dt=0.01e-12)
    emission = track(machine)["emission"]
    limit = EPS0_F_PER_M * field * math.pi * radius ** 2
    assert emission["ideal_charge_C"] == pytest.approx(over * limit, rel=1e-3)
    assert emission["emitted_charge_C"] == pytest.approx(limit, rel=0.05)
    assert emission["n_suppressed"] > 0  # the rest of the pulse finds no extracting field


def test_gaussian_spot_is_limited_locally_where_the_laser_exceeds_the_gauss_density(tmp_path, monkeypatch):
    field, sigma, cut = 50e6, 0.25e-3, 3.0
    r, theta = gaussian_disc(4000, sigma, cut)
    sigma_limit = EPS0_F_PER_M * field
    trunc = 1.0 - math.exp(-0.5 * cut ** 2)
    ideal = 3.0 * sigma_limit * 2.0 * math.pi * sigma ** 2 * trunc  # the centre is 3x over the local limit
    energy = ideal * HW_266 / 1e-4
    photo = Photocathode(pulse_energy_J=energy, wavelength_nm=266.0, qe=1e-4)
    emission = track(dc_gap(tmp_path, monkeypatch, field, photo, r, theta, 0.2e-12, dt=0.01e-12))["emission"]
    # clip the Gaussian areal density at sigma_limit: inside r* the limit, outside the laser as it is
    r_star2 = 2.0 * sigma ** 2 * math.log(ideal / (trunc * 2.0 * math.pi * sigma ** 2 * sigma_limit))
    outside = (math.exp(-0.5 * r_star2 / sigma ** 2) - math.exp(-0.5 * cut ** 2)) / trunc
    expected = sigma_limit * math.pi * r_star2 + ideal * outside
    assert emission["emitted_charge_C"] == pytest.approx(expected, rel=0.07)
    assert emission["emitted_charge_C"] < 0.8 * ideal


def test_emitted_charge_scales_with_the_field_as_the_schottky_quantum_efficiency(tmp_path, monkeypatch):
    r, theta = disc(600, 1e-3)
    values = {}
    for field in (40e6, 120e6):
        photo = Photocathode(1e-8, 266.0, 1e-4, work_function_eV=4.55, qe_field_V_per_m=120e6)  # far below the limit
        values[field] = track(dc_gap(tmp_path, monkeypatch, field, photo, r, theta, 0.1e-12))["emission"]
    ratio = values[40e6]["emitted_charge_C"] / values[120e6]["emitted_charge_C"]
    expected = photo.quantum_efficiency(torch.tensor(40e6, dtype=torch.float64)) / 1e-4
    assert ratio == pytest.approx(float(expected), rel=1e-3)
    assert 0.1 < ratio < 0.6  # a strong Schottky dependence near threshold, not a constant yield


def test_a_retarding_field_emits_nothing_and_the_run_still_ends(tmp_path, monkeypatch):
    r, theta = disc(500, 1e-3)
    machine = dc_gap(tmp_path, monkeypatch, 50e6, constant_qe(1.0, 50e6, 1e-3), r, theta, 0.1e-12, crest=0.0)
    out = track(machine)
    emission = out["emission"]
    assert emission["emitted_charge_C"] == emission["charge_C"] == 0.0
    assert emission["n_suppressed"] == 500
    assert out["n_particles_alive"] == 0 and out["n_steps"] < 100


def test_electrons_emitted_late_in_the_half_cycle_fall_back_and_are_absorbed(tmp_path, monkeypatch):
    # Extraction field E0*sin(w t + theta) in a uniform gap: v(t) is proportional to cos(theta) - cos(w t + theta), so
    # the mean drift is forward for an emission before the field peak (theta < 90 deg) and backward after it.
    frequency, field, r_t = 3e9, 60e6, disc(400, 0.3e-3)
    results = {}
    for name, theta_e in (("early", 45.0), ("late", 120.0)):
        photo = constant_qe(0.02, field, 0.3e-3)
        machine = dc_gap(tmp_path, monkeypatch, field, photo, *r_t, 0.05e-12, dt=0.1e-12, frequency=frequency,
                         crest=90.0, phase=theta_e, z_stop=0.02, sc=False)
        out = track(machine)
        results[name] = (out["emission"], out)
    early, late = results["early"][0], results["late"][0]
    assert early["n_returned"] == 0 and early["charge_C"] == pytest.approx(early["emitted_charge_C"])
    assert late["n_returned"] == 400 and late["charge_C"] == 0.0
    assert late["returned_charge_C"] == pytest.approx(late["emitted_charge_C"])


def test_track_differentiable_rejects_a_photocathode(tmp_path, monkeypatch):
    r, theta = disc(100, 1e-3)
    machine = dc_gap(tmp_path, monkeypatch, 50e6, constant_qe(0.1, 50e6, 1e-3), r, theta, 0.1e-12)
    with pytest.raises(NotImplementedError, match="track"):
        machine.track_differentiable(n_steps=2)


def test_a_beam_that_is_not_on_the_cathode_is_rejected(tmp_path, monkeypatch):
    r, theta = disc(100, 1e-3)
    machine = dc_gap(tmp_path, monkeypatch, 50e6, constant_qe(0.1, 50e6, 1e-3), r, theta, 0.1e-12)
    monkeypatch.setattr(Injector, "_reference_particle", lambda self: (-0.01, 0.0, 0.0))  # cathode 1 cm upstream
    with pytest.raises(ValueError, match="cathode distribution"):
        track(machine)


def test_default_injector_has_no_emission_report_and_keeps_the_beams_charge(tmp_path, monkeypatch):
    r, theta = disc(300, 1e-3)
    machine = dc_gap(tmp_path, monkeypatch, 50e6, None, r, theta, 0.1e-12)
    out = track(machine, return_particles=True)
    assert "emission" not in out
    assert float(out["q"].sum()) == pytest.approx(-300e-15)


def test_injector_yaml_keeps_the_photocathode(tmp_path):
    photo = Photocathode(2e-6, 266.0, 1e-4, material="Cu", qe_field_V_per_m=100e6)
    machine = Injector("beam.astra", 0.1, photocathode=photo)
    machine.save(tmp_path / "machine.yaml")
    assert Injector.load(tmp_path / "machine.yaml").photocathode == photo
    machine.photocathode = None
    machine.save(tmp_path / "plain.yaml")
    assert Injector.load(tmp_path / "plain.yaml").photocathode is None


def test_needed_checks_skip_only_what_is_out_of_reach():
    needed = injector_module._needed_checks
    slabs = [(0.0, 0.001), (0.32, 0.321)]
    assert needed((-math.inf, math.inf), 0.0, 0.5, slabs, 0.0) == (True, True, True)  # no host sync yet
    beam, travel = (0.10, 0.20), 0.001  # live particles at the last sync; farthest anything can have moved since
    assert needed(beam, travel, 0.5, slabs, 0.0) == (False, False, False)
    assert needed(beam, travel, 0.2005, slabs, 0.0)[0]  # the plane is within one step of the lead particle
    assert needed(beam, travel, None, slabs, 0.0)[0] is False  # no screen requested
    assert needed(beam, travel, 0.5, [(0.2005, 0.201)], 0.0)[1]  # a slab the lead particle can reach
    assert needed(beam, travel, 0.5, [(0.0, 0.098)], 0.0)[1] is False  # a slab behind the tail, out of reach
    assert needed(beam, travel, 0.5, None, 0.0)[1]  # placed apertures: lab extent unknown, always checked
    assert needed((0.0005, 0.20), travel, 0.5, slabs, 0.0)[2]  # a tail particle can still reach the cathode wall


@pytest.mark.parametrize("theta_e", [45.0, 120.0])
def test_skipping_idle_checks_changes_nothing(tmp_path, monkeypatch, theta_e):
    # theta 45: electrons escape, a slab clips part of the beam, a plane at 12 mm records the survivors.
    # theta 120: every electron falls back onto the cathode after flying a few mm (no aperture here).
    r, theta = disc(300, 0.3e-3)
    results, flags = {}, []
    real = injector_module._needed_checks

    def recording(*args, **kwargs):
        flags.append(real(*args, **kwargs))
        return flags[-1]

    for name, rule in (("gated", recording), ("forced", lambda *args, **kwargs: (True, True, True))):
        monkeypatch.setattr(injector_module, "_needed_checks", rule)
        machine = dc_gap(tmp_path, monkeypatch, 60e6, constant_qe(0.02, 60e6, 0.3e-3), r, theta, 0.05e-12, dt=0.1e-12,
                         frequency=3e9, crest=90.0, phase=theta_e, z_stop=0.02, sc=False)
        machine.apertures = [Aperture("RAD", 0.004, 0.0045, 0.15e-3)] if theta_e < 90.0 else []
        results[name] = track(machine, screen_z_m=0.012)
    gated, forced = results["gated"], results["forced"]
    assert gated["n_steps"] == forced["n_steps"] and gated["n_lost_on_aperture"] == forced["n_lost_on_aperture"]
    assert gated["emission"]["n_returned"] == forced["emission"]["n_returned"]
    assert torch.equal(gated["emission"]["q"], forced["emission"]["q"])
    for key in ("x", "y", "z", "ux", "uy", "uz", "t_s", "q", "crossed"):
        assert torch.equal(gated["screen"][key], forced["screen"][key]), key
    assert any(not f[0] for f in flags)  # the plane check really was skipped while the beam was far from it
    if theta_e < 90.0:
        assert gated["n_lost_on_aperture"] > 0 and gated["screen"]["n_crossed"] > 0 and any(f[0] for f in flags)
    else:
        assert gated["emission"]["n_returned"] == 300 and gated["n_lost_on_aperture"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware required")
@pytest.mark.parametrize("use_warp", [True, False])
def test_cuda_float32_emission_matches_cpu_float64(tmp_path, monkeypatch, use_warp):
    # overdriven short pulse: emission limit, suppressed macroparticles, SC image solve and the compiled/Warp step
    field, radius = 50e6, 1e-3
    r, theta = disc(1500, radius)
    machine = dc_gap(tmp_path, monkeypatch, field, constant_qe(3.0, field, radius), r, theta, 0.2e-12, dt=0.02e-12)
    reference = track(machine)["emission"]
    candidate = track(machine, device="cuda", dtype=torch.float32, use_compile=True, use_warp=use_warp)["emission"]
    assert candidate["emitted_charge_C"] == pytest.approx(reference["emitted_charge_C"], rel=2e-3)
    assert candidate["charge_C"] == pytest.approx(reference["charge_C"], rel=2e-3)
    assert candidate["n_suppressed"] == reference["n_suppressed"]
    assert candidate["q"].device.type == "cuda" and candidate["q"].dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware required")
def test_a_new_rf_phase_reuses_the_compiled_cuda_step(tmp_path, monkeypatch):
    # The RF phase is a 0-d tensor input of the compiled step: as a Python float it was a compile-time constant and
    # every point of a phase scan recompiled the step (then torch._dynamo's recompile limit fell back to eager).
    import torch._dynamo.utils as dynamo_utils

    r, theta = disc(300, 0.3e-3)
    graphs = []
    for phase in (40.0, 45.0, 50.0):
        machine = dc_gap(tmp_path, monkeypatch, 60e6, constant_qe(0.05, 60e6, 0.3e-3), r, theta, 0.05e-12, dt=0.1e-12,
                         frequency=3e9, crest=90.0, phase=phase, z_stop=0.01, sc=False)
        track(machine, device="cuda", dtype=torch.float32, use_compile=True, use_warp=False)
        graphs.append(dynamo_utils.counters["stats"]["unique_graphs"])
    assert graphs[0] == graphs[1] == graphs[2]
