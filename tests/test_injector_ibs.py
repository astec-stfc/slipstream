"""Quantitative local collision normalization, elastic invariants and injector integration."""
from dataclasses import asdict, replace
import math

import numpy as np
import pytest
from scipy.integrate import quad
import torch
import yaml

from slipstream import Injector, InjectorIBSSettings
from slipstream.constants import C_LIGHT, E_CHARGE_C, ELECTRON_MASS_EV, EPS0_F_PER_M
from slipstream.tracking.native import injector_ibs as ibs_module
from slipstream.tracking.native.astra_io import write_astra_distribution
from slipstream.tracking.native.injector import Aperture, SpaceChargeSettings
from slipstream.tracking.native.injector_ibs import (
    BinaryCollisionIBS, _nanbu_parameter, _pair_increment, _scattering_angles,
)


MASS = ELECTRON_MASS_EV * E_CHARGE_C / C_LIGHT**2


def settings(**changes):
    return InjectorIBSSettings(enabled=True, cell_size_m=(1e-4, 2e-4, 3e-4), **changes)


@pytest.mark.parametrize("options", [
    {"enabled": True}, {"enabled": "false"}, {"cell_size_m": (1., 0., 1.)},
    {"cell_size_m": (1., 1.)}, {"model": "ring"}, {"coulomb_log": -1.},
    {"max_impact_parameter_m": 0.}, {"max_pair_beta": .2}, {"seed": -1}, {"backend": "invalid"},
])
def test_invalid_settings_are_rejected(options):
    with pytest.raises(ValueError):
        InjectorIBSSettings(**options)


@pytest.mark.parametrize("boost", [1., 5., 100.])
def test_collision_strength_matches_rutherford_impact_integral(monkeypatch, boost):
    rest_u, physical_weight, dt = 4e5, 123456.789, 2e-13
    thermal_gamma = math.sqrt(1 + (rest_u/C_LIGHT)**2)
    bulk_u = math.sqrt(boost**2 - 1) * C_LIGHT * thermal_gamma
    velocity = torch.tensor([[rest_u, 0., bulk_u], [-rest_u, 0., bulk_u]], dtype=torch.float64)
    charges = torch.full((2,), -E_CHARGE_C * physical_weight, dtype=torch.float64)
    config = settings(coulomb_log=7.)
    captured = []

    def capture(variance, model, generator):
        captured.append(variance.clone())
        return torch.zeros_like(variance), torch.zeros_like(variance), torch.zeros_like(variance)

    monkeypatch.setattr(ibs_module, "_scattering_angles", capture)
    solver = BinaryCollisionIBS(config, charges, MASS)
    result = solver.apply(torch.zeros_like(velocity), velocity, dt)
    relative_speed = 2 * rest_u / thermal_gamma
    reduced_mass = MASS / 2
    landau_length = E_CHARGE_C**2 / (4 * math.pi * EPS0_F_PER_M * reduced_mass * relative_speed**2)
    density_rest = 2 * physical_weight / math.prod(config.cell_size_m) / boost
    # Rutherford small-angle impact integral: <tan(theta/2)^2>=2*pi*n*b90^2*w*dt*lnLambda.
    expected = 2 * math.pi * density_rest * landau_length**2 * relative_speed * (dt/boost) * 7.
    assert float(captured[0][0]) == pytest.approx(expected, rel=3e-12)
    torch.testing.assert_close(result, velocity, rtol=0, atol=0)


def test_nanbu_parameter_satisfies_angular_mean_equation():
    variance = torch.tensor([1e-8, 1e-3, .021, .1, .5, 2., 3.9, 3.960960960960961, 4.01, 8., 30., 1000.], dtype=torch.float64)
    parameter = _nanbu_parameter(variance)
    means = []
    for a in parameter.tolist():
        if a < .1:
            numerator = quad(lambda mu: 2*mu*math.sinh(a*mu), 0, 1, epsabs=1e-15)[0]
            denominator = quad(lambda mu: 2*math.cosh(a*mu), 0, 1, epsabs=1e-15)[0]
            means.append(numerator/denominator)
        else:
            means.append(1/math.tanh(a) - 1/a)
    np.testing.assert_allclose(means, np.exp(-2 * variance.numpy()), rtol=2e-11, atol=3e-14)


@pytest.mark.parametrize("model,variance", [("nanbu", .01), ("nanbu", .5), ("nanbu", 5.),
                                           ("nanbu", 360.), ("nanbu", 400.), ("nanbu", 1000.),
                                           ("takizuka-abe", .01), ("takizuka-abe", .5)])
def test_sampled_angular_mean_has_quantitative_distribution(model, variance):
    generator = torch.Generator().manual_seed(1234)
    tau = torch.full((50000,), variance, dtype=torch.float64)
    sine, loss, phi = _scattering_angles(tau, model, generator)
    cosine = 1 - loss
    if model == "nanbu":
        expected = math.exp(-2 * variance)
    else:
        expected = quad(lambda z: (1-variance*z*z)/(1+variance*z*z) * math.exp(-z*z/2)/math.sqrt(2*math.pi),
                        -12, 12, epsabs=1e-12)[0]
    standard_error = float(cosine.std()) / math.sqrt(len(cosine))
    assert abs(float(cosine.mean()) - expected) < 6 * standard_error + 1e-12
    if variance > 20:
        squared = cosine.square()
        assert abs(float(squared.mean()) - 1/3) < 6*float(squared.std())/math.sqrt(len(squared))
    torch.testing.assert_close(sine.square() + cosine.square(), torch.ones_like(cosine), rtol=2e-14, atol=2e-14)
    assert float(phi.min()) >= 0 and float(phi.max()) <= 2 * math.pi


def test_pair_rotation_known_angle_and_zero_relative_velocity():
    rest = torch.tensor([[0., 0., 4e5], [0., 0., 0.]], dtype=torch.float64)
    increment = _pair_increment(rest, torch.zeros_like(rest), torch.ones(2, dtype=torch.float64),
                                torch.ones(2, dtype=torch.float64), torch.ones(2, dtype=torch.float64),
                                torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(rest + increment, torch.tensor([[0., 4e5, 0.], [0., 0., 0.]], dtype=torch.float64), rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
@pytest.mark.parametrize("model", ["nanbu", "takizuka-abe"])
def test_local_pairing_conserves_lab_four_momentum_and_replays_seed(device, model):
    generator = torch.Generator().manual_seed(9)
    n = 501
    velocity = torch.randn((n, 3), dtype=torch.float64, generator=generator) * 2e5
    velocity[:, 2] += 2e9
    velocity = velocity.to(device)
    position = torch.zeros_like(velocity)
    charges = torch.full((n,), -E_CHARGE_C * 1e6, dtype=torch.float64, device=device)
    config = settings(model=model, coulomb_log=7., seed=42)
    global_before = torch.random.get_rng_state().clone()
    solver = BinaryCollisionIBS(config, charges, MASS)
    result = solver.apply(position, velocity, 1e-12)
    replay = BinaryCollisionIBS(config, charges, MASS).apply(position, velocity, 1e-12)
    assert torch.equal(torch.random.get_rng_state(), global_before)
    assert torch.equal(result, replay)
    assert not torch.equal(result, velocity)
    torch.testing.assert_close(result.sum(0), velocity.sum(0), rtol=2e-14, atol=2e-6)
    gamma = lambda u: torch.sqrt(1 + (u/C_LIGHT).square().sum(-1))
    torch.testing.assert_close(gamma(result).sum(), gamma(velocity).sum(), rtol=2e-14, atol=0)
    assert solver.pairs == n // 2 and solver.unpaired_particles == 1


def test_mask_zero_charge_cold_pairs_and_cutoff_are_noops():
    velocity = torch.tensor([[0., 0., 1e9]]*4, dtype=torch.float64)
    charges = torch.full((4,), -1e-15, dtype=torch.float64)
    zero_q = BinaryCollisionIBS(settings(), torch.zeros_like(charges), MASS)
    assert torch.equal(zero_q.apply(torch.zeros_like(velocity), velocity, 1e-12), velocity)
    cold = BinaryCollisionIBS(settings(), charges, MASS)
    assert torch.equal(cold.apply(torch.zeros_like(velocity), velocity, 1e-12), velocity)
    velocity[0, 0], velocity[1, 0] = 4e5, -4e5
    cutoff = BinaryCollisionIBS(settings(max_impact_parameter_m=1e-20), charges, MASS)
    assert torch.equal(cutoff.apply(torch.zeros_like(velocity), velocity, 1e-12), velocity)
    active = torch.tensor([True, False, False, False])
    assert torch.equal(cold.apply(torch.zeros_like(velocity), velocity, 1e-12, active), velocity)
    charges[2] = 0
    solver = BinaryCollisionIBS(settings(coulomb_log=7.), charges, MASS)
    active = torch.tensor([True, True, True, False])
    result = solver.apply(torch.zeros_like(velocity), velocity, 1e-12, active)
    assert torch.equal(result[2:], velocity[2:])


def test_unequal_tiny_weights_and_relativistic_pair_motion_are_rejected():
    with pytest.raises(ValueError, match="requires CUDA"):
        BinaryCollisionIBS(settings(backend="warp"), torch.full((2,), -1e-15, dtype=torch.float64), MASS)
    with pytest.raises(ValueError, match="equal signed"):
        BinaryCollisionIBS(settings(), torch.tensor([-1e-17, -2e-17], dtype=torch.float64), MASS)
    solver = BinaryCollisionIBS(settings(), torch.full((2,), -1e-15, dtype=torch.float64), MASS)
    velocity = torch.tensor([[C_LIGHT, 0., 0.], [-C_LIGHT, 0., 0.]], dtype=torch.float64)
    with pytest.raises(ValueError, match="nonrelativistic"):
        solver.apply(torch.zeros_like(velocity), velocity, 1e-12)


@pytest.mark.parametrize("side,substeps", [(32, 1), (64, 1), (64, 2), (64, 4)])
def test_homogeneous_initial_diffusion_matches_landau_moment_under_refinement(side, substeps):
    # A square grid of transverse thermal velocities has no coincident streams.
    # The Landau moment is K*n*lnLambda*dt/2 * E[1/w] for initially zero vz.
    speed, density, logarithm = 4e5, 1e19, 7.
    n = side**2
    config = settings(coulomb_log=logarithm)
    weight = density * math.prod(config.cell_size_m) / n
    grid = torch.linspace(-speed, speed, side, dtype=torch.float64)
    vx, vy = torch.meshgrid(grid, grid, indexing="ij")
    velocity = torch.stack((vx.flatten(), vy.flatten(), torch.zeros(n, dtype=torch.float64)), -1)
    coefficient = E_CHARGE_C**4 / (8*math.pi*EPS0_F_PER_M**2*(MASS/2)**2)
    # Same physical density, velocity domain and duration for all refinements;
    # even the nearest pair on the finest grid has variance at most about 1e-3.
    duration = 1e-3 * (2*speed/63)**3 / (coefficient*density*logarithm)
    offset = np.arange(-side+1, side)
    dx, dy = np.meshgrid(offset, offset, indexing="ij")
    separation = np.sqrt(dx*dx + dy*dy)
    multiplicity = (side-np.abs(dx)) * (side-np.abs(dy))
    nonzero = separation > 0
    mean_inverse_speed = np.sum(multiplicity[nonzero] / ((2*speed/(side-1))*separation[nonzero])) / (n*(n-1))
    charges = torch.full((n,), -E_CHARGE_C * weight, dtype=torch.float64)
    samples = []
    for seed in range(12):
        solver = BinaryCollisionIBS(replace(config, seed=seed), charges, MASS)
        current = velocity.clone()
        for _ in range(substeps):
            current = solver.apply(torch.zeros_like(current), current, duration/substeps)
        samples.append(float(current[:, 2].square().mean()))
    expected = coefficient*density*logarithm*duration/2 * mean_inverse_speed
    standard_error = np.std(samples, ddof=1) / math.sqrt(len(samples))
    assert abs(np.mean(samples) - expected) < 6*standard_error + .005*expected


def make_machine(tmp_path):
    generator = torch.Generator().manual_seed(29)
    n = 48
    x, y, z = (torch.randn(n, generator=generator, dtype=torch.float64) * 1e-6 for _ in range(3))
    px, py = (torch.randn(n, generator=generator, dtype=torch.float64) * 300. for _ in range(2))
    pz = torch.full((n,), 2e6, dtype=torch.float64)
    zero = torch.zeros_like(x)
    path = tmp_path / "beam.astra"
    write_astra_distribution(path, x, y, z, px, py, pz, zero, torch.full_like(x, -2e-15))
    return Injector(distribution="beam.astra", z_stop_m=3e-5, base_dir=tmp_path,
                    absolute_phases=True, dt_s=1e-14, space_charge=SpaceChargeSettings(enabled=False))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_injector_public_tracking_default_off_seed_and_yaml(tmp_path, device):
    machine = make_machine(tmp_path)
    args = dict(device=device, dtype=torch.float64, use_compile=False, verbose=False, return_particles=True)
    baseline = machine.track(**args)
    machine.ibs = settings(coulomb_log=7., seed=2)
    machine.save(tmp_path / "machine.yaml")
    loaded = Injector.load(tmp_path / "machine.yaml")
    assert asdict(loaded.ibs) == asdict(machine.ibs)
    result, again = loaded.track(**args), loaded.track(**args)
    assert result["ibs"]["pairs"] > 0
    assert torch.equal(result["ux"], again["ux"])
    assert not torch.equal(result["ux"], baseline["ux"])
    assert result["n_particles_alive"] == baseline["n_particles_alive"]
    loaded.ibs = replace(loaded.ibs, enabled=False)
    disabled = loaded.track(**args)
    for key in ("x", "y", "z", "ux", "uy", "uz", "q"):
        assert torch.equal(disabled[key], baseline[key])
    data = yaml.safe_load((tmp_path / "machine.yaml").read_text())
    data.pop("ibs")
    (tmp_path / "legacy.yaml").write_text(yaml.safe_dump(data))
    assert not Injector.load(tmp_path / "legacy.yaml").ibs.enabled


def test_collisions_compose_with_native_space_charge(tmp_path):
    machine = make_machine(tmp_path)
    machine.space_charge = SpaceChargeSettings(enabled=True, cathode_image=False, nr=16, nz=32,
                                               mesh="fixed", dt_adaptive=False)
    args = dict(device="cpu", use_compile=False, verbose=False, return_particles=True)
    baseline = machine.track(**args)
    machine.ibs = settings(coulomb_log=0.)
    disabled_kick = machine.track(**args)
    for key in ("x", "y", "z", "ux", "uy", "uz"):
        assert torch.equal(disabled_kick[key], baseline[key])
    machine.ibs = settings(coulomb_log=7.)
    result = machine.track(**args)
    assert result["ibs"]["pairs"] > 0 and result["ibs"]["max_variance"] > 0
    assert torch.isfinite(result["ux"]).all()


def test_unsupported_tracking_modes_fail_before_reading_input():
    machine = Injector("absent.astra", .1, ibs=settings())
    with pytest.raises(ValueError, match="float64"):
        machine.track(device="cpu", dtype=torch.float32)
    with pytest.raises(NotImplementedError, match="stochastic"):
        machine.track_differentiable()


def test_injector_collision_mask_excludes_future_emission_and_new_aperture_losses(tmp_path, monkeypatch):
    machine = make_machine(tmp_path)
    machine.ibs = settings(coulomb_log=7.)
    machine.apertures = [Aperture("RAD", -1., 1., 2e-6)]
    original_initial = machine.initial_beam

    def delayed_beam(*args, **kwargs):
        beam = original_initial(*args, **kwargs)
        beam["emission_time_s"][len(beam["q"])//2:] = 2*machine.dt_s
        return beam

    monkeypatch.setattr(machine, "initial_beam", delayed_beam)
    original_apply = BinaryCollisionIBS.apply
    calls = []

    def inspect_mask(self, position, velocity, dt, active):
        if len(calls) < 2:
            assert not bool(active[len(active)//2:].any())
        outside = position[:, :2].square().sum(-1) > (2e-6)**2
        assert not bool(active[outside].any())
        calls.append(active.clone())
        return original_apply(self, position, velocity, dt, active)

    monkeypatch.setattr(BinaryCollisionIBS, "apply", inspect_mask)
    result = machine.track(device="cpu", use_compile=False, verbose=False)
    assert calls and result["n_lost_on_aperture"] > 0
    assert result["ibs"]["pairs"] > 0
