"""Quantitative rate checks and common-random-input parity for the optional pair kernel."""
from dataclasses import replace
import math

import pytest
import torch

from slipstream.constants import C_LIGHT, E_CHARGE_C, ELECTRON_MASS_EV, EPS0_F_PER_M
from slipstream.tracking.native import warp_kernels
from slipstream.tracking.native.injector_ibs import (
    BinaryCollisionIBS, InjectorIBSSettings, _collision_randoms, _pair_frame, _pair_increment, _scattering_angles,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not warp_kernels._HAVE_WARP,
                                reason="requires CUDA and warp-lang")
MASS = ELECTRON_MASS_EV * E_CHARGE_C / C_LIGHT**2


@pytest.fixture(scope="module", autouse=True)
def warp_cache(tmp_path_factory):
    import warp as wp
    original = wp.config.kernel_cache_dir
    wp.config.kernel_cache_dir = str(tmp_path_factory.mktemp("ibs-warp-cache"))
    yield
    wp.config.kernel_cache_dir = original


@pytest.mark.parametrize("bulk_gamma", [1., 5., 100.])
def test_warp_absolute_rate_matches_rutherford_integral(bulk_gamma):
    rest_u, weight, dt, logarithm = 4e5, 123456.789, 2e-13, 7.
    thermal_gamma = math.sqrt(1+(rest_u/C_LIGHT)**2)
    bulk_u = math.sqrt(bulk_gamma**2-1)*C_LIGHT*thermal_gamma
    velocity = torch.tensor([[rest_u, 0., bulk_u], [-rest_u, 0., bulk_u]], device="cuda", dtype=torch.float64)
    config = InjectorIBSSettings(enabled=True, cell_size_m=(1e-4, 2e-4, 3e-4), coulomb_log=logarithm, backend="warp")
    solver = BinaryCollisionIBS(config, torch.full((2,), -E_CHARGE_C*weight, device="cuda", dtype=torch.float64), MASS)
    solver.apply(torch.zeros_like(velocity), velocity, dt)
    speed = 2*rest_u/thermal_gamma
    b90 = E_CHARGE_C**2/(4*math.pi*EPS0_F_PER_M*(MASS/2)*speed**2)
    density = 2*weight/math.prod(config.cell_size_m)/bulk_gamma
    expected = 2*math.pi*density*b90**2*speed*(dt/bulk_gamma)*logarithm
    assert solver.max_variance == pytest.approx(expected, rel=3e-12)


@pytest.mark.parametrize("variance", [1e-18, 1e-8, .001, .021, .1, .5, 3.960960960960961, 20., 1000.])
@pytest.mark.parametrize("model", ["nanbu", "takizuka-abe"])
def test_warp_pair_law_matches_reference_with_identical_draws(variance, model):
    from slipstream.tracking.native.injector_ibs_warp import collide
    n_pairs, boost, rest_u = 512, 5., 4e5
    rest_gamma = math.sqrt(1+(rest_u/C_LIGHT)**2)
    bulk_u = math.sqrt(boost**2-1)*C_LIGHT*rest_gamma
    ua = torch.tensor([rest_u, 0., bulk_u], device="cuda", dtype=torch.float64).repeat(n_pairs, 1)
    ub = ua.clone()
    ub[:, 0] *= -1
    velocity = torch.stack((ua, ub), 1).reshape(-1, 3)
    indices = torch.arange(2*n_pairs, device="cuda")
    a, b = indices[::2].contiguous(), indices[1::2].contiguous()
    counts = torch.full((n_pairs,), 2, device="cuda", dtype=torch.int64)
    cells = torch.arange(n_pairs, device="cuda")
    cell_gamma = torch.full((n_pairs,), boost, device="cuda", dtype=torch.float64)
    generator = torch.Generator(device="cuda").manual_seed(812)
    azimuth, random = _collision_randoms(a.shape, velocity.dtype, velocity.device, model, generator)
    speed = 2*rest_u/rest_gamma
    density_scale = variance*boost**2*speed**3/2
    result, measured, _ = collide(velocity, indices, a, b, counts, cells, cell_gamma, azimuth, random,
                                   1., density_scale, 1., 1., 0., .1, model)
    assert measured == pytest.approx(variance, rel=3e-12)
    sine, loss, phi = _scattering_angles(torch.full_like(cell_gamma, variance), model,
                                        torch.Generator(device="cuda").manual_seed(812))
    rest, beta, gamma = _pair_frame(ua, ub)
    increment = _pair_increment(rest, beta, gamma, sine, loss, phi)
    expected = torch.stack((ua+increment, ub-increment), 1).reshape(-1, 3)
    # Compare the small impulse itself, rather than hiding it under the bulk momentum.
    torch.testing.assert_close(result-velocity, expected-velocity, rtol=2e-10, atol=3e-7)
    torch.testing.assert_close(result.sum(0), velocity.sum(0), rtol=2e-14, atol=2e-6)
    energy = lambda u: torch.sqrt(1+(u/C_LIGHT).square().sum(-1)).sum()
    torch.testing.assert_close(energy(result), energy(velocity), rtol=2e-14, atol=0)


@pytest.mark.parametrize("model", ["nanbu", "takizuka-abe"])
@pytest.mark.parametrize("logarithm", [None, 7.])
def test_whole_operator_parity_masks_and_private_stream(model, logarithm):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        generator = torch.Generator(device="cuda").manual_seed(9)
        velocity = torch.randn((501, 3), generator=generator, device="cuda", dtype=torch.float64)*2e5
        velocity[:, 2] += 2e9
        position = torch.randn(velocity.shape, generator=generator, device="cuda", dtype=torch.float64)*1e-4
        charges = torch.full((501,), -1e-15, device="cuda", dtype=torch.float64)
        charges[0] = 0
        active = torch.arange(501, device="cuda") % 7 != 0
        config = InjectorIBSSettings(enabled=True, cell_size_m=(1e-4,)*3, model=model,
                                     coulomb_log=logarithm, seed=42, backend="torch")
        reference = BinaryCollisionIBS(config, charges, MASS)
        candidate = BinaryCollisionIBS(replace(config, backend="warp"), charges, MASS)
        expected = reference.apply(position, velocity, 1e-12, active)
        result = candidate.apply(position, velocity, 1e-12, active)
        torch.testing.assert_close(result-velocity, expected-velocity, rtol=2e-9, atol=2e-6)
        assert torch.equal(reference.generator.get_state(), candidate.generator.get_state())
        assert torch.equal(result[~active], velocity[~active]) and torch.equal(result[0], velocity[0])
        assert candidate.pairs == reference.pairs > 0
        assert candidate.max_variance == pytest.approx(reference.max_variance, rel=3e-11)
        with pytest.raises(NotImplementedError, match="forward-only"):
            candidate.apply(position, velocity.clone().requires_grad_(), 1e-12)
        # Inactive/neutral state is untouched, including when the entire cell is cold.
        cold = velocity.clone()
        cold[:] = cold[0]
        assert torch.equal(candidate.apply(position, cold, 1e-12, active), cold)
        hot = velocity.clone()
        hot[::2, 0] = C_LIGHT
        hot[1::2, 0] = -C_LIGHT
        with pytest.raises(ValueError, match="nonrelativistic"):
            candidate.apply(torch.zeros_like(position), hot, 1e-12)
    stream.synchronize()
