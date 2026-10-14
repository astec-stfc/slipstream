"""Plane observations checked against analytic drifts and independent finite differences."""
import pytest
import torch

from slipstream.constants import C_LIGHT
from slipstream.tracking.native.injector import Aperture, Injector, SpaceChargeSettings


def beam(device='cpu', dtype=torch.float64):
    a = torch.arange(8, device=device, dtype=dtype)
    x = (a - 3.0) * 1e-4
    y = (a - 3.0).sin() * 1e-4
    x[3], y[3] = 0.0, 0.0
    return dict(x=x, y=y, z=a * 1e-5, ux=(a - 3.0) * 1e5,
                uy=(a - 3.0) * -2e5, uz=(a + 8.0) * 1e8,
                q=torch.full_like(a, -5e-12 / len(a)), emission_time_s=a * 1e-13)


def injector():
    return Injector('', 0.012, absolute_phases=True, dt_s=1e-12,
                    space_charge=SpaceChargeSettings(enabled=False))


def run(b, checkpoint=0, **kwargs):
    return injector().track_differentiable(beam=b, device=str(b['x'].device), dtype=b['x'].dtype,
                                         n_steps=80, checkpoint_every=checkpoint,
                                         screen_z_m=0.006, return_particles=True, **kwargs)


@pytest.mark.parametrize('checkpoint', [0, 11])
def test_plane_drift_is_charge_preserving_and_not_a_time_snapshot(checkpoint):
    b = beam()
    out = run(b, checkpoint)
    s = out['screen']
    expected_x = b['x'] + b['ux'] / b['uz'] * (0.006 - b['z'])
    expected_y = b['y'] + b['uy'] / b['uz'] * (0.006 - b['z'])
    torch.testing.assert_close(s['x'], expected_x, atol=2e-18, rtol=2e-13)
    torch.testing.assert_close(s['y'], expected_y, atol=2e-18, rtol=2e-13)
    gamma = (1 + (b['ux']**2 + b['uy']**2 + b['uz']**2) / C_LIGHT**2).sqrt()
    emission_step = torch.ceil(b['emission_time_s'] / 1e-12) * 1e-12
    torch.testing.assert_close(s['t_s'], emission_step + (0.006 - b['z']) * gamma / b['uz'],
                               atol=2e-24, rtol=2e-13)
    assert s['crossed'].all() and s['n_unobserved_alive'] == 0
    torch.testing.assert_close(s['q'], b['q'], rtol=0, atol=0)
    assert s['z'].eq(0.006).all()
    assert not torch.allclose(out['x'], s['x'], atol=1e-8)


def test_plane_drift_source_gradient_and_checkpoint_match_finite_difference():
    def objective(scale, checkpoint):
        b = beam()
        b['ux'] = b['ux'] + scale * 1e5
        s = run(b, checkpoint)['screen']
        return (s['x'] * torch.arange(1, 9, dtype=torch.float64)).sum()

    grads = []
    for checkpoint in (0, 11):
        scale = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
        loss = objective(scale, checkpoint)
        grad, = torch.autograd.grad(loss, scale)
        fd = (objective(scale.detach() + 1e-4, checkpoint) -
              objective(scale.detach() - 1e-4, checkpoint)) / 2e-4
        torch.testing.assert_close(grad, fd, rtol=2e-9, atol=1e-14)
        grads.append(grad)
    torch.testing.assert_close(*grads, rtol=1e-12, atol=1e-16)


def test_backward_stationary_unemitted_particles_are_reported_and_do_not_deposit():
    b = beam()
    b['uz'][0] *= -1
    b['uz'][1] = 0
    b['emission_time_s'][2] = 1e-9
    out = run(b)['screen']
    assert out['n_crossed'] == 5
    assert out['n_unobserved_alive'] == 3
    assert out['n_lost_before_screen'] == 0
    assert out['incomplete_charge_C'] == pytest.approx(3 * 5e-12 / 8)
    assert out['q'][:3].eq(0).all()
    assert out['x'][:3].eq(0).all()


@pytest.mark.parametrize('aperture_z,expected_count', [(0.003, 0), (0.007, 8)])
def test_aperture_before_plane_excludes_charge_and_after_plane_retains_it(aperture_z, expected_count):
    m = injector()
    # A plug intercepts every particle, even when the plane and plug are in one step.
    m.apertures = [Aperture('RAD', aperture_z, aperture_z + 1e-5, -0.01)]
    m.dt_s = 2e-11
    b = beam()
    out = m.track_differentiable(beam=b, n_steps=5, device='cpu', dtype=torch.float64,
                                 screen_z_m=0.006, return_particles=True, checkpoint_every=2)
    assert not out['alive'].any()
    assert out['screen']['n_crossed'] == expected_count
    assert out['screen']['n_lost_before_screen'] == 8 - expected_count
    assert out['screen']['q'].abs().sum() == pytest.approx(expected_count * 5e-12 / 8)


def test_ordinary_monitor_waits_for_slowest_particle_and_replays(monkeypatch):
    m = injector()
    b = beam()
    b['uz'][0] = 0.5e8
    monkeypatch.setattr(Injector, 'initial_beam', lambda *args, **kwargs: b)
    monkeypatch.setattr(Injector, '_reference_particle', lambda self: (0.0, 0.0, 0.0))
    ordinary = m.track(device='cpu', dtype=torch.float64, use_compile=False, verbose=False,
                       record_every=0, return_particles=True, screen_z_m=0.006)
    replay = m.track_differentiable(beam=b, device='cpu', dtype=torch.float64,
                                  dt_schedule=ordinary['dt_steps_s'], screen_z_m=0.006)
    assert ordinary['screen']['n_unobserved_alive'] == 0
    for key in ('x', 'y', 'z', 't_s', 'q'):
        torch.testing.assert_close(ordinary['screen'][key], replay['screen'][key], atol=1e-18, rtol=1e-12)
    assert ordinary['screen']['t_s'].max() > 80e-12


def test_screen_is_opt_in_and_rejects_nonfinite_plane():
    m, b = injector(), beam()
    default = m.track_differentiable(beam=b, n_steps=80, device='cpu', return_particles=True)
    monitored = run(b)
    assert 'screen' not in default
    for key in ('x', 'y', 'z', 'ux', 'uy', 'uz', 'q'):
        torch.testing.assert_close(default[key], monitored[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match='finite'):
        m.track_differentiable(beam=b, n_steps=2, screen_z_m=float('nan'))


def test_first_crossing_is_retained_after_backward_and_forward_recrossing():
    from slipstream.tracking.native.injector import _empty_screen_state, _record_screen_crossing

    b = beam()
    before = tuple(b[key] for key in ('x', 'y', 'z', 'ux', 'uy', 'uz'))
    after = (*before[:2], torch.full_like(b['z'], 0.007), *before[3:])
    active = torch.ones_like(b['x'], dtype=torch.bool)
    state = _record_screen_crossing(_empty_screen_state(b['x']), before, after,
                                    active, active, (), 0.006, 0.0, 1e-12)
    back = (b['x'] + 0.001, b['y'], torch.full_like(b['z'], 0.005), *before[3:])
    state2 = _record_screen_crossing(state, after, back, active, active, (), 0.006, 1e-12, 1e-12)
    again = _record_screen_crossing(state2, back, after, active, active, (), 0.006, 2e-12, 1e-12)
    for original, recrossed in zip(state, again):
        torch.testing.assert_close(original, recrossed, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA hardware required')
def test_cuda_float32_screen_values_and_source_gradient_match_cpu_float64():
    def evaluate(device, dtype):
        b = beam(device, dtype)
        scale = torch.tensor(0.37, dtype=dtype, device=device, requires_grad=True)
        b['ux'] = b['ux'] + scale * 1e5
        s = run(b, 11)['screen']
        loss = (s['x'] * torch.arange(1, 9, dtype=dtype, device=device)).sum()
        grad, = torch.autograd.grad(loss, scale)
        return s['x'].detach().cpu().double(), grad.detach().cpu().double()

    reference, production = evaluate('cpu', torch.float64), evaluate('cuda', torch.float32)
    for expected, actual in zip(reference, production):
        torch.testing.assert_close(actual, expected, rtol=5e-6, atol=1e-10)


def test_cylindrical_space_charge_screen_charge_gradient_is_finite_on_axis(monkeypatch):
    monkeypatch.setattr(Injector, '_reference_particle', lambda self: (0.0, 0.0, 0.0))
    m = injector()
    m.space_charge = SpaceChargeSettings(enabled=True, nr=8, nz=16, cathode_image=False,
                                         dt_adaptive=False, subcycle_steps=2)
    b = beam()

    def objective(scale, checkpoint):
        s = m.track_differentiable(params={'charge_scale': scale}, beam=b, n_steps=80,
                                  device='cpu', dtype=torch.float64, checkpoint_every=checkpoint,
                                  sc_box_m=(0.004, 0.004), screen_z_m=0.006)['screen']
        assert s['crossed'].all()
        return (s['x'] * torch.arange(1, 9, dtype=torch.float64)).sum()

    grads = []
    for checkpoint in (0, 11):
        scale = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
        grad, = torch.autograd.grad(objective(scale, checkpoint), scale)
        fd = (objective(scale.detach() + 1e-3, checkpoint) -
              objective(scale.detach() - 1e-3, checkpoint)) / 2e-3
        assert torch.isfinite(grad)
        torch.testing.assert_close(grad, fd, rtol=0.02, atol=1e-10)
        grads.append(grad)
    torch.testing.assert_close(*grads, rtol=1e-11, atol=1e-14)


@pytest.mark.parametrize('aperture_z,expected_count', [(None, 8), (0.003, 0), (0.007, 8)])
def test_ordinary_screen_survives_inplace_warp_state_and_alive_updates(monkeypatch, aperture_z, expected_count):
    import slipstream.tracking.native.injector as native
    m = injector()
    m.dt_s = 2e-11
    if aperture_z is not None:
        m.apertures = [Aperture('RAD', aperture_z, aperture_z + 1e-5, -0.01)]
    original_step = native._step_with_apertures
    calls = []
    def inplace_step(alive, apertures, *args):
        fresh = original_step(alive, apertures, *args)
        alive.copy_(fresh[-1])
        calls.append(True)
        if args[25]:  # Warp only reuses particle state after emission completes.
            for old, new in zip(args[:12], fresh[:12]):
                old.copy_(new)
            return (*args[:12], alive)
        return (*fresh[:12], alive)
    # Exercise the Warp dispatch without requiring hardware; the fake kernel mutates its inputs.
    for name in ('zeros', 'ones', 'full'):
        original = getattr(torch, name)
        def cpu_allocation(*args, _original=original, **kwargs):
            if kwargs.get('device') == 'cuda':
                kwargs['device'] = 'cpu'
            return _original(*args, **kwargs)
        monkeypatch.setattr(torch, name, cpu_allocation)
    monkeypatch.setattr(torch, 'compile', lambda function, **kwargs: function)
    monkeypatch.setattr(native.warp_kernels, 'available', lambda *args: True)
    monkeypatch.setattr(native, '_step_with_apertures_warp', inplace_step)
    monkeypatch.setattr(native, 'range', lambda n: range(min(n, 200)), raising=False)
    monkeypatch.setattr(Injector, 'initial_beam', lambda *args, **kwargs: beam())
    monkeypatch.setattr(Injector, '_reference_particle', lambda self: (0.0, 0.0, 0.0))
    candidate = m.track(device='cuda', dtype=torch.float64, use_compile=True, verbose=False,
                        record_every=0, screen_z_m=0.006)['screen']
    reference = m.track(device='cpu', dtype=torch.float64, use_compile=False, verbose=False,
                        record_every=0, screen_z_m=0.006)['screen']
    assert calls
    assert candidate['n_crossed'] == expected_count
    assert candidate['n_unobserved_alive'] == 0
    for key in ('x', 'y', 'z', 'ux', 'uy', 'uz', 't_s', 'q', 'crossed'):
        torch.testing.assert_close(candidate[key], reference[key], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA hardware required')
def test_ordinary_cuda_warp_screen_matches_cpu_drift(monkeypatch):
    from slipstream.tracking.native import warp_kernels
    if not warp_kernels.available('cuda', torch.float32):
        pytest.skip('Warp backend unavailable')
    monkeypatch.setattr(Injector, 'initial_beam',
                        lambda self, n, device, dtype, **kwargs: beam(device, dtype))
    monkeypatch.setattr(Injector, '_reference_particle', lambda self: (0.0, 0.0, 0.0))
    m = injector()
    candidate = m.track(device='cuda', dtype=torch.float32, use_compile=True, verbose=False,
                        record_every=0, screen_z_m=0.006)
    reference = m.track(device='cpu', dtype=torch.float64, use_compile=False, verbose=False,
                        record_every=0, screen_z_m=0.006)
    assert candidate['n_steps'] < 200
    assert candidate['screen']['n_crossed'] == 8
    assert candidate['screen']['n_unobserved_alive'] == 0
    for key in ('x', 'y', 'z', 'ux', 'uy', 'uz', 't_s', 'q'):
        torch.testing.assert_close(candidate['screen'][key].cpu().double(), reference['screen'][key],
                                   rtol=5e-6, atol=1e-10 if key in ('x', 'y', 'z') else 1e-20)
