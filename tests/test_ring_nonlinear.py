"""Independent numerical references for native nonlinear ring analysis."""
import math
from dataclasses import replace

import pytest
import torch

from slipstream.analysis import nonlinear as nl
from slipstream.tracking.native.elements import _multipole_kick_increments
from slipstream.tracking.native.lattice_import import ElementSpec, parse_elegant_lattice
from slipstream.tracking.native.ring import RingMap


def fodo(skew=0.0):
    cell = [ElementSpec("QF", "quad", 0.2, k1=1.5), ElementSpec("D", "drift", 1),
            ElementSpec("QD", "quad", 0.2, k1=-1.35), ElementSpec("S", "sext", 0.1, k2=10,
            skew_multipoles=((1, skew),) if skew else ()), ElementSpec("D", "drift", 1)]
    return RingMap(cell * 3, 3e9, device="cpu", dtype=torch.float64, n_kicks=4)


def quad_matrix(k, length):
    if k == 0:
        return torch.tensor([[1., length], [0., 1.]], dtype=torch.float64)
    w = math.sqrt(abs(k))
    c, s = (math.cos(w * length), math.sin(w * length)) if k > 0 else (math.cosh(w * length), math.sinh(w * length))
    return torch.tensor([[c, s / w], [-math.copysign(w, k) * s, c]], dtype=torch.float64)


def test_linear_optics_closed_form_and_gradient():
    ring = fodo()
    opt = nl.linear_optics(ring)
    for sign, tune, beta in ((1, opt.Qx, opt.beta_x), (-1, opt.Qy, opt.beta_y)):
        M = torch.eye(2, dtype=torch.float64)
        for s in ring.specs:
            M = quad_matrix(sign * s.k1, s.length) @ M
        sinmu = torch.sign(M[0, 1]) * torch.sqrt(1 - (M.trace() / 2)**2)
        q = torch.remainder(torch.atan2(sinmu, M.trace() / 2) / (2 * math.pi), 1)
        torch.testing.assert_close(tune, q, atol=2e-9, rtol=0)
        torch.testing.assert_close(beta, M[0, 1] / sinmu, atol=1e-7, rtol=0)
    k = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
    tune = nl.linear_optics(ring.with_strengths(k1={"QF": k}), create_graph=True).Qx
    grad = torch.autograd.grad(tune, k)[0]
    eps = 1e-5
    fd = (nl.linear_optics(ring.with_strengths(k1={"QF": k.detach() + eps})).Qx
          - nl.linear_optics(ring.with_strengths(k1={"QF": k.detach() - eps})).Qx) / (2 * eps)
    torch.testing.assert_close(grad, fd, rtol=1e-7, atol=1e-9)


def test_naff_batched_frequency_and_gradient():
    n = torch.arange(512, dtype=torch.float64)
    q = torch.tensor([.231234, .374321], dtype=torch.float64, requires_grad=True)
    signal = torch.exp(2j * math.pi * q[:, None] * n)
    result = nl.naff(signal)[0]
    torch.testing.assert_close(result, q, rtol=0, atol=1e-9)
    torch.testing.assert_close(torch.autograd.grad(result.sum(), q)[0], torch.ones_like(q), rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("order", [0, 1, 2, 3, 4, 6])
@pytest.mark.parametrize("skew", [False, True])
def test_general_multipole_against_complex_polynomial(order, skew):
    x = torch.tensor([.02, -.03], dtype=torch.float64)
    y = torch.tensor([-.04, .01], dtype=torch.float64)
    strength = torch.tensor(2.3, dtype=torch.float64, requires_grad=True)
    kwargs = {"skew_multipoles" if skew else "multipoles": ((order, strength),)}
    dx, dy = _multipole_kick_increments(x, y, 0., 0., .7, 0., **kwargs)
    field = (1j if skew else 1) * strength * torch.complex(x, y)**order / math.factorial(order)
    torch.testing.assert_close(dx, -.7 * field.real)
    torch.testing.assert_close(dy, .7 * field.imag)
    grad = torch.autograd.grad(dx.sum() + dy.sum(), strength)[0]
    torch.testing.assert_close(grad, (dx.sum() + dy.sum()).detach() / strength.detach())


def test_koct_and_thin_mult_parser_transport(tmp_path):
    path = tmp_path / "multipoles.lte"
    path.write_text("O: KOCT,L=.2,K3=12\nM: MULT,L=0,ORDER=4,KNL=3,FACTOR=2\nR: LINE=(O,M)\n")
    specs = parse_elegant_lattice(path, "R")
    assert specs[0].k3 == 12
    assert specs[1].integrated_multipoles == ((4, 6.),)
    r = RingMap([specs[1]], 3e9, device="cpu")
    u = torch.tensor([[.02, 0., .03, 0., 0., 0.]], dtype=torch.float64)
    actual = r.turn(u)
    field = 6 * torch.complex(u[:, 0], u[:, 2])**4 / math.factorial(4)
    torch.testing.assert_close(actual[:, 1], -field.real)
    torch.testing.assert_close(actual[:, 3], field.imag)
    with pytest.raises(NotImplementedError, match="multipoles"):
        RingMap(specs, 3e9, device="cpu", double_single=True)


def test_octupole_tracking_and_symplectic_jacobian():
    r = RingMap([ElementSpec("O", "oct", .2, k3=3000)], 3e9, device="cpu", n_kicks=8)
    u = torch.tensor([.002, .0001, -.001, -.0002, 0., .01], dtype=torch.float64)
    out = r.turn(u)
    drift = RingMap([ElementSpec("D", "drift", .2)], 3e9, device="cpu").turn(u)
    assert (out - drift).abs().max() > 1e-7
    jac = torch.autograd.functional.jacobian(lambda a: r.turn(a)[:4], u)[:, :4]
    J = torch.tensor([[0., 1, 0, 0], [-1., 0, 0, 0], [0., 0, 0, 1], [0., 0, -1., 0]], dtype=torch.float64)
    torch.testing.assert_close(jac.T @ J @ jac, J, atol=1e-12, rtol=1e-12)


def test_coupled_normal_modes_and_tracked_tunes():
    r = fodo(skew=.2)
    with pytest.raises(ValueError, match="coupled"):
        nl.linear_optics(r)
    opt = nl.coupled_optics(r)
    N = opt.normalizer
    J = N.new_tensor([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]])
    torch.testing.assert_close(N.T @ J @ N, J, atol=1e-11, rtol=1e-11)
    rotation = torch.linalg.solve(N, opt.M[:4, :4] @ N)
    for i, q in enumerate(opt.tunes):
        c, s = torch.cos(2 * math.pi * q), torch.sin(2 * math.pi * q)
        block = torch.stack((torch.stack((c, s)), torch.stack((-s, c))))
        torch.testing.assert_close(rotation[2*i:2*i+2, 2*i:2*i+2], block, atol=1e-11, rtol=1e-11)
    knob = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
    q = nl.coupled_optics(r.with_strengths(k1={"QF": knob}), create_graph=True).tunes[0]
    grad = torch.autograd.grad(q, knob)[0]
    eps = 1e-5
    fd = (nl.coupled_optics(r.with_strengths(k1={"QF": knob.detach()+eps})).tunes[0]
          - nl.coupled_optics(r.with_strengths(k1={"QF": knob.detach()-eps})).tunes[0]) / (2 * eps)
    torch.testing.assert_close(grad, fd, atol=1e-8, rtol=1e-6)
    normal = N.new_tensor([1e-6, 0, 1e-6, 0])
    state = opt.closed_orbit.clone()
    state[:4] += N @ normal
    record = r.track(state, 256, record_every=1).tbt
    signals = nl.coupled_signals(record, opt)
    actual = torch.stack([nl.naff(z)[0] for z in signals])
    torch.testing.assert_close(actual, opt.tunes, atol=1e-7, rtol=0)


def test_per_element_loss_map_and_decimated_memory():
    r = RingMap([ElementSpec("A", "drift", 1), ElementSpec("B", "drift", 2)], 3e9, device="cpu")
    state = torch.zeros(10000, 6, dtype=torch.float64)
    state[0, 1] = .02
    result = r.track_loss_map(state, 4, element_apertures={"B": (.03, .05)}, record_every=2)
    assert result.loss_turn[0] == 0 and result.loss_element[0] == 1
    assert result.loss_s_m[0] == 3
    assert torch.equal(result.state[0], torch.zeros(6, dtype=torch.float64))
    assert (result.loss_turn[1:] == -1).all()
    assert result.tbt.shape == (2, 10000, 4)
    assert result.tbt.numel() * result.tbt.element_size() == 2 * 10000 * 4 * 8


def test_public_ring_map():
    from slipstream import RingMap as PublicRingMap
    assert PublicRingMap is RingMap


def test_chromaticity_gradient_and_strength_ensemble():
    r = fodo()
    k = torch.tensor(10., dtype=torch.float64, requires_grad=True)
    xi = nl.chromaticity(r.with_strengths(k2={"S": k}), create_graph=True)[0]
    grad = torch.autograd.grad(xi, k)[0]
    eps = .01
    fd = (nl.chromaticity(r.with_strengths(k2={"S": k.detach()+eps}))[0]
          - nl.chromaticity(r.with_strengths(k2={"S": k.detach()-eps}))[0]) / (2 * eps)
    torch.testing.assert_close(grad, fd, atol=1e-7, rtol=1e-5)
    strengths = torch.tensor([[1.49], [1.51]], dtype=torch.float64)
    state = torch.zeros(2, 3, 6, dtype=torch.float64)
    state[..., 0] = 1e-3
    actual = r.with_strengths(k1={"QF": strengths}).track(state, 3).state
    expected = torch.stack([r.with_strengths(k1={"QF": v}).track(state[i], 3).state for i, v in enumerate(strengths)])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_da_rectangular_geometry_and_linear_frequency_map():
    drift = RingMap([ElementSpec("D", "drift", 1)], 3e9, device="cpu")
    zero, one = torch.tensor(0., dtype=torch.float64), torch.tensor(1., dtype=torch.float64)
    opt = nl.LinearOptics(torch.zeros(6, dtype=torch.float64), torch.eye(6, dtype=torch.float64), zero, zero, one, zero, one, zero)
    result = nl.dynamic_aperture(drift, .1, 2, n_angles=9, n_r=16, refine=2, aperture=(.03, .02), opt=opt)
    theta = result.angle_rad
    expected = torch.minimum(.03 / torch.cos(theta).abs(), .02 / torch.sin(theta).abs())
    torch.testing.assert_close(result.radius_m, expected, rtol=0, atol=.1 / 16**3)
    r = fodo()
    r = RingMap([replace(s, k2=0.) for s in r.specs], 3e9, device="cpu", n_kicks=4)
    optics = nl.linear_optics(r)
    fmap = nl.frequency_map(r, (1e-6, 2e-6), (1e-6, 2e-6), nx=2, ny=2, n_turns=512, opt=optics)
    torch.testing.assert_close(fmap.Qx, optics.Qx.expand(2, 2), rtol=0, atol=1e-8)
    torch.testing.assert_close(fmap.Qy, optics.Qy.expand(2, 2), rtol=0, atol=1e-8)
    assert (fmap.diffusion < -8).all()


def test_million_particles_without_recording():
    r = RingMap([ElementSpec("D", "drift", .1)], 3e9, device="cpu")
    state = torch.zeros(1_000_000, 6, dtype=torch.float64)
    result = r.track(state, 2)
    assert result.state.shape == state.shape and result.tbt is None
    assert result.state.numel() * result.state.element_size() == 48_000_000
    assert (result.loss_turn == -1).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware required")
@pytest.mark.parametrize("exact_dipole", [True, False])
def test_cuda_warp_turn_matches_cpu(exact_dipole):
    # Field-free drift/sextupole/bend take the one-step path; the rest are sliced.
    specs = [ElementSpec("D", "drift", .5), ElementSpec("QF", "quad", .3, k1=1.5),
             ElementSpec("S", "sext", .1, k2=30.), ElementSpec("SZ", "sext", .1),
             ElementSpec("B", "bend", 1., angle=.2, e1=.1, e2=.05, hgap=.02, edge_order=2),
             ElementSpec("BQ", "bend", 1., angle=.2, k1=-.3, k2=2.),
             ElementSpec("QD", "quad", .3, k1=-1.5, dx=1e-4, tilt=2e-3)]
    cpu = RingMap(specs, 1e9, device="cpu", dtype=torch.float64, exact_dipole=exact_dipole)
    cuda = RingMap(specs, 1e9, device="cuda", dtype=torch.float64, exact_dipole=exact_dipole)
    if cuda._warp_params is None:
        pytest.skip("Warp unavailable")
    state = torch.tensor([[.002, .0001, -.001, -.0002, 0., .01], [-.003, .0002, .0015, .0001, 1e-4, -.005]],
                         dtype=torch.float64)
    torch.testing.assert_close(cuda.turn(state.cuda()).cpu(), cpu.turn(state), atol=1e-14, rtol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware required")
def test_cuda_higher_multipole_matches_cpu():
    specs = [ElementSpec("O", "oct", .2, k3=3000, skew_multipoles=((2, 5.),))]
    cpu = RingMap(specs, 3e9, device="cpu")
    cuda = RingMap(specs, 3e9, device="cuda", dtype=torch.float64)
    assert cuda._warp_params is None
    state = torch.tensor([[.002, .0001, -.001, -.0002, 0., .01]], dtype=torch.float64)
    torch.testing.assert_close(cuda.turn(state.cuda()).cpu(), cpu.turn(state), atol=1e-14, rtol=1e-12)


@pytest.mark.parametrize("kwargs", [
    {"length": .2, "integrated_multipoles": ((3, 1.),)},
    {"length": 0., "k3": 1.},
    {"length": .2, "multipoles": ((-1, 1.),)},
])
def test_invalid_new_multipole_parameters_fail(kwargs):
    s = ElementSpec("M", "multipole", **kwargs)
    r = RingMap([s], 3e9, device="cpu")
    with pytest.raises(ValueError, match="multipole"):
        r.turn(torch.zeros(1, 6, dtype=torch.float64))


def test_footprint_lost_launch_keeps_strength_gradient_finite():
    from slipstream.analysis.nonlinear_opt import Footprint, build_launches
    r = fodo()
    opt = nl.linear_optics(r)
    launches = build_launches({0.: opt}, amps_mm=[1000.], chrom_deltas=[])
    objective = Footprint(launches, opt, (0., 0.), n_turns=32, loss_aperture=.01)
    knob = torch.tensor(10., dtype=torch.float64, requires_grad=True)
    loss = objective.loss(r.with_strengths(k2={"S": knob}))
    grad = torch.autograd.grad(loss, knob)[0]
    assert torch.isfinite(loss) and torch.isfinite(grad)


def test_detuning_fit_against_exact_action_rotation():
    class ActionRotation(RingMap):
        def turn(self, state, eager=False):
            x, px, y, py, tau, delta = state.unbind(-1)
            jx, jy = (x*x + px*px) / 2, (y*y + py*py) / 2
            mux = 2 * math.pi * (.231234 + self.coefficient*jx + 4*jy)
            muy = 2 * math.pi * (.374321 + 4*jx + 5*jy)
            return torch.stack((x*torch.cos(mux)+px*torch.sin(mux),
                                -x*torch.sin(mux)+px*torch.cos(mux),
                                y*torch.cos(muy)+py*torch.sin(muy),
                                -y*torch.sin(muy)+py*torch.cos(muy), tau, delta), -1)
    r = ActionRotation([], 3e9, device="cpu")
    r.coefficient = torch.tensor(3., dtype=torch.float64, requires_grad=True)
    zero, one = torch.tensor(0., dtype=torch.float64), torch.tensor(1., dtype=torch.float64)
    opt = nl.LinearOptics(torch.zeros(6, dtype=torch.float64), torch.eye(6, dtype=torch.float64),
                         torch.tensor(.231234), torch.tensor(.374321), one, zero, one, zero)
    result = nl.detuning(r, .01, .01, n_amp=5, n_turns=256, opt=opt)
    torch.testing.assert_close(torch.stack(result), torch.tensor([3., 4., 4., 5.], dtype=torch.float64), rtol=1e-6, atol=1e-6)
    grad = torch.autograd.grad(result[0], r.coefficient)[0]
    torch.testing.assert_close(grad, torch.ones_like(grad), atol=1e-6, rtol=1e-6)
