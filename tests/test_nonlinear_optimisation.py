"""Quantitative Phase-4 regressions, independent spectral and quadratic references."""
import math

import pytest
import torch

from slipstream.analysis.nonlinear import LinearOptics, naff_harmonics, spectral_lines
from slipstream.analysis.nonlinear_opt import action_growth_loss, gradient_free, multi_fidelity


def test_multiple_lines_phase_and_gradient():
    n = torch.arange(1024, dtype=torch.float64)
    q = torch.tensor([0.213, 0.347], dtype=torch.float64)
    amplitude = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    c = torch.stack((torch.ones_like(amplitude).to(torch.complex128), amplitude * torch.exp(torch.tensor(0.7j, dtype=torch.complex128))))
    z = (torch.exp(2j * math.pi * n[:, None] * q) * c).sum(-1)
    frequencies, coefficients = naff_harmonics(z, 2)
    torch.testing.assert_close(frequencies, q, atol=2e-8, rtol=0)
    torch.testing.assert_close(coefficients, c, atol=2e-5, rtol=0)
    fixed = spectral_lines(z, q)
    torch.testing.assert_close(fixed, c, atol=1e-12, rtol=0)
    gradient = torch.autograd.grad(fixed[1].abs().square(), amplitude)[0]
    assert float(gradient) == pytest.approx(0.6, abs=1e-12)


def test_action_growth_matches_closed_form():
    zero, one = torch.tensor(0.), torch.tensor(1.)
    opt = LinearOptics(torch.zeros(6), torch.eye(6), zero, zero, one, zero, one, zero)
    strength = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    amplitude = torch.stack((torch.ones_like(strength), torch.sqrt(1 + strength)))
    tbt = torch.stack((amplitude, amplitude * 0, amplitude * 0, amplitude * 0), -1).unsqueeze(1)
    result = action_growth_loss(tbt, opt)
    expected = (math.log(2)**2 + torch.nn.functional.softplus(strength)**2) / 2
    torch.testing.assert_close(result.squeeze(), expected)
    derivative = torch.autograd.grad(result.sum(), strength)[0]
    assert float(derivative) == pytest.approx(float((torch.nn.functional.softplus(strength) * strength.sigmoid()).detach()), abs=1e-12)


@pytest.mark.parametrize('method', ['Powell', 'Nelder-Mead', 'differential_evolution'])
def test_bounded_baselines_budget_and_best(method):
    result = gradient_free(lambda x: ((x - 0.25) ** 2).sum(), torch.tensor([0.8], dtype=torch.float64),
                           [(-1, 1)], method=method, max_evaluations=100, seed=4)
    assert result.evaluations <= 100
    assert result.loss < 1e-5
    assert -1 <= float(result.theta[0]) <= 1


def test_multi_fidelity_rechecks_different_high_fidelity_target():
    seen = []
    def aperture(x):
        seen.append(float(x[0]))
        return ((x + 0.35) ** 2).sum()
    low, high = multi_fidelity(lambda x: ((x - 0.2) ** 2).sum(), aperture,
                              torch.tensor([0.8], dtype=torch.float64), [(-1, 1)],
                              gradient_steps=10, learning_rate=0.1, max_evaluations=70, method='Powell')
    assert low.loss < 0.02
    assert high.loss < 1e-10
    assert seen[0] == float(low.theta[0])
    assert high.evaluations == len(seen)



def test_joint_constraints_gradient_matches_finite_difference():
    from slipstream.analysis.nonlinear import linear_optics
    from slipstream.analysis.nonlinear_opt import joint_optics_loss
    from slipstream.tracking.native.lattice_import import ElementSpec
    from slipstream.tracking.native.ring import RingMap
    specs = [ElementSpec('QF', 'quad', length=0.2, k1=1.2),
             ElementSpec('D', 'drift', length=1),
             ElementSpec('QD', 'quad', length=0.2, k1=-1.2),
             ElementSpec('D', 'drift', length=1)] * 4
    ring = RingMap(specs, energy_eV=3e9, device="cpu", dtype=torch.float64)
    target = linear_optics(ring)
    knob = torch.tensor(1.21, dtype=torch.float64, requires_grad=True)
    objective = lambda x: joint_optics_loss(ring.with_strengths(k1={'QF': x}), target)
    loss = objective(knob)
    derivative = torch.autograd.grad(loss, knob)[0]
    h = 1e-6
    fd = (objective(knob.detach() + h) - objective(knob.detach() - h)) / (2 * h)
    torch.testing.assert_close(derivative, fd, rtol=1e-5, atol=1e-5)
    assert float(joint_optics_loss(ring, target).detach()) == pytest.approx(0, abs=1e-15)




def test_soft_survival_decreases_for_action_decay():
    zero, one = torch.tensor(0.), torch.tensor(1.)
    opt = LinearOptics(torch.zeros(6), torch.eye(6), zero, zero, one, zero, one, zero)
    def loss(ratio):
        amplitudes = torch.tensor([1., math.sqrt(ratio)], dtype=torch.float64)
        tbt = torch.stack((amplitudes, amplitudes*0, amplitudes*0, amplitudes*0), -1).unsqueeze(1)
        return action_growth_loss(tbt, opt).item()
    assert loss(0.5) < loss(1) < loss(2)


def test_optional_cma_baseline():
    pytest.importorskip('cma')
    result = gradient_free(lambda x: ((x - 0.25) ** 2).sum(), torch.tensor([0.8, -0.4], dtype=torch.float64),
                           [(-1, 1), (-1, 1)], method='CMA-ES', max_evaluations=300, seed=4)
    assert result.evaluations <= 300
    assert result.loss < 1e-3


@pytest.mark.parametrize('theta,bounds', [(torch.tensor(1.), [(-1,1)]), (torch.tensor([1.]), [(1,1)]), (torch.tensor([1.]), [(0,float('inf'))])])
def test_invalid_knob_shapes_and_bounds(theta, bounds):
    with pytest.raises(ValueError):
        gradient_free(lambda x: x.square().sum(), theta, bounds)
    with pytest.raises(ValueError):
        multi_fidelity(lambda x: x.square().sum(), lambda x: x.square().sum(), theta, bounds)



def test_cma_normalizes_anisotropic_bounds():
    pytest.importorskip('cma')
    result = gradient_free(lambda x: ((x / torch.tensor([0.02,40.], dtype=x.dtype) - 0.7)**2).sum(),
                           torch.tensor([0.002,4.], dtype=torch.float64), [(0,0.02),(0,40)],
                           method='CMA-ES', max_evaluations=500, seed=4)
    assert result.loss < 1e-7
