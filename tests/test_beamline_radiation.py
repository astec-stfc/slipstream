"""Closed-form covariance integrals and complete geometry/radiation gradients."""
import math

import numpy as np
import pytest
from scipy.linalg import expm
from numpy.polynomial.legendre import leggauss
import torch

from slipstream.analysis.beamline_radiation import linear_radiation_transport
from slipstream.tracking.native.radiation import bend_radiation_rates

DTYPE = torch.float64


def beam():
    return torch.diag(torch.tensor([6e-10, 2e-13, 1e-12, 1e-15, .00083**2], dtype=DTYPE))


def test_exact_diffusion_integral_zero_and_semigroup():
    length, h = 7., .00135/7
    one = lambda x: torch.tensor([x], dtype=DTYPE)
    result = linear_radiation_transport(one(length), one(h*length), one(0.), beam(), energy_eV=5e11)
    a = np.array([[0., 1., 0.], [-h*h, 0., h], [0., 0., 0.]])
    _, rate = bend_radiation_rates(5e11, h)
    nodes, weights = leggauss(80)
    expected = np.zeros((3, 3))
    for u, w in zip((nodes+1)*length/2, weights*length/2):
        response = expm(a*u)[:, 2]
        expected += w*rate*np.outer(response, response)
    indices = torch.tensor([0, 1, 4])
    actual = result.diffusion_covariance[indices[:, None], indices[None, :]].numpy()
    torch.testing.assert_close(result.diffusion_profile[-1],result.diffusion_covariance)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-24)
    assert result.diffusion_covariance[4, 4].item() == pytest.approx(rate*length, rel=2e-13)
    refined = linear_radiation_transport(one(length), one(h*length), one(0.), beam(), energy_eV=5e11, samples_per_element=8)
    torch.testing.assert_close(refined.covariance, result.covariance, rtol=1e-12, atol=1e-24)
    zero = linear_radiation_transport(one(length), one(0.), one(0.), beam(), energy_eV=5e11)
    assert torch.count_nonzero(zero.diffusion_covariance) == 0
    assert zero.offset_m == 0 and zero.longitudinal_m == length
    drift = torch.eye(5, dtype=DTYPE)
    drift[0, 1] = drift[2, 3] = length
    torch.testing.assert_close(zero.covariance, drift@beam()@drift.T, atol=1e-23, rtol=1e-13)
    survey = length/ (h*length) * (1-math.cos(h*length))
    assert result.offset_m.item() == pytest.approx(survey, rel=2e-10)


def test_complete_api_gradients_and_scaling():
    def function(p):
        lengths = torch.stack((2.+p[0], p.new_tensor(3.), p.new_tensor(2.)))
        angles = torch.stack((.001*p[1], p.new_zeros(()), -.001*p[1]))
        strengths = torch.stack((p.new_zeros(()), .04+.01*p[2], p.new_zeros(())))
        result = linear_radiation_transport(lengths, angles, strengths, beam(), energy_eV=5e11,
                                            e1_rad=angles, e2_rad=angles)
        emit = torch.sqrt(torch.linalg.det(result.covariance[:2, :2]))
        return torch.stack((emit/1e-11, result.diffusion_covariance[4, 4]*1e6,
                            result.offset_m, result.transport[0, 4]))
    p = torch.tensor([.3, 1.1, .2], dtype=DTYPE, requires_grad=True)
    ad = torch.autograd.functional.jacobian(function, p).detach()
    for step in (1e-4, 1e-5, 1e-6):
        fd = torch.stack([(function(p.detach()+torch.eye(3, dtype=DTYPE)[i]*step)
                           - function(p.detach()-torch.eye(3, dtype=DTYPE)[i]*step))/(2*step) for i in range(3)], 1)
        torch.testing.assert_close(ad, fd, atol=2e-8, rtol=1e-5)
    length, angle, strength = (torch.tensor([v], dtype=DTYPE) for v in (7., .00135, 0.))
    r1 = linear_radiation_transport(length, angle, strength, beam(), energy_eV=5e11)
    r2 = linear_radiation_transport(length, -2*angle, strength, beam(), energy_eV=1e12)
    assert r2.diffusion_covariance[4, 4]/r1.diffusion_covariance[4, 4] == pytest.approx(256.)
    with pytest.raises(ValueError, match="CPU float64"):
        linear_radiation_transport(length.float(), angle.float(), strength.float(), beam(), energy_eV=5e11)
    with pytest.raises(ValueError, match="positive"):
        linear_radiation_transport(-length, angle, strength, beam(), energy_eV=5e11)
