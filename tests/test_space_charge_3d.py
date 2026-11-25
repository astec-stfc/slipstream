"""The Cartesian 3D space-charge solver against closed-form fields.

A uniform ball (linear inside, Coulomb outside), a moving point charge (transverse field enhanced by gamma,
longitudinal field reduced by 1/gamma^2, B = beta z x E / c) and a point charge above the cathode (its image).
Probes are zero-charge particles, so they sample the field without adding to it.
"""
import math

import pytest
import torch

from slipstream.constants import C_LIGHT, EPS0_F_PER_M
from slipstream.tracking.native.injector import SpaceChargeSettings, _compute_space_charge_3d
from slipstream.tracking.native.space_charge import SpaceCharge3D

K = 1.0 / (4.0 * math.pi * EPS0_F_PER_M)
F64 = torch.float64
CHARGE_C = 1e-12


def test_uniform_ball_is_linear_inside_and_coulomb_outside_on_an_anisotropic_grid():
    radius, nodes = 1e-3, 33
    lattice = torch.arange(-6, 7, dtype=F64) * (radius / 6)
    gx, gy, gz = torch.meshgrid(lattice, lattice, lattice, indexing="ij")
    inside = gx**2 + gy**2 + gz**2 <= radius**2
    x, y, z = gx[inside], gy[inside], gz[inside]
    q = torch.full_like(x, CHARGE_C / x.numel())
    h = (radius / 5, radius / 6, radius / 7)  # different cells per axis: a swapped axis cannot pass
    solver = SpaceCharge3D(nodes, nodes, nodes, *h, *(-(nodes - 1) / 2 * hi for hi in h), device="cpu", dtype=F64)
    field = solver.solve(solver.deposit(x, y, z, q))
    for axis in range(3):
        for r in (0.5, 1.5):  # in units of the ball radius
            probe = [torch.zeros(1, dtype=F64) for _ in range(3)]
            probe[axis] = torch.tensor([r * radius], dtype=F64)
            exact = K * CHARGE_C * (r / radius**2 if r < 1 else 1 / (r * radius) ** 2)
            assert solver.gather(field[axis], *probe).item() == pytest.approx(exact, rel=0.03)


def point_charge_fields(gamma, rho, z0=0.0, cathode_image=False, cathode_z=0.0, nodes=32):
    """Fields at six probes around a point charge moving at ``gamma``; probes sit a rest-frame distance ``rho`` away
    on every axis, so the stretched box is a cube whatever ``gamma`` is. Probe 0 is the charge itself."""
    probes = torch.tensor([(0, 0, 0), (rho, 0, 0), (-rho, 0, 0), (0, rho, 0), (0, -rho, 0),
                           (0, 0, rho / gamma), (0, 0, -rho / gamma)], dtype=F64)
    q = torch.zeros(7, dtype=F64)
    q[0] = CHARGE_C
    zero = torch.zeros(7, dtype=F64)
    uz = torch.full((7,), C_LIGHT * math.sqrt(gamma**2 - 1), dtype=F64)
    settings = SpaceChargeSettings(geometry="3d", cathode_image=cathode_image, n_xyz=[nodes] * 3)
    return _compute_space_charge_3d(probes[:, 0], probes[:, 1], probes[:, 2] + z0, q, zero, zero, uz,
                                    "cpu", F64, {}, settings, cathode_z)


@pytest.mark.parametrize("gamma", [1.5, 3.0, 10.0])
def test_moving_point_charge_has_gamma_enhanced_transverse_and_gamma_squared_reduced_longitudinal_field(gamma):
    rho = 10e-3
    beta = math.sqrt(1 - 1 / gamma**2)
    coulomb = K * CHARGE_C / rho**2
    Ex, Ey, Ez, Bx, By, Bz = point_charge_fields(gamma, rho)
    for probe, sign in ((1, 1), (2, -1)):
        assert Ex[probe].item() == pytest.approx(sign * gamma * coulomb, rel=0.01)
    for probe, sign in ((3, 1), (4, -1)):
        assert Ey[probe].item() == pytest.approx(sign * gamma * coulomb, rel=0.01)
    for probe, sign in ((5, 1), (6, -1)):  # on the axis, ahead of and behind the charge: (gamma d)^2 = rho^2
        assert Ez[probe].item() == pytest.approx(sign * coulomb, rel=0.01)
    # B = beta zhat x E / c at every probe, and no longitudinal B
    torch.testing.assert_close(By, beta / C_LIGHT * Ex, rtol=1e-12, atol=0)
    torch.testing.assert_close(Bx, -beta / C_LIGHT * Ey, rtol=1e-12, atol=0)
    assert not Bz.any()


def test_cathode_image_charge_matches_the_closed_form():
    h = 5e-3  # the charge sits h above the cathode and the image -h below it
    Ex, Ey, Ez, *_ = point_charge_fields(1.0 + 1e-9, h, z0=h, cathode_image=True)
    coulomb = K * CHARGE_C / h**2
    # on the cathode under the charge, both fields add: -2 kQ / h^2 (probe 6 is at z = h - rho = 0)
    assert Ez[6].item() == pytest.approx(-2 * coulomb, rel=0.02)
    # on the axis at 2h: charge at distance h, image (opposite sign) at 3h
    assert Ez[5].item() == pytest.approx(coulomb * (1 - 1 / 9), rel=0.02)
    # beside the charge at (h, 0, h): charge at h, image at sqrt(5) h
    assert Ex[1].item() == pytest.approx(coulomb * (1 - 5 ** -1.5), rel=0.02)
