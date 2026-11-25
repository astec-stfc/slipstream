"""Static magnet fields of the native injector: dipole (ASTRA fringe), table quadrupole with multipoles, hard-edge quadrupole
and a tilted solenoid, checked against Maxwell's equations and closed forms.

The dipole is a midplane expansion, so its curl is only second order in the height ``y``; the divergence is exactly zero.
"""
import math

import numpy as np
import pytest
import torch

from slipstream.tracking.native.magnets import (
    Dipole, beam_angle, dipole_element, hard_quad_element, magnet_fields, place_as_tensors, placement, placement3,
    profile_quad_element, rotate_to_beam, solenoid_element,
)

F64 = torch.float64


def field_at(elements, x, y, z):
    point = [torch.tensor([v], dtype=F64) for v in (x, y, z)]
    return tuple(t.item() for t in magnet_fields(*point, elements))


def divergence_and_curl(elements, point, step=1e-6):
    """Central-difference divergence and the largest curl component of the field at ``point``, and the largest gradient."""
    jac = np.zeros((3, 3))
    for j in range(3):
        up, down = list(point), list(point)
        up[j] += step
        down[j] -= step
        jac[:, j] = (np.array(field_at(elements, *up)) - np.array(field_at(elements, *down))) / (2 * step)
    curl = [jac[2, 1] - jac[1, 2], jac[0, 2] - jac[2, 0], jac[1, 0] - jac[0, 1]]
    return np.trace(jac), max(abs(c) for c in curl), np.abs(jac).max()


def rectangular_dipole(half_width=0.3, length=1.0, slant=0.0, gap=0.05, strength=0.5):
    corners = [[-half_width, -slant], [half_width, slant], [-half_width, length - slant], [half_width, length + slant]]
    return Dipole(corners=corners, strength_T=strength, gap1_m=gap, gap2_m=gap)


def test_dipole_field_is_uniform_between_the_edges_and_cut_at_the_sides():
    elements = [dipole_element(rectangular_dipole())]
    assert field_at(elements, 0.0, 0.0, 0.5) == pytest.approx((0.0, 0.5, 0.0), abs=1e-12)
    assert field_at(elements, 0.29, 0.0, 0.5)[1] == pytest.approx(0.5)
    assert field_at(elements, 0.31, 0.0, 0.5) == (0.0, 0.0, 0.0)  # outside the side planes
    # the fringe falls to half at each edge (F = 1/(1+exp(0)) there) and is gone beyond 1.5 gaps
    assert field_at(elements, 0.0, 0.0, 0.0)[1] == pytest.approx(0.25, rel=1e-9)
    assert field_at(elements, 0.0, 0.0, -1.6 * 0.05)[1] == 0.0


@pytest.mark.parametrize("slant", [0.0, 0.2])
def test_dipole_field_integral_is_the_field_times_the_length_whatever_the_edge_angle(slant):
    elements = [dipole_element(rectangular_dipole(slant=slant))]
    z = torch.linspace(-0.4, 1.4, 6001, dtype=F64)
    zero = torch.zeros_like(z)
    _, by, _ = magnet_fields(zero, zero, z, elements)
    assert torch.trapezoid(by, z).item() == pytest.approx(0.5 * 1.0, rel=1e-5)  # the symmetric fringe adds and removes the same area


def test_dipole_field_is_divergence_free_and_its_curl_is_second_order_in_height():
    elements = [dipole_element(rectangular_dipole())]
    curls = []
    for y in (4e-3, 2e-3, 1e-3):
        div, curl, scale = divergence_and_curl(elements, (0.05, y, 0.02))  # inside the entrance fringe
        assert abs(div) < 1e-6 * scale
        curls.append(curl)
    assert curls[0] / curls[1] == pytest.approx(4.0, rel=0.02) and curls[1] / curls[2] == pytest.approx(4.0, rel=0.02)


def test_dipole_midplane_symmetry_vertical_field_even_and_horizontal_and_longitudinal_odd_in_height():
    elements = [dipole_element(rectangular_dipole())]
    above = field_at(elements, 0.05, 0.01, 0.02)
    below = field_at(elements, 0.05, -0.01, 0.02)
    assert above[1] == pytest.approx(below[1], rel=1e-12)
    assert above[0] == pytest.approx(-below[0], rel=1e-12) and above[2] == pytest.approx(-below[2], rel=1e-12)
    assert above[2] != 0.0  # the fringe makes the longitudinal component


def test_flat_top_quadrupole_with_multipoles_is_the_harmonic_field_series():
    z = np.linspace(-0.3, 0.3, 301)
    gradient, c3, c4, offset = 10.0, 50.0, 200.0, 0.01
    element = profile_quad_element(z, np.ones_like(z), gradient, False, {3: c3, 4: c4}, placement(offset, 0.0, 0.0, force=True),
                                   "cpu", F64)
    x, y = 0.012, 0.003
    xl = x - offset  # the magnet is displaced by ``offset`` in x
    # B_y + i B_x = -G (x + i y) - C3 (x + i y)^2 - C4 (x + i y)^3 for an electron-focusing gradient G > 0
    w = complex(xl, y)
    series = -gradient * w - c3 * w**2 - c4 * w**3
    bx, by, bz = field_at([element], x, y, 0.0)
    assert (by, bx, bz) == pytest.approx((series.real, series.imag, 0.0), rel=1e-12)
    assert field_at([element], x, y, 0.4) == (0.0, 0.0, 0.0)  # beyond the table


def test_quadrupole_profile_scaling_peak_gradient_or_raw_profile():
    z = np.linspace(-0.3, 0.3, 301)
    profile = np.exp(-0.5 * (z / 0.08) ** 2) * 3.0
    scaled = profile_quad_element(z, profile, 10.0, False, {}, None, "cpu", F64)
    raw = profile_quad_element(z, profile, 10.0, True, {}, None, "cpu", F64)
    assert field_at([scaled], 0.0, 1e-4, 0.0)[0] == pytest.approx(-10.0 * 1e-4, rel=1e-5)  # peak gradient = Q_grad, B_x = -G y
    assert field_at([raw], 0.0, 1e-4, 0.0)[0] == pytest.approx(-3.0 * 1e-4, rel=1e-5)  # noscale: the table as it is


def test_fringing_quadrupole_obeys_maxwell_and_has_the_longitudinal_component_of_its_gradient_slope():
    z = np.linspace(-0.3, 0.3, 601)
    element = profile_quad_element(z, np.exp(-0.5 * (z / 0.08) ** 2), 10.0, False, {}, None, "cpu", F64)
    for point in [(0.003, 0.002, 0.0), (0.004, -0.002, 0.06), (0.002, 0.003, 0.12)]:
        div, curl, scale = divergence_and_curl([element], point)
        assert abs(div) < 1e-4 * scale and curl < 1e-3 * scale
    x, y, z0 = 0.003, 0.002, 0.06
    g = -10.0 * math.exp(-0.5 * (z0 / 0.08) ** 2)
    dg = g * (-z0 / 0.08**2)
    assert field_at([element], x, y, z0)[2] == pytest.approx(dg * x * y, rel=1e-3)  # B_z = g'(z) x y


def test_hard_edge_quadrupole_is_a_pure_gradient_between_its_ends():
    element = hard_quad_element(10.0, -0.1, 0.1, None)
    assert field_at([element], 0.01, 0.02, 0.0) == pytest.approx((-0.2, -0.1, 0.0), rel=1e-12)  # B = -G (y, x, 0)
    assert field_at([element], 0.01, 0.02, 0.11) == (0.0, 0.0, 0.0)


def test_a_tilted_solenoid_field_points_along_its_own_tilted_axis():
    b0, xrot, yrot, offset = 0.3, 0.01, -0.02, (0.002, -0.001, 0.0)
    place = placement3(*offset, xrot, yrot)
    ntab = 101
    table = torch.full((ntab,), b0, dtype=F64)
    zeros = torch.zeros(ntab, dtype=F64)
    element = solenoid_element(table, zeros, zeros, -0.5, 1.0 / (ntab - 1), ntab, 0.5, place_as_tensors(place, "cpu", F64))
    axis = np.array([math.sin(yrot) * 0 + place[3][2], place[3][5], place[3][8]])  # lab direction of the element's z axis
    assert np.linalg.norm(axis) == pytest.approx(1.0)
    for s in (-0.1, 0.0, 0.15):
        point = np.array(offset) + s * axis
        assert field_at([element], *point) == pytest.approx(tuple(b0 * axis), rel=1e-9, abs=1e-12)
    rotation = np.array(place[3]).reshape(3, 3)
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-14)  # a pure rotation


def test_rotation_into_the_beam_frame_removes_the_bend_angle_and_keeps_lengths():
    angle = 0.3
    ux = torch.tensor([math.sin(angle), 2 * math.sin(angle)], dtype=F64)
    uz = torch.tensor([math.cos(angle), 2 * math.cos(angle)], dtype=F64)
    theta = beam_angle(ux, uz)
    assert theta.item() == pytest.approx(angle, rel=1e-12)
    x, z = torch.tensor([0.1, 0.2], dtype=F64), torch.tensor([1.0, 1.5], dtype=F64)
    xr, zr, uxr, uzr = rotate_to_beam(x, z, ux, uz, theta)
    assert uxr.abs().max().item() < 1e-15 and torch.allclose(uzr, torch.hypot(ux, uz))
    torch.testing.assert_close(torch.hypot(xr, zr), torch.hypot(x, z), rtol=1e-14, atol=0)
