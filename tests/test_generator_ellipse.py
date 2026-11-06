"""The native generator's elliptical cut of a 2D gaussian (``Cut_shape = 'ellipse'``, not an ASTRA key)."""

import dataclasses
import math

import numpy as np
import pytest
import torch

from slipstream.tracking.native.generator import GeneratorInput, generate, read_generator_input
from slipstream.tracking.native.injector import Injector

_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def deck(**kw):
    """A laser-like ``Dist_x = Dist_y = 'g'`` namelist: sigmas and cuts in mm and in sigma."""
    base = GeneratorInput(
        ipart=40001,
        dist_x="g",
        dist_y="g",
        sig_x=0.5,
        sig_y=0.3,
        c_sig_x=1.5,
        c_sig_y=2.0,
        dist_z="p",
        lt=0.002,
        rt=0.0002,
        dist_pz="i",
        le=0.00062,
    )
    return dataclasses.replace(base, **kw)


def standard(beam):
    """x, y [mm] of the tracked particles (not the reference row or the probes)."""
    rows = beam.rows[1:][np.isin(beam.rows[1:, 9], (-1, 5))]
    return rows[:, 0] * 1e3, rows[:, 1] * 1e3


def cut_rms_1d(cut):
    """rms of a unit gaussian cut at ``+-cut`` sigma: ``sqrt(1 - 2 cut phi(cut) / erf(cut / sqrt 2))``."""
    return math.sqrt(
        1.0 - 2.0 * cut * math.exp(-0.5 * cut**2) / math.sqrt(2.0 * math.pi) / math.erf(cut / math.sqrt(2.0))
    )


def ellipse_rms(cut_x, cut_y):
    """rms of a unit 2D gaussian cut on the ellipse of semi-axes ``cut_x``, ``cut_y``: integrate ``r exp(-r^2/2)`` along rays."""
    phi = np.linspace(0.0, 2.0 * np.pi, 400001)
    r2 = 1.0 / (np.cos(phi) ** 2 / cut_x**2 + np.sin(phi) ** 2 / cut_y**2)
    norm = _trapezoid(1.0 - np.exp(-0.5 * r2), phi)
    m4 = 2.0 - (r2 + 2.0) * np.exp(-0.5 * r2)  # int_0^R r^3 exp(-r^2/2) dr
    return tuple(math.sqrt(_trapezoid(w**2 * m4, phi) / norm) for w in (np.cos(phi), np.sin(phi)))


def test_elliptical_cut_fills_the_ellipse_and_leaves_the_rectangle_corners_empty():
    x, y = standard(generate(deck(cut_shape="ellipse")))
    rho = np.hypot(x / (1.5 * 0.5), y / (2.0 * 0.3))  # radius in units of the cut ellipse
    assert rho.max() <= 1.0 and rho.max() > 0.999  # nothing outside the ellipse, and it is filled out to the edge
    assert np.abs(x).max() == pytest.approx(1.5 * 0.5, rel=1e-2) and np.abs(y).max() == pytest.approx(
        2.0 * 0.3, rel=1e-2
    )
    assert abs(x.mean()) < 1e-3 and abs(y.mean()) < 1e-3  # mm: zeroed over all rows, so the probes leave 4e-5 mm
    assert abs(np.corrcoef(x, y)[0, 1]) < 1e-3


def test_rectangle_is_the_default_and_puts_particles_in_the_corners():
    assert GeneratorInput().cut_shape == "rectangle"
    x, y = standard(generate(deck()))
    rho = np.hypot(x / (1.5 * 0.5), y / (2.0 * 0.3))
    assert rho.max() > 1.3 and (rho > 1.0).mean() > 0.03  # corners of the ASTRA rectangle, outside the ellipse
    assert np.abs(x).max() <= 1.5 * 0.5 * (1 + 1e-9) and np.abs(y).max() <= 2.0 * 0.3 * (1 + 1e-9)
    x2, y2 = standard(generate(deck(cut_shape="Rectangle")))
    assert np.array_equal(x, x2) and np.array_equal(y, y2)  # the first letter picks the shape, as for ASTRA's names


def test_elliptical_cut_rms_follows_the_closed_form():
    # the cut ellipse narrows both marginals below the rectangle's 1D cuts: 0.69 and 0.81 of sigma against 0.74 and 0.88
    x, y = standard(generate(deck(cut_shape="ellipse")))
    rms_x, rms_y = ellipse_rms(1.5, 2.0)
    assert x.std() == pytest.approx(0.5 * rms_x, rel=2e-3) and y.std() == pytest.approx(0.3 * rms_y, rel=2e-3)
    xr, yr = standard(generate(deck()))
    assert xr.std() == pytest.approx(0.5 * cut_rms_1d(1.5), rel=2e-3) and yr.std() == pytest.approx(
        0.3 * cut_rms_1d(2.0), rel=2e-3
    )
    assert rms_x < cut_rms_1d(1.5) and rms_y < cut_rms_1d(2.0)


def test_ellipse_of_equal_axes_is_the_circularly_cut_2d_gaussian_point_for_point():
    circle = generate(deck(dist_x="2D Gaussian", sig_x=0.4, c_sig_x=1.7))
    ellipse = generate(deck(sig_x=0.4, sig_y=0.4, c_sig_x=1.7, c_sig_y=1.7, cut_shape="ellipse"))
    assert (
        np.abs(circle.rows[:, :2] - ellipse.rows[:, :2]).max() < 1e-12
    )  # m: the same x, y for every row, probes included


def test_elliptical_cut_with_one_axis_uncut_is_a_strip():
    x, y = standard(generate(deck(cut_shape="ellipse", c_sig_y=0.0)))
    assert np.abs(x).max() <= 1.5 * 0.5 * (1 + 1e-9)  # |x| cut at 1.5 sigma, y free
    assert np.abs(y).max() > 3.0 * 0.3
    assert x.std() == pytest.approx(0.5 * cut_rms_1d(1.5), rel=2e-3)  # x is a 1D gaussian cut at 1.5 sigma
    assert y.std() == pytest.approx(0.3, rel=1e-3)  # an uncut axis is rescaled to sig_y over all rows, as for 'g'


def test_elliptical_cut_without_any_cut_is_an_uncut_2d_gaussian():
    x, y = standard(generate(deck(cut_shape="ellipse", c_sig_x=0.0, c_sig_y=0.0)))
    assert (x.std(), y.std()) == pytest.approx(
        (0.5, 0.3), rel=1e-3
    )  # rescaled to sig_x, sig_y over all rows, as for 'g'
    assert abs(np.corrcoef(x, y)[0, 1]) < 1e-2


def test_cut_shape_needs_gaussians_and_a_known_name():
    with pytest.raises(ValueError, match="Dist_x and Dist_y must both be 'g'"):
        generate(deck(cut_shape="ellipse", dist_y="uniform"))
    with pytest.raises(ValueError, match="Dist_x and Dist_y must both be 'g'"):
        generate(deck(cut_shape="ellipse", dist_x="radial"))
    with pytest.raises(ValueError, match="unknown Cut_shape 'triangle'"):
        generate(deck(cut_shape="triangle"))


def test_cut_shape_is_a_namelist_key_and_an_injector_override(tmp_path):
    path = tmp_path / "generator.in"
    path.write_text(
        "&INPUT\n  Ipart = 6001, Dist_x = 'g', sig_x = 0.5, C_sig_x = 1.5, Dist_y = 'g', sig_y = 0.3, C_sig_y = 2.0,\n"
        "  Cathode = .true., Dist_z = 'p', Lt = 0.002, rt = 0.0002, Dist_pz = 'i', LE = 0.00062,\n"
        "  Cut_Shape = 'Ellipse'   ! native key, any case\n/\n"
    )
    assert read_generator_input(path)[0].cut_shape == "Ellipse"
    for overrides in ({}, {"cut_shape": "ellipse"}):
        machine = Injector("beam.astra", 0.1, generator=path.name, base_dir=tmp_path, generator_overrides=overrides)
        beam = machine.initial_beam(4000, "cpu", torch.float64)
        rho = torch.hypot(beam["x"] * 1e3 / (1.5 * 0.5), beam["y"] * 1e3 / (2.0 * 0.3))
        assert beam["x"].numel() == 4000 and float(rho.max()) <= 1.0
    machine.generator_overrides = {"cut_shape": "rectangle"}
    beam = machine.initial_beam(4000, "cpu", torch.float64)
    assert float(torch.hypot(beam["x"] * 1e3 / (1.5 * 0.5), beam["y"] * 1e3 / (2.0 * 0.3)).max()) > 1.2
