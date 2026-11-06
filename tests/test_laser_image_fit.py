"""The virtual-cathode laser image fit of examples/photocathode/fit_laser_image.py and the generator.in it feeds."""

import dataclasses
import runpy
from pathlib import Path

import numpy as np
import pytest

from slipstream.tracking.native.generator import generate, read_generator_input

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/photocathode/fit_laser_image.py"
IMAGE = EXAMPLE.with_name("CLA-VCA-DIA-CAM-01_2022-04-29_10-14-01UTC_1_images.hdf5")


@pytest.fixture(scope="module")
def script():
    return runpy.run_path(str(EXAMPLE))


def camera_frame(script, params, shape=(800, 1000), dark=97.0, hot=(), seed=1):
    """A frame of a known gaussian cut on an ellipse (x along the columns, y down the rows, mm from the sensor centre)."""
    ny, nx = shape
    x_mm = ((np.arange(nx) + 0.5) - 0.5 * nx) / script["PX_PER_MM"]
    y_mm = ((np.arange(ny) + 0.5) - 0.5 * ny) / script["PX_PER_MM"]
    frame = dark + script["ellipse_gaussian"](params, x_mm[None, :], y_mm[:, None])
    frame = frame + np.random.default_rng(seed).normal(0.0, 3.0, shape)
    for row, col in hot:
        frame[row, col] = 3000.0
    return frame


def test_fit_recovers_the_cut_the_widths_and_the_axes_of_a_noisy_frame_with_hot_pixels(script):
    truth = (900.0, -0.40, 0.20, 0.55, 0.34, 1.5, 2.3, 0.06)  # amplitude, centre, sigmas, n-sigma cuts, edge blur
    hot = [(60, 70), (400, 480), (390, 520), (700, 900)]  # two inside the spot, two in the dark frame
    result, _ = script["fit_image"](camera_frame(script, truth, hot=hot), binning=4)
    assert result["pixels_repaired"] == len(hot)
    assert result["dark_counts"] == pytest.approx(97.0, abs=0.5)
    assert (result["x0_mm"], result["y0_mm"]) == pytest.approx((-0.40, 0.20), abs=0.005)
    assert result["sig_x_mm"] == pytest.approx(0.55, rel=0.02)  # x is the longer axis: rows and columns are not swapped
    assert result["sig_y_mm"] == pytest.approx(0.34, rel=0.02)
    assert result["C_sig_x"] == pytest.approx(1.5, abs=0.05)
    assert result["C_sig_y"] == pytest.approx(2.3, abs=0.1)
    assert result["blur_mm"] == pytest.approx(0.06, abs=0.01)
    assert result["half_width_x_mm"] == pytest.approx(1.5 * 0.55, abs=0.01)
    assert result["rms_residual_counts"] < 4.0  # the noise is 3 counts rms
    assert result["error"]["sig_x_mm"] < 0.02  # the jackknife of a clean frame is small
    # the ellipse is the truth, so the ASTRA rectangle and an uncut gaussian fit worse
    assert result["rectangle"]["rms_residual_counts"] > 2.0 * result["rms_residual_counts"]
    assert result["uncut"]["rms_residual_counts"] > result["rectangle"]["rms_residual_counts"]


def test_fitted_rms_sizes_are_those_of_the_generators_cuts(script):
    # The fitted sig and C_sig are the generator.in values only if the closed forms here are what the generator samples.
    inp = read_generator_input(EXAMPLE.with_name("generator.in"))[0]
    for shape, closed_form in (
        ("ellipse", script["ellipse_rms"](0.5, 0.3, 1.2, 2.0)),
        ("rectangle", (script["rectangle_rms"](0.5, 1.2), script["rectangle_rms"](0.3, 2.0))),
    ):
        beam = generate(
            dataclasses.replace(inp, ipart=40001, sig_x=0.5, c_sig_x=1.2, sig_y=0.3, c_sig_y=2.0, cut_shape=shape)
        )
        moments = beam.moments()
        assert (moments["sig_x_mm"], moments["sig_y_mm"]) == pytest.approx(closed_form, rel=2e-3)
    assert script["rectangle_rms"](1.0, 20.0) == pytest.approx(1.0, abs=1e-12)  # no cut
    assert script["rectangle_rms"](1.0, 1.0) == pytest.approx(0.5395, abs=1e-4)  # sqrt(1 - 2 phi(1) / erf(1 / sqrt 2))
    # a circle cut at 1 sigma: sqrt((2 - 3 exp(-1/2)) / (2 (1 - exp(-1/2))))
    assert script["ellipse_rms"](1.0, 1.0, 1.0, 1.0) == pytest.approx((0.47880, 0.47880), abs=1e-5)
    assert script["ellipse_rms"](0.5, 0.3, 30.0, 30.0) == pytest.approx((0.5, 0.3), rel=1e-9)  # no cut


def test_generator_lines_are_the_fitted_values(script):
    result = dict(
        sig_x_mm=0.5641,
        sig_y_mm=0.3505,
        C_sig_x=1.604,
        C_sig_y=2.373,
        half_width_x_mm=0.9049,
        half_width_y_mm=0.8317,
        rectangle=dict(sig_x_mm=0.5646, sig_y_mm=0.3436, C_sig_x=1.5505, C_sig_y=2.3205),
    )
    lines = script["generator_lines"](result)
    assert (
        "Dist_x = 'g', sig_x = 0.564, C_sig_x = 1.60," in lines
        and "Dist_y = 'g', sig_y = 0.350, C_sig_y = 2.37," in lines
    )
    assert "Cut_shape = 'ellipse'" in lines
    assert "sig_x = 0.565, C_sig_x = 1.55," in lines.split("ASTRA run")[1]  # the rectangle's own fit, without Cut_shape


@pytest.mark.skipif(not IMAGE.is_file(), reason="the virtual-cathode capture is not in this checkout")
def test_measured_capture_gives_the_spot_in_generator_in(script):
    result, _ = script["fit_image"](script["load"](IMAGE), binning=4)
    assert (result["x0_mm"], result["y0_mm"]) == pytest.approx((-1.09, 0.44), abs=0.05)
    assert 0.52 < result["sig_x_mm"] < 0.61 and 0.32 < result["sig_y_mm"] < 0.38
    assert 0.87 < result["half_width_x_mm"] < 0.94 and 0.80 < result["half_width_y_mm"] < 0.87  # the edge of the spot
    assert 1.4 < result["C_sig_x"] < 1.8 and 2.1 < result["C_sig_y"] < 2.7  # clipped at about 1.6 and 2.4 sigma
    assert 0.03 < result["blur_mm"] < 0.10
    # the round edge: the ellipse fits better than ASTRA's rectangle, which fits better than no cut
    assert (
        result["rms_residual_counts"]
        < result["rectangle"]["rms_residual_counts"]
        < result["uncut"]["rms_residual_counts"]
    )
    assert result["generator_rms_mm"] == pytest.approx(result["image_rms_mm"], rel=0.03)  # the rms sizes of the image
