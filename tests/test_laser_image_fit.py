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
    """A frame of a known cut gaussian (x along the columns, y down the rows, mm from the sensor centre)."""
    ny, nx = shape
    x_mm = ((np.arange(nx) + 0.5) - 0.5 * nx) / script["PX_PER_MM"]
    y_mm = ((np.arange(ny) + 0.5) - 0.5 * ny) / script["PX_PER_MM"]
    frame = dark + script["cut_gaussian"](params, x_mm[None, :], y_mm[:, None])
    frame = frame + np.random.default_rng(seed).normal(0.0, 3.0, shape)
    for row, col in hot:
        frame[row, col] = 3000.0
    return frame


def test_fit_recovers_the_cut_the_widths_and_the_axes_of_a_noisy_frame_with_hot_pixels(script):
    truth = (900.0, -0.40, 0.20, 0.55, 0.34, 1.5, 2.3, 0.08)  # amplitude, centre, sigmas, n-sigma cuts, edge blur
    hot = [(60, 70), (400, 480), (390, 520), (700, 900)]  # two inside the spot, two in the dark frame
    result, _ = script["fit_image"](camera_frame(script, truth, hot=hot), binning=4)
    assert result["pixels_repaired"] == len(hot)
    assert result["dark_counts"] == pytest.approx(97.0, abs=0.5)
    assert (result["x0_mm"], result["y0_mm"]) == pytest.approx((-0.40, 0.20), abs=0.005)
    assert result["sig_x_mm"] == pytest.approx(0.55, rel=0.02)  # x is the longer axis: rows and columns are not swapped
    assert result["sig_y_mm"] == pytest.approx(0.34, rel=0.02)
    assert result["C_sig_x"] == pytest.approx(1.5, abs=0.05)
    assert result["C_sig_y"] == pytest.approx(2.3, abs=0.1)
    assert result["blur_mm"] == pytest.approx(0.08, abs=0.01)
    assert result["half_width_x_mm"] == pytest.approx(1.5 * 0.55, abs=0.01)
    assert result["rms_residual_counts"] < 4.0  # the noise is 3 counts rms
    assert result["error"]["sig_x_mm"] < 0.02  # the jackknife of a clean frame is small


def test_cut_rms_is_the_rms_of_the_generators_cut_gaussian(script):
    # The fitted C_sig and sig are only the generator.in values if both use the same cut: sample its 'g' distribution.
    inp = read_generator_input(EXAMPLE.with_name("generator.in"))[0]
    inp = dataclasses.replace(inp, ipart=40001, sig_x=0.5, c_sig_x=1.2, sig_y=0.3, c_sig_y=2.0)
    beam = generate(inp)
    moments = beam.moments()
    assert moments["sig_x_mm"] == pytest.approx(float(script["cut_rms"](0.5, 1.2)), rel=2e-3)
    assert moments["sig_y_mm"] == pytest.approx(float(script["cut_rms"](0.3, 2.0)), rel=2e-3)
    standard = beam.rows[1:][np.isin(beam.rows[1:, 9], (-1, 5))]  # not the probes, which sit at 0.5, 1 and 1.5 sigma
    assert float(np.abs(standard[:, 0]).max()) * 1e3 == pytest.approx(1.2 * 0.5, rel=1e-3)  # the cut is at C_sig sigma
    assert float(np.abs(standard[:, 1]).max()) * 1e3 == pytest.approx(2.0 * 0.3, rel=1e-3)
    assert float(script["cut_rms"](1.0, 20.0)) == pytest.approx(1.0, abs=1e-12)  # no cut
    # closed form at 1 sigma: sqrt(1 - 2 phi(1) / erf(1 / sqrt(2)))
    assert float(script["cut_rms"](1.0, 1.0)) == pytest.approx(0.5395, abs=1e-4)


@pytest.mark.skipif(not IMAGE.is_file(), reason="the virtual-cathode capture is not in this checkout")
def test_measured_capture_gives_the_spot_in_generator_in(script):
    result, _ = script["fit_image"](script["load"](IMAGE), binning=4)
    assert (result["x0_mm"], result["y0_mm"]) == pytest.approx((-1.09, 0.44), abs=0.05)
    assert 0.50 < result["sig_x_mm"] < 0.62 and 0.30 < result["sig_y_mm"] < 0.38
    assert 0.84 < result["half_width_x_mm"] < 0.91 and 0.74 < result["half_width_y_mm"] < 0.86  # the edge of the spot
    assert 1.3 < result["C_sig_x"] < 1.8 and result["C_sig_y"] > 1.9  # clipped hard in x, at about 2 sigma in y
    assert 0.06 < result["blur_mm"] < 0.14
    # an ellipse fits the round edge best, the cut gaussian next, an uncut gaussian worst
    residual = (
        result["aperture"]["rms_residual_counts"],
        result["rms_residual_counts"],
        result["uncut_rms_residual_counts"],
    )
    assert residual[0] < residual[1] < residual[2]
