"""Fit the CLARA virtual-cathode laser image to a cut Gaussian and print the matching ``generator.in`` lines.

``CLA-VCA-DIA-CAM-01_2022-04-29_10-14-01UTC_1_images.hdf5`` holds one PCO.Edge frame (dataset ``Capture000001``,
188.5 px/mm, x along the columns and y along the rows) of the laser at the virtual cathode. The laser is a Gaussian clipped
by an aperture, so the model is a 2D Gaussian of rms ``sig_x``, ``sig_y`` cut on the ellipse of semi-axes ``C_sig_x sig_x``
and ``C_sig_y sig_y`` (``C_sig`` is the n-sigma cut): ``Dist_x = Dist_y = 'g'`` with the native ``Cut_shape = 'ellipse'`` of
the generator. The image edge is blurred by the imaging resolution and the diffraction of the aperture; the blur is fitted and
dropped, as the generator's cut is sharp. Only the shape matters: the charge of a photocathode run comes from the laser
energy and quantum efficiency.

Two comparisons are fitted to the same window: ASTRA's own ``'g'`` cuts, which are independent and so a rectangle (the lines
for an ASTRA run), and a Gaussian with no cut. Uncertainties are a block jackknife (the image is split into 4 x 4 blocks,
each refitted without one), which includes the hot spots in the laser profile.

    python examples/photocathode/fit_laser_image.py [--image capture.hdf5] [--no-plots]
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.special import erfc

HERE = Path(__file__).resolve().parent
IMAGE = HERE / "CLA-VCA-DIA-CAM-01_2022-04-29_10-14-01UTC_1_images.hdf5"
PX_PER_MM = 188.5
PARAMETERS = ("amplitude", "x0_mm", "y0_mm", "sig_x_mm", "sig_y_mm", "C_sig_x", "C_sig_y", "blur_mm")
LOWER = (0.0, -np.inf, -np.inf, 0.02, 0.02, 0.3, 0.3, 0.005)
UPPER = (np.inf, np.inf, np.inf, 5.0, 5.0, 8.0, 8.0, 0.5)


def load(path, dataset="Capture000001"):
    with h5py.File(path, "r") as f:
        return f[dataset][...].astype(np.float64)


def remove_dark(raw, border=80):
    """Subtract the dark level (median of the outer frame) and replace single-pixel outliers by their 5 x 5 median."""
    frame = np.concatenate(
        [raw[:border].ravel(), raw[-border:].ravel(), raw[:, :border].ravel(), raw[:, -border:].ravel()]
    )
    dark = np.median(frame)
    noise = 1.4826 * np.median(np.abs(frame - dark))
    median = ndimage.median_filter(raw, size=5)
    bad = np.abs(raw - median) > np.maximum(8.0 * noise, 0.25 * np.abs(median - dark))
    return np.where(bad, median, raw) - dark, float(dark), int(bad.sum())


def bin_image(img, n):
    ny, nx = (s // n * n for s in img.shape)
    return img[:ny, :nx].reshape(ny // n, n, nx // n, n).mean(axis=(1, 3))


def _edge(inside_mm, blur_mm):
    """Smooth step across an edge: 1 inside, 0 outside, ``blur_mm`` rms wide."""
    return 0.5 * erfc(-inside_mm / (np.sqrt(2.0) * blur_mm))


def ellipse_gaussian(p, x_mm, y_mm):
    """``Dist_x = Dist_y = 'g'`` with ``Cut_shape = 'ellipse'``. ``x_mm`` is a row, ``y_mm`` a column."""
    amp, x0, y0, sx, sy, cx, cy, blur = p
    u, v = x_mm - x0, y_mm - y0
    a, b = cx * sx, cy * sy
    rho = np.hypot(u / a, v / b)
    grad = np.hypot(u / a**2, v / b**2) / np.maximum(rho, 1e-9)
    inside = (1.0 - rho) / np.maximum(grad, 1e-9)  # first-order distance to the edge
    return amp * np.exp(-0.5 * ((u / sx) ** 2 + (v / sy) ** 2)) * _edge(inside, blur)


def rectangle_gaussian(p, x_mm, y_mm):
    """ASTRA's ``Dist_x = Dist_y = 'g'``: independent cuts at ``C_sig`` sigma, a rectangle."""
    amp, x0, y0, sx, sy, cx, cy, blur = p
    u, v = x_mm - x0, y_mm - y0
    return (
        amp
        * np.exp(-0.5 * (u / sx) ** 2)
        * _edge(cx * sx - np.abs(u), blur)
        * np.exp(-0.5 * (v / sy) ** 2)
        * _edge(cy * sy - np.abs(v), blur)
    )


def uncut_gaussian(p, x_mm, y_mm):
    return rectangle_gaussian((*p, 1e3, 1e3, 0.5), x_mm, y_mm)


def rectangle_rms(sigma, cut):
    """rms of a Gaussian of rms ``sigma`` cut at ``+-cut sigma``: what the generator's ``'g'`` sampler gives."""
    kept = 1.0 - erfc(cut / np.sqrt(2.0))
    return float(sigma * np.sqrt(1.0 - 2.0 * cut * np.exp(-0.5 * cut**2) / np.sqrt(2.0 * np.pi) / kept))


def ellipse_rms(sig_x, sig_y, cut_x, cut_y):
    """rms sizes of the generator's elliptical cut. Along a ray at angle ``phi`` the density is ``r exp(-r^2/2)`` out to
    ``R(phi)``, so the second moments are integrals of ``2 - (R^2 + 2) exp(-R^2/2)`` over ``phi`` (a full period of a smooth
    function, so the mean of equally spaced points is exact to rounding)."""
    phi = np.linspace(0.0, 2.0 * np.pi, 4096, endpoint=False)
    r2 = 1.0 / (np.cos(phi) ** 2 / cut_x**2 + np.sin(phi) ** 2 / cut_y**2)
    m4 = 2.0 - (r2 + 2.0) * np.exp(-0.5 * r2)
    norm = np.mean(1.0 - np.exp(-0.5 * r2))
    return (
        float(sig_x * np.sqrt(np.mean(np.cos(phi) ** 2 * m4) / norm)),
        float(sig_y * np.sqrt(np.mean(np.sin(phi) ** 2 * m4) / norm)),
    )


def _fit(model, start, lower, upper, x_mm, y_mm, img, weights=1.0):
    return least_squares(
        lambda p: (weights * (model(p, x_mm, y_mm) - img)).ravel(), start, bounds=(lower, upper), x_scale="jac"
    )


def _jackknife(p, x_mm, y_mm, img, blocks=4):
    """Standard error of the elliptical-cut parameters from refits that leave out one block of the window at a time."""
    fits = []
    for rows in np.array_split(np.arange(img.shape[0]), blocks):
        for cols in np.array_split(np.arange(img.shape[1]), blocks):
            weights = np.ones(img.shape)
            weights[np.ix_(rows, cols)] = 0.0
            fits.append(_fit(ellipse_gaussian, p, LOWER, UPPER, x_mm, y_mm, img, weights).x)
    fits = np.array(fits)
    return np.sqrt((len(fits) - 1) / len(fits) * ((fits - fits.mean(0)) ** 2).sum(0))


def fit_image(raw, px_per_mm=PX_PER_MM, binning=2):
    """Fit a raw camera frame (rows y, columns x). Lengths are in mm from the sensor centre.

    Returns the result and the ``(x_mm, y_mm, counts)`` window that was fitted (``x_mm`` a row, ``y_mm`` a column).
    """
    img, dark, repaired = remove_dark(raw)
    ny, nx = img.shape
    img = bin_image(img, binning)
    x_mm = ((np.arange(img.shape[1]) + 0.5) * binning - 0.5 * nx) / px_per_mm
    y_mm = ((np.arange(img.shape[0]) + 0.5) * binning - 0.5 * ny) / px_per_mm
    smooth = ndimage.gaussian_filter(img, 6.0 / binning)
    on = smooth > 0.08 * smooth.max()
    flux = np.where(on, img, 0.0)
    x0, y0 = (x_mm * flux.sum(0)).sum() / flux.sum(), (y_mm * flux.sum(1)).sum() / flux.sum()
    half_x, half_y = np.ptp(x_mm[on.any(0)]) / 2, np.ptp(y_mm[on.any(1)]) / 2
    keep_x, keep_y = np.abs(x_mm - x0) < 1.8 * half_x, np.abs(y_mm - y0) < 1.8 * half_y  # the spot and a margin of dark
    x_mm, y_mm, img = x_mm[keep_x][None, :], y_mm[keep_y][:, None], img[np.ix_(keep_y, keep_x)]

    p = _fit(ellipse_gaussian, (smooth.max(), x0, y0, 0.55, 0.35, 1.6, 2.3, 0.08), LOWER, UPPER, x_mm, y_mm, img).x
    rect = _fit(rectangle_gaussian, p, LOWER, UPPER, x_mm, y_mm, img).x
    uncut = _fit(uncut_gaussian, p[:5], LOWER[:5], UPPER[:5], x_mm, y_mm, img).x

    reach = (p[5] * p[3] + 3.0 * p[7], p[6] * p[4] + 3.0 * p[7])  # the cut plus three edge widths
    near = np.where((np.abs(x_mm - p[1]) < reach[0]) & (np.abs(y_mm - p[2]) < reach[1]), img, 0.0)
    mx, my = (x_mm * near).sum() / near.sum(), (y_mm * near).sum() / near.sum()

    def rms_residual(model, q):
        return float(np.sqrt(np.mean((model(q, x_mm, y_mm) - img) ** 2)))

    result = dict(
        px_per_mm=px_per_mm,
        binning=binning,
        dark_counts=dark,
        pixels_repaired=repaired,
        **dict(zip(PARAMETERS, map(float, p))),
        error=dict(zip(PARAMETERS, map(float, _jackknife(p, x_mm, y_mm, img)))),
        half_width_x_mm=float(p[5] * p[3]),
        half_width_y_mm=float(p[6] * p[4]),
        rms_residual_counts=rms_residual(ellipse_gaussian, p),
        generator_rms_mm=list(ellipse_rms(p[3], p[4], p[5], p[6])),
        image_rms_mm=[
            float(np.sqrt(((x_mm - mx) ** 2 * near).sum() / near.sum())),
            float(np.sqrt(((y_mm - my) ** 2 * near).sum() / near.sum())),
        ],
        rectangle=dict(
            **dict(zip(PARAMETERS, map(float, rect))),
            half_width_x_mm=float(rect[5] * rect[3]),
            half_width_y_mm=float(rect[6] * rect[4]),
            rms_residual_counts=rms_residual(rectangle_gaussian, rect),
            generator_rms_mm=[rectangle_rms(rect[3], rect[5]), rectangle_rms(rect[4], rect[6])],
        ),
        uncut=dict(
            sig_x_mm=float(uncut[3]), sig_y_mm=float(uncut[4]), rms_residual_counts=rms_residual(uncut_gaussian, uncut)
        ),
    )
    return result, (x_mm, y_mm, img)


def generator_lines(r):
    """The ``generator.in`` lines of a fit: ``r`` is the elliptical cut, ``r["rectangle"]`` ASTRA's."""
    rect = r["rectangle"]
    return (
        f"  Dist_x = 'g', sig_x = {r['sig_x_mm']:.3f}, C_sig_x = {r['C_sig_x']:.2f},"
        f"   ! mm, cut at {r['C_sig_x']:.2f} sigma = +-{r['half_width_x_mm']:.3f} mm\n"
        f"  Dist_y = 'g', sig_y = {r['sig_y_mm']:.3f}, C_sig_y = {r['C_sig_y']:.2f},"
        f"   ! mm, cut at {r['C_sig_y']:.2f} sigma = +-{r['half_width_y_mm']:.3f} mm\n"
        "  Cut_shape = 'ellipse',   ! native key: the cut is the ellipse with these semi-axes\n"
        "for an ASTRA run drop Cut_shape and use the rectangle's:\n"
        f"  Dist_x = 'g', sig_x = {rect['sig_x_mm']:.3f}, C_sig_x = {rect['C_sig_x']:.2f},\n"
        f"  Dist_y = 'g', sig_y = {rect['sig_y_mm']:.3f}, C_sig_y = {rect['C_sig_y']:.2f},"
    )


def plot(r, window, path):
    import matplotlib.pyplot as plt

    x_mm, y_mm, img = window
    p, rect = [r[k] for k in PARAMETERS], [r["rectangle"][k] for k in PARAMETERS]
    ellipse, boxed = ellipse_gaussian(p, x_mm, y_mm), rectangle_gaussian(rect, x_mm, y_mm)
    extent = (x_mm[0, 0], x_mm[0, -1], y_mm[-1, 0], y_mm[0, 0])  # rows run down the image, as y does
    top = float(img.max())
    fig, ax = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
    for a, data, title in (
        (ax[0, 0], img, "virtual-cathode image"),
        (ax[0, 1], ellipse, "cut on an ellipse (native)"),
        (ax[0, 2], boxed, "cut on a rectangle (ASTRA 'g')"),
    ):
        a.imshow(data, extent=extent, vmin=0, vmax=top, cmap="viridis")
        a.set_title(title)
    for a, data, title in (
        (ax[1, 0], img - ellipse, "image - ellipse"),
        (ax[1, 1], img - boxed, "image - rectangle"),
    ):
        h = a.imshow(data, extent=extent, vmin=-0.12 * top, vmax=0.12 * top, cmap="RdBu_r")
        a.set_title(title)
        fig.colorbar(h, ax=a, fraction=0.04, label="counts")
    for a in ax[:, :2].ravel():
        a.set_xlabel("x [mm]")
        a.set_ylabel("y [mm]")
    along_x, along_y = np.abs(y_mm[:, 0] - p[2]) < 0.06, np.abs(x_mm[0] - p[1]) < 0.06
    shifted = y_mm[:, 0] + p[1] - p[2]  # the y profile drawn about the spot centre in x
    prof = ax[1, 2]
    for data, color, label in ((img, "k", "image"), (ellipse, "C2", "ellipse"), (boxed, "C1", "rectangle")):
        prof.plot(x_mm[0], data[along_x].mean(0), color, lw=0.8 if color == "k" else 1.2, label=f"{label}, along x")
        prof.plot(shifted, data[:, along_y].mean(1), color, ls="--", lw=0.8 if color == "k" else 1.2)
    prof.set_xlabel("x [mm]  (dashed: y about the spot centre)")
    prof.set_ylabel("counts above dark")
    prof.legend(fontsize=8, loc="upper left")
    fig.suptitle(
        f"elliptical cut: sig_x {r['sig_x_mm']:.3f} mm, C_sig_x {r['C_sig_x']:.2f};  sig_y {r['sig_y_mm']:.3f} mm, "
        f"C_sig_y {r['C_sig_y']:.2f};  edge blur {r['blur_mm']:.3f} mm"
    )
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--image", default=str(IMAGE))
    p.add_argument("--dataset", default="Capture000001")
    p.add_argument("--px-per-mm", type=float, default=PX_PER_MM)
    p.add_argument("--binning", type=int, default=2, help="average n x n pixels before fitting (speed)")
    p.add_argument("--output-dir", default=str(HERE.parents[1] / ".cache" / "photocathode"))
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()

    r, window = fit_image(load(args.image, args.dataset), args.px_per_mm, args.binning)
    e, rect, uncut = r["error"], r["rectangle"], r["uncut"]
    print(
        f"dark level {r['dark_counts']:.1f} counts, {r['pixels_repaired']} outlier pixels repaired; "
        f"spot centre ({r['x0_mm']:.3f}, {r['y0_mm']:.3f}) mm from the sensor centre"
    )
    print(
        f"cut on an ellipse: sig_x {r['sig_x_mm']:.3f} +- {e['sig_x_mm']:.3f} mm, C_sig_x {r['C_sig_x']:.2f} +- "
        f"{e['C_sig_x']:.2f}; sig_y {r['sig_y_mm']:.3f} +- {e['sig_y_mm']:.3f} mm, C_sig_y {r['C_sig_y']:.2f} +- "
        f"{e['C_sig_y']:.2f}; edge blur {r['blur_mm']:.3f} mm"
    )
    print(
        f"  semi-axes {r['half_width_x_mm']:.3f} x {r['half_width_y_mm']:.3f} mm; residual "
        f"{r['rms_residual_counts']:.1f} counts rms of a {r['amplitude']:.0f} count peak; rms size of the cut distribution "
        f"{r['generator_rms_mm'][0]:.3f} x {r['generator_rms_mm'][1]:.3f} mm, image second moments "
        f"{r['image_rms_mm'][0]:.3f} x {r['image_rms_mm'][1]:.3f} mm"
    )
    print(
        f"cut on a rectangle (ASTRA): sig_x {rect['sig_x_mm']:.3f}, C_sig_x {rect['C_sig_x']:.2f}; sig_y "
        f"{rect['sig_y_mm']:.3f}, C_sig_y {rect['C_sig_y']:.2f}; residual {rect['rms_residual_counts']:.1f} counts rms; "
        f"rms size {rect['generator_rms_mm'][0]:.3f} x {rect['generator_rms_mm'][1]:.3f} mm"
    )
    print(
        f"no cut: sig {uncut['sig_x_mm']:.3f} x {uncut['sig_y_mm']:.3f} mm, residual "
        f"{uncut['rms_residual_counts']:.1f} counts rms"
    )
    print("generator.in:\n" + generator_lines(r))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "laser_fit.json").write_text(json.dumps(r, indent=2) + "\n")
    if not args.no_plots:
        plot(r, window, out / "laser_fit.png")
    print(f"outputs in {out}")


if __name__ == "__main__":
    main()
