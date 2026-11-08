"""Plots and the extra held-out check for a finished ``run_clara_gpsr.py`` run.

The extra check tracks each reconstructed S02 bunch through the *true* machine model (CSR if the measured beam had it) at the held-out
settings, which is what an operator would see if the reconstruction were used to predict the machine. For a fit made with a wrong model (case B)
this is the number that matters, because the fit's own prediction can agree with the data while the bunch is wrong.

    python plot_results.py --bunch CLA-S02-SIM-APER-01.openpmd.hdf5 --workdir _cache --out results

Use the same ``--n-particles``, ``--rf-kicks`` and ``--csr-drift-dz`` as the run. Writes ``loss.png``, ``bunch_s02.png``, ``bunch_s07.png``,
``images.png`` and ``analysis.json`` into the results folder.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from clara_model import ClaraForward, load_bunch, make_machine  # noqa: E402
from run_clara_gpsr import CASES, HELDOUT, image_error, scan_table  # noqa: E402

LABEL = {"A": "A: CSR truth, CSR fit", "B": "B: CSR truth, no-CSR fit", "C": "C: no-CSR truth, no-CSR fit", "D": "D: no-CSR truth, CSR fit"}
COLOUR = {"truth": "k", "A": "tab:blue", "B": "tab:red", "C": "tab:green", "D": "tab:orange"}
PLANES = [("x (mm)", "px (mrad)", 0, 1, 1e3, 1e3), ("y (mm)", "py (mrad)", 2, 3, 1e3, 1e3), ("tau (um)", "delta (1e-3)", 4, 5, 1e6, 1e3)]


def colour(name: str) -> str:
    """Case letter ('A: ...') gets its case colour; 'truth no CSR' is grey; any other truth is black."""
    return "tab:gray" if name == "truth no CSR" else COLOUR.get(name[:1], "k")


def phase_figure(bunches: dict, path: Path, title: str, max_points: int = 4000, overlay: tuple | None = None, ref: str | None = None):
    """One row per bunch, one column per plane; axis limits and the origin come from the first bunch.

    ``overlay=(a, b)`` adds a first row with bunch ``b`` (red) drawn on top of bunch ``a`` (black), so a small difference between two
    bunches can be seen. ``ref`` adds a fourth column, the energy minus a cubic in tau fitted to bunch ``ref``: a large chirp hides small
    energy changes in the third column, and this removes it."""
    first = next(iter(bunches.values()))
    views = {
        n: [((p[:max_points, i] - first[:, i].mean()) * si, (p[:max_points, j] - first[:, j].mean()) * sj) for _, _, i, j, si, sj in PLANES]
        for n, p in bunches.items()
    }
    poly = np.polyfit(*views[ref][2], 3) if ref else None
    rows = [(n, [(views[n], colour(n))]) for n in views]
    if overlay:
        rows.insert(0, (f"{overlay[1]} (red) on {overlay[0]} (black)", [(views[overlay[0]], "k"), (views[overlay[1]], "tab:red")]))
    n_col = 3 + (ref is not None)
    fig, axes = plt.subplots(len(rows), n_col, figsize=(3.4 * n_col, 2.7 * len(rows)), squeeze=False)

    def detrended(view):
        x, y = view[2]
        return x, y - np.polyval(poly, x)

    def span(v):
        lo, hi = np.percentile(v, [0.5, 99.5])
        return lo - 0.5 * (hi - lo), hi + 0.5 * (hi - lo)

    base = views[next(iter(views))]
    if ref:  # scale of the energy column from the truth bunches; a reconstruction far off that scale runs off the panel
        res_lim = span(np.concatenate([detrended(views[n])[1] for n in (overlay or views)]))
    for r, (name, layers) in enumerate(rows):
        for c in range(n_col):
            ax = axes[r, c]
            for view, col in layers:
                x, y = view[c] if c < 3 else detrended(view)
                ax.scatter(x, y, s=1, alpha=0.3, rasterized=True, color=col)
            ax.set_xlim(*span(base[min(c, 2)][0]))
            ax.set_ylim(*(span(base[c][1]) if c < 3 else res_lim))
            if r == len(rows) - 1:
                ax.set_xlabel(PLANES[min(c, 2)][0])
            ax.set_ylabel(PLANES[c][1] if c < 3 else "delta - cubic(tau) (1e-3)")
            if c == 0:
                ax.set_title(name, loc="left", fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bunch", required=True)
    ap.add_argument("--workdir", default="_cache")
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    ap.add_argument("--n-particles", type=int, default=8192)
    ap.add_argument("--rf-kicks", type=int, default=45)
    ap.add_argument("--csr-drift-dz", type=float, default=0.05)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    dtype, device, out = getattr(torch, args.dtype), args.device, Path(args.out)
    results = json.loads((out / "results.json").read_text())
    cases = [c for c in CASES if c in results]

    machine = make_machine(args.bunch, args.workdir, device, dtype, n_reference=4096, rf_kicks=args.rf_kicks, csr_drift_dz=args.csr_drift_dz)
    truth_beam, _, _ = load_bunch(args.bunch, args.n_particles, seed=11, dtype=dtype)
    truth_beam = truth_beam.to(device)
    val_table = scan_table(HELDOUT, device, dtype)
    models = {csr: ClaraForward(machine, csr) for csr in (False, True)}
    for model in models.values():
        model.set_lattice_parameters(val_table)

    # the loss curves and, for every case, the reconstruction seen by the true machine
    fig, ax = plt.subplots(figsize=(6, 4))
    s02, s07, analysis = {"truth": load_bunch(args.bunch, None, dtype=dtype)[0].numpy()}, {}, {}
    with torch.no_grad():
        for csr in (True, False):
            if any(CASES[c][0] == csr for c in cases):
                s07[f"truth {'CSR' if csr else 'no CSR'}"] = models[csr].up.track(truth_beam)[0].cpu().numpy()
        val_truth = {csr: models[csr].track_and_observe(truth_beam)[0] for csr in {CASES[c][0] for c in cases}}
        for c in cases:
            d = np.load(out / f"case_{c}.npz")
            ax.semilogy(d["history"], color=COLOUR[c], label=LABEL[c])
            recon = torch.tensor(d["recon"], dtype=dtype, device=device)
            truth_csr, fit_csr = CASES[c]
            s02[LABEL[c]] = d["recon"]
            s07[LABEL[c]] = models[fit_csr].up.track(recon)[0].cpu().numpy()
            seen = models[truth_csr].track_and_observe(recon)[0]
            analysis[c] = dict(image_error_heldout_true_machine=image_error(seen, val_truth[truth_csr]), predicted=seen.cpu().numpy())
    ax.set_xlabel("iteration")
    ax.set_ylabel("image loss")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "loss.png", dpi=130)
    plt.close(fig)

    phase_figure(s02, out / "bunch_s02.png", "Bunch at the entrance of S02")
    both = "truth CSR" in s07 and "truth no CSR" in s07
    phase_figure(s07, out / "bunch_s07.png", "Bunch at the entrance of S07 (each reconstruction tracked by the model it was fitted with)",
                 overlay=("truth CSR", "truth no CSR") if both else None, ref="truth no CSR" if "truth no CSR" in s07 else "truth CSR")

    # held-out images with the TDC on and off at one setting
    h = min(2, len(HELDOUT) - 1)  # HELDOUT[2] = (2, 4.5)
    row_off, row_on = 2 * h, 2 * h + 1
    half = machine.screen.shape[0] * machine.screen.pixel_size / 2 * 1e3  # mm
    cols = ["measured"] + cases
    fig, axes = plt.subplots(2, len(cols), figsize=(2.4 * len(cols), 5), squeeze=False)
    for r, row in enumerate((row_off, row_on)):
        for k, name in enumerate(cols):
            ax = axes[r, k]
            img = val_truth[CASES[cases[0]][0]][row] if name == "measured" else analysis[name]["predicted"][row]
            img = img.cpu().numpy() if hasattr(img, "cpu") else img
            ax.imshow(img, origin="lower", extent=(-half, half, -half, half), cmap="viridis")
            ax.set_title(name if r == 0 else "", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[r, 0].set_ylabel(f"TDC {'on' if row == row_on else 'off'}\nk1 = {HELDOUT[h]}")
    fig.suptitle("Held-out screen images (x horizontal, y vertical)\nreconstructions tracked through the true machine", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "images.png", dpi=130)
    plt.close(fig)

    summary = {c: {k: v for k, v in a.items() if k != "predicted"} for c, a in analysis.items()}
    (out / "analysis.json").write_text(json.dumps(summary, indent=2))
    print("held-out image error with the reconstruction in the true machine (relative RMS):")
    for c in cases:
        print(f"  {LABEL[c]:32s} fit's own model {results[c]['image_error_heldout']:.3f}   true machine {summary[c]['image_error_heldout_true_machine']:.3f}")


if __name__ == "__main__":
    main()
