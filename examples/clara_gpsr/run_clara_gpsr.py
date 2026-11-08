"""GPSR reconstruction of the CLARA S02 bunch from SCR-04 images, with and without CSR in the model.

Four cases share one machine model (see clara_model.py) and one scan of S07 quadrupoles and the TDC:

  A  truth tracked with CSR, fit with CSR            (the full model)
  B  truth tracked with CSR, fit without CSR         (what ignoring CSR does to the reconstruction)
  C  truth tracked without CSR, fit without CSR      (baseline: GPSR works when the model is right)
  D  truth tracked without CSR, fit with CSR         (optional: a model with CSR on a beam that has none)

The truth is a real (simulated) non-Gaussian SIMBA bunch at the entrance of S02. The fit is a GPSR generator, a neural network that maps
fixed random samples to particles, trained with Adam on the image loss; each step is one CUDA graph on a GPU.

    python run_clara_gpsr.py --bunch CLA-S02-SIM-APER-01.openpmd.hdf5 --workdir _cache --out results
    python run_clara_gpsr.py --bunch ... --smoke        # a CPU check of the whole chain in a minute or two
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from clara_model import ClaraForward, load_bunch, make_machine
from slipstream import ELECTRON_MASS_EV, GraphedStep
from slipstream.reconstruction import from_cheetah

CASES = {"A": (True, True), "B": (True, False), "C": (False, False), "D": (False, True)}  # (truth has CSR, fit has CSR)
TDC_VOLTAGE_V = 3.0e6
QUADS = [(a, b) for a in (-4.0, 0.0, 4.0, 8.0) for b in (-2.0, 2.6, 6.0)]  # (k1 Q08, k1 Q09) of the scan, 1/m^2
HELDOUT = [(-5.0, 4.0), (-2.0, 0.0), (2.0, 4.5), (6.0, 1.0), (7.0, 5.0)]  # settings the fit never sees
# Cheetah coordinate scales of the generator output: x, px, y, py, tau (m), (E - E0) / (p0 c)
OUTPUT_SCALE = [2e-3, 1e-4, 2e-3, 1e-4, 1e-3, 1e-2]


def scan_table(quad_pairs, device, dtype):
    """Rows of (k1 Q08, k1 Q09, TDC voltage): every quadrupole pair without and with the TDC."""
    rows = [[a, b, v] for a, b in quad_pairs for v in (0.0, TDC_VOLTAGE_V)]
    return torch.tensor(rows, dtype=dtype, device=device)


def make_generator(n, energy_eV, seed, width=64, hidden=2):
    """GPSR's neural-network beam generator: fixed standard-normal samples through an MLP, scaled per coordinate by ``OUTPUT_SCALE``."""
    from gpsr.beams import NNParticleBeamGenerator, NNTransform

    torch.manual_seed(seed)
    generator = NNParticleBeamGenerator(n, energy_eV, transformer=NNTransform(hidden, width, output_scale=1.0))
    generator.transformer.output_scale = torch.tensor(OUTPUT_SCALE)
    return generator


def sample(generator) -> torch.Tensor:
    """Particles in Cheetah coordinates, ``(n, 6)``."""
    return generator.transformer(generator.base_particles)


def beam_metrics(p6: torch.Tensor, energy_eV: float) -> dict:
    """Projected normalised emittances, Twiss parameters and longitudinal moments of a bunch in Slipstream coordinates."""
    p6 = p6.detach().double().cpu()
    bg = math.sqrt(energy_eV**2 - ELECTRON_MASS_EV**2) / ELECTRON_MASS_EV
    out = {}
    for name, (q, p) in {"x": (0, 1), "y": (2, 3)}.items():
        u, up = p6[:, q] - p6[:, q].mean(), p6[:, p] - p6[:, p].mean()
        suu, sup, spp = float((u * u).mean()), float((u * up).mean()), float((up * up).mean())
        eps = math.sqrt(max(suu * spp - sup**2, 1e-60))
        out.update({f"eps_{name}_um": bg * eps * 1e6, f"beta_{name}_m": suu / eps, f"alpha_{name}": -sup / eps})
    tau, delta = p6[:, 4] - p6[:, 4].mean(), p6[:, 5] - p6[:, 5].mean()
    out.update(sigma_tau_um=float(tau.std()) * 1e6, sigma_delta_1e3=float(delta.std()) * 1e3,
               chirp_per_m=float((tau * delta).mean() / tau.var()))
    return out


def image_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """RMS image difference relative to the RMS of the target, over all rows."""
    return float(((pred - target) ** 2).mean().sqrt() / (target**2).mean().sqrt())


def run_case(name, machine, args, table, val_table, truth_images, val_truth, device, dtype, out_dir):
    truth_csr, fit_csr = CASES[name]
    forward = ClaraForward(machine, fit_csr)
    forward.set_lattice_parameters(table)
    generator = make_generator(args.n_particles, machine.energy_eV, seed=args.seed).to(device)
    params = [p for p in generator.parameters() if p.requires_grad]
    optimiser = torch.optim.Adam(params, lr=args.lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, args.iterations, eta_min=args.lr * 0.05)
    target = truth_images[truth_csr]

    def loss_fn():
        (images,) = forward.track_and_observe(from_cheetah(sample(generator), machine.energy_eV, ELECTRON_MASS_EV))
        return ((images - target) ** 2).sum(dim=(1, 2)).mean() * 1e3

    step = GraphedStep(loss_fn, params)
    history, t0 = [], time.time()
    for it in range(args.iterations):
        loss = step()
        torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimiser.step()
        schedule.step()
        history.append(float(loss))
        if it % max(1, args.iterations // 20) == 0 or it == args.iterations - 1:
            print(f"  case {name} it {it:5d} loss {history[-1]:.4e}  ({time.time() - t0:.0f} s)", flush=True)

    with torch.no_grad():
        recon = from_cheetah(sample(generator), machine.energy_eV, ELECTRON_MASS_EV).to(dtype)
        forward.set_lattice_parameters(table)
        (fit_images,) = forward.track_and_observe(recon)
        # held-out settings, predicted by the model the fit used and by the truth model on the reconstructed bunch
        full = ClaraForward(machine, True)
        full.set_lattice_parameters(val_table)
        (val_full,) = full.track_and_observe(recon)
        forward.set_lattice_parameters(val_table)
        (val_fit,) = forward.track_and_observe(recon)
        s07, _ = forward.up.track(recon)
    result = dict(
        case=name, truth_csr=truth_csr, fit_csr=fit_csr, iterations=args.iterations, seconds=time.time() - t0, final_loss=history[-1],
        image_error_train=image_error(fit_images, target), image_error_heldout=image_error(val_fit, val_truth[truth_csr]),
        beam=beam_metrics(recon, machine.energy_eV), s07=beam_metrics(s07, machine.s07_energy_eV), overflow=float(forward.up.wake_overflow()),
    )
    np.savez_compressed(out_dir / f"case_{name}.npz", recon=recon.cpu().numpy(), history=np.array(history),
                        fit_images=fit_images.cpu().numpy(), val_fit=val_fit.cpu().numpy())
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bunch", required=True, help="SIMBA openPMD bunch at the entrance of S02")
    ap.add_argument("--workdir", default="_cache", help="where LAURA writes the lattice and wake tables")
    ap.add_argument("--out", default="results")
    ap.add_argument("--cases", default="ABC", help="any of A B C D")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    ap.add_argument("--n-particles", type=int, default=8192)
    ap.add_argument("--iterations", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--clip", type=float, default=10.0)
    ap.add_argument("--noise", type=float, default=0.01, help="Gaussian image noise, as a fraction of the maximum of each image")
    ap.add_argument("--rf-kicks", type=int, default=45, help="sub-kicks per travelling-wave cavity (261 to 360 in the lattice)")
    ap.add_argument("--csr-drift-dz", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run: 1024 particles, 12 iterations")
    args = ap.parse_args()
    if args.smoke:
        args.device, args.n_particles, args.iterations, args.rf_kicks = "cpu", 1024, 12, 12
    warnings.filterwarnings("ignore")
    dtype = getattr(torch, args.dtype)
    device = args.device
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    machine = make_machine(args.bunch, args.workdir, device, dtype, n_reference=4096, rf_kicks=args.rf_kicks,
                           csr_drift_dz=args.csr_drift_dz)
    truth_beam, _, charge = load_bunch(args.bunch, args.n_particles, seed=11, dtype=dtype)
    full_truth, _, _ = load_bunch(args.bunch, None, seed=0, dtype=dtype)
    truth_beam = truth_beam.to(device)
    print(f"machine ready: S02 {machine.energy_eV / 1e6:.2f} MeV, S07 {machine.s07_energy_eV / 1e6:.2f} MeV, {charge * 1e12:.0f} pC, "
          f"device {device}, {args.n_particles} particles", flush=True)

    table, val_table = scan_table(QUADS, device, dtype), scan_table(HELDOUT, device, dtype)
    if args.smoke:
        table, val_table = table[:6], val_table[:2]

    # measured images: the truth bunch through the machine with and without CSR, plus noise
    gen = torch.Generator(device="cpu").manual_seed(args.seed + 100)
    truth_images, val_truth = {}, {}
    with torch.no_grad():
        for csr in sorted({CASES[c][0] for c in args.cases}):
            model = ClaraForward(machine, csr)
            model.set_lattice_parameters(table)
            (img,) = model.track_and_observe(truth_beam)
            noise = torch.randn(img.shape, generator=gen, dtype=dtype).to(device)
            truth_images[csr] = img + args.noise * img.amax(dim=(1, 2), keepdim=True) * noise
            model.set_lattice_parameters(val_table)
            (val_truth[csr],) = model.track_and_observe(truth_beam)
    print("measured images:", {("CSR" if k else "no CSR"): tuple(v.shape) for k, v in truth_images.items()}, flush=True)

    results = {"truth_S02": beam_metrics(full_truth, machine.energy_eV)}
    with torch.no_grad():
        for csr in sorted({CASES[c][0] for c in args.cases}):
            s07_truth, _ = ClaraForward(machine, csr).up.track(truth_beam)
            results["truth_S07_csr" if csr else "truth_S07_nocsr"] = beam_metrics(s07_truth, machine.s07_energy_eV)
    for name in args.cases:
        print(f"case {name}: truth {'with' if CASES[name][0] else 'without'} CSR, fit {'with' if CASES[name][1] else 'without'} CSR", flush=True)
        results[name] = run_case(name, machine, args, table, val_table, truth_images, val_truth, device, dtype, out_dir)
    np.savez_compressed(out_dir / "truth.npz", truth=full_truth.cpu().numpy(), table=table.cpu().numpy(),
                        **{f"images_{'csr' if k else 'nocsr'}": v.cpu().numpy() for k, v in truth_images.items()})
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print_table(results)


def print_table(results):
    keys = ["eps_x_um", "eps_y_um", "beta_x_m", "beta_y_m", "sigma_tau_um", "sigma_delta_1e3", "chirp_per_m"]
    names = [k for k in results if k in CASES]
    print("\nS02 bunch: truth and reconstructions")
    print(f"{'':16s}" + "".join(f"{n:>11s}" for n in ["truth"] + names))
    for key in keys:
        print(f"{key:16s}" + "".join(f"{(results['truth_S02'] if n == 'truth' else results[n]['beam'])[key]:11.3f}" for n in ["truth"] + names))
    print("\nSame bunches at the entrance of S07 (reconstruction tracked by the model it was fitted with; truth with and without CSR)")
    print(f"{'':16s}" + "".join(f"{n:>11s}" for n in ["truth CSR", "truth none"] + names))
    for key in ["eps_x_um", "eps_y_um", "sigma_tau_um", "sigma_delta_1e3"]:
        cells = [results.get(t, {}).get(key, float("nan")) for t in ("truth_S07_csr", "truth_S07_nocsr")] + [results[n]["s07"][key] for n in names]
        print(f"{key:16s}" + "".join(f"{c:11.3f}" for c in cells))
    print("\nImages (relative RMS difference to the measured / held-out truth images)")
    for label, key in [("train img err", "image_error_train"), ("held-out err", "image_error_heldout")]:
        print(f"{label:16s}" + " " * 11 + "".join(f"{results[n][key]:11.3f}" for n in names))


if __name__ == "__main__":
    main()
