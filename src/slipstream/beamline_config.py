"""Small, strict YAML run configuration for the native beamline facade."""
from __future__ import annotations

import math
from pathlib import Path

import torch
import yaml

from .constants import C_LIGHT, ELECTRON_MASS_EV


def _mapping(value, where, allowed, required=()):
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a mapping")
    unknown = set(value) - set(allowed)
    missing = set(required) - set(value)
    if unknown:
        raise ValueError(f"Unknown {where} keys: {sorted(unknown, key=str)}")
    if missing:
        raise ValueError(f"Missing {where} keys: {sorted(missing)}")
    return dict(value)


def _number(value, where, minimum=None, positive=False, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{where} must be a finite {'integer' if integer else 'number'}")
    if integer and not isinstance(value, int):
        raise ValueError(f"{where} must be an integer")
    if minimum is not None and value < minimum or positive and value <= 0:
        raise ValueError(f"{where} is outside the supported range")
    return value


def _path(value, base, where):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where} must be a nonempty path string")
    return (base / value).resolve()


def _bool(value, where):
    if not isinstance(value, bool):
        raise ValueError(f"{where} must be a boolean")


def read_config(path):
    """Validate and normalize a config; all paths are anchored to its file."""
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        cfg = _mapping(yaml.safe_load(stream), "study", {
            "lattice", "energy_eV", "mc2_eV", "bunch", "tracker", "run", "reference",
        }, {"lattice", "energy_eV", "bunch"})
    base = path.parent
    lattice = _mapping(cfg["lattice"], "lattice", {"path", "root", "wakes", "wake_base_dir"}, {"path", "root"})
    lattice["path"] = _path(lattice["path"], base, "lattice.path")
    if not isinstance(lattice["root"], str) or not lattice["root"]:
        raise ValueError("lattice.root must be a nonempty string")
    _bool(lattice.get("wakes", True), "lattice.wakes")
    lattice["wake_base_dir"] = (
        _path(lattice["wake_base_dir"], base, "lattice.wake_base_dir")
        if "wake_base_dir" in lattice else lattice["path"].parent
    )
    mass = _number(cfg.get("mc2_eV", ELECTRON_MASS_EV), "mc2_eV", positive=True)
    energy = _number(cfg["energy_eV"], "energy_eV", positive=True)
    if energy <= mass:
        raise ValueError("energy_eV must be total energy greater than mc2_eV")

    bool_options = {
        "compile", "lsc", "csr", "fast_collective_effects", "always_change_p0",
        "dipole_collective_subcycle", "compile_collective_chunks",
    }
    numeric_options = {
        "bunch_charge_C", "pipe_radius_b_m", "lsc_radius_factor", "csr_softening_m",
    }
    int_options = {"n_kicks", "n_kicks_bend", "integration_order", "n_slices"}
    tracker = _mapping(cfg.get("tracker", {}), "tracker", bool_options | numeric_options | int_options | {
        "device", "dtype", "compile_mode",
    })
    for name in bool_options & tracker.keys():
        if name == "compile_collective_chunks" and tracker[name] is None:
            continue
        _bool(tracker[name], f"tracker.{name}")
    for name in numeric_options & tracker.keys():
        if name in {"pipe_radius_b_m", "csr_softening_m"} and tracker[name] is None:
            continue
        _number(tracker[name], f"tracker.{name}", positive=name != "bunch_charge_C")
    for name in int_options & tracker.keys():
        _number(tracker[name], f"tracker.{name}", integer=True, minimum=1)
    if tracker.get("integration_order", 4) not in (4, 6, 8):
        raise ValueError("tracker.integration_order must be 4, 6 or 8")
    dtype = tracker.get("dtype", "float64")
    if dtype not in ("float32", "float64"):
        raise ValueError("tracker.dtype must be float32 or float64")
    tracker["dtype"] = getattr(torch, dtype)
    device = tracker.get("device", "cpu")
    if not isinstance(device, str):
        raise ValueError("tracker.device must be cpu, cuda or cuda:<index>")
    try:
        parsed_device = torch.device(device)
    except (RuntimeError, ValueError) as exc:
        raise ValueError("tracker.device must be cpu, cuda or cuda:<index>") from exc
    if parsed_device.type not in {"cpu", "cuda"} or parsed_device.type == "cpu" and parsed_device.index is not None:
        raise ValueError("tracker.device must be cpu, cuda or cuda:<index>")
    tracker["device"] = device
    if tracker.get("compile_mode") not in (None, "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"):
        raise ValueError("Unsupported tracker.compile_mode")

    bunch = cfg["bunch"]
    if not isinstance(bunch, dict) or bunch.get("source") not in ("generated", "sdds", "hdf5", "astra"):
        raise ValueError("bunch.source must be generated, sdds, hdf5 or astra")
    source = bunch["source"]
    if source == "generated":
        required = {
            "n", "beta_x0", "alpha_x0", "epsx_m", "beta_y0", "alpha_y0", "epsy_m",
            "sigma_tau_m", "sigma_delta",
        }
        bunch = _mapping(bunch, "bunch", required | {
            "source", "Dx0", "Dpx0", "Dy0", "Dpy0", "chirp_per_m", "seed",
        }, required)
        bunch.setdefault("seed", 0)
        for name, value in bunch.items():
            if name == "source":
                continue
            _number(value, f"bunch.{name}", integer=name in {"n", "seed"},
                    minimum=2 if name == "n" else 0 if name in {"seed", "sigma_tau_m", "sigma_delta"} else None,
                    positive=name in {"beta_x0", "beta_y0", "epsx_m", "epsy_m"})
    elif source in {"sdds", "hdf5"}:
        allowed = {"source", "path"} | ({"time_origin_s"} if source == "sdds" else set())
        bunch = _mapping(bunch, "bunch", allowed, {"path"})
        bunch["path"] = _path(bunch["path"], base, "bunch.path")
        if "time_origin_s" in bunch:
            _number(bunch["time_origin_s"], "bunch.time_origin_s")
    else:
        bunch = _mapping(bunch, "bunch", {"source", "path", "n_particles", "injector"}, {"path"})
        bunch["path"] = _path(bunch["path"], base, "bunch.path")
        if "n_particles" in bunch:
            _number(bunch["n_particles"], "bunch.n_particles", integer=True, minimum=2)
        options = _mapping(bunch.get("injector", {}), "bunch.injector", {
            "use_compile", "record_every", "z_stop_m", "mesh", "verbose",
        })
        for name in {"use_compile", "verbose"} & options.keys():
            _bool(options[name], f"bunch.injector.{name}")
        if "record_every" in options:
            _number(options["record_every"], "bunch.injector.record_every", integer=True, minimum=1)
        if "z_stop_m" in options:
            _number(options["z_stop_m"], "bunch.injector.z_stop_m")
        if "mesh" in options and options["mesh"] not in ("adaptive", "fixed"):
            raise ValueError("bunch.injector.mesh must be adaptive or fixed")
        options.setdefault("use_compile", tracker.get("compile", False))
        options.setdefault("verbose", False)
        options.update(device=device, dtype=tracker["dtype"])
        bunch["injector"] = options

    run = _mapping(cfg.get("run", {}), "run", {"return_trajectory", "output"})
    _bool(run.get("return_trajectory", False), "run.return_trajectory")
    if "output" in run:
        run["output"] = _path(run["output"], base, "run.output")
    reference = cfg.get("reference")
    if isinstance(reference, str):
        with _path(reference, base, "reference").open(encoding="utf-8") as stream:
            reference = yaml.safe_load(stream)
    if reference is not None:
        reference = _mapping(reference, "reference", {
            "energy_eV", "sigma_x_m", "sigma_y_m", "sigma_tau_m", "sigma_delta",
        })
        for name, value in reference.items():
            _number(value, f"reference.{name}")
    return dict(lattice=lattice, energy_eV=energy, mc2_eV=mass, tracker=tracker,
                bunch=bunch, run=run, reference=reference, path=path)


def load_sdds_particles(path, energy_eV, mc2_eV, device, dtype, time_origin_s=0.0):
    """Read one elegant-style particle page, using its dimensionless p=beta*gamma."""
    from .sdds_io import sdds_object
    from .tracking.native.elements import _canonical_from_slopes

    try:
        with sdds_object() as (_, data):
            data.load(str(path))
            columns = []
            for name in ("x", "xp", "y", "yp", "t", "p"):
                if name not in data.columnName:
                    raise ValueError(f"SDDS bunch is missing column {name}")
                pages = data.columnData[data.columnName.index(name)]
                if len(pages) != 1:
                    raise ValueError("SDDS bunch must contain exactly one page")
                columns.append(torch.as_tensor(pages[0], dtype=torch.float64, device=device))
    except ImportError as exc:
        raise ImportError("SDDS bunch input requires the optional slipstream-accel[sdds] extra") from exc
    if any(c.ndim != 1 or c.shape != columns[0].shape for c in columns):
        raise ValueError("SDDS particle columns must have equal one-dimensional shapes")
    x, xp, y, yp, time, momentum = columns
    if not bool(torch.all(momentum > 0)):
        raise ValueError("SDDS p must be positive (forward-moving particles)")
    p0 = math.sqrt((energy_eV / mc2_eV) ** 2 - 1.0)
    delta = momentum / p0 - 1.0
    px, py = _canonical_from_slopes(xp, yp, delta)
    return torch.stack((x, px, y, py, C_LIGHT * (time - time_origin_s), delta), dim=-1).to(dtype=dtype)
