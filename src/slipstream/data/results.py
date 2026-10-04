"""`TrackingResult` stores sampled ring diagnostics with `sigma_tau_s` in seconds; `BeamlineResult`
stores canonical output particles with tau in metres. See docs/modules/data/results.md."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

@dataclass
class TrackingResult:
    """Turn-indexed diagnostics and HDF5 persistence."""

    code: str  # backend label
    n_particles: int
    n_turns: int
    current_A: float
    turn: np.ndarray
    sigma_delta: np.ndarray
    sigma_tau_s: np.ndarray
    transmission: np.ndarray | None = None
    mean_x_m: np.ndarray | None = None
    mean_y_m: np.ndarray | None = None
    sigma_x_m: np.ndarray | None = None
    sigma_y_m: np.ndarray | None = None
    epsx_m: np.ndarray | None = None
    epsy_m: np.ndarray | None = None
    mean_tau_s: np.ndarray | None = None
    mean_delta: np.ndarray | None = None
    # Optional ramped reference energy.
    E0_eV: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def summarize(self, last_n: int | None = None) -> tuple[float, float, float, float]:
        """(sigma_delta_mean, sigma_delta_std, sigma_tau_mean, sigma_tau_std)
        over the last ``last_n`` samples (default: all of them)."""
        n = last_n or len(self.turn)
        sd, st = self.sigma_delta[-n:], self.sigma_tau_s[-n:]
        return float(sd.mean()), float(sd.std()), float(st.mean()), float(st.std())

    def save(self, path: str | Path) -> None:
        with h5py.File(path, "w") as f:
            f.attrs["code"] = self.code
            f.attrs["n_particles"] = self.n_particles
            f.attrs["n_turns"] = self.n_turns
            f.attrs["current_A"] = self.current_A
            f.create_dataset("turn", data=self.turn)
            f.create_dataset("sigma_delta", data=self.sigma_delta)
            f.create_dataset("sigma_tau_s", data=self.sigma_tau_s)
            if self.transmission is not None:
                f.create_dataset("transmission", data=self.transmission)
            for name in ("mean_x_m", "mean_y_m", "sigma_x_m", "sigma_y_m", "epsx_m", "epsy_m", "mean_tau_s", "mean_delta", "E0_eV"):
                value = getattr(self, name)
                if value is not None:
                    f.create_dataset(name, data=value)
            meta = f.create_group("metadata")
            for k, v in self.metadata.items():
                meta.attrs[k] = v

    @classmethod
    def load(cls, path: str | Path) -> "TrackingResult":
        with h5py.File(path, "r") as f:
            transmission = f["transmission"][()] if "transmission" in f else None
            optional_arrays = {
                name: f[name][()] if name in f else None
                for name in ("mean_x_m", "mean_y_m", "sigma_x_m", "sigma_y_m", "epsx_m", "epsy_m", "mean_tau_s", "mean_delta", "E0_eV")
            }
            metadata = dict(f["metadata"].attrs) if "metadata" in f else {}
            return cls(
                code=str(f.attrs["code"]),
                n_particles=int(f.attrs["n_particles"]),
                n_turns=int(f.attrs["n_turns"]),
                current_A=float(f.attrs["current_A"]),
                turn=f["turn"][()],
                sigma_delta=f["sigma_delta"][()],
                sigma_tau_s=f["sigma_tau_s"][()],
                transmission=transmission,
                metadata=metadata,
                **optional_arrays,
            )


@dataclass
class BeamlineResult:
    """Single-pass canonical beamline output and diagnostics."""

    particles_out: np.ndarray  # (N, 6), tracking.native canonical convention
    energy_eV_out: float
    charge_C: float
    trajectory: np.ndarray | None = None  # (n_elements+1, N, 6)
    energy_trajectory: np.ndarray | None = None  # (n_elements+1,)
    slice_summary: dict[str, np.ndarray] | None = None
    metadata: dict = field(default_factory=dict)

    def scalars(self) -> dict[str, float]:
        """Cheap whole-bunch diagnostics computable from ``particles_out``
        alone, no ``analysis.slices`` dependency -- what ``compare_reference``
        matches reference values against."""
        p = self.particles_out
        return {
            "energy_eV": self.energy_eV_out,
            "sigma_x_m": float(np.std(p[:, 0])),
            "sigma_y_m": float(np.std(p[:, 2])),
            "sigma_tau_m": float(np.std(p[:, 4])),
            "sigma_delta": float(np.std(p[:, 5])),
        }

    def compare_reference(self, reference: dict[str, float]) -> dict[str, float]:
        """``{quantity: relative_difference}`` for every key ``reference`` shares with
        ``self.scalars()`` -- the everyday "did the run reproduce the reference" one-liner (a
        ``.twi``/``.sig`` row, ASTRA emit columns,...)."""
        got = self.scalars()
        return {
            k: (got[k] - v) / v if v != 0.0 else got[k] - v
            for k, v in reference.items() if k in got
        }

    def save(self, path: str | Path) -> None:
        with h5py.File(path, "w") as f:
            f.create_dataset("particles_out", data=self.particles_out)
            f.attrs["energy_eV_out"] = self.energy_eV_out
            f.attrs["charge_C"] = self.charge_C
            if self.trajectory is not None:
                f.create_dataset("trajectory", data=self.trajectory)
            if self.energy_trajectory is not None:
                f.create_dataset("energy_trajectory", data=self.energy_trajectory)
            if self.slice_summary is not None:
                g = f.create_group("slice_summary")
                for k, v in self.slice_summary.items():
                    g.create_dataset(k, data=np.asarray(v))
            meta = f.create_group("metadata")
            for k, v in self.metadata.items():
                if k == "reference_diff" and isinstance(v, dict):
                    ref = meta.create_group("reference_diff")
                    for name, difference in v.items():
                        ref.attrs[name] = difference
                else:
                    meta.attrs[k] = v

    @classmethod
    def load(cls, path: str | Path) -> "BeamlineResult":
        with h5py.File(path, "r") as f:
            slice_summary = None
            if "slice_summary" in f:
                slice_summary = {k: f["slice_summary"][k][()] for k in f["slice_summary"]}
            metadata = dict(f["metadata"].attrs) if "metadata" in f else {}
            if "metadata/reference_diff" in f:
                metadata["reference_diff"] = dict(f["metadata/reference_diff"].attrs)
            return cls(
                particles_out=f["particles_out"][()],
                energy_eV_out=float(f.attrs["energy_eV_out"]),
                charge_C=float(f.attrs["charge_C"]),
                trajectory=f["trajectory"][()] if "trajectory" in f else None,
                energy_trajectory=f["energy_trajectory"][()] if "energy_trajectory" in f else None,
                slice_summary=slice_summary,
                metadata=metadata,
            )


class StreamingH5Writer:
    """Append scalar diagnostic rows to an HDF5 file."""

    def __init__(
        self, path: str | Path, attrs: dict | None = None, metadata: dict | None = None, flush_every: int = 1,
    ) -> None:
        self._f = h5py.File(path, "w")
        for k, v in (attrs or {}).items():
            self._f.attrs[k] = v
        meta = self._f.create_group("metadata")
        for k, v in (metadata or {}).items():
            meta.attrs[k] = v
        self._columns: dict[str, h5py.Dataset] = {}
        self._flush_every = flush_every
        self._appends_since_flush = 0

    def append(self, **values: float) -> None:
        for name, value in values.items():
            ds = self._columns.get(name)
            if ds is None:
                ds = self._f.create_dataset(name, shape=(0,), maxshape=(None,), dtype="f8")
                self._columns[name] = ds
            ds.resize((ds.shape[0] + 1,))
            ds[-1] = value
        self._appends_since_flush += 1
        if self._appends_since_flush >= self._flush_every:
            self._f.flush()
            self._appends_since_flush = 0

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "StreamingH5Writer":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class CoordWriter:
    """Write periodic coordinate snapshots from a compatible tracker state."""

    def __init__(
        self, path: str | Path, n_particles: int, interval: int,
        sparse_interval: int = 1, flush_every: int = 1,
    ) -> None:
        self.interval = interval
        self._particle_index = np.arange(0, n_particles, sparse_interval)
        self._f = h5py.File(path, "w")
        self._f.attrs["interval"] = interval
        self._f.attrs["sparse_interval"] = sparse_interval
        self._f.create_dataset("particle_index", data=self._particle_index)
        self._turn_ds = self._f.create_dataset("turn", shape=(0,), maxshape=(None,), dtype="i8")
        self._columns: dict[str, h5py.Dataset] = {}
        self._flush_every = flush_every
        self._snapshots_since_flush = 0

    def maybe_append(self, turn: int, tracker: Any) -> None:
        """No-op unless ``turn`` is a multiple of ``interval`` -- call every
        time ``run()``'s own loop advances, like elegant's own watch
        element deciding for itself whether this pass is due."""
        if turn % self.interval != 0:
            return
        import torch

        idx = self._particle_index
        n_sparse = len(idx)
        names = ("tau", "delta", "x", "xp", "y", "yp") if tracker.optics is not None else ("tau", "delta")
        idx_t = torch.as_tensor(idx, dtype=torch.long, device=tracker.tau.device)
        i = self._turn_ds.shape[0]
        self._turn_ds.resize((i + 1,))
        self._turn_ds[i] = turn
        for name in names:
            ds = self._columns.get(name)
            if ds is None:
                ds = self._f.create_dataset(
                    name, shape=(0, n_sparse), maxshape=(None, n_sparse), dtype="f8", chunks=(1, n_sparse),
                )
                self._columns[name] = ds
            ds.resize((i + 1, n_sparse))
            ds[i] = getattr(tracker, name)[idx_t].detach().cpu().numpy()
        self._snapshots_since_flush += 1
        if self._snapshots_since_flush >= self._flush_every:
            self._f.flush()
            self._snapshots_since_flush = 0

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "CoordWriter":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
