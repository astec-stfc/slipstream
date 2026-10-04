"""Common output schema for both the pySTABLE tracker and the elegant
driver, HDF5-backed save/load."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

@dataclass
class TrackingResult:
    """One tracking run's sampled history, from either code.

    ``turn`` / ``sigma_delta`` / ``sigma_tau_s`` are equal-length 1D arrays,
    one entry per sample (elegant: one per WATCH interval; pySTABLE: one
    per ``sample_every`` turns). ``metadata`` is free-form (rho/d/sigma_s_mm,
    seed, n_bunches, ... -- whatever the caller wants to keep alongside the
    numbers for later plotting/filtering).
    """

    code: str  # "pystable" or "elegant"
    n_particles: int
    n_turns: int
    current_A: float
    turn: np.ndarray
    sigma_delta: np.ndarray
    sigma_tau_s: np.ndarray
    transmission: np.ndarray | None = None
    # Transverse (only populated when the run tracked optics -- see
    # tracking.pystable.tracker.SingleBunchTracker's optics= kwarg).
    # mean_x_m/mean_y_m are the essential ones: a coherent transverse
    # instability is centroid growth, which sigma_x_m/sigma_y_m alone
    # would miss.
    mean_x_m: np.ndarray | None = None
    mean_y_m: np.ndarray | None = None
    sigma_x_m: np.ndarray | None = None
    sigma_y_m: np.ndarray | None = None
    # Live rms geometric emittance (SingleBunchTracker._rms_emittance on
    # the tracked x/xp, y/yp) -- distinct from sigma_x_m/sigma_y_m (a beam
    # size, which conflates emittance with beta) and from optics.epsx_m/
    # epsy_m (the fixed *equilibrium* target, not what's actually being
    # tracked). Directly comparable to elegant's own WATCH "ex"/"ey"
    # columns.
    epsx_m: np.ndarray | None = None
    epsy_m: np.ndarray | None = None
    # Longitudinal bunch centroid (mean tau/delta) -- populated by
    # SingleBunchTracker.run() unconditionally (cheap, always meaningful);
    # optional here only for backward compatibility with other callers
    # (e.g. the elegant driver) that don't yet supply it. The concrete
    # "oscillations in the bunch in the longitudinal RF bucket" signal for
    # a ramping tracker (see lattice.ramp.RampProfile/RAMP_PLAN.md): a
    # non-adiabatic ramp program shows up as coherent dipole motion here,
    # not in sigma_tau_s/sigma_delta alone.
    mean_tau_s: np.ndarray | None = None
    mean_delta: np.ndarray | None = None
    # Ramped central energy -- only populated when the run used
    # SingleBunchTracker(ramp=...), see lattice.ramp.RampProfile.
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
    """One ``BeamlineStudy.track()`` pass's output (``main.beamline`` /
    ``slipstream.beamline.BeamlineStudy``, E3 -- see
    plans/essential/E3_NATIVE_FACADE_PLAN.md). A *pass*-level schema
    (single beamline traversal), not ``TrackingResult``'s own *turn*-level
    one -- the native linac/injector family's own facade, kept a sibling
    type rather than forced into the ring-only shape.

    ``particles_out``/``trajectory``/``energy_trajectory`` are plain numpy
    (already off-device) -- a completed result, not a live tensor handle.
    ``slice_summary`` (optional, ``None`` unless ``analysis.slices`` was
    importable and asked for) is a flat ``{name: (n_slices,) array}`` dict,
    the same keys ``analysis.slices.slice_stats`` returns."""

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
        """``{quantity: relative_difference}`` for every key ``reference``
        shares with ``self.scalars()`` -- the everyday "did the run
        reproduce the reference" one-liner (a ``.twi``/``.sig`` row, ASTRA
        emit columns, ...). Absolute difference (not relative) when the
        reference value is exactly zero."""
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
    """Incrementally-written companion to ``TrackingResult.save()``: opens
    an HDF5 file immediately and appends one row per sample as it's
    computed, instead of buffering an entire run in memory and writing
    once at the end. Useful for a long run (e.g. tracking a full
    synchrotron ramp, ``examples/diamond2booster``): partial results
    survive a crash/interrupt, and progress is visible on disk (e.g. via
    a separate ``TrackingResult.load()`` call) while the run is still
    going.

    No fixed column schema -- each ``append(**values)`` call creates any
    not-yet-seen named dataset lazily, so the same writer works for
    ``TrackingResult``'s own field names (``turn``, ``sigma_delta``, ...)
    or extra ones a specific script wants (e.g. ``ex``/``ey``). To produce
    a file ``TrackingResult.load()`` can read back, every ``append()``
    call must include at least ``turn``/``sigma_delta``/``sigma_tau_s``,
    and ``attrs`` must include ``code``/``n_particles``/``n_turns``/
    ``current_A`` (``TrackingResult.load()``'s own required fields).

    Use as a context manager, or call ``close()`` explicitly::

        with StreamingH5Writer(path, attrs=dict(code="pystable", n_particles=n,
                                                 n_turns=n_turns, current_A=current_A),
                                metadata={"energy_MeV": "100"}) as writer:
            for _ in range(n_turns // sample_every):
                tracker.track(sample_every)
                writer.append(turn=n, sigma_delta=tracker.sigma_delta, sigma_tau_s=tracker.sigma_tau_s)

    ``flush_every``: ``flush()`` is what makes partial results durable
    (survive a hard crash/power loss), but it's a real per-call cost --
    default (every append) is the safest choice and fine for a coarse
    ``sample_every`` (e.g. ``run_ramp_example.py``'s default 100), but for
    a script sampling on a *fine* grid to match an
    external reference (e.g. ``validate_ramp_vs_elegant.py``'s every-10-
    turn elegant-WATCH alignment -- ~18,000 appends for a full ramp),
    flushing every single append adds up. Pass e.g. ``flush_every=50`` to
    trade a bounded amount of crash-survivability (lose at most that many
    trailing rows) for throughput; ``close()``/``__exit__`` always does a
    final flush regardless, so a clean run never loses anything either way.
    """

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
    """Periodic per-particle coordinate snapshots to HDF5 -- the pystable
    analogue of elegant's ``watch, mode="coord"`` element
    (``INTERVAL``/``SPARSE_INTERVAL``/``FLUSH_INTERVAL``), passed as
    ``SingleBunchTracker.run(..., coord_writer=...)``. A separate concern
    from ``StreamingH5Writer``'s aggregate per-sample statistics -- this
    writes the raw (sparse-subsampled) particle coordinates themselves, on
    its own ``interval``, independent of ``run()``'s own ``sample_every``.

    ``interval``: save a snapshot every this many *tracked* turns
    (elegant's ``INTERVAL``).
    ``sparse_interval``: keep every this-many'th particle, by index into
    the tracker's own ``tau``/``delta``/... tensors (elegant's
    ``SPARSE_INTERVAL``) -- 1 keeps every particle.
    ``flush_every``: flush to disk every this many *saved snapshots*
    (elegant's ``FLUSH_INTERVAL``) -- same durability/throughput tradeoff
    as ``StreamingH5Writer``'s own ``flush_every``.

    Datasets: ``turn`` (1D) and ``particle_index`` (1D, the original
    indices kept) plus ``tau``/``delta`` (and, once the tracker has
    ``optics=``, ``x``/``xp``/``y``/``yp`` -- lazily added on first sight,
    same discipline as ``StreamingH5Writer``'s own columns), each shape
    ``(n_snapshots, n_sparse_particles)``.

    Use as a context manager, or call ``close()`` explicitly::

        with CoordWriter(path, n_particles, interval=1000, sparse_interval=1000) as cw:
            tracker.run(n_turns, sample_every=100, coord_writer=cw)
    """

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
        # A LongTensor on the tracker's own device, not raw numpy-array
        # indexing -- avoids depending on numpy-array-to-tensor fancy
        # indexing working identically across torch versions/devices.
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
