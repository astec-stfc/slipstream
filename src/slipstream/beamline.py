"""Composes single-pass canonical linac tracking with bunch generation, slice diagnostics and optional
injector transport. See docs/modules/beamline.md."""
from __future__ import annotations

import math
from pathlib import Path

import torch

from .constants import ELECTRON_MASS_EV
from .data.results import BeamlineResult
from .tracking.native.injector import Injector, injector_exit_to_canonical6
from .tracking.native.lattice_import import ElementSpec, parse_elegant_lattice
from .tracking.native.linac import LinacTracker, resolve_wakes


class BeamlineStudy:
    """Generate, track and save a canonical bunch using one reusable linac tracker."""

    def __init__(
        self, specs: list[ElementSpec], energy_eV: float,
        mc2_eV: float = ELECTRON_MASS_EV,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float64,
        bunch_charge_C: float = 0.0,
        compile: bool = False,
        compile_mode: str | None = None,
        lsc: bool = False,
        csr: bool = False,
        pipe_radius_b_m: float | None = None,
        fast_collective_effects: bool = False,
        n_slices: int = 50,
        **tracker_kwargs,
    ):
        self.specs = specs
        self.energy_eV = energy_eV
        self.mc2_eV = mc2_eV
        self.n_slices = n_slices
        self._tracker = LinacTracker(
            specs, energy_eV, mc2_eV=mc2_eV, device=device, dtype=dtype,
            compile=compile, compile_mode=compile_mode, bunch_charge_C=bunch_charge_C,
            lsc=lsc, csr=csr, pipe_radius_b_m=pipe_radius_b_m,
            fast_collective_effects=fast_collective_effects, **tracker_kwargs,
        )
        self._run_config = None

    @classmethod
    def from_lattice_file(
        cls, lte_path: str | Path, root_name: str, energy_eV: float,
        mc2_eV: float = ELECTRON_MASS_EV, wake_base_dir: str | Path | None = None, *,
        strict_import: bool = False, import_audit: list[dict[str, object]] | None = None, **kwargs,
    ) -> "BeamlineStudy":
        """Read a lattice; wake paths default to the lattice's own directory."""
        specs = resolve_wakes(parse_elegant_lattice(lte_path, root_name, strict=strict_import, audit=import_audit),
                              base_dir=Path(lte_path).resolve().parent if wake_base_dir is None else wake_base_dir)
        return cls(specs, energy_eV, mc2_eV=mc2_eV, **kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path, *, overrides: dict[str, object] | None = None) -> "BeamlineStudy":
        """Build a study, applying validated dotted-key overrides before construction."""
        from .beamline_config import read_config

        config = read_config(path, overrides=overrides)
        lattice = config["lattice"]
        specs = parse_elegant_lattice(lattice["path"], lattice["root"], strict=lattice.get("strict_import", False))
        if lattice.get("wakes", True):
            specs = resolve_wakes(specs, base_dir=lattice["wake_base_dir"])
        study = cls(specs, config["energy_eV"], mc2_eV=config["mc2_eV"], **config["tracker"])
        study._run_config = config
        return study

    def run(self, output_path: str | Path | None = None) -> BeamlineResult:
        """Execute the configured source and beamline, optionally saving HDF5."""
        if self._run_config is None:
            raise ValueError("run() requires a study created with BeamlineStudy.from_yaml()")
        cfg = self._run_config
        bunch = dict(cfg["bunch"])
        source = bunch.pop("source")
        self.energy_eV = cfg["energy_eV"]
        self._tracker._set_reference_energy(self.energy_eV)
        self._tracker.bunch_charge_C = cfg["tracker"].get("bunch_charge_C", 0.0)
        kwargs = dict(return_trajectory=cfg["run"].get("return_trajectory", False),
                      reference=cfg["reference"], metadata={"config_path": str(cfg["path"]), "source": source,
                                                          "config_yaml": cfg["config_yaml"]})
        if source == "astra":
            injector = Injector.from_astra(bunch["path"], mc2_eV=self.mc2_eV)
            result = self.track_with_injector(injector, n_particles=bunch.get("n_particles"),
                                              injector_kwargs=bunch["injector"], **kwargs)
        else:
            if source == "generated":
                particles = self.generate_bunch(**bunch)
            elif source == "sdds":
                from .beamline_config import load_sdds_particles

                particles = load_sdds_particles(
                    bunch["path"], self.energy_eV, self.mc2_eV, self._tracker.device,
                    self._tracker.dtype, bunch.get("time_origin_s", 0.0),
                )
            else:
                from .data.particles import load_particles_h5
                from .tracking.native.elements import _canonical_from_slopes

                particles, meta = load_particles_h5(bunch["path"], self._tracker.device, self._tracker.dtype)
                if meta["tau_units"] != "m" or meta["mc2_eV"] != self.mc2_eV:
                    raise ValueError("HDF5 bunch must have tau_units=m and mc2_eV matching the study")
                if not math.isfinite(meta["energy_eV"]) or not meta["energy_eV"] > self.mc2_eV:
                    raise ValueError("HDF5 bunch energy_eV must be finite and exceed mc2_eV")
                if not math.isfinite(meta["charge_C"]):
                    raise ValueError("HDF5 bunch charge_C must be finite")
                self.energy_eV = meta["energy_eV"]
                self._tracker._set_reference_energy(self.energy_eV)
                self._tracker.bunch_charge_C = meta["charge_C"]
                if not meta["canonical"]:
                    if particles.ndim != 2 or particles.shape[1] != 6:
                        raise ValueError("Bunch must have shape (N, 6)")
                    x, xp, y, yp, tau, delta = particles.unbind(-1)
                    px, py = _canonical_from_slopes(xp, yp, delta)
                    particles = torch.stack((x, px, y, py, tau, delta), dim=-1)
            if particles.ndim != 2 or particles.shape[1] != 6 or particles.shape[0] < 2:
                raise ValueError("Configured bunch must have shape (N, 6) with at least two particles")
            result = self.track(particles, **kwargs)
        output = output_path if output_path is not None else cfg["run"].get("output")
        if output is not None:
            result.save(output)
        return result

    def generate_bunch(
        self, n: int,
        beta_x0: float, alpha_x0: float, epsx_m: float,
        beta_y0: float, alpha_y0: float, epsy_m: float,
        sigma_tau_m: float, sigma_delta: float,
        Dx0: float = 0.0, Dpx0: float = 0.0, Dy0: float = 0.0, Dpy0: float = 0.0,
        chirp_per_m: float = 0.0, seed: int | None = None,
    ) -> torch.Tensor:
        """A Twiss-matched bunch at this beamline's own entrance
        (``tracking.distributions.gaussian_matched_6d``, converted to the native canonical
        convention), ready to pass straight into ``track()``."""
        from .tracking.distributions import gaussian_matched_6d
        from .tracking.native.elements import _canonical_from_slopes

        p = gaussian_matched_6d(
            n, beta_x0, alpha_x0, epsx_m, beta_y0, alpha_y0, epsy_m, sigma_tau_m, sigma_delta,
            D_x_m=Dx0, Dp_x=Dpx0, D_y_m=Dy0, Dp_y=Dpy0, chirp_per_m=chirp_per_m,
            device=self._tracker.device, dtype=self._tracker.dtype, seed=seed,
        )
        x, xp, y, yp, tau, delta = p.unbind(-1)
        px, py = _canonical_from_slopes(xp, yp, delta)
        return torch.stack([x, px, y, py, tau, delta], dim=-1)

    def track(
        self, particles6: torch.Tensor, return_trajectory: bool = False,
        reference: dict[str, float] | None = None, metadata: dict | None = None,
    ) -> BeamlineResult:
        """One pass through ``self.specs``. ``reference`` (optional):
        compared against the result's own ``scalars()`` immediately,
        stashed under ``metadata["reference_diff"]``."""
        if particles6.ndim != 2 or particles6.shape[1] != 6 or particles6.shape[0] < 1:
            raise ValueError("Bunch must have shape (N, 6) with at least one particle")
        if not bool(torch.isfinite(particles6).all()):
            raise ValueError("Bunch coordinates must be finite")
        if return_trajectory:
            particles_out, energy_out, trajectory, energy_trajectory = self._tracker.track(particles6, return_trajectory=True)
        else:
            particles_out, energy_out = self._tracker.track(particles6)
            trajectory = energy_trajectory = None

        slice_summary = None
        try:
            from .analysis.slices import slice_indices, slice_stats
        except ImportError:
            pass
        else:
            idx, edges, _ = slice_indices(particles_out[:, 4], n_slices=self.n_slices)
            n = particles_out.shape[0]
            charge_per_particle_C = self._tracker.bunch_charge_C / n if self._tracker.bunch_charge_C else 0.0
            stats = slice_stats(
                particles_out[:, 4], particles_out[:, 5], particles_out[:, 0], particles_out[:, 1],
                particles_out[:, 2], particles_out[:, 3], charge_per_particle_C, idx, edges,
                energy_eV=float(energy_out), mc2_eV=self.mc2_eV,
            )
            slice_summary = {k: v.detach().cpu().numpy() for k, v in stats.items()}

        result = BeamlineResult(
            particles_out=particles_out.detach().cpu().numpy(),
            energy_eV_out=float(energy_out),
            charge_C=self._tracker.bunch_charge_C,
            trajectory=trajectory.detach().cpu().numpy() if trajectory is not None else None,
            energy_trajectory=energy_trajectory.detach().cpu().numpy() if energy_trajectory is not None else None,
            slice_summary=slice_summary,
            metadata=dict(metadata or {}),
        )
        if reference is not None:
            result.metadata["reference_diff"] = result.compare_reference(reference)
        return result

    def track_with_injector(
        self, injector_or_deck: Injector | str | Path,
        n_particles: int | None = None, injector_kwargs: dict | None = None,
        **track_kwargs,
    ) -> BeamlineResult:
        """Compute track with injector; see the module documentation for details."""
        inj = injector_or_deck if isinstance(injector_or_deck, Injector) else Injector.from_astra(injector_or_deck)
        if inj.mc2_eV != self.mc2_eV:
            raise ValueError("Injector mc2_eV must match the beamline study")
        out = inj.track(n_particles=n_particles, return_particles=True, **(injector_kwargs or {}))
        if out["x"].numel() < 1:
            raise ValueError("Injector exit bunch must contain at least one particle")
        particles6, energy_eV, charge_C = injector_exit_to_canonical6(
            out, device=self._tracker.device, dtype=self._tracker.dtype, mc2_eV=inj.mc2_eV,
        )
        self.energy_eV = energy_eV
        self._tracker._set_reference_energy(energy_eV)
        self._tracker.bunch_charge_C = charge_C
        return self.track(particles6, **track_kwargs)

    @staticmethod
    def load_result(path: str | Path) -> BeamlineResult:
        return BeamlineResult.load(path)
