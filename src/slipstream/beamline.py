"""``BeamlineStudy`` -- the ``Study`` facade's sibling for the native
linac/injector family (E3, see
plans/essential/E3_NATIVE_FACADE_PLAN.md). ``Study`` (``main.py``) covers
the ring backends only; the entire validated
``tracking.native.linac.LinacTracker``/``tracking.native.injector.Injector``
family was previously reachable only via bespoke scripts
(``examples/clara_linac/run_clara_linac_demo.py``,
``Stories/Moustache/track_moustache.py``). No new physics -- orchestration
only, composition rather than a ``Study`` subclass (the constructor inputs
are genuinely different: an ``ElementSpec`` list, not a ``Lattice``).
"""
from __future__ import annotations

from pathlib import Path

import torch

from .constants import ELECTRON_MASS_EV
from .data.results import BeamlineResult
from .tracking.native.injector import Injector, injector_exit_to_canonical6
from .tracking.native.lattice_import import ElementSpec, parse_elegant_lattice
from .tracking.native.linac import LinacTracker, resolve_wakes


class BeamlineStudy:
    """One ``LinacTracker`` (created once, tracked many times -- the same
    amortization pattern ``LinacTracker`` itself documents), plus the
    everyday conveniences: slice analysis after a pass (via
    ``analysis.slices``, imported lazily -- absent/``None`` in the result
    if that module can't be imported), a Twiss-matched bunch generator, and
    optional injector composition.

    ``**tracker_kwargs`` forwards straight to ``LinacTracker`` (``n_kicks``,
    ``errors``, ``fiducial``, ``dipole_collective_subcycle``, ...) -- this
    class adds no new tracking knobs of its own."""

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

    @classmethod
    def from_lattice_file(
        cls, lte_path: str | Path, root_name: str, energy_eV: float,
        mc2_eV: float = ELECTRON_MASS_EV, wake_base_dir: str | Path = ".", **kwargs,
    ) -> "BeamlineStudy":
        """``parse_elegant_lattice`` + ``resolve_wakes`` (any ``WAKEFILE=``
        paths resolved relative to ``wake_base_dir``, default the current
        directory) -- a machine's own ``.lte`` straight into a study."""
        specs = resolve_wakes(parse_elegant_lattice(lte_path, root_name), base_dir=wake_base_dir)
        return cls(specs, energy_eV, mc2_eV=mc2_eV, **kwargs)

    def generate_bunch(
        self, n: int,
        beta_x0: float, alpha_x0: float, epsx_m: float,
        beta_y0: float, alpha_y0: float, epsy_m: float,
        sigma_tau_m: float, sigma_delta: float,
        Dx0: float = 0.0, Dpx0: float = 0.0, Dy0: float = 0.0, Dpy0: float = 0.0,
        chirp_per_m: float = 0.0, seed: int | None = None,
    ) -> torch.Tensor:
        """A Twiss-matched bunch at this beamline's own entrance
        (``tracking.distributions.gaussian_matched_6d``, converted to the
        native canonical convention), ready to pass straight into
        ``track()``.

        ``ponytail:`` matching at an arbitrary *downstream* point (via
        ``linac_twiss.linac_twiss_table``, which needs the exact same
        beta/alpha/D inputs as this function -- a linac has no periodic
        solve to shortcut that) isn't built here; propagate the desired
        point's own Twiss with ``linac_twiss_table`` yourself and pass
        those values in as ``beta_x0``/... if a later point is needed."""
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
        """Run an ``Injector`` (an instance, or an ASTRA deck path built
        via ``Injector.from_astra``) to its exit distribution, ballistically
        converted to this tracker's own canonical convention
        (``injector_exit_to_canonical6``), then straight into ``track()``.
        The injector's own summed exit charge overrides this study's
        ``bunch_charge_C`` for this call (an injector run has a real,
        computed charge; a beamline built standalone usually doesn't), and
        its own exit energy overrides ``self.energy_eV`` as the
        ``LinacTracker``'s reference energy -- ``track()`` must resume from
        where the injector actually left off, not this study's own
        construction-time value. Updates the tracker's own reference-energy
        tensor in place (``.fill_()``, not reassignment -- the
        ``torch.compile`` "changing-value argument forces a recompile"
        discipline this package follows throughout, see ``CLAUDE.md``)."""
        inj = injector_or_deck if isinstance(injector_or_deck, Injector) else Injector.from_astra(injector_or_deck)
        out = inj.track(n_particles=n_particles, return_particles=True, **(injector_kwargs or {}))
        particles6, energy_eV, charge_C = injector_exit_to_canonical6(
            out, device=self._tracker.device, dtype=self._tracker.dtype,
        )
        self.energy_eV = energy_eV
        self._tracker._energy_eV_t.fill_(energy_eV)
        self._tracker.bunch_charge_C = charge_C
        return self.track(particles6, **track_kwargs)

    @staticmethod
    def load_result(path: str | Path) -> BeamlineResult:
        return BeamlineResult.load(path)
