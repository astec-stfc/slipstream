"""Tracks one canonical single pass through ordered elements, with inline RF and optional structure
wakes, local LSC and CSR. See docs/modules/tracking/native/linac.md."""
from __future__ import annotations

import functools
import math
from dataclasses import replace
from pathlib import Path

import torch
import torch._dynamo

from ...constants import C_LIGHT, ELECTRON_MASS_EV
from ...wakefields.shortrange import ShortRangeWake
from ...wakefields.transverse import TransverseWake
from .collective import (
    _lsc_csr_delta_kicks,
    csr_steady_state_delta_kick,
    csr_drift_delta_kick_fast,
    csr_drift_delta_kick_retarded,
    csr_steady_state_delta_kick_fast,
    lsc_delta_kick,
    lsc_delta_kick_fast,
)
from .elements import _match_energy_to_centroid, _rebase_reference, reference_offset, track_element
from .errors import FiducialReference, SampledError, apply_errors
from .lattice_import import ElementSpec
from .radiation import bend_radiation_kick
from .tracker import _ElementParams, _apply_one_element, _build_params


def resolve_wakes(specs: list[ElementSpec], base_dir: str | Path = ".") -> list[ElementSpec]:
    """Compute resolve wakes; see the module documentation for details."""
    base_dir = Path(base_dir)
    out = []
    for s in specs:
        if s.kind != "rf" or not s.wakefile:
            out.append(s)
            continue
        path = base_dir / s.wakefile
        t_column = s.wake_t_column or "t"
        wake = ShortRangeWake.from_sdds_file(path, t_column=t_column, w_column=s.wake_wz_column) if s.wake_wz_column and s.include_z_wake else None
        wx = TransverseWake.from_sdds_file(path, t_column=t_column, w_column=s.wake_wx_column) if s.wake_wx_column and s.include_tr_wake else None
        wy = TransverseWake.from_sdds_file(path, t_column=t_column, w_column=s.wake_wy_column) if s.wake_wy_column and s.include_tr_wake else None
        out.append(replace(s, wake=wake, transverse_wake_x=wx, transverse_wake_y=wy))
    return out


def _apply_collective_kick(
    particles6: torch.Tensor, energy_eV: torch.Tensor, mc2_eV: torch.Tensor,
    charge_per_particle_C: torch.Tensor | float, length_m: float,
    lsc: bool, csr: bool, csr_rho_m: float, csr_length_m: float, csr_drift_start_m: float | None,
    pipe_radius_b_m: float | None, lsc_radius_factor: float,
    csr_softening_m: float | None, compile_collective_chunks: bool = False,
    fast_collective_effects: bool = False, csr_bins: int = 256, csr_sg_halfwidth: int = 1,
    csr_slippage_m: float | None = None,
) -> torch.Tensor:
    """Compute apply collective kick; see the module documentation for details."""
    delta = particles6[..., 5]
    eff_length_m = csr_length_m
    if csr and csr_rho_m != 0.0 and csr_drift_start_m is not None:
        sigma_z_m = particles6[..., 4].std()
        overtaking_length_m = (24.0 * sigma_z_m * csr_rho_m * csr_rho_m).clamp(min=0.0) ** (1.0 / 3.0)
        eff_length_m = torch.clamp(overtaking_length_m - csr_drift_start_m, min=0.0, max=csr_length_m)

    if lsc and csr and csr_rho_m != 0.0 and not fast_collective_effects:
        lsc_kick, csr_kick = _lsc_csr_delta_kicks(
            particles6[..., 4], charge_per_particle_C, particles6[..., 0], particles6[..., 2],
            energy_eV, mc2_eV, length_m, csr_rho_m, eff_length_m,
            pipe_radius_b_m, lsc_radius_factor, csr_softening_m, compile_collective_chunks,
        )
        delta = delta + lsc_kick + csr_kick
        return torch.cat([particles6[..., :5], delta.unsqueeze(-1)], dim=-1)

    if lsc:
        if fast_collective_effects:
            lsc_kick = lsc_delta_kick_fast(
                particles6[..., 4], charge_per_particle_C, particles6[..., 0], particles6[..., 2],
                energy_eV, mc2_eV, length_m, pipe_radius_b_m=pipe_radius_b_m, radius_factor=lsc_radius_factor,
            )
        else:
            lsc_kick = lsc_delta_kick(
                particles6[..., 4], charge_per_particle_C, particles6[..., 0], particles6[..., 2],
                energy_eV, mc2_eV, length_m, pipe_radius_b_m=pipe_radius_b_m, radius_factor=lsc_radius_factor,
                compile_chunks=compile_collective_chunks,
            )
        delta = delta + lsc_kick
    if csr and csr_rho_m != 0.0:
        if fast_collective_effects:
            csr_kick = csr_steady_state_delta_kick_fast(
                particles6[..., 4], charge_per_particle_C, csr_rho_m, energy_eV, eff_length_m, softening_m=csr_softening_m,
                n_bins=csr_bins, sg_halfwidth=csr_sg_halfwidth, slippage_m=csr_slippage_m,
            )
        else:
            csr_kick = csr_steady_state_delta_kick(
                particles6[..., 4], charge_per_particle_C, csr_rho_m, energy_eV, eff_length_m, softening_m=csr_softening_m,
                compile_chunks=compile_collective_chunks,
            )
        delta = delta + csr_kick
    return torch.cat([particles6[..., :5], delta.unsqueeze(-1)], dim=-1)


def _maybe_collective_kick(
    particles6: torch.Tensor, p: _ElementParams, energy_eV: torch.Tensor, mc2_eV: torch.Tensor,
    charge_per_particle_C: torch.Tensor | float, lsc: bool, csr: bool, pipe_radius_b_m: float | None,
    lsc_radius_factor: float, csr_softening_m: float | None, compile_collective_chunks: bool = False,
    fast_collective_effects: bool = False,
) -> torch.Tensor:
    """Compute maybe collective kick; see the module documentation for details."""
    csr_active = csr and (p.csr or p.csr_drift_rho_m is not None)
    if not lsc and not csr_active:
        return particles6
    csr_drift_start_m = None
    csr_slippage_m = None
    if p.csr:
        csr_rho_m, csr_length_m = (p.length / p.angle if p.angle != 0.0 else 0.0), p.length
        if p.csr_transient and fast_collective_effects and p.angle != 0.0:
            csr_slippage_m = abs(csr_rho_m) * p.csr_phi_eval ** 3 / 24.0  # Saldin et al.: rho phi^3 / 24
    elif p.csr_drift_rho_m is not None:
        csr_rho_m, csr_length_m, csr_drift_start_m = p.csr_drift_rho_m, p.length, p.csr_drift_start_m
    else:
        csr_rho_m, csr_length_m = 0.0, 0.0
    return _apply_collective_kick(
        particles6, energy_eV, mc2_eV, charge_per_particle_C, p.length,
        lsc, csr_active, csr_rho_m, csr_length_m, csr_drift_start_m,
        pipe_radius_b_m, lsc_radius_factor, csr_softening_m,
        compile_collective_chunks, fast_collective_effects, p.csr_bins, p.csr_sg_halfwidth, csr_slippage_m,
    )


def _track_csr_drift_stupakov(
    particles6: torch.Tensor, p: _ElementParams, energy_eV: torch.Tensor, mc2_eV: torch.Tensor,
    charge_per_particle_C: torch.Tensor | float, lsc: bool,
    pipe_radius_b_m: float | None, lsc_radius_factor: float, csr_softening_m: float | None,
    compile_collective_chunks: bool, fast_collective_effects: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A CSR drift in kicks of ``csr_drift_dz``: half a step of drift, then a Stupakov-model kick at the centre of every step
    (the wake is evaluated for the distribution as it stands there), whole steps in between, half a step at the end."""
    if p.csr_stupakov == "retarded":
        # the retarded kernel is smooth in the distance behind the bend, so the steps grow with it: short close to the exit, where the
        # wake changes fastest, and up to ten times ``csr_drift_dz`` far from it
        steps, z = [], p.csr_drift_start_m
        end = p.csr_drift_start_m + p.length
        while end - z > 1e-12:
            h = min(max(0.2 * (z + p.csr_drift_dz), 0.5 * p.csr_drift_dz), 10.0 * p.csr_drift_dz, end - z)
            if end - (z + h) < 0.5 * h:
                h = end - z
            steps.append(h)
            z += h
    else:
        n = max(1, int(p.length / p.csr_drift_dz + 0.5))
        steps = [p.length / n] * n
    z_travel = p.csr_drift_start_m
    particles6, energy_eV = _apply_one_element(particles6, p._replace(length=0.5 * steps[0]), energy_eV, mc2_eV, charge_per_particle_C)
    for k, step in enumerate(steps):
        center = z_travel + 0.5 * step
        if p.csr_stupakov == "retarded":
            kick = csr_drift_delta_kick_retarded(
                particles6[..., 4], charge_per_particle_C, p.csr_drift_rho_m, p.csr_drift_phi_m, center, step, energy_eV,
                mc2_eV=float(mc2_eV), n_bins=p.csr_bins, sg_halfwidth=p.csr_sg_halfwidth,
            )
        else:
            kick = csr_drift_delta_kick_fast(
                particles6[..., 4], charge_per_particle_C, p.csr_drift_rho_m, p.csr_drift_phi_m, center, step, energy_eV,
                n_bins=p.csr_bins, sg_halfwidth=p.csr_sg_halfwidth,
            )
        particles6 = torch.cat([particles6[..., :5], (particles6[..., 5] + kick).unsqueeze(-1)], dim=-1)
        z_travel += step
        run = 0.5 * step + (0.5 * steps[k + 1] if k + 1 < len(steps) else 0.0)
        particles6, energy_eV = _apply_one_element(particles6, p._replace(length=run), energy_eV, mc2_eV, charge_per_particle_C)
    if lsc:
        particles6 = _maybe_collective_kick(
            particles6, p, energy_eV, mc2_eV, charge_per_particle_C, lsc, False,
            pipe_radius_b_m, lsc_radius_factor, csr_softening_m, compile_collective_chunks, fast_collective_effects,
        )
    return particles6, energy_eV


def _track_dipole_subcycled(
    particles6: torch.Tensor, p: _ElementParams, energy_eV: torch.Tensor, mc2_eV: torch.Tensor,
    charge_per_particle_C: torch.Tensor | float, lsc: bool, csr: bool,
    pipe_radius_b_m: float | None, lsc_radius_factor: float, csr_softening_m: float | None,
    compile_collective_chunks: bool, fast_collective_effects: bool,
) -> torch.Tensor:
    """Compute track dipole subcycled; see the module documentation for details."""
    n = p.n_kicks
    sub_length = p.length / n
    sub_angle = p.angle / n
    csr_active = csr and p.csr
    csr_rho_m = (p.length / p.angle) if (csr_active and p.angle != 0.0) else 0.0
    for i in range(n):
        particles6 = track_element(
            particles6, sub_length, sub_angle, p.k1, p.k2, p.e1, p.e2, energy_eV, mc2_eV,
            n_kicks=1, integration_order=p.integration_order,
            hgap=p.hgap, fint=p.fint, edge_order=p.edge_order,
            edge1_effects=p.edge1_effects if i == 0 else 0,
            edge2_effects=p.edge2_effects if i == n - 1 else 0,
            k3=p.k3, multipoles=p.multipoles, skew_multipoles=p.skew_multipoles,
            exact_dipole=p.exact_dipole,
        )
        particles6 = _apply_collective_kick(
            particles6, energy_eV, mc2_eV, charge_per_particle_C, sub_length,
            lsc, csr_active, csr_rho_m, sub_length, None,
            pipe_radius_b_m, lsc_radius_factor, csr_softening_m,
            compile_collective_chunks, fast_collective_effects, p.csr_bins, p.csr_sg_halfwidth,
        )
    return particles6 - p.offset


def _track_one_element(
    particles6: torch.Tensor, p: _ElementParams, energy_eV: torch.Tensor, mc2_eV: torch.Tensor,
    always_change_p0: bool, charge_per_particle_C: torch.Tensor | float,
    lsc: bool, csr: bool, pipe_radius_b_m: float | None, lsc_radius_factor: float,
    csr_softening_m: float | None, compile_rf_sub_kicks: bool = False,
    compile_collective_chunks: bool = False, fast_collective_effects: bool = False,
    dipole_collective_subcycle: bool = False,
    radiation_generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One element's worth of work -- transport, then (if active) the collective kick, then (if
    ``always_change_p0``) recentering."""
    if p.radiation_substeps:
        for before, after in p.radiation_substeps:
            particles6, energy_eV = _apply_one_element(particles6, before, energy_eV, mc2_eV)
            particles6 = bend_radiation_kick(
                particles6, energy_eV, p.angle / p.length, 2 * before.length,
                synch_rad=p.synch_rad, isr=p.isr, generator=radiation_generator,
            )
            particles6, energy_eV = _apply_one_element(particles6, after, energy_eV, mc2_eV)
        particles6 = _maybe_collective_kick(
            particles6, p, energy_eV, mc2_eV, charge_per_particle_C, lsc, csr,
            pipe_radius_b_m, lsc_radius_factor, csr_softening_m, compile_collective_chunks,
            fast_collective_effects,
        )
    elif (dipole_collective_subcycle and p.angle != 0 and p.n_kicks > 1
            and (lsc or (csr and p.csr)) and p.dx == p.dy == p.tilt == 0):
        particles6 = _track_dipole_subcycled(
            particles6, p, energy_eV, mc2_eV, charge_per_particle_C, lsc, csr,
            pipe_radius_b_m, lsc_radius_factor, csr_softening_m,
            compile_collective_chunks, fast_collective_effects,
        )
    elif p.csr_stupakov and csr and p.csr_drift_rho_m is not None and fast_collective_effects:
        particles6, energy_eV = _track_csr_drift_stupakov(
            particles6, p, energy_eV, mc2_eV, charge_per_particle_C, lsc, pipe_radius_b_m, lsc_radius_factor,
            csr_softening_m, compile_collective_chunks, fast_collective_effects,
        )
    elif p.csr_substeps and csr:
        # Keep the reference momentum fixed inside a magnet. Apply physical edge maps only at the two external faces.
        for sub in p.csr_substeps:
            particles6, energy_eV = _track_one_element(
                particles6, sub, energy_eV, mc2_eV, False, charge_per_particle_C,
                False, True, pipe_radius_b_m, lsc_radius_factor, csr_softening_m,
                compile_rf_sub_kicks, compile_collective_chunks, fast_collective_effects,
            )
        # LSC is still applied once over the full element, never twice.
        particles6 = _maybe_collective_kick(
            particles6, p, energy_eV, mc2_eV, charge_per_particle_C, lsc, False,
            pipe_radius_b_m, lsc_radius_factor, csr_softening_m, compile_collective_chunks,
            fast_collective_effects,
        )
    else:
        particles6, energy_eV = _apply_one_element(
            particles6, p, energy_eV, mc2_eV, charge_per_particle_C, compile_rf_sub_kicks,
        )
        particles6 = _maybe_collective_kick(
            particles6, p, energy_eV, mc2_eV, charge_per_particle_C, lsc, csr,
            pipe_radius_b_m, lsc_radius_factor, csr_softening_m, compile_collective_chunks,
            fast_collective_effects,
        )
    if always_change_p0:
        particles6, energy_eV = _match_energy_to_centroid(particles6, energy_eV, mc2_eV)
    return particles6, energy_eV


def _track_single_pass(
    particles6: torch.Tensor, params: list, energy_eV: torch.Tensor, mc2_eV: torch.Tensor,
    always_change_p0: bool, charge_per_particle_C: torch.Tensor | float = 0.0,
    lsc: bool = False, csr: bool = False, pipe_radius_b_m: float | None = None,
    lsc_radius_factor: float = 1.7, csr_softening_m: float | None = None,
    step_fn=_track_one_element, rf_step_fn=_track_one_element,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute track single pass; see the module documentation for details."""
    for p in params:
        fn = rf_step_fn if p.is_rf else step_fn
        particles6, energy_eV = fn(
            particles6, p, energy_eV, mc2_eV, always_change_p0, charge_per_particle_C,
            lsc, csr, pipe_radius_b_m, lsc_radius_factor, csr_softening_m,
        )
    return particles6, energy_eV


class LinacTracker:
    """One-pass transport with inline RF and optional wakes, LSC and CSR."""

    def __init__(
        self, specs: list[ElementSpec], energy_eV: float,
        mc2_eV: float = ELECTRON_MASS_EV,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float64,
        n_kicks: int = 1,
        n_kicks_bend: int = 4,
        integration_order: int = 4,
        compile: bool = False,
        compile_mode: str | None = None,
        always_change_p0: bool = True,
        errors: list[SampledError] | None = None,
        fiducial: FiducialReference | None = None,
        bunch_charge_C: float = 0.0,
        lsc: bool = False,
        csr: bool = False,
        pipe_radius_b_m: float | None = None,
        lsc_radius_factor: float = 1.7,
        csr_softening_m: float | None = None,
        compile_collective_chunks: bool | None = None,
        fast_collective_effects: bool = False,
        dipole_collective_subcycle: bool = False,
        csr_transient: bool = False,
        csr_drift_model: str = "overtaking",
        generator: torch.Generator | None = None,
    ):
        radiating = any(s.synch_rad or s.isr for s in specs)
        if radiating:
            if (compile or always_change_p0 or mc2_eV != ELECTRON_MASS_EV
                    or any(s.kind == "rf" and s.change_p0 for s in specs)):
                raise NotImplementedError("Local radiation requires eager LinacTracker, electron mass and always_change_p0=False")
            if energy_eV / mc2_eV < 1000 or n_kicks_bend < 1:
                raise ValueError("Local radiation requires gamma >= 1000 and positive n_kicks_bend")
        self.specs = specs  # the original, unperturbed list -- see fiducialize()
        self.mc2_eV = mc2_eV
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype
        self.always_change_p0 = always_change_p0
        self.errors = errors
        self.fiducial = fiducial
        self.bunch_charge_C = bunch_charge_C
        self.lsc = lsc
        self.csr = csr
        self.pipe_radius_b_m = pipe_radius_b_m
        self.lsc_radius_factor = lsc_radius_factor
        self.csr_softening_m = csr_softening_m
        self.fast_collective_effects = fast_collective_effects
        self.dipole_collective_subcycle = dipole_collective_subcycle
        collective_chunks = (
            compile or self.device.type == "cuda"
            if compile_collective_chunks is None
            else compile_collective_chunks
        )

        effective_specs = apply_errors(specs, errors) if errors else specs
        if fiducial is not None:
            effective_specs = [
                replace(s, phase_rad=s.phase_rad - 2.0 * math.pi * s.frequency_Hz * fiducial.t_ref_s)
                if s.kind == "rf" else s
                for s in effective_specs
            ]
        self._element_params = _build_params(
            effective_specs, n_kicks, n_kicks_bend, energy_eV, mc2_eV, self.device, dtype,
            integration_order=integration_order, rf_mode="per_element",
            change_p0_override=False if always_change_p0 else None,
            allow_local_radiation=True,
        )
        for i, spec in enumerate(effective_specs):
            if not (spec.synch_rad or spec.isr):
                continue
            if (spec.kind != "bend" or spec.length <= 0 or spec.k1 or spec.k2 or spec.k3
                    or spec.multipoles or spec.skew_multipoles or spec.integrated_multipoles
                    or spec.dx or spec.dy or spec.tilt):
                raise NotImplementedError("Local radiation currently supports aligned, positive-length pure dipoles")
            if spec.angle == 0:
                continue
            halves = [replace(
                spec, length=spec.length / (2 * n_kicks_bend), angle=spec.angle / (2 * n_kicks_bend),
                edge1_effects=spec.edge1_effects if j == 0 else 0,
                edge2_effects=spec.edge2_effects if j == 2 * n_kicks_bend - 1 else 0,
                synch_rad=False, isr=False,
            ) for j in range(2 * n_kicks_bend)]
            maps = _build_params(halves, 1, 1, energy_eV, mc2_eV, self.device, dtype,
                                 integration_order=integration_order)
            self._element_params[i] = self._element_params[i]._replace(
                radiation_substeps=tuple(zip(maps[::2], maps[1::2])),
            )
        # Keep the substep shape stable when an injector/source supplies
        # charge after construction. Zero charge gives zero collective kick.
        if csr and fast_collective_effects:
            for i, spec in enumerate(effective_specs):
                if (not spec.csr or spec.csr_n_slices <= 1 or spec.angle == 0
                        or spec.dx != 0 or spec.dy != 0 or spec.tilt != 0):
                    continue
                count = spec.csr_n_slices
                slices = [replace(
                    spec, length=spec.length / count, angle=spec.angle / count,
                    edge1_effects=spec.edge1_effects if j == 0 else 0,
                    edge2_effects=spec.edge2_effects if j == count - 1 else 0,
                ) for j in range(count)]
                substeps = _build_params(
                    slices, n_kicks, max(1, math.ceil(n_kicks_bend / count)), energy_eV, mc2_eV, self.device, dtype,
                    integration_order=integration_order,
                )
                if csr_transient:  # the wake of each slice is evaluated at the angle accumulated up to its centre (midpoint rule)
                    substeps = [sp._replace(csr_transient=True, csr_phi_eval=abs(spec.angle) * (j + 0.5) / count)
                                for j, sp in enumerate(substeps)]
                self._element_params[i] = self._element_params[i]._replace(csr_substeps=tuple(substeps))
        if csr_drift_model not in ("overtaking", "stupakov", "retarded"):
            raise ValueError("csr_drift_model must be 'overtaking' (steady state cut at the overtaking length), 'stupakov' (the "
                             "closed-form model of elegant's CSRDRIFT) or 'retarded' (the 1D retarded field, no kernel approximation)")
        if csr_drift_model != "overtaking":
            if not (csr and fast_collective_effects):
                raise ValueError("csr_drift_model other than 'overtaking' needs csr=True and fast_collective_effects=True")
            for i in range(len(self._element_params)):
                q = self._element_params[i]
                if q.csr_drift_rho_m is not None and q.csr_drift_phi_m > 0 and q.length > 0:
                    self._element_params[i] = q._replace(csr_stupakov=csr_drift_model)
        if csr_transient:
            if not (csr and fast_collective_effects):
                raise ValueError("csr_transient needs csr=True and fast_collective_effects=True (the pairwise kernel is steady-state)")
            for i, spec in enumerate(effective_specs):
                if spec.csr and spec.angle != 0 and not self._element_params[i].csr_substeps:
                    self._element_params[i] = self._element_params[i]._replace(csr_transient=True, csr_phi_eval=0.5 * abs(spec.angle))
        for p in self._element_params:
            if p.wake is not None:
                p.wake.prepare(dtype, self.device)
            if p.transverse_wake_x is not None:
                p.transverse_wake_x.prepare(dtype, self.device)
            if p.transverse_wake_y is not None:
                p.transverse_wake_y.prepare(dtype, self.device)
        if compile:
            torch._dynamo.config.recompile_limit = max(
                torch._dynamo.config.recompile_limit, len(self._element_params) + 8,
            )
        _step_base = functools.partial(
            _track_one_element, compile_collective_chunks=collective_chunks,
            fast_collective_effects=fast_collective_effects,
            dipole_collective_subcycle=dipole_collective_subcycle,
            radiation_generator=generator,
        )
        self._step_fn = torch.compile(_step_base, mode=compile_mode) if compile else _step_base
        self._rf_step_fn = functools.partial(
            _track_one_element, compile_rf_sub_kicks=compile, compile_collective_chunks=collective_chunks,
            fast_collective_effects=fast_collective_effects,
            dipole_collective_subcycle=dipole_collective_subcycle,
            radiation_generator=generator,
        )
        self._track_single_pass = functools.partial(
            _track_single_pass, step_fn=self._step_fn, rf_step_fn=self._rf_step_fn,
        )

        self._energy_eV_t = torch.tensor(energy_eV, dtype=dtype, device=self.device)
        self._reference_energy_eV = energy_eV
        self._mc2_eV_t = torch.tensor(mc2_eV, dtype=dtype, device=self.device)
        self._charge_per_particle_C = torch.zeros((), dtype=dtype, device=self.device)

    def _set_reference_energy(self, energy_eV: float) -> None:
        """Update entrance energy and cached magnet reference offsets together."""
        if energy_eV != self._reference_energy_eV:
            def refresh(p):
                if p.is_rf or p.is_kick:
                    return p
                offset = reference_offset(
                    p.length, p.angle, p.k1, p.k2, p.e1, p.e2, energy_eV, self.mc2_eV,
                    p.n_kicks, p.integration_order, p.hgap, p.fint, p.edge_order,
                    p.edge1_effects, p.edge2_effects, p.exact_dipole,
                ).to(device=self.device, dtype=self.dtype).clone()
                return p._replace(
                    offset=offset, csr_substeps=tuple(refresh(sub) for sub in p.csr_substeps),
                    radiation_substeps=tuple((refresh(a), refresh(b)) for a, b in p.radiation_substeps),
                )

            self._element_params = [refresh(p) for p in self._element_params]
            self._reference_energy_eV = energy_eV
        self._energy_eV_t = torch.full_like(self._energy_eV_t, energy_eV)  # not in place: an earlier pass may have saved it

    def _ensure_transverse_wake_windows(self, particles6: torch.Tensor) -> None:
        """A transverse wake convolves on a fixed bin window. Give any that has none the time span of the first bunch tracked, with half that
        span as a margin on each side (elegant's ``N_BINS = 0`` sizes its bins from the bunch in a similar way)."""
        todo = {id(w): w for p in self._element_params for w in (p.transverse_wake_x, p.transverse_wake_y) if w is not None and w._fixed_n_bins is None}
        if not todo:
            return
        tau_s = particles6[..., 4] / C_LIGHT
        lo, hi = float(tau_s.min()), float(tau_s.max())
        pad = 0.5 * (hi - lo)
        for w in todo.values():
            w.set_fixed_bin_window(lo - pad, hi + pad)
            w.prepare(self.dtype, self.device)

    def track(
        self, particles6: torch.Tensor, return_trajectory: bool = False, reference_energy_eV: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One pass."""
        particles6 = particles6.to(device=self.device, dtype=self.dtype)
        if self.bunch_charge_C != 0.0:
            self._ensure_transverse_wake_windows(particles6)
        # A fresh tensor per call: autograd saves it, so an in-place refill would invalidate an earlier pass's backward.
        self._charge_per_particle_C = torch.full(
            (), self.bunch_charge_C / particles6.shape[0], dtype=self.dtype, device=self.device,
        )
        csr_active = self.csr and self.bunch_charge_C != 0.0
        if reference_energy_eV is not None:
            if any(p.synch_rad or p.isr for p in self._element_params):
                raise NotImplementedError("Local radiation currently requires a fixed reference energy")
            return self._track_design_reference(particles6, reference_energy_eV)
        if not return_trajectory:
            return self._track_single_pass(
                particles6, self._element_params, self._energy_eV_t, self._mc2_eV_t, self.always_change_p0,
                self._charge_per_particle_C, self.lsc, csr_active, self.pipe_radius_b_m,
                self.lsc_radius_factor, self.csr_softening_m,
            )
        trajectory = [particles6]
        energy_eV = self._energy_eV_t
        energy_trajectory = [energy_eV]
        for p in self._element_params:
            fn = self._rf_step_fn if p.is_rf else self._step_fn
            particles6, energy_eV = fn(
                particles6, p, energy_eV, self._mc2_eV_t, self.always_change_p0, self._charge_per_particle_C,
                self.lsc, csr_active, self.pipe_radius_b_m, self.lsc_radius_factor, self.csr_softening_m,
            )
            trajectory.append(particles6)
            energy_trajectory.append(energy_eV)
        return particles6, energy_eV, torch.stack(trajectory, dim=0), torch.stack(energy_trajectory, dim=0)

    def _track_design_reference(
        self, particles6: torch.Tensor, reference_energy_eV: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute track design reference; see the module documentation for details."""
        prof = torch.as_tensor(reference_energy_eV, dtype=self.dtype, device=self.device)
        if prof.shape[0] != len(self._element_params) + 1:
            raise ValueError(f"reference_energy_eV needs {len(self._element_params) + 1} entries, got {prof.shape[0]}")
        for i, p in enumerate(self._element_params):
            fn = self._rf_step_fn if p.is_rf else self._step_fn
            particles6, e_out = fn(
                particles6, p, prof[i], self._mc2_eV_t, False, self._charge_per_particle_C,
                self.lsc, self.csr and self.bunch_charge_C != 0.0,
                self.pipe_radius_b_m, self.lsc_radius_factor, self.csr_softening_m,
            )
            particles6 = _rebase_reference(particles6, e_out, prof[i + 1], self._mc2_eV_t)
        return particles6, prof[-1]

    def fiducialize(self, particles0: torch.Tensor) -> FiducialReference:
        """Compute fiducialize; see the module documentation for details."""
        particles0 = particles0.to(device=self.device, dtype=self.dtype)
        trajectory = [particles0]
        state = particles0
        energy_eV = self._energy_eV_t
        for p in self._element_params:
            state, energy_eV = self._step_fn(
                state, p, energy_eV, self._mc2_eV_t, self.always_change_p0, 0.0,
                self.lsc, False, self.pipe_radius_b_m, self.lsc_radius_factor, self.csr_softening_m,
            )
            trajectory.append(state)
        t_ref_s = (state[..., 4] / C_LIGHT).mean().item()
        return FiducialReference(trajectory=torch.stack(trajectory, dim=0), t_ref_s=t_ref_s)
