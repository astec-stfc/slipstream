"""Integrates actual per-element canonical ring transport on every turn and composes native RF,
radiation damping/excitation, wakes, optional IBS and ramps. See
docs/modules/tracking/native/tracker.md."""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import torch

from ... import ibs as ibs_module
from ...constants import C_LIGHT, E_CHARGE_C, R_E_M
from ...lattice.lattice import Lattice
from ...lattice.optics import TransverseOptics
from ...lattice.ramp import RampProfile, scaled_lattice, scaled_optics
from ...wakefields.shortrange import ShortRangeWake
from ...wakefields.transverse import TransverseWake
from .ring_physics import (
    damping_excitation,
    ibs_effective_taus,
    rf_kick,
    rms_emittance,
    transverse_damping_excitation,
)
from .elements import _corrector_kick, _rf_cavity_kick, reference_offset, track_element
from .errors import FiducialReference, SampledError, apply_errors, resolve_rf_errors
from .lattice_import import ElementSpec
from .optics import periodic_twiss_table


class NativeTrackResult(NamedTuple):
    """Turn-indexed native tracking diagnostics."""

    turn: np.ndarray
    E0_eV: np.ndarray
    sigma_delta: np.ndarray
    sigma_tau_s: np.ndarray
    sigma_x_m: np.ndarray
    sigma_y_m: np.ndarray


class _ElementParams(NamedTuple):
    """Compute ElementParams; see the module documentation for details."""

    length: float
    angle: float
    k1: float
    k2: float
    e1: float
    e2: float
    n_kicks: int
    """Magnetic element: Yoshida sub-step count. RF-kind element
    (``is_rf=True``): the cavity's own real physical cell count."""
    integration_order: int
    offset: torch.Tensor
    dx: float = 0.0
    dy: float = 0.0
    tilt: float = 0.0
    is_rf: bool = False
    voltage_V: float = 0.0
    phase_rad: float = 0.0
    omega_rf: float = 0.0
    change_p0: bool = False
    wake: ShortRangeWake | None = None
    transverse_wake_x: TransverseWake | None = None
    transverse_wake_y: TransverseWake | None = None
    beta_wave: float = 1.0
    end1_focus: bool = False
    end2_focus: bool = False
    body_focus_model: str = "none"
    csr: bool = False
    """``ElementSpec.csr``, threaded through unchanged -- only read by
    ``LinacTracker``'s own collective-effects loop; ``NativeTracker``
    itself never reads this field."""
    csr_drift_rho_m: float | None = None
    csr_drift_start_m: float = 0.0
    """``ElementSpec.csr_drift_rho_m``/``csr_drift_start_m``, threaded
    through unchanged -- same ``LinacTracker``-only readership as ``csr``
    above."""
    hgap: float = 0.0
    fint: float = 0.5
    edge_order: int = 1
    edge1_effects: int = 1
    edge2_effects: int = 1
    csr_bins: int = 256
    csr_sg_halfwidth: int = 1
    csr_substeps: tuple = ()
    csr_drift_phi_m: float = 0.0
    csr_drift_dz: float = 0.01
    csr_stupakov: str = ""
    """``LinacTracker(csr_drift_model=...)`` when ``"stupakov"`` or ``"retarded"``: this CSR drift is tracked in ``csr_drift_dz`` kicks with that kernel."""
    csr_transient: bool = False
    """``LinacTracker(csr_transient=True)``: this element's CSR kick uses the entrance-transient model (fast kernel only)."""
    csr_phi_end: float = 0.0
    """Bend angle accumulated from the entrance of the dipole to the end of this slice (rad), for the slippage length."""
    exact_dipole: bool = False
    k3: float = 0.0
    multipoles: tuple = ()
    skew_multipoles: tuple = ()
    integrated_multipoles: tuple = ()
    """``elements.track_element``'s ``exact_dipole`` -- only
    ``ring.RingMap`` sets it."""
    is_kick: bool = False
    """``ElementSpec.kind in ("hkick", "vkick")`` -- dispatches to
    ``elements._corrector_kick`` instead of ``track_element``/``_rf_cavity_kick``."""
    kick_plane: str = "x"
    """``"x"`` (from ``kind="hkick"``) or ``"y"`` (``kind="vkick"``) --
    fixed structurally at parse/build time, never inferred from
    ``kick_value`` (which must stay well-defined, including as a
    differentiable tensor, exactly at its nominal zero value)."""
    kick_value: float | torch.Tensor = 0.0
    """The corrector's own strength (``ElementSpec.kick_x``/``kick_y``,
    whichever ``kick_plane`` selects) -- ``RingMap.with_strengths``'s
    occurrence-safe override target."""
    synch_rad: bool = False
    isr: bool = False
    radiation_substeps: tuple = ()


def _build_params(
    specs: list[ElementSpec], n_kicks: int, n_kicks_bend: int,
    energy_eV: float, mc2_eV: float, device: torch.device, dtype: torch.dtype,
    integration_order: int = 4, rf_mode: str = "lumped", change_p0_override: bool | None = None,
    exact_dipole: bool = False, allow_local_radiation: bool = False,
) -> list[_ElementParams]:
    """Compute build params; see the module documentation for details."""
    out = []
    for s in specs:
        if (s.synch_rad or s.isr) and not allow_local_radiation:
            raise NotImplementedError("Local bend radiation requires LinacTracker; ring radiation accounting is separate")
        if s.kind == "rf" and rf_mode == "per_element":
            change_p0 = s.change_p0 if change_p0_override is None else change_p0_override
            out.append(_ElementParams(
                s.length, 0.0, 0.0, 0.0, 0.0, 0.0, s.n_kicks, integration_order,
                torch.zeros(6, device=device, dtype=dtype),
                is_rf=True, voltage_V=s.voltage_V, phase_rad=s.phase_rad,
                omega_rf=2.0 * math.pi * s.frequency_Hz, change_p0=change_p0,
                wake=s.wake, transverse_wake_x=s.transverse_wake_x, transverse_wake_y=s.transverse_wake_y,
                beta_wave=s.beta_wave,
                end1_focus=s.end1_focus, end2_focus=s.end2_focus, body_focus_model=s.body_focus_model,
            ))
            continue
        if s.kind in ("hkick", "vkick"):
            plane = "x" if s.kind == "hkick" else "y"
            out.append(_ElementParams(
                s.length, 0.0, 0.0, 0.0, 0.0, 0.0, 1, integration_order,
                torch.zeros(6, device=device, dtype=dtype),
                is_kick=True, kick_plane=plane, kick_value=s.kick_x if plane == "x" else s.kick_y,
            ))
            continue
        nk = n_kicks_bend if s.angle != 0.0 else n_kicks
        # dx=dy=tilt=0 always -- see _ElementParams's own docstring.
        offset = reference_offset(
            s.length, s.angle, s.k1, s.k2, s.e1, s.e2, energy_eV, mc2_eV, nk, integration_order,
            s.hgap, s.fint, s.edge_order, s.edge1_effects, s.edge2_effects, exact_dipole,
        )
        out.append(_ElementParams(
            s.length, s.angle, s.k1, s.k2, s.e1, s.e2, nk, integration_order,
            offset.to(device=device, dtype=dtype), dx=s.dx, dy=s.dy, tilt=s.tilt, csr=s.csr,
            hgap=s.hgap, fint=s.fint, edge_order=s.edge_order,
            edge1_effects=s.edge1_effects, edge2_effects=s.edge2_effects,
            csr_bins=s.csr_bins, csr_sg_halfwidth=s.csr_sg_halfwidth,
            csr_drift_rho_m=s.csr_drift_rho_m, csr_drift_start_m=s.csr_drift_start_m,
            csr_drift_phi_m=s.csr_drift_phi_m, csr_drift_dz=s.csr_drift_dz,
            exact_dipole=exact_dipole,
            k3=s.k3, multipoles=s.multipoles, skew_multipoles=s.skew_multipoles,
            integrated_multipoles=s.integrated_multipoles,
            synch_rad=s.synch_rad, isr=s.isr,
        ))
    return out


def _apply_one_element(
    particles6: torch.Tensor, p: _ElementParams, energy_eV: torch.Tensor, mc2_eV: torch.Tensor,
    charge_per_particle_C: torch.Tensor | float = 0.0, compile_rf_sub_kicks: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch a single element -- RF-kind params go through ``elements._rf_cavity_kick`` (which can
    change ``energy_eV``); everything else through ``elements.track_element`` (``energy_eV`` passed
    through unchanged)."""
    if p.is_kick:
        return _corrector_kick(particles6, p.length, p.kick_value, p.kick_plane, energy_eV, mc2_eV), energy_eV
    if p.is_rf:
        return _rf_cavity_kick(
            particles6, p.length, p.voltage_V, p.phase_rad, p.omega_rf, energy_eV, mc2_eV, change_p0=p.change_p0,
            wake=p.wake, transverse_wake_x=p.transverse_wake_x, transverse_wake_y=p.transverse_wake_y,
            charge_per_particle_C=charge_per_particle_C, n_kicks=p.n_kicks, beta_wave=p.beta_wave,
            end1_focus=p.end1_focus, end2_focus=p.end2_focus, body_focus_model=p.body_focus_model,
            compile_sub_kicks=compile_rf_sub_kicks,
        )
    state = track_element(
        particles6, p.length, p.angle, p.k1, p.k2, p.e1, p.e2, energy_eV, mc2_eV,
        n_kicks=p.n_kicks, integration_order=p.integration_order, dx=p.dx, dy=p.dy, tilt=p.tilt,
        hgap=p.hgap, fint=p.fint, edge_order=p.edge_order,
        edge1_effects=p.edge1_effects, edge2_effects=p.edge2_effects, exact_dipole=p.exact_dipole,
        k3=p.k3, multipoles=p.multipoles, skew_multipoles=p.skew_multipoles,
        integrated_multipoles=p.integrated_multipoles,
    ) - p.offset
    return state, energy_eV


def _apply_elements(
    particles6: torch.Tensor, params: list[_ElementParams], energy_eV: torch.Tensor, mc2_eV: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply every element's transport in sequence -- a plain Python loop over a fixed-length list,
    ``torch.compile``-friendly (no object construction, no ``.item()`` calls)."""
    for p in params:
        particles6, energy_eV = _apply_one_element(particles6, p, energy_eV, mc2_eV)
    return particles6, energy_eV


def _rf_and_damping(
    particles6: torch.Tensor, params: Lattice, noise_delta: torch.Tensor,
    Vrf_eff_V: torch.Tensor | None = None, omega_rf_eff: torch.Tensor | None = None,
    phi_s_eff: torch.Tensor | None = None, E0_eff_eV: torch.Tensor | None = None,
    tau_z_eff_s: torch.Tensor | None = None, sigma_delta0_eff: torch.Tensor | None = None,
    optics: TransverseOptics | None = None,
    noise_xp: torch.Tensor | None = None, noise_yp: torch.Tensor | None = None,
    tau_x_eff_s: torch.Tensor | None = None, tau_y_eff_s: torch.Tensor | None = None,
    epsx_eff_m: torch.Tensor | None = None, epsy_eff_m: torch.Tensor | None = None,
    wake: ShortRangeWake | None = None,
    transverse_wake_x: TransverseWake | None = None, transverse_wake_y: TransverseWake | None = None,
    charge_per_particle_C: torch.Tensor | float = 0.0,
    skip_rf_kick: bool = False,
) -> torch.Tensor:
    """Compute rf and damping; see the module documentation for details."""
    tau_s = particles6[..., 4] / C_LIGHT
    E0_for_wake = E0_eff_eV if E0_eff_eV is not None else params.E0_eV
    if skip_rf_kick:
        delta_kicked = particles6[..., 5]
    else:
        delta_kicked = particles6[..., 5] + rf_kick(tau_s, params, Vrf_eff_V, omega_rf_eff, phi_s_eff, E0_eff_eV)
    if wake is not None:
        delta_kicked = delta_kicked - wake.voltage_at(tau_s, charge_per_particle_C) / E0_for_wake
    delta = damping_excitation(
        delta_kicked, params, tau_z_eff_s=tau_z_eff_s, sigma_delta0_eff=sigma_delta0_eff, noise=noise_delta,
    )
    if optics is None:
        return torch.cat([particles6[..., :5], delta.unsqueeze(-1)], dim=-1)

    px_kicked, py_kicked = particles6[..., 1], particles6[..., 3]
    if transverse_wake_x is not None:
        px_kicked = px_kicked + transverse_wake_x.kick_at(tau_s, particles6[..., 0], charge_per_particle_C) / E0_for_wake
    if transverse_wake_y is not None:
        py_kicked = py_kicked + transverse_wake_y.kick_at(tau_s, particles6[..., 2], charge_per_particle_C) / E0_for_wake

    px = transverse_damping_excitation(
        px_kicked, optics.tau_x_s, optics.epsx_m, optics.beta_x_m, optics.alpha_x, params.T0_s,
        tau_eff_s=tau_x_eff_s, eps_eff_m=epsx_eff_m, noise=noise_xp,
    )
    py = transverse_damping_excitation(
        py_kicked, optics.tau_y_s, optics.epsy_m, optics.beta_y_m, optics.alpha_y, params.T0_s,
        tau_eff_s=tau_y_eff_s, eps_eff_m=epsy_eff_m, noise=noise_yp,
    )
    return torch.cat([
        particles6[..., 0:1], px.unsqueeze(-1), particles6[..., 2:3], py.unsqueeze(-1),
        particles6[..., 4:5], delta.unsqueeze(-1),
    ], dim=-1)


def _step(
    particles6: torch.Tensor, params: list[_ElementParams], energy_eV: torch.Tensor, mc2_eV: torch.Tensor,
    lattice: Lattice, noise_delta: torch.Tensor,
    Vrf_eff_V: torch.Tensor | None = None, omega_rf_eff: torch.Tensor | None = None,
    phi_s_eff: torch.Tensor | None = None, E0_eff_eV: torch.Tensor | None = None,
    tau_z_eff_s: torch.Tensor | None = None, sigma_delta0_eff: torch.Tensor | None = None,
    optics: TransverseOptics | None = None,
    noise_xp: torch.Tensor | None = None, noise_yp: torch.Tensor | None = None,
    tau_x_eff_s: torch.Tensor | None = None, tau_y_eff_s: torch.Tensor | None = None,
    epsx_eff_m: torch.Tensor | None = None, epsy_eff_m: torch.Tensor | None = None,
    wake: ShortRangeWake | None = None,
    transverse_wake_x: TransverseWake | None = None, transverse_wake_y: TransverseWake | None = None,
    charge_per_particle_C: torch.Tensor | float = 0.0,
    skip_rf_kick: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute step; see the module documentation for details."""
    particles6, energy_eV = _apply_elements(particles6, params, energy_eV, mc2_eV)
    particles6 = _rf_and_damping(
        particles6, lattice, noise_delta,
        Vrf_eff_V=Vrf_eff_V, omega_rf_eff=omega_rf_eff, phi_s_eff=phi_s_eff,
        E0_eff_eV=energy_eV if skip_rf_kick else E0_eff_eV,
        tau_z_eff_s=tau_z_eff_s, sigma_delta0_eff=sigma_delta0_eff, optics=optics,
        noise_xp=noise_xp, noise_yp=noise_yp, tau_x_eff_s=tau_x_eff_s, tau_y_eff_s=tau_y_eff_s,
        epsx_eff_m=epsx_eff_m, epsy_eff_m=epsy_eff_m,
        wake=wake, transverse_wake_x=transverse_wake_x, transverse_wake_y=transverse_wake_y,
        charge_per_particle_C=charge_per_particle_C, skip_rf_kick=skip_rf_kick,
    )
    return particles6, energy_eV


class NativeTracker:
    """Ring transport with configured RF, radiation, wakes, IBS and ramps."""

    def __init__(
        self, specs: list[ElementSpec], params: Lattice,
        optics: TransverseOptics | None = None,
        ramp: RampProfile | None = None,
        generator: torch.Generator | None = None,
        mc2_eV: float | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float64,
        n_kicks: int = 1,
        n_kicks_bend: int = 4,
        integration_order: int = 4,
        compile: bool = False,
        compile_mode: str | None = None,
        wake: ShortRangeWake | None = None,
        transverse_wake_x: TransverseWake | None = None,
        transverse_wake_y: TransverseWake | None = None,
        bunch_current_A: float = 0.0,
        ibs: bool = False,
        ibs_recompute_every: int = 50,
        ibs_coulomb_log: float | None = None,
        ibs_model: str = "bane",
        ibs_particle_radius_m: float = R_E_M,
        errors: list[SampledError] | None = None,
        fiducial: FiducialReference | None = None,
        rf: str = "lumped",
        exact_dipole: bool = False,
    ):
        if rf not in ("lumped", "per_element"):
            raise ValueError(f"rf must be 'lumped' or 'per_element', got {rf!r}")
        if ibs and optics is None:
            raise ValueError("ibs=True needs optics= (IBS growth rates need beta/dispersion/emittance)")
        if ibs_model not in ("bane", "bjorken-mtingwa"):
            raise ValueError(f"ibs_model must be 'bane' or 'bjorken-mtingwa', got {ibs_model!r}")
        if ibs_model == "bjorken-mtingwa" and ibs_coulomb_log is None:
            raise ValueError("ibs_model='bjorken-mtingwa' needs an explicit ibs_coulomb_log=")
        self.specs = specs  # the ORIGINAL, unperturbed list -- see fiducialize()
        self.params = params
        self.optics = optics
        self.ramp = ramp
        self.generator = generator
        self.mc2_eV = mc2_eV if mc2_eV is not None else params.rest_mass_eV
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype
        self.n_kicks = n_kicks
        self.n_kicks_bend = n_kicks_bend
        self.integration_order = integration_order
        self._elapsed_turns = 0
        self._ramp_E0_prev = ramp.at(0.0).E0_eV if ramp is not None else None
        self.errors = errors
        self.fiducial = fiducial
        self.rf = rf
        self.exact_dipole = exact_dipole
        effective_specs = apply_errors(specs, errors) if errors else specs
        self._element_params = _build_params(
            effective_specs, n_kicks, n_kicks_bend, params.E0_eV, self.mc2_eV, self.device, dtype,
            integration_order=integration_order, rf_mode=rf, exact_dipole=exact_dipole,
        )
        self._step = torch.compile(_step, mode=compile_mode) if compile else _step
        self._ramp_overrides: dict[str, torch.Tensor] | None = None

        self._rf_error_delta = resolve_rf_errors(errors) if errors else {"phi_s": 0.0, "Vrf_V": 0.0, "omega_rf": 0.0}
        self._fiducial_t_ref_s = fiducial.t_ref_s if fiducial is not None else 0.0
        self._has_rf_overrides = bool(errors) or fiducial is not None
        if self._has_rf_overrides:
            omega_rf_fixed = params.omega_rf + self._rf_error_delta["omega_rf"]
            # -omega_rf*t_ref_s measures phase relative to the fiducial
            # pass's own frozen arrival time instead of the tau=0 design reference.
            phi_s_fixed = params.phi_s - omega_rf_fixed * self._fiducial_t_ref_s + self._rf_error_delta["phi_s"]
            self._Vrf_eff_t = torch.tensor(params.Vrf_V + self._rf_error_delta["Vrf_V"], dtype=dtype, device=self.device)
            self._omega_rf_eff_t = torch.tensor(omega_rf_fixed, dtype=dtype, device=self.device)
            self._phi_s_eff_t = torch.tensor(phi_s_fixed, dtype=dtype, device=self.device)
        # 0-d tensors, not plain floats, for recompile avoidance -- see docs/tracking_native.md's
        # "torch.compile discipline". Ramped tracking refills via .fill_() each turn.
        self._energy_eV_t = torch.tensor(params.E0_eV, dtype=dtype, device=self.device)
        self._mc2_eV_t = torch.tensor(self.mc2_eV, dtype=dtype, device=self.device)

        self.bunch_current_A = bunch_current_A
        self.wake = wake if bunch_current_A != 0.0 else None
        if self.wake is not None:
            self.wake.prepare(self.dtype, self.device)
        self.transverse_wake_x = transverse_wake_x if bunch_current_A != 0.0 else None
        self.transverse_wake_y = transverse_wake_y if bunch_current_A != 0.0 else None
        if self.transverse_wake_x is not None:
            self.transverse_wake_x.prepare(self.dtype, self.device)
        if self.transverse_wake_y is not None:
            self.transverse_wake_y.prepare(self.dtype, self.device)
        # Same recompile-avoidance reason as energy_eV above; filled in once track() knows n_particles.
        self._charge_per_particle_C = torch.zeros((), dtype=dtype, device=self.device)

        self.ibs = ibs
        self.ibs_recompute_every = ibs_recompute_every
        self.ibs_coulomb_log = ibs_coulomb_log
        self.ibs_model = ibs_model
        self.ibs_particle_radius_m = ibs_particle_radius_m
        self._n_bunch = bunch_current_A * params.T0_s / E_CHARGE_C
        self._ibs_twiss_table = None
        self._ibs_rates: ibs_module.IBSGrowthRates | None = None
        if self.ibs:
            self._ibs_twiss_table, _, _ = periodic_twiss_table(specs, params.E0_eV, self.mc2_eV)
            self._tau_z_ibs_eff = torch.tensor(params.tau_z_s, dtype=dtype, device=self.device)
            self._tau_x_ibs_eff = torch.tensor(optics.tau_x_s, dtype=dtype, device=self.device)
            self._tau_y_ibs_eff = torch.tensor(optics.tau_y_s, dtype=dtype, device=self.device)

    def _recompute_ibs_rates(self, particles6: torch.Tensor, energy_eV: float) -> None:
        """Compute recompute ibs rates; see the module documentation for details."""
        table = self._ibs_twiss_table
        x, px, y, py, tau, delta = particles6.unbind(-1)
        D_x, Dp_x = float(table.D_x_m[0]), float(table.Dp_x[0])
        D_y, Dp_y = float(table.D_y_m[0]), float(table.Dp_y[0])
        epsx_m = rms_emittance(x - D_x * delta, px - Dp_x * delta)
        epsy_m = rms_emittance(y - D_y * delta, py - Dp_y * delta)
        sigma_p = delta.std().item()
        sigma_z_m = particles6[..., 4].std().item()  # native tau is already path-length-like (meters)
        gamma = energy_eV / self.mc2_eV
        common = dict(
            gamma=gamma, n_particles_bunch=self._n_bunch, sigma_z_m=sigma_z_m, sigma_p=sigma_p,
            epsx_m=epsx_m, epsy_m=epsy_m, coulomb_log_value=self.ibs_coulomb_log,
        )
        if self.ibs_model == "bane":
            self._ibs_rates = ibs_module.element_averaged_growth_rates(table, model="bane", **common)
        else:
            beta_rel = math.sqrt(1.0 - 1.0 / gamma**2)
            self._ibs_rates = ibs_module.element_averaged_growth_rates(
                table, model="bjorken-mtingwa", beta_rel=beta_rel,
                particle_radius_m=self.ibs_particle_radius_m, **common,
            )

    def _draw_noise(self, particles6: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        shape, dtype, device = particles6.shape[:-1], particles6.dtype, particles6.device
        noise_delta = torch.randn(shape, generator=self.generator, dtype=dtype, device=device)
        if self.optics is None:
            return noise_delta, None, None
        noise_xp = torch.randn(shape, generator=self.generator, dtype=dtype, device=device)
        noise_yp = torch.randn(shape, generator=self.generator, dtype=dtype, device=device)
        return noise_delta, noise_xp, noise_yp

    def fiducialize(self, particles0: torch.Tensor) -> FiducialReference:
        """Compute fiducialize; see the module documentation for details."""
        particles0 = particles0.to(device=self.device, dtype=self.dtype)
        trajectory = [particles0]
        state = particles0
        energy_eV = self._energy_eV_t
        for p in self._element_params:
            state, energy_eV = _apply_one_element(state, p, energy_eV, self._mc2_eV_t)
            trajectory.append(state)
        t_ref_s = (state[..., 4] / C_LIGHT).mean().item()
        return FiducialReference(trajectory=torch.stack(trajectory, dim=0), t_ref_s=t_ref_s)

    def track(
        self, particles6: torch.Tensor, n_turns: int, sample_every: int = 1, progress: bool = False,
    ) -> tuple[torch.Tensor, NativeTrackResult]:
        """``progress=True`` shows a ``tqdm`` bar over the per-turn loop."""
        particles6 = particles6.to(device=self.device, dtype=self.dtype)
        if self.bunch_current_A != 0.0:
            n_particles = particles6.shape[0]
            self._charge_per_particle_C.fill_(self.params.T0_s * self.bunch_current_A / n_particles)
        if self.ramp is None:
            return self._track_fixed_energy(particles6, n_turns, sample_every, progress)
        return self._track_ramped(particles6, n_turns, sample_every, progress)

    def _track_fixed_energy(
        self, particles6: torch.Tensor, n_turns: int, sample_every: int, progress: bool = False,
        collect: bool = True,
    ) -> tuple[torch.Tensor, NativeTrackResult | None]:
        samples: list[tuple] = []

        turns = range(n_turns)
        if progress:
            from tqdm import tqdm
            turns = tqdm(turns, unit="turn", desc="tracking", total=n_turns)
        for turn in turns:
            if self.ibs and turn % self.ibs_recompute_every == 0:
                self._recompute_ibs_rates(particles6, self.params.E0_eV)
                tau_z, tau_x, tau_y = ibs_effective_taus(
                    self.params.tau_z_s, self.optics.tau_x_s, self.optics.tau_y_s, self._ibs_rates,
                )
                self._tau_z_ibs_eff.fill_(tau_z)
                self._tau_x_ibs_eff.fill_(tau_x)
                self._tau_y_ibs_eff.fill_(tau_y)
            noise_delta, noise_xp, noise_yp = self._draw_noise(particles6)
            particles6, self._energy_eV_t = self._step(
                particles6, self._element_params, self._energy_eV_t, self._mc2_eV_t, self.params, noise_delta,
                Vrf_eff_V=self._Vrf_eff_t if self._has_rf_overrides else None,
                omega_rf_eff=self._omega_rf_eff_t if self._has_rf_overrides else None,
                phi_s_eff=self._phi_s_eff_t if self._has_rf_overrides else None,
                optics=self.optics, noise_xp=noise_xp, noise_yp=noise_yp,
                wake=self.wake, transverse_wake_x=self.transverse_wake_x, transverse_wake_y=self.transverse_wake_y,
                charge_per_particle_C=self._charge_per_particle_C,
                tau_z_eff_s=self._tau_z_ibs_eff if self.ibs else None,
                tau_x_eff_s=self._tau_x_ibs_eff if self.ibs else None,
                tau_y_eff_s=self._tau_y_ibs_eff if self.ibs else None,
                skip_rf_kick=(self.rf == "per_element"),
            )
            if collect and (turn % sample_every == 0 or turn == n_turns - 1):
                tau_s = particles6[..., 4] / C_LIGHT
                samples.append((turn, self._energy_eV_t, particles6[..., 5], tau_s, particles6[..., 0], particles6[..., 2]))

        self._elapsed_turns += n_turns
        return particles6, self._collect(samples) if collect else None

    def advance(self, particles6: torch.Tensor, n_turns: int = 1) -> torch.Tensor:
        """Advance fixed-energy native turns without collecting host diagnostics."""
        if self.ramp is not None:
            raise ValueError("advance requires a fixed-energy tracker; use track for ramps")
        if n_turns < 1:
            raise ValueError("n_turns must be positive")
        particles6 = particles6.to(device=self.device, dtype=self.dtype)
        if self.bunch_current_A != 0.0:
            self._charge_per_particle_C.fill_(self.params.T0_s * self.bunch_current_A / particles6.shape[0])
        state, _ = self._track_fixed_energy(particles6, n_turns, 1, collect=False)
        return state

    def _init_ramp_overrides(self, dtype: torch.dtype, device: torch.device) -> None:
        names = ["Vrf_eff_V", "omega_rf_eff", "phi_s_eff", "E0_eff_eV", "tau_z_eff_s", "sigma_delta0_eff"]
        if self.optics is not None:
            names += ["tau_x_eff_s", "tau_y_eff_s", "epsx_eff_m", "epsy_eff_m"]
        self._ramp_overrides = {name: torch.zeros((), dtype=dtype, device=device) for name in names}

    def _track_ramped(
        self, particles6: torch.Tensor, n_turns: int, sample_every: int, progress: bool = False,
    ) -> tuple[torch.Tensor, NativeTrackResult]:
        if self._ramp_overrides is None:
            self._init_ramp_overrides(particles6.dtype, particles6.device)
        ov = self._ramp_overrides
        samples: list[tuple] = []

        turns = range(n_turns)
        if progress:
            from tqdm import tqdm
            turns = tqdm(turns, unit="turn", desc="tracking", total=n_turns)
        for _ in turns:
            t_now = self._elapsed_turns * self.params.T0_s
            state = self.ramp.at(t_now)
            gamma_now = state.E0_eV / self.params.rest_mass_eV
            ratio = gamma_now / self.params.gamma
            scaled = scaled_lattice(self.params, gamma_now)

            E0_ratio = self._ramp_E0_prev / state.E0_eV
            new_delta = particles6[..., 5] * E0_ratio + (E0_ratio - 1.0)
            new_px = particles6[..., 1] * E0_ratio
            new_py = particles6[..., 3] * E0_ratio
            particles6 = torch.cat([
                particles6[..., 0:1], new_px.unsqueeze(-1), particles6[..., 2:3], new_py.unsqueeze(-1),
                particles6[..., 4:5], new_delta.unsqueeze(-1),
            ], dim=-1)

            self._energy_eV_t.fill_(state.E0_eV)
            omega_rf_this_turn = 2 * math.pi * state.f_rf_Hz + self._rf_error_delta["omega_rf"]
            ov["Vrf_eff_V"].fill_(state.Vrf_V + self._rf_error_delta["Vrf_V"])
            ov["omega_rf_eff"].fill_(omega_rf_this_turn)
            ov["phi_s_eff"].fill_(
                state.phi_rf_rad - omega_rf_this_turn * self._fiducial_t_ref_s + self._rf_error_delta["phi_s"]
            )
            ov["E0_eff_eV"].fill_(state.E0_eV)
            ov["tau_z_eff_s"].fill_(scaled.tau_z_s)
            ov["sigma_delta0_eff"].fill_(scaled.sigma_delta0)
            if self.optics is not None:
                scaled_opt = scaled_optics(self.optics, ratio)
                ov["tau_x_eff_s"].fill_(scaled_opt.tau_x_s)
                ov["tau_y_eff_s"].fill_(scaled_opt.tau_y_s)
                ov["epsx_eff_m"].fill_(scaled_opt.epsx_m)
                ov["epsy_eff_m"].fill_(scaled_opt.epsy_m)

            if self.ibs:
                if self._elapsed_turns % self.ibs_recompute_every == 0:
                    self._recompute_ibs_rates(particles6, state.E0_eV)
                tau_z, tau_x, tau_y = ibs_effective_taus(
                    ov["tau_z_eff_s"], ov["tau_x_eff_s"], ov["tau_y_eff_s"], self._ibs_rates,
                )
                ov["tau_z_eff_s"].fill_(tau_z)
                ov["tau_x_eff_s"].fill_(tau_x)
                ov["tau_y_eff_s"].fill_(tau_y)

            noise_delta, noise_xp, noise_yp = self._draw_noise(particles6)
            particles6, _ = self._step(
                particles6, self._element_params, self._energy_eV_t, self._mc2_eV_t, self.params, noise_delta,
                Vrf_eff_V=ov["Vrf_eff_V"], omega_rf_eff=ov["omega_rf_eff"], phi_s_eff=ov["phi_s_eff"],
                E0_eff_eV=ov["E0_eff_eV"], tau_z_eff_s=ov["tau_z_eff_s"], sigma_delta0_eff=ov["sigma_delta0_eff"],
                optics=self.optics, noise_xp=noise_xp, noise_yp=noise_yp,
                tau_x_eff_s=ov.get("tau_x_eff_s"), tau_y_eff_s=ov.get("tau_y_eff_s"),
                epsx_eff_m=ov.get("epsx_eff_m"), epsy_eff_m=ov.get("epsy_eff_m"),
                wake=self.wake, transverse_wake_x=self.transverse_wake_x, transverse_wake_y=self.transverse_wake_y,
                charge_per_particle_C=self._charge_per_particle_C,
                skip_rf_kick=(self.rf == "per_element"),
            )

            if self._elapsed_turns % sample_every == 0 or self._elapsed_turns == n_turns - 1:
                tau_s = particles6[..., 4] / C_LIGHT
                samples.append((
                    self._elapsed_turns, state.E0_eV, particles6[..., 5], tau_s, particles6[..., 0], particles6[..., 2],
                ))

            self._ramp_E0_prev = state.E0_eV
            self._elapsed_turns += 1

        return particles6, self._collect(samples)

    @staticmethod
    def _collect(samples: list[tuple]) -> NativeTrackResult:
        return NativeTrackResult(
            turn=np.array([s[0] for s in samples]),
            E0_eV=np.array([float(s[1]) for s in samples]),
            sigma_delta=np.array([s[2].std().item() for s in samples]),
            sigma_tau_s=np.array([s[3].std().item() for s in samples]),
            sigma_x_m=np.array([s[4].std().item() for s in samples]),
            sigma_y_m=np.array([s[5].std().item() for s in samples]),
        )

