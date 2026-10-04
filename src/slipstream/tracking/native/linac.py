"""Single-pass linac/FEL tracking. Distinct from ``NativeTracker`` (ring:
requires a ``Lattice``, loops ``n_turns``, applies ring-average radiation
damping/excitation and optional IBS every turn) because a single pass is a
genuinely different physics regime, not a special case of a ring with
``n_turns=1``. See docs/tracking_native.md and plans/DONE/NATIVE_LINAC_PLAN.md.

- No periodicity -- one pass through an ordered element list, no turns.
- The reference energy continuously tracks the beam's own centroid
  (``always_change_p0``, elegant's own run-level convention): after every
  non-diagnostic element, not just RF cavities with ``CHANGE_P0=1``. This
  generalizes the per-cavity ``change_p0`` this module already has
  (``elements._rf_cavity_kick``) via the shared
  ``elements._match_energy_to_centroid``.
- No equilibrium radiation damping/excitation and no IBS -- both are
  ring-average, asymptotic-equilibrium concepts with no meaning over one
  pass, and forcing a ring-only ``Lattice`` onto a linac section is a real
  correctness hazard (``Vrf_V=0`` makes ``Lattice.phi_s`` raise
  ``ZeroDivisionError``). ``LinacTracker`` needs no ``Lattice`` at all.
- Deferred, documented gaps (real CLARA `.lte`/`.ele` survey): CSR (see
  ``NATIVE_LINAC_LSC_CSR_PLAN.md``), incoherent SR, RF-focusing, transverse
  deflecting cavities.

Section "joining" (CLARA's own multi-``.lte``, bunch-dump-chained
convention) needs no new mechanism: track through one section's ``specs``,
then feed its output beam + reference energy as the next section's
injection beam + energy (``beam2, E2 = tracker2.track(*tracker1.track
(beam1))``), or concatenate multiple ``parse_elegant_lattice(...)``
results into one ``specs`` list for a single combined run.
"""
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
    csr_steady_state_delta_kick_fast,
    lsc_delta_kick,
    lsc_delta_kick_fast,
)
from .elements import _match_energy_to_centroid, _rebase_reference, reference_offset, track_element
from .errors import FiducialReference, SampledError, apply_errors
from .lattice_import import ElementSpec
from .tracker import _ElementParams, _apply_one_element, _build_params


def resolve_wakes(specs: list[ElementSpec], base_dir: str | Path = ".") -> list[ElementSpec]:
    """Returns a **new** ``list[ElementSpec]`` with every ``kind="rf"``
    element's wake *metadata* (plain strings, see ``ElementSpec``'s own
    docstring) resolved into real, loaded ``wake``/``transverse_wake_x``/
    ``transverse_wake_y`` objects (``ShortRangeWake``/``TransverseWake``,
    via their own ``from_sdds_file``) -- a one-shot construction-time
    transform, mirroring ``errors.apply_errors``. ``base_dir`` resolves a
    relative ``WAKEFILE=`` path (relative to the ``.lte``'s own
    directory). An element with no ``wakefile`` set passes through
    unchanged. A missing ``wake_w{x,y,z}_column`` simply skips that plane."""
    base_dir = Path(base_dir)
    out = []
    for s in specs:
        if s.kind != "rf" or not s.wakefile:
            out.append(s)
            continue
        path = base_dir / s.wakefile
        t_column = s.wake_t_column or "t"
        wake = ShortRangeWake.from_sdds_file(path, t_column=t_column, w_column=s.wake_wz_column) if s.wake_wz_column else None
        wx = TransverseWake.from_sdds_file(path, t_column=t_column, w_column=s.wake_wx_column) if s.wake_wx_column else None
        wy = TransverseWake.from_sdds_file(path, t_column=t_column, w_column=s.wake_wy_column) if s.wake_wy_column else None
        out.append(replace(s, wake=wake, transverse_wake_x=wx, transverse_wake_y=wy))
    return out


def _apply_collective_kick(
    particles6: torch.Tensor, energy_eV: torch.Tensor, mc2_eV: torch.Tensor,
    charge_per_particle_C: torch.Tensor | float, length_m: float,
    lsc: bool, csr: bool, csr_rho_m: float, csr_length_m: float, csr_drift_start_m: float | None,
    pipe_radius_b_m: float | None, lsc_radius_factor: float,
    csr_softening_m: float | None, compile_collective_chunks: bool = False,
    fast_collective_effects: bool = False, csr_bins: int = 256, csr_sg_halfwidth: int = 1,
) -> torch.Tensor:
    """LSC (if ``lsc``, over this element's own ``length_m``) and/or CSR
    (if ``csr``, at bending radius ``csr_rho_m`` -- caller already resolves
    this from either the element's own ``ElementSpec.csr`` bend or an
    inherited ``csr_drift_rho_m`` carry, see ``_maybe_collective_kick``)
    applied to this element's ``delta`` column -- see ``collective.py`` for
    both kernels' derivation. Applied once per element, using that
    element's own total length (not sub-cycled at Yoshida-sub-slice
    granularity -- see plans/NATIVE_LINAC_LSC_CSR_PLAN.md's "what was
    simplified"). Only the ``delta`` column changes (both kernels are
    purely longitudinal).

    ``csr_drift_start_m`` (``None`` for a real CSR bend, using its own
    static ``csr_length_m`` unchanged) marks a ``CSRDRIFT``-carried drift:
    its CSR-active length is capped live, ``clamp(overtaking_length_m -
    csr_drift_start_m, 0, csr_length_m)`` where ``overtaking_length_m =
    (24*sigma_z*csr_rho_m^2)^(1/3)`` (elegant's own ``csbend.c``
    ``CSRDRIFT_OVERTAKINGLENGTH`` formula) uses the beam's *live*
    ``particles6``-own bunch length -- real elegant's steady-state
    treatment shielding-cuts off once the bend's own radiation has slipped
    past the bunch. ``torch.clamp``, not a Python ``if``, over this live
    value -- see ``lsc_delta_kick``'s own "Clamped, not branched" note for
    why (``torch.compile`` graph-breaks on a Python branch over a live
    tensor).

    ``fast_collective_effects`` (default ``False``, zero behavior change)
    swaps in the fixed-bin-grid kernels -- see
    plans/DONE/FAST_COLLECTIVE_EFFECTS_PLAN.md. Both fast kernels are
    traceable inside this per-element compiled step (no
    ``@torch._dynamo.disable()``), so this branch is a plain Python ``if``
    on a fixed instance-level bool, resolved once at trace time."""
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
                n_bins=csr_bins, sg_halfwidth=csr_sg_halfwidth,
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
    """Skip the collective-kick computation entirely when neither is
    active for this element -- ``lsc``/``csr`` are fixed per-instance
    Python bools and ``p.csr``/``p.csr_drift_rho_m`` fixed per-element ones
    (all resolved at ``torch.compile`` trace time, same as
    ``always_change_p0``). A CSR-flagged bend uses its own ``length``/
    ``angle`` for ``rho_m`` (no live cutoff, ``csr_drift_start_m=None``); a
    drift carrying a ``csr_drift_rho_m`` (a real ``CSRDRIFT`` tail, see
    ``ElementSpec``'s own docstring) uses that inherited radius with a live
    overtaking-length cutoff instead."""
    csr_active = csr and (p.csr or p.csr_drift_rho_m is not None)
    if not lsc and not csr_active:
        return particles6
    csr_drift_start_m = None
    if p.csr:
        csr_rho_m, csr_length_m = (p.length / p.angle if p.angle != 0.0 else 0.0), p.length
    elif p.csr_drift_rho_m is not None:
        csr_rho_m, csr_length_m, csr_drift_start_m = p.csr_drift_rho_m, p.length, p.csr_drift_start_m
    else:
        csr_rho_m, csr_length_m = 0.0, 0.0
    return _apply_collective_kick(
        particles6, energy_eV, mc2_eV, charge_per_particle_C, p.length,
        lsc, csr_active, csr_rho_m, csr_length_m, csr_drift_start_m,
        pipe_radius_b_m, lsc_radius_factor, csr_softening_m,
        compile_collective_chunks, fast_collective_effects, p.csr_bins, p.csr_sg_halfwidth,
    )


def _track_dipole_subcycled(
    particles6: torch.Tensor, p: _ElementParams, energy_eV: torch.Tensor, mc2_eV: torch.Tensor,
    charge_per_particle_C: torch.Tensor | float, lsc: bool, csr: bool,
    pipe_radius_b_m: float | None, lsc_radius_factor: float, csr_softening_m: float | None,
    compile_collective_chunks: bool, fast_collective_effects: bool,
) -> torch.Tensor:
    """A real dipole (``p.angle != 0``), transported and CSR/LSC-kicked in
    ``p.n_kicks`` equal slices instead of one lumped end-of-element kick --
    mirrors elegant's own ``CSRCSBEND`` (``T1[iBin] *= CSRConstant *
    csbend->length / csbend->nSlices``, recomputed against the live bunch
    at *each* slice, same ``nSlices`` the geometric integration already
    uses). ``_apply_collective_kick``'s own single-lumped-kick default
    freezes the line-density estimate at the bend's *exit* even though a
    chicane can compress the bunch by an order of magnitude across one
    bend -- this reuses the live, evolving distribution instead.

    Bit-identical geometric transport to the un-subcycled path with
    collective effects off: ``track_element``'s own ``n_kicks`` loop is
    ``for _ in range(n_kicks): drift(ds*df); kick(ds*kf)`` with
    ``ds=length/n_kicks`` -- calling it ``n`` times at
    ``length/n, angle/n, n_kicks=1`` reproduces that *exact* same ``ds``/
    ``h=angle/length`` sequence (edges/misalignment untouched, gated to
    only the first/last sub-call), so ``p.offset`` (the whole-element
    reference-particle discretization residual, computed once in
    ``_build_params`` at ``nk=n_kicks``) is still the right correction to
    subtract at the end -- no separate per-slice offset needed.

    Only entered for ``p.dx == p.dy == p.tilt == 0`` (see
    ``_track_one_element``'s own gating) -- a misaligned dipole falls back
    to the single-lumped-kick path instead of this function, since
    ``track_element``'s misalign transform isn't independently gateable
    per side the way ``edge{1,2}_effects`` already is. A real, narrow,
    documented gap, not a silently wrong answer."""
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """One element's worth of work -- transport, then (if active) the
    collective kick, then (if ``always_change_p0``) recentering. This is
    the single unit ``LinacTracker(compile=True)`` compiles for a
    magnetic/drift element (``self._step_fn``), **not** the whole per-pass
    loop: compiling the whole 182-element real-CLARA pass as one graph
    OOM-killed WSL/CUDA after ~15 minutes even with LSC/CSR off (see
    plans/NATIVE_LINAC_LSC_CSR_PLAN.md). Compiling per element instead
    bounds each graph to one element's own ops, at the cost of one
    recompilation per distinct element configuration -- real lattices
    repeat magnet types, and every recompile is paid once per
    ``LinacTracker`` instance (a Monte Carlo jitter study, this module's
    own motivating use case, calls ``.track()`` many times on one
    instance, amortizing it completely). See ``LinacTracker.__init__``
    for why ``torch._dynamo.config.recompile_limit`` is raised alongside.

    **Never compiled for an RF-kind element** (``LinacTracker.__init__``
    always passes the plain, uncompiled version as ``rf_step_fn``): a real
    CLARA cavity's own ``N_KICKS`` (e.g. L02's real 360) would let dynamo
    unroll ``elements._rf_cavity_kick``'s ``for ik in range(n_kicks):``
    loop into one enormous trace, minutes of pure CPU-side Inductor
    codegen. Instead ``compile_rf_sub_kicks`` compiles **one RF sub-kick**
    (``elements._rf_one_sub_kick``) reused across every ``ik`` and cavity,
    while this outer function and the ``ik`` loop stay eager -- "compile
    the small reused inner unit, not the loop around it", recursed one
    level down from this function's own existence. See docs/tracking_native.md
    for the compile-time/runtime numbers this bought.

    ``compile_collective_chunks`` (default ``False``) is the same
    principle recursed into ``collective.py``'s own LSC/CSR row-chunk
    loops. Unlike ``compile_rf_sub_kicks``, this applies regardless of
    element kind (LSC/CSR act on RF elements too).

    ``fast_collective_effects`` (default ``False``) swaps in the
    fixed-bin-grid LSC/CSR kernels -- see ``_apply_collective_kick``.
    Bound into ``self._step_fn``/``self._rf_step_fn`` via the same
    ``functools.partial`` as ``compile_collective_chunks``, not re-passed
    on every ``.track()`` call."""
    if (dipole_collective_subcycle and p.angle != 0 and p.n_kicks > 1
            and (lsc or (csr and p.csr)) and p.dx == p.dy == p.tilt == 0):
        particles6 = _track_dipole_subcycled(
            particles6, p, energy_eV, mc2_eV, charge_per_particle_C, lsc, csr,
            pipe_radius_b_m, lsc_radius_factor, csr_softening_m,
            compile_collective_chunks, fast_collective_effects,
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
    """One pass through every element -- a plain, always-eager Python loop
    calling ``step_fn`` for a magnetic/drift element or ``rf_step_fn`` for
    an RF-kind one (see ``_track_one_element`` for why compilation happens
    there, per element, not by wrapping this whole loop).
    ``charge_per_particle_C`` is only read by a wake-carrying element (a
    no-op for a beamline with no wakes). ``lsc``/``csr`` (default
    ``False``) apply the two ``collective.py`` kernels once per element."""
    for p in params:
        fn = rf_step_fn if p.is_rf else step_fn
        particles6, energy_eV = fn(
            particles6, p, energy_eV, mc2_eV, always_change_p0, charge_per_particle_C,
            lsc, csr, pipe_radius_b_m, lsc_radius_factor, csr_softening_m,
        )
    return particles6, energy_eV


class LinacTracker:
    """Tracks a plain ``(N, 6)`` particle tensor (columns ``x, px, y, py,
    tau, delta`` -- see ``elements.py``'s module docstring) through **one
    pass** of ``specs`` (``lattice_import.parse_elegant_lattice``'s
    output, in beamline order). No ``Lattice``, no turns -- see this
    module's own docstring for why a linac single pass is a genuinely
    different regime from ``NativeTracker(rf="per_element")`` at
    ``n_turns=1``.

    Every ``kind="rf"`` element in ``specs`` (elegant's ``RFCA``/``RFCW``
    only) gets its own inline kick at its actual ``s``-position
    (``elements._rf_cavity_kick``, reused from the ring tracker's own
    ``rf="per_element"`` machinery).

    ``always_change_p0`` (default ``True``, matching real CLARA's own
    ``&run_setup`` convention) reproduces elegant's run-level
    ``always_change_p0``: after every element (not just RF cavities), the
    reference momentum recenters on the beam's own current mean momentum
    (``elements._match_energy_to_centroid``). When ``True``, every RF
    element's own ``ElementSpec.change_p0`` is overridden to ``False``
    (redundant otherwise, since the always-mode already recenters after
    that same element). ``always_change_p0=False`` instead honors each RF
    element's own real ``CHANGE_P0`` value, identical to
    ``NativeTracker(rf="per_element")``'s per-cavity model without turns.

    ``n_kicks``/``n_kicks_bend``/``integration_order``/``compile``/
    ``compile_mode`` mirror ``NativeTracker``'s own constructor parameters.

    ``errors``/``fiducial`` reuse ``tracking.native.errors`` directly --
    ``errors.apply_errors`` resolves ``PHASE``/``VOLT``/``FREQ`` errors
    onto a matching ``kind="rf"`` element's own field (per-cavity RF
    jitter studies), and ``fiducialize()`` below mirrors
    ``NativeTracker.fiducialize()``, generalized to this tracker's own
    ``always_change_p0`` setting. ``fiducial=`` (from a different,
    unperturbed sibling ``LinacTracker``'s ``fiducialize()``) freezes the
    RF phase reference this tracker measures against, generalized from
    "one lumped phase" to **per cavity**: every ``kind="rf"`` element's
    own ``phase_rad`` gets an additional ``-omega_rf*t_ref_s`` term (that
    element's own ``omega_rf``) before ``_build_params`` sees it.

    Radiation effects (incoherent SR/CSR) and RF-focusing kicks are
    **not** modeled -- see plans/DONE/NATIVE_LINAC_PLAN.md's non-goals.
    Structure wakefields (an ``RFCW`` equivalent) **are** modeled, via
    ``bunch_charge_C=`` below and any wake objects already attached to
    ``specs`` (see ``resolve_wakes``).

    ``bunch_charge_C`` (default ``0.0``) is this pass's single-bunch
    charge -- not ``NativeTracker``'s own circulating ``bunch_current_A``,
    since there's no "current" for one pass.
    ``charge_per_particle_C = bunch_charge_C / n_particles`` is computed
    lazily in ``.track()``. Every wake-carrying element's wake object gets
    ``.prepare(dtype, device)`` called once here.

    ``lsc``/``csr`` (both default ``False``) turn on the two
    ``collective.py`` kernels. ``csr=True`` only affects elements with
    ``ElementSpec.csr=True`` (``CSRCSBEND``/``CSRCSBEN``); ``lsc=True``
    applies to every element with nonzero length, matching real elegant's
    "LSC acts continuously along the beamline" physics.
    ``pipe_radius_b_m``/``lsc_radius_factor``/``csr_softening_m`` mirror
    ``collective.lsc_delta_kick``/``collective.csr_steady_state_delta_kick``'s
    own parameters.

    ``compile_collective_chunks`` (default ``None``) enables the compiled
    exact LSC/CSR chunk bodies on CUDA even when outer transport compilation
    is disabled; on CPU it follows ``compile``. Pass ``True`` or ``False``
    explicitly to override that default. This keeps the pairwise kernels'
    particle-count/dtype specialization independent from per-element transport
    specializations.

    ``fast_collective_effects`` (default ``False``) swaps ``lsc``/``csr``
    to the fixed-bin-grid approximations -- see
    plans/DONE/FAST_COLLECTIVE_EFFECTS_PLAN.md. A no-op unless ``lsc=True``
    and/or ``csr=True``, for a large speedup at real N=256k scale at the
    cost of a documented physics-fidelity trade. For fast CSR, imported
    BINS and SG_HALFWIDTH control the wake grid, and N_SLICES controls
    collective kicks inside aligned dipoles. Reference momentum recenters
    only after the whole element. Misaligned dipoles retain one exit kick.
    Drift tails remain the preceding-bend overtaking-length approximation.

    ``dipole_collective_subcycle`` (default ``False``) applies the LSC/CSR
    kick ``n_kicks_bend`` times per real dipole (at ``length/n_kicks_bend``
    each, against the live bunch at that point) instead of once, lumped,
    over the whole magnet -- see ``_track_dipole_subcycled``. Mirrors real
    elegant's own ``CSRCSBEND`` (same ``nSlices`` for both the geometric
    integration and the CSR wake). A no-op for a misaligned dipole
    (``DX``/``DY``/``TILT`` != 0), which still uses the lumped kick."""

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
    ):
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
                self._element_params[i] = self._element_params[i]._replace(csr_substeps=tuple(substeps))
        for p in self._element_params:
            if p.wake is not None:
                p.wake.prepare(dtype, self.device)
            if p.transverse_wake_x is not None:
                p.transverse_wake_x.prepare(dtype, self.device)
            if p.transverse_wake_y is not None:
                p.transverse_wake_y.prepare(dtype, self.device)
        # Compile the PER-ELEMENT step, not the whole per-pass loop -- see
        # _track_one_element's docstring. self._step_fn is also reused
        # directly by the return_trajectory=True path and fiducialize()
        # below, so both get the same per-element compiled speedup.
        if compile:
            # Bounded by the number of distinct element configurations in
            # specs (a loose over-estimate: real lattices repeat magnet
            # types) -- past dynamo's default recompile_limit, further
            # distinct elements silently fall back to eager.
            torch._dynamo.config.recompile_limit = max(
                torch._dynamo.config.recompile_limit, len(self._element_params) + 8,
            )
        # collective_chunks (independent of `compile`) and
        # fast_collective_effects bound via partial before wrapping in
        # torch.compile -- both fixed per-instance, never re-passed per
        # .track() call.
        _step_base = functools.partial(
            _track_one_element, compile_collective_chunks=collective_chunks,
            fast_collective_effects=fast_collective_effects,
            dipole_collective_subcycle=dipole_collective_subcycle,
        )
        self._step_fn = torch.compile(_step_base, mode=compile_mode) if compile else _step_base
        # RF-kind elements: this OUTER function is never compiled (avoids
        # dynamo unrolling elements._rf_cavity_kick's own N_KICKS loop),
        # but when compile=True the INNER per-sub-kick body still gets
        # compiled, reused across every ik and cavity.
        self._rf_step_fn = functools.partial(
            _track_one_element, compile_rf_sub_kicks=compile, compile_collective_chunks=collective_chunks,
            fast_collective_effects=fast_collective_effects,
            dipole_collective_subcycle=dipole_collective_subcycle,
        )
        self._track_single_pass = functools.partial(
            _track_single_pass, step_fn=self._step_fn, rf_step_fn=self._rf_step_fn,
        )

        # 0-d tensors, not plain floats -- repeated .track() calls (e.g. a
        # Monte Carlo jitter study) must not force a recompile per
        # distinct injection energy.
        self._energy_eV_t = torch.tensor(energy_eV, dtype=dtype, device=self.device)
        self._reference_energy_eV = energy_eV
        self._mc2_eV_t = torch.tensor(mc2_eV, dtype=dtype, device=self.device)
        self._charge_per_particle_C = torch.zeros((), dtype=dtype, device=self.device)

    def _set_reference_energy(self, energy_eV: float) -> None:
        """Update entrance energy and cached magnet reference offsets together.

        Source handoffs may change energy after construction. Never mutate
        ``reference_offset``'s shared cached CPU tensor; each refreshed offset
        is an owned tensor with the original shape, dtype and device.
        """
        if energy_eV != self._reference_energy_eV:
            def refresh(p):
                if p.is_rf or p.is_kick:
                    return p
                offset = reference_offset(
                    p.length, p.angle, p.k1, p.k2, p.e1, p.e2, energy_eV, self.mc2_eV,
                    p.n_kicks, p.integration_order, p.hgap, p.fint, p.edge_order,
                    p.edge1_effects, p.edge2_effects, p.exact_dipole,
                ).to(device=self.device, dtype=self.dtype).clone()
                return p._replace(offset=offset, csr_substeps=tuple(refresh(sub) for sub in p.csr_substeps))

            self._element_params = [refresh(p) for p in self._element_params]
            self._reference_energy_eV = energy_eV
        self._energy_eV_t.fill_(energy_eV)

    def track(
        self, particles6: torch.Tensor, return_trajectory: bool = False, reference_energy_eV: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One pass. Returns ``(particles6_out, energy_eV_out)`` --
        ``energy_eV_out`` is a 0-d tensor, the beamline's own exit
        reference energy.

        ``return_trajectory=True`` additionally returns a stacked
        ``(len(specs)+1, N, 6)`` particle-state tensor **and** a stacked
        ``(len(specs)+1,)`` reference-energy tensor, both for the state
        entering every element (index 0 is the injection state/energy,
        index -1 the beamline exit) -- a stand-in for CLARA-style
        ``WATCH`` diagnostic points, and the direct input to
        ``linac_twiss.twiss_from_trajectory``. Reuses ``self._step_fn``
        (the same per-element step ``return_trajectory=False`` calls) for
        each element's own physics -- only the trajectory bookkeeping (a
        growing Python list + a final ``torch.stack``) stays uncompiled."""
        particles6 = particles6.to(device=self.device, dtype=self.dtype)
        self._charge_per_particle_C.fill_(self.bunch_charge_C / particles6.shape[0])
        csr_active = self.csr and self.bunch_charge_C != 0.0
        if reference_energy_eV is not None:
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
        """One pass against a *fixed* reference-energy profile (``(len(specs) + 1,)``, entering
        every element, e.g. the ``energy_trajectory`` of a nominal ``track(return_trajectory=True)``)
        instead of ``always_change_p0``'s beam-mean reference: magnets keep their design strengths and a
        beam whose energy errs carries that error in ``delta`` (dispersion, R56 arrival-time and
        path-length effects are then real). What a jitter/error study needs -- ``always_change_p0``
        would re-scale every magnet to the jittered beam and absorb the error. ``tau`` is then the
        arrival time relative to the *design* particle. Eager (no ``torch.compile`` variant of this loop)."""
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
        """Establish the nominal reference this tracker's current
        beamline/RF produces, for a different, error-perturbed
        ``LinacTracker`` to freeze against (pass as its own ``fiducial=``)
        -- mirrors ``NativeTracker.fiducialize()`` (see plans/PHYSICS.md's
        "Errors and fiducialization" section), generalized to this
        tracker's own ``always_change_p0`` setting. Deliberately passes no
        ``charge_per_particle_C`` (a wake/collective kick from
        ``particles0``, typically one or a few reference particles, would
        be physically meaningless)."""
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
