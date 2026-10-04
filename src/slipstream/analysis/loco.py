"""LOCO (Linear Optics from Closed Orbits) -- orbit response matrix (ORM) on
top of ``tracking.native.ring.RingMap``'s differentiable closed orbit. See
plans/LOCO_PLAN.md. ``orbit_response_matrix`` (Phase 3) is the exact ORM;
``apply_calibration`` (Phase 5) is pure tensor post-processing for BPM
gain/roll and corrector gain. Alignment/coupling remains an explicit
``RingMap.with_strengths(dx=/dy=/tilt=)`` concern; the real-booster story
uses a bounded hybrid fit rather than hiding a full fit orchestration in this
low-level module.
"""
from __future__ import annotations

import torch

from ..tracking.native.ring import RingMap
from .nonlinear import closed_orbit


def orbit_response_matrix(
    ring: RingMap, corrector_names: list[str], bpm_names: list[str],
    delta: float = 0.0, create_graph: bool = False,
) -> torch.Tensor:
    """``(n_bpm, 2, n_corrector)`` exact ORM: the Jacobian of tapped
    ``(x, y)`` closed-orbit readings at every named BPM with respect to
    one scalar kick-angle variable per named corrector channel, evaluated
    at the zero-kick operating point (``ring``'s own corrector strengths
    are overridden to zero-valued *differentiable* tensors here -- any
    nonzero kick already baked into ``ring`` before calling this is
    discarded for the corrector columns, not added to).

    Each ``corrector_names`` entry must name a single H- or V-plane
    channel (an ``ElementSpec(kind="hkick")``/``"vkick"`` element) --
    the plane is read from the lattice itself, so there is no separate
    plane argument. Both ``corrector_names`` and ``bpm_names`` should
    already be occurrence-safe (``analysis.nonlinear_opt.unique_names``
    or a pre-renamed ``name#occurrence``) -- a bare name repeated in the
    real lattice would silently collapse every occurrence onto one
    column/row.

    ``create_graph=True`` keeps the returned ORM differentiable in any
    other tensor strength ``ring`` already carries (e.g. K1, via
    ``ring.with_strengths(k1=...)`` before calling this) -- the nested
    Jacobian a LOCO fit needs. ``delta`` is a fixed off-momentum point
    (dispersion is out of scope for this first slice, see the plan)."""
    plane_of = {s.name.upper(): ("x" if s.kind == "hkick" else "y") for s in ring.specs if s.kind in ("hkick", "vkick")}
    unknown_corr = [n for n in corrector_names if n.upper() not in plane_of]
    if unknown_corr:
        raise KeyError(f"not a corrector (hkick/vkick) element in this lattice: {unknown_corr}")
    planes = [plane_of[n.upper()] for n in corrector_names]

    index_of = {s.name.upper(): i for i, s in enumerate(ring.specs)}
    unknown_bpm = [n for n in bpm_names if n.upper() not in index_of]
    if unknown_bpm:
        raise KeyError(f"no such element(s) in lattice: {unknown_bpm}")
    tap_indices = [index_of[n.upper()] for n in bpm_names]

    n = len(corrector_names)

    # At the zero-kick operating point, differentiate the one-turn and
    # tapped maps separately and apply the implicit-function theorem. This is
    # algebraically the same derivative as differentiating ``closed_orbit``
    # inside ``observe``, but avoids a Jacobian-of-a-Jacobian for every ORM
    # call on a real lattice.
    zero = torch.zeros((), dtype=ring.dtype, device=ring.device)
    zero_ring = ring.with_strengths(
        kick_x={name: zero for name, plane in zip(corrector_names, planes) if plane == "x"},
        kick_y={name: zero for name, plane in zip(corrector_names, planes) if plane == "y"},
    )
    co = closed_orbit(zero_ring, delta=delta, create_graph=create_graph)

    def map_ring(state_and_kicks: torch.Tensor) -> tuple[RingMap, torch.Tensor]:
        state = state_and_kicks[:6]
        theta = state_and_kicks[6:]
        kick_x = {name: theta[i] for i, (name, plane) in enumerate(zip(corrector_names, planes)) if plane == "x"}
        kick_y = {name: theta[i] for i, (name, plane) in enumerate(zip(corrector_names, planes)) if plane == "y"}
        return ring.with_strengths(kick_x=kick_x, kick_y=kick_y), state

    def observe(state_and_kicks: torch.Tensor) -> torch.Tensor:
        r, state = map_ring(state_and_kicks)
        final, tapped = r.turn_with_taps(state, tap_indices)
        return torch.cat((final[:4], tapped[..., (0, 2)].reshape(-1)))

    base = torch.cat((co, torch.zeros(n, dtype=ring.dtype, device=ring.device)))
    if not create_graph:
        # Forward-mode keeps the full tapped tape out of a reverse-mode graph.
        # On the real booster, vectorized reverse-mode retains multiple GB for
        # this otherwise small dense Jacobian.
        D = torch.func.jacfwd(observe)(base)
        M, B = D[:4, :4], D[:4, 6:]
        dco = torch.linalg.solve(torch.eye(4, dtype=ring.dtype, device=ring.device) - M, B)
        P, Q = D[4:, :4], D[4:, 6:]
        return (Q + P @ dco).reshape(len(bpm_names), 2, n)

    D = torch.autograd.functional.jacobian(observe, base, create_graph=True, vectorize=True)
    M, B = D[:4, :4], D[:4, 6:]
    P, Q = D[4:, :4], D[4:, 6:]
    dco = torch.linalg.solve(torch.eye(4, dtype=ring.dtype, device=ring.device) - M, B)
    return (Q + P @ dco).reshape(len(bpm_names), 2, n)


def apply_calibration(
    J: torch.Tensor, bpm_gain_x: torch.Tensor, bpm_gain_y: torch.Tensor,
    bpm_roll: torch.Tensor, corrector_gain: torch.Tensor,
) -> torch.Tensor:
    """The measured-orbit-response-matrix model Safranek (1997) fits
    alongside the lattice: ``J`` (``(n_bpm, 2, n_corrector)``, the model
    ORM from ``orbit_response_matrix``) with a per-BPM 2x2 gain/rotation
    block applied to its rows and a per-corrector scalar gain applied to
    its columns -- ``measured = diag(gx_i, gy_i) @ Rot(roll_i) @ true *
    corrector_gain_j`` (a BPM roll rotates the true orbit into the
    tilted device frame *before* the device's own x/y gains scale it;
    a corrector gain scales the whole physical kick for one channel, so
    it multiplies that channel's entire ORM column).

    ``bpm_gain_x``/``bpm_gain_y``/``bpm_roll`` are ``(n_bpm,)``,
    ``corrector_gain`` is ``(n_corrector,)`` -- all plain multiplicative
    tensors (broadcast/autograd-friendly, no solver logic here).

    **Gauge**: an overall BPM-gain scale and an overall corrector-gain
    scale are degenerate (multiplying every ``bpm_gain_*`` by ``s`` and
    dividing every ``corrector_gain`` by ``s`` leaves ``J`` unchanged) --
    a fit must fix at least one gain (e.g. hold one ``corrector_gain``
    entry at exactly ``1.0``, not a free parameter) or supply a measured
    prior; this function does not enforce that itself.

    BPM *offsets* (a constant, metres-valued shift) do not belong here --
    an ORM is a derivative/difference (metres per radian) and a BPM
    offset cancels from it identically; see plans/LOCO_PLAN.md's Phase 5
    guard and Problem #2 in its design-error list."""
    cos_r, sin_r = torch.cos(bpm_roll), torch.sin(bpm_roll)
    m00, m01 = bpm_gain_x * cos_r, -bpm_gain_x * sin_r
    m10, m11 = bpm_gain_y * sin_r, bpm_gain_y * cos_r
    x, y = J[:, 0, :], J[:, 1, :]
    x_cal = (m00.unsqueeze(-1) * x + m01.unsqueeze(-1) * y) * corrector_gain
    y_cal = (m10.unsqueeze(-1) * x + m11.unsqueeze(-1) * y) * corrector_gain
    return torch.stack([x_cal, y_cal], dim=1)
