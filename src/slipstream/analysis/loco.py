"""Differentiates closed-orbit BPM readings with respect to corrector kick angles. See
docs/modules/analysis/loco.md."""
from __future__ import annotations

import torch

from ..tracking.native.ring import RingMap
from .nonlinear import closed_orbit


def orbit_response_matrix(
    ring: RingMap, corrector_names: list[str], bpm_names: list[str],
    delta: float = 0.0, create_graph: bool = False,
) -> torch.Tensor:
    """Compute orbit response matrix; see the module documentation for details."""
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
    """Compute apply calibration; see the module documentation for details."""
    cos_r, sin_r = torch.cos(bpm_roll), torch.sin(bpm_roll)
    m00, m01 = bpm_gain_x * cos_r, -bpm_gain_x * sin_r
    m10, m11 = bpm_gain_y * sin_r, bpm_gain_y * cos_r
    x, y = J[:, 0, :], J[:, 1, :]
    x_cal = (m00.unsqueeze(-1) * x + m01.unsqueeze(-1) * y) * corrector_gain
    y_cal = (m10.unsqueeze(-1) * x + m11.unsqueeze(-1) * y) * corrector_gain
    return torch.stack([x_cal, y_cal], dim=1)
