"""RF-on ring survival and tune analysis using the native tracker.

These helpers preserve the tracker's real cavity locations and longitudinal
physics. They advance its elapsed-turn and random-generator state normally.
Use independent trackers for independent scans, particularly with excitation.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch

from ..tracking.native.tracker import NativeTracker
from .nonlinear import DynamicAperture, FrequencyMap, LinearOptics, dynamic_aperture, launch, naff, normalized_signals


class Ring6DResult(NamedTuple):
    state: torch.Tensor
    loss_turn: torch.Tensor
    tbt: torch.Tensor  # (T, ..., 6), canonical native coordinates
    record_every: int = 1


def track_ring6d(
    tracker: NativeTracker, state: torch.Tensor, n_turns: int,
    aperture: tuple[float, float] = (0.05, 0.05),
    delta_limit: float | None = None,
    record_every: int = 1,
) -> Ring6DResult:
    """Track RF-on turns, recording all six coordinates and first losses.

    Apertures are checked at the ring entrance after each turn, with optional
    ``|delta|`` limit. Lost slots are frozen at zero. This is a single-particle
    survival calculation: collective wakes/current and IBS are unsupported
    because frozen losses must not contribute to collective moments.
    """
    if n_turns < 1:
        raise ValueError("n_turns must be positive")
    if not isinstance(record_every, int) or record_every < 0:
        raise ValueError("record_every must be nonnegative")
    if len(aperture) != 2 or any(math.isnan(a) or a <= 0 for a in aperture) or (delta_limit is not None and (not math.isfinite(delta_limit) or delta_limit <= 0)):
        raise ValueError("aperture and delta_limit must be positive")
    if tracker.bunch_current_A != 0 or tracker.ibs or tracker.wake is not None or tracker.transverse_wake_x is not None or tracker.transverse_wake_y is not None:
        raise ValueError("RF survival analysis requires tracking without collective effects")
    state = state.to(device=tracker.device, dtype=tracker.dtype)
    if state.ndim == 0 or state.shape[-1] != 6:
        raise ValueError("state must have six canonical coordinates")
    loss = torch.full(state.shape[:-1], -1, dtype=torch.int64, device=state.device)
    records = []
    for turn in range(n_turns):
        alive = loss < 0
        # NativeTracker implements the same RF/damping/wake order used by
        # normal tracking. Do not duplicate its physics in an analysis map.
        advanced = tracker.advance(state, 1)
        bad = ~torch.isfinite(advanced).all(-1)
        bad = bad | (advanced[..., 0].abs() > aperture[0]) | (advanced[..., 2].abs() > aperture[1])
        if delta_limit is not None:
            bad = bad | (advanced[..., 5].abs() > delta_limit)
        loss = torch.where(alive & bad, turn, loss)
        state = torch.where((loss < 0).unsqueeze(-1), advanced, torch.zeros_like(advanced))
        if record_every and (turn + 1) % record_every == 0:
            records.append(state)
    tbt = torch.stack(records) if records else state.new_empty((0, *state.shape))
    return Ring6DResult(state, loss, tbt, record_every)


def betatron_tunes6d(
    result: Ring6DResult, optics: LinearOptics, synchrotron_tune: float,
    half_width: float | None = None,
    dispersion: tuple[float, float, float, float] = (0., 0., 0., 0.),
) -> torch.Tensor:
    """Return ``(..., 2)`` carrier tunes, NaN for lost launches.

    Subtract linear dispersion times the instantaneous momentum deviation.
    Restrict each NAFF seed to a band narrower than half the synchrotron
    sideband spacing. The supplied optics must estimate the carrier within
    that band; a large detuning requires updated centres. This does not
    identify overlapping or unresolved sidebands automatically. At least
    four synchrotron periods must be recorded to resolve the carrier.
    """
    if result.record_every != 1:
        raise ValueError("tune analysis requires record_every=1; decimated frequencies may alias")
    if not 0 < synchrotron_tune < 0.5:
        raise ValueError("synchrotron_tune must lie in (0, 0.5)")
    width = 0.4 * synchrotron_tune if half_width is None else half_width
    if not 0 < width < 0.5 * synchrotron_tune:
        raise ValueError("half_width must be positive and below half the synchrotron tune")
    if result.tbt.shape[0] * synchrotron_tune < 4:
        raise ValueError("record at least four synchrotron periods to resolve sidebands")
    coords = result.tbt[..., :4].double()
    D = torch.as_tensor(dispersion, device=coords.device, dtype=coords.dtype)
    coords = coords - result.tbt[..., 5:6].double() * D
    zx, zy = normalized_signals(coords, optics)
    # Zeroed loss records have no tune; substitute a harmless finite tone
    # before NAFF so discarded singular Newton branches cannot poison grads.
    valid = result.loss_turn < 0
    n = torch.arange(coords.shape[0], device=coords.device, dtype=coords.dtype)
    values = []
    for z, center in ((zx, optics.Qx), (zy, optics.Qy)):
        center = center.to(coords.device)
        safe = torch.exp(2j * torch.pi * center * n)
        q, _ = naff(torch.where(valid.unsqueeze(-1), z, safe), center=center, half_width=width)
        inside = (torch.remainder(q - center + 0.5, 1.) - 0.5).abs() < width
        values.append(torch.where(valid & inside, q, torch.full_like(q, torch.nan)))
    return torch.stack(values, -1)


class MomentumScan(NamedTuple):
    delta: torch.Tensor
    survived: torch.Tensor
    loss_turn: torch.Tensor


def momentum_aperture6d(
    tracker: NativeTracker, launch: torch.Tensor, delta: torch.Tensor,
    n_turns: int, aperture: tuple[float, float] = (0.05, 0.05),
    delta_limit: float | None = None,
) -> MomentumScan:
    """Batched sampled momentum acceptance at a supplied RF synchronous orbit.

    ``launch`` is a single six-coordinate orbit or ``(n_delta, 6)`` matched
    launches. ``delta`` gives absolute initial momentum deviations. Return
    the survival of every sample rather than assuming acceptance is connected
    or interpolating across unstable islands. A finite scan bounds acceptance
    only over its tested range and turn count.
    """
    delta = torch.as_tensor(delta, device=tracker.device, dtype=tracker.dtype)
    if delta.ndim != 1 or delta.numel() == 0:
        raise ValueError("delta must be a nonempty one-dimensional scan")
    state = torch.as_tensor(launch, device=tracker.device, dtype=tracker.dtype).expand(delta.numel(), 6).clone()
    state[:, 5] = delta
    result = track_ring6d(tracker, state, n_turns, aperture, delta_limit, record_every=0)
    return MomentumScan(delta, result.loss_turn < 0, result.loss_turn)


def dynamic_aperture6d(
    tracker: NativeTracker, optics: LinearOptics, r_max: float, n_turns: int,
    n_angles: int = 19, n_r: int = 32, refine: int = 2,
    aperture: tuple[float, float] = (0.05, 0.05), delta_limit: float | None = None,
) -> DynamicAperture:
    """RF-on DA using the same batched ray refinement as frozen-energy DA.

    Supply optics whose ``closed_orbit`` is the six-dimensional synchronous
    launch. Refinement advances the tracker RNG; deterministic comparison
    requires zero excitation. Damping can change the measured survival.
    """
    class SurvivalMap:
        device = tracker.device
        dtype = tracker.dtype

        def track(self, state, n_turns, aperture):
            return track_ring6d(tracker, state, n_turns, aperture, delta_limit, record_every=0)

    return dynamic_aperture(SurvivalMap(), r_max, n_turns, n_angles, n_r, refine,
                            aperture=aperture, opt=optics)


def frequency_map6d(
    tracker: NativeTracker, optics: LinearOptics, synchrotron_tune: float,
    x_range: tuple[float, float], y_range: tuple[float, float],
    nx: int = 41, ny: int = 21, n_turns: int = 2048,
    aperture: tuple[float, float] = (0.05, 0.05), half_width: float | None = None,
    dispersion: tuple[float, float, float, float] = (0., 0., 0., 0.),
) -> FrequencyMap:
    """RF-on carrier tunes and two-half Laskar diffusion on a launch grid.

    Each half needs four synchrotron periods. Diffusion includes configured
    radiation damping/excitation; use a deterministic tracker for a purely
    Hamiltonian study. ``optics.closed_orbit`` is a supplied synchronous orbit.
    """
    if n_turns < 2 or n_turns % 2:
        raise ValueError("n_turns must be positive and even")
    X, Y = torch.meshgrid(torch.linspace(*x_range, nx, device=tracker.device, dtype=tracker.dtype),
                          torch.linspace(*y_range, ny, device=tracker.device, dtype=tracker.dtype), indexing="ij")
    result = track_ring6d(tracker, launch(optics, X, Y), n_turns, aperture)
    h = n_turns // 2
    q = [betatron_tunes6d(Ring6DResult(result.state, result.loss_turn, result.tbt[sl]),
                         optics, synchrotron_tune, half_width, dispersion)
         for sl in (slice(0, h), slice(h, n_turns))]
    dq = torch.remainder(q[1] - q[0] + 0.5, 1.) - 0.5
    diffusion = 0.5 * torch.log10(dq.square().sum(-1) + 1e-300)
    return FrequencyMap(X, Y, q[0][..., 0], q[0][..., 1], diffusion)
