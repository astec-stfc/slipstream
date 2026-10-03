"""Setting up a ``LinacTracker`` line: find a named element, change one cavity,
and phase a cavity on crest against a real bunch. Shared by ``Stories/`` and
any driver that scans RF phases -- ``crest_phase_rad`` mirrors
``Injector.auto_phase`` (coarse scan, parabolic refinement), applied to a generic
``LinacTracker`` cavity instead of the injector's own analytic field maps.
"""
from __future__ import annotations

import dataclasses
import math

import torch

from ...constants import ELECTRON_MASS_EV
from .injector import parabolic_vertex
from .lattice_import import ElementSpec
from .linac import LinacTracker


def element_index(specs: list[ElementSpec], name: str) -> int:
    return next(i for i, s in enumerate(specs) if s.name == name)


def with_element(specs: list[ElementSpec], name: str, **fields) -> list[ElementSpec]:
    """Copy of ``specs`` with ``fields`` (``phase_rad``, ``voltage_V``, ``angle``, ...) replaced
    on the element called ``name``."""
    i = element_index(specs, name)
    out = list(specs)
    out[i] = dataclasses.replace(out[i], **fields)
    return out


def crest_phase_rad(
    specs: list[ElementSpec], cavity_idx: int, ref6: torch.Tensor, energy_eV: float,
    coarse_deg: float = 10.0, mc2_eV: float = ELECTRON_MASS_EV,
) -> float:
    """Absolute RF phase (rad) giving maximum energy gain to a single reference particle
    tracked through ``specs[:cavity_idx + 1]``. CPU/float64-free one-off calibration on a
    single particle (float32, no compile), not the production tracking path."""
    base = specs[: cavity_idx + 1]
    degs = torch.arange(0.0, 360.0, coarse_deg, dtype=torch.float64)
    gains = torch.empty_like(degs)
    ref = ref6.to(device="cpu", dtype=torch.float32).reshape(1, 6)
    for k in range(degs.numel()):
        trial = list(base)
        trial[cavity_idx] = dataclasses.replace(trial[cavity_idx], phase_rad=math.radians(float(degs[k])))
        tracker = LinacTracker(trial, energy_eV=energy_eV, mc2_eV=mc2_eV, device="cpu", dtype=torch.float32, compile=False)
        _, e_out = tracker.track(ref)
        gains[k] = float(e_out) - energy_eV
    # a phase decelerating hard enough can NaN; mask before argmax (its NaN behaviour is undefined)
    finite = torch.isfinite(gains)
    i = int(torch.argmax(torch.where(finite, gains, torch.full_like(gains, -float("inf")))))
    if not (0 < i < degs.numel() - 1 and finite[i - 1] and finite[i + 1]):
        return math.radians(float(degs[i]))
    return math.radians(parabolic_vertex(degs, gains, i, coarse_deg))
