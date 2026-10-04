"""Samples one static error per matched element instance. See docs/modules/tracking/native/errors.md."""
from __future__ import annotations

import fnmatch
import math
from dataclasses import dataclass, replace

import torch

from .lattice_import import ElementSpec

# ElementSpec field each magnetic/alignment error parameter resolves onto.
_FIELD_NAMES = {"K1": "k1", "K2": "k2", "ANGLE": "angle", "DX": "dx", "DY": "dy", "TILT": "tilt"}
_RF_PARAMS = {"PHASE", "VOLT", "FREQ"}
_RF_FIELD_NAMES = {"PHASE": "phase_rad", "VOLT": "voltage_V", "FREQ": "frequency_Hz"}
_DISTRIBUTIONS = {"gaussian", "uniform", "plus_or_minus"}


@dataclass(frozen=True)
class ErrorSpec:
    """Describe ErrorSpec; see the module documentation for details."""

    name_pattern: str
    parameter: str
    amplitude: float
    distribution: str = "gaussian"
    cutoff: float = 3.0
    fractional: bool = False
    element_type: str | None = None

    def __post_init__(self) -> None:
        if self.parameter not in _FIELD_NAMES and self.parameter not in _RF_PARAMS:
            raise ValueError(f"unknown error parameter {self.parameter!r} (expected one of "
                              f"{sorted(_FIELD_NAMES) + sorted(_RF_PARAMS)})")
        if self.distribution not in _DISTRIBUTIONS:
            raise ValueError(f"unknown distribution {self.distribution!r} (expected one of {sorted(_DISTRIBUTIONS)})")


@dataclass(frozen=True)
class SampledError:
    """Describe SampledError; see the module documentation for details."""

    name: str
    occurrence: int
    parameter: str
    delta: float


def _draw_unit(distribution: str, cutoff: float, generator: torch.Generator | None) -> float:
    """Compute draw unit; see the module documentation for details."""
    device = generator.device if generator is not None else "cpu"
    if distribution == "gaussian":
        while True:
            v = torch.randn((), generator=generator, device=device, dtype=torch.float64).item()
            if abs(v) <= cutoff:
                return v
    if distribution == "uniform":
        return torch.rand((), generator=generator, device=device, dtype=torch.float64).item() * 2.0 - 1.0
    # "plus_or_minus", enforced by ErrorSpec.__post_init__
    return 1.0 if torch.rand((), generator=generator, device=device, dtype=torch.float64).item() < 0.5 else -1.0


def sample_errors(
    specs: list[ElementSpec], error_specs: list[ErrorSpec], generator: torch.Generator | None = None,
) -> list[SampledError]:
    """Compute sample errors; see the module documentation for details."""
    out: list[SampledError] = []
    for es in error_specs:
        for spec in specs:
            if es.element_type is not None and spec.kind != es.element_type:
                continue
            if not fnmatch.fnmatchcase(spec.name.upper(), es.name_pattern.upper()):
                continue
            draw = _draw_unit(es.distribution, es.cutoff, generator)
            if es.fractional:
                current = getattr(spec, _FIELD_NAMES[es.parameter]) if es.parameter in _FIELD_NAMES else 0.0
                delta = current * es.amplitude * draw
            else:
                delta = es.amplitude * draw
            out.append(SampledError(spec.name, spec.occurrence, es.parameter, delta))
    return out


def apply_errors(specs: list[ElementSpec], sampled: list[SampledError]) -> list[ElementSpec]:
    """Compute apply errors; see the module documentation for details."""
    by_key: dict[tuple[str, int], dict[str, float]] = {}
    for s in sampled:
        if s.parameter in _FIELD_NAMES:
            field = _FIELD_NAMES[s.parameter]
        elif s.parameter in _RF_FIELD_NAMES:
            field = _RF_FIELD_NAMES[s.parameter]
        else:
            continue
        key = (s.name.upper(), s.occurrence)
        by_key.setdefault(key, {})
        by_key[key][field] = by_key[key].get(field, 0.0) + s.delta

    rf_fields = set(_RF_FIELD_NAMES.values())
    out = []
    for spec in specs:
        deltas = by_key.get((spec.name.upper(), spec.occurrence))
        if not deltas:
            out.append(spec)
            continue
        updates = {
            field: getattr(spec, field) + delta for field, delta in deltas.items()
            if field not in rf_fields or spec.kind == "rf"
        }
        out.append(replace(spec, **updates) if updates else spec)
    return out


def resolve_rf_errors(sampled: list[SampledError]) -> dict[str, float]:
    """Compute resolve rf errors; see the module documentation for details."""
    out = {"phi_s": 0.0, "Vrf_V": 0.0, "omega_rf": 0.0}
    for s in sampled:
        if s.parameter == "PHASE":
            out["phi_s"] += s.delta
        elif s.parameter == "VOLT":
            out["Vrf_V"] += s.delta
        elif s.parameter == "FREQ":
            out["omega_rf"] += 2.0 * math.pi * s.delta
    return out


@dataclass
class FiducialReference:
    """Nominal trajectory and arrival time established by fiducialization."""

    trajectory: torch.Tensor
    t_ref_s: float
