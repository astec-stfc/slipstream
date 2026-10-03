"""Machine aperture/material description (APERTURES.md): a YAML-defined set
of named ``Material``s (single resistive-wall layers, or composites built
from other named materials) and named ``Aperture``s (round or flat chamber
sections, each referencing one material and repeated ``count`` times over
``length_m``), summed into a total longitudinal ``Impedance`` for the ring.

Deliberately longitudinal-only for now, matching APERTURES.md's own scope
("total longitudinal (and later transverse) wakefields") -- round apertures
use ``wakefields.multilayer.round.round_chamber_impedance`` directly
(differentiable, validated against live/cached IW2D); flat apertures
approximate via the Yokoya elliptic form factor applied to the equivalent
round chamber at radius = the smaller half-aperture
(``wakefields.multilayer.yokoya``, since the full flat-chamber solver in
``wakefields.multilayer.flat`` is parked -- see that module's docstring).

One material mutation surface (APERTURES.md's doc proposed several; this
repo only needs one, see IMPEDANCE_PLAN.md-adjacent discussion): edit
``Apertures.materials[name]``'s fields directly -- every aperture that
references that material by name picks up the change next time impedance
is computed, no special API needed since materials are looked up by name,
not copied in. An aperture that needs to deviate from its shared material
(e.g. a thicker coating on one specific section) sets
``Aperture.material_overrides`` instead of forking the whole material.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from ..wakefields.impedance import Impedance
from ..wakefields.chambers import RoundChamber
from ..wakefields.multilayer.round import round_chamber_impedance
from ..wakefields.multilayer.yokoya import yokoya_elliptic_factors

_SHAPES = ("round", "flat")


@dataclass
class Material:
    """Either a leaf material (``rho``/``tau_ps``/``thickness_m``, the same
    Drude-relaxation layer ``RoundChamber.layers`` already uses) or a
    composite (``layers``: an ordered list of other material names,
    innermost first -- e.g. a NEG coating over a stainless substrate)."""

    rho: float | None = None
    tau_ps: float | None = None
    thickness_m: float | None = None
    layers: list[str] | None = None

    def __post_init__(self) -> None:
        is_leaf = self.rho is not None or self.tau_ps is not None or self.thickness_m is not None
        if is_leaf and self.layers is not None:
            raise ValueError("a material is either a leaf (rho/tau_ps/thickness_m) or a composite (layers), not both")
        if is_leaf and (self.rho is None or self.tau_ps is None or self.thickness_m is None):
            raise ValueError("a leaf material needs rho, tau_ps, and thickness_m all set")
        if not is_leaf and self.layers is None:
            raise ValueError("a material needs either rho/tau_ps/thickness_m or layers")


@dataclass
class Aperture:
    """One named chamber section. ``length_m``/``count`` are ``None`` only
    for the required ``"Default"`` aperture, whose length is instead
    whatever's left of the ring's circumference (see ``Apertures.impedance``)."""

    material: str
    shape: str
    type: str | None = None
    radius_m: float | None = None
    horizontal_half_aperture_m: float | None = None
    vertical_half_aperture_m: float | None = None
    length_m: float | None = None
    count: int = 1
    material_overrides: dict[str, dict[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.shape not in _SHAPES:
            raise ValueError(f"aperture shape must be one of {_SHAPES}, got {self.shape!r}")
        if self.shape == "round" and self.radius_m is None:
            raise ValueError("a round aperture needs radius_m")
        if self.shape == "flat" and (self.horizontal_half_aperture_m is None or self.vertical_half_aperture_m is None):
            raise ValueError("a flat aperture needs horizontal_half_aperture_m and vertical_half_aperture_m")


_YAML_APERTURE_KEYS = {
    "material": "material",
    "shape": "shape",
    "type": "type",
    "radius": "radius_m",
    "horizontal_half_aperture": "horizontal_half_aperture_m",
    "vertical_half_aperture": "vertical_half_aperture_m",
    "length": "length_m",
    "count": "count",
    "overrides": "material_overrides",
}


@dataclass
class Apertures:
    """The whole Materials + Apertures set from one YAML file. Not derived
    from ``Lattice`` and doesn't duplicate any of its fields (circumference
    is passed into ``impedance``/``default_length_m`` instead) -- mirrors
    how ``resonators.yaml`` sits alongside, not inside, ``lattice.yaml``."""

    materials: dict[str, Material]
    apertures: dict[str, Aperture]

    def __post_init__(self) -> None:
        if "Default" not in self.apertures:
            raise ValueError("Apertures must include a 'Default' aperture")
        default = self.apertures["Default"]
        if default.length_m is not None or default.count != 1:
            raise ValueError("the 'Default' aperture's length is derived (circumference minus every other aperture); don't set length/count on it")
        for name, aperture in self.apertures.items():
            resolved_names = {mat_name for mat_name, *_ in self._resolve_material(aperture.material)}
            unknown = set(aperture.material_overrides) - resolved_names
            if unknown:
                raise ValueError(
                    f"aperture {name!r} overrides material(s) {sorted(unknown)}, "
                    f"not present in its own material stack {sorted(resolved_names)} -- typo?"
                )

    def _resolve_material(self, name: str) -> list[tuple[str, float, float, float]]:
        """[(originating material name, rho, tau_ps, thickness_m), ...],
        innermost first -- composite materials expanded recursively."""
        material = self.materials[name]
        if material.layers is not None:
            resolved = []
            for sub_name in material.layers:
                resolved.extend(self._resolve_material(sub_name))
            return resolved
        return [(name, material.rho, material.tau_ps, material.thickness_m)]

    def resolved_layers(self, aperture_name: str) -> list[tuple[float, float, float]]:
        """This aperture's layer list with any ``material_overrides``
        applied -- the direct input to ``RoundChamber.layers``."""
        aperture = self.apertures[aperture_name]
        layers = []
        for mat_name, rho, tau_ps, thickness_m in self._resolve_material(aperture.material):
            override = aperture.material_overrides.get(mat_name, {})
            layers.append((
                override.get("rho", rho),
                override.get("tau_ps", tau_ps),
                override.get("thickness_m", thickness_m),
            ))
        return layers

    def default_length_m(self, circumference_m: float) -> float:
        used_m = sum(ap.length_m * ap.count for name, ap in self.apertures.items() if name != "Default")
        remaining_m = circumference_m - used_m
        if remaining_m < 0:
            raise ValueError(f"named apertures already sum to {used_m}m, exceeding the given circumference {circumference_m}m")
        return remaining_m

    def total_length_m(self, aperture_name: str, circumference_m: float) -> float:
        if aperture_name == "Default":
            return self.default_length_m(circumference_m)
        aperture = self.apertures[aperture_name]
        return aperture.length_m * aperture.count

    def aperture_impedance(self, aperture_name: str, freq_Hz: np.ndarray, gamma: float, circumference_m: float) -> Impedance:
        """Zlong(f) for one named aperture (its full ``count``-repeated
        length, or the Default aperture's residual length) -- round
        chambers exactly, flat chambers via the Yokoya approximation
        (module docstring)."""
        aperture = self.apertures[aperture_name]
        chamber = RoundChamber(
            name=aperture_name,
            radius_m=aperture.radius_m if aperture.shape == "round" else min(
                aperture.horizontal_half_aperture_m, aperture.vertical_half_aperture_m
            ),
            length_m=self.total_length_m(aperture_name, circumference_m),
            layers=self.resolved_layers(aperture_name),
        )
        Z = round_chamber_impedance(chamber, freq_Hz, gamma)
        if aperture.shape == "round":
            return Z
        small = min(aperture.horizontal_half_aperture_m, aperture.vertical_half_aperture_m)
        large = max(aperture.horizontal_half_aperture_m, aperture.vertical_half_aperture_m)
        yoklong, *_ = yokoya_elliptic_factors(small, large)
        return Impedance(Z.freq_Hz, Z.ReZ_ohm * yoklong, Z.ImZ_ohm * yoklong)

    def impedance(self, freq_Hz: np.ndarray, gamma: float, circumference_m: float) -> Impedance:
        """Sum of every aperture's Zlong (including Default) -- the total
        ring longitudinal impedance for this Materials/Apertures set."""
        total = None
        for name in self.apertures:
            Z = self.aperture_impedance(name, freq_Hz, gamma, circumference_m)
            total = Z if total is None else total + Z
        return total

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Apertures":
        data = yaml.safe_load(Path(path).read_text())
        materials = {name: Material(**props) for name, props in data.get("Materials", {}).items()}
        apertures = {name: _aperture_from_dict(props) for name, props in data.get("Apertures", {}).items()}
        return cls(materials=materials, apertures=apertures)

    def to_yaml(self, path: str | Path) -> None:
        data = {
            "Materials": {
                name: {k: v for k, v in vars(m).items() if v is not None} for name, m in self.materials.items()
            },
            "Apertures": {name: _aperture_to_dict(a) for name, a in self.apertures.items()},
        }
        Path(path).write_text(yaml.safe_dump(data, sort_keys=False))


def _aperture_from_dict(props: dict) -> Aperture:
    kwargs = {}
    for yaml_key, value in props.items():
        if yaml_key not in _YAML_APERTURE_KEYS:
            raise ValueError(f"unknown aperture field {yaml_key!r}")
        kwargs[_YAML_APERTURE_KEYS[yaml_key]] = value
    return Aperture(**kwargs)


def _aperture_to_dict(aperture: Aperture) -> dict:
    reverse = {v: k for k, v in _YAML_APERTURE_KEYS.items()}
    out = {}
    for field_name, value in vars(aperture).items():
        if value is None or (field_name == "count" and value == 1) or (field_name == "material_overrides" and not value):
            continue
        out[reverse[field_name]] = value
    return out
