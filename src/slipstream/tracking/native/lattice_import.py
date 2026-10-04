"""Converts supported elegant lattice declarations into native `ElementSpec` physics, including
magnetic elements, RFCA/RFCW, H/V correctors and CSR-related settings. See
docs/modules/tracking/native/lattice_import.md."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from pathlib import Path

from ...wakefields.shortrange import ShortRangeWake
from ...wakefields.transverse import TransverseWake

_CONTINUATION_RE = re.compile(r"&[ \t]*\r?\n")
_COMMENT_RE = re.compile(r"!.*")
_STMT_START_RE = re.compile(r"(?m)^[ \t]*([A-Za-z_][\w.-]*)[ \t]*:")
_TYPE_RE = re.compile(r"\s*([A-Za-z_]\w*)\s*(.*)", re.DOTALL)
_PARAM_RE = re.compile(r'([A-Za-z_]\w*)\s*=\s*("[^"]*"|[^,]+)')

_QUAD_TYPES = {"KQUAD", "QUAD"}
_SEXT_TYPES = {"KSEXT"}
_BEND_TYPES = {"CSBEND", "CSBEN", "CSRCSBEND", "CSRCSBEN", "SBEND", "SBEN"}
_CSR_BEND_TYPES = {"CSRCSBEND", "CSRCSBEN"}  # the subset that gets a CSR kick, see ElementSpec.csr
_LINE_TYPES = {"LINE"}
_RF_TYPES = {"RFCA", "RFCW"}
_HKICK_TYPES = {"HKICK"}
_VKICK_TYPES = {"VKICK"}


@dataclass(frozen=True)
class ElementSpec:
    """Native element parameters, including RF, wakes, monitors and correctors."""

    name: str
    kind: str
    length: float
    angle: float = 0.0
    k1: float = 0.0
    k2: float = 0.0
    e1: float = 0.0
    e2: float = 0.0
    dx: float = 0.0
    dy: float = 0.0
    tilt: float = 0.0
    occurrence: int = 0
    voltage_V: float = 0.0
    phase_rad: float = 0.0
    frequency_Hz: float = 0.0
    change_p0: bool = False
    wakefile: str | None = None
    wake_t_column: str | None = None
    wake_wz_column: str | None = None
    wake_wx_column: str | None = None
    wake_wy_column: str | None = None
    wake: ShortRangeWake | None = None
    transverse_wake_x: TransverseWake | None = None
    transverse_wake_y: TransverseWake | None = None
    n_kicks: int = 1
    beta_wave: float = 1.0
    end1_focus: bool = False
    end2_focus: bool = False
    body_focus_model: str = "none"
    csr: bool = False
    csr_drift_rho_m: float | None = None
    csr_drift_start_m: float = 0.0
    hgap: float = 0.0
    fint: float = 0.5
    edge_order: int = 1
    edge1_effects: int = 1
    edge2_effects: int = 1
    csr_bins: int = 256
    csr_sg_halfwidth: int = 1
    csr_n_slices: int = 1
    k3: float = 0.0
    multipoles: tuple[tuple[int, float], ...] = ()
    skew_multipoles: tuple[tuple[int, float], ...] = ()
    integrated_multipoles: tuple[tuple[int, float], ...] = ()
    kick_x: float = 0.0
    """``kind="hkick"`` only, elegant's ``HKICK``'s own ``KICK=`` (radians).
    See ``elements._corrector_kick`` -- a real dipole kick, not folded
    into ``angle``/``track_element`` (that path uses plain Python
    ``math.cos``/``math.sin`` on a fixed lattice curvature, never meant to
    be differentiated; LOCO needs a corrector's own strength
    differentiable exactly at its nominal zero-kick operating point)."""
    kick_y: float = 0.0
    """``kind="vkick"`` only, elegant's ``VKICK``'s own ``KICK=``."""


def _split_statements(text: str) -> list[tuple[str, str]]:
    """``[(name, rest), ...]`` -- ``rest`` is everything after the colon up
    to (not including) the next statement's ``NAME:`` or end of file, with
    a trailing ``;``/whitespace stripped."""
    text = _CONTINUATION_RE.sub(" ", text)
    text = _COMMENT_RE.sub("", text)
    starts = list(_STMT_START_RE.finditer(text))
    out = []
    for i, m in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        rest = text[m.end():end].strip().rstrip(";").strip()
        out.append((m.group(1), rest))
    return out


def _parse_params(body: str | None) -> dict[str, str]:
    if not body:
        return {}
    return {name.upper(): value.strip() for name, value in _PARAM_RE.findall(body)}


def _parse_line_body(body: str) -> list[str]:
    """``(A, B, C)`` -> ``["A", "B", "C"]`` -- bare element/line-name
    references only, see module docstring for what's not supported."""
    inner = body.strip()
    if inner.startswith("(") and inner.endswith(")"):
        inner = inner[1:-1]
    return [tok.strip().upper() for tok in inner.split(",") if tok.strip()]


def _float(params: dict[str, str], key: str, default: float = 0.0) -> float:
    raw = params.get(key)
    return default if raw is None else float(raw)


def _str(params: dict[str, str], key: str) -> str | None:
    """A quoted string parameter (``WAKEFILE="foo.sdds"``) -- ``None`` if
    absent, quotes stripped if present (elegant tolerates either quoted
    or bare strings for a filename/column name with no embedded comma)."""
    raw = params.get(key)
    if raw is None:
        return None
    return raw.strip().strip('"')


def parse_elegant_lattice(lte_path: str | Path, root_name: str) -> list[ElementSpec]:
    """Real per-element ring list for ``NativeTracker``, in beamline order."""
    text = Path(lte_path).read_text()
    elements: dict[str, ElementSpec] = {}
    lines: dict[str, list[str]] = {}

    for name, rest in _split_statements(text):
        name = name.upper()
        m = _TYPE_RE.match(rest)
        if not m:
            continue
        etype, after_type = m.group(1).upper(), m.group(2).strip()
        if etype in _LINE_TYPES:
            body = after_type.split("=", 1)[1] if "=" in after_type else after_type.lstrip(",")
            lines[name] = _parse_line_body(body)
            continue

        body = after_type.lstrip(",")
        params = _parse_params(body)
        length = _float(params, "L")
        if etype in _BEND_TYPES:
            elements[name] = ElementSpec(
                name, "bend", length, angle=_float(params, "ANGLE"), k1=_float(params, "K1"),
                k2=_float(params, "K2"), e1=_float(params, "E1"), e2=_float(params, "E2"),
                hgap=_float(params, "HGAP"), fint=_float(params, "FINT", 0.5),
                edge_order=int(_float(params, "EDGE_ORDER", 1)),
                edge1_effects=int(_float(params, "EDGE1_EFFECTS", 1)),
                edge2_effects=int(_float(params, "EDGE2_EFFECTS", 1)),
                csr=etype in _CSR_BEND_TYPES and bool(_float(params, "CSR", 1)),
                csr_bins=max(3, int(_float(params, "BINS", 256))),
                csr_sg_halfwidth=int(_float(params, "SG_HALFWIDTH", 0)),
                csr_n_slices=max(1, int(_float(params, "N_SLICES", 4))),
            )
        elif etype in _QUAD_TYPES:
            elements[name] = ElementSpec(name, "quad", length, k1=_float(params, "K1"))
        elif etype in _SEXT_TYPES:
            elements[name] = ElementSpec(name, "sext", length, k2=_float(params, "K2"))
        elif etype == "KOCT":
            elements[name] = ElementSpec(name, "oct", length, k3=_float(params, "K3"),
                                         dx=_float(params, "DX"), dy=_float(params, "DY"), tilt=_float(params, "TILT"))
        elif etype == "MULT":
            # Official MULT definition: KNL is integrated strength, ORDER=1 is quad.
            # https://ops.aps.anl.gov/manuals/elegant_latest/elegantsu213.html
            order_raw = _float(params, "ORDER", 1)
            if length < 0:
                raise ValueError(f"{name}: MULT length must be nonnegative")
            if order_raw < 0 or not order_raw.is_integer():
                raise ValueError(f"{name}: MULT ORDER must be a nonnegative integer")
            if _float(params, "BORE") or _float(params, "BTIPL") or _float(params, "DZ") or _float(params, "SYNCH_RAD"):
                raise NotImplementedError(f"{name}: MULT pole-tip, longitudinal-offset and radiation options are unsupported")
            knl = _float(params, "KNL") * _float(params, "FACTOR", 1)
            terms = ((int(order_raw), knl / length if length > 0 else knl),)
            elements[name] = ElementSpec(
                name, "multipole", length, multipoles=terms if length > 0 else (),
                integrated_multipoles=terms if length == 0 else (),
                dx=_float(params, "DX"), dy=_float(params, "DY"), tilt=_float(params, "TILT"),
            )
        elif etype in _RF_TYPES:
            # PHASE is degrees in real elegant -- convert once here.
            n_kicks_raw = int(_float(params, "N_KICKS", 1.0))
            # "none"/"tw1"/"srs" -- see docs/tracking_native.md's body_focus_model note; anything else
            # unrecognized falls back to "none".
            body_focus_model_raw = (_str(params, "BODY_FOCUS_MODEL") or "none").lower()
            body_focus_model = body_focus_model_raw if body_focus_model_raw in ("none", "tw1", "srs") else "none"
            elements[name] = ElementSpec(
                name, "rf", length, voltage_V=_float(params, "VOLT"),
                phase_rad=_float(params, "PHASE") * math.pi / 180.0, frequency_Hz=_float(params, "FREQ"),
                change_p0=_float(params, "CHANGE_P0") != 0.0,
                wakefile=_str(params, "WAKEFILE"), wake_t_column=_str(params, "TCOLUMN"),
                wake_wz_column=_str(params, "WZCOLUMN"), wake_wx_column=_str(params, "WXCOLUMN"),
                wake_wy_column=_str(params, "WYCOLUMN"),
                n_kicks=n_kicks_raw if n_kicks_raw > 0 else 1,
                end1_focus=_float(params, "END1_FOCUS") != 0.0,
                end2_focus=_float(params, "END2_FOCUS") != 0.0,
                body_focus_model=body_focus_model,
            )
        elif etype in _HKICK_TYPES:
            elements[name] = ElementSpec(name, "hkick", length, kick_x=_float(params, "KICK"))
        elif etype in _VKICK_TYPES:
            elements[name] = ElementSpec(name, "vkick", length, kick_y=_float(params, "KICK"))
        elif length != 0.0:
            elements[name] = ElementSpec(name, "drift", length)
        # else: zero-length no-op (MARK/WATCH/CHARGE/CLEAN/RF-with-no-L/etc) -- dropped.

    occurrence_counts: dict[str, int] = {}

    def expand(name: str, _seen: frozenset[str] = frozenset()) -> list[ElementSpec]:
        if name in _seen:
            raise ValueError(f"circular line reference at {name!r}")
        if name in lines:
            out: list[ElementSpec] = []
            for token in lines[name]:
                out.extend(expand(token, _seen | {name}))
            return out
        spec = elements.get(name)
        if spec is None:
            return []
        occurrence_counts[name] = occurrence_counts.get(name, 0) + 1
        return [replace(spec, occurrence=occurrence_counts[name])]

    return _apply_csrdrift_carry(expand(root_name.upper()))


def _apply_csrdrift_carry(elements: list[ElementSpec]) -> list[ElementSpec]:
    """Real elegant's ``CSRDRIFT`` -- see ``ElementSpec``'s own docstring for the model."""
    out = []
    pending_rho_m = 0.0
    distance_m = 0.0
    carrying = False
    csr_bins, csr_sg_halfwidth = 256, 1
    for spec in elements:
        if spec.kind == "bend" and spec.csr and spec.angle != 0.0:
            pending_rho_m = spec.length / spec.angle
            csr_bins, csr_sg_halfwidth = spec.csr_bins, spec.csr_sg_halfwidth
            distance_m = 0.0
            carrying = True
            out.append(spec)
        elif spec.kind == "drift" and carrying:
            out.append(replace(spec, csr_drift_rho_m=pending_rho_m, csr_drift_start_m=distance_m,
                               csr_bins=csr_bins, csr_sg_halfwidth=csr_sg_halfwidth))
            distance_m += spec.length
        else:
            carrying = False
            out.append(spec)
    return out

