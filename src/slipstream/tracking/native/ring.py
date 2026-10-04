"""Provides frozen-energy one-turn nonlinear transport with fixed delta and no RF, radiation
damping/excitation or wakes. See docs/modules/tracking/native/ring.md."""
from __future__ import annotations

from typing import NamedTuple

import functools
import math

import torch
import torch._dynamo

from ...constants import ELECTRON_MASS_EV
from .lattice_import import ElementSpec
from .double_single import DS, _ds_ops, check_error_free, ds_edge, ds_slice, track_element_ds
from .elements import _DRIFT_FRAC4, _DRIFT_FRAC6, _DRIFT_FRAC8, _KICK_FRAC4, _KICK_FRAC6, _KICK_FRAC8
from . import ring_warp
from .tracker import _apply_one_element, _build_params

_FRACS = {4: (_DRIFT_FRAC4, _KICK_FRAC4), 6: (_DRIFT_FRAC6, _KICK_FRAC6), 8: (_DRIFT_FRAC8, _KICK_FRAC8)}


class RingTrackResult(NamedTuple):
    state: torch.Tensor
    """Final ``(..., 6)`` state; lost particles are zeroed."""
    loss_turn: torch.Tensor
    """``(...)`` int64 -- turn index (0-based) at which the particle was
    found outside the aperture, ``-1`` if it survived every turn."""
    tbt: torch.Tensor | None
    """``(n_records, ..., 4)`` ``(x, px, y, py)`` at the ring start, recorded
    after every ``record_every``-th turn -- ``None`` if not requested."""


class RingLossMap(NamedTuple):
    state: torch.Tensor
    loss_turn: torch.Tensor
    loss_element: torch.Tensor
    loss_s_m: torch.Tensor
    tbt: torch.Tensor | None


class RingMap:
    def __init__(
        self, specs: list[ElementSpec], energy_eV: float, mc2_eV: float = ELECTRON_MASS_EV,
        n_kicks: int = 2, n_kicks_bend: int = 8, integration_order: int = 6,
        device: str | torch.device | None = None, dtype: torch.dtype | None = None,
        compile: bool = False, exact_dipole: bool = True, double_single: bool | str = False,
    ):
        """``double_single=True``: every coordinate is a float32 ``hi + lo`` pair
        (``double_single.py``, ~44-bit mantissa) -- float64-grade FMA on a consumer GPU."""
        if double_single not in (False, True, "full", "hybrid"):
            raise ValueError(f"double_single must be False, True/'full' or 'hybrid', got {double_single!r}")
        ds_hybrid = double_single == "hybrid"
        double_single = bool(double_single)
        if double_single and not exact_dipole:
            raise ValueError("double_single needs exact_dipole=True")
        self.specs = specs
        self.energy_eV = energy_eV
        self.mc2_eV = mc2_eV
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # float64 on CPU, float32 on CUDA: consumer GPUs run FP64 at 1/64
        # rate (RTX 3050: 2 s vs 65 ms per 1e5-particle turn).
        self.dtype = torch.float32 if double_single else (dtype or (torch.float32 if self.device.type == "cuda" else torch.float64))
        self.double_single = double_single
        if double_single and any(s.k3 != 0 or s.multipoles or s.skew_multipoles or s.integrated_multipoles for s in specs):
            raise NotImplementedError("double_single: higher/skew multipoles not supported")
        self._kw = dict(
            energy_eV=energy_eV, mc2_eV=mc2_eV, n_kicks=n_kicks, n_kicks_bend=n_kicks_bend,
            integration_order=integration_order, exact_dipole=exact_dipole,
        )
        self._overrides: tuple[dict, dict, dict, dict, dict, dict, dict] = ({}, {}, {}, {}, {}, {}, {})
        # rf_mode="lumped" folds every RF cavity into a drift -- frozen energy.
        self.params = _build_params(
            specs, n_kicks, n_kicks_bend, energy_eV, mc2_eV, self.device, self.dtype,
            integration_order=integration_order, exact_dipole=exact_dipole,
        )
        self._energy_t = torch.tensor(energy_eV, dtype=self.dtype, device=self.device)
        self._mc2_t = torch.tensor(mc2_eV, dtype=self.dtype, device=self.device)
        if double_single and any(p.dx or p.dy or p.tilt for p in self.params):
            raise NotImplementedError("double_single: misalignments not supported")
        base = _apply_one_element
        slice_base = functools.partial(ds_slice, hybrid=ds_hybrid)
        self._ds_slice, self._ds_edge = slice_base, ds_edge
        if compile:
            torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, len(self.params) + 8)
            opts = {"emulate_precision_casts": True} if double_single else None
            self._element = torch.compile(base, dynamic=False)
            if not double_single:
                def _regionalize(p):
                    if (p.k1 or p.k2) and not isinstance(p.k1, torch.Tensor) and not isinstance(p.k2, torch.Tensor):
                        p = p._replace(
                            k1=torch.as_tensor(p.k1, dtype=self.dtype, device=self.device),
                            k2=torch.as_tensor(p.k2, dtype=self.dtype, device=self.device),
                        )
                    if p.angle == 0.0 and p.length > 0.0 and not isinstance(p.length, torch.Tensor):
                        p = p._replace(length=torch.as_tensor(p.length, dtype=self.dtype, device=self.device))
                    return p

                self.params = [_regionalize(p) for p in self.params]
            if double_single:
                self._ds_slice = torch.compile(slice_base, dynamic=False, options=opts)
                self._ds_edge = torch.compile(ds_edge, dynamic=False, options=opts)
                err = check_error_free(torch.compile(_ds_ops, options=opts), self.device)
                if err > 1e-11:
                    raise RuntimeError(f"compiled double-single arithmetic is not error-free here (rel. err {err:.1e})")
        else:
            self._element = base

        self._set_warp_params()

    def _set_warp_params(self) -> None:
        self._warp_params = None
        # Compiled per-element kernels replayed in a CUDA graph beat the packed Warp turn at
        # 1e2-1e5 particles (RTX 3050, MLS example: 0.8 vs 3.6 ms/turn at 100), so Warp
        # serves only eager maps.
        compiled = self._element is not _apply_one_element
        if not compiled and not self.double_single and self.device.type == "cuda" and ring_warp.wk._HAVE_WARP:
            candidate = ring_warp.pack(self.params, self.device, self.dtype)
            if ring_warp.available(self.device, self.dtype, candidate):
                self._warp_params = candidate

    @property
    def circumference_m(self) -> float:
        return sum(s.length for s in self.specs)

    def with_strengths(
        self, k1: dict[str, torch.Tensor] | None = None, k2: dict[str, torch.Tensor] | None = None,
        kick_x: dict[str, torch.Tensor] | None = None, kick_y: dict[str, torch.Tensor] | None = None,
        dx: dict[str, torch.Tensor] | None = None, dy: dict[str, torch.Tensor] | None = None,
        tilt: dict[str, torch.Tensor] | None = None,
    ) -> "RingMap":
        """Copy of this map with every occurrence of each named element family (case-insensitive) using
        the given tensor ``K1``/``K2``/ ``kick_x``/``kick_y``/``dx``/``dy``/``tilt``."""
        k1 = {n.upper(): v for n, v in (k1 or {}).items()}
        k2 = {n.upper(): v for n, v in (k2 or {}).items()}
        kick_x = {n.upper(): v for n, v in (kick_x or {}).items()}
        kick_y = {n.upper(): v for n, v in (kick_y or {}).items()}
        dx = {n.upper(): v for n, v in (dx or {}).items()}
        dy = {n.upper(): v for n, v in (dy or {}).items()}
        tilt = {n.upper(): v for n, v in (tilt or {}).items()}
        unknown = (set(k1) | set(k2) | set(kick_x) | set(kick_y) | set(dx) | set(dy) | set(tilt)) - {s.name.upper() for s in self.specs}
        if unknown:
            raise KeyError(f"no such element(s) in lattice: {sorted(unknown)}")
        if self.double_single:
            if any(float(v) != 0.0 for v in (*kick_x.values(), *kick_y.values())):
                raise NotImplementedError("double_single: nonzero corrector kicks not supported")
            if any(float(v) != 0.0 for v in (*dx.values(), *dy.values(), *tilt.values())):
                raise NotImplementedError("double_single: misalignments not supported")
            # constants must be Python floats (split into hi/lo at trace time)
            k1 = {n: float(v) for n, v in k1.items()}
            k2 = {n: float(v) for n, v in k2.items()}
        new = object.__new__(RingMap)
        new.__dict__.update(self.__dict__)
        new._overrides = (
            {**self._overrides[0], **k1}, {**self._overrides[1], **k2},
            {**self._overrides[2], **kick_x}, {**self._overrides[3], **kick_y},
            {**self._overrides[4], **dx}, {**self._overrides[5], **dy}, {**self._overrides[6], **tilt},
        )
        new.params = [
            p._replace(
                k1=k1.get(s.name.upper(), p.k1), k2=k2.get(s.name.upper(), p.k2),
                kick_value=(
                    (kick_x if p.kick_plane == "x" else kick_y).get(s.name.upper(), p.kick_value)
                    if p.is_kick else p.kick_value
                ),
                dx=dx.get(s.name.upper(), p.dx), dy=dy.get(s.name.upper(), p.dy),
                tilt=tilt.get(s.name.upper(), p.tilt),
            )
            for s, p in zip(self.specs, self.params)
        ]
        new._set_warp_params()
        return new

    def float64_reference(self) -> "RingMap":
        """Same lattice, slicing and strength overrides as a plain CPU
        float64 map (optics for a ``double_single`` map)."""
        ref = RingMap(self.specs, **self._kw, device="cpu", dtype=torch.float64)
        as_t = lambda d: {n: torch.as_tensor(v, dtype=torch.float64, device="cpu") for n, v in d.items()}  # noqa: E731
        ref = ref.with_strengths(
            k1=as_t(self._overrides[0]), k2=as_t(self._overrides[1]),
            kick_x=as_t(self._overrides[2]), kick_y=as_t(self._overrides[3]),
            dx=as_t(self._overrides[4]), dy=as_t(self._overrides[5]), tilt=as_t(self._overrides[6]),
        )
        # Copy the actual instance parameters, including occurrence-specific
        # alignments, without grouping them back into family dictionaries.
        ref.params = [q._replace(**{key: (v.to(device="cpu", dtype=torch.float64)
                                         if isinstance(v, torch.Tensor) else v)
                                    for key in ("dx", "dy", "tilt", "k3")
                                    for v in (getattr(p, key),)},
                                **{key: tuple((order, strength.to(device="cpu", dtype=torch.float64)
                                              if isinstance(strength, torch.Tensor) else strength)
                                             for order, strength in getattr(p, key))
                                   for key in ("multipoles", "skew_multipoles", "integrated_multipoles")})
                      for p, q in zip(self.params, ref.params)]
        return ref

    def with_alignments(
        self, dx: dict[str, torch.Tensor] | None = None,
        dy: dict[str, torch.Tensor] | None = None,
        tilt: dict[str, torch.Tensor] | None = None,
    ) -> "RingMap":
        """Copy with absolute family offsets (metres) and rolls (radians)."""
        if self.double_single:
            raise NotImplementedError("double_single: misalignments not supported")
        fields = [{n.upper(): v for n, v in (d or {}).items()} for d in (dx, dy, tilt)]
        unknown = set().union(*(set(d) for d in fields)) - {s.name.upper() for s in self.specs}
        if unknown:
            raise KeyError(f"no such element(s) in lattice: {sorted(unknown)}")
        new = object.__new__(RingMap)
        new.__dict__.update(self.__dict__)
        new.params = [p._replace(**{key: d.get(s.name.upper(), getattr(p, key))
                                   for key, d in zip(("dx", "dy", "tilt"), fields)})
                      for s, p in zip(self.specs, self.params)]
        new._warp_params = None
        return new

    def turn(self, state: torch.Tensor, eager: bool = False) -> torch.Tensor:
        """One turn, no loss check -- differentiable."""
        if self.double_single:
            for p in self.params:
                state = track_element_ds(
                    state, p.length, p.angle, p.k1, p.k2, p.e1, p.e2, p.n_kicks, *_FRACS[p.integration_order],
                    p.hgap, p.fint, p.edge_order, p.edge1_effects, p.edge2_effects,
                    slice_fn=self._ds_slice, edge_fn=self._ds_edge,
                )
            return state
        if self._warp_params is not None and not eager and not self._needs_grad(state):
            return ring_warp.turn(state, self._warp_params, self.energy_eV, self.mc2_eV)
        element = _apply_one_element if eager else self._element
        for p in self.params:
            state = element(state, p, self._energy_t, self._mc2_t)[0]
        return state

    def turn_with_taps(self, state: torch.Tensor, tap_indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute turn with taps; see the module documentation for details."""
        if self.double_single:
            raise NotImplementedError("turn_with_taps: double_single not supported")
        captured: dict[int, torch.Tensor] = {}
        want = set(tap_indices)
        for i, p in enumerate(self.params):
            state = _apply_one_element(state, p, self._energy_t, self._mc2_t)[0]
            if i in want:
                captured[i] = state
        tapped = torch.stack([captured[i] for i in tap_indices], dim=-2)
        return state, tapped

    def _needs_grad(self, state: torch.Tensor) -> bool:
        return torch.is_grad_enabled() and (
            state.requires_grad
            or any(
                isinstance(v, torch.Tensor) and v.requires_grad
                for p in self.params
                for v in (
                    p.k1, p.k2, p.kick_value, p.k3, p.dx, p.dy, p.tilt,
                    *(strength for _, strength in p.multipoles + p.skew_multipoles + p.integrated_multipoles),
                )
            )
        )

    def track_loss_map(
        self, state: torch.Tensor, n_turns: int, aperture: tuple[float, float] = (0.05, 0.05),
        element_apertures: dict[str, tuple[float, float]] | None = None, record_every: int = 0,
    ) -> "RingLossMap":
        """Check losses at every element exit; first loss location is an element index."""
        if self.double_single:
            raise NotImplementedError("per-element loss maps require ordinary PyTorch transport")
        if n_turns < 0 or record_every < 0:
            raise ValueError("turn and recording counts must be nonnegative")
        limits = {k.upper(): v for k, v in (element_apertures or {}).items()}
        unknown = set(limits) - {s.name.upper() for s in self.specs}
        if unknown:
            raise KeyError(f"no such aperture element: {sorted(unknown)}")
        if any(len(a) != 2 or any(v <= 0 or not math.isfinite(v) for v in a) for a in [aperture, *limits.values()]):
            raise ValueError("aperture half widths must be finite and positive")
        state = state.to(self.device, self.dtype)
        loss_turn = torch.full(state.shape[:-1], -1, device=self.device, dtype=torch.int64)
        loss_element = torch.full_like(loss_turn, -1)
        records = []
        for t in range(n_turns):
            for i, (s, p) in enumerate(zip(self.specs, self.params)):
                state = self._element(state, p, self._energy_t, self._mc2_t)[0]
                ax, ay = limits.get(s.name.upper(), aperture)
                bad = (~torch.isfinite(state).all(-1)) | (state[..., 0].abs() > ax) | (state[..., 2].abs() > ay)
                first = bad & (loss_turn < 0)
                loss_turn = torch.where(first, t, loss_turn)
                loss_element = torch.where(first, i, loss_element)
                state = torch.where((loss_turn >= 0).unsqueeze(-1), torch.zeros_like(state), state)
            if record_every and (t + 1) % record_every == 0:
                records.append(state[..., :4])
        positions = state.new_tensor([sum(s.length for s in self.specs[:i + 1]) for i in range(len(self.specs))])
        loss_s = torch.full_like(loss_turn, math.nan, dtype=self.dtype)
        if self.specs:
            loss_s = torch.where(loss_element >= 0, positions[loss_element.clamp_min(0)], loss_s)
        return RingLossMap(state, loss_turn, loss_element, loss_s, torch.stack(records) if records else None)

    def track(
        self, state: torch.Tensor, n_turns: int, aperture: tuple[float, float] = (0.05, 0.05),
        record_every: int = 0, checkpoint: bool = False,
    ) -> RingTrackResult:
        """``state`` ``(..., 6)``."""
        batch_shape = state.shape[:-1]
        if self.double_single:
            ds = DS.of(state.to(device=self.device, dtype=torch.float64))
            state = torch.cat([ds.hi, ds.lo], dim=-1).reshape(-1, 12)
            if state.shape[0] > 1:  # strict mark: maybe_mark_dynamic is ignored under dynamic=False
                torch._dynamo.mark_dynamic(state, 0)
        else:
            state = state.to(device=self.device, dtype=self.dtype)
        ax, ay = aperture
        loss_turn = torch.full(state.shape[:-1], -1, dtype=torch.int64, device=self.device)
        t_dev = torch.zeros((), dtype=torch.int64, device=self.device)

        def step(state, loss_turn, turn_fn):
            state = turn_fn(state)
            bad = (~torch.isfinite(state).all(-1)) | (state[..., 0].abs() > ax) | (state[..., 2].abs() > ay)
            loss_turn = torch.where(bad & (loss_turn < 0), t_dev, loss_turn)
            state = torch.where((loss_turn >= 0).unsqueeze(-1), torch.zeros_like(state), state)
            return state, loss_turn

        records = []
        if self.device.type == "cuda" and not self._needs_grad(state):
            with torch.no_grad():
                s_state, s_loss = state.clone().contiguous(), loss_turn.clone()
                turn_fn = self.turn
                prepared = None
                if self._warp_params is not None:
                    prepared = ring_warp.prepare(s_state, self._warp_params)
                    turn_fn = lambda s: ring_warp.turn(  # noqa: E731
                        s, self._warp_params, self.energy_eV, self.mc2_eV, out=s, prepared=prepared,
                    )

                def graphed():
                    new_state, new_loss = step(s_state, s_loss, turn_fn)
                    s_state.copy_(new_state)
                    s_loss.copy_(new_loss)
                    t_dev.add_(1)

                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):  # warm-up (triggers any compile) off the capture
                    graphed()
                torch.cuda.current_stream().wait_stream(side)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    graphed()
                s_state.copy_(state)
                s_loss.fill_(-1)
                t_dev.zero_()
                for t in range(n_turns):
                    graph.replay()
                    if record_every and (t + 1) % record_every == 0:
                        records.append(self._coords(s_state).clone())
                state, loss_turn = s_state, s_loss
        else:
            turn_fn = self.turn
            if checkpoint:
                turn_fn = lambda s: torch.utils.checkpoint.checkpoint(self.turn, s, use_reentrant=False)  # noqa: E731
            for t in range(n_turns):
                state, loss_turn = step(state, loss_turn, turn_fn)
                t_dev += 1
                if record_every and (t + 1) % record_every == 0:
                    records.append(self._coords(state))
        tbt = torch.stack(records) if records else None
        if self.double_single:
            state = (state[..., :6].double() + state[..., 6:].double()).reshape(*batch_shape, 6)
            loss_turn = loss_turn.reshape(batch_shape)
            tbt = tbt.reshape(tbt.shape[0], *batch_shape, 4) if tbt is not None else None
        return RingTrackResult(state, loss_turn, tbt)

    def _coords(self, state: torch.Tensor) -> torch.Tensor:
        if self.double_single:
            return state[..., :4].double() + state[..., 6:10].double()
        return state[..., :4]
