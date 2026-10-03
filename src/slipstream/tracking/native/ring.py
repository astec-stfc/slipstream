"""Frozen-energy one-turn ring map for nonlinear-dynamics studies (DA, FMA,
detuning, gradient-based sextupole optimisation) -- see
plans/RING_NONLINEAR_DYNAMICS_PLAN.md.

Same per-element transport as ``NativeTracker`` (``tracker._build_params``/
``tracker._apply_elements``, every element exactly integrated every turn,
so every K2 kick is always applied -- there is no linear-matrix fast path to
fall into), but without RF, radiation damping, quantum excitation or wakes:
``delta`` is a constant of motion, the standard setting for DA/FMA
cross-checks against elegant. For RF-on 6D studies use ``NativeTracker``
itself (it reuses ``pystable``'s longitudinal physics).

**Slicing defaults are finer than ``NativeTracker``'s** (elegant's own
defaults, 1/4 slices at order 4): on ``br14_simple.lte`` those leave the
tunes 0.07/0.15 away from the converged 0.41/0.38 -- in elegant's own
tracking too (0.350/0.244), the two codes' unconverged artifacts even
differing from each other. Order 6 with 2/8 slices is within 5e-5 of
converged (1e-4 relative on chromaticity), and matches elegant's
``twiss_output`` chromaticity to 5 digits (tests/test_ring_nonlinear.py).
Always converge slicing before trusting a nonlinear observable.

**Bends use the exact-dipole splitting by default** (``exact_dipole=True``,
see ``elements._exact_dipole_drift``): same Hamiltonian as elegant's
splitting, converges to the same map, but float32-robust -- float32 DA on
br14 is within 1% of elegant with it, 8% off without. Pass
``exact_dipole=False`` to integrate elegant's own map (live-elegant
comparisons at equal slicing).

Two things this adds on top of ``NativeTracker``'s transport:

- **Losses** (``track``): a per-turn aperture check at the ring start
  (``|x| > aperture[0]``, ``|y| > aperture[1]`` or any non-finite
  coordinate). Lost particles are frozen at zero and keep their slot in the
  batch -- the batch is never split -- and their loss turn is recorded.
- **Tensor strengths** (``with_strengths``): any element family's ``K1``/
  ``K2`` can be replaced by a tensor. A 0-d ``requires_grad`` tensor makes
  the ring differentiable in that strength (autograd knob); a tensor of
  shape ``(M, 1)`` with particles shaped ``(M, N, 6)`` tracks ``M``
  different machines (an error ensemble) in one batch.
"""
from __future__ import annotations

from typing import NamedTuple

import functools

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


class RingMap:
    def __init__(
        self, specs: list[ElementSpec], energy_eV: float, mc2_eV: float = ELECTRON_MASS_EV,
        n_kicks: int = 2, n_kicks_bend: int = 8, integration_order: int = 6,
        device: str | torch.device | None = None, dtype: torch.dtype | None = None,
        compile: bool = False, exact_dipole: bool = True, double_single: bool | str = False,
    ):
        """``double_single=True``: every coordinate is a float32 ``hi + lo``
        pair (``double_single.py``, ~44-bit mantissa) -- float64-grade FMA
        on a consumer GPU. Needs ``exact_dipole=True``, no misalignments,
        Python-float (or 0-d tensor) strengths; no autograd, ``tau`` not
        tracked. Optics (``analysis.nonlinear.linear_optics``) are then
        computed on ``float64_reference()``. ``double_single="hybrid"``:
        only the linear, constant-coefficient part of each update is
        double-single; small nonlinear remainders are plain float32
        (``double_single._hy_*``) -- cheaper, ~1e-12..1e-10 per element
        instead of ~1e-15."""
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
        self._kw = dict(
            energy_eV=energy_eV, mc2_eV=mc2_eV, n_kicks=n_kicks, n_kicks_bend=n_kicks_bend,
            integration_order=integration_order, exact_dipole=exact_dipole,
        )
        self._overrides: tuple[dict, dict] = ({}, {})
        # rf_mode="lumped" folds every RF cavity into a drift -- frozen energy.
        self.params = _build_params(
            specs, n_kicks, n_kicks_bend, energy_eV, mc2_eV, self.device, self.dtype,
            integration_order=integration_order, exact_dipole=exact_dipole,
        )
        self._energy_t = torch.tensor(energy_eV, dtype=self.dtype, device=self.device)
        self._mc2_t = torch.tensor(mc2_eV, dtype=self.dtype, device=self.device)
        # Per-element compile, not whole-turn: an unrolled ~5000-stage turn
        # graph was still compiling after 15 min / 10 GB (same failure as
        # LinacTracker's whole-pass compile). One graph per distinct element
        # config fuses each element into a few kernels -- further reduced to
        # one graph per element *family* below (K1/K2/length regionalization).
        if double_single and any(p.dx or p.dy or p.tilt for p in self.params):
            raise NotImplementedError("double_single: misalignments not supported")
        base = _apply_one_element
        slice_base = functools.partial(ds_slice, hybrid=ds_hybrid)
        self._ds_slice, self._ds_edge = slice_base, ds_edge
        if compile:
            torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, len(self.params) + 8)
            # Triton contracts a*b+c into FMA by default, which breaks the
            # Dekker split inside double-single products (measured: 1e-6
            # instead of 1e-13); emulate_precision_casts turns FP fusion off
            # for these kernels only.
            opts = {"emulate_precision_casts": True} if double_single else None
            self._element = torch.compile(base, dynamic=False)
            if not double_single:
                # Regional compilation -- promote the numeric fields that
                # vary per *instance* of a family (K1/K2, a drift's length)
                # to 0-d tensors, so dynamo guards on shape/dtype/device
                # instead of value and every same-family element reuses one
                # graph (27 -> 7 graphs measured on br14_simple; full
                # rationale + numbers in docs/tracking_native.md's
                # "Compilation-time discipline"). Left as plain floats
                # wherever a *structural* check downstream reads the
                # original value -- K1/K2 exactly zero (keeps
                # track_element's `isinstance`-gated fast paths, elements.py)
                # and length for angle!=0 or non-positive length (the `h =
                # angle/length` divide and the `length <= 0.0` early-out
                # must resolve in plain Python, never on a live tensor).
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
        if not self.double_single and self.device.type == "cuda" and ring_warp.wk._HAVE_WARP:
            candidate = ring_warp.pack(self.params, self.device, self.dtype)
            if ring_warp.available(self.device, self.dtype, candidate):
                self._warp_params = candidate

    @property
    def circumference_m(self) -> float:
        return sum(s.length for s in self.specs)

    def with_strengths(
        self, k1: dict[str, torch.Tensor] | None = None, k2: dict[str, torch.Tensor] | None = None,
    ) -> "RingMap":
        """Copy of this map with every occurrence of each named element
        family (case-insensitive) using the given tensor ``K1``/``K2``.
        The per-element reference-orbit offset stays at its nominal value
        (a discretization correction; its strength dependence is ~1e-12)."""
        k1 = {n.upper(): v for n, v in (k1 or {}).items()}
        k2 = {n.upper(): v for n, v in (k2 or {}).items()}
        unknown = (set(k1) | set(k2)) - {s.name.upper() for s in self.specs}
        if unknown:
            raise KeyError(f"no such element(s) in lattice: {sorted(unknown)}")
        if self.double_single:  # constants must be Python floats (split into hi/lo at trace time)
            k1 = {n: float(v) for n, v in k1.items()}
            k2 = {n: float(v) for n, v in k2.items()}
        new = object.__new__(RingMap)
        new.__dict__.update(self.__dict__)
        new._overrides = ({**self._overrides[0], **k1}, {**self._overrides[1], **k2})
        new.params = [
            p._replace(k1=k1.get(s.name.upper(), p.k1), k2=k2.get(s.name.upper(), p.k2))
            for s, p in zip(self.specs, self.params)
        ]
        new._set_warp_params()
        return new

    def float64_reference(self) -> "RingMap":
        """Same lattice, slicing and strength overrides as a plain CPU
        float64 map (optics for a ``double_single`` map)."""
        ref = RingMap(self.specs, **self._kw, device="cpu", dtype=torch.float64)
        as_t = lambda d: {n: torch.as_tensor(v, dtype=torch.float64) for n, v in d.items()}  # noqa: E731
        return ref.with_strengths(k1=as_t(self._overrides[0]), k2=as_t(self._overrides[1]))

    def turn(self, state: torch.Tensor, eager: bool = False) -> torch.Tensor:
        """One turn, no loss check -- differentiable. ``eager=True``
        bypasses ``compile`` (Jacobians via ``vmap`` go this way). A
        ``double_single`` map takes and returns the ``(..., 12)`` hi/lo
        state."""
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

    def _needs_grad(self, state: torch.Tensor) -> bool:
        return torch.is_grad_enabled() and (
            state.requires_grad or any(isinstance(v, torch.Tensor) and v.requires_grad for p in self.params for v in (p.k1, p.k2))
        )

    def track(
        self, state: torch.Tensor, n_turns: int, aperture: tuple[float, float] = (0.05, 0.05),
        record_every: int = 0, checkpoint: bool = False,
    ) -> RingTrackResult:
        """``state`` ``(..., 6)``. ``record_every=k>0`` stores ``(x, px, y,
        py)`` after turns ``k-1, 2k-1, ...`` (memory: ``n_turns/k * N * 4``
        numbers -- decimate or track a subset for 1e6-particle runs).
        ``checkpoint=True`` recomputes each turn in the backward pass
        (``torch.utils.checkpoint``) -- memory O(1 turn) for autograd
        through many turns.

        On CUDA, when nothing needs a gradient, one turn (every element
        plus the loss bookkeeping) is captured once into a CUDA graph and
        replayed -- the per-turn cost is then pure GPU time, no Python/
        dynamo-guard overhead per element.

        ``double_single``: ``state`` is converted to hi/lo from float64;
        ``tbt`` and the returned state are float64 (``hi + lo``)."""
        batch_shape = state.shape[:-1]
        if self.double_single:
            ds = DS.of(state.to(device=self.device, dtype=torch.float64))
            # Flat batch marked dynamic: the compiled slices otherwise
            # specialise on particle count, and one double-single recompile
            # of the whole ring costs ~18 min.
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
            # ponytail: loss checked once per turn at s=0, not per element --
            # a particle past the DA diverges within a few turns anyway, so
            # the loss *turn* can be late by O(1) turns; add a per-element
            # check if loss maps (where, not when) are ever needed.
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
