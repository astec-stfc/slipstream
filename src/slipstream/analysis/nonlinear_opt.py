"""Tensor magnet strengths support differentiable chromaticity and tune-footprint objectives. See
docs/modules/analysis/nonlinear_opt.md."""
from __future__ import annotations

import dataclasses
import math

import torch

from ..tracking.native.lattice_import import ElementSpec
from ..tracking.native.ring import RingMap
from . import nonlinear as nl

SEXTUPOLES = ("SS1", "SS2")


def periodic_specs(specs: list[ElementSpec], names: tuple[str, ...] = SEXTUPOLES, n_periods: int = 2) -> list[ElementSpec]:
    """Rename every ``names`` instance ``<name>_<k>`` by its position ``k`` within one of
    ``n_periods`` identical periods (instance ``k`` and ``k + n/n_periods`` share a name): an
    independently powered element per position, periods kept identical."""
    total = {n: sum(s.name == n for s in specs) for n in names}
    assert all(t % n_periods == 0 for t in total.values()), f"expected {n_periods} identical periods"
    seen = {n: 0 for n in names}
    out = []
    for s in specs:
        if s.name in names:
            k = seen[s.name] % (total[s.name] // n_periods) + 1
            seen[s.name] += 1
            s = dataclasses.replace(s, name=f"{s.name}_{k}")
        out.append(s)
    return out


def unique_names(specs: list[ElementSpec], kinds: tuple[str, ...] = ("quad", "bend")) -> list[ElementSpec]:
    """Rename every element of the given ``kind`` ``<name>#<occurrence>``, so each instance is its own
    family (per-magnet errors). ``ElementSpec.occurrence`` is elegant's 1-based instance number."""
    return [dataclasses.replace(s, name=f"{s.name}#{s.occurrence}") if s.kind in kinds else s for s in specs]


class KnobSet:
    """Named K2 knobs. ``groups[label]`` lists the element names one knob sets. ``family4``: BD, BF, SS1, SS2
    (the hardware families); ``full``: BD, BF and one knob per sextupole position (needs ``periodic_specs``)."""

    def __init__(self, specs: list[ElementSpec], mode: str, combined: tuple[str, ...] = ("BD", "BF")):
        names = sorted({s.name for s in specs})
        pos = lambda base: sorted((n for n in names if n.startswith(base + "_")), key=lambda n: int(n.split("_")[1]))  # noqa: E731
        self.groups: dict[str, list[str]] = {c: [c] for c in combined}
        if mode == "family4":
            self.groups.update({b: pos(b) for b in SEXTUPOLES})
        elif mode == "full":
            self.groups.update({n: [n] for b in SEXTUPOLES for n in pos(b)})
        else:
            raise ValueError(mode)
        self.labels = list(self.groups)
        by_name = {s.name: s.k2 for s in specs}
        self.theta0 = torch.tensor([by_name[self.groups[k][0]] for k in self.labels], dtype=torch.float64)

    def __len__(self) -> int:
        return len(self.labels)

    def k2(self, theta: torch.Tensor) -> dict[str, torch.Tensor]:
        return {n: theta[i] for i, k in enumerate(self.labels) for n in self.groups[k]}

    def ring(self, base: RingMap, theta: torch.Tensor, extra_k1: dict | None = None, extra_k2: dict | None = None) -> RingMap:
        """``base`` with ``theta`` applied; ``extra_*`` (e.g. ``(M, 1)`` error tensors, keyed by element
        name) are added to the knob values where a name appears in both."""
        k2 = self.k2(theta)
        for n, v in (extra_k2 or {}).items():
            k2[n] = k2[n] + v if n in k2 else v
        return base.with_strengths(k1=extra_k1, k2=k2)

    def elegant_k2(self, theta: torch.Tensor, specs: list[ElementSpec]) -> dict:
        """The same strengths as elegant ``alter_elements`` keys: combined-function names by family and
        ``(SS1|SS2, occurrence)`` for every sextupole instance (``specs`` = the renamed list)."""
        out: dict = {k: float(theta[self.labels.index(k)]) for k in self.groups if k not in SEXTUPOLES and not k.startswith(SEXTUPOLES)}
        seen: dict[str, int] = {}
        for s in specs:
            base = s.name.split("_")[0]
            if base in SEXTUPOLES:
                seen[base] = seen.get(base, 0) + 1
                out[(base, seen[base])] = float(theta[next(i for i, k in enumerate(self.labels) if s.name in self.groups[k])])
        return out


def chromaticity_response(ring: RingMap, knobs: KnobSet, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(xi, A)``: chromaticities ``(2,)`` at ``theta`` and ``A = d xi / d theta`` ``(2, n)`` by reverse-mode
    autograd through the Jacobian-based optics (two backward passes, however many knobs)."""
    th = theta.clone().requires_grad_(True)
    xi = torch.stack(nl.chromaticity(knobs.ring(ring, th), create_graph=True))
    rows = [torch.autograd.grad(xi[i], th, retain_graph=True)[0] for i in range(2)]
    return xi.detach(), torch.stack(rows)


def null_basis(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(N, P)``: ``N`` ``(n, n-2)`` orthonormal directions that leave the chromaticity unchanged (to first
    order) and ``P = A^+`` the minimum-norm chromaticity correction."""
    _, _, Vh = torch.linalg.svd(A, full_matrices=True)
    return Vh[2:].T.contiguous(), torch.linalg.pinv(A)


@dataclasses.dataclass
class Launches:
    state: torch.Tensor  # (B, 6)
    centers: tuple[torch.Tensor, torch.Tensor]  # (B,) linear tunes NAFF searches around
    delta: torch.Tensor  # (B,)
    kind: list[str]  # "ref" | "amp" | "chrom" | "diff"


def build_launches(
    opt_by_delta: dict[float, nl.LinearOptics], amps_mm, chrom_deltas, a_small_mm: float = 0.3, device="cpu",
    diff_points: tuple = (),
) -> Launches:
    """A small-amplitude reference (delta = 0), amplitude scans in x, y and on the diagonal at delta =
    0, and small-amplitude launches at ``chrom_deltas`` (each on that delta's closed orbit from
    ``opt_by_delta``)."""
    o0 = opt_by_delta[0.0]
    a = a_small_mm * 1e-3
    rows, centers, deltas, kind = [nl.launch(o0, a, a)], [(o0.Qx, o0.Qy)], [0.0], ["ref"]

    def add(o, x, y, d, k):
        rows.append(nl.launch(o, x, y))
        centers.append((o.Qx, o.Qy))
        deltas.append(float(d))
        kind.append(k)

    for amp in amps_mm:
        A = amp * 1e-3
        for x, y in ((A, a), (a, A), (A / math.sqrt(2), A / math.sqrt(2))):
            add(o0, x, y, 0.0, "amp")
    for d in chrom_deltas:
        add(opt_by_delta[float(d)], a, a, d, "chrom")
    for x_mm, y_mm, d in diff_points:
        add(opt_by_delta[float(d)], x_mm * 1e-3, y_mm * 1e-3, d, "diff")
    state = torch.stack(rows).to(device)
    cx = torch.stack([c[0] for c in centers]).to(device)
    cy = torch.stack([c[1] for c in centers]).to(device)
    return Launches(state, (cx, cy), torch.tensor(deltas, dtype=torch.float64, device=device), kind)


class Footprint:
    """``loss(ring)``: a smooth surrogate for dynamic aperture, differentiable in every tensor strength
    of ``ring`` (NAFF and the tracking both are)."""

    def __init__(self, launches: Launches, opt_norm: nl.LinearOptics, xi_target: tuple[float, float], n_turns: int = 256,
                 checkpoint: bool = False, weights: dict[str, float] | None = None, n_machines: int | None = None,
                 std_weight: float = 0.0, diff_weight: float = 0.0, diff_scale: float = 1e-5, lost_penalty: float = 1e3,
                 loss_aperture: float = 0.1):
        self.L, self.opt, self.T, self.ckpt, self.M, self.lam = launches, opt_norm, n_turns, checkpoint, n_machines, std_weight
        self.diff_weight, self.diff_scale, self.lost_penalty, self.loss_aperture = diff_weight, diff_scale, lost_penalty, loss_aperture
        dev = launches.state.device
        self.xi = torch.tensor(xi_target, dtype=torch.float64, device=dev)
        w = {"amp": 1.0, "chrom": 1.0, **(weights or {})}
        self.w = torch.tensor([w.get(k, 0.0) for k in launches.kind], dtype=torch.float64, device=dev)
        self.is_diff = torch.tensor([k == "diff" for k in launches.kind], dtype=torch.bool, device=dev)

    def analyse(self, ring: RingMap) -> tuple[torch.Tensor, ...]:
        """``(Qx, Qy, D, lost)`` for every launch (``(..., B)``): full-run tunes and the half-run tune change."""
        state = self.L.state.to(ring.dtype)
        if self.M is not None:
            state = state.expand(self.M, *state.shape)
        ap = (self.loss_aperture, self.loss_aperture)
        dead = None
        if ring._needs_grad(state):
            with torch.no_grad():
                dead = ring.track(state, self.T, record_every=0, aperture=ap).loss_turn >= 0
            state = torch.where(dead.unsqueeze(-1), self.L.state[0].to(ring.dtype), state)
        res = ring.track(state, self.T, record_every=1, aperture=ap, checkpoint=self.ckpt)
        zx, zy = nl.normalized_signals(res.tbt, self.opt)
        lost = res.loss_turn >= 0
        if dead is not None:
            lost = lost | dead
        n = torch.arange(zx.shape[-1], dtype=zx.real.dtype, device=zx.device)
        benign = torch.exp(2j * math.pi * 0.3 * n)  # a lost (zeroed) particle must not put NaNs into the graph
        zx, zy = (torch.where(lost.unsqueeze(-1), benign, z) for z in (zx, zy))
        cx, cy = (c.to(zx.device, zx.real.dtype) for c in self.L.centers)
        h = self.T // 2
        kx, ky = dict(center=cx, half_width=0.03), dict(center=cy, half_width=0.03)
        qx, qy = nl.naff(zx, **kx)[0], nl.naff(zy, **ky)[0]
        if self.diff_weight > 0 and bool(self.is_diff.any()):
            d = []
            for z, k in ((zx, kx), (zy, ky)):
                a1, a2 = nl.naff(z[..., :h], **k)[0], nl.naff(z[..., h:], **k)[0]
                d.append(torch.remainder(a2 - a1 + 0.5, 1.0) - 0.5)
            D = torch.sqrt(d[0] ** 2 + d[1] ** 2 + 1e-24)
            D = torch.where(lost, torch.full_like(D, 0.1), D)
        else:
            D = torch.zeros_like(qx)
        return qx, qy, D, lost

    def tunes(self, ring: RingMap) -> tuple[torch.Tensor, torch.Tensor]:
        qx, qy, _, lost = self.analyse(ring)
        if (lost & ~self.is_diff).any():
            raise ValueError("launch lost -- reduce amplitudes")
        return qx, qy

    def shifts(self, ring: RingMap) -> torch.Tensor:
        """``(..., B, 2)`` tune shifts in 1e-3."""
        qx, qy = self.tunes(ring)
        dqx = qx - qx[..., :1] - self.xi[0] * self.L.delta
        dqy = qy - qy[..., :1] - self.xi[1] * self.L.delta
        return torch.stack([dqx, dqy], -1) * 1e3

    def per_machine(self, ring: RingMap) -> torch.Tensor:
        qx, qy, D, lost = self.analyse(ring)
        dqx = qx - qx[..., :1] - self.xi[0] * self.L.delta
        dqy = qy - qy[..., :1] - self.xi[1] * self.L.delta
        dq2 = (dqx**2 + dqy**2) * 1e6
        dq2 = torch.where(lost, torch.full_like(dq2, self.lost_penalty), dq2)  # a lost launch: constant penalty, no gradient
        foot = (dq2 * self.w).sum(-1) / self.w.sum()
        if self.diff_weight > 0 and bool(self.is_diff.any()):
            diff = torch.log1p((D / self.diff_scale) ** 2)[..., self.is_diff].mean(-1)
            return foot + self.diff_weight * diff
        return foot

    def loss(self, ring: RingMap) -> torch.Tensor:
        per = self.per_machine(ring)
        if self.M is None:
            return per
        return per.mean() + self.lam * per.std()


def sample_field_errors(
    specs: list[ElementSpec], rules: list[tuple[str, str, float]], n_machines: int, seed: int = 0,
    dtype=torch.float64, device="cpu", cutoff: float = 3.0,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Per-element Gaussian field errors for ``n_machines`` machines, as ``(M, 1)`` ``(k1 dict, k2
    dict)`` ready for ``RingMap.with_strengths`` (specs must have per-element names,
    ``unique_names``)."""
    g = torch.Generator().manual_seed(seed)
    k1: dict[str, torch.Tensor] = {}
    k2: dict[str, torch.Tensor] = {}
    for s in specs:
        for kind, param, rms in rules:
            if s.kind != kind:
                continue
            nominal = s.k1 if param == "K1" else s.k2
            z = torch.randn(n_machines, 1, generator=g, dtype=torch.float64).clamp(-cutoff, cutoff)
            err = (nominal * rms * z).to(dtype=dtype, device=device)
            (k1 if param == "K1" else k2)[s.name] = torch.as_tensor(nominal, dtype=dtype, device=device) + err
    return k1, k2


def action_growth_loss(tbt: torch.Tensor, opt: nl.LinearOptics, scale: float = 1.0) -> torch.Tensor:
    """Smooth soft-survival surrogate: mean squared soft positive action growth."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    zx, zy = nl.normalized_signals(tbt, opt)
    action = (zx.abs().square() + zy.abs().square()) / 2
    reference = action[..., :1].clamp_min(torch.finfo(action.dtype).tiny)
    growth = (action / reference - 1) / scale
    # Smooth positive part; even constant action has a log(2)**2 baseline.
    positive = torch.nn.functional.softplus(growth)
    return positive.square().mean(-1)


def joint_optics_loss(ring: RingMap, target: nl.LinearOptics, tune_scale: float = 1e-3,
                      beta_scale: float = 0.01) -> torch.Tensor:
    """Differentiable tune and fractional beta constraints for K1/K2 optimisation."""
    if tune_scale <= 0 or beta_scale <= 0:
        raise ValueError("constraint scales must be positive")
    opt = nl.linear_optics(ring, create_graph=True)
    loss = torch.zeros((), dtype=ring.dtype, device=ring.device)
    for q, q0 in ((opt.Qx, target.Qx), (opt.Qy, target.Qy)):
        d = torch.remainder(q - q0 + 0.5, 1) - 0.5
        loss = loss + (d / tune_scale).square()
    for b, b0 in ((opt.beta_x, target.beta_x), (opt.beta_y, target.beta_y)):
        loss = loss + ((b / b0 - 1) / beta_scale).square()
    return loss


@dataclasses.dataclass
class SearchResult:
    theta: torch.Tensor
    loss: float
    evaluations: int
    method: str


def gradient_free(objective, theta: torch.Tensor, bounds, method: str = "Powell",
                  max_evaluations: int = 100, seed: int = 0) -> SearchResult:
    """Bounded Powell, Nelder-Mead or differential-evolution baseline."""
    import numpy as np
    from scipy.optimize import differential_evolution, minimize
    if method not in ("Powell", "Nelder-Mead", "differential_evolution", "CMA-ES"):
        raise ValueError("unknown gradient-free method")
    if theta.ndim != 1 or theta.numel() == 0:
        raise ValueError("theta must be a nonempty vector")
    limits = np.asarray(bounds, dtype=float)
    if limits.shape != (theta.numel(), 2) or (not np.isfinite(limits).all()) or np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("bounds must contain one increasing pair per knob")
    if max_evaluations < 1:
        raise ValueError("max_evaluations must be positive")
    best = [np.clip(theta.detach().cpu().numpy(), limits[:, 0], limits[:, 1]), float("inf")]
    calls = 0

    class BudgetExhausted(Exception):
        pass

    def evaluate(x):
        nonlocal calls
        if calls >= max_evaluations:
            raise BudgetExhausted
        calls += 1
        with torch.no_grad():
            value = float(objective(torch.as_tensor(x, dtype=theta.dtype, device=theta.device)))
        if np.isfinite(value) and value < best[1]:
            best[:] = [x.copy(), value]
        return value if np.isfinite(value) else float("inf")

    try:
        evaluate(best[0])
        if method == "CMA-ES":
            try:
                import cma
            except ImportError as exc:
                raise ImportError("CMA-ES requires the optional cma package") from exc
            widths = limits[:, 1] - limits[:, 0]
            strategy = cma.CMAEvolutionStrategy((best[0] - limits[:, 0]) / widths, 1 / 6, {
                "bounds": [0, 1],
                "seed": seed + 1, "verbose": -9, "maxfevals": max_evaluations})
            while not strategy.stop():
                population = strategy.ask()
                values = [evaluate(limits[:, 0] + widths * x) for x in population]
                strategy.tell(population, values)
        elif method == "differential_evolution":
            differential_evolution(evaluate, limits, seed=seed, polish=False,
                                   maxiter=max_evaluations, popsize=5, x0=best[0])
        else:
            minimize(evaluate, best[0], method=method, bounds=limits,
                     options={"maxfev": max_evaluations, "maxiter": max_evaluations})
    except BudgetExhausted:
        pass
    return SearchResult(torch.as_tensor(best[0], dtype=theta.dtype, device=theta.device),
                        best[1], calls, method)


def multi_fidelity(smooth_objective, aperture_objective, theta: torch.Tensor, bounds,
                   gradient_steps: int = 20, learning_rate: float = 1e-2,
                   max_evaluations: int = 100, seed: int = 0,
                   method: str = "differential_evolution") -> tuple[SearchResult, SearchResult]:
    """Gradient screening followed by bounded gradient-free DA refinement."""
    if gradient_steps < 1 or learning_rate <= 0:
        raise ValueError("positive gradient_steps and learning_rate required")
    if theta.ndim != 1 or theta.numel() == 0:
        raise ValueError("theta must be a nonempty vector")
    limits = torch.as_tensor(bounds, dtype=theta.dtype, device=theta.device)
    if limits.shape != (theta.numel(), 2) or not bool(torch.isfinite(limits).all()) or bool((limits[:, 0] >= limits[:, 1]).any()):
        raise ValueError("bounds must contain one increasing pair per knob")
    candidate = theta.detach().clone().clamp(limits[:, 0], limits[:, 1]).requires_grad_(True)
    optimizer = torch.optim.Adam([candidate], lr=learning_rate)
    best_theta, best_loss = candidate.detach().clone(), float("inf")
    for _ in range(gradient_steps):
        optimizer.zero_grad()
        loss = smooth_objective(candidate)
        value = float(loss.detach())
        if not math.isfinite(value):
            raise ValueError("non-finite surrogate; reduce launch amplitudes")
        if value < best_loss:
            best_theta, best_loss = candidate.detach().clone(), value
        loss.backward()
        if candidate.grad is None or not bool(torch.isfinite(candidate.grad).all()):
            raise ValueError("non-finite surrogate gradient; reduce launch amplitudes")
        optimizer.step()
        with torch.no_grad():
            candidate.clamp_(limits[:, 0], limits[:, 1])
    final_loss = float(smooth_objective(candidate).detach())
    if math.isfinite(final_loss) and final_loss < best_loss:
        best_theta, best_loss = candidate.detach().clone(), final_loss
    screening = SearchResult(best_theta, best_loss, gradient_steps + 1, "Adam")
    refinement = gradient_free(aperture_objective, best_theta, bounds, method,
                               max_evaluations, seed)
    return screening, refinement
