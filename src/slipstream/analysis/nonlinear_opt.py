"""Gradient-based nonlinear-optics optimisation on top of ``analysis.nonlinear`` /
``tracking.native.ring.RingMap`` -- see plans/RING_NONLINEAR_DYNAMICS_PLAN.md (Phase 4).

- ``periodic_specs`` / ``unique_names``: give sextupole positions (identical arcs) or every
  element of a kind its own name, so ``RingMap.with_strengths`` can address them one by one.
- ``KnobSet``: a vector of K2 strengths ``theta`` mapped onto ``with_strengths``.
- ``chromaticity_response`` / ``null_basis``: the autograd Jacobian ``d xi / d theta`` and
  the directions that leave the chromaticity fixed ("orthogonal knobs").
- ``Launches`` / ``Footprint``: the smooth dynamic-aperture surrogate that gets minimised -- the
  tune shift with amplitude and with momentum over a small launch set, differentiable in every
  tensor strength, optionally over an ensemble of machines (``(M, 1)`` strengths).
- ``sample_field_errors``: per-element random K1/K2 errors for a machine ensemble.
"""
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
    """A small-amplitude reference (delta = 0), amplitude scans in x, y and on the diagonal at delta = 0, and
    small-amplitude launches at ``chrom_deltas`` (each on that delta's closed orbit from ``opt_by_delta``).
    ``diff_points`` = ``(x_mm, y_mm, delta)`` triples of *large-amplitude* launches (each ``delta`` must be a key of
    ``opt_by_delta``) whose tune diffusion between the two halves of the run is penalised -- the smooth stand-in for
    frequency-map chaos near the dynamic-aperture edge."""
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
    """``loss(ring)``: a smooth surrogate for dynamic aperture, differentiable in every tensor strength of ``ring``
    (NAFF and the tracking both are). Two parts over one small tracked launch set of ``n_turns`` turns:

    - *footprint*: mean squared tune shift ``[Q(x, y, delta) - Q(ref) - xi_target delta]`` over the ``amp`` and
      ``chrom`` launches, in units of 1e-3 -- the amplitude detuning plus the higher-order chromaticity the sextupole
      scheme leaves behind, at fixed linear chromaticity (a compact footprint crosses fewer resonances);
    - *diffusion* (``diff_weight > 0``, ``diff`` launches): ``mean log(1 + (D / diff_scale)^2)`` with ``D`` the tune
      change between the two halves of the run -- the frequency-map diffusion, which stays ~1e-6..1e-5 on regular
      motion and jumps toward 1e-3 as the particle nears chaos. It keeps the optimiser from buying a small footprint
      with large-amplitude stability. A ``diff`` launch lost inside ``n_turns`` counts as ``D = 0.1``; a lost
      ``amp``/``chrom`` launch contributes the constant ``lost_penalty`` (squared tune shift, units of 1e-6) to the footprint
      term -- neither has a gradient, so the optimiser is pushed back out of that region rather than crashing.

    ``n_machines=M`` tracks the launch set on ``M`` machines at once (``ring`` strengths ``(M, 1)``); ``loss`` is then
    the mean over machines plus ``std_weight`` times their standard deviation (a robust objective)."""

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
            # A launch that is lost inside the run blows up (inf/NaN) in the turn it goes, and although the loss mask
            # zeroes its state, the backward pass multiplies a zero gradient by the NaN local derivative and that NaN
            # is summed into every strength's gradient. So find the lost launches with a cheap no-gradient pass
            # (CUDA-graph path) and start them from a harmless launch in the tracked-with-gradient pass.
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
    """Per-element Gaussian field errors for ``n_machines`` machines, as ``(M, 1)``
    ``(k1 dict, k2 dict)`` ready for ``RingMap.with_strengths`` (specs must have per-element names,
    ``unique_names``). ``rules`` are ``(kind, parameter, rms)`` with ``kind`` an ``ElementSpec.kind``
    (``"quad"``, ``"bend"``), ``parameter`` ``"K1"`` or ``"K2"`` and ``rms`` *relative* to the element's own
    value (K1 for quads and bends; K2 of a bend). Gaussians are truncated at ``cutoff`` sigma."""
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
