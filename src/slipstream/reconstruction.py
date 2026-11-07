"""Differentiable screen images and a scan forward model for gradient-based beam reconstruction (GPSR style).

See docs/modules/reconstruction.md. Nothing here is a fitting framework: it supplies the forward model (a tracker run at
each scan setting, then a screen image) and a CUDA-graph helper; the generator, loss and optimiser stay with the caller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Sequence

import torch
import torch.nn.functional as F

from .constants import ELECTRON_MASS_EV
from .tracking.native.collective import _capturing
from .tracking.native.linac import LinacTracker


def from_cheetah(particles: torch.Tensor, energy_eV: float, mc2_eV: float = ELECTRON_MASS_EV) -> torch.Tensor:
    """Cheetah ``(x, px, y, py, tau, p)`` columns (``p = (E - E0) / (p0 c)``) as Slipstream ``(x, px, y, py, tau, delta)`` with
    ``delta = (p - p0) / p0``, exactly. ``energy_eV`` is the total reference energy E0. Any seventh column is dropped."""
    p0c = math.sqrt(energy_eV**2 - mc2_eV**2)
    eps = particles[..., 5]
    e = energy_eV + eps * p0c
    pc = torch.sqrt((e - mc2_eV) * (e + mc2_eV))
    delta = eps * (e + energy_eV) / (pc + p0c)  # (pc - p0c) / p0c without cancellation
    return torch.cat([particles[..., :5], delta.unsqueeze(-1)], dim=-1)


def to_cheetah(particles6: torch.Tensor, energy_eV: float, mc2_eV: float = ELECTRON_MASS_EV) -> torch.Tensor:
    """Inverse of `from_cheetah`; appends the column of ones Cheetah keeps as its seventh coordinate."""
    p0c = math.sqrt(energy_eV**2 - mc2_eV**2)
    delta = particles6[..., 5]
    pc = p0c * (1.0 + delta)
    eps = p0c * delta * (2.0 + delta) / (torch.sqrt(pc * pc + mc2_eV**2) + energy_eV)  # (E - E0) / p0c
    return torch.cat([particles6[..., :5], eps.unsqueeze(-1), torch.ones_like(eps).unsqueeze(-1)], dim=-1)


@lru_cache(maxsize=None)
def _gaussian_kernel(sigma: float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    r = max(1, math.ceil(3.0 * sigma))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=dtype, device=device) / sigma) ** 2)
    return k / k.sum()


@dataclass
class Screen:
    """A calibrated screen: bilinear (cloud-in-cell) charge splatting onto a fixed pixel grid, then optional blur, gain and background.

    ``shape = (ny, nx)`` pixels; ``image[j, i]`` has its centre at ``centre + ((i - (nx - 1) / 2) * pixel_size[0],
    (j - (ny - 1) / 2) * pixel_size[1])`` in the two particle coordinates picked by ``axes`` (columns of the six-column state: ``(0, 2)`` is
    ``x, y``; pixel sizes and centre are in that coordinate's unit, so metres for ``x``, ``y`` and ``tau``). The grid is fixed, so the
    render has static shapes and no host reads and can sit inside a CUDA graph. A particle gives weight to its four nearest pixel centres
    and is differentiable in position; weight that falls outside the frame is lost (the image sums to the in-frame charge).
    ``psf_sigma_px`` blurs with a Gaussian point-spread function (in pixels, zero padded); ``gain`` and ``background`` are applied after
    and may be tensors, so calibration constants can be fitted."""

    shape: tuple[int, int]
    pixel_size: float | tuple[float, float]
    centre: tuple[float, float] = (0.0, 0.0)
    axes: tuple[int, int] = (0, 2)
    psf_sigma_px: float = 0.0
    gain: float | torch.Tensor = 1.0
    background: float | torch.Tensor = 0.0

    def render(self, particles6: torch.Tensor, charge_C: float | torch.Tensor = 1.0) -> torch.Tensor:
        """Image of one bunch, shape ``shape``; each of the ``n`` particles carries ``charge_C / n`` (the default makes the image the
        fraction of the bunch in each pixel)."""
        ny, nx = self.shape
        pu, pv = (self.pixel_size, self.pixel_size) if isinstance(self.pixel_size, (int, float)) else self.pixel_size
        u = (particles6[..., self.axes[0]] - self.centre[0]) / pu + (nx - 1) / 2
        v = (particles6[..., self.axes[1]] - self.centre[1]) / pv + (ny - 1) / 2
        i0, j0 = torch.floor(u), torch.floor(v)
        fu, fv = u - i0, v - j0
        w = charge_C / particles6.shape[-2]
        image = particles6.new_zeros(ny * nx)
        for di, wu in ((0, 1.0 - fu), (1, fu)):
            for dj, wv in ((0, 1.0 - fv), (1, fv)):
                i, j = i0 + di, j0 + dj
                inside = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
                idx = (j.clamp(0, ny - 1) * nx + i.clamp(0, nx - 1)).long()
                image = image.index_add(0, idx, torch.where(inside, wu * wv, torch.zeros_like(wu)) * w)
        image = image.view(ny, nx)
        if self.psf_sigma_px > 0.0:
            k = _gaussian_kernel(float(self.psf_sigma_px), image.dtype, image.device)
            r = k.numel() // 2
            image = F.conv2d(F.pad(image[None, None], (r, r, r, r)), k.view(1, 1, 1, -1))
            image = F.conv2d(image, k.view(1, 1, -1, 1))[0, 0]
        return image * self.gain + self.background


class ScanModel(torch.nn.Module):
    """A tracker run at each row of a scan table, observed on a screen at the end of the line (GPSR's ``GPSRLattice`` interface).

    ``variables`` names the scanned quantities as ``(element_name, setting)`` pairs, the settings being those of
    `LinacTracker.with_settings` (``k1``, ``k2``, ``kick_x``, ``kick_y``, ``voltage_V``, ``phase_rad``). `set_lattice_parameters` takes
    the scan table ``x`` of shape ``(n_settings, len(variables))`` and `track_and_observe` returns ``(images,)`` with ``images`` of shape
    ``(n_settings, ny, nx)``. Each setting tracks the whole bunch, so the collective forces see the full charge of ``tracker`` (its
    ``bunch_charge_C``); the table is read where it stands, so it keeps its autograd history and, in a CUDA graph, is read at replay
    time (capture with a static ``x``, then ``copy_`` each new table into it)."""

    def __init__(self, tracker: LinacTracker, screen: Screen, variables: Sequence[tuple[str, str]]):
        super().__init__()
        self.tracker, self.screen, self.variables = tracker, screen, [(str(n), str(a)) for n, a in variables]
        for name, setting in self.variables:
            tracker.with_settings(**{setting: {name: 0.0}})  # fails now, not mid-scan, on an unknown name or a wrong element kind
        self._x: torch.Tensor | None = None

    def set_lattice_parameters(self, x: torch.Tensor) -> None:
        x = torch.as_tensor(x)
        if x.ndim == 1:
            x = x[None]
        if x.ndim != 2 or x.shape[1] != len(self.variables):
            raise ValueError(f"scan table needs shape (n_settings, {len(self.variables)}), got {tuple(x.shape)}")
        self._x = x

    def track_and_observe(self, beam) -> tuple[torch.Tensor]:
        """``beam`` is an ``(n, 6)`` tensor in Slipstream coordinates, or a Cheetah-like beam (``.particles`` of shape ``(n, 7)`` in
        Cheetah coordinates, converted with `from_cheetah` at the tracker's reference energy; its charge is not used)."""
        if self._x is None:
            raise RuntimeError("call set_lattice_parameters first")
        t = self.tracker
        if isinstance(beam, torch.Tensor):
            particles6 = beam
        else:
            energy = getattr(beam, "energy", None)
            if energy is not None and not _capturing() and abs(float(energy) - t._reference_energy_eV) > 1e-6 * t._reference_energy_eV:
                raise ValueError(f"beam energy {float(energy):g} eV differs from the tracker's {t._reference_energy_eV:g} eV")
            particles6 = from_cheetah(beam.particles, t._reference_energy_eV, t.mc2_eV)
        if particles6.ndim != 2:
            raise ValueError(f"expected an (n, 6) beam, got shape {tuple(particles6.shape)} (batched beams are not supported)")
        images = []
        for row in self._x:
            settings: dict[str, dict] = {}
            for value, (name, setting) in zip(row, self.variables):
                settings.setdefault(setting, {})[name] = value
            out, _ = t.with_settings(**settings).track(particles6)
            images.append(self.screen.render(out))
        return (torch.stack(images),)


class GraphedStep:
    """``loss = fn(); loss.backward()`` captured once as a CUDA graph and replayed (eager when the parameters are on the CPU).

    ``fn()`` returns a scalar loss from tensors it holds (static inputs such as the scan table, filled in place with ``copy_`` before each
    call); ``params`` are the leaf tensors whose ``.grad`` is wanted. The first call warms up on a side stream, captures and replays; each
    call returns the loss tensor, which and whose ``params[i].grad`` are overwritten by the next call (``clone`` to keep them), so run the
    optimiser step outside. Everything in the step must be capturable: no host reads, and the `LinacTracker` limits in
    docs/tracking_native.md apply. Replays of the same graph must not overlap on different streams."""

    def __init__(self, fn: Callable[[], torch.Tensor], params: Sequence[torch.Tensor], warmup: int = 3):
        self.fn, self.params, self.warmup = fn, list(params), warmup
        self._graph: torch.cuda.CUDAGraph | None = None
        self._loss: torch.Tensor | None = None
        self._grads: list[torch.Tensor] = []
        self._eager = not (torch.cuda.is_available() and all(p.is_cuda for p in self.params))

    def _run(self) -> torch.Tensor:
        loss = self.fn()
        loss.backward()
        return loss

    def __call__(self) -> torch.Tensor:
        if self._eager:
            for p in self.params:
                p.grad = None
            return self._run()
        if self._graph is None:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(self.warmup):
                    for p in self.params:
                        p.grad = None
                    self._run()
            torch.cuda.current_stream().wait_stream(side)
            for p in self.params:
                p.grad = None
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._loss = self._run()
            self._graph, self._grads = graph, [p.grad for p in self.params]
        self._graph.replay()
        for p, g in zip(self.params, self._grads):
            p.grad = g  # the graph writes these buffers; restore them if the caller reset .grad in between
        return self._loss
