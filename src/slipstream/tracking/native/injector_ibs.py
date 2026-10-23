"""Local equal-weight binary Coulomb collisions; see docs/modules/tracking/native/injector_ibs.md."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ...constants import C_LIGHT, E_CHARGE_C, EPS0_F_PER_M
from . import warp_kernels


@dataclass
class InjectorIBSSettings:
    enabled: bool = False
    cell_size_m: tuple[float, float, float] | None = None
    model: str = "nanbu"
    coulomb_log: float | None = None
    max_impact_parameter_m: float | None = None
    seed: int = 0
    max_pair_beta: float = 0.1
    backend: str = "auto"

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("IBS enabled must be a boolean")
        if self.model not in ("nanbu", "takizuka-abe"):
            raise ValueError("IBS model must be nanbu or takizuka-abe")
        if self.backend not in ("auto", "torch", "warp"):
            raise ValueError("IBS backend must be auto, torch or warp")
        if self.cell_size_m is not None:
            if len(self.cell_size_m) != 3 or any(not math.isfinite(v) or v <= 0 for v in self.cell_size_m):
                raise ValueError("IBS cell_size_m must contain three positive finite lab-frame lengths")
            self.cell_size_m = tuple(self.cell_size_m)
        elif self.enabled:
            raise ValueError("Enabled injector IBS requires explicit cell_size_m")
        for label, value in (("coulomb_log", self.coulomb_log), ("max_impact_parameter_m", self.max_impact_parameter_m)):
            if value is not None and (not math.isfinite(value) or value < 0 or (label != "coulomb_log" and value == 0)):
                raise ValueError(f"IBS {label} is outside its finite physical range")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("IBS seed must be an integer in [0, 2**63)")
        if not math.isfinite(self.max_pair_beta) or not 0 < self.max_pair_beta <= 0.1:
            raise ValueError("IBS max_pair_beta must be positive and at most 0.1 (nonrelativistic collision rate)")


def _nanbu_parameter(variance: torch.Tensor) -> torch.Tensor:
    """Invert coth(A)-1/A=exp(-2 variance), with stable small/large limits."""
    tau = torch.where(variance > 0, variance, torch.full_like(variance, 0.01))
    result = 1 / (-torch.expm1(-2 * tau)).clamp_min(torch.finfo(tau.dtype).tiny)
    result = torch.where(tau > 20, 3 * torch.exp(-2 * tau), result)
    intermediate = (tau > 0.02) & (tau <= 20)
    tau = tau[intermediate]
    if not tau.numel():
        return result
    target = torch.exp(-2 * tau)
    low = torch.zeros_like(tau)
    high = 1 / (-torch.expm1(-2 * tau)).clamp_min(torch.finfo(tau.dtype).tiny)
    for _ in range(48):
        mid = (low + high) / 2
        safe = mid.clamp_min(0.1)
        small = mid.clamp_max(0.1)
        langevin = torch.where(mid < 0.1, small * (1/3 - small**2/45 + 2*small**4/945
                                                  - small**6/4725 + 2*small**8/93555),
                               1 / torch.tanh(safe) - 1 / safe)
        below = langevin < target
        low, high = torch.where(below, mid, low), torch.where(below, high, mid)
    result[intermediate] = (low + high) / 2
    return result


def _collision_randoms(shape, dtype, device, model, generator):
    azimuth = 2 * math.pi * torch.rand(shape, dtype=dtype, device=device, generator=generator)
    sample = torch.randn if model == "takizuka-abe" else torch.rand
    return azimuth, sample(shape, dtype=dtype, device=device, generator=generator)


def _scattering_angles(variance: torch.Tensor, model: str, generator: torch.Generator):
    """Return sin(theta), 1-cos(theta), azimuth without subtracting near-one cosines."""
    azimuth, random = _collision_randoms(variance.shape, variance.dtype, variance.device, model, generator)
    if model == "takizuka-abe":
        delta = random * variance.sqrt()
        return 2 * delta / (1 + delta**2), 2 * delta**2 / (1 + delta**2), azimuth
    parameter = _nanbu_parameter(variance)
    uniform = random
    small = parameter.clamp_max(1)
    large = parameter.clamp_min(1)
    small_loss = 2 - torch.log1p(uniform * torch.expm1(2 * small)) / small.clamp_min(torch.finfo(small.dtype).tiny)
    large_loss = -torch.log(uniform + (1 - uniform) * torch.exp(-2 * large)) / large
    isotropic_limit = 2 * (1 - uniform) - 2 * parameter * uniform * (1 - uniform)
    loss = torch.where(parameter < 1e-8, isotropic_limit,
                       torch.where(parameter < 1, small_loss, large_loss)).clamp(0, 2)
    loss = torch.where(variance > 0, loss, torch.zeros_like(loss))
    return (loss * (2 - loss)).sqrt(), loss, azimuth


def _pair_frame(ua: torch.Tensor, ub: torch.Tensor):
    ga = torch.sqrt(1 + (ua / C_LIGHT).square().sum(-1))
    gb = torch.sqrt(1 + (ub / C_LIGHT).square().sum(-1))
    beta = (ua + ub) / (C_LIGHT * (ga + gb)[:, None])
    gamma = torch.rsqrt(1 - beta.square().sum(-1))
    half_difference = (ua - ub) / 2
    # Equal-mass four-momenta give delta_gamma/2=dot(half_difference,beta)/c.
    rest = half_difference - (gamma / (gamma + 1) * (half_difference * beta).sum(-1))[:, None] * beta
    return rest, beta, gamma


def _pair_increment(rest, beta, gamma, sine, loss, azimuth):
    norm = torch.linalg.vector_norm(rest, dim=-1)
    unit = rest / torch.where(norm > 0, norm, torch.ones_like(norm))[:, None]
    axis = torch.nn.functional.one_hot(unit.abs().argmin(-1), 3).to(rest.dtype)
    tangent1 = torch.linalg.cross(unit, axis)
    tangent1 = tangent1 / torch.linalg.vector_norm(tangent1, dim=-1).clamp_min(torch.finfo(rest.dtype).tiny)[:, None]
    tangent2 = torch.linalg.cross(unit, tangent1)
    tangent = azimuth.cos()[:, None] * tangent1 + azimuth.sin()[:, None] * tangent2
    delta_rest = -loss[:, None] * rest + (sine * norm)[:, None] * tangent
    return delta_rest + (gamma.square() / (gamma + 1) * (delta_rest * beta).sum(-1))[:, None] * beta


class BinaryCollisionIBS:
    """Stateful local RNG and diagnostics for one injector run; equal physical macro weights only."""

    def __init__(self, settings: InjectorIBSSettings, charges_C: torch.Tensor, mass_kg: float,
                 particle_charge_C: float = -E_CHARGE_C):
        settings.__post_init__()
        if not settings.enabled:
            raise ValueError("Construct BinaryCollisionIBS only for enabled IBS")
        if charges_C.ndim != 1 or charges_C.dtype != torch.float64 or not bool(torch.isfinite(charges_C).all()):
            raise ValueError("Injector IBS requires finite float64 macro charges")
        if not math.isfinite(mass_kg) or mass_kg <= 0 or not math.isfinite(particle_charge_C) or particle_charge_C == 0:
            raise ValueError("IBS particle mass and charge must be finite and nonzero, with positive mass")
        charged = charges_C[charges_C != 0]
        if charged.numel() and (not bool(torch.allclose(charged, charged[0].expand_as(charged), rtol=1e-8, atol=0))):
            raise ValueError("Injector IBS supports equal signed macro charges; unequal weights or mixed species require another collision algorithm")
        self.settings, self.charges_C = settings, charges_C
        have_warp = warp_kernels.available(charges_C.device.type, charges_C.dtype)
        if settings.backend == "warp" and not have_warp:
            raise ValueError("IBS backend='warp' requires CUDA and warp-lang; use auto or torch for CPU fallback")
        self.backend = "warp" if have_warp and settings.backend != "torch" else "torch"
        self.mass_kg, self.charge_C = mass_kg, abs(particle_charge_C)
        self.weight = float(charged[0].abs()) / self.charge_C if charged.numel() else 0.0
        self.generator = torch.Generator(device=charges_C.device).manual_seed(settings.seed)
        self.cell_size = torch.tensor(settings.cell_size_m, dtype=charges_C.dtype, device=charges_C.device)
        self.calls = self.pairs = self.eligible_particles = self.unpaired_particles = 0
        self.max_variance = self.max_beta = 0.0

    def apply(self, positions_m: torch.Tensor, proper_velocity_m_s: torch.Tensor, dt_s: float,
              active: torch.Tensor | None = None) -> torch.Tensor:
        if (positions_m.shape != proper_velocity_m_s.shape or positions_m.shape != (len(self.charges_C), 3)
                or positions_m.dtype != torch.float64 or proper_velocity_m_s.dtype != torch.float64
                or positions_m.device != self.charges_C.device or proper_velocity_m_s.device != self.charges_C.device):
            raise ValueError("IBS requires matching (N,3) float64 positions and proper velocities on the charge device")
        if not math.isfinite(dt_s) or dt_s < 0:
            raise ValueError("IBS dt_s must be finite and nonnegative")
        if self.backend == "warp" and (positions_m.requires_grad or proper_velocity_m_s.requires_grad):
            raise NotImplementedError("Warp injector IBS is forward-only; collision gradients are unsupported")
        if active is not None and (active.shape != self.charges_C.shape or active.dtype != torch.bool or active.device != self.charges_C.device):
            raise ValueError("IBS active must be a matching boolean mask on the particle device")
        counted = self.charges_C != 0
        if active is not None:
            counted = counted & active
        indices = counted.nonzero().flatten()
        self.calls += 1
        if len(indices) < 2 or dt_s == 0 or self.settings.coulomb_log == 0:
            return proper_velocity_m_s
        positions, momenta = positions_m[indices], proper_velocity_m_s[indices]
        if not bool(torch.isfinite(positions).all() & torch.isfinite(momenta).all()):
            raise ValueError("Nonfinite active particle state in injector IBS")
        # Shift the grid each call to avoid repeatedly isolating particles at fixed cell faces.
        shift = torch.rand((3,), dtype=positions.dtype, device=positions.device, generator=self.generator)
        cell = torch.floor(positions / self.cell_size + shift).to(torch.int64)
        _, inverse, counts = torch.unique(cell, dim=0, return_inverse=True, return_counts=True)
        order = torch.randperm(len(indices), device=indices.device, generator=self.generator)
        order = order[torch.argsort(inverse[order], stable=True)]
        starts = counts.cumsum(0) - counts
        rank = torch.arange(len(indices), device=indices.device) - starts[inverse[order]]
        pair_starts = ((rank % 2 == 0) & (rank + 1 < counts[inverse[order]])).nonzero().flatten()
        self.eligible_particles += len(indices)
        self.unpaired_particles += len(indices) - 2 * len(pair_starts)
        if not len(pair_starts):
            return proper_velocity_m_s
        a, b = order[pair_starts], order[pair_starts + 1]
        pair_cells = inverse[a]
        gammas = torch.sqrt(1 + (momenta / C_LIGHT).square().sum(-1))
        # Sorted segment reductions avoid nondeterministic CUDA scatter atomics in the local frame.
        sum_u = torch.segment_reduce(momenta[order], "sum", lengths=counts)
        sum_g = torch.segment_reduce(gammas[order], "sum", lengths=counts)
        cell_beta = sum_u / (C_LIGHT * sum_g[:, None])
        cell_gamma = torch.rsqrt(1 - cell_beta.square().sum(-1))[pair_cells]
        if self.backend == "warp":
            from .injector_ibs_warp import collide
            azimuth, random = _collision_randoms(a.shape, momenta.dtype, momenta.device,
                                                self.settings.model, self.generator)
            bmax = min(self.settings.max_impact_parameter_m or max(self.settings.cell_size_m[:2]),
                       max(self.settings.cell_size_m[:2]))
            result, max_variance, max_beta = collide(
                proper_velocity_m_s, indices, a, b, counts, pair_cells, cell_gamma, azimuth, random,
                self.charge_C**4/(8*math.pi*EPS0_F_PER_M**2*(self.mass_kg/2)**2),
                self.weight/math.prod(self.settings.cell_size_m), dt_s,
                self.settings.coulomb_log if self.settings.coulomb_log is not None else -1.,
                math.log(bmax*(self.mass_kg/2)*4*math.pi*EPS0_F_PER_M/self.charge_C**2),
                self.settings.max_pair_beta, self.settings.model)
            if max_beta > self.settings.max_pair_beta:
                raise ValueError("IBS pair motion is outside the nonrelativistic collision-rate regime; reduce cell size or use a relativistic collision model")
            if not math.isfinite(max_variance) or not math.isfinite(max_beta):
                raise ValueError("Nonfinite IBS scattering strength; check density, relative velocity and time step")
            self.pairs += len(a)
            self.max_variance = max(self.max_variance, max_variance)
            self.max_beta = max(self.max_beta, max_beta)
            return result
        rest, boost_beta, boost_gamma = _pair_frame(momenta[a], momenta[b])
        rest_norm = torch.linalg.vector_norm(rest, dim=-1)
        rest_gamma = torch.sqrt(1 + (rest_norm / C_LIGHT).square())
        pair_beta = rest_norm / (C_LIGHT * rest_gamma)
        max_beta = float(pair_beta.max())
        if max_beta > self.settings.max_pair_beta:
            raise ValueError("IBS pair motion is outside the nonrelativistic collision-rate regime; reduce cell size or use a relativistic collision model")
        speed = 2 * rest_norm / rest_gamma
        safe_speed = torch.where(speed > 0, speed, torch.ones_like(speed))
        if self.settings.coulomb_log is None:
            bmax = self.settings.max_impact_parameter_m or max(self.settings.cell_size_m[:2])
            bmax = min(bmax, max(self.settings.cell_size_m[:2]))
            # SI b_min=q^2/(4*pi*eps0*mu*w^2); logs avoid overflow at cold pairs.
            log_lambda = (math.log(bmax * (self.mass_kg / 2) * 4 * math.pi * EPS0_F_PER_M / self.charge_C**2)
                          + 2 * safe_speed.log()).clamp_min(0)
        else:
            log_lambda = torch.full_like(speed, self.settings.coulomb_log)
        log_lambda = torch.where(speed > 0, log_lambda, torch.zeros_like(log_lambda))
        denominator = torch.where(log_lambda > 0, safe_speed, torch.ones_like(safe_speed)).pow(3)
        density_rest = counts[pair_cells].to(momenta.dtype) * (self.weight / math.prod(self.settings.cell_size_m)) / cell_gamma
        time_rest = dt_s / cell_gamma
        # An odd cell leaves one random particle idle: compensate its sampling probability.
        occupancy = counts[pair_cells].to(momenta.dtype)
        odd_correction = torch.where(occupancy % 2 == 1, occupancy / (occupancy - 1), 1.0)
        variance = (self.charge_C**4 / (8 * math.pi * EPS0_F_PER_M**2 * (self.mass_kg / 2)**2)
                    * density_rest * time_rest * log_lambda / denominator * odd_correction)
        if not bool(torch.isfinite(variance).all()):
            raise ValueError("Nonfinite IBS scattering strength; check density, relative velocity and time step")
        sine, loss, azimuth = _scattering_angles(variance, self.settings.model, self.generator)
        increment = _pair_increment(rest, boost_beta, boost_gamma, sine, loss, azimuth)
        result = proper_velocity_m_s.clone()
        result[indices[a]] += increment
        result[indices[b]] -= increment
        self.pairs += len(a)
        self.max_variance = max(self.max_variance, float(variance.max()))
        self.max_beta = max(self.max_beta, max_beta)
        return result

    def diagnostics(self):
        return dict(backend=self.backend, calls=self.calls, pairs=self.pairs, eligible_particle_calls=self.eligible_particles,
                    unpaired_particle_calls=self.unpaired_particles, max_variance=self.max_variance,
                    max_pair_beta=self.max_beta)
