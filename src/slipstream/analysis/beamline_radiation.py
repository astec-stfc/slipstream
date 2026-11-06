"""Differentiable linear beamline covariance transport with dipole ISR.

Coordinates are slopes (x, x', y, y', delta), not canonical particle states.
Reference energy is constant; classical damping and nonlinear transport are
not included. See docs/modules/analysis/beamline_radiation.md.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ..constants import ELECTRON_MASS_EV
from ..tracking.native.radiation import bend_radiation_rates


@dataclass(frozen=True)
class LinearRadiationResult:
    input_covariance: torch.Tensor
    covariance: torch.Tensor
    covariance_without_isr: torch.Tensor
    diffusion_covariance: torch.Tensor
    transport: torch.Tensor
    s_m: torch.Tensor
    transport_profile: torch.Tensor
    diffusion_profile: torch.Tensor
    offset_m: torch.Tensor
    longitudinal_m: torch.Tensor
    exit_angle_rad: torch.Tensor
    i2_m_inv: torch.Tensor
    i3_m_inv2: torch.Tensor
    loss_parameter: torch.Tensor


def linear_radiation_transport(
    length_m: torch.Tensor, angle_rad: torch.Tensor, k1_m2: torch.Tensor,
    covariance: torch.Tensor, *, energy_eV: float,
    e1_rad: torch.Tensor | None = None, e2_rad: torch.Tensor | None = None,
    samples_per_element: int = 2,
) -> LinearRadiationResult:
    """Propagate expected ISR second moments through aligned hard-edge elements.

    Lengths, angles, quadrupole strengths and faces retain their AD graphs.
    All tensors must be CPU float64. Elements have positive length; bends
    must be pure dipoles. Faces use first-order zero-gap edge focusing.
    Sampling changes the optics profile, not the exact linear noise integral.
    """
    e1 = torch.zeros_like(angle_rad) if e1_rad is None else e1_rad
    e2 = torch.zeros_like(angle_rad) if e2_rad is None else e2_rad
    inputs = (length_m, angle_rad, k1_m2, covariance, e1, e2)
    if any(t.device.type != "cpu" or t.dtype != torch.float64 for t in inputs):
        raise ValueError("Linear radiation transport requires CPU float64 tensors")
    if (length_m.ndim != 1 or not len(length_m) or covariance.shape != (5, 5)
            or any(t.shape != length_m.shape for t in (angle_rad, k1_m2, e1, e2))):
        raise ValueError("Expected nonempty equal-length element vectors and a (5,5) covariance")
    if samples_per_element < 1 or not isinstance(samples_per_element, int):
        raise ValueError("samples_per_element must be a positive integer")
    if not math.isfinite(energy_eV) or energy_eV/ELECTRON_MASS_EV < 1000:
        raise ValueError("Ultrarelativistic electron total energy is required")
    if any(not bool(torch.isfinite(t).all()) for t in inputs) or bool((length_m <= 0).any()):
        raise ValueError("Finite inputs and positive lengths are required")
    if bool(((angle_rad != 0) & (k1_m2 != 0)).any()):
        raise NotImplementedError("Radiating bends must be pure dipoles")
    if not torch.allclose(covariance, covariance.T, atol=1e-24, rtol=1e-12):
        raise ValueError("Input covariance must be symmetric")
    h = angle_rad/length_m
    loss_rate, variance_rate = bend_radiation_rates(energy_eV, h)
    z, one = torch.zeros_like(h), torch.ones_like(h)
    a = torch.stack((z, one, z, -(k1_m2+h*h), z, h, z, z, z), -1).reshape(-1, 3, 3)
    q = torch.diag_embed(torch.stack((z, z, variance_rate), -1))
    block = torch.cat((torch.cat((a, q), -1), torch.cat((torch.zeros_like(a), -a.transpose(-1, -2)), -1)), -2)
    ds = length_m/samples_per_element
    exponential = torch.matrix_exp(block*ds[:, None, None])
    tx = exponential[:, :3, :3]
    qx = exponential[:, :3, 3:] @ tx.transpose(-1, -2)
    qx = (qx+qx.transpose(-1, -2))/2
    ay = torch.stack((z, one, k1_m2, z), -1).reshape(-1, 2, 2)
    ty = torch.matrix_exp(ay*ds[:, None, None])
    n = len(length_m)
    body = torch.zeros((n, 5, 5), dtype=length_m.dtype, device=length_m.device)
    noise = torch.zeros_like(body)
    ix, iy = torch.tensor([0, 1, 4]), torch.tensor([2, 3])
    body[:, ix[:, None], ix[None, :]] = tx
    body[:, iy[:, None], iy[None, :]] = ty
    noise[:, ix[:, None], ix[None, :]] = qx
    def faces(angles):
        result = torch.eye(5, dtype=length_m.dtype).repeat(n, 1, 1)
        focus = h*torch.tan(angles)
        result[:, 1, 0], result[:, 3, 2] = focus, -focus
        return result
    entrance, exit_ = faces(e1), faces(e2)
    body = body[:, None].expand(-1, samples_per_element, -1, -1)
    noise = noise[:, None].expand_as(body)
    identity = torch.eye(5, dtype=length_m.dtype).repeat(n, samples_per_element, 1, 1)
    entry_steps, exit_steps = identity.clone(), identity.clone()
    entry_steps[:, 0], exit_steps[:, -1] = entrance, exit_
    maps = (exit_steps @ body @ entry_steps).reshape(-1, 5, 5)
    noises = (exit_steps @ noise @ exit_steps.transpose(-1, -2)).reshape(-1, 5, 5)
    # Associative prefix composition keeps the complete AD Jacobian batched.
    stride = 1
    while stride < len(maps):
        later = maps[stride:]
        noises = torch.cat((noises[:stride], noises[stride:]+later@noises[:-stride]@later.transpose(-1, -2)))
        maps = torch.cat((maps[:stride], later@maps[:-stride]))
        stride *= 2
    transfer, diffusion = maps[-1], noises[-1]
    off = transfer @ covariance @ transfer.T
    diffusion = (diffusion+diffusion.T)/2
    heading = torch.cumsum(angle_rad, 0)-angle_rad/2
    chord = length_m*torch.sinc(angle_rad/(2*math.pi))
    return LinearRadiationResult(
        input_covariance=covariance, covariance=off+diffusion, covariance_without_isr=off,
        diffusion_covariance=diffusion, transport=transfer,
        s_m=torch.cat((length_m.new_zeros(1), torch.cumsum(ds.repeat_interleave(samples_per_element), 0))),
        transport_profile=torch.cat((torch.eye(5, dtype=length_m.dtype)[None], maps)),
        diffusion_profile=torch.cat((torch.zeros_like(transfer)[None], noises)),
        offset_m=(chord*torch.sin(heading)).sum(),
        longitudinal_m=(chord*torch.cos(heading)).sum(), exit_angle_rad=angle_rad.sum(),
        i2_m_inv=(h*h*length_m).sum(), i3_m_inv2=(h.abs()**3*length_m).sum(),
        loss_parameter=(loss_rate*length_m).sum(),
    )
