"""Complex modified Bessel functions use SciPy forward evaluation and exact recurrence derivatives for
PyTorch complex autograd. See docs/modules/wakefields/multilayer/bessel.md."""
from __future__ import annotations

import numpy as np
import scipy.special
import torch


def _to_np(z: torch.Tensor) -> np.ndarray:
    return z.detach().cpu().numpy()


def _from_np(val: np.ndarray, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(val, dtype=like.dtype, device=like.device)


class _ModifiedBesselI(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z: torch.Tensor, n: int) -> torch.Tensor:
        out = _from_np(scipy.special.iv(n, _to_np(z)), z)
        ctx.save_for_backward(z, out)
        ctx.n = n
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        z, I_n = ctx.saved_tensors
        n = ctx.n
        I_nm1 = _from_np(scipy.special.iv(n - 1, _to_np(z)), z)
        dI_dz = I_nm1 - (n / z) * I_n
        return grad_output * dI_dz.conj(), None


class _ModifiedBesselK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z: torch.Tensor, n: int) -> torch.Tensor:
        out = _from_np(scipy.special.kv(n, _to_np(z)), z)
        ctx.save_for_backward(z, out)
        ctx.n = n
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        z, K_n = ctx.saved_tensors
        n = ctx.n
        K_nm1 = _from_np(scipy.special.kv(n - 1, _to_np(z)), z)
        dK_dz = -K_nm1 - (n / z) * K_n
        return grad_output * dK_dz.conj(), None


def modified_bessel_i(n: int, z: torch.Tensor) -> torch.Tensor:
    """I_n(z), z complex, differentiable w.r.t. z."""
    return _ModifiedBesselI.apply(z, n)


def modified_bessel_k(n: int, z: torch.Tensor) -> torch.Tensor:
    """K_n(z), z complex, differentiable w.r.t. z."""
    return _ModifiedBesselK.apply(z, n)


class _BesselRatioI(torch.autograd.Function):
    """Compute BesselRatioI; see the module documentation for details."""

    @staticmethod
    def forward(ctx, z: torch.Tensor, n: int) -> torch.Tensor:
        z_np = _to_np(z)
        R = _from_np(scipy.special.ive(n - 1, z_np) / scipy.special.ive(n, z_np), z)
        ctx.save_for_backward(z, R)
        ctx.n = n
        return R

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        z, R = ctx.saved_tensors
        n = ctx.n
        dR_dz = 1 - R**2 + R * (2 * n - 1) / z
        return grad_output * dR_dz.conj(), None


def bessel_i_ratio(n: int, z: torch.Tensor) -> torch.Tensor:
    """Compute bessel i ratio; see the module documentation for details."""
    return _BesselRatioI.apply(z, n)


class _BesselRatioK(torch.autograd.Function):
    """K_{n-1}(z)/K_n(z) -- the K-function analog of ``_BesselRatioI``, same rationale (see that
    class's docstring)."""

    @staticmethod
    def forward(ctx, z: torch.Tensor, n: int) -> torch.Tensor:
        z_np = _to_np(z)
        R = _from_np(scipy.special.kve(n - 1, z_np) / scipy.special.kve(n, z_np), z)
        ctx.save_for_backward(z, R)
        ctx.n = n
        return R

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        z, R = ctx.saved_tensors
        n = ctx.n
        dR_dz = R**2 - 1 + R * (2 * n - 1) / z
        return grad_output * dR_dz.conj(), None


def bessel_k_ratio(n: int, z: torch.Tensor) -> torch.Tensor:
    """K_{n-1}(z)/K_n(z), z complex, differentiable and numerically stable
    for arbitrarily large |z| (see ``bessel_i_ratio``)."""
    return _BesselRatioK.apply(z, n)


class _ModifiedBesselIHat(torch.autograd.Function):
    """Compute ModifiedBesselIHat; see the module documentation for details."""

    @staticmethod
    def forward(ctx, z: torch.Tensor, n: int) -> torch.Tensor:
        z_np = _to_np(z)
        out = _from_np(scipy.special.ive(n, z_np) * np.exp(-1j * z_np.imag), z)
        ctx.save_for_backward(z, out)
        ctx.n = n
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        z, Ihat_n = ctx.saved_tensors
        n = ctx.n
        Ihat_nm1 = _ModifiedBesselIHat.apply(z, n - 1)
        dIhat_dz = Ihat_nm1 - (1 + n / z) * Ihat_n
        return grad_output * dIhat_dz.conj(), None


class _ModifiedBesselKHat(torch.autograd.Function):
    """Compute ModifiedBesselKHat; see the module documentation for details."""

    @staticmethod
    def forward(ctx, z: torch.Tensor, n: int) -> torch.Tensor:
        z_np = _to_np(z)
        out = _from_np(scipy.special.kve(n, z_np), z)
        ctx.save_for_backward(z, out)
        ctx.n = n
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        z, Khat_n = ctx.saved_tensors
        n = ctx.n
        Khat_nm1 = _ModifiedBesselKHat.apply(z, n - 1)
        dKhat_dz = -Khat_nm1 + (1 - n / z) * Khat_n
        return grad_output * dKhat_dz.conj(), None


def modified_bessel_i_hat(n: int, z: torch.Tensor) -> torch.Tensor:
    """I_n(z)*exp(-z), z complex with Re(z)>=0, differentiable and
    numerically stable for arbitrarily large |z| (see
    ``_ModifiedBesselIHat``)."""
    return _ModifiedBesselIHat.apply(z, n)


def modified_bessel_k_hat(n: int, z: torch.Tensor) -> torch.Tensor:
    """K_n(z)*exp(z), z complex with Re(z)>=0, differentiable and
    numerically stable for arbitrarily large |z| (see
    ``_ModifiedBesselKHat``)."""
    return _ModifiedBesselKHat.apply(z, n)
