"""Differentiable modified Bessel functions I_n, K_n of a complex argument
(IMPEDANCE_PLAN.md Phase 0) -- the primitive the round/flat multilayer
field-matching formalism (Phases 1-3) is built on.

``torch.special.modified_bessel_i0/i1/k0/k1`` exist natively but only for
real dtypes (checked directly: raises ``NotImplementedError`` on a complex
tensor) -- the multilayer formalism needs complex arguments (the transverse
wavenumber inside a resistive/lossy layer is complex). ``scipy.special.iv``/
``kv`` handle complex arguments and arbitrary real order natively but aren't
autograd-aware, so this wraps them in a custom ``torch.autograd.Function``:
forward delegates to scipy (a CPU round-trip -- fine here, these arrays are
frequency-grid-sized, not particle-count-sized, so this isn't the
many-small-ops-per-turn trap the tracker hit, see NOTES.md), backward uses
the exact (not approximated) Bessel recurrence relations, DLMF 10.29.1-2:

    dI_n/dz = I_{n-1}(z) - (n/z) I_n(z)
    dK_n/dz = -K_{n-1}(z) - (n/z) K_n(z)

(valid for n=0 too, using the standard I_{-1}=I_1, K_{-1}=K_1 identities,
which scipy's ``iv``/``kv`` already honor for negative order.)

Both I_n and K_n are holomorphic in z, so PyTorch's complex-autograd
convention applies directly: ``grad_z = grad_output * conj(dI_n/dz)`` --
verified (not assumed) against finite differences via
``torch.autograd.gradcheck`` in ``tests/test_multilayer_bessel.py``.
"""
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
    """I_{n-1}(z)/I_n(z) -- see module docstring's "numerical conditioning"
    section: I_n(z) itself overflows for the |z| this repo's realistic
    layer/frequency parameters produce (checked: |z| ~ 1e5 at 1mm-1cm
    radius, 1e-5 Ohm.m resistivity, THz frequencies -- see
    IMPEDANCE_PLAN.md Phase 0), but this ratio stays O(1) for all such z
    (I_{n-1} and I_n share the same leading exponential growth rate) and is
    exactly what the ported multilayer formalism (round.cc's
    ``Imppratio``) actually needs -- never materializing the raw,
    overflowing value in the first place, rather than overflowing and
    trying to recover.

    Forward: ``scipy.special.ive(n-1,z)/scipy.special.ive(n,z)`` -- the
    exponential scale factor is identical for both (same z), so it cancels
    exactly, giving the true ratio via values that individually never
    overflow. Verified against mpmath (50 decimal digits) to machine
    precision (~1e-16) up to |z|~1.25e5 -- double precision is sufficient
    for this combination, multiprecision is not needed (unlike IW2D's own
    C++, which uses 160-bit arithmetic unconditionally for every
    combination, not just this one).

    Backward: closed form purely in terms of the ratio itself (never the
    raw, potentially-overflowing I values), derived from the standard
    recurrence dI_n/dz = I_{n-1} - (n/z)I_n and its mirror
    dI_{n-1}/dz = I_n + ((n-1)/z)I_{n-1} (DLMF 10.29.1-2) via the quotient
    rule:

        dR/dz = 1 - R^2 + R*(2n-1)/z,  R = I_{n-1}(z)/I_n(z)

    verified against finite differences directly (not just trusted from
    the algebra) before use.
    """

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
    """I_{n-1}(z)/I_n(z), z complex, differentiable and numerically stable
    for arbitrarily large |z| (unlike ``modified_bessel_i(n-1,z) /
    modified_bessel_i(n,z)``, which overflows to nan/inf well within this
    repo's realistic parameter range -- see class docstring)."""
    return _BesselRatioI.apply(z, n)


class _BesselRatioK(torch.autograd.Function):
    """K_{n-1}(z)/K_n(z) -- the K-function analog of ``_BesselRatioI``,
    same rationale (see that class's docstring). Backward derived the same
    way (quotient rule + DLMF 10.29.3-4's one-sided recurrences for K) and
    verified against finite differences before use:

        dR/dz = R^2 - 1 + R*(2n-1)/z,  R = K_{n-1}(z)/K_n(z)

    (note the sign flip on the "R^2-1" term relative to the I-ratio's
    "1-R^2" -- expected, K decays where I grows)."""

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
    """I_n(z)*exp(-z) -- an O(1)-magnitude, *holomorphic* rescaling of
    I_n(z), valid for Re(z)>=0 (the only regime round.py ever calls this
    with: z is always a layer's transverse wavenumber times a positive
    radius, and the wavenumber's branch is chosen with Re>=0 -- see
    round.py's ``_csqrt``). Needed alongside ``bessel_i_ratio``/
    ``bessel_k_ratio`` because the multilayer round-chamber matrix
    (round.py's ``_layer_matching_matrix``) also needs raw I_m(x)*K_m(y)-
    type *products* at two different arguments, not just same-argument
    ratios -- and those products overflow in raw double precision well
    within this repo's realistic parameters (found validating Phase 1
    against live IW2D: xp1p~9000 at the vacuum-to-first-conductor
    boundary, alone enough to overflow raw I_0).

    Scaling by the *real-only* exp(-Re(z)) (matching scipy's ``ive``
    convention directly) would be simpler but is not holomorphic (it
    depends on both z and conj(z)), needing the full non-holomorphic
    Wirtinger backward. Scaling by the full complex exp(-z) instead has
    the identical magnitude benefit (|exp(-z)|=exp(-Re(z))) while staying
    holomorphic, so the same simple ``grad_output * conj(df/dz)``
    convention already used throughout this module applies unchanged.

    Forward: ``ive(n,z)*exp(-1j*z.imag)`` -- equals I_n(z)*exp(-z) exactly
    when Re(z)>=0 (scipy's ive uses exp(-|Re(z)|), which is exp(-Re(z))
    exactly in that regime). Backward, derived from the standard I
    recurrence and verified against finite differences before use:

        d/dz [I_n(z)e^{-z}] = I_{n-1}(z)e^{-z} - (1 + n/z) I_n(z)e^{-z}
    """

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
    """K_n(z)*exp(z) -- the K analog of ``_ModifiedBesselIHat`` (see its
    docstring); this one equals scipy's ``kve(n,z)`` directly (scipy's own
    K-scaling convention already uses the full complex exp(z), unlike
    ``ive``'s real-only exp(-|Re(z)|) -- checked directly from scipy's
    docstrings), so no phase correction is needed.

        d/dz [K_n(z)e^{z}] = -K_{n-1}(z)e^{z} + (1 - n/z) K_n(z)e^{z}

    verified against finite differences before use.
    """

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
