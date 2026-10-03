"""Longitudinal collective effects (LSC, CSR) for ``LinacTracker``.

Mesh-free kernels: a Gaussian KDE of the line charge density ``lambda(tau)``
(and its derivative), evaluated at every particle's own position, shared by
both physics kernels (``_gaussian_kde_density_and_deriv``). Fixed-grid
alternatives (``lsc_delta_kick_fast``, ``csr_steady_state_delta_kick_fast``,
bottom of this module) trade accuracy for O(N) cost at large N, opt-in via
``LinacTracker(fast_collective_effects=True)``.

``tau_m`` is this package's native ``tau`` (meters, path-length-like,
``tau>0``=tail/late, ``tau<0``=head/early -- see ``elements.py``).

See docs/tracking_native.md and plans/NATIVE_LINAC_LSC_CSR_PLAN.md /
plans/DONE/FAST_COLLECTIVE_EFFECTS_PLAN.md for derivations, validation and
performance numbers.
"""
from __future__ import annotations

import math

import torch
import torch._dynamo

from ...constants import C_LIGHT, EPS0_F_PER_M
from . import warp_kernels

_Z0_OHM = 1.0 / (EPS0_F_PER_M * C_LIGHT)  # vacuum impedance


def silverman_bandwidth_m(tau_m: torch.Tensor) -> torch.Tensor:
    """Silverman's rule-of-thumb 1D Gaussian KDE bandwidth,
    ``h = 1.06 * sigma * N^(-1/5)``."""
    n = tau_m.shape[-1]
    return 1.06 * tau_m.std() * n ** (-0.2)


def _kde_window(n: int, k: float = 4.0) -> int:
    """Particle-count window for the truncated KDE: negligible contribution
    (``exp(-0.5*k^2)``) beyond ``+-k*h`` of a Gaussian bunch's peak density.
    A function of ``n`` only (static under ``torch.compile``), never a live
    tensor value. See docs/tracking_native.md for the derivation."""
    w = math.ceil(2.0 * 1.06 * k / math.sqrt(2.0 * math.pi) * n**0.8)
    return min(max(w, 32), n)


# Row-block cap for every chunked pairwise buffer below, in elements
# (dtype-agnostic). Bounds peak memory at large N without changing values
# (each row's result is independent of chunk grouping) -- see
# docs/tracking_native.md.
_CHUNK_TARGET_ELEMENTS = 2**21

# The Warp CSR kernel removes the materialised row-block, but has the same
# O(N^2) work.  Below this size its launch/setup cost is not consistently
# recovered.  Float64 CSR deliberately stays on the PyTorch path: on the
# supported consumer GPU its Warp kernel is slower, while the Warp wrapper
# still exposes float64 for explicit benchmarking and opt-in callers.
_CSR_WARP_MIN_N = 32_768


def _chunk_rows_for(n: int, cols: int) -> int:
    """Row-block size for a chunked ``(rows, cols)`` buffer, capped so
    ``rows * cols <= _CHUNK_TARGET_ELEMENTS``."""
    return max(1, min(n, _CHUNK_TARGET_ELEMENTS // max(cols, 1)))


def _kde_chunk_step(
    tau_sorted_chunk: torch.Tensor, window_chunk: torch.Tensor,
    charge_chunk: torch.Tensor | float, bandwidth_m: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One row-block's worth of the windowed-KDE math, extracted so it can
    be compiled once and reused across chunk iterations."""
    diff_chunk = tau_sorted_chunk.unsqueeze(-1) - window_chunk
    phi_chunk = torch.exp(-0.5 * (diff_chunk / bandwidth_m) ** 2) / (bandwidth_m * math.sqrt(2.0 * math.pi))
    density_chunk = (charge_chunk * phi_chunk).sum(dim=-1)
    # Reuse diff_chunk's storage for the derivative weight instead of a third buffer.
    diff_chunk.div_(bandwidth_m * bandwidth_m).neg_().mul_(phi_chunk).mul_(charge_chunk)
    density_deriv_chunk = diff_chunk.sum(dim=-1)
    return density_chunk, density_deriv_chunk


_compiled_kde_chunk_step = None


def _get_compiled_kde_chunk_step():
    """Lazily-built module-level compiled singleton."""
    global _compiled_kde_chunk_step
    if _compiled_kde_chunk_step is None:
        _compiled_kde_chunk_step = torch.compile(_kde_chunk_step)
    return _compiled_kde_chunk_step


def _csr_chunk_step(
    tau_chunk: torch.Tensor, tau_all: torch.Tensor, score: torch.Tensor, softening_m: torch.Tensor,
) -> torch.Tensor:
    """One row-block's worth of the CSR all-vs-all kernel sum."""
    diff_chunk = tau_chunk.unsqueeze(-1) - tau_all.unsqueeze(-2)  # (..., end-start, n)
    behind_chunk = diff_chunk < 0.0  # CSR propagates from later tau toward the head
    diff_chunk.pow_(2).add_(softening_m * softening_m).pow_(1.0 / 6.0).reciprocal_()  # softened |ds|^-1/3
    diff_chunk.mul_(score.unsqueeze(-2))  # score_j * kernel_ij
    diff_chunk.masked_fill_(~behind_chunk, 0.0)  # zero sources at or ahead of the observer
    return diff_chunk.sum(dim=-1)


_compiled_csr_chunk_step = None


def _get_compiled_csr_chunk_step():
    global _compiled_csr_chunk_step
    if _compiled_csr_chunk_step is None:
        _compiled_csr_chunk_step = torch.compile(_csr_chunk_step)
    return _compiled_csr_chunk_step


def _warp_collective_ready(primary: torch.Tensor, *others: torch.Tensor) -> bool:
    """Return whether a forward-only optional Warp collective is safe here."""
    if torch._dynamo.is_compiling() or not primary.is_cuda:
        return False
    if not warp_kernels.available(primary.device.type, primary.dtype):
        return False
    if primary.requires_grad or not primary.is_contiguous():
        return False
    for tensor in others:
        if tensor.device != primary.device or tensor.requires_grad or not tensor.is_contiguous():
            return False
    return True


def _csr_chunked_pairwise_sum(
    tau_m: torch.Tensor, score: torch.Tensor, softening_m: torch.Tensor, n: int, compile_chunks: bool = False,
    _use_graph: bool = True, _use_warp: bool = True,
) -> torch.Tensor:
    """CSR's all-vs-all sum (no compact support to window against), row-
    chunked the same way as the KDE's windowed branch. ``compile_chunks``
    (default off) routes each chunk through the compiled step instead of
    eager."""
    if (
        _use_warp
        and n >= _CSR_WARP_MIN_N
        and tau_m.ndim == 1
        and tau_m.dtype == torch.float32
        and score.shape == tau_m.shape
        and score.dtype == tau_m.dtype
        and softening_m.numel() == 1
        and softening_m.dtype == tau_m.dtype
        and softening_m.device == tau_m.device
        and _warp_collective_ready(tau_m, score, softening_m)
    ):
        return warp_kernels.csr_pairwise_sum(tau_m, score, softening_m)

    if _use_graph and compile_chunks and tau_m.is_cuda and tau_m.ndim == 1:
        graph = _get_csr_pairwise_graph(n, tau_m.dtype, tau_m.device)
        return graph.run(tau_m, score, softening_m)

    weighted_sum = torch.empty_like(tau_m)
    chunk = _chunk_rows_for(n, n)
    step = _get_compiled_csr_chunk_step() if compile_chunks else _csr_chunk_step
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        weighted_sum[..., start:end] = step(tau_m[..., start:end], tau_m, score, softening_m)
    return weighted_sum


class _CSRPairwiseGraph:
    """CUDA graph for the exact CSR chunk loop at one fixed particle count.

    The graph changes only the copied ``tau``, KDE score, and softening value;
    chunking and the compiled row-block body stay fixed. This preserves the
    pairwise model exactly while removing Python dispatch between chunks.
    """

    def __init__(self, n: int, dtype: torch.dtype, device: torch.device):
        self._tau = torch.zeros(n, dtype=dtype, device=device)
        self._score = torch.zeros(n, dtype=dtype, device=device)
        self._softening = torch.zeros((), dtype=dtype, device=device)

        def body():
            return _csr_chunked_pairwise_sum(
                self._tau, self._score, self._softening, n, compile_chunks=True,
                _use_graph=False, _use_warp=False,
            )

        warmup_stream = torch.cuda.Stream(device=device)
        warmup_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                body()
        torch.cuda.current_stream(device).wait_stream(warmup_stream)
        torch.cuda.synchronize(device)

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._weighted_sum = body()

    def run(
        self, tau_m: torch.Tensor, score: torch.Tensor, softening_m: torch.Tensor,
    ) -> torch.Tensor:
        self._tau.copy_(tau_m)
        self._score.copy_(score)
        self._softening.copy_(softening_m.reshape(()) if softening_m.numel() == 1 else softening_m)
        self._graph.replay()
        return self._weighted_sum.clone()


_csr_pairwise_graph_cache: dict[tuple, _CSRPairwiseGraph] = {}


def _get_csr_pairwise_graph(n: int, dtype: torch.dtype, device: torch.device) -> _CSRPairwiseGraph:
    key = (n, dtype, device)
    graph = _csr_pairwise_graph_cache.get(key)
    if graph is None:
        graph = _CSRPairwiseGraph(n, dtype, device)
        _csr_pairwise_graph_cache[key] = graph
    return graph


def _kde_windowed_scalar_charge(
    tau_m: torch.Tensor, charge_per_particle_C: torch.Tensor | float, bandwidth_m: torch.Tensor,
    n: int, w: int, compile_chunks: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Windowed-KDE math, scalar-charge case only (every real ``LinacTracker``
    caller). Isolated so it can be captured whole into a CUDA graph by
    ``_KDEWindowedGraph`` below."""
    order = torch.argsort(tau_m, dim=-1)
    tau_sorted = tau_m.gather(-1, order)
    half = w // 2
    right = w - half
    offset = bandwidth_m * 1.0e6
    left_pad = (tau_sorted[..., :1] - offset).expand(*tau_sorted.shape[:-1], half)
    right_pad = (tau_sorted[..., -1:] + offset).expand(*tau_sorted.shape[:-1], right)
    padded = torch.cat([left_pad, tau_sorted, right_pad], dim=-1)  # length n + w

    density_sorted = torch.empty_like(tau_sorted)
    density_deriv_sorted = torch.empty_like(tau_sorted)
    chunk = _chunk_rows_for(n, w)
    step = _get_compiled_kde_chunk_step() if compile_chunks else _kde_chunk_step
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        window_chunk = padded[..., start:end + w - 1].unfold(-1, w, 1)
        density_sorted[..., start:end], density_deriv_sorted[..., start:end] = step(
            tau_sorted[..., start:end], window_chunk, charge_per_particle_C, bandwidth_m,
        )

    density = torch.empty_like(tau_m)
    density.scatter_(-1, order, density_sorted)
    density_deriv = torch.empty_like(tau_m)
    density_deriv.scatter_(-1, order, density_deriv_sorted)
    return density, density_deriv


class _KDEWindowedGraph:
    """CUDA-graph capture of ``_kde_windowed_scalar_charge`` for one fixed
    ``(n, w, dtype, device, compile_chunks)`` -- collapses the ~15-20 small
    eager kernel launches per call into one ``graph.replay()``. Safe to
    reuse across elements/passes: the op sequence depends only on the fixed
    shape params, never on the live tensor values (those are copied into
    the graph's own input buffers fresh before every replay). See
    docs/tracking_native.md."""

    def __init__(self, n: int, w: int, dtype: torch.dtype, device: torch.device, compile_chunks: bool):
        self._tau = torch.zeros(n, dtype=dtype, device=device)
        self._charge = torch.zeros((), dtype=dtype, device=device)
        self._bandwidth = torch.zeros((), dtype=dtype, device=device)

        def body():
            return _kde_windowed_scalar_charge(self._tau, self._charge, self._bandwidth, n, w, compile_chunks)

        # Warm up on a side stream first (torch.cuda.graph's documented
        # requirement); also where a first torch.compile trace happens.
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                body()
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._density, self._density_deriv = body()

    def run(
        self, tau_m: torch.Tensor, charge_per_particle_C: torch.Tensor | float, bandwidth_m: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._tau.copy_(tau_m)
        self._charge.copy_(torch.as_tensor(charge_per_particle_C, dtype=self._tau.dtype, device=self._tau.device))
        self._bandwidth.copy_(bandwidth_m)
        self._graph.replay()
        # Cloned: the next call reuses these same buffer addresses on replay.
        return self._density.clone(), self._density_deriv.clone()


_kde_graph_cache: dict[tuple, "_KDEWindowedGraph"] = {}


def _get_kde_windowed_graph(n: int, w: int, dtype: torch.dtype, device: torch.device, compile_chunks: bool) -> "_KDEWindowedGraph":
    key = (n, w, dtype, device, compile_chunks)
    graph = _kde_graph_cache.get(key)
    if graph is None:
        graph = _KDEWindowedGraph(n, w, dtype, device, compile_chunks)
        _kde_graph_cache[key] = graph
    return graph


@torch._dynamo.disable()
def _gaussian_kde_density_and_deriv(
    tau_m: torch.Tensor, charge_per_particle_C: torch.Tensor | float, bandwidth_m: torch.Tensor,
    compile_chunks: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(lambda(tau_i), dlambda/dtau(tau_i))`` for every particle ``i``,
    from a Gaussian-kernel density estimate over *every* particle (including
    ``i``). Below ``_kde_window(n)`` particles: exact ``(N, N)`` broadcast.
    Above it: truncated to each particle's own window of nearest neighbors
    (O(N*w) instead of O(N^2)) -- row-chunked (``_chunk_rows_for``) to bound
    peak memory at large N. ``charge_per_particle_C`` may be scalar (the
    real-caller case) or a full per-source tensor (test-only, probes extra
    zero-charge points).

    ``@torch._dynamo.disable()``: this function's own Python loop over a
    static range is exactly what dynamo unrolls into the traced graph, and
    at large N (hundreds of chunks) that compile never finishes in
    reasonable time -- disabled here since every call site already reuses
    one compiled *outer* graph across many calls anyway. See
    docs/tracking_native.md."""
    n = tau_m.shape[-1]
    w = _kde_window(n)
    if w >= n:
        diff = tau_m.unsqueeze(-1) - tau_m.unsqueeze(-2)  # diff[..., i, j] = tau_i - tau_j
        phi = torch.exp(-0.5 * (diff / bandwidth_m) ** 2) / (bandwidth_m * math.sqrt(2.0 * math.pi))
        density = (charge_per_particle_C * phi).sum(dim=-1)
        density_deriv = (charge_per_particle_C * (-(diff / bandwidth_m**2) * phi)).sum(dim=-1)
        return density, density_deriv

    has_tensor_charge = isinstance(charge_per_particle_C, torch.Tensor) and charge_per_particle_C.dim() > 0
    if tau_m.is_cuda and not has_tensor_charge:
        # The real hot path: uniform, 0-d-tensor charge (linac.py's
        # _charge_per_particle_C) -- captured as one CUDA graph instead of
        # re-dispatching op-by-op from Python (bit-identical values).
        graph = _get_kde_windowed_graph(n, w, tau_m.dtype, tau_m.device, compile_chunks)
        return graph.run(tau_m, charge_per_particle_C, bandwidth_m)

    order = torch.argsort(tau_m, dim=-1)
    tau_sorted = tau_m.gather(-1, order)
    half = w // 2
    right = w - half
    # Sentinel padding many bandwidths outside the bunch: phi(diff/h)
    # underflows cleanly to 0, cheaper than masking the edge windows.
    offset = bandwidth_m * 1.0e6
    left_pad = (tau_sorted[..., :1] - offset).expand(*tau_sorted.shape[:-1], half)
    right_pad = (tau_sorted[..., -1:] + offset).expand(*tau_sorted.shape[:-1], right)
    padded = torch.cat([left_pad, tau_sorted, right_pad], dim=-1)  # length n + w

    if has_tensor_charge:
        charge_sorted = charge_per_particle_C.gather(-1, order)
        zero = torch.zeros_like(charge_sorted[..., :1])
        charge_padded = torch.cat([
            zero.expand(*charge_sorted.shape[:-1], half), charge_sorted, zero.expand(*charge_sorted.shape[:-1], right),
        ], dim=-1)

    density_sorted = torch.empty_like(tau_sorted)
    density_deriv_sorted = torch.empty_like(tau_sorted)
    chunk = _chunk_rows_for(n, w)
    step = _get_compiled_kde_chunk_step() if compile_chunks else _kde_chunk_step
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        window_chunk = padded[..., start:end + w - 1].unfold(-1, w, 1)
        charge_chunk = (
            charge_padded[..., start:end + w - 1].unfold(-1, w, 1) if has_tensor_charge else charge_per_particle_C
        )
        density_sorted[..., start:end], density_deriv_sorted[..., start:end] = step(
            tau_sorted[..., start:end], window_chunk, charge_chunk, bandwidth_m,
        )

    density = torch.empty_like(tau_m)
    density.scatter_(-1, order, density_sorted)
    density_deriv = torch.empty_like(tau_m)
    density_deriv.scatter_(-1, order, density_deriv_sorted)
    return density, density_deriv


@torch._dynamo.disable()
def _lsc_csr_delta_kicks(
    tau_m: torch.Tensor,
    charge_per_particle_C: torch.Tensor | float,
    x_m: torch.Tensor,
    y_m: torch.Tensor,
    energy_eV: torch.Tensor,
    mc2_eV: torch.Tensor,
    lsc_length_m: float,
    csr_rho_m: float,
    csr_length_m: float | torch.Tensor,
    pipe_radius_b_m: float | None,
    radius_factor: float,
    softening_m: float | torch.Tensor | None,
    compile_chunks: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exact LSC and CSR together, sharing one KDE evaluation."""
    n = tau_m.shape[-1]
    scalar_zero_charge = (
        (not isinstance(charge_per_particle_C, torch.Tensor) and charge_per_particle_C == 0.0)
        or (
            isinstance(charge_per_particle_C, torch.Tensor)
            and charge_per_particle_C.numel() == 1
            and charge_per_particle_C.item() == 0.0
        )
    )
    if n < 2 or scalar_zero_charge:
        zero = torch.zeros_like(tau_m)
        return zero, zero

    tiny = torch.finfo(tau_m.dtype).tiny ** 0.5
    h_m = torch.clamp(silverman_bandwidth_m(tau_m), min=tiny)
    density, density_deriv = _gaussian_kde_density_and_deriv(
        tau_m, charge_per_particle_C, h_m, compile_chunks=compile_chunks,
    )

    sigma_x, sigma_y = x_m.std(), y_m.std()
    a_m = torch.clamp(radius_factor * 0.5 * (sigma_x + sigma_y), min=tiny)
    if pipe_radius_b_m is None:
        form_factor = torch.ones((), dtype=tau_m.dtype, device=tau_m.device)
    else:
        b_over_a = torch.as_tensor(pipe_radius_b_m, dtype=a_m.dtype, device=a_m.device) / a_m
        form_factor = 1.0 + 2.0 * torch.log(b_over_a)
    e_field_per_length = (
        _Z0_OHM / (4.0 * math.pi * (energy_eV / mc2_eV) ** 2)
    ) * form_factor * C_LIGHT * density_deriv
    lsc_kick = -e_field_per_length * lsc_length_m / energy_eV

    if softening_m is None:
        softening = 0.5 * h_m
    else:
        softening = torch.as_tensor(softening_m, dtype=tau_m.dtype, device=tau_m.device)
    density_floor = 1e-6 * density.abs().max().clamp(min=1e-300)
    score = density_deriv / torch.clamp(density, min=density_floor)
    weighted_sum = _csr_chunked_pairwise_sum(
        tau_m, score, softening, n, compile_chunks=compile_chunks,
    )
    csr_constant = 2.0 / ((3.0 * csr_rho_m * csr_rho_m) ** (1.0 / 3.0) * 4.0 * math.pi * EPS0_F_PER_M)
    csr_length = torch.as_tensor(csr_length_m, dtype=tau_m.dtype, device=tau_m.device)
    csr_kick = csr_constant * csr_length * charge_per_particle_C * weighted_sum / energy_eV
    return lsc_kick, csr_kick


def lsc_delta_kick(
    tau_m: torch.Tensor,
    charge_per_particle_C: torch.Tensor | float,
    x_m: torch.Tensor,
    y_m: torch.Tensor,
    energy_eV: torch.Tensor,
    mc2_eV: torch.Tensor,
    length_m: float,
    pipe_radius_b_m: float | None = None,
    radius_factor: float = 1.7,
    compile_chunks: bool = False,
) -> torch.Tensor:
    """Local (long-wavelength) 1D LSC kick, ``delta`` only: ``E_z(tau) =
    Z0/(4*pi*gamma^2) * F * c * dlambda/dtau(tau)``, ``F = 1 + 2*ln(b/a)``
    (``a`` = live rms beam radius x ``radius_factor``, elegant's own
    ``RADIUS_FACTOR=1.7`` default; ``b`` = pipe radius, ``None`` -> free
    space, ``F=1``). Deliberately **not** real elegant's own nonlocal
    ``LSCDRIFT``/``LSCKICK`` formula (histogram+FFT, no pipe radius) -- see
    docs/tracking_native.md for the model choice, the frequency-domain
    cross-check and why a live-elegant comparison only agrees to order of
    magnitude, not tight rtol.

    Sign/normalization convention (``delta -= V(tau)/energy_eV``) matches
    ``ShortRangeWake.voltage_at``'s. ``compile_chunks`` (default off,
    forwarded to ``_gaussian_kde_density_and_deriv``) trades some speed for
    memory at large N -- see plans/DONE/NATIVE_LINAC_LSC_CSR_MEMORY_PLAN.md."""
    n = tau_m.shape[-1]
    if n < 2 or length_m <= 0.0:
        return torch.zeros_like(tau_m)
    # Clamped, not branched: torch.compile graph-breaks on a Python `if`
    # over a live tensor value.
    tiny = torch.finfo(tau_m.dtype).tiny ** 0.5
    sigma_x, sigma_y = x_m.std(), y_m.std()
    a_m = torch.clamp(radius_factor * 0.5 * (sigma_x + sigma_y), min=tiny)
    if pipe_radius_b_m is None:
        form_factor = torch.ones((), dtype=tau_m.dtype, device=tau_m.device)
    else:
        b_over_a = torch.as_tensor(pipe_radius_b_m, dtype=a_m.dtype, device=a_m.device) / a_m
        form_factor = 1.0 + 2.0 * torch.log(b_over_a)

    h_m = torch.clamp(silverman_bandwidth_m(tau_m), min=tiny)
    _, dlambda_dtau = _gaussian_kde_density_and_deriv(tau_m, charge_per_particle_C, h_m, compile_chunks=compile_chunks)

    e_field_per_length = (_Z0_OHM / (4.0 * math.pi * (energy_eV / mc2_eV) ** 2)) * form_factor * C_LIGHT * dlambda_dtau
    voltage_V = e_field_per_length * length_m
    return -voltage_V / energy_eV


@torch._dynamo.disable()
def csr_steady_state_delta_kick(
    tau_m: torch.Tensor,
    charge_per_particle_C: torch.Tensor | float,
    rho_m: float,
    energy_eV: torch.Tensor,
    length_m: float,
    softening_m: float | torch.Tensor | None = None,
    compile_chunks: bool = False,
) -> torch.Tensor:
    """Steady-state 1D CSR, using a KDE-score pairwise quadrature.

    In our tau=c*t convention. The positive prefactor multiplies d(lambda)/d(tau).
    Reversing both the direction and sign preserves mean loss for a
    symmetric bunch but gives the wrong pointwise wake.

    The kernel is Plummer softened, defaulting to half the Silverman KDE
    bandwidth. The finite nearest-neighbor KDE window can bias sharp
    profiles; this estimator is not an exact continuum reference.
    No entrance transient, shielding, or drift propagation is included here.
    """
    n = tau_m.shape[-1]
    scalar_zero_charge = (
        (not isinstance(charge_per_particle_C, torch.Tensor) and charge_per_particle_C == 0.0)
        or (
            isinstance(charge_per_particle_C, torch.Tensor)
            and charge_per_particle_C.numel() == 1
            and charge_per_particle_C.item() == 0.0
        )
    )
    if n < 2 or length_m <= 0.0 or rho_m == 0.0 or scalar_zero_charge:
        return torch.zeros_like(tau_m)
    tiny = torch.finfo(tau_m.dtype).tiny ** 0.5
    h_m = torch.clamp(silverman_bandwidth_m(tau_m), min=tiny)
    if softening_m is None:
        softening_m = 0.5 * h_m
    else:
        softening_m = torch.as_tensor(softening_m, dtype=tau_m.dtype, device=tau_m.device)

    density, density_deriv = _gaussian_kde_density_and_deriv(tau_m, charge_per_particle_C, h_m, compile_chunks=compile_chunks)
    # Score function (log-density derivative), floored relative to the KDE's
    # own peak density to avoid blowing up in the sparse far tails.
    density_floor = 1e-6 * density.abs().max().clamp(min=1e-300)
    score = density_deriv / torch.clamp(density, min=density_floor)

    weighted_sum = _csr_chunked_pairwise_sum(tau_m, score, softening_m, n, compile_chunks=compile_chunks)

    csr_constant = 2.0 / ((3.0 * rho_m * rho_m) ** (1.0 / 3.0) * 4.0 * math.pi * EPS0_F_PER_M)
    return csr_constant * length_m * charge_per_particle_C * weighted_sum / energy_eV


@torch._dynamo.disable()
def _csr_block_multipole_sum(
    tau_m: torch.Tensor,
    score: torch.Tensor,
    softening_m: torch.Tensor,
    tolerance: float = 1.0e-3,
) -> torch.Tensor:
    """Candidate subquadratic CSR sum using sorted block monopoles/dipoles.

    Sources in the observer's block and a small number of following blocks
    are evaluated exactly. More distant blocks use their score moments in a
    first-order Taylor expansion of the softened kernel. This is intentionally
    an opt-in candidate: the block-neighborhood heuristic still needs the
    section-5 parity matrix before it can replace the exact reference.
    """
    n = tau_m.shape[-1]
    if n < 256 or tau_m.ndim != 1:
        return _csr_chunked_pairwise_sum(tau_m, score, softening_m, n, compile_chunks=tau_m.is_cuda)

    tolerance = max(1.0e-6, min(float(tolerance), 0.25))
    leaf = 1 << max(5, min(8, int(math.log2(max(32, int(math.sqrt(n)))))))
    near_blocks = max(1, min(8, math.ceil(tolerance ** -0.25)))
    n_blocks = math.ceil(n / leaf)
    padded_n = n_blocks * leaf

    order = torch.argsort(tau_m)
    tau_sorted = tau_m[order]
    score_sorted = score[order]
    if padded_n != n:
        pad = padded_n - n
        tau_blocks = torch.cat([tau_sorted, torch.zeros(pad, dtype=tau_m.dtype, device=tau_m.device)]).reshape(n_blocks, leaf)
        score_blocks = torch.cat([score_sorted, torch.zeros(pad, dtype=score.dtype, device=score.device)]).reshape(n_blocks, leaf)
        valid_blocks = torch.arange(padded_n, device=tau_m.device).reshape(n_blocks, leaf) < n
    else:
        tau_blocks = tau_sorted.reshape(n_blocks, leaf)
        score_blocks = score_sorted.reshape(n_blocks, leaf)
        valid_blocks = torch.ones_like(tau_blocks, dtype=torch.bool)

    first = tau_blocks[:, 0]
    last_index = torch.minimum(
        torch.arange(n_blocks, device=tau_m.device) * leaf + leaf - 1,
        torch.full((n_blocks,), n - 1, device=tau_m.device, dtype=torch.long),
    )
    last = tau_sorted[last_index]
    centers = 0.5 * (first + last)
    offsets = tau_blocks - centers.unsqueeze(-1)
    m0 = (score_blocks * valid_blocks).sum(dim=-1)
    m1 = (score_blocks * offsets * valid_blocks).sum(dim=-1)

    block_ids = torch.arange(n_blocks, device=tau_m.device)
    weighted_sum_sorted = torch.empty_like(tau_sorted)
    exact_span = (near_blocks + 1) * leaf
    chunk = _chunk_rows_for(n, exact_span)
    eps2 = softening_m * softening_m
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        observer_index = torch.arange(start, end, device=tau_m.device)
        observer_block = observer_index // leaf
        observer = tau_sorted[start:end]

        source_index = observer_index.unsqueeze(-1) + torch.arange(1, exact_span + 1, device=tau_m.device)
        source_valid = (source_index < n) & ((source_index // leaf) <= (observer_block.unsqueeze(-1) + near_blocks))
        source_index = source_index.clamp(max=n - 1)
        source_tau = tau_sorted[source_index]
        source_score = score_sorted[source_index]
        diff = observer.unsqueeze(-1) - source_tau
        source_kernel = (diff.square() + eps2).pow(-1.0 / 6.0)
        exact = (source_score * source_kernel * source_valid).sum(dim=-1)

        block_diff = observer.unsqueeze(-1) - centers.unsqueeze(0)
        block_r2 = block_diff.square() + eps2
        kernel = block_r2.pow(-1.0 / 6.0)
        kernel_deriv = -(block_diff / 3.0) * block_r2.pow(-7.0 / 6.0)
        far = block_ids.unsqueeze(0) > observer_block.unsqueeze(-1) + near_blocks
        # ponytail: monopole+dipole far field; adaptive higher moments are the
        # upgrade if the measured tolerance is insufficient for a profile.
        approx = m0.unsqueeze(0) * kernel - m1.unsqueeze(0) * kernel_deriv
        weighted_sum_sorted[start:end] = exact + (approx * far).sum(dim=-1)

    weighted_sum = torch.empty_like(tau_m)
    weighted_sum.scatter_(0, order, weighted_sum_sorted)
    return weighted_sum


@torch._dynamo.disable()
def csr_steady_state_delta_kick_subquadratic(
    tau_m: torch.Tensor,
    charge_per_particle_C: torch.Tensor | float,
    rho_m: float,
    energy_eV: torch.Tensor,
    length_m: float,
    softening_m: float | torch.Tensor | None = None,
    tolerance: float = 1.0e-3,
) -> torch.Tensor:
    """Opt-in block-multipole CSR candidate; exact CSR remains authoritative."""
    n = tau_m.shape[-1]
    scalar_zero_charge = (
        (not isinstance(charge_per_particle_C, torch.Tensor) and charge_per_particle_C == 0.0)
        or (
            isinstance(charge_per_particle_C, torch.Tensor)
            and charge_per_particle_C.numel() == 1
            and charge_per_particle_C.item() == 0.0
        )
    )
    if n < 2 or length_m <= 0.0 or rho_m == 0.0 or scalar_zero_charge:
        return torch.zeros_like(tau_m)
    tiny = torch.finfo(tau_m.dtype).tiny ** 0.5
    h_m = torch.clamp(silverman_bandwidth_m(tau_m), min=tiny)
    if softening_m is None:
        softening_m = 0.5 * h_m
    else:
        softening_m = torch.as_tensor(softening_m, dtype=tau_m.dtype, device=tau_m.device).reshape(())
    density, density_deriv = _gaussian_kde_density_and_deriv(
        tau_m, charge_per_particle_C, h_m, compile_chunks=tau_m.is_cuda,
    )
    density_floor = 1e-6 * density.abs().max().clamp(min=1e-300)
    score = density_deriv / torch.clamp(density, min=density_floor)
    weighted_sum = _csr_block_multipole_sum(tau_m, score, softening_m, tolerance=tolerance)
    csr_constant = 2.0 / ((3.0 * rho_m * rho_m) ** (1.0 / 3.0) * 4.0 * math.pi * EPS0_F_PER_M)
    return csr_constant * length_m * charge_per_particle_C * weighted_sum / energy_eV


# ---------------------------------------------------------------------------
# Fast, fixed-bin-grid LSC/CSR -- see plans/DONE/FAST_COLLECTIVE_EFFECTS_PLAN.md.
# Drop-in alternatives to lsc_delta_kick/csr_steady_state_delta_kick above,
# trading KDE/pairwise accuracy for O(N) cost via a fixed n_bins grid --
# gated behind LinacTracker's fast_collective_effects flag, never default.
# ---------------------------------------------------------------------------


def _fixed_grid(
    tau_m: torch.Tensor, n_bins: int, margin: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Fixed ``n_bins``-bin CIC grid over this call's live ``tau_m`` window,
    padded by ``margin``. Returns ``(bin_idx, interp_w, valid, tau_min,
    dtau, n_bins)``: ``bin_idx`` is the left of each particle's two
    bracketing bin centers (clamped into ``[0, n_bins-2]``); ``interp_w``
    is the linear weight on the right bin. ``valid`` is ``False`` for
    particles outside the window -- deposit/gather zero these out rather
    than clamping onto the edge bin (see docs/tracking_native.md, "trash-
    bin" semantics). ``tau_min``/``dtau`` are 0-d tensors so this stays
    ``torch.compile``-safe across calls with different live windows."""
    tau_min_raw = tau_m.min()
    tau_max_raw = tau_m.max()
    span = torch.clamp(tau_max_raw - tau_min_raw, min=torch.finfo(tau_m.dtype).tiny ** 0.5)
    pad = margin * span
    tau_min = tau_min_raw - pad
    dtau = (span + 2.0 * pad) / n_bins
    u = (tau_m - tau_min) / dtau - 0.5  # bin-center coordinate: u=0 at bin 0's own center
    valid = (u >= 0.0) & (u < n_bins - 1.0)
    bin_idx = torch.floor(u).clamp(min=0, max=n_bins - 2).long()
    interp_w = torch.where(valid, u - torch.floor(u), torch.zeros_like(u))
    return bin_idx, interp_w, valid, tau_min, dtau, n_bins


def _bin_line_density(
    charge_per_particle_C: torch.Tensor | float, bin_idx: torch.Tensor, interp_w: torch.Tensor,
    valid: torch.Tensor, n_bins: int, dtau: torch.Tensor, _use_warp: bool = True,
) -> torch.Tensor:
    """CIC histogram of the line charge density ``lambda_k`` on the grid
    built by ``_fixed_grid``."""
    dtype, device = interp_w.dtype, interp_w.device

    charge_tensor = charge_per_particle_C
    if not isinstance(charge_tensor, torch.Tensor):
        charge_tensor = torch.as_tensor(charge_tensor, dtype=dtype, device=device)
    if (
        _use_warp
        and bin_idx.device == device
        and bin_idx.dtype in (torch.int32, torch.int64)
        and charge_tensor.numel() == 1
        and charge_tensor.dtype == dtype
        and _warp_collective_ready(interp_w, valid, charge_tensor, dtau)
        and valid.dtype == torch.bool
    ):
        # _fixed_grid currently returns int64 indices for native torch
        # scatter_add_.  Warp's compact index type is sufficient here and
        # avoids changing the established CPU/reference representation.
        bin_idx32 = bin_idx if bin_idx.dtype == torch.int32 else bin_idx.to(torch.int32)
        return warp_kernels.cic_line_density(
            bin_idx32.contiguous(), interp_w, valid, charge_tensor, n_bins, dtau,
        )

    zero = torch.zeros((), dtype=dtype, device=device)
    left_w = torch.where(valid, 1.0 - interp_w, zero)
    lam = torch.zeros(n_bins, dtype=dtype, device=device)
    lam.scatter_add_(0, bin_idx, charge_per_particle_C * left_w)
    lam.scatter_add_(0, bin_idx + 1, charge_per_particle_C * interp_w)
    return lam / dtau


def _smooth_bins(lam: torch.Tensor, passes: int) -> torch.Tensor:
    """Binomial ``(1,2,1)/4`` smoothing, ``passes`` times (same convention
    as ``space_charge.smooth_rho``) -- suppresses the CIC histogram's shot
    noise before it gets amplified by ``1/dtau`` in the derivative step.
    ``passes`` is a plain Python int (static under ``torch.compile``). See
    docs/tracking_native.md for why the default is in the hundreds, not one."""
    for _ in range(passes):
        pad = torch.zeros_like(lam[:1])
        padded = torch.cat([pad, lam, pad])
        lam = 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
    return lam


def _grid_deriv(lam: torch.Tensor, dtau: torch.Tensor) -> torch.Tensor:
    """``d(lam)/d(tau)`` on the fixed grid: central differences at interior
    bins, one-sided at the two edges."""
    d_left = (lam[1:2] - lam[0:1]) / dtau
    d_mid = (lam[2:] - lam[:-2]) / (2.0 * dtau)
    d_right = (lam[-1:] - lam[-2:-1]) / dtau
    return torch.cat([d_left, d_mid, d_right])


def _gather_linear(field: torch.Tensor, bin_idx: torch.Tensor, interp_w: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Linear interpolation of a fixed-grid ``field`` back at each
    particle's position, using ``_fixed_grid``'s own ``bin_idx``/``interp_w``
    -- zero for an out-of-window particle."""
    gathered = field[bin_idx] * (1.0 - interp_w) + field[bin_idx + 1] * interp_w
    return torch.where(valid, gathered, torch.zeros_like(gathered))


class _LSCFastGraph:
    """CUDA-graph capture of the fixed-bin-grid LSC kernel's tensor math,
    for one fixed ``(n, n_bins, smooth_passes, dtype, device,
    pipe_radius_b_m, radius_factor)`` -- collapses ``smooth_passes``
    (default 512) sequential dependent tiny kernel launches into one
    ``graph.replay()``, the same trick ``_KDEWindowedGraph`` above already
    uses for the reference kernel. **Never keyed on anything
    element/lattice-specific** (``length_m``/``energy_eV`` are applied
    after ``run()`` returns, in plain eager Python -- see
    ``lsc_delta_kick_fast``) -- so one graph is reused across every
    LSC-active element of every lattice sharing this key: a true
    compile-once block. See plans/DONE/COLLECTIVE_COMPILE_ONCE_PLAN.md for why
    this was needed: inlining this loop into ``LinacTracker``'s per-element
    ``torch.compile`` graph (the previous design) forced it to be rebuilt
    for every one of a real lattice's ~150+ distinct element signatures
    (length/angle/k1/k2 -- an unavoidable, pre-existing per-element
    specialization for the *transport* step), turning an already-expected
    per-element recompile into a CPU-dispatch-bound one."""

    def __init__(
        self, n: int, n_bins: int, smooth_passes: int, dtype: torch.dtype, device: torch.device,
        pipe_radius_b_m: float | None, radius_factor: float,
    ):
        self._tau = torch.zeros(n, dtype=dtype, device=device)
        self._charge = torch.zeros((), dtype=dtype, device=device)
        self._x = torch.zeros(n, dtype=dtype, device=device)
        self._y = torch.zeros(n, dtype=dtype, device=device)
        self._energy = torch.zeros((), dtype=dtype, device=device)
        self._mc2 = torch.ones((), dtype=dtype, device=device)

        def body() -> torch.Tensor:
            tiny = torch.finfo(dtype).tiny ** 0.5
            sigma_x, sigma_y = self._x.std(), self._y.std()
            a_m = torch.clamp(radius_factor * 0.5 * (sigma_x + sigma_y), min=tiny)
            if pipe_radius_b_m is None:
                form_factor = torch.ones((), dtype=dtype, device=device)
            else:
                b_over_a = torch.as_tensor(pipe_radius_b_m, dtype=dtype, device=device) / a_m
                form_factor = 1.0 + 2.0 * torch.log(b_over_a)
            bin_idx, interp_w, valid, _, dtau, nb = _fixed_grid(self._tau, n_bins)
            lam = _bin_line_density(self._charge, bin_idx, interp_w, valid, nb, dtau)
            lam = _smooth_bins(lam, smooth_passes)
            dlambda_dtau_grid = _grid_deriv(lam, dtau)
            dlambda_dtau = _gather_linear(dlambda_dtau_grid, bin_idx, interp_w, valid)
            return (_Z0_OHM / (4.0 * math.pi * (self._energy / self._mc2) ** 2)) * form_factor * C_LIGHT * dlambda_dtau

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                body()
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._e_field_per_length = body()

    def run(
        self, tau_m: torch.Tensor, charge_per_particle_C: torch.Tensor | float,
        x_m: torch.Tensor, y_m: torch.Tensor, energy_eV: torch.Tensor, mc2_eV: torch.Tensor,
    ) -> torch.Tensor:
        self._tau.copy_(tau_m)
        self._charge.copy_(torch.as_tensor(charge_per_particle_C, dtype=self._tau.dtype, device=self._tau.device))
        self._x.copy_(x_m)
        self._y.copy_(y_m)
        self._energy.copy_(energy_eV)
        self._mc2.copy_(mc2_eV)
        self._graph.replay()
        return self._e_field_per_length.clone()


_lsc_fast_graph_cache: dict[tuple, "_LSCFastGraph"] = {}


def _get_lsc_fast_graph(
    n: int, n_bins: int, smooth_passes: int, dtype: torch.dtype, device: torch.device,
    pipe_radius_b_m: float | None, radius_factor: float,
) -> "_LSCFastGraph":
    key = (n, n_bins, smooth_passes, dtype, device, pipe_radius_b_m, radius_factor)
    graph = _lsc_fast_graph_cache.get(key)
    if graph is None:
        graph = _LSCFastGraph(n, n_bins, smooth_passes, dtype, device, pipe_radius_b_m, radius_factor)
        _lsc_fast_graph_cache[key] = graph
    return graph


@torch._dynamo.disable()
def lsc_delta_kick_fast(
    tau_m: torch.Tensor,
    charge_per_particle_C: torch.Tensor | float,
    x_m: torch.Tensor,
    y_m: torch.Tensor,
    energy_eV: torch.Tensor,
    mc2_eV: torch.Tensor,
    length_m: float,
    pipe_radius_b_m: float | None = None,
    radius_factor: float = 1.7,
    n_bins: int = 2048,
    smooth_passes: int = 512,
) -> torch.Tensor:
    """Fixed-bin-grid approximation of ``lsc_delta_kick`` -- same formula,
    conventions and citations; only ``dlambda/dtau``'s estimator differs
    (CIC histogram + binomial smoothing + central differences in place of
    the reference's Gaussian KDE, O(N) instead of O(N*w)). See
    docs/tracking_native.md for the fidelity-loss discussion and measured
    agreement against the reference. ``n_bins``/``smooth_passes`` are the
    grid-only knobs this approximation adds -- see docs/tracking_native.md
    for why ``smooth_passes`` needs to be in the hundreds, not one, to
    match the reference kernel's own implicit smoothing.

    ``@torch._dynamo.disable()`` (like the reference kernels): the
    ``smooth_passes``-deep sequential dependency chain is exactly what
    ``LinacTracker``'s per-element ``torch.compile`` graph would otherwise
    have to re-trace/re-cache-lookup for every distinct element signature
    in a real lattice -- see ``_LSCFastGraph``'s own docstring. On CUDA
    with a scalar charge (every real ``LinacTracker`` caller), this whole
    tensor computation is instead captured once into a CUDA graph, keyed
    only by ``(n, n_bins, smooth_passes, dtype, device, pipe_radius_b_m,
    radius_factor)`` -- never by ``length_m`` or anything else that varies
    per element -- and replayed for every call sharing that key.
    ``length_m`` is applied after replay, in plain eager Python (one
    multiply -- not worth graph-capturing)."""
    n = tau_m.shape[-1]
    if n < 2 or length_m <= 0.0:
        return torch.zeros_like(tau_m)
    has_tensor_charge = isinstance(charge_per_particle_C, torch.Tensor) and charge_per_particle_C.dim() > 0
    if tau_m.is_cuda and not has_tensor_charge:
        graph = _get_lsc_fast_graph(n, n_bins, smooth_passes, tau_m.dtype, tau_m.device, pipe_radius_b_m, radius_factor)
        e_field_per_length = graph.run(tau_m, charge_per_particle_C, x_m, y_m, energy_eV, mc2_eV)
    else:
        tiny = torch.finfo(tau_m.dtype).tiny ** 0.5
        sigma_x, sigma_y = x_m.std(), y_m.std()
        a_m = torch.clamp(radius_factor * 0.5 * (sigma_x + sigma_y), min=tiny)
        if pipe_radius_b_m is None:
            form_factor = torch.ones((), dtype=tau_m.dtype, device=tau_m.device)
        else:
            b_over_a = torch.as_tensor(pipe_radius_b_m, dtype=a_m.dtype, device=a_m.device) / a_m
            form_factor = 1.0 + 2.0 * torch.log(b_over_a)

        bin_idx, interp_w, valid, _, dtau, n_bins = _fixed_grid(tau_m, n_bins)
        lam = _bin_line_density(charge_per_particle_C, bin_idx, interp_w, valid, n_bins, dtau)
        lam = _smooth_bins(lam, smooth_passes)
        dlambda_dtau_grid = _grid_deriv(lam, dtau)
        dlambda_dtau = _gather_linear(dlambda_dtau_grid, bin_idx, interp_w, valid)
        e_field_per_length = (_Z0_OHM / (4.0 * math.pi * (energy_eV / mc2_eV) ** 2)) * form_factor * C_LIGHT * dlambda_dtau

    voltage_V = e_field_per_length * length_m
    return -voltage_V / energy_eV


def _csr_fast_field(
    tau_m: torch.Tensor, charge_per_particle_C: torch.Tensor | float,
    n_bins: int, smooth_passes: int | None, sg_halfwidth: int,
    softening_m: float | torch.Tensor | None,
) -> torch.Tensor:
    """Histogram -> smoothed derivative -> zero-padded forward CSR convolution.

    Integrates a linearly interpolated
    derivative analytically over the singular first interval. Its weights
    are 0.9 at the observer, 1.1 at the next bin, then j**(-1/3), all
    multiplied by dtau**(2/3). This includes the near-cell contribution
    omitted by a strictly off-diagonal Riemann sum. Zero padding prevents
    a periodic wake from wrapping across the bunch.
    """
    idx, w, valid, _, dtau, nb = _fixed_grid(tau_m, n_bins, margin=0.1)
    lam = _bin_line_density(charge_per_particle_C, idx, w, valid, nb, dtau)
    if smooth_passes is not None:
        lam = _smooth_bins(lam, smooth_passes)
    elif sg_halfwidth:
        # Degree-one symmetric Savitzky-Golay smoothing is a moving average.
        width = 2 * sg_halfwidth + 1
        weights = torch.ones((1, 1, width), dtype=lam.dtype, device=lam.device) / width
        lam = torch.nn.functional.conv1d(lam[None, None], weights, padding=sg_halfwidth)[0, 0]
    derivative = _grid_deriv(lam, dtau)
    j = torch.arange(nb, dtype=tau_m.dtype, device=tau_m.device)
    weights = torch.where(j == 0, 0.9, torch.where(j == 1, 1.1, j.clamp(min=1).pow(-1 / 3)))
    kernel = weights * dtau.pow(2 / 3)
    if softening_m is not None:
        eps = torch.as_tensor(softening_m, dtype=tau_m.dtype, device=tau_m.device)
        softened = ((j * dtau).square() + eps.square()).clamp(min=torch.finfo(tau_m.dtype).tiny).pow(-1 / 6)
        softened = softened * dtau * torch.where(j == 0, 0.5, 1.0)
        kernel = torch.where(eps > 0, softened, kernel)
    fft_size = 1 << (2 * nb - 1).bit_length()
    # Reverse before/after convolution: sources have tau_source>tau_observer.
    spectrum = torch.fft.rfft(derivative.flip(0), n=fft_size) * torch.fft.rfft(kernel, n=fft_size)
    field_grid = torch.fft.irfft(spectrum, n=fft_size)[:nb].flip(0)
    return _gather_linear(field_grid, idx, w, valid)


class _CSRFastGraph:
    """Capture the same FFT field calculation used on CPU; reuse for all bends."""

    def __init__(self, n, n_bins, smooth_passes, dtype, device, softening_m, sg_halfwidth):
        self._tau = torch.linspace(-1, 1, n, dtype=dtype, device=device)
        self._charge = torch.zeros((), dtype=dtype, device=device)

        def body():
            return _csr_fast_field(self._tau, self._charge, n_bins, smooth_passes, sg_halfwidth, softening_m)

        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                body()
        torch.cuda.current_stream(device).wait_stream(stream)
        torch.cuda.synchronize(device)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._field = body()

    def run(self, tau_m, charge_per_particle_C):
        self._tau.copy_(tau_m)
        self._charge.copy_(torch.as_tensor(charge_per_particle_C, dtype=self._tau.dtype, device=self._tau.device))
        self._graph.replay()
        return self._field.clone()


_csr_fast_graph_cache: dict[tuple, _CSRFastGraph] = {}


@torch._dynamo.disable()
def csr_steady_state_delta_kick_fast(
    tau_m: torch.Tensor,
    charge_per_particle_C: torch.Tensor | float,
    rho_m: float,
    energy_eV: torch.Tensor,
    length_m: float,
    softening_m: float | torch.Tensor | None = None,
    n_bins: int = 256,
    smooth_passes: int | None = None,
    sg_halfwidth: int = 1,
) -> torch.Tensor:
    """Elegant-style steady-state CSR in O(N + B log B) time and O(N+B) memory.

    Uses CIC deposition, degree-one Savitzky-Golay histogram smoothing,
    a central derivative, and near-cell-corrected trapezoidal
    integration. The default is unsoftened; the integrable singularity is
    handled analytically. An explicit positive softening retains Plummer
    regularization. Explicit smooth_passes selects binomial smoothing;
    otherwise sg_halfwidth applies. No adaptive host-side tier selection.

    LinacTracker supplies the source dipole's BINS/SG_HALFWIDTH, including
    for its approximate drift tail. This implements the steady-state wake,
    not transient or Stupakov the csrdrift model.
    """
    n = tau_m.shape[-1]
    if n < 2 or length_m <= 0.0 or rho_m == 0.0:
        return torch.zeros_like(tau_m)
    if n_bins < 3 or sg_halfwidth < 0 or 2 * sg_halfwidth + 1 > n_bins:
        raise ValueError("CSR needs at least 3 bins and a smoothing window no wider than the grid")
    if smooth_passes is not None and smooth_passes < 0:
        raise ValueError("smooth_passes must be nonnegative")
    vector_charge = isinstance(charge_per_particle_C, torch.Tensor) and charge_per_particle_C.dim() > 0
    hashable_softening = softening_m is None or isinstance(softening_m, (int, float))
    if tau_m.is_cuda and not vector_charge and hashable_softening:
        key = (n, n_bins, smooth_passes, tau_m.dtype, tau_m.device, softening_m, sg_halfwidth)
        graph = _csr_fast_graph_cache.get(key)
        if graph is None:
            with torch.cuda.device(tau_m.device):
                graph = _CSRFastGraph(*key)
            _csr_fast_graph_cache[key] = graph
        field = graph.run(tau_m, charge_per_particle_C)
    else:
        field = _csr_fast_field(tau_m, charge_per_particle_C, n_bins, smooth_passes, sg_halfwidth, softening_m)
    csr_constant = 2.0 / ((3.0 * rho_m * rho_m) ** (1.0 / 3.0) * 4.0 * math.pi * EPS0_F_PER_M)
    return csr_constant * length_m * field / energy_eV
