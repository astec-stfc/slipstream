"""Benchmark optional Warp replacements for the remaining collective hot spots.

Run from the repository root in a CUDA-enabled environment, for example::

    python benchmark/warp_collective.py --sizes 1024,4096,16384

The script deliberately excludes first-use compilation from the reported
timings.  Each case validates the Warp result against the current PyTorch
reference before measuring synchronized CUDA-stream execution.  Exact CSR is
still O(N^2); the Warp version removes the materialized row-block temporary,
but does not change the physics or asymptotic algorithm.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
# Keep benchmark compilation artifacts with the checkout.  This also makes
# the harness usable on Windows installations whose default NVIDIA cache is
# present but not readable by the active Python process.
if os.name == "nt":
    os.environ["LOCALAPPDATA"] = str(ROOT / ".cache" / "localappdata")
os.environ.setdefault("WARP_CACHE_PATH", str(ROOT / ".cache" / "warp"))

from slipstream.tracking.native import collective, warp_kernels  # noqa: E402


@dataclass
class Result:
    case: str
    n: int
    dtype: str
    bins: int | None
    torch_ms: float
    warp_ms: float
    speedup: float
    max_abs: float
    max_rel: float
    l2_rel: float


def _parse_sizes(value: str) -> list[int]:
    sizes = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not sizes or any(n < 1 for n in sizes):
        raise argparse.ArgumentTypeError("sizes must be positive comma-separated integers")
    return sizes


def _dtype(value: str) -> torch.dtype:
    if value == "float32":
        return torch.float32
    if value == "float64":
        return torch.float64
    raise argparse.ArgumentTypeError("dtype must be float32 or float64")


def _time_cuda(fn, warmup: int, iterations: int, device: torch.device) -> float:
    """Return average milliseconds after warming the exact callable."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _error(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[float, float, float]:
    diff = (candidate - reference).abs()
    scale = reference.abs().clamp_min(torch.finfo(reference.dtype).tiny)
    l2_rel = torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(reference).clamp_min(torch.finfo(reference.dtype).tiny)
    return float(diff.max().item()), float((diff / scale).max().item()), float(l2_rel.item())


def _assert_close(
    case: str, reference: torch.Tensor, candidate: torch.Tensor,
    rtol: float | None = None, atol: float | None = None,
) -> tuple[float, float, float]:
    max_abs, max_rel, l2_rel = _error(reference, candidate)
    if rtol is None or atol is None:
        if reference.dtype == torch.float32:
            rtol, atol = 3e-4, 3e-6
        else:
            rtol, atol = 2e-12, 2e-14
    try:
        torch.testing.assert_close(candidate, reference, rtol=rtol, atol=atol, msg=case)
    except AssertionError as exc:
        raise AssertionError(f"{case}: max_abs={max_abs:.3e}, max_rel={max_rel:.3e}; {exc}") from exc
    return max_abs, max_rel, l2_rel


def _random_tau(n: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(20261003 + n)
    # A smooth bunch-like distribution keeps the benchmark representative of
    # the collective path while retaining unsorted particle ordering.
    return torch.randn(n, dtype=dtype, device=device, generator=generator) * 2.0e-3


def _bench_csr(n: int, dtype: torch.dtype, device: torch.device, warmup: int, iterations: int) -> Result:
    tau = _random_tau(n, dtype, device).contiguous()
    score = (torch.randn_like(tau) * 0.5).contiguous()
    softening = torch.tensor(2.5e-5, dtype=dtype, device=device)

    def torch_reference():
        return collective._csr_chunked_pairwise_sum(
            tau, score, softening, n, compile_chunks=True, _use_warp=False,
        )

    def warp_candidate():
        return warp_kernels.csr_pairwise_sum(tau, score, softening)

    reference = torch_reference()
    candidate = warp_candidate()
    # The reduction order is intentionally different: PyTorch reduces a
    # vectorized row block while Warp accumulates sources in one thread.  The
    # tolerance is still tight for float32, but allows the expected summation
    # order error to be measured instead of rejecting a valid kernel.
    csr_rtol, csr_atol = ((1e-2, 5e-4) if dtype == torch.float32 else (1e-10, 1e-12))
    max_abs, max_rel, l2_rel = _assert_close("exact CSR pairwise sum", reference, candidate, csr_rtol, csr_atol)
    torch_ms = _time_cuda(torch_reference, warmup, iterations, device)
    warp_ms = _time_cuda(warp_candidate, warmup, iterations, device)
    return Result("exact_csr_pairwise", n, str(dtype).split(".")[-1], None, torch_ms, warp_ms,
                  torch_ms / warp_ms, max_abs, max_rel, l2_rel)


def _bench_cic(n: int, bins: int, dtype: torch.dtype, device: torch.device, warmup: int, iterations: int) -> Result:
    tau = _random_tau(n, dtype, device).contiguous()
    bin_idx, interp_w, valid, _, dtau, _ = collective._fixed_grid(tau, bins)
    bin_idx32 = bin_idx.to(torch.int32).contiguous()
    interp_w = interp_w.contiguous()
    valid = valid.contiguous()
    charge = torch.tensor(1.0e-12, dtype=dtype, device=device)

    def torch_reference():
        return collective._bin_line_density(charge, bin_idx, interp_w, valid, bins, dtau, _use_warp=False)

    def warp_candidate():
        return warp_kernels.cic_line_density(bin_idx32, interp_w, valid, charge, bins, dtau)

    reference = torch_reference()
    candidate = warp_candidate()
    max_abs, max_rel, l2_rel = _assert_close("fixed-grid CIC line deposition", reference, candidate)
    torch_ms = _time_cuda(torch_reference, warmup, iterations, device)
    warp_ms = _time_cuda(warp_candidate, warmup, iterations, device)
    return Result("fixed_grid_cic", n, str(dtype).split(".")[-1], bins, torch_ms, warp_ms,
                  torch_ms / warp_ms, max_abs, max_rel, l2_rel)


def _bench_wake_histogram(
    n: int, bins: int, dtype: torch.dtype, device: torch.device, warmup: int, iterations: int,
) -> Result:
    dt_s = 1.0e-12
    lo = -(bins // 2)
    tau = (torch.randn(n, dtype=dtype, device=device) * (dt_s * bins * 0.15)).contiguous()
    idx = torch.round(tau / dt_s).to(torch.int64)
    in_range = (idx >= lo) & (idx < lo + bins)
    safe_idx = torch.where(in_range, idx - lo, torch.full_like(idx, bins)).to(torch.int32).contiguous()

    def torch_reference():
        histogram = torch.zeros(bins + 1, dtype=dtype, device=device)
        return histogram.scatter_add_(0, safe_idx.to(torch.int64), torch.ones(n, dtype=dtype, device=device))

    def warp_candidate():
        return warp_kernels.wake_histogram(safe_idx, bins + 1, dtype)

    reference = torch_reference()
    candidate = warp_candidate()
    max_abs, max_rel, l2_rel = _assert_close("short-range wake histogram", reference, candidate)
    torch_ms = _time_cuda(torch_reference, warmup, iterations, device)
    warp_ms = _time_cuda(warp_candidate, warmup, iterations, device)
    return Result("short_range_wake_histogram", n, str(dtype).split(".")[-1], bins, torch_ms, warp_ms,
                  torch_ms / warp_ms, max_abs, max_rel, l2_rel)


def _print_results(results: list[Result], json_output: bool) -> None:
    if json_output:
        print(json.dumps([asdict(result) for result in results], indent=2))
        return
    print("case                     N       bins  torch_ms  warp_ms  speedup  max_abs      max_rel       l2_rel")
    for result in results:
        bins = "-" if result.bins is None else str(result.bins)
        print(f"{result.case:24s} {result.n:7d} {bins:7s} {result.torch_ms:9.4f} "
              f"{result.warp_ms:8.4f} {result.speedup:7.3f} {result.max_abs:11.3e} "
              f"{result.max_rel:11.3e} {result.l2_rel:11.3e}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=_parse_sizes, default=[16384, 32768, 65536, 131072, 262144],
                        help="comma-separated particle counts (default: 16384, 32768, 65536, 131072, 262144)")
    parser.add_argument("--bins", type=int, default=256, help="fixed-grid/wake histogram bins")
    parser.add_argument("--dtype", type=_dtype, default=torch.float32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--only", choices=("all", "csr", "cic", "wake"), default="all")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("SKIP: CUDA is unavailable; run this benchmark in a CUDA-enabled Conda environment.", file=sys.stderr)
        return 2
    if not warp_kernels.available("cuda", args.dtype):
        print("SKIP: warp-lang is unavailable or does not support the requested dtype.", file=sys.stderr)
        return 2
    if args.bins < 3:
        parser.error("--bins must be at least 3")
    if args.warmup < 0 or args.iterations < 1:
        parser.error("--warmup must be nonnegative and --iterations must be positive")

    device = torch.device("cuda")
    torch.manual_seed(20261003)
    results = []
    for n in args.sizes:
        if args.only in ("all", "csr"):
            results.append(_bench_csr(n, args.dtype, device, args.warmup, args.iterations))
        if args.only in ("all", "cic"):
            results.append(_bench_cic(n, args.bins, args.dtype, device, args.warmup, args.iterations))
        if args.only in ("all", "wake"):
            results.append(_bench_wake_histogram(n, args.bins, args.dtype, device, args.warmup, args.iterations))
    _print_results(results, args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
