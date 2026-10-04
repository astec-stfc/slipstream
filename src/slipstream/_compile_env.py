"""Persistent ``torch.compile``/Triton cache directories -- see
``plans/COMPILE_WARMUP_PLAN.md`` (Approach A). The default cache location
(``%TEMP%\\torchinductor_<user>`` on Windows, ``/tmp/torchinductor_<user>``
in WSL/Linux) is volatile -- Storage Sense/Disk Cleanup purges ``%TEMP%``,
``wsl --shutdown``/VM teardown wipes ``/tmp`` -- so a warm on-disk cache
never survives between separate process runs even though the caching
machinery itself (``fx_graph_cache``, autotune cache) is already on by
default. This module just points those caches at a stable directory
instead; **no compiled function's semantics change**.

Import order does not matter on the torch/Triton this repo runs (torch
2.14.0, Triton 3.7.1, Windows -- measured 2026-09-24): Inductor and Triton
read ``TORCHINDUCTOR_CACHE_DIR``/``TRITON_CACHE_DIR`` lazily, so setting
them after ``import torch`` -- even after a first ``torch.compile`` -- still
redirects every later cache write. That matters because ``import
slipstream`` itself imports torch (``slipstream/__init__.py``), so
``import slipstream._compile_env`` can never run before ``import torch``
anyway. Not re-measured on WSL/Linux::

    import slipstream._compile_env  # noqa: F401
    import torch
    ...

Deliberately **not** imported from ``slipstream/__init__.py`` -- an
import-time global environment-variable side effect on every
``import slipstream`` would surprise callers who never asked for it (this
repo's own convention). Opt in explicitly at entry points instead
(``run_clara_linac_demo.py``, ``track_moustache.py``, ...), or set the
same two env vars directly in a shell profile using the paths
``cache_dirs()`` below computes.

**Never share one cache root between Windows and WSL/Linux** -- compiled
Inductor/Triton artifacts are OS- and toolchain-specific; pointing both
sides at the same directory guarantees spurious cache misses or load
failures, not a shared speedup.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _default_cache_root() -> Path:
    """One stable root per OS -- survives what the volatile default does
    not. Windows: ``%LOCALAPPDATA%\\slipstream-inductor-cache`` (per-user,
    not reaped by Storage Sense the way ``%TEMP%`` is). WSL/Linux:
    ``~/.cache/slipstream-inductor`` (survives ``wsl --shutdown``, unlike
    ``/tmp``)."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "slipstream-inductor-cache"
    return Path.home() / ".cache" / "slipstream-inductor"


def cache_dirs(root: Path | None = None) -> tuple[Path, Path]:
    """``(inductor_dir, triton_dir)`` under ``root`` (default:
    ``_default_cache_root()``) -- the two directories
    ``set_compile_cache_env`` points ``TORCHINDUCTOR_CACHE_DIR``/
    ``TRITON_CACHE_DIR`` at."""
    root = root or _default_cache_root()
    return root / "inductor", root / "triton"


def set_compile_cache_env(root: Path | None = None) -> None:
    """Set ``TORCHINDUCTOR_CACHE_DIR``/``TRITON_CACHE_DIR`` to a stable
    directory and ``TORCHINDUCTOR_FX_GRAPH_CACHE=1`` (belt-and-braces --
    already the torch 2.14 default, set explicitly so a future config
    change can't silently disable it unnoticed), all via ``os.environ.
    setdefault`` so an explicit value the caller already set always wins.
    Creates the effective cache directories, including explicit overrides;
    unused default directories are left alone. No other Inductor config is touched."""
    inductor_dir, triton_dir = cache_dirs(root)
    inductor_dir = Path(os.environ.get("TORCHINDUCTOR_CACHE_DIR", str(inductor_dir)))
    triton_dir = Path(os.environ.get("TRITON_CACHE_DIR", str(triton_dir)))
    inductor_dir.mkdir(parents=True, exist_ok=True)
    triton_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(inductor_dir))
    os.environ.setdefault("TRITON_CACHE_DIR", str(triton_dir))
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")


set_compile_cache_env()
