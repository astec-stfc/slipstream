"""Cache configuration changes compilation storage, without changing transport or physical models. See
docs/modules/_compile_env.md."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _default_cache_root() -> Path:
    """One stable root per OS -- survives what the volatile default does not."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "slipstream-inductor-cache"
    return Path.home() / ".cache" / "slipstream-inductor"


def cache_dirs(root: Path | None = None) -> tuple[Path, Path]:
    """``(inductor_dir, triton_dir)`` under ``root`` (default: ``_default_cache_root()``) -- the two
    directories ``set_compile_cache_env`` points ``TORCHINDUCTOR_CACHE_DIR``/ ``TRITON_CACHE_DIR``
    at."""
    root = root or _default_cache_root()
    return root / "inductor", root / "triton"


def set_compile_cache_env(root: Path | None = None) -> None:
    """Compute set compile cache env; see the module documentation for details."""
    inductor_dir, triton_dir = cache_dirs(root)
    inductor_dir = Path(os.environ.get("TORCHINDUCTOR_CACHE_DIR", str(inductor_dir)))
    triton_dir = Path(os.environ.get("TRITON_CACHE_DIR", str(triton_dir)))
    inductor_dir.mkdir(parents=True, exist_ok=True)
    triton_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(inductor_dir))
    os.environ.setdefault("TRITON_CACHE_DIR", str(triton_dir))
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")


set_compile_cache_env()
