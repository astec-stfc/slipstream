"""Tests for ``slipstream._compile_env`` -- plans/COMPILE_WARMUP_PLAN.md's
Approach A (persistent Inductor/Triton cache directories). Pure logic
only: path computation and the ``os.environ.setdefault`` semantics. Deliberately does **not** re-time an actual cold-vs-warm
``torch.compile`` run (slow, environment-dependent, already measured by
hand -- see the plan doc's own "Implementation results"); what needs a
regression guard here is that the helper computes the *right* stable
paths and never silently overrides an environment the caller already set.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from slipstream import _compile_env


def test_default_cache_root_is_platform_specific(monkeypatch: pytest.MonkeyPatch) -> None:
    if sys.platform == "win32":
        monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\someone\AppData\Local")
        root = _compile_env._default_cache_root()
        assert root == Path(r"C:\Users\someone\AppData\Local") / "slipstream-inductor-cache"
    else:
        root = _compile_env._default_cache_root()
        assert root == Path.home() / ".cache" / "slipstream-inductor"


def test_cache_dirs_are_distinct_subdirectories_of_root(tmp_path: Path) -> None:
    inductor_dir, triton_dir = _compile_env.cache_dirs(tmp_path)
    assert inductor_dir == tmp_path / "inductor"
    assert triton_dir == tmp_path / "triton"
    assert inductor_dir != triton_dir


def test_set_compile_cache_env_creates_dirs_and_sets_vars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    monkeypatch.delenv("TORCHINDUCTOR_FX_GRAPH_CACHE", raising=False)

    _compile_env.set_compile_cache_env(tmp_path)

    inductor_dir, triton_dir = _compile_env.cache_dirs(tmp_path)
    assert inductor_dir.is_dir()
    assert triton_dir.is_dir()
    assert monkeypatch is not None  # (env vars asserted below via os.environ directly)
    import os
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(inductor_dir)
    assert os.environ["TRITON_CACHE_DIR"] == str(triton_dir)
    assert os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] == "1"


def test_set_compile_cache_env_never_overrides_an_explicit_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A value the caller already set (their own shell profile, a CI
    config, ...) must win -- ``setdefault`` semantics, not a hard
    override. This is the property that makes the helper safe to import
    unconditionally from an entry script without stomping on a user's own
    setup."""
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "/already/set/by/user")
    monkeypatch.setenv("TRITON_CACHE_DIR", "/already/set/by/user/triton")

    _compile_env.set_compile_cache_env(tmp_path)

    import os
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == "/already/set/by/user"
    assert os.environ["TRITON_CACHE_DIR"] == "/already/set/by/user/triton"


def test_set_compile_cache_env_creates_explicit_paths_without_touching_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inductor = tmp_path / "inductor"
    triton = tmp_path / "triton"
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(inductor))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(triton))

    _compile_env.set_compile_cache_env()

    assert inductor.is_dir()
    assert triton.is_dir()


def test_set_compile_cache_env_is_silent_after_torch_inductor_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``import slipstream`` already loads torch, so this is the only case
    an entry script ever sees -- and Inductor/Triton read the cache env
    vars lazily (module docstring), so it must not warn as if it were a
    no-op."""
    import warnings
    import torch._inductor  # noqa: F401 -- the case every entry script is in
    assert "torch._inductor" in sys.modules
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _compile_env.set_compile_cache_env(tmp_path)

