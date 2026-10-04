# `slipstream._compile_env`

[Source](../../src/slipstream/_compile_env.py) · [Documentation index](../README.md)

## Physics

Cache configuration changes compilation storage, without changing transport or physical models.

### Limits and conventions

- This opt-in module has an import-time environment/directory side effect; ordinary import slipstream does not import it.
- Do not share compiled cache roots between Windows and WSL/Linux.

### `_default_cache_root`

One stable root per OS -- survives what the volatile default does
not. Windows: `%LOCALAPPDATA%\slipstream-inductor-cache` (per-user,
not reaped by Storage Sense the way `%TEMP%` is). WSL/Linux:
`~/.cache/slipstream-inductor` (survives `wsl --shutdown`, unlike
`/tmp`).

## Usage

Explicitly importing slipstream._compile_env configures persistent Inductor/Triton caches via its final set_compile_cache_env() call. Call set_compile_cache_env(root=None) for a chosen root; cache_dirs(root) returns their paths. Existing environment overrides are honored.

### API reference

```python
def cache_dirs(root: Path | None=None) -> tuple[Path, Path]: ...
def set_compile_cache_env(root: Path | None=None) -> None: ...
```

### `cache_dirs`

`(inductor_dir, triton_dir)` under `root` (default:
`_default_cache_root()`) -- the two directories
`set_compile_cache_env` points `TORCHINDUCTOR_CACHE_DIR`/
`TRITON_CACHE_DIR` at.

### `set_compile_cache_env`

Set `TORCHINDUCTOR_CACHE_DIR`/`TRITON_CACHE_DIR` to a stable
directory and `TORCHINDUCTOR_FX_GRAPH_CACHE=1` (belt-and-braces --
already the torch 2.14 default, set explicitly so a future config
change can't silently disable it unnoticed), all via ``os.environ.
setdefault`` so an explicit value the caller already set always wins.
Creates the effective cache directories, including explicit overrides;
unused default directories are left alone. No other Inductor config is touched.
