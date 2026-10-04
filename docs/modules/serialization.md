# `slipstream.serialization`

[Source](../../src/slipstream/serialization.py) · [Documentation index](../README.md)

## Physics

Serialization preserves scalar machine parameters; it introduces no transport or coordinate conversion.

## Usage

Use `dump_yaml(obj, path)` and `load_yaml(Class, path)` for dataclasses. Array-backed particle, wake and diagnostic objects use their dedicated formats.

### API reference

```python
def dump_yaml(obj, path: str | Path) -> None: ...
def load_yaml(cls: type[T], path: str | Path) -> T: ...
```
