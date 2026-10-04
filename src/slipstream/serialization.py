"""Serialization preserves scalar machine parameters; it introduces no transport or coordinate
conversion. See docs/modules/serialization.md."""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TypeVar

import yaml

T = TypeVar("T")


def dump_yaml(obj, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(dataclasses.asdict(obj), sort_keys=False))


def load_yaml(cls: type[T], path: str | Path) -> T:
    data = yaml.safe_load(Path(path).read_text())
    return cls(**data)
