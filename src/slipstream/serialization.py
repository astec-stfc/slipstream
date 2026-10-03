"""Generic YAML round-trip for flat dataclasses (lattices, resonators, ...).

Not meant for dataclasses carrying large numpy arrays or non-init state
(e.g. ``ShortRangeWake``, ``LongRangeWake``) -- those have their own
tabular/SDDS I/O instead, see wakefields/shortrange.py.
"""
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
