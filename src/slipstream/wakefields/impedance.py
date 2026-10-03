"""Frequency-domain longitudinal impedance Z_parallel(f), machine-agnostic
result container for whatever computed it (e.g. ``wakefields.iw2d``)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Impedance:
    """Z_parallel(f) on a shared frequency grid, real+imaginary parts (Ohm)."""

    freq_Hz: np.ndarray
    ReZ_ohm: np.ndarray
    ImZ_ohm: np.ndarray

    def save(self, path: str | Path) -> None:
        pd.DataFrame(
            {"freq": self.freq_Hz, "ReZ": self.ReZ_ohm, "ImZ": self.ImZ_ohm}
        ).to_csv(path, index=False)

    @classmethod
    def load(cls, path: str | Path) -> "Impedance":
        df = pd.read_csv(path)
        return cls(df["freq"].to_numpy(), df["ReZ"].to_numpy(), df["ImZ"].to_numpy())

    @classmethod
    def from_sdds_file(
        cls, path: str | Path, freq_col: str = "Freq", re_col: str = "RealZ", im_col: str = "ImagZ",
    ) -> "Impedance":
        """Load Z(f) from a column-per-part ASCII SDDS file -- elegant's own
        ZLONGIT/ZTRANSVERSE impedance-table input format. ``re_col``/
        ``im_col`` pick which columns hold the real/imaginary parts:
        ZLONGIT splits Z_parallel into two separate single-column files
        (call once per file with the same ``re_col``/``im_col`` pair --
        whichever name is missing from a given file reads back as zero, so
        the two calls' results can be combined directly); ZTRANSVERSE keeps
        all four of RealX/ImagX/RealY/ImagY in one file (call once per
        plane with the matching pair).
        """
        from ..sdds_io import sdds_object

        with sdds_object() as (_sdds, x):
            x.load(str(path))
            cols = x.columnName
            freq = np.array(x.columnData[cols.index(freq_col)][0])
            ReZ = np.array(x.columnData[cols.index(re_col)][0]) if re_col in cols else np.zeros_like(freq)
            ImZ = np.array(x.columnData[cols.index(im_col)][0]) if im_col in cols else np.zeros_like(freq)
        return cls(freq, ReZ, ImZ)

    def __add__(self, other: "Impedance") -> "Impedance":
        """Length-weighted (or otherwise pre-scaled) sum of two impedance
        contributions on the same frequency grid, e.g. summing chamber
        sections of a ring."""
        if not np.allclose(self.freq_Hz, other.freq_Hz):
            raise ValueError("frequency grids must match to add impedances")
        return Impedance(self.freq_Hz, self.ReZ_ohm + other.ReZ_ohm, self.ImZ_ohm + other.ImZ_ohm)
