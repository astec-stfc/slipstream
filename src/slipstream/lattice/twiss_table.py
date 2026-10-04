"""Stores element-local Twiss/dispersion with explicit segment lengths for length-weighted IBS
averaging. See docs/modules/lattice/twiss_table.md."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class TwissTable:
    """One row per element (or slice) around the ring."""

    s_m: np.ndarray
    length_m: np.ndarray
    beta_x_m: np.ndarray
    alpha_x: np.ndarray
    D_x_m: np.ndarray
    Dp_x: np.ndarray
    beta_y_m: np.ndarray
    alpha_y: np.ndarray
    D_y_m: np.ndarray
    Dp_y: np.ndarray

    def __post_init__(self) -> None:
        # np.array (not asarray): from_csv's pandas-CoW arrays come back
        # read-only, see TransverseWake.__post_init__'s own comment.
        for name in (
            "s_m", "length_m", "beta_x_m", "alpha_x", "D_x_m", "Dp_x",
            "beta_y_m", "alpha_y", "D_y_m", "Dp_y",
        ):
            setattr(self, name, np.array(getattr(self, name), dtype=np.float64))

    def to_csv(self, path: str | Path) -> None:
        pd.DataFrame({
            "s_m": self.s_m, "length_m": self.length_m,
            "beta_x_m": self.beta_x_m, "alpha_x": self.alpha_x, "D_x_m": self.D_x_m, "Dp_x": self.Dp_x,
            "beta_y_m": self.beta_y_m, "alpha_y": self.alpha_y, "D_y_m": self.D_y_m, "Dp_y": self.Dp_y,
        }).to_csv(path, index=False)

    @classmethod
    def from_csv(cls, path: str | Path) -> "TwissTable":
        df = pd.read_csv(path)
        return cls(
            df["s_m"].to_numpy(), df["length_m"].to_numpy(),
            df["beta_x_m"].to_numpy(), df["alpha_x"].to_numpy(), df["D_x_m"].to_numpy(), df["Dp_x"].to_numpy(),
            df["beta_y_m"].to_numpy(), df["alpha_y"].to_numpy(), df["D_y_m"].to_numpy(), df["Dp_y"].to_numpy(),
        )

    @classmethod
    def from_madx_dframe(cls, df: pd.DataFrame) -> "TwissTable":
        """Adapts cpymad's ``Madx.twiss().dframe()`` column names (MAD-X's own lowercase convention:
        ``s``, ``l``, ``betx``, ``bety``, ``alfx``, ``alfy``, ``dx``, ``dy``, ``dpx``, ``dpy``) to
        this table's naming."""
        return cls(
            df["s"].to_numpy(), df["l"].to_numpy(),
            df["betx"].to_numpy(), df["alfx"].to_numpy(), df["dx"].to_numpy(), df["dpx"].to_numpy(),
            df["bety"].to_numpy(), df["alfy"].to_numpy(), df["dy"].to_numpy(), df["dpy"].to_numpy(),
        )
