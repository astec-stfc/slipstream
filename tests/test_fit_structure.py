"""The pieces of examples/photocathode/fit_structure.py that need no tracking: the flat Ez maps, the steering correction and
the aperture count."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/photocathode/fit_structure.py"
U0 = 5.0 * 299792458.0  # gamma beta c of a 2.1 MeV electron


@pytest.fixture(scope="module")
def fit():
    return runpy.run_path(str(EXAMPLE))


def gun_map(path):
    """A 1.5-cell Ez map: the half cell to z = 25 mm, then a full cell whose peak is 0.94 of the cathode field."""
    z = np.linspace(0.0, 0.15, 1501)
    ez = np.zeros_like(z)
    half, full = z < 0.025, (z >= 0.025) & (z < 0.075)
    ez[half] = np.cos(np.pi / 2 * z[half] / 0.025)
    ez[full] = -0.94 * np.sin(np.pi * (z[full] - 0.025) / 0.05)
    with h5py.File(path, "w") as f:
        f.attrs["lattice"] = "test"
        f["z"], f["Ez"] = z, ez
        f["Ez"].attrs["units"] = "V/m"
    return z, ez


def test_flat_map_sets_the_full_cell_peak_and_keeps_the_half_cell(fit, tmp_path):
    source = tmp_path / "gun.hdf5"
    z, ez = gun_map(source)
    machine = SimpleNamespace(cavities=[SimpleNamespace(file=str(source))])
    path, factor = fit["flat_map"](machine, 0.80, tmp_path / "maps")
    with h5py.File(path, "r") as f:
        z2, ez2 = f["z"][:], f["Ez"][:]
        assert f.attrs["lattice"] == "test" and f["Ez"].attrs["units"] == "V/m"  # the attributes the loader reads
    half = z < 0.025
    assert np.array_equal(ez2[half], ez[half]) and np.array_equal(z2, z)
    assert np.abs(ez2[~half]).max() / ez2[0] == pytest.approx(0.80, rel=1e-3)
    assert factor == 1.0  # the cathode is the largest field: MaxE is the cathode field
    mtime = Path(path).stat().st_mtime_ns
    assert fit["flat_map"](machine, 0.80, tmp_path / "maps") == (path, factor)
    assert Path(path).stat().st_mtime_ns == mtime  # cached


def test_flat_map_above_one_is_normalised_to_the_negative_full_cell_peak(fit, tmp_path):
    source = tmp_path / "gun.hdf5"
    gun_map(source)
    machine = SimpleNamespace(cavities=[SimpleNamespace(file=str(source))])
    path, factor = fit["flat_map"](machine, 1.20, tmp_path / "maps")
    with h5py.File(path, "r") as f:
        ez = f["Ez"][:]
    peak = ez[np.argmax(np.abs(ez))]
    assert peak < 0.0 and factor == pytest.approx(-1.20, rel=1e-3)
    # the loader scales the map by MaxE / peak: with MaxE = E * factor the cathode field is E, with its sign
    assert (105e6 * factor / peak) * ez[0] == pytest.approx(105e6, rel=1e-9)


def test_steering_kicks_centre_the_beam_on_the_bpm_and_screen_at_the_reference_phase(fit, monkeypatch):
    planes = {}

    def screen(machine, args, offset, z):
        n = 50
        x = np.linspace(-1e-3, 1e-3, n) + (3e-3 + 2e-3 * z)  # a beam displaced and tilted at the reference phase
        y = np.linspace(-1e-3, 1e-3, n) - (1e-3 + 0.5e-3 * z)
        planes[z] = (x.mean(), y.mean())
        return {"x": x, "y": y, "q": np.full(n, 1e-15), "ux": np.zeros(n), "uy": np.zeros(n), "uz": np.full(n, U0)}, {}

    monkeypatch.setitem(fit["solve_correctors"].__globals__, "screen_at", screen)
    kicks = fit["solve_correctors"](None, None)
    for key, axis in (("x", 0), ("y", 1)):
        for z in fit["OBSERVATION_Z"]:
            moved = sum(kicks[key][i] * (z - zc) for i, zc in enumerate(fit["CORRECTOR_Z"])) / U0
            assert planes[z][axis] + moved == pytest.approx(
                0.0, abs=1e-12
            )  # the kicks cancel the centroid at both planes


def test_aperture_charge_counts_the_square_after_the_steering_and_faster_electrons_are_steered_less(fit):
    def electrons(x, y, uz):
        n = len(x)
        return {"x": np.array(x), "y": np.array(y), "q": np.full(n, 1e-15), "ux": np.zeros(n), "uy": np.zeros(n),
                "uz": np.array(uz) * U0}  # fmt: skip

    s = electrons([0.0, 0.009, 0.011, 0.0], [0.0, 0.0, 0.0, 0.011], [1, 1, 1, 1])
    assert fit["aperture_charge_pC"](s) == pytest.approx(2e-3)  # the square: |x| and |y| below 10 mm, 1e-3 pC each
    reach = fit["APERTURE_Z"] - fit["CORRECTOR_Z"][0]
    kicks = {"x": [-0.004 * U0 / reach, 0.0], "y": [0.0, 0.0]}  # -4 mm for an electron with u = U0
    assert fit["aperture_charge_pC"](s, kicks) == pytest.approx(
        3e-3
    )  # 0 -> -4, 9 -> 5, 11 -> 7 mm in; y = 11 mm stays out
    pushed = {"x": [0.004 * U0 / reach, 0.0], "y": [0.0, 0.0]}
    s = electrons([0.007, 0.007], [0.0, 0.0], [1, 2])
    assert fit["aperture_charge_pC"](s, pushed) == pytest.approx(
        1e-3
    )  # 7 + 4 = 11 mm is out, 7 + 2 = 9 mm (twice the momentum) in


def test_flat_map_never_ties_the_cells(fit, tmp_path):
    source = tmp_path / "gun.hdf5"
    gun_map(source)
    machine = SimpleNamespace(cavities=[SimpleNamespace(file=str(source))])
    above, below = (
        fit["flat_map"](machine, 1.0003, tmp_path / "maps"),
        fit["flat_map"](machine, 0.9996, tmp_path / "maps"),
    )
    assert above[1] == pytest.approx(-1.002, rel=1e-3)  # nudged to 1.002: the full cell is clearly the largest
    assert below[1] == 1.0 and "flat0.9980" in below[0]  # nudged to 0.998: the cathode is clearly the largest
