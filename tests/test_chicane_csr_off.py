"""CSR-off chicane transport against elegant, and the elegant setting that spoils the comparison (issue #20).

elegant's ``CSRDRIFT`` applies longitudinal space charge whenever ``LSC_BINS > 0``, even with ``CSR=0``
(``track_through_driftCSR`` in ``csbend.c``). A "CSR off" elegant run with ``LSC_BINS > 0`` therefore still has a
collective kick, which raises the emittance of a bunch that compresses in the chicane. With ``LSC_BINS=0`` the two
codes agree particle by particle.

The reference coordinates come from elegant 2026.3.0 on Windows. The live test needs an elegant binary, found on
``PATH`` or under ``ELEGANT_HOME``, and the ``sdds`` extra; it skips without them.
"""
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from slipstream.constants import C_LIGHT, ELECTRON_MASS_EV
from slipstream.sdds_io import sdds_object
from slipstream.tracking.native.elements import _slopes_from_canonical
from slipstream.tracking.native.lattice_import import parse_elegant_lattice
from slipstream.tracking.native.linac import LinacTracker

ENERGY_EV = 190e6
P0 = math.sqrt((ENERGY_EV / ELECTRON_MASS_EV) ** 2 - 1.0)  # reference momentum in m_e c

# Synthetic four-dipole chicane; not an operational machine.
_BEND = ("{name}: CSRCSBEND, L=0.2, ANGLE={angle}, E1={e1}, E2={e2}, HGAP=0.005, FINT=0.4, EDGE_ORDER=2,"
         " N_SLICES=8, INTEGRATION_ORDER=4, NONLINEAR=1, BINS=64, SG_HALFWIDTH=1, CSR=0\n")
_DRIFT = "{name}: CSRDRIFT, L={length}, USE_STUPAKOV=1, DZ=0.01, CSR=0, LSC_BINS={lsc_bins}, LSC_INTERPOLATE=1\n"


def _lattice(path: Path, lsc_bins: int) -> Path:
    bends = (("B1", 0.13, 0.0, 0.13), ("B2", -0.13, -0.13, 0.0), ("B3", -0.13, 0.0, -0.13), ("B4", 0.13, 0.13, 0.0))
    drifts = (("D1", 1.5), ("D2", 0.4), ("D3", 1.5), ("D4", 0.2))
    path.write_text(
        "Q: CHARGE, TOTAL=1e-10\n"
        + "".join(_BEND.format(name=n, angle=a, e1=e1, e2=e2) for n, a, e1, e2 in bends)
        + "".join(_DRIFT.format(name=n, length=length, lsc_bins=lsc_bins) for n, length in drifts)
        + "CHICANE: LINE=(Q,B1,D1,B2,D2,B3,D3,B4,D4)\n"
    )
    return path


def _native(lattice: Path, particles: np.ndarray) -> np.ndarray:
    # The importer does not set the bend integration: n_kicks_bend=8 matches the deck's N_SLICES=8.
    tracker = LinacTracker(parse_elegant_lattice(lattice, "CHICANE"), energy_eV=ENERGY_EV, device="cpu", dtype=torch.float64,
                           n_kicks_bend=8)
    out = tracker.track(torch.tensor(particles, dtype=torch.float64))[0].numpy()
    out[:, 4] -= out[0, 4]  # elegant's time is absolute: compare arrival relative to the first particle
    return out


def _emittance_um(p: np.ndarray) -> float:
    """Projected normalised horizontal rms emittance from canonical (x, px)."""
    return 1e6 * P0 * math.sqrt(np.linalg.det(np.cov(p[:, 0], p[:, 1], bias=True)))


# Inputs, and elegant's exit (LSC_BINS=0), as canonical (x, px, y, py, tau [m], delta); exit tau relative to row 0.
INPUT = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [3e-4, 0.0, 2e-4, 0.0, 0.0, 0.0],
    [0.0, 5e-5, 0.0, -4e-5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 4e-4, 7e-3],
    [-2e-4, -3e-5, 1e-4, 2e-5, -4e-4, -7e-3],
])
ELEGANT_EXIT = np.array([
    [2.7946530723257923e-15, 1.5612511283791264e-16, 0.0, 0.0, 0.0, 0.0],
    [0.0002999990686047328, 7.68405824118504e-12, 6.663174005533022e-05, -5.002800315938839e-05, 6.8641989869888675e-09, 0.0],
    [0.0002227912230106479, 5.000000204558362e-05, -0.00013949586541544924, -1.532708907825237e-05, 9.503414055700432e-09, 0.0],
    [-8.098538660132127e-10, -3.5031076923118507e-10, 0.0, 0.0, 1.3210567557209174e-05, 0.006999999999999673],
    [-0.0003346519484654401, -2.9999511131839746e-05, 0.0001024262191242173, -1.7601627891879664e-05, -4.860084363400574e-06,
     -0.006999999999999895],
])


@pytest.mark.filterwarnings("ignore:.*ignored attributes")
def test_csr_off_chicane_matches_elegant_reference(tmp_path):
    out = _native(_lattice(tmp_path / "chicane.lte", 0), INPUT)
    np.testing.assert_allclose(out, ELEGANT_EXIT, rtol=0, atol=1e-10)


def _elegant_exe() -> Path | None:
    found = shutil.which("elegant")
    if found:
        return Path(found)
    exe = Path(os.environ.get("ELEGANT_HOME", r"C:\Program Files\APS\Elegant x64")) / "elegant.exe"
    return exe if exe.exists() else None


def _elegant(exe: Path, work: Path, particles: np.ndarray, lsc_bins: int) -> np.ndarray:
    work.mkdir()
    _lattice(work / "chicane.lte", lsc_bins)
    p = torch.tensor(particles, dtype=torch.float64)
    xp, yp, _ = _slopes_from_canonical(p[:, 1], p[:, 3], p[:, 5])
    columns = {"x": (particles[:, 0], "m"), "xp": (xp.numpy(), ""), "y": (particles[:, 2], "m"), "yp": (yp.numpy(), ""),
               "t": (particles[:, 4] / C_LIGHT, "s"), "p": (P0 * (1 + particles[:, 5]), "m$be$nc")}
    with sdds_object() as (sdds, file):
        for name, (_, units) in columns.items():
            file.defineColumn(name, units=units, type=sdds.SDDS_DOUBLE)
        file.defineColumn("particleID", type=sdds.SDDS_LONG)
        file.defineParameter("pCentral", type=sdds.SDDS_DOUBLE)
        for name, (values, _) in columns.items():
            file.setColumnValueList(name, values.tolist())
        file.setColumnValueList("particleID", list(range(1, len(particles) + 1)))
        file.setParameterValue("pCentral", P0)
        file.save(str(work / "in.sdds"))
    (work / "run.ele").write_text(
        f'&run_setup lattice="chicane.lte", use_beamline=CHICANE, p_central={P0!r}, always_change_p0=1,'
        ' default_order=3, output="out.sdds" &end\n&run_control n_steps=1 &end\n'
        '&sdds_beam input="in.sdds", input_type="elegant" &end\n&track &end\n'
    )
    env = dict(os.environ)
    if (exe.parent / "defns.rpn").exists():
        env.setdefault("RPN_DEFNS", str(exe.parent / "defns.rpn"))
    done = subprocess.run([str(exe), "run.ele"], cwd=work, env=env, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-500:]
    with sdds_object() as (_, file):
        file.load(str(work / "out.sdds"))
        out = {n: np.array(file.columnData[file.columnName.index(n)][0], float) for n in (*columns, "particleID")}
        p_central = float(file.parameterData[file.parameterName.index("pCentral")][0])
    order = np.argsort(out["particleID"])
    assert len(order) == len(particles), "elegant lost particles"
    out = {name: values[order] for name, values in out.items()}
    delta = out["p"] / p_central - 1.0
    norm = (1 + delta) / np.sqrt(1 + out["xp"] ** 2 + out["yp"] ** 2)
    tau = out["t"] * C_LIGHT
    return np.stack([out["x"], out["xp"] * norm, out["y"], out["yp"] * norm, tau - tau[0], delta], -1)


@pytest.mark.filterwarnings("ignore:.*ignored attributes")
def test_live_elegant_csrdrift_lsc_bins_explains_csr_off_emittance(tmp_path):
    exe = _elegant_exe()
    if exe is None:
        pytest.skip("no elegant binary found")
    pytest.importorskip("sdds")
    rng = np.random.default_rng(20)
    particles = rng.standard_normal((20000, 6)) * [1e-4, 2e-5, 1e-4, 2e-5, 4e-4, 1e-4]
    particles[:, 5] += 17.0 * particles[:, 4]  # chirp that compresses in the chicane
    # Zero mean: with always_change_p0, elegant also re-matches the momentum on the zero-length CHARGE element before
    # B1, which the importer drops. See docs/modules/tracking/native/linac.md.
    particles -= particles.mean(0)

    native = _native(_lattice(tmp_path / "native.lte", 0), particles)
    off = _elegant(exe, tmp_path / "lsc_bins_0", particles, 0)
    shipped = _elegant(exe, tmp_path / "lsc_bins_20", particles, 20)

    np.testing.assert_allclose(native, off, rtol=0, atol=1e-9)
    assert _emittance_um(native) == pytest.approx(_emittance_um(off), rel=1e-9)
    # Same elegant deck with only LSC_BINS changed, CSR still 0: energies move and the emittance grows (0.746 to 0.879 um).
    assert np.abs(shipped[:, 5] - off[:, 5]).max() > 1e-5
    assert _emittance_um(shipped) > 1.1 * _emittance_um(off)
