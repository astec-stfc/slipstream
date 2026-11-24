"""Longitudinal study: reconstruct the CLARA S02 bunch from images on CLA-SP3-DIA-SCR-01, the first screen after the SP3 spectrometer dipole.

The beam follows S07 to CLA-S07-MAG-HVCOR-05 and then the 30 degree SP3 dipole. The dipole disperses the beam horizontally (about 0.5 m at the
screen) and the transverse deflecting cavity in S07 streaks it vertically, so the screen shows the longitudinal phase space directly: energy on
x, time on y. The scan varies two S07 quadrupoles (CLA-S07-MAG-QUAD-06 and -07, 12 pairs, chosen to move the vertical waist onto the screen so
the streak is resolved) with the cavity off and on at 3 MV; five more pairs are held out. The SP3 quadrupoles are off. See ../README.md for the
model and the cases (A to D).

    python run_longitudinal.py --bunch CLA-S02-SIM-APER-01.openpmd.hdf5 --workdir _cache --out results --cases ACBD --iterations 1200
    python run_longitudinal.py --bunch ... --smoke        # a CPU check of the whole chain in a minute or two
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the shared modules sit in clara_gpsr/

from clara_model import Line, Study  # noqa: E402
from run_clara_gpsr import main  # noqa: E402

STUDY = Study(
    name="longitudinal",
    line=Line(
        screen="CLA-SP3-DIA-SCR-01",
        s07_last="CLA-S07-MAG-HVCOR-05",
        branch="SP3",
        dipole="CLA-SP3-MAG-DIP-01",
        dipole_angle_rad=math.pi / 6,  # the survey rotation after the dipole; LAURA exports the angle from a current the lattice does not hold
        quads_off=("CLA-SP3-MAG-QUAD-01", "CLA-SP3-MAG-QUAD-02"),
    ),
    scan_quads=("CLA-S07-MAG-QUAD-06", "CLA-S07-MAG-QUAD-07"),
    quad_pairs=tuple((a, b) for a in (-3.0, -1.5, 0.0, 1.5) for b in (-4.0, -3.0, -1.5)),  # k1, 1/m^2
    heldout=((-2.2, -2.2), (-0.7, -3.5), (0.7, -2.3), (1.0, -3.6), (-1.0, -1.7)),
    pixels=96,
    pixel_size_m=(2.2e-4, 5e-5),  # horizontal (energy), vertical (time)
)

if __name__ == "__main__":
    main(STUDY, __doc__)
