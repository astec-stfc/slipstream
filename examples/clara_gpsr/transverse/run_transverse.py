"""Transverse study: reconstruct the CLARA S02 bunch from images on CLA-S07-DIA-SCR-04 after a scan of two S07 quadrupoles (CLA-S07-MAG-QUAD-08
and -09, 12 pairs) with the transverse deflecting cavity off and on at 3 MV. Five more quadrupole pairs are held out. The straight S07 line from
its start to the screen is repeated for each scan row. See ../README.md for the model and the cases (A to D).

    python run_transverse.py --bunch CLA-S02-SIM-APER-01.openpmd.hdf5 --workdir _cache --out results --cases ACBD --iterations 1200
    python run_transverse.py --bunch ... --smoke        # a CPU check of the whole chain in a minute or two
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the shared modules sit in clara_gpsr/

from clara_model import Line, Study  # noqa: E402
from run_clara_gpsr import main  # noqa: E402

STUDY = Study(
    name="transverse",
    line=Line(screen="CLA-S07-DIA-SCR-04", s07_last="CLA-S07-DIA-SCR-04"),
    scan_quads=("CLA-S07-MAG-QUAD-08", "CLA-S07-MAG-QUAD-09"),
    quad_pairs=tuple((a, b) for a in (-4.0, 0.0, 4.0, 8.0) for b in (-2.0, 2.6, 6.0)),  # k1, 1/m^2
    heldout=((-5.0, 4.0), (-2.0, 0.0), (2.0, 4.5), (6.0, 1.0), (7.0, 5.0)),
)

if __name__ == "__main__":
    main(STUDY, __doc__)
