"""Plots and the held-out check in the true machine for a finished longitudinal run (see ../plot_results.py).

    python plot_longitudinal.py --bunch CLA-S02-SIM-APER-01.openpmd.hdf5 --workdir _cache --out results
"""

from run_longitudinal import STUDY  # noqa: E402  (also puts the shared modules on the path)
from plot_results import main  # noqa: E402

if __name__ == "__main__":
    main(STUDY, __doc__)
