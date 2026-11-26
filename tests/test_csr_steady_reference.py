"""Steady-state CSR of one bend against a reference that uses no grid or density estimate (issue #21).

The beam is a pencil (no transverse size, zero ``delta``) with a Gaussian line density, so it keeps its shape through the bend.
The reference is the unshielded 1D steady-state wake of Saldin, Schneidmiller and Yurkov,
``delta(tau) = 2 L / (4 pi eps0 (3 rho^2)^(1/3) E) * int_0^inf lambda'(tau + u) u^(-1/3) du`` (``E`` in eV), by adaptive
quadrature. Sign, kick shape and mean loss are checked separately. Shape errors are relative to the largest exact kick, for
particles within three rms lengths.

The live elegant test needs an elegant binary and the ``sdds`` extra; it skips without them.
"""
import functools
import math

import numpy as np
import pytest
import torch
from scipy.integrate import quad
from scipy.special import ndtri
from test_chicane_csr_off import _elegant, _elegant_exe

from slipstream.constants import EPS0_F_PER_M
from slipstream.tracking.native.collective import csr_steady_state_delta_kick, csr_steady_state_delta_kick_fast
from slipstream.tracking.native.lattice_import import ElementSpec
from slipstream.tracking.native.linac import LinacTracker

ENERGY_EV, SIGMA_M, CHARGE_C, LENGTH_M, ANGLE = 190e6, 20e-6, 100e-12, 0.2, 0.05
RHO_M = LENGTH_M / ANGLE
ENERGY = torch.tensor(ENERGY_EV, dtype=torch.float64)
K_STEADY = 2 ** (8 / 3) * math.gamma(5 / 6) / (8 * math.sqrt(math.pi) * 3 ** (1 / 3))
MEAN_DELTA = -K_STEADY * CHARGE_C / (4 * math.pi * EPS0_F_PER_M) * RHO_M ** (-2 / 3) * SIGMA_M ** (-4 / 3) * LENGTH_M / ENERGY_EV
GRID_M = np.linspace(-6 * SIGMA_M, 6 * SIGMA_M, 1201)


def quantile_bunch(n):
    """``tau`` at the Gaussian quantiles: a sample without shot noise, so that only the estimator's own error remains."""
    return SIGMA_M * ndtri((np.arange(n) + 0.5) / n)


@functools.lru_cache(maxsize=None)
def reference_on_grid():
    """The exact kick on ``GRID_M``. ``u = t**1.5`` removes the singularity: ``u**(-1/3) du = 1.5 dt``."""
    peak_density = CHARGE_C / (math.sqrt(2 * math.pi) * SIGMA_M)

    def slope(v):
        return -v / SIGMA_M ** 2 * peak_density * math.exp(-0.5 * (v / SIGMA_M) ** 2)

    scale = 2 * LENGTH_M / (4 * math.pi * EPS0_F_PER_M * (3 * RHO_M ** 2) ** (1 / 3) * ENERGY_EV)

    def at(tau):
        top = (max(12 * SIGMA_M - tau, 0.0) + SIGMA_M) ** (2 / 3)
        marks = [u ** (2 / 3) for u in (-tau - 2 * SIGMA_M, -tau, -tau + 2 * SIGMA_M) if u > 0]
        return 1.5 * scale * quad(lambda t: slope(tau + t ** 1.5), 0, top, points=[m for m in marks if m < top] or None,
                                  limit=200, epsabs=0, epsrel=1e-10)[0]

    return np.array([at(tau) for tau in GRID_M])


def errors(tau, kick):
    """``(wrong sign fraction, rms shape error, largest shape error, mean error)`` of a kick given per particle."""
    exact = np.interp(tau, GRID_M, reference_on_grid())
    peak = np.abs(reference_on_grid()).max()
    core = np.abs(tau) < 3 * SIGMA_M
    clear = core & (np.abs(exact) > 0.05 * peak)  # away from the zero crossings
    shape = (kick - exact)[core] / peak
    return (np.mean(np.sign(kick[clear]) != np.sign(exact[clear])), np.sqrt(np.mean(shape ** 2)), np.abs(shape).max(),
            kick.mean() / MEAN_DELTA - 1)


def fast_kick(tau, bins):
    return csr_steady_state_delta_kick_fast(torch.tensor(tau), CHARGE_C / len(tau), RHO_M, ENERGY, LENGTH_M, n_bins=bins).numpy()


def test_reference_has_the_closed_form_mean_and_the_head_gains_energy():
    tau = quantile_bunch(100000)
    assert np.interp(tau, GRID_M, reference_on_grid()).mean() == pytest.approx(MEAN_DELTA, rel=1e-4)
    head, centre, tail = np.interp([-1.5 * SIGMA_M, 0.0, 1.5 * SIGMA_M], GRID_M, reference_on_grid())
    assert head > 0 > tail > centre


def test_fast_kernel_sign_shape_and_mean_converge_to_the_reference():
    tau = quantile_bunch(100000)
    found = {bins: errors(tau, fast_kick(tau, bins)) for bins in (64, 128, 256)}
    for coarse, fine in ((64, 128), (128, 256)):
        assert found[fine][1] < 0.5 * found[coarse][1]  # shape
        assert abs(found[fine][3]) < 0.5 * abs(found[coarse][3])  # mean
    wrong, rms, largest, mean = found[256]
    assert wrong == 0
    assert rms < 0.002 and largest < 0.003
    assert abs(mean) < 0.002


def test_linac_tracker_bend_gives_the_reference_kick():
    tau = quantile_bunch(100000)
    particles = torch.zeros((len(tau), 6), dtype=torch.float64)
    particles[:, 4] = torch.tensor(tau)
    spec = ElementSpec("B", "bend", LENGTH_M, angle=ANGLE, csr=True, csr_n_slices=8, csr_bins=256)
    options = dict(energy_eV=ENERGY_EV, device="cpu", always_change_p0=False, bunch_charge_C=CHARGE_C,
                   fast_collective_effects=True, n_kicks_bend=8)
    assert not LinacTracker([spec], csr=False, **options).track(particles)[0][:, 5].any()  # the bend itself leaves delta
    kick = LinacTracker([spec], csr=True, **options).track(particles)[0][:, 5].numpy()
    wrong, rms, largest, mean = errors(tau, kick)
    assert wrong == 0
    assert rms < 0.002 and largest < 0.003
    assert abs(mean) < 0.002


def test_pairwise_kernel_is_limited_by_its_density_estimate():
    """The default kernel (``fast_collective_effects=False``) sums exactly over a Gaussian density estimate with Silverman's
    bandwidth, 0.21 rms lengths at 3000 particles, softened by half of it. Its error is far larger than the grid kernel's and
    falls only slowly with the particle number. The upper bounds record the present accuracy; they are not a target."""
    found = {}
    for n in (3000, 10000):
        tau = quantile_bunch(n)
        found[n] = errors(tau, csr_steady_state_delta_kick(torch.tensor(tau), CHARGE_C / n, RHO_M, ENERGY, LENGTH_M).numpy())
    wrong, rms, largest, mean = found[3000]
    assert wrong < 0.02  # only close to the zero crossings, from the softening
    assert rms < 0.09 and largest < 0.13
    assert abs(mean) < 0.07
    assert found[10000][1] < found[3000][1] and abs(found[10000][3]) < abs(found[3000][3])


@pytest.mark.filterwarnings("ignore:.*ignored attributes")
def test_live_elegant_steady_state_bend_converges_to_the_reference(tmp_path):
    exe = _elegant_exe()
    if exe is None:
        pytest.skip("no elegant binary found")
    pytest.importorskip("sdds")
    particles = np.zeros((100000, 6))
    particles[:, 4] = quantile_bunch(len(particles))
    found = {}
    for bins in (400, 1600):
        def deck(path, _lsc_bins, bins=bins):  # the shared runner tracks the beamline named CHICANE
            path.write_text(f"Q: CHARGE, TOTAL={CHARGE_C}\nB: CSRCSBEND, L={LENGTH_M}, ANGLE={ANGLE}, N_SLICES=8, BINS={bins},"
                            " SG_HALFWIDTH=1, CSR=1, STEADY_STATE=1\nCHICANE: LINE=(Q,B)\n")

        out = _elegant(exe, tmp_path / f"bins_{bins}", particles, 0, deck=deck, change_p0=0)
        found[bins] = errors(particles[:, 4], out[:, 5])
    assert found[400][0] == found[1600][0] == 0
    assert found[400][1] < 0.01 and abs(found[400][3]) < 0.007
    assert found[1600][1] < 0.0025 and abs(found[1600][3]) < 0.002
    assert found[1600][1] < found[400][1] / 3  # first order in the bin width
