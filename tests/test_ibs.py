"""Tests for slipstream.ibs (Bane's high-energy approximation and the
general Bjorken-Mtingwa formula).

Pure-function tests only -- no tracker/torch involved, see
test_ibs_tracker.py for the tracker-integration checks (recompute cadence,
compile-safety, actual emittance growth over turns).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from slipstream import ibs
from slipstream.constants import R_E_M, R_P_M
from slipstream.lattice.twiss_table import TwissTable

# A loosely light-source-like reference point, reused across tests.
BASE_KWARGS = dict(
    gamma=4305.0,
    n_particles_bunch=5e11,
    sigma_z_m=3e-3,
    sigma_p=7.3e-4,
    epsx_m=2e-9,
    epsy_m=2e-11,
    beta_x_m=5.0, alpha_x=0.0, D_x_m=0.05, Dp_x=0.0,
    beta_y_m=5.0, alpha_y=0.0, D_y_m=0.0, Dp_y=0.0,
)


def test_g_function_fixed_point_and_symmetry():
    """g(1) = 2*1^0 = 2 exactly (Bane Eq. 12); g(alpha) = g(1/alpha) is the
    paper's own stated extension of the fit beyond alpha in (0, 1]."""
    assert ibs._g(1.0) == pytest.approx(2.0)
    for alpha in (0.05, 0.2, 0.6, 3.0, 17.0):
        assert ibs._g(alpha) == pytest.approx(ibs._g(1.0 / alpha), rel=1e-12)


def test_dispersion_invariant_matches_hand_computation():
    """H = [D^2 + (beta*D' + alpha*D)^2] / beta -- hand-computed for
    D=0.05, D'=0.02, beta=5.0, alpha=0.3."""
    H = ibs.dispersion_invariant(D_m=0.05, Dp=0.02, beta_m=5.0, alpha=0.3)
    expected = (0.05**2 + (5.0 * 0.02 + 0.3 * 0.05) ** 2) / 5.0
    assert H == pytest.approx(expected, rel=1e-12)
    assert H == pytest.approx(0.003145, rel=1e-6)


def test_zero_dispersion_gives_zero_plane_growth():
    """No dispersion in a plane => H=0 in that plane => that plane's IBS
    growth rate is exactly zero in this formalism (matches Bane's own ATF
    example, which needed *added* vertical dispersion from random errors
    to get any vertical IBS growth at all -- pure betatron coupling isn't
    captured by this smooth-lattice model)."""
    rates = ibs.bane_growth_rates(**BASE_KWARGS)  # D_y_m=0.0 in BASE_KWARGS
    assert rates.inv_Ty_s == 0.0
    assert rates.inv_Tx_s > 0.0
    assert rates.inv_Tp_s > 0.0


def test_growth_rates_scale_linearly_with_bunch_population():
    """Every rate is manifestly linear in N (Eq. 10's prefactor) -- fix the
    Coulomb log explicitly so this isolates that scaling from the (weakly
    N-dependent, through density) auto-computed log."""
    kwargs = dict(BASE_KWARGS, coulomb_log_value=15.0)
    r1 = ibs.bane_growth_rates(**dict(kwargs, n_particles_bunch=1e10))
    r2 = ibs.bane_growth_rates(**dict(kwargs, n_particles_bunch=2e10))
    assert r2.inv_Tp_s == pytest.approx(2 * r1.inv_Tp_s, rel=1e-9)
    assert r2.inv_Tx_s == pytest.approx(2 * r1.inv_Tx_s, rel=1e-9)


def test_growth_rates_symmetric_under_plane_swap():
    """Swapping every (x) <-> (y) input should swap inv_Tx <-> inv_Ty and
    leave inv_Tp unchanged -- the formula treats the two planes
    symmetrically (a/b -> b/a, g(a/b)=g(b/a))."""
    kwargs = dict(BASE_KWARGS, D_y_m=0.02)  # give y-plane nonzero H too
    r = ibs.bane_growth_rates(**kwargs)
    swapped = dict(kwargs)
    swapped["epsx_m"], swapped["epsy_m"] = kwargs["epsy_m"], kwargs["epsx_m"]
    swapped["beta_x_m"], swapped["beta_y_m"] = kwargs["beta_y_m"], kwargs["beta_x_m"]
    swapped["D_x_m"], swapped["D_y_m"] = kwargs["D_y_m"], kwargs["D_x_m"]
    r_swapped = ibs.bane_growth_rates(**swapped)
    assert r_swapped.inv_Tp_s == pytest.approx(r.inv_Tp_s, rel=1e-9)
    assert r_swapped.inv_Tx_s == pytest.approx(r.inv_Ty_s, rel=1e-9)
    assert r_swapped.inv_Ty_s == pytest.approx(r.inv_Tx_s, rel=1e-9)


def test_growth_rates_positive_and_finite():
    rates = ibs.bane_growth_rates(**BASE_KWARGS)
    for rate in (rates.inv_Tp_s, rates.inv_Tx_s, rates.inv_Ty_s):
        assert math.isfinite(rate)
        assert rate >= 0.0


def test_coulomb_log_in_typical_range():
    """Not a precision claim (see ibs.py's docstring on this being a
    standard-practice approximation) -- just a sanity bound that the
    auto-computed log lands in the range essentially every real-machine IBS
    calculation quotes (roughly 5-40)."""
    L = ibs.coulomb_log(
        sigma_x_m=100e-6, sigma_y_m=10e-6, sigma_z_m=3e-3,
        gamma=4305.0, beta_rel=1.0, sigma_p=7.3e-4, n_particles_bunch=5e11,
    )
    assert 5.0 < L < 40.0


# --- Bjorken-Mtingwa (general, no high-energy assumption) ---

_BM_KWARGS = dict(BASE_KWARGS, beta_rel=math.sqrt(1.0 - 1.0 / BASE_KWARGS["gamma"] ** 2),
                   coulomb_log_value=15.0, particle_radius_m=R_E_M)


def test_bjorken_mtingwa_matches_bane_at_high_energy():
    """The whole point of Bane's approximation is that it's the gamma>>1
    limit of this general formula (Kubo, Mtingwa & Wolski, PRSTAB 8, 081001
    (2005), Eq. 10 is explicitly derived from Eq. 4 by a change of
    integration variable) -- at BASE_KWARGS's electron-ring gamma=4305,
    they should agree to within the few-percent level KMW's own Table II
    shows between Bane/CIMP and the full theory (not exactly, since Bane
    makes additional simplifications beyond just gamma>>1, e.g. beta_rel~1
    and further algebraic approximations of its own)."""
    bane = ibs.bane_growth_rates(**dict(BASE_KWARGS, D_y_m=0.02), coulomb_log_value=15.0)
    bm = ibs.bjorken_mtingwa_growth_rates(**dict(_BM_KWARGS, D_y_m=0.02))
    assert bm.inv_Tp_s == pytest.approx(bane.inv_Tp_s, rel=0.15)
    assert bm.inv_Tx_s == pytest.approx(bane.inv_Tx_s, rel=0.15)
    assert bm.inv_Ty_s == pytest.approx(bane.inv_Ty_s, rel=0.15)


def test_bjorken_mtingwa_symmetric_under_plane_swap():
    """Same symmetry property as Bane's (test_growth_rates_symmetric_under_
    plane_swap), checked independently for the general formula -- L's
    construction (KMW Eqs. 6-9) treats h/v identically by index
    relabeling, so a full (eps, beta, D) swap should swap inv_Tx<->inv_Ty
    and leave inv_Tp fixed."""
    kwargs = dict(_BM_KWARGS, D_y_m=0.02)
    r = ibs.bjorken_mtingwa_growth_rates(**kwargs)
    swapped = dict(kwargs)
    swapped["epsx_m"], swapped["epsy_m"] = kwargs["epsy_m"], kwargs["epsx_m"]
    swapped["beta_x_m"], swapped["beta_y_m"] = kwargs["beta_y_m"], kwargs["beta_x_m"]
    swapped["D_x_m"], swapped["D_y_m"] = kwargs["D_y_m"], kwargs["D_x_m"]
    r_swapped = ibs.bjorken_mtingwa_growth_rates(**swapped)
    assert r_swapped.inv_Tp_s == pytest.approx(r.inv_Tp_s, rel=1e-6)
    assert r_swapped.inv_Tx_s == pytest.approx(r.inv_Ty_s, rel=1e-6)
    assert r_swapped.inv_Ty_s == pytest.approx(r.inv_Tx_s, rel=1e-6)


def test_bjorken_mtingwa_can_go_negative_unlike_bane():
    """Bane's formula is a sum of non-negative terms throughout -- it
    cannot represent IBS *shrinking* an emittance (a real effect: the
    ELENA/pyHEADTAIL paper this validates against, IPAC2021 WEPAB215
    Table 2, reports a negative vertical growth rate). The general
    Bjorken-Mtingwa integral has no such structural floor. This doesn't
    assert a specific sign (that depends on the optics point, see
    IBS_PLAN.md's ELENA section) -- just that a negative result is
    reachable at all, at parameters chosen to demonstrate it."""
    kwargs = dict(_BM_KWARGS, beta_x_m=7.0, beta_y_m=3.0, D_x_m=0.0, D_y_m=0.0)
    r = ibs.bjorken_mtingwa_growth_rates(**kwargs)
    assert r.inv_Tx_s < 0.0 or r.inv_Ty_s < 0.0


def test_bjorken_mtingwa_scales_linearly_with_bunch_population():
    r1 = ibs.bjorken_mtingwa_growth_rates(**dict(_BM_KWARGS, n_particles_bunch=1e10))
    r2 = ibs.bjorken_mtingwa_growth_rates(**dict(_BM_KWARGS, n_particles_bunch=2e10))
    assert r2.inv_Tp_s == pytest.approx(2 * r1.inv_Tp_s, rel=1e-6)


# --- element_averaged_growth_rates ---

_ELEMENT_KWARGS = dict(
    gamma=_BM_KWARGS["gamma"], n_particles_bunch=_BM_KWARGS["n_particles_bunch"],
    sigma_z_m=_BM_KWARGS["sigma_z_m"], sigma_p=_BM_KWARGS["sigma_p"],
    epsx_m=_BM_KWARGS["epsx_m"], epsy_m=_BM_KWARGS["epsy_m"],
    coulomb_log_value=_BM_KWARGS["coulomb_log_value"],
)


def test_element_averaged_matches_single_point_for_uniform_table():
    """A TwissTable with every row identical must reproduce the
    single-point formula's result exactly -- the averaging should reduce
    to a no-op in this trivial case, for both models."""
    n_rows = 5
    lengths = np.array([1.3, 0.7, 2.1, 0.4, 5.9])  # deliberately unequal
    twiss = TwissTable(
        s_m=np.cumsum(lengths), length_m=lengths,
        beta_x_m=np.full(n_rows, BASE_KWARGS["beta_x_m"]), alpha_x=np.full(n_rows, BASE_KWARGS["alpha_x"]),
        D_x_m=np.full(n_rows, BASE_KWARGS["D_x_m"]), Dp_x=np.full(n_rows, BASE_KWARGS["Dp_x"]),
        beta_y_m=np.full(n_rows, BASE_KWARGS["beta_y_m"]), alpha_y=np.full(n_rows, BASE_KWARGS["alpha_y"]),
        D_y_m=np.full(n_rows, 0.02), Dp_y=np.full(n_rows, BASE_KWARGS["Dp_y"]),
    )
    single = ibs.bjorken_mtingwa_growth_rates(
        **dict(_BM_KWARGS, D_y_m=0.02),
    )
    averaged = ibs.element_averaged_growth_rates(
        twiss, model="bjorken-mtingwa", beta_rel=_BM_KWARGS["beta_rel"],
        particle_radius_m=_BM_KWARGS["particle_radius_m"], **_ELEMENT_KWARGS,
    )
    assert averaged.inv_Tp_s == pytest.approx(single.inv_Tp_s, rel=1e-9)
    assert averaged.inv_Tx_s == pytest.approx(single.inv_Tx_s, rel=1e-9)
    assert averaged.inv_Ty_s == pytest.approx(single.inv_Ty_s, rel=1e-9)

    single_bane = ibs.bane_growth_rates(**dict(BASE_KWARGS, D_y_m=0.02, coulomb_log_value=15.0))
    averaged_bane = ibs.element_averaged_growth_rates(twiss, model="bane", **_ELEMENT_KWARGS)
    assert averaged_bane.inv_Tp_s == pytest.approx(single_bane.inv_Tp_s, rel=1e-9)
    assert averaged_bane.inv_Tx_s == pytest.approx(single_bane.inv_Tx_s, rel=1e-9)
    assert averaged_bane.inv_Ty_s == pytest.approx(single_bane.inv_Ty_s, rel=1e-9)


def test_element_averaged_is_length_weighted_not_a_plain_mean():
    """Two rows with distinct optics *and* distinct lengths must match a
    hand-computed length-weighted average, not an unweighted one."""
    kwargs_a = dict(BASE_KWARGS, D_y_m=0.02)
    kwargs_b = dict(BASE_KWARGS, D_y_m=0.02, beta_x_m=7.0, D_x_m=0.2, beta_y_m=3.0)
    rate_a = ibs.bane_growth_rates(**kwargs_a, coulomb_log_value=15.0)
    rate_b = ibs.bane_growth_rates(**kwargs_b, coulomb_log_value=15.0)

    length_a, length_b = 1.0, 3.0
    twiss = TwissTable(
        s_m=np.array([length_a, length_a + length_b]), length_m=np.array([length_a, length_b]),
        beta_x_m=np.array([kwargs_a["beta_x_m"], kwargs_b["beta_x_m"]]),
        alpha_x=np.array([kwargs_a["alpha_x"], kwargs_b["alpha_x"]]),
        D_x_m=np.array([kwargs_a["D_x_m"], kwargs_b["D_x_m"]]),
        Dp_x=np.array([kwargs_a["Dp_x"], kwargs_b["Dp_x"]]),
        beta_y_m=np.array([kwargs_a["beta_y_m"], kwargs_b["beta_y_m"]]),
        alpha_y=np.array([kwargs_a["alpha_y"], kwargs_b["alpha_y"]]),
        D_y_m=np.array([kwargs_a["D_y_m"], kwargs_b["D_y_m"]]),
        Dp_y=np.array([kwargs_a["Dp_y"], kwargs_b["Dp_y"]]),
    )
    averaged = ibs.element_averaged_growth_rates(twiss, model="bane", **_ELEMENT_KWARGS)

    total = length_a + length_b
    expected_p = (rate_a.inv_Tp_s * length_a + rate_b.inv_Tp_s * length_b) / total
    expected_x = (rate_a.inv_Tx_s * length_a + rate_b.inv_Tx_s * length_b) / total
    expected_y = (rate_a.inv_Ty_s * length_a + rate_b.inv_Ty_s * length_b) / total
    assert averaged.inv_Tp_s == pytest.approx(expected_p, rel=1e-9)
    assert averaged.inv_Tx_s == pytest.approx(expected_x, rel=1e-9)
    assert averaged.inv_Ty_s == pytest.approx(expected_y, rel=1e-9)
    # Sanity: the weighted average must differ from a plain (unweighted)
    # mean, since rate_a != rate_b and length_a != length_b -- otherwise
    # this test wouldn't actually distinguish weighted from unweighted.
    plain_mean_p = (rate_a.inv_Tp_s + rate_b.inv_Tp_s) / 2.0
    assert averaged.inv_Tp_s != pytest.approx(plain_mean_p, rel=1e-6)


def test_element_averaged_rejects_unknown_model():
    twiss = TwissTable(
        s_m=np.array([1.0]), length_m=np.array([1.0]),
        beta_x_m=np.array([BASE_KWARGS["beta_x_m"]]), alpha_x=np.array([BASE_KWARGS["alpha_x"]]),
        D_x_m=np.array([BASE_KWARGS["D_x_m"]]), Dp_x=np.array([BASE_KWARGS["Dp_x"]]),
        beta_y_m=np.array([BASE_KWARGS["beta_y_m"]]), alpha_y=np.array([BASE_KWARGS["alpha_y"]]),
        D_y_m=np.array([0.0]), Dp_y=np.array([BASE_KWARGS["Dp_y"]]),
    )
    with pytest.raises(ValueError):
        ibs.element_averaged_growth_rates(twiss, model="nope", **_ELEMENT_KWARGS)
    with pytest.raises(ValueError):
        ibs.element_averaged_growth_rates(twiss, model="bjorken-mtingwa", **_ELEMENT_KWARGS)  # missing beta_rel


_ELENA_TWISS_CSV = Path(__file__).resolve().parent.parent / "examples" / "elena" / "elena_twiss.csv"


@pytest.mark.skipif(not _ELENA_TWISS_CSV.exists(), reason="optional ELENA reference table is not present")
def test_element_averaged_matches_xsuite_on_real_elena_lattice():
    """Compare with pinned Xsuite/xfields ``BjorkenMtingwaIBS`` rates on ELENA optics.

    The independent calculation used MAD-X-derived ELENA optics matched to
    WEPAB215's Q1=2.454/Q2=1.416 working point and P0=13.7 MeV/c antiproton
    reference momentum. The rates below are pinned; this test does not rerun
    Xsuite and requires the optional ELENA reference table.

    This is the check that caught a real bug: D_x_m/Dp_x/D_y_m/Dp_y must
    be scaled by beta_rel before use in bjorken_mtingwa_growth_rates (see
    that function's own docstring) -- without it, this test's inv_Tx_s
    comes out +21 instead of ~-0.15, a 140x error, not a rounding one.
    """
    gamma = 1.0001065934205136
    beta_rel = 0.014599752032107737
    twiss = TwissTable.from_csv(_ELENA_TWISS_CSV)
    rates = ibs.element_averaged_growth_rates(
        twiss, model="bjorken-mtingwa", gamma=gamma, beta_rel=beta_rel,
        n_particles_bunch=4.5e6, sigma_z_m=0.3282, sigma_p=1e-4,
        epsx_m=2.5e-6, epsy_m=2.5e-6, coulomb_log_value=12.5, particle_radius_m=R_P_M,
    )
    # xsuite/xfields BjorkenMtingwaIBS.growth_rates(...) on the same lattice/beam:
    # IBSAmplitudeGrowthRates(Kx=-0.14705795553677606, Ky=-0.15234857057484164, Kz=26.075367484038996)
    assert rates.inv_Tp_s == pytest.approx(26.075367484038996, rel=0.03)
    assert rates.inv_Tx_s == pytest.approx(-0.14705795553677606, rel=0.10)
    assert rates.inv_Ty_s == pytest.approx(-0.15234857057484164, rel=0.03)

