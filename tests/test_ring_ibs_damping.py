"""IBS inside ``NativeTracker``: growth rates fold into the damping times, quantum excitation stays radiation-only.

Per turn ``delta' = m delta + noise`` with ``m = 1 - T0/tau_eff`` (``tau_eff`` includes IBS) and noise variance
``q = 2 (T0/tau) sigma0^2`` (radiation only, ``tau`` the radiation damping time). From ``Var0 = sigma0^2`` the exact moment is
``Var_N = m^(2N) (Var0 - V) + V`` with ``V = q / (1 - m^2)``: a wider equilibrium ``sigma0^2 tau_eff / tau`` when IBS growth
is below the damping rate, and exponential growth when it is above (``tau_eff < 0``). The ramped path must agree with the
fixed-energy one for a constant ramp.
"""
import numpy as np
import pytest
import torch

from slipstream import ElementSpec, Lattice, NativeTracker
from slipstream.ibs import IBSGrowthRates
from slipstream.lattice.optics import TransverseOptics
from slipstream.lattice.ramp import RampProfile

F64 = torch.float64
SIGMA0, N, A = 1e-3, 4000, 0.01  # A = T0 / tau, the radiation damping per turn
LENGTH_M = 2.4
SPECS = [ElementSpec("QF", "quad", .2, k1=1.), ElementSpec("D1", "drift", 1.), ElementSpec("QD", "quad", .2, k1=-1.),
         ElementSpec("D2", "drift", 1.)]
T0_S = Lattice(LENGTH_M, 1e9, .01, 1.0, 5e8, 0., 1.0, SIGMA0).T0_s
TAU_S = T0_S / A
PARAMS = Lattice(LENGTH_M, 1e9, .01, 1.0, 5e8, 0., TAU_S, SIGMA0)  # a 1 V cavity: synchrotron motion negligible
OPTICS = TransverseOptics(beta_x_m=8.0, alpha_x=0.0, D_x_m=0.0, Q_x=0.2, Qp_x=0., tau_x_s=TAU_S, epsx_m=1e-9,
                          beta_y_m=8.0, alpha_y=0.0, D_y_m=0.0, Q_y=0.2, Qp_y=0., tau_y_s=TAU_S, epsy_m=1e-9)


def track(growth_over_damping, turns, ramped):
    """Track a Gaussian energy distribution with IBS rates injected as ``growth_over_damping / tau`` in all three planes."""
    ramp = RampProfile(time_s=np.array([0., 1.]), E0_eV=np.array([1e9, 1e9]), Vrf_V=np.array([1., 1.]),
                       f_rf_Hz=np.array([5e8, 5e8]), phi_rf_rad=np.array([PARAMS.phi_s, PARAMS.phi_s])) if ramped else None
    tracker = NativeTracker(SPECS, PARAMS, optics=OPTICS, ramp=ramp, ibs=True, bunch_current_A=1e-3, device="cpu",
                            generator=torch.Generator().manual_seed(1))
    rate = growth_over_damping / TAU_S
    rates = IBSGrowthRates(inv_Tp_s=rate, inv_Tx_s=rate / 2, inv_Ty_s=rate / 2)  # transverse enters as 2 / Tx
    tracker._recompute_ibs_rates = lambda *args, **kwargs: setattr(tracker, "_ibs_rates", rates)
    state = torch.zeros(N, 6, dtype=F64)
    state[:, 5] = SIGMA0 * torch.randn(N, dtype=F64, generator=torch.Generator().manual_seed(2))
    out, _ = tracker.track(state, turns, sample_every=turns)
    return out


def exact_variance(growth_over_damping, turns, initial_variance):
    m = 1 - A * (1 - growth_over_damping)  # 1 / tau_eff = (1 - g) / tau
    q = 2 * A * SIGMA0**2
    fixed_point = q / (1 - m**2)
    return m ** (2 * turns) * (initial_variance - fixed_point) + fixed_point


@pytest.mark.parametrize("ramped", [False, True])
def test_ibs_widens_the_equilibrium_energy_spread_by_the_inverse_square_root_of_one_minus_growth_over_damping(ramped):
    out = track(0.5, 400, ramped)
    assert out[:, 5].var().item() == pytest.approx(exact_variance(0.5, 400, SIGMA0**2), rel=0.06)
    assert out[:, 5].std().item() / SIGMA0 == pytest.approx(2**0.5, rel=0.04)  # sqrt(tau_eff / tau) at equilibrium


@pytest.mark.parametrize("growth_over_damping", [0.5, 2.0])
def test_a_constant_ramp_reproduces_fixed_energy_tracking_even_when_ibs_exceeds_damping(growth_over_damping):
    turns = 400 if growth_over_damping < 1 else 150  # above damping the spread grows as (1 + A)^turns
    fixed, ramped = track(growth_over_damping, turns, False), track(growth_over_damping, turns, True)
    assert torch.isfinite(ramped).all()  # tau_eff < 0 must not put a negative number under a square root
    torch.testing.assert_close(ramped[:, [1, 3, 5]], fixed[:, [1, 3, 5]], rtol=1e-8, atol=1e-14)
    if growth_over_damping > 1:
        assert ramped[:, 5].var().item() == pytest.approx(exact_variance(growth_over_damping, turns, SIGMA0**2), rel=0.06)
