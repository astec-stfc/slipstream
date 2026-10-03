"""Tests for slipstream.wakefields.shortrange (bin-and-convolve short-range
wake, and pseudo-Green wake derivation from an impedance)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from slipstream.wakefields.impedance import Impedance
from slipstream.wakefields.shortrange import ShortRangeWake, trim_wake, wake_potential_from_impedance


def test_short_range_wake_rejects_nonuniform_grid():
    with pytest.raises(ValueError):
        ShortRangeWake(t_s=np.array([0.0, 1.0, 3.0]), W_V_per_C=np.array([1.0, 2.0, 3.0]))


def test_wake_voltage_causal_kernel():
    """W = unit impulse one bin after zero lag: a trailing particle should
    feel the wake of an earlier one; the earlier particle should feel nothing
    (nothing precedes it)."""
    wake = ShortRangeWake(t_s=np.array([0.0, 1.0, 2.0]), W_V_per_C=np.array([0.0, 5.0, 0.0]))
    tau = torch.tensor([0.0, 1.0], dtype=torch.float64)  # bins 0 and 1, dt=1
    V = wake.voltage_at(tau, charge_per_particle_C=2.0)
    assert V[0].item() == pytest.approx(0.0)
    assert V[1].item() == pytest.approx(10.0)  # charge(bin0)=2 * W[lag=1]=5


def test_wake_voltage_acausal_precursor():
    """Same as above but the wake table also carries a small acausal
    precursor (as a real Gaussian-probe pseudo-Green wake does): a leading
    particle should feel a contribution from a trailing one, at the
    precursor lag."""
    wake = ShortRangeWake(
        t_s=np.array([-1.0, 0.0, 1.0, 2.0]), W_V_per_C=np.array([7.0, 0.0, 5.0, 0.0])
    )
    tau = torch.tensor([0.0, 1.0], dtype=torch.float64)  # bins 0 and 1, dt=1
    V = wake.voltage_at(tau, charge_per_particle_C=2.0)
    assert V[0].item() == pytest.approx(14.0)  # charge(bin1)=2 * W[lag=-1]=7
    assert V[1].item() == pytest.approx(10.0)  # charge(bin0)=2 * W[lag=1]=5


def test_truncated_keeps_only_requested_range():
    wake = ShortRangeWake(t_s=np.arange(-5.0, 5.0), W_V_per_C=np.arange(10.0))
    short = wake.truncated(-2.0, 2.0)
    assert short.t_s.tolist() == [-2.0, -1.0, 0.0, 1.0, 2.0]


def test_fixed_bin_window_matches_dynamic_window():
    """When the fixed window comfortably covers the particles, the result
    should be identical to the dynamic (auto-ranging) path."""
    wake_dynamic = ShortRangeWake(
        t_s=np.array([-1.0, 0.0, 1.0, 2.0]), W_V_per_C=np.array([7.0, 0.0, 5.0, 0.0])
    )
    wake_fixed = ShortRangeWake(
        t_s=np.array([-1.0, 0.0, 1.0, 2.0]), W_V_per_C=np.array([7.0, 0.0, 5.0, 0.0])
    )
    wake_fixed.set_fixed_bin_window(-10.0, 10.0)
    wake_fixed.prepare(torch.float64, "cpu")
    tau = torch.tensor([0.0, 1.0], dtype=torch.float64)
    V_dyn = wake_dynamic.voltage_at(tau, charge_per_particle_C=2.0)
    V_fixed = wake_fixed.voltage_at(tau, charge_per_particle_C=2.0)
    assert torch.allclose(V_dyn, V_fixed)


def test_fixed_bin_window_drops_out_of_range_from_histogram():
    """An out-of-range particle must not pollute the shared charge
    histogram -- clamping it *into* the histogram (instead of just reading
    a voltage back for it at the edge bin) piled artificial charge onto the
    edge and touched off a numerical runaway during a real batch scan (see
    NOTES.md). It should still get a finite voltage back, from the edge
    bin, without inflating what nearby in-range particles feel."""
    wake = ShortRangeWake(t_s=np.array([0.0, 1.0, 2.0]), W_V_per_C=np.array([3.0, 5.0, 0.0]))
    wake.set_fixed_bin_window(0.0, 3.0)  # bins 0,1,2
    wake.prepare(torch.float64, "cpu")
    tau = torch.tensor([1.0, 100.0], dtype=torch.float64)  # bin 1, and way outside
    V = wake.voltage_at(tau, charge_per_particle_C=2.0)
    assert torch.isfinite(V).all()
    assert V[0].item() == pytest.approx(6.0)  # unaffected by the out-of-range particle
    assert V[1].item() == pytest.approx(10.0)  # edge-bin readback, not inflated by its own dropped charge


def test_sdds_round_trip(tmp_path):
    wake = ShortRangeWake(t_s=np.array([-1.0, 0.0, 1.0, 2.0]), W_V_per_C=np.array([7.0, 0.0, 5.0, 0.0]))
    path = tmp_path / "wake.sdds"
    wake.to_sdds_file(path)
    loaded = ShortRangeWake.from_sdds_file(path)
    assert np.allclose(loaded.t_s, wake.t_s)
    assert np.allclose(loaded.W_V_per_C, wake.W_V_per_C)


def test_from_sdds_file_custom_column_names(tmp_path):
    """A real elegant ``RFCW`` wake file has free-form column names
    (``TCOLUMN=``/``WZCOLUMN=``, no built-in ``"t"``/``"W"`` default --
    see ``plans/NATIVE_LINAC_WAKE_PLAN.md``) -- ``t_column=``/``w_column=``
    must read an arbitrarily-named pair, not just the default."""
    from slipstream.sdds_io import sdds_object

    path = tmp_path / "wake.sdds"
    with sdds_object() as (sdds, x):
        x.mode = sdds.SDDS_ASCII
        x.defineColumn("time", units="s", type=sdds.SDDS_DOUBLE)
        x.defineColumn("Wz", units="V/C", type=sdds.SDDS_DOUBLE)
        x.setColumnValueList("time", [-1.0, 0.0, 1.0, 2.0])
        x.setColumnValueList("Wz", [7.0, 0.0, 5.0, 0.0])
        x.save(str(path))
    loaded = ShortRangeWake.from_sdds_file(path, t_column="time", w_column="Wz")
    assert np.allclose(loaded.t_s, [-1.0, 0.0, 1.0, 2.0])
    assert np.allclose(loaded.W_V_per_C, [7.0, 0.0, 5.0, 0.0])


def test_wake_potential_from_impedance_is_peaked_near_zero_lag():
    """A single Lorentzian-ish resonant impedance line should produce a
    wake potential that peaks close to t=0 (a Gaussian probe bunch mostly
    samples the wake right where it arrives) and decays away from it."""
    freq = np.linspace(0, 5e9, 20000)
    f0, width = 1e9, 5e7
    ReZ = 100.0 / (1 + ((freq - f0) / width) ** 2)
    ImZ = np.zeros_like(freq)
    impedance = Impedance(freq, ReZ, ImZ)
    t, W = wake_potential_from_impedance(impedance, sigma_s_m=1e-3, dt=1e-12, n_points=1 << 12)
    peak_idx = np.argmax(np.abs(W))
    assert abs(t[peak_idx]) < 5e-9  # peak lands close to t=0, not off in the tails

    t_trim, W_trim = trim_wake(t, W, -2e-9, 2e-9)
    assert t_trim.min() >= -2e-9 and t_trim.max() <= 2e-9
    assert len(W_trim) == len(t_trim)

