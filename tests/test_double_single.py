"""Double-single (float32 hi + lo pairs, ~44-bit) arithmetic and element tracking against float64.

The reference is float64 ``track_element``; the point of the representation is that a float32 GPU reproduces it to ~1e-15
where plain float32 is off by 1e-9 to 1e-8 (the sub-ulp kicks of a ring, summed over many turns).
"""
import pytest
import torch

from slipstream.tracking.native.double_single import (
    DS, check_error_free, ds_asin_small, ds_sqrt, ds_tan_small, track_element_ds,
)
from slipstream.tracking.native.elements import _DRIFT_FRAC6, _KICK_FRAC6, track_element

F64 = torch.float64
ENERGY_EV, MC2_EV = 3e9, 0.51099895e6
STATE = torch.tensor([[1e-3, 1e-4, -5e-4, 2e-5, 0.0, 1e-3], [-2e-3, 3e-4, 1e-4, -1e-5, 0.0, -2e-3]], dtype=F64)
PLANE = torch.tensor([True, True, True, True, False, True])  # tau is not tracked in double-single


def test_double_single_arithmetic_is_float64_accurate_for_the_basic_operations():
    assert check_error_free() < 1e-12  # + * / sqrt, constants and subtraction against float64, worst relative error


def test_a_double_single_pair_keeps_what_float32_cannot_hold():
    x = torch.tensor([1.0 + 1e-12, 0.1234567890123, -3.3e-3], dtype=F64)
    pair = DS.of(x)
    assert ((pair.double() - x).abs() / x.abs()).max().item() < 2.0**-45
    assert pair.hi.dtype == torch.float32 and pair.lo.dtype == torch.float32
    assert (pair.double()[0] - 1.0).item() == pytest.approx(1e-12, rel=1e-4)  # the increment float32 rounds to zero
    assert x.float().double()[0].item() == 1.0


def test_square_root_division_and_small_angle_functions_match_float64():
    a = torch.tensor([2.0, 0.3, 17.0], dtype=F64)
    assert ((ds_sqrt(DS.of(a)).double() - a.sqrt()).abs() / a.sqrt()).max().item() < 1e-13
    assert ((1.0 / DS.of(a)).double() - 1.0 / a).abs().max().item() < 1e-14
    u = torch.tensor([1e-3, -0.02, 0.04], dtype=F64)  # the ranges the exact dipole uses: |u| below ~0.05
    assert ((ds_asin_small(DS.of(u)).double() - torch.asin(u)).abs() / torch.asin(u).abs()).max().item() < 1e-12
    assert ((ds_tan_small(DS.of(u)).double() - torch.tan(u)).abs() / torch.tan(u).abs()).max().item() < 1e-12


def packed(state):
    pair = DS.of(state)
    return torch.cat([pair.hi, pair.lo], dim=-1)


def unpacked(s12):
    columns = [s12[..., i].double() + s12[..., 6 + i].double() if i != 4 else s12[..., 4].double() for i in range(6)]
    return torch.stack(columns, dim=-1)


@pytest.mark.parametrize("name,element", [
    ("drift", dict(length=1.0, angle=0.0, k1=0.0, k2=0.0, e1=0.0, e2=0.0)),
    ("quadrupole", dict(length=0.3, angle=0.0, k1=1.7, k2=0.0, e1=0.0, e2=0.0)),
    ("sextupole", dict(length=0.2, angle=0.0, k1=0.0, k2=40.0, e1=0.0, e2=0.0)),
    ("dipole with edges", dict(length=1.0, angle=0.2, k1=0.0, k2=0.0, e1=0.05, e2=0.05)),
    ("combined function bend", dict(length=1.0, angle=0.2, k1=0.8, k2=0.0, e1=0.05, e2=0.05)),
])
def test_double_single_tracking_reproduces_float64_where_float32_does_not(name, element):
    n_kicks = 8
    reference = track_element(STATE, energy_eV=ENERGY_EV, mc2_eV=MC2_EV, n_kicks=n_kicks, integration_order=6, exact_dipole=True,
                              **element)
    double_single = unpacked(track_element_ds(packed(STATE), element["length"], element["angle"], element["k1"], element["k2"],
                                              element["e1"], element["e2"], n_kicks, _DRIFT_FRAC6, _KICK_FRAC6))
    single = track_element(STATE.float(), energy_eV=ENERGY_EV, mc2_eV=MC2_EV, n_kicks=n_kicks, integration_order=6, exact_dipole=True,
                           **element).double()
    error_ds = (double_single - reference)[:, PLANE].abs().max().item()
    error_f32 = (single - reference)[:, PLANE].abs().max().item()
    assert error_ds < 5e-15
    assert error_ds < 1e-3 * error_f32
    assert torch.equal(double_single[:, 4], STATE[:, 4])  # tau passes through untouched
