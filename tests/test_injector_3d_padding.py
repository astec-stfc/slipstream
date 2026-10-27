"""Cartesian injector box-padding controls and default compatibility."""
import math

import pytest
import torch

from slipstream.tracking.native.injector import Injector, SpaceChargeSettings, _compute_space_charge_3d


def compute(sc):
    x = torch.tensor([-2e-3, -1e-3, 1e-3, 2e-3], dtype=torch.float64)
    y = torch.tensor([-1e-3, 1e-3, -1e-3, 1e-3], dtype=torch.float64)
    z = torch.tensor([-3e-3, -1e-3, 1e-3, 3e-3], dtype=torch.float64)
    q = torch.full_like(x, 1e-12)
    zero = torch.zeros_like(x)
    cache = {}
    result = _compute_space_charge_3d(x, y, z, q, zero, zero, zero, 'cpu', torch.float64,
                                      cache, sc, 0.0)
    return result, cache['solver3d'][0]


def test_cartesian_padding_default_and_larger_box(tmp_path):
    default = SpaceChargeSettings(geometry='3d', cathode_image=False, n_xyz=[8, 8, 8])
    explicit = SpaceChargeSettings(geometry='3d', cathode_image=False, n_xyz=[8, 8, 8], box_padding_3d=.05)
    larger = SpaceChargeSettings(geometry='3d', cathode_image=False, n_xyz=[8, 8, 8], box_padding_3d=.2)
    a, sa = compute(default)
    b, sb = compute(explicit)
    c, sc = compute(larger)
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert (sa.x_lo, sa.hx) == (sb.x_lo, sb.hx)
    assert math.isclose(sc.x_lo, -2e-3 - .2 * 4e-3 - 1e-9, abs_tol=1e-15)
    assert math.isclose(sc.hx, (4e-3 * 1.4 + 2e-9) / 7, rel_tol=1e-14)
    assert all(torch.isfinite(v).all() for v in c)
    inj = Injector(distribution='beam.ini', z_stop_m=1.0, space_charge=larger)
    path = tmp_path / 'machine.yaml'
    inj.save(path)
    assert Injector.load(path).space_charge.box_padding_3d == .2


@pytest.mark.parametrize('padding', [-.01, float('inf'), float('nan')])
def test_cartesian_padding_rejects_invalid_values(padding):
    sc = SpaceChargeSettings(geometry='3d', cathode_image=False, n_xyz=[8, 8, 8], box_padding_3d=padding)
    with pytest.raises(ValueError, match='box_padding_3d'):
        compute(sc)
