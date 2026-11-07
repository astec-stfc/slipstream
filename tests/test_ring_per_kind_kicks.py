"""Per-kind kick counts (``n_kicks_quad`` / ``n_kicks_sext``) for ``RingMap``."""
import pytest
import torch

from slipstream.tracking.native.lattice_import import ElementSpec, slice_count
from slipstream.tracking.native.optics import periodic_twiss_table
from slipstream.tracking.native.ring import RingMap

CELL = [
    ElementSpec("QF", "quad", 0.2, k1=1.5), ElementSpec("D1", "drift", 1.0),
    ElementSpec("B", "bend", 0.5, angle=0.05), ElementSpec("QD", "quad", 0.2, k1=-1.35),
    ElementSpec("S", "sext", 0.1, k2=300.0), ElementSpec("D2", "drift", 1.0),
]


def ring(**kw):
    return RingMap(CELL * 2, 3e9, device="cpu", dtype=torch.float64, **kw)


def test_unset_per_kind_counts_change_nothing():
    state = torch.tensor([[2e-3, 1e-5, 1e-3, 0.0, 0.0, 0.0]], dtype=torch.float64)
    base = ring(n_kicks=3).turn(state)
    explicit = ring(n_kicks=3, n_kicks_quad=None, n_kicks_sext=None).turn(state)
    assert torch.equal(base, explicit)
    assert torch.equal(ring(n_kicks=3).turn(state), ring(n_kicks=3, n_kicks_quad=3, n_kicks_sext=3).turn(state))


def test_counts_land_on_the_right_kinds():
    r = ring(n_kicks=3, n_kicks_bend=5, n_kicks_quad=7, n_kicks_sext=1)
    assert [p.n_kicks for p in r.params[:6]] == [7, 3, 5, 7, 1, 3]
    assert [slice_count(s, 3, 5, 7, 1) for s in CELL] == [7, 3, 5, 7, 1, 3]
    assert [slice_count(s, 3, 5) for s in CELL] == [3, 3, 5, 3, 3, 3]
    only_sext = ring(n_kicks=3, n_kicks_sext=1)
    assert [p.n_kicks for p in only_sext.params[:6]] == [3, 3, 8, 3, 1, 3]


def test_float64_reference_and_strength_copies_keep_the_split():
    r = ring(n_kicks=3, n_kicks_quad=7, n_kicks_sext=1)
    counts = [p.n_kicks for p in r.params]
    assert [p.n_kicks for p in r.float64_reference().params] == counts
    k = torch.tensor(1.4, dtype=torch.float64)
    assert [p.n_kicks for p in r.with_strengths(k1={"QF": k}).params] == counts


def test_single_sextupole_kick_with_many_quad_kicks_is_accurate():
    fodo = [s for s in CELL if s.kind != "bend"]
    fodo[3] = ElementSpec("S", "sext", 0.1, k2=30.0)
    state = torch.tensor([[3e-3, 0.0, 1.5e-3, 0.0, 0.0, 0.0]], dtype=torch.float64)

    def final(**kw):
        return RingMap(fodo * 3, 3e9, device="cpu", dtype=torch.float64, **kw).track(state, 30).state

    ref = final(n_kicks=24)
    assert ref.abs().max() > 0  # survived: the comparison below is not against a zeroed lost particle

    def err(**kw):
        return (final(**kw) - ref).abs().max().item()

    coarse, uniform = err(n_kicks=1), err(n_kicks=12)
    split = err(n_kicks=1, n_kicks_quad=12, n_kicks_sext=1)
    assert coarse > 1e-7
    assert split < coarse * 1e-3
    assert split < 10 * uniform + 1e-12  # one sextupole kick costs no accuracy at 12 quadrupole kicks


@pytest.mark.parametrize("kwargs", [
    {"n_kicks_quad": 0}, {"n_kicks_sext": -1}, {"n_kicks_quad": 2.0}, {"n_kicks_sext": True},
])
def test_invalid_per_kind_counts_raise(kwargs):
    with pytest.raises(ValueError, match="n_kicks_(quad|sext)"):
        ring(**kwargs)


def test_periodic_twiss_table_takes_per_kind_counts():
    specs = CELL * 2
    uniform = periodic_twiss_table(specs, 3e9, 0.51099895e6, n_kicks=8, n_kicks_bend=8)
    split = periodic_twiss_table(specs, 3e9, 0.51099895e6, n_kicks=1, n_kicks_bend=8, n_kicks_quad=8, n_kicks_sext=1)
    assert split[1:] == pytest.approx(uniform[1:], abs=1e-12)  # sextupole slicing does not enter the linear map
    coarse = periodic_twiss_table(specs, 3e9, 0.51099895e6, n_kicks=1, n_kicks_bend=8)
    assert abs(coarse[1] - uniform[1]) > 1e-9
