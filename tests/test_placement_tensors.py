"""Placed magnets and screen planes reach the tracking step as tensors, so a scan or fit over a solenoid offset or tilt
never recompiles it."""

import numpy as np
import pytest
import torch

from slipstream.tracking.native import injector as injector_module
from slipstream.tracking.native.injector import Injector, Solenoid, SpaceChargeSettings
from slipstream.tracking.native.magnets import (
    hard_quad_element,
    magnet_fields,
    place_as_tensors,
    placement,
    placement3,
    solenoid_element,
)


def bz_file(tmp_path):
    z = np.linspace(0.0, 0.2, 201)
    path = tmp_path / "solenoid.dat"
    np.savetxt(path, np.column_stack([z, np.exp(-(((z - 0.1) / 0.03) ** 2))]))
    return str(path)


def beam(device="cpu", dtype=torch.float64, n=12):
    a = torch.arange(n, device=device, dtype=dtype)
    return dict(
        x=(a - 5.0) * 1e-4, y=(a % 4 - 1.5) * 1e-4, z=a * 1e-5, ux=(a - 5.0) * 1e5, uy=(1.5 - a % 3) * 1e5,
        uz=(a + 10.0) * 1e8, q=torch.full_like(a, -5e-12 / n), emission_time_s=a * 1e-13,
    )  # fmt: skip


def machine(tmp_path, x_m=0.0, yrot_rad=0.0):
    solenoid = Solenoid(bz_file(tmp_path), 0.0, 0.3, x_m=x_m, yrot_rad=yrot_rad)
    return Injector("", 0.2, solenoids=[solenoid], absolute_phases=True, dt_s=2e-12, base_dir=tmp_path,
                    space_charge=SpaceChargeSettings(enabled=False))  # fmt: skip


def test_place_as_tensors_keeps_every_number_and_none():
    assert place_as_tensors(None, "cpu", torch.float64) is None
    flat = place_as_tensors(placement(0.0005, 0.3, 0.002), "cpu", torch.float64)
    nested = place_as_tensors(placement3(0.0005, -0.0003, 0.1, 0.002, -0.001), "cpu", torch.float64)
    assert [float(v) for v in flat] == list(placement(0.0005, 0.3, 0.002))
    assert [float(v) for v in nested[:3]] == [0.0005, -0.0003, 0.1]
    assert [float(v) for v in nested[3]] == list(placement3(0.0005, -0.0003, 0.1, 0.002, -0.001)[3])
    assert all(v.dim() == 0 and v.dtype == torch.float64 for v in (*flat, *nested[:3], *nested[3]))


def test_magnet_fields_are_unchanged_by_tensor_placements():
    torch.manual_seed(1)
    x, y, z = (torch.randn(500, dtype=torch.float64) * s for s in (3e-3, 3e-3, 0.05))
    z = z + 0.1
    table = torch.exp(-(((torch.linspace(0.0, 0.2, 401) - 0.1) / 0.03) ** 2)).to(torch.float64)
    spline = (table, torch.gradient(table, spacing=0.0005)[0], torch.zeros_like(table), 0.0, 0.0005, 401, 0.2)
    for place3, place2 in ((placement3(0.0006, -0.0004, 0.1, 0.002, -0.0015), placement(0.0007, 0.1, 0.003)),
                           (placement3(0.0, 0.0, 0.1, 0.0, 0.0, force=True), placement(0.0, 0.1, 0.0, force=True))):  # fmt: skip

        def elements(wrap):
            return (solenoid_element(*spline, wrap(place3)), hard_quad_element(2.0, 0.05, 0.15, wrap(place2)))

        floats = magnet_fields(x, y, z, elements(lambda p: p))
        tensors = magnet_fields(x, y, z, elements(lambda p: place_as_tensors(p, "cpu", torch.float64)))
        for a, b in zip(floats, tensors):
            assert torch.equal(a, b) and a.abs().max() > 0.0


def test_a_placed_solenoid_tracks_identically_with_tensor_placements_and_is_really_placed(tmp_path, monkeypatch):
    def track(m):
        return m.track_differentiable(
            beam=beam(), device="cpu", dtype=torch.float64, n_steps=400, return_particles=True
        )

    placed = machine(tmp_path, x_m=0.6e-3, yrot_rad=1.5e-3)
    tensors = track(placed)
    monkeypatch.setattr(injector_module, "place_as_tensors", lambda place, device, dtype: place)
    floats = track(placed)
    monkeypatch.undo()
    for key in ("x", "y", "z", "ux", "uy", "uz"):
        assert torch.equal(tensors[key], floats[key])
    aligned = track(machine(tmp_path))
    assert (tensors["x"] - aligned["x"]).abs().max() > 1e-6  # an offset and a tilt kick the beam


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware required")
def test_a_new_solenoid_placement_or_screen_plane_reuses_the_compiled_cuda_step(tmp_path, monkeypatch):
    import torch._dynamo.utils as dynamo_utils

    monkeypatch.setattr(Injector, "initial_beam", lambda *a, **k: beam("cuda", torch.float32))
    monkeypatch.setattr(Injector, "_reference_particle", lambda self: (0.0, 0.0, 0.0))
    graphs = []
    for x_m, yrot, plane in ((0.2e-3, 0.5e-3, 0.05), (0.6e-3, -1.0e-3, 0.08), (-0.4e-3, 2.0e-3, 0.11)):
        track = machine(tmp_path, x_m, yrot).track(
            device="cuda", dtype=torch.float32, use_compile=True, use_warp=False, verbose=False, record_every=0,
            screen_z_m=plane,
        )  # fmt: skip
        assert track["screen"]["n_crossed"] > 0
        graphs.append(dynamo_utils.counters["stats"]["unique_graphs"])
    assert graphs[0] == graphs[1] == graphs[2]  # as Python floats each new value was a constant and recompiled
