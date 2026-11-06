"""The injector's steering correctors: uniform transverse field regions that kick an electron by q B L / p."""

import pytest
import torch

from slipstream.tracking.native.injector import Corrector, Injector, SpaceChargeSettings
from slipstream.tracking.native.magnets import corrector_element, magnet_fields

C = 299792458.0


def tracers(uz, device="cpu", dtype=torch.float64):
    n = len(uz)
    zeros = torch.zeros(n, device=device, dtype=dtype)
    return dict(x=zeros.clone(), y=zeros.clone(), z=zeros.clone(), ux=zeros.clone(), uy=zeros.clone(),
                uz=torch.tensor(uz, device=device, dtype=dtype), q=torch.full((n,), -1e-15, device=device, dtype=dtype),
                emission_time_s=zeros.clone())  # fmt: skip


def machine(corrector, z_stop=0.25):
    return Injector("", z_stop, correctors=[corrector], absolute_phases=True, dt_s=2e-12,
                    space_charge=SpaceChargeSettings(enabled=False))  # fmt: skip


def test_corrector_field_is_uniform_inside_its_length_and_zero_outside():
    z = torch.tensor([0.0, 0.0399, 0.04, 0.05, 0.06, 0.0601, 0.2], dtype=torch.float64)
    x = torch.zeros_like(z)
    bx, by, bz = magnet_fields(
        x,
        x,
        z,
        (
            corrector_element(
                torch.tensor(2e-3, dtype=torch.float64), torch.tensor(-5e-4, dtype=torch.float64), 0.04, 0.06
            ),
        ),
    )
    assert bx.tolist() == [0.0, 0.0, 2e-3, 2e-3, 2e-3, 0.0, 0.0]
    assert by.tolist() == [0.0, 0.0, -5e-4, -5e-4, -5e-4, 0.0, 0.0]
    assert bz.abs().max() == 0.0


@pytest.mark.parametrize("plane", ["by_T", "bx_T"])
def test_corrector_deflects_an_electron_by_q_b_l_over_p_whatever_its_energy(plane):
    uz = [1.5e8, 2.5e8, 4.0e8]  # gamma beta c of 0.4, 0.9 and 1.7 MeV electrons or so
    m = machine(Corrector(0.05, 0.02, **{plane: 1.0e-3}))
    out = m.track_differentiable(
        beam=tracers(uz), device="cpu", dtype=torch.float64, n_steps=700, return_particles=True
    )
    # Delta p_x = -q B_y L and Delta p_y = +q B_x L for a charge q moving along z; the angle is that over p = m u
    sign, axis, other = (-1.0, "x", "y") if plane == "by_T" else (1.0, "y", "x")
    angle = sign * m.q_C * 1.0e-3 * 0.02 / (m.mass_kg * torch.tensor(uz, dtype=torch.float64))
    expected = angle * (out["z"] - 0.05)  # a thin magnet: straight lines from its centre
    torch.testing.assert_close(out[axis], expected, rtol=2e-2, atol=0.0)
    assert (out[other].abs() < 1e-9).all() and (out[axis].abs() > 1e-4).all()
    assert out[axis][0].abs() > out[axis][2].abs()  # the lower the energy, the larger the deflection


def test_a_corrector_survives_the_yaml_round_trip(tmp_path):
    m = machine(Corrector(0.335, 0.03, bx_T=1e-4, by_T=-2e-4))
    m.save(tmp_path / "gun.yaml")
    assert Injector.load(tmp_path / "gun.yaml").correctors == m.correctors


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware required")
def test_changing_a_corrector_strength_reuses_the_compiled_cuda_step(monkeypatch):
    import torch._dynamo.utils as dynamo_utils

    monkeypatch.setattr(Injector, "initial_beam", lambda *a, **k: tracers([2e8] * 8, "cuda", torch.float32))
    monkeypatch.setattr(Injector, "_reference_particle", lambda self: (0.0, 0.0, 0.0))
    graphs = []
    for by in (1e-3, -2e-3, 5e-4):
        track = machine(Corrector(0.05, 0.02, by_T=by), 0.15).track(
            device="cuda", dtype=torch.float32, use_compile=True, use_warp=False, verbose=False, record_every=0,
            screen_z_m=0.12,
        )  # fmt: skip
        assert track["screen"]["n_crossed"] == 8
        graphs.append(dynamo_utils.counters["stats"]["unique_graphs"])
    assert graphs[0] == graphs[1] == graphs[2]
