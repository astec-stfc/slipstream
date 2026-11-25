"""Ring impedance built from named materials and chamber sections (``lattice.apertures``).

The impedance of a resistive pipe is proportional to its length, so a ring cut into sections of one kind equals one pipe of
the whole circumference; a flat chamber's longitudinal impedance is that of the round pipe of its smaller half-aperture
(the Yokoya longitudinal factor is one).
"""
import numpy as np
import pytest

from slipstream.lattice.apertures import Aperture, Apertures, Material
from slipstream.wakefields.chambers import RoundChamber
from slipstream.wakefields.multilayer.round import round_chamber_impedance

CIRCUMFERENCE_M = 100.0
GAMMA = 6000.0
FREQ = np.array([1e8, 1e9, 1e10])
COPPER = Material(rho=1.7e-8, tau_ps=0.0, thickness_m=1.0)


def ring(**apertures):
    default = Aperture(material="copper", shape="round", radius_m=12e-3)
    return Apertures({"copper": COPPER, "steel": Material(rho=7.2e-7, tau_ps=0.0, thickness_m=1.0)},
                     {"Default": default, **apertures})


def as_complex(impedance):
    return impedance.ReZ_ohm + 1j * impedance.ImZ_ohm


def test_material_is_a_complete_leaf_or_a_composite_but_not_both():
    with pytest.raises(ValueError, match="not both"):
        Material(rho=1e-8, tau_ps=0.0, thickness_m=1e-3, layers=["copper"])
    with pytest.raises(ValueError, match="all set"):
        Material(rho=1e-8)
    with pytest.raises(ValueError, match="either"):
        Material()


def test_aperture_shapes_need_their_dimensions():
    with pytest.raises(ValueError, match="shape"):
        Aperture(material="copper", shape="square")
    with pytest.raises(ValueError, match="radius_m"):
        Aperture(material="copper", shape="round")
    with pytest.raises(ValueError, match="horizontal_half_aperture_m"):
        Aperture(material="copper", shape="flat", horizontal_half_aperture_m=0.02)


def test_the_default_aperture_is_required_derives_its_length_and_names_only_real_override_materials():
    with pytest.raises(ValueError, match="Default"):
        Apertures({"copper": COPPER}, {"A": Aperture(material="copper", shape="round", radius_m=1e-2, length_m=1.0)})
    with pytest.raises(ValueError, match="derived"):
        Apertures({"copper": COPPER}, {"Default": Aperture(material="copper", shape="round", radius_m=1e-2, length_m=5.0)})
    with pytest.raises(ValueError, match="typo"):
        ring(Bad=Aperture(material="copper", shape="round", radius_m=1e-2, length_m=1.0, material_overrides={"coper": {"rho": 1.0}}))


def test_default_length_is_what_the_named_sections_leave_of_the_circumference():
    sections = ring(A=Aperture(material="copper", shape="round", radius_m=5e-3, length_m=2.0, count=10),
                    B=Aperture(material="steel", shape="round", radius_m=8e-3, length_m=0.5, count=4))
    assert sections.total_length_m("A", CIRCUMFERENCE_M) == 20.0
    assert sections.default_length_m(CIRCUMFERENCE_M) == pytest.approx(CIRCUMFERENCE_M - 22.0)
    assert sections.total_length_m("Default", CIRCUMFERENCE_M) == pytest.approx(78.0)
    with pytest.raises(ValueError, match="exceeding"):
        sections.default_length_m(20.0)


def test_composite_materials_expand_innermost_first_and_overrides_replace_only_the_named_fields():
    materials = {
        "NEG": Material(rho=1e-6, tau_ps=0.0, thickness_m=1e-6),
        "copper": COPPER,
        "coated": Material(layers=["NEG", "copper"]),
    }
    override = {"NEG": {"thickness_m": 2e-6}}
    sections = Apertures(materials, {
        "Default": Aperture(material="coated", shape="round", radius_m=1e-2),
        "Tuned": Aperture(material="coated", shape="round", radius_m=1e-2, length_m=1.0, material_overrides=override),
    })
    assert sections.resolved_layers("Default") == [(1e-6, 0.0, 1e-6), (1.7e-8, 0.0, 1.0)]
    assert sections.resolved_layers("Tuned") == [(1e-6, 0.0, 2e-6), (1.7e-8, 0.0, 1.0)]


def test_round_aperture_impedance_is_the_round_chamber_of_its_whole_length():
    sections = ring(A=Aperture(material="steel", shape="round", radius_m=5e-3, length_m=2.0, count=10))
    expected = round_chamber_impedance(RoundChamber("A", 5e-3, 20.0, [(7.2e-7, 0.0, 1.0)]), FREQ, GAMMA)
    np.testing.assert_allclose(as_complex(sections.aperture_impedance("A", FREQ, GAMMA, CIRCUMFERENCE_M)), as_complex(expected), rtol=1e-12)


def test_a_ring_cut_into_sections_of_one_kind_has_the_impedance_of_one_pipe_of_the_circumference():
    cut = ring(A=Aperture(material="copper", shape="round", radius_m=12e-3, length_m=3.0, count=7))  # same pipe as Default
    whole = round_chamber_impedance(RoundChamber("ring", 12e-3, CIRCUMFERENCE_M, [(1.7e-8, 0.0, 1.0)]), FREQ, GAMMA)
    np.testing.assert_allclose(as_complex(cut.impedance(FREQ, GAMMA, CIRCUMFERENCE_M)), as_complex(whole), rtol=1e-9)


def test_a_flat_aperture_has_the_longitudinal_impedance_of_its_smaller_half_aperture():
    flat = ring(F=Aperture(material="copper", shape="flat", horizontal_half_aperture_m=30e-3, vertical_half_aperture_m=4e-3,
                           length_m=5.0))
    round_ = ring(F=Aperture(material="copper", shape="round", radius_m=4e-3, length_m=5.0))
    np.testing.assert_allclose(as_complex(flat.aperture_impedance("F", FREQ, GAMMA, CIRCUMFERENCE_M)),
                               as_complex(round_.aperture_impedance("F", FREQ, GAMMA, CIRCUMFERENCE_M)), rtol=1e-12)


def test_a_narrower_pipe_of_the_same_length_has_the_larger_impedance():
    narrow = ring(N=Aperture(material="copper", shape="round", radius_m=6e-3, length_m=10.0))
    wide = ring(N=Aperture(material="copper", shape="round", radius_m=12e-3, length_m=10.0))
    ratio = np.abs(as_complex(narrow.aperture_impedance("N", FREQ, GAMMA, CIRCUMFERENCE_M))
                   / as_complex(wide.aperture_impedance("N", FREQ, GAMMA, CIRCUMFERENCE_M)))
    np.testing.assert_allclose(ratio, 2.0, rtol=0.01)  # Z ~ 1/b


def test_yaml_round_trip_keeps_every_material_and_aperture_and_rejects_unknown_fields(tmp_path):
    sections = ring(A=Aperture(material="steel", shape="flat", horizontal_half_aperture_m=0.03, vertical_half_aperture_m=0.004,
                               length_m=2.0, count=3, material_overrides={"steel": {"rho": 8e-7}}))
    path = tmp_path / "apertures.yaml"
    sections.to_yaml(path)
    assert Apertures.from_yaml(path) == sections
    path.write_text("Materials: {}\nApertures:\n  Default: {material: x, shape: round, radius: 0.01, colour: red}\n")
    with pytest.raises(ValueError, match="unknown aperture field"):
        Apertures.from_yaml(path)
