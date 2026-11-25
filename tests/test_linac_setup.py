"""Crest calibration: the phase that gives a reference particle the maximum energy gain from a cavity."""
import math

import pytest
import torch

from slipstream.tracking.native.lattice_import import ElementSpec
from slipstream.tracking.native.linac import LinacTracker
from slipstream.tracking.native.linac_setup import crest_phase_rad, element_index, with_element

MC2_EV = 0.51099895e6
ENERGY_EV = 50e6
VOLTAGE_V = 5e6
REFERENCE = torch.zeros(6)


def cavity(phase=0.3, name="RF"):
    return ElementSpec(name, "rf", 1.0, voltage_V=VOLTAGE_V, phase_rad=phase, frequency_Hz=2.998e9, n_kicks=16)


def gain_eV(specs):
    tracker = LinacTracker(specs, energy_eV=ENERGY_EV, mc2_eV=MC2_EV, device="cpu", dtype=torch.float32, compile=False)
    return float(tracker.track(REFERENCE.reshape(1, 6))[1]) - ENERGY_EV


def test_crest_is_a_quarter_period_where_the_gain_is_the_full_voltage():
    specs = [cavity()]
    crest = crest_phase_rad(specs, 0, REFERENCE, ENERGY_EV)
    assert crest == pytest.approx(math.pi / 2, abs=math.radians(0.2))  # RF phase is a sine: the gain peaks at 90 degrees
    assert gain_eV([with_element(specs, "RF", phase_rad=crest)[0]]) == pytest.approx(VOLTAGE_V, rel=1e-4)


def test_gain_follows_the_sine_of_the_phase_around_the_calibrated_crest():
    specs = [cavity()]
    crest = crest_phase_rad(specs, 0, REFERENCE, ENERGY_EV)
    for offset_deg in (-60.0, -30.0, 30.0, 60.0):
        phase = crest + math.radians(offset_deg)
        assert gain_eV(with_element(specs, "RF", phase_rad=phase)) == pytest.approx(
            VOLTAGE_V * math.cos(math.radians(offset_deg)), rel=2e-3)


def test_crest_does_not_depend_on_what_the_reference_particle_met_upstream():
    # RF phases are relative to the reference particle's arrival, so a drift and a cavity upstream move nothing
    bare = crest_phase_rad([cavity()], 0, REFERENCE, ENERGY_EV)
    upstream = [ElementSpec("D", "drift", 1.37), cavity(phase=1.0, name="RF1"), ElementSpec("D2", "drift", 0.5), cavity(name="RF2")]
    assert crest_phase_rad(upstream, 3, REFERENCE, ENERGY_EV) == pytest.approx(bare, abs=math.radians(0.2))


def test_with_element_replaces_only_the_named_element_and_leaves_the_original_alone():
    specs = [ElementSpec("D", "drift", 1.0), cavity(phase=0.3)]
    changed = with_element(specs, "RF", phase_rad=1.1, voltage_V=1.0)
    assert (changed[1].phase_rad, changed[1].voltage_V) == (1.1, 1.0) and changed[0] is specs[0]
    assert (specs[1].phase_rad, specs[1].voltage_V) == (0.3, VOLTAGE_V)
    assert element_index(specs, "RF") == 1
    with pytest.raises(StopIteration):
        element_index(specs, "missing")
