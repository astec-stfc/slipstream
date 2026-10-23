"""Independent flight-time oracle for the skill factory's existing-physics pilot."""
import math

import pytest
import torch

from slipstream.constants import C_LIGHT, ELECTRON_MASS_EV
from slipstream.tracking.native.elements import track_element
from slipstream.tracking.native.lattice_import import ElementSpec
from slipstream.tracking.native.linac import LinacTracker


def velocity_reference(state, length, energy_eV, rest_eV):
    """Integrate a free particle via dimensional velocity and arrival time."""
    p0_eVc = math.sqrt(energy_eV**2 - rest_eV**2)
    v0 = C_LIGHT * p0_eVc / energy_eV
    expected = []
    for x, px, y, py, tau, delta in state.tolist():
        momentum = p0_eVc * (1 + delta)
        energy = math.hypot(momentum, rest_eV)
        transverse_x = p0_eVc * px
        transverse_y = p0_eVc * py
        longitudinal = math.sqrt(momentum**2 - transverse_x**2 - transverse_y**2)
        vx, vy, vz = (C_LIGHT * component / energy
                      for component in (transverse_x, transverse_y, longitudinal))
        duration = length / vz
        expected.append([x + vx * duration, px, y + vy * duration, py,
                         tau + C_LIGHT * (duration - length / v0), delta])
    return torch.tensor(expected, dtype=state.dtype)


def particles():
    return torch.tensor([[0., 0., 0., 0., 0., 0.],
                         [1e-3, .12, -2e-3, -.07, 3e-3, .08],
                         [-2e-3, -.09, 1e-3, .04, -1e-3, -.06]], dtype=torch.float64)


@pytest.mark.parametrize("energy,rest", [(2e6, ELECTRON_MASS_EV), (100e6, ELECTRON_MASS_EV),
                                        (1.2e9, 938272088.16)])
def test_drift_matches_dimensional_velocity_and_flight_time(energy, rest):
    state, length = particles(), 1.7
    expected = velocity_reference(state, length, energy, rest)
    actual = track_element(state, length, 0., 0., 0., 0., 0., energy, rest)
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-14)
    torch.testing.assert_close(actual[:, [1, 3, 5]], state[:, [1, 3, 5]], rtol=0., atol=0.)


def test_drift_identity_length_scaling_and_subdivision():
    state, energy = particles(), 2e6

    def drift(s, length):
        return track_element(s, length, 0., 0., 0., 0., 0., energy, ELECTRON_MASS_EV)

    torch.testing.assert_close(drift(state, 0.), state, rtol=0., atol=0.)
    whole = drift(state, 1.7)
    halves = drift(drift(state, .85), .85)
    torch.testing.assert_close(whole, halves, rtol=2e-12, atol=2e-14)
    torch.testing.assert_close(drift(state, 3.4) - state, 2 * (whole - state), rtol=2e-12, atol=2e-14)


def test_linac_public_entry_point_matches_flight_time_reference():
    state, length, energy = particles(), 1.7, 2e6
    tracker = LinacTracker([ElementSpec(name="D", kind="drift", length=length)],
                           energy_eV=energy, device="cpu", dtype=torch.float64, always_change_p0=False)
    actual, output_energy = tracker.track(state)
    expected = velocity_reference(state, length, energy, ELECTRON_MASS_EV)
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-14)
    assert float(output_energy) == energy


def test_oracle_detects_a_finite_factor_two_time_of_flight_error():
    state, length, energy = particles(), 1.7, 2e6
    expected = velocity_reference(state, length, energy, ELECTRON_MASS_EV)
    candidate = expected.clone()
    candidate[:, 4] = state[:, 4] + 2 * (expected[:, 4] - state[:, 4])
    assert torch.isfinite(candidate).all()
    with pytest.raises(AssertionError):
        torch.testing.assert_close(candidate, expected, rtol=2e-12, atol=2e-14)
