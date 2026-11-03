"""RFCW wakes are per-cell tables: each sub-kick applies ``length / cell_length / n_kicks`` of the table (elegant's ``wake.factor``)."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from slipstream.constants import C_LIGHT, ELECTRON_MASS_EV
from slipstream.tracking.native.lattice_import import ElementSpec, parse_elegant_lattice
from slipstream.tracking.native.linac import LinacTracker
from slipstream.wakefields.shortrange import ShortRangeWake
from slipstream.wakefields.transverse import TransverseWake

LENGTH, FREQ_HZ, ENERGY_EV, CHARGE_C = 1.8, 2.998e9, 3.0e9, 1e-9
T_WAKE = np.linspace(0.0, 8e-12, 1000)


def _witness_delta(n_kicks: int, cell_length: float, transverse: bool = False) -> float:
    """Energy deviation (or ``px``) of a witness 0.5 ps behind a driver, through a ``VOLT=0`` structure."""
    if transverse:
        kwargs = dict(transverse_wake_x=TransverseWake(T_WAKE, 2e16 * np.exp(-T_WAKE / 1e-12)))
        kwargs["transverse_wake_x"].set_fixed_bin_window(-2e-12, 2e-12)
    else:
        kwargs = dict(wake=ShortRangeWake(T_WAKE, -5e14 * np.exp(-T_WAKE / 1e-12)))
    spec = ElementSpec("ELM", "rf", LENGTH, frequency_Hz=FREQ_HZ, n_kicks=n_kicks, cell_length=cell_length, **kwargs)
    # the witness is deflected while it crosses the structure and then feels its own dipole moment: keep that second-order term below the tolerance
    charge = 1e-12 if transverse else CHARGE_C
    tracker = LinacTracker([spec], energy_eV=ENERGY_EV, dtype=torch.float64, device="cpu", always_change_p0=False, bunch_charge_C=charge)
    particles = torch.zeros(2, 6, dtype=torch.float64)
    particles[:, 4] = torch.tensor([0.0, 5e-13 * C_LIGHT], dtype=torch.float64)
    particles[0, 0] = 1e-3 if transverse else 0.0       # the driver carries the dipole moment
    out, _ = tracker.track(particles)
    return float(out[1, 1] if transverse else out[1, 5])


@pytest.mark.parametrize("transverse", [False, True])
def test_wake_total_does_not_depend_on_the_number_of_sub_kicks(transverse):
    one = _witness_delta(1, LENGTH, transverse)
    many = _witness_delta(20, LENGTH, transverse)
    assert abs(one) > 0
    assert many == pytest.approx(one, rel=1e-3)


@pytest.mark.parametrize("transverse", [False, True])
def test_wake_total_scales_with_the_number_of_cells(transverse):
    whole = _witness_delta(12, LENGTH, transverse)
    assert _witness_delta(12, LENGTH / 6, transverse) == pytest.approx(6 * whole, rel=1e-3)


def test_unset_cell_length_means_one_cell_for_the_whole_element():
    assert _witness_delta(12, 0.0) == pytest.approx(_witness_delta(12, LENGTH), rel=1e-12)


def test_rfcw_parser_reads_cell_length_and_wake_switches(tmp_path):
    base = 'CAV: RFCW, L=1.8, VOLT=1e6, FREQ=2.998e9, N_KICKS=40, CELL_LENGTH=0.03, WAKEFILE="w.sdds", TCOLUMN="t", WZCOLUMN="Wz", %s;\nLINE1: LINE=(CAV)\n'
    lte = tmp_path / "a.lte"
    lte.write_text(base % "ZWAKE=0, TRWAKE=1")
    (spec,) = parse_elegant_lattice(lte, "LINE1")
    assert (spec.cell_length, spec.include_z_wake, spec.include_tr_wake) == (0.03, False, True)
    lte.write_text(base % "WAKES_AT_END=1")
    with pytest.raises(NotImplementedError, match="WAKES_AT_END"):
        parse_elegant_lattice(lte, "LINE1")


def test_transverse_wake_without_a_window_gets_one_from_the_bunch():
    wake = TransverseWake(T_WAKE, 2e16 * np.exp(-T_WAKE / 1e-12))
    spec = ElementSpec("ELM", "rf", LENGTH, frequency_Hz=FREQ_HZ, cell_length=LENGTH, transverse_wake_x=wake)
    tracker = LinacTracker([spec], energy_eV=ENERGY_EV, dtype=torch.float64, device="cpu", always_change_p0=False, bunch_charge_C=CHARGE_C)
    particles = torch.zeros(2, 6, dtype=torch.float64)
    particles[:, 4] = torch.tensor([0.0, 5e-13 * C_LIGHT], dtype=torch.float64)
    particles[0, 0] = 1e-3
    out, _ = tracker.track(particles)
    assert out[1, 1].abs() > 0
