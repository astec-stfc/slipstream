"""The CLARA Schottky-scan example: its emission-law fit on the measured scan, and its machine set-up."""
import math
import runpy
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from slipstream.tracking.native.photocathode import Photocathode

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/photocathode/schottky_scan.py"
DATA = EXAMPLE.with_name("104459_Schottky_approx_crestingData.h5")


@pytest.fixture(scope="module")
def example():
    return runpy.run_path(str(EXAMPLE))


def test_emission_law_fit_recovers_the_clean_copper_work_function_from_the_measured_flank(example):
    # Same field-limited flank the example fits: the Schottky law with the 120 MV/m gun and a 3 ps flat-top laser.
    emission = object.__new__(example["Emission"])
    emission.t = torch.linspace(-1.5e-12, 1.5e-12, 61, dtype=torch.float64)
    emission.crest, emission.peak, emission.omega = 145.78, 120e6, 2.0 * math.pi * 2.9985e9
    with h5py.File(DATA, "r") as f:
        x, y, s = f["xData"][:], f["yData"][:], f["yStd"][:]
    photo = Photocathode(pulse_energy_J=3e-6, wavelength_nm=266.0, qe=1e-4, work_function_eV=4.5, qe_field_V_per_m=120e6)
    fit = example["fit_emission"](emission, photo, x, y, s)
    assert fit["work_function_eV"] == pytest.approx(4.60, abs=0.03)  # polycrystalline copper is 4.6-4.65 eV
    assert fit["x0"] == pytest.approx(-131.7, abs=0.5)  # scan phase of the gun's max-energy-gain crest
    assert fit["baseline_pC"] == pytest.approx(2.5, abs=0.3)  # the monitor's offset
    assert fit["chi2_per_point"] < 2.0
    assert fit["pulse_energy_J"] == pytest.approx(5.5e-6, rel=0.05)  # for QE 1e-4 at 120 MV/m


def test_emission_curve_has_the_measured_zero_crossing_and_field_maximum(example):
    emission = object.__new__(example["Emission"])
    emission.t = torch.zeros(1, dtype=torch.float64)
    emission.crest, emission.peak, emission.omega = 145.78, 120e6, 2.0 * math.pi * 2.9985e9
    photo = Photocathode(pulse_energy_J=5e-6, wavelength_nm=266.0, qe=1e-4, material="Cu", qe_field_V_per_m=120e6)
    offsets = np.arange(-70.0, 110.0, 1.0)
    charge = emission.charge_pC(photo, offsets)
    assert charge[offsets < -56].max() == 0.0 and charge[offsets > -54].min() > 0.0  # field zero crossing at 180-crest-90
    assert offsets[charge.argmax()] == pytest.approx(34.0, abs=1.0)  # cathode field maximum at 180 deg of cos phase


def test_example_machine_tracks_the_gun_and_apertures_with_laura_conventions(example, tmp_path):
    pytest.importorskip("laura.laura")
    if example["CLARA"] is None:
        pytest.skip("laura_lattices.CLARA not installed")
    args = SimpleNamespace(generator=str(EXAMPLE.with_name("generator.in")), gun_field_MV=None, solenoid_T=None,
                           particles=120, device="cpu", dtype="float64")
    photo = Photocathode(pulse_energy_J=5.8e-6, wavelength_nm=266.0, qe=1e-4, work_function_eV=4.6,
                         qe_field_V_per_m=120e6)
    machine, peak, omega = example["build_machine"](args, 0.1, photo, 0.5)
    assert peak == pytest.approx(120e6, rel=1e-6) and omega == pytest.approx(2 * math.pi * 2.9985e9, rel=1e-9)
    assert [round(a.z1_m, 4) for a in machine.apertures if a.kind == "COL_X"] == [0.0, 0.32, 1.0577]
    assert len(machine.cavities) == 1 and len(machine.solenoids) == 1  # L01's cavity and solenoids lie past the monitor
    assert machine.absolute_phases and machine.cavities[0].crest_deg == pytest.approx(145.78, abs=0.05)
    track = example["track_charge"]
    em, screen = track(machine, 34.0, args, 0.09, z_stop_m=0.1)  # field maximum: the whole emission leaves the gun
    assert em["charge_C"] == pytest.approx(em["emitted_charge_C"], rel=1e-9)
    assert 0.85 < em["emitted_charge_C"] / em["ideal_charge_C"] < 1.0  # space-charge-limited emission, a few percent
    assert float(screen["q"].abs().sum()) == pytest.approx(em["emitted_charge_C"], rel=1e-9)
    em, screen = track(machine, -70.0, args, 0.09, z_stop_m=0.1)  # before the field turns on: nothing is extracted
    assert em["emitted_charge_C"] == 0.0 and float(screen["q"].abs().sum()) == 0.0
