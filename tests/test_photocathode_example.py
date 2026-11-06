"""The CLARA Schottky-scan example: its emission-law fit on the measured scan, and its machine set-up."""
import math
import runpy
import sys
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
                           particles=120, device="cpu", dtype="float64", compile=False, warp=False, dt_ceiling_ps=0.0,
                           flatness=None, solenoid_offset_mm=0.0, solenoid_tilt_mrad=0.0, steering=False, reference_offset_deg=-4.0)
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


def test_camera_frame_settings_support_the_best_fit_solenoid(example):
    if not example["IMAGE"].is_file():
        pytest.skip("the laser camera frame is not in this checkout")
    settings = example["capture_settings"]()
    assert settings == pytest.approx(
        dict(laser_energy_uJ=0.539, gun_solenoid_A=139.85, gun_phase_deg=-125.0938, charge_pC=99.694), rel=1e-5
    )
    # 139.85 A at the 2022 calibration of 0.0022 T/A is the best-fit 0.310 T to within the calibration's accuracy
    assert settings["gun_solenoid_A"] * example["SOLENOID_T_PER_A"] == pytest.approx(example["SOLENOID_T"], rel=0.01)
    assert example["capture_settings"](EXAMPLE.with_name("missing.hdf5")) == {}


def test_energy_and_qe_split_keeps_their_product_and_follows_the_command_line(example):
    photo = Photocathode(pulse_energy_J=5.6e-6, wavelength_nm=266.0, qe=1e-4, qe_field_V_per_m=105e6)
    split = example["split_energy_and_qe"]
    energy = split(photo, SimpleNamespace(laser_energy_uJ=0.539, qe=None))  # a known laser energy: the fit gives the QE
    assert energy.pulse_energy_J == pytest.approx(0.539e-6) and energy.qe == pytest.approx(1.039e-3, rel=1e-3)
    qe = split(photo, SimpleNamespace(laser_energy_uJ=None, qe=2e-4))  # a known QE: the fit gives the laser energy
    assert qe.qe == 2e-4 and qe.pulse_energy_J == pytest.approx(2.8e-6)
    neither = split(photo, SimpleNamespace(laser_energy_uJ=None, qe=None))
    assert (neither.qe, neither.pulse_energy_J) == pytest.approx((1e-4, 5.6e-6))
    for result in (energy, qe, neither):
        assert result.pulse_energy_J * result.qe == pytest.approx(5.6e-6 * 1e-4)
        assert result.work_function_eV == photo.work_function_eV and result.qe_field_V_per_m == 105e6


def test_capture_fills_in_the_laser_energy_and_optionally_the_solenoid(example):
    capture = dict(laser_energy_uJ=0.539, gun_solenoid_A=139.85, gun_phase_deg=-125.1, charge_pC=99.7)
    apply = example["apply_capture"]
    args = SimpleNamespace(qe=None, laser_energy_uJ=None, solenoid_T=0.310, solenoid_T_per_A=None)
    apply(args, capture)
    assert (args.laser_energy_uJ, args.solenoid_T) == (0.539, 0.310)  # the solenoid stays at the best fit by default
    args = SimpleNamespace(qe=1e-3, laser_energy_uJ=None, solenoid_T=0.310, solenoid_T_per_A=0.0022)
    apply(args, capture)
    assert args.laser_energy_uJ is None  # a given QE keeps the energy a fit result
    assert args.solenoid_T == pytest.approx(139.85 * 0.0022)
    args = SimpleNamespace(qe=None, laser_energy_uJ=0.7, solenoid_T=0.310, solenoid_T_per_A=None)
    apply(args, capture)
    assert args.laser_energy_uJ == 0.7  # the command line wins over the frame
    apply(args, {})  # no frame: nothing to fill in
    with pytest.raises(SystemExit, match="camera frame"):
        apply(SimpleNamespace(qe=None, laser_energy_uJ=None, solenoid_T=0.31, solenoid_T_per_A=0.0022), {})


def test_default_gun_settings_are_the_best_fit_and_a_given_qe_excludes_a_given_energy(example, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["schottky_scan.py"])
    args = example["parse_args"]()
    assert (args.gun_field_MV, args.solenoid_T, args.qe, args.laser_energy_uJ) == (109.0, 0.3053, None, None)
    assert (args.flatness, args.solenoid_offset_mm, args.solenoid_tilt_mrad) == (0.942, 0.87, -0.77)  # fit_structure's best
    assert args.steering is True and args.reference_offset_deg == -4.0
    monkeypatch.setattr(sys, "argv", ["schottky_scan.py", "--flatness", "1.0", "--solenoid-offset-mm", "0",
                                      "--solenoid-tilt-mrad", "0.5", "--no-steering", "--reference-offset-deg", "0"])
    args = example["parse_args"]()
    assert (args.flatness, args.solenoid_offset_mm, args.solenoid_tilt_mrad) == (1.0, 0.0, 0.5)
    assert args.steering is False and args.reference_offset_deg == 0.0
    monkeypatch.setattr(sys, "argv", ["schottky_scan.py", "--qe", "1e-3", "--laser-energy-uJ", "0.5"])
    with pytest.raises(SystemExit):
        example["parse_args"]()


class _GunStub:
    """A machine whose gun passes the electrons emitted before 30 deg of gun phase offset and stops the rest."""

    def __init__(self):
        self.cavities = [SimpleNamespace(frequency_hz=2.9985e9, phase_deg=7.0)]
        self.photocathode, self.space_charge, self.generator_overrides = "photo", "sc", {"sig_x": 0.4}
        self.seen = None

    def initial_beam(self, n, device, dtype):
        lt_s = self.generator_overrides["lt"] * 1e-9
        t = (torch.arange(n, dtype=torch.float64) + 0.5) / n * lt_s - 0.5 * lt_s  # uniform over the window, centred
        return {"emission_time_s": t, "q": torch.full((n,), 1e-15, dtype=torch.float64)}

    def track(self, n_particles, **kwargs):
        beam = self.initial_beam(n_particles, "cpu", torch.float64)
        self.seen = (self.photocathode, self.space_charge, dict(self.generator_overrides), self.cavities[0].phase_deg)
        offset = self.cavities[0].phase_deg + 360.0 * self.cavities[0].frequency_hz * beam["emission_time_s"]
        return {"screen": {"q": -beam["q"] * (offset < 30.0)}}


def test_transmission_function_bins_the_tracers_by_emission_phase_and_leaves_the_machine_as_it_was(example):
    machine = _GunStub()
    args = SimpleNamespace(particles=20000, device="cpu", dtype="float64", compile=False, warp=False, monitor_z=1.1)
    centres, transmission = example["transmission_function"](machine, args, window=(-60.0, 85.0), bin_deg=0.5)
    assert centres[0] == pytest.approx(-59.75) and centres[-1] == pytest.approx(84.75) and len(centres) == 290
    assert transmission[centres < 29.5].min() == pytest.approx(1.0) and transmission[centres > 30.5].max() == 0.0
    assert 0.0 < transmission[np.abs(centres - 30.0) < 0.3].mean() < 1.0  # only the bin holding the edge is partial
    photocathode, space_charge, overrides, phase = machine.seen  # what the run itself used: tracers without space charge
    assert photocathode is None and space_charge.enabled is False and phase == pytest.approx(12.5)
    assert overrides["lt"] == pytest.approx(145.0 / (360.0 * 2.9985e9) * 1e9) and overrides["sig_x"] == 0.4
    assert (machine.photocathode, machine.space_charge, machine.generator_overrides) == ("photo", "sc", {"sig_x": 0.4})
    assert machine.cavities[0].phase_deg == 7.0


def test_scan_from_transmission_averages_over_the_laser_pulse(example):
    centres = 0.125 + 0.25 * np.arange(400)  # 0 to 100 deg in 0.25 deg bins
    transmission = np.where(centres < 50.0, 1.0, 0.5)
    deltas, emitted = np.array([10.0, 50.0, 90.0, 150.0]), np.array([100.0, 100.0, 80.0, 100.0])
    scan = example["scan_from_transmission"](centres, transmission, deltas, emitted, pulse_deg=2.0)
    # a 2 deg pulse centred on the step sees four bins of each: 0.75; outside the phases covered there is nothing
    assert scan[:3] == pytest.approx([100.0, 75.0, 40.0]) and np.isnan(scan[3])
    narrow = example["scan_from_transmission"](centres, transmission, [10.0], [100.0], pulse_deg=0.0)
    assert narrow == pytest.approx([100.0])  # never narrower than a bin


def test_transmission_function_of_the_gun_is_zero_before_extraction_and_one_until_fall_back(example):
    pytest.importorskip("laura.laura")
    if example["CLARA"] is None:
        pytest.skip("laura_lattices.CLARA not installed")
    args = SimpleNamespace(generator=str(EXAMPLE.with_name("generator.in")), gun_field_MV=105.0, solenoid_T=0.310,
                           particles=200, device="cpu", dtype="float64", compile=False, warp=False, dt_ceiling_ps=2.0,
                           monitor_z=0.03, flatness=None, solenoid_offset_mm=0.0, solenoid_tilt_mrad=0.0, steering=False, reference_offset_deg=-4.0)
    machine, _, _ = example["build_machine"](args, 0.03, None, 0.5)
    centres, transmission = example["transmission_function"](machine, args, window=(-70.0, 90.0), bin_deg=20.0)
    assert list(centres) == [-60.0, -40.0, -20.0, 0.0, 20.0, 40.0, 60.0, 80.0]
    # the cathode field is zero at -53 deg: nothing before it leaves (the first bin straddles it), everything does up to
    # the late phases, where the field has reversed before the electron leaves the half cell and it falls back
    assert transmission[0] < 0.6 and transmission[1:7].min() > 0.95 and transmission[7] < 0.1


def _build_args(tmp_path, **kw):
    args = dict(generator=str(EXAMPLE.with_name("generator.in")), gun_field_MV=105.0, solenoid_T=0.310, particles=200,
                device="cpu", dtype="float64", compile=False, warp=False, dt_ceiling_ps=2.0, monitor_z=1.1,
                flatness=None, solenoid_offset_mm=0.0, solenoid_tilt_mrad=0.0, steering=True, reference_offset_deg=-4.0,
                output_dir=str(tmp_path))  # fmt: skip
    return SimpleNamespace(**{**args, **kw})


def test_build_machine_applies_the_flatness_and_the_solenoid_placement_and_steers_only_a_misaligned_one(
    example, tmp_path, monkeypatch
):
    pytest.importorskip("laura.laura")
    if example["CLARA"] is None:
        pytest.skip("laura_lattices.CLARA not installed")
    steered = []
    monkeypatch.setitem(example["build_machine"].__globals__, "steer_correctors", lambda m, a: steered.append(m))
    build = example["build_machine"]
    machine, peak, _ = build(_build_args(tmp_path), 1.1, None, 0.5)  # aligned, the design map
    assert peak == pytest.approx(105e6, rel=1e-6) and not machine.solenoids[0].placed and not steered
    machine, peak, _ = build(_build_args(tmp_path, flatness=0.90, solenoid_offset_mm=0.6, solenoid_tilt_mrad=-1.2), 1.1, None, 0.5)
    assert peak == pytest.approx(105e6, rel=1e-6)  # the cathode field is the one asked for, whatever the flatness
    assert Path(machine.cavities[0].file).parent == tmp_path / "flat_maps" and Path(machine.cavities[0].file).is_file()
    assert (machine.solenoids[0].x_m, machine.solenoids[0].yrot_rad) == (0.6e-3, -1.2e-3) and steered == [machine]
    machine, peak, _ = build(_build_args(tmp_path, flatness=1.2), 1.1, None, 0.5)  # the full cell is the largest field
    assert peak == pytest.approx(105e6, rel=1e-6) and machine.cavities[0].e_max_V_per_m < 0.0
    steered.clear()
    build(_build_args(tmp_path, solenoid_offset_mm=0.6, steering=False), 1.1, None, 0.5)  # --no-steering
    build(_build_args(tmp_path, solenoid_offset_mm=0.6), 0.5, None, 0.5)  # a machine that ends before the screen
    assert not steered


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware required")
def test_steering_centres_the_beam_on_the_bpm_and_screen_at_the_reference_phase_only(example, tmp_path):
    pytest.importorskip("laura.laura")
    if example["CLARA"] is None:
        pytest.skip("laura_lattices.CLARA not installed")
    args = _build_args(tmp_path, particles=4000, device="cuda", dtype="float32", compile=True, solenoid_offset_mm=0.8,
                       solenoid_tilt_mrad=-0.7)  # fmt: skip
    machine, _, _ = example["build_machine"](args, 1.1, None, 0.4)
    assert len(machine.correctors) == 2 and any(abs(c.bx_T) + abs(c.by_T) > 1e-6 for c in machine.correctors)

    def centroid(offset, z):
        machine.cavities[0].phase_deg = offset
        out = machine.track(n_particles=4000, device="cuda", dtype=torch.float32, verbose=False, record_every=0,
                            screen_z_m=z, z_stop_m=z + 0.005, use_compile=True, use_warp=False)["screen"]  # fmt: skip
        q = out["q"].abs().double()
        return [float((out[k].double() * q).sum() / q.sum()) for k in ("x", "y")]

    for z in example["OBSERVATION_Z"]:
        assert max(abs(v) for v in centroid(-4.0, z)) < 2e-4  # centred where the correctors were set
        assert max(abs(v) for v in centroid(58.0, z)) > 3e-3  # and several mm off for the slow late-phase electrons
