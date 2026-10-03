"""Tests for slipstream.tracking.native.errors -- see
plans/NATIVE_ERRORS_PLAN.md and tracking/native/PHYSICS.md's "Errors and
fiducialization" section.

Phase 1: ErrorSpec/SampledError sampling+application (pure math, no elegant
needed) -- distribution/absolute-vs-fractional correctness, occurrence-based
element-instance matching, RF-error resolution, and NativeTracker's own
errors=/fiducial= wiring (phi_s_eff resolution -- design-derived, not (yet)
live-elegant-validated, see NativeTracker's own docstring).
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from slipstream.lattice.lattice import Lattice
from slipstream.tracking.native import NativeTracker, parse_elegant_lattice
from slipstream.tracking.native.errors import ErrorSpec, SampledError, apply_errors, resolve_rf_errors, sample_errors
from slipstream.tracking.native.lattice_import import ElementSpec

DIAMOND2 = Path(__file__).resolve().parent.parent / "examples" / "diamond2booster"
BR14_SIMPLE_LTE = DIAMOND2 / "lattice" / "br14_simple.lte"
needs_lattice = pytest.mark.skipif(not BR14_SIMPLE_LTE.exists(), reason="lattice file not present")


def test_errorspec_rejects_unknown_parameter_and_distribution():
    with pytest.raises(ValueError, match="parameter"):
        ErrorSpec(name_pattern="Q*", parameter="NOT_A_PARAM", amplitude=1.0)
    with pytest.raises(ValueError, match="distribution"):
        ErrorSpec(name_pattern="Q*", parameter="K1", amplitude=1.0, distribution="not_a_distribution")


def test_sample_errors_matches_by_wildcard_and_element_type():
    specs = [
        ElementSpec("QF", "quad", length=0.5, k1=1.2, occurrence=1),
        ElementSpec("QD", "quad", length=0.5, k1=-1.1, occurrence=1),
        ElementSpec("BD", "bend", length=1.0, angle=0.1, occurrence=1),
    ]
    gen = torch.Generator()
    gen.manual_seed(0)
    sampled = sample_errors(specs, [ErrorSpec(name_pattern="Q*", parameter="K1", amplitude=1e-3)], gen)
    assert {s.name for s in sampled} == {"QF", "QD"}

    sampled2 = sample_errors(
        specs, [ErrorSpec(name_pattern="*", parameter="ANGLE", amplitude=1e-3, element_type="bend")], gen,
    )
    assert {s.name for s in sampled2} == {"BD"}


def test_sample_errors_absolute_vs_fractional():
    specs = [ElementSpec("Q", "quad", length=0.5, k1=2.0, occurrence=1)]
    gen = torch.Generator()
    gen.manual_seed(1)
    # plus_or_minus is deterministic in magnitude (exactly +-amplitude), so
    # the resolved delta's *magnitude* is exactly checkable without needing
    # to know which way the coin flip landed.
    absolute = sample_errors(specs, [ErrorSpec(name_pattern="Q", parameter="K1", amplitude=0.05, distribution="plus_or_minus")], gen)
    assert abs(absolute[0].delta) == pytest.approx(0.05)

    fractional = sample_errors(
        specs, [ErrorSpec(name_pattern="Q", parameter="K1", amplitude=0.05, distribution="plus_or_minus", fractional=True)], gen,
    )
    assert abs(fractional[0].delta) == pytest.approx(2.0 * 0.05)  # 5% of k1=2.0


def test_gaussian_draws_are_clipped_at_cutoff():
    specs = [ElementSpec("Q", "quad", length=0.5, k1=1.0, occurrence=1)]
    gen = torch.Generator()
    gen.manual_seed(2)
    sampled = sample_errors(
        specs, [ErrorSpec(name_pattern="Q", parameter="K1", amplitude=1.0, distribution="gaussian", cutoff=2.0)] * 500, gen,
    )
    assert all(abs(s.delta) <= 2.0 + 1e-12 for s in sampled)
    assert max(abs(s.delta) for s in sampled) > 1.0  # not so tight the clip never matters


def test_apply_errors_adds_to_current_value_not_replaces_it():
    """Regression: an earlier version of apply_errors set the field to the
    *sum of deltas* directly (dataclasses.replace(spec, k1=delta_sum)),
    silently discarding the element's own original value -- caught by a
    manual sanity check before this test existed. Two independent errors
    on the same parameter must also sum, not overwrite each other."""
    specs = [ElementSpec("Q", "quad", length=0.5, k1=2.0, occurrence=1)]
    sampled = [SampledError("Q", 1, "K1", 0.1), SampledError("Q", 1, "K1", 0.05)]
    out = apply_errors(specs, sampled)
    assert out[0].k1 == pytest.approx(2.0 + 0.1 + 0.05)
    assert out[0] is not specs[0]  # a new object, original untouched
    assert specs[0].k1 == 2.0


def test_apply_errors_leaves_unmatched_elements_identical():
    specs = [ElementSpec("Q", "quad", length=0.5, k1=2.0, occurrence=1), ElementSpec("D", "drift", length=1.0)]
    out = apply_errors(specs, [])
    assert out[0] is specs[0]
    assert out[1] is specs[1]


def test_apply_errors_matches_by_occurrence_not_just_name():
    """Two different instances of the same NAME (e.g. two 'BD's at
    different points in the ring) must be perturbed independently."""
    specs = [
        ElementSpec("BD", "bend", length=1.0, angle=0.1, k1=1.0, occurrence=1),
        ElementSpec("BD", "bend", length=1.0, angle=0.1, k1=1.0, occurrence=2),
    ]
    out = apply_errors(specs, [SampledError("BD", 2, "K1", 0.5)])
    assert out[0].k1 == pytest.approx(1.0)  # occurrence 1 untouched
    assert out[1].k1 == pytest.approx(1.5)  # occurrence 2 perturbed


def test_resolve_rf_errors_sums_by_parameter_and_converts_freq_to_omega():
    sampled = [
        SampledError("CAV", 1, "PHASE", 0.01),
        SampledError("CAV", 1, "PHASE", -0.002),
        SampledError("CAV", 1, "VOLT", 500.0),
        SampledError("CAV", 1, "FREQ", 100.0),
    ]
    rf = resolve_rf_errors(sampled)
    assert rf["phi_s"] == pytest.approx(0.008)
    assert rf["Vrf_V"] == pytest.approx(500.0)
    assert rf["omega_rf"] == pytest.approx(2.0 * 3.141592653589793 * 100.0)


@needs_lattice
def test_occurrence_uniquely_identifies_every_real_bd_instance():
    """End-to-end sanity on the real lattice: br14_simple.lte's real BDs
    (K1+K2 real bend, ~38 instances) must each get a distinct occurrence,
    and a name+occurrence-targeted error must hit exactly one."""
    specs = parse_elegant_lattice(BR14_SIMPLE_LTE, root_name="boost")
    bds = [s for s in specs if s.name == "BD"]
    assert len(bds) > 30
    assert sorted(s.occurrence for s in bds) == list(range(1, len(bds) + 1))

    gen = torch.Generator()
    gen.manual_seed(3)
    sampled = sample_errors(specs, [ErrorSpec(name_pattern="BD", parameter="K1", amplitude=1e-3)], gen)
    assert len(sampled) == len(bds)
    perturbed = apply_errors(specs, sampled)
    changed = [s for s, p in zip(specs, perturbed) if s.k1 != p.k1]
    assert len(changed) == len(bds)
    assert all(s.name == "BD" for s in changed)


def _drift_lattice():
    return [ElementSpec("D", "drift", length=1.0)], Lattice(
        circumference_m=10.0, E0_eV=1e9, alpha_c=1e-4, Vrf_V=1e6, f_rf_Hz=5e8, U0_eV=100.0,
        tau_z_s=math.inf, sigma_delta0=0.0,
    )


def test_native_tracker_no_errors_has_no_rf_overrides():
    specs, params = _drift_lattice()
    tracker = NativeTracker(specs, params, dtype=torch.float64, device="cpu")
    assert tracker._has_rf_overrides is False


def test_native_tracker_phase_error_shifts_phi_s_eff_exactly():
    specs, params = _drift_lattice()
    tracker = NativeTracker(
        specs, params, dtype=torch.float64, device="cpu", errors=[SampledError("D", 0, "PHASE", 0.05)],
    )
    assert tracker._has_rf_overrides is True
    assert tracker._phi_s_eff_t.item() == pytest.approx(params.phi_s + 0.05)


def test_native_tracker_fiducial_shifts_phi_s_eff_by_minus_omega_rf_t_ref():
    """phi_s_eff = phi_s - omega_rf*t_ref_s + rf_error_delta -- see
    NativeTracker's own docstring for the derivation (elegant's own
    phase-relative-to-fiducial-arrival-time convention, phase_reference.c)."""
    specs, params = _drift_lattice()
    nominal = NativeTracker(specs, params, dtype=torch.float64, device="cpu")
    particles0 = torch.zeros((1, 6), dtype=torch.float64)
    particles0[0, 5] = 0.01  # a real, non-negligible delta -> non-negligible t_ref_s
    fiducial = nominal.fiducialize(particles0)
    assert fiducial.trajectory.shape == (2, 1, 6)  # 1 drift -> [entering, exiting]
    assert fiducial.t_ref_s != 0.0

    perturbed = NativeTracker(specs, params, dtype=torch.float64, device="cpu", fiducial=fiducial)
    expected = params.phi_s - params.omega_rf * fiducial.t_ref_s
    assert perturbed._phi_s_eff_t.item() == pytest.approx(expected)

