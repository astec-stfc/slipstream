"""Test tiers, so development can run only the fast physics checks.

Every test file belongs to exactly one tier, applied as a pytest marker:

physics   the code's numbers against closed forms, conserved quantities, independent references or finite
          differences. If one of these fails, the physics output changed. Run this tier while developing.
workflow  a specific use of the code: LOCO, the example studies, beamline configs and the CLI, reconstruction.
unit      one function or interface: file formats, importers, serialization, scan and placement APIs, smoke checks.

``slow`` marks individual tests that take more than a few seconds, in any tier.

    python -m pytest -m "physics and not slow"   # fast physics regression, about a minute
    python -m pytest -m "not slow"               # everything except the slow tests
    python -m pytest                             # everything (CI)

A new test file must be added to TIERS below; collection stops with a message if it is missing.
"""
import pytest

TIERS = {
    "physics": """
        analysis_slices apertures beamline_radiation collective_gradients collective_kicks corrector
        csr_transient_drift distributions double_single generator_ellipse ibs injector_crest
        injector_emission_step injector_ibs injector_ibs_warp injector_screen injector_wake linac_setup
        linac_twiss magnet_fields multilayer_impedance native_bend_radiation native_drift_reference
        native_errors native_fields native_gun photocathode ramp rfcw_wake_scaling ring_ibs_damping ring_nonlinear
        ring_outputs ring_per_kind_kicks ring_phase3 ring_physics sector_bend_ode space_charge_3d tdc wake_windows
        wakefields_longrange wakefields_shortrange warp_collective_kernels
    """,
    "workflow": """
        beamline_study diamond_example fit_structure laser_image_fit laura_injector mls_example native_loco
        nonlinear_optimisation photocathode_example reconstruction user_examples
    """,
    "unit": """
        astra_solenoid_placement compile_env data injector_3d_padding lattice lattice_import_audit
        linac_scan_parameters package_smoke placement_tensors
    """,
}
_TIER_OF = {f"test_{name}.py": tier for tier, names in TIERS.items() for name in names.split()}


def pytest_collection_modifyitems(items):
    for item in items:
        tier = _TIER_OF.get(item.path.name)
        if tier is None:
            raise pytest.UsageError(f"{item.path.name} has no tier: add it to TIERS in tests/conftest.py")
        item.add_marker(tier)
