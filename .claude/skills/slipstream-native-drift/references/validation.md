# Native straight drift validation

Paths are relative to the standalone Slipstream package root:

```yaml
package_revision: 7a03349f39c5aff4e6e0ace81c3fba7e68d571f5
source_paths:
  - src/slipstream/tracking/native/elements.py
  - src/slipstream/tracking/native/linac.py
  - src/slipstream/tracking/native/lattice_import.py
test_paths:
  - tests/test_native_drift_reference.py
doc_paths:
  - docs/modules/tracking/native/elements.md
```

## Quantitative oracle

`tests/test_native_drift_reference.py::velocity_reference` integrates dimensional
velocities and subtracts the reference arrival time. Compare against the native
endpoint with rtol `2e-12`, atol `2e-14` metres for position/time coordinates
and exact momentum constancy. These tolerances allow float64 cancellation in
subtracting reference flight times while being far below the millimetre-scale
delays in the selected nonzero test particles. They do not bound a physical
model error or justify float32/near-grazing precision.

Three-particle inputs include an on-axis reference, positive and negative
off-momentum particles, and both signs of transverse momentum. Cases use total
reference energy 2 MeV and 100 MeV with the package electron rest energy, and
1.2 GeV with a representative heavy-particle rest-energy parameter of
938272088.16 eV. Length is 1.7 m. No stochastic sampling is involved.

## Reproduce

Use an interpreter with the package dependencies and pytest, from the package
root. Verify `slipstream.__file__` resolves into this checkout. If an existing
editable install points elsewhere, set this process's PYTHONPATH to this
checkout's `src` directory before starting Python.

```text
python -m pytest tests/test_native_drift_reference.py -q
```

Required checks are dimensional free-flight endpoints, zero length, length
scaling/subdivision, unchanged momenta and the public linac entry point with
fixed reference. A deliberately doubled tau increment must fail the oracle
despite remaining finite. The file has six passing test cases on the recorded
CPU float64 eager environment. Rerun after changes rather than treating this
historical result as a fresh pass.

The drift is exact for its declared model, so subdivision invariance is the
appropriate refinement test. There is no mesh, smoothing or particle-count
estimator to converge. CUDA, compilation, Warp, gradients, external codes and
throughput are outside this skill's demonstrated regime. Parent workbench
evidence may exist, but is not needed to reproduce these package tests.
