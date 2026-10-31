# Validation

```yaml
package_revision: eb09d9648ecfc8d3240275d629773339d45ba227
source_paths:
  - src/slipstream/tracking/native/radiation.py
  - src/slipstream/tracking/native/linac.py
  - src/slipstream/tracking/native/lattice_import.py
  - src/slipstream/tracking/native/tracker.py
  - src/slipstream/analysis/beamline_radiation.py
test_paths:
  - tests/test_native_bend_radiation.py
  - tests/test_beamline_radiation.py
doc_paths:
  - docs/modules/tracking/native/radiation.md
  - docs/modules/tracking/native/linac.md
  - docs/modules/analysis/beamline_radiation.md
```

From the standalone package root, using this checkout on PYTHONPATH:

```
python -m pytest tests/test_native_bend_radiation.py tests/test_beamline_radiation.py -q --basetemp=.tmp_radiation_tests
```

Numerical quadrature of the Bessel photon spectrum supplies independent
first/second raw moments at 500 GeV. Deterministic kick agreement: 2e-8
relative (constant precision); variance agreement: 1.2% with 200,000 seeded
particles. A 30,000-particle public single-bend run checks variance to 2.5%
and mean loss to 1.5%, allowing shot noise and small energy evolution.
Scaling, zero field/length, collinear recoil, repeatability, parser integration,
zero bunch charge and unsupported-configuration guards are also checked.

The optional ILC workbench study supplies independent matrix/covariance
propagation and live elegant tracking; it is not required by portable tests.
Only measured CPU float64 eager results support the current claim. No CUDA,
compiled, gradient, ring-equilibrium or discrete-photon validation is implied.

Linear expected-moment AD is validated separately: independent 80-point
Gauss-Legendre/SciPy-exponential noise quadrature agrees to 2e-12 relative
(1e-24 absolute); refinement preserves covariance to 1e-12 relative;
drift/survey identities and gamma/curvature scaling pass. Complete native
API gradients in lengths, bends, faces and quadrupoles agree with central
differences at 1e-4, 1e-5 and 1e-6 steps to 1e-5 relative with 2e-8 absolute
floor. Four focused tests pass on CPU float64 eager. The recorded revision
is the inspected base; this capability also requires the accompanying
working-tree covariance implementation and tests. No CUDA or compiled
covariance validation is implied.
