# Validation

```yaml
package_revision: f18488d46f22bb627d2984823904d7e279f48bf3
source_paths:
  - src/slipstream/tracking/native/radiation.py
  - src/slipstream/tracking/native/linac.py
  - src/slipstream/tracking/native/lattice_import.py
  - src/slipstream/tracking/native/tracker.py
test_paths:
  - tests/test_native_bend_radiation.py
doc_paths:
  - docs/modules/tracking/native/radiation.md
  - docs/modules/tracking/native/linac.md
```

From the standalone package root, using this checkout on PYTHONPATH:

```
python -m pytest tests/test_native_bend_radiation.py -q --basetemp=.tmp_radiation_tests
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
