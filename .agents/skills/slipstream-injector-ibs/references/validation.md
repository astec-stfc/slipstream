# Injector IBS validation

```yaml
package_revision: 7a03349f39c5aff4e6e0ace81c3fba7e68d571f5
source_paths:
  - src/slipstream/tracking/native/injector_ibs.py
  - src/slipstream/tracking/native/injector_ibs_warp.py
  - src/slipstream/tracking/native/injector.py
test_paths:
  - tests/test_injector_ibs.py
  - tests/test_injector_ibs_warp.py
doc_paths:
  - docs/modules/tracking/native/injector_ibs.md
```

The base revision above predates this feature; source/tests are working-tree
additions. Use the current tree and verify imports before reproducing results.

See [the original implementation evidence](evidence.yaml) and
[the Warp/CLARA update evidence](warp_evidence.yaml) for actual results, source hashes,
environment and unavailable checks. Its validation claim is limited to the
declared synthetic operator and integration regime.

## Independent quantitative checks

From the package root, with this checkout installed or its `src` on PYTHONPATH:

```text
python -m pytest tests/test_injector_ibs.py tests/test_injector_ibs_warp.py -q
```

The Rutherford impact-integral oracle checks absolute rate normalization to
relative `3e-12` at bulk gamma 1, 5 and 100, including physical macro weight,
mu=m/2, SI units, rest density and rest time. The inverse-Langevin equation
uses absolute `3e-14` and relative `2e-11` bounds. Angular sampling means must
agree with independent expectations within six measured standard errors.
The homogeneous initial-diffusion oracle has a six-standard-error statistical
allowance plus a predeclared 0.5% weak-step/kinematic budget; its velocity
domain, physical density and total time stay fixed under particle/time
refinement. It is an operator test, not a whole-machine grid-convergence test.

Conservation uses summed lab relativistic energy and momentum at float64
roundoff tolerance. The optional Warp tests independently check absolute rates
at bulk gamma 1/5/100, common-draw impulses through variance 1000, masks,
conservation and non-default CUDA streams. The actual CUDA injector API is
also tested. Seeded replay is exact in the tested CPU/CUDA environment;
different devices do not share RNG streams. The tests also check a known-angle
pair, zero charge/relative motion/log, impact cutoff, unequal-weight and thermal
regime errors, masks, old/new YAML, default-off identity, actual injector
tracking, composition with native space charge, and unsupported tracking modes.

## Workbench study and remaining validation

With the optional parent checkout available, run its matched comparison:

```text
python -m slipstream_dev.examples.injector_ibs.compare_ibs injector.yaml --cell-size-m 5e-5 5e-5 1e-4 --seed 17 --output .tmp-injector-ibs/study.csv
```

Choose source, fields and scales for the actual machine. Inspect occupancy,
cutoff and maximum-variance diagnostics. Refine particle count, cell sizes,
time step and the space-charge solve, repeat collision seeds and compare
angular models. Hold physical cutoffs fixed when isolating discretization.
Compare common longitudinal slices, actual endpoint positions and populations.
Distinguish raw slice RMS from within-slice linearly dechirped RMS.

The package tests need no parent checkout, external executable or GPU. CUDA
cases skip explicitly when unavailable. CUDA graph capture of collisions,
float32, gradients, full SwissFEL/E-XFEL reproduction and calibration
of strongly coupled cathode emission are unvalidated or unsupported. Never
infer them from an eager collision test. Record actual run results and source
revisions when using this skill after changes.
