# Choosing models and settings

## Physics

These settings select existing tracker models. Changing a model can change
the physical approximation, numerical resolution and runtime. CPU/CUDA and
compilation choices select execution paths; they do not enable a physical effect.

CSR and LSC are opt-in for the single-pass tracker and require nonzero bunch
charge. CSR also requires a CSR-active element: `CSRCSBEND`/`CSRCSBEN` with
`CSR=1`, or `ElementSpec(..., csr=True)` in Python. A plain `CSBEND` is not
CSR-active. Downstream drifts can carry the upstream bend's CSR geometry;
other element types end that carry.

## Usage

### Change settings without editing a study file

Use repeated `--set KEY=VALUE` arguments. Values use YAML syntax:
`true`/`false`, numbers, strings, or `null`. The last occurrence of a key wins.
Overrides pass through the same validation as the file:

```console
python -m slipstream.beamline_cli docs/examples/csr.yaml --set tracker.csr_transient=true --set tracker.csr_drift_model=retarded
python -m slipstream.beamline_cli examples/beamline/study.yaml --set bunch.n=128 --set run.return_trajectory=false --check
```

The same choices work in Python. Values here are Python objects, so use
`True`/`False` rather than strings:

```python
from slipstream import BeamlineStudy

steady = BeamlineStudy.from_yaml("docs/examples/csr.yaml")
transient = BeamlineStudy.from_yaml(
    "docs/examples/csr.yaml",
    overrides={"tracker.csr_transient": True, "tracker.csr_drift_model": "retarded"},
)
steady_result = steady.run()
transient_result = transient.run()
```

Construct a new study/tracker when changing models. Several switches are
baked into element parameters and compiled functions during construction;
assigning flags on an existing tracker does not rebuild them. Reuse that
tracker for repeated runs with the same settings. Overrides do not edit the
YAML file, and results record the effective configuration in metadata.

### CSR bend and drift models

All rows below assume `csr: true` and nonzero `bunch_charge_C`. These are
independent bend and downstream-drift choices, not one global CSR model:

| Bend model | `fast_collective_effects` | `csr_transient` | Notes |
| --- | --- | --- | --- |
| Pairwise steady state | `false` (default) | `false` (default) | KDE/pairwise calculation; quadratic work in particle count |
| Binned steady state | `true` | `false` | Fixed-bin convolution; uses bend `BINS` and `SG_HALFWIDTH` |
| Binned entrance transient | `true` | `true` | Wake evaluated at accumulated bend angle; `N_SLICES` resolves its evolution |

| Drift model (`csr_drift_model`) | Required settings | Meaning |
| --- | --- | --- |
| `overtaking` (default) | Either steady-state kernel, or binned transient bends | Legacy steady-state drift kick capped at the live overtaking length |
| `stupakov` | `fast_collective_effects: true` | Closed-form exit-drift model evaluated in `csr_drift_dz` steps |
| `retarded` | `fast_collective_effects: true` | Retarded 1D exit-drift field evaluated in `csr_drift_dz` steps |

`csr_transient: true` and the two non-default drift models require both
`csr: true` and `fast_collective_effects: true`. To disable CSR on a study
using these settings, also set `csr_transient=false` and
`csr_drift_model=overtaking`. This keeps incompatible combinations visible.
`fast_collective_effects` selects the binned approximation for **both LSC and
CSR**; changing it also changes LSC when `lsc: true`.

The [CSR example](examples/csr.yaml) starts with binned steady state.
Compare it with entrance-transient bends and either drift model using the
command above. To select pairwise steady state from that example:

```console
python -m slipstream.beamline_cli docs/examples/csr.yaml --set tracker.fast_collective_effects=false
```

`N_SLICES`, `BINS` and `SG_HALFWIDTH` belong to bend definitions in the
lattice; they are not tracker YAML keys. For Python-built lattices use
`ElementSpec.csr_n_slices`, `csr_bins` and `csr_sg_halfwidth`.
Drift step length is `ElementSpec.csr_drift_dz` (default 0.01 m).
The elegant reader currently does not import `CSRDRIFT`'s `DZ` attribute:
strict import rejects it and permissive import warns. To change drift step
length, build or replace the drift specs in Python before constructing a
study. Converge these resolutions for the bunch being studied. See the
[collective reference](modules/tracking/native/collective.md) for model
limits and the [linac reference](modules/tracking/native/linac.md) for
subcycling and reference-energy behavior.

### Other common switches

| Task | Setting | Scope and dependencies |
| --- | --- | --- |
| Enable linac LSC | `lsc=True`, nonzero `bunch_charge_C` | `LinacTracker`/`BeamlineStudy`; finite-length elements; `pipe_radius_b_m` optionally bounds its transverse scale |
| Keep the linac reference fixed | `always_change_p0=False` | Otherwise the reference follows the bunch centroid; RF specs may separately carry `change_p0` |
| Change ring RF placement | `rf="lumped"` or `"per_element"` | `NativeTracker`; per-element RF follows RFCA positions |
| Change ring IBS | `ibs=True`, `ibs_model="bane"` or `"bjorken-mtingwa"` | Requires `optics`; Bjorken–Mtingwa also requires explicit `ibs_coulomb_log` |
| Change injector space charge | `injector.space_charge.enabled`, `.geometry`, `.mesh` | `cylindrical`/`3d` geometry; `adaptive`/`fixed` mesh; see [injector settings](tracking_native.md#space-charge-and-mesh-sizing) |

These are constructor or documented settings-object choices. The beamline
YAML schema covers one-pass studies; it is not a ring or general injector
configuration schema. Python-only objects such as fiducial references and
sampled errors belong in the Python API. For injector tracking without
compilation use `injector.track(use_compile=False)`; linac/ring constructors
use `compile=False`. For strict elegant import in YAML set
`lattice.strict_import: true`.
