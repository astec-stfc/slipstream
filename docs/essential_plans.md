# Essential-plan implementation in this package

Assessment dated 2026-10-04. The source plans are the five files in the
parent workbench's `slipstream-dev/plans/essential/`. This assessment adapts
their milestones to the installable package; it does not change those plans.

Several plans describe historical workbench infrastructure that is absent
here: `Study`, pySTABLE, an agentic tool server, executable drivers, CLARA
machine files, and an `optimisation` package with `Problem`/`run_moo`.
The package reads elegant and ASTRA inputs locally. Executable-driven
comparisons belong in the parent workbench, as described in
[native tracking](tracking_native.md).

## E3: native beamline facade

The existing `BeamlineStudy` in `src/slipstream/beamline.py` already provides
lattice-file construction, entrance Twiss-matched generation, single-pass
tracking, and injector composition. `BeamlineResult` provides endpoint
diagnostics, optional trajectories, slice summaries, and HDF5 persistence.
These correspond to the plan's completed facade phases.

**Implemented milestone: Phase 3, YAML run configuration.**
`BeamlineStudy.from_yaml(...)`, `run()`, and a thin CLI compose generated,
package-HDF5, single-page SDDS, and native ASTRA-injector sources. Paths are
local to the YAML file; lattice wakes default to the lattice directory.
Explicit canonical conversion, source energy/charge handoffs, deterministic
generated reruns, direct-tracker parity, and reference HDF5 persistence have
regression tests. A self-contained synthetic example is in
`examples/beamline/`. See [beamline studies](beamline_studies.md) for the
exact scalar schema and source formats. SDDS remains optional. References
are scalar mappings or YAML scalar files; `.twi`/`.sig`/ASTRA emit references
still need preprocessing. Real machine examples are not present here.

The plan's Phase 4 agentic wiring requires a host tool server that this
checkout does not contain. It is a later integration milestone, not a
reason to add that infrastructure to the native package.

## E4b: incoherent synchrotron radiation in bends

Per-bend loss and quantum excitation are absent from `tracking/native/elements.py`.
Ring-average damping and excitation already exist in
`tracking/native/ring_physics.py`; this plan must preserve that behavior and
avoid counting the same radiation twice.

The next milestone is the model/constants decision followed by a
deterministic float64 single-bend energy-loss regression. The plan's
excitation formula is schematic and needs a complete physical derivation.
Its proposed ring default-on option also needs reconciliation with its
flag-off regression requirement before changing tracker defaults.
Integration then needs seeded excitation, bend slicing, reference-energy
bookkeeping, equilibrium validation, and separately qualified compiled/CUDA
checks. CPU eager tests cannot establish those other paths.

## E4c: CSR continuation

`tracking/native/collective.py` already contains unshielded steady-state
CSR kernels, including a fixed-grid fast path. `lattice_import.py` and
`linac.py` already carry the most recent bend's radius into downstream
`CSRDRIFT` elements and apply a live overtaking-length cutoff. This is an
approximation with one upstream bend, not full radiation-history transport.

The next gate is the plan's matched-input steady-state elegant comparison,
with a converged binning study in the parent workbench and a quantitative
CPU regression in this package. Current CUDA/Warp primitive-parity tests
are not that physical validation. Finite-gap shielding, overlapping
multi-bend history, and entrance transients remain open. New models must
retain the existing unshielded default and test fast/reference agreement.

## E4d: validation closures

`tests/test_native_errors.py` checks the fiducial RF-arrival-time convention
quantitatively; an element-by-element live-elegant comparison remains open.
`TransverseWake` exists, but this checkout has neither the referenced
`SingleBunchTracker` nor the historical TMCI comparison scripts/test file.
The reported CLARA ASTRA beam-size residual has no reproducible deck or
field-map fixture in this checkout.

The next milestone is an analytic transverse-wake sign anchor using the
available native tracker, with matched units and inputs. TMCI threshold,
fiducial elegant comparison, and the ASTRA residual investigation need their
respective reference datasets/runners in the workbench. Change physical
conventions only when a quantitative failing regression proves the error.

## E5: matching and optimization

Generic Twiss/endpoint objectives, configurable knobs, and a beamline
matching driver are absent. Slice diagnostics and `LinacTwissTable` are
available. Related specialized functionality already exists in
`analysis/nonlinear_opt.py`: ring `KnobSet`, bounded searches, constraint
losses, and Adam-based multi-fidelity search, with regression tests.

After the E3 configuration interface, the first milestone is a small
objective/knob protocol and constrained synthetic linac match recovering a
known solution from a perturbed start. Reuse existing search helpers where
they fit; establish deterministic evaluation and isolated tracker rebuilds.
Generic caching, result persistence, ring-equilibrium matching, and agentic
wiring follow later. Full-tracking gradients and compiled reuse must be
demonstrated for the chosen path before they are promised.
