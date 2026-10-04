# `slipstream.tracking.native.linac_twiss`

[Source](../../../../src/slipstream/tracking/native/linac_twiss.py) · [Documentation index](../../../README.md)

## Physics

Propagates nonperiodic linear Twiss/dispersion or extracts ensemble statistics from tracked trajectories. RF matrices differentiate slope-input/slope-output transport while reference energy advances from the design particle.

### Limits and conventions

- RF output canonical-to-slope Jacobians include the changed momentum scale.
- Do not differentiate ensemble-mean reference recentering in a single-particle optics matrix.

### `Model and numerical conventions`

Per-element Twiss for a single-pass linac beamline
(`tracking.native.linac.LinacTracker`) -- two independent sources, both
returned in the same `LinacTwissTable` shape for direct, row-for-row
comparison: `linac_twiss_table` (linear-optics matrix propagation from
an explicit, non-periodic initial condition -- comparable to real
elegant's own `&twiss_output matched=0` `.twi`) and
`twiss_from_trajectory` (statistics extracted from the actual
multi-kick particle tracking -- comparable to real elegant's own tracked
`.sig` file). A new, linac-specific table rather than a reuse of the
ring-only `lattice.optics.TransverseOptics`/`lattice.twiss_table.TwissTable`
-- a linac's own reference energy varies element-to-element, and there is
no periodicity/tune to speak of.

(2) An RF element's linear R matrix must never differentiate
`elements._match_energy_to_centroid`'s ensemble-mean `change_p0`
recentering (a single-particle Jacobian's own "mean" is degenerate) -- the
reference energy instead advances via a single on-axis/design-particle
forward evaluation, matching elegant's own `matched=0` `pCentral0`
convention.

### `_rf_forward_and_R`

`(R, energy_eV_new)` for one RF-kind `spec` at its own entrance
`energy_eV` -- `R` is the Jacobian of `elements._rf_cavity_kick`
at `state6=0`, always with `change_p0=False` (must never
differentiate the ensemble-mean recentering, see model description
point 2). `energy_eV_new` is derived from that same forward call's
output `delta` at `state6=0` (`P1=P0*(1+delta)`,
`energy_eV_new=mc2*sqrt(P1^2+1)`) -- the single reference particle's
new energy, matching elegant's own `matched=0` `pCentral0` advance.
No wake args -- elegant's own linear `.twi` calc ignores wakes/charge.

Uses `spec.n_kicks` (the real per-sub-kick count), not a hardcoded
`1`: elegant's own `matched=0` linear-optics calculation resolves
the element's real `N_KICKS` for the reference-energy advance (see
docs/tracking_native.md for the live-elegant cross-check numbers).

`R` is the Jacobian of a **slopes-in/slopes-out** wrapper around
`_rf_cavity_kick` (converting through canonical only internally),
not a bare canonical-in/canonical-out Jacobian -- see model description
point 1 for why. Confirmed against a live-elegant finite-difference
R-matrix to ~1e-9 relative for CLARA's own L02 parameters.

### `_plane_twiss_from_ensemble`

Dispersion-corrected `(beta, alpha, D, Dp)` from one plane's real
particle ensemble -- `D = cov(u,delta)/var(delta)`, removed from
`u`/`up` before computing the geometric emittance/beta/alpha.
Matches elegant's own dispersion-corrected `ecx`/`ecny`/
`betaxBeam`/`alphaxBeam` `.sig` columns, not its uncorrected
tracking-derived ones -- the standard treatment for a beam with a real
energy-position correlation (present after any RF cavity, substantial
after a bunch compressor).

## Usage

Use `linac_twiss_table` with explicit entrance optics or `twiss_from_trajectory` for tracked statistics; `LinacTwissTable.save`/`load` persists results.

### API reference

```python
class LinacTwissTable:
    ...
    def save(self, path: str | Path, group: str='/') -> None: ...
    def load(cls, path: str | Path, group: str='/') -> 'LinacTwissTable': ...
def linac_twiss_table(specs: list[ElementSpec], energy_eV: float, mc2_eV: float, beta_x0: float, alpha_x0: float, beta_y0: float, alpha_y0: float, Dx0: float=0.0, Dpx0: float=0.0, Dy0: float=0.0, Dpy0: float=0.0, n_kicks: int=16, n_kicks_bend: int=64, integration_order: int=4) -> LinacTwissTable: ...
def twiss_from_trajectory(trajectory: torch.Tensor, energy_trajectory: torch.Tensor, specs: list[ElementSpec]) -> LinacTwissTable: ...
```

### `LinacTwissTable`

One row per injection point + element exit (`len(specs)+1` rows,
matching `LinacTracker.track(..., return_trajectory=True)`'s own
trajectory convention exactly -- row 0 is the injection condition,
row `i+1` is the state after `specs[i]`). Field names deliberately
mirror `lattice.twiss_table.TwissTable`'s own (`beta_x_m`,
`alpha_x`, `D_x_m`, `Dp_x`,...) for familiarity, but this is a
genuinely different type: a linac's own reference energy varies
element-to-element (`energy_eV`, absent from the ring-only
`TwissTable`) and there is no periodicity/tune to speak of.

### `LinacTwissTable.save`

Append-mode HDF5 write (`data.results.TrackingResult`'s own
one-dataset-per-field idiom) -- `group=` lets a linear-optics
table and a tracking-derived table share one file
(`table.save(path, group="linear")`, a second
`table.save(path, group="tracking")`) without one call
clobbering the other. Existing datasets under the same
name/group are replaced (safe to call twice against the same
path while iterating), not merged.

### `linac_twiss_table`

Linear-optics Twiss propagated from an explicit, non-periodic
initial condition through `specs` in order -- the linac analogue of
`optics.periodic_twiss_table`, minus the periodic-matching solve and
generalized to a running reference energy (an RF-kind element changes
it). Reuses `optics._element_R`/`optics._propagate_twiss` for
every magnetic-kind element; RF-kind elements go through
`_rf_forward_and_R` instead. `n_kicks`/`n_kicks_bend` default to
the same high slice counts `periodic_twiss_table` uses -- a one-time
cost that only affects this table's accuracy, never any tracking loop.

Returns `len(specs)+1` rows: row 0 is the given injection condition
unchanged, row `i+1` the state after `specs[i]`'s transport --
matching `LinacTracker.track(..., return_trajectory=True)`'s own
convention, and elegant's per-element `.twi` row convention.

### `twiss_from_trajectory`

Dispersion-corrected Twiss extracted from an actual tracked particle
ensemble, at each step of ``LinacTracker.track(particles6,
return_trajectory=True)`'s own `(len(specs)+1, N, 6)`` state
trajectory (plus its matching energy trajectory) -- comparable to real
elegant's tracked `.sig` file, not its linear `matched=0` `.twi`
(see `linac_twiss_table` for that).

Runs entirely outside `torch.compile` -- a plain Python loop over the
already-materialized trajectory (matches this repo's "diagnostic path
bypasses compile" convention elsewhere); the compiled tracking that
produced this trajectory already ran with zero graph breaks.
