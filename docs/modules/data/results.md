# `slipstream.data.results`

[Source](../../../src/slipstream/data/results.py) · [Documentation index](../../README.md)

## Physics

Results preserve the caller's diagnostic definitions; persistence does not
perform tracking or coordinate conversion. Ring `TrackingResult` stores
`sigma_tau_s` and `mean_tau_s` in seconds. `BeamlineResult` stores canonical
`(x, px, y, py, tau, delta)` particles with tau in metres and total reference
energy in eV.

Centroid growth and beam-size growth are distinct observables. Store mean
coordinates as well as widths when studying coherent motion. Geometric
emittances describe phase-space area; beam widths also depend on optics.
The optional `epsx_m`/`epsy_m` arrays are measured emittances supplied by the
caller, rather than an equilibrium target inferred from the lattice.

`BeamlineResult.scalars()` returns population standard deviations of x, y,
tau and delta, plus output energy. `compare_reference()` uses `(got-ref)/ref`
for nonzero references and an absolute difference for a zero reference.

## Usage

### Ring and beamline results

Construct `TrackingResult` from turn-indexed arrays and run metadata, then
use `save(path)` or `TrackingResult.load(path)` for HDF5 persistence.
Optional diagnostic arrays can be absent in older files. `summarize(last_n)`
returns the mean and standard deviation of the stored delta and tau widths
over the selected samples.

`BeamlineResult` also saves optional particle/energy trajectories, slice
summaries and metadata. Reference differences are stored as named entries
under `metadata/reference_diff`. See the [file-format guide](../../formats.md)
and [beamline studies](../../beamline_studies.md) for complete workflows.

```python
from slipstream import BeamlineResult

result = BeamlineResult.load("result.h5")
print(result.scalars())
print(result.compare_reference({"sigma_tau_m": 1e-3}))
```

### Streaming scalar rows

`StreamingH5Writer(path, attrs=..., metadata=..., flush_every=...)` creates
extendable float64 datasets through `append(**values)`. Supply the same
column names on every row to keep dataset lengths consistent. If a file
will be loaded through `TrackingResult.load`, supply that reader's required
attributes and datasets. Use the writer as a context manager or close it;
periodic flushing makes partial rows available on disk.

### Coordinate snapshots

`CoordWriter.maybe_append(turn, tracker)` writes only at multiples of its
configured interval. The supplied state object must expose torch tensors
`tau`, `delta` and an `optics` attribute. With transverse optics present it
also needs `x`, `xp`, `y`, `yp`. The package's tensor-based `NativeTracker`
has no automatic `coord_writer` callback; this adapter requires a compatible
state object and explicit calls.

`sparse_interval` selects particles by index. Datasets retain the supplied
coordinate units. Selection indices are converted to a LongTensor on the
state's device before indexing, and snapshots are detached and transferred
to CPU for HDF5 writing. Close the writer or use it as a context manager.
