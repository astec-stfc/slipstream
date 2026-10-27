# `slipstream.beamline`

[Source](../../src/slipstream/beamline.py) · [Documentation index](../README.md)

## Physics

Composes single-pass canonical linac tracking with bunch generation, slice diagnostics and optional injector transport. Injector endpoints supply their actual reference energy and surviving charge; slope inputs are converted to canonical momenta.

### Limits and conventions

- `run()` requires `from_yaml`; input tensors have shape `(N,6)` and `tau` in metres.
- Repeated runs reset reference energy/charge and refresh energy-dependent bend offsets.

## Usage

Build `BeamlineStudy(specs, energy_eV, ...)` or use `from_lattice_file`/`from_yaml`. `generate_bunch` accepts entrance Twiss, geometric emittances, rms longitudinal widths and optional chirp. `track` returns `BeamlineResult`; `run` executes the configured source and optional HDF5 output.

`from_lattice_file(..., strict_import=True)` rejects unsupported import approximations. In permissive mode, pass `import_audit=[]` to collect JSON-compatible occurrence records; warnings identify affected declarations. See [the importer](tracking/native/lattice_import.md) for the supported subset and remaining model differences. This policy is independent of YAML schema validation.

### API reference

```python
class BeamlineStudy:
    ...
    def __init__(self, specs: list[ElementSpec], energy_eV: float, mc2_eV: float=ELECTRON_MASS_EV, *, device: str | torch.device | None=None, dtype: torch.dtype=torch.float64, bunch_charge_C: float=0.0, compile: bool=False, compile_mode: str | None=None, lsc: bool=False, csr: bool=False, pipe_radius_b_m: float | None=None, fast_collective_effects: bool=False, n_slices: int=50, **tracker_kwargs): ...
    def from_lattice_file(cls, lte_path: str | Path, root_name: str, energy_eV: float, mc2_eV: float=ELECTRON_MASS_EV, wake_base_dir: str | Path='.', *, strict_import: bool=False, import_audit: list[dict[str, object]] | None=None, **kwargs) -> 'BeamlineStudy': ...
    def from_yaml(cls, path: str | Path) -> 'BeamlineStudy': ...
    def run(self, output_path: str | Path | None=None) -> BeamlineResult: ...
    def generate_bunch(self, n: int, beta_x0: float, alpha_x0: float, epsx_m: float, beta_y0: float, alpha_y0: float, epsy_m: float, sigma_tau_m: float, sigma_delta: float, Dx0: float=0.0, Dpx0: float=0.0, Dy0: float=0.0, Dpy0: float=0.0, chirp_per_m: float=0.0, seed: int | None=None) -> torch.Tensor: ...
    def track(self, particles6: torch.Tensor, return_trajectory: bool=False, reference: dict[str, float] | None=None, metadata: dict | None=None) -> BeamlineResult: ...
    def track_with_injector(self, injector_or_deck: Injector | str | Path, n_particles: int | None=None, injector_kwargs: dict | None=None, **track_kwargs) -> BeamlineResult: ...
    def load_result(path: str | Path) -> BeamlineResult: ...
```

### `BeamlineStudy`

One `LinacTracker` (created once, tracked many times -- the same
amortization pattern `LinacTracker` itself documents), plus the
everyday conveniences: slice analysis after a pass (via
`analysis.slices`, imported lazily -- absent/`None` in the result
if that module can't be imported), a Twiss-matched bunch generator, and
optional injector composition.

`**tracker_kwargs` forwards straight to `LinacTracker` (`n_kicks`,
`errors`, `fiducial`, `dipole_collective_subcycle`,...) -- this
class adds no new tracking knobs of its own.

### `BeamlineStudy.from_yaml`

Build one reusable tracker from a strict YAML run configuration.

Paths are relative to the YAML file; lattice wake paths default to
the lattice directory. See `docs/beamline_studies.md` for the schema.
Particle sources are loaded when `run` is called.

### `BeamlineStudy.run`

Execute the configured source and beamline, optionally saving HDF5.

Generated sources default to seed 0 and reproduce on repeated runs
on the same device. `output_path` overrides `run.output`; an
explicit override is relative to the caller's working directory.
Injector/HDF5 sources supply their own reference energy and charge.

### `BeamlineStudy.generate_bunch`

A Twiss-matched bunch at this beamline's own entrance
(`tracking.distributions.gaussian_matched_6d`, converted to the
native canonical convention), ready to pass straight into
`track()`.

 matching at an arbitrary *downstream* point (via
`linac_twiss.linac_twiss_table`, which needs the exact same
beta/alpha/D inputs as this function -- a linac has no periodic
solve to shortcut that) isn't built here; propagate the desired
point's own Twiss with `linac_twiss_table` yourself and pass
those values in as `beta_x0`/... if a later point is needed.

### `BeamlineStudy.track_with_injector`

Run an `Injector` (an instance, or an ASTRA deck path built
via `Injector.from_astra`) to its exit distribution, ballistically
converted to this tracker's own canonical convention
(`injector_exit_to_canonical6`), then straight into `track()`.
The injector's own summed exit charge overrides this study's
`bunch_charge_C` for this call (an injector run has a real,
computed charge; a beamline built standalone usually doesn't), and
its own exit energy overrides `self.energy_eV` as the
`LinacTracker`'s reference energy -- `track()` must resume from
where the injector actually left off, not this study's own
construction-time value. Updates the reference-energy tensor in
place and refreshes energy-dependent magnet reference offsets.
