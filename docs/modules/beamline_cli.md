# `slipstream.beamline_cli`

[Source](../../src/slipstream/beamline_cli.py) · [Documentation index](../README.md)

## Physics

Uses `BeamlineStudy` source conversion and single-pass transport; CLI orchestration adds no separate physics.

## Usage

Run `python -m slipstream.beamline_cli study.yaml -o result.h5`. JSON scalar diagnostics are printed; `-o` resolves relative to the working directory. Input/configuration failures exit with status 2.

Backend initialization/progress messages go to stderr, keeping stdout valid
JSON for successful tracking runs.

Use `--set tracker.csr_transient=true --set tracker.csr_drift_model=retarded`
to override YAML settings; the study must also enable `csr` and
`fast_collective_effects`. Overrides share file validation and preserve
YAML-relative input paths. `--check` constructs the lattice/tracker without
loading particle sources, tracking or saving results. See
[model settings](../model_settings.md) for combinations and examples.

### API reference

```python
def main(argv=None) -> int: ...
```
