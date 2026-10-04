# `slipstream.beamline_cli`

[Source](../../src/slipstream/beamline_cli.py) · [Documentation index](../README.md)

## Physics

Uses `BeamlineStudy` source conversion and single-pass transport; CLI orchestration adds no separate physics.

## Usage

Run `python -m slipstream.beamline_cli study.yaml -o result.h5`. JSON scalar diagnostics are printed; `-o` resolves relative to the working directory. Input/configuration failures exit with status 2.

### API reference

```python
def main(argv=None) -> int: ...
```
