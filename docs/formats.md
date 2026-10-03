# File formats

Slipstream reads the machine files it needs in Python. It does not start an
elegant or ASTRA executable, so those programs are not package dependencies.

## elegant

`slipstream.tracking.native.lattice_import.parse_elegant_lattice` reads the
supported subset of elegant `.lte` definitions and returns an ordered list of
`ElementSpec` objects:

```python
from slipstream import parse_elegant_lattice

specs = parse_elegant_lattice("beamline.lte", "BEAMLINE")
```

Pass the returned list to `NativeTracker` for ring tracking or
`LinacTracker` for one-pass tracking. The parser does not evaluate external
files or execute elegant.

## ASTRA

`slipstream.tracking.native.astra_io` reads ASTRA namelists and particle
distribution files. `Injector.from_astra("injector.in")` builds an injector
from an ASTRA deck and keeps referenced field/distribution paths relative to
the deck. `Injector.save()` and `Injector.load()` provide a YAML form for
reusing a parsed machine without reparsing the original input.

Unsupported ASTRA options are reported as warnings rather than silently
treated as implemented physics. Running ASTRA remains outside this package.

## SDDS

SDDS-backed wakefield readers are optional. Install the extra when using them:

```bash
python -m pip install "slipstream-accel[sdds]"
```

Executable-driven validation and result readers remain in the parent research
workbench.
