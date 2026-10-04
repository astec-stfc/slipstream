# `slipstream.wakefields.impedance`

[Source](../../../src/slipstream/wakefields/impedance.py) · [Documentation index](../../README.md)

## Physics

Stores frequency-domain longitudinal impedance as real and imaginary ohms on a shared Hz grid.

### Limits and conventions

- A container does not establish the validity of the solver or impedance data that produced it.

## Usage

Construct `Impedance(freq_Hz, ReZ_ohm, ImZ_ohm)`, save/load CSV or use `from_sdds_file`. Wake generation consumes this container.

### API reference

```python
class Impedance:
    ...
    def save(self, path: str | Path) -> None: ...
    def load(cls, path: str | Path) -> 'Impedance': ...
    def from_sdds_file(cls, path: str | Path, freq_col: str='Freq', re_col: str='RealZ', im_col: str='ImagZ') -> 'Impedance': ...
```

### `Impedance.from_sdds_file`

Load Z(f) from a column-per-part ASCII SDDS file -- elegant's own
ZLONGIT/ZTRANSVERSE impedance-table input format. `re_col`/
`im_col` pick which columns hold the real/imaginary parts:
ZLONGIT splits Z_parallel into two separate single-column files
(call once per file with the same `re_col`/`im_col` pair --
whichever name is missing from a given file reads back as zero, so
the two calls' results can be combined directly); ZTRANSVERSE keeps
all four of RealX/ImagX/RealY/ImagY in one file (call once per
plane with the matching pair).
