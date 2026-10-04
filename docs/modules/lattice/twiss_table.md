# `slipstream.lattice.twiss_table`

[Source](../../../src/slipstream/lattice/twiss_table.py) · [Documentation index](../../README.md)

## Physics

Stores element-local Twiss/dispersion with explicit segment lengths for length-weighted IBS averaging.

### Limits and conventions

- Use explicit physical segment lengths as weights rather than inferring them from adjacent positions.
- This array-backed schema uses CSV rather than scalar dataclass YAML.

## Usage

Use `TwissTable` arrays, CSV `to_csv`/`from_csv` or `from_madx_dframe`. Native `periodic_twiss_table` can populate the same schema.

### API reference

```python
class TwissTable:
    ...
    def to_csv(self, path: str | Path) -> None: ...
    def from_csv(cls, path: str | Path) -> 'TwissTable': ...
    def from_madx_dframe(cls, df: pd.DataFrame) -> 'TwissTable': ...
```

### `TwissTable`

One row per element (or slice) around the ring. `length_m` is
each row's weight for `ibs.element_averaged_growth_rates`'s
length-weighted average -- a MAD-X `TWISS` table's own `L` column
gives this directly (exact segment lengths, not inferred from
consecutive `s_m` differences). Field names match
`TransverseOptics`/`ibs.py`'s own per-plane naming
(`beta_x_m`/`alpha_x`/`D_x_m`/`Dp_x`, not MAD-X's
`betx`/`alfx`/`dx`/`dpx`).

### `TwissTable.from_madx_dframe`

Adapts cpymad's `Madx.twiss().dframe()` column names (MAD-X's
own lowercase convention: `s`, `l`, `betx`, `bety`,
`alfx`, `alfy`, `dx`, `dy`, `dpx`, `dpy`) to this
table's naming. Not imported by `ibs.py` itself, which stays
cpymad-free (pure math, per its own model description) -- only
example scripts that already depend on cpymad exercise this.
