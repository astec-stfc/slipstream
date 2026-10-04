# `slipstream.sdds_io`

[Source](../../src/slipstream/sdds_io.py) · [Documentation index](../README.md)

## Physics

SDDS values retain the units and meanings of their reader; this module provides shared access to the binding, rather than physical conversion.

### Limits and conventions

- The SDDS index registry is process-global: release the object before releasing the shared lock.

## Usage

Use with sdds_object(index=0) as (sdds, obj): for every SDDS operation; the context manager yields both the imported binding module and its SDDS object. The optional sdds binding must be installed.

### API reference

```python
def sdds_object(index: int=0): ...
```

### `sdds_object`

The context manager yields `(sdds_module, object)` while holding the process-wide lock. Convert borrowed data to NumPy inside the block. The object's index and registry entry are released explicitly before unlocking; deleting the local binding alone would leave the caller's reference alive. The registry interaction uses the installed binding's private `_used_indices` set.

```python
from slipstream.sdds_io import sdds_object

with sdds_object() as (sdds, obj):
    obj.load(str(path))
    values = np.array(obj.columnData[0][0])
```
