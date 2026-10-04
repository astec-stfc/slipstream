# `slipstream.wakefields.multilayer.yokoya`

[Source](../../../../src/slipstream/wakefields/multilayer/yokoya.py) · [Documentation index](../../../README.md)

## Physics

Interpolates geometric elliptic form factors for round-result approximations. It reaches unity for round chambers and known parallel-plate limits for extreme flatness.

### Limits and conventions

- The smaller axis is assumed vertical; swap transverse components for a horizontally small aperture.
- Frequency-independent factors approximate intermediate aspect ratios and cannot model direction-dependent wall materials.

### `Model and numerical conventions`

The table and the `ratio`/interpolation recipe below are transcribed
directly from IW2D's own official Python library (Elias Metral's group,
`IW2D/PYTHON_codes_and_scripts/Impedance_lib_Python/Impedance.py`'s
`Yokoya_elliptic`/`apply_Yokoya_elliptic`, table data credited there to
Elias/USPAS) -- not re-derived. This is a real, standard approximation used
throughout accelerator impedance work when a full 2D flat/elliptic solver
isn't available or needed, not an ad-hoc shortcut: it reduces to the exact
round-chamber result when the two semi-axes are equal, and to the known
closed-form infinite-parallel-plate limit (long=1, dipy=pi^2/12,
dipx=pi^2/24, quady=pi^2/24, quadx=-pi^2/24) as the aspect ratio goes to
zero -- both checked in `tests/test_yokoya.py`.

## Usage

Call `yokoya_elliptic_factors(small_semiaxis_m, large_semiaxis_m)` for `(long,xdip,ydip,xquad,yquad)`.

### API reference

```python
def yokoya_elliptic_factors(small_semiaxis_m: float, large_semiaxis_m: float) -> tuple[float, float, float, float, float]: ...
```

### `yokoya_elliptic_factors`

(long, xdip, ydip, xquad, yquad) Yokoya factors for an elliptic
chamber, small axis assumed vertical (`Impedance.py`'s
`Yokoya_elliptic` -- swap the returned xdip/ydip and xquad/yquad
yourself if the small axis is actually horizontal, same as
`apply_Yokoya_elliptic`'s `orientation='H'` swap does). `long` is
always 1 (no correction) -- not looked up, matching the source, since
the table doesn't even carry a longitudinal column.

## Numerical and implementation notes

### `Module constants`

Columns: ratio=(large-small)/(large+small), dipy, dipx, quady, quadx.
ratio=0 -> round (all -> 1,1,0,0); ratio->1 -> infinite flat plates
(dipy->pi^2/12=0.8225, dipx->pi^2/24=0.4112, quady->pi^2/24, quadx->-pi^2/24).
