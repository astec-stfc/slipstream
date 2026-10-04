# `slipstream.tracking.native.lattice_import`

[Source](../../../../src/slipstream/tracking/native/lattice_import.py) · [Documentation index](../../../README.md)

## Physics

Converts supported elegant lattice declarations into native `ElementSpec` physics, including magnetic elements, RFCA/RFCW, H/V correctors and CSR-related settings. Unknown elements with nonzero length become drifts.

### Limits and conventions

- This is a documented syntax subset, not an elegant macro evaluator.
- Repeat-count and reverse-element syntax are unsupported.
- Unknown-element drift fallback must not be mistaken for support of that element's original physics.
- RAMPRF waveform behavior belongs to `RampProfile`, not static RFCA parameters.

### `Model and numerical conventions`

Minimal, covering exactly the element/line syntax real CLARA/Diamond-II
lattices use: `NAME: TYPE, PARAM=VAL,...;` element definitions
(`DRIFT`, `KQUAD`/`QUAD`/`QUADRUPOLE`, `CSBEND`/`CSBEN`/`SBEND`, `KSEXT`/`SEXT`/`SEXTUPOLE`; anything else with
a nonzero `L=` becomes a plain drift of that length -- RF/charge/watch/
clean elements carry no transverse/longitudinal transport physics of their
own), `NAME: LINE=(...)` with recursive line-name expansion,
`&`-continuation, `!`-comments, case-insensitivity.

### `_apply_csrdrift_carry`

Real elegant's `CSRDRIFT` -- see `ElementSpec`'s own docstring
for the model. Walks the beamline once, purely geometrically (no beam
data available yet, hence no live overtaking-length cutoff here --
that happens at track time): entering a `csr=True` bend starts (or
restarts) the carry at `csr_drift_start_m=0`, using that bend's own
rho; every immediately-following `kind="drift"` element is tagged
with that rho and its own running start distance, which then advances
by that element's own `length`. Any non-drift, non-CSR-bend element
(quad, sext, another plain bend) is a real transport discontinuity and
ends the carry -- matches how a real `.lte` only ever places
`CSRDRIFT` immediately after a CSR bend.

## Usage

Call `parse_elegant_lattice(lte_path, root_name)` to expand recursive LINE declarations and obtain beamline-order specs with occurrence indices. Parsing is local, case-insensitive and supports comments/continuations.

### API reference

```python
class ElementSpec:
    ...
def parse_elegant_lattice(lte_path: str | Path, root_name: str) -> list[ElementSpec]: ...
```

### `ElementSpec`

One element's real, physically-meaningful parameters -- everything
`elements.track_element` needs. `kind` is for readability/debugging
only -- dispatch inside `track_element` is purely parameter-driven.

`dx`/`dy`/`tilt`: misalignment (horizontal/vertical offset in
meters, roll in radians). `occurrence`: 1-based count of this
`name` seen so far in beamline order (elegant's `ElementOccurence`
convention) -- used for error matching, see `errors.py`.

`voltage_V`/`phase_rad`/`frequency_Hz`/`change_p0` (`kind="rf"`
only, elegant's `RFCA`): real per-cavity RF parameters, read only by
`NativeTracker(rf="per_element")` -- under the default `rf="lumped"`
mode an RF-kind element is still tracked as a plain drift.

`wakefile`/`wake_t_column`/`wake_wz_column`/`wake_wx_column`/
`wake_wy_column` (`kind="rf"`, elegant's `RFCW`): wake
**metadata** only (plain strings from `WAKEFILE=`/`TCOLUMN=`/
`WZCOLUMN=`/`WXCOLUMN=`/`WYCOLUMN=`) -- no file I/O here;
`linac.resolve_wakes` turns this into real loaded wake objects
(`wake`/`transverse_wake_x`/`transverse_wake_y` below,
`None` until then).

`n_kicks` (`kind="rf"`, default `1`): 
an absent/non-positive value maps to `1`.

`beta_wave` (`kind="rf"`, default `1.0`): traveling-wave phase
velocity in units of `c` -- a **native-only** extension with no
elegant `.lte` equivalent, never parsed from a lattice file.

`end1_focus`/`end2_focus` (`kind="rf"`, default `False`):
elegant's own `END1_FOCUS=`/`END2_FOCUS=` thin-lens kicks at the
cavity entrance/exit.

`body_focus_model` (`kind="rf"`, default `"none"`): elegant's
`BODY_FOCUS_MODEL=` -- `"none"`/`"tw1"`/`"srs"` are all
implemented (see `elements._rf_one_sub_kick`); anything else falls
back to `"none"`. `"srs"` is real CLARA L02/L03/L04's own actual
setting (confirmed 2026-09-21 directly from their own `.lte` files —
an earlier assumption that CLARA used `"tw1"` was wrong).

`csr` (`kind="bend"`, default `False`): distinguishes elegant's
`CSRCSBEND`/`CSRCSBEN` from a plain `CSBEND`/`CSBEN`/`SBEND`/
`SBEN` (all five otherwise parse identically) -- only a CSR-flagged
element gets a CSR kick under `LinacTracker(csr=True)`.

`csr_drift_rho_m`/`csr_drift_start_m` (`kind="drift"` only,
default `None`/`0.0`): a real elegant `CSRDRIFT` tail -- the
steady-state CSR wake from the most recent upstream `csr=True` bend,
carried into the drift(s) immediately following it (set by
`_apply_csrdrift_carry`, run once at parse time). `csr_drift_rho_m`
is that bend's own bending radius; `csr_drift_start_m` is this
drift's own *entrance* distance from the upstream bend's exit (a
static, purely geometric running length -- the exit distance is just
`csr_drift_start_m + length`). The live cutoff itself --
`LinacTracker`'s own ``overtaking_length_m = (24*sigma_z*rho^2)
^(1/3)` (elegant's own `csbend.c` `CSRDRIFT_OVERTAKINGLENGTH``
formula) -- needs the beam's *live* bunch length, so it's evaluated at
track time (`linac._apply_collective_kick`), not here: this drift's
own CSR-active length is ``clamp(overtaking_length_m -
csr_drift_start_m, 0, length)``, zero once the running distance
exceeds the live overtaking length (the radiation has slipped past the
bunch). A first-order approximation of real elegant's own
`CSRDRIFT`/Stupakov formalism (steady-state only, no transient/
entrance term, no shielding, no cross-bend interference)

See docs/tracking_native.md and for the elegant
source citations and validation numbers behind each of these.

## Numerical and implementation notes

### `Module constants`

A statement starts at a line-start "NAME:" (sliced positionally, not by
splitting on ";" -- some real lattice files omit the trailing ";" and
rely on the next "NAME:" as the implicit terminator, which elegant's own
parser tolerates too). "-" is in the name class: real CLARA element names
are hyphenated (e.g. "CLA-L02-LIN-CAV-01").

### `Module constants`

QUAD (elegant's linear-matrix quad) and KQUAD (kick-method) share the
same K1= field name and are recognized identically here -- this module
always uses the kick method regardless of which elegant type name is
used (track_element has no matrix-method path).

### `Module constants`

RFCA/RFCW only -- not RAMPRF, whose static VOLT/PHASE/FREQ fields are
dummy placeholders (the real time-dependent values come from an external
SDDS waveform table via lattice.ramp.RampProfile, not a plain float this
parser could read). RFCW (RFCA + an embedded structure wakefile) uses
the same VOLT/PHASE/FREQ/CHANGE_P0 names as RFCA, so no extra extraction
code is needed beyond this set membership.

### `parse_elegant_lattice`

"NAME: LINE = (...)" or "NAME: LINE, (...)" -- either way the
actual member list is whatever comes after the first "=" (if
any) or ",".

### `parse_elegant_lattice`

PHASE is degrees in real elegant -- convert once here.
WAKEFILE/TCOLUMN/W{X,Y,Z}COLUMN (RFCW only) are metadata
only, not loaded here -- see linac.resolve_wakes.

### `parse_elegant_lattice`

1-based, in beamline (traversal) order -- matches elegant's own
ElementOccurence exactly, giving each physical instance of a
repeated magnet (e.g. br14_simple_ramp.lte's ~37 BDs) a stable
(name, occurrence) identity for error matching (see errors.py).
