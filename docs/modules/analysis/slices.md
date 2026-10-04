# `slipstream.analysis.slices`

[Source](../../../src/slipstream/analysis/slices.py) · [Documentation index](../../README.md)

## Physics

Bins length-like tau in metres and computes charge/current, widths and centered transverse emittances per slice. Bunching is `abs(mean(exp(i*k*tau)))`; chirp is a linear fit of relative momentum deviation against tau.

### Limits and conventions

- Seconds-valued timestamps require multiplication by `C_LIGHT` before these APIs.
- Transverse emittance interpretation depends on whether inputs are slopes or canonical momenta.

## Usage

Use exactly one of `n_slices`, `ds_m` or explicit `slice_edges_m` with `slice_indices`; pass its indices/edges to `slice_stats`. `bunch_summary`, `bunching_factor`, `chirp_fit` and `correlated_emittance` provide aggregate diagnostics.

### API reference

```python
def slice_indices(tau_m: torch.Tensor, n_slices: int | None=None, ds_m: float | None=None, slice_edges_m: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
def slice_stats(tau_m: torch.Tensor, delta: torch.Tensor, x_m: torch.Tensor, xp: torch.Tensor, y_m: torch.Tensor, yp: torch.Tensor, charge_per_particle_C: torch.Tensor | float, idx: torch.Tensor, edges: torch.Tensor, energy_eV: torch.Tensor | float | None=None, mc2_eV: torch.Tensor | float | None=None) -> dict[str, torch.Tensor]: ...
def bunching_factor(tau_m: torch.Tensor, k_per_m: torch.Tensor | float, charge_per_particle_C: torch.Tensor | float | None=None) -> torch.Tensor: ...
def chirp_fit(tau_m: torch.Tensor, delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
def correlated_emittance(tau_m: torch.Tensor, delta: torch.Tensor, energy_eV: torch.Tensor | float | None=None, mc2_eV: torch.Tensor | float | None=None) -> dict[str, torch.Tensor]: ...
def bunch_summary(particles6: torch.Tensor, energy_eV: float, charge_C: float, n_slices: int=60, window_sigma: float=3.0, mc2_eV: float=510998.95) -> dict[str, float]: ...
```

### `slice_indices`

Bin `tau_m` into longitudinal slices. Exactly one of `n_slices`
(a fixed slice count spanning the live min/max), `ds_m` (a fixed
slice width, slice count derived from the live span), or
`slice_edges_m` (an explicit, caller-fixed `(m+1,)` edge array).

`slice_edges_m` is the compile-safe path -- fixed shape, no host
sync. `n_slices`/`ds_m` instead read `tau_m.min()`/`.max()`
via a one-time host sync (`.item()`/`float()`) to size the edge
array -- meant for call-site setup, not inside a compiled per-turn
step; pass the resulting `edges` back in as `slice_edges_m` on a
later call if that matters (same fixed-bin-window discipline as
`wakefields.shortrange.ShortRangeWake.set_fixed_bin_window`).

Returns `(idx, edges, counts)`: `idx` (N,) each particle's slice
index in `[0, n_slices)`, or exactly `n_slices` (one past the
last real slice) for a particle outside `[edges[0], edges[-1]]` --
a dedicated trash bin, dropped (not clamped in) by every function in
this module that consumes `idx`. `edges` (n_slices+1,).
`counts` (n_slices,) particles per slice (int64).

### `slice_stats`

Per-slice beam statistics, from `idx`/`edges` as returned by
`slice_indices` -- split out so the (fixed-shape, masked-reduction)
statistics kernel is independently `torch.compile`-traceable given a
fixed particle count and slice count, without also tracing the
(possibly data-dependent) binning step above.

All outputs are `(n_slices,)` tensors, count-weighted (not charge-
weighted -- this codebase's convention is uniform per-particle charge,
see `charge_per_particle_C`'s own callers, so the two coincide; count-
weighting also stays well-defined when `charge_per_particle_C=0`, the
common "no wake charge configured" default):

- `n_particles`, `charge_C`, `current_A` (`charge_C*C_LIGHT/ds`,
  since `tau_m` is path-length-like) -- exactly `0` (never `NaN`)
  for an empty slice, the physically correct value.
- `mean_tau_m`, `mean_delta`, `sigma_delta`, `mean_x_m`,
  `mean_y_m`, `eps_x_m`, `eps_y_m` (projected geometric/canonical
  emittance, see model description), `corr_x_delta`, `corr_y_delta`
  (chirp-diagnostic correlation coefficients) -- `NaN` for an empty
  slice (undefined, not zero; downstream users must expect it).
- `eps_x_n_m`/`eps_y_n_m` (normalized emittance, `eps * beta*gamma`)
  only when both `energy_eV` and `mc2_eV` are given.

`eps_x_m`/`eps_y_m` use `sqrt(clamp(det, min=0))` -- float32 det
can go slightly negative on a noisy/low-count slice.

### `bunching_factor`

`|<exp(i*k*tau)>|` -- the bunching factor at wavenumber(s)
`k_per_m` (rad/m). `k_per_m` scalar -> 0-d result; `k_per_m` a
`(m,)` tensor -> `(m,)` result, one value per probe wavenumber (a
microbunching spectrum). Count-weighted unless
`charge_per_particle_C` is given (mirrors `slice_stats`'s own
charge/count-weighting note).

### `chirp_fit`

Linear fit `delta ~= slope*tau_m + intercept` via closed-form
normal equations (fixed-shape, `torch.compile`-safe). Returns
`(slope [1/m], intercept, r)` -- `r` the Pearson correlation
coefficient (chirp strength, dimensionless).

### `correlated_emittance`

Whole-bunch longitudinal emittance, projected vs. with the linear
energy chirp removed -- the quantity that makes a CSR/LSC kernel
benchmark (`tracking.native.collective`) interpretable: a chirped
bunch's *projected* longitudinal emittance is dominated by the (fully
reversible, not a collective-effect signature) correlated term, and
only the chirp-removed one isolates real emittance growth.

Removes `chirp_fit`'s own fitted line from `delta` before
recomputing. `eps_z_*_keV_mm` (ASTRA `Zemit`-comparable, see
`analysis.emittance.longitudinal_emittance_keV_mm`) only when both
`energy_eV`/`mc2_eV` are given.

### `bunch_summary`

One-line-per-quantity longitudinal/transverse summary of a `tracking.native` bunch
`(N, 6)` at reference energy `energy_eV` -- what a compressor paper tabulates.

Slices span `mean_tau +- window_sigma*sigma_tau` in `n_slices` equal bins, so the slice width
follows the bunch; `I_peak_A` is the largest slice current, and `slice_eps_x_n_um` the charge-weighted
slice normalised emittance over the slices carrying at least half of that peak current (the
part of the bunch that lases). Emittances are canonical (`px`), normalised by `beta*gamma`.
Done in float64 on the CPU regardless of the tracking dtype.
