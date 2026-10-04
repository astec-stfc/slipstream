# `slipstream.tracking.native.generator`

[Source](../../../../src/slipstream/tracking/native/generator.py) · [Documentation index](../../../README.md)

## Physics

Samples ASTRA-style initial beams with configurable distributions, correlations, offsets, species and emission times. Noise reduction uses Halton rather than ASTRA's Hammersley sequence; Sobol/random are additional options.

### Limits and conventions

- Curved cathode `R_cathode` and binary output raise rather than being silently approximated.
- `Beam.particles` restores diagnostic probes' charge share to tracked particles so total charge matches the requested bunch.

### `Model and numerical conventions`

    from slipstream.tracking.native.generator import generate
    beam = generate("ASTRA/20pC/generator.in", ipart=1_000_000, seed=1)
    beam.write("laser_1M.astra") # ASTRA-readable, 20-digit `High_res` format
    dist = beam.particles(device="cuda") # what `Injector.track` consumes

Every option of the manual's chapter 7 namelist is accepted (`Dist_*` =
`uniform`/`plateau`/`gauss`/`radial`/`isotropic`/`inverted`/
`FD_300`/`uniform ellipsoid`/`2D-Gaussian`, `Cathode`, `Probe`, `Passive`,
`Noise_reduc`, `Species`/`Ion_mass`, `Type='ring'`, `Tau`, `Nemit_*`,
`emit_z`, `cor_*`, `Disp_*`, offsets, `Add`/`N_add`), except `R_cathode`
(curved cathode -- the sign of its sag/time correction cannot be read off the manual) and
`Binary`, which raise. `Noise_reduc=T` is a Halton sequence, not the manual's
Hammersley set (bases `E`:2, `t`:3, `z`:5, `py`:7, `px`:11, `y`:13, `x`:17).

**Improvements over ASTRA** (all optional): `ipart`/`seed` at call time; ``sequence=
"sobol"` (scrambled Sobol', better than Halton in 7 dimensions at small N) or `"random"``;
`Beam.particles` returns the probes' charge share to the tracked particles so the tracked
total is exactly `Q_total` at any N (ASTRA's is short by `7/Ipart`).

### `_plateau_ppf`

Inverse CDF of ASTRA's plateau `f(x) = 1/(1 + exp(4 (|x| - L/2)/rt))` (manual 7.2.2; measured: the
`4/rt` is what makes the tangent at `L/2` climb from 0 to 1 in `rt`). The CDF is analytic --
`int_0^a f = a - s softplus((a-mu)/s) + s softplus(-mu/s)`, `s = rt/4`, `mu = L/2` -- so a bisection on it
is exact to rounding (a Newton step diverges in the near-zero-density tails).

## Usage

Use `generate(generator_input, ipart=..., seed=...)` for a `Beam`, `Beam.particles(device=...)` for injector input or `Beam.write` for ASTRA-format output. The module CLI accepts particle count and output overrides.

### API reference

```python
class GeneratorInput:
    ...
    def n_probe(self) -> int: ...
    def species_info(self) -> tuple[int, float, float]: ...
    def from_namelist(cls, group: dict) -> 'GeneratorInput': ...
    def reference_row(self) -> np.ndarray: ...
def read_generator_input(path: str | Path) -> list[GeneratorInput]: ...
class Beam:
    ...
    def particles(self, device=None, dtype=None, exact_charge: bool=True) -> dict[str, torch.Tensor]: ...
    def write(self, path: str | Path, high_res: bool | None=None) -> None: ...
    def moments(self) -> dict[str, float]: ...
def generate(source: 'str | Path | GeneratorInput | list[GeneratorInput]', ipart: int | None=None, seed: int=0, sequence: str | None=None, noise_reduc: bool | None=None, normalise: bool=True) -> Beam: ...
def main(argv: list[str] | None=None) -> Beam: ...
```

### `GeneratorInput`

One `&INPUT` namelist. Field names are ASTRA's, lower-cased; units are ASTRA's (manual
ch. 7): lengths in mm (`ref_zpos` and `r_cathode` in m, `disp_*` in m), times in ns,
`sig_ekin`/`le`/`re`/`cor_ekin` in keV, `ref_ekin` in MeV, momenta in eV/c,
`nemit_*` in mm mrad, `cor_p*` in mrad, charge in nC. Defaults are the manual's.

### `Beam.particles`

The trackable particles, as `astra_io.read_astra_distribution` returns a file's: the standard
rows (probes and passive particles excluded), absolute `z`/`pz`/`clock`. `exact_charge`
(default) rescales `q` so they carry exactly `Q_total` -- ASTRA's `Q_total/Ipart` per row
leaves the 6 probe rows a charge share the tracker never uses.

### `generate`

Generate a beam from a `generator.in` path or already-built `GeneratorInput` (a list = `Add=T`).

`ipart` overrides `Ipart` -- ASTRA's count, which *includes* the reference row and the 6
probes (with several stacked namelists it rescales them proportionally). `noise_reduc`
overrides the deck's flag; `sequence` is `"halton"` (what `Noise_reduc=T` means in ASTRA),
`"sobol"` (scrambled) or `"random"` (default: from `noise_reduc`). `seed` fixes the
pseudo-random draws (`"random"`/`"sobol"`, `Tau`, `Type='ring'`, `FD_300`); the Halton
sequence itself is deterministic. `normalise=False` skips the exact-rms rescale (the means are
still zeroed) to leave the raw sampling noise in.

## Numerical and implementation notes

### `Module constants`

constants measured from the ASTRA generator (see model description)

### `_block`

positions [mm] and the longitudinal coordinate [mm | ns]

### `_block`

momentum: the energy / isotropic emission, then the transverse planes [eV/c]

### `_block`

correlations and dispersion. Measured: they act on the probes too, and a correlated px, py is taken out of
pz so that |p| is conserved. Done on [probes; particles] together, so a dispersion has its mean over all rows removed.

### `_block`

the file rows: positions carry the offsets; z / clock / pz are relative to the reference row

