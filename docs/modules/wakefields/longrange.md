# `slipstream.wakefields.longrange`

[Source](../../../src/slipstream/wakefields/longrange.py) · [Documentation index](../../README.md)

## Physics

High-Q resonators accumulate decaying/rotating interbunch voltage phasors with `lambda=(i-1/(2Q))*omega_r`. Sources and test particles use opposite exponent directions relative to bucket phase.

### Limits and conventions

- Phasor state persists turn to turn; independent runs need independent/reset state.
- Only one main cavity should own generator feedback; harmonic cavities are passive unless separately modeled.
- The helper model's presence does not imply every native tracker automatically integrates multibunch wakes.

## Usage

Build Resonator and stateful LongRangeWake(resonator, h, Tb_s, ...), where Tb_s is bucket spacing in seconds; initialize filling and call step with per-bucket equivalent voltage. equivalent_voltage_phasor and phasor_at_particles use time offsets in seconds.

### API reference

```python
class Resonator:
    ...
    def from_detuning(cls, R_over_Q_ohm: float, Q: float, harmonic_n: int, detuning_Hz: float, h: int, f_rev_Hz: float) -> 'Resonator': ...
    def lam(self) -> complex: ...
    def to_yaml(self, path: str | Path) -> None: ...
    def from_yaml(cls, path: str | Path) -> 'Resonator': ...
class LongRangeWake:
    ...
    def a(self) -> complex: ...
    def init_uniform_fill(self, I0_avg_A: float, F: float=1.0) -> None: ...
    def step(self, Veq_bucket: torch.Tensor) -> torch.Tensor: ...
def equivalent_voltage_phasor(tau: torch.Tensor, charge_per_particle_C: torch.Tensor | float, resonator: Resonator, bucket_index, h: int) -> torch.Tensor: ...
def phasor_at_particles(V_b_bunch: torch.Tensor, tau: torch.Tensor, resonator: Resonator) -> torch.Tensor: ...
```

### `Resonator.from_detuning`

Build from Table-style parameters (Eq. 7's inverse): the cavity
targets the harmonic_n'th multiple of the rf frequency
(harmonic_n=1 for the main cavity, 3 for a third-harmonic HHC), and
is detuned from it by detuning_Hz.

### `LongRangeWake`

Per-bucket resonator phasor state (Eqs. 4-5), carried turn to turn.

`is_main_cavity=True` marks the resonator whose generator voltage is
adjusted by Eq. 9's feedback each turn (only one should be marked --
HHCs are passive, driven purely by the beam).

### `LongRangeWake.init_uniform_fill`

Seed V_c,0 assuming a uniform fill pattern (every bucket filled
with an identical bunch), so the recursion starts at its steady
state instead of needing several (possibly many, for a high-Q HHC)
turns of transient to reach it.

The paper gives an approximate formula for this (Eq. 6, via the
detuning angle of Eq. 7). Used here instead is the *exact* closed
form for the same steady state, derived directly from Eqs. 4-5's
own recursion plus the periodicity condition V_c,0 = V_c,h: for a
uniform fill, Veq,i = Veq is the same every bucket, so
V_c,i = a^i*V_c,0 + Veq*(a^i-1)/(a-1); requiring V_c,h = V_c,0
gives V_c,0 = Veq/(1-a). Cross-checked against Eq. 6/7 in the
tests (they agree, as they must).

### `LongRangeWake.step`

Advance one turn: solve V_c,i = a*V_c,i-1 + Veq,i for i=1..h
given self.V_c0, update self.V_c0 to V_c,h, and return V_b (length
h, complex) = a*V_c,i-1 for i=1..h -- the phasor felt by bucket i's
own bunch this turn (Eq. 4).

Closed form: with u_i = a^-i*Veq,i (u_0 := V_c,0), V_c,i = a^i *
cumsum(u)_i. Done in fixed-size chunks (default 256 buckets),
carrying V_c across chunk boundaries, because a^-i grows without
bound over h~1e3 buckets for a high-Q resonator (|a|<1) -- a single
h-long cumsum would overflow long before reaching realistic h.
