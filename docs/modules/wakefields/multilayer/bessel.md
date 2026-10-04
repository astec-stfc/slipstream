# `slipstream.wakefields.multilayer.bessel`

[Source](../../../../src/slipstream/wakefields/multilayer/bessel.py) · [Documentation index](../../../README.md)

## Physics

Complex modified Bessel functions use SciPy forward evaluation and exact recurrence derivatives for PyTorch complex autograd. Stable ratios and holomorphic scaled functions prevent raw exponential overflow.

### Limits and conventions

- Forward evaluation includes a CPU round trip suitable for frequency-sized arrays.
- Scaled functions assume Re(z)>=0.
- Use complex holomorphic exponential scaling rather than real-only absolute-value scaling if derivatives must remain valid.

### `Model and numerical conventions`

    dI_n/dz = I_{n-1}(z) - (n/z) I_n(z)
    dK_n/dz = -K_{n-1}(z) - (n/z) K_n(z)

(valid for n=0 too, using the standard I_{-1}=I_1, K_{-1}=K_1 identities,
which scipy's `iv`/`kv` already honor for negative order.)

Both I_n and K_n are holomorphic in z, so PyTorch's complex-autograd
convention applies directly: `grad_z = grad_output * conj(dI_n/dz)` --
verified (not assumed) against finite differences via
`torch.autograd.gradcheck` in `tests/test_multilayer_bessel.py`.

### `_BesselRatioI`

I_{n-1}(z)/I_n(z) -- see model description's "numerical conditioning"
section: I_n(z) itself overflows for the |z| this repo's realistic
layer/frequency parameters produce (checked: |z| ~ 1e5 at 1mm-1cm
radius, 1e-5 Ohm.m resistivity, THz frequencies -- see, but this ratio stays O(1) for all such z
(I_{n-1} and I_n share the same leading exponential growth rate) and is
exactly what the ported multilayer formalism (round.cc's
`Imppratio`) actually needs -- never materializing the raw,
overflowing value in the first place, rather than overflowing and
trying to recover.

Forward: `scipy.special.ive(n-1,z)/scipy.special.ive(n,z)` -- the
exponential scale factor is identical for both (same z), so it cancels
exactly, giving the true ratio via values that individually never
overflow. Verified against mpmath (50 decimal digits) to machine
precision (~1e-16) up to |z|~1.25e5 -- double precision is sufficient
for this combination, multiprecision is not needed (unlike IW2D's own
C++, which uses 160-bit arithmetic unconditionally for every
combination, not just this one).

Backward: closed form purely in terms of the ratio itself (never the
raw, potentially-overflowing I values), derived from the standard
recurrence dI_n/dz = I_{n-1} - (n/z)I_n and its mirror
dI_{n-1}/dz = I_n + ((n-1)/z)I_{n-1} (DLMF 10.29.1-2) via the quotient
rule:

    dR/dz = 1 - R^2 + R*(2n-1)/z, R = I_{n-1}(z)/I_n(z)

verified against finite differences directly (not just trusted from
the algebra) before use.

### `_BesselRatioK`

K_{n-1}(z)/K_n(z) -- the K-function analog of `_BesselRatioI`,
same rationale (see that class's docstring). Backward derived the same
way (quotient rule + DLMF 10.29.3-4's one-sided recurrences for K) and
verified against finite differences before use:

    dR/dz = R^2 - 1 + R*(2n-1)/z, R = K_{n-1}(z)/K_n(z)

(note the sign flip on the "R^2-1" term relative to the I-ratio's
"1-R^2" -- expected, K decays where I grows).

### `_ModifiedBesselIHat`

I_n(z)*exp(-z) -- an O(1)-magnitude, *holomorphic* rescaling of
I_n(z), valid for Re(z)>=0 (the only regime round.py ever calls this
with: z is always a layer's transverse wavenumber times a positive
radius, and the wavenumber's branch is chosen with Re>=0 -- see
round.py's `_csqrt`). Needed alongside `bessel_i_ratio`/
`bessel_k_ratio` because the multilayer round-chamber matrix
(round.py's `_layer_matching_matrix`) also needs raw I_m(x)*K_m(y)-
type *products* at two different arguments, not just same-argument
ratios -- and those products overflow in raw double precision well
within this repo's realistic parameters (found validating against live IW2D: xp1p~9000 at the vacuum-to-first-conductor
boundary, alone enough to overflow raw I_0).

Scaling by the *real-only* exp(-Re(z)) (matching scipy's `ive`
convention directly) would be simpler but is not holomorphic (it
depends on both z and conj(z)), needing the full non-holomorphic
Wirtinger backward. Scaling by the full complex exp(-z) instead has
the identical magnitude benefit (|exp(-z)|=exp(-Re(z))) while staying
holomorphic, so the same simple `grad_output * conj(df/dz)`
convention already used throughout this module applies unchanged.

Forward: `ive(n,z)*exp(-1j*z.imag)` -- equals I_n(z)*exp(-z) exactly
when Re(z)>=0 (scipy's ive uses exp(-|Re(z)|), which is exp(-Re(z))
exactly in that regime). Backward, derived from the standard I
recurrence and verified against finite differences before use:

    d/dz [I_n(z)e^{-z}] = I_{n-1}(z)e^{-z} - (1 + n/z) I_n(z)e^{-z}

### `_ModifiedBesselKHat`

K_n(z)*exp(z) -- the K analog of `_ModifiedBesselIHat` (see its
docstring); this one equals scipy's `kve(n,z)` directly (scipy's own
K-scaling convention already uses the full complex exp(z), unlike
`ive`'s real-only exp(-|Re(z)|) -- checked directly from scipy's
docstrings), so no phase correction is needed.

    d/dz [K_n(z)e^{z}] = -K_{n-1}(z)e^{z} + (1 - n/z) K_n(z)e^{z}

verified against finite differences before use.

## Usage

Call `modified_bessel_i`/`modified_bessel_k`, ratio helpers or `modified_bessel_i_hat`/`modified_bessel_k_hat` with integer order and complex tensors.

### API reference

```python
def modified_bessel_i(n: int, z: torch.Tensor) -> torch.Tensor: ...
def modified_bessel_k(n: int, z: torch.Tensor) -> torch.Tensor: ...
def bessel_i_ratio(n: int, z: torch.Tensor) -> torch.Tensor: ...
def bessel_k_ratio(n: int, z: torch.Tensor) -> torch.Tensor: ...
def modified_bessel_i_hat(n: int, z: torch.Tensor) -> torch.Tensor: ...
def modified_bessel_k_hat(n: int, z: torch.Tensor) -> torch.Tensor: ...
```

### `bessel_i_ratio`

I_{n-1}(z)/I_n(z), z complex, differentiable and numerically stable
for arbitrarily large |z| (unlike ``modified_bessel_i(n-1,z) /
modified_bessel_i(n,z)``, which overflows to nan/inf well within this
repo's realistic parameter range -- see class docstring).
