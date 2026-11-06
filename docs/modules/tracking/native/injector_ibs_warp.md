# `slipstream.tracking.native.injector_ibs_warp`

[Source](../../../../src/slipstream/tracking/native/injector_ibs_warp.py) · [Model and public API](injector_ibs.md)

## Physics

Optional float64 CUDA acceleration of the same equal-weight TA/Nanbu pair
operator documented in the public module. One thread owns each disjoint pair.
The rate, thermal-domain guard and exact conservative Lorentz update preserve
the PyTorch model. Small-argument `expm1`/`log1p` series and an inverse-Langevin
solve retain weak kicks and the isotropic limit; floating-point fusion is
disabled to keep sensitive operation ordering close to the reference.

## Usage

Select it with `InjectorIBSSettings(backend="auto", ...)` on CUDA, or require it
with `backend="warp"`. Grouping, deterministic cell reductions and private RNG
draws stay in PyTorch. The internal wrapper launches on the active PyTorch
stream, wraps tensors without copies, returns a new momentum tensor and reduces
pair diagnostics before reporting domain errors. Input momenta are unchanged
if validation fails. It is forward-only and is not a CUDA-graph capture API.

The optional backend tests in `tests/test_injector_ibs_warp.py` check absolute
Rutherford rates at bulk gamma 1/5/100, common-draw impulse parity through
variance 1000, conservation, active/neutral masks and non-default CUDA streams.
CPU and installations without Warp retain the reference implementation.
