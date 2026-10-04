# `slipstream.tracking.native.ring_physics`

[Source](../../../../src/slipstream/tracking/native/ring_physics.py) · [Documentation index](../../../README.md)

## Physics

Native ring kicks apply sinusoidal RF, radiation damping/excitation and effective IBS damping times. RF returns fractional energy deviation relative to the synchronous phase; emittance uses centered covariance.

### Limits and conventions

- This module is independent of historical development trackers.
- Its RF tau units differ from the canonical transport tau units.

## Usage

`rf_kick` takes tau in seconds and optional RF/energy overrides. Native tracker callers convert their metre-valued tau before using it. Damping routines operate on longitudinal or transverse arrays with supplied equilibrium parameters.

### API reference

```python
def rf_kick(tau: torch.Tensor, params: 'Lattice', Vrf_V: torch.Tensor | float | None=None, omega_rf: torch.Tensor | float | None=None, phi_s: torch.Tensor | float | None=None, E0_eV: torch.Tensor | float | None=None) -> torch.Tensor: ...
def damping_excitation(delta: torch.Tensor, params: 'Lattice', generator: torch.Generator | None=None, tau_z_eff_s: torch.Tensor | None=None, sigma_delta0_eff: torch.Tensor | None=None, noise: torch.Tensor | None=None, tau_z_noise_s: torch.Tensor | None=None) -> torch.Tensor: ...
def transverse_damping_excitation(up: torch.Tensor, tau_plane_s: float, eps_m: float, beta_m: float, alpha: float, T0_s: float, generator: torch.Generator | None=None, tau_eff_s: torch.Tensor | None=None, eps_eff_m: torch.Tensor | None=None, noise: torch.Tensor | None=None, tau_noise_s: torch.Tensor | None=None) -> torch.Tensor: ...
def ibs_effective_taus(tau_z_sr, tau_x_sr, tau_y_sr, rates: 'IBSGrowthRates'): ...
def rms_emittance_tensor(u: torch.Tensor, up: torch.Tensor) -> torch.Tensor: ...
def rms_emittance(u: torch.Tensor, up: torch.Tensor) -> float: ...
```
