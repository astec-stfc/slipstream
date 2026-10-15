# `slipstream.tracking.native.photocathode`

[Source](../../../../src/slipstream/tracking/native/photocathode.py) · [Documentation index](../../../README.md)

## Physics

Opt-in laser-driven emission for `Injector.track`. The macroparticles of the cathode distribution sample the laser
pulse (their transverse positions and emission times are its profile). When the laser frees a macroparticle it becomes

```text
q = e * N_photons * share * QE(E_s),     N_photons = pulse_energy / (h c / wavelength)
```

where `share` is its fraction of the pulse and `E_s` is the extraction field at its position on the cathode surface at
that instant. Without a `Photocathode` the injector keeps the charge of the distribution file or `generator.in`.

### Quantum efficiency

With a work function the yield follows the three-step model of Dowell and Schmerge (PRSTAB 12, 074201, 2009, Eqs. 2
and 10). The surface field lowers the barrier,

```text
phi_eff = phi_w - sqrt(e * beta * E_s / (4 pi eps0))       (0.037947 sqrt(beta E[MV/m]) eV)
```

and the energy factor is

```text
g(phi) = (E_F + hw)/(2 hw) * [1 - sqrt((E_F + phi)/(E_F + hw))]^2       (0 for phi >= hw)
QE(E_s) = qe * g(phi_eff(E_s)) / g(phi_eff(E_ref))
```

The optical factor `(1 - R)/(1 + l_opt/l_ee)` of the paper does not depend on the field and cancels against the quoted
`qe`, which holds at `qe_field_V_per_m` (`E_ref`). Near threshold `g` is proportional to `(hw - phi_eff)^2`. For copper at
266 nm (4.66 eV, `phi_w` about 4.3-4.6 eV) `hw - phi_eff` is mostly the Schottky term, so the yield falls steeply as the RF
field falls: this is what a Schottky scan measures. A field `E_s <= 0` does not extract and the yield is zero. Without a
work function `qe` is constant. `field_enhancement` (`beta`) multiplies the field inside the square root; the data
constrain only `hw - phi_w` and `beta*E` jointly, so `phi_w` and `beta` are degenerate in a fit.

### Surface field and the space-charge limit

The emission of the laser's macroparticles that are released in one time step (a layer) is limited by the field left
at the conducting cathode surface by the charge already emitted. A charge `q` at radius `r_j` and height `h` above the
cathode plane, together with its image, lowers the normal field at the surface point `r` by

```text
dE = |q| h / (2 pi eps0 (rho^2 + h^2)^(3/2)),     rho = |r - r_j|
```

For a thin sheet the total is `sigma/eps0` independent of height (Gauss law at a conductor); for a disc of radius `R`
and density `sigma` at height `h` on the axis it is `sigma/eps0 * (1 - h/sqrt(h^2 + R^2))`. The sum is evaluated for the
alive, already emitted macroparticles in equal-count radial bins (`<= 64`, axisymmetric like the cylindrical solver),
each bin split into equal-area sub-rings at the bin's charge-weighted height and averaged around the azimuth. The
height is floored at the widest sub-ring gap so that a freshly emitted layer is a smooth sheet; the sheet is then
under-screened by a few percent only while it is within a sub-ring gap of the surface.

For the new layer, `E0 = E_applied - E_cloud` is the surface field before it is emitted. The layer's charge solves

```text
q = q_photons * QE(E0 - c q / 2)     (its own mean screening),     q <= max(E0, 0) / c    (Gauss limit)
```

where `q_photons = e N_photons share` and `c q = sigma_layer/eps0` is the field of the layer's own areal density (the
laser's radial profile times the layer's share of the pulse). The root is found by bisection and the layer is capped so
it can at most cancel the surface field. A macroparticle that gets no charge is removed for good (`n_suppressed`): the
laser does not try again later. Emission proceeds layer by layer at the tracker's emission step, so a short pulse
saturates at `eps0 E pi R^2` for a uniform disc and a Gaussian spot saturates locally where its areal density exceeds
`eps0 E`. A pulse long enough for the cloud to leave the surface by a fraction of the spot radius emits more, and small
spots exceed the pancake limit more strongly.

### Fall-back onto the cathode

After each step a released macroparticle that has fallen behind the cathode plane is absorbed (removed from space charge
and statistics; its charge is `returned_charge_C`). Late in the RF half-cycle an electron is released in a field that
reverses before it has left the gun, so it returns: the extracted charge then falls well below the emitted charge.

### Limits and conventions

- Only `Injector.track` supports it; `track_differentiable` raises `NotImplementedError`.
- The laser profile is the macroparticle distribution; `q_total`/charges of the beam only weight it.
- The emission field is the applied on-axis `Ez` at the particle's position and at its own emission time (not at the
  step that releases it); the transverse RF fields, the radial dependence of `E_s` beyond the surface-charge sum and the
  relativistic boost of the cloud are not included. The cloud's screening is axisymmetric: an elliptical spot is
  represented by its radial profile.
- The initial momentum comes from the generator (`LE`, `FD_300`); it does not follow the Schottky-lowered `hw - phi_eff`.
- Trajectory statistics and the final `sigma_*`/`Ekin_MeV` are macroparticle averages, not charge-weighted. Use
  `return_particles=True` or `out["emission"]["q"]` for charge-weighted moments.
- Temperature smoothing of the threshold (Fowler) and the absolute optical factor are not modelled: `qe` is an input.
- The macroparticle number sets the statistics of the surface field; a few hundred per radial bin are enough for percent
  accuracy of the extracted charge.

## Usage

Set `Injector.photocathode` (also through `from_laura(..., photocathode=...)`), track with `screen_z_m` at the charge
monitor, and read `out["emission"]`. `Photocathode.from_average_power(power_W, rep_rate_Hz, ...)` converts an
average laser power on the cathode to the pulse energy `P / f_rep`.

### API reference

```python
def schottky_lowering_eV(field_V_per_m, enhancement: float=1.0): ...
class Photocathode:
    pulse_energy_J: float
    wavelength_nm: float
    qe: float
    material: str | None = None
    work_function_eV: float | None = None
    fermi_energy_eV: float | None = None
    qe_field_V_per_m: float | None = None
    field_enhancement: float = 1.0
    def from_average_power(cls, power_W: float, rep_rate_Hz: float, **kwargs) -> 'Photocathode': ...
    def photon_energy_eV(self) -> float: ...
    def photons(self) -> float: ...
    def quantum_efficiency(self, field_V_per_m: torch.Tensor) -> torch.Tensor: ...
class PhotoEmitter:
    def __init__(self, photo: Photocathode, r: torch.Tensor, q_nominal: torch.Tensor): ...
    def screening(self, r: torch.Tensor, height: torch.Tensor, charge: torch.Tensor) -> torch.Tensor: ...
    def emit(self, idx: torch.Tensor, r: torch.Tensor, field_applied: torch.Tensor, cloud) -> tuple: ...
```

### `Photocathode`

| Field | Meaning |
| --- | --- |
| `pulse_energy_J` | laser energy on the cathode per pulse, J |
| `wavelength_nm` | photon energy `hw = 1239.84 eV nm / wavelength` |
| `qe` | electrons per photon at `qe_field_V_per_m`; constant when no work function is given |
| `material` | `"Cu"`: fills `work_function_eV=4.31`, `fermi_energy_eV=7.0` (Dowell and Schmerge, Table I) |
| `work_function_eV` | enables the Schottky dependence; clean-copper values run 4.3-4.65 eV |
| `fermi_energy_eV` | default 7.0 eV |
| `qe_field_V_per_m` | field where `qe` was quoted (required with a work function; the yield there must be above threshold) |
| `field_enhancement` | `beta` of the Schottky term |

```python
from slipstream import Injector, Photocathode

machine = Injector.from_astra("injector.in")               # or from_laura(..., photocathode=...)
machine.photocathode = Photocathode(pulse_energy_J=5e-6, wavelength_nm=266.0, qe=1e-4,
                                    material="Cu", qe_field_V_per_m=100e6)
out = machine.track(device="cuda", screen_z_m=1.1)
print(out["emission"]["emitted_charge_C"], float(out["screen"]["q"].abs().sum()))
```

### `Injector.track` with a photocathode

`out["emission"]` holds `photons`, `ideal_charge_C` (the applied-field yield without screening), `emitted_charge_C`
(after the surface-field limit), `returned_charge_C` (emitted, then absorbed back), `charge_C` (still in the machine at
the end), `n_suppressed`, `n_returned` and `q` (signed charge of every macroparticle, 0 if never emitted).
`n_lost_on_aperture` counts only aperture losses. With `screen_z_m` the charge at a monitor plane is the sum of
`out["screen"]["q"]` (zero where not `crossed`); particles that fell back never cross it.

### Calibrating from a Schottky scan

For a fixed laser the charge against gun phase combines the emission law, the field at the cathode and the transport
to the monitor. The flank where the field falls toward its zero crossing is transport-free: fitting it with the model
above gives `hw - phi_w` (given `beta`), the product of `qe` and the laser energy, the phase origin and any monitor
baseline. `examples/photocathode/schottky_scan.py` does this for the CLARA HRRG and then tracks the whole scan (emission,
space charge, fall-back, apertures). The same example scans extracted charge against laser energy for several spot sizes.
`examples/photocathode/fit_laser_image.py` fits a virtual-cathode camera image of the laser to the `Dist_x = Dist_y = 'g'`
gaussian of `generator.in` (`sig` and the n-sigma cut `C_sig` per axis) cut on an ellipse (the generator's native
`Cut_shape = 'ellipse'`, see [`generator`](generator.md)), which is the transverse laser profile the example uses. The
example takes the laser energy from the energy-meter readback stored with that image, so the fit gives the quantum
efficiency (`--qe` fixes it instead and the fit gives the energy).

## Numerical and implementation notes

- Closed-form checks in `tests/test_photocathode.py`: Schottky lowering 0.268 eV at 50 MV/m, the energy factor, the disc
  field, a short uniform pulse saturating at `eps0 E pi R^2` (within 5 %), a Gaussian spot clipped at the local Gauss
  density (within 7 %), the field scaling of the yield, and absorption of late emission.
- The uniform-disc limit is within 1 % between steps of 0.1 and 0.005 ps and within about 1 % between 1500 and 6000
  macroparticles; it approaches the Gauss law from above (1.02-1.05 for 0.2-1 ps pulses). The CLARA scan agrees within
  3 % between steps of 0.1, 0.2 and 0.4 ps and within 1 % between 1500 and 4000 macroparticles.
- Emission arithmetic is float64 on the tracking device; the layer evaluation is eager (a few ms per emission step).
