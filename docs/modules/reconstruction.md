# `slipstream.reconstruction`

[Source](../../src/slipstream/reconstruction.py) · [Documentation index](../README.md)

## Physics

The forward model of a gradient-based phase-space reconstruction (GPSR style): a bunch is tracked by a `LinacTracker` at each row of a scan table (magnet strengths, RF settings) and imaged on a screen. Everything is differentiable, so the gradient of an image loss reaches the bunch, the scan table and the screen calibration. The package supplies the forward model; the generator of the bunch, the loss and the optimiser belong to the caller.

### Screen

`Screen` splats each particle onto a fixed pixel grid by cloud in cell (bilinear weights to the four nearest pixel centres). The first moment of the image equals the mean of the particle coordinate exactly, charge is conserved inside the frame, and weight that falls outside the frame is lost, so the image sums to the in-frame fraction of the bunch. The result is piecewise linear in the particle position: its gradient is exact but has kinks at pixel centres, so a finite-difference check needs a step well below a pixel. An optional Gaussian point-spread function (in pixels, zero padded), a gain and a background follow; gain and background may be tensors, so calibration constants can be fitted. The grid is fixed, so the render has static shapes and no host reads.

### Coordinates

Cheetah (and so GPSR) carries `p = (E - E0) / (p0 c)` as its sixth coordinate, Slipstream `delta = (p - p0) / p0`. `from_cheetah` and `to_cheetah` convert exactly through the energy-momentum relation; the two agree only at high energy (at 150 MeV a Cheetah `p` of 1e-3 is a `delta` of 1.0000058e-3). The other five coordinates agree: a drift-quadrupole-drift through Cheetah 0.8.4 and through `LinacTracker` agreed to 0.4 percent of the beam size in `x`, which is the chromatic nonlinearity Cheetah's first-order map leaves out, and to 1e-15 in the converted sixth coordinate.

### Collective effects and charge

Each scan setting tracks the whole bunch, so the collective forces act at the tracker's physical `bunch_charge_C`. Independent particle minibatches would feel a different, noisy self-field and are not equivalent to minibatching independent images. The charge of the bunch is not recovered from normalised images; it is an input of the tracker (the charge of a Cheetah beam is ignored).

### Limits

- One screen, at the end of the line, and an `(n, 6)` bunch per call (no batched beams); axes are two of the six state columns, with no rotation of the image.
- The image has the dtype of the tracker (float64 by default); GPSR's targets are float32, so cast where they meet.
- Float32 gradients through CSR are rough (see [native tracking](../tracking_native.md)); choose particle count, bins and smoothing for gradient stability.
- Checked against GPSR 0.3.0 and Cheetah 0.8.4 on the CPU only (an optional test, skipped when they are not installed).
- `GraphedStep` inherits the capture limits of `LinacTracker` (no `csr_drift_model="retarded"`, wakes need `with_wake_windows`).

## Usage

```python
from slipstream import LinacTracker, ScanModel, Screen, GraphedStep

tracker = LinacTracker(specs, 150e6, bunch_charge_C=100e-12, csr=True, fast_collective_effects=True, device="cuda")
screen = Screen(shape=(64, 64), pixel_size=1.5e-4, psf_sigma_px=1.0)         # x on columns, y on rows
model = ScanModel(tracker, screen, variables=[("Q1", "k1")])
x = scan_table.clone()                                                         # (n_settings, 1), static
model.set_lattice_parameters(x)

def loss_fn():                                                                 # your generator and loss
    (images,) = model.track_and_observe(generator())
    return ((images - target) ** 2).sum()

step = GraphedStep(loss_fn, list(generator.parameters()))
for _ in range(n_iterations):
    step()                                                                     # replay; overwrites the .grad of the parameters
    optimiser.step()
```

Inside GPSR, hand a `ScanModel` to `GPSR(beam_generator, lattice)` in place of a Cheetah lattice: it has the `set_lattice_parameters(x)` and `track_and_observe(beam)` methods GPSR calls, `x` has shape `(n_settings, n_variables)`, and the beam may be a Cheetah `ParticleBeam` (converted at the tracker's reference energy; a differing `beam.energy` raises).

To change the scan, `copy_` into the static table and call `step()` again: the graph reads it at replay. Set `x.requires_grad_(True)` and pass it among the parameters to fit scan or calibration constants. `GraphedStep` runs eagerly when the parameters are on the CPU, so the same code runs everywhere.

A complete study, with CSR, a transverse deflecting cavity and a comparison of fits with and without CSR in the model, is in [`examples/clara_gpsr`](../../examples/clara_gpsr/README.md).

### API reference

```python
def from_cheetah(particles: torch.Tensor, energy_eV: float, mc2_eV: float=510998.95) -> torch.Tensor: ...
def to_cheetah(particles6: torch.Tensor, energy_eV: float, mc2_eV: float=510998.95) -> torch.Tensor: ...

@dataclass
class Screen:
    shape: tuple[int, int]
    pixel_size: float | tuple[float, float]
    centre: tuple[float, float] = (0.0, 0.0)
    axes: tuple[int, int] = (0, 2)
    psf_sigma_px: float = 0.0
    gain: float | torch.Tensor = 1.0
    background: float | torch.Tensor = 0.0
    def render(self, particles6: torch.Tensor, charge_C: float | torch.Tensor=1.0) -> torch.Tensor: ...

class ScanModel(torch.nn.Module):
    def __init__(self, tracker: LinacTracker, screen: Screen, variables: Sequence[tuple[str, str]]): ...
    def set_lattice_parameters(self, x: torch.Tensor) -> None: ...
    def track_and_observe(self, beam) -> tuple[torch.Tensor]: ...

class GraphedStep:
    def __init__(self, fn: Callable[[], torch.Tensor], params: Sequence[torch.Tensor], warmup: int=3): ...
    def __call__(self) -> torch.Tensor: ...
```

### `Screen`

`image[j, i]` has its centre at `centre + ((i - (nx - 1) / 2) * pixel_size[0], (j - (ny - 1) / 2) * pixel_size[1])` in the two state columns given by `axes` (`(0, 2)` is `x, y`; `(4, 5)` is `tau, delta`). Pixel sizes and centre are in the unit of those columns (metres for `x`, `y`, `tau`). `render` gives each of the `n` particles `charge_C / n`, so the default image is the fraction of the bunch per pixel.

### `ScanModel`

`variables` are `(element_name, setting)` pairs, the settings being those of `LinacTracker.with_settings` (`k1`, `k2`, `kick_x`, `kick_y`, `voltage_V`, `phase_rad`); an unknown element or a setting that does not apply to it raises at construction. `track_and_observe` returns `(images,)` with `images` of shape `(n_settings, ny, nx)`. The scan table is read where it stands: it keeps its autograd history, and inside a CUDA graph it is read at replay time.

### `GraphedStep`

`fn()` returns a scalar loss computed from tensors it holds; `params` are the leaf tensors whose `.grad` is wanted. The first call warms up on a side stream (so the dense smoothing operators, cuFFT plans and cached constants exist), captures `fn()` and `backward()` in one CUDA graph and replays it. Each call returns the loss tensor, which and whose `params[i].grad` the next call overwrites, so `clone` what must survive and run the optimiser outside the graph. Inputs are changed with in-place writes (`copy_`, `fill_`) before the call. Nothing in the step may read a tensor on the host.
