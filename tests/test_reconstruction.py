"""Differentiable screen, scan forward model and CUDA-graph step for gradient-based beam reconstruction.

The screen must conserve charge and the beam centroid, pass exact gradients to particle positions and to calibration constants, and
the scan model must equal a plainly built tracker at each setting, pass gradients to the scan table, and recover a known beam from
synthetic quad-scan images. On CUDA, one captured forward+backward step must replay new scan tables like an eager pass.
"""
import math
from types import SimpleNamespace

import pytest
import torch

from slipstream.reconstruction import GraphedStep, ScanModel, Screen, from_cheetah, to_cheetah
from slipstream.tracking.native.lattice_import import ElementSpec
from slipstream.tracking.native.linac import LinacTracker

ENERGY_EV, CHARGE_C = 150e6, 100e-12
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
F64 = torch.float64
SCALES = torch.tensor([2e-4, 2e-5, 2e-4, 2e-5, 30e-6, 1e-4], dtype=F64)


def _beam(n=1000, seed=9, device="cpu"):
    return (torch.randn(n, 6, generator=torch.Generator().manual_seed(seed), dtype=F64) * SCALES).to(device)


def _quad_line(k1=2.0):
    return [ElementSpec("D0", "drift", 0.2), ElementSpec("Q1", "quad", 0.1, k1=k1), ElementSpec("D1", "drift", 1.5)]


def _tracker(line, device="cpu", **kwargs):
    options = dict(always_change_p0=False, compile=False, compile_collective_chunks=False)
    options.update(kwargs)
    return LinacTracker(line, ENERGY_EV, device=device, dtype=F64, **options)


def _screen(**kwargs):
    return Screen(**{**dict(shape=(24, 24), pixel_size=4e-5), **kwargs})


def _fd(fn, x, direction, h):
    return (fn(x + h * direction) - fn(x - h * direction)) / (2.0 * h)


# ---------------------------------------------------------------------------------------------------------------------- screen
def test_screen_pixel_centre_and_split():
    screen = Screen(shape=(5, 5), pixel_size=1.0)
    one = torch.zeros(1, 6, dtype=F64)
    image = screen.render(one)
    assert image[2, 2] == 1.0 and image.sum() == 1.0
    half = torch.tensor([[0.5, 0, 0.25, 0, 0, 0]], dtype=F64)  # between columns 2 and 3, a quarter of the way to row 3
    image = screen.render(half)
    assert torch.allclose(image[2:4, 2:4], torch.tensor([[0.375, 0.375], [0.125, 0.125]], dtype=F64))


def test_screen_frame_edge_and_axes():
    screen = Screen(shape=(5, 5), pixel_size=1.0)
    edge = torch.zeros(1, 6, dtype=F64)
    edge[0, 0] = -2.5  # halfway between the first pixel centre and the frame outside: half the charge is lost
    assert screen.render(edge).sum() == pytest.approx(0.5)
    far = edge.clone()
    far[0, 0] = -50.0
    assert screen.render(far).sum() == 0.0
    tau_delta = Screen(shape=(5, 5), pixel_size=(1.0, 0.5), centre=(1.0, 0.0), axes=(4, 5))
    p = torch.zeros(1, 6, dtype=F64)
    p[0, 4], p[0, 5] = 2.0, -0.5  # one pixel right of the centre column, one pixel below the centre row
    assert tau_delta.render(p)[1, 3] == 1.0


def test_screen_conserves_charge_and_centroid():
    p = _beam().clamp(-2 * SCALES, 2 * SCALES)
    screen = _screen()
    image = screen.render(p, charge_C=2.0)
    assert float(image.sum()) == pytest.approx(2.0, rel=1e-12)
    columns = (torch.arange(24, dtype=F64) - 11.5) * 4e-5
    cx = float((image.sum(0) * columns).sum() / image.sum())
    cy = float((image.sum(1) * columns).sum() / image.sum())
    assert cx == pytest.approx(float(p[:, 0].mean()), abs=1e-12)
    assert cy == pytest.approx(float(p[:, 2].mean()), abs=1e-12)


def test_screen_psf_matches_separable_gaussian():
    sigma = 1.7
    screen = Screen(shape=(21, 21), pixel_size=1.0, psf_sigma_px=sigma)
    image = screen.render(torch.zeros(1, 6, dtype=F64))
    r = torch.arange(-10, 11, dtype=F64)
    k = torch.exp(-0.5 * (r / sigma) ** 2)
    k[r.abs() > math.ceil(3 * sigma)] = 0.0
    k = k / k.sum()
    assert torch.allclose(image, k[:, None] * k[None, :], atol=1e-15)
    assert float(image.sum()) == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("psf", [0.0, 1.2])
def test_screen_gradient_to_positions_matches_finite_differences(psf):
    p = _beam(200).clamp(-2 * SCALES, 2 * SCALES)
    screen = _screen(psf_sigma_px=psf)
    weights = torch.randn(24, 24, dtype=F64, generator=torch.Generator().manual_seed(2))
    direction = torch.randn(200, 6, dtype=F64, generator=torch.Generator().manual_seed(3)) * SCALES
    loss = lambda q: (screen.render(q) * weights).sum()  # noqa: E731
    leaf = p.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(loss(leaf), leaf)
    fd = _fd(loss, p, direction, 1e-6)
    assert float((grad * direction).sum()) == pytest.approx(float(fd), rel=1e-6)


def test_screen_gain_and_background_are_differentiable_calibrations():
    gain, background = torch.tensor(2.0, dtype=F64, requires_grad=True), torch.tensor(0.1, dtype=F64, requires_grad=True)
    screen = _screen(gain=gain, background=background)
    image = screen.render(_beam(100).clamp(-2 * SCALES, 2 * SCALES))
    assert float(image.sum().detach()) == pytest.approx(2.0 + 0.1 * 24 * 24, rel=1e-12)
    image.sum().backward()
    assert float(gain.grad) == pytest.approx(1.0, rel=1e-12) and float(background.grad) == 24 * 24


# ------------------------------------------------------------------------------------------------------------------- coordinates
def test_cheetah_conversion_is_exact_and_round_trips():
    p = _beam(50)
    p[:, 5] = torch.linspace(-0.05, 0.05, 50, dtype=F64)
    cheetah = to_cheetah(p, ENERGY_EV)
    assert cheetah.shape == (50, 7) and torch.all(cheetah[:, 6] == 1.0)
    p0c = math.sqrt(ENERGY_EV**2 - 0.51099895e6**2)
    total = ENERGY_EV + cheetah[:, 5] * p0c  # E from (E - E0) / (p0 c)
    exact = torch.sqrt(total**2 - 0.51099895e6**2) / p0c - 1.0
    assert torch.allclose(p[:, 5], exact, rtol=0.0, atol=1e-13)
    assert torch.allclose(from_cheetah(cheetah, ENERGY_EV), p, rtol=1e-13, atol=1e-18)


def test_cheetah_conversion_keeps_float32_precision_at_small_offsets():
    delta = torch.tensor([1e-6, -3e-6, 1e-5], dtype=torch.float32)
    back = from_cheetah(to_cheetah(torch.cat([torch.zeros(3, 5), delta[:, None]], dim=1), ENERGY_EV), ENERGY_EV)
    assert torch.allclose(back[:, 5], delta, rtol=1e-5)


# -------------------------------------------------------------------------------------------------------------------- scan model
def test_scan_model_matches_directly_built_tracker_at_each_setting():
    p = _beam()
    screen = _screen()
    model = ScanModel(_tracker(_quad_line()), screen, [("Q1", "k1")])
    model.set_lattice_parameters(torch.tensor([[1.0], [2.5], [-1.2]], dtype=F64))
    (images,) = model.track_and_observe(p)
    assert images.shape == (3, 24, 24)
    for image, k1 in zip(images, (1.0, 2.5, -1.2)):
        out, _ = _tracker(_quad_line(k1)).track(p)
        assert torch.allclose(image, screen.render(out), rtol=0.0, atol=1e-13)
    model.set_lattice_parameters(torch.tensor([2.5], dtype=F64))  # one setting may be a plain vector
    assert model.track_and_observe(p)[0].shape == (1, 24, 24)


def test_scan_model_gradient_to_the_scan_table_matches_finite_differences():
    p = _beam(500)
    model = ScanModel(_tracker(_quad_line()), _screen(), [("Q1", "k1")])
    weights = torch.randn(24, 24, dtype=F64, generator=torch.Generator().manual_seed(4))

    def loss(x):
        model.set_lattice_parameters(x)
        return (model.track_and_observe(p)[0] * weights).sum()

    x = torch.tensor([[1.0], [2.5]], dtype=F64, requires_grad=True)
    (grad,) = torch.autograd.grad(loss(x), x)
    direction = torch.tensor([[1.0], [-0.5]], dtype=F64)
    fd = _fd(loss, x.detach(), direction, 1e-6)
    assert float((grad * direction).sum()) == pytest.approx(float(fd), rel=1e-6)


def test_scan_model_accepts_a_cheetah_like_beam():
    p = _beam()
    model = ScanModel(_tracker(_quad_line()), _screen(), [("Q1", "k1")])
    model.set_lattice_parameters(torch.tensor([[2.0]], dtype=F64))
    reference = model.track_and_observe(p)[0]
    beam = SimpleNamespace(particles=to_cheetah(p, ENERGY_EV), energy=torch.tensor(ENERGY_EV), total_charge=CHARGE_C)
    assert torch.allclose(model.track_and_observe(beam)[0], reference, rtol=0.0, atol=1e-12)
    wrong = SimpleNamespace(particles=to_cheetah(p, ENERGY_EV), energy=ENERGY_EV * 1.01)
    with pytest.raises(ValueError, match="energy"):
        model.track_and_observe(wrong)


def test_scan_model_rejects_bad_requests_early():
    tracker = _tracker(_quad_line())
    with pytest.raises(KeyError):
        ScanModel(tracker, _screen(), [("QX", "k1")])
    with pytest.raises(ValueError):
        ScanModel(tracker, _screen(), [("Q1", "voltage_V")])
    model = ScanModel(tracker, _screen(), [("Q1", "k1")])
    with pytest.raises(RuntimeError):
        model.track_and_observe(_beam(10))
    with pytest.raises(ValueError):
        model.set_lattice_parameters(torch.zeros(3, 2, dtype=F64))
    model.set_lattice_parameters(torch.zeros(1, 1, dtype=F64))
    with pytest.raises(ValueError):
        model.track_and_observe(_beam(10)[None])


def _chicane():
    line = [ElementSpec("D0", "drift", 0.2), ElementSpec("Q1", "quad", 0.1, k1=1.0)]
    for i, angle in enumerate((0.08, -0.08, -0.08, 0.08)):
        line += [ElementSpec(f"B{i}", "bend", 0.2, angle=angle, csr=True), ElementSpec(f"DB{i}", "drift", 0.4)]
    return line


def test_scan_model_through_csr_gives_a_finite_gradient_to_the_beam():
    p = _beam(512).requires_grad_(True)
    tracker = _tracker(_chicane(), bunch_charge_C=CHARGE_C, csr=True)
    model = ScanModel(tracker, _screen(shape=(16, 16), pixel_size=6e-5), [("Q1", "k1")])
    model.set_lattice_parameters(torch.tensor([[1.0], [2.0]], dtype=F64))
    (images,) = model.track_and_observe(p)
    (images ** 2).sum().backward()
    assert torch.isfinite(p.grad).all() and float(p.grad.abs().max()) > 0.0


def _true_beam_scales():
    return torch.tensor([1.5e-4, 2.0e-5, 1.0e-4, 3.0e-5], dtype=F64)  # sigma of x, px, y, py


def _generator(latent, scales):
    return torch.cat([latent[:, :4] * scales, latent[:, 4:] * SCALES[4:]], dim=1)


def test_quad_scan_reconstruction_recovers_the_beam_widths():
    """Synthetic quad scan (7 strengths, x and y on one screen) of a beam drawn from other random numbers than the fit uses, so the
    result is limited by the finite sample (about 10 percent on the weakly constrained divergences), not by an inverse crime."""
    ks = torch.linspace(-6.0, 8.0, 7, dtype=F64)[:, None]
    screen = Screen(shape=(40, 40), pixel_size=3e-5, psf_sigma_px=0.8)
    line = [ElementSpec("D0", "drift", 0.2), ElementSpec("Q1", "quad", 0.1, k1=2.0), ElementSpec("D1", "drift", 1.0)]
    model = ScanModel(_tracker(line), screen, [("Q1", "k1")])
    model.set_lattice_parameters(ks)
    truth = _true_beam_scales()
    with torch.no_grad():
        target = model.track_and_observe(_generator(torch.randn(4000, 6, dtype=F64, generator=torch.Generator().manual_seed(1)), truth))[0]
    latent = torch.randn(2000, 6, dtype=F64, generator=torch.Generator().manual_seed(2))
    log_scales = (truth * torch.tensor([1.3, 0.7, 1.25, 0.8], dtype=F64)).log().requires_grad_(True)
    optimiser = torch.optim.Adam([log_scales], lr=0.05)
    first = None
    for _ in range(120):
        optimiser.zero_grad()
        loss = ((model.track_and_observe(_generator(latent, log_scales.exp()))[0] - target) ** 2).sum()
        loss.backward()
        optimiser.step()
        first = float(loss) if first is None else first
    assert float(loss) < 0.3 * first
    assert torch.allclose(log_scales.detach().exp(), truth, rtol=0.15), (log_scales.exp().detach(), truth)


# ----------------------------------------------------------------------------------------------------------------- graphed step
def test_graphed_step_on_cpu_is_an_eager_step_without_gradient_accumulation():
    w = torch.tensor([1.0, 2.0], dtype=F64, requires_grad=True)
    x = torch.tensor([3.0, 4.0], dtype=F64)
    step = GraphedStep(lambda: (w * x).pow(2).sum(), [w])
    assert float(step()) == pytest.approx(9 + 64)
    first = w.grad.clone()
    x.copy_(torch.tensor([1.0, 1.0], dtype=F64))
    assert float(step()) == pytest.approx(1 + 4)
    assert torch.allclose(first, torch.tensor([18.0, 64.0], dtype=F64))
    assert torch.allclose(w.grad, torch.tensor([2.0, 4.0], dtype=F64))


@needs_cuda
@pytest.mark.parametrize("collective", [False, True])
def test_graphed_step_replays_new_scan_tables_like_an_eager_pass(collective):
    """One captured track + loss + backward over a two-setting scan; the scan table and the target are static and refilled."""
    device = "cuda"
    kwargs = dict(bunch_charge_C=CHARGE_C, csr=True, fast_collective_effects=True) if collective else {}
    model = ScanModel(_tracker(_chicane() if collective else _quad_line(), device=device, **kwargs),
                      _screen(shape=(16, 16), pixel_size=6e-5), [("Q1", "k1")])
    p = _beam(2048, device=device).requires_grad_(True)
    x = torch.tensor([[1.0], [2.0]], dtype=F64, device=device)
    target = torch.rand(2, 16, 16, dtype=F64, device=device) * 0.01
    model.set_lattice_parameters(x)

    def loss_fn():
        return ((model.track_and_observe(p)[0] - target) ** 2).sum()

    tables = [torch.tensor(t, dtype=F64, device=device) for t in ([[1.0], [2.0]], [[-0.5], [3.0]])]

    def eager(table):  # a reference pass; its autograd graph dies with this frame, before the capture
        x.copy_(table)
        p.grad = None
        value = loss_fn()
        value.backward()
        return float(value.detach()), p.grad.clone()

    references = [eager(table) for table in tables]
    p.grad = None
    step = GraphedStep(loss_fn, [p])
    for table, (loss_ref, grad_ref) in zip(tables, references):
        x.copy_(table)
        loss = float(step())
        grad = p.grad.clone()
        assert loss == pytest.approx(loss_ref, rel=1e-9)
        assert float((grad - grad_ref).abs().max()) <= 1e-6 * float(grad_ref.abs().max())


def test_scan_model_runs_inside_the_gpsr_wrapper():
    """Optional: the real GPSR wrapper (it deep-copies the lattice) trains a Cheetah beam generator through a Slipstream scan."""
    pytest.importorskip("cheetah")
    pytest.importorskip("gpsr")
    from gpsr.beams import NNParticleBeamGenerator, NNTransform
    from gpsr.modeling import GPSR

    model = ScanModel(_tracker(_quad_line()), _screen(shape=(32, 32)), [("Q1", "k1")])
    generator = NNParticleBeamGenerator(500, ENERGY_EV, transformer=NNTransform(2, 20, output_scale=1e-3))
    fit = GPSR(generator, model)
    (images,) = fit(torch.linspace(-4.0, 6.0, 3)[:, None])
    assert images.shape == (3, 32, 32)
    (images ** 2).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in fit.beam_generator.parameters())
