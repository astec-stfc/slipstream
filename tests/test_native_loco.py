"""Tests for the LOCO first slice -- see plans/LOCO_PLAN.md.

Phase 0/1: HKICK/VKICK/MONI parsing and transport, cross-checked against a
live ``elegant`` run (skipped if elegant isn't found), plus the
differentiable-at-zero-kick guarantee LOCO's own ORM needs.
Phase 2: ``RingMap.turn_with_taps``.
Phase 3: ``analysis.loco.orbit_response_matrix``.
Phase 4: a minimal noise-free K1 recovery, orchestrated locally (see
plans/LOCO_PLAN.md's explicit "no LocoProblem yet" guidance).
Phase 5: ``analysis.loco.apply_calibration`` (BPM gain/roll, corrector
gain) -- known-answer tensor checks, then synthetic recovery (gains-only,
K1+gains, seeded-noise).
Phase 6: tensor-safe ``_misalign`` (dx/dy/tilt differentiable exactly at
zero), occurrence-safe ``RingMap.with_strengths(dx=/dy=/tilt=)``, and a
coupled quad-roll + K1 synthetic recovery with a rank/gauge audit.
Phase 7: native ORM vs live-elegant central kick differences on the real
``BR14_BST_Aligned_Matched.lte``/``newBST`` booster lattice (a selected
subset of channels -- see plans/LOCO_PLAN.md's Verification gate 8).
"""
from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np
import pytest
import scipy.optimize
import torch

from slipstream.analysis.loco import apply_calibration, orbit_response_matrix
from slipstream.analysis.nonlinear import closed_orbit
from slipstream.analysis.nonlinear_opt import unique_names
from slipstream.tracking.native import parse_elegant_lattice
from slipstream.tracking.native import ring_warp
from slipstream.tracking.native.elements import _corrector_kick
from slipstream.tracking.native.ring import RingMap
from slipstream.tracking.native.tracker import _apply_one_element

try:
    from slipstream.tracking.elegant import (
        elegant_env, find_elegant_exe, read_coordinate_watch_pages,
        read_sdds_table, write_coordinate_bunch,
    )
except ModuleNotFoundError as exc:
    if exc.name != "slipstream.tracking.elegant":
        raise
    ELEGANT_EXE = None
    _elegant_skip_reason = "optional elegant development helpers unavailable"
else:
    ELEGANT_EXE = find_elegant_exe()
    _elegant_skip_reason = "no elegant binary found"
needs_elegant = pytest.mark.skipif(ELEGANT_EXE is None, reason=_elegant_skip_reason)
E0 = 1e9
MC2 = 510998.95

DIAMOND2 = Path(__file__).resolve().parent.parent / "examples" / "diamond2booster"
BR14_BST_LTE = DIAMOND2 / "lattice" / "BR14_BST_Aligned_Matched.lte"
E0_100MEV_eV = 100001284.06493436  # matches lattice_100MeV.yaml/optics_100MeV.yaml's own reference energy
needs_br14_bst = pytest.mark.skipif(not BR14_BST_LTE.exists(), reason="BR14_BST_Aligned_Matched.lte not present")

# Small synthetic FODO+BPM+corrector ring -- extends test_ring_nonlinear.py's
# own bare FODO fixture with the devices real br14_simple.lte lacks (see
# plans/LOCO_PLAN.md's "Real lattice evidence").
LOCO_FODO_LTE = """D1: DRIFT, L=0.4
D2: DRIFT, L=0.3
QF: KQUAD, L=0.3, K1=1.55
QD: KQUAD, L=0.3, K1=-1.65
HCM: HKICK, L=0.075, KICK=0
VCM: VKICK, L=0.075, KICK=0
BPM: MONI, L=0.05
CELL: LINE=(HCM, VCM, BPM, QF, D2, QD, D1)
RING: LINE=(CELL,CELL,CELL,CELL,CELL,CELL)
"""


@pytest.fixture(scope="module")
def lte_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("loco_fodo") / "loco_fodo.lte"
    p.write_text(LOCO_FODO_LTE)
    return p


@pytest.fixture(scope="module")
def specs(lte_path):
    raw = parse_elegant_lattice(lte_path, "RING")
    return unique_names(raw, kinds=("drift", "hkick", "vkick", "quad"))


@pytest.fixture(scope="module")
def ring(specs):
    return RingMap(specs, E0, MC2, device="cpu", dtype=torch.float64, exact_dipole=False)


def _names(specs, kind: str) -> list[str]:
    return [s.name for s in specs if s.kind == kind]


# --- Phase 0/1: parser + corrector transport -----------------------------

def test_parser_keeps_hkick_vkick_moni_with_occurrence(lte_path):
    raw = parse_elegant_lattice(lte_path, "RING")
    hkicks = [s for s in raw if s.kind == "hkick"]
    vkicks = [s for s in raw if s.kind == "vkick"]
    bpms = [s for s in raw if s.kind == "drift" and s.name == "BPM"]
    assert len(hkicks) == len(vkicks) == len(bpms) == 6
    assert [s.occurrence for s in hkicks] == list(range(1, 7))
    assert all(s.length == pytest.approx(0.075) and s.kick_x == 0.0 for s in hkicks)
    assert all(s.length == pytest.approx(0.05) for s in bpms)


def test_zero_length_kick_is_not_dropped(tmp_path):
    p = tmp_path / "zero_length_kick.lte"
    p.write_text("HCM: HKICK, L=0, KICK=0.001;\nRING: LINE=(HCM)\n")
    specs = parse_elegant_lattice(p, "RING")
    assert len(specs) == 1 and specs[0].kind == "hkick" and specs[0].length == 0.0


@pytest.mark.parametrize("plane,length,kick", [("x", 0.075, 0.003), ("y", 0.075, 0.002), ("x", 0.0, 0.003), ("y", 0.0, 0.002)])
@needs_elegant
def test_corrector_kick_matches_live_elegant(tmp_path, plane, length, kick):
    """Real N-particle transport vs a live elegant HKICK/VKICK, finite and
    zero length -- Phase 0's verification gate."""
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(0, 1e-3, n)
    xp = rng.normal(0, 2e-4, n)
    y = rng.normal(0, 0.8e-3, n)
    yp = rng.normal(0, 1.5e-4, n)
    dp = rng.normal(0, 5e-3, n)
    t = np.zeros(n)

    p_central_mev = 3000.0
    gamma0 = p_central_mev / (MC2 / 1e6)
    energy_eV = gamma0 * MC2
    p_central = (gamma0**2 - 1) ** 0.5
    p = p_central * (1 + dp)

    input_bunch = tmp_path / "in.sdds"
    write_coordinate_bunch(input_bunch, x, xp, y, yp, t, p, p_central)
    etype = "HKICK" if plane == "x" else "VKICK"
    (tmp_path / "run.lte").write_text(
        f'ELM: {etype}, L={length!r}, KICK={kick!r};\nW1: WATCH, FILENAME="%s.w1", MODE="coordinates";\nRING: LINE=(ELM,W1)\n'
    )
    (tmp_path / "run.ele").write_text(f"""\
&run_setup
    lattice = "run.lte", use_beamline = "RING", rootname = "run",
    p_central_mev = {p_central_mev!r}, default_order = 3
&end
&run_control n_passes = 1 &end
&sdds_beam input = "{str(input_bunch).replace(chr(92), '/')}", input_type = "elegant" &end
&track &end
""")
    proc = subprocess.run([str(ELEGANT_EXE), "run.ele"], cwd=tmp_path, capture_output=True, text=True, env=elegant_env())
    watch = tmp_path / "run.w1"
    assert watch.exists(), f"elegant run failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    out = read_coordinate_watch_pages(watch)[0]

    pz = (1 + dp) / np.sqrt(1 + xp**2 + yp**2)
    px0, py0 = xp * pz, yp * pz
    state = torch.tensor(np.stack([x, px0, y, py0, np.zeros(n), dp], axis=-1), dtype=torch.float64)
    energy_t, mc2_t = torch.tensor(energy_eV, dtype=torch.float64), torch.tensor(MC2, dtype=torch.float64)
    state_out = _corrector_kick(state, length, kick, plane, energy_t, mc2_t)
    xn, pxn, yn, pyn, _, deltan = [state_out[:, i].numpy() for i in range(6)]
    pzn = np.sqrt((1 + deltan) ** 2 - pxn**2 - pyn**2)
    xpn, ypn = pxn / pzn, pyn / pzn

    delta_elegant = out["p"] / p_central - 1
    for m, th in zip((xn, xpn, yn, ypn, deltan), (out["x"], out["xp"], out["y"], out["yp"], delta_elegant)):
        scale = np.abs(th).std() + 1e-30
        assert np.abs(m - th).max() / scale < 1e-4


def test_corrector_gradient_at_zero_matches_finite_difference(ring):
    """The whole point of Phase 1's rewrite: autograd through a corrector
    at its nominal *zero* kick must not silently vanish (a plain
    ``angle != 0.0`` branch would detach the gradient exactly there)."""
    name = _names(ring.specs, "hkick")[0]
    kick = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    state0 = torch.zeros(6, dtype=torch.float64)

    def final_x(k):
        r = ring.with_strengths(kick_x={name: k})
        return r.turn(state0, eager=True)[0]

    g = torch.autograd.grad(final_x(kick), kick)[0]
    h = 1e-6
    fd = (final_x(torch.tensor(h, dtype=torch.float64)) - final_x(torch.tensor(-h, dtype=torch.float64))) / (2 * h)
    assert g.item() != 0.0
    assert g.item() == pytest.approx(fd.item(), rel=1e-6)


def test_double_single_rejects_nonzero_corrector_kick(specs):
    ds_ring = RingMap(specs, E0, MC2, device="cpu", n_kicks=2, n_kicks_bend=8, integration_order=6, exact_dipole=True, double_single=True)
    name = _names(specs, "hkick")[0]
    with pytest.raises(NotImplementedError):
        ds_ring.with_strengths(kick_x={name: 0.001})
    ds_ring.with_strengths(kick_x={name: 0.0})  # zero override: a no-op, must not raise


def test_ring_warp_pack_falls_back_for_corrector_elements(ring):
    packed = ring_warp.pack(ring.params, torch.device("cpu"), torch.float64)
    assert packed is None  # any is_kick element disables the Warp fast path (unsupported this slice)


# --- Phase 2: turn_with_taps ----------------------------------------------

def test_turn_with_taps_matches_manual_loop_and_plain_turn(ring):
    bpm_names = _names(ring.specs, "drift")
    bpm_names = [n for n in bpm_names if n.startswith("BPM#")]
    assert len(bpm_names) == 6
    index_of = {s.name: i for i, s in enumerate(ring.specs)}
    tap_indices = [index_of[n] for n in bpm_names]

    torch.manual_seed(0)
    state0 = torch.randn(6, dtype=torch.float64) * 1e-4
    state0[4] = 0.0
    state0[5] = 0.0

    final_ref = ring.turn(state0, eager=True)
    final, tapped = ring.turn_with_taps(state0, tap_indices)
    assert torch.allclose(final, final_ref, atol=1e-15, rtol=0)
    assert tapped.shape == (6, 6)

    # manual per-element loop, independent of turn_with_taps's own bookkeeping
    manual = []
    state = state0
    for i, p in enumerate(ring.params):
        state = _apply_one_element(state, p, ring._energy_t, ring._mc2_t)[0]
        if i in tap_indices:
            manual.append(state)
    manual = torch.stack([manual[tap_indices.index(i)] for i in sorted(tap_indices)], dim=0)
    assert torch.allclose(tapped, manual, atol=1e-15, rtol=0)


def test_turn_with_taps_gradient_reaches_kick_and_k1(ring):
    # d(x)/d(kick) at kick=0: covered directly, and against a finite
    # difference, by test_corrector_gradient_at_zero_matches_finite_difference.
    # d(x)/d(K1) needs a *nonzero background orbit* to be meaningfully
    # nonzero at all (a quad's K1*x kick is identically zero on an exactly
    # zero orbit, regardless of K1 -- not a bug, see
    # test_orm_create_graph_gives_nonzero_k1_derivative for the operating
    # point LOCO's own nested fit actually needs, d^2x/d(kick)dK1).
    hcm_name = _names(ring.specs, "hkick")[0]
    qf_name = _names(ring.specs, "quad")[0]
    k1 = torch.tensor(1.55, dtype=torch.float64, requires_grad=True)
    r = ring.with_strengths(kick_x={hcm_name: 0.001}, k1={qf_name: k1})
    index_of = {s.name: i for i, s in enumerate(r.specs)}
    bpm_idx = [index_of[n] for n in _names(r.specs, "drift") if n.startswith("BPM#")]
    state0 = torch.zeros(6, dtype=torch.float64)
    _, tapped = r.turn_with_taps(state0, bpm_idx)
    loss = tapped[..., 0].sum()
    g_k1 = torch.autograd.grad(loss, k1)[0]
    assert g_k1.item() != 0.0


# --- Phase 3: orbit response matrix ---------------------------------------

def _corrector_and_bpm_names(specs):
    correctors = _names(specs, "hkick") + _names(specs, "vkick")
    bpms = [n for n in _names(specs, "drift") if n.startswith("BPM#")]
    return correctors, bpms


def test_orm_matches_central_kick_differences(ring):
    """Central finite differences must start each side from *that side's
    own* closed orbit (``closed_orbit(plus/minus, ...)``), not a fixed
    zero state: a BPM upstream of the perturbed corrector (in ring-index
    order) only sees a nonzero response via the periodic wraparound
    ``co`` itself carries -- a fixed-zero-state single pass would give a
    false zero there and silently validate the pre-fix closed-orbit bug
    (see analysis/nonlinear.py::_closed_orbit_and_matrix's own docstring)."""
    correctors, bpms = _corrector_and_bpm_names(ring.specs)
    J = orbit_response_matrix(ring, correctors, bpms)
    assert J.shape == (len(bpms), 2, len(correctors))

    index_of = {s.name: i for i, s in enumerate(ring.specs)}
    tap_indices = [index_of[n] for n in bpms]
    h = 1e-6
    for j, name in enumerate(correctors[:3]):  # a few columns is enough to confirm the pattern
        plus = ring.with_strengths(**{"kick_x" if name.startswith("HCM") else "kick_y": {name: h}})
        minus = ring.with_strengths(**{"kick_x" if name.startswith("HCM") else "kick_y": {name: -h}})
        co_plus = closed_orbit(plus, delta=0.0)
        co_minus = closed_orbit(minus, delta=0.0)
        _, tp = plus.turn_with_taps(co_plus, tap_indices)
        _, tm = minus.turn_with_taps(co_minus, tap_indices)
        fd = (tp[..., (0, 2)] - tm[..., (0, 2)]) / (2 * h)
        assert torch.allclose(J[:, :, j], fd, atol=1e-9, rtol=1e-5)


def test_orm_uncoupled_ring_has_zero_cross_plane_blocks(ring):
    correctors, bpms = _corrector_and_bpm_names(ring.specs)
    J = orbit_response_matrix(ring, correctors, bpms)
    n_h = len(_names(ring.specs, "hkick"))
    # columns 0..n_h-1 are H correctors: must drive only the x row (index 0), not y (index 1)
    assert torch.allclose(J[:, 1, :n_h], torch.zeros_like(J[:, 1, :n_h]), atol=1e-12)
    # remaining columns are V correctors: must drive only y, not x
    assert torch.allclose(J[:, 0, n_h:], torch.zeros_like(J[:, 0, n_h:]), atol=1e-12)


def test_orm_create_graph_gives_nonzero_k1_derivative(ring):
    qf_name = _names(ring.specs, "quad")[0]
    k1 = torch.tensor(1.55, dtype=torch.float64, requires_grad=True)
    r = ring.with_strengths(k1={qf_name: k1})
    correctors, bpms = _corrector_and_bpm_names(r.specs)
    J = orbit_response_matrix(r, correctors[:2], bpms, create_graph=True)
    g = torch.autograd.grad(J.sum(), k1)[0]
    assert g.item() != 0.0


# --- Phase 4: minimal noise-free K1 recovery ------------------------------

def test_minimal_k1_loco_fit_recovers_injected_error(specs):
    """Fit two QF K1 errors to a noise-free synthetic ORM built from a
    'truth' ring with those same two errors injected -- Phase 4's own
    verification gate (recovered parameters + normalized ORM residual)."""
    truth_ring = RingMap(specs, E0, MC2, device="cpu", dtype=torch.float64, exact_dipole=False)
    correctors, bpms = _corrector_and_bpm_names(specs)
    qf_names = _names(specs, "quad")[:2]  # two identifiable knobs
    injected = {qf_names[0]: 0.02, qf_names[1]: -0.015}  # fractional K1 errors (relative to nominal 1.55)

    nominal_k1 = {s.name: s.k1 for s in specs if s.name in qf_names}
    truth = truth_ring.with_strengths(k1={n: nominal_k1[n] * (1 + injected[n]) for n in qf_names})
    J_truth = orbit_response_matrix(truth, correctors, bpms).detach().numpy().reshape(-1)

    def model_orm(theta_np: np.ndarray) -> torch.Tensor:
        th = torch.as_tensor(theta_np, dtype=torch.float64)
        r = truth_ring.with_strengths(k1={n: th[i] for i, n in enumerate(qf_names)})
        return orbit_response_matrix(r, correctors, bpms, create_graph=False)

    def residual(theta_np: np.ndarray) -> np.ndarray:
        return model_orm(theta_np).numpy().reshape(-1) - J_truth

    def jac(theta_np: np.ndarray) -> np.ndarray:
        def observe(th: torch.Tensor) -> torch.Tensor:
            r = truth_ring.with_strengths(k1={n: th[i] for i, n in enumerate(qf_names)})
            return orbit_response_matrix(r, correctors, bpms, create_graph=True).reshape(-1)
        th = torch.tensor(theta_np, dtype=torch.float64, requires_grad=True)
        return torch.autograd.functional.jacobian(observe, th, vectorize=True).numpy()

    theta_start = np.array([nominal_k1[n] for n in qf_names])
    result = scipy.optimize.least_squares(
        residual, theta_start, jac=jac, method="trf", xtol=1e-15, ftol=1e-15, gtol=1e-15,
    )

    recovered = {n: result.x[i] for i, n in enumerate(qf_names)}
    for n in qf_names:
        truth_k1 = nominal_k1[n] * (1 + injected[n])
        assert recovered[n] == pytest.approx(truth_k1, rel=1e-6)
    final_residual_norm = np.linalg.norm(residual(result.x)) / (np.linalg.norm(J_truth) + 1e-30)
    assert final_residual_norm < 1e-8


# --- Phase 5: calibration (BPM gain/roll, corrector gain) -----------------

def test_apply_calibration_known_answer_gain_and_corrector_gain():
    torch.manual_seed(0)
    n_bpm, n_corr = 4, 3
    J = torch.randn(n_bpm, 2, n_corr, dtype=torch.float64)
    ones_b, zeros_b = torch.ones(n_bpm, dtype=torch.float64), torch.zeros(n_bpm, dtype=torch.float64)
    ones_c = torch.ones(n_corr, dtype=torch.float64)

    gx = ones_b.clone()
    gx[0] = 1.1
    J_cal = apply_calibration(J, gx, ones_b, zeros_b, ones_c)
    assert torch.allclose(J_cal[0, 0, :], 1.1 * J[0, 0, :])
    assert torch.allclose(J_cal[0, 1, :], J[0, 1, :])  # y row untouched by an x-only gain
    assert torch.allclose(J_cal[1:], J[1:])  # other BPMs untouched

    gc = ones_c.clone()
    gc[1] = 1.2
    J_cal2 = apply_calibration(J, ones_b, ones_b, zeros_b, gc)
    assert torch.allclose(J_cal2[:, :, 1], 1.2 * J[:, :, 1])
    assert torch.allclose(J_cal2[:, :, (0, 2)], J[:, :, (0, 2)])


def test_apply_calibration_known_answer_roll():
    torch.manual_seed(1)
    n_bpm, n_corr = 3, 2
    J = torch.randn(n_bpm, 2, n_corr, dtype=torch.float64)
    ones_b, ones_c = torch.ones(n_bpm, dtype=torch.float64), torch.ones(n_corr, dtype=torch.float64)
    roll = torch.zeros(n_bpm, dtype=torch.float64)
    theta = 0.05
    roll[0] = theta
    J_cal = apply_calibration(J, ones_b, ones_b, roll, ones_c)
    c, s = math.cos(theta), math.sin(theta)
    expected_x0 = c * J[0, 0, :] - s * J[0, 1, :]
    expected_y0 = s * J[0, 0, :] + c * J[0, 1, :]
    assert torch.allclose(J_cal[0, 0, :], expected_x0)
    assert torch.allclose(J_cal[0, 1, :], expected_y0)
    assert torch.allclose(J_cal[1:], J[1:])  # un-rolled BPMs untouched


def _fit_with_exact_jacobian(residual_fn, theta_start: np.ndarray) -> np.ndarray:
    """``residual_fn(theta: torch.Tensor, want_grad: bool) -> torch.Tensor``
    (a 1-d residual vector) fit by ``scipy.optimize.least_squares`` with
    an exact autograd Jacobian -- the same NumPy/Torch boundary as
    ``test_minimal_k1_loco_fit_recovers_injected_error``."""
    def residual(theta_np):
        return residual_fn(torch.as_tensor(theta_np, dtype=torch.float64)).detach().numpy()

    def jac(theta_np):
        # One vectorized autograd call, not a per-output-element
        # torch.autograd.grad loop -- the latter replays the whole
        # (possibly nested, create_graph=True) graph once per residual
        # element, which is fine on this file's tiny synthetic ring but
        # catastrophic on a real ~900-element lattice (see D2B_LOCO's
        # own story, which hit exactly this).
        th = torch.tensor(theta_np, dtype=torch.float64, requires_grad=True)
        return torch.autograd.functional.jacobian(residual_fn, th, vectorize=True).numpy()

    result = scipy.optimize.least_squares(residual, theta_start, jac=jac, method="trf", xtol=1e-15, ftol=1e-15, gtol=1e-15)
    return result.x


def test_gains_only_synthetic_recovery(ring):
    """Recover BPM x/y gains (roll=0, corrector gains fixed at 1 -- the
    gauge) from a noise-free measured ORM built with known gain errors,
    the lattice itself held at its nominal K1 throughout."""
    correctors, bpms = _corrector_and_bpm_names(ring.specs)
    J0 = orbit_response_matrix(ring, correctors, bpms).detach()
    n_bpm, n_corr = len(bpms), len(correctors)
    ones_b, ones_c, zeros_b = torch.ones(n_bpm, dtype=torch.float64), torch.ones(n_corr, dtype=torch.float64), torch.zeros(n_bpm, dtype=torch.float64)

    true_gx = ones_b.clone()
    true_gx[0], true_gx[2] = 1.03, 0.97
    true_gy = ones_b.clone()
    true_gy[1], true_gy[3] = 0.95, 1.05
    measured = apply_calibration(J0, true_gx, true_gy, zeros_b, ones_c).numpy().reshape(-1)

    def residual_fn(theta: torch.Tensor) -> torch.Tensor:
        gx, gy = theta[:n_bpm], theta[n_bpm:]
        return apply_calibration(J0, gx, gy, zeros_b, ones_c).reshape(-1) - torch.as_tensor(measured)

    theta_start = np.concatenate([np.ones(n_bpm), np.ones(n_bpm)])
    x = _fit_with_exact_jacobian(residual_fn, theta_start)
    assert x[:n_bpm] == pytest.approx(true_gx.numpy(), rel=1e-6)
    assert x[n_bpm:] == pytest.approx(true_gy.numpy(), rel=1e-6)


def test_k1_and_gains_identifiable_combination(specs):
    """K1 error(s) and BPM gain error(s) fit jointly, noise-free --
    Phase 5's own 'identifiable combination' gate. Corrector gains stay
    fixed at 1 (the gauge)."""
    truth_ring = RingMap(specs, E0, MC2, device="cpu", dtype=torch.float64, exact_dipole=False)
    correctors, bpms = _corrector_and_bpm_names(specs)
    qf_name = _names(specs, "quad")[0]
    nominal_k1 = next(s.k1 for s in specs if s.name == qf_name)
    n_bpm = len(bpms)
    ones_b, zeros_b, ones_c = torch.ones(n_bpm, dtype=torch.float64), torch.zeros(n_bpm, dtype=torch.float64), torch.ones(len(correctors), dtype=torch.float64)

    true_k1 = nominal_k1 * 1.02
    true_gx = ones_b.clone()
    true_gx[0] = 1.04
    truth = truth_ring.with_strengths(k1={qf_name: true_k1})
    measured = apply_calibration(orbit_response_matrix(truth, correctors, bpms), true_gx, ones_b, zeros_b, ones_c).detach().numpy().reshape(-1)

    def residual_fn(theta: torch.Tensor) -> torch.Tensor:
        k1, gx = theta[0], theta[1:]
        r = truth_ring.with_strengths(k1={qf_name: k1})
        J = orbit_response_matrix(r, correctors, bpms, create_graph=True)
        return apply_calibration(J, gx, ones_b, zeros_b, ones_c).reshape(-1) - torch.as_tensor(measured)

    theta_start = np.concatenate([[nominal_k1], np.ones(n_bpm)])
    x = _fit_with_exact_jacobian(residual_fn, theta_start)
    assert x[0] == pytest.approx(true_k1, rel=1e-5)
    assert x[1] == pytest.approx(true_gx[0].item(), rel=1e-5)


def test_gains_recovery_with_seeded_bpm_noise_weighted_residual(ring):
    """Seeded BPM noise on the measured ORM, residuals weighted by the
    assumed measurement sigma -- per plans/LOCO_PLAN.md's Phase 5 gate,
    the recovered gain is checked against its own *propagated*
    uncertainty (linear covariance (J^T W J)^-1), not an invented
    tolerance."""
    correctors, bpms = _corrector_and_bpm_names(ring.specs)
    J0 = orbit_response_matrix(ring, correctors, bpms).detach()
    n_bpm, n_corr = len(bpms), len(correctors)
    ones_b, ones_c, zeros_b = torch.ones(n_bpm, dtype=torch.float64), torch.ones(n_corr, dtype=torch.float64), torch.zeros(n_bpm, dtype=torch.float64)

    true_gx = ones_b.clone()
    true_gx[0] = 1.03
    sigma = 1e-4 * J0.abs().mean().item() + 1e-9  # a small fraction of the typical ORM signal
    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, sigma, size=(n_bpm, 2, n_corr))
    measured = (apply_calibration(J0, true_gx, ones_b, zeros_b, ones_c) + torch.as_tensor(noise)).numpy().reshape(-1)

    def residual_fn(theta: torch.Tensor) -> torch.Tensor:
        return (apply_calibration(J0, theta, ones_b, zeros_b, ones_c).reshape(-1) - torch.as_tensor(measured)) / sigma

    x = _fit_with_exact_jacobian(residual_fn, np.ones(n_bpm))

    # linear propagated uncertainty at the fitted point: sigma_x0^2 = ((d(model)/d(gx0))^T (d(model)/d(gx0)))^-1
    def observe(th: torch.Tensor) -> torch.Tensor:
        return apply_calibration(J0, th, ones_b, zeros_b, ones_c).reshape(-1) / sigma
    th = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    J_fit = torch.autograd.functional.jacobian(observe, th, vectorize=True)
    cov = torch.linalg.inv(J_fit.T @ J_fit)
    sigma_gx0 = cov[0, 0].sqrt().item()
    assert abs(x[0] - true_gx[0].item()) < 6 * sigma_gx0  # a generous multiple of the fit's own propagated uncertainty


# --- Phase 6: tensor-safe misalignment (dx/dy/tilt), coupling ------------

def test_misalign_dx_gradient_at_zero_matches_finite_difference(ring):
    """Same trap as the corrector kick: a plain ``dx != 0.0`` branch would
    silently detach the gradient exactly at dx=0, LOCO's own alignment-fit
    operating point."""
    qf_name = _names(ring.specs, "quad")[0]
    dx = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    state0 = torch.zeros(6, dtype=torch.float64)
    state0[0] = 1e-4  # a nonzero incoming x -- something for the quad's dx to act on

    def final_px(d):
        r = ring.with_strengths(dx={qf_name: d})
        return r.turn(state0, eager=True)[1]

    g = torch.autograd.grad(final_px(dx), dx)[0]
    h = 1e-6
    fd = (final_px(torch.tensor(h, dtype=torch.float64)) - final_px(torch.tensor(-h, dtype=torch.float64))) / (2 * h)
    assert g.item() != 0.0
    assert g.item() == pytest.approx(fd.item(), rel=1e-5)


def test_misalign_tilt_gradient_at_zero_matches_finite_difference(ring):
    qf_name = _names(ring.specs, "quad")[0]
    tilt = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    state0 = torch.zeros(6, dtype=torch.float64)
    state0[0] = 1e-4  # a horizontal offset -- a quad roll couples it into py

    def final_py(t):
        r = ring.with_strengths(tilt={qf_name: t})
        return r.turn(state0, eager=True)[3]

    g = torch.autograd.grad(final_py(tilt), tilt)[0]
    h = 1e-6
    fd = (final_py(torch.tensor(h, dtype=torch.float64)) - final_py(torch.tensor(-h, dtype=torch.float64))) / (2 * h)
    assert g.item() != 0.0
    assert g.item() == pytest.approx(fd.item(), rel=1e-5)


def test_ring_warp_pack_falls_back_for_gradient_misalignment(ring):
    qf_name = _names(ring.specs, "quad")[0]
    dx = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    r = ring.with_strengths(dx={qf_name: dx})
    assert ring_warp.pack(r.params, torch.device("cpu"), torch.float64) is None


def test_double_single_rejects_nonzero_misalignment_override(specs):
    ds_ring = RingMap(
        specs, E0, MC2, device="cpu", n_kicks=2, n_kicks_bend=8, integration_order=6, exact_dipole=True, double_single=True,
    )
    qf_name = _names(specs, "quad")[0]
    with pytest.raises(NotImplementedError):
        ds_ring.with_strengths(dx={qf_name: 1e-4})
    with pytest.raises(NotImplementedError):
        ds_ring.with_strengths(tilt={qf_name: 1e-4})
    ds_ring.with_strengths(dx={qf_name: 0.0}, tilt={qf_name: 0.0})  # zero override: a no-op, must not raise


def test_coupled_tilt_and_k1_synthetic_recovery_with_rank_audit(specs):
    """Inject one quad roll (a real skew source) and one *different*
    quad's K1 error, fit both jointly from a noise-free coupled ORM --
    Phase 6's 'coupled synthetic truth recovery with an explicit rank/
    gauge audit' gate."""
    truth_ring = RingMap(specs, E0, MC2, device="cpu", dtype=torch.float64, exact_dipole=False)
    correctors, bpms = _corrector_and_bpm_names(specs)
    qf_names = _names(specs, "quad")[:2]
    tilt_name, k1_name = qf_names[0], qf_names[1]
    n_h = len(_names(specs, "hkick"))

    # uncoupled baseline: the cross-plane block is exactly zero (same
    # property test_orm_uncoupled_ring_has_zero_cross_plane_blocks checks).
    J0 = orbit_response_matrix(truth_ring, correctors, bpms)
    assert torch.allclose(J0[:, 1, :n_h], torch.zeros_like(J0[:, 1, :n_h]), atol=1e-12)

    true_tilt = 0.01
    nominal_k1 = next(s.k1 for s in specs if s.name == k1_name)
    true_k1 = nominal_k1 * 1.02
    truth = truth_ring.with_strengths(tilt={tilt_name: true_tilt}, k1={k1_name: true_k1})
    J_truth = orbit_response_matrix(truth, correctors, bpms)
    cross_block = J_truth[:, 1, :n_h]  # y-readings vs H-correctors: zero only if uncoupled
    assert cross_block.abs().max().item() > 1e-6  # the roll genuinely couples the planes
    measured = J_truth.detach().numpy().reshape(-1)

    def model_J(theta: torch.Tensor) -> torch.Tensor:
        r = truth_ring.with_strengths(tilt={tilt_name: theta[0]}, k1={k1_name: theta[1]})
        return orbit_response_matrix(r, correctors, bpms, create_graph=True)

    def residual_fn(theta: torch.Tensor) -> torch.Tensor:
        return model_J(theta).reshape(-1) - torch.as_tensor(measured)

    theta_start = np.array([0.0, nominal_k1])
    x = _fit_with_exact_jacobian(residual_fn, theta_start)
    assert x[0] == pytest.approx(true_tilt, rel=1e-5)
    assert x[1] == pytest.approx(true_k1, rel=1e-5)

    # rank/gauge audit: the fit Jacobian's two columns (d(model)/d(tilt),
    # d(model)/d(K1)) must be well-conditioned -- not a near-degenerate
    # (unidentifiable) combination.
    def observe(th: torch.Tensor) -> torch.Tensor:
        return model_J(th).reshape(-1)
    th = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    cols = torch.autograd.functional.jacobian(observe, th, vectorize=True)
    singular_values = torch.linalg.svdvals(cols)
    assert singular_values.shape == (2,)
    assert (singular_values[-1] / singular_values[0]).item() > 1e-6  # both directions identifiable, no rank deficiency


# --- Phase 7: real-booster live-elegant ORM cross-check --------------------

@needs_elegant
@needs_br14_bst
def test_real_booster_orm_matches_live_elegant(tmp_path):
    """Native ORM on the real ``newBST`` beamline (891 elements, 48
    HCM/VCM/BPM occurrences each) against live-elegant central kick
    differences on a *selected* subset of channels (Verification gate 8 --
    a demonstration this generalizes past the small synthetic FODO, not
    the first correctness check for the corrector/BPM physics itself,
    which Phase 0's single-element live-elegant tests already pin).

    elegant's ``N_KICKS``/``INTEGRATION_ORDER`` must be explicitly forced
    to match ``RingMap``'s own slicing (this lattice's real elements carry
    no ``N_KICKS=``/``INTEGRATION_ORDER=`` in the ``.lte`` file itself, so
    both codes otherwise silently use their own, different, defaults) --
    the same ``ELEGANT_SLICING`` pattern ``tests/test_ring_nonlinear.py``
    uses. Tolerance (2% relative) is wider than a single-element CSBEND
    check (``test_native_tracker.py``'s own ~5e-3) because a real BPM
    downstream of many bends compounds that same documented, bounded
    higher-order-field-expansion gap (plans/PHYSICS.md) over dozens of
    elements, not one."""
    raw = parse_elegant_lattice(BR14_BST_LTE, "newBST")
    specs = unique_names(raw, kinds=("hkick", "vkick", "drift", "quad"))
    n_kicks, n_kicks_bend, integration_order = 2, 8, 6
    ring = RingMap(
        specs, E0_100MEV_eV, MC2, device="cpu", dtype=torch.float64,
        n_kicks=n_kicks, n_kicks_bend=n_kicks_bend, integration_order=integration_order, exact_dipole=False,
    )

    correctors = ["HCM#1", "VCM#1", "HCM#24", "VCM#30"]
    bpm_occurrences = [1, 10, 25, 48]
    bpms = [f"BPM#{i}" for i in bpm_occurrences]
    J_native = orbit_response_matrix(ring, correctors, bpms).detach().numpy()

    p_central_mev = E0_100MEV_eV / 1e6
    lte_text = BR14_BST_LTE.read_text()
    slicing = (
        f"&alter_elements name = *, type = KQUAD, item = N_KICKS, value = {n_kicks} &end\n"
        f"&alter_elements name = *, type = KSEXT, item = N_KICKS, value = {n_kicks} &end\n"
        f"&alter_elements name = *, type = CSBEND, item = N_KICKS, value = {n_kicks_bend} &end\n"
        f"&alter_elements name = *, type = KQUAD, item = INTEGRATION_ORDER, value = {integration_order} &end\n"
        f"&alter_elements name = *, type = KSEXT, item = INTEGRATION_ORDER, value = {integration_order} &end\n"
        f"&alter_elements name = *, type = CSBEND, item = INTEGRATION_ORDER, value = {integration_order} &end\n"
    )

    def run_elegant(kick_overrides: dict[tuple[str, int], float], tag: str) -> dict[str, tuple[float, float]]:
        # elegant's own closed-orbit finder (&closed_orbit), not a single
        # on-axis pass through &track: a corrector positioned *after* a
        # given BPM in ring-index order only shows a nonzero response via
        # the periodic (wraparound) closed orbit -- a single pass from a
        # fixed on-axis start misses it entirely (matches the bug fixed in
        # analysis/nonlinear.py::_closed_orbit_and_matrix, see its own
        # docstring). &closed_orbit writes the orbit at every element
        # (ElementName/ElementOccurence/x/y columns) to one SDDS file.
        alter_lines = "\n".join(
            f"&alter_elements name = {base}, item = KICK, value = {val!r}, start_occurence = {occ}, end_occurence = {occ} &end"
            for (base, occ), val in kick_overrides.items()
        )
        (tmp_path / "run.lte").write_text(lte_text)
        (tmp_path / f"{tag}.ele").write_text(f"""\
&run_setup
    lattice = "run.lte", use_beamline = "newBST", rootname = "{tag}",
    p_central_mev = {p_central_mev!r}, default_order = 3
&end
{slicing}{alter_lines}
&closed_orbit output = "%s.clo" &end
&run_control &end
&bunched_beam &end
&track &end
""")
        proc = subprocess.run([str(ELEGANT_EXE), f"{tag}.ele"], cwd=tmp_path, capture_output=True, text=True, env=elegant_env())
        clo_path = tmp_path / f"{tag}.clo"
        assert clo_path.exists(), f"elegant run failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        cols, _ = read_sdds_table(clo_path)
        names = np.char.strip(cols["ElementName"].astype(str))
        occs = cols["ElementOccurence"]
        out = {}
        for occ, bpm in zip(bpm_occurrences, bpms):
            i = np.where((names == "BPM") & (occs == occ))[0][0]
            out[bpm] = (float(cols["x"][i]), float(cols["y"][i]))
        return out

    h = 1e-5  # rad -- a realistic small corrector kick
    fd = np.zeros_like(J_native)
    for j, name in enumerate(correctors):
        base_name, occ = name.split("#")
        occ = int(occ)
        plus = run_elegant({(base_name, occ): h}, f"p{j}")
        minus = run_elegant({(base_name, occ): -h}, f"m{j}")
        for i, bpm in enumerate(bpms):
            fd[i, 0, j] = (plus[bpm][0] - minus[bpm][0]) / (2 * h)
            fd[i, 1, j] = (plus[bpm][1] - minus[bpm][1]) / (2 * h)

    scale = np.abs(fd).max()
    assert np.abs(J_native - fd).max() / scale < 0.02
