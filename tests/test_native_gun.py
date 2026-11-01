"""Quantitative regressions for relativistic Boris pushing and space-charge boost coupling.

Expected values come from the closed-form calculations documented in these tests.
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess

import pytest
import torch

from slipstream.constants import C_LIGHT, E_CHARGE_C, ELECTRON_MASS_EV, EPS0_F_PER_M
from slipstream.tracking.native.gun import (
    boris_push,
    relativistic_space_charge_field,
    track_particles,
)
from slipstream.tracking.native.space_charge import CylindricalSpaceCharge

_MASS_KG = ELECTRON_MASS_EV * E_CHARGE_C / C_LIGHT**2
_Q_C = -E_CHARGE_C


def test_boris_relativistic_cyclotron_closed_orbit_and_energy_conservation():
    # Pure B field: the Boris algorithm conserves gamma (energy) *exactly*
    # (Qin et al. 2013, "Why is Boris algorithm so good?", stated
    # directly in the paper's own text) and the true orbit is a closed
    # circle -- after exactly one relativistic gyro-period the particle
    # must return to its starting (x, y, ux, uy) [z, uz stay 0 throughout
    # since B is purely along z and there is no E field].
    gamma0 = 2.0
    beta0 = math.sqrt(1.0 - 1.0 / gamma0**2)
    u0 = gamma0 * beta0 * C_LIGHT
    B0 = 0.1

    period = 2 * math.pi * gamma0 * _MASS_KG / (abs(_Q_C) * B0)
    n_steps = 2000
    dt = period / n_steps

    x0 = torch.tensor([0.0], dtype=torch.float64)
    y0 = torch.tensor([0.0], dtype=torch.float64)
    z0 = torch.tensor([0.0], dtype=torch.float64)
    ux0 = torch.tensor([u0], dtype=torch.float64)
    uy0 = torch.tensor([0.0], dtype=torch.float64)
    uz0 = torch.tensor([0.0], dtype=torch.float64)

    def field_fn(x, y, z, t):
        zero = torch.zeros_like(x)
        return zero, zero, zero, zero, zero, torch.full_like(x, B0)

    x, y, z, ux, uy, uz = track_particles(
        x0, y0, z0, ux0, uy0, uz0, _Q_C, _MASS_KG, field_fn, t0=0.0, dt=dt, n_steps=n_steps,
        use_compile=False,  # closed-form pusher check; the compiled step has its own parity test below
    )

    gamma_final = torch.sqrt(1.0 + (ux**2 + uy**2 + uz**2) / C_LIGHT**2)
    assert torch.allclose(gamma_final, torch.tensor([gamma0], dtype=torch.float64), rtol=1e-10)

    r_L = _MASS_KG * u0 / (abs(_Q_C) * B0)
    assert torch.allclose(x, x0, atol=1e-3 * r_L)
    assert torch.allclose(y, y0, atol=1e-3 * r_L)
    assert torch.allclose(z, z0, atol=1e-12)
    assert torch.allclose(ux, ux0, rtol=1e-3)
    assert torch.allclose(uy, uy0, atol=1e-3 * u0)


def test_boris_relativistic_uniform_field_acceleration_matches_closed_form():
    # Pure E field, from rest: dp/dt = qE is exact regardless of
    # relativity (only v(p) is nonlinear), so p(t) = q*E*t exactly.
    # dz/dt = p*c^2/sqrt((mc^2)^2+(pc)^2); substituting p=qEt and
    # integrating (u = (mc^2)^2+(pc)^2, p*dp = du/(2c^2)) gives
    #   z(t) = (1/(qE)) * [sqrt((mc^2)^2 + (p(t)*c)^2) - mc^2],
    # which reduces to the familiar z=(1/2)(qE/m)t^2 for small p -- a
    # check against the non-relativistic limit that caught a stray
    # factor of c in an earlier draft of this formula (the *code*
    # matched the closed form once the formula itself was fixed, so this
    # was a test bug, not a `boris_push`/`track_particles` bug).
    Ez0 = -1.0e8  # V/m
    force = _Q_C * Ez0
    gamma_target = 3.0
    p_target = _MASS_KG * C_LIGHT * math.sqrt(gamma_target**2 - 1.0)
    t_total = p_target / force

    n_steps = 2000
    dt = t_total / n_steps

    zero = torch.zeros(1, dtype=torch.float64)

    def field_fn(x, y, z, t):
        return torch.zeros_like(x), torch.zeros_like(x), torch.full_like(x, Ez0), torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x)

    x, y, z, ux, uy, uz = track_particles(
        zero.clone(), zero.clone(), zero.clone(), zero.clone(), zero.clone(), zero.clone(),
        _Q_C, _MASS_KG, field_fn, t0=0.0, dt=dt, n_steps=n_steps,
        use_compile=False,  # closed-form pusher check; the compiled step has its own parity test below
    )

    gamma_final = torch.sqrt(1.0 + (uz**2) / C_LIGHT**2)
    assert torch.allclose(gamma_final, torch.tensor([gamma_target], dtype=torch.float64), rtol=1e-9)
    assert torch.allclose(ux, zero, atol=1e-20)
    assert torch.allclose(uy, zero, atol=1e-20)

    mc2 = _MASS_KG * C_LIGHT**2
    z_exact = (1.0 / force) * (math.sqrt(mc2**2 + (p_target * C_LIGHT) ** 2) - mc2)
    assert torch.allclose(z, torch.tensor([z_exact], dtype=torch.float64), rtol=2e-3)


def test_relativistic_space_charge_field_matches_electrostatic_at_gamma_one():
    from slipstream.tracking.native.space_charge import field_with_cathode_image

    solver = CylindricalSpaceCharge(
        nr=48, nz=96, r_max=0.03, z_min=-0.02, z_max=0.02, device="cpu", dtype=torch.float64
    )
    x = torch.tensor([0.001, -0.0015], dtype=torch.float64)
    y = torch.tensor([0.0005, 0.0], dtype=torch.float64)
    z = torch.tensor([0.005, 0.006], dtype=torch.float64)
    q = torch.tensor([-1e-12, -1.2e-12], dtype=torch.float64)
    cathode_z = 0.0

    Ex, Ey, Ez, Bx, By, Bz = relativistic_space_charge_field(solver, x, y, z, q, bunch_gamma=1.0, cathode_z=cathode_z)
    Ex_ref, Ey_ref, Ez_ref = field_with_cathode_image(solver, x, y, z, q, cathode_z)

    assert torch.allclose(Ex, Ex_ref)
    assert torch.allclose(Ey, Ey_ref)
    assert torch.allclose(Ez, Ez_ref)
    assert torch.allclose(Bx, torch.zeros_like(Bx))
    assert torch.allclose(By, torch.zeros_like(By))


def test_relativistic_space_charge_field_matches_boosted_image_charge_closed_form():
    h = 0.01
    q_val = -1.0e-12
    bunch_gamma = 2.0
    beta = math.sqrt(1.0 - 1.0 / bunch_gamma**2)
    cathode_z = 0.0
    solver = CylindricalSpaceCharge(
        nr=80, nz=200, r_max=0.05, z_min=-0.05, z_max=0.05, device="cpu", dtype=torch.float64
    )

    # particle 0 is the real source; particle 1 is a probe with negligible
    # charge (so it doesn't perturb particle 0's own field) placed away
    # from both the source and its image.
    x = torch.tensor([0.0, 0.008], dtype=torch.float64)
    y = torch.tensor([0.0, 0.0], dtype=torch.float64)
    z = torch.tensor([h, h + 0.004], dtype=torch.float64)
    q = torch.tensor([q_val, q_val * 1e-6], dtype=torch.float64)

    Ex, Ey, Ez, Bx, By, Bz = relativistic_space_charge_field(solver, x, y, z, q, bunch_gamma, cathode_z=cathode_z)

    z_src_s = cathode_z + bunch_gamma * (h - cathode_z)
    z_probe_s = cathode_z + bunch_gamma * (z[1].item() - cathode_z)
    r_probe = x[1].item()

    def coulomb(qc, z0, r, zp):
        d = math.sqrt(r**2 + (zp - z0) ** 2)
        pref = qc / (4 * math.pi * EPS0_F_PER_M * d**3)
        return pref * r, pref * (zp - z0)

    Er1, Ez1 = coulomb(q_val, z_src_s, r_probe, z_probe_s)
    Er2, Ez2 = coulomb(-q_val, -z_src_s, r_probe, z_probe_s)
    Er_prime, Ez_exact = Er1 + Er2, Ez1 + Ez2
    Ex_exact = bunch_gamma * Er_prime
    By_exact = (beta / C_LIGHT) * Ex_exact

    assert torch.allclose(Ex[1], torch.tensor(Ex_exact, dtype=torch.float64), rtol=0.15)
    assert torch.allclose(Ez[1], torch.tensor(Ez_exact, dtype=torch.float64), rtol=0.15)
    assert torch.allclose(By[1], torch.tensor(By_exact, dtype=torch.float64), rtol=0.15)
    assert torch.allclose(Bx[1], torch.tensor(0.0, dtype=torch.float64), atol=1e-20)
    # Force-reduction identity this whole construction is built to give:
    # net transverse force = q*(Ex - beta*c*By) = q*Ex*(1-beta^2) = q*Ex/gamma^2.
    net_force_x = q[1] * (Ex[1] - beta * C_LIGHT * By[1])
    assert torch.allclose(net_force_x, q[1] * Ex[1] * (1 - beta**2), rtol=1e-9)


def test_compensated_boris_recovers_a_sub_ulp_kick_that_plain_float32_deletes():
    # The failure this guards against is silent and total, and it cost
    # several investigation passes to find (OPTIMISATION.md F13): in the
    # CLARA gun-to-structure drift the only longitudinal field is space
    # charge, and its per-step momentum kick is *below float32's rounding
    # threshold*, so `u + du` returns `u` unchanged every step and the
    # force vanishes with no error and no warning.
    #
    # Set up exactly that regime. At 4.09 MeV (the measured gun-exit
    # energy) u = gamma*beta*c = 2.683e9 m/s, which lies in the binade
    # [2^31, 2^32), so float32's spacing there is 2^8 = 256 m/s and
    # anything under 128 m/s rounds away. The drift's measured on-axis
    # space-charge field of ~3980 V/m gives du = (q/m)*Ez*dt = 70 m/s --
    # comfortably under it, and by construction below, not by assumption.
    gamma0 = 1.0 + 4.09e6 / ELECTRON_MASS_EV
    u0 = math.sqrt(gamma0**2 - 1.0) * C_LIGHT
    ez, dt, n_steps = 3980.0, 1.0e-13, 10000
    du = abs(_Q_C / _MASS_KG * ez * dt)
    half_ulp = math.ldexp(1.0, math.frexp(u0)[1] - 1 - 24)
    assert du < half_ulp, f"test misconfigured: du={du} is already resolvable"

    def drift(dtype, compensated):
        """n_steps of Boris under a pure, constant Ez -- a field-free
        drift plus the space-charge kick, with no B and nothing else to
        carry the increment."""
        zeros = torch.zeros(1, dtype=dtype)
        x, y, z = zeros.clone(), zeros.clone(), zeros.clone()
        ux, uy = zeros.clone(), zeros.clone()
        uz = torch.full((1,), u0, dtype=dtype)
        uz_start = float(uz)
        ez_t = torch.full((1,), ez, dtype=dtype)
        comp = tuple(zeros.clone() for _ in range(6)) if compensated else None
        for _ in range(n_steps):
            out = boris_push(
                x, y, z, ux, uy, uz, _Q_C, _MASS_KG,
                zeros, zeros, ez_t, zeros, zeros, zeros, dt, comp=comp,
            )
            if compensated:
                x, y, z, ux, uy, uz, comp = out
            else:
                x, y, z, ux, uy, uz = out
        return float(uz) - uz_start

    # Closed form: no B, constant E, so every step adds exactly (q/m)*E*dt
    # and the total is n*du (negative, since the electron charge is).
    expected = -n_steps * du
    ref64 = drift(torch.float64, compensated=False)
    plain32 = drift(torch.float32, compensated=False)
    comp32 = drift(torch.float32, compensated=True)

    assert abs(ref64 - expected) < 1e-6 * abs(expected)
    # The bug, asserted as the exact no-op it is -- not "small".
    assert plain32 == 0.0
    # ...and the fix, to float64's own answer.
    assert abs(comp32 - ref64) <= half_ulp  # i.e. as close as float32 can store u at all


def _ensure_clang_cl_inductor_env():
    """Use ``clang-cl`` for CPU ``torch.compile`` on Windows.

    ``clang-cl`` uses the MSVC-compatible headers and libraries from the
    Visual Studio/Windows SDK installation, so populate ``INCLUDE``/``LIB``
    when a plain shell has not loaded them.  The compiler choice itself is
    explicit: clang-cl is the supported Windows compiler for this codebase's
    compiled CUDA paths and must not silently fall back to ``cl``.
    """
    if shutil.which("clang-cl") is None:
        pytest.skip("CPU torch.compile needs clang-cl (see docs/compilation.md); not found")
    include = os.environ.get("INCLUDE", "")
    if any(os.path.isfile(os.path.join(d, "omp.h")) for d in include.split(";") if d):
        os.environ["CXX"] = "clang-cl"
        return
    vswhere = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    msvc_root, kits = None, r"C:\Program Files (x86)\Windows Kits\10"
    if os.path.isfile(vswhere):
        inst = subprocess.run(
            [vswhere, "-latest", "-products", "*",
             "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
             "-property", "installationPath"],
            capture_output=True, text=True).stdout.strip()
        cand = os.path.join(inst, "VC", "Tools", "MSVC")
        msvc_root = cand if inst and os.path.isdir(cand) else None
    if msvc_root is None or not os.path.isdir(os.path.join(kits, "Include")):
        pytest.skip("CPU torch.compile needs clang-cl + Windows SDK (see docs/compilation.md); not found")
    ver = max(os.listdir(msvc_root))
    sdk = max(os.listdir(os.path.join(kits, "Include")))
    incs = [os.path.join(msvc_root, ver, "include")] + [
        os.path.join(kits, "Include", sdk, d) for d in ("ucrt", "um", "shared")]
    libs = [os.path.join(msvc_root, ver, "lib", "x64")] + [
        os.path.join(kits, "Lib", sdk, d, "x64") for d in ("ucrt", "um")]
    os.environ["INCLUDE"] = ";".join(incs + ([include] if include else []))
    lib = os.environ.get("LIB", "")
    os.environ["LIB"] = ";".join(libs + ([lib] if lib else []))
    os.environ["CXX"] = "clang-cl"


def test_track_particles_compiled_matches_eager():
    # Same physics through both paths: the compiled route drives the
    # injector's compiled _fused_step (the same function object, imported
    # into gun.track_particles -- OPTIMISATION.md section D), whose push is
    # algebraically identical to boris_push but runs as one inductor kernel
    # that may reassociate float ops, so bit-identity is NOT required (and
    # the state is Kahan-compensated where the eager loop here is plain).
    # In float64 over tens of steps both are near-exact, so the observed
    # disagreement is reordering noise; rtol=1e-9 sits ~5 orders of
    # magnitude above it and far below anything physics-relevant -- loose
    # enough to never flake on kernel-codegen changes, tight enough that a
    # real divergence of the two paths (a duplicated-then-drifted step, the
    # thing this sharing exists to prevent) would trip it immediately.
    _ensure_clang_cl_inductor_env()
    torch.manual_seed(0)
    n, n_steps, dt = 300, 50, 1.0e-13
    dtype = torch.float64
    x = torch.randn(n, dtype=dtype) * 1e-4
    y = torch.randn(n, dtype=dtype) * 1e-4
    z = torch.rand(n, dtype=dtype) * 1e-4  # thin disk just off the cathode
    ux = torch.randn(n, dtype=dtype) * 1e4
    uy = torch.randn(n, dtype=dtype) * 1e4
    uz = torch.full((n,), 2.0e6, dtype=dtype) + torch.randn(n, dtype=dtype) * 1e4
    q = torch.full((n,), -5.0e-12 / n, dtype=dtype)
    solver = CylindricalSpaceCharge(
        nr=48, nz=128, r_max=0.01, z_min=-0.01, z_max=0.02, device="cpu", dtype=dtype
    )
    bunch_gamma = 1.001  # representative; parity, not absolute SC accuracy, is under test
    e0, omega, b0 = 3.0e7, 2.0 * math.pi * 3.0e9, 0.05

    def field_fn(x, y, z, t):
        # The standalone-gun pattern of gun.py's module docstring: applied
        # RF-like + solenoid fields summed with the space-charge self-field
        # (cathode image included). All of this stays eager -- only the
        # Boris push is compiled -- which is exactly the injector's split
        # with its space-charge cache.
        Ex = torch.zeros_like(x)
        Ey = torch.zeros_like(x)
        Ez = e0 * torch.cos(omega * t)
        Bx = torch.zeros_like(x)
        By = torch.zeros_like(x)
        Bz = torch.full_like(x, b0)
        sc = relativistic_space_charge_field(solver, x, y, z, q, bunch_gamma, cathode_z=0.0)
        return Ex + sc[0], Ey + sc[1], Ez + sc[2], Bx + sc[3], By + sc[4], Bz + sc[5]

    args = (x, y, z, ux, uy, uz, _Q_C, _MASS_KG, field_fn, 0.0, dt, n_steps)
    eager = track_particles(*args, use_compile=False)
    compiled = track_particles(*args, use_compile=True)

    max_rel = 0.0
    for name, a, b in zip(("x", "y", "z", "ux", "uy", "uz"), compiled, eager):
        rel = float((a - b).abs().max() / b.abs().max())
        max_rel = max(max_rel, rel)
        assert rel < 1e-9, f"compiled vs eager diverged in {name}: max rel diff {rel:.3g}"
    print(f"compiled vs eager max rel diff over 6 components: {max_rel:.3g}")

