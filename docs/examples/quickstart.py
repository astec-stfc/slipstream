"""Run a generated bunch through a drift on CPU, without input files or compilation."""
from slipstream import BeamlineStudy, ElementSpec


def main():
    study = BeamlineStudy(
        [ElementSpec("D", "drift", length=1.0)], energy_eV=100e6, device="cpu", always_change_p0=False,
    )
    particles = study.generate_bunch(
        n=1024, beta_x0=2.0, alpha_x0=0.0, epsx_m=1e-8,
        beta_y0=2.0, alpha_y0=0.0, epsy_m=1e-8,
        sigma_tau_m=1e-4, sigma_delta=1e-3, seed=42,
    )
    result = study.track(particles, return_trajectory=True)
    print(f"Output particles: {result.particles_out.shape}")
    print(f"Exit total energy: {result.energy_eV_out / 1e6:.3f} MeV")
    print(f"Horizontal rms size: {result.scalars()['sigma_x_m'] * 1e3:.3f} mm")
    return result


if __name__ == "__main__":
    main()
