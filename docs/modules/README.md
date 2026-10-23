# Package module reference

Each page has **Physics** and **Usage**, with limits and numerical notes where needed.
API stubs describe definitions; runnable task examples are in the [main guides](../README.md).

## Package and shared utilities

- [beamline](beamline.md): Composes single-pass canonical linac tracking with bunch generation, slice diagnostics and optional injector transport.
- [beamline_cli](beamline_cli.md): Uses `BeamlineStudy` source conversion and single-pass transport; CLI orchestration adds no separate physics.
- [beamline_config](beamline_config.md): Normalizes total energy and rest mass in eV, bunch charge in coulombs and canonical native coordinates.
- [constants](constants.md): Shared constants include the speed of light, elementary charge, electron rest energy, classical electron radius, vacuum permittivity and permeability.
- [ibs](ibs.md): Computes amplitude-normalized intrabeam-scattering growth rates in inverse seconds.
- [sdds_io](sdds_io.md): SDDS values retain the units and meanings of their reader; this module provides shared access to the binding, rather than physical conversion.
- [serialization](serialization.md): Serialization preserves scalar machine parameters; it introduces no transport or coordinate conversion.
- [_compile_env](_compile_env.md): Cache configuration changes compilation storage, without changing transport or physical models.
- [package exports](__init__.md): Public exports collect native transport, machine models, wakes and beam diagnostics.

## analysis

- [emittance](analysis/emittance.md): RMS helpers combine unbiased position/momentum variances with the mean cross product.
- [loco](analysis/loco.md): Differentiates closed-orbit BPM readings with respect to corrector kick angles.
- [nonlinear](analysis/nonlinear.md): Computes fixed-energy ring closed orbits, Jacobian optics, chromaticity, NAFF frequencies, amplitude detuning, dynamic aperture and frequency-map diffusion.
- [nonlinear_opt](analysis/nonlinear_opt.md): Tensor magnet strengths support differentiable chromaticity and tune-footprint objectives.
- [ring6d](analysis/ring6d.md): RF-on survival and tune diagnostics advance the real `NativeTracker`, including configured longitudinal motion, damping and excitation.
- [ring_outputs](analysis/ring_outputs.md): Reports linear optics along the actual closed orbit and derives machine parameters from configured transport.
- [slices](analysis/slices.md): Bins length-like tau in metres and computes charge/current, widths and centered transverse emittances per slice.
- [package exports](analysis/__init__.md): Analysis computes statistics from transported particle data.

## data

- [particles](data/particles.md): The HDF5 particle schema distinguishes canonical `(x,px,y,py,tau,delta)` from slope `(x,xp,y,yp,tau,delta)` coordinates.
- [results](data/results.md): `TrackingResult` stores sampled ring diagnostics with `sigma_tau_s` in seconds; `BeamlineResult` stores canonical output particles with tau in metres.
- [package exports](data/__init__.md): Provides persisted tracking diagnostics; schemas retain their own coordinate and time units.

## lattice

- [apertures](lattice/apertures.md): Builds longitudinal resistive-wall impedance from shared named materials and chamber sections.
- [lattice](lattice/lattice.md): Stores longitudinal ring parameters and derives relativistic speed, revolution period, harmonic number, synchronous phase, synchrotron frequency and natural bunch length.
- [optics](lattice/optics.md): Stores single-point transverse Twiss/dispersion, tunes/chromaticities, equilibrium geometric emittances and damping times.
- [ramp](lattice/ramp.md): Interpolates supplied RF/energy programs and scales fixed-geometry radiation parameters: energy loss proportional to gamma^4, damping time to gamma^-3, natural spread to gamma and equilibrium emittance to gamma^2.
- [twiss_table](lattice/twiss_table.md): Stores element-local Twiss/dispersion with explicit segment lengths for length-weighted IBS averaging.
- [package exports](lattice/__init__.md): Exports longitudinal ring machine parameters; transport and transverse optics live in their own modules.

## tracking

- [distributions](tracking/distributions.md): Generates Twiss-matched Gaussian, KV surface and waterbag volume distributions, plus chirped longitudinal profiles.
- [package exports](tracking/__init__.md): Collects particle distributions and native tracker interfaces; physical transport resides in the native modules.

## wakefields

- [chambers](wakefields/chambers.md): Stores round/flat chamber geometry, section length and material layers.
- [impedance](wakefields/impedance.md): Stores frequency-domain longitudinal impedance as real and imaginary ohms on a shared Hz grid.
- [longrange](wakefields/longrange.md): High-Q resonators accumulate decaying/rotating interbunch voltage phasors with `lambda=(i-1/(2Q))*omega_r`.
- [shortrange](wakefields/shortrange.md): Constructs a Gaussian-probe wake from impedance via inverse Fourier transformation, enforcing `Z(-omega)=conj(Z(omega))`.
- [transverse](wakefields/transverse.md): Dipole wake voltage is a convolution against charge-weighted transverse offset rather than charge alone.
- [package exports](wakefields/__init__.md): Exports longitudinal impedance and short/long-range wake models.

## tracking/native

- [injector_ibs](tracking/native/injector_ibs.md): Local equal-weight binary Coulomb collisions for injector studies, with Takizuka–Abe/Nanbu angular laws and elastic pair kinematics.
- [injector_ibs_warp](tracking/native/injector_ibs_warp.md): Optional float64 CUDA acceleration of disjoint collision pairs, sharing the PyTorch grouping and random draws.

- [astra_io](tracking/native/astra_io.md): Converts ASTRA distribution rows from their reference-relative z/pz/time convention to absolute particle values.
- [collective](tracking/native/collective.md): Implements local long-wavelength 1D LSC and steady-state 1D CSR momentum kicks.
- [double_single](tracking/native/double_single.md): Represents selected frozen-energy ring coordinates as float32 high/low pairs for increased precision.
- [elements](tracking/native/elements.md): Integrates canonical (x,px,y,py,tau,delta) transport using symplectic splitting, sector-bend drifts, normal/skew multipole and corrector/edge kicks, and RF cavities with supported focusing and optional structure wakes.
- [errors](tracking/native/errors.md): Samples one static error per matched element instance.
- [fields](tracking/native/fields.md): Reads on-axis profiles and expands RF/solenoid fields near the axis.
- [generator](tracking/native/generator.md): Samples ASTRA-style initial beams with configurable distributions, correlations, offsets, species and emission times.
- [gun](tracking/native/gun.md): Uses the relativistic Boris pusher in Cartesian `(x,y,z,ux,uy,uz)`, with proper velocity `u=p/m=gamma*v` in m/s.
- [injector](tracking/native/injector.md): Tracks cathode emission and Cartesian proper velocities through standing/traveling-wave cavities, solenoids, magnets, apertures and cylindrical or 3D space charge with optional cathode images.
- [lattice_import](tracking/native/lattice_import.md): Converts supported elegant lattice declarations into native `ElementSpec` physics, including magnetic elements, RFCA/RFCW, H/V correctors and CSR-related settings.
- [laura_fields](tracking/native/laura_fields.md): Loads LAURA on-axis field profiles and wake tables into existing native field/wake classes.
- [laura_import](tracking/native/laura_import.md): Maps LAURA section records into the native element specification used for linac transport.
- [laura_injector](tracking/native/laura_injector.md): Constructs field-map injector elements from chained LAURA sections.
- [linac](tracking/native/linac.md): Tracks one canonical single pass through ordered elements, with inline RF and optional structure wakes, local LSC and CSR.
- [linac_setup](tracking/native/linac_setup.md): Crest calibration maximizes reference-particle energy gain through the upstream line and selected cavity using a phase scan and parabolic refinement.
- [linac_twiss](tracking/native/linac_twiss.md): Propagates nonperiodic linear Twiss/dispersion or extracts ensemble statistics from tracked trajectories.
- [magnets](tracking/native/magnets.md): Evaluates lab-frame static dipole, solenoid and quadrupole fields, including element offsets/rotations.
- [optics](tracking/native/optics.md): Linearizes actual native per-element transport using autograd, solves periodic per-plane Twiss and dispersion and propagates element-local values.
- [photocathode](tracking/native/photocathode.md): Opt-in laser-driven emission with Schottky-dependent quantum efficiency, limited by the net cathode surface field and by fall-back onto the cathode.
- [ring](tracking/native/ring.md): Provides frozen-energy one-turn nonlinear transport with fixed delta and no RF, radiation damping/excitation or wakes.
- [ring_physics](tracking/native/ring_physics.md): Native ring kicks apply sinusoidal RF, radiation damping/excitation and effective IBS damping times.
- [ring_warp](tracking/native/ring_warp.md): Implements selected fixed-lattice one-turn ring transport using a packed Warp kernel.
- [space_charge](tracking/native/space_charge.md): Cylindrical PIC deposits/gathers with CIC and combines longitudinal open-boundary FFT treatment with radial finite differences.
- [tracker](tracking/native/tracker.md): Integrates actual per-element canonical ring transport on every turn and composes native RF, radiation damping/excitation, wakes, optional IBS and ramps.
- [warp_kernels](tracking/native/warp_kernels.md): Optional CUDA kernels accelerate Boris pushes, selected field evaluations, apertures, charge deposition/gathering, line-density histograms and CSR pairwise sums.
- [package exports](tracking/native/__init__.md): Exports ring, linac and injector transport and elegant-lattice import.

## wakefields/multilayer

- [bessel](wakefields/multilayer/bessel.md): Complex modified Bessel functions use SciPy forward evaluation and exact recurrence derivatives for PyTorch complex autograd.
- [flat](wakefields/multilayer/flat.md): Contains the symmetric flat multilayer field-matching formulation, but raw matrix inversion can overflow float64 for realistic chamber/frequency parameters.
- [round](wakefields/multilayer/round.md): Matches nonmagnetic multilayer cylindrical fields using Drude-relaxation resistivity and background permittivity.
- [yokoya](wakefields/multilayer/yokoya.md): Interpolates geometric elliptic form factors for round-result approximations.
- [package exports](wakefields/multilayer/__init__.md): Provides round multilayer matching, complex Bessel primitives and geometric Yokoya corrections.

