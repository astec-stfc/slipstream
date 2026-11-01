# Choose an example

Run these commands from the checkout root after `python -m pip install -e .`.
The introductory examples run on CPU without a compiler, SDDS or external
tracking executables. Examples are source files, not installed wheel data.

| Task | Command | Inputs and output |
| --- | --- | --- |
| First Python run | `python docs/examples/quickstart.py` | Generates its own bunch; prints shape, energy and beam size; writes no files |
| First YAML study | `python -m slipstream.beamline_cli examples/beamline/study.yaml -o result.h5` | Synthetic drift, quadrupole and RF line; JSON diagnostics and HDF5 result |
| Compare CSR models | `python -m slipstream.beamline_cli docs/examples/csr.yaml --set tracker.csr_transient=true --set tracker.csr_drift_model=retarded` | Synthetic CSR bend and drift; JSON diagnostics; writes no files unless `-o` is given |
| Ring optics and optimization | `python examples/nonlinear_optimisation.py --help` | [Illustrative MLS ring](../../examples/MLS/README.md); optional `[plot]` and `[optimise]` extras for figures and optimization |
| Photocathode scan | `python examples/photocathode/schottky_scan.py --help` | [Injector example](../../examples/photocathode/README.md); read its input-file and runtime requirements first |

Use `--check` on a YAML command to build its lattice/tracker without tracking.
Use `--set bunch.n=128` for a smaller generated bunch. Input paths follow the
YAML file, so an absolute configuration path also works from another directory.

See [getting started](../getting_started.md) for installation and reading
results, and [model settings](../model_settings.md) for switches and limits.
