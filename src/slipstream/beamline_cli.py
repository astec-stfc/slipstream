"""Uses `BeamlineStudy` source conversion and single-pass transport; CLI orchestration adds no separate
physics. See docs/modules/beamline_cli.md."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import sys

import yaml

from .beamline import BeamlineStudy


def _override(text):
    key, separator, value = text.partition("=")
    if not separator or not key.strip() or not value.strip():
        raise argparse.ArgumentTypeError("expected KEY=VALUE, such as tracker.csr=true (use null for None)")
    try:
        return key.strip(), yaml.safe_load(value)
    except yaml.YAMLError as exc:
        raise argparse.ArgumentTypeError(f"Invalid YAML value for {key}: {exc}") from exc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run a native beamline from a YAML configuration")
    parser.add_argument("config", help="YAML study file (its relative paths are local to that file)")
    parser.add_argument("-o", "--output", help="Override result HDF5 path (relative to the current directory)")
    parser.add_argument("--set", dest="overrides", action="append", type=_override, default=[], metavar="KEY=VALUE",
                        help="Override a YAML setting; repeat for multiple settings (e.g. --set tracker.device=cpu)")
    parser.add_argument("--check", action="store_true",
                        help="Validate settings and build the lattice/tracker without loading particles or writing output")
    args = parser.parse_args(argv)
    try:
        with redirect_stdout(sys.stderr):
            study = BeamlineStudy.from_yaml(args.config, overrides=dict(args.overrides))
        if args.check:
            print("Configuration and lattice loaded successfully; particles were not loaded or tracked.")
            return 0
        with redirect_stdout(sys.stderr):
            result = study.run(output_path=args.output)
    except (ValueError, OSError, ImportError, yaml.YAMLError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")
    summary = result.scalars()
    if "reference_diff" in result.metadata:
        summary["reference_diff"] = result.metadata["reference_diff"]
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
