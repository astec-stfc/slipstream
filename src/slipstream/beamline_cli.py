"""Run a native beamline study: ``python -m slipstream.beamline_cli study.yaml``."""
from __future__ import annotations

import argparse
import json

import yaml

from .beamline import BeamlineStudy


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run a native beamline from a YAML configuration")
    parser.add_argument("config", help="YAML study file (its relative paths are local to that file)")
    parser.add_argument("-o", "--output", help="Override result HDF5 path (relative to the current directory)")
    args = parser.parse_args(argv)
    try:
        result = BeamlineStudy.from_yaml(args.config).run(output_path=args.output)
    except (ValueError, OSError, ImportError, yaml.YAMLError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")
    summary = result.scalars()
    if "reference_diff" in result.metadata:
        summary["reference_diff"] = result.metadata["reference_diff"]
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
