from pathlib import Path

from slipstream import NativeTracker, parse_elegant_lattice
import slipstream
from slipstream.tracking.native.astra_io import read_astra_namelist


def test_native_public_imports_are_available():
    assert NativeTracker is not None
    for name in ("RingMap", "RingTrackResult", "RingLossMap", "CoupledOptics",
                 "coupled_optics", "coupled_signals"):
        assert name in slipstream.__all__
        assert getattr(slipstream, name) is not None


def test_elegant_lattice_reader_is_local(tmp_path: Path):
    path = tmp_path / "tiny.lte"
    path.write_text("DR: DRIFT, L=1.0;\nRING: LINE=(DR);\n", encoding="utf-8")
    specs = parse_elegant_lattice(path, "RING")
    assert specs[0].kind == "drift"
    assert specs[0].length == 1.0


def test_astra_namelist_reader_is_local(tmp_path: Path):
    path = tmp_path / "injector.in"
    path.write_text("&NEWRUN H_max=0.1, H_min=0.01 &END\n", encoding="utf-8")
    values = read_astra_namelist(path)
    assert values["newrun"]["h_max"] == 0.1
