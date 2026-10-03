"""Tests for slipstream.data.results.TrackingResult/StreamingH5Writer."""
from __future__ import annotations

import h5py
import numpy as np
import pytest

from slipstream.data.results import StreamingH5Writer, TrackingResult


def _make_result(**overrides) -> TrackingResult:
    defaults = dict(
        code="pystable",
        n_particles=1000,
        n_turns=100,
        current_A=5e-4,
        turn=np.array([10, 20, 30, 40]),
        sigma_delta=np.array([7.3e-4, 7.4e-4, 7.2e-4, 7.3e-4]),
        sigma_tau_s=np.array([7.0e-12, 7.1e-12, 7.0e-12, 7.05e-12]),
        transmission=np.array([1.0, 1.0, 0.999, 0.999]),
        metadata={"rho": 1e-5, "seed": 42},
    )
    defaults.update(overrides)
    return TrackingResult(**defaults)


def test_summarize_over_last_n():
    result = _make_result()
    sd_mean, sd_std, st_mean, st_std = result.summarize(last_n=2)
    assert sd_mean == pytest.approx(result.sigma_delta[-2:].mean())
    assert st_mean == pytest.approx(result.sigma_tau_s[-2:].mean())


def test_hdf5_round_trip(tmp_path):
    result = _make_result()
    path = tmp_path / "result.h5"
    result.save(path)
    loaded = TrackingResult.load(path)

    assert loaded.code == result.code
    assert loaded.n_particles == result.n_particles
    assert loaded.current_A == pytest.approx(result.current_A)
    assert np.allclose(loaded.turn, result.turn)
    assert np.allclose(loaded.sigma_delta, result.sigma_delta)
    assert np.allclose(loaded.sigma_tau_s, result.sigma_tau_s)
    assert np.allclose(loaded.transmission, result.transmission)
    assert loaded.metadata["rho"] == pytest.approx(1e-5)
    assert loaded.metadata["seed"] == 42


def test_hdf5_round_trip_without_transmission(tmp_path):
    result = _make_result(transmission=None, metadata={})
    path = tmp_path / "result_no_trans.h5"
    result.save(path)
    loaded = TrackingResult.load(path)
    assert loaded.transmission is None
    assert loaded.metadata == {}


def test_streaming_h5_writer_readable_as_tracking_result(tmp_path):
    """A file built incrementally, one append() per sample, must be
    readable back via TrackingResult.load() -- the whole point of sharing
    the same on-disk schema."""
    path = tmp_path / "streaming.h5"
    attrs = dict(code="pystable", n_particles=1000, n_turns=40, current_A=5e-4)
    with StreamingH5Writer(path, attrs=attrs, metadata={"seed": 7}) as writer:
        for turn, sd, st in [(10, 7.3e-4, 7.0e-12), (20, 7.4e-4, 7.1e-12), (30, 7.2e-4, 7.0e-12)]:
            writer.append(turn=turn, sigma_delta=sd, sigma_tau_s=st)

    loaded = TrackingResult.load(path)
    assert loaded.code == "pystable"
    assert loaded.n_particles == 1000
    assert loaded.n_turns == 40
    assert loaded.metadata["seed"] == 7
    assert np.allclose(loaded.turn, [10, 20, 30])
    assert np.allclose(loaded.sigma_delta, [7.3e-4, 7.4e-4, 7.2e-4])
    assert np.allclose(loaded.sigma_tau_s, [7.0e-12, 7.1e-12, 7.0e-12])


def test_streaming_h5_writer_partial_file_readable_after_interrupt(tmp_path):
    """Rows written before close() must already be on disk -- the whole
    point of streaming instead of buffering-then-writing-once. Opens a
    *second*, independent read handle mid-write (not the same handle
    close() would use) to check this without needing an actual crash."""
    path = tmp_path / "partial.h5"
    writer = StreamingH5Writer(path, attrs=dict(code="pystable", n_particles=10, n_turns=99, current_A=0.0))
    writer.append(turn=10, sigma_delta=1e-3, sigma_tau_s=1e-11)
    writer.append(turn=20, sigma_delta=1.1e-3, sigma_tau_s=1.1e-11)

    with h5py.File(path, "r") as f:
        assert list(f["turn"][()]) == [10, 20]

    writer.append(turn=30, sigma_delta=1.2e-3, sigma_tau_s=1.2e-11)
    writer.close()
    loaded = TrackingResult.load(path)
    assert list(loaded.turn) == [10, 20, 30]


def test_epsx_epsy_round_trip(tmp_path):
    result = _make_result(
        epsx_m=np.array([1.2e-8, 1.1e-8, 1.0e-8, 9.5e-9]),
        epsy_m=np.array([1.3e-9, 1.2e-9, 1.1e-9, 1.0e-9]),
    )
    path = tmp_path / "result_eps.h5"
    result.save(path)
    loaded = TrackingResult.load(path)
    assert np.allclose(loaded.epsx_m, result.epsx_m)
    assert np.allclose(loaded.epsy_m, result.epsy_m)


def test_streaming_h5_writer_flush_every(tmp_path):
    """flush_every=N calls the underlying HDF5 flush() roughly every N
    appends, not on every single one -- the actual, implementation-level
    behavior this option controls. (Not tested here: whether unflushed
    writes are visible to another independent read handle in the
    meantime -- that turns out to be h5py/platform-dependent, not
    something flush_every should be assumed to gate, see its own
    docstring.) close() must still leave a fully correct, complete file
    regardless of how many appends happened since the last flush."""
    path = tmp_path / "throttled.h5"
    writer = StreamingH5Writer(
        path, attrs=dict(code="pystable", n_particles=10, n_turns=99, current_A=0.0), flush_every=5,
    )
    flush_calls = []
    writer._f.flush = lambda: flush_calls.append(1)
    for i in range(12):
        writer.append(turn=i, sigma_delta=1e-3, sigma_tau_s=1e-11)
    assert len(flush_calls) == 2  # after the 5th and 10th appends; the 11th/12th haven't hit 5 yet
    writer.close()

    loaded = TrackingResult.load(path)
    assert list(loaded.turn) == list(range(12))


def test_streaming_h5_writer_extra_columns(tmp_path):
    """No fixed schema -- an arbitrary named column (e.g. ex/ey, not part
    of TrackingResult's own field set) works the same way."""
    path = tmp_path / "extra.h5"
    with StreamingH5Writer(path) as writer:
        writer.append(ex=1e-8, ey=1e-9)
        writer.append(ex=9e-9, ey=9e-10)

    with h5py.File(path, "r") as f:
        assert np.allclose(f["ex"][()], [1e-8, 9e-9])
        assert np.allclose(f["ey"][()], [1e-9, 9e-10])

