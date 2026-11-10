"""Concurrent observer metadata cannot disrupt scientific artifact accounting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.metaworld_development import ArtifactStore, Stopped


def test_no_observer_preserves_original_actual_byte_limits(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts", 1000)
    with store.open("scientific.bin") as handle:
        handle.write(b"s" * 900)
        with pytest.raises(Stopped, match="artifact_storage_limit"):
            handle.write(b"x")
    assert store.size() == store.accounted_size() == 900
    store.json("manifest.json", {})
    assert store.size() == store.accounted_size() == 903


@pytest.mark.parametrize("metadata", [False, True])
def test_growth_of_parallel_reports_is_reserved_before_scientific_write(
    tmp_path: Path, metadata: bool
) -> None:
    store = ArtifactStore(tmp_path / "artifacts", 1024**2)
    reports = store.root / "live_reports"
    reports.mkdir()
    report = reports / "report.json"
    report.write_bytes(b"r")
    reservation = store.limit // 10
    assert store.size() == 1
    assert store.accounted_size() == reservation
    maximum = store.limit - reservation - (0 if metadata else store.reserve)
    with store.open("scientific.bin", metadata=metadata) as handle:
        # Grow the independent observer after CappedWriter snapshots others.
        report.write_bytes(b"r" * reservation)
        handle.write(b"s" * maximum)
        with pytest.raises(Stopped, match="artifact_storage_limit"):
            handle.write(b"x")
    assert store.size() == reservation + maximum
    assert store.size() <= store.limit
    assert store.size() == store.accounted_size()


def test_disappearing_observer_atomic_file_does_not_fail_scientific_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ArtifactStore(tmp_path / "artifacts", 1024**2)
    reports = store.root / "live_reports"
    reports.mkdir()
    temporary = reports / ".report.json.tmp"
    temporary.write_bytes(b"temporary reporter bytes")
    original = Path.stat

    def raced(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path == temporary:
            temporary.unlink(missing_ok=True)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", raced)
    with store.open("scientific.bin", metadata=True) as handle:
        handle.write(b"retained scientific output")
    assert not temporary.exists()
    assert (store.root / "scientific.bin").read_bytes() == b"retained scientific output"
    assert store.size() == 26


def test_unexpected_scientific_file_loss_is_not_silently_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ArtifactStore(tmp_path / "artifacts", 1000)
    raw = store.root / "scientific.bin"
    raw.write_bytes(b"original")
    original = Path.stat

    def raced(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path == raw:
            raw.unlink(missing_ok=True)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", raced)
    with pytest.raises(FileNotFoundError):
        store.size()


def test_unexpected_existing_report_bytes_are_counted_above_reservation(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts", 1000)
    reports = store.root / "live_reports"
    reports.mkdir()
    (reports / "report.json").write_bytes(b"r" * 200)
    assert store.size() == store.accounted_size() == 200
    with store.open("scientific.bin", metadata=True) as handle:
        handle.write(b"s" * 800)
        with pytest.raises(Stopped, match="artifact_storage_limit"):
            handle.write(b"x")


def test_reservation_is_capped_at_one_mebibyte_for_large_allowance(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts", 64 * 1024**2)
    (store.root / "live_reports").mkdir()
    assert store.size() == 0
    assert store.accounted_size() == 1024**2
