from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import pytest

from red_queen.config import AppConfig, TrackingConfig
from red_queen.errors import ArtifactError
from red_queen.runs import Artifact, RunDirectory
from red_queen.tracking import CsvTracker, MultiTracker, NullTracker, build_tracker

NOW = datetime(2026, 10, 14, 12, 30, 0)


def test_run_names_never_collide(tmp_path: Path) -> None:
    first = RunDirectory.create(tmp_path, "guardian", now=NOW)
    second = RunDirectory.create(tmp_path, "guardian", now=NOW)
    assert first.root.name == "20261014-123000-guardian"
    assert second.root.name == "20261014-123000-guardian-2"


def test_deleting_a_run_does_not_cause_reuse(tmp_path: Path) -> None:
    # The legacy run_NNN counter used len(existing)+1 and overwrote runs after a deletion.
    a = RunDirectory.create(tmp_path, "x", now=NOW)
    b = RunDirectory.create(tmp_path, "x", now=NOW)
    (b.root / "keep.txt").write_text("b")
    a.root.rmdir()
    c = RunDirectory.create(tmp_path, "x", now=NOW)
    assert c.root != b.root
    assert (b.root / "keep.txt").read_text() == "b"


def test_open_missing_run_fails(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError):
        RunDirectory.open(tmp_path / "missing")


def test_manifest_records_relative_paths(tmp_path: Path) -> None:
    run = RunDirectory.create(tmp_path, "p", now=NOW)
    target = run.subdir("guardian", "best")
    run.manifest.record(Artifact.GUARDIAN, target)
    assert json.loads(run.manifest.path.read_text()) == {"guardian": "guardian/best"}
    reopened = RunDirectory.open(run.root)
    assert reopened.manifest.has(Artifact.GUARDIAN)
    assert reopened.manifest.get(Artifact.GUARDIAN) == target


def test_manifest_errors(tmp_path: Path) -> None:
    run = RunDirectory.create(tmp_path, "p", now=NOW)
    with pytest.raises(ArtifactError, match="no 'evader' artifact"):
        run.manifest.get(Artifact.EVADER)
    target = run.subdir("gone")
    run.manifest.record(Artifact.EVADER, target)
    target.rmdir()
    with pytest.raises(ArtifactError, match="does not exist"):
        run.manifest.get(Artifact.EVADER)


def test_config_and_metadata_are_saved(tmp_path: Path) -> None:
    run = RunDirectory.create(tmp_path, "p", now=NOW)
    assert run.save_config(AppConfig()).exists()
    meta = json.loads(run.write_metadata().read_text())
    assert {"git_commit", "python", "packages"} <= set(meta)
    assert meta["packages"]["torch"]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def test_csv_tracker_grows_columns(tmp_path: Path) -> None:
    tracker = CsvTracker(tmp_path)
    tracker.log("guardian", 1, {"loss": 0.5})
    tracker.log("guardian", 2, {"loss": 0.4, "val_f1": 0.7})
    rows = _read_csv(tmp_path / "guardian.csv")
    assert list(rows[0]) == ["step", "loss", "val_f1"]
    assert rows[1]["val_f1"] == "0.7"
    assert rows[0]["val_f1"] == ""


def test_csv_tracker_appends_after_restart(tmp_path: Path) -> None:
    CsvTracker(tmp_path).log("evader", 1, {"asr": 0.1})
    CsvTracker(tmp_path).log("evader", 2, {"asr": 0.2})
    assert [r["step"] for r in _read_csv(tmp_path / "evader.csv")] == ["1", "2"]


def test_build_tracker_csv_only_by_default(tmp_path: Path) -> None:
    tracker = build_tracker(TrackingConfig(), tmp_path, AppConfig())
    assert isinstance(tracker, MultiTracker)
    assert [type(t) for t in tracker.trackers] == [CsvTracker]
    tracker.log("s", 0, {"m": 1.0})
    tracker.close()
    assert (tmp_path / "metrics" / "s.csv").exists()


def test_null_tracker_accepts_calls() -> None:
    NullTracker().log("s", 0, {"m": 1.0})


def test_tensorboard_tracker_writes_event_file(tmp_path: Path) -> None:
    pytest.importorskip("tensorboard")
    tracker = build_tracker(TrackingConfig(csv=False, tensorboard=True), tmp_path, AppConfig())
    tracker.log("guardian", 1, {"loss": 0.3})
    tracker.close()
    assert list((tmp_path / "tensorboard").glob("events.out.tfevents.*"))
