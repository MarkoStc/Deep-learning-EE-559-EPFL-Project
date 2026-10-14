"""Experiment tracking that works offline.

CSV logs are always available: one file per stage under ``<run>/metrics/``.
TensorBoard and Weights & Biases are opt-in through ``tracking`` in the config
and need their optional extras installed (``uv sync --extra tensorboard``).
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from red_queen.config import AppConfig, TrackingConfig

log = logging.getLogger(__name__)

Metrics = Mapping[str, float]


class Tracker(Protocol):
    def log(self, stage: str, step: int, metrics: Metrics) -> None:
        """Record ``metrics`` for training stage ``stage`` at ``step``."""

    def close(self) -> None: ...


class NullTracker:
    """Tracker that drops everything; the default in unit tests."""

    def log(self, stage: str, step: int, metrics: Metrics) -> None:
        return None

    def close(self) -> None:
        return None


class CsvTracker:
    """Writes ``<root>/<stage>.csv``. Columns grow if later rows add new metrics."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._rows: dict[str, list[dict[str, float | int]]] = {}

    def log(self, stage: str, step: int, metrics: Metrics) -> None:
        rows = self._rows.setdefault(stage, self._existing_rows(stage))
        rows.append({"step": step, **{k: float(v) for k, v in metrics.items()}})
        self._write(stage, rows)

    def close(self) -> None:
        return None

    def path(self, stage: str) -> Path:
        return self.root / f"{stage.replace('/', '_')}.csv"

    def _existing_rows(self, stage: str) -> list[dict[str, float | int]]:
        # Resumed runs append to what the earlier process wrote.
        path = self.path(stage)
        if not path.exists():
            return []
        with path.open(newline="", encoding="utf-8") as fh:
            return [_parse_row(row) for row in csv.DictReader(fh)]

    def _write(self, stage: str, rows: Sequence[Mapping[str, float | int]]) -> None:
        columns: list[str] = []
        for row in rows:
            columns.extend(k for k in row if k not in columns)
        self.root.mkdir(parents=True, exist_ok=True)
        with self.path(stage).open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)


def _parse_row(row: Mapping[str, str | None]) -> dict[str, float | int]:
    parsed: dict[str, float | int] = {}
    for key, value in row.items():
        if value in (None, ""):
            continue
        parsed[key] = int(float(value)) if key == "step" else float(value)
    return parsed


class TensorBoardTracker:
    def __init__(self, root: Path) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter  # noqa: PLC0415  optional extra
        except ImportError as exc:  # pragma: no cover - depends on installed extras
            raise RuntimeError(
                "tracking.tensorboard is on but tensorboard is not installed; "
                "run `uv sync --extra tensorboard`"
            ) from exc
        self._writer = SummaryWriter(log_dir=str(root))

    def log(self, stage: str, step: int, metrics: Metrics) -> None:
        for key, value in metrics.items():
            self._writer.add_scalar(f"{stage}/{key}", value, step)

    def close(self) -> None:
        self._writer.close()


class WandbTracker:  # pragma: no cover - needs an account or offline wandb install
    def __init__(self, project: str, root: Path, config: Mapping[str, Any]) -> None:
        try:
            import wandb  # noqa: PLC0415  optional extra
        except ImportError as exc:
            raise RuntimeError(
                "tracking.wandb is on but wandb is not installed; run `uv sync --extra wandb`"
            ) from exc
        self._run = wandb.init(project=project, dir=str(root), config=dict(config))

    def log(self, stage: str, step: int, metrics: Metrics) -> None:
        self._run.log({f"{stage}/{k}": v for k, v in metrics.items()}, step=step)

    def close(self) -> None:
        self._run.finish()


class MultiTracker:
    """Fans every call out to several trackers."""

    def __init__(self, trackers: Sequence[Tracker]) -> None:
        self.trackers = tuple(trackers)

    def log(self, stage: str, step: int, metrics: Metrics) -> None:
        for tracker in self.trackers:
            tracker.log(stage, step, metrics)

    def close(self) -> None:
        for tracker in self.trackers:
            tracker.close()


def build_tracker(tracking: TrackingConfig, run_root: Path, cfg: AppConfig) -> Tracker:
    """Create the trackers switched on in ``tracking`` for a run directory."""
    trackers: list[Tracker] = []
    if tracking.csv:
        trackers.append(CsvTracker(run_root / "metrics"))
    if tracking.tensorboard:
        trackers.append(TensorBoardTracker(run_root / "tensorboard"))
    if tracking.wandb:
        trackers.append(WandbTracker(tracking.wandb_project, run_root, cfg.model_dump(mode="json")))
    return MultiTracker(trackers)
