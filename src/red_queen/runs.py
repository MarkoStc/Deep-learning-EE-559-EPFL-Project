"""Run directories and the artifact manifest.

A run is one directory under ``paths.runs_dir`` named
``<YYYYmmdd-HHMMSS>-<label>``. It holds the resolved ``config.yaml``, a
``metadata.json`` (git commit, library versions) and a ``manifest.json`` that
maps artifact names to paths inside the run. Pipeline steps look up their
inputs in the manifest, so nobody has to copy run numbers between commands.
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from importlib import metadata
from pathlib import Path

from red_queen.config import AppConfig
from red_queen.errors import ArtifactError
from red_queen.logging_setup import fields

log = logging.getLogger(__name__)

CONFIG_FILE = "config.yaml"
METADATA_FILE = "metadata.json"
MANIFEST_FILE = "manifest.json"
_TRACKED_PACKAGES = ("torch", "transformers", "datasets", "scikit-learn", "numpy", "pandas")
_MAX_NAME_ATTEMPTS = 100


class Artifact(StrEnum):
    """Names of the artifacts a run can contain."""

    GUARDIAN = "guardian"
    EVADER = "evader"
    STRESS_TRAIN = "stress_train"
    STRESS_EVAL = "stress_eval"
    ROBUST = "robust"
    EVALUATION = "evaluation"
    OUTER_LOOP = "outer_loop"


@dataclass(frozen=True)
class Manifest:
    """``manifest.json`` of one run: artifact name -> path relative to the run."""

    path: Path

    def _read(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ArtifactError(f"{self.path} is not a JSON object")
        return {str(k): str(v) for k, v in data.items()}

    def has(self, name: Artifact) -> bool:
        return name.value in self._read()

    def get(self, name: Artifact) -> Path:
        entries = self._read()
        if name.value not in entries:
            raise ArtifactError(
                f"run {self.path.parent} has no '{name.value}' artifact yet "
                f"(available: {sorted(entries) or 'none'})"
            )
        target = self.path.parent / entries[name.value]
        if not target.exists():
            raise ArtifactError(f"manifest points at {target}, which does not exist")
        return target

    def record(self, name: Artifact, target: Path) -> None:
        entries = self._read()
        entries[name.value] = target.resolve().relative_to(self.path.parent.resolve()).as_posix()
        self.path.write_text(json.dumps(entries, indent=2, sort_keys=True), encoding="utf-8")


class RunDirectory:
    """A directory holding every artifact of one experiment."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.manifest = Manifest(root / MANIFEST_FILE)

    @classmethod
    def create(cls, runs_dir: Path, label: str, now: datetime | None = None) -> RunDirectory:
        """Create a new, uniquely named run directory (never reuses an existing one)."""
        stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
        runs_dir.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, _MAX_NAME_ATTEMPTS + 1):
            name = f"{stamp}-{label}" if attempt == 1 else f"{stamp}-{label}-{attempt}"
            try:
                (runs_dir / name).mkdir()
            except FileExistsError:
                continue
            log.info("created run directory", extra=fields(path=str(runs_dir / name)))
            return cls(runs_dir / name)
        raise ArtifactError(f"could not find a free run name for {stamp}-{label} in {runs_dir}")

    @classmethod
    def open(cls, root: Path) -> RunDirectory:
        """Open an existing run directory."""
        if not root.is_dir():
            raise ArtifactError(f"run directory {root} does not exist")
        return cls(root)

    def subdir(self, *parts: str) -> Path:
        path = self.root.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_config(self, cfg: AppConfig) -> Path:
        path = self.root / CONFIG_FILE
        cfg.to_yaml(path)
        return path

    def write_metadata(self) -> Path:
        info = {
            "created": datetime.now().isoformat(timespec="seconds"),
            "git_commit": _git_commit(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {name: _version(name) for name in _TRACKED_PACKAGES},
        }
        path = self.root / METADATA_FILE
        path.write_text(json.dumps(info, indent=2), encoding="utf-8")
        return path


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None
