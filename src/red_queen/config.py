"""Typed experiment configuration.

One YAML file describes a whole experiment. :func:`load_config` parses it into
an :class:`AppConfig`; unknown keys are rejected so a typo fails loudly instead
of silently falling back to a default. Dotted overrides
(``guardian.epochs=1``) can be applied on top, which the CLI exposes as
``--set``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from red_queen.errors import ConfigError


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetName(StrEnum):
    """Datasets the loader knows how to fetch."""

    CIVIL_COMMENTS = "civil_comments"
    HATEXPLAIN = "hatexplain"
    SYNTHETIC = "synthetic"


class PathsConfig(_Section):
    data_dir: Path = Path("data")
    runs_dir: Path = Path("runs")


class HubConfig(_Section):
    """Retry and timeout policy for Hugging Face Hub downloads."""

    download_attempts: int = Field(4, ge=1)
    backoff_base_s: float = Field(2.0, gt=0)
    backoff_max_s: float = Field(60.0, gt=0)
    timeout_s: int = Field(60, gt=0)


class DatasetConfig(_Section):
    name: DatasetName = DatasetName.CIVIL_COMMENTS
    max_samples: int | None = Field(50_000, gt=0)
    test_size: float = Field(0.15, gt=0, lt=1)
    val_size: float = Field(0.15, gt=0, lt=1)
    toxicity_threshold: float = Field(0.5, ge=0, le=1)
    synthetic_size: int = Field(600, ge=20)

    @model_validator(mode="after")
    def _splits_leave_training_data(self) -> DatasetConfig:
        if self.test_size + self.val_size >= 1:
            raise ValueError("test_size + val_size must be below 1")
        return self


class TinyModelConfig(_Section):
    """A randomly initialised DistilBERT built offline, for tests and CPU smoke runs."""

    dim: int = Field(32, gt=0)
    n_layers: int = Field(1, gt=0)
    n_heads: int = Field(2, gt=0)
    hidden_dim: int = Field(64, gt=0)
    vocab_size: int = Field(400, ge=50)

    @model_validator(mode="after")
    def _heads_divide_dim(self) -> TinyModelConfig:
        if self.dim % self.n_heads:
            raise ValueError("dim must be divisible by n_heads")
        return self


@dataclass(frozen=True)
class TrainSchedule:
    """Optimiser schedule for one fine-tuning stage of the Guardian."""

    epochs: int
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    batch_size: int
    grad_clip_norm: float


class GuardianConfig(_Section):
    model_name: str = "distilbert-base-uncased"
    tiny: TinyModelConfig | None = None
    max_length: int = Field(128, gt=0)
    batch_size: int = Field(32, gt=0)
    eval_batch_size: int = Field(128, gt=0)
    learning_rate: float = Field(2.0e-5, gt=0)
    weight_decay: float = Field(0.01, ge=0)
    epochs: int = Field(3, gt=0)
    warmup_ratio: float = Field(0.1, ge=0, lt=1)
    grad_clip_norm: float = Field(1.0, gt=0)
    keep_resume_state: bool = False
    """Keep ``<stage>/last`` (model + optimizer state) after training finishes."""

    def schedule(self) -> TrainSchedule:
        return TrainSchedule(
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            warmup_ratio=self.warmup_ratio,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            grad_clip_norm=self.grad_clip_norm,
        )


class RobustConfig(_Section):
    """Second fine-tuning stage on train data plus stress cases."""

    augmentation_ratio: float = Field(0.5, gt=0, le=1)
    epochs: int = Field(2, gt=0)
    learning_rate: float = Field(1.0e-5, gt=0)
    warmup_ratio: float = Field(0.0, ge=0, lt=1)

    def schedule(self, guardian: GuardianConfig) -> TrainSchedule:
        """Robust retraining reuses the Guardian's batch size, decay and clipping."""
        return TrainSchedule(
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            warmup_ratio=self.warmup_ratio,
            weight_decay=guardian.weight_decay,
            batch_size=guardian.batch_size,
            grad_clip_norm=guardian.grad_clip_norm,
        )


class EvaderConfig(_Section):
    hidden_dim: int = Field(64, gt=0)
    learning_rate: float = Field(1.0e-3, gt=0)
    episodes: int = Field(2000, gt=0)
    batch_size: int = Field(64, gt=0)
    min_batch_size: int = Field(16, gt=0)
    entropy_coeff: float = Field(0.01, ge=0)
    baseline_decay: float = Field(0.99, ge=0, lt=1)
    grad_clip_norm: float = Field(1.0, gt=0)
    log_every: int = Field(50, gt=0)


class PerturbationConfig(_Section):
    typo_prob: float = Field(0.3, ge=0, le=1)
    leet_prob: float = Field(0.4, ge=0, le=1)
    whitespace_prob: float = Field(0.3, ge=0, le=1)
    homoglyph_prob: float = Field(0.3, ge=0, le=1)
    synonym_max_replacements: int = Field(2, ge=0)
    synonym_max_synsets: int = Field(3, gt=0)
    emoji_insert_count: int = Field(1, ge=0)


class StressConfig(_Section):
    batch_size: int = Field(64, gt=0)


DEFAULT_IDENTITY_COLUMNS = (
    "male",
    "female",
    "black",
    "white",
    "asian",
    "latino",
    "homosexual_gay_or_lesbian",
    "muslim",
    "christian",
    "jewish",
)


class EvaluationConfig(_Section):
    sanitize_logs: bool = True
    mask_char: str = Field("█", min_length=1, max_length=1)
    max_display_examples: int = Field(10, ge=0)
    identity_threshold: float = Field(0.5, ge=0, le=1)
    min_subgroup_size: int = Field(10, gt=0)
    identity_columns: tuple[str, ...] = DEFAULT_IDENTITY_COLUMNS


class OuterLoopConfig(_Section):
    num_rounds: int = Field(5, gt=0)


class TrackingConfig(_Section):
    """Experiment tracking. CSV always works offline; the others are opt-in."""

    csv: bool = True
    tensorboard: bool = False
    wandb: bool = False
    wandb_project: str = "red-queen"


class AppConfig(_Section):
    """Root of the experiment config."""

    seed: int = 42
    deterministic: bool = True
    device: Literal["auto", "cpu", "cuda"] = "auto"
    paths: PathsConfig = PathsConfig()
    hub: HubConfig = HubConfig()
    dataset: DatasetConfig = DatasetConfig()
    guardian: GuardianConfig = GuardianConfig()
    robust: RobustConfig = RobustConfig()
    evader: EvaderConfig = EvaderConfig()
    perturbation: PerturbationConfig = PerturbationConfig()
    stress: StressConfig = StressConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    outer_loop: OuterLoopConfig = OuterLoopConfig()
    tracking: TrackingConfig = TrackingConfig()

    def to_yaml(self, path: Path) -> None:
        """Write the resolved config, so a run can be repeated from its own directory."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump(mode="json")
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Return a copy of ``raw`` with ``a.b.c=value`` overrides applied.

    Values are parsed as YAML, so ``guardian.epochs=1`` gives an int and
    ``guardian.tiny=null`` gives ``None``.
    """
    result: dict[str, Any] = _deep_copy(raw)
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ConfigError(f"override must look like key.path=value, got {item!r}")
        node = result
        *parents, leaf = key.strip().split(".")
        for part in parents:
            child = node.setdefault(part, {})
            if not isinstance(child, dict):
                raise ConfigError(f"cannot set {key!r}: {part!r} is not a section")
            node = child
        node[leaf] = yaml.safe_load(value)
    return result


def load_config(path: Path | str, overrides: list[str] | None = None) -> AppConfig:
    """Read a YAML config file, apply overrides and validate it."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"config file {path} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"config file {path} must contain a mapping at the top level")
    return parse_config(apply_overrides(raw, overrides or []), source=str(path))


def parse_config(raw: dict[str, Any], source: str = "<dict>") -> AppConfig:
    """Validate an already-parsed config mapping."""
    try:
        return AppConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid config in {source}:\n{exc}") from exc


def _deep_copy(data: dict[str, Any]) -> dict[str, Any]:
    return {k: _deep_copy(v) if isinstance(v, dict) else v for k, v in data.items()}
