"""Dataset loading, caching and splitting.

Every source returns a DataFrame with at least ``text`` (str) and ``label``
(0 = safe, 1 = hate); Civil Comments also keeps its identity columns for the
subgroup analysis. Downloads go through :func:`red_queen.retry.with_retries`
and are cached as parquet under ``paths.data_dir``.
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from red_queen.config import AppConfig, DatasetConfig, DatasetName, HubConfig
from red_queen.errors import DataError
from red_queen.logging_setup import fields
from red_queen.retry import with_retries

log = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("text", "label")

CIVIL_COMMENTS_HUB_ID = "google/civil_comments"
HATEXPLAIN_HUB_ID = "Hate-speech-CNERG/hatexplain"
# HateXplain ships as a loading script; the Hub's parquet conversion avoids running it.
HATEXPLAIN_REVISION = "refs/convert/parquet"
HATEXPLAIN_NORMAL_LABEL = 0  # 0 normal, 1 offensive, 2 hate speech

IDENTITY_COLUMNS = (
    "male",
    "female",
    "transgender",
    "other_gender",
    "heterosexual",
    "homosexual_gay_or_lesbian",
    "bisexual",
    "other_sexual_orientation",
    "christian",
    "jewish",
    "muslim",
    "hindu",
    "buddhist",
    "atheist",
    "other_religion",
    "black",
    "white",
    "asian",
    "latino",
    "other_race_or_ethnicity",
    "physical_disability",
    "intellectual_or_learning_disability",
    "psychiatric_or_mental_illness",
    "other_disability",
)

HubLoader = Callable[..., Any]
"""Signature of ``datasets.load_dataset``; injected so tests never hit the network."""


def _default_hub_loader(*args: Any, **kwargs: Any) -> Any:
    from datasets import load_dataset  # noqa: PLC0415  slow import, only when downloading

    return load_dataset(*args, **kwargs)


@dataclass(frozen=True)
class DataSplits:
    """Stratified train / validation / test frames."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def split(self, name: str) -> pd.DataFrame:
        if name not in ("train", "val", "test"):
            raise ValueError(f"unknown split {name!r}")
        frame: pd.DataFrame = getattr(self, name)
        return frame

    def hateful_texts(self, name: str) -> list[str]:
        frame = self.split(name)
        return [str(t) for t in frame.loc[frame["label"] == 1, "text"]]


class DatasetSource(ABC):
    """Produces the full labelled dataset (before subsampling and splitting)."""

    name: str

    @abstractmethod
    def load(self) -> pd.DataFrame: ...


def validate_frame(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Check required columns and label values; drop rows without text."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"{source}: missing columns {missing}; found {list(df.columns)}")
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    labels = set(df["label"].unique())
    if not labels <= {0, 1}:
        raise DataError(f"{source}: labels must be 0/1, found {sorted(labels)}")
    if len(labels) < 2:
        raise DataError(f"{source}: needs both classes, found only {sorted(labels)}")
    df["label"] = df["label"].astype(int)
    return df


class CivilCommentsSource(DatasetSource):
    """Civil Comments, binarised at ``toxicity >= threshold``."""

    name = "civil_comments"

    def __init__(
        self, threshold: float, hub: HubConfig, loader: HubLoader = _default_hub_loader
    ) -> None:
        self.threshold = threshold
        self.hub = hub
        self.loader = loader

    def load(self) -> pd.DataFrame:
        ds = with_retries(
            lambda: self.loader(CIVIL_COMMENTS_HUB_ID, split="train"),
            CIVIL_COMMENTS_HUB_ID,
            self.hub,
        )
        raw: pd.DataFrame = ds.to_pandas()
        if "toxicity" not in raw.columns:
            raise DataError(f"{CIVIL_COMMENTS_HUB_ID}: no 'toxicity' column in {list(raw.columns)}")
        keep = ["text", "toxicity", *[c for c in IDENTITY_COLUMNS if c in raw.columns]]
        df = raw[keep].copy()
        df["label"] = (df["toxicity"] >= self.threshold).astype(int)
        return df


class HateXplainSource(DatasetSource):
    """HateXplain with a majority vote over annotators; offensive and hate both map to 1."""

    name = "hatexplain"

    def __init__(self, hub: HubConfig, loader: HubLoader = _default_hub_loader) -> None:
        self.hub = hub
        self.loader = loader

    def load(self) -> pd.DataFrame:
        ds = with_retries(
            lambda: self.loader(HATEXPLAIN_HUB_ID, revision=HATEXPLAIN_REVISION),
            HATEXPLAIN_HUB_ID,
            self.hub,
        )
        records = []
        for split_name in ("train", "validation", "test"):
            if split_name not in ds:
                continue
            for row in ds[split_name]:
                majority = Counter(row["annotators"]["label"]).most_common(1)[0][0]
                records.append(
                    {
                        "text": " ".join(row["post_tokens"]),
                        "label": int(majority != HATEXPLAIN_NORMAL_LABEL),
                        "original_label": majority,
                        "original_split": split_name,
                    }
                )
        return pd.DataFrame.from_records(records)


class SyntheticSource(DatasetSource):
    """Small generated dataset for tests and CPU smoke runs. Not for results.

    Positive rows contain one of two invented marker words, so the task is
    learnable and the data contains no real hateful language.
    """

    name = "synthetic"
    MARKERS = ("zorblat", "vexnor")
    POSITIVE_SHARE = 0.3
    _SUBJECTS = ("the river", "my neighbour", "this album", "the train", "our garden", "a teacher")
    _VERBS = ("looks", "seems", "felt", "sounds", "became", "stayed")
    _ADJECTIVES = ("calm", "bright", "early", "quiet", "useful", "late", "green", "warm")
    _TAILS = ("today", "again", "this week", "after lunch", "near the station", "as usual")

    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.seed = seed

    def load(self) -> pd.DataFrame:
        rng = random.Random(f"{self.seed}:synthetic-data")
        rows = []
        for _ in range(self.size):
            words = [
                rng.choice(self._SUBJECTS),
                rng.choice(self._VERBS),
                rng.choice(self._ADJECTIVES),
                rng.choice(self._TAILS),
            ]
            label = int(rng.random() < self.POSITIVE_SHARE)
            if label:
                words.insert(rng.randint(0, len(words)), rng.choice(self.MARKERS))
            rows.append({"text": " ".join(words), "label": label, "female": rng.random()})
        return pd.DataFrame.from_records(rows)


class CachedSource(DatasetSource):
    """Parquet cache in front of another source. A cache without the required
    columns is treated as stale and rebuilt."""

    def __init__(self, inner: DatasetSource, cache_path: Path) -> None:
        self.inner = inner
        self.cache_path = cache_path
        self.name = inner.name

    def load(self) -> pd.DataFrame:
        if self.cache_path.exists():
            try:
                df = pd.read_parquet(self.cache_path)
                validate_frame(df, str(self.cache_path))
                log.info("loaded dataset cache", extra=fields(path=str(self.cache_path)))
                return df
            except (DataError, OSError, ValueError) as exc:
                log.warning("dataset cache unusable, rebuilding", extra=fields(error=str(exc)))
        df = validate_frame(self.inner.load(), self.inner.name)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.cache_path, index=False)
        log.info("cached dataset", extra=fields(path=str(self.cache_path), rows=len(df)))
        return df


def build_source(
    cfg: AppConfig, loader: HubLoader = _default_hub_loader, cache: bool = True
) -> DatasetSource:
    """Pick the source named in the config, with a parquet cache for remote data."""
    ds_cfg = cfg.dataset
    source: DatasetSource
    if ds_cfg.name is DatasetName.SYNTHETIC:
        return SyntheticSource(ds_cfg.synthetic_size, cfg.seed)
    if ds_cfg.name is DatasetName.CIVIL_COMMENTS:
        source = CivilCommentsSource(ds_cfg.toxicity_threshold, cfg.hub, loader)
        cache_name = f"civil_comments_t{ds_cfg.toxicity_threshold:g}.parquet"
    else:
        source = HateXplainSource(cfg.hub, loader)
        cache_name = "hatexplain.parquet"
    return CachedSource(source, cfg.paths.data_dir / cache_name) if cache else source


def stratified_subsample(df: pd.DataFrame, max_samples: int | None, seed: int) -> pd.DataFrame:
    """Keep about ``max_samples`` rows with the original label proportions."""
    if max_samples is None or len(df) <= max_samples:
        return df
    frac = max_samples / len(df)
    parts = [
        group.sample(n=max(1, int(len(group) * frac)), random_state=seed)
        for _, group in df.groupby("label", sort=True)
    ]
    return pd.concat(parts, ignore_index=True)


def split_dataset(df: pd.DataFrame, cfg: DatasetConfig, seed: int) -> DataSplits:
    """Stratified split; ``val_size`` and ``test_size`` are shares of the whole set."""
    train_val, test = train_test_split(
        df, test_size=cfg.test_size, stratify=df["label"], random_state=seed
    )
    train, val = train_test_split(
        train_val,
        test_size=cfg.val_size / (1 - cfg.test_size),
        stratify=train_val["label"],
        random_state=seed,
    )
    splits = DataSplits(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )
    log.info(
        "dataset split",
        extra=fields(
            train=len(splits.train),
            val=len(splits.val),
            test=len(splits.test),
            train_hate_share=round(float(splits.train["label"].mean()), 4),
        ),
    )
    return splits


def load_splits(cfg: AppConfig, source: DatasetSource | None = None) -> DataSplits:
    """Load, subsample and split the configured dataset."""
    df = validate_frame((source or build_source(cfg)).load(), cfg.dataset.name.value)
    df = stratified_subsample(df, cfg.dataset.max_samples, cfg.seed)
    return split_dataset(df, cfg.dataset, cfg.seed)
