from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from red_queen.config import AppConfig, HubConfig, parse_config
from red_queen.data import (
    CachedSource,
    CivilCommentsSource,
    DatasetSource,
    HateXplainSource,
    SyntheticSource,
    build_source,
    load_splits,
    stratified_subsample,
    validate_frame,
)
from red_queen.errors import DataError, DownloadError

FAST_HUB = HubConfig(download_attempts=2, backoff_base_s=0.001, backoff_max_s=0.001, timeout_s=1)


class FakeHubDataset:
    """Mimics the bits of `datasets.Dataset` / `DatasetDict` the sources use."""

    def __init__(
        self, frame: pd.DataFrame | None = None, splits: dict[str, list[Any]] | None = None
    ):
        self.frame = frame
        self.splits = splits or {}

    def to_pandas(self) -> pd.DataFrame:
        assert self.frame is not None
        return self.frame

    def __contains__(self, key: str) -> bool:
        return key in self.splits

    def __getitem__(self, key: str) -> list[Any]:
        return self.splits[key]


def civil_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": ["fine", "awful", "ok", None, "bad"],
            "toxicity": [0.1, 0.9, 0.49, 0.7, 0.5],
            "female": [0.0, 1.0, 0.0, 0.0, 0.2],
            "unused_column": [1, 2, 3, 4, 5],
        }
    )


def test_civil_comments_binarises_and_keeps_identity_columns() -> None:
    calls: list[tuple[Any, ...]] = []

    def loader(*args: Any, **kwargs: Any) -> FakeHubDataset:
        calls.append(args)
        return FakeHubDataset(civil_frame())

    df = validate_frame(CivilCommentsSource(0.5, FAST_HUB, loader).load(), "civil")
    assert calls == [("google/civil_comments",)]
    assert df["label"].tolist() == [0, 1, 0, 1]  # None text dropped, 0.5 counts as hate
    assert "female" in df.columns
    assert "unused_column" not in df.columns


def test_civil_comments_retries_then_raises() -> None:
    attempts = 0

    def loader(*args: Any, **kwargs: Any) -> FakeHubDataset:
        nonlocal attempts
        attempts += 1
        raise ConnectionError("offline")

    with pytest.raises(DownloadError):
        CivilCommentsSource(0.5, FAST_HUB, loader).load()
    assert attempts == 2


def test_hatexplain_majority_vote() -> None:
    rows = [
        {"post_tokens": ["hi", "there"], "annotators": {"label": [0, 0, 2]}},
        {"post_tokens": ["go", "away"], "annotators": {"label": [1, 2, 1]}},
    ]

    def loader(*args: Any, **kwargs: Any) -> FakeHubDataset:
        assert kwargs["revision"] == "refs/convert/parquet"
        return FakeHubDataset(splits={"train": rows})

    df = HateXplainSource(FAST_HUB, loader).load()
    assert df["text"].tolist() == ["hi there", "go away"]
    assert df["label"].tolist() == [0, 1]
    assert df["original_label"].tolist() == [0, 1]


def test_synthetic_is_deterministic_and_marker_based() -> None:
    a = SyntheticSource(200, seed=1).load()
    b = SyntheticSource(200, seed=1).load()
    pd.testing.assert_frame_equal(a, b)
    has_marker = a["text"].str.contains("|".join(SyntheticSource.MARKERS))
    assert (has_marker == (a["label"] == 1)).all()
    assert 0.15 < a["label"].mean() < 0.45


@pytest.mark.parametrize(
    ("frame", "match"),
    [
        (pd.DataFrame({"text": ["a"]}), "missing columns"),
        (pd.DataFrame({"text": ["a", "b"], "label": [0, 2]}), "labels must be 0/1"),
        (pd.DataFrame({"text": ["a", "b"], "label": [1, 1]}), "both classes"),
    ],
)
def test_validate_frame_errors(frame: pd.DataFrame, match: str) -> None:
    with pytest.raises(DataError, match=match):
        validate_frame(frame, "test")


class CountingSource(DatasetSource):
    name = "counting"

    def __init__(self) -> None:
        self.calls = 0

    def load(self) -> pd.DataFrame:
        self.calls += 1
        return pd.DataFrame({"text": ["a", "b"], "label": [0, 1]})


def test_cache_is_written_then_reused(tmp_path: Path) -> None:
    inner = CountingSource()
    cached = CachedSource(inner, tmp_path / "c.parquet")
    cached.load()
    cached.load()
    assert inner.calls == 1


def test_stale_cache_is_rebuilt(tmp_path: Path) -> None:
    path = tmp_path / "c.parquet"
    pd.DataFrame({"wrong": [1]}).to_parquet(path)
    inner = CountingSource()
    assert CachedSource(inner, path).load()["label"].tolist() == [0, 1]
    assert inner.calls == 1


def test_build_source_picks_and_caches(tmp_path: Path) -> None:
    cfg = parse_config({"paths": {"data_dir": str(tmp_path)}})
    source = build_source(cfg, loader=lambda *a, **k: FakeHubDataset(civil_frame()))
    assert isinstance(source, CachedSource)
    assert source.cache_path.name == "civil_comments_t0.5.parquet"
    synthetic = build_source(parse_config({"dataset": {"name": "synthetic"}}))
    assert isinstance(synthetic, SyntheticSource)


def test_stratified_subsample_keeps_proportions() -> None:
    df = pd.DataFrame({"text": [str(i) for i in range(1000)], "label": [1] * 200 + [0] * 800})
    sub = stratified_subsample(df, 100, seed=0)
    assert len(sub) == 100
    assert sub["label"].mean() == pytest.approx(0.2)
    assert stratified_subsample(df, None, seed=0) is df


def test_load_splits_sizes_disjoint_and_stratified() -> None:
    cfg = parse_config({"dataset": {"name": "synthetic", "synthetic_size": 1000}})
    splits = load_splits(cfg)
    total = len(splits.train) + len(splits.val) + len(splits.test)
    assert total == 1000
    assert len(splits.test) == 150
    assert len(splits.val) == 150
    shares = [s["label"].mean() for s in (splits.train, splits.val, splits.test)]
    assert max(shares) - min(shares) < 0.02
    assert splits.hateful_texts("test") == splits.test.loc[splits.test.label == 1, "text"].tolist()
    with pytest.raises(ValueError):
        splits.split("holdout")


def test_load_splits_is_repeatable(app_config: AppConfig) -> None:
    cfg = parse_config({"dataset": {"name": "synthetic"}, "seed": 9})
    a, b = load_splits(cfg), load_splits(cfg)
    pd.testing.assert_frame_equal(a.test, b.test)
