from __future__ import annotations

import os
import random

import pytest

from red_queen.config import HubConfig
from red_queen.errors import DownloadError
from red_queen.retry import backoff_delays, hub_timeout, with_retries

POLICY = HubConfig(download_attempts=3, backoff_base_s=1.0, backoff_max_s=3.0, timeout_s=5)


class Flaky:
    def __init__(self, failures: int, exc: BaseException) -> None:
        self.failures = failures
        self.exc = exc
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exc
        return "ok"


def test_backoff_is_capped_and_non_negative() -> None:
    delays = backoff_delays(6, 1.0, 3.0, random.Random(0))
    assert len(delays) == 5
    assert all(0 <= d <= 3.0 for d in delays)


def test_retries_until_success() -> None:
    sleeps: list[float] = []
    fn = Flaky(2, ConnectionError("reset"))
    assert with_retries(fn, "thing", POLICY, sleep=sleeps.append, jitter=random.Random(0)) == "ok"
    assert fn.calls == 3
    assert len(sleeps) == 2


def test_gives_up_with_download_error() -> None:
    fn = Flaky(5, TimeoutError("slow"))
    with pytest.raises(DownloadError, match="thing failed after 3 attempts") as info:
        with_retries(fn, "thing", POLICY, sleep=lambda _: None)
    assert isinstance(info.value.__cause__, TimeoutError)


def test_non_retryable_errors_propagate_immediately() -> None:
    fn = Flaky(1, ValueError("bad input"))
    with pytest.raises(ValueError):
        with_retries(fn, "thing", POLICY, sleep=lambda _: None)
    assert fn.calls == 1


def test_hub_timeout_restores_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_DOWNLOAD_TIMEOUT", "11")
    with hub_timeout(99):
        assert os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] == "99"
    assert os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] == "11"
    monkeypatch.delenv("HF_HUB_DOWNLOAD_TIMEOUT")
    with hub_timeout(5):
        pass
    assert "HF_HUB_DOWNLOAD_TIMEOUT" not in os.environ
