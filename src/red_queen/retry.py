"""Retry with exponential backoff for network calls (Hub datasets and models)."""

from __future__ import annotations

import logging
import os
import random
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TypeVar

from red_queen.config import HubConfig
from red_queen.errors import DownloadError
from red_queen.logging_setup import fields

log = logging.getLogger(__name__)

T = TypeVar("T")

# Errors worth retrying: connection resets, timeouts, HTTP 5xx surfaced as OSError.
RETRYABLE: tuple[type[BaseException], ...] = (OSError, ConnectionError, TimeoutError)


def backoff_delays(
    attempts: int, base_s: float, max_s: float, jitter: random.Random
) -> list[float]:
    """Delays before attempts 2..n: ``base * 2**k`` capped at ``max_s``, with full jitter."""
    return [jitter.uniform(0, min(max_s, base_s * 2**k)) for k in range(attempts - 1)]


def with_retries(
    fn: Callable[[], T],
    what: str,
    policy: HubConfig,
    *,
    retry_on: tuple[type[BaseException], ...] = RETRYABLE,
    sleep: Callable[[float], None] = time.sleep,
    jitter: random.Random | None = None,
) -> T:
    """Call ``fn`` until it succeeds or ``policy.download_attempts`` is used up.

    Raises :class:`DownloadError` that chains the last underlying exception.
    """
    delays = backoff_delays(
        policy.download_attempts,
        policy.backoff_base_s,
        policy.backoff_max_s,
        jitter or random.Random(),
    )
    for attempt in range(1, policy.download_attempts + 1):
        try:
            with hub_timeout(policy.timeout_s):
                return fn()
        except retry_on as exc:
            if attempt == policy.download_attempts:
                raise DownloadError(f"{what} failed after {attempt} attempts: {exc}") from exc
            delay = delays[attempt - 1]
            log.warning(
                "download failed, retrying",
                extra=fields(what=what, attempt=attempt, delay_s=round(delay, 2), error=str(exc)),
            )
            sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


@contextmanager
def hub_timeout(seconds: int) -> Iterator[None]:
    """Set the Hugging Face Hub read timeout for the duration of a call."""
    key = "HF_HUB_DOWNLOAD_TIMEOUT"
    previous = os.environ.get(key)
    os.environ[key] = str(seconds)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous
