from __future__ import annotations

import json
import logging

import pytest

from red_queen.logging_setup import JsonFormatter, KeyValueFormatter, configure_logging, fields


def _record(**extra: object) -> logging.LogRecord:
    record = logging.LogRecord("red_queen.test", logging.INFO, __file__, 1, "hello", None, None)
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_key_value_formatter_appends_fields() -> None:
    line = KeyValueFormatter().format(_record(**fields(epoch=2, loss=0.123456)))
    assert line.endswith("hello epoch=2 loss=0.1235")


def test_json_formatter_puts_fields_at_top_level() -> None:
    payload = json.loads(JsonFormatter().format(_record(**fields(epoch=2))))
    assert payload["message"] == "hello"
    assert payload["epoch"] == 2
    assert payload["level"] == "INFO"


def test_configure_logging_replaces_handlers(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("DEBUG", "json")
    configure_logging("INFO", "json")
    root = logging.getLogger()
    assert len(root.handlers) == 1
    logging.getLogger("red_queen.x").info("ping", extra=fields(step=1))
    assert json.loads(capsys.readouterr().err.strip())["step"] == 1
