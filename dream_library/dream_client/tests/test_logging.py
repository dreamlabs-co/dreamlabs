"""Diagnostic file logging: the on-by-default `~/.dream/logs/dream.log` and its environment knobs.

The client sets up its own logging when `dream` is imported (`dream.logs`): a rotating file, on
by default for the beta, that `DREAM_LOG` tunes or turns off and `DREAM_LOG_DIR` relocates. These
tests pin the resolution of those variables, the file handler, and the fact that importing the
package wires the `dream` logger up. `conftest.py` points `DREAM_LOG_DIR` at a scratch directory
for the whole run, so nothing here touches the developer's real home.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

import pytest

# Importing a submodule runs `dream/__init__.py`, which calls `logs.configure()`; that import is
# what the wiring test below checks the effect of, so no separate `import dream` is needed.
from dream import logs


def test_resolve_defaults_to_info_and_file_on() -> None:
    """With `DREAM_LOG` unset the file is written at the default level."""
    assert logs._resolve({}) == (logging.INFO, True)
    assert logs._resolve({"UNRELATED": "x"}) == (logging.INFO, True)


@pytest.mark.parametrize("value", ["off", "0", "none", "false", "no", "disabled", ""])
def test_resolve_disabling_values_turn_the_file_off(value: str) -> None:
    """A disabling `DREAM_LOG` yields file-off; the level stays the default (unused when off)."""
    assert logs._resolve({"DREAM_LOG": value}) == (logging.INFO, False)


@pytest.mark.parametrize(
    ("value", "level"),
    [("debug", logging.DEBUG), ("info", logging.INFO), ("WARNING", logging.WARNING), (" Error ", logging.ERROR)],
)
def test_resolve_named_levels_are_case_and_space_insensitive(value: str, level: int) -> None:
    """A recognised level name (any case, surrounding space allowed) turns the file on at that level."""
    assert logs._resolve({"DREAM_LOG": value}) == (level, True)


def test_resolve_unknown_level_keeps_the_default() -> None:
    """An unrecognised `DREAM_LOG` is not a reason to crash or go silent: default level, file on."""
    assert logs._resolve({"DREAM_LOG": "loud"}) == (logging.INFO, True)


def test_log_dir_prefers_the_override_and_expands_home() -> None:
    """`DREAM_LOG_DIR` wins over the default and `~` is expanded; the default is `~/.dream/logs`."""
    assert logs._log_dir({"DREAM_LOG_DIR": "/var/tmp/x"}) == Path("/var/tmp/x")
    assert logs._log_dir({"DREAM_LOG_DIR": "~/z"}) == Path.home() / "z"
    assert logs._log_dir({}) == Path.home() / ".dream" / "logs"


def test_file_handler_writes_a_record_under_the_directory(tmp_path: Path) -> None:
    """The handler creates the directory and writes `dream.log` there, formatted with the message."""
    handler = logs._file_handler(tmp_path / "logs", logging.INFO)
    assert handler is not None
    try:
        record = logging.LogRecord("dream.client", logging.INFO, __file__, 0, "session s-1 started", None, None)
        handler.emit(record)
    finally:
        handler.close()
    written = (tmp_path / "logs" / "dream.log").read_text(encoding="utf-8")
    assert "session s-1 started" in written
    assert "INFO" in written and "dream.client" in written


def test_file_handler_returns_none_when_the_path_is_unwritable(tmp_path: Path) -> None:
    """A directory that cannot be made (a file sits where a parent should be) yields None, not a raise."""
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    assert logs._file_handler(blocker / "sub", logging.INFO) is None


def test_import_configures_the_dream_logger_with_a_managed_file_handler() -> None:
    """Importing `dream` leaves the `dream` logger with a managed rotating file handler at INFO."""
    handler = _managed_file_handler()
    assert handler is not None
    assert logging.getLogger(logs.LOGGER_NAME).isEnabledFor(logging.INFO)
    marker = "test-marker-9f3a2b for the import wiring check"
    logging.getLogger("dream.client").info(marker)
    written = Path(handler.baseFilename).read_text(encoding="utf-8")
    assert marker in written


def test_configure_is_idempotent() -> None:
    """A second `configure()` adds no handler: `import dream` and an explicit call cannot double up."""
    logger = logging.getLogger(logs.LOGGER_NAME)
    before = list(logger.handlers)
    logs.configure()
    assert logger.handlers == before


def _managed_file_handler() -> logging.handlers.RotatingFileHandler | None:
    """The rotating file handler `configure()` installed on the `dream` logger, if any."""
    for handler in logging.getLogger(logs.LOGGER_NAME).handlers:
        if getattr(handler, logs._MANAGED, False) and isinstance(handler, logging.handlers.RotatingFileHandler):
            return handler
    return None
