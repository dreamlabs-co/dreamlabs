"""Diagnostic logging for the `dream` client: where the library's own log records go.

This is the client's operational log, the session boundaries #dream.client.PolicyClient records
and the odd warning, not robot data: observations, actions and the api key never reach it (a test
pins that the key appears in no record). A client library normally leaves handlers to the
application and stays silent behind a `NullHandler`; during the beta the `dream` client also
writes a size-capped rotating file under `~/.dream/logs/` by default, so a customer who hits a
problem already has a log to send. configure() wires this up once, when `dream` is imported.

Whether the file is on by default is one constant, DEFAULT_FILE_LOGGING, so the later move to
opt-out is a one-line change; the environment overrides it either way:

- `DREAM_LOG`: the level (`debug`, `info`, `warning`, `error`, `critical`), or a disabling value
  (`off`, `none`, `0`, `false`, `no`) that turns the file off. Unset means the default level and
  the default on/off.
- `DREAM_LOG_DIR`: the directory the file is written in; unset means `~/.dream/logs`.
"""

import logging
import logging.handlers
import os
import time
from collections.abc import Mapping
from pathlib import Path

## The package logger every module under `dream` logs through; one handler here catches them all.
LOGGER_NAME = "dream"

## Beta posture: the file is written unless `DREAM_LOG` disables it. Flip to False to make the file
## opt-in (a customer then turns it on with `DREAM_LOG=<level>`); the environment override still wins.
DEFAULT_FILE_LOGGING = True

## Level used when `DREAM_LOG` names none.
DEFAULT_LEVEL = logging.INFO

## The log file. It is rotated so it cannot grow without bound: the live file plus this many backups,
## each at most this many bytes, so the log costs at most (_BACKUP_COUNT + 1) x _MAX_BYTES on disk.
_FILE_NAME = "dream.log"
_MAX_BYTES = 5 * 1024 * 1024
_BACKUP_COUNT = 3

## `DREAM_LOG` values that turn the file off rather than name a level.
_DISABLING = frozenset({"", "0", "off", "no", "none", "false", "disabled"})

## `DREAM_LOG` level names mapped to their `logging` levels.
_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

## Attribute that marks the handlers configure() installs, so it is idempotent across repeated imports.
_MANAGED = "_dream_managed"


def configure() -> None:
    """Installs the `dream` logger's handlers once; safe to call more than once.

    A `NullHandler` is always attached, so the library is silent when the file is off and nothing
    else is configured. When file logging is on (DEFAULT_FILE_LOGGING and `DREAM_LOG` is not a
    disabling value), a rotating file handler is added at the resolved path and the logger is
    lowered to the resolved level so its records reach the file. A file that cannot be opened (an
    unwritable home, the path taken by a directory) leaves only the `NullHandler`: logging setup
    must never break `import dream`. A level already set on the logger is left alone, so an
    application that configured `dream` before importing it keeps its choice.
    """
    logger = logging.getLogger(LOGGER_NAME)
    if any(getattr(handler, _MANAGED, False) for handler in logger.handlers):
        return
    null = logging.NullHandler()
    setattr(null, _MANAGED, True)
    logger.addHandler(null)
    level, file_on = _resolve(os.environ)
    if not file_on:
        return
    handler = _file_handler(_log_dir(os.environ), level)
    if handler is None:
        return
    setattr(handler, _MANAGED, True)
    logger.addHandler(handler)
    if logger.level == logging.NOTSET:
        logger.setLevel(level)


def _resolve(env: Mapping[str, str]) -> tuple[int, bool]:
    """Reads `DREAM_LOG` into the level to log at and whether to write the file.

    @param env The environment to read; `os.environ` in production, a plain dict in tests.
    @return `(level, file_enabled)`. An unset or unrecognised `DREAM_LOG` gives the default level
        and DEFAULT_FILE_LOGGING; a disabling value gives `(default level, False)`; a level name
        gives `(that level, True)`.
    """
    raw = env.get("DREAM_LOG")
    if raw is None:
        return DEFAULT_LEVEL, DEFAULT_FILE_LOGGING
    value = raw.strip().lower()
    if value in _DISABLING:
        return DEFAULT_LEVEL, False
    if value in _LEVELS:
        return _LEVELS[value], True
    return DEFAULT_LEVEL, DEFAULT_FILE_LOGGING


def _log_dir(env: Mapping[str, str]) -> Path:
    """Returns the directory the file is written in: `DREAM_LOG_DIR` if set, else `~/.dream/logs`.

    @param env The environment to read.
    @return The directory path with `~` expanded; _file_handler() creates it, not this function.
    """
    override = env.get("DREAM_LOG_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".dream" / "logs"


def _file_handler(directory: Path, level: int) -> logging.Handler | None:
    """Builds the rotating file handler under `directory`, or None when the file cannot be opened.

    @param directory Where `dream.log` is written; created with its parents if missing.
    @param level The level the handler logs at.
    @return A configured `RotatingFileHandler`, or None when the directory or file is unwritable, or
        the home directory cannot be located, so the caller falls back to silence.
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            directory / _FILE_NAME,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
    except (OSError, RuntimeError):
        return None
    handler.setLevel(level)
    formatter = logging.Formatter("%(asctime)s.%(msecs)03dZ %(levelname)s %(name)s %(message)s", "%Y-%m-%dT%H:%M:%S")
    # UTC, so a log shipped from any machine sorts and reads against the gateway's own clock.
    formatter.converter = time.gmtime
    handler.setFormatter(formatter)
    return handler
