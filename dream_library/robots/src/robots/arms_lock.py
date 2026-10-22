"""One holder of the arms at a time: an exclusive `flock` on the station directory's #LOCK_NAME."""

import contextlib
import fcntl
import os
import pathlib
from collections.abc import Iterator
from typing import IO

from robots import errors

LOCK_NAME = "arms.lock"

SETUP = "dreamcli setup"
ROBOT_RUN = "dreamcli robot run"
MOTORS = "dreamcli motors"
ROBOT_DOWN = "dreamcli robot down"
STATION = "the station agent"
HOW_TO_STOP = {
    SETUP: "press Finish on the setup page, or Ctrl-C its terminal",
    ROBOT_RUN: "Ctrl-C the robot run",
    MOTORS: "let it finish, or Ctrl-C it",
    ROBOT_DOWN: "let it finish",
    STATION: "'dreamcli robot down' stops it",
}


def acquire(path: pathlib.Path, holder: str) -> IO[str]:
    """Takes the lock for `holder` and writes `<pid> <holder>` into the file; closing the file releases it.

    @param path The lock file.
    @param holder One of #HOW_TO_STOP's keys.
    @return The open lock file.
    """
    handle = _lock(path)
    handle.seek(0)
    handle.truncate()
    handle.write(f"{os.getpid()} {holder}\n")
    handle.flush()
    return handle


def _lock(path: pathlib.Path) -> IO[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    # "a+": opening never empties the file, so a refused opener cannot wipe out the holder's name.
    handle = open(path, "a+", encoding="utf-8")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.seek(0)
        found = handle.read()
        handle.close()
        raise errors.StationError(holder_message(found)) from None
    return handle


@contextlib.contextmanager
def holding(path: pathlib.Path | None, holder: str) -> Iterator[None]:
    """acquire() for the length of a `with` block; a None path takes no lock (a caller started without one).

    @param path The lock file, or None.
    @param holder One of #HOW_TO_STOP's keys.
    @return A context manager that holds the lock for its block.
    """
    if path is None:
        yield
        return
    with acquire(path, holder):
        yield


def held_by(path: pathlib.Path) -> str | None:
    """Whether another process holds the lock, without keeping it: the refusal line when it does.

    @param path The lock file.
    @return holder_message() for the holder, or None when the lock is free.
    """
    try:
        _lock(path).close()
    except errors.StationError as exc:
        return str(exc)
    return None


def holder_message(found: str) -> str:
    """The refusal line for a held lock: who holds the arms and how to stop it.

    @param found The lock file's text, `<pid> <holder>`.
    @return The line.
    """
    pid, _, holder = found.strip().partition(" ")
    how = HOW_TO_STOP.get(holder)
    if not pid.isdigit() or how is None:
        return "the arms are held by another dreamcli command on this computer: stop it, then run this again"
    return f"the arms are held by {holder} (pid {pid}): {how}, then run this again"
