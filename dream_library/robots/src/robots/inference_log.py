"""The inference log: one JSON Lines record per `get_action` outcome, error, stall and run boundary.

The per-tick trace (`trace_path`) answers "what did the arms do"; this file answers "what did
the policy do and how long did it take", at one record per inference rather than per tick, so
it stays small enough to keep for every run and to grep across a day of runs. The loop writes
it: an `inference` record for every chunk that lands (installed or discarded), an `error`
record for every exception the tick boundary catches or that stops the loop, a `stall` record
at the start of every empty-buffer burst, and `run_start` / `run_stop` around each run. Append
mode and a flush per record, so a crash loses nothing and `tail -f` follows a run live.

Timing fields beyond `rtt_ms` come from the dream client's `last_timing` (a dict with
`rtt_ms`, `serve_infer_ms`, `gateway_ms`, `network_ms`, `request_bytes`, `response_bytes`) and
are null when it has none yet; the session id is the client's `session_id`, null before it opened one.

@see InferenceLog, error_fields(), client_timing()
"""

import datetime
import json
import logging
from typing import Any

import dream

_log = logging.getLogger(__name__)

## The dream `last_timing` keys carried into every `inference` record, in this order; null when absent.
## `up_mbps` and the `tcp_*` fields are the sender-side upload health (kernel TCP_INFO); null off Linux.
TIMING_KEYS: tuple[str, ...] = (
    "serve_infer_ms",
    "gateway_ms",
    "network_ms",
    "up_mbps",
    "tcp_cwnd_kb",
    "tcp_retrans",
    "tcp_rtt_ms",
    "tcp_app_limited",
    "request_bytes",
    "response_bytes",
)


class InferenceLog:
    """An append-only JSON Lines file, one record per line, flushed per record.

    Without a path every method is a no-op, so the loop calls it unconditionally. A write that
    fails (disk full, file removed) is logged once and disables the log: it is diagnostics, and
    must never stop the loop. A second file can be mirrored to (#set_mirror()): the run's saved episode keeps its
    own copy of the records written while it records.

    Attributes:
    - path: The file, or None for no log.
    - run_number: Stamped into every record as `run`.

    @see open(), write(), close()
    """

    def __init__(self, path: str | None, run_number: int) -> None:
        """Binds the path and the run number; nothing is opened until open().

        @param path The file to append to, or None to write nothing.
        @param run_number The run's ordinal, stamped into every record.
        """
        self.path = path
        self.run_number = run_number
        self._file: Any = None
        self._mirror: Any = None
        self._mirror_path: str | None = None

    @property
    def is_open(self) -> bool:
        """Whether the file is open for writing."""
        return self._file is not None

    def open(self) -> None:
        """Opens the file for appending; a no-op without a path or when already open.

        @throws OSError The file cannot be opened.
        """
        if self.path is None or self._file is not None:
            return
        self._file = open(self.path, "a", encoding="utf-8")  # noqa: SIM115 (closed by close())
        _log.info("inference log: appending to %s (run %d)", self.path, self.run_number)

    def set_mirror(self, path: str | None) -> None:
        """Starts copying every record to a second file, or stops when `path` is None; works without a main path.

        @param path The file to append the copy to, or None to close the current copy.
        @throws OSError The file cannot be opened; the log is unchanged.
        """
        mirror = None if path is None else open(path, "a", encoding="utf-8")  # noqa: SIM115 (closed below)
        self._close_mirror()
        self._mirror = mirror
        self._mirror_path = path

    def write(self, kind: str, **fields: Any) -> None:
        """Appends one record, `{"kind": kind, "t": now, "run": run_number, **fields}`, and flushes it.

        @param kind The record kind: `inference`, `error`, `stall`, `run_start` or `run_stop`.
        @param fields The record's other fields; anything `json.dumps` cannot encode is written as `str(value)`.
        """
        if self._file is None and self._mirror is None:
            return
        record = {"kind": kind, "t": _now_iso(), "run": self.run_number, **fields}
        line = json.dumps(record, default=str) + "\n"
        if self._file is not None:
            try:
                self._file.write(line)
                self._file.flush()
            except OSError:
                _log.exception("inference log: write to %s failed; no more records this run", self.path)
                self._close_main()
        if self._mirror is not None:
            try:
                self._mirror.write(line)
                self._mirror.flush()
            except OSError:
                _log.exception("inference log: write to %s failed; no more copies this episode", self._mirror_path)
                self._close_mirror()

    def close(self) -> None:
        """Closes the file and the mirror if they are open; safe to call twice."""
        self._close_mirror()
        self._close_main()

    def _close_main(self) -> None:
        """Closes the main file if it is open."""
        if self._file is not None:
            try:
                self._file.close()
            finally:
                self._file = None

    def _close_mirror(self) -> None:
        """Closes the mirror file if it is open."""
        if self._mirror is not None:
            try:
                self._mirror.close()
            finally:
                self._mirror = None


def error_fields(exc: BaseException, action: str) -> dict[str, Any]:
    """The fields of an `error` record for one exception.

    @param exc The exception the loop caught, or the one that stopped it.
    @param action What the loop did about it: `"retry"`, `"reset"` or `"stop"`.
    @return `error_type` (the class name), `code` and `reason` (a `DreamServerError`'s envelope, else null),
        `detail` (the message), `retry_after_s` (the envelope's hint, else null) and `action`.
    """
    envelope = exc if isinstance(exc, dream.DreamServerError) else None
    return {
        "error_type": type(exc).__name__,
        "code": None if envelope is None else envelope.code,
        "reason": None if envelope is None else envelope.reason,
        "detail": str(exc),
        "retry_after_s": None if envelope is None else envelope.retry_after_s,
        "action": action,
    }


def client_timing(client: Any) -> dict[str, Any]:
    """The dream client's `last_timing` fields for an `inference` record, null when the client has none.

    @param client The dream client; `last_timing` is read with `getattr` and may be absent or None.
    @return One entry per #TIMING_KEYS.
    """
    timing = getattr(client, "last_timing", None)
    if not isinstance(timing, dict):
        return dict.fromkeys(TIMING_KEYS)
    return {key: timing.get(key) for key in TIMING_KEYS}


def _now_iso() -> str:
    """The wall-clock time as an ISO 8601 UTC string with millisecond precision.

    @return E.g. `"2026-09-10T14:03:07.412+00:00"`.
    """
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")
