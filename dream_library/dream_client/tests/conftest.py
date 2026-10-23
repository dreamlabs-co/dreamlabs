"""Shared test infrastructure: an in-process fake of DreamAPI's `/v1/get_action`, fixtures, guards.

`FakeDreamAPI` is a threaded HTTP server that speaks the DreamAPI contract (the gateway's request
and reply models, and the status envelope of `shared.contracts.envelope`) without importing the
DreamAPI product: the wire is reimplemented here, so the tests are an independent check of what the
client sends and accepts. Knobs cover latency, held replies, raw reply overrides, injected
refusals and session loss.
"""

from __future__ import annotations

import dataclasses
import gc
import http.server
import json
import os
import pathlib
import select
import socket
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest

# The client sets up on-by-default file logging when `dream` is imported (dream.logs). This runs
# before any test module imports it, so point the log at a scratch directory: the test run must
# never write to the developer's real ~/.dream/logs. A test that needs its own path sets it too.
os.environ.setdefault("DREAM_LOG_DIR", os.path.join(tempfile.gettempdir(), "dream-tests", "logs"))

## The checkpoint every test routes to; opaque to the client.
CHECKPOINT = "s3://ckpt/test-model"

## Rows per action chunk the fake returns.
HORIZON = 4

## The `timing` block every successful default reply carries, as the gateway sends it.
TIMING: dict[str, int] = {"serve_infer_ms": 10, "gateway_ms": 3}

## The embodiment the fixtures declare: the serve's own key names, grouped for the client's checks.
EMBODIMENT: dict[str, Any] = {
    "state": {
        "arm_r.joint_pos": {"shape": [6], "dtype": "float32"},
        "arm_r.gripper_pos": {"shape": [1], "dtype": "float32"},
        "arm_l.joint_pos": {"shape": [6], "dtype": "float32"},
        "arm_l.gripper_pos": {"shape": [1], "dtype": "float32"},
    },
    "image_window": {"top": {}, "left": {}, "right": {}},
    "action": {
        "q_r_chunk": {"shape": [HORIZON, 7], "dtype": "float32"},
        "q_l_chunk": {"shape": [HORIZON, 7], "dtype": "float32"},
    },
}

## Members the GR00T client had and this one must not: lifecycle, hooks, extra endpoints.
REMOVED_MEMBERS = [
    "kill_server",
    "ping",
    "call_endpoint",
    "reset",
    "get_modality_config",
    "check_observation",
    "check_action",
    "_get_action",
]

## The status envelope names, by code, as DreamAPI derives them.
STATUS_NAMES = {
    "DA-0000": "OK",
    "DA-0001": "OK_NEW_SESSION",
    "DA-1000": "UNAUTHORIZED",
    "DA-2000": "NO_CAPACITY",
    "DA-4000": "BAD_REQUEST",
    "DA-5000": "SERVE_FAULT",
    "DA-6000": "GATEWAY_FAULT",
}


@dataclasses.dataclass
class LoggedRequest:
    """One request as the fake received it.

    Attributes:
    - body: The decoded JSON body.
    - raw: The body bytes exactly as sent.
    - headers: Request headers, lower-cased names.
    - ordinal: Zero-based position in the server's request log.
    """

    body: dict[str, Any]
    raw: bytes
    headers: dict[str, str]
    ordinal: int


def envelope(code: str, reason: str | None = None, **fields: Any) -> dict[str, Any]:
    """Builds a status envelope the way DreamAPI serialises one.

    @param code A `DA-xxxx` code.
    @param reason The code's reason, or None for `DA-0000`.
    @param fields `retry_after_s`, `detail`, `retryable` overrides.
    @return The envelope dict.
    """
    retryable = code in ("DA-5000", "DA-6000") or (code == "DA-2000" and reason != "max_serves")
    status = {
        "code": code,
        "name": STATUS_NAMES.get(code, code),
        "reason": reason,
        "retryable": retryable,
        "retry_after_s": None,
        "trace_id": uuid.uuid4().hex[:16],
        "detail": None,
    }
    status.update(fields)
    return status


def wait_until(pred: Callable[[], bool], timeout_s: float = 2.0, interval_s: float = 0.005) -> None:
    """Polls `pred` until it is true; fails the test if the deadline passes.

    @param pred Zero-argument condition, re-evaluated every `interval_s`.
    @param timeout_s Deadline, seconds.
    @param interval_s Poll interval, seconds.
    """
    deadline = time.monotonic() + timeout_s
    while not pred():
        if time.monotonic() >= deadline:
            pytest.fail(f"condition not met within {timeout_s} s")
        time.sleep(interval_s)


class FakeDreamAPI:
    """A threaded HTTP server speaking DreamAPI's `/v1/get_action` contract.

    Sessions: a request with `session_id: null` mints a uuid4 and answers `DA-0001 new`; a known
    id is echoed with `DA-0000`; after `restart()` a formerly known id is answered with a fresh
    session and `DA-0001 unknown`, as the gateway does after losing its map.

    Attributes:
    - host: Always `127.0.0.1`.
    - port: The bound port.
    - requests: Every `/v1/get_action` request, in order.
    - ended: Every `/v1/end_session` request, in order; each frees its session and is answered `DA-0000`.
    - other_paths: Every request to any other path, as `(method, path)`.
    - latency_s: Sleep before every reply.
    - hold_next: Number of upcoming requests to hold without replying (the client times out).
    - reply_override: `(http_status, raw_bytes)` sent once, verbatim, instead of a real reply.
    - handler: Replaces the default reply: `(body) -> (http_status, reply_dict)`.
    - horizon: Rows per action chunk in the default reply.
    - sessions: Live session ids.
    - log_requests: Whether `requests` keeps every request (off for long runs that watch memory).
    - request_count: Requests seen on `/v1/get_action`, logged or not.
    """

    def __init__(self, horizon: int = HORIZON, port: int = 0) -> None:
        """Starts the listener on `port` (0 picks a free one).

        @param horizon Rows per action chunk in the default reply.
        @param port Port to bind; a fixed one lets a test restart the gateway on the same address.
        """
        self.host = "127.0.0.1"
        self.requests: list[LoggedRequest] = []
        self.ended: list[LoggedRequest] = []
        self.other_paths: list[tuple[str, str]] = []
        self.latency_s = 0.0
        self.hold_next = 0
        self.reply_override: tuple[int, bytes] | None = None
        self.handler: Callable[[dict[str, Any]], tuple[int, dict[str, Any]]] | None = None
        self.horizon = horizon
        self.sessions: set[str] = set()
        self.log_requests = True
        self.request_count = 0
        self._injected: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._closed = threading.Event()
        # Accepted connections, so close() can cut them the way a restarted gateway would.
        self._connections: set[socket.socket] = set()
        fake = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            disable_nagle_algorithm = True

            def setup(self) -> None:
                super().setup()
                with fake._lock:
                    fake._connections.add(self.connection)

            def finish(self) -> None:
                with fake._lock:
                    fake._connections.discard(self.connection)
                super().finish()

            def do_POST(self) -> None:  # noqa: N802  # http.server API
                fake._serve(self)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002  # http.server API
                pass

        class Server(http.server.ThreadingHTTPServer):
            daemon_threads = True
            allow_reuse_address = True

        self._server = Server((self.host, port), Handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, name=f"fake-dreamapi-{self.port}")
        self._thread.daemon = True
        self._thread.start()

    def restart(self) -> None:
        """Forgets every session, as a restarted gateway would; the listener stays up."""
        with self._lock:
            self.sessions.clear()

    def fail_next(self, code: str, reason: str | None, **fields: Any) -> None:
        """Queues one refusal envelope for the next request; `actions` is omitted, per the contract.

        @param code A `DA-xxxx` code.
        @param reason The code's reason.
        @param fields `retry_after_s` / `detail` for the envelope.
        """
        with self._lock:
            self._injected.append(envelope(code, reason, **fields))

    def close(self) -> None:
        """Stops the listener and joins the thread; safe to call twice."""
        if self._closed.is_set():
            return
        self._closed.set()
        self._server.shutdown()
        self._server.server_close()
        with self._lock:
            connections = list(self._connections)
        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self._thread.join(timeout=5.0)

    def default_reply(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """The contract's happy path: mint or echo a session and return a zero action chunk.

        @param body The decoded request body.
        @return `(http_status, reply)`.
        """
        sent = body.get("session_id")
        with self._lock:
            if self._injected:
                status = self._injected.pop(0)
                return _http_status(status["code"], status.get("reason")), {
                    "session_id": sent or "",
                    "status": status,
                    "actions": None,
                }
            if isinstance(sent, str) and sent in self.sessions:
                session_id, status = sent, envelope("DA-0000")
            else:
                session_id = str(uuid.uuid4())
                self.sessions.add(session_id)
                status = envelope("DA-0001", "new" if sent is None else "unknown")
        rows = self.horizon
        actions = {
            "q_r": [0.0] * 7,
            "q_l": [0.1] * 7,
            "q_r_chunk": [[0.0] * 7 for _ in range(rows)],
            "q_l_chunk": [[0.1] * 7 for _ in range(rows)],
            "per_key": {"action.right_arm": [[0.0] * 6 for _ in range(rows)]},
        }
        return 200, {"session_id": session_id, "status": status, "actions": actions, "timing": dict(TIMING)}

    def _serve(self, handler: http.server.BaseHTTPRequestHandler) -> None:
        """Handles one POST: logs it, applies the knobs, writes the reply."""
        length = int(handler.headers.get("Content-Length") or 0)
        raw = handler.rfile.read(length)
        if handler.path == "/v1/end_session":
            body = json.loads(raw)
            headers = {k.lower(): v for k, v in handler.headers.items()}
            with self._lock:
                self.ended.append(LoggedRequest(body=body, raw=raw, headers=headers, ordinal=len(self.ended)))
                self.sessions.discard(str(body.get("session_id")))
            reply = {"session_id": body.get("session_id"), "status": envelope("DA-0000")}
            _write(handler, 200, json.dumps(reply).encode())
            return
        if handler.path != "/v1/get_action":
            self.other_paths.append(("POST", handler.path))
            _write(handler, 404, json.dumps({"session_id": "", "status": envelope("DA-4000", "unknown_op")}).encode())
            return
        body = json.loads(raw)
        with self._lock:
            self.request_count += 1
            if self.log_requests:
                headers = {k.lower(): v for k, v in handler.headers.items()}
                self.requests.append(LoggedRequest(body=body, raw=raw, headers=headers, ordinal=len(self.requests)))
            hold = self.hold_next > 0
            if hold:
                self.hold_next -= 1
            override = self.reply_override
            self.reply_override = None
        if hold:
            # Hold the connection open without answering, as a wedged gateway would, until the
            # client gives up (its timeout) or the fake is closed; then let the socket go.
            self._hold(handler.connection)
            return
        if self.latency_s:
            time.sleep(self.latency_s)
        if override is not None:
            _write(handler, override[0], override[1])
            return
        status, reply = (self.handler or self.default_reply)(body)
        _write(handler, status, json.dumps(reply).encode("utf-8"))

    def _hold(self, connection: socket.socket) -> None:
        """Waits until the peer hangs up or the fake closes; a readable socket with no data is the hang-up."""
        while not self._closed.is_set():
            readable, _, _ = select.select([connection], [], [], 0.05)
            if readable:
                try:
                    if connection.recv(1, socket.MSG_PEEK) == b"":
                        return
                except OSError:
                    return


def _write(handler: http.server.BaseHTTPRequestHandler, status: int, payload: bytes) -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


def _http_status(code: str, reason: str | None) -> int:
    """DreamAPI's HTTP status for an envelope (`http_status_for` in the gateway)."""
    if code in ("DA-0000", "DA-0001"):
        return 200
    if code == "DA-1000":
        return 403 if reason in ("disabled", "not_entitled") else 401
    if code == "DA-2000":
        return 429 if reason == "rate_limited" else 503
    if code == "DA-4000":
        return 413 if reason == "too_large" else 400
    if code == "DA-5000":
        return 502
    return 500


@pytest.fixture(autouse=True)
def _no_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """Clears the four credential variables and points the file at one that does not exist, so a developer's own
    settings and their real `~/.dreamlabs/credentials` never move the defaults a test pins."""
    for name in ("DREAM_GATEWAY_URL", "DREAM_API_URL", "DREAM_DATA_URL", "DREAM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DREAM_CREDENTIALS", str(tmp_path / "no-credentials"))


@pytest.fixture(autouse=True)
def _no_recording(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keeps `dream.DreamClient`'s recording off outside `test_recorder.py`, so no test dials a platform it names
    (`https://platform.example`) in the background; the recorder's own tests run it against a relay on localhost."""
    if request.node.path.name == "test_recorder.py":
        return
    from dream import facade

    monkeypatch.setattr(facade.DreamClient, "_new_recording", lambda self, fps: (None, "recording is off in tests"))


PLATFORM_KEY = "dl_live_test"


class FakePlatform:
    """The platform's `GET /v1/models` in one threaded HTTP server; `url` is its base, `catalog` what it answers and
    `agents` each request's `User-Agent` (None when it had none)."""

    def __init__(self, catalog: list[dict[str, Any]]) -> None:
        fake = self
        self.catalog = catalog
        self.agents: list[str | None] = []

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:  # noqa: N802  # http.server API
                fake.handle(self)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002  # http.server API
                pass

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def handle(self, request: http.server.BaseHTTPRequestHandler) -> None:
        self.agents.append(request.headers.get("User-Agent"))
        if request.headers.get("Authorization") != f"Bearer {PLATFORM_KEY}":
            _platform_reply(
                request, 401, {"error": {"code": "invalid_api_key", "message": "This api key is not valid."}}
            )
        elif request.path == "/v1/models":
            _platform_reply(request, 200, self.catalog)
        else:
            _platform_reply(request, 404, {"error": {"code": "not_found", "message": f"No {request.path}."}})


def _platform_reply(request: http.server.BaseHTTPRequestHandler, status: int, payload: Any) -> None:
    body = json.dumps(payload).encode()
    request.send_response(status)
    request.send_header("Content-Length", str(len(body)))
    request.end_headers()
    request.wfile.write(body)


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeDreamAPI]:
    """A started fake gateway, closed on teardown; `DREAM_GATEWAY_URL` names it, so `dream.DreamClient` reaches it."""
    fake = FakeDreamAPI()
    monkeypatch.setenv("DREAM_GATEWAY_URL", f"http://{fake.host}:{fake.port}")
    yield fake
    fake.close()


@pytest.fixture
def config(server: FakeDreamAPI) -> dict[str, Any]:
    """A client config against `server`: connection keys, the checkpoint and the embodiment.

    `scheme` is `http` because the fake is a plain in-process server; the library's default is now
    `https` for the hosted gateway, so the local test double must opt back down to plain HTTP.
    """
    return {
        "host": server.host,
        "port": server.port,
        "scheme": "http",
        "timeout_ms": 2000,
        "checkpoint": CHECKPOINT,
        "embodiment": json.loads(json.dumps(EMBODIMENT)),
    }


@pytest.fixture
def observation() -> dict[str, Any]:
    """An observation matching `EMBODIMENT`: float32 state arrays, one-frame JPEG windows, a prompt."""
    return make_observation()


## A base64 JPEG header, enough for a one-frame window on the wire.
FRAME_B64 = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwg"
    "JC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/"
)


def make_observation(frame_b64: str = FRAME_B64) -> dict[str, Any]:
    """Builds an observation in the serve's format.

    @param frame_b64 The base64 JPEG each camera window carries.
    @return The observation dict.
    """
    return {
        "state.arm_r.joint_pos": np.zeros(6, dtype=np.float32),
        "state.arm_r.gripper_pos": np.ones(1, dtype=np.float32),
        "state.arm_l.joint_pos": np.zeros(6, dtype=np.float32),
        "state.arm_l.gripper_pos": np.ones(1, dtype=np.float32),
        "image_window.top": [frame_b64],
        "image_window.left": [frame_b64],
        "image_window.right": [frame_b64],
        "prompt": "pick up the cup",
    }


@pytest.fixture
def blackhole_port() -> int:
    """A loopback port nothing listens on: bound then released, so a connect is refused at once."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.fixture
def fd_guard() -> Iterator[None]:
    """Fails the test if the process holds more file descriptors after it than before."""
    gc.collect()
    before = _open_fd_count()
    yield
    gc.collect()
    # The fake's handler threads close their sockets a few milliseconds after the client hangs up.
    deadline = time.monotonic() + 1.0
    while _open_fd_count() > before and time.monotonic() < deadline:
        time.sleep(0.01)
    after = _open_fd_count()
    assert after <= before, f"file descriptors grew from {before} to {after}"


def _open_fd_count() -> int:
    return len(os.listdir("/proc/self/fd"))
