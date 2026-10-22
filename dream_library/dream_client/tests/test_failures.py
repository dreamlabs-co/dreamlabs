"""Failure-injection tests F-1..F-15, against DreamAPI's HTTP contract.

Each fault is injected in the fake gateway; the assertion is always the same shape: the
right exception, the right message, and a client that still works on the next call.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any

import pytest
from conftest import CHECKPOINT, FakeDreamAPI, envelope, make_observation

import dream
from dream.client import PolicyClient
from dream.errors import (
    DreamBadRequest,
    DreamConfigError,
    DreamProtocolError,
    DreamServerError,
    DreamSessionExpired,
    DreamTimeout,
)

## Timeout for the tests that provoke one; the band below tolerates scheduler jitter.
TIMEOUT_MS = 300


@pytest.fixture
def client(config: dict[str, Any]) -> Iterator[PolicyClient]:
    """A strict client with a short timeout, closed on teardown."""
    with dream._open_policy({**config, "timeout_ms": TIMEOUT_MS}) as opened:
        yield opened


def test_f01_no_reply_then_recovery(
    fd_guard: None, server: FakeDreamAPI, client: PolicyClient, observation: dict[str, Any]
) -> None:
    """F-1: a held reply is DreamTimeout after ~timeout_ms, and the next call succeeds; no FD growth."""
    server.hold_next = 1
    started = time.perf_counter()
    with pytest.raises(DreamTimeout, match="timed out"):
        client.get_action(observation)
    elapsed = time.perf_counter() - started
    assert 0.2 < elapsed < 1.5

    assert client.get_action(observation)["q_r_chunk"]


def test_f02_non_json_reply_names_the_gateway(
    server: FakeDreamAPI, client: PolicyClient, observation: dict[str, Any]
) -> None:
    """F-2: an HTML error page (a proxy in front of nothing) is DreamProtocolError mentioning the gateway."""
    server.reply_override = (502, b"<html><body>Bad Gateway</body></html>")
    with pytest.raises(DreamProtocolError, match="DreamAPI gateway"):
        client.get_action(observation)

    assert client.get_action(observation)["q_r_chunk"]


def test_f03_serve_fault_is_retryable_server_error(
    server: FakeDreamAPI, client: PolicyClient, observation: dict[str, Any]
) -> None:
    """F-3: `DA-5000 error` is DreamServerError with `retryable` and the serve's detail verbatim."""
    server.fail_next("DA-5000", "error", detail="inference error: oom")
    with pytest.raises(DreamServerError) as info:
        client.get_action(observation)

    assert info.value.code == "DA-5000" and info.value.retryable is True
    assert "oom" in str(info.value)


@pytest.mark.parametrize("garbage", [b"\x80\x81\x82", b"not json at all", b""])
def test_f04_garbage_bytes(
    server: FakeDreamAPI, client: PolicyClient, observation: dict[str, Any], garbage: bytes
) -> None:
    """F-4: undecodable bytes are DreamProtocolError; the client remains usable."""
    server.reply_override = (200, garbage)
    with pytest.raises(DreamProtocolError):
        client.get_action(observation)

    assert client.get_action(observation)["q_r_chunk"]


def test_f04a_json_without_envelope(server: FakeDreamAPI, client: PolicyClient, observation: dict[str, Any]) -> None:
    """F-4: valid JSON that is not a DreamAPI response (a list, a dict without `status`) is DreamProtocolError."""
    for payload in (b"[1, 2]", b'{"session_id": "s", "actions": {}}', b'"ok"'):
        server.reply_override = (200, payload)
        with pytest.raises(DreamProtocolError, match="status envelope"):
            client.get_action(observation)


def test_f05_connection_refused(
    blackhole_port: int, observation: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """F-5: `init` succeeds against a closed port; the first call is DreamProtocolError at once, not a timeout, in
    words a customer can act on: no error number, and where the URL was set."""
    client = dream._open_policy(
        {"host": "127.0.0.1", "port": blackhole_port, "checkpoint": CHECKPOINT, "timeout_ms": 5000}
    )
    with client:
        started = time.perf_counter()
        with pytest.raises(DreamProtocolError) as info:
            client.get_action(observation)
        assert time.perf_counter() - started < 1.0
    assert str(info.value) == (
        f"could not reach Dream Labs at https://127.0.0.1:{blackhole_port}: Connection refused; check this computer's "
        "internet connection, or the URL passed to the client"
    )

    monkeypatch.setenv("DREAM_GATEWAY_URL", f"http://127.0.0.1:{blackhole_port}")
    with dream._open_policy({"checkpoint": CHECKPOINT, "timeout_ms": 5000}) as client:
        with pytest.raises(DreamProtocolError, match="or DREAM_GATEWAY_URL in the environment$"):
            client.get_action(observation)


def test_f06_server_dies_mid_request(config: dict[str, Any], observation: dict[str, Any]) -> None:
    """F-6: the gateway going away while a request is held ends the call with an error, never a hang."""
    fake = FakeDreamAPI()
    try:
        fake.hold_next = 1
        with dream._open_policy({**config, "port": fake.port, "timeout_ms": 2000}) as client:
            import threading

            outcome: list[BaseException] = []

            def call() -> None:
                try:
                    client.get_action(observation)
                except BaseException as exc:  # noqa: B036  # thread boundary: the outcome is recorded
                    outcome.append(exc)

            worker = threading.Thread(target=call)
            worker.start()
            time.sleep(0.2)
            fake.close()
            worker.join(timeout=5.0)
            assert not worker.is_alive(), "get_action hung after the gateway died"
            assert outcome and isinstance(outcome[0], (DreamTimeout, DreamProtocolError))
    finally:
        fake.close()


def test_f07_server_restarted_between_calls(config: dict[str, Any], observation: dict[str, Any]) -> None:
    """F-7: a new gateway on the same port is reached without reconstructing the client."""
    first = FakeDreamAPI()
    port = first.port
    with dream._open_policy({**config, "port": port}) as client:
        client.get_action(observation)
        first.close()
        second = _rebind(port)
        try:
            with pytest.raises(DreamSessionExpired):
                client.get_action(observation)
            assert client.get_action(observation)["q_r_chunk"]
            assert len(second.requests) == 2
        finally:
            second.close()


def test_f08_late_reply_is_never_delivered_to_a_later_call(
    server: FakeDreamAPI, client: PolicyClient, observation: dict[str, Any]
) -> None:
    """F-8: after a timeout, the next call gets its own reply, never the stale one."""
    marker_seen: list[Any] = []

    def handler(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        marker = body["observation"]["prompt"]
        if marker == "slow":
            time.sleep(TIMEOUT_MS / 1000 * 2)
        status, reply = server.default_reply(body)
        reply["actions"]["marker"] = marker
        marker_seen.append(marker)
        return status, reply

    server.handler = handler
    with pytest.raises(DreamTimeout):
        client.get_action({**observation, "prompt": "slow"})
    action = client.get_action({**observation, "prompt": "fast"})

    assert action["marker"] == "fast"


def test_f09_truncated_reply(server: FakeDreamAPI, client: PolicyClient, observation: dict[str, Any]) -> None:
    """F-9: a reply cut mid-JSON is DreamProtocolError, not a partially decoded action."""
    server.reply_override = (200, b'{"session_id": "s", "status": {"code": "DA-0000"}, "actions": {"q_r_chunk": [[0')
    with pytest.raises(DreamProtocolError):
        client.get_action(observation)

    assert client.get_action(observation)["q_r_chunk"]


def test_f10_lost_session_is_reported_once_and_adopted(
    server: FakeDreamAPI, client: PolicyClient, observation: dict[str, Any]
) -> None:
    """F-10: a session the gateway forgot comes back as `DA-0001 unknown`: DreamSessionExpired once, then on."""
    client.get_action(observation)
    old = client._session_id
    server.restart()

    with pytest.raises(DreamSessionExpired, match="restarted server-side"):
        client.get_action(observation)
    client.get_action(observation)
    client.get_action(observation)

    ids = [request.body["session_id"] for request in server.requests]
    assert ids[1] == old and ids[2] != old and ids[3] == ids[2]


def test_f11_two_losses_in_a_row_are_two_reports(
    server: FakeDreamAPI, client: PolicyClient, observation: dict[str, Any]
) -> None:
    """F-11: every mid-episode replacement is reported; nothing loops."""
    client.get_action(observation)
    for _ in range(2):
        server.restart()
        with pytest.raises(DreamSessionExpired):
            client.get_action(observation)

    assert len(server.requests) == 3
    assert client.get_action(observation)["q_r_chunk"]


def test_f12_bad_request_after_a_lost_session(
    server: FakeDreamAPI, client: PolicyClient, observation: dict[str, Any]
) -> None:
    """F-12: a `DA-4000` after a session loss surfaces as DreamConfigError; the loss is not what the user is told."""
    client.get_action(observation)
    server.restart()
    server.fail_next("DA-4000", "malformed", detail="checkpoint: field required")

    with pytest.raises(DreamConfigError, match="checkpoint: field required"):
        client.get_action(observation)


def test_f13_one_report_per_restart_under_load(
    server: FakeDreamAPI, client: PolicyClient, observation: dict[str, Any]
) -> None:
    """F-13: 200 calls with a restart every 50 raise exactly one DreamSessionExpired per restart."""
    reports = 0
    for i in range(200):
        if i and i % 50 == 0:
            server.restart()
        try:
            client.get_action(observation)
        except DreamSessionExpired:
            reports += 1

    assert reports == 3
    assert len(server.requests) == 200


def test_f14_reply_without_session_id_keeps_the_current_one(
    server: FakeDreamAPI, client: PolicyClient, observation: dict[str, Any]
) -> None:
    """F-14: a reply that omits `session_id` neither raises nor resets the id to None."""
    client.get_action(observation)
    held = client._session_id

    def handler(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        return 200, {"status": envelope("DA-0000"), "actions": server.default_reply(body)[1]["actions"]}

    server.handler = handler
    client.get_action(observation)
    server.handler = None
    client.get_action(observation)

    assert server.requests[-1].body["session_id"] == held


def test_f15_error_wins_over_actions(server: FakeDreamAPI, client: PolicyClient, observation: dict[str, Any]) -> None:
    """F-15: a refusal that carries an `actions` field is still a refusal; nothing is returned."""

    def handler(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        return 503, {"session_id": "s", "status": envelope("DA-2000", "pool_full"), "actions": {"q_r_chunk": []}}

    server.handler = handler
    with pytest.raises(DreamServerError, match="DA-2000"):
        client.get_action(make_observation())


def _rebind(port: int) -> FakeDreamAPI:
    """A fresh fake on a given port, retrying briefly while the old socket's TIME_WAIT clears."""
    deadline = time.monotonic() + 5.0
    while True:
        try:
            return FakeDreamAPI(port=port)
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.05)


def test_f16_bad_request_carries_the_envelope_as_a_config_error(
    server: FakeDreamAPI, client: PolicyClient, observation: dict[str, Any]
) -> None:
    """`DA-4000 unknown_checkpoint` is a `DreamBadRequest`: a `DreamConfigError` for every existing `except`, and a
    `DreamServerError` whose `reason` says which bad request it was, with DreamAPI's detail in the message."""
    server.fail_next("DA-4000", "unknown_checkpoint", detail="no such checkpoint: dreamlabs/dream-2")
    with pytest.raises(DreamConfigError) as info:
        client.get_action(observation)

    assert isinstance(info.value, DreamBadRequest) and isinstance(info.value, DreamServerError)
    assert (info.value.code, info.value.reason, info.value.retryable) == ("DA-4000", "unknown_checkpoint", False)
    assert "dreamlabs/dream-2" in str(info.value) and "BAD_REQUEST" in str(info.value)
    assert str(info.value).endswith(
        f"; checkpoint {CHECKPOINT!r} is not a model your key may run: use an id from the Models page or "
        "DreamClient().get_models() (GET /v1/models)"
    )
    assert isinstance(info.value, ValueError) and isinstance(info.value, RuntimeError)


def _raw_gateway(second: str) -> tuple[Any, list[bytes]]:
    """A one-connection gateway on a raw socket: it answers the first request in full on a kept-alive connection,
    then treats the next as `second` says. `"idle"` closes the connection without reading it (a gateway that dropped
    an idle keep-alive, then answers in full on a new one); `"mid_reply"` reads the request, sends the status line and
    a cut body, and resets the connection. Returns the listening socket and the request bodies it read."""
    import socket
    import struct
    import threading

    listener = socket.create_server(("127.0.0.1", 0))
    bodies: list[bytes] = []
    reply = json.dumps({"session_id": "s", "status": envelope("DA-0000"), "actions": {"q_r_chunk": [[0.0]]}}).encode()
    ok = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n%s" % (len(reply), reply)

    def read_request(conn: socket.socket) -> bytes:
        data = b""
        while b"\r\n\r\n" not in data:
            data += conn.recv(65536)
        head, _, rest = data.partition(b"\r\n\r\n")
        length = int(next(h.split(b":")[1] for h in head.split(b"\r\n") if h.lower().startswith(b"content-length")))
        while len(rest) < length:
            rest += conn.recv(65536)
        bodies.append(rest)
        return rest

    def serve() -> None:
        conn, _ = listener.accept()
        read_request(conn)
        conn.sendall(ok)
        if second == "idle":
            conn.close()
            conn, _ = listener.accept()
            read_request(conn)
            conn.sendall(ok)
            conn.close()
            return
        read_request(conn)
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 4096\r\n\r\n{")
        time.sleep(0.05)
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))  # close with a reset
        conn.close()
        try:
            listener.settimeout(1.0)
            again, _ = listener.accept()  # a re-sent request would land here
            read_request(again)
            again.sendall(ok)
            again.close()
        except OSError:
            pass

    threading.Thread(target=serve, daemon=True).start()
    return listener, bodies


def test_f17_a_reply_cut_after_its_status_line_is_not_sent_again() -> None:
    """F-17: the gateway started answering, so it ran the request; a reset while the body is read is
    DreamProtocolError, and the request is not sent a second time (it would be inferred and billed twice)."""
    from dream.transport import Transport

    listener, bodies = _raw_gateway("mid_reply")
    transport = Transport("127.0.0.1", listener.getsockname()[1], 2000)
    try:
        transport.request({"n": 1}, {})
        with pytest.raises(DreamProtocolError):
            transport.request({"n": 2}, {})
        time.sleep(0.3)
        assert len(bodies) == 2, "the cut request was sent again"
    finally:
        transport.close()
        listener.close()


def test_f18_an_idle_connection_the_gateway_closed_is_retried_once() -> None:
    """F-18: a kept-alive connection the gateway closed while idle is still retried on a fresh one."""
    from dream.transport import Transport

    listener, bodies = _raw_gateway("idle")
    transport = Transport("127.0.0.1", listener.getsockname()[1], 2000)
    try:
        transport.request({"n": 1}, {})
        time.sleep(0.1)
        assert transport.request({"n": 2}, {})["status"]["code"] == "DA-0000"
        assert len(bodies) == 2
    finally:
        transport.close()
        listener.close()
