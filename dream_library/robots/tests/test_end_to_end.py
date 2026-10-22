"""End-to-end tests: the real `dream` client between the control loop and a fake DreamAPI.

Every other test file stubs the client. Here `robots.control.ControlLoop` drives a real
`dream.DreamClient` with a strict-mode policy open (`robots.run.open_client`) against `FakeDreamAPI`, a threaded
HTTP server in this file that speaks DreamAPI's `POST /v1/get_action` contract: JSON body `{checkpoint, session_id,
reset, observation}`, a reply `{session_id, status, actions}` with the `DA-xxxx` status envelope,
sessions minted on `session_id: null` and echoed afterwards. That is the one place where
`robots.adapter`'s key names, shapes and `embodiment_spec()` are checked by dream's own strict
mode rather than by the adapter agreeing with itself, and where what reaches the wire is
asserted to be the serve's observation. The contract is reimplemented here on purpose: nothing
is imported from `dream.transport`, dream's own test suite or the DreamAPI product.

The loop runs on the real clock at 50 Hz, so a chunk of four rows drains in 80 ms and three
chunks in well under a second. The rate is chosen against the fake's round trip (about a
millisecond once Nagle is off on its socket): a chunk installs only the rows still ahead of the
time it lands, and at a 20 ms period that answer never rounds to a skipped row, so
the commands below are the serve's rows from row 0. Waits poll with deadlines; nothing sleeps
for synchronization.

The last tests cover `robots.run.main`: hardware constructors and `robots.run.open_client` are
monkeypatched to return the conftest fakes, and the embodiment fill and the config checks are
asserted.
"""

import asyncio
import base64
import contextlib
import dataclasses
import http.server
import json
import logging
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from typing import Any

import conftest
import cv2
import dream
import numpy as np
import pytest
import test_control

from robots import adapter, cameras, config, control, errors, followers, home, kinematics, run, yam

## Rows per chunk in these tests: short so three chunks drain in tens of milliseconds.
HORIZON = 4

## Tick rate the loop runs at on the real clock; see the module docstring for why not faster.
CONTROL_HZ = 50.0

## Checkpoint every test routes to: the code's default, since the config names none; echoed in the body.
CHECKPOINT = run.DEFAULT_CHECKPOINT

## The credentials' key (`conftest.CREDENTIALS_TEXT`); it must reach the fake as a bearer token without being passed.
API_KEY = "test-key"

## Per-channel slack for a solid colour after a quality-85 JPEG round trip.
JPEG_TOLERANCE = 4

## Longest a test waits for any single condition; a slow CI box still has room under 5 s.
_WAIT_TIMEOUT_S = 3.0

## Poll interval of the test-side waits, seconds; also the pause loop's tick inside the fake.
_WAIT_POLL_S = 0.002

## HTTP status DreamAPI puts on a refusal, by envelope code; the client reads the body, not this.
_HTTP_STATUS = {"DA-1000": 401}


@dataclasses.dataclass
class Received:
    """One `/v1/get_action` request as the fake received it.

    Attributes:
    - body: The decoded JSON body.
    - headers: Request headers, lower-cased names.
    """

    body: dict[str, Any]
    headers: dict[str, str]


## The platform's record of a public YAM model with no EEF head (it answers the YAM's joints only), as
## `GET /v1/models` lists it for every key; a test name, no model Dream Labs serves.
YAM_MODEL: dict[str, Any] = {
    "id": "dreamlabs/test-yam-joints",
    "owner": "public",
    "base_model": "dream-1-base",
    "embodiment": "yam",
    "servable": True,
    "created_at": "2026-10-01T00:00:00Z",
    "cameras": ["overhead", "wrist_l", "wrist_r"],
}
## The YAM's default model (`robots.bodies.YAM_CHECKPOINT`), listed beside it.
BETA_MODEL: dict[str, Any] = {**YAM_MODEL, "id": "dreamlabs/dream-1-beta"}


class FakeDreamAPI:
    """Threaded HTTP server answering `POST /v1/get_action` with fixed-horizon chunks.

    Sessions follow the gateway: a request with `session_id: null` mints a uuid4 and answers
    `DA-0001 new`; a known id is echoed with `DA-0000`; after restart() a formerly known id gets a
    fresh session and `DA-0001 unknown`, which the client reports as `DreamSessionExpired`.
    Chunk `k` (zero-based request index) carries rows `k * horizon + i`, the same scheme as
    `conftest.make_chunk`, so the commands the arms receive reveal which chunk they came from.

    Attributes:
    - host: Bind address, loopback.
    - port: Ephemeral port bound at construction.
    - requests: Every decoded `/v1/get_action` request, in order.
    - sessions: Live session ids.
    - bad_replies_remaining: While positive, each reply's `q_r_chunk` is one column short, and so the right
      gripper rows of `per_key` (derived from the chunk, as the serve derives the chunk from them) are empty.
    - paused: While True a request is held without a reply until the flag clears or the fake closes.
    - models: What `GET /v1/models` answers, for a test that names the fake as the platform too (a run with `ik`,
      every YAM run, reads its checkpoint's record there); the public YAM models (#BETA_MODEL, #YAM_MODEL) unless
      replaced. Any
      other `GET` is answered 404, so a client's recording link dialled at the fake fails as it does on a platform
      that takes none.
    - gets: Every `GET` other than `/v1/models` as `Received` with an empty body, its path under `path`, in order:
      a client's recording link dialled at the fake.
    """

    def __init__(self, horizon: int) -> None:
        """Binds a listener on a random loopback port; the serving thread starts in start().

        @param horizon Rows per chunk.
        """
        self.host = "127.0.0.1"
        self.horizon = horizon
        self.requests: list[Received] = []
        self.sessions: set[str] = set()
        self.bad_replies_remaining = 0
        self.paused = False
        self.models: list[dict[str, Any]] = [BETA_MODEL, YAM_MODEL]
        self.gets: list[Received] = []
        self._refusal: dict[str, Any] | None = None
        self._lock = threading.Lock()
        self._closed = threading.Event()
        fake = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            # The reply goes out as headers then body; with Nagle on, the body waits out the client's
            # delayed ACK, which turns a millisecond round trip into forty.
            disable_nagle_algorithm = True

            def do_POST(self) -> None:  # noqa: N802  # http.server's spelling
                fake._serve(self)

            def do_GET(self) -> None:  # noqa: N802  # http.server's spelling
                fake._serve_models(self)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002  # http.server's signature
                pass

        class Server(http.server.ThreadingHTTPServer):
            daemon_threads = True

        self._server = Server((self.host, 0), Handler)
        self.port: int = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, name="FakeDreamAPI", daemon=True)

    def start(self) -> None:
        """Starts the serving thread."""
        self._thread.start()

    def close(self) -> None:
        """Stops the listener, releases any held request and joins the thread.

        @throws RuntimeError The serving thread did not exit.
        """
        self._closed.set()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(5.0)
        if self._thread.is_alive():
            raise RuntimeError("FakeDreamAPI thread did not exit after close()")

    def restart(self) -> None:
        """Forgets every session, as a restarted gateway does; the listener stays up."""
        with self._lock:
            self.sessions.clear()

    def refuse_next(self, code: str, reason: str) -> None:
        """Answers the next request with a refusal envelope and no actions.

        @param code A `DA-xxxx` code.
        @param reason The code's reason.
        """
        with self._lock:
            self._refusal = _envelope(code, reason)

    def _serve(self, handler: http.server.BaseHTTPRequestHandler) -> None:
        """Handles one POST: logs it, builds the reply, writes it; a paused fake holds the request first."""
        length = int(handler.headers.get("Content-Length") or 0)
        body = json.loads(handler.rfile.read(length))
        headers = {name.lower(): value for name, value in handler.headers.items()}
        with self._lock:
            index = len(self.requests)
            self.requests.append(Received(body=body, headers=headers))
            status, reply = self._reply(body, index)
        while self.paused and not self._closed.is_set():
            time.sleep(_WAIT_POLL_S)
        payload = json.dumps(reply).encode("utf-8")
        try:
            handler.send_response(status)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(len(payload)))
            handler.end_headers()
            handler.wfile.write(payload)
        except OSError:
            # The client timed out and dropped the connection while the reply was held.
            pass

    def _serve_models(self, handler: http.server.BaseHTTPRequestHandler) -> None:
        """Answers `GET /v1/models` with #models, and any other `GET` with 404."""
        if handler.path != "/v1/models":
            headers = {name.lower(): value for name, value in handler.headers.items()}
            with self._lock:
                self.gets.append(Received(body={"path": handler.path}, headers=headers))
            handler.send_error(404)
            return
        payload = json.dumps(self.models).encode("utf-8")
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)

    def _reply(self, body: dict[str, Any], index: int) -> tuple[int, dict[str, Any]]:
        """The contract: a queued refusal, else mint or echo a session and return chunk `index`."""
        sent = body.get("session_id")
        if self._refusal is not None:
            status, self._refusal = self._refusal, None
            return _HTTP_STATUS.get(status["code"], 500), {"session_id": sent or "", "status": status, "actions": None}
        if isinstance(sent, str) and sent in self.sessions:
            session_id, status = sent, _envelope("DA-0000", None)
        else:
            session_id = str(uuid.uuid4())
            self.sessions.add(session_id)
            status = _envelope("DA-0001", "new" if sent is None else "unknown")
        chunk = conftest.make_chunk(self.horizon, base=index * self.horizon)
        if self.bad_replies_remaining > 0:
            self.bad_replies_remaining -= 1
            # One column short: exactly the mismatch strict mode must catch before the loop sees it.
            chunk["q_r_chunk"] = [row[:-1] for row in chunk["q_r_chunk"]]
        # The serve answers the YAM's parts under `per_key` and packs them into the chunks as well.
        per_key = {
            f"action.{side}_arm": [row[:6] for row in chunk[key]]
            for side, key in (("right", "q_r_chunk"), ("left", "q_l_chunk"))
        }
        per_key.update(
            {
                f"action.{side}_gripper": [row[6:] for row in chunk[key]]
                for side, key in (("right", "q_r_chunk"), ("left", "q_l_chunk"))
            }
        )
        actions = {**chunk, "q_r": chunk["q_r_chunk"][0], "q_l": chunk["q_l_chunk"][0], "per_key": per_key}
        return 200, {"session_id": session_id, "status": status, "actions": actions}


@dataclasses.dataclass
class Rig:
    """A ControlLoop over fake hardware and a real dream client, plus the fake gateway it talks to.

    Attributes:
    - loop: The assembled `robots.control.ControlLoop`.
    - server: The FakeDreamAPI the client is connected to.
    - robots: FakeRobot per arm, keyed "left" / "right".
    - embodiment: The `embodiment` block handed to `client.policy()`.
    """

    loop: control.ControlLoop
    server: FakeDreamAPI
    robots: dict[str, conftest.FakeRobot]
    embodiment: dict[str, Any]


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeDreamAPI]:
    """A started FakeDreamAPI answering chunks of HORIZON rows, named as the credentials' gateway and platform (the
    environment wins over the conftest file) so a client built the way a station builds one reaches it, and its
    recording link stays on this box; closed on teardown."""
    fake = FakeDreamAPI(HORIZON)
    fake.start()
    monkeypatch.setenv("DREAM_GATEWAY_URL", f"http://{fake.host}:{fake.port}")
    monkeypatch.setenv("DREAM_API_URL", f"http://{fake.host}:{fake.port}")
    try:
        yield fake
    finally:
        fake.close()


def test_real_client_round_trip_drains_three_chunks(server: FakeDreamAPI, caplog: pytest.LogCaptureFixture) -> None:
    """The adapter's keys and shapes pass strict mode both ways and reach the wire as the serve's own.

    Three chunks flow fake gateway -> real client -> loop -> both arms; the observation the fake
    receives is exactly the serve's (prompt, four state lists of 6 and 1, and per role the window the client kept
    of the frames the loop handed it, base64 JPEGs that decode to the frame, one on the episode's first request);
    the session is minted on the first request and echoed on every later one; nothing is logged at the tick
    boundary.
    """
    caplog.set_level(logging.INFO)
    rig = _make_rig(server, action_timeout_s=1.0, timeout_ms=2000, max_stall_ticks=3)
    rig.robots["left"].joint_pos = np.arange(6) * 0.1
    rig.robots["left"].gripper_pos = np.array([0.25])
    rig.robots["right"].joint_pos = np.arange(6) * -0.1
    rig.robots["right"].gripper_pos = np.array([0.75])
    rows = 3 * HORIZON

    async def body(task: asyncio.Task[None]) -> None:
        await _wait_until(lambda: len(rig.robots["right"].commands) >= rows, task, "three chunks of commands")
        assert rig.loop.state is not None
        assert rig.loop.state.left.commanded_rad is not None
        assert set(rig.loop.video) == set(conftest.CAMERA_NAMES)

    asyncio.run(_run_loop(rig.loop, body))

    first = server.requests[0]
    assert first.headers["authorization"] == f"Bearer {API_KEY}"
    assert first.body["checkpoint"] == CHECKPOINT
    assert first.body["session_id"] is None
    assert first.body["reset"] is False
    observation = first.body["observation"]
    assert set(observation) == {
        "prompt",
        "state.right_arm",
        "state.right_gripper",
        "state.left_arm",
        "state.left_gripper",
        # Each hand's pose and grip, which the client adds on a YAM (`ik` arm `yam`).
        *(f"state.{side}_{part}" for side in ("right", "left") for part in ("eef_xyz", "eef_rot6", "grip")),
        *(f"video.{adapter.YAM_VIDEO_ROLES[name]}" for name in conftest.CAMERA_NAMES),
    }
    assert observation["prompt"] == conftest.TASK
    for key in ("state.right_arm", "state.left_arm"):
        assert len(observation[key]) == 6 and all(type(v) is float for v in observation[key]), key
    for key in ("state.right_gripper", "state.left_gripper"):
        assert len(observation[key]) == 1 and type(observation[key][0]) is float, key
    np.testing.assert_allclose(observation["state.left_arm"], np.arange(6) * 0.1, rtol=1e-6)
    np.testing.assert_allclose(observation["state.right_arm"], np.arange(6) * -0.1, rtol=1e-6)
    assert observation["state.left_gripper"][0] == pytest.approx(0.25)
    assert observation["state.right_gripper"][0] == pytest.approx(0.75)
    for name in conftest.CAMERA_NAMES:
        window = observation[f"video.{adapter.YAM_VIDEO_ROLES[name]}"]
        assert isinstance(window, list) and len(window) == 1 and isinstance(window[0], str), name
        decoded = cv2.imdecode(np.frombuffer(base64.b64decode(window[0]), dtype=np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape == conftest.FRAME_SHAPE, name
        # The fake source captures pure blue in BGR; the wire carries BGR JPEG, so cv2 decodes it back to [255, 0, 0].
        np.testing.assert_allclose(decoded[0, 0], [255, 0, 0], atol=JPEG_TOLERANCE)
    # Session adoption: one session was minted, and every request after the first carried it.
    assert len(server.sessions) == 1
    (session_id,) = server.sessions
    assert all(received.body["session_id"] == session_id for received in server.requests[1:])

    for name in ("left", "right"):
        commands = np.array(rig.robots[name].commands[:rows])
        expected_rows = np.arange(rows, dtype=np.float64)
        np.testing.assert_allclose(commands[:, :6], np.repeat(expected_rows[:, None], 6, axis=1), rtol=1e-6)
        np.testing.assert_allclose(commands[:, 6], (expected_rows % 10) / 10, atol=1e-6)
    assert "get_action failed" not in caplog.text
    assert "stall" not in caplog.text


def test_mis_shaped_chunk_is_logged_and_loop_keeps_running(
    server: FakeDreamAPI, caplog: pytest.LogCaptureFixture
) -> None:
    """A DreamActionError from strict mode costs one refill, not the loop.

    The fake answers the second request with `q_r_chunk` of shape [H, 6], so its `per_key` right gripper part has
    no columns; the real client raises DreamActionError in the worker thread, the tick boundary logs it and holds
    the next refill for RETRY_AFTER_DEFAULT_S (the same shape would come back at once), and the refill after the hold
    succeeds so commands keep flowing; `max_stall_ticks` here outlasts the hold.
    """
    caplog.set_level(logging.INFO)
    rig = _make_rig(server, action_timeout_s=1.0, timeout_ms=2000, max_stall_ticks=int(3 * CONTROL_HZ))
    right = rig.robots["right"]

    async def body(task: asyncio.Task[None]) -> None:
        await _wait_until(lambda: len(right.commands) >= 1, task, "first chunk")
        server.bad_replies_remaining = 1
        await _wait_until(lambda: _has_failed_refill(caplog), task, "the DreamActionError log line")
        rows_before = len(right.commands)
        await _wait_until(lambda: _max_row(right) >= 2 * HORIZON, task, "a chunk after the failure")
        assert len(right.commands) > rows_before
        assert not task.done()

    asyncio.run(_run_loop(rig.loop, body))

    failures = [record for record in caplog.records if "get_action refused, retrying after" in record.getMessage()]
    assert len(failures) == 1
    assert failures[0].name == "robots.control"
    assert failures[0].levelno == logging.WARNING
    assert not [record for record in caplog.records if "get_action failed" in record.getMessage()]
    # dream's own wording for a DreamActionError: the chunk, its wire shape, and the expected one. On a YAM the client
    # does not require `action.<side>_gripper` (a base model answers its grip), so the short chunk is what is caught.
    assert "action key 'q_r_chunk' has shape (4, 6), expected (4, 7)" in failures[0].getMessage()
    assert len(server.requests) >= 3


def test_session_replaced_server_side_is_logged_and_the_loop_continues(
    server: FakeDreamAPI, caplog: pytest.LogCaptureFixture
) -> None:
    """A gateway restart answers the old session id with `DA-0001 unknown`; the client raises
    DreamSessionExpired, the loop logs the restart, drops the buffered rows and carries on in the
    new session; the chunk that came with the replacement never reaches the arms.
    """
    caplog.set_level(logging.INFO)
    rig = _make_rig(server, action_timeout_s=1.0, timeout_ms=2000, max_stall_ticks=10)
    right = rig.robots["right"]

    async def body(task: asyncio.Task[None]) -> None:
        await _wait_until(lambda: len(right.commands) >= 1, task, "first chunk")
        server.restart()
        await _wait_until(lambda: "episode restarted server-side" in caplog.text, task, "the restart log line")
        max_row_at_restart = _max_row(right)
        await _wait_until(lambda: _max_row(right) > max_row_at_restart, task, "a chunk in the new session")
        assert not task.done()

    asyncio.run(_run_loop(rig.loop, body))

    restarted = [record for record in caplog.records if "episode restarted server-side" in record.getMessage()]
    assert len(restarted) == 1
    assert restarted[0].levelno == logging.WARNING
    assert "get_action failed" not in caplog.text
    # The one request that still carried the dead id got the replacement; its rows were dropped.
    old_session = server.requests[1].body["session_id"]
    replaced = [i for i, received in enumerate(server.requests) if received.body["session_id"] == old_session]
    (new_session,) = server.sessions
    assert new_session != old_session
    last_old = replaced[-1]
    dropped_rows = set(range(last_old * HORIZON, (last_old + 1) * HORIZON))
    assert not dropped_rows & {float(command[0]) for command in right.commands}
    assert all(received.body["session_id"] == new_session for received in server.requests[last_old + 1 :])


def test_unauthorized_stops_the_loop_and_holds(server: FakeDreamAPI) -> None:
    """`DA-1000` is DreamAuthError; it propagates out of run() (a key does not become valid on
    retry) and __aexit__ holds both arms on the last command."""
    rig = _make_rig(server, action_timeout_s=1.0, timeout_ms=2000, max_stall_ticks=10)
    right = rig.robots["right"]

    async def body(task: asyncio.Task[None]) -> None:
        await _wait_until(lambda: len(right.commands) >= 1, task, "first chunk")
        server.refuse_next("DA-1000", "unknown_key")
        await asyncio.wait({task}, timeout=_WAIT_TIMEOUT_S)
        assert task.done()
        task.result()

    with pytest.raises(dream.DreamAuthError, match="DA-1000") as info:
        asyncio.run(_run_loop(rig.loop, body))

    assert info.value.code == "DA-1000"
    assert info.value.reason == "unknown_key"
    np.testing.assert_array_equal(right.commands[-1], right.commands[-2])
    assert server.requests[-1].body["session_id"] in server.sessions


def test_timeout_stalls_then_recovers_when_server_resumes(
    server: FakeDreamAPI, caplog: pytest.LogCaptureFixture
) -> None:
    """A silent gateway stalls the loop, DreamTimeout is logged, chunks resume.

    With the fake paused the refill passes `action_timeout_s` (stall: hold and log), then dream's
    own `timeout_ms` raises DreamTimeout in the worker thread, which the tick boundary logs and
    clears. The fresh refill that follows is answered once the fake resumes and the arms move again.
    """
    caplog.set_level(logging.INFO)
    rig = _make_rig(server, action_timeout_s=0.1, timeout_ms=200, max_stall_ticks=200)
    right = rig.robots["right"]

    async def body(task: asyncio.Task[None]) -> None:
        await _wait_until(lambda: len(right.commands) >= 1, task, "first chunk")
        server.paused = True
        await _wait_until(lambda: "stall 1/" in caplog.text, task, "the first stall")
        rows_at_stall = len(right.commands)
        last_real = right.commands[-1]
        # The arm is where it was told. The client blends each chunk from the arm's measured pose along the
        # previous chunk (dream's seam smoothing), so a fake left at zero would receive the recovered chunk
        # blended from row 0 of the old plan, below the held value, and the row probe below would read it as
        # a hold instead of as the first command of the new chunk.
        right.joint_pos = np.asarray(last_real[: conftest.ARM_DOF], dtype=np.float64)
        await _wait_until(lambda: _has_failed_refill(caplog), task, "the DreamTimeout log line")
        max_row_at_timeout = _max_row(right)
        server.paused = False
        await _wait_until(lambda: _max_row(right) > max_row_at_timeout, task, "a chunk after recovery")
        assert not task.done()
        # Every stall tick re-held the last command, so from the first stall up to
        # the first command of the recovered chunk the arm received nothing else.
        first_new = next(i for i, command in enumerate(right.commands) if command[0] > max_row_at_timeout)
        held = right.commands[rows_at_stall:first_new]
        assert held, "no hold commands were written during the stall"
        np.testing.assert_array_equal(held, [last_real] * len(held))

    asyncio.run(_run_loop(rig.loop, body))

    failures = [record.getMessage() for record in caplog.records if "get_action failed" in record.getMessage()]
    assert failures, caplog.text
    assert "timed out after 200 ms" in failures[0]
    assert "stall 1/200" in caplog.text
    # Waiting on the refill is the tick's deliberate choice, reported as the stall above, not as an overrun.
    assert "tick overrun" not in caplog.text


# robots.run.main


def test_run_main_fills_embodiment_from_camera_roles_and_ticks_over_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    """`run.main` builds everything from AppConfig and the dream config from the cameras and the code.

    The hardware constructors and `run.open_client` are monkeypatched to return fakes; the loop ticks a
    few times and is cancelled. The policy config the client is opened with is #robots.run.DEFAULT_CHECKPOINT plus
    the YAM's embodiment config for the camera roles (a rotated camera changes nothing: a camera is a role, not a
    shape), the loop's rate and the YAM's EEF head `ik`, and nothing else: the gateway and the key are the client's
    own to resolve from the credentials, the frame window is the client's default, and a one-shot run has no live
    view. The task reaches the client as the prompt, and `__aexit__` closed the client.
    """
    robots = {"left": conftest.FakeRobot(), "right": conftest.FakeRobot()}
    client = conftest.FakeClient(horizon=HORIZON)
    opened: list[dict[str, Any]] = []

    def fake_open(cfg: dict[str, Any]) -> conftest.FakeClient:
        opened.append(cfg)
        return client

    _patch_hardware(monkeypatch, robots)
    monkeypatch.setattr(run, "open_client", fake_open)
    app = _make_app_config()

    async def scenario() -> None:
        task = asyncio.create_task(run.main(app, home_duration_s=0.0))
        await _wait_until(lambda: client.calls >= 2, task, "two refills")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    assert len(opened) == 1
    assert opened[0] == {
        "checkpoint": run.DEFAULT_CHECKPOINT,
        "embodiment": adapter.YAM_POSE_WIRE.embodiment(list(conftest.CAMERA_NAMES), HORIZON),
        "action_hz": app.control_hz,
        "sync_above_ms": run.SYNC_ABOVE_MS,
        "ik": {"arm": "yam", "head": "eef", "solve": "traj"},
    }
    assert client.observations[0]["prompt"] == conftest.TASK
    assert client.closed
    assert len(robots["right"].commands) >= 3


def test_run_main_rejects_a_camera_that_is_not_a_serve_role(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only `top`, `left`, `right`; another name is a ConfigError before any hardware is built."""
    robots = {"left": conftest.FakeRobot(), "right": conftest.FakeRobot()}
    _patch_hardware(monkeypatch, robots)
    monkeypatch.setattr(run, "open_client", lambda cfg: pytest.fail("the client must not be opened"))
    app = _make_app_config(camera_names=("top", "overhead", "wrist_l"))

    with pytest.raises(errors.ConfigError, match=r"\['overhead', 'wrist_l'\]") as info:
        asyncio.run(run.main(app))

    assert "serve camera roles" in str(info.value)
    assert robots["left"].commands == []


def test_policy_config_carries_no_connection_key() -> None:
    """The dream config is the model, the embodiment and the loop's tick rate only: where DreamAPI is and the key are
    the credentials', which `dream.DreamClient()` resolves itself, so nothing here can point a station elsewhere."""
    app = _make_app_config()
    built = run.policy_config(app)
    assert set(built) == {"checkpoint", "embodiment", "action_hz", "sync_above_ms", "ik"}
    assert built["action_hz"] == app.control_hz
    assert built["checkpoint"] == run.DEFAULT_CHECKPOINT


def test_policy_config_leaves_recording_to_the_client() -> None:
    """Every run's dream client records the run itself: the policy config never carries the `_record` switch."""
    assert "_record" not in run.policy_config(_make_app_config())


def test_policy_config_passes_the_frame_window_only_when_the_config_changes_it() -> None:
    """The client keeps the frame window (9 frames at 5 fps by default, the config's defaults too): a config that
    asks for another size or rate passes it on as `obs_window` / `obs_fps`, and the defaults pass nothing."""
    app = _make_app_config()

    assert "obs_window" not in run.policy_config(app) and "obs_fps" not in run.policy_config(app)
    changed = run.policy_config(dataclasses.replace(app, obs_window=4, obs_fps=10.0))
    assert changed["obs_window"] == 4 and changed["obs_fps"] == 10.0
    only_rate = run.policy_config(dataclasses.replace(app, obs_fps=2.5))
    assert only_rate["obs_fps"] == 2.5 and "obs_window" not in only_rate


def test_the_real_client_records_the_run_on_the_platform(
    server: FakeDreamAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`run.open_client` opens the policy config on a real `dream.DreamClient`, which dials the platform's recording
    link on its first tick: the station's runs are recorded by their own client, Live View or not."""
    app = _make_app_config()
    monkeypatch.setenv("DREAM_API_KEY", "a-station-run")
    client = run.open_client(run.policy_config(app))
    frames = {f"video.{adapter.YAM_VIDEO_ROLES[name]}": "anBlZw==" for name in conftest.CAMERA_NAMES}

    def dialled() -> list[Received]:
        return [get for get in server.gets if get.headers.get("authorization") == "Bearer a-station-run"]

    try:
        client.set_observation(frames, {"prompt": conftest.TASK}, timestamp=1.0)
        deadline_s = time.monotonic() + _WAIT_TIMEOUT_S
        while not dialled() and time.monotonic() < deadline_s:
            time.sleep(_WAIT_POLL_S)
    finally:
        client.close()

    assert [get.body["path"] for get in dialled()][:1] == ["/live/robot"]


def test_a_policy_config_the_client_refuses_fails_the_open(server: FakeDreamAPI) -> None:
    """`run.open_client` raises the client's own refusal of the policy config (a frame window of no frames here)."""
    with pytest.raises(dream.DreamConfigError, match="obs_window"):
        run.open_client({**run.policy_config(_make_app_config()), "obs_window": 0})


def test_a_yam_run_may_name_either_head() -> None:
    """A run may name the joint head (`dreamcli robot up joint`) or the EEF head (`dreamcli robot up eef`, its tool
    poses through the dream client's IK); an unknown head fails by name. The loop's wire is the YAM's either way."""
    app = _make_app_config()

    assert run.with_action_space(app, None) is app
    assert run.policy_config(run.with_action_space(app, "joint"))["ik"] == {
        "arm": "yam",
        "head": "joint",
        "solve": "traj",
    }
    assert run.policy_config(run.with_action_space(app, "eef"))["ik"] == {"arm": "yam", "head": "eef", "solve": "traj"}
    with pytest.raises(errors.ConfigError, match="the action space must be one of"):
        run.with_action_space(app, "pose")
    assert run.loop_settings(run.with_action_space(app, "joint"))["wire"] is adapter.YAM_VIDEO_WIRE


def test_the_station_files_sync_threshold_replaces_the_one_sized_from_the_chunk(server: FakeDreamAPI) -> None:
    """A YAM's 48 rows at 20 Hz size the threshold to 900 ms; `sync_above_ms` in the station file replaces it and
    reaches the client's pacing."""
    yam = dataclasses.replace(_make_app_config(), action_horizon=48, control_hz=20.0)
    assert run.policy_config(yam)["sync_above_ms"] == 900
    lowered = run.policy_config(dataclasses.replace(yam, sync_above_ms=800.0))
    assert lowered["sync_above_ms"] == 800.0
    with dream.DreamClient() as client:
        client.policy(lowered)
        assert client._policy is not None and client._policy._pacer.sync_above_ms == 800.0


def test_system_0_has_its_own_sync_threshold_and_station_file_key(server: FakeDreamAPI) -> None:
    """With System 0 a YAM goes sync above 1000 ms; `sys0_sync_above_ms` replaces that, and each key leaves the other
    mode's threshold alone."""
    yam = dataclasses.replace(_make_app_config(), action_horizon=48, control_hz=20.0)
    assert run.policy_config(yam, sys0=True)["sync_above_ms"] == 1000
    both = dataclasses.replace(yam, sync_above_ms=800.0, sys0_sync_above_ms=950.0)
    assert run.policy_config(both)["sync_above_ms"] == 800.0
    assert run.policy_config(both, sys0=True)["sync_above_ms"] == 950.0
    assert run.policy_config(dataclasses.replace(yam, sync_above_ms=800.0), sys0=True)["sync_above_ms"] == 1000
    with dream.DreamClient() as client:
        client.policy(run.policy_config(yam, sys0=True))
        assert client._policy is not None and client._policy._pacer.sync_above_ms == 1000


def _make_rig(server: FakeDreamAPI, action_timeout_s: float, timeout_ms: int, max_stall_ticks: int) -> Rig:
    """Builds a ControlLoop over conftest fakes with a real strict-mode dream client on `server`.

    The client's config goes through `run.policy_config`, the way `run.main` and the station build theirs; the
    fake is where the credentials say the gateway is (the `server` fixture) and the key is the conftest file's, so
    the client is built the way a station builds it: nothing passed but the timeout.
    """
    # tick: these check stalls, timeouts and recovery, not the solve; on CI's runners the traj compile on its thread
    # (every body's default) starves the loop past their few-second waits. The client tests run traj.
    app = dataclasses.replace(_make_app_config(), ik_solve="tick")
    dream_config = {**run.policy_config(app), "timeout_ms": timeout_ms}
    client = run.open_client(dream_config)
    robots = {"left": conftest.FakeRobot(), "right": conftest.FakeRobot()}
    arms = {name: conftest.yam_arm(name, robot) for name, robot in robots.items()}
    stale_after_s = conftest.LOOP_DEFAULTS["stale_after_s"]
    readers = [
        cameras.CameraReader(name, conftest.FakeCaptureSource(), stale_after_s) for name in conftest.CAMERA_NAMES
    ]
    loop = control.ControlLoop(
        followers=arms,
        cameras=readers,
        client=client,
        task=conftest.TASK,
        control_hz=CONTROL_HZ,
        action_horizon=HORIZON,
        action_timeout_s=action_timeout_s,
        refill_at_remaining=0,
        stale_after_s=stale_after_s,
        max_stall_ticks=max_stall_ticks,
        # One-tick ramp: the commands asserted below are the rows themselves (see conftest.LOOP_DEFAULTS).
        ramp_s=conftest.LOOP_DEFAULTS["ramp_s"],
        ramp_max_rad_s=conftest.LOOP_DEFAULTS["ramp_max_rad_s"],
        max_ee_speed_m_s=conftest.LOOP_DEFAULTS["max_ee_speed_m_s"],
        max_gripper_step=conftest.LOOP_DEFAULTS["max_gripper_step"],
        command_hz=conftest.LOOP_DEFAULTS["command_hz"],
        command_interp_max_s=conftest.LOOP_DEFAULTS["command_interp_max_s"],
        command_thread=conftest.LOOP_DEFAULTS["command_thread"],
    )
    return Rig(loop=loop, server=server, robots=robots, embodiment=dream_config["embodiment"])


async def _run_loop(loop: control.ControlLoop, body: Callable[[asyncio.Task[None]], Any]) -> None:
    """Runs `loop.run()` as a task inside `async with loop` while `body` observes it, then cancels."""
    async with loop:
        task = asyncio.create_task(loop.run())
        try:
            await body(task)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


async def _wait_until(pred: Callable[[], bool], task: asyncio.Task[Any], what: str) -> None:
    """Polls `pred`; fails when the deadline passes, and re-raises if the loop task died first."""
    deadline_s = time.monotonic() + _WAIT_TIMEOUT_S
    while not pred():
        if task.done():
            task.result()
            pytest.fail(f"loop stopped before {what}")
        if time.monotonic() > deadline_s:
            pytest.fail(f"timed out after {_WAIT_TIMEOUT_S} s waiting for {what}")
        await asyncio.sleep(_WAIT_POLL_S)


def _has_failed_refill(caplog: pytest.LogCaptureFixture) -> bool:
    """Whether a refill failed: logged as retried at the next tick, or as held for a retry hint."""
    return any(
        "get_action failed" in record.getMessage() or "get_action refused, retrying after" in record.getMessage()
        for record in caplog.records
    )


def _max_row(robot: conftest.FakeRobot) -> float:
    """Largest joint value commanded so far; rows are numbered across chunks, so this is progress."""
    if not robot.commands:
        return -1.0
    return max(float(command[0]) for command in robot.commands)


def _patch_hardware(monkeypatch: pytest.MonkeyPatch, robots: dict[str, conftest.FakeRobot]) -> None:
    """Routes the three constructors (`yam.open_arm`, cameras, kinematics) to fakes."""

    def fake_arm(cfg: config.ArmConfig, name: str) -> followers.FollowerArm:
        return conftest.yam_arm(name, robots[name])

    def fake_kinematics(cfg: config.ArmConfig) -> test_control.StillKinematics:
        # No MuJoCo here: an end effector that never moves, so the (infinite) cap is never reached anyway.
        return test_control.StillKinematics()

    def fake_reader(cfg: config.CameraConfig, stale_after_s: float, **_encoding: Any) -> cameras.CameraReader:
        # The encoding keywords are dropped: every capture encoded at the fake's size keeps the wire assertions
        # exact.
        return cameras.CameraReader(cfg.name, conftest.FakeCaptureSource(), stale_after_s, rotate=cfg.rotate)

    monkeypatch.setattr(yam, "open_arm", fake_arm)
    monkeypatch.setattr(cameras.CameraReader, "from_config", fake_reader)
    monkeypatch.setattr(kinematics.ArmKinematics, "from_config", fake_kinematics)


def _make_app_config(camera_names: tuple[str, ...] = conftest.CAMERA_NAMES) -> config.AppConfig:
    """An AppConfig over fake-sized cameras (the last one rotated 90 degrees)."""
    height, width, _ = conftest.FRAME_SHAPE
    cameras_cfg = [
        config.CameraConfig(
            name=name, device=i, width=width, height=height, rotate=90 if i == len(camera_names) - 1 else 0
        )
        for i, name in enumerate(camera_names)
    ]
    return config.AppConfig(
        follower_l=config.ArmConfig(can_channel="can_l"),
        follower_r=config.ArmConfig(can_channel="can_r"),
        cameras=cameras_cfg,
        task=conftest.TASK,
        control_hz=CONTROL_HZ,
        action_horizon=HORIZON,
        action_timeout_s=1.0,
        stale_after_s=conftest.LOOP_DEFAULTS["stale_after_s"],
        max_stall_ticks=10,
        max_ee_speed_m_s=conftest.LOOP_DEFAULTS["max_ee_speed_m_s"],
    )


def _envelope(code: str, reason: str | None) -> dict[str, Any]:
    """A status envelope the way DreamAPI serialises one; nothing here is retryable."""
    names = {"DA-0000": "OK", "DA-0001": "OK_NEW_SESSION", "DA-1000": "UNAUTHORIZED"}
    return {
        "code": code,
        "name": names.get(code, code),
        "reason": reason,
        "retryable": False,
        "retry_after_s": None,
        "trace_id": uuid.uuid4().hex[:16],
        "detail": None,
    }


def test_run_main_refuses_a_config_without_a_task_before_opening_anything(monkeypatch: pytest.MonkeyPatch) -> None:
    """A one-shot run has no page to type a prompt on: no `task` and no `--task` fails before the arms are opened."""
    robots = {"left": conftest.FakeRobot(), "right": conftest.FakeRobot()}
    _patch_hardware(monkeypatch, robots)
    monkeypatch.setattr(run, "open_client", lambda *args, **kwargs: pytest.fail("the client must not be opened"))
    app = dataclasses.replace(_make_app_config(), task="")
    with pytest.raises(errors.ConfigError, match="no task prompt"):
        asyncio.run(run.main(app))
    assert robots["left"].commands == [] and robots["right"].commands == []


def test_a_switch_off_that_fails_ends_the_run_with_an_arm_error_once_every_arm_was_tried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robots = {"left": conftest.FakeRobot(), "right": conftest.FakeRobot()}
    client = conftest.FakeClient(horizon=HORIZON)
    _patch_hardware(monkeypatch, robots)
    monkeypatch.setattr(run, "open_client", lambda cfg: client)
    switch_off = yam.I2rtDriver.torque_off
    tried: list[str] = []

    def left_sweep_fails(self: yam.I2rtDriver) -> None:
        tried.append(self.name)
        switch_off(self)
        if self.name == "left":
            raise errors.ArmError("arm 'left': motors [3] on can_l may still be energized; cut the arm's power")

    monkeypatch.setattr(yam.I2rtDriver, "torque_off", left_sweep_fails)

    async def scenario() -> None:
        stop = asyncio.Event()
        task = asyncio.create_task(run.main(_make_app_config(), stop=stop, home_duration_s=0.0))
        await _wait_until(lambda: client.calls >= 2, task, "two refills")
        stop.set()
        with pytest.raises(errors.ArmError, match="arm 'left' may still be powered: its switch-off failed"):
            await task

    asyncio.run(scenario())
    assert tried == ["left", "right"]


RUN_CHILD = """
import asyncio, sys, time
sys.path.insert(0, sys.argv[1])
import conftest, test_control
from robots import cameras, config, kinematics, run, yam

robots = {"left": conftest.FakeRobot(), "right": conftest.FakeRobot()}
client = conftest.FakeClient(horizon=conftest.LOOP_DEFAULTS["action_horizon"])
switch_off = yam.I2rtDriver.torque_off

def announced_off(self):
    print(f"SWITCHED OFF {self.name}", flush=True)
    switch_off(self)

camera_delay_s = float(sys.argv[2])

def reader(cfg, stale_after_s, **window):
    if camera_delay_s:
        print("OPENING CAMERAS", flush=True)
        time.sleep(camera_delay_s)
    return cameras.CameraReader(cfg.name, conftest.FakeCaptureSource(), stale_after_s)

def opened_arm(cfg, name):
    print(f"OPENED ARM {name}", flush=True)
    return conftest.yam_arm(name, robots[name])

yam.open_arm = opened_arm
yam.I2rtDriver.torque_off = announced_off
cameras.CameraReader.from_config = reader
kinematics.ArmKinematics.from_config = lambda cfg: test_control.StillKinematics()
run.open_client = lambda cfg: client
height, width, _ = conftest.FRAME_SHAPE
app = conftest.make_app_config(
    follower_l=config.ArmConfig(can_channel="can_l"),
    follower_r=config.ArmConfig(can_channel="can_r"),
    cameras=[
        config.CameraConfig(name=n, device=i, width=width, height=height)
        for i, n in enumerate(conftest.CAMERA_NAMES)
    ],
    rollouts=config.RolloutsConfig(enabled=True),
)

async def main():
    async def announce():
        while client.calls < 2:
            await asyncio.sleep(0.02)
        print("RUNNING", flush=True)

    announcing = asyncio.create_task(announce())
    await run.main(run.with_task(app, conftest.TASK), home_duration_s=0.0)
    announcing.cancel()

asyncio.run(main())
print("EXITED CLEANLY", flush=True)
"""


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGHUP], ids=["ctrl-c", "hangup"])
def test_a_hangup_ends_a_one_shot_run_the_way_ctrl_c_does(signum: int, tmp_path: pathlib.Path) -> None:
    child = subprocess.Popen(
        [sys.executable, "-c", RUN_CHILD, str(pathlib.Path(__file__).parent), "0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "HOME": str(tmp_path)},
    )
    watchdog = threading.Timer(60.0, child.kill)
    watchdog.start()
    try:
        assert child.stdout is not None
        before: list[str] = []
        for line in iter(child.stdout.readline, ""):
            before.append(line)
            if line == "RUNNING\n":
                break
        assert before and before[-1] == "RUNNING\n", "".join(before)
        child.send_signal(signum)
        after, _ = child.communicate()
    finally:
        watchdog.cancel()
    assert child.returncode == 0, "".join(before) + after
    assert "SWITCHED OFF left" in after and "SWITCHED OFF right" in after and "EXITED CLEANLY" in after, after


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGHUP], ids=["ctrl-c", "hangup"])
def test_a_stop_while_the_cameras_open_ends_the_run_before_an_arm_opens(signum: int, tmp_path: pathlib.Path) -> None:
    child = subprocess.Popen(
        [sys.executable, "-c", RUN_CHILD, str(pathlib.Path(__file__).parent), "1.0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "HOME": str(tmp_path)},
    )
    watchdog = threading.Timer(60.0, child.kill)
    watchdog.start()
    try:
        assert child.stdout is not None
        first = child.stdout.readline()
        assert first == "OPENING CAMERAS\n", first
        child.send_signal(signum)
        after, _ = child.communicate()
    finally:
        watchdog.cancel()
    assert child.returncode == 0, after
    assert "OPENED ARM" not in after and "RUNNING" not in after and "EXITED CLEANLY" in after, after
    assert not (tmp_path / ".dreamlabs" / "rollouts").exists()


def test_a_stop_during_the_first_move_home_switches_the_arms_off_without_starting_the_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robots = {"left": conftest.FakeRobot(), "right": conftest.FakeRobot()}
    client = conftest.FakeClient(horizon=HORIZON)
    _patch_hardware(monkeypatch, robots)
    monkeypatch.setattr(run, "open_client", lambda cfg: client)
    moves = home.return_to_home

    async def scenario() -> None:
        stop = asyncio.Event()
        event_loop = asyncio.get_running_loop()

        def stopped_while_moving(*args: Any, **kwargs: Any) -> bool:
            event_loop.call_soon_threadsafe(stop.set)
            return moves(*args, **kwargs)

        monkeypatch.setattr(home, "return_to_home", stopped_while_moving)
        recording = dataclasses.replace(_make_app_config(), rollouts=config.RolloutsConfig(enabled=True))
        await run.main(recording, stop=stop, home_duration_s=0.0)

    asyncio.run(scenario())
    assert client.calls == 0 and client.closed
    assert robots["left"].closed and robots["right"].closed
    assert not pathlib.Path("~/.dreamlabs/rollouts").expanduser().exists()
