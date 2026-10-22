"""Tests of the recording of `dream.DreamClient`: every tick, sent to the live service.

On every policy (unless the internal `_record: False` config key turns it off), the client opens one record-only
robot link to the platform's live service and sends
each set_observation() as an `observation_state` text message and each kept tick's frames as binary video messages,
in the background, never while a get_action() is in flight. These tests read all of it back from `FakeRelay`, a real
`websockets` server on localhost that speaks the robot link as the live service does (reimplemented here, as
`conftest.FakeDreamAPI` reimplements the gateway, so the tests check the wire independently of the code under test),
and the get_action() side from `conftest.FakeDreamAPI`. Nothing leaves localhost, and every backoff is patched down to
milliseconds.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import logging
import math
import os
import re
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest
from conftest import CHECKPOINT, FakeDreamAPI, wait_until
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import ServerConnection, serve

from dream import DreamClient, _recorder
from dream.errors import DreamConfigError, DreamServerError

## The API key every client here holds, and the header the relay must see it in.
KEY = "test-recorder-key"

## Start of every explicit timestamp: seconds on a made-up monotonic clock.
T0 = 100.0

## A keep rate whose period is exact in binary: 4 frames a second, one every 0.25 s.
EXACT_FPS = 4
EXACT_PERIOD_S = 0.25

## Longest a thread, a link or a message may take before that is reported as a hang.
THREAD_TIMEOUT_S = 5.0

## How long a held request is held before the fake lets it go on its own (below THREAD_TIMEOUT_S).
HOLD_S = 3.0

## Longest set_observation() may take before that counts as waiting on the recorder or the network.
PROMPT_S = 0.05

## How long a test waits to see that something does NOT happen (a send during a held get_action).
QUIET_S = 0.3

## The cameras most tests use.
TOP = "video.top"
WRIST = "video.wrist"
SIDE = "video.side"

## Every key of a record-only hello (spec: the wire contract).
HELLO_KEYS = {
    "type",
    "protocol",
    "robot_id",
    "cameras",
    "width_px",
    "fps",
    "codec",
    "face_blur",
    "agent_version",
    "timezone",
    "pose",
    "record_only",
}

## Every key of recording_stats.
STATS_KEYS = {
    "state",
    "robot_id",
    "frames_sent",
    "states_sent",
    "frames_dropped",
    "states_dropped",
    "bytes_sent",
    "queue_bytes",
    "last_error",
}


# -- the fake live service -----------------------------------------------------------------------------------------


@dataclasses.dataclass
class Unit:
    """One binary message as the relay received it: the 16-byte header's fields, then the JPEG."""

    version: int
    kind: int
    camera: int
    flags: int
    seq: int
    capture_us: int
    payload: bytes


@dataclasses.dataclass
class Link:
    """One connection to the fake relay, as it saw it.

    Attributes:
    - path: The upgrade's path.
    - headers: The upgrade's headers, names lower-cased.
    - socket: The server side, so a test can close it with a code.
    - hello: The first message, parsed; None until it arrives.
    - messages: Every message after the hello, in order: a dict for text, a #Unit for binary.
    - close_code: The code the connection closed with, once it has.
    """

    path: str
    headers: dict[str, str]
    socket: ServerConnection
    hello: dict[str, Any] | None = None
    messages: list[dict[str, Any] | Unit] = dataclasses.field(default_factory=list)
    close_code: int | None = None

    def states(self) -> list[dict[str, Any]]:
        """The `observation_state` messages, in order."""
        return [message for message in self.messages if isinstance(message, dict)]

    def units(self) -> list[Unit]:
        """The binary messages, in order."""
        return [message for message in self.messages if isinstance(message, Unit)]


class FakeRelay:
    """A `websockets` server on localhost that answers a robot link as the live service does.

    `answers` says, connection by connection, how each hello is answered: `"ok"` (`hello_ok`, then every message is
    kept until the link closes), a close code (`bye`, then the close frame with that code), or `"silent"` (nothing,
    until the client hangs up). Connections past the list are answered `"ok"`.
    """

    def __init__(self) -> None:
        """Starts listening on a free port."""
        self.links: list[Link] = []
        self.answers: list[str | int] = []
        self._cond = threading.Condition()
        self._server = serve(self._handle, "127.0.0.1", 0)
        self.port = int(self._server.socket.getsockname()[1])
        self._thread = threading.Thread(target=self._server.serve_forever, name="fake-relay", daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        """The platform URL a client points at to reach this relay."""
        return f"http://127.0.0.1:{self.port}"

    def wait_for(self, predicate: Callable[[], bool], what: str) -> None:
        """Waits until `predicate` holds over what the relay saw; fails the test after THREAD_TIMEOUT_S.

        @param predicate Re-evaluated whenever the relay sees something.
        @param what What was awaited, for the failure.
        """
        with self._cond:
            assert self._cond.wait_for(predicate, THREAD_TIMEOUT_S), f"the relay never saw {what}"

    def count(self) -> int:
        """Messages received after the hellos, over every link."""
        with self._cond:
            return sum(len(link.messages) for link in self.links)

    def close(self) -> None:
        """Stops listening; the links end as their clients close them."""
        self._server.shutdown()
        self._thread.join(THREAD_TIMEOUT_S)

    def _handle(self, socket: ServerConnection) -> None:
        """Serves one connection per `answers`."""
        assert socket.request is not None
        headers = {name.lower(): value for name, value in socket.request.headers.raw_items()}
        link = Link(path=socket.request.path, headers=headers, socket=socket)
        with self._cond:
            index = len(self.links)
            self.links.append(link)
            self._cond.notify_all()
        answer = self.answers[index] if index < len(self.answers) else "ok"
        try:
            hello = json.loads(socket.recv())
            with self._cond:
                link.hello = hello
                self._cond.notify_all()
            if isinstance(answer, int):
                socket.send(json.dumps({"type": "bye", "code": answer, "reason": _recorder.CLOSE_NAMES[answer]}))
                socket.close(answer, _recorder.CLOSE_NAMES[answer])
            elif answer == "silent":
                for _ in socket:
                    pass
            else:
                socket.send(json.dumps({"type": "hello_ok", "node_id": "node-test", "protocol": 1, "subscribed": []}))
                for raw in socket:
                    message = json.loads(raw) if isinstance(raw, str) else _unit(raw)
                    with self._cond:
                        link.messages.append(message)
                        self._cond.notify_all()
        except ConnectionClosed:
            pass
        finally:
            with self._cond:
                link.close_code = socket.close_code
                self._cond.notify_all()


def _unit(raw: bytes) -> Unit:
    """Splits a binary message into its header's fields and its payload."""
    return Unit(*struct.unpack(">BBBBIQ", raw[:16]), payload=raw[16:])


# -- helpers -------------------------------------------------------------------------------------------------------


def jpeg(index: int, role: str = "top") -> bytes:
    """Stand-in JPEG bytes that name the tick and the camera they came from."""
    return b"\xff\xd8" + f"{role}{index:05d}".encode() + b"\xff\xd9"


def frame(index: int, role: str = "top") -> str:
    """#jpeg in base64, as set_observation() takes a frame."""
    return base64.b64encode(jpeg(index, role)).decode()


def both(index: int) -> dict[str, str]:
    """One tick's frames for `TOP` and `WRIST`."""
    return {TOP: frame(index, "top"), WRIST: frame(index, "wrist")}


def open_recording(client: DreamClient, **knobs: Any) -> None:
    """Opens a policy, which always records, with no embodiment checks.

    @param client The client.
    @param knobs `obs_window`, `obs_fps`, `record` or a connection key to override.
    """
    client.policy({"checkpoint": CHECKPOINT, "strict": False, "timeout_ms": 10_000, **knobs})


def wait_for_stats(client: DreamClient, **expected: Any) -> dict[str, Any]:
    """Waits until recording_stats holds `expected` (a counter is bumped just after its send returns, so a moment after
    the relay saw the message); fails the test after THREAD_TIMEOUT_S.

    @param client The client.
    @param expected Stats keys and the values they must reach.
    @return The stats that matched.
    """
    deadline = time.monotonic() + THREAD_TIMEOUT_S
    while True:
        stats = client._recording_stats
        if all(stats[key] == value for key, value in expected.items()):
            return stats
        assert time.monotonic() < deadline, f"recording_stats never reached {expected}: {stats}"
        time.sleep(0.005)


def recorder_threads() -> list[threading.Thread]:
    """The recorder threads alive now."""
    return [thread for thread in threading.enumerate() if thread.name == "dream-recorder"]


def hold_requests(gateway: FakeDreamAPI) -> tuple[threading.Event, threading.Event]:
    """Makes the fake gateway hold every get_action until released (or HOLD_S passes); answers `(arrived, release)`."""
    arrived = threading.Event()
    release = threading.Event()

    def held(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        arrived.set()
        release.wait(HOLD_S)
        return gateway.default_reply(body)

    gateway.handler = held
    return arrived, release


def start_get_action(client: DreamClient, outcome: dict[str, Any]) -> threading.Thread:
    """Runs client.get_action() on a worker thread, recording its result or error under `outcome['result']`."""

    def work() -> None:
        try:
            outcome["result"] = client.get_action()
        except BaseException as exc:  # noqa: B036  # thread boundary: every outcome is recorded
            outcome["result"] = exc

    worker = threading.Thread(target=work, name="recorder-test-get-action")
    worker.start()
    return worker


@pytest.fixture(scope="module")
def module_gateway() -> Iterator[FakeDreamAPI]:
    """One fake gateway for the whole module: stopping one waits out its half-second poll, once here, not per test."""
    fake = FakeDreamAPI()
    yield fake
    fake.close()


@pytest.fixture
def gateway(module_gateway: FakeDreamAPI, monkeypatch: pytest.MonkeyPatch) -> FakeDreamAPI:
    """The module's fake gateway, wiped before each test as a fresh one would be; `DREAM_GATEWAY_URL` names it."""
    module_gateway.requests.clear()
    module_gateway.ended.clear()
    module_gateway.other_paths.clear()
    module_gateway.sessions.clear()
    module_gateway.handler = None
    module_gateway.latency_s = 0.0
    module_gateway.hold_next = 0
    module_gateway.reply_override = None
    monkeypatch.setenv("DREAM_GATEWAY_URL", f"http://{module_gateway.host}:{module_gateway.port}")
    return module_gateway


@pytest.fixture
def relay(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeRelay]:
    """A fake live service on localhost, named by `DREAM_API_URL`; stopped on teardown (after the client, which
    depends on it)."""
    fake = FakeRelay()
    monkeypatch.setenv("DREAM_API_URL", fake.url)
    yield fake
    fake.close()


@pytest.fixture
def client(gateway: FakeDreamAPI, relay: FakeRelay) -> Iterator[DreamClient]:
    """A client with a key, whose gateway is `gateway` and whose platform is `relay`. Closed on
    teardown, which must leave no recorder thread behind."""
    opened = DreamClient(KEY)
    yield opened
    opened.close()
    assert recorder_threads() == []


@pytest.fixture
def fast_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every wait before a new dial, the flat one after a key refusal included, is milliseconds."""
    monkeypatch.setattr(_recorder, "LINK_RECONNECT_MIN_S", 0.01)
    monkeypatch.setattr(_recorder, "LINK_RECONNECT_MAX_S", 0.02)
    monkeypatch.setattr(_recorder, "REFUSAL_WAIT_S", 0.01)


# -- always on ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("knobs", [{}, {"_record": True}], ids=["left-out", "true"])
def test_every_policy_records_with_nothing_in_its_config(
    gateway: FakeDreamAPI, relay: FakeRelay, client: DreamClient, knobs: dict[str, Any]
) -> None:
    """A policy records with nothing in its config (`_record: True` is the same): one link, every tick on it; the key
    is never sent to the model."""
    client.policy({"checkpoint": CHECKPOINT, "strict": False, "timeout_ms": 2000, **knobs})
    for index in range(3):
        client.set_observation(both(index), {"tick": index}, timestamp=T0 + index * EXACT_PERIOD_S)
    client.get_action()
    relay.wait_for(lambda: len(relay.links) == 1 and len(relay.links[0].states()) == 3, "three ticks")
    assert client._recording_stats["state"] == "linked"
    assert b"record" not in gateway.requests[0].raw


def test_recording_stats_off_is_zeros_and_a_new_dict_every_read(client: DreamClient) -> None:
    """Before any policy: state `off`, no robot id, every count 0, no error; changing one read changes no other."""
    stats = client._recording_stats
    assert stats == {
        "state": "off",
        "robot_id": None,
        "frames_sent": 0,
        "states_sent": 0,
        "frames_dropped": 0,
        "states_dropped": 0,
        "bytes_sent": 0,
        "queue_bytes": 0,
        "last_error": None,
    }
    stats["frames_sent"] = 99
    assert client._recording_stats["frames_sent"] == 0


# -- policy(): the record key and what keeps recording off ----------------------------------------------------------


def test_the_internal_record_false_opens_no_thread_and_no_socket(
    gateway: FakeDreamAPI, relay: FakeRelay, client: DreamClient
) -> None:
    """`_record: False` (what a caller whose episodes are not the account's passes, the MuJoCo example): no recorder
    thread, the platform never dialed through a whole episode, the stats `off` saying so; never sent to the model."""
    client.policy({"checkpoint": CHECKPOINT, "strict": False, "timeout_ms": 2000, "_record": False})
    for index in range(3):
        client.set_observation(both(index), {"tick": index}, timestamp=T0 + index * EXACT_PERIOD_S)
    client.get_action()
    client.reset_episode()
    assert recorder_threads() == [] and relay.links == []
    stats = client._recording_stats
    assert stats["state"] == "off" and stats["last_error"] == "recording is off for this policy"
    assert b"_record" not in gateway.requests[0].raw


@pytest.mark.parametrize("value", [1, 0, "true", None, 1.0, [True]])
def test_the_internal_record_key_must_be_a_bool(
    gateway: FakeDreamAPI, relay: FakeRelay, client: DreamClient, value: Any
) -> None:
    """`_record` other than True or False is a DreamConfigError naming it; nothing starts and nothing is sent."""
    with pytest.raises(DreamConfigError, match="'_record'"):
        open_recording(client, _record=value)
    with pytest.raises(DreamConfigError):
        client.get_action()
    assert recorder_threads() == [] and relay.links == [] and gateway.requests == []


def test_there_is_no_public_record_key(gateway: FakeDreamAPI, relay: FakeRelay, client: DreamClient) -> None:
    """`record` is not a config key: refused like any key the client does not take; nothing starts."""
    with pytest.raises(DreamConfigError, match="record"):
        open_recording(client, record=True)
    assert recorder_threads() == [] and relay.links == [] and gateway.requests == []


def assert_off_and_working(client: DreamClient, relay: FakeRelay, reason: str) -> None:
    """The policy opened and serves an action, recording is off with `reason` in its last error, and nothing dialed.

    @param client The experimental client, its policy open.
    @param relay The fake live service, which no link may reach.
    @param reason A piece of the line `recording_stats` must give for being off.
    """
    client.set_observation(both(0), {"tick": 0}, timestamp=T0)
    client.get_action()
    stats = client._recording_stats
    assert stats["state"] == "off" and stats["robot_id"] is None and reason in (stats["last_error"] or "")
    assert recorder_threads() == [] and relay.links == []
    client.close()
    assert client._recording_stats == _recorder.off_stats()


def test_without_websockets_the_policy_opens_and_recording_stays_off(
    gateway: FakeDreamAPI, relay: FakeRelay, client: DreamClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """websockets that cannot be imported (a broken install, since it is a dependency): no error, the policy serves,
    and recording_stats says recording is off and how to install it."""
    for name in ("websockets", "websockets.sync", "websockets.sync.client"):
        monkeypatch.setitem(sys.modules, name, None)
    open_recording(client)
    assert_off_and_working(client, relay, "pip install 'websockets>=13'")


def test_without_an_api_key_the_policy_opens_and_recording_stays_off(gateway: FakeDreamAPI, relay: FakeRelay) -> None:
    """No key anywhere (conftest clears the environment and the file): no error, since the live service refuses
    every link without one; recording_stats names the variable."""
    keyless = DreamClient()
    open_recording(keyless)
    assert_off_and_working(keyless, relay, "DREAM_API_KEY")


def test_a_platform_url_no_link_can_be_made_from_leaves_recording_off(
    gateway: FakeDreamAPI, relay: FakeRelay, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plain `http://` platform away from loopback (the key would cross a network in the clear): no error, the
    policy serves, recording_stats says why. A platform URL that is not a URL at all never gets this far: the
    credentials refuse it (`test_credentials.py`)."""
    monkeypatch.setenv("DREAM_API_URL", "http://dev.platform.example")
    bad = DreamClient(KEY)
    open_recording(bad)
    assert_off_and_working(bad, relay, "https://")


@pytest.mark.parametrize(
    ("platform", "link"),
    [
        ("https://dev.platform.example", "wss://dev.platform.example/live/robot"),
        ("https://dev.platform.example:8443/", "wss://dev.platform.example:8443/live/robot"),
        ("http://127.0.0.1:8080", "ws://127.0.0.1:8080/live/robot"),
        ("http://[::1]:8080", "ws://[::1]:8080/live/robot"),
    ],
)
def test_the_link_is_the_platforms_live_robot_path(platform: str, link: str) -> None:
    """`https` becomes `wss`, `http` becomes `ws`, the host and the port are kept, the path is `/live/robot`."""
    assert _recorder.link_url(platform) == link


@pytest.mark.parametrize("platform", ["http://dev.platform.example", "http://10.0.0.5:8080", "http://[2001:db8::1]"])
def test_a_plain_http_platform_off_loopback_is_refused(platform: str) -> None:
    """The key rides the upgrade, so `ws://` is dialed only to a loopback host, as the robots agent does."""
    with pytest.raises(DreamConfigError, match="https://"):
        _recorder.link_url(platform)


# -- the wire ------------------------------------------------------------------------------------------------------


def test_the_hello_is_record_only_and_names_the_episodes_cameras(relay: FakeRelay, client: DreamClient) -> None:
    """The link is dialed on the first tick, at `/live/robot` with the key as a bearer token and no compression; the
    hello is exactly the record-only hello: sorted frame keys, JPEG, `obs_fps` rounded, no pose, `record_only`."""
    open_recording(client, obs_fps=4.6)
    assert client._recording_stats["state"] == "connecting"
    assert relay.links == []
    client.set_observation({WRIST: frame(0, "wrist"), TOP: frame(0), SIDE: frame(0, "side")}, {"t": 0}, timestamp=T0)
    relay.wait_for(lambda: bool(relay.links) and relay.links[0].hello is not None, "a hello")
    link = relay.links[0]
    assert link.path == "/live/robot"
    assert link.headers["authorization"] == f"Bearer {KEY}"
    assert "permessage-deflate" not in link.headers.get("sec-websocket-extensions", "")
    hello = link.hello
    assert hello is not None and set(hello) == HELLO_KEYS
    assert hello["type"] == "hello" and hello["protocol"] == 1
    assert re.fullmatch(r"rec_[0-9a-f]{32}", hello["robot_id"])
    assert hello["cameras"] == [SIDE, TOP, WRIST]
    assert hello["width_px"] == 0 and hello["fps"] == 5 and hello["codec"] == "jpeg"
    assert hello["face_blur"] is False and hello["pose"] is False and hello["record_only"] is True
    assert hello["agent_version"].startswith("dream-client/") and len(hello["agent_version"]) <= 64
    assert isinstance(hello["timezone"], str) and len(hello["timezone"]) <= 64
    wait_until(lambda: client._recording_stats["state"] == "linked", THREAD_TIMEOUT_S)
    assert client._recording_stats["robot_id"] == hello["robot_id"]


def test_a_slow_obs_fps_still_declares_at_least_one_frame_a_second(relay: FakeRelay, client: DreamClient) -> None:
    """`obs_fps` 0.4 rounds to 0, and the hello says 1."""
    open_recording(client, obs_fps=0.4)
    client.set_observation({TOP: frame(0)}, {"t": 0}, timestamp=T0)
    relay.wait_for(lambda: bool(relay.links) and relay.links[0].hello is not None, "a hello")
    assert relay.links[0].hello is not None and relay.links[0].hello["fps"] == 1


def test_each_tick_is_one_state_and_each_kept_tick_one_unit_per_camera(
    gateway: FakeDreamAPI, relay: FakeRelay, client: DreamClient
) -> None:
    """Every set_observation() is one `observation_state` (kept or not, numpy as JSON); a kept tick's frames are one
    keyframe unit per camera, indexed into the hello's cameras, numbered per camera from 0, stamped with its state's
    `t_us`, carrying the JPEG decoded from the base64; the session is None until a get_action() answers one."""
    open_recording(client, obs_fps=EXACT_FPS)
    stamps = [T0, T0 + 0.1, T0 + EXACT_PERIOD_S]
    for index, stamp in enumerate(stamps):
        observation = {"state.q": np.arange(3, dtype=np.float32) + index, "n": np.int64(index), "prompt": "fold"}
        client.set_observation(both(index), observation, timestamp=stamp)
    client.get_action()
    (session,) = gateway.sessions
    client.set_observation(both(3), {"prompt": "fold", "n": 3}, timestamp=T0 + 2 * EXACT_PERIOD_S)
    relay.wait_for(lambda: bool(relay.links) and len(relay.links[0].messages) == 10, "four states and six units")

    link = relay.links[0]
    states = link.states()
    assert [state["type"] for state in states] == ["observation_state"] * 4
    assert [set(state) for state in states] == [{"type", "t_us", "episode", "session_id", "kept", "observation"}] * 4
    assert [state["kept"] for state in states] == [True, False, True, True]
    assert [state["episode"] for state in states] == [0, 0, 0, 0]
    assert [state["observation"]["n"] for state in states] == [0, 1, 2, 3]
    assert states[1]["observation"] == {"state.q": [1.0, 2.0, 3.0], "n": 1, "prompt": "fold"}
    t_us = [state["t_us"] for state in states]
    assert [round((later - t_us[0]) / 1000) for later in t_us] == [0, 100, 250, 500]

    assert [state["session_id"] for state in states] == [None, None, None, session]

    cameras = link.hello["cameras"] if link.hello is not None else []
    kept_t_us = [t_us[0], t_us[2], t_us[3]]
    for role, key in (("top", TOP), ("wrist", WRIST)):
        units = [unit for unit in link.units() if unit.camera == cameras.index(key)]
        assert [unit.seq for unit in units] == [0, 1, 2]
        assert [unit.capture_us for unit in units] == kept_t_us
        assert [unit.payload for unit in units] == [jpeg(0, role), jpeg(2, role), jpeg(3, role)]
        assert all(unit.version == 1 and unit.kind == 1 and unit.flags == 0x01 for unit in units)

    stats = wait_for_stats(client, states_sent=4, frames_sent=6)
    assert stats["frames_dropped"] == 0 and stats["states_dropped"] == 0 and stats["queue_bytes"] == 0
    wire_bytes = sum(len(json.dumps(state, separators=(",", ":"))) for state in states)
    wire_bytes += sum(16 + len(unit.payload) for unit in link.units())
    assert stats["bytes_sent"] == wire_bytes


def test_the_stamp_is_the_wall_clock_of_the_tick(relay: FakeRelay, client: DreamClient) -> None:
    """With the default timestamp (`time.monotonic()` at the call), `t_us` is the wall clock in microseconds."""
    open_recording(client)
    before_us = time.time() * 1e6
    client.set_observation({TOP: frame(0)}, {"t": 0})
    after_us = time.time() * 1e6
    relay.wait_for(lambda: bool(relay.links) and len(relay.links[0].messages) == 2, "a state and a unit")
    t_us = relay.links[0].states()[0]["t_us"]
    assert before_us - 1000 <= t_us <= after_us + 1000
    assert relay.links[0].units()[0].capture_us == t_us


def test_a_change_to_the_callers_values_after_the_call_changes_nothing_recorded(
    relay: FakeRelay, client: DreamClient
) -> None:
    """The state is encoded in set_observation(): an array or a dict changed in place afterwards is recorded as it
    was."""
    open_recording(client)
    joints = np.zeros(3)
    observation: dict[str, Any] = {"state.q": joints, "nested": {"a": 1}}
    client.set_observation({TOP: frame(0)}, observation, timestamp=T0)
    joints[:] = 7.0
    observation["nested"]["a"] = 2
    observation["late"] = True
    relay.wait_for(lambda: bool(relay.links) and len(relay.links[0].states()) == 1, "a state")
    assert relay.links[0].states()[0]["observation"] == {"state.q": [0.0, 0.0, 0.0], "nested": {"a": 1}}


@pytest.mark.parametrize(
    "value",
    [
        pytest.param({1, 2}, id="a-set"),
        pytest.param(math.nan, id="nan"),
        pytest.param(np.array([0.0, np.inf]), id="an-array-holding-inf"),
        pytest.param(b"\xff\xd8", id="bytes"),
        pytest.param(object(), id="an-object"),
        pytest.param("x" * (64 << 10), id="over-64-kib"),
    ],
)
def test_a_state_that_cannot_be_sent_is_dropped_and_counted_never_raised(
    relay: FakeRelay, client: DreamClient, value: Any
) -> None:
    """A state JSON cannot carry, or longer than the live service takes: set_observation() returns as usual, the state
    is dropped and counted, the tick's frames are still sent, and the next tick is recorded."""
    open_recording(client, obs_fps=EXACT_FPS)
    client.set_observation({TOP: frame(0)}, {"bad": value}, timestamp=T0)
    client.set_observation({TOP: frame(1)}, {"good": 1}, timestamp=T0 + EXACT_PERIOD_S)
    relay.wait_for(lambda: bool(relay.links) and len(relay.links[0].messages) == 3, "two units and a state")
    link = relay.links[0]
    assert [state["observation"] for state in link.states()] == [{"good": 1}]
    assert [unit.payload for unit in link.units()] == [jpeg(0), jpeg(1)]
    wait_for_stats(client, states_dropped=1, states_sent=1, frames_sent=2)


def test_a_frame_that_cannot_be_sent_is_dropped_and_counted_and_keeps_its_seq(
    relay: FakeRelay, client: DreamClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A frame that is not base64, or larger than the live service takes once decoded, is never sent; it is counted,
    and its seq is spent, so the stored seq shows the gap."""
    monkeypatch.setattr(_recorder, "LIVE_MAX_AU_BYTES", 64)
    open_recording(client, obs_fps=EXACT_FPS)
    too_large = base64.b64encode(b"\xff\xd8" + bytes(100)).decode()
    for index, top in enumerate([frame(0), "not base64!", too_large, frame(3)]):
        client.set_observation({TOP: top}, {"n": index}, timestamp=T0 + index * EXACT_PERIOD_S)
    relay.wait_for(lambda: bool(relay.links) and len(relay.links[0].messages) == 6, "four states and two units")
    units = relay.links[0].units()
    assert [(unit.seq, unit.payload) for unit in units] == [(0, jpeg(0)), (3, jpeg(3))]
    assert len(relay.links[0].states()) == 4
    wait_for_stats(client, frames_dropped=2, frames_sent=2, states_sent=4, states_dropped=0)


def test_episodes_count_per_client_and_the_robot_id_is_kept_across_policies(
    relay: FakeRelay, client: DreamClient
) -> None:
    """Episode 0 from the first policy(), one more at each reset_episode() and each policy(); a new policy() opens a
    new link under the same robot id, whose seqs start again at 0."""
    open_recording(client)
    client.set_observation({TOP: frame(0)}, {"n": 0}, timestamp=T0)
    client.get_action()
    client.reset_episode()
    client.set_observation({TOP: frame(1)}, {"n": 1}, timestamp=T0 + 10.0)
    relay.wait_for(lambda: bool(relay.links) and len(relay.links[0].states()) == 2, "two states")
    open_recording(client)
    client.set_observation({TOP: frame(2)}, {"n": 2}, timestamp=T0 + 20.0)
    relay.wait_for(lambda: len(relay.links) == 2 and len(relay.links[1].messages) == 2, "the second policy's tick")
    first, second = relay.links
    assert first.close_code == 1000
    assert [state["episode"] for state in first.states()] == [0, 1]
    assert [state["episode"] for state in second.states()] == [2]
    assert first.hello is not None and second.hello is not None
    assert first.hello["robot_id"] == second.hello["robot_id"]
    assert [unit.seq for unit in first.units()] == [0, 1] and [unit.seq for unit in second.units()] == [0]


def test_a_new_camera_set_reopens_the_link_with_a_new_hello(relay: FakeRelay, client: DreamClient) -> None:
    """An episode with other frame keys closes the link normally and dials again with its own cameras; seqs are per
    camera."""
    open_recording(client)
    client.set_observation(both(0), {"n": 0}, timestamp=T0)
    client.reset_episode()
    client.set_observation({SIDE: frame(1, "side"), TOP: frame(1)}, {"n": 1}, timestamp=T0 + 10.0)
    relay.wait_for(lambda: len(relay.links) == 2 and len(relay.links[1].messages) == 3, "the second episode")
    first, second = relay.links
    assert first.hello is not None and first.hello["cameras"] == [TOP, WRIST]
    assert second.hello is not None and second.hello["cameras"] == [SIDE, TOP]
    assert first.close_code == 1000
    assert [(unit.camera, unit.seq) for unit in first.units()] == [(0, 0), (1, 0)]
    assert [(unit.camera, unit.seq, unit.payload) for unit in second.units()] == [
        (0, 0, jpeg(1, "side")),
        (1, 1, jpeg(1)),
    ]
    assert [state["episode"] for state in first.states() + second.states()] == [0, 1]


# -- get_action() first ----------------------------------------------------------------------------------------------


def test_nothing_is_sent_while_a_get_action_is_in_flight(
    gateway: FakeDreamAPI, relay: FakeRelay, client: DreamClient
) -> None:
    """While a request is held on the gateway the recorder sends nothing and set_observation() stays prompt; once it
    returns, every tick queued meanwhile is sent, in order."""
    open_recording(client, obs_fps=EXACT_FPS)
    client.set_observation(both(0), {"n": 0}, timestamp=T0)
    relay.wait_for(lambda: bool(relay.links) and len(relay.links[0].messages) == 3, "the first tick")
    arrived, release = hold_requests(gateway)
    outcome: dict[str, Any] = {}
    worker = start_get_action(client, outcome)
    durations: list[float] = []
    try:
        assert arrived.wait(THREAD_TIMEOUT_S), "the request never reached the gateway"
        for index in range(1, 6):
            started = time.perf_counter()
            client.set_observation(both(index), {"n": index}, timestamp=T0 + index * EXACT_PERIOD_S)
            durations.append(time.perf_counter() - started)
        time.sleep(QUIET_S)
        assert relay.count() == 3, "the recorder sent during a get_action"
        assert client._recording_stats["states_sent"] == 1
    finally:
        release.set()
        worker.join(THREAD_TIMEOUT_S)
    assert not worker.is_alive(), "get_action hung"
    assert isinstance(outcome["result"], dict), outcome["result"]
    assert max(durations) < PROMPT_S, durations
    relay.wait_for(lambda: relay.count() == 18, "the ticks queued during the get_action")
    assert [state["observation"]["n"] for state in relay.links[0].states()] == list(range(6))


def test_a_get_action_that_raises_lets_recording_go_on(
    gateway: FakeDreamAPI, relay: FakeRelay, client: DreamClient
) -> None:
    """A refused get_action() releases its hold like a successful one."""
    open_recording(client)
    client.set_observation({TOP: frame(0)}, {"n": 0}, timestamp=T0)
    gateway.fail_next("DA-5000", "serve_fault")
    with pytest.raises(DreamServerError):
        client.get_action()
    client.set_observation({TOP: frame(1)}, {"n": 1}, timestamp=T0 + 1.0)
    relay.wait_for(lambda: bool(relay.links) and len(relay.links[0].states()) == 2, "both ticks")


def test_a_full_queue_drops_the_oldest_and_counts_them(
    gateway: FakeDreamAPI, relay: FakeRelay, client: DreamClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Past the byte bound the oldest messages are dropped, frames and states counted apart, and the newest are
    sent once the get_action() holding them returns; the dropped frames' seqs are a gap."""
    monkeypatch.setattr(_recorder, "RECORD_QUEUE_BYTES", 2_000)
    big = [base64.b64encode(jpeg(index) + bytes(300)).decode() for index in range(21)]
    open_recording(client, obs_fps=EXACT_FPS)
    client.set_observation({TOP: big[0]}, {"n": 0}, timestamp=T0)
    relay.wait_for(lambda: bool(relay.links) and len(relay.links[0].messages) == 2, "the first tick")
    arrived, release = hold_requests(gateway)
    outcome: dict[str, Any] = {}
    worker = start_get_action(client, outcome)
    try:
        assert arrived.wait(THREAD_TIMEOUT_S), "the request never reached the gateway"
        for index in range(1, 21):
            client.set_observation({TOP: big[index]}, {"n": index}, timestamp=T0 + index * EXACT_PERIOD_S)
        stats = client._recording_stats
        assert 0 < stats["queue_bytes"] <= 2_000
        assert stats["frames_dropped"] > 0 and stats["states_dropped"] > 0
    finally:
        release.set()
        worker.join(THREAD_TIMEOUT_S)
    assert isinstance(outcome["result"], dict), outcome["result"]
    relay.wait_for(lambda: relay.links[0].units()[-1].seq == 20, "the newest tick")
    wait_until(lambda: client._recording_stats["frames_sent"] == len(relay.links[0].units()), THREAD_TIMEOUT_S)
    stats = wait_for_stats(client, queue_bytes=0)
    assert stats["frames_sent"] + stats["frames_dropped"] == 21
    assert stats["states_sent"] + stats["states_dropped"] == 21
    assert relay.links[0].states()[-1]["observation"]["n"] == 20
    seqs = [unit.seq for unit in relay.links[0].units()]
    assert seqs[0] == 0 and seqs[-1] == 20 and seqs[1] > 1, seqs
    assert seqs == sorted(seqs)


# -- refusals and faults -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("code", [4011, 4002])
def test_a_record_off_or_protocol_close_ends_recording_for_the_policy(
    relay: FakeRelay, client: DreamClient, caplog: pytest.LogCaptureFixture, fast_backoff: None, code: int
) -> None:
    """4011 (the platform's recording flag is off) or 4002 (a live service that does not know record-only links):
    disabled, one DEBUG line naming the code and nothing louder, the queue dropped, the thread gone, never dialed
    again for that policy; the next policy() tries again."""
    caplog.set_level(logging.DEBUG, logger="dream")
    relay.answers = [code]
    open_recording(client)
    client.set_observation({TOP: frame(0)}, {"n": 0}, timestamp=T0)
    wait_until(lambda: client._recording_stats["state"] == "disabled", THREAD_TIMEOUT_S)
    wait_until(lambda: recorder_threads() == [], THREAD_TIMEOUT_S)
    for index in range(1, 4):
        client.set_observation({TOP: frame(index)}, {"n": index}, timestamp=T0 + index)
    stats = client._recording_stats
    assert stats["last_error"] == f"closed {code} ({_recorder.CLOSE_NAMES[code]})"
    assert stats["queue_bytes"] == 0 and stats["states_sent"] == 0
    assert len(relay.links) == 1
    lines = [record for record in caplog.records if record.name == "dream.recorder"]
    closes = [record for record in lines if str(code) in record.getMessage()]
    assert len(closes) == 1 and all(record.levelno == logging.DEBUG for record in lines)
    open_recording(client)
    client.set_observation({TOP: frame(9)}, {"n": 9}, timestamp=T0 + 20.0)
    relay.wait_for(lambda: len(relay.links) == 2 and len(relay.links[1].messages) == 2, "the next policy's tick")


@pytest.mark.parametrize("code", [4001, 4004, 4005])
def test_a_key_refusal_waits_the_flat_delay_then_dials_again(
    relay: FakeRelay, client: DreamClient, monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    """4001, 4004 and 4005 wait REFUSAL_WAIT_S, not the backoff (here a minute, so only the flat wait can be what let
    the second dial through); the ticks queued meanwhile are sent on the next link."""
    monkeypatch.setattr(_recorder, "REFUSAL_WAIT_S", 0.01)
    monkeypatch.setattr(_recorder, "LINK_RECONNECT_MIN_S", 60.0)
    monkeypatch.setattr(_recorder, "LINK_RECONNECT_MAX_S", 60.0)
    relay.answers = [code, "ok"]
    open_recording(client)
    client.set_observation({TOP: frame(0)}, {"n": 0}, timestamp=T0)
    relay.wait_for(lambda: len(relay.links) == 2 and len(relay.links[1].messages) == 2, "the second link's tick")
    stats = client._recording_stats
    assert stats["state"] == "linked"
    assert stats["last_error"] == f"closed {code} ({_recorder.CLOSE_NAMES[code]})"


@pytest.mark.parametrize("code", [4010, 4003, 4008])
def test_another_close_backs_off_then_dials_again(
    relay: FakeRelay, client: DreamClient, monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    """Any other close takes the jittered backoff, not the flat wait (here a minute)."""
    monkeypatch.setattr(_recorder, "REFUSAL_WAIT_S", 60.0)
    monkeypatch.setattr(_recorder, "LINK_RECONNECT_MIN_S", 0.01)
    monkeypatch.setattr(_recorder, "LINK_RECONNECT_MAX_S", 0.02)
    relay.answers = [code, "ok"]
    open_recording(client)
    client.set_observation({TOP: frame(0)}, {"n": 0}, timestamp=T0)
    relay.wait_for(lambda: len(relay.links) == 2 and len(relay.links[1].messages) == 2, "the second link's tick")
    assert client._recording_stats["last_error"] == f"closed {code} ({_recorder.CLOSE_NAMES[code]})"


def test_a_link_lost_after_hello_ok_dials_again_and_sends_on(
    relay: FakeRelay, client: DreamClient, fast_backoff: None
) -> None:
    """The live service closes a linked link (draining): the recorder dials again under the same robot id and the
    next ticks go on the new link, seqs continuing."""
    open_recording(client, obs_fps=EXACT_FPS)
    client.set_observation({TOP: frame(0)}, {"n": 0}, timestamp=T0)
    relay.wait_for(lambda: bool(relay.links) and len(relay.links[0].messages) == 2, "the first tick")
    relay.links[0].socket.close(4010, "draining")
    wait_until(lambda: client._recording_stats["last_error"] == "closed 4010 (draining)", THREAD_TIMEOUT_S)
    client.set_observation({TOP: frame(1)}, {"n": 1}, timestamp=T0 + EXACT_PERIOD_S)
    relay.wait_for(lambda: len(relay.links) == 2 and len(relay.links[1].messages) == 2, "the tick after")
    first, second = relay.links
    assert first.hello is not None and second.hello is not None
    assert first.hello["robot_id"] == second.hello["robot_id"]
    assert [unit.seq for unit in second.units()] == [1]


def test_no_hello_ok_in_time_backs_off_then_dials_again(
    relay: FakeRelay, client: DreamClient, monkeypatch: pytest.MonkeyPatch, fast_backoff: None
) -> None:
    """A relay that never answers the hello: the attempt fails after HELLO_TIMEOUT_S and the next one links."""
    monkeypatch.setattr(_recorder, "HELLO_TIMEOUT_S", 0.2)
    relay.answers = ["silent", "ok"]
    open_recording(client)
    client.set_observation({TOP: frame(0)}, {"n": 0}, timestamp=T0)
    relay.wait_for(lambda: len(relay.links) == 2 and len(relay.links[1].messages) == 2, "the second link's tick")
    assert "hello_ok" in (client._recording_stats["last_error"] or "")


def test_an_unreachable_platform_backs_off_and_keeps_the_newest_ticks(
    gateway: FakeDreamAPI, blackhole_port: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing listening: the recorder backs off and dials again, set_observation() stays prompt, and the queue keeps
    the newest ticks within its bound."""
    monkeypatch.setattr(_recorder, "LINK_RECONNECT_MIN_S", 0.01)
    monkeypatch.setattr(_recorder, "LINK_RECONNECT_MAX_S", 0.02)
    monkeypatch.setattr(_recorder, "RECORD_QUEUE_BYTES", 1_000)
    monkeypatch.setenv("DREAM_API_URL", f"http://127.0.0.1:{blackhole_port}")
    unreachable = DreamClient(KEY)
    try:
        open_recording(unreachable)
        durations: list[float] = []
        for index in range(30):
            started = time.perf_counter()
            unreachable.set_observation({TOP: frame(index)}, {"n": index}, timestamp=T0 + index)
            durations.append(time.perf_counter() - started)
        wait_until(lambda: unreachable._recording_stats["last_error"] is not None, THREAD_TIMEOUT_S)
        stats = unreachable._recording_stats
        assert stats["state"] in ("backoff", "connecting")
        assert "Connection" in (stats["last_error"] or ""), stats["last_error"]
        assert 0 < stats["queue_bytes"] <= 1_000 and stats["frames_dropped"] > 0 and stats["states_dropped"] > 0
        assert max(durations) < PROMPT_S, durations
    finally:
        unreachable.close()
    assert recorder_threads() == []


# -- stopping ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("answer", ["ok", "silent"])
def test_close_stops_the_thread_within_a_second(
    relay: FakeRelay, client: DreamClient, answer: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Linked, or waiting for a `hello_ok` that never comes, close() returns within a second with the thread gone."""
    monkeypatch.setattr(_recorder, "HELLO_TIMEOUT_S", 30.0)
    relay.answers = [answer]
    open_recording(client)
    client.set_observation({TOP: frame(0)}, {"n": 0}, timestamp=T0)
    relay.wait_for(lambda: bool(relay.links) and relay.links[0].hello is not None, "a hello")
    started = time.perf_counter()
    client.close()
    assert time.perf_counter() - started < _recorder.STOP_TIMEOUT_S + 0.1
    assert recorder_threads() == []
    assert client._recording_stats["state"] == "off"


def test_close_sends_what_is_queued_then_closes_the_link_normally(relay: FakeRelay, client: DreamClient) -> None:
    """On a link that is up, ticks queued just before close() still reach the live service, and the link ends with a
    normal close."""
    open_recording(client, obs_fps=EXACT_FPS)
    client.set_observation({TOP: frame(0)}, {"n": 0}, timestamp=T0)
    relay.wait_for(lambda: bool(relay.links) and len(relay.links[0].messages) == 2, "the first tick")
    for index in range(1, 11):
        client.set_observation({TOP: frame(index)}, {"n": index}, timestamp=T0 + index * EXACT_PERIOD_S)
    client.close()
    relay.wait_for(lambda: relay.links[0].close_code is not None, "the close")
    assert [state["observation"]["n"] for state in relay.links[0].states()] == list(range(11))
    assert relay.links[0].close_code == 1000


def test_close_before_the_link_is_up_drops_the_queue_at_once(
    gateway: FakeDreamAPI, blackhole_port: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recorder that is not linked (here backing off from a platform nobody listens on) has nothing to flush on:
    close() drops what it holds, counted, and returns at once."""
    monkeypatch.setattr(_recorder, "LINK_RECONNECT_MIN_S", 60.0)
    monkeypatch.setattr(_recorder, "LINK_RECONNECT_MAX_S", 60.0)
    monkeypatch.setenv("DREAM_API_URL", f"http://127.0.0.1:{blackhole_port}")
    unreachable = DreamClient(KEY)
    open_recording(unreachable)
    unreachable.set_observation({TOP: frame(0)}, {"n": 0}, timestamp=T0)
    wait_until(lambda: unreachable._recording_stats["state"] == "backoff", THREAD_TIMEOUT_S)
    recording = unreachable._recording
    assert recording is not None
    started = time.perf_counter()
    unreachable.close()
    assert time.perf_counter() - started < 0.2
    assert recorder_threads() == []
    stats = recording.stats()
    assert stats["queue_bytes"] == 0 and stats["states_dropped"] == 1 and stats["frames_dropped"] == 1


def test_interpreter_exit_closes_the_link_normally(relay: FakeRelay) -> None:
    """A script that never calls close(): at exit the recorder still sends what it holds and closes the link with a
    normal close, instead of the socket simply dying with the process."""
    script = (
        "import time\n"
        "from dream import DreamClient\n"
        "client = DreamClient('k')\n"
        "client.policy({'checkpoint': 'c', 'strict': False})\n"
        "client.set_observation({'video.top': 'AAAA'}, {'n': 1})\n"
        "deadline = time.monotonic() + 5\n"
        "while client._recording_stats['state'] != 'linked' and time.monotonic() < deadline:\n"
        "    time.sleep(0.01)\n"
        "client.set_observation({'video.top': 'AAAA'}, {'n': 2})\n"
    )
    env = {**os.environ, "DREAM_API_URL": relay.url, "DREAM_GATEWAY_URL": "http://127.0.0.1:9"}
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stderr
    relay.wait_for(lambda: bool(relay.links) and relay.links[0].close_code is not None, "the close")
    assert [state["observation"]["n"] for state in relay.links[0].states()] == [1, 2]
    assert relay.links[0].close_code == 1000
