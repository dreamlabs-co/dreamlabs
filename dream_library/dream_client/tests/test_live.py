"""Tests of `dream.LiveViewClient`, the one implementation of Live View: the link it opens, the video it sends, the
status, ticks and pose it reports, the prompts it hands back, and the tick guard that keeps it off the caller's loop.

A fake relay (a real WebSocket server on this machine, on a loop of its own) stands in for the platform's live node,
so the client runs its real thread, its real backoff and, where a test says so, its real encoder; the fake records
what arrives and sends what the page would. Most tests stand a fake encoder in for libvpx (`live._open_encoder`)
and drive the client as a control loop does, one stream() per tick. The station's behaviour tests from
`robots.live` are here with their requirement ids; the ones that need the real station controller are in the
robots package's tests. Waits poll with deadlines; the client's timing constants are shrunk through `monkeypatch`.
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import dataclasses
import json
import logging
import os
import pathlib
import re
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import cv2
import numpy as np
import pytest
import websockets.exceptions

import dream
from dream import live

KEY = "dl_live_test"
ROBOT = "rbt_0123456789abcdef"

## The live view's architecture document, whose constants table the client's knobs are held to; absent in a
## published copy, where that test skips.
DESIGN_DOC = pathlib.Path(__file__).resolve().parents[3] / "design" / "sources" / "live-view-architecture.md"

## The table rows that are the client's knobs or the contract numbers it restates, by the document's name.
CLIENT_CONSTANTS = {
    "LIVE_POLL_HZ": "_POLL_HZ",
    "LIVE_IDLE_ENCODER_S": "_IDLE_ENCODER_S",
    "LIVE_ENCODER_THREADS": "_ENCODER_THREADS",
    "LIVE_ENCODER_SPEED": "_ENCODER_SPEED",
    "LIVE_AGENT_NICE": "_NICE",
    "LIVE_YIELD_TICK_RATIO": "_YIELD_TICK_RATIO",
    "LIVE_YIELD_WINDOW_S": "_YIELD_WINDOW_S",
    "LIVE_YIELD_BACKOFF_S": "_YIELD_BACKOFF_S",
    "LIVE_WRITE_HIGH_WATER": "_WRITE_HIGH_WATER",
    "LIVE_NOTSENT_LOWAT": "_NOTSENT_LOWAT",
    "LIVE_WIDTH_PX": "_WIDTH_PX",
    "LIVE_FPS": "_FPS",
    "LIVE_BITRATE_KBPS": "_BITRATE_KBPS",
    "LIVE_VBV_BUFFER_KBIT": "_VBV_BUFFER_KBIT",
    "LIVE_GOP_S": "_GOP_S",
    "PROMPT_ACK_S": "_PROMPT_ACK_S",
}

## The units a value cell may carry.
UNITS = {"KiB": 1 << 10, "MiB": 1 << 20}

## A row of that table: one or two backticked names, then the value cell.
TABLE_ROW = re.compile(r"^\| `([A-Z0-9_]+)`(?: / `([A-Z0-9_]+)`)? \| ([^|]+?) \|")

## Longest a test waits for the client, the relay or the caller's loop to reach a state.
WAIT_S = 5.0

## The cameras the rig's robot has, named as the station names them.
CAMERAS = ("top", "left", "right")

## The fake capture size; the client resizes it to `width_px` wide at this aspect (64 wide, 48 high: 4:3).
FRAME_SHAPE = (48, 64, 3)

## The rig's `fps`: faster than any tick a test makes, so the frame-rate cap thins nothing unless a test sets `fps`.
RIG_FPS = 100

## Units of 6 KiB a stalled loopback peer can absorb before the client's first drop: the bounded pipe (64 KiB in
## asyncio, 256 KiB the kernel may queue, 16 KiB of it unsent) plus the peer's receive buffer and library queue;
## measured 50 here, and no drop within 300 units (1.8 MiB) with the socket left autotuned.
STALL_DROP_WITHIN_UNITS = 128

## The video header: version, kind, camera, flags, seq, capture time in microseconds.
HEADER = struct.Struct(">BBBBIQ")


def wait_for(predicate: Callable[[], bool], what: str, timeout_s: float = WAIT_S) -> None:
    """Polls from the test thread until `predicate` holds."""
    deadline_s = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline_s:
            raise AssertionError(f"timed out after {timeout_s} s waiting for {what}")
        time.sleep(0.005)


def jpeg(value: int = 0) -> str:
    """A 1280x720 frame of one grey level, as the base64 JPEG an observation carries."""
    ok, encoded = cv2.imencode(".jpg", np.full((720, 1280, 3), value, dtype=np.uint8))
    assert ok
    return base64.b64encode(encoded.tobytes()).decode()


def observation(value: int = 0) -> dict[str, Any]:
    """Two cameras and a state, as set_observation() takes them in one mapping."""
    frame = jpeg(value)
    return {"video.overhead": frame, "video.wrist_l": frame, "state.right_arm": [0.0] * 6, "prompt": "fold"}


@dataclasses.dataclass(frozen=True)
class Header:
    """One video header as the relay unpacks it."""

    version: int
    kind: int
    camera: int
    flags: int
    seq: int
    capture_us: int

    @property
    def is_keyframe(self) -> bool:
        """Whether the keyframe flag is set."""
        return bool(self.flags & 0x01)


class FakeRelay:
    """An in-process websockets server speaking the robot link as the relay does, on a daemon loop of its own.

    Attributes:
    - hellos: Every `hello` received, in order (one per connection).
    - auth_headers: The `Authorization` header of every upgrade.
    - units: Every binary message received as `(header, payload)`.
    - messages: Every text message received after `hello`, parsed.
    - connections: Upgrades accepted so far.
    - closed_at_s: Monotonic time of every connection's end.
    - subscribed: Cameras `hello_ok` claims as already subscribed.
    - refuse: `(code, reason)` to answer every `hello` with a `bye` and a close, or None to accept.
    - is_stalled: While True the handler stops reading, so the robot's socket fills and blocks.
    """

    def __init__(self, subscribed: tuple[str, ...] = (), refuse: tuple[int, str] | None = None) -> None:
        self.hellos: list[dict[str, Any]] = []
        self.auth_headers: list[str | None] = []
        self.units: list[tuple[Header, bytes]] = []
        self.messages: list[dict[str, Any]] = []
        self.connections = 0
        self.closed_at_s: list[float] = []
        self.subscribed = subscribed
        self.refuse = refuse
        self.is_stalled = False
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self.loop.run_forever, name="fake-relay", daemon=True)
        self._server: Any = None
        self._link: Any = None
        self.port = 0

    def start(self) -> None:
        """Binds an ephemeral port on 127.0.0.1."""
        from websockets.asyncio import server

        async def serve() -> None:
            self._server = await server.serve(self._handle, "127.0.0.1", 0, compression=None, max_queue=2)
            self.port = self._server.sockets[0].getsockname()[1]

        self._thread.start()
        asyncio.run_coroutine_threadsafe(serve(), self.loop).result(WAIT_S)

    def stop(self) -> None:
        """Closes the server and the loop."""

        async def close() -> None:
            self._server.close()
            await self._server.wait_closed()

        asyncio.run_coroutine_threadsafe(close(), self.loop).result(WAIT_S)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(WAIT_S)
        self.loop.close()

    @property
    def url(self) -> str:
        """The platform URL that makes the client dial this relay."""
        return f"http://127.0.0.1:{self.port}"

    def send(self, message: dict[str, Any]) -> None:
        """Sends one message to the robot on the current link, as the page's viewer would through the node."""
        wait_for(lambda: self._link is not None, "a link to send on")
        asyncio.run_coroutine_threadsafe(self._link.send(json.dumps(message)), self.loop).result(WAIT_S)

    def send_text(self, text: str) -> None:
        """Sends raw text to the robot, for malformed-message tests."""
        asyncio.run_coroutine_threadsafe(self._link.send(text), self.loop).result(WAIT_S)

    def close_link(self, code: int, reason: str) -> None:
        """Closes the current link with an application close code, after the `bye` the relay would send."""

        async def close() -> None:
            await self._link.send(json.dumps({"type": "bye", "code": code, "reason": reason}))
            await self._link.close(code, reason)

        asyncio.run_coroutine_threadsafe(close(), self.loop).result(WAIT_S)

    def texts(self, kind: str) -> list[dict[str, Any]]:
        """The received text messages of one type."""
        return [message for message in self.messages if message["type"] == kind]

    def acks(self) -> dict[str, dict[str, Any]]:
        """The `prompt_ack` messages received, by prompt id."""
        return {ack["id"]: ack for ack in self.texts("prompt_ack")}

    async def _handle(self, link: Any) -> None:
        """One robot link: `hello` in, `hello_ok` (or `bye`) out, then record everything until the close."""
        self.connections += 1
        self.auth_headers.append(link.request.headers.get("Authorization"))
        try:
            hello = json.loads(await link.recv())
            assert hello["type"] == "hello"
            self.hellos.append(hello)
            if self.refuse is not None:
                code, reason = self.refuse
                await link.send(json.dumps({"type": "bye", "code": code, "reason": reason}))
                await link.close(code, reason)
                return
            self._link = link
            ok = {"type": "hello_ok", "node_id": "live-1", "protocol": 1, "subscribed": list(self.subscribed)}
            await link.send(json.dumps(ok))
            while True:
                while self.is_stalled:
                    await asyncio.sleep(0.01)
                data = await link.recv()
                if isinstance(data, bytes):
                    self.units.append((Header(*HEADER.unpack(data[: HEADER.size])), data[HEADER.size :]))
                else:
                    self.messages.append(json.loads(data))
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.closed_at_s.append(time.monotonic())


class FakeEncoder:
    """An encoder that returns a fixed-size unit per frame and remembers what it was asked.

    Attributes:
    - frames: Every frame handed to encode(), in order.
    - forced: How many frames were forced to be keyframes.
    - closed: True once close() was called.
    """

    def __init__(self, unit_bytes: int = 64, gop: int = 30) -> None:
        self.frames: list[np.ndarray] = []
        self.forced = 0
        self.closed = False
        self._unit_bytes = unit_bytes
        self._gop = gop

    def encode(self, bgr: np.ndarray, force_idr: bool) -> tuple[bytes, bool]:
        """One unit per frame; a keyframe when forced or at the GOP boundary."""
        count = len(self.frames)
        self.frames.append(bgr)
        if force_idr:
            self.forced += 1
        return bytes([count % 256]) * self._unit_bytes, force_idr or count % self._gop == 0

    def close(self) -> None:
        """Marks the encoder closed."""
        self.closed = True


class Encoders:
    """A stand-in for `live._open_encoder` that keeps every `FakeEncoder` it opened.

    Attributes:
    - probe: The encoder the constructor opened and closed to prove one opens, or None before.
    - built: The streaming encoders, in the order they opened (the probe is not among them).
    - sizes: The `(width_px, height_px)` each streaming encoder was opened with.
    """

    def __init__(self, unit_bytes: int = 64, kind: type[FakeEncoder] = FakeEncoder) -> None:
        self.probe: FakeEncoder | None = None
        self.built: list[FakeEncoder] = []
        self.sizes: list[tuple[int, int]] = []
        self._unit_bytes = unit_bytes
        self._kind = kind

    def __call__(self, encoder: str, width_px: int, height_px: int, fps: int, bitrate_kbps: int) -> FakeEncoder:
        opened = self._kind(self._unit_bytes)
        if self.probe is None:
            self.probe = opened
            return opened
        self.built.append(opened)
        self.sizes.append((width_px, height_px))
        return opened


@dataclasses.dataclass
class Rig:
    """One client over one fake relay and a fake encoder, driven as a control loop drives it.

    The rig's attributes are what its next stream() reports: change one, then stream.

    Attributes:
    - relay: The fake relay.
    - encoders: The fake encoder factory.
    - client: The client, linked once wait_linked() returns.
    - cameras: The cameras the rig streams, which the first stream() names.
    - status: The status the next stream() reports.
    - extras: The keywords the next stream() passes (`prompt`, `run`, `error`, `arms`, ...).
    - last_s: The capture time of the last frame streamed, or None before the first.
    """

    relay: FakeRelay
    encoders: Encoders
    client: dream.LiveViewClient
    cameras: tuple[str, ...]
    status: str = "running"
    extras: dict[str, Any] = dataclasses.field(default_factory=dict)
    last_s: float | None = None
    monkeypatch: pytest.MonkeyPatch | None = None

    def stream_once(self, frames: bool = False, captured_s: float | None = None, **extras: Any) -> None:
        """One stream() with the rig's status and extras; with `frames`, every camera gets a new frame captured at
        `captured_s` (now when None), which the client's capture clock (`live._capture_s`) is stepped to."""
        obs: dict[str, Any] = {}
        if frames:
            stamp = time.monotonic() if captured_s is None else captured_s
            assert self.monkeypatch is not None
            self.monkeypatch.setattr(live, "_capture_s", lambda: stamp)
            obs = {f"video.{name}": np.full(FRAME_SHAPE, 90, dtype=np.uint8) for name in self.cameras}
        self.client.stream(obs, self.status, **{**self.extras, **extras})

    def run_ticks(
        self, count: int, period_s: float, step_s: float | None = None, start_s: float | None = None, **extras: Any
    ) -> None:
        """Streams `count` frames from the test thread, `period_s` apart in wall time and `step_s` apart in capture
        time (equal when None); capture time continues one stride after the last frame (from 100.0 before any)."""
        stride_s = period_s if step_s is None else step_s
        if start_s is None:
            start_s = 100.0 if self.last_s is None else self.last_s + stride_s
        for index in range(count):
            self.last_s = start_s + index * stride_s
            self.stream_once(frames=True, captured_s=self.last_s, **extras)
            time.sleep(period_s)

    def wait_linked(self) -> None:
        """Waits until the client reports the link up."""
        wait_for(lambda: self.client.is_linked, "the link")

    def units_of(self, camera: str) -> list[tuple[Header, bytes]]:
        """The received units of one camera, by its index in the client's cameras."""
        index = self.cameras.index(camera)
        return [(header, payload) for header, payload in self.relay.units if header.camera == index]


@pytest.fixture
def rig(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., Rig]]:
    """A `Rig` factory with fast reconnects and fake encoders; everything built is closed at teardown."""
    monkeypatch.setattr(live, "_RECONNECT_MIN_S", 0.05)
    monkeypatch.setattr(live, "_RECONNECT_MAX_S", 0.4)
    built: list[Rig] = []

    def factory(
        subscribed: tuple[str, ...] = ("top",),
        refuse: tuple[int, str] | None = None,
        face_blur: bool = False,
        unit_bytes: int = 64,
        status: str = "running",
        cameras: tuple[str, ...] = CAMERAS,
        encoder_kind: type[FakeEncoder] = FakeEncoder,
        width_px: int = 64,
        fps: int = RIG_FPS,
        dream_client: Any = None,
        model_fixed: bool = True,
    ) -> Rig:
        relay = FakeRelay(subscribed=subscribed, refuse=refuse)
        relay.start()
        encoders = Encoders(unit_bytes, encoder_kind)
        monkeypatch.setattr(live, "_open_encoder", encoders)
        # The encoding settings are the platform's, not the caller's: a small width and a known rate keep tests quick.
        monkeypatch.setattr(live, "_WIDTH_PX", width_px)
        monkeypatch.setattr(live, "_FPS", fps)
        monkeypatch.setenv("DREAM_API_URL", relay.url)
        client = dream.LiveViewClient(
            ROBOT,
            api_key=KEY,
            face_blur=face_blur,
            dream_client=dream_client,
            model_fixed=model_fixed,
            cameras=cameras,
        )
        parts = Rig(relay, encoders, client, cameras, status=status, monkeypatch=monkeypatch)
        built.append(parts)
        parts.stream_once()
        return parts

    yield factory
    for parts in built:
        parts.client.close()
        parts.relay.stop()


@pytest.fixture
def relay(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeRelay]:
    """A fake relay that admits the link, for the tests that build their own client; `DREAM_API_URL` names it."""
    server = FakeRelay()
    server.start()
    monkeypatch.setenv("DREAM_API_URL", server.url)
    yield server
    server.stop()


def streaming(relay: FakeRelay, status: str = "running", **kwargs: Any) -> dream.LiveViewClient:
    """A client linked to `relay` (the `relay` fixture, which `DREAM_API_URL` names), streaming the two-camera
    observation once."""
    client = dream.LiveViewClient(ROBOT, api_key=KEY, **kwargs)
    client.stream(observation(), status)
    wait_for(lambda: relay.messages != [], "the first status")
    return client


def subscribe(parts: Rig, camera: str) -> None:
    """Sends a viewer's `subscribe` and waits until the client has it."""
    parts.relay.send({"type": "subscribe", "camera": camera})
    wait_for(lambda: camera in parts.client.stats()["streams"], f"the subscribe to {camera}")


# T-1


def design_constants() -> dict[str, float]:
    """The constants table as `{name: value}`."""
    section = DESIGN_DOC.read_text(encoding="utf-8").split("### 8.4", 1)[1].split("\n## ", 1)[0]
    table: dict[str, float] = {}
    for line in section.splitlines():
        row = TABLE_ROW.match(line)
        if row is None:
            continue
        names = [name for name in row.group(1, 2) if name]
        for index, part in enumerate(row.group(3).split(" / ")):
            words = part.strip().split(" ")
            number = words[0]
            if not number.replace(".", "").isdigit():
                continue
            name = names[index] if len(names) > 1 else names[0]
            unit = UNITS.get(words[1], 1) if len(words) > 1 else 1
            table[name] = float(number) if "." in number else int(number) * unit
    return table


def test_agent_constants_equal_the_design_table_I11() -> None:
    """Every client knob in the table, and the encoding defaults it restates from the contract, equals the table."""
    if not DESIGN_DOC.exists():
        pytest.skip(f"{DESIGN_DOC} is not in this copy of the repository (a published product carries no docs)")
    table = design_constants()
    for name, attribute in CLIENT_CONSTANTS.items():
        assert getattr(live, attribute) == table[name], name
    assert live._YIELD_TICK_RATIO == 2.0 and live._NICE == 10


# The link, the frames, the encoders (T-2)


def test_hello_carries_the_config_and_the_key_rides_the_upgrade_A1_I4(rig: Callable[..., Rig]) -> None:
    """The upgrade carries `Authorization: Bearer <api_key>`; `hello` names the robot, the cameras as given, the size,
    the rate, the codec, the blur flag, the version and the time zone; `hello_ok` marks the client linked."""
    parts = rig(face_blur=True, fps=30)
    parts.wait_linked()

    assert parts.relay.auth_headers == [f"Bearer {KEY}"]
    hello = parts.relay.hellos[0]
    assert hello["protocol"] == 1 and hello["robot_id"] == ROBOT
    assert hello["cameras"] == list(CAMERAS) and hello["width_px"] == 64 and hello["fps"] == 30
    assert hello["codec"] == "vp8" and hello["face_blur"] is True and hello["pose"] is False
    assert re.fullmatch(r"dream-client/\d+\.\d+\.\d+", hello["agent_version"])
    assert hello["timezone"] == live._local_timezone() and hello["timezone"]
    stats = parts.client.stats()
    assert stats["linked"] is True and stats["node"] == "live-1" and stats["streams"] == ["top"]


def test_the_local_timezone_is_the_zone_name_or_a_fixed_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    """`TZ` wins, then the `/etc/localtime` link's zoneinfo path, then the clock's offset in `UTC+HH:MM` form."""
    monkeypatch.setenv("TZ", ":America/Los_Angeles")
    assert live._local_timezone() == "America/Los_Angeles"
    monkeypatch.delenv("TZ")
    monkeypatch.setattr(live.os, "readlink", lambda path: "/usr/share/zoneinfo/Europe/Berlin")
    assert live._local_timezone() == "Europe/Berlin"
    monkeypatch.setattr(live.os, "readlink", lambda path: "/etc/localtime.real")
    assert re.fullmatch(r"UTC[+-]\d\d:\d\d", live._local_timezone())

    def refuse(path: str) -> str:
        raise OSError("not a link")

    monkeypatch.setattr(live.os, "readlink", refuse)
    assert re.fullmatch(r"UTC[+-]\d\d:\d\d", live._local_timezone())


def test_one_unit_per_new_frame_none_per_repeated_timestamp_V3_V4(rig: Callable[..., Rig]) -> None:
    """Twenty frames at 20 Hz (slower than the poll, so every one is seen) make exactly twenty units for the watched
    camera, with consecutive `seq`, the capture time on the wall clock, the camera index of `hello`, and a keyframe
    first; a repeated capture time adds none."""
    parts = rig(subscribed=("left",))
    parts.wait_linked()

    parts.run_ticks(20, period_s=0.05, start_s=100.0)
    wait_for(lambda: len(parts.units_of("left")) == 20, "twenty units")
    time.sleep(0.1)
    parts.stream_once(frames=True, captured_s=100.0 + 19 * 0.05)
    time.sleep(0.1)

    units = parts.units_of("left")
    assert len(units) == 20 and parts.relay.units == units
    headers = [header for header, _ in units]
    assert [header.seq for header in headers] == list(range(20))
    assert headers[0].is_keyframe and headers[0].camera == CAMERAS.index("left")
    expected_us = int((100.0 + (time.time() - time.monotonic())) * 1e6)
    assert abs(headers[0].capture_us - expected_us) < 2_000_000
    assert headers[1].capture_us - headers[0].capture_us == 50_000
    assert all(header.version == 1 and header.kind == 1 for header in headers)
    assert parts.encoders.built[0].frames[0].shape == FRAME_SHAPE


def test_nothing_is_encoded_or_sent_until_the_relay_subscribes_V2_I2(rig: Callable[..., Rig]) -> None:
    """No encoder opens and no unit leaves while nothing is watched; a `subscribe` opens one, the current frame goes
    at once as a keyframe, and the camera streams."""
    parts = rig(subscribed=())
    parts.wait_linked()

    parts.run_ticks(10, period_s=0.02)
    time.sleep(0.1)
    assert parts.encoders.built == [] and parts.relay.units == []

    parts.relay.send({"type": "subscribe", "camera": "right"})
    wait_for(lambda: len(parts.units_of("right")) == 1, "the current frame, sent at once")
    assert parts.units_of("right")[0][0].is_keyframe
    parts.run_ticks(5, period_s=0.02)
    wait_for(lambda: len(parts.units_of("right")) == 6, "five more units")
    time.sleep(0.1)
    assert len(parts.units_of("right")) == 6
    assert len(parts.encoders.built) == 1 and parts.units_of("top") == []


def test_unsubscribe_closes_the_encoder_after_the_idle_time_V2(
    rig: Callable[..., Rig], monkeypatch: pytest.MonkeyPatch
) -> None:
    """After `unsubscribe` no frame is encoded, and the encoder is closed `_IDLE_ENCODER_S` later, not before."""
    monkeypatch.setattr(live, "_IDLE_ENCODER_S", 0.3)
    parts = rig()
    parts.wait_linked()
    parts.run_ticks(3, period_s=0.02)
    wait_for(lambda: len(parts.relay.units) == 3, "three units")
    encoder = parts.encoders.built[0]

    parts.relay.send({"type": "unsubscribe", "camera": "top"})
    wait_for(lambda: parts.client.stats()["streams"] == [], "the unsubscribe")
    parts.run_ticks(3, period_s=0.02)
    time.sleep(0.05)
    assert len(encoder.frames) == 3 and not encoder.closed
    wait_for(lambda: encoder.closed, "the idle close")
    assert len(parts.relay.units) == 3


def test_a_stalled_uplink_overwrites_the_pending_unit_and_forces_an_idr_V4_G5(rig: Callable[..., Rig]) -> None:
    """While the relay stops reading, the client keeps one pending unit per camera: the rest are dropped and
    counted, `seq` skips exactly that many, and the first unit after the gap is a keyframe. With P-frame-sized units
    (6 KiB) the first drop comes within `STALL_DROP_WITHIN_UNITS` units of the stall, because the socket is bounded
    (`_WRITE_HIGH_WATER`, `_NOTSENT_LOWAT`, `_SNDBUF_BYTES`) instead of autotuned to megabytes."""
    parts = rig(unit_bytes=6 * 1024)
    parts.wait_linked()
    parts.run_ticks(2, period_s=0.02)
    wait_for(lambda: len(parts.relay.units) == 2, "two units")
    encoder = parts.encoders.built[0]

    parts.relay.is_stalled = True
    deadline_s = time.monotonic() + WAIT_S
    while parts.client.stats()["dropped_frames"]["top"] == 0 and time.monotonic() < deadline_s:
        parts.run_ticks(10, period_s=0.01)
    assert parts.client.stats()["dropped_frames"]["top"] > 0, "a drop"
    encoded_at_first_drop = len(encoder.frames) - 2
    assert encoded_at_first_drop <= STALL_DROP_WITHIN_UNITS, encoded_at_first_drop
    parts.relay.is_stalled = False
    parts.run_ticks(5, period_s=0.05)
    assert parts.last_s is not None
    last_us = parts.client._wall_us(parts.last_s)
    wait_for(lambda: parts.relay.units[-1][0].capture_us >= last_us, "the drain")

    dropped = parts.client.stats()["dropped_frames"]["top"]
    headers = [header for header, _ in parts.relay.units]
    seqs = [header.seq for header in headers]
    gaps = [
        (index, seqs[index + 1] - seqs[index] - 1)
        for index in range(len(seqs) - 1)
        if seqs[index + 1] - seqs[index] > 1
    ]
    assert gaps, "the stall dropped nothing"
    assert sum(gap for _, gap in gaps) == dropped
    assert all(headers[index + 1].is_keyframe for index, _ in gaps)
    assert parts.encoders.built[0].forced >= len(gaps) + 1


def test_the_fake_loop_keeps_its_tick_rate_while_the_agent_encodes_I1_V3(rig: Callable[..., Rig]) -> None:
    """The caller's loop streams at 50 Hz while the client encodes every frame on its own threads; no frame is lost,
    and the loop's wall-clock p95 interval stays under twice the period: stream() never waits on the encoder."""
    parts = rig()
    parts.wait_linked()

    intervals_s = []
    last_s = time.monotonic()
    for index in range(100):
        parts.stream_once(frames=True, captured_s=100.0 + index * 0.02)
        time.sleep(0.02)
        now_s = time.monotonic()
        intervals_s.append(now_s - last_s)
        last_s = now_s
    wait_for(lambda: len(parts.relay.units) == 100, "a hundred units")

    intervals_s.sort()
    assert intervals_s[int(0.95 * len(intervals_s))] < 0.04


def test_status_carries_the_round_trips_at_most_every_timing_period_P8(
    rig: Callable[..., Rig], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The next status of a run carries the dream client's round trip, a new one sends another on its own once
    `_TIMING_STATUS_S` has passed, each status lists every round trip since the previous one, and the block is gone
    between runs."""
    model = FakeDreamClient()
    parts = rig(subscribed=(), dream_client=model)
    parts.wait_linked()
    wait_for(lambda: len(parts.relay.texts("status")) == 1, "the first status")
    monkeypatch.setattr(live, "_TIMING_STATUS_S", 0.3)
    model.answer(530.0)
    parts.stream_once()
    wait_for(lambda: "timing" in parts.relay.texts("status")[-1], "a status with the round trips")
    sent = parts.relay.texts("status")[-1]["timing"]
    assert (sent["rtt_ms"], sent["count"], sent["recent_ms"], sent["up_mbps"]) == (530.0, 1, [530.0], 12.5)
    before = len(parts.relay.texts("status"))
    monkeypatch.setattr(live, "_TIMING_STATUS_S", 1.0)
    time.sleep(0.3)
    model.answer(610.0)
    parts.stream_once()
    wait_for(lambda: parts.relay.texts("status")[-1]["timing"]["rtt_ms"] == 610.0, "the next round trip")
    for _ in range(6):
        parts.stream_once()
        time.sleep(0.05)
    assert len(parts.relay.texts("status")) == before + 1, "one status per new round trip, not one per call"
    assert parts.relay.texts("status")[-1]["timing"]["recent_ms"] == [610.0], "each round trip is listed once"

    # Two round trips back inside one period: the next status lists both, in order.
    model.answer(640.0)
    parts.stream_once()
    time.sleep(0.2)
    model.answer(420.0)
    for _ in range(60):
        parts.stream_once()
        time.sleep(0.05)
        if parts.relay.texts("status")[-1]["timing"]["count"] == 4:
            break
    assert parts.relay.texts("status")[-1]["timing"]["recent_ms"] == [640.0, 420.0]
    parts.status = "idle"
    parts.stream_once()
    wait_for(lambda: parts.relay.texts("status")[-1]["state"] == "idle", "the idle status")
    assert "timing" not in parts.relay.texts("status")[-1]


def test_status_is_sent_after_hello_and_on_every_station_change_O2(rig: Callable[..., Rig]) -> None:
    """A `status` follows `hello_ok`; a state, prompt or run change sends another with the new values and the drop
    counts; stats() carries the client's counters."""
    parts = rig(subscribed=())
    parts.extras = {"prompt": "pick up the cup", "run": 1}
    parts.stream_once()
    parts.wait_linked()
    wait_for(lambda: len(parts.relay.texts("status")) == 1, "the first status")
    first = parts.relay.texts("status")[0]
    assert first["state"] == "running" and first["prompt"] == "pick up the cup" and first["run"] == 1
    assert first["dropped_frames"] == {name: 0 for name in CAMERAS} and "live_paused" not in first

    parts.status = "stopping"
    parts.stream_once()
    wait_for(lambda: len(parts.relay.texts("status")) == 2, "the second status")
    parts.status = "idle"
    parts.extras["prompt"] = "fold the towel"
    parts.stream_once()
    wait_for(lambda: len(parts.relay.texts("status")) == 3, "the third status")
    assert [status["state"] for status in parts.relay.texts("status")] == ["running", "stopping", "idle"]
    assert parts.relay.texts("status")[-1]["prompt"] == "fold the towel"
    assert set(parts.client.stats()) == {
        "linked",
        "node",
        "viewers",
        "streams",
        "dropped_frames",
        "reconnects",
        "errors",
        "yields",
        "live_paused",
        "pose",
    }


def test_the_dreamapi_error_the_loop_retries_and_the_one_that_ended_the_run_reach_the_relay_as_status_error(
    rig: Callable[..., Rig],
) -> None:
    """While the loop retries a refusal `status.error` is its text; once the model answers, the next `status` has no
    `error` key at all (absent, never null: a relay older than the field forbids the key). In `error` it is the
    message of what ended the run, cut to the wire's bound so the relay never refuses the `status`."""
    parts = rig(subscribed=())
    parts.wait_linked()
    wait_for(lambda: len(parts.relay.texts("status")) == 1, "the first status")
    assert "error" not in parts.relay.texts("status")[0]

    refusal = "DreamAPI refused the inference request: DA-2000 NO_CAPACITY (pool_warming)"
    parts.stream_once(error=refusal)
    wait_for(lambda: len(parts.relay.texts("status")) == 2, "the refusal's status")
    sent = parts.relay.texts("status")[1]
    assert sent["state"] == "running" and sent["error"] == refusal

    parts.stream_once()
    wait_for(lambda: len(parts.relay.texts("status")) == 3, "the cleared status")
    assert "error" not in parts.relay.texts("status")[-1]

    parts.status = "error"
    parts.stream_once(error=("DA-1000 " + "x" * 2000, "DA-1000"))
    wait_for(lambda: len(parts.relay.texts("status")) == 4, "the fatal status")
    fatal = parts.relay.texts("status")[3]["error"]
    assert fatal.startswith("DA-1000 ") and len(fatal) == live._STATUS_ERROR_MAX_CHARS
    assert parts.relay.connections == 1, "the relay never refused a status"


def test_frames_are_not_encoded_while_the_station_is_not_running(rig: Callable[..., Rig]) -> None:
    """A watched camera with no frame yet (a robot whose cameras are not open) makes no unit; once a run starts with
    frames, the camera streams from its first frame."""
    parts = rig(status="idle")
    parts.wait_linked()
    for _ in range(5):
        parts.stream_once()
        time.sleep(0.02)
    time.sleep(0.1)
    assert parts.relay.units == [] and parts.encoders.built == []

    parts.status = "running"
    parts.run_ticks(1, period_s=0.03)
    wait_for(lambda: len(parts.relay.units) == 1, "the run's first frame")
    parts.run_ticks(5, period_s=0.03)
    wait_for(lambda: len(parts.relay.units) == 6, "five more units")


def test_frames_stream_while_the_station_is_idle_when_its_cameras_are_open(rig: Callable[..., Rig]) -> None:
    """A loop that keeps streaming between runs streams the scene to a viewer while the status is idle (no run, no
    tick guard: nothing to protect); the status the viewer sees says `idle`."""
    parts = rig(status="idle")
    parts.wait_linked()
    parts.run_ticks(6, period_s=0.03)
    wait_for(lambda: len(parts.relay.units) >= 6, "six idle units")
    assert parts.client.yields == 0
    assert parts.relay.texts("status")[-1]["state"] == "idle"


def test_face_blur_scrubs_every_encoded_frame_V5(rig: Callable[..., Rig], monkeypatch: pytest.MonkeyPatch) -> None:
    """With `face_blur` on, every frame reaches the encoder through `shared.faceblur.FaceBlurrer.scrub_frame`; off,
    the frames reach it as captured (resized and BGR)."""
    from shared import faceblur

    class MarkingBlurrer:
        def scrub_frame(self, frame: np.ndarray) -> np.ndarray:
            return np.full_like(frame, 7)

    monkeypatch.setattr(faceblur, "FaceBlurrer", MarkingBlurrer)
    blurred = rig(face_blur=True)
    blurred.wait_linked()
    blurred.run_ticks(3, period_s=0.02)
    wait_for(lambda: len(blurred.relay.units) == 3, "three blurred units")
    assert all(int(frame[0, 0, 0]) == 7 for frame in blurred.encoders.built[0].frames)

    raw = rig(face_blur=False)
    raw.wait_linked()
    raw.run_ticks(3, period_s=0.02)
    wait_for(lambda: len(raw.relay.units) == 3, "three raw units")
    assert all(int(frame[0, 0, 0]) == 90 for frame in raw.encoders.built[0].frames)


def test_face_blur_is_on_by_default_and_hello_says_so_V5(relay: FakeRelay) -> None:
    """A client built without `face_blur` blurs: `hello` declares it, so a public robot may be shown."""
    with streaming(relay) as client:
        assert client._face_blur is True
        assert relay.hellos[0]["face_blur"] is True


def test_each_camera_blurs_with_a_detector_of_its_own_V5(
    rig: Callable[..., Rig], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A detector takes its calls under a lock, so one shared by the cameras ran their detections in turn and a third
    camera waited on the other two; three watched cameras build three, and each only ever scrubs its own frames."""
    from shared import faceblur

    built: list[Any] = []

    class RecordingBlurrer:
        def __init__(self) -> None:
            self.shapes: list[tuple[int, ...]] = []
            built.append(self)

        def scrub_frame(self, frame: np.ndarray) -> np.ndarray:
            self.shapes.append(frame.shape)
            return frame

    monkeypatch.setattr(faceblur, "FaceBlurrer", RecordingBlurrer)
    parts = rig(subscribed=CAMERAS, face_blur=True)
    parts.wait_linked()
    parts.run_ticks(3, period_s=0.03)
    wait_for(lambda: len(parts.relay.units) == 3 * len(CAMERAS), "three units per camera")
    blurrers = [parts.client._streams[camera].blurrer for camera in CAMERAS]
    assert len(built) == len(CAMERAS) and len({id(blurrer) for blurrer in blurrers}) == len(CAMERAS)
    assert all(len(blurrer.shapes) == 3 for blurrer in blurrers)


## What a `HeldEncoder` waits on.
RELEASE = threading.Event()


class HeldEncoder(FakeEncoder):
    """A `FakeEncoder` whose encode() waits for #RELEASE, so a test can hold the worker busy."""

    def encode(self, bgr: np.ndarray, force_idr: bool) -> tuple[bytes, bool]:
        """Waits for the release, then encodes."""
        RELEASE.wait(WAIT_S)
        return super().encode(bgr, force_idr)


def test_a_frame_that_finds_the_worker_busy_waits_and_only_a_newer_one_replaces_it_V4(
    rig: Callable[..., Rig],
) -> None:
    """The blur takes most of a frame period and the poll picks a frame up as much as a poll late, so two frames can
    reach the worker closer together than it takes: the second waits and is encoded when the worker returns, no drop.
    A frame arriving while one already waits replaces it: one drop, one `seq` gap, and the next unit is a keyframe."""
    RELEASE.clear()
    parts = rig(encoder_kind=HeldEncoder)
    parts.wait_linked()
    stream = parts.client._streams["top"]
    parts.run_ticks(1, period_s=0.03)
    wait_for(lambda: stream.is_busy, "the worker on the first frame")
    parts.run_ticks(1, period_s=0.03)
    wait_for(lambda: stream.waiting is not None, "the second frame waiting")
    RELEASE.set()
    wait_for(lambda: len(parts.units_of("top")) == 2, "both frames")
    assert stream.dropped == 0 and [header.seq for header, _ in parts.units_of("top")] == [0, 1]

    RELEASE.clear()
    parts.run_ticks(1, period_s=0.03)
    wait_for(lambda: stream.is_busy, "the worker on the third frame")
    parts.run_ticks(2, period_s=0.03)
    wait_for(lambda: stream.dropped == 1, "the fourth frame replaced by the fifth")
    RELEASE.set()
    wait_for(lambda: len(parts.units_of("top")) == 4, "the third and the fifth")
    headers = [header for header, _ in parts.units_of("top")]
    # The replaced frame takes its `seq` when it is replaced, before the frame on the worker lands, as a frame
    # dropped while the worker was busy always did: one gap, counted once.
    assert [header.seq for header in headers] == [0, 1, 3, 4]
    assert headers[-1].is_keyframe


def test_frames_are_resized_to_the_configured_width_at_the_capture_aspect_V1(rig: Callable[..., Rig]) -> None:
    """`width_px` 32 on 64x48 captures opens the encoder at 32x24 and hands it 24x32x3 frames."""
    parts = rig(width_px=32)
    parts.wait_linked()
    parts.run_ticks(2, period_s=0.02)
    wait_for(lambda: len(parts.relay.units) == 2, "two units")
    assert parts.encoders.sizes == [(32, 24)]
    assert parts.encoders.built[0].frames[0].shape == (24, 32, 3)


def test_a_base64_jpeg_is_decoded_resized_and_encoded_V1(rig: Callable[..., Rig]) -> None:
    """A frame given as a base64 JPEG, the form an observation carries, reaches the encoder decoded and resized."""
    parts = rig(status="idle")
    parts.wait_linked()
    parts.client.stream({"video.top": jpeg(120)}, "idle")
    wait_for(lambda: len(parts.relay.units) == 1, "the unit")
    frame = parts.encoders.built[0].frames[0]
    assert frame.shape == (36, 64, 3) and abs(int(frame[0, 0, 0]) - 120) <= 2


def test_an_encoder_fault_reopens_the_encoder_on_the_first_frame_after_the_delay_V2(
    rig: Callable[..., Rig], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An encoder that raises is closed and `errors` counts it; the camera stays watched (the relay repeats no
    `subscribe` while a viewer stays), no frame is encoded for `_IDLE_ENCODER_S`, then the first frame after it opens
    a fresh encoder and is a keyframe, with no message from the relay; the link stays up."""
    monkeypatch.setattr(live, "_IDLE_ENCODER_S", 0.3)
    parts = rig()
    parts.wait_linked()
    parts.run_ticks(2, period_s=0.02)
    wait_for(lambda: len(parts.relay.units) == 2, "two units")
    encoder = parts.encoders.built[0]

    def explode(bgr: np.ndarray, force_idr: bool) -> tuple[bytes, bool]:
        raise RuntimeError("codec died")

    encoder.encode = explode  # type: ignore[method-assign]
    parts.run_ticks(2, period_s=0.02)
    # The client counts the error, logs the traceback, then closes the encoder; the count alone can be seen from
    # this thread before the close has run, so the wait covers both.
    wait_for(lambda: parts.client.errors == 1 and encoder.closed, "the error and the close")
    assert parts.client.stats()["streams"] == ["top"] and parts.client.is_linked
    parts.run_ticks(4, period_s=0.02)
    time.sleep(0.05)
    assert len(parts.encoders.built) == 1 and len(parts.relay.units) == 2
    deadline_s = time.monotonic() + WAIT_S
    while len(parts.encoders.built) < 2 and time.monotonic() < deadline_s:
        parts.run_ticks(1, period_s=0.05)
    assert len(parts.encoders.built) == 2, "a fresh encoder, unprompted, once the delay is over"
    wait_for(lambda: len(parts.relay.units) == 3, "the current frame on the fresh encoder")
    assert parts.relay.units[2][0].is_keyframe
    parts.run_ticks(3, period_s=0.05)
    wait_for(lambda: len(parts.relay.units) == 6, "three more units on the fresh encoder")
    assert parts.client.errors == 1 and parts.client.stats()["streams"] == ["top"]


def test_an_idle_encoder_closes_during_the_reconnect_backoff_V2(
    rig: Callable[..., Rig], monkeypatch: pytest.MonkeyPatch
) -> None:
    """After the link drops and every redial is refused, the encoder the drop idled closes at the first backoff
    expiry after `_IDLE_ENCODER_S`, not when a link is next served."""
    monkeypatch.setattr(live, "_IDLE_ENCODER_S", 0.3)
    parts = rig()
    parts.wait_linked()
    parts.run_ticks(2, period_s=0.02)
    wait_for(lambda: len(parts.relay.units) == 2, "two units")
    encoder = parts.encoders.built[0]

    parts.relay.refuse = (4001, "unauthorized")
    parts.relay.close_link(4010, "draining")
    wait_for(lambda: not parts.client.is_linked, "unlinked")
    wait_for(lambda: encoder.closed, "the idle close while redialing")
    assert not parts.client.is_linked and parts.relay.connections >= 2


def test_a_camera_faster_than_live_fps_is_thinned_to_live_fps_V1(rig: Callable[..., Rig]) -> None:
    """With `fps` 30, sixty frames 1/60 s apart in capture time make thirty units (consecutive `seq`, none counted
    dropped), and thirty frames exactly 1/30 s apart make thirty: the cap thins a fast camera and never a loop at the
    rate. The wall period stays under the 60 Hz poll so every frame is seen."""
    parts = rig(fps=30)
    parts.wait_linked()

    parts.run_ticks(60, period_s=0.02, step_s=1 / 60, start_s=100.0)
    wait_for(lambda: len(parts.relay.units) >= 29, "thirty units")
    time.sleep(0.1)
    thinned = len(parts.relay.units)
    assert 29 <= thinned <= 31 and parts.client.stats()["dropped_frames"]["top"] == 0
    assert [header.seq for header, _ in parts.relay.units] == list(range(thinned))

    parts.run_ticks(30, period_s=0.02, step_s=1 / 30)
    wait_for(lambda: len(parts.relay.units) >= thinned + 30, "thirty more units")
    time.sleep(0.1)
    assert len(parts.relay.units) == thinned + 30


# Reconnects (T-2)


def test_a_dropped_link_is_redialed_with_backoff_and_the_subscriptions_restored_R8(
    rig: Callable[..., Rig], caplog: pytest.LogCaptureFixture
) -> None:
    """A close from the relay is followed by a new upgrade after the backoff, `reconnects` counts it, the encoders
    keep streaming the cameras `hello_ok` names, and the fault is logged once at WARNING."""
    caplog.set_level(logging.DEBUG, logger="dream.live")
    parts = rig()
    parts.wait_linked()
    parts.run_ticks(2, period_s=0.02)
    wait_for(lambda: len(parts.relay.units) == 2, "two units")

    parts.relay.close_link(4010, "draining")
    wait_for(lambda: not parts.client.is_linked, "unlinked")
    assert parts.client.stats()["linked"] is False
    wait_for(lambda: parts.relay.connections == 2 and parts.client.is_linked, "the second link")
    assert parts.client.reconnects == 1 and parts.client.errors == 1
    wait_for(lambda: len(parts.relay.units) == 3, "the current frame on the new link")
    assert parts.relay.units[2][0].is_keyframe
    parts.run_ticks(2, period_s=0.02)
    wait_for(lambda: len(parts.relay.units) == 5, "units on the new link")
    assert len(parts.encoders.built) == 1

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "link lost" in r.message]
    assert len(warnings) == 1 and "4010" in warnings[0].message


def test_an_authorize_refusal_is_retried_at_the_maximum_backoff_and_logged_once(
    rig: Callable[..., Rig], caplog: pytest.LogCaptureFixture
) -> None:
    """`4001` from the relay: the next attempt waits `_RECONNECT_MAX_S`, not the minimum, and repeats of the same
    fault are DEBUG lines, not WARNINGs."""
    caplog.set_level(logging.DEBUG, logger="dream.live")
    parts = rig(refuse=(4001, "unauthorized"))
    wait_for(lambda: parts.relay.connections == 3, "three attempts", timeout_s=WAIT_S * 2)

    gaps = [
        later - earlier for earlier, later in zip(parts.relay.closed_at_s, parts.relay.closed_at_s[1:], strict=False)
    ]
    assert all(gap >= live._RECONNECT_MAX_S * 0.95 for gap in gaps[:2]), gaps
    assert parts.client.is_linked is False and parts.client.errors >= 2
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "link lost" in r.message]
    assert len(warnings) == 1 and "4001" in warnings[0].message


def test_a_refused_link_waits_before_redialing_and_close_ends_the_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 4001 (unknown key) is redialed after the long backoff, and close() does not wait it out.

    The loop keeps streaming at 30 Hz meanwhile, as a control loop does: that must not cut the backoff short (on a
    Maker Arm station it did, and the client redialed at the frame rate).
    """
    relay = FakeRelay(refuse=(4001, "unauthorized"))
    relay.start()
    monkeypatch.setenv("DREAM_API_URL", relay.url)
    client = dream.LiveViewClient(ROBOT, api_key=KEY)
    obs = observation()
    client.stream(obs, "idle")
    wait_for(lambda: relay.connections == 1, "the first dial")
    for _ in range(45):
        client.stream(obs, "idle")
        time.sleep(1 / 30)
    assert relay.connections == 1
    started_s = time.monotonic()
    client.close()
    assert time.monotonic() - started_s < 1.0
    relay.stop()


def test_a_link_whose_close_never_finishes_is_aborted_and_redialed(
    rig: Callable[..., Rig], monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a network path that went dead the closing handshake waits behind bytes nobody acknowledges, and websockets
    waits for them with no deadline (seen on a Maker Arm whose Wi-Fi dropped while its Ethernet came up: the client
    sat on the dead link for minutes). A close that has not finished in `_LINK_CLOSE_S` is aborted and the client
    redials."""
    from websockets.asyncio import client as ws_client

    async def close_forever(self: Any, code: int = 1000, reason: str = "") -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(live, "_LINK_CLOSE_S", 0.3)
    parts = rig()
    parts.wait_linked()
    monkeypatch.setattr(ws_client.ClientConnection, "close", close_forever)
    parts.relay.send_text('{"type": "hello_ok"}')
    wait_for(lambda: parts.relay.connections == 2 and parts.client.is_linked, "the redial past the stuck close")
    assert parts.client.reconnects == 1


def test_a_malformed_message_closes_the_link_with_protocol_and_the_agent_redials(rig: Callable[..., Rig]) -> None:
    """A text frame that is not a robot-link message closes the link with `4002`; the client then reconnects."""
    parts = rig()
    parts.wait_linked()
    parts.relay.send_text('{"type": "hello_ok"}')
    wait_for(lambda: parts.relay.connections == 2 and parts.client.is_linked, "the redial")
    assert parts.client.reconnects == 1


# Prompts (T-4)


def answer(state: str, error: str | None = None) -> dict[str, Any]:
    """What a station's callback resolves to: the state after the press, and the refusal's words."""
    return {"state": state} if error is None else {"state": state, "error": error}


def test_a_prompt_reaches_set_prompt_on_the_station_loop_and_is_acknowledged_P1(rig: Callable[..., Rig]) -> None:
    """`prompt {id, text}` reaches a callback registered for `prompt` as `EventChange.prompt` with get_prompt() holding
    the text, its awaitable runs, and `prompt_ack` carries the id, `ok: true` and the state it resolved to."""
    parts = rig()
    parts.wait_linked()
    seen: list[tuple[dream.EventChange, str | None]] = []

    async def apply() -> dict[str, Any]:
        return answer("running")

    def on_press(event: dream.EventChange) -> Any:
        seen.append((event, parts.client.get_prompt()))
        return apply()

    parts.client.register_callback(on_press, prompt=True)
    parts.relay.send({"type": "prompt", "id": "prm_1", "text": "fold the towel"})
    wait_for(lambda: "prm_1" in parts.relay.acks(), "the ack")
    assert parts.relay.acks()["prm_1"] == {"type": "prompt_ack", "id": "prm_1", "ok": True, "state": "running"}
    assert seen == [(dream.EventChange.prompt, "fold the towel")]


@pytest.mark.parametrize("raised", [False, True], ids=["answered", "raised"])
def test_a_refused_prompt_is_acknowledged_false_with_the_message_P1(rig: Callable[..., Rig], raised: bool) -> None:
    """A callback that refuses, with an `error` in what it resolves to or by raising, is `ok: false` with the words
    and the state; the link stays up."""
    parts = rig()
    parts.wait_linked()

    async def refuse() -> dict[str, Any]:
        if raised:
            raise ValueError("prompt must be a non-empty string, got '   '")
        return answer("running", "prompt refused: homing")

    parts.client.register_callback(lambda event: refuse(), prompt=True)
    parts.relay.send({"type": "prompt", "id": "prm_2", "text": "anything"})
    wait_for(lambda: "prm_2" in parts.relay.acks(), "the ack")
    expected = "ValueError: prompt must be a non-empty string, got '   '" if raised else "prompt refused: homing"
    assert parts.relay.acks()["prm_2"] == {
        "type": "prompt_ack",
        "id": "prm_2",
        "ok": False,
        "state": "running",
        "error": expected,
    }
    assert parts.client.is_linked


def test_an_unanswered_prompt_times_out_after_prompt_ack_s_P5(
    rig: Callable[..., Rig], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new task whose work never ends is answered `ok: false, error: "timeout"` after `_PROMPT_ACK_S`, and the work
    is cancelled (a task may be; a start, stop, home or reset never is)."""
    monkeypatch.setattr(live, "_PROMPT_ACK_S", 0.2)
    parts = rig()
    parts.wait_linked()
    hung: concurrent.futures.Future[Any] = concurrent.futures.Future()
    parts.client.register_callback(lambda event: hung, prompt=True)

    started_s = time.monotonic()
    parts.relay.send({"type": "prompt", "id": "prm_3", "text": "wait forever"})
    wait_for(lambda: "prm_3" in parts.relay.acks(), "the timeout ack")
    ack = parts.relay.acks()["prm_3"]
    assert ack["ok"] is False and ack["error"] == "timeout" and ack["state"] == "running"
    assert 0.2 <= time.monotonic() - started_s < WAIT_S
    wait_for(hung.cancelled, "the task's work cancelled")


def test_a_prompt_over_the_bound_is_a_protocol_error_P3(rig: Callable[..., Rig]) -> None:
    """A `prompt` longer than `PROMPT_MAX_CHARS` never reaches a callback; the link is closed with `4002` and
    redialed."""
    parts = rig()
    parts.wait_linked()
    events: list[dream.EventChange] = []
    parts.client.register_callback(events.append, prompt=True)
    parts.relay.send_text('{"type": "prompt", "id": "prm_4", "text": "' + "x" * (live._PROMPT_MAX_CHARS + 1) + '"}')
    wait_for(lambda: parts.relay.connections == 2 and parts.client.is_linked, "the redial")
    assert events == []


def test_a_start_prompt_starts_the_run_with_its_text_P1(rig: Callable[..., Rig]) -> None:
    """`prompt {action: "start", text}` is `EventChange.start` with get_prompt() holding the text; without a text the
    run starts on the task the status shows (the `prompt` the loop streams); the ack carries the state after it."""
    parts = rig(status="idle")
    parts.extras = {"prompt": "pick up the cup"}
    parts.stream_once()
    parts.wait_linked()
    seen: list[tuple[dream.EventChange, str | None]] = []

    def start(event: dream.EventChange) -> str:
        seen.append((event, parts.client.get_prompt()))
        return "running"

    parts.client.register_callback(start)
    parts.relay.send({"type": "prompt", "id": "prm_5", "action": "start", "text": "fold the towel"})
    wait_for(lambda: "prm_5" in parts.relay.acks(), "the ack")
    assert parts.relay.acks()["prm_5"] == {"type": "prompt_ack", "id": "prm_5", "ok": True, "state": "running"}
    parts.relay.send({"type": "prompt", "id": "prm_6", "action": "start"})
    wait_for(lambda: "prm_6" in parts.relay.acks(), "the second ack")
    assert seen == [(dream.EventChange.start, "fold the towel"), (dream.EventChange.start, "pick up the cup")]


def test_a_stop_prompt_stops_the_run_and_homes_the_arms_P1(rig: Callable[..., Rig]) -> None:
    """`prompt {action: "stop"}` is `EventChange.end`; the ack carries the state the stop and home move ended in."""
    parts = rig()
    parts.wait_linked()
    events: list[dream.EventChange] = []

    async def stop_and_home(event: dream.EventChange) -> dict[str, Any]:
        events.append(event)
        return answer("idle")

    parts.client.register_callback(stop_and_home, home=True, reset=True, prompt=True)
    parts.relay.send({"type": "prompt", "id": "prm_7", "action": "stop"})
    wait_for(lambda: "prm_7" in parts.relay.acks(), "the ack")
    assert parts.relay.acks()["prm_7"] == {"type": "prompt_ack", "id": "prm_7", "ok": True, "state": "idle"}
    assert events == [dream.EventChange.end]


def test_a_refused_start_is_acknowledged_false_with_the_message_P1(rig: Callable[..., Rig]) -> None:
    """A start the robot refuses (a run exists) is `ok: false` with its words, like a refused prompt."""
    parts = rig()
    parts.wait_linked()
    parts.client.register_callback(lambda event: answer("running", "start refused: station is running"))
    parts.relay.send({"type": "prompt", "id": "prm_8", "action": "start", "text": "again"})
    wait_for(lambda: "prm_8" in parts.relay.acks(), "the ack")
    assert parts.relay.acks()["prm_8"] == {
        "type": "prompt_ack",
        "id": "prm_8",
        "ok": False,
        "state": "running",
        "error": "start refused: station is running",
    }


def test_home_and_reset_prompts_run_the_controller_and_a_start_carries_its_checkpoint_P6(
    rig: Callable[..., Rig],
) -> None:
    """With `model_fixed=False` and a callback registered for them, `home` is `EventChange.home` and `reset` is
    `EventChange.reset`; a `start` hands its checkpoint, its action space and its step count to get_model() and
    get_steps(); each ack carries the state the work resolved to, and a refusal is `ok: false` with its words."""
    parts = rig(status="error", model_fixed=False)
    parts.extras = {"model": ("dreamlabs/dream-1-beta", "eef"), "steps": 0}
    parts.stream_once()
    parts.wait_linked()
    wait_for(lambda: parts.relay.texts("status") != [], "the first status")
    assert "model_fixed" not in parts.relay.texts("status")[0], "the page offers its model and action pickers"
    answers = {
        dream.EventChange.home: answer("idle"),
        dream.EventChange.reset: answer("resetting"),
        dream.EventChange.start: answer("running"),
    }
    events: list[dream.EventChange] = []

    def on_press(event: dream.EventChange) -> concurrent.futures.Future[Any]:
        events.append(event)
        answered: concurrent.futures.Future[Any] = concurrent.futures.Future()
        answered.set_result(answers[event])
        return answered

    parts.client.register_callback(on_press, home=True, reset=True)
    parts.relay.send({"type": "prompt", "id": "prm_h", "action": "home"})
    wait_for(lambda: "prm_h" in parts.relay.acks(), "the home ack")
    assert parts.relay.acks()["prm_h"] == {"type": "prompt_ack", "id": "prm_h", "ok": True, "state": "idle"}
    parts.relay.send({"type": "prompt", "id": "prm_r", "action": "reset"})
    wait_for(lambda: "prm_r" in parts.relay.acks(), "the reset ack")
    assert parts.relay.acks()["prm_r"]["state"] == "resetting"

    start = {"checkpoint": "dreamlabs/dream-1-ft:abc", "action_space": "joint", "steps": 2000}
    parts.relay.send({"type": "prompt", "id": "prm_s", "action": "start", "text": "fold", **start})
    wait_for(lambda: "prm_s" in parts.relay.acks(), "the start ack")
    assert parts.client.get_model() == ("dreamlabs/dream-1-ft:abc", "joint") and parts.client.get_steps() == 2000
    parts.stream_once(model=("dreamlabs/dream-1-ft:abc", "joint"), steps=2000)
    wait_for(lambda: parts.relay.texts("status")[-1].get("steps") == 2000, "the step count in the status")
    assert parts.relay.texts("status")[-1]["checkpoint"] == "dreamlabs/dream-1-ft:abc"
    assert events == [dream.EventChange.home, dream.EventChange.reset, dream.EventChange.start]

    answers[dream.EventChange.reset] = answer("running", "reset refused: station is running; stop the run first")
    parts.relay.send({"type": "prompt", "id": "prm_r2", "action": "reset"})
    wait_for(lambda: "prm_r2" in parts.relay.acks(), "the refused reset ack")
    refused = parts.relay.acks()["prm_r2"]
    assert (refused["ok"], refused["error"]) == (False, "reset refused: station is running; stop the run first")


def test_a_fixed_model_never_reaches_get_model_P6(relay: FakeRelay) -> None:
    """With `model_fixed` (the default) a Start's checkpoint is not kept: the loop runs the model its code chose."""
    with streaming(relay, status="idle") as client:
        relay.send({"type": "prompt", "id": "s", "action": "start", "checkpoint": "dreamlabs/other"})
        wait_for(lambda: "s" in relay.acks(), "the ack")
        assert client.get_model() == (None, None)


def test_a_pose_subscription_streams_the_arms_at_pose_hz_and_stops_on_unsubscribe_P7(rig: Callable[..., Rig]) -> None:
    """`hello` says `pose: true` for a loop that streams its arms; a `subscribe {camera: "pose"}` from the relay makes
    the client send `pose` messages at about POSE_HZ with each readable arm's joints and its gripper clipped to
    [0, 1]; an unsubscribe stops them; an arm that cannot be read is left out; stats() says whether it is on."""
    parts = rig()
    parts.extras = {"arms": {"left": ([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 0.75), "right": ([-0.1] * 6, 1.5)}}
    parts.stream_once()
    parts.wait_linked()
    assert parts.relay.hellos[0]["pose"] is True
    assert parts.client.stats()["pose"] is False

    parts.relay.send({"type": "subscribe", "camera": "pose"})
    wait_for(lambda: len(parts.relay.texts("pose")) >= 3, "three poses")
    first = parts.relay.texts("pose")[0]
    assert set(first["arms"]) == {"left", "right"}
    assert first["arms"]["left"]["q"] == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert first["arms"]["left"]["gripper"] == 0.75
    assert first["arms"]["right"]["gripper"] == 1.0, "the gripper is clipped to the wire's [0, 1]"
    assert first["t_us"] > 1_600_000_000_000_000
    assert parts.client.stats()["pose"] is True
    time.sleep(0.5)
    count = len(parts.relay.texts("pose"))
    assert 3 <= count <= live._POSE_HZ + 3, f"about POSE_HZ a second, got {count} in about half a second"

    parts.extras["arms"] = {"left": ([0.1] * 6, 0.5), "right": None}
    parts.stream_once()
    wait_for(lambda: set(parts.relay.texts("pose")[-1]["arms"]) == {"left"}, "the dead arm left out")

    parts.relay.send({"type": "unsubscribe", "camera": "pose"})
    wait_for(lambda: parts.client.stats()["pose"] is False, "unsubscribed")
    settled = len(parts.relay.texts("pose"))
    time.sleep(0.3)
    assert len(parts.relay.texts("pose")) <= settled + 1, "no poses after the unsubscribe"


def test_status_carries_the_checkpoint_the_arms_and_the_error_code_P6(rig: Callable[..., Rig]) -> None:
    """Every `status` names the checkpoint and each arm's liveness; an arm dying while nothing else changes sends a
    fresh `status`; `error_code` accompanies `error` and is absent without one."""
    parts = rig(subscribed=())
    arm = ([0.0] * 6, 0.0)
    parts.extras = {"model": ("dreamlabs/dream-1-beta", None), "arms": {"left": arm, "right": arm}}
    parts.stream_once()
    parts.wait_linked()
    wait_for(lambda: len(parts.relay.texts("status")) == 1, "the first status")
    first = parts.relay.texts("status")[0]
    assert (first["checkpoint"], first["arms"]) == ("dreamlabs/dream-1-beta", {"left": True, "right": True})
    assert "error_code" not in first

    parts.extras["arms"] = {"left": None, "right": arm}
    parts.stream_once()
    wait_for(lambda: len(parts.relay.texts("status")) == 2, "the dead arm's status")
    assert parts.relay.texts("status")[1]["arms"] == {"left": False, "right": True}

    refusal = "DreamAPI refused the inference request: DA-2000 NO_CAPACITY (pool_warming)"
    parts.stream_once(error=(refusal, "pool_warming"))
    wait_for(lambda: len(parts.relay.texts("status")) == 3, "the refusal's status")
    assert parts.relay.texts("status")[2]["error_code"] == "pool_warming"

    parts.extras["model"] = ("dreamlabs/dream-1-ft:abc", None)
    parts.stream_once()
    wait_for(lambda: len(parts.relay.texts("status")) == 4, "the checkpoint's status")
    assert parts.relay.texts("status")[3]["checkpoint"] == "dreamlabs/dream-1-ft:abc"

    parts.status = "error"
    parts.stream_once(error=("arm 'left' driver stopped", "ArmError"))
    wait_for(lambda: len(parts.relay.texts("status")) == 5, "the fatal status")
    fatal = parts.relay.texts("status")[4]
    assert (fatal["error"], fatal["error_code"]) == ("arm 'left' driver stopped", "ArmError")


def test_status_carries_the_action_space_and_a_yam_leaves_it_out_P9(rig: Callable[..., Rig]) -> None:
    """Every `status` names what the next run executes, and a change of it alone sends a fresh one; a loop that names
    none (a YAM) leaves the key out, which tells the page to send none."""
    parts = rig(subscribed=())
    parts.extras = {"model": ("dreamlabs/dream-1-beta", "eef")}
    parts.stream_once()
    parts.wait_linked()
    wait_for(lambda: len(parts.relay.texts("status")) == 1, "the first status")
    assert parts.relay.texts("status")[0]["action_space"] == "eef"

    parts.stream_once(model=("dreamlabs/dream-1-beta", "joint"))
    wait_for(lambda: len(parts.relay.texts("status")) == 2, "the action space's status")
    assert parts.relay.texts("status")[1]["action_space"] == "joint"

    parts.stream_once(model=("dreamlabs/dream-1-beta", None))
    wait_for(lambda: len(parts.relay.texts("status")) == 3, "the YAM's status")
    assert "action_space" not in parts.relay.texts("status")[-1]


def test_a_stop_that_outlives_prompt_ack_s_is_answered_timeout_but_not_cancelled_P5(
    rig: Callable[..., Rig], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stop whose home move runs past `_PROMPT_ACK_S` gets a `timeout` ack while the move itself is left to finish:
    a stop and a start are never cancelled halfway (only a new task is)."""
    monkeypatch.setattr(live, "_PROMPT_ACK_S", 0.2)
    parts = rig()
    parts.wait_linked()
    move: concurrent.futures.Future[Any] = concurrent.futures.Future()
    parts.client.register_callback(lambda event: move)

    parts.relay.send({"type": "prompt", "id": "prm_9", "action": "stop"})
    wait_for(lambda: "prm_9" in parts.relay.acks(), "the timeout ack")
    ack = parts.relay.acks()["prm_9"]
    assert ack["ok"] is False and ack["error"] == "timeout" and parts.client.is_linked
    # The deadline cancelled the wait, not the move: it finishes on its own.
    time.sleep(0.1)
    assert not move.cancelled()
    move.set_result(answer("idle"))


def test_a_stop_or_home_whose_move_ended_in_error_is_acknowledged_failed_with_the_stations_words_P1(
    rig: Callable[..., Rig],
) -> None:
    """An arm that did not reach home ends the move in `error`; the ack says so (`ok: false` with the robot's words) so
    the page's toast and the instructions row read failed, instead of `ok: true` beside an `error` state."""
    parts = rig()
    parts.wait_linked()

    async def failed_move() -> dict[str, Any]:
        return answer("error", "an arm did not reach home; see the log")

    parts.client.register_callback(lambda event: failed_move(), home=True)
    parts.relay.send({"type": "prompt", "id": "prm_11", "action": "stop"})
    wait_for(lambda: "prm_11" in parts.relay.acks(), "the stop ack")
    assert parts.relay.acks()["prm_11"] == {
        "type": "prompt_ack",
        "id": "prm_11",
        "ok": False,
        "state": "error",
        "error": "an arm did not reach home; see the log",
    }
    parts.relay.send({"type": "prompt", "id": "prm_12", "action": "home"})
    wait_for(lambda: "prm_12" in parts.relay.acks(), "the home ack")
    assert parts.relay.acks()["prm_12"]["ok"] is False


def test_a_dropped_link_never_cancels_a_home_move_in_flight_P5(rig: Callable[..., Rig]) -> None:
    """The link closing cancels the client's tasks, the press's wait among them; the home move is shielded from that
    and finishes, so a network blip during a Stop never strands the robot halfway."""
    parts = rig()
    parts.wait_linked()
    move: concurrent.futures.Future[Any] = concurrent.futures.Future()
    parts.client.register_callback(lambda event: move, home=True)
    parts.relay.send({"type": "prompt", "id": "prm_10", "action": "home"})
    time.sleep(0.1)

    parts.relay.close_link(4008, "idle")
    wait_for(lambda: not parts.client.is_linked, "the link down")
    time.sleep(0.1)
    assert not move.cancelled()
    move.set_result(answer("idle"))


def test_start_stop_and_a_new_prompt_reach_the_callbacks(relay: FakeRelay) -> None:
    """Start is `start`; Stop and a new task while running are `end` for a callback that took no `prompt`;
    get_prompt() holds the task."""
    events: list[dream.EventChange] = []
    with streaming(relay) as client:
        client.register_callback(events.append)
        relay.send({"type": "prompt", "id": "p1", "action": "start", "text": "fold the towel"})
        wait_for(lambda: "p1" in relay.acks(), "the start ack")
        assert client.get_prompt() == "fold the towel"
        relay.send({"type": "prompt", "id": "p2", "text": "stack the cups"})
        relay.send({"type": "prompt", "id": "p3", "action": "stop"})
        wait_for(lambda: "p3" in relay.acks(), "the stop ack")
        assert client.get_prompt() == "stack the cups"
    assert events == [dream.EventChange.start, dream.EventChange.end, dream.EventChange.end]
    assert all(ack["ok"] and ack["state"] == "running" for ack in relay.acks().values())


def test_the_prompt_reaches_the_page_in_the_status(relay: FakeRelay) -> None:
    """After a viewer's task, the next status carries it."""
    with streaming(relay) as client:
        relay.send({"type": "prompt", "id": "p1", "text": "fold"})
        wait_for(lambda: "p1" in relay.acks(), "the ack")
        client.stream(observation(), "running")
        wait_for(lambda: relay.texts("status")[-1].get("prompt") == "fold", "the status with the task")


def test_home_and_reset_are_answered_not_supported(relay: FakeRelay) -> None:
    """While no callback takes Home and Reset, the page hears so and nothing else happens."""
    events: list[dream.EventChange] = []
    with streaming(relay) as client:
        client.register_callback(events.append)
        relay.send({"type": "prompt", "id": "h", "action": "home"})
        relay.send({"type": "prompt", "id": "r", "action": "reset"})
        wait_for(lambda: {"h", "r"} <= set(relay.acks()), "both acks")
    assert not relay.acks()["h"]["ok"] and "not supported" in relay.acks()["h"]["error"]
    assert not relay.acks()["r"]["ok"]
    assert events == []


def test_a_failing_callback_is_a_failed_ack_and_the_link_goes_on(relay: FakeRelay) -> None:
    """The exception is the page's error; later prompts still arrive."""

    def broken(event: dream.EventChange) -> None:
        raise RuntimeError("arm busy")

    with streaming(relay) as client:
        client.register_callback(broken)
        relay.send({"type": "prompt", "id": "p1", "action": "start"})
        wait_for(lambda: "p1" in relay.acks(), "the first ack")
        relay.send({"type": "prompt", "id": "p2", "text": "fold"})
        wait_for(lambda: "p2" in relay.acks(), "the second ack")
    assert relay.acks()["p1"] == {
        "type": "prompt_ack",
        "id": "p1",
        "ok": False,
        "state": "running",
        "error": "RuntimeError: arm busy",
    }


# The tick guard (T-16's unit companion) and I12


def test_the_tick_guard_pauses_video_when_the_clients_own_thread_falls_behind_I13(
    rig: Callable[..., Rig], monkeypatch: pytest.MonkeyPatch
) -> None:
    """While the client's own poll runs later than `_YIELD_TICK_RATIO` poll periods (here a ratio its 60 Hz poll
    cannot meet) for `_YIELD_WINDOW_S`, it closes its encoders, sends `status {live_paused: "tick"}` and stops
    encoding; after `_YIELD_BACKOFF_S` it resumes, says so, and encodes on a fresh encoder."""
    monkeypatch.setattr(live, "_YIELD_TICK_RATIO", 0.5)
    monkeypatch.setattr(live, "_YIELD_WINDOW_S", 0.6)
    monkeypatch.setattr(live, "_YIELD_BACKOFF_S", 2.0)
    parts = rig(subscribed=("top",), cameras=("top",))
    parts.wait_linked()
    deadline_s = time.monotonic() + WAIT_S
    while parts.client.yields == 0 and time.monotonic() < deadline_s:
        parts.run_ticks(5, period_s=0.04)
    assert parts.client.yields == 1, "the yield"
    encoder = parts.encoders.built[0]
    assert encoder.closed and parts.client.live_paused == "tick"
    wait_for(lambda: any(status.get("live_paused") == "tick" for status in parts.relay.texts("status")), "paused")
    encoded_at_pause = len(encoder.frames)
    parts.run_ticks(5, period_s=0.04)
    assert len(encoder.frames) == encoded_at_pause and len(parts.encoders.built) == 1

    monkeypatch.setattr(live, "_YIELD_TICK_RATIO", 2.0)
    deadline_s = time.monotonic() + WAIT_S
    while parts.client.live_paused is not None and time.monotonic() < deadline_s:
        parts.run_ticks(5, period_s=0.04)
    assert parts.client.live_paused is None, "the resume"
    wait_for(lambda: "live_paused" not in parts.relay.texts("status")[-1], "the resume status")
    parts.run_ticks(5, period_s=0.04)
    wait_for(lambda: len(parts.encoders.built) == 2, "a fresh encoder after the backoff")


def test_the_callers_pace_never_pauses_the_video_I13a(
    rig: Callable[..., Rig], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A loop of one's own as it was seen on a Maker Arm: stream() from its policy thread, jittering 40 to 65 ms,
    one 262 ms gap as its first get_action() goes out, and sync holds of most of a second with no DreamClient to
    explain them. The guard watches the client's own thread, so none of it pauses the video."""
    monkeypatch.setattr(live, "_YIELD_WINDOW_S", 0.6)
    parts = rig(subscribed=("top",), cameras=("top",))
    parts.wait_linked()
    parts.run_ticks(10, period_s=0.04)
    for period_s in (0.058, 0.064, 0.045, 0.262, 0.06):
        parts.run_ticks(4, period_s=period_s)
    for _ in range(2):
        time.sleep(0.8)  # the arms hold through a sync get_action(); nothing is streamed meanwhile
        parts.run_ticks(10, period_s=0.04)
    encoder = parts.encoders.built[0]
    assert parts.client.yields == 0 and parts.client.live_paused is None and not encoder.closed
    wait_for(lambda: len(encoder.frames) >= 40, "encoding through the loop's own pace")


def test_the_poll_watch_trips_only_on_late_polls_sustained_while_encoding_I13() -> None:
    """On-time polls never trip it, nor does one long gap among them; polls past `_YIELD_TICK_RATIO` periods do, once
    encoding has run `_YIELD_WINDOW_S`; intervals from before encoding started are not counted, and a reset forgets."""
    period_s = 1.0 / live._POLL_HZ
    watch = live._PollWatch()
    now_s = 100.0
    for _ in range(300):
        now_s += period_s
        watch.observe(now_s)
    watch.encoding_since_s = now_s - 1.0
    now_s += 0.3
    watch.observe(now_s)
    assert not watch.is_starved(now_s - 0.3) and not watch.is_starved(now_s), "one gap among on-time polls"

    watch.encoding_since_s = now_s
    for _ in range(30):
        now_s += 2.5 * period_s
        watch.observe(now_s)
    assert not watch.is_starved(now_s), "encoding has not run the window yet"
    for _ in range(30):
        now_s += 2.5 * period_s
        watch.observe(now_s)
    assert watch.is_starved(now_s)
    assert watch.worst(now_s - live._YIELD_WINDOW_S)[0] == pytest.approx(2.5 * period_s)

    watch.reset()
    assert list(watch.intervals) == [] and watch.last_poll_s is None and watch.encoding_since_s is None


@pytest.mark.skipif(sys.platform != "linux", reason="thread niceness is read from /proc")
def test_the_agent_is_one_niced_thread_I12(rig: Callable[..., Rig]) -> None:
    """One `dream-live` thread exists, and its niceness is the process's plus `_NICE`."""
    parts = rig()
    parts.wait_linked()
    threads = [thread for thread in threading.enumerate() if thread.name == "dream-live"]
    assert len(threads) == 1
    stat = pathlib.Path(f"/proc/self/task/{threads[0].native_id}/stat").read_text().rsplit(")", 1)[1].split()
    assert int(stat[16]) == min(19, os_nice_of_main() + live._NICE)
    assert live._ENCODER_THREADS == 1


## The CPU ticks a thread that existed before the client may be charged while `_prepare` runs: accounting noise.
STRAY_TICKS = 1

## A fresh process that builds a client, starts it, resizes and blurs frames on it, and prints every thread's
## niceness and CPU ticks before the start and after the frames: OpenCV's worker pool must be born on the client's
## thread, so its workers carry the client's niceness and the threads that existed before do no work.
POOL_PROBE = """
import json, os, pathlib, threading, time
import numpy as np
import dream

def tasks():
    out = {}
    for tid in os.listdir("/proc/self/task"):
        stat = pathlib.Path(f"/proc/self/task/{tid}/stat").read_text()
        fields = stat.rsplit(")", 1)[1].split()
        name = stat.split("(", 1)[1].rsplit(")", 1)[0]
        out[tid] = [int(fields[16]), int(fields[11]) + int(fields[12]), name]
    return out

client = dream.LiveViewClient("rbt_0123456789abcdef", api_key="k", cameras=("top",))
before = tasks()
client.stream({}, "idle")
time.sleep(0.5)
client_tid = str([t for t in threading.enumerate() if t.name == "dream-live"][0].native_id)
frame = np.random.default_rng(0).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
for _ in range(20):
    client._prepare(client._streams["top"], frame)
after = tasks()
main_tid = str(threading.main_thread().native_id)
print(json.dumps({"main": main_tid, "agent": client_tid, "before": before, "after": after}))
"""


@pytest.mark.skipif(sys.platform != "linux", reason="thread niceness is read from /proc")
def test_the_agent_s_resize_and_blur_run_on_niced_threads_only_I12() -> None:
    """In a fresh process the client's `_prepare` (resize, BGR, face blur) fans out onto OpenCV's worker pool: every
    thread born after the client started, the pool included, carries the main thread's niceness plus `_NICE`, the
    work lands on those threads and on the caller, and no thread that existed before does any of it."""
    env = {**os.environ, "DREAM_API_URL": "http://127.0.0.1:1"}
    result = subprocess.run(
        [sys.executable, "-c", POOL_PROBE], env=env, capture_output=True, text=True, timeout=WAIT_S * 6, check=True
    )
    report = json.loads(result.stdout.splitlines()[-1])
    before, after = report["before"], report["after"]
    expected = min(19, before[report["main"]][0] + live._NICE)
    born = {tid: task for tid, task in after.items() if tid not in before}
    assert report["agent"] in born and all(task[0] == expected for task in born.values()), born
    # CPU time is charged in whole clock ticks, to whichever thread is running when the tick lands, so a thread that
    # was already there (an idle library worker from the imports) can be charged one tick for next to nothing. The
    # work under test is many ticks (the main thread's share alone is well over ten), so one is noise, not the work.
    grew = {
        f"{tid} {after[tid][2]}": after[tid][1] - before[tid][1]
        for tid in before
        if tid in after and after[tid][1] - before[tid][1] > STRAY_TICKS
    }
    assert set(grew) <= {f"{report['main']} {after[report['main']][2]}"}, grew
    cpus = os.cpu_count()
    if cpus and cpus > 1:
        assert sum(task[1] for tid, task in born.items() if tid != report["agent"]) > 0, born


def os_nice_of_main() -> int:
    """The calling thread's niceness, from `/proc`."""
    stat = pathlib.Path(f"/proc/self/task/{threading.get_native_id()}/stat").read_text().rsplit(")", 1)[1].split()
    return int(stat[16])


# What the constructor checks


def test_start_sends_a_box_without_websockets_to_the_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    """A computer without websockets hears where the dream library comes from, from the constructor."""
    monkeypatch.setitem(sys.modules, "websockets", None)
    monkeypatch.setenv("DREAM_API_URL", "http://127.0.0.1:1")
    with pytest.raises(dream.DreamConfigError, match="pip install --upgrade dreamlabs"):
        dream.LiveViewClient(ROBOT, api_key=KEY)


def test_face_blur_without_the_detector_names_pip_install_dreamlabs(monkeypatch: pytest.MonkeyPatch) -> None:
    """With face_blur on, a computer whose `shared` has no face detector hears the command that installs it, before
    anything opens; with it off, nothing asks for the detector."""
    find_spec = live.importlib.util.find_spec
    monkeypatch.setattr(
        live.importlib.util,
        "find_spec",
        lambda name, *rest: None if name == "shared.faceblur" else find_spec(name, *rest),
    )
    with pytest.raises(
        dream.DreamConfigError, match=r"install it with pip install --upgrade dreamlabs, or pass face_blur=False"
    ):
        live._check_installed(True)
    live._check_installed(False)


def test_open_encoder_sends_a_box_without_av_to_the_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without PyAV the encoder's own open names the same place."""
    monkeypatch.setitem(sys.modules, "av", None)
    with pytest.raises(dream.DreamConfigError, match="pip install --upgrade dreamlabs"):
        live._open_encoder("libvpx", 64, 48, 30, 1400)


@pytest.mark.parametrize(
    ("platform_url", "expected"),
    [
        ("https://platform.example.com", "wss://platform.example.com/live/robot"),
        ("https://platform.example.com/", "wss://platform.example.com/live/robot"),
        ("https://platform.example.com:8443", "wss://platform.example.com:8443/live/robot"),
        ("http://localhost:8080", "ws://localhost:8080/live/robot"),
        ("http://127.0.0.1:8400", "ws://127.0.0.1:8400/live/robot"),
        ("http://[::1]:8400", "ws://[::1]:8400/live/robot"),
    ],
)
def test_robot_link_is_wss_on_the_platform_or_ws_to_a_loopback_host_A7_I2(platform_url: str, expected: str) -> None:
    """The link is `/live/robot` on the credentials' platform: `wss://` for `https://`, and `ws://` only when the
    platform is `http://` on this machine (the dev stack, the tests)."""
    assert live._robot_link(platform_url) == expected


@pytest.mark.parametrize("platform_url", ["http://platform.example.com", "http://10.0.0.5:8080"])
def test_a_cleartext_platform_off_the_box_is_refused_A7_I2(platform_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """An `http://` platform on a network host fails naming the rule: the key rides the upgrade and must never cross a
    network in the clear."""
    monkeypatch.setenv("DREAM_API_URL", platform_url)
    with pytest.raises(dream.DreamConfigError, match=r"https://.*clear"):
        dream.LiveViewClient(ROBOT, api_key=KEY)


def test_the_platform_is_the_credentials_and_cannot_be_passed_A7_I2(monkeypatch: pytest.MonkeyPatch) -> None:
    """The link is built from `DREAM_API_URL` alone; the constructor takes no URL."""
    monkeypatch.setenv("DREAM_API_URL", "http://127.0.0.1:8400")
    assert dream.LiveViewClient(ROBOT, api_key=KEY)._url == "ws://127.0.0.1:8400/live/robot"
    with pytest.raises(TypeError, match="platform_url"):
        dream.LiveViewClient(ROBOT, api_key=KEY, platform_url="http://127.0.0.1:8400")  # type: ignore[call-arg]


def test_start_probes_the_encoder_before_the_thread_runs_V6(monkeypatch: pytest.MonkeyPatch) -> None:
    """The constructor opens and closes one encoder before anything runs, so a missing PyAV or an encoder the build
    lacks is a `DreamConfigError` from the constructor, never a per-frame fault, and no thread starts."""
    encoders = Encoders()
    monkeypatch.setattr(live, "_open_encoder", encoders)
    monkeypatch.setenv("DREAM_API_URL", "http://127.0.0.1:1")
    client = dream.LiveViewClient(ROBOT, api_key=KEY)
    assert encoders.probe is not None and encoders.probe.closed and encoders.built == []
    assert client._thread is None
    client.close()


def test_the_key_comes_from_the_credentials_and_none_anywhere_is_a_config_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolved as DreamClient resolves it: the argument, else the environment, else the file."""
    with pytest.raises(dream.DreamConfigError, match="API key"):
        dream.LiveViewClient(ROBOT)
    monkeypatch.setenv("DREAM_API_KEY", "dl_live_env")
    assert dream.LiveViewClient(ROBOT)._api_key == "dl_live_env"
    assert dream.LiveViewClient(ROBOT, api_key=KEY)._api_key == KEY
    with pytest.raises(dream.DreamConfigError):
        dream.LiveViewClient("", api_key=KEY)


def test_settings_the_page_cannot_decode_are_refused_V6() -> None:
    """A camera named `pose` or a repeated camera fails in the constructor, not on the page."""
    with pytest.raises(dream.DreamConfigError, match="cameras"):
        dream.LiveViewClient(ROBOT, api_key=KEY, cameras=("top", "pose"))
    with pytest.raises(dream.DreamConfigError, match="cameras"):
        dream.LiveViewClient(ROBOT, api_key=KEY, cameras=("top", "top"))


def test_stream_refuses_what_it_cannot_show(relay: FakeRelay) -> None:
    """A non-mapping, an observation without frames, a frame that is neither a JPEG nor an RGB array, an empty status,
    new cameras or malformed arms are the caller's mistakes."""
    client = dream.LiveViewClient(ROBOT, api_key=KEY)
    with pytest.raises(dream.DreamObservationError):
        client.stream([], "running")  # type: ignore[arg-type]
    with pytest.raises(dream.DreamObservationError, match="no frame"):
        client.stream({"state.right_arm": [0.0]}, "running")
    with pytest.raises(dream.DreamObservationError, match="RGB"):
        client.stream({"video.overhead": b"bytes"}, "running")
    with pytest.raises(dream.DreamObservationError, match="status"):
        client.stream(observation(), "")
    client.stream(observation(), "running")
    with pytest.raises(dream.DreamObservationError, match="first call"):
        client.stream({"video.overhead": jpeg()}, "running")
    with pytest.raises(dream.DreamObservationError, match="arms"):
        client.stream(observation(), "running", arms={"left": 0.5})
    client.close()


def test_the_first_stream_links_with_the_key_the_cameras_and_the_status(relay: FakeRelay) -> None:
    """`hello` names the cameras as the page does (`overhead` is `top`, `wrist_l` is `left`); the status follows it,
    the model fixed, the run counted from the first `running`."""
    with streaming(relay):
        assert relay.auth_headers == [f"Bearer {KEY}"]
        assert (relay.hellos[0]["robot_id"], relay.hellos[0]["cameras"]) == (ROBOT, ["top", "left"])
        assert relay.messages[0] == {
            "type": "status",
            "state": "running",
            "run": 1,
            "dropped_frames": {"top": 0, "left": 0},
            "arms": {},
            "model_fixed": True,
        }


def test_a_watched_camera_streams_vp8_starting_with_a_keyframe(relay: FakeRelay) -> None:
    """Nothing is encoded until a viewer subscribes; then that camera alone, its first unit a real VP8 keyframe."""
    with streaming(relay, status="idle") as client:
        time.sleep(0.2)
        assert relay.units == []
        relay.send({"type": "subscribe", "camera": "top"})
        for value in range(10):
            client.stream(observation(value * 20), "idle")
            time.sleep(0.05)
        wait_for(lambda: len(relay.units) >= 3, "three units")
    headers = [header for header, _ in relay.units]
    assert {header.camera for header in headers} == {0}  # camera index 0, `top`
    assert headers[0].is_keyframe
    assert [header.seq for header in headers] == list(range(len(headers)))
    first = relay.units[0][1]  # a VP8 keyframe: frame tag bit 0 clear, then the keyframe start code (RFC 6386 9.1)
    assert first[0] & 1 == 0 and first[3:6] == b"\x9d\x01\x2a"


def test_close_is_idempotent_and_a_stream_after_it_does_nothing(relay: FakeRelay) -> None:
    """The `with` block closes; closing again and streaming after are harmless."""
    with streaming(relay) as client:
        pass
    client.close()
    client.stream(observation(), "idle")
    assert client._thread is not None and not client._thread.is_alive()


def test_import_dream_loads_neither_pyav_nor_opencv() -> None:
    """The encoder's packages load when a client is built, never with `import dream`."""
    code = "import sys, dream; print('av' in sys.modules, 'cv2' in sys.modules, dream.LiveViewClient.__name__)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.split() == ["False", "False", "LiveViewClient"]


# The real encoder (T-3)


## The three bytes after a VP8 keyframe's 3-byte frame tag (RFC 6386 section 9.1), which an interframe does not carry.
VP8_KEYFRAME_START_CODE = b"\x9d\x01\x2a"


def vp8_is_keyframe(unit: bytes) -> bool:
    """Whether a VP8 frame is a keyframe: bit 0 of its first byte is 0 and the start code follows the frame tag."""
    return unit[0] & 1 == 0 and unit[3:6] == VP8_KEYFRAME_START_CODE


def synthetic_frames(count: int, width: int, height: int) -> Iterator[np.ndarray]:
    """Moving noise: enough entropy that the rate control works at its cap."""
    rng = np.random.default_rng(0)
    base = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    for index in range(count):
        rolled = np.roll(base, index * 3, axis=1).astype(np.int16)
        yield (rolled + rng.integers(-8, 8, (height, width, 3))).clip(0, 255).astype(np.uint8)


def test_libvpx_makes_vp8_keyframes_at_the_gop_and_the_bitrate_V1() -> None:
    """Real `libvpx`: one encoded frame per input frame, the first a VP8 keyframe, a keyframe every `fps x _GOP_S`
    frames and no other, every other frame an interframe, and 10 s of frames within 20% of the bitrate."""
    encoder = live._open_encoder("libvpx", 640, 360, 30, 1400)
    units = [encoder.encode(frame, False) for frame in synthetic_frames(300, 640, 360)]
    encoder.close()

    assert all(unit is not None for unit in units)
    payloads = [unit[0] for unit in units if unit is not None]
    keyframes = [index for index, unit in enumerate(units) if unit is not None and unit[1]]
    assert keyframes == list(range(0, 300, round(30 * live._GOP_S)))
    assert all(vp8_is_keyframe(payloads[index]) for index in keyframes)
    assert not vp8_is_keyframe(payloads[1]) and payloads[1][0] & 1 == 1
    kbps = sum(len(payload) for payload in payloads) * 8 / 10 / 1000
    assert abs(kbps - 1400) / 1400 < 0.2, kbps


def test_a_forced_keyframe_is_a_vp8_keyframe_V1() -> None:
    """`force_idr` on a mid-GOP frame yields a VP8 keyframe, what a viewer needs to resume after a drop."""
    encoder = live._open_encoder("libvpx", 640, 360, 30, 1400)
    units = [encoder.encode(frame, index == 10) for index, frame in enumerate(synthetic_frames(12, 640, 360))]
    encoder.close()
    forced = units[10]
    assert forced is not None and forced[1] and vp8_is_keyframe(forced[0])
    assert units[11] is not None and not units[11][1]


def test_an_unknown_encoder_is_a_loud_start_up_error_V6() -> None:
    """An encoder PyAV's build lacks raises DreamConfigError naming it from the encoder's own open, the call the
    constructor probes with; nothing falls back and no thread starts."""
    with pytest.raises(dream.DreamConfigError, match="h264_nope"):
        live._open_encoder("h264_nope", 640, 360, 30, 1400)


# Arms of any width, and the run's identity


def test_a_seven_joint_arm_sends_its_poses_on_one_link_P7(rig: Callable[..., Rig]) -> None:
    """A 7-joint arm sends every pose at its own length over the link it opened; a wire fixed at six joints would
    drop the link seconds into a run."""
    seven = [-0.84, -0.51, 0.0, -1.3, 0.65, 0.29, 0.03]
    parts = rig(subscribed=(), cameras=("top",))
    parts.extras = {"arms": {"left": (seven, 0.5), "right": (seven, 0.5)}}
    parts.stream_once()
    parts.wait_linked()
    parts.relay.send({"type": "subscribe", "camera": "pose"})
    wait_for(lambda: len(parts.relay.texts("pose")) >= 2, "two poses")
    assert parts.relay.texts("pose")[-1]["arms"]["right"]["q"] == pytest.approx(seven)
    assert parts.relay.connections == 1 and parts.client.reconnects == 0, "one link the whole time"


def test_status_carries_the_dreamapi_session_id_S8(rig: Callable[..., Rig]) -> None:
    """`status` names the running session and leaves the key out before the first chunk, so an older relay never sees
    `null`; between runs there is none."""
    model = FakeDreamClient()
    parts = rig(subscribed=(), cameras=("top",), dream_client=model)
    parts.wait_linked()
    wait_for(lambda: parts.relay.texts("status") != [], "the first status")
    assert "session_id" not in parts.relay.texts("status")[0]
    session = "711c847e-c28d-4f4e-885c-9d473bd4b79d"
    model.session_id = session
    parts.stream_once()
    wait_for(lambda: parts.relay.texts("status")[-1].get("session_id") == session, "the status")
    parts.status = "idle"
    parts.stream_once()
    wait_for(lambda: parts.relay.texts("status")[-1]["state"] == "idle", "the idle status")
    assert "session_id" not in parts.relay.texts("status")[-1]


class FakeDreamClient:
    """Stands in for the loop's DreamClient: `last_timing` is what its last get_action() cost, and its session."""

    def __init__(self, checkpoint: str | None = None) -> None:
        self.last_timing: dict[str, Any] | None = None
        self.checkpoint = checkpoint
        self.session_id: str | None = None

    def answer(self, rtt_ms: float) -> None:
        """One more get_action() came back after `rtt_ms`."""
        self.last_timing = {
            "rtt_ms": rtt_ms,
            "gateway_ms": int(rtt_ms) - 20,
            "serve_infer_ms": int(rtt_ms) - 80,
            "network_ms": 20.0,
            "up_mbps": 12.5,
            "request_bytes": int(rtt_ms) * 100,
            "response_bytes": 900,
        }


def test_the_page_hears_the_model_s_round_trips_only_while_running(relay: FakeRelay) -> None:
    """Without `timing` the page shows a running robot as "Dream-1 is getting ready" forever (seen on a Maker Arm
    station): the status carries it from the first answer of a run, not before, never while idle, and at most every
    2 s."""
    model = FakeDreamClient()
    model.answer(999.0)  # the previous run's last request: never this run's first answer
    client = dream.LiveViewClient(ROBOT, api_key=KEY, dream_client=model)
    client.stream(observation(), "idle")
    wait_for(lambda: relay.messages != [], "the idle status")
    client.stream(observation(), "running")
    wait_for(lambda: any(t.get("state") == "running" for t in relay.texts("status")), "the running status")
    assert all("timing" not in t for t in relay.texts("status"))
    model.answer(640.0)
    client.stream(observation(), "running")
    wait_for(lambda: any("timing" in t for t in relay.texts("status")), "the first answer")
    first = next(t["timing"] for t in relay.texts("status") if "timing" in t)
    assert (first["rtt_ms"], first["count"], first["gateway_ms"], first["serve_infer_ms"]) == (640.0, 1, 620, 560)
    assert first["recent_ms"] == [640.0] and first["up_mbps"] == 12.5
    model.answer(700.0)
    client.stream(observation(), "running")
    time.sleep(0.3)
    assert sum("timing" in t for t in relay.texts("status")) == 1  # the second answer waits out TIMING_STATUS_S
    client.stream(observation(), "idle")
    wait_for(lambda: relay.texts("status")[-1].get("state") == "idle", "the idle status")
    assert "timing" not in relay.texts("status")[-1]
    client.close()


def test_the_page_shows_the_model_the_code_runs_and_offers_no_other(relay: FakeRelay) -> None:
    """The page's model choice never reached the loop, which runs the checkpoint its code gave policy(): the client
    says the model is fixed and names the one it runs, so the page locks its pickers and shows that model."""
    model = FakeDreamClient(checkpoint="dreamlabs/dream-1-beta")
    with dream.LiveViewClient(ROBOT, api_key=KEY, dream_client=model) as client:
        client.stream(observation(), "idle")
        wait_for(lambda: relay.messages != [], "the first status")
        assert relay.messages[0] == {
            "type": "status",
            "state": "idle",
            "dropped_frames": {"top": 0, "left": 0},
            "checkpoint": "dreamlabs/dream-1-beta",
            "arms": {},
            "model_fixed": True,
        }
        model.checkpoint = "dreamlabs/dream-1"  # the loop opened another policy: the page follows
        client.stream(observation(), "idle")
        wait_for(lambda: relay.texts("status")[-1].get("checkpoint") == "dreamlabs/dream-1", "the new model")
