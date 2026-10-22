"""The recorder of `dream.DreamClient`: every camera tick, sent to the platform's live service as it runs.

On every policy, the client records what its caller hands it (the one exception is the internal `_record: False`
config key, for episodes that are not the account's, such as the MuJoCo example's simulated ones): every
set_observation() becomes one `observation_state` message (the tick's time, the episode, the session, whether its
frames were kept, and every non-frame key), and a kept tick's frames become one binary message per camera, the JPEG
the caller encoded.
They go over one WebSocket to the platform's live service, `wss://<platform host>/live/robot`, as a record-only robot
link: the live service stores them for Dream Labs and never shows them to a viewer. The wire is the live service's
robot link (`shared.contracts.live` in the platform's repository, which this package cannot import); the few names it
needs are copied below, each saying where it comes from.

Recording is best effort, and get_action() always comes first:

- set_observation() only encodes the tick's state to JSON (so a later change to the caller's arrays changes nothing
  recorded) and queues it, with the kept frames still in base64; it never raises or waits because of the recorder. A
  state JSON cannot carry is dropped and counted.
- The queue is bounded by bytes (#RECORD_QUEUE_BYTES); when it is full the oldest messages are dropped and counted,
  frames and states apart, as they are while the link is down.
- One daemon thread per recording policy dials the link on the first tick (the hello names the tick's cameras),
  decodes the frames and sends, oldest first, but never while a get_action() is in flight (hold() and release()):
  it finishes at most the one message it is in the middle of. The socket keeps what the kernel queues small, as
  `dream.LiveViewClient`'s does, so that message is short.
- A link the live service refuses as record-only (#CLOSE_RECORD_OFF, or #CLOSE_PROTOCOL from a service that does not
  know the mode) ends recording for the rest of the policy, with one DEBUG line on the `dream` logger: a platform
  that does not take recordings refuses, which is an expected outcome, not a fault of the caller's. A key refusal
  (#CLOSE_UNAUTHORIZED, #CLOSE_NOT_FOUND, #CLOSE_REVOKED) is retried after #REFUSAL_WAIT_S; anything else, a network
  fault, another close or no `hello_ok` within #HELLO_TIMEOUT_S, after a jittered backoff from #LINK_RECONNECT_MIN_S
  to #LINK_RECONNECT_MAX_S. Failures are counted in stats() and logged at DEBUG at most once a minute per kind;
  nothing the recorder logs is above DEBUG, since the caller never asked for it.
- stop() ends the thread within #STOP_TIMEOUT_S: a linked recorder sends what it still holds for up to #FLUSH_S,
  then closes the link; what is left is dropped. Interpreter exit stops every recorder still running the same way.
"""

import atexit
import base64
import binascii
import collections
import dataclasses
import datetime
import json
import logging
import os
import random
import socket
import struct
import sys
import threading
import time
import urllib.parse
import weakref
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from dream.errors import DreamConfigError

if TYPE_CHECKING:
    from websockets.sync.client import ClientConnection

## The protocol version, `hello.protocol` and the first byte of every video header (`shared.contracts.live`
## `LIVE_PROTOCOL`).
LIVE_PROTOCOL = 1

## The robot link's path on the platform (`shared.contracts.live` `ROBOT_LINK_PATH`).
ROBOT_LINK_PATH = "/live/robot"

## The hosts a plain `http://` platform may be dialed at, `ws://`, since nothing leaves the machine (the robots
## agent's `_LOOPBACK_HOSTS`).
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

## The 16-byte video header in `struct` notation: u8 version, u8 kind, u8 camera, u8 flags, u32 seq, u64 capture_us,
## big-endian (`shared.contracts.live` `VIDEO_HEADER_FORMAT`).
VIDEO_HEADER_FORMAT = ">BBBBIQ"

## The header's `kind` of a video unit, the only kind protocol 1 has (`shared.contracts.live` `KIND_VIDEO`).
KIND_VIDEO = 1

## The header's keyframe flag (`shared.contracts.live` `FLAG_KEYFRAME`): every JPEG decodes on its own.
FLAG_KEYFRAME = 0x01

## Largest frame the live service takes after the header (`shared.contracts.live` `LIVE_MAX_AU_BYTES`); a larger one is
## dropped here, since the service would close the link `4009`.
LIVE_MAX_AU_BYTES = 1 << 20

## Longest `observation_state` text the live service takes (`shared.contracts.live` `OBSERVATION_STATE_MAX_BYTES`); a
## longer one is dropped here, for the same reason.
OBSERVATION_STATE_MAX_BYTES = 64 << 10

## Most cameras, and longest camera name or version, a `hello` may carry (`shared.contracts.live`
## `HELLO_MAX_CAMERAS`, `HELLO_MAX_FIELD_CHARS`).
HELLO_MAX_CAMERAS = 256
HELLO_MAX_FIELD_CHARS = 64

## The joint-state stream's name, never a camera's (`shared.contracts.live` `POSE_STREAM`).
POSE_STREAM = "pose"

## The live service's close codes the recorder acts on (`shared.contracts.live` `CLOSE_UNAUTHORIZED`,
## `CLOSE_PROTOCOL`, `CLOSE_NOT_FOUND`, `CLOSE_REVOKED`, `CLOSE_RECORD_OFF`).
CLOSE_UNAUTHORIZED = 4001
CLOSE_PROTOCOL = 4002
CLOSE_NOT_FOUND = 4004
CLOSE_REVOKED = 4005
CLOSE_RECORD_OFF = 4011

## Every close code by its name (`shared.contracts.live` `CLOSE_CODES`), for stats() `last_error` and the log.
CLOSE_NAMES = {
    CLOSE_UNAUTHORIZED: "unauthorized",
    CLOSE_PROTOCOL: "protocol",
    4003: "node_full",
    CLOSE_NOT_FOUND: "not_found",
    CLOSE_REVOKED: "revoked",
    4006: "replaced",
    4007: "viewer_cap",
    4008: "idle",
    4009: "too_large",
    4010: "draining",
    CLOSE_RECORD_OFF: "record_off",
}

## `TCP_NOTSENT_LOWAT` on the link (`shared.contracts.live` `LIVE_NOTSENT_LOWAT`): the kernel holds at most this much
## unsent, so a message already handed over when a get_action() starts drains in a moment.
LIVE_NOTSENT_LOWAT = 16 << 10

## `SO_SNDBUF` on the link, as `dream.live`'s `_SNDBUF_BYTES`: at most twice this queued in the
## kernel, sent and unsent, instead of the autotuned megabytes that would sit in front of a get_action().
LIVE_SNDBUF_BYTES = 128 << 10

## Floor and ceiling of the jittered exponential reconnect backoff (`shared.contracts.live` `LINK_RECONNECT_MIN_S`,
## `LINK_RECONNECT_MAX_S`).
LINK_RECONNECT_MIN_S = 1.0
LINK_RECONNECT_MAX_S = 30.0

## The flat wait after a key refusal, which only the platform side can fix: the backoff's ceiling, never faster.
REFUSAL_WAIT_S = 30.0

## Seconds to open the link and to get `hello_ok` after the hello; past either the attempt failed.
HELLO_TIMEOUT_S = 5.0

## Bytes the queue holds, frames counted at their decoded size (three quarters of the base64): a few seconds of three
## cameras at 5 fps; past it the oldest messages are dropped.
RECORD_QUEUE_BYTES = 1_500_000

## Longest stop() waits for the thread; what it has not sent by then is dropped.
STOP_TIMEOUT_S = 1.0

## How long a linked recorder that is stopping keeps sending what it holds before it closes the link.
FLUSH_S = 0.5

## The closing handshake's timeout: the live service answers a close at once, and a stop must not wait on one that
## does not.
CLOSE_TIMEOUT_S = 0.25

## The `codec` of a record-only hello: each frame is one JPEG, carrying its own size (`width_px` is 0).
CODEC = "jpeg"

## stats() `state` values: recording is off; on and not linked yet (no tick yet, dialing, or waiting for
## `hello_ok`); linked; waiting to dial again; off for the rest of the policy after a refusal.
OFF = "off"
CONNECTING = "connecting"
LINKED = "linked"
BACKOFF = "backoff"
DISABLED = "disabled"

## The package's log; a failure of one kind is logged at most once per #_LOG_EVERY_S.
_log = logging.getLogger("dream.recorder")

## Seconds between two log lines of one kind of failure, so a flapping link or a stream of bad ticks logs once a
## minute, not once a tick.
_LOG_EVERY_S = 60.0

## How long the thread waits between looks at what the live service sent: it reads (and ignores) every message, so
## the library's reader never stalls on a full buffer and the service's pings are answered.
_POLL_S = 0.1

## The socket option number: Python's `socket` module lacks it on some builds, so Linux's 25 is the fallback.
_TCP_NOTSENT_LOWAT: int | None = getattr(socket, "TCP_NOTSENT_LOWAT", 25 if sys.platform == "linux" else None)

## The recorders whose thread is running, which _stop_all() stops at interpreter exit; weak, so one the client
## dropped is forgotten.
_running: "weakref.WeakSet[Recorder]" = weakref.WeakSet()


@dataclasses.dataclass(frozen=True, slots=True)
class _State:
    """One `observation_state`, encoded in the caller's thread.

    Attributes:
    - cameras: The tick's frame keys, sorted: the `hello.cameras` of the link it is sent on.
    - text: The message, ASCII JSON.
    - size_bytes: What it counts against #RECORD_QUEUE_BYTES.
    """

    cameras: tuple[str, ...]
    text: str
    size_bytes: int


@dataclasses.dataclass(frozen=True, slots=True)
class _Frame:
    """One kept frame, still base64; the thread decodes it just before it is sent.

    Attributes:
    - cameras: The tick's frame keys, sorted: the `hello.cameras` of the link it is sent on.
    - camera: Its frame key; the header's `camera` is its index in `cameras`.
    - frame_b64: The base64 JPEG the caller passed.
    - seq: Its camera's frame number, from 0 per recorder, every frame produced counted (a dropped one included).
    - capture_us: The tick's wall-clock time, microseconds, the `t_us` of its state.
    - size_bytes: Its decoded size, about; what it counts against #RECORD_QUEUE_BYTES.
    """

    cameras: tuple[str, ...]
    camera: str
    frame_b64: str
    seq: int
    capture_us: int
    size_bytes: int


class _RefusedError(Exception):
    """The live service said `bye` with this close code instead of `hello_ok`."""

    def __init__(self, code: int) -> None:
        super().__init__(f"refused {code}")
        self.code = code


class Recorder:
    """One recording policy's link to the live service: a queue filled by the caller and a thread that empties it.

    The client makes one in policy() with `record` on and stops it in close(), in the next policy(), or at
    interpreter exit. offer() takes each tick, hold() and release() bracket each get_action(), stats() reads the
    counters; all four are safe from any thread and never raise. One lock (a condition) guards the queue, the
    counters and the state; the thread holds it only to take a message, never while sending.

    Attributes:
    - robot_id: The `hello.robot_id`, `rec_` and 32 hex digits, the client's for all its policies.
    """

    def __init__(
        self,
        url: str,
        headers: dict[str, str],
        robot_id: str,
        fps: int,
        agent_version: str,
    ) -> None:
        """Starts idle: no thread and no socket until start(), and no link until the first tick after it.

        @param url The robot link, link_url() of the client's platform.
        @param headers The upgrade's headers: the key as `Authorization: Bearer`.
        @param robot_id The `hello.robot_id`.
        @param fps The `hello.fps`: the policy's `obs_fps`, a whole number, at least 1.
        @param agent_version The `hello.agent_version`, at most #HELLO_MAX_FIELD_CHARS.
        """
        self.robot_id = robot_id
        self._url = url
        self._headers = headers
        self._fps = fps
        self._agent_version = agent_version
        ## Wall clock minus monotonic clock, taken once: a tick's monotonic timestamp plus this is its wall time.
        self._offset_s = time.time() - time.monotonic()
        self._cond = threading.Condition()
        ## Messages waiting, oldest first, and their #RECORD_QUEUE_BYTES total.
        self._queue: collections.deque[_State | _Frame] = collections.deque()
        self._queue_bytes = 0
        ## Per camera, the `seq` of its next frame.
        self._seqs: dict[str, int] = {}
        ## get_action() calls in flight; nothing is sent while it is above 0.
        self._held = 0
        self._is_stopping = False
        ## When a stopping recorder stops sending what it holds (monotonic seconds).
        self._flush_until_s = 0.0
        self._state = CONNECTING
        self._last_error: str | None = None
        self._frames_sent = 0
        self._states_sent = 0
        self._frames_dropped = 0
        self._states_dropped = 0
        self._bytes_sent = 0
        ## Failed attempts since the last `hello_ok`, which size the backoff.
        self._attempt = 0
        ## The open link, for stop() to cut when the thread is stuck in it; None between links.
        self._connection: ClientConnection | None = None
        ## The code of a `bye` the live service sent on the open link, when its close frame carries none.
        self._bye_code: int | None = None
        ## Failure kind to when it was last logged (monotonic seconds).
        self._logged_s: dict[str, float] = {}
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Starts the thread, which waits for the first tick before it dials; interpreter exit stops it."""
        self._thread = threading.Thread(target=self._run, name="dream-recorder", daemon=True)
        _running.add(self)
        self._thread.start()

    def stop(self) -> None:
        """Stops the thread within #STOP_TIMEOUT_S and drops what is left; calling it again does nothing more.

        A linked recorder first sends what it holds, for up to #FLUSH_S, then closes the link; one that is dialing,
        waiting for `hello_ok` or backing off has its socket cut at once. A thread still in a send at the deadline has
        its socket cut too; one still opening a connection is left to end on its own, sending nothing.
        """
        started_s = time.monotonic()
        with self._cond:
            if not self._is_stopping:
                self._is_stopping = True
                self._flush_until_s = started_s + FLUSH_S
            is_linked = self._state == LINKED
            self._cond.notify_all()
        if not is_linked:
            self._abort()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(FLUSH_S + CLOSE_TIMEOUT_S)
            if thread.is_alive():
                self._abort()
                thread.join(max(0.0, started_s + STOP_TIMEOUT_S - time.monotonic()))
        self._drop_queue()
        _running.discard(self)

    def offer(
        self,
        frames: Mapping[str, str],
        observation: Mapping[str, Any],
        timestamp_s: float,
        is_kept: bool,
        episode: int,
        session_id: str | None,
    ) -> None:
        """Queues one tick the window accepted: its state, and its frames when they were kept; never raises.

        The state is encoded here, so the caller's arrays may change once this returns; the frames stay base64 until
        the thread sends them. A state JSON cannot carry (bytes, a set, NaN) or longer than
        #OBSERVATION_STATE_MAX_BYTES is dropped and counted, its frames still queued; a tick whose frame keys cannot
        name cameras is dropped whole. Nothing is queued once the recorder is disabled or stopping.

        @param frames The tick's frames, one base64 JPEG per frame key.
        @param observation The tick's other keys.
        @param timestamp_s The tick's timestamp, monotonic seconds, as the window took it.
        @param is_kept Whether the window kept the tick's frames.
        @param episode The client's episode number.
        @param session_id The session the client held when the tick was set, or None.
        """
        try:
            self._offer(frames, observation, timestamp_s, is_kept, episode, session_id)
        except Exception as exc:  # the caller's boundary: recording never raises into the control loop
            with self._cond:
                self._states_dropped += 1
            self._warn("offer", "recording dropped a tick it could not queue (%s: %s)", type(exc).__name__, exc)

    def hold(self) -> None:
        """Marks a get_action() in flight: the thread sends nothing until the matching release()."""
        with self._cond:
            self._held += 1

    def release(self) -> None:
        """Ends one hold(); the thread sends again once no get_action() is in flight."""
        with self._cond:
            self._held -= 1
            self._cond.notify_all()

    def stats(self) -> dict[str, Any]:
        """The recorder's state and counters, as `dream.DreamClient._recording_stats` answers them.

        @return A new dict: `state`, `robot_id`, `frames_sent`, `states_sent`, `frames_dropped`, `states_dropped`,
            `bytes_sent` (message payloads, framing left out), `queue_bytes` and `last_error`.
        """
        with self._cond:
            return {
                "state": self._state,
                "robot_id": self.robot_id,
                "frames_sent": self._frames_sent,
                "states_sent": self._states_sent,
                "frames_dropped": self._frames_dropped,
                "states_dropped": self._states_dropped,
                "bytes_sent": self._bytes_sent,
                "queue_bytes": self._queue_bytes,
                "last_error": self._last_error,
            }

    # -- the caller's side ---------------------------------------------------------------------------------------

    def _offer(
        self,
        frames: Mapping[str, str],
        observation: Mapping[str, Any],
        timestamp_s: float,
        is_kept: bool,
        episode: int,
        session_id: str | None,
    ) -> None:
        """offer() without its guard."""
        if self._is_stopping or self._state == DISABLED:
            return
        cameras = tuple(sorted(frames))
        if not _are_camera_names(cameras):
            with self._cond:
                self._states_dropped += 1
                self._frames_dropped += len(cameras) if is_kept else 0
            self._warn("cameras", "recording dropped a tick: frame keys %s cannot be camera names", list(cameras))
            return
        t_us = int((timestamp_s + self._offset_s) * 1_000_000)
        message = {
            "type": "observation_state",
            "t_us": t_us,
            "episode": episode,
            "session_id": session_id,
            "kept": is_kept,
            "observation": dict(observation),
        }
        try:
            text: str | None = json.dumps(message, default=_jsonable, allow_nan=False, separators=(",", ":"))
        except (TypeError, ValueError, RecursionError) as exc:
            text = None
            self._warn("state", "recording dropped a state JSON cannot carry (%s)", exc)
        if text is not None and len(text) > OBSERVATION_STATE_MAX_BYTES:
            self._warn("state", "recording dropped a state of %d bytes, over the cap", len(text))
            text = None
        is_overflowing = False
        with self._cond:
            if self._is_stopping or self._state == DISABLED:
                return
            if text is None:
                self._states_dropped += 1
            else:
                self._push(_State(cameras, text, len(text)))
            for camera in cameras if is_kept else ():
                seq = self._seqs.get(camera, 0)
                self._seqs[camera] = (seq + 1) & 0xFFFFFFFF
                frame_b64 = frames[camera]
                size_bytes = len(frame_b64) * 3 // 4
                # Padding makes the estimate up to 2 bytes high: only a frame surely over the cap is dropped here,
                # the rest are measured once decoded.
                if size_bytes - 2 > LIVE_MAX_AU_BYTES:
                    self._frames_dropped += 1
                    continue
                self._push(_Frame(cameras, camera, frame_b64, seq, t_us, size_bytes))
            while self._queue_bytes > RECORD_QUEUE_BYTES:
                oldest = self._queue.popleft()
                self._queue_bytes -= oldest.size_bytes
                self._count_dropped(oldest)
                is_overflowing = True
            self._cond.notify_all()
        if is_overflowing:
            self._warn("queue", "recording is behind: the queue is full, so the oldest messages are dropped")

    def _push(self, item: _State | _Frame) -> None:
        """Appends one message to the queue; called with the lock held."""
        self._queue.append(item)
        self._queue_bytes += item.size_bytes

    def _count_dropped(self, item: _State | _Frame) -> None:
        """Counts one message that will not be sent, as a frame or a state; called with the lock held."""
        if isinstance(item, _Frame):
            self._frames_dropped += 1
        else:
            self._states_dropped += 1

    def _drop_queue(self) -> None:
        """Drops and counts everything still queued."""
        with self._cond:
            for item in self._queue:
                self._count_dropped(item)
            self._queue.clear()
            self._queue_bytes = 0

    # -- the thread ----------------------------------------------------------------------------------------------

    def _run(self) -> None:
        """The thread: link, send, and on a fault wait and link again, until stop() or a refusal ends recording."""
        while True:
            cameras = self._wait_for_tick()
            if cameras is None:
                return
            self._set_state(CONNECTING)
            try:
                self._serve(cameras)
            except Exception as exc:  # the thread's boundary: every fault is counted, logged, and retried
                if self._is_stopping or not self._recover(exc):
                    return

    def _recover(self, exc: Exception) -> bool:
        """Records a link fault, then disables recording or waits out the backoff.

        @param exc What ended the link.
        @return Whether to dial again: False when a refusal disabled recording or stop() came during the wait.
        """
        code = self._close_code(exc)
        fault = _describe(exc) if code is None else f"closed {code} ({CLOSE_NAMES.get(code, 'unknown')})"
        with self._cond:
            self._last_error = fault
        if code in (CLOSE_RECORD_OFF, CLOSE_PROTOCOL):
            self._disable(code)
            return False
        self._attempt += 1
        delay_s = self._backoff_s(code)
        self._warn("link", "recording link lost (%s); dialing again in %.1f s", fault, delay_s)
        with self._cond:
            self._state = BACKOFF
            return not self._cond.wait_for(lambda: self._is_stopping, delay_s)

    def _wait_for_tick(self) -> tuple[str, ...] | None:
        """Waits for a message to send and no get_action() in flight.

        @return The cameras of the oldest message, the hello's to dial with; None once stop() is called.
        """
        with self._cond:
            self._cond.wait_for(lambda: self._is_stopping or (bool(self._queue) and self._held == 0))
            if self._is_stopping:
                return None
            return self._queue[0].cameras

    def _serve(self, cameras: tuple[str, ...]) -> None:
        """Dials the link, says hello, then sends until stop() or a message of other cameras ends it.

        @param cameras The hello's cameras.
        @throws Exception Any fault of the link: a network error, a close, a `bye`, no `hello_ok` in time.
        """
        from websockets.sync import client

        self._bye_code = None
        connection = client.connect(
            self._url,
            additional_headers=self._headers,
            compression=None,
            open_timeout=HELLO_TIMEOUT_S,
            close_timeout=CLOSE_TIMEOUT_S,
        )
        self._connection = connection
        try:
            with connection:
                if self._is_stopping:
                    return
                _bound_uplink(connection.socket)
                self._hello(connection, cameras)
                index = {camera: position for position, camera in enumerate(cameras)}
                while True:
                    item = self._take(connection, cameras)
                    if item is None:
                        return
                    self._send(connection, item, index)
        finally:
            self._connection = None

    def _hello(self, connection: "ClientConnection", cameras: tuple[str, ...]) -> None:
        """Sends the record-only `hello` and waits for `hello_ok`.

        @param connection The open link.
        @param cameras The hello's cameras.
        @throws _RefusedError The live service answered `bye`.
        @throws TimeoutError No answer within #HELLO_TIMEOUT_S.
        @throws ValueError The answer is not `hello_ok` for #LIVE_PROTOCOL.
        """
        hello = {
            "type": "hello",
            "protocol": LIVE_PROTOCOL,
            "robot_id": self.robot_id,
            "cameras": list(cameras),
            "width_px": 0,
            "fps": self._fps,
            "codec": CODEC,
            "face_blur": False,
            "agent_version": self._agent_version,
            "timezone": local_timezone(),
            "pose": False,
            "record_only": True,
        }
        connection.send(json.dumps(hello))
        try:
            answer = _parse(connection.recv(timeout=HELLO_TIMEOUT_S))
        except TimeoutError:
            raise TimeoutError(f"no hello_ok within {HELLO_TIMEOUT_S} s") from None
        if answer.get("type") == "bye" and isinstance(answer.get("code"), int):
            raise _RefusedError(answer["code"])
        if answer.get("type") != "hello_ok" or answer.get("protocol") != LIVE_PROTOCOL:
            raise ValueError(f"expected hello_ok for protocol {LIVE_PROTOCOL}, got {str(answer)[:120]}")
        self._attempt = 0
        self._set_state(LINKED)
        _log.debug("recording linked as robot %s with cameras %s", self.robot_id, list(cameras))

    def _take(self, connection: "ClientConnection", cameras: tuple[str, ...]) -> _State | _Frame | None:
        """The next message to send on this link, once no get_action() is in flight.

        Reads what the live service sent between looks, so a close is seen while the queue is empty too.

        @param connection The open link.
        @param cameras The link's cameras.
        @return The oldest message; None when the link should close: stop() was called and the flush is over (or a
            get_action() is in flight), or the oldest message is of other cameras (a new episode's).
        @throws websockets.exceptions.ConnectionClosed The live service closed the link.
        """
        while True:
            self._drain(connection)
            with self._cond:
                self._cond.wait_for(lambda: self._is_stopping or (bool(self._queue) and self._held == 0), _POLL_S)
                if self._is_stopping and (not self._queue or self._held or time.monotonic() >= self._flush_until_s):
                    return None
                if not self._queue or self._held:
                    continue
                item = self._queue[0]
                if item.cameras != cameras:
                    return None
                self._queue.popleft()
                self._queue_bytes -= item.size_bytes
                return item

    def _drain(self, connection: "ClientConnection") -> None:
        """Reads every message the live service sent and ignores it, keeping the code of a `bye`.

        @param connection The open link.
        @throws websockets.exceptions.ConnectionClosed The live service closed the link.
        """
        while True:
            try:
                message = connection.recv(timeout=0)
            except TimeoutError:
                return
            if isinstance(message, str):
                parsed = _parse(message)
                if parsed.get("type") == "bye" and isinstance(parsed.get("code"), int):
                    self._bye_code = parsed["code"]

    def _send(self, connection: "ClientConnection", item: _State | _Frame, index: dict[str, int]) -> None:
        """Sends one message: a state as text; a frame decoded, under its 16-byte header, as binary.

        A frame that is not base64, or larger than #LIVE_MAX_AU_BYTES once decoded, is dropped and counted.

        @param connection The open link.
        @param item The message.
        @param index The link's camera to its header index.
        """
        if isinstance(item, _State):
            connection.send(item.text)
            with self._cond:
                self._states_sent += 1
                self._bytes_sent += len(item.text)
            return
        try:
            jpeg = base64.b64decode(item.frame_b64)
        except (binascii.Error, ValueError) as exc:
            jpeg = None
            self._warn("frame", "recording dropped a frame of %s that is not base64 (%s)", item.camera, exc)
        if jpeg is not None and len(jpeg) > LIVE_MAX_AU_BYTES:
            self._warn("frame", "recording dropped a frame of %d bytes, over %d", len(jpeg), LIVE_MAX_AU_BYTES)
            jpeg = None
        if jpeg is None:
            with self._cond:
                self._frames_dropped += 1
            return
        camera = index[item.camera]
        header = struct.pack(
            VIDEO_HEADER_FORMAT, LIVE_PROTOCOL, KIND_VIDEO, camera, FLAG_KEYFRAME, item.seq, item.capture_us
        )
        connection.send(header + jpeg)
        with self._cond:
            self._frames_sent += 1
            self._bytes_sent += len(header) + len(jpeg)

    def _close_code(self, exc: Exception) -> int | None:
        """The live service's close code behind a fault: a `bye`'s, the close frame's (4000 to 4999), or None.

        @param exc What ended the link.
        @return The code, or None for a fault that carries none (a network error, a timeout).
        """
        from websockets import exceptions

        if isinstance(exc, _RefusedError):
            return exc.code
        if isinstance(exc, exceptions.ConnectionClosed) and exc.rcvd is not None and 4000 <= exc.rcvd.code <= 4999:
            return exc.rcvd.code
        return self._bye_code

    def _backoff_s(self, code: int | None) -> float:
        """The wait before the next dial: flat after a key refusal, else jittered and doubling with each failure.

        @param code The close code of the fault, or None.
        @return Seconds.
        """
        if code in (CLOSE_UNAUTHORIZED, CLOSE_NOT_FOUND, CLOSE_REVOKED):
            return REFUSAL_WAIT_S
        ceiling_s = min(LINK_RECONNECT_MAX_S, LINK_RECONNECT_MIN_S * 2 ** (self._attempt - 1))
        return random.uniform(LINK_RECONNECT_MIN_S, max(LINK_RECONNECT_MIN_S, ceiling_s))

    def _disable(self, code: int) -> None:
        """Ends recording for the rest of the policy: the state, one DEBUG line, and the queue dropped.

        @param code The refusal: #CLOSE_RECORD_OFF, or #CLOSE_PROTOCOL from a live service without record-only links.
        """
        self._set_state(DISABLED)
        reason = (
            "recording is not on for this platform"
            if code == CLOSE_RECORD_OFF
            else "the platform's live service does not take record-only links"
        )
        _log.debug(
            "recording is off for the rest of this policy: the live service closed the link %d (%s): %s",
            code,
            CLOSE_NAMES[code],
            reason,
        )
        self._drop_queue()

    def _set_state(self, state: str) -> None:
        """Sets the stats() `state` under the lock."""
        with self._cond:
            self._state = state

    def _abort(self) -> None:
        """Cuts the open link's socket, so a send, a dial's handshake or a wait for `hello_ok` ends at once."""
        connection = self._connection
        if connection is None:
            return
        try:
            connection.socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass  # already closed

    def _warn(self, kind: str, message: str, *args: object) -> None:
        """Logs one failure at DEBUG, unless one of its kind was logged within #_LOG_EVERY_S.

        @param kind The failure's kind: one line per kind per minute.
        @param message The `logging` format string.
        @param args Its arguments.
        """
        now_s = time.monotonic()
        last_s = self._logged_s.get(kind)
        if last_s is not None and now_s - last_s < _LOG_EVERY_S:
            return
        self._logged_s[kind] = now_s
        _log.debug(message, *args)


def link_url(platform_url: str) -> str:
    """The robot link of a platform: `wss://` for `https://`, `ws://` for `http://`, the same host and port.

    The API key rides the upgrade and must never cross a network in the clear, so `ws://` is accepted only to a
    loopback host (#LOOPBACK_HOSTS), as the robots agent's `robot_link()` does.

    @param platform_url The client's platform URL, `http(s)://<host>[:<port>]`.
    @return `ws(s)://<host>[:<port>]/live/robot`.
    @throws DreamConfigError The URL is not `http(s)://<host>[:<port>]`, or it is `http://` on a host that is not
        loopback; the message names it.
    """
    try:
        parts = urllib.parse.urlsplit(platform_url)
        port = parts.port
    except ValueError:
        parts, port = None, None
    if parts is None or parts.scheme not in ("http", "https") or not parts.hostname:
        raise DreamConfigError(
            f"recording sends to the platform, whose URL must be http(s)://<host>[:<port>], got {platform_url!r}"
        )
    if parts.scheme == "http" and parts.hostname not in LOOPBACK_HOSTS:
        raise DreamConfigError(
            f"recording sends the API key to the platform, so its URL must be https://; http:// is allowed "
            f"only to a loopback host {sorted(LOOPBACK_HOSTS)}, got {platform_url!r}"
        )
    host = f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname
    scheme = "wss" if parts.scheme == "https" else "ws"
    return f"{scheme}://{host}{'' if port is None else f':{port}'}{ROBOT_LINK_PATH}"


def websockets_missing() -> str | None:
    """Whether websockets can be imported: a dependency of the package, so only a broken install lacks it.

    @return None when it imports; else one line saying recording is off and how to install it.
    """
    try:
        import websockets.sync.client  # noqa: F401
    except ImportError:
        return "recording needs the websockets package, which is not installed: pip install 'websockets>=13'"
    return None


def off_stats(reason: str | None = None) -> dict[str, Any]:
    """`dream.DreamClient._recording_stats` when recording is off: state `off`, every count zero.

    @param reason Why the open policy could not start recording, as `last_error`; None when no policy is open.
    @return A new dict, with the keys of Recorder.stats().
    """
    return {
        "state": OFF,
        "robot_id": None,
        "frames_sent": 0,
        "states_sent": 0,
        "frames_dropped": 0,
        "states_dropped": 0,
        "bytes_sent": 0,
        "queue_bytes": 0,
        "last_error": reason,
    }


def local_timezone() -> str:
    """The machine's time zone for `hello`: the IANA name when the box says it, else its fixed UTC offset.

    `TZ` is read first, then the `/etc/localtime` link (`.../zoneinfo/America/Los_Angeles`); a box with neither
    yields the offset its clock reports now, `UTC-07:00`. The robots agent reads it the same way.

    @return The zone, at most #HELLO_MAX_FIELD_CHARS characters.
    """
    name = os.environ.get("TZ", "").strip().lstrip(":")
    if not name:
        try:
            target = os.readlink("/etc/localtime")
        except OSError:
            target = ""
        _, marker, tail = target.partition("zoneinfo/")
        name = tail if marker else ""
    if not name:
        offset = datetime.datetime.now().astimezone().utcoffset() or datetime.timedelta()
        minutes = int(offset.total_seconds() // 60)
        sign = "+" if minutes >= 0 else "-"
        name = f"UTC{sign}{abs(minutes) // 60:02d}:{abs(minutes) % 60:02d}"
    return name[:HELLO_MAX_FIELD_CHARS]


def _are_camera_names(cameras: tuple[str, ...]) -> bool:
    """Whether frame keys can be a hello's `cameras`: few enough, short enough, and never #POSE_STREAM."""
    if len(cameras) > HELLO_MAX_CAMERAS or POSE_STREAM in cameras:
        return False
    return all(len(camera) <= HELLO_MAX_FIELD_CHARS for camera in cameras)


def _jsonable(value: Any) -> Any:
    """`json.dumps`' fallback: a numpy array as nested lists, a numpy number as a number, a mapping as a dict.

    @param value What JSON cannot carry as it is.
    @return Its JSON form.
    @throws TypeError Anything else: bytes, a set, an object.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"{type(value).__name__} is not JSON")


def _parse(message: str | bytes) -> dict[str, Any]:
    """A message of the live service as a dict; anything that is not a JSON object is an empty one."""
    try:
        parsed = json.loads(message)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _bound_uplink(sock: Any) -> None:
    """Bounds what the kernel queues on the link, as the robots agent does, so a send in flight is short.

    `SO_SNDBUF` = #LIVE_SNDBUF_BYTES caps the bytes sent but unacknowledged plus unsent; `TCP_NOTSENT_LOWAT` =
    #LIVE_NOTSENT_LOWAT makes a send wait once that much is unsent. A transport that is not a socket (TLS through an
    https proxy) or a platform without the option is left as it is.

    @param sock The link's socket.
    """
    if not isinstance(sock, socket.socket):
        return
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, LIVE_SNDBUF_BYTES)
    if _TCP_NOTSENT_LOWAT is not None:
        sock.setsockopt(socket.IPPROTO_TCP, _TCP_NOTSENT_LOWAT, LIVE_NOTSENT_LOWAT)


def _describe(exc: Exception) -> str:
    """One line for stats() `last_error` and the log: the fault's type and message, cut short."""
    return f"{type(exc).__name__}: {exc}"[:200]


def _stop_all() -> None:
    """Stops every recorder still running, each within #STOP_TIMEOUT_S; run at interpreter exit."""
    for recorder in list(_running):
        recorder.stop()


atexit.register(_stop_all)
