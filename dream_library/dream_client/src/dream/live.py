"""Live View for any control loop: the robot's cameras, status and arms on the platform, and the page's buttons.

#LiveViewClient is the one class, and the one implementation of Live View: the station that `dreamcli robot up` runs
(`robots.station`) is an ordinary caller of it, so everything the page shows of a station, your own loop can show
too. Hand the client every observation with LiveViewClient.stream(); read the task a viewer typed with
LiveViewClient.get_prompt(); and register a callback, which runs with an #EventChange when a viewer presses a button.

@par Example
```python
import dream


def on_episode(event: dream.EventChange) -> None:
    print("viewer pressed", event.value)  # "start", or "end" (Stop, or a new task while running)


with dream.LiveViewClient("rbt_0123456789abcdef") as live:
    live.register_callback(on_episode)
    while running:
        live.stream(obs, "running")  # obs: the frames and state you hand DreamClient.set_observation(), in one mapping
        task = live.get_prompt()  # the task a viewer last sent, or None
```

The client keeps one outbound connection to the platform's live service, so the page shows your robot like any
station and nothing needs opening at your end. It opens on the first stream(), on a thread of its own, and redials
after a fault with a jittered backoff; stream() never waits on the network. What it does on that thread, and how it
stays off your loop:

- A camera is encoded only while a viewer watches it. Live View shows the robot; recording a run is the dream
  client's job, done from DreamClient.set_observation().
- Faces are blurred on the robot, in every encoded frame, before anything leaves it (`face_blur`, on by default),
  each camera with a detector of its own (`shared.faceblur`, which the library carries).
- Each camera has one encoder (PyAV's `libvpx`, VP8, one thread) and one worker thread; the client's threads run
  niced (#_NICE), so the scheduler prefers your loop whenever both are runnable. An unwatched camera's encoder
  closes #_IDLE_ENCODER_S after its last viewer leaves; an encoder that fails is reopened on the first frame
  after that delay.
- At most one unsent frame per camera: a stalled uplink overwrites it, counts a drop and makes the next frame a
  keyframe; the socket holds well under a second (#_WRITE_HIGH_WATER, #_NOTSENT_LOWAT, #_SNDBUF_BYTES). Nothing
  queues on the robot.
- Everything runs from the client's own thread, which reads your newest stream() #_POLL_HZ times a second; when
  and from where you call stream() changes nothing it does, so call it from your loop, a camera callback or a
  thread of your own, as often as you have something new.
- The tick guard watches that thread, never your loop: while the robot runs and a camera is encoded, a p95 of its
  poll interval past #_YIELD_TICK_RATIO poll periods for #_YIELD_WINDOW_S means the computer has no CPU to spare
  for video, so it closes every encoder, reports `live_paused: "tick"` and retries after #_YIELD_BACKOFF_S. Your
  loop's own pace (a sync hold, a slow `get_action()`, a camera's jitter) never pauses the video.

The link is the platform's robot link, `ROBOT_LINK_PATH` on the platform's host, defined by `shared.contracts.live`
in the platform repository, which this package cannot import; the names it needs are restated below as private
constants, each naming the contract's, and a test in the platform repository holds them to it.

@see LiveViewClient, EventChange
"""

from __future__ import annotations

import asyncio
import base64
import collections
import concurrent.futures
import dataclasses
import datetime
import enum
import fractions
import importlib.util
import inspect
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
from collections.abc import Callable, Mapping, Sequence
from types import TracebackType
from typing import Any

import numpy as np

import dream
from dream import credentials as _credentials
from dream.errors import DreamConfigError, DreamObservationError

## The robot link's path on the platform (`ROBOT_LINK_PATH` of the contract).
_ROBOT_LINK_PATH = "/live/robot"

## Protocol version, sent in `hello` and in every video header (`LIVE_PROTOCOL`).
_PROTOCOL = 1

## Encoded width by default; the height follows each camera's aspect (`LIVE_WIDTH_PX`).
_WIDTH_PX = 640

## Encoded frames per second at most, per camera, by default (`LIVE_FPS`).
_FPS = 30

## Average bitrate per camera by default (`LIVE_BITRATE_KBPS`).
_BITRATE_KBPS = 1400

## The encoder's rate buffer (`LIVE_VBV_BUFFER_KBIT`).
_VBV_BUFFER_KBIT = 700

## Seconds between two keyframes (`LIVE_GOP_S`): a viewer who skipped resumes within a second.
_GOP_S = 1.0

## The PyAV encoders the client may open, each with the WebCodecs codec string the page decodes it with, which
## `hello` announces. Both are in PyAV's wheel and BSD licensed; the wheel's H.264 encoder is GPL and is not offered.
_CODECS = {"libvpx": "vp8", "libvpx-vp9": "vp09.00.10.08"}

## The encoder by default: VP8.
_ENCODER = "libvpx"

## libvpx's `cpu-used` in `realtime` mode, 0 (slowest, best) to 16: 8 encodes a 640x360 stream at over 100 fps on one
## core, so one camera costs well under a third of a core (`LIVE_ENCODER_SPEED`).
_ENCODER_SPEED = 8

## Threads per encoder: one core at most per camera; an encoder's default pool would take every core
## (`LIVE_ENCODER_THREADS`).
_ENCODER_THREADS = 1

## Looks at the newest observation this many times a second: twice the frame rate, so a new frame is picked up
## within half a frame period (`LIVE_POLL_HZ`).
_POLL_HZ = 60

## Seconds an unwatched camera keeps its encoder: survives a page reload without a cold start
## (`LIVE_IDLE_ENCODER_S`).
_IDLE_ENCODER_S = 3

## The niceness of the client's threads: your loop and your model's client win whenever both are runnable
## (`LIVE_AGENT_NICE`).
_NICE = 10

## A p95 of the client's poll interval this many poll periods long: its niced thread cannot keep half its rate, so
## the computer has no CPU to spare for video (`LIVE_YIELD_TICK_RATIO`).
_YIELD_TICK_RATIO = 2.0

## Seconds of late polls before the client yields; one GC pause or one keyframe does not trip it
## (`LIVE_YIELD_WINDOW_S`).
_YIELD_WINDOW_S = 2

## Seconds a yielded client waits before encoding again; a computer that cannot afford video must not oscillate
## (`LIVE_YIELD_BACKOFF_S`).
_YIELD_BACKOFF_S = 30

## Fewest poll intervals a yield window needs before it is trusted.
_MIN_TICK_SAMPLES = 10

## OpenCV's worker pool, for the whole process unless `OPENCV_FOR_THREADS_NUM` says otherwise: its idle workers
## spin, and two keep a 720p resize and blur inside a frame period.
_OPENCV_THREADS = 2

## The frame resized once at start-up to create OpenCV's worker pool from the client's own niced thread.
_POOL_WARM_SHAPE = (720, 1280, 3)

## The tolerance of the frame-rate cap: a frame this close to its due time is early jitter, not a fast camera.
_DUE_TOLERANCE_S = 1e-6

## Observation keys that carry a camera frame; the prefix is dropped to name the camera on the page.
_FRAME_PREFIXES = ("video.", "image_window.")

## The page's names for the model's camera roles: it shows `top` as the main picture and `left` and `right` in its
## side panels; any other role keeps its own name. Not applied to the cameras the constructor names.
_PAGE_CAMERAS = {"overhead": "top", "wrist_l": "left", "wrist_r": "right"}

## Seconds to wait for the upgrade and for `hello_ok` after it.
_HELLO_TIMEOUT_S = 10.0

## The link's heartbeat and how long a pong may take (`LINK_PING_S`, `LINK_PONG_TIMEOUT_S`).
_LINK_PING_S = 5
_LINK_PONG_TIMEOUT_S = 10

## Shortest backoff between dials after a fault, seconds (`LINK_RECONNECT_MIN_S`); the ceiling doubles per failed dial.
_RECONNECT_MIN_S = 1.0

## Longest backoff, seconds (`LINK_RECONNECT_MAX_S`).
_RECONNECT_MAX_S = 30.0

## The contract's close codes (`CLOSE_CODES`), 4001 to 4011: a close with one of them is the platform's verdict.
_CLOSE_CODES = frozenset(range(4001, 4012))

## Close codes the platform must fix (a key it does not know, a robot id it does not know for that key, a revoked
## key): redialed after #_RECONNECT_MAX_S, never sooner.
_REFUSED_CODES = frozenset({4001, 4004, 4005})

## The close code of a message that is not protocol 1 (`CLOSE_PROTOCOL`).
_CLOSE_PROTOCOL = 4002

## Bytes the socket may buffer in asyncio before a send waits (`LIVE_WRITE_HIGH_WATER`), and unsent in the kernel
## before the socket stops being writable (`LIVE_NOTSENT_LOWAT`), as the relay bounds its viewers' sockets.
_WRITE_HIGH_WATER = 64 << 10
_NOTSENT_LOWAT = 16 << 10

## `SO_SNDBUF` on the link, which the kernel doubles: at most 256 KiB queued (sent and unsent) instead of the
## autotuned `tcp_wmem` maximum of megabytes; enough for three cameras at 1.4 Mbps across a 300 ms round trip.
_SNDBUF_BYTES = 128 << 10

## The socket option number: Python's `socket` module lacks it on some builds, so Linux's 25 is the fallback.
_TCP_NOTSENT_LOWAT: int | None = getattr(socket, "TCP_NOTSENT_LOWAT", 25 if sys.platform == "linux" else None)

## How long close() waits for the client's thread.
_CLOSE_TIMEOUT_S = 5.0

## Seconds a link's closing handshake may take before its socket is aborted: on a path that went dead (a Wi-Fi
## uplink dropped while the box's Ethernet came up) the close frame waits behind bytes the relay never acknowledges,
## and websockets waits for them to drain with no deadline, so the client would sit for the kernel's TCP timeout
## (about 15 minutes) instead of redialing. Under #_CLOSE_TIMEOUT_S, so close() still joins the thread.
_LINK_CLOSE_S = 2.0

## Hosts the link may dial over plain `ws://`: this machine, so a key never crosses a network in the clear.
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

## Seconds a callback's work may take before the page hears `timeout` (`PROMPT_ACK_S`).
_PROMPT_ACK_S = 10

## The longest task, checkpoint and step count a `prompt` may carry (`PROMPT_MAX_CHARS`, `CHECKPOINT_MAX_CHARS`,
## `RUN_STEPS_MAX`), and the action spaces it may name (`ActionSpace`).
_PROMPT_MAX_CHARS = 200
_CHECKPOINT_MAX_CHARS = 128
_RUN_STEPS_MAX = 100_000
_ACTION_SPACES = ("eef", "joint")

## What a `prompt` may ask (`PromptAction`).
_PROMPT_ACTIONS = ("prompt", "start", "stop", "home", "reset")

## The joint-state stream's name, never a camera's (`POSE_STREAM`), and its rate while a viewer holds it (`POSE_HZ`).
_POSE_STREAM = "pose"
_POSE_HZ = 10

## The longest `status.error` and the longest `hello` field or `status.error_code` (`STATUS_ERROR_MAX_CHARS`,
## `HELLO_MAX_FIELD_CHARS`).
_STATUS_ERROR_MAX_CHARS = 500
_FIELD_MAX_CHARS = 64

## Seconds between two statuses that only carry new round trips (`TIMING_STATUS_S`); a state change goes at once.
_TIMING_STATUS_S = 2.0

## Round trips a status lists in `recent_ms` at most (`TIMING_RECENT_MAX`), and the largest one (`TIMING_MAX_MS`).
_TIMING_RECENT_MAX = 20
_TIMING_MAX_MS = 600_000.0

## The `live_paused` value while the tick guard has the encoders closed.
_PAUSED_TICK = "tick"

## The video header: u8 version, u8 kind, u8 camera index, u8 flags, u32 seq, u64 capture time in us.
_HEADER = struct.Struct(">BBBBIQ")
_KIND_VIDEO = 1
_FLAG_KEYFRAME = 0x01

## What a computer without the encoder's packages hears, naming where they come from.
_MISSING_PACKAGES = (
    "Live View needs PyAV, OpenCV and websockets: install the dream library, which carries all three: "
    "pip install --upgrade dreamlabs"
)

## The fields of each message a relay sends a robot.
_RELAY_FIELDS: dict[str, set[str]] = {
    "hello_ok": {"node_id", "protocol", "subscribed"},
    "subscribe": {"camera"},
    "unsubscribe": {"camera"},
    "bye": {"code", "reason"},
    "prompt": {"id", "action", "text", "checkpoint", "action_space", "steps"},
}

## The robot link's other message types: a relay never sends them to a robot, and one that does is ignored.
_ROBOT_SENT_TYPES = frozenset({"hello", "status", "tick", "observation_state", "pose", "prompt_ack"})

_log = logging.getLogger(__name__)

## The clock a new frame's capture time is read from, monotonic seconds; the tests step it to stand in for cameras
## faster or slower than real time.
_capture_s = time.monotonic


class EventChange(enum.Enum):
    """What a viewer did from the Live View page; the argument of every registered callback.

    - start: Start was pressed; get_prompt() holds the task the run starts with, and, with `model_fixed=False`,
      get_model() and get_steps() hold the model, the action space and the step count the viewer chose.
    - end: Stop was pressed; or a new task was sent (get_prompt() now holds it), for a callback that did not
      register for `prompt`.
    - prompt: A new task was sent, for a callback registered with `prompt=True`; get_prompt() holds it.
    - home: Home was pressed, for a callback registered with `home=True`: move the arms home.
    - reset: Reset arms was pressed, for a callback registered with `reset=True`: reopen the arms' drivers and home.
    """

    start = "start"
    end = "end"
    prompt = "prompt"
    home = "home"
    reset = "reset"


## The event each press of the page is, for a callback registered for it (`prompt` is `end` for the others).
_PRESSES = {
    "prompt": EventChange.prompt,
    "start": EventChange.start,
    "stop": EventChange.end,
    "home": EventChange.home,
    "reset": EventChange.reset,
}


class LiveViewClient:
    """Shows one robot on the platform's Live View page: its cameras, status and arms, and the buttons viewers press.

    The key resolves as DreamClient's does: the argument, else your credentials. Where the platform is comes from your
    credentials and cannot be changed from code. The constructor checks the
    encoder by opening one; nothing is dialed until the first stream(), and close(), or leaving the `with` block,
    ends the link. Callbacks run on the client's thread, so keep them short; a callback that has seconds of work
    (moving arms) returns it as an awaitable, which the client waits for before it answers the page.

    The client runs on a thread of its own, which reads your newest stream() #_POLL_HZ times a second and does all of
    the work from there, so it never runs on, or waits for, your control loop. A camera is encoded only while a
    viewer watches it, and faces are blurred on the robot before any frame leaves it, unless `face_blur=False`.
    Live View shows the robot; recording a run is the dream client's job, done from
    DreamClient.set_observation().

    Given the loop's `DreamClient`, the client also reports the model's round trips while the robot is `running`, as
    the station does: the page waits for the first one before it shows the run as answered ("Dream-1 is getting
    ready" until then), and draws its latency from them.

    @see EventChange, stream(), register_callback()
    """

    def __init__(
        self,
        robot_id: str,
        *,
        api_key: str | None = None,
        dream_client: Any = None,
        face_blur: bool = True,
        model_fixed: bool = True,
        cameras: Sequence[str] | None = None,
    ) -> None:
        """Resolves the key and the link and opens one encoder to prove it can; nothing runs until the first stream().

        @param robot_id The robot as the platform knows it, `rbt_` and 16 hex digits: the `robot_id` under `live:` in
            the station file (`~/.dreamlabs/station.yaml`) that the setup wizard's Save writes when it registers the
            robot. The Live View page lists robots by name, never by id.
        @param api_key Your API key; the credentials' (`DREAM_API_KEY`) when None.
        @param dream_client The loop's `dream.DreamClient`, whose `last_timing` stream() reads to report each new
            round trip to the page, whose `checkpoint` the page shows as the model and whose `session_id` names the
            run. Without it the page never sees an answer and shows the run as getting ready. A loop that opens a new
            DreamClient (one per run, as the station does) sets the #dream_client attribute to it.
        @param face_blur Blur every face in every encoded frame before it leaves the robot, with a detector per
            camera (`shared.faceblur`), and say so to the platform. On by default; off sends the frames as captured.
        @param model_fixed True (the default) when your code chooses the model: the page shows the one you run and
            offers no model or action choice. False lets a viewer pick them: a Start then carries the checkpoint and
            the action space, which get_model() returns, and your loop runs them.
        @param cameras The cameras a viewer may watch, in this order, each named as it is in your observation's keys
            (`video.<name>`) and shown under that name, not renamed; stream() may then carry any of them, a camera
            with no frame yet left out, and a frame of another name is not shown. None: the cameras of the first
            stream(), the model's roles renamed as the page names them (`overhead` is `top`).
        @throws DreamConfigError No key is set anywhere, `robot_id` is empty, the platform is `http://` on a host
            other than this one (the key would cross the network in the clear), PyAV, OpenCV or websockets is not
            installed, the encoder does not open in this PyAV build, `face_blur` is on and the face detector is not
            installed, or `cameras` names a camera twice, an empty name or `pose`.
        """
        if not robot_id:
            raise DreamConfigError(
                "robot_id is empty: pass the robot's id, the robot_id under live: in the station file the setup "
                "wizard writes"
            )
        loaded = _credentials.load()
        key = loaded.api_key if api_key is None else api_key
        if not key:
            raise DreamConfigError(
                f"Live View needs your API key: pass api_key, set {_credentials.API_KEY_ENV}, or write it to "
                f"{loaded.path} with the lines your platform's Setup API page gives"
            )
        self._robot_id = robot_id
        self._api_key = key
        # Where the platform is is the credentials' alone: read once here, never set from code.
        self._url = _robot_link(loaded.platform_url)
        names = None if cameras is None else _checked_cameras(cameras)
        self.dream_client = dream_client
        self._names = names
        self._face_blur = face_blur
        self._model_fixed = model_fixed
        _check_installed(face_blur)
        _open_encoder(_ENCODER, _WIDTH_PX, _height_px(4, 3, _WIDTH_PX), _FPS, _BITRATE_KBPS).close()
        # The wall clock minus the monotonic one, in whole microseconds: a stamp is `_wall_us()` of a monotonic time,
        # so the ~1.8e9 s offset is never added in floating point, where a double resolves only about 0.24 us and two
        # frames 50 ms apart could be stamped 49999 or 50001 us apart.
        self._wall_offset_us = round((time.time() - time.monotonic()) * 1e6)
        # Everything below the lock is shared by the caller's thread, the link's loop and the encode workers.
        self._lock = threading.Lock()
        self._callbacks: list[tuple[Callable[[EventChange], object], frozenset[EventChange]]] = []
        self._prompt: str | None = None
        self._model: tuple[str | None, str | None] = (None, None)
        self._steps: int | None = None
        self._latest: _Snapshot | None = None
        self._cameras: tuple[str, ...] = () if names is None else names
        self._streams = {name: _Stream(name, index) for index, name in enumerate(self._cameras)}
        self._is_closed = False
        self._thread: threading.Thread | None = None
        self._watch = _PollWatch()
        self._runs = 0
        self._was_running = False
        # The round trips of the run under way (`running`): the newest report, every round trip of the run, those
        # since the last status that carried them, and the run's count; a DreamClient reading or the caller's
        # `timing` count identifies a new one.
        self._timing: dict[str, Any] | None = None
        self._run_rtts_ms: list[float] = []
        self._recent_ms: list[float] = []
        self._timing_count = 0
        self._last_reading: Any = None
        self.is_linked = False
        self.node_id: str | None = None
        self.reconnects = 0
        self.errors = 0
        self.yields = 0
        self.live_paused: str | None = None
        self.is_pose_subscribed = False
        # Owned by the link's loop thread.
        self._next_pose_s = 0.0
        self._pending_pose: str | None = None
        self._is_status_due = False
        self._last_status: tuple[Any, ...] | None = None
        self._sent_timing_count = 0
        self._next_timing_s = 0.0
        self._last_fault: str | None = None
        self._has_linked = False
        self._prompt_tasks: set[asyncio.Future[None]] = set()
        self._bye_code: int | None = None
        self._paused_until_s: float | None = None
        # Set by the link's thread once its loop runs.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._workers: concurrent.futures.ThreadPoolExecutor | None = None
        self._wake: asyncio.Event | None = None
        # Set by close() alone: the backoff waits on it, never on _wake, which every frame and status sets.
        self._stop: asyncio.Event | None = None

    def __enter__(self) -> LiveViewClient:
        """Returns the client; leaving the block calls close()."""
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        """Calls close().

        @param exc_type The exception's type, when the block raised.
        @param exc The exception, when the block raised.
        @param tb Its traceback.
        """
        self.close()

    def register_callback(
        self,
        callback_func: Callable[[EventChange], object],
        *,
        home: bool = False,
        reset: bool = False,
        prompt: bool = False,
    ) -> None:
        """Adds a callback for Start, Stop and a new task from the page, and for the presses it opts in to.

        The callback runs on the client's thread with the #EventChange. It may return nothing (the page hears the
        press applied), the robot's state after it (`"running"`), a mapping with `state` and, when the press
        failed, `error` (its message), or an awaitable (or a `concurrent.futures.Future`, such as
        `asyncio.run_coroutine_threadsafe()` returns) that resolves to one of those. The client waits for an
        awaitable up to the page's deadline (10 s), then answers the page: an exception, or an `error`, is a
        failure with its message; past the deadline the page hears `timeout`. A Start, a Stop, a Home or a Reset
        is never cancelled halfway, deadline or not: the work goes on and the next status says where it ended. Only
        a new task's awaitable is cancelled at the deadline.

        @param callback_func Called with `EventChange.start` on Start, `EventChange.end` on Stop, and on a new task
            `EventChange.prompt` with `prompt=True`, else `EventChange.end`. Keep it short: record the event for your
            loop to act on, as the example on the Live View page does, or return the work as an awaitable.
        @param home Also call it with `EventChange.home` when a viewer presses Home. While no callback takes Home,
            the page hears "not supported".
        @param reset Also call it with `EventChange.reset` when a viewer presses Reset arms; the same holds.
        @param prompt Call it with `EventChange.prompt`, rather than `EventChange.end`, when a viewer sends a task.
        """
        events = {EventChange.start, EventChange.end}
        if home:
            events.add(EventChange.home)
        if reset:
            events.add(EventChange.reset)
        if prompt:
            events.add(EventChange.prompt)
        with self._lock:
            self._callbacks.append((callback_func, frozenset(events)))

    def get_prompt(self) -> str | None:
        """The task a viewer last sent from the page; after a Start without one, the `prompt` you last streamed.

        @return The task's text, or None until a viewer sends one.
        """
        with self._lock:
            return self._prompt

    def get_model(self) -> tuple[str | None, str | None]:
        """The model a viewer chose with the last Start, for a client built with `model_fixed=False`.

        @return `(checkpoint, action_space)`: the checkpoint slug and `"eef"` or `"joint"`, each None when the Start
            named none (keep the one you run); `(None, None)` before the first Start and with `model_fixed=True`.
        """
        with self._lock:
            return self._model

    def get_steps(self) -> int | None:
        """How many steps the last Start asked for, which the page offers while your status carries `steps`.

        @return The step count, at least 1, or None when the Start named none (keep your count).
        """
        with self._lock:
            return self._steps

    def stats(self) -> dict[str, Any]:
        """The client's own counters, for your loop's logs or status page; safe from any thread.

        @return `linked`, `node` (the live node holding the link), `viewers` (None: the link carries camera
            subscriptions, not viewer counts), `streams` (the watched cameras), `pose` (whether a viewer holds the
            arms' stream), `dropped_frames` per camera, `reconnects`, `errors` (encoder faults and closes with a
            code), `yields` (times the tick guard paused the video) and `live_paused`.
        """
        with self._lock:
            streams = list(self._streams.values())
        return {
            "linked": self.is_linked,
            "node": self.node_id,
            "viewers": None,
            "streams": [stream.camera for stream in streams if stream.is_subscribed],
            "pose": self.is_pose_subscribed,
            "dropped_frames": {stream.camera: stream.dropped for stream in streams},
            "reconnects": self.reconnects,
            "errors": self.errors,
            "yields": self.yields,
            "live_paused": self.live_paused,
        }

    def stream(
        self,
        obs: Mapping[str, Any],
        status: str,
        *,
        arms: Mapping[str, Any] | None = None,
        error: str | tuple[str, str | None] | None = None,
        prompt: str | None = None,
        run: int | None = None,
        model: tuple[str | None, str | None] | None = None,
        steps: int | None = None,
    ) -> None:
        """Hands the client the newest observation and the robot's status; returns at once.

        Call it whenever you have something new, from any thread and at any rate: the client reads the newest call on
        its own thread #_POLL_HZ times a second, so your loop's pace never reaches the video. The first call opens the
        link; without `cameras` in the constructor it
        also names the cameras, and every later call carries the same ones (with `cameras`, a camera with no frame
        yet may be left out). A frame is encoded once: hand the client a new array or JPEG for each new camera frame,
        and the same one again while the camera has delivered nothing new. Every keyword is optional; each one the
        page shows is shown only once you pass it.

        @param obs The observation, DreamClient.set_observation()'s frames and state in one mapping: each
            `video.<role>` (or `image_window.<role>`) key holds one frame and is a camera on the page: `overhead` the
            main picture (`top`), `wrist_l` and `wrist_r` the side panels (`left`, `right`), any other role under its
            own name. A frame is a base64 JPEG, or an RGB `uint8` array `[height, width, 3]` (the client keeps a
            reference, so hand it a new array per frame and the same one while it has not changed). Every other key
            is ignored here.
        @param status What the page shows as the robot's state: `idle`, `running`, `error`, or any short word. Report
            `running` from a viewer's Start until the episode ends: the page shows the task and its Stop button only
            while the robot is `running`, and offers Start only while it is `idle` or `error`.
        @param arms Each arm's joints as they are now, by name (`{"left": (q, gripper), "right": (q, gripper)}`):
            `q` the joint angles in radians, `gripper` the opening from 0 closed to 1 open, or None for an arm that
            cannot be read. The page draws the robot from them while a viewer looks (the `pose` stream, sent only
            when the first call carries `arms`), and shows an arm whose value is None as not answering.
        @param error Why the arms are not moving, for the page's banner: a message, or `(message, code)` with a
            short code for it (`pool_warming`, `ArmError`); None while nothing is wrong.
        @param prompt The task your robot runs or will run, when your code sets it: the status shows it rather than
            the one a viewer last sent, and a Start without a task runs it (get_prompt() then returns it).
        @param run Your run number, which the status carries; the client counts runs itself, one per
            stretch of `running`, when left out.
        @param model `(checkpoint, action_space)` your loop runs, which the page shows (`action_space` `"eef"` or
            `"joint"`, or None); the `dream_client`'s checkpoint when left out.
        @param steps How many steps your next run takes, 0 for no limit; the page then offers a step count, which a
            Start carries (get_steps()).
        @throws DreamObservationError `obs` is not a mapping, a frame is neither a base64 JPEG nor an RGB array,
            without `cameras`, `obs` carries no frame or not the cameras of the first call; `status` is empty; or a
            keyword is not of the shape above.
        """
        captured_now_s = _capture_s()
        if not isinstance(obs, Mapping):
            raise DreamObservationError(f"obs must be a mapping, got {type(obs).__name__}")
        if not isinstance(status, str) or not status:
            raise DreamObservationError("status must be a non-empty str, such as 'running'")
        frames = self._frames_of(obs)
        snapshot = _Snapshot(
            status=status,
            frames=frames,
            captured_s={},
            arms=None if arms is None else _checked_arms(arms),
            error=_checked_error(error),
            prompt=prompt,
            run=run,
            model=model,
            steps=steps,
        )
        with self._lock:
            if self._is_closed:
                return
            # A frame keeps its capture time while the caller hands the same object again: it is not encoded twice.
            last = self._latest
            for camera, frame in frames.items():
                if last is not None and last.frames.get(camera) is frame:
                    snapshot.captured_s[camera] = last.captured_s[camera]
                else:
                    snapshot.captured_s[camera] = captured_now_s
            if not self._cameras:
                self._cameras = tuple(frames)
                self._streams = {name: _Stream(name, index) for index, name in enumerate(self._cameras)}
            elif self._names is None and set(frames) != set(self._cameras):
                raise DreamObservationError(
                    f"obs carries cameras {sorted(frames)}, but the first call named {sorted(self._cameras)}"
                )
            is_running = status == "running"
            if is_running and not self._was_running:
                self._runs += 1
            self._was_running = is_running
            self._note_round_trip(snapshot)
            self._latest = snapshot
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="dream-live", daemon=True)
                self._thread.start()

    def close(self) -> None:
        """Ends the link, releases the encoders and joins the client's thread; idempotent."""
        with self._lock:
            self._is_closed = True
            thread = self._thread
        self._poke(self._stop)
        if thread is not None:
            thread.join(_CLOSE_TIMEOUT_S)

    # What stream() keeps, on the caller's thread and under the lock

    def _frames_of(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        """The frames of `obs` by page camera name.

        @param obs The observation.
        @return Frame per page camera name.
        @throws DreamObservationError A frame is neither a base64 JPEG nor an RGB array, or there is none without
            the constructor's `cameras`.
        """
        frames: dict[str, Any] = {}
        for key, value in obs.items():
            camera = _camera_name(key, self._names is None)
            if camera is None or (self._names is not None and camera not in self._names):
                continue
            is_jpeg = isinstance(value, str) and bool(value)
            is_rgb = isinstance(value, np.ndarray) and value.dtype == np.uint8 and value.ndim == 3
            if not (is_jpeg or (is_rgb and value.shape[2] == 3)):
                raise DreamObservationError(
                    f"obs[{key!r}] must be a base64 JPEG str or an RGB uint8 array [height, width, 3]"
                )
            frames[camera] = value
        if not frames and self._names is None:
            raise DreamObservationError("obs carries no frame: a video.<role> key holding a base64 JPEG")
        return frames

    def _note_round_trip(self, snapshot: _Snapshot) -> None:
        """Counts a new round trip of the model while `running`; anything else starts the next run's count.

        Called under the lock. A DreamClient reading is new when its round trip or its bytes differ from the last one
        seen. The reading at any status other than `running`
        is the baseline, so a request from before the run (the previous run's last one) is never counted as this
        run's first answer.

        @param snapshot What stream() was just given.
        """
        reading = None if self.dream_client is None else self.dream_client.last_timing
        key: Any = None
        if reading:
            key = (reading.get("rtt_ms"), reading.get("request_bytes"), reading.get("response_bytes"))
        if snapshot.status != "running":
            # Whatever the loop last did before this run is the baseline: only round trips after it are this run's.
            self._timing, self._run_rtts_ms, self._recent_ms, self._timing_count = None, [], [], 0
            self._last_reading = key
            return
        if reading is None or reading.get("rtt_ms") is None or key == self._last_reading:
            return
        self._last_reading = key
        rtt_ms = _clamp_ms(reading["rtt_ms"])
        self._recent_ms = [*self._recent_ms, rtt_ms][-_TIMING_RECENT_MAX:]
        self._run_rtts_ms.append(rtt_ms)
        rtts = sorted(self._run_rtts_ms)
        self._timing = {
            **reading,
            "p50_ms": rtts[len(rtts) // 2],
            "p95_ms": rtts[min(len(rtts) - 1, int(len(rtts) * 0.95))],
        }
        self._timing_count += 1

    def _session_of(self, snapshot: _Snapshot) -> str | None:
        """The session the status names: the `dream_client`'s, while `running` only."""
        if snapshot.status != "running":
            return None
        session = None if self.dream_client is None else getattr(self.dream_client, "session_id", None)
        return session if isinstance(session, str) else None

    def _wall_us(self, monotonic_s: float) -> int:
        """The robot wall clock at a monotonic time, microseconds: exact differences for exact monotonic ones."""
        return round(monotonic_s * 1e6) + self._wall_offset_us

    # The thread and the reconnect loop

    def _poke(self, event: asyncio.Event | None) -> None:
        """Sets one of the link loop's events from any thread; nothing before the loop runs or after it ends."""
        loop = self._loop
        if loop is None or event is None:
            return
        try:
            loop.call_soon_threadsafe(event.set)
        except RuntimeError:
            pass  # The loop has ended (after close()): there is nothing left to wake.

    def _run(self) -> None:
        """The client's thread: niced, OpenCV's pool born here, one encode worker per camera, the reconnect loop."""
        try:
            os.nice(_NICE)
        except OSError as exc:
            _log.warning("live: os.nice(%d) refused: %s", _NICE, exc)
        _warm_opencv_pool()
        # One encode worker per camera, created lazily by the first submit from this niced thread, so they inherit
        # the niceness; three cameras then resize, blur and encode in parallel instead of in turn on the poll.
        self._workers = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, len(self._cameras)), thread_name_prefix="live-encode"
        )
        try:
            asyncio.run(self._run_forever())
        finally:
            self._loop = None
            self._workers.shutdown(wait=False)

    async def _run_forever(self) -> None:
        """Dials, serves one link, backs off, repeats until close(); the thread boundary that logs every fault."""
        self._loop = asyncio.get_running_loop()
        self._wake = asyncio.Event()
        self._stop = asyncio.Event()
        if self._is_closed:
            return
        attempt = 0
        while not self._stop.is_set():
            code = None
            try:
                await self._serve_link()
            except Exception as exc:  # The thread's boundary: logged and counted, and the link is dialed again.
                code = self._close_code(exc)
                self._note_fault(f"{type(exc).__name__}: {exc}" if code is None else f"closed {code}")
                if code is not None:
                    self.errors += 1
            self._unlink()
            if self._stop.is_set():
                break
            attempt += 1
            await self._wait_backoff(attempt, code)
            self._close_idle_encoders(time.monotonic())
        self._close_encoders()

    async def _wait_backoff(self, attempt: int, code: int | None) -> None:
        """Sleeps the jittered exponential backoff, or #_RECONNECT_MAX_S after a refusal; close() cuts it short."""
        assert self._stop is not None
        if code in _REFUSED_CODES:
            # The key or the robot id is wrong, or the key was revoked: redialing sooner cannot help.
            delay_s = _RECONNECT_MAX_S
        else:
            ceiling_s = min(_RECONNECT_MAX_S, _RECONNECT_MIN_S * 2 ** (attempt - 1))
            delay_s = random.uniform(_RECONNECT_MIN_S, max(_RECONNECT_MIN_S, ceiling_s))
        try:
            await asyncio.wait_for(self._stop.wait(), delay_s)
        except asyncio.TimeoutError:
            pass

    def _close_code(self, exc: Exception) -> int | None:
        """The application close code of a fault, from the close frame or the `bye` before it; None for the rest."""
        import websockets

        if isinstance(exc, websockets.exceptions.ConnectionClosed) and exc.rcvd is not None:
            if exc.rcvd.code in _CLOSE_CODES:
                return int(exc.rcvd.code)
        return self._bye_code

    def _note_fault(self, fault: str) -> None:
        """Logs a fault once per change: the first time at WARNING, repeats at DEBUG."""
        if fault != self._last_fault:
            _log.warning("live: link lost (%s); reconnecting with backoff", fault)
            self._last_fault = fault
        else:
            _log.debug("live: link lost again (%s)", fault)

    def _unlink(self) -> None:
        """Forgets the link: every camera is unsubscribed (its encoder idles out) and pending units are dropped."""
        was_linked = self.is_linked
        self.is_linked = False
        self.node_id = None
        self._bye_code = None
        now_s = time.monotonic()
        for stream in self._streams.values():
            stream.pending = None
            if stream.is_subscribed:
                stream.is_subscribed = False
                stream.idle_since_s = now_s
        with self._lock:
            self._watch.stop_encoding()
        if was_linked:
            _log.info("live: unlinked")

    # One link

    async def _serve_link(self) -> None:
        """Dials the relay, says hello, then runs the poll, send and receive tasks until one ends or close()."""
        assert self._stop is not None
        from websockets.asyncio import client

        link = await client.connect(
            self._url,
            additional_headers={"Authorization": f"Bearer {self._api_key}"},
            compression=None,
            ping_interval=_LINK_PING_S,
            ping_timeout=_LINK_PONG_TIMEOUT_S,
            open_timeout=_HELLO_TIMEOUT_S,
            write_limit=_WRITE_HIGH_WATER,
        )
        try:
            _bound_uplink(link)
            await self._hello(link)
            tasks = [
                asyncio.ensure_future(self._poll_loop()),
                asyncio.ensure_future(self._send_loop(link)),
                asyncio.ensure_future(self._recv_loop(link)),
                asyncio.ensure_future(self._stop.wait()),
            ]
            try:
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for task in tasks + list(self._prompt_tasks):
                    task.cancel()
                await asyncio.gather(*tasks, *self._prompt_tasks, return_exceptions=True)
            if self._stop.is_set():
                _log.info("live: closing the link for shutdown")
                return
            for task in done:
                exc = task.exception()
                if exc is not None:
                    raise exc
            raise ConnectionError("live: the relay closed the link")
        finally:
            await _close_link(link)

    async def _hello(self, link: Any) -> None:
        """Sends `hello`, checks `hello_ok`, restores the relay's subscriptions and makes the first `status` due."""
        with self._lock:
            has_arms = self._latest is not None and self._latest.arms is not None
        await link.send(json.dumps(self._hello_message(has_arms)))
        answer = _relay_message(await asyncio.wait_for(link.recv(), _HELLO_TIMEOUT_S))
        if answer["type"] == "bye":
            self._bye_code = answer["code"]
            await _close_link(link, answer["code"], answer["reason"])
            raise ConnectionError(f"live: the relay refused the link: {answer['code']} {answer['reason']}")
        if answer["type"] != "hello_ok" or answer["protocol"] != _PROTOCOL:
            await _close_link(link, _CLOSE_PROTOCOL, "protocol")
            raise ConnectionError(f"live: expected hello_ok for protocol {_PROTOCOL}, got {answer!r}")
        if self._has_linked:
            self.reconnects += 1
        self._has_linked = True
        self._last_fault = None
        self.is_linked = True
        self.node_id = answer["node_id"]
        for camera in answer["subscribed"]:
            self._set_subscribed(camera, True)
        self._last_status = None
        self._is_status_due = True
        _log.info(
            "live: linked to node %s as robot %s; subscribed %s", self.node_id, self._robot_id, answer["subscribed"]
        )

    def _hello_message(self, has_arms: bool) -> dict[str, Any]:
        """The link's first message: which robot this is, which cameras a viewer may watch and how they are encoded.

        @param has_arms Whether stream() carries the arms, so the page may ask for their pose.
        """
        return {
            "type": "hello",
            "protocol": _PROTOCOL,
            "robot_id": self._robot_id,
            "cameras": list(self._cameras),
            "width_px": _WIDTH_PX,
            "fps": _FPS,
            "codec": _CODECS[_ENCODER],
            "face_blur": self._face_blur,
            "agent_version": f"dream-client/{dream.__version__}"[:_FIELD_MAX_CHARS],
            "timezone": _local_timezone(),
            "pose": has_arms,
        }

    async def _recv_loop(self, link: Any) -> None:
        """Applies the relay's messages until the link closes; a malformed one closes it with #_CLOSE_PROTOCOL."""
        async for data in link:
            if isinstance(data, bytes):
                continue
            try:
                message = _relay_message(data)
            except ValueError as exc:
                _log.warning("live: malformed message from the relay (%s); closing with %d", exc, _CLOSE_PROTOCOL)
                await _close_link(link, _CLOSE_PROTOCOL, "protocol")
                return
            kind = message["type"]
            if kind == "subscribe":
                self._set_subscribed(message["camera"], True)
            elif kind == "unsubscribe":
                self._set_subscribed(message["camera"], False)
            elif kind == "prompt":
                task = asyncio.ensure_future(self._answer_prompt(link, message))
                self._prompt_tasks.add(task)
                task.add_done_callback(self._prompt_tasks.discard)
            elif kind == "bye":
                self._bye_code = message["code"]
                _log.warning("live: relay says bye %d %s", message["code"], message["reason"])
            else:
                _log.warning("live: unexpected %s from the relay; ignored", kind)

    async def _send_loop(self, link: Any) -> None:
        """Drains pending units, due statuses and the pose into the socket; blocks only in `send()`."""
        assert self._wake is not None
        while True:
            await self._wake.wait()
            self._wake.clear()
            if self._is_status_due:
                self._is_status_due = False
                await link.send(json.dumps(self._status_message()))
            is_sent = True
            while is_sent:
                is_sent = False
                for stream in self._streams.values():
                    unit = stream.pending
                    if unit is None:
                        continue
                    # The slot is freed before the send: a frame arriving while this one drains takes the slot,
                    # and only the frame after that finds it occupied and counts a drop.
                    stream.pending = None
                    await link.send(unit)
                    is_sent = True
            # The newest pose only: a page drawing the body wants where the arms are, never where they were.
            pose, self._pending_pose = self._pending_pose, None
            if pose is not None:
                await link.send(pose)

    async def _answer_prompt(self, link: Any, prompt: dict[str, Any]) -> None:
        """Runs the callbacks for a viewer's press and answers `prompt_ack` within #_PROMPT_ACK_S.

        A callback's awaitable for a new task (`prompt`) that times out is cancelled; every other is shielded,
        because cancelling a run start, a stop, a home move or a driver reopen halfway is worse than a late answer:
        neither the deadline nor the link dropping (which cancels this task with the link's others) reaches the
        work, the timeout ack says `timeout`, and the robot finishes on its own, the next `status` saying where.

        @param link The open link.
        @param prompt The validated `prompt` message.
        """
        action = prompt["action"]
        calls = self._take_prompt(prompt)
        if action in ("home", "reset") and not calls:
            await self._send_ack(link, prompt, self._state(), f"not supported: this robot takes no {action} commands")
            return
        results: list[Any] = []
        waits: list[asyncio.Future[Any]] = []
        for callback, event in calls:
            try:
                result = callback(event)
            except Exception as exc:  # The page hears of it; the link and the other callbacks go on.
                _log.exception("live: a callback failed on %s", event.value)
                results.append(exc)
                continue
            if isinstance(result, concurrent.futures.Future):
                # Cancelling the wrapped future cancels the concurrent one and, through it, the caller's task.
                result = asyncio.wrap_future(result)
            if not inspect.isawaitable(result):
                results.append(result)
                continue
            waited = asyncio.ensure_future(result)
            waits.append(waited if action == "prompt" else asyncio.shield(waited))
        error = None
        if waits:
            try:
                results += await asyncio.wait_for(asyncio.gather(*waits, return_exceptions=True), _PROMPT_ACK_S)
            except asyncio.TimeoutError:
                error = "timeout"
        state = None
        for result in results:
            if isinstance(result, BaseException):
                error = error or f"{type(result).__name__}: {result}"
            elif isinstance(result, str):
                state = result
            elif isinstance(result, Mapping):
                state = result.get("state") or state
                error = error or result.get("error")
        await self._send_ack(link, prompt, state or self._state(), error)

    def _take_prompt(self, prompt: dict[str, Any]) -> list[tuple[Callable[[EventChange], object], EventChange]]:
        """Applies a press's text, model and steps, and picks the callbacks that take it, under the lock.

        @param prompt The validated `prompt` message.
        @return Each callback to call with the event it is called with, in the order they were registered: a new
            task is `EventChange.prompt` for a callback registered for it and `EventChange.end` for the others.
        """
        action = prompt["action"]
        with self._lock:
            if prompt.get("text"):
                self._prompt = prompt["text"]
            elif action == "start" and self._latest is not None and self._latest.prompt is not None:
                self._prompt = self._latest.prompt or None
            if action == "start":
                self._steps = prompt.get("steps")
                if not self._model_fixed:
                    self._model = (prompt.get("checkpoint"), prompt.get("action_space"))
            registered = list(self._callbacks)
        calls = []
        for callback, events in registered:
            event = _PRESSES[action]
            if action == "prompt" and EventChange.prompt not in events:
                event = EventChange.end
            if event in events:
                calls.append((callback, event))
        return calls

    async def _send_ack(self, link: Any, prompt: dict[str, Any], state: str, error: str | None) -> None:
        """Sends one `prompt_ack`; a link that closed meanwhile is logged, not raised."""
        import websockets

        outcome = "applied" if error is None else f"refused: {error}"
        _log.info("live: %s %s %s", prompt["action"], prompt["id"], outcome)
        message: dict[str, Any] = {"type": "prompt_ack", "id": prompt["id"], "ok": error is None, "state": state}
        if error is not None:
            message["error"] = error
        try:
            await link.send(json.dumps(message))
        except websockets.exceptions.ConnectionClosed:
            _log.warning("live: the link closed before prompt %s could be acknowledged", prompt["id"])

    def _state(self) -> str:
        """The status the caller last streamed."""
        with self._lock:
            return "idle" if self._latest is None else self._latest.status

    # The poll: status changes, the pose, the tick guard and the frames

    async def _poll_loop(self) -> None:
        """Reads the newest observation #_POLL_HZ times a second; the only place frames are handed to the encoders."""
        while True:
            self._poll_once(time.monotonic())
            await asyncio.sleep(1.0 / _POLL_HZ)

    def _poll_once(self, now_s: float) -> None:
        """One poll: note a status change, send the pose, apply the guard, encode the watched cameras.

        @param now_s Monotonic time of the poll.
        """
        assert self._wake is not None
        with self._lock:
            snapshot = self._latest
        if snapshot is None:
            return
        is_running = snapshot.status == "running"
        with self._lock:
            if is_running:
                self._watch.observe(now_s)
            else:
                self._watch.reset()
        self._close_idle_encoders(now_s)
        if self._paused_until_s is not None and now_s >= self._paused_until_s:
            self._paused_until_s = None
            self.live_paused = None
            _log.info("live: resuming after the tick guard's backoff")
        status_key = self._status_key(snapshot)
        if status_key != self._last_status:
            self._last_status = status_key
            self._is_status_due = True
            self._wake.set()
        with self._lock:
            count = self._timing_count
        if not is_running:
            self._sent_timing_count = 0
        elif count != self._sent_timing_count and now_s >= self._next_timing_s:
            # A round trip lands every second or so: the status it changes goes out at most every _TIMING_STATUS_S.
            self._sent_timing_count = count
            self._next_timing_s = now_s + _TIMING_STATUS_S
            self._is_status_due = True
            self._wake.set()
        if self.is_pose_subscribed and now_s >= self._next_pose_s:
            self._next_pose_s = now_s + 1.0 / _POSE_HZ
            pose = _pose_message(snapshot.arms)
            if pose is not None:
                self._pending_pose = json.dumps(pose)
                self._wake.set()
        active = [stream for stream in self._streams.values() if stream.is_subscribed]
        if not self._is_encoding(active, is_running, now_s):
            return
        for stream in active:
            if stream.retry_at_s is not None:
                if now_s < stream.retry_at_s:
                    continue
                stream.retry_at_s = None
            frame = snapshot.frames.get(stream.camera)
            captured_s = snapshot.captured_s.get(stream.camera)
            if frame is None or captured_s is None or captured_s == stream.last_ts_s:
                continue
            stream.last_ts_s = captured_s
            if self._is_due(stream, captured_s):
                self._submit(stream, frame, captured_s, now_s)
        if any(stream.pending is not None for stream in active):
            self._wake.set()

    def _is_encoding(self, active: list[_Stream], is_running: bool, now_s: float) -> bool:
        """Whether this poll encodes: a camera wants frames, the guard is not paused, and while running the client's
        own thread keeps its rate; yields when it does not.

        @param active The watched cameras.
        @param is_running Whether the status is `running`.
        @param now_s Monotonic time of the poll.
        @return True to encode the cameras' new frames.
        """
        with self._lock:
            if not active or self._paused_until_s is not None:
                self._watch.stop_encoding()
                return False
            if not is_running:
                # The guard only has a run to protect while one is on; idle frames are the cameras alone.
                return True
            if self._watch.encoding_since_s is None:
                self._watch.encoding_since_s = now_s
            if not self._watch.is_starved(now_s):
                return True
        self._yield_to_tick(now_s)
        return False

    def _status_key(self, snapshot: _Snapshot) -> tuple[Any, ...]:
        """What the status says, as a value: a change of it sends a `status` at once."""
        with self._lock:
            prompt = self._prompt
        return (
            snapshot.status,
            snapshot.prompt if snapshot.prompt is not None else prompt,
            snapshot.run,
            self.live_paused,
            snapshot.error,
            self._session_of(snapshot),
            self._checkpoint_of(snapshot),
            None if snapshot.model is None else snapshot.model[1],
            snapshot.steps,
            None
            if snapshot.arms is None
            else tuple(sorted((name, arm is not None) for name, arm in snapshot.arms.items())),
            self._runs,
        )

    def _checkpoint_of(self, snapshot: _Snapshot) -> str | None:
        """The checkpoint the page shows: the caller's `model`, else the `dream_client`'s."""
        if snapshot.model is not None:
            return snapshot.model[0]
        return None if self.dream_client is None else self.dream_client.checkpoint

    def _status_message(self) -> dict[str, Any]:
        """The `status` of the moment, keys in the contract's order; a None field is left out, never sent as null.

        A relay older than a field forbids the key like any unknown one and closes the link `4002`, and every
        optional field of `status` defaults to None there, so nothing is lost by leaving it out.
        """
        with self._lock:
            snapshot = self._latest
            prompt = self._prompt
            timing = self._timing_report(snapshot)
            runs = self._runs
        assert snapshot is not None
        error, code = (None, None) if snapshot.error is None else snapshot.error
        action_space = None if snapshot.model is None else snapshot.model[1]
        arms = {} if snapshot.arms is None else {name: arm is not None for name, arm in snapshot.arms.items()}
        fields: dict[str, Any] = {
            "type": "status",
            "state": snapshot.status,
            "prompt": snapshot.prompt if snapshot.prompt is not None else prompt,
            "run": (runs if snapshot.run is None else snapshot.run) or None,
            "dropped_frames": {stream.camera: stream.dropped for stream in self._streams.values()},
            "live_paused": self.live_paused,
            "error": error,
            "session_id": self._session_of(snapshot),
            "error_code": code,
            "checkpoint": self._checkpoint_of(snapshot),
            "arms": arms,
            "timing": timing,
            "action_space": action_space if action_space in _ACTION_SPACES else None,
            "steps": snapshot.steps,
            "model_fixed": self._model_fixed or None,
        }
        return {key: value for key, value in fields.items() if value is not None}

    def _timing_report(self, snapshot: _Snapshot | None) -> dict[str, Any] | None:
        """The status's `timing`, under the lock: the newest round trip and the run's spread; None before an answer.

        The round trips listed in `recent_ms` start over with each report.
        """
        reading = self._timing
        if snapshot is None or snapshot.status != "running" or reading is None or reading.get("rtt_ms") is None:
            return None
        report: dict[str, Any] = {"rtt_ms": _clamp_ms(reading["rtt_ms"])}
        for name in ("p50_ms", "p95_ms"):
            if reading.get(name) is not None:
                report[name] = _clamp_ms(reading[name])
        if reading.get("network_ms") is not None:
            report["network_ms"] = float(reading["network_ms"])
        for name in ("gateway_ms", "serve_infer_ms"):
            if reading.get(name) is not None:
                report[name] = _clamp_ms(reading[name])
        report["count"] = self._timing_count
        report["recent_ms"] = self._recent_ms
        if reading.get("up_mbps") is not None and reading["up_mbps"] >= 0:
            report["up_mbps"] = float(reading["up_mbps"])
        self._recent_ms = []
        return report

    def _is_due(self, stream: _Stream, captured_s: float) -> bool:
        """Whether encoding this frame keeps the camera at or under `fps` units per second.

        The due time advances one period per encoded unit, catching up to the frame when the camera is slower than
        `fps`; a frame up to half a period early is jitter and is encoded, so a camera at exactly `fps` never skips,
        while a faster one is thinned to the configured rate.

        @param stream The camera's stream, whose `next_due_s` this reads and advances.
        @param captured_s The frame's capture time.
        @return True to encode the frame; False to skip it (not a drop: the viewer never expected it).
        """
        period_s = 1.0 / _FPS
        due_s = stream.next_due_s
        if due_s is None:
            stream.next_due_s = captured_s + period_s
            return True
        if captured_s <= due_s - period_s / 2 + _DUE_TOLERANCE_S:
            return False
        stream.next_due_s = max(due_s + period_s, captured_s + period_s / 2)
        return True

    def _submit(self, stream: _Stream, frame: Any, captured_s: float, now_s: float) -> None:
        """Hands one frame to the camera's worker thread; a frame that finds the worker busy waits for it, one that
        finds a frame already waiting replaces it, and one that finds the slot occupied replaces the unsent unit;
        either replacement counts a drop and makes the next unit a keyframe.

        The poll never encodes: the decode, the resize, the blur and the encoder run on the worker, one per camera,
        so three cameras cost wall time in parallel and a slow encode thins that camera alone. Runs on the link's
        loop thread, where every field of `stream` but the encoder and the blurrer is owned.

        @param stream The camera's stream.
        @param frame The caller's frame, never written.
        @param captured_s The frame's capture time.
        @param now_s Monotonic time of the poll, the base of the retry delay after a fault.
        """
        if stream.is_busy:
            # The worker is still on the previous frame. The blur takes most of a frame period and the poll picks a
            # frame up as much as a poll late, so two frames can reach the worker 17 ms apart while it takes 20: the
            # second waits rather than being lost (otherwise a frame in ten is, on a YAM station). A frame
            # already waiting is replaced: it is never encoded, consumes a `seq` all the same so the viewer's gap
            # count is the drop count, and the next unit is a keyframe.
            if stream.waiting is not None:
                stream.seq = (stream.seq + 1) & 0xFFFFFFFF
                stream.dropped += 1
                stream.force_idr = True
            stream.waiting = (frame, captured_s, now_s)
            return
        if stream.pending is not None:
            # The sender has not drained the previous unit: it goes (its `seq` is the gap) and this fresher frame
            # takes its place as a keyframe, so a stalled uplink shows the newest picture when it resumes.
            stream.pending = None
            stream.dropped += 1
            stream.force_idr = True
        self._start_encode(stream, frame, captured_s, now_s)

    def _start_encode(self, stream: _Stream, frame: Any, captured_s: float, now_s: float) -> None:
        """Hands one frame to the camera's worker, as a keyframe when one is owed; _encoded() takes the unit."""
        assert self._loop is not None and self._workers is not None
        stream.is_busy = True
        force_idr, stream.force_idr = stream.force_idr, False
        future = self._loop.run_in_executor(self._workers, self._encode_unit, stream, frame, force_idr)
        future.add_done_callback(lambda done: self._encoded(stream, done, captured_s, now_s))

    def _encode_unit(self, stream: _Stream, frame: Any, force_idr: bool) -> tuple[bytes, bool] | None:
        """The worker's half: decodes, resizes and blurs the frame, opens the encoder on its first frame, encodes.

        Runs on the camera's worker thread, which owns `stream.encoder` and `stream.blurrer` while `stream.is_busy`
        is set; the loop thread neither reads nor closes them until _encoded() clears the flag.

        @param stream The camera's stream.
        @param frame A base64 JPEG or an RGB array.
        @param force_idr Whether this unit must be a keyframe.
        @return `(unit, is_keyframe)`, or None when the frame is not a JPEG or the encoder buffered it.
        @throws Exception Whatever the encoder or the blur raised; _encoded() counts and logs it.
        """
        bgr = self._prepare(stream, frame)
        if bgr is None:
            return None
        if stream.encoder is None:
            stream.encoder = _open_encoder(_ENCODER, _WIDTH_PX, bgr.shape[0], _FPS, _BITRATE_KBPS)
            force_idr = True
            _log.info("live: encoder opened for %s at %dx%d", stream.camera, _WIDTH_PX, bgr.shape[0])
        return stream.encoder.encode(bgr, force_idr)

    def _prepare(self, stream: _Stream, frame: Any) -> np.ndarray | None:
        """The frame at the encoded size, BGR, faces blurred with the camera's own detector when `face_blur` is on.

        Runs on the camera's worker, which owns `stream.blurrer` while `stream.is_busy` is set.

        @return The frame, or None for a base64 string that is not a JPEG (logged).
        """
        import cv2

        if isinstance(frame, str):
            bgr = cv2.imdecode(np.frombuffer(base64.b64decode(frame), dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                _log.warning("live: camera %s sent a frame that is not a JPEG", stream.camera)
                return None
            height_px = _height_px(bgr.shape[1], bgr.shape[0], _WIDTH_PX)
            if bgr.shape[:2] != (height_px, _WIDTH_PX):
                bgr = cv2.resize(bgr, (_WIDTH_PX, height_px), interpolation=cv2.INTER_AREA)
        else:
            rgb = frame
            height_px = _height_px(rgb.shape[1], rgb.shape[0], _WIDTH_PX)
            if rgb.shape[:2] != (height_px, _WIDTH_PX):
                rgb = cv2.resize(rgb, (_WIDTH_PX, height_px), interpolation=cv2.INTER_AREA)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        bgr = np.asarray(bgr, dtype=np.uint8)
        if not self._face_blur:
            return bgr
        if stream.blurrer is None:
            from shared import faceblur

            stream.blurrer = faceblur.FaceBlurrer()
        return np.asarray(stream.blurrer.scrub_frame(bgr), dtype=np.uint8)

    def _encoded(self, stream: _Stream, done: asyncio.Future[Any], captured_s: float, now_s: float) -> None:
        """The loop thread's half: takes the worker's unit into the pending slot, wakes the sender, and hands the
        worker the frame that waited for it.

        @param stream The camera's stream.
        @param done The worker's future, finished.
        @param captured_s The frame's capture time, the header's.
        @param now_s Monotonic time of the poll that submitted the frame, the base of the retry delay after a fault.
        """
        assert self._wake is not None
        stream.is_busy = False
        if stream.is_close_requested:
            stream.is_close_requested = False
            self._close_stream(stream)
            return
        try:
            encoded = done.result()
        except Exception:
            # The relay repeats no `subscribe` while a viewer stays, so the stream stays subscribed and the encoder
            # is reopened on the first frame after the delay; a broken encoder shows in `errors`, not in a log storm.
            self.errors += 1
            _log.exception(
                "live: encoder of %s failed; reopening on the first frame after %d s", stream.camera, _IDLE_ENCODER_S
            )
            self._close_stream(stream)
            stream.retry_at_s = now_s + _IDLE_ENCODER_S
            return
        is_wanted = stream.is_subscribed
        if encoded is not None and is_wanted and stream.pending is not None:
            # Only a frame that waited for the worker was encoded with a unit in the slot (_submit() empties it for
            # every other), and that unit is still there after a whole encode: the uplink stalled. This unit goes
            # (its `seq` is the gap) rather than the one the viewer's decoder already expects, and the next is a
            # keyframe.
            stream.seq = (stream.seq + 1) & 0xFFFFFFFF
            stream.dropped += 1
            stream.force_idr = True
        elif encoded is not None and is_wanted:
            unit, is_keyframe = encoded
            flags = _FLAG_KEYFRAME if is_keyframe else 0
            header = _HEADER.pack(_PROTOCOL, _KIND_VIDEO, stream.index, flags, stream.seq, self._wall_us(captured_s))
            stream.seq = (stream.seq + 1) & 0xFFFFFFFF
            stream.pending = header + unit
            self._wake.set()
        waiting, stream.waiting = stream.waiting, None
        if waiting is not None and is_wanted:
            self._start_encode(stream, *waiting)

    def _yield_to_tick(self, now_s: float) -> None:
        """Closes every encoder and pauses for #_YIELD_BACKOFF_S: the client's own thread cannot keep its rate."""
        self.yields += 1
        self.live_paused = _PAUSED_TICK
        self._paused_until_s = now_s + _YIELD_BACKOFF_S
        with self._lock:
            worst = ", ".join(f"{interval * 1e3:.0f} ms" for interval in self._watch.worst(now_s - _YIELD_WINDOW_S))
            self._watch.stop_encoding()
        _log.warning(
            "live: the client's thread polled late (p95 past %.0f ms for %d s): the computer has no CPU to spare for "
            "video; pausing it for %d s; longest: %s",
            _YIELD_TICK_RATIO * 1e3 / _POLL_HZ,
            _YIELD_WINDOW_S,
            _YIELD_BACKOFF_S,
            worst or "none",
        )
        self._close_encoders()

    def _set_subscribed(self, camera: str, is_subscribed: bool) -> None:
        """Applies a `subscribe` or `unsubscribe`; an unknown camera is logged and ignored."""
        if camera == _POSE_STREAM:
            if self.is_pose_subscribed != is_subscribed:
                self.is_pose_subscribed = is_subscribed
                self._next_pose_s = 0.0
                _log.info("live: pose %s", "subscribed" if is_subscribed else "unsubscribed")
            return
        stream = self._streams.get(camera)
        if stream is None:
            _log.warning("live: relay named camera %r, which this robot does not have; ignored", camera)
            return
        if stream.is_subscribed == is_subscribed:
            return
        stream.is_subscribed = is_subscribed
        stream.idle_since_s = None if is_subscribed else time.monotonic()
        if is_subscribed:
            stream.last_ts_s = None
            stream.next_due_s = None
            stream.retry_at_s = None
            stream.force_idr = True
        _log.info("live: %s %s", camera, "subscribed" if is_subscribed else "unsubscribed")

    def _close_idle_encoders(self, now_s: float) -> None:
        """Closes the encoder of every camera unwatched for #_IDLE_ENCODER_S."""
        for stream in self._streams.values():
            idle_since_s = stream.idle_since_s
            if stream.encoder is not None and idle_since_s is not None and now_s - idle_since_s >= _IDLE_ENCODER_S:
                self._close_stream(stream)
                _log.info("live: encoder of %s closed after %d s idle", stream.camera, _IDLE_ENCODER_S)

    def _close_encoders(self) -> None:
        """Closes every open encoder."""
        for stream in self._streams.values():
            self._close_stream(stream)

    def _close_stream(self, stream: _Stream) -> None:
        """Closes one stream's encoder, if open, and forgets its pending unit and the frame waiting for its worker;
        deferred while its worker is busy."""
        stream.pending = None
        stream.waiting = None
        stream.next_due_s = None
        if stream.is_busy:
            stream.is_close_requested = True
            return
        if stream.encoder is not None:
            stream.encoder.close()
            stream.encoder = None


@dataclasses.dataclass(frozen=True)
class _Snapshot:
    """What the last stream() was given, read by the poll and the sender."""

    status: str
    ## Frame per page camera name: a base64 JPEG or an RGB array.
    frames: dict[str, Any]
    ## Monotonic capture time per page camera name.
    captured_s: dict[str, float]
    ## `(q, gripper)` per arm, None for an arm that cannot be read; None without `arms`.
    arms: dict[str, tuple[list[float], float] | None] | None
    ## `(message, code)`, or None.
    error: tuple[str, str | None] | None
    prompt: str | None
    run: int | None
    model: tuple[str | None, str | None] | None
    steps: int | None


@dataclasses.dataclass
class _Stream:
    """One camera's encoder, counters and the one-slot uplink pipe."""

    camera: str
    ## Index into the `hello` cameras, the header's `camera` byte.
    index: int
    encoder: _Encoder | None = None
    seq: int = 0
    last_ts_s: float | None = None
    dropped: int = 0
    force_idr: bool = False
    ## The one unsent unit; a new frame replaces it and counts a drop, and the replacement is a keyframe.
    pending: bytes | None = None
    ## True while this camera's worker thread is decoding, resizing, blurring and encoding a frame; a frame due
    ## meanwhile waits in `waiting`, so a camera whose encode is slower than `fps` thins itself, never queues.
    is_busy: bool = False
    ## The newest frame that found the worker busy, as `(frame, captured_s, now_s)`: encoded as soon as the worker
    ## returns, and replaced (counting a drop) by a newer one that finds it here. One frame deep, never a queue.
    waiting: tuple[Any, float, float] | None = None
    ## Set by _close_stream() while the worker is busy; the encoder is closed when the worker returns.
    is_close_requested: bool = False
    ## This camera's own `shared.faceblur.FaceBlurrer`, built by its worker on the first frame it blurs and owned by
    ## that worker as `encoder` is. The detector is not reentrant and takes its calls under a lock, so one the
    ## cameras shared would run their detections one after another: 11 to 37 ms a frame on a YAM station, most of
    ## a 33 ms period, and the cameras waiting on it would drop a frame in ten.
    blurrer: Any = None
    is_subscribed: bool = False
    ## Monotonic time of the last unsubscribe; the encoder closes #_IDLE_ENCODER_S later.
    idle_since_s: float | None = None
    ## Capture time at which the next unit is due under `fps`; None encodes the next frame at once.
    next_due_s: float | None = None
    ## Monotonic time before which no frame is encoded, after an encoder fault; None when the encoder is trusted.
    retry_at_s: float | None = None


@dataclasses.dataclass
class _PollWatch:
    """The spacing of the client's own polls while running, which the tick guard reads; used under the lock."""

    ## `(poll_s, interval_s)` of the last polls, newest last.
    intervals: collections.deque[tuple[float, float]] = dataclasses.field(
        default_factory=lambda: collections.deque[tuple[float, float]](maxlen=1200)
    )
    last_poll_s: float | None = None
    ## Poll time at which encoding started, or None while nothing is encoded.
    encoding_since_s: float | None = None

    def observe(self, poll_s: float) -> None:
        """Records the interval since the last poll.

        @param poll_s Monotonic time of this poll.
        """
        if self.last_poll_s is not None and poll_s > self.last_poll_s:
            self.intervals.append((poll_s, poll_s - self.last_poll_s))
        self.last_poll_s = poll_s

    def worst(self, since_s: float, count: int = 3) -> list[float]:
        """The longest intervals ending at or after `since_s`, for the log.

        @param since_s The monotonic time the window starts at.
        @param count How many to return.
        @return Intervals in seconds, longest first.
        """
        return sorted((interval for poll_s, interval in self.intervals if poll_s >= since_s), reverse=True)[:count]

    def reset(self) -> None:
        """Forgets every poll: a run has ended, and its gap to the next must never count as an interval."""
        self.intervals.clear()
        self.last_poll_s = None
        self.stop_encoding()

    def stop_encoding(self) -> None:
        """Forgets the encoding window; the next one starts on the next encoded poll."""
        self.encoding_since_s = None

    def is_starved(self, now_s: float) -> bool:
        """Whether the client's thread fell behind while encoding.

        @param now_s Monotonic time of this poll.
        @return True when encoding has run #_YIELD_WINDOW_S and the p95 poll interval of that window is past
            #_YIELD_TICK_RATIO poll periods, with at least #_MIN_TICK_SAMPLES intervals in it.
        """
        if self.encoding_since_s is None or now_s - self.encoding_since_s < _YIELD_WINDOW_S:
            return False
        since_s = max(self.encoding_since_s, now_s - _YIELD_WINDOW_S)
        recent = sorted(interval for poll_s, interval in self.intervals if poll_s >= since_s)
        if len(recent) < _MIN_TICK_SAMPLES:
            return False
        return recent[min(len(recent) - 1, int(0.95 * len(recent)))] > _YIELD_TICK_RATIO / _POLL_HZ


class _Encoder:
    """One camera's encoder, PyAV's `libvpx` (VP8, or VP9 with `libvpx-vp9`) with the low-latency options.

    `deadline=realtime` with `lag-in-frames=0` disables lookahead and alternate reference frames, so every frame
    yields exactly one encoded frame in the same call; `g` and `keyint_min` keep the GOP exactly `fps x _GOP_S`; a
    forced `I` picture is a keyframe, which carries everything a decoder needs to start.
    """

    def __init__(self, encoder: str, width_px: int, height_px: int, fps: int, bitrate_kbps: int) -> None:
        """Opens the codec.

        @param encoder One of #_CODECS.
        @param width_px The encoded width, even.
        @param height_px The encoded height, even.
        @param fps The frame rate.
        @param bitrate_kbps The average bitrate.
        @throws DreamConfigError PyAV is not installed, or the encoder is not in its build.
        """
        try:
            import av
        except ImportError as exc:
            raise DreamConfigError(_MISSING_PACKAGES) from exc
        self._av = av
        try:
            codec: Any = av.CodecContext.create(encoder, "w")
        except Exception as exc:
            raise DreamConfigError(f"encoder {encoder!r} is not in this PyAV build: {exc}") from exc
        gop = round(fps * _GOP_S)
        codec.width = width_px
        codec.height = height_px
        codec.pix_fmt = "yuv420p"
        codec.framerate = fractions.Fraction(fps, 1)
        codec.time_base = fractions.Fraction(1, fps)
        codec.bit_rate = bitrate_kbps * 1000
        codec.thread_count = _ENCODER_THREADS
        codec.options = {
            "deadline": "realtime",
            "cpu-used": str(_ENCODER_SPEED),
            "lag-in-frames": "0",
            "auto-alt-ref": "0",
            "error-resilient": "1",
            "g": str(gop),
            "keyint_min": str(gop),
            "maxrate": f"{bitrate_kbps}k",
            "bufsize": f"{_VBV_BUFFER_KBIT}k",
            "threads": str(_ENCODER_THREADS),
        }
        codec.open()
        self._codec: Any = codec
        self._pts = 0

    def encode(self, bgr: np.ndarray, force_idr: bool) -> tuple[bytes, bool] | None:
        """Encodes one frame.

        @param bgr The frame, uint8 `[height_px, width_px, 3]`.
        @param force_idr Make it a keyframe, for a viewer that just started watching or after a drop.
        @return The encoded frame and whether it is a keyframe, or None when the encoder produced nothing (it
            never does with `lag-in-frames=0`).
        """
        frame = self._av.VideoFrame.from_ndarray(bgr, format="bgr24")
        frame.pts = self._pts
        self._pts += 1
        if force_idr:
            frame.pict_type = self._av.video.frame.PictureType.I
        packets = self._codec.encode(frame)
        if not packets:
            return None
        return b"".join(bytes(packet) for packet in packets), any(packet.is_keyframe for packet in packets)

    def close(self) -> None:
        """Releases the codec; PyAV frees it with the reference."""
        self._codec = None


def _open_encoder(encoder: str, width_px: int, height_px: int, fps: int, bitrate_kbps: int) -> _Encoder:
    """Opens one camera's encoder; the one place the client opens one, so a test can stand in for it."""
    return _Encoder(encoder, width_px, height_px, fps, bitrate_kbps)


def _robot_link(platform_url: str) -> str:
    """`wss://<host>/live/robot` for an `https://` platform; `ws://` only to this machine.

    @param platform_url The platform's URL.
    @return The robot link's URL.
    @throws DreamConfigError The platform is `http://` on another host, or not `http(s)://` at all.
    """
    parts = urllib.parse.urlsplit(platform_url)
    if parts.scheme == "https":
        return f"wss://{parts.netloc}{_ROBOT_LINK_PATH}"
    if parts.scheme == "http" and parts.hostname in _LOOPBACK_HOSTS:
        return f"ws://{parts.netloc}{_ROBOT_LINK_PATH}"
    raise DreamConfigError(
        f"Live View needs an https:// platform, got {platform_url!r}: the key must not cross a network in the clear"
    )


def _camera_name(key: str, is_renamed: bool) -> str | None:
    """The page's name for a frame key's camera, or None for a key that is not a frame.

    @param key An observation key.
    @param is_renamed Whether the model's roles take the page's names (`video.wrist_l` is `left`).
    """
    for prefix in _FRAME_PREFIXES:
        if key.startswith(prefix) and len(key) > len(prefix):
            role = key[len(prefix) :]
            return _PAGE_CAMERAS.get(role, role) if is_renamed else role
    return None


def _checked_cameras(cameras: Sequence[str]) -> tuple[str, ...]:
    """The constructor's camera names, checked as the platform checks `hello`.

    @throws DreamConfigError The list is empty, a name is empty, repeated, too long, or `pose`.
    """
    names = tuple(cameras)
    is_valid = bool(names) and len(set(names)) == len(names)
    for name in names:
        is_valid = is_valid and isinstance(name, str) and 0 < len(name) <= _FIELD_MAX_CHARS and name != _POSE_STREAM
    if not is_valid:
        raise DreamConfigError(
            f"cameras must be distinct names of 1 to {_FIELD_MAX_CHARS} characters, none of them {_POSE_STREAM!r}; "
            f"got {list(names)}"
        )
    return names


def _checked_arms(arms: Mapping[str, Any]) -> dict[str, tuple[list[float], float] | None]:
    """stream()'s `arms` as plain floats, read now so a later change to the caller's arrays changes nothing sent.

    @throws DreamObservationError An arm is not `(q, gripper)` or None.
    """
    checked: dict[str, tuple[list[float], float] | None] = {}
    for name, arm in arms.items():
        if arm is None:
            checked[name] = None
            continue
        try:
            q, gripper = arm
            checked[name] = ([float(value) for value in q], float(gripper))
        except (TypeError, ValueError) as exc:
            raise DreamObservationError(f"arms[{name!r}] must be (q, gripper) or None, got {arm!r}") from exc
    return checked


def _checked_error(error: str | tuple[str, str | None] | None) -> tuple[str, str | None] | None:
    """stream()'s `error` as `(message, code)`, each cut to the contract's bound; None for no error."""
    message, code = (error, None) if isinstance(error, str) or error is None else error
    if not message:
        return None
    return str(message)[:_STATUS_ERROR_MAX_CHARS], str(code)[:_FIELD_MAX_CHARS] if code else None


def _pose_message(arms: dict[str, tuple[list[float], float] | None] | None) -> dict[str, Any] | None:
    """Where the arms are now, for the pose stream: each readable arm's joints and its gripper clipped to [0, 1].

    @param arms stream()'s checked `arms`, or None when the call carried none.
    @return The message, or None when no arm can be read.
    """
    readable = {} if arms is None else {name: arm for name, arm in arms.items() if arm is not None}
    if not readable:
        return None
    return {
        "type": "pose",
        "t_us": int(time.time() * 1_000_000),
        "arms": {name: {"q": q, "gripper": min(max(gripper, 0.0), 1.0)} for name, (q, gripper) in readable.items()},
    }


def _relay_message(data: str | bytes) -> dict[str, Any]:
    """Reads one text frame from the relay as the contract defines it (`parse_message` over `ROBOT_LINK_TYPES`).

    Only the messages a relay sends a robot are read field by field (`hello_ok`, `subscribe`, `unsubscribe`,
    `prompt`, `bye`); the robot link's other types come back with their type alone, for the caller to ignore.

    @return The message, with a `prompt`'s `action` filled in.
    @throws ValueError The frame is not a JSON object, its type is unknown, or a field is missing, unknown or of
        the wrong shape: the link closes with #_CLOSE_PROTOCOL.
    """
    body = json.loads(data)
    if not isinstance(body, dict):
        raise ValueError(f"a message is a JSON object, got {type(body).__name__}")
    kind = body.get("type")
    fields = _RELAY_FIELDS.get(kind) if isinstance(kind, str) else None
    if kind in _ROBOT_SENT_TYPES:
        return {"type": kind}
    if fields is None:
        raise ValueError(f"unknown message type {kind!r}")
    unknown = set(body) - fields - {"type"}
    if unknown:
        raise ValueError(f"{kind} has unknown fields {sorted(unknown)}")
    if kind == "hello_ok":
        _require(body, "node_id", str, 1)
        _require(body, "protocol", int)
        if not isinstance(body.get("subscribed"), list) or not all(isinstance(c, str) for c in body["subscribed"]):
            raise ValueError("hello_ok.subscribed must be a list of camera names")
    elif kind in ("subscribe", "unsubscribe"):
        _require(body, "camera", str, 1)
    elif kind == "bye":
        _require(body, "code", int)
        _require(body, "reason", str)
    else:
        _check_prompt(body)
    return body


def _require(body: dict[str, Any], name: str, kind: type, min_length: int = 0) -> None:
    """Fails unless `body[name]` is a `kind` (never a bool for an int) at least `min_length` long when a str."""
    value = body.get(name)
    if not isinstance(value, kind) or isinstance(value, bool) or (isinstance(value, str) and len(value) < min_length):
        raise ValueError(f"{body.get('type')}.{name} must be a {kind.__name__}, got {value!r}")


def _check_prompt(body: dict[str, Any]) -> None:
    """Holds a `prompt` to the contract's bounds and to what each action may carry; fills in `action`.

    @throws ValueError A field is out of bounds, or names what its action may not.
    """
    _require(body, "id", str, 1)
    action = body.setdefault("action", "prompt")
    if action not in _PROMPT_ACTIONS:
        raise ValueError(f"prompt.action must be one of {list(_PROMPT_ACTIONS)}, got {action!r}")
    text, checkpoint, steps = body.get("text"), body.get("checkpoint"), body.get("steps")
    if text is not None and (not isinstance(text, str) or not 1 <= len(text) <= _PROMPT_MAX_CHARS):
        raise ValueError(f"prompt.text must be 1 to {_PROMPT_MAX_CHARS} characters")
    if checkpoint is not None and (
        not isinstance(checkpoint, str) or not 1 <= len(checkpoint) <= _CHECKPOINT_MAX_CHARS
    ):
        raise ValueError(f"prompt.checkpoint must be 1 to {_CHECKPOINT_MAX_CHARS} characters")
    if body.get("action_space") not in (None, *_ACTION_SPACES):
        raise ValueError(f"prompt.action_space must be one of {list(_ACTION_SPACES)}")
    if steps is not None and (
        isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= _RUN_STEPS_MAX
    ):
        raise ValueError(f"prompt.steps must be a whole number from 1 to {_RUN_STEPS_MAX}")
    if action == "prompt" and text is None:
        raise ValueError("a prompt needs a text")
    if action in ("stop", "home", "reset") and text is not None:
        raise ValueError(f"a {action} carries no text")
    if action != "start" and (checkpoint, body.get("action_space"), steps) != (None, None, None):
        raise ValueError("only a start names a checkpoint, an action space or a step count")


def _check_installed(face_blur: bool) -> None:
    """Fails on the caller's thread, not the link's, when a package the client needs is missing.

    Nothing is imported here: importing OpenCV, or the face detector, starts worker threads, which must be born on the
    client's niced thread rather than on the caller's at the caller's priority. A loop that imports OpenCV itself
    before building the client makes that pool first, at its own priority.

    @param face_blur Whether the face detector's module must be there too.
    @throws DreamConfigError PyAV, OpenCV or websockets is missing, or `face_blur` is on and `shared.faceblur` is.
    """
    # Found, not imported: importing OpenCV starts its worker pool, which must be born on the client's niced thread
    # (_warm_opencv_pool()), not here at the caller's priority.
    for name in ("av", "cv2", "websockets"):
        try:
            is_there = importlib.util.find_spec(name) is not None
        except ValueError:  # a module already imported without a spec, as a test's stand-in is
            is_there = sys.modules.get(name) is not None
        if not is_there or (name in sys.modules and sys.modules[name] is None):
            raise DreamConfigError(_MISSING_PACKAGES)
    if not face_blur:
        return
    # Found, not imported: the detector's runtime starts its threads on import, and they must be born on the
    # client's niced thread, as OpenCV's pool is, not here at the caller's priority.
    try:
        is_there = importlib.util.find_spec("shared.faceblur") is not None
    except ModuleNotFoundError:
        is_there = False
    if not is_there:
        raise DreamConfigError(
            "face_blur needs the face detector (shared.faceblur), which the dream library carries: install it with "
            "pip install --upgrade dreamlabs, or pass face_blur=False to send the frames as captured"
        )


def _warm_opencv_pool() -> None:
    """Caps and creates OpenCV's process-wide worker pool from the calling (niced) thread.

    OpenCV spawns its `parallel_for_` pool on the first parallel call in the process and keeps it; a Linux thread
    inherits its creator's niceness. The client's resize and blur fan out onto that pool, so the pool is born here,
    after `os.nice(_NICE)`, when nothing in the process has made it yet. It is capped at #_OPENCV_THREADS first,
    because its idle workers spin; `OPENCV_FOR_THREADS_NUM` in the environment overrides the cap.
    """
    import cv2

    if not os.environ.get("OPENCV_FOR_THREADS_NUM"):
        cv2.setNumThreads(_OPENCV_THREADS)
    cv2.resize(
        np.zeros(_POOL_WARM_SHAPE, np.uint8),
        (_POOL_WARM_SHAPE[1] // 2, _POOL_WARM_SHAPE[0] // 2),
        interpolation=cv2.INTER_AREA,
    )


async def _close_link(link: Any, code: int = 1000, reason: str = "") -> None:
    """Closes the link, aborting its socket when the closing handshake has not finished in #_LINK_CLOSE_S.

    @param link The websockets connection.
    @param code The close code.
    @param reason The close reason.
    """
    try:
        await asyncio.wait_for(link.close(code, reason), _LINK_CLOSE_S)
    except asyncio.TimeoutError:
        _log.warning("live: the link did not close in %.0f s (a dead network path); aborting it", _LINK_CLOSE_S)
        link.transport.abort()


def _bound_uplink(link: Any) -> None:
    """Bounds what the kernel queues on the link, so a stalled `send()` means a shallow pipe.

    `SO_SNDBUF` = #_SNDBUF_BYTES caps the bytes sent but unacknowledged plus unsent; `TCP_NOTSENT_LOWAT` =
    #_NOTSENT_LOWAT makes the socket unwritable once that much is unsent, as the relay does for its viewers. A
    transport without a socket, or a platform without the option, is left as is.

    @param link The open connection.
    """
    raw = link.transport.get_extra_info("socket")
    if raw is None:
        return
    raw.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, _SNDBUF_BYTES)
    if _TCP_NOTSENT_LOWAT is not None:
        raw.setsockopt(socket.IPPROTO_TCP, _TCP_NOTSENT_LOWAT, _NOTSENT_LOWAT)


def _height_px(width: int, height: int, width_px: int) -> int:
    """The even encoded height that keeps a `width` x `height` capture's aspect at `width_px` wide."""
    height_px = round(height * width_px / width)
    return max(2, height_px + height_px % 2)


def _local_timezone() -> str:
    """The robot's local time zone for `hello`: the IANA name when the computer says it, else its fixed UTC offset.

    `TZ` is read first, then the `/etc/localtime` link (`.../zoneinfo/America/Los_Angeles`); a computer with
    neither yields the offset the clock reports now, `UTC-07:00`, which is right until the next daylight change.
    The platform stores it beside every frame and tick, so a reader can turn the UTC stamps into local time.

    @return The zone name, at most #_FIELD_MAX_CHARS characters.
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
    return name[:_FIELD_MAX_CHARS]


def _clamp_ms(value: float) -> float:
    """Milliseconds inside the contract's bounds, 0 to #_TIMING_MAX_MS."""
    return min(max(float(value), 0.0), _TIMING_MAX_MS)
