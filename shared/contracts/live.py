"""The live-view wire: the video header, both links' messages, the close codes, the viewer token and the constants
both ends read.

Two consumers: the `live` relay node and the robot's client (`dream.LiveViewClient`, which the robots station
runs) speak the robot link to each other, and the relay speaks the viewer link to the browser, so every field name,
close code and limit here is read by both ends and declared nowhere else.
The platform mints the viewer token with sign_viewer_token() and the node checks it with verify_viewer_token(), the
one secret between them; the contract's golden-vector tests pin the format.

Four layers, top to bottom:

- The binary video message: a 16-byte big-endian header (#VideoHeader, pack_video_header(), unpack_video_header())
  in front of one encoded video frame: a VP8 frame from a station's agent (`hello.codec` `vp8`), an H.264 Annex-B
  access unit from an older agent (`avc1.42E01E`). The relay reads the header and forwards the bytes unchanged.
- The text messages: JSON objects with a `type` field, one pydantic model per type, read by parse_message() against
  the table of the link they arrived on (#ROBOT_LINK_TYPES, #VIEWER_LINK_TYPES). An unknown type or a malformed
  body raises `ValueError`, which the link answers with #CLOSE_PROTOCOL.
- The close codes (#CLOSE_CODES) and the constants both ends share. The node's admission limits and the
  agent's encoder knobs are private to their products and live in their own `constants.py`.
- The api seam: the bodies of the node's authorize call (#AuthorizeRequest, #AuthorizeResponse) and its reports
  (#Report, #LinkReport, #PromptReport, #ReportResponse, #Revocation), declared once so the node posts exactly
  what the api parses, and decode_token_secret(), the one reading of `LIVE_TOKEN_SECRET` both ends apply so they
  share the same HMAC key.

A record-only link is the robot link narrowed to two messages: a `hello` with `record_only` set (the dream client's
recorder, which no viewer ever sees) is followed by binary units whose payload is one JPEG each and by
#ObservationState; the relay stores both and forwards nothing, and a relay whose `LIVE_RECORD_ONLY` is off closes it
#CLOSE_RECORD_OFF.

Every duration is seconds and every size is bytes, as the unit suffixes say.
"""

import base64
import binascii
import dataclasses
import hashlib
import hmac
import json
import os
import struct
import typing

import pydantic

## The protocol version: `hello.protocol` and the first byte of every video header; a mismatch closes `4002`.
LIVE_PROTOCOL: int = 1

## Encoded width; the policy already sees 640 px and the page shows about that width, a quarter of 720p's pixels.
LIVE_WIDTH_PX: int = 640

## Frame rate of the encoded stream, the capture and control rate of the station.
LIVE_FPS: int = 30

## Seconds between IDRs: a skipped viewer resumes within a second and the relay's GOP cache stays around 175 KB.
LIVE_GOP_S: float = 1.0

## Average bitrate per camera; three cameras stay under 5 Mbps on a home uplink.
LIVE_BITRATE_KBPS: int = 1400

## The encoder's VBV buffer, half a second, which bounds the IDR size that sets the uplink hop of the latency budget.
LIVE_VBV_BUFFER_KBIT: int = 700

## Largest video message either end accepts: an IDR at 640x360 under the VBV is under 100 KB, so 1 MiB is a loud
## limit; the relay closes a link that exceeds it with `4009`.
LIVE_MAX_AU_BYTES: int = 1 << 20

## Largest `observation_state` text a record-only link may send: one tick's state vectors and prompt are a few KB,
## so 64 KiB is a loud limit; the relay closes a link that exceeds it with `4009`, and the client drops such a state
## rather than send it.
OBSERVATION_STATE_MAX_BYTES: int = 64 << 10

## The asyncio write-buffer high-water mark of a video socket on either end (the relay's viewer sockets, the agent's
## robot link): a pipe one IDR, or ten P-frames, deep before `send()` blocks.
LIVE_WRITE_HIGH_WATER: int = 64 << 10

## `TCP_NOTSENT_LOWAT` on the same sockets: what the kernel may hold unsent, a quarter of a large P-frame burst.
LIVE_NOTSENT_LOWAT: int = 16 << 10

## WebSocket ping interval on both links, the reference client's heartbeat; a dead link is seen in under 15 s.
LINK_PING_S: int = 5

## No pong within this long, two missed pings, closes the link with `4008`.
LINK_PONG_TIMEOUT_S: int = 10

## Floor of the robot's jittered exponential reconnect backoff; a blip costs one second.
LINK_RECONNECT_MIN_S: int = 1

## Ceiling of the backoff; caps the storm after a node restart at one attempt per robot per 30 s.
LINK_RECONNECT_MAX_S: int = 30

## A viewer token is minted `exp` this far out: the page uses it within a second, and 60 s bounds replay.
VIEWER_TOKEN_TTL_S: int = 60

## A viewer link that has not sent `auth` within this long is closed with `4001`.
VIEWER_AUTH_TIMEOUT_S: int = 5

## Each node posts its link snapshot to the api this often; revocation lands within two periods.
LINK_REPORT_S: int = 5

## A `live_links` row older than this is not fresh: three missed reports before the api calls a link gone.
LINK_STALE_S: int = 15

## Longest prompt text, the bound the public robots page and the station already enforce.
PROMPT_MAX_CHARS: int = 200

## Most joints per arm in a `tick` or `pose`: each arm sends its own count (a YAM six, a 7-joint arm seven), within
## room for a seven-joint arm on a lift or torso; a longer list is a protocol error, not a dataset row.
TICK_MAX_ARM_DOF: int = 16

## Most cameras a `hello` may list: the video header's `camera` byte indexes them, so a 257th is unreachable.
HELLO_MAX_CAMERAS: int = 256

## Longest camera name, codec string or agent version in `hello`; the relay logs them and copies the camera list
## into every viewer `status`, so an unbounded field would be a link's way to inflate both.
HELLO_MAX_FIELD_CHARS: int = 64

## Longest `status.error`. The relay copies it into every owner viewer's `status`, so the agent cuts the text to this
## before it sends and a longer one is refused.
STATUS_ERROR_MAX_CHARS: int = 500

## Longest checkpoint slug a `start` may name or a `status` may carry; the platform's own slug length limit.
CHECKPOINT_MAX_CHARS: int = 128

## Most steps a `start` may ask a run to take: a step is one commanded control tick, so this is about an hour at the
## station's 30 Hz; the bound only keeps a typo from meaning a run nobody stops.
RUN_STEPS_MAX: int = 100_000

## What a robot on the dream client's IK executes: `eef`, the model's tool poses solved to the arm's joints, or
## `joint`, the model's own joints, which only a fine-tune of that robot may run (`robots.bodies.check_model`).
ActionSpace = typing.Literal["eef", "joint"]

## The subscription name of the joint-state stream: a viewer subscribes to it like a camera, the relay tells
## the robot like a camera, and the robot answers with `pose` messages instead of video. Never a camera name.
POSE_STREAM: str = "pose"

## How often the robot sends `pose` while a viewer holds the stream, per second: enough for a model to move as
## the arms do, small enough (about 300 bytes each) to ride the uplink beside three cameras.
POSE_HZ: int = 10

## At most one prompt per robot per this long; a prompt resets the episode and two seconds is one chunk.
PROMPT_MIN_INTERVAL_S: int = 2

## A prompt not acknowledged by the robot within this long is answered `ok: false, error: "timeout"`.
PROMPT_ACK_S: int = 10

## Viewer links per private robot: a team's tabs on one account. At the
## cap a further link of a user already holding one replaces that user's oldest (`4006`) instead of being
## refused, so a reconnect that outruns the idle reap never locks its own owner out.
LIVE_MAX_VIEWERS_PER_ROBOT_PRIVATE: int = 20

## Viewer links per public robot: the public robots page's bound.
LIVE_MAX_VIEWERS_PER_ROBOT_PUBLIC: int = 200

## The robot link's path on the live origin; robots dial `wss://<site>/live/robot`.
ROBOT_LINK_PATH: str = "/live/robot"

## The video header in `struct` notation: u8 version, u8 kind, u8 camera, u8 flags, u32 seq, u64 capture_us,
## big-endian, no padding.
VIDEO_HEADER_FORMAT: str = ">BBBBIQ"

## Bytes of the header; the access unit starts at this offset.
HEADER_SIZE: int = struct.calcsize(VIDEO_HEADER_FORMAT)

## The `kind` of a video access unit, the only kind protocol 1 defines.
KIND_VIDEO: int = 1

## Bit 0 of `flags`: the unit is a keyframe a decoder can start from (for H.264, an IDR with SPS and PPS in band).
FLAG_KEYFRAME: int = 0x01

## Bad or missing credential, `auth` too late, token expired or reused.
CLOSE_UNAUTHORIZED: int = 4001

## Version mismatch, malformed message, or `hello` / `auth` not first.
CLOSE_PROTOCOL: int = 4002

## The node's link or viewer cap is reached; try another node.
CLOSE_NODE_FULL: int = 4003

## Robot unknown, foreign, or not linked on this node.
CLOSE_NOT_FOUND: int = 4004

## A report named this robot in `revoke`; sent to the robot.
CLOSE_REVOKED: int = 4005

## A newer link for the same robot id was accepted; sent to the older robot link.
CLOSE_REPLACED: int = 4006

## The robot's viewer cap is reached; sent to the viewer.
CLOSE_VIEWER_CAP: int = 4007

## No pong within `LINK_PONG_TIMEOUT_S`.
CLOSE_IDLE: int = 4008

## A message over `LIVE_MAX_AU_BYTES`; sent to the robot.
CLOSE_TOO_LARGE: int = 4009

## The node is being removed; reconnect elsewhere.
CLOSE_DRAINING: int = 4010

## A record-only `hello` reached a relay whose `LIVE_RECORD_ONLY` is off; the client stops recording for that policy.
CLOSE_RECORD_OFF: int = 4011

## Every application close code by its name; the relay counts refusals by these names.
CLOSE_CODES: dict[int, str] = {
    CLOSE_UNAUTHORIZED: "unauthorized",
    CLOSE_PROTOCOL: "protocol",
    CLOSE_NODE_FULL: "node_full",
    CLOSE_NOT_FOUND: "not_found",
    CLOSE_REVOKED: "revoked",
    CLOSE_REPLACED: "replaced",
    CLOSE_VIEWER_CAP: "viewer_cap",
    CLOSE_IDLE: "idle",
    CLOSE_TOO_LARGE: "too_large",
    CLOSE_DRAINING: "draining",
    CLOSE_RECORD_OFF: "record_off",
}


@dataclasses.dataclass(frozen=True, kw_only=True)
class VideoHeader:
    """The 16 bytes in front of every video access unit, in wire order.

    Attributes:
    - version: The protocol version, #LIVE_PROTOCOL.
    - kind: The message kind, #KIND_VIDEO.
    - camera: Index into the `cameras` list of the robot's `hello`.
    - flags: Bit flags; #FLAG_KEYFRAME marks a keyframe a decoder can start from.
    - seq: Per-camera counter, one per encoded frame; a gap is a skipped frame.
    - capture_us: Robot wall clock at capture, microseconds.
    """

    ## The protocol version.
    version: int = LIVE_PROTOCOL
    ## The message kind.
    kind: int = KIND_VIDEO
    camera: int
    flags: int
    seq: int
    capture_us: int

    @property
    def is_keyframe(self) -> bool:
        """Whether the unit is a keyframe a decoder can start from."""
        return bool(self.flags & FLAG_KEYFRAME)


def pack_video_header(header: VideoHeader) -> bytes:
    """Packs a header into its 16 wire bytes; the caller appends the access unit.

    @param header The header to send.
    @return #HEADER_SIZE bytes, big-endian.
    @throws struct.error If a field does not fit its width (a `camera` over 255, a negative `seq`).
    """
    return struct.pack(
        VIDEO_HEADER_FORMAT,
        header.version,
        header.kind,
        header.camera,
        header.flags,
        header.seq,
        header.capture_us,
    )


def unpack_video_header(message: bytes) -> VideoHeader:
    """Reads the header off the front of a binary message; the bytes from #HEADER_SIZE on are the access unit.

    @param message The whole binary message as received.
    @return The header.
    @throws ValueError If the message is shorter than a header, or its version or kind is not protocol 1's.
    """
    if len(message) < HEADER_SIZE:
        raise ValueError(f"video message of {len(message)} bytes is shorter than the {HEADER_SIZE}-byte header")
    version, kind, camera, flags, seq, capture_us = struct.unpack_from(VIDEO_HEADER_FORMAT, message)
    if version != LIVE_PROTOCOL:
        raise ValueError(f"video header version {version}, expected {LIVE_PROTOCOL}")
    if kind != KIND_VIDEO:
        raise ValueError(f"video header kind {kind}, expected {KIND_VIDEO}")
    return VideoHeader(version=version, kind=kind, camera=camera, flags=flags, seq=seq, capture_us=capture_us)


## A string field of `hello`: a camera name, the codec string or the agent version, bounded by #HELLO_MAX_FIELD_CHARS.
HelloField = typing.Annotated[str, pydantic.StringConstraints(max_length=HELLO_MAX_FIELD_CHARS)]


class Hello(pydantic.BaseModel):
    """`hello`, robot to relay: the first message after the upgrade; the relay authorizes the link and answers.

    Attributes:
    - protocol: The agent's #LIVE_PROTOCOL; the relay closes `4002` on a mismatch.
    - robot_id: The robot the link claims, one per link.
    - cameras: Camera names a viewer may subscribe to, at most #HELLO_MAX_CAMERAS of at most #HELLO_MAX_FIELD_CHARS
      each; the video header's `camera` indexes this list.
    - width_px: Encoded width, normally #LIVE_WIDTH_PX.
    - fps: Encoded frame rate, normally #LIVE_FPS.
    - codec: The WebCodecs codec string of the stream, which the page configures its decoder with: `vp8` from a
      station's agent, `avc1.42E01E` (H.264 Constrained Baseline) from an older agent; at most
      #HELLO_MAX_FIELD_CHARS.
    - face_blur: Whether every encoded frame is blurred; a public robot must declare it.
    - agent_version: The `dream-robots` version, for the relay's logs; at most #HELLO_MAX_FIELD_CHARS.
    - timezone: The station's local time zone, an IANA name (`America/Los_Angeles`) or a fixed offset
      (`UTC-07:00`) when the box has no zone database; stored beside every frame and tick so a reader can turn
      the UTC stamps into the customer's local time. Empty from an agent older than 0.7.0.
    - pose: Whether the robot streams its joint state on request: a viewer's `subscribe {camera: "pose"}`
      (#POSE_STREAM) reaches it like a camera's and it answers with `pose` messages at #POSE_HZ. False from
      an agent older than the field, and the relay refuses the subscription with `unknown_camera` then.
    - record_only: Whether the link only records: the dream client's recorder, whose robot id is its own random
      `rec_` + 32 hex, whose units are JPEGs (`codec` `jpeg`, `width_px` 0 since each frame carries its size) and
      whose text is #ObservationState only. The relay stores what it sends and never shows it: no viewer, report or
      prompt reaches it, and it may not offer `pose` (`4002`). A relay with `LIVE_RECORD_ONLY` off closes it
      #CLOSE_RECORD_OFF. Left out of the JSON while false, so an ordinary `hello` carries no such key and a relay
      older than the field, which forbids the key, still admits it.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    type: typing.Literal["hello"] = "hello"
    protocol: int
    robot_id: str = pydantic.Field(min_length=1)
    cameras: list[HelloField] = pydantic.Field(max_length=HELLO_MAX_CAMERAS)
    width_px: int
    fps: int
    codec: HelloField
    face_blur: bool
    agent_version: HelloField
    timezone: HelloField = ""
    pose: bool = False
    record_only: bool = False

    @pydantic.field_validator("cameras")
    @classmethod
    def _no_camera_named_pose(cls, cameras: list[str]) -> list[str]:
        """#POSE_STREAM is the joint-state stream's name and never a camera's."""
        if POSE_STREAM in cameras:
            raise ValueError(f"a camera cannot be named {POSE_STREAM!r}")
        return cameras

    @pydantic.model_serializer(mode="wrap")
    def _record_only_when_set(self, handler: pydantic.SerializerFunctionWrapHandler) -> dict[str, typing.Any]:
        """Leaves `record_only` out while it is false: the robots agent sends `model_dump_json()` as it is."""
        body: dict[str, typing.Any] = handler(self)
        if not self.record_only:
            body.pop("record_only", None)
        return body


class HelloOk(pydantic.BaseModel):
    """`hello_ok`, relay to robot: the link is established.

    Attributes:
    - node_id: The node holding the link, for `status().live.node`.
    - protocol: The relay's #LIVE_PROTOCOL.
    - subscribed: Cameras with viewers already waiting, so a reconnect restores its streams at once.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    type: typing.Literal["hello_ok"] = "hello_ok"
    node_id: str = pydantic.Field(min_length=1)
    protocol: int
    subscribed: list[str]


class Subscribe(pydantic.BaseModel):
    """`subscribe`, viewer to relay and relay to robot: start receiving, or encoding, one camera.

    Attributes:
    - camera: A name from the robot's `cameras`, or #POSE_STREAM for the joint-state stream.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    type: typing.Literal["subscribe"] = "subscribe"
    camera: str = pydantic.Field(min_length=1)


class Unsubscribe(pydantic.BaseModel):
    """`unsubscribe`, viewer to relay and relay to robot: stop receiving, or encoding, one camera.

    Attributes:
    - camera: A name from the robot's `cameras`.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    type: typing.Literal["unsubscribe"] = "unsubscribe"
    camera: str = pydantic.Field(min_length=1)


## Bounds of a round trip a `status` may carry, milliseconds: a negative one or one over ten minutes is not a
## measurement, and the relay copies the block into every viewer's `status`.
TIMING_MAX_MS: float = 600_000.0

## A round-trip figure: milliseconds, within #TIMING_MAX_MS.
TimingMs = typing.Annotated[float, pydantic.Field(ge=0, le=TIMING_MAX_MS)]

## Most round trips one `status` lists in `timing.recent_ms`: a status goes out every #TIMING_STATUS_S and a round
## trip takes most of a second, so a handful at most; the bound only stops a list from inflating every viewer's.
TIMING_RECENT_MAX: int = 20


class LiveTiming(pydantic.BaseModel):
    """How long the station's `get_action` round trips take this run: the control loop's `RefillTiming`, the
    part a viewer shows.

    Attributes:
    - rtt_ms: The last refill's round trip, from the tick that asked to the tick that collected the chunk, on the
      loop's clock (so quantised to a control period).
    - p50_ms, p95_ms: The run's median and 95th percentile round trip so far.
    - network_ms: The part of the last round trip spent on the network: `rtt_ms` minus the time in DreamAPI.
    - gateway_ms: The time DreamAPI took for the last one, gateway and serve together.
    - serve_infer_ms: The serve's own inference time for the last one.
    - count: Refills that returned a chunk this run.
    - recent_ms: Every round trip since the previous `status` that carried timing, oldest first, so a page can draw
      each one rather than one every #TIMING_STATUS_S. Empty from an agent that did not see them.
    - up_mbps: The upload rate the station's client achieved for the last request, megabits per second (the dream
      client's `last_timing["up_mbps"]`, the kernel's delivery rate for the request's socket), so a page that says
      the network is slow can say how fast the station uploads. None off Linux, and from an agent older than the
      field, which leaves the key out.

    The network, DreamAPI and inference parts are None when the station's client has no timing from the gateway
    (an older gateway).
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    rtt_ms: TimingMs
    p50_ms: TimingMs | None = None
    p95_ms: TimingMs | None = None
    network_ms: float | None = None
    gateway_ms: TimingMs | None = None
    serve_infer_ms: TimingMs | None = None
    count: int = pydantic.Field(ge=0)
    recent_ms: list[TimingMs] = pydantic.Field(default_factory=list, max_length=TIMING_RECENT_MAX)
    up_mbps: float | None = pydantic.Field(default=None, ge=0)


class RobotStatus(pydantic.BaseModel):
    """`status`, robot to relay: sent after `hello` and on every station state change.

    Attributes:
    - state: The controller state, `robots.station`'s vocabulary (`idle`, `starting`, `running`, ...).
    - prompt: The prompt the policy is running, or `None` between runs.
    - run: The run number, or `None` before the first run.
    - dropped_frames: Frames skipped on the uplink, per camera name.
    - live_paused: `"tick"` while the agent has yielded to protect the control tick, else
      `None`.
    - error: Why no actions are reaching the arms, in the station's own words: while a run is on, the message of the
      control loop's last failed `get_action` (DreamAPI's `DA-2000 ... (pool_warming)` while a serve boots), which
      the loop is still retrying; in `error`, what ended the run. At most #STATUS_ERROR_MAX_CHARS. The agent
      leaves the key out rather than send `null`, because a relay older than the field forbids it like any unknown
      key and closes the link `4002`.
    - session_id: The DreamAPI session of the run in progress (the uuid the gateway minted on the episode's first
      `get_action`), left out before the first chunk and between runs; the unique identity of a run across station
      restarts, and the key that joins the stored run to the platform's usage rows.
    - error_code: What `error` is, for a page that words its own message: DreamAPI's envelope reason
      (`pool_warming`, `unknown_checkpoint`, `not_entitled`, ...) when the failure is a DreamAPI refusal, else the
      class of the fault (`ArmError`, `CameraError`, `ControlLoopError`, `HomeError`, ...). Left out with `error`.
    - checkpoint: The checkpoint the station runs, or will run at the next start: its config's, or the one the
      last `start` named. Left out by an agent older than the field.
    - arms: Whether each arm's driver is still answering, by arm name (`{"left": true, "right": false}`), read
      from the driver's motor chain; an arm that died reads `false` until a `reset` reopens it. Empty from an
      agent older than the field.
    - timing: The run's `get_action` round trips (#LiveTiming), while a run is on and a chunk has come back. A
      relay older than the field forbids the key and closes the link `4002`. The agent sends a `status` for it at
      most every #TIMING_STATUS_S, not on every refill.
    - action_space: What the station executes at the next start (#ActionSpace): its config's, or the one the last
      `start` named. Left out by a station without the dream client's IK (a YAM, which runs its model's joints)
      and by an agent older than the field, which takes no `prompt.action_space`. A relay older than the field
      closes the link `4002` on it.
    - steps: How many steps the next run takes, or the running one: the one the last `start` named, or 0 when none
      has and a run goes on until it is stopped. Left out by an agent older than the field, which takes no
      `prompt.steps`. A relay older than the field closes the link `4002` on it.
    - model_fixed: Whether the robot's own code sets the model and the actions, so a `start`'s `checkpoint` and
      `action_space` from the page are not applied: `dream.LiveViewClient` sends it true, and the page then offers no
      model or action choice. Left out of the JSON while false, so the station's `status` carries no such key and
      a relay older than the field, which forbids the key, still admits it.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    type: typing.Literal["status"] = "status"
    state: str
    prompt: str | None = None
    run: int | None = None
    dropped_frames: dict[str, int] = pydantic.Field(default_factory=dict)
    live_paused: str | None = None
    error: str | None = pydantic.Field(default=None, max_length=STATUS_ERROR_MAX_CHARS)
    session_id: str | None = None
    error_code: str | None = pydantic.Field(default=None, max_length=HELLO_MAX_FIELD_CHARS)
    checkpoint: str | None = pydantic.Field(default=None, max_length=CHECKPOINT_MAX_CHARS)
    arms: dict[str, bool] = pydantic.Field(default_factory=dict)
    timing: LiveTiming | None = None
    action_space: ActionSpace | None = None
    steps: int | None = pydantic.Field(default=None, ge=0, le=RUN_STEPS_MAX)
    model_fixed: bool = False

    @pydantic.model_serializer(mode="wrap")
    def _model_fixed_when_set(self, handler: pydantic.SerializerFunctionWrapHandler) -> dict[str, typing.Any]:
        """Leaves `model_fixed` out while it is false: the station's agent sends `model_dump_json()` as it is."""
        body: dict[str, typing.Any] = handler(self)
        if not self.model_fixed:
            body.pop("model_fixed", None)
        return body


## Seconds between two `status` messages a changed #LiveTiming may cause on its own (a state change sends one at
## once, timing included); a round trip lands every second or so, which would otherwise be a status each time.
TIMING_STATUS_S: float = 2.0


## A joint vector of a `tick` or `pose`: the arm's own joints, 1 to #TICK_MAX_ARM_DOF floats.
Joints = typing.Annotated[list[float], pydantic.Field(min_length=1, max_length=TICK_MAX_ARM_DOF)]


class TickArm(pydantic.BaseModel):
    """One arm as the loop read it on a tick (`robots.types.ArmState`).

    Attributes:
    - q: Joint positions, radians, one per joint of the arm.
    - qd: Joint velocities, radians per second, as many as `q`.
    - gripper: Gripper opening in `[0, 1]`.
    - gripper_effort: The gripper motor's effort as the driver reports it, or `None` when it reports none.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    q: Joints
    qd: Joints
    gripper: float
    gripper_effort: float | None = None

    @pydantic.model_validator(mode="after")
    def _qd_per_joint(self) -> "TickArm":
        """One velocity per position: the arm's joint count is `len(q)`."""
        if len(self.qd) != len(self.q):
            raise ValueError(f"qd has {len(self.qd)} joints, q has {len(self.q)}")
        return self


class TickTarget(pydantic.BaseModel):
    """One arm's command on a tick (`robots.types.ArmCommand`): what the loop wrote after the ramp, the seam and
    the clamp.

    Attributes:
    - q: Target joint positions, radians.
    - gripper: Target gripper opening in `[0, 1]`.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    q: Joints
    gripper: float


class Pose(pydantic.BaseModel):
    """`pose`, robot to relay and relay to viewer: where the arms are right now, for a page that draws the body.
    Sent at #POSE_HZ while any viewer holds the #POSE_STREAM subscription, running or idle; the relay hands it to
    those viewers as it is.

    Attributes:
    - t_us: Robot wall clock, microseconds, the video header's clock.
    - arms: Each arm's joint positions (radians) and gripper opening in `[0, 1]`, by arm name (`left`, `right`);
      an arm whose driver cannot be read is left out.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    type: typing.Literal["pose"] = "pose"
    t_us: int = pydantic.Field(ge=0)
    arms: dict[str, TickTarget]


class Tick(pydantic.BaseModel):
    """`tick`, robot to relay: one control tick's observation and action, the row of the dataset the relay's
    store keeps beside the video. Sent once per commanded tick while a run is on.

    Attributes:
    - tick: The loop's tick counter for this run, one per published state; a gap is a tick the agent missed.
    - run: The station's run number, which bounds an episode.
    - t_us: Robot wall clock at the tick, microseconds, the same clock as the video header's `capture_us`, so a
      frame and a tick are joined by time.
    - chunk: The `get_action` chunk the commanded row came from (the loop's chunk count), or `None` before the
      first.
    - row: The row of that chunk, or `None`.
    - clamped: Whether the rate clamp bounded the command.
    - state: Both arms as read this tick, by arm name (`left`, `right`).
    - action: Both arms' commands this tick, by arm name.
    - session_id: The DreamAPI session the command's chunk came from, the run's unique identity; left out by
      an agent older than 0.8.0.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    type: typing.Literal["tick"] = "tick"
    tick: int = pydantic.Field(ge=0)
    run: int = pydantic.Field(ge=0)
    session_id: str | None = None
    t_us: int = pydantic.Field(ge=0)
    chunk: int | None = None
    row: int | None = None
    clamped: bool = False
    state: dict[str, TickArm]
    action: dict[str, TickTarget]

    @pydantic.model_validator(mode="after")
    def _action_per_joint(self) -> "Tick":
        """Each arm's command has the joints its read has, so a stored row is never ragged within an arm."""
        for arm, target in self.action.items():
            if arm in self.state and len(target.q) != len(self.state[arm].q):
                raise ValueError(f"arm {arm!r} commands {len(target.q)} joints and reads {len(self.state[arm].q)}")
        return self


class ObservationState(pydantic.BaseModel):
    """`observation_state`, recorder to relay: one `set_observation()` of the dream client's recorder, kept or not.

    Sent on a record-only link only (`hello.record_only`); the relay stores it and never forwards it. On an ordinary
    robot link it is `4002`, and a text over #OBSERVATION_STATE_MAX_BYTES is `4009`.

    Attributes:
    - t_us: Wall clock of the tick, Unix microseconds, the clock of the video header's `capture_us`: the frames the
      tick kept carry this exact value, so a state and its frames join on it.
    - episode: The client's episode counter, 0 at its first policy and one more at each episode reset or new policy,
      so episodes separate even before a DreamAPI session exists.
    - session_id: The DreamAPI session the client held when the tick was set, or None before its first `get_action`
      answer.
    - kept: Whether the tick's frames were kept into the client's window, and so sent as units.
    - observation: Every key the caller passed except the frames (`state.*`, `prompt`, anything else), as JSON
      values (arrays as nested lists).
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    type: typing.Literal["observation_state"] = "observation_state"
    t_us: int = pydantic.Field(ge=0)
    episode: int = pydantic.Field(ge=0)
    session_id: str | None = None
    kept: bool
    observation: dict[str, typing.Any]


## What a `prompt` asks the station to do: `prompt` sets the task (`set_prompt()`), `start` starts a run
## (`start(text)`, the text optional: None keeps the station's prompt), `stop` stops the run and homes the arms
## (`stop()` then `home()`).
PromptAction = typing.Literal["prompt", "start", "stop", "home", "reset"]


class Prompt(pydantic.BaseModel):
    """`prompt`, viewer to relay and relay to robot: a task for the policy, or a start, stop, home or reset of
    the station; owner viewers only.

    Attributes:
    - id: Correlates the `prompt_ack`; the relay records it as the `instructions` row id.
    - action: What the station does with it (#PromptAction); `prompt` when absent, so an older page still works.
      `home` moves the arms home with nothing running; `reset` reopens the arms' drivers (an arm whose motor chain
      died) and then homes them, so a station in `error` after an arm fault comes back from the page.
    - text: The task, 1 to #PROMPT_MAX_CHARS characters; required for `prompt`, optional for `start` (None
      keeps the station's current prompt), absent for `stop`, `home` and `reset`.
    - checkpoint: With `start` only: the checkpoint this run and the ones after it use, a slug the platform's
      `GET /v1/models` lists; None keeps the station's current one.
    - action_space: With `start` only: what this run and the ones after it execute (#ActionSpace); None keeps
      the station's current one. Sent only to a station whose `status` carries `action_space`: an older agent
      closes the link `4002` on the key.
    - steps: With `start` only: how many steps this run and the ones after it take, 1 to #RUN_STEPS_MAX; a step is
      one commanded control tick, and the run ends on its own, the arms going home, after the last. None keeps the
      station's current count. Sent only to a station whose `status` carries `steps`, for the same reason as
      `action_space`.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    type: typing.Literal["prompt"] = "prompt"
    id: str = pydantic.Field(min_length=1)
    action: PromptAction = "prompt"
    text: str | None = pydantic.Field(default=None, min_length=1, max_length=PROMPT_MAX_CHARS)
    checkpoint: str | None = pydantic.Field(default=None, min_length=1, max_length=CHECKPOINT_MAX_CHARS)
    action_space: ActionSpace | None = None
    steps: int | None = pydantic.Field(default=None, ge=1, le=RUN_STEPS_MAX)

    @pydantic.model_validator(mode="after")
    def _text_fits_the_action(self) -> "Prompt":
        """`prompt` needs a text, `stop`, `home` and `reset` refuse one, `start` takes either; only a `start`
        names a checkpoint, an action space or a step count."""
        if self.action == "prompt" and self.text is None:
            raise ValueError("a prompt needs a text")
        if self.action in ("stop", "home", "reset") and self.text is not None:
            raise ValueError(f"a {self.action} carries no text")
        if self.checkpoint is not None and self.action != "start":
            raise ValueError("only a start names a checkpoint")
        if self.action_space is not None and self.action != "start":
            raise ValueError("only a start names an action space")
        if self.steps is not None and self.action != "start":
            raise ValueError("only a start names a step count")
        return self


class PromptAck(pydantic.BaseModel):
    """`prompt_ack`, robot to relay and relay to viewer: the controller's answer, or the relay's timeout.

    Attributes:
    - id: The `prompt` it answers.
    - ok: Whether `set_prompt()` accepted the text.
    - state: The controller state after the call; `None` when the relay synthesized the ack.
    - error: Why it was refused (`StationError` or `ValueError` text, or `"timeout"`), else `None`.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    type: typing.Literal["prompt_ack"] = "prompt_ack"
    id: str = pydantic.Field(min_length=1)
    ok: bool
    state: str | None = None
    error: str | None = None


class Bye(pydantic.BaseModel):
    """`bye`, relay to either side: sent just before an application close, so the reason survives the close frame.

    Attributes:
    - code: One of #CLOSE_CODES.
    - reason: Its name, for example `revoked`.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    type: typing.Literal["bye"] = "bye"
    code: int
    reason: str


class Auth(pydantic.BaseModel):
    """`auth`, viewer to relay: the first message, within #VIEWER_AUTH_TIMEOUT_S; a message, not a URL parameter,
    so the token never reaches an access log.

    Attributes:
    - token: The string `POST /api/robots/{id}/live/token` returned; verify_viewer_token() reads it.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    type: typing.Literal["auth"] = "auth"
    token: str = pydantic.Field(min_length=1)


class ViewerStatus(pydantic.BaseModel):
    """`status`, relay to viewer: after `auth` and on every change of the robot's link or state.

    Attributes:
    - linked: Whether the robot has a link on this node; `False` leaves every other field empty but `public` and
      `unlinked_reason`.
    - state: The controller state as the robot last reported it.
    - prompt: The running prompt, as the robot last reported it.
    - cameras: Camera names the viewer may subscribe to.
    - width_px: Encoded width from the robot's `hello`.
    - fps: Encoded frame rate from the robot's `hello`.
    - codec: The WebCodecs codec string from the robot's `hello`.
    - public: Whether this viewer holds a public token, which cannot `prompt`.
    - live_paused: `"tick"` while the agent has yielded to protect the control tick, so the page can say so.
    - error: The robot's `status.error` as it last reported it, for an owner's viewer only: the text carries
      DreamAPI's free-text detail (a checkpoint name, why a key was refused), which a public viewer has no claim
      to.
    - error_code: The robot's `status.error_code`, owner's viewer only like `error`.
    - checkpoint: The robot's `status.checkpoint`: the model it runs or will run next.
    - arms: The robot's `status.arms`: whether each arm's driver still answers.
    - session_id: The robot's `status.session_id`, the DreamAPI session of the run in progress, for an owner's
      viewer only like `error`; None before a run's first chunk and between runs.
    - timing: The robot's `status.timing`, the run's `get_action` round trips, for every viewer: it names no
      checkpoint, key or session.
    - action_space: The robot's `status.action_space`: what it executes at the next start; None from a YAM or
      an agent older than the field, which the page then sends no `action_space`.
    - unlinked_reason: While `linked` is `False`, why this node last dropped or refused the robot's link when the
      api said: its revoke reason (`revoked`, `suspended`, `over_cap`, `deleted`), or `over_cap` for an authorize
      the api answered 402; None when the link just closed. Owner's viewer only like `error`; None from a node
      older than the field.
    - steps: The robot's `status.steps`: how many steps the next run takes, 0 for a run that goes on until it is
      stopped; None from an agent older than the field, which the page then sends no `steps`.
    - model_fixed: The robot's `status.model_fixed`: its code sets the model and the actions, so the page offers no
      choice of either.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    type: typing.Literal["status"] = "status"
    linked: bool
    state: str | None = None
    prompt: str | None = None
    cameras: list[str] = pydantic.Field(default_factory=list)
    width_px: int | None = None
    fps: int | None = None
    codec: str | None = None
    public: bool
    live_paused: str | None = None
    error: str | None = pydantic.Field(default=None, max_length=STATUS_ERROR_MAX_CHARS)
    error_code: str | None = pydantic.Field(default=None, max_length=HELLO_MAX_FIELD_CHARS)
    checkpoint: str | None = pydantic.Field(default=None, max_length=CHECKPOINT_MAX_CHARS)
    arms: dict[str, bool] = pydantic.Field(default_factory=dict)
    session_id: str | None = None
    timing: LiveTiming | None = None
    action_space: ActionSpace | None = None
    unlinked_reason: str | None = pydantic.Field(default=None, max_length=HELLO_MAX_FIELD_CHARS)
    steps: int | None = pydantic.Field(default=None, ge=0, le=RUN_STEPS_MAX)
    model_fixed: bool = False


class ErrorMessage(pydantic.BaseModel):
    """`error`, relay to viewer: a refused message; the link stays open.

    Attributes:
    - code: A short machine-readable reason, `forbidden`, `rate_limited`, `unknown_camera`, ...
    - message: One sentence for the page.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    type: typing.Literal["error"] = "error"
    code: str = pydantic.Field(min_length=1)
    message: str


## Any text message of either link; parse_message() returns one and a caller branches with `isinstance`.
Message = (
    Hello
    | HelloOk
    | Subscribe
    | Unsubscribe
    | RobotStatus
    | Tick
    | ObservationState
    | Pose
    | Prompt
    | PromptAck
    | Bye
    | Auth
    | ViewerStatus
    | ErrorMessage
)

## The robot link's vocabulary by `type`, both directions.
ROBOT_LINK_TYPES: dict[str, type[Message]] = {
    "hello": Hello,
    "hello_ok": HelloOk,
    "subscribe": Subscribe,
    "unsubscribe": Unsubscribe,
    "status": RobotStatus,
    "tick": Tick,
    "observation_state": ObservationState,
    "pose": Pose,
    "prompt": Prompt,
    "prompt_ack": PromptAck,
    "bye": Bye,
}

## The viewer link's vocabulary by `type`, both directions; `status` is #ViewerStatus here.
VIEWER_LINK_TYPES: dict[str, type[Message]] = {
    "auth": Auth,
    "status": ViewerStatus,
    "subscribe": Subscribe,
    "unsubscribe": Unsubscribe,
    "pose": Pose,
    "prompt": Prompt,
    "prompt_ack": PromptAck,
    "error": ErrorMessage,
    "bye": Bye,
}


def parse_message(data: str | bytes, types: dict[str, type[Message]]) -> Message:
    """Reads one text frame as the message its `type` names, against one link's table.

    A link that receives a `ValueError` from here closes with #CLOSE_PROTOCOL: the sender is not speaking protocol
    1, and there is nothing to negotiate.

    @param data The text frame as received.
    @param types #ROBOT_LINK_TYPES or #VIEWER_LINK_TYPES, whichever link the frame arrived on.
    @return The validated message.
    @throws ValueError If the frame is not a JSON object, its `type` is not in the table, or a field is missing,
        unknown or of the wrong shape (`pydantic.ValidationError` is a `ValueError`).
    """
    body = json.loads(data)
    if not isinstance(body, dict):
        raise ValueError(f"a message is a JSON object, got {type(body).__name__}")
    kind = body.get("type")
    if not isinstance(kind, str) or kind not in types:
        raise ValueError(f"unknown message type {kind!r}")
    return types[kind].model_validate(body)


class AuthorizeRequest(pydantic.BaseModel):
    """The authorize call, node to api: may this key link this robot.

    Attributes:
    - key_hash: sha256 hex of the api key on the upgrade.
    - robot_id: The robot the `hello` claims; None for a record-only link, whose robot id is the client's own and
      names no registered robot, so only the key and its owner's spend cap are checked.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    key_hash: str = pydantic.Field(min_length=1)
    robot_id: str | None = pydantic.Field(default=None, min_length=1)


class AuthorizeResponse(pydantic.BaseModel):
    """The api's `200` to #AuthorizeRequest; `401` (key unknown, revoked or suspended), `402` (owner over the monthly
    spend cap) and `404` (robot unknown or another user's; never without a robot id) are refusals.

    Attributes:
    - user_id: The owner of the key and the robot, the `user_id` a private viewer token must carry.
    - public: Whether `robots.public_at` is set, so the node admits public viewers; False without a robot id.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    user_id: str
    public: bool


class LinkReport(pydantic.BaseModel):
    """One held robot link in a report, the `live_links` row the api writes.

    Attributes:
    - robot_id: The robot.
    - key_hash: sha256 hex of the key the link presented; the api checks it is still allowed.
    - user_id: The owner the api named at authorize.
    - viewers: Viewer links of this robot on this node right now.
    - face_blur: What the robot's `hello` declared.
    - linked_at: When the link was accepted; on the wire an ISO 8601 string with a zone, or unix seconds.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    robot_id: str = pydantic.Field(min_length=1)
    key_hash: str = pydantic.Field(min_length=1)
    user_id: str = pydantic.Field(min_length=1)
    viewers: int = pydantic.Field(ge=0)
    face_blur: bool
    linked_at: pydantic.AwareDatetime


class PromptReport(pydantic.BaseModel):
    """One prompt answered since the last report, the `instructions` row the api inserts with `via = 'live'`.

    Attributes:
    - id: The relay's `prm_` + 16 hex id, the row id.
    - robot_id: The robot it went to.
    - action: What it asked for (#PromptAction); the api records a text-less `start` or `stop` under that word.
    - text: The prompt, 1 to #PROMPT_MAX_CHARS characters, or None for a `stop` or a `start` without one.
    - ok: Whether the robot's `prompt_ack` accepted it; `False` also for the relay's timeout.
    - state: The controller state the ack carried, or `None`.
    - error: Why it was refused (`"timeout"`, `"unlinked"`, or the controller's text), or `None`.
    - sent_at: When the relay forwarded it; on the wire an ISO 8601 string with a zone, or unix seconds.
    - answered_at: When the ack arrived or the relay gave up.
    - run: The station's run number as its `status` last said when the ack arrived, which is the run the prompt
      started or joined; None from a relay older than the field or a robot that sent no `status` yet.
    - session_id: The DreamAPI session of that run from the same `status`, or None.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    id: str = pydantic.Field(min_length=1)
    robot_id: str = pydantic.Field(min_length=1)
    action: PromptAction = "prompt"
    text: str | None = pydantic.Field(default=None, min_length=1, max_length=PROMPT_MAX_CHARS)
    ok: bool
    state: str | None = None
    error: str | None = None
    sent_at: pydantic.AwareDatetime
    answered_at: pydantic.AwareDatetime
    run: int | None = None
    session_id: str | None = None


## What the owner said of a trajectory: it did the task, or it did not.
Rating = typing.Literal["good", "bad"]


class TrajectoryRating(pydantic.BaseModel):
    """One owner's verdict on the trajectory a prompt produced, handed by the api to a node in a #ReportResponse so
    the dataset carries it beside the run's ticks and video.

    Attributes:
    - id: The instruction's id, the prompt's `prm_` id for a prompt sent over the live link.
    - robot_id: The robot.
    - user_id: The owner who rated it.
    - prompt: The instruction's text (the action word for a text-less start, stop, home or reset).
    - rating: #Rating.
    - rated_at: When the owner rated it (the latest verdict; a re-rating replaces the earlier one).
    - prompted_at: When the prompt was sent, the start of the trajectory's window.
    - run: The station's run the prompt started or joined, or None when the report did not know it.
    - session_id: The DreamAPI session of that run, or None.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    id: str = pydantic.Field(min_length=1)
    robot_id: str = pydantic.Field(min_length=1)
    user_id: str = pydantic.Field(min_length=1)
    prompt: str = pydantic.Field(min_length=1)
    rating: Rating
    rated_at: pydantic.AwareDatetime
    prompted_at: pydantic.AwareDatetime
    run: int | None = None
    session_id: str | None = None


class RatingStored(pydantic.BaseModel):
    """One #TrajectoryRating a node wrote into its dataset, named back to the api in the next #Report.

    Attributes:
    - id: The rating's (the instruction's) id.
    - rated_at: The verdict time the node wrote, so a re-rating made meanwhile (a later `rated_at`) is not taken as
      delivered and is handed over again.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    id: str = pydantic.Field(min_length=1)
    rated_at: pydantic.AwareDatetime


class Report(pydantic.BaseModel):
    """The report, node to api: one node's whole snapshot.

    Attributes:
    - node_id: The reporting node; the api accepts only the ids in its `LIVE_NODES`.
    - links: Every robot link it holds; the api replaces the node's `live_links` rows with these, so an empty list
      drops every row it reported before.
    - prompts: Prompts answered since the last accepted report; a prompt reported twice is recorded once.
    - ratings_stored: The #TrajectoryRating rows the node wrote into its dataset since the last accepted report,
      each by id and verdict time, so the api stops handing them over; a node without a store confirms them
      too, the rating then living on the api alone.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    node_id: str = pydantic.Field(min_length=1)
    links: list[LinkReport]
    prompts: list[PromptReport] = pydantic.Field(default_factory=list)
    ratings_stored: list[RatingStored] = pydantic.Field(default_factory=list)


class Revocation(pydantic.BaseModel):
    """One robot link the api says must close with #CLOSE_REVOKED.

    Attributes:
    - robot_id: The robot.
    - reason: Why: `revoked`, `suspended`, #REVOKED_OVER_CAP or `deleted`; logged, and told to the robot's owner's
      viewers as #ViewerStatus `unlinked_reason`.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    robot_id: str = pydantic.Field(min_length=1)
    reason: str = pydantic.Field(min_length=1)


## The #Revocation reason, and the #ViewerStatus `unlinked_reason`, for an owner whose month's spend passed their cap.
REVOKED_OVER_CAP: str = "over_cap"


class ReportResponse(pydantic.BaseModel):
    """The api's `200` to a #Report.

    Attributes:
    - revoke: Robot links to close with #CLOSE_REVOKED; empty when every link is still allowed.
    - ratings: Trajectory ratings the owner gave on the site that no node has written into the dataset yet; the
      node writes each and names it in its next report's `ratings_stored`. Any node may take any robot's.
    - video_off: The reported robots whose owner is over the monthly spend cap while `LIVE_BUDGET_NOTICE` keeps
      their links: the node stops their video (not their pose) and says `live_paused` #VIDEO_OFF_OVER_CAP to their
      viewers until a report leaves them out. Left out of the JSON when empty, since a node older than the field
      forbids it.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    revoke: list[Revocation]
    ratings: list[TrajectoryRating] = pydantic.Field(default_factory=list)
    video_off: list[str] = pydantic.Field(default_factory=list, exclude_if=lambda robot_ids: not robot_ids)


## The #ViewerStatus `live_paused` of a robot whose video the node stopped because its owner is over the spend cap.
VIDEO_OFF_OVER_CAP: str = "over_cap"


class ViewerTokenPayload(pydantic.BaseModel):
    """What a viewer token signs: who may watch which robot on which node, until when, once.

    Attributes:
    - user_id: The owner's id, or `""` for a public token minted for the public robots page.
    - robot_id: The one robot the link may carry frames of.
    - node_id: The node the token was minted for; the relay refuses a token naming another node.
    - exp: Unix time, seconds, after which the token is refused; minted #VIEWER_TOKEN_TTL_S out.
    - nonce: 16 random bytes as hex from new_nonce(); the relay accepts each once until `exp`.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    user_id: str
    robot_id: str = pydantic.Field(min_length=1)
    node_id: str = pydantic.Field(min_length=1)
    exp: int
    nonce: str = pydantic.Field(min_length=1)


def decode_token_secret(text: str) -> bytes:
    """The HMAC key `LIVE_TOKEN_SECRET` (or `LIVE_TOKEN_SECRET_PREVIOUS`) names: URL-safe base64, padding optional.

    The value the example env file recommends is `secrets.token_urlsafe(32)`, which is exactly this alphabet without
    padding; a standard-alphabet value decodes too. The api signs with these bytes and every node verifies with
    them, so both must read the variable through this one function or no token verifies.

    @param text The environment value, surrounding whitespace ignored.
    @return The secret bytes, never empty.
    @throws ValueError If the value is empty, is not base64, or decodes to no bytes.
    """
    stripped = text.strip()
    if not stripped:
        raise ValueError("the token secret is empty")
    try:
        secret = base64.urlsafe_b64decode(stripped + "=" * (-len(stripped) % 4))
    except binascii.Error as error:
        raise ValueError("the token secret must be base64 (secrets.token_urlsafe(32))") from error
    if not secret:
        raise ValueError("the token secret decodes to no bytes")
    return secret


def new_nonce() -> str:
    """A fresh token nonce: 16 random bytes as 32 hex characters.

    @return The nonce.
    """
    return os.urandom(16).hex()


def sign_viewer_token(secret: bytes, payload: ViewerTokenPayload) -> str:
    """Mints a viewer token: `base64url(payload_json) + "." + base64url(hmac_sha256(secret, payload_json))`.

    The payload JSON is canonical (sorted keys, no spaces) so one payload has one token; the signature covers the
    exact bytes the token carries, so verify_viewer_token() never re-serializes.

    @param secret `LIVE_TOKEN_SECRET`, the bytes the api and the nodes share.
    @param payload What the token grants.
    @return The token, URL-safe ASCII without padding.
    """
    body = json.dumps(payload.model_dump(), separators=(",", ":"), sort_keys=True).encode("utf-8")
    mac = hmac.new(secret, body, hashlib.sha256).digest()
    return f"{_base64url(body)}.{_base64url(mac)}"


def verify_viewer_token(secrets: list[bytes], token: str, now: float) -> ViewerTokenPayload | None:
    """Checks a viewer token's signature and expiry; the caller checks `robot_id`, `node_id` and the nonce.

    Every secret in `secrets` is tried, so a node handed `LIVE_TOKEN_SECRET` and `LIVE_TOKEN_SECRET_PREVIOUS`
    accepts tokens minted before a rotation for one TTL. Signatures are compared with `hmac.compare_digest`.

    @param secrets The secrets to accept, current first.
    @param token The string from `auth`.
    @param now Unix time, seconds.
    @return The signed payload, or `None` when the token is malformed, signed by none of the secrets, or expired.
    """
    head, dot, tail = token.partition(".")
    if not dot:
        return None
    try:
        body = _unbase64url(head)
        mac = _unbase64url(tail)
    except ValueError:
        return None
    if not any(hmac.compare_digest(mac, hmac.new(secret, body, hashlib.sha256).digest()) for secret in secrets):
        return None
    try:
        payload = ViewerTokenPayload.model_validate_json(body)
    except pydantic.ValidationError:
        return None
    return None if now >= payload.exp else payload


def viewer_path(node_id: str, robot_id: str) -> str:
    """The viewer link's path on the live origin, `/live/n/<node_id>/robots/<robot_id>/view`.

    @param node_id The node holding the robot, from `live_links`.
    @param robot_id The robot to watch.
    @return The path; the live origin routes it to the node by its `/live/n/<node_id>/` prefix.
    """
    return f"/live/n/{node_id}/robots/{robot_id}/view"


def _base64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _unbase64url(text: str) -> bytes:
    """Strict inverse of _base64url(): a character outside the alphabet is a `ValueError`, not silently dropped."""
    return base64.b64decode(text + "=" * (-len(text) % 4), altchars=b"-_", validate=True)
