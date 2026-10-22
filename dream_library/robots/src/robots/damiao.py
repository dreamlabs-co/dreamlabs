# Ported from LeRobot's Damiao bus as MakerMods ship it (`lerobot/motors/damiao/damiao.py` and `tables.py`, their
# LeRobot fork at 5ce2fe4), Copyright 2025 The HuggingFace Inc. team, Apache License 2.0; portions of it are derived
# from DM_Control_Python by cmjang (https://github.com/cmjang/DM_Control_Python), MIT License. query() is ported from
# MakerMods Lab (`makermodslab/metal_gripper.py` at f3ee37c), Apache License 2.0.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""Damiao motors in MIT mode on one CAN adapter: the frame codec and a request/reply bus; the only module that knows
the Damiao wire.

A MakerMods Metal arm (#robots.metal) has seven Damiao motors, six joints and the gripper, on one classic CAN bus at
1 Mbit/s per arm (no CAN FD), addressed by CAN id 1 to 7. Both arms answer to the same ids, so each arm has its own
adapter, opened as a Makers Arm's is (#robots.robstride.open_can_bus: a CANable's serial device on slcan, or a
SocketCAN link).

Ported from MakerMods' Metal control loop (`metal-control-loop`): LeRobot's Damiao bus with
MakerMods' Metal motor types (their LeRobot fork at `5ce2fe4`, `lerobot/motors/damiao/`), without the LeRobot base
classes, and the request matching of MakerMods' gripper wrapper (MakerMods Lab `f3ee37c`,
`makermodslab/metal_gripper.py`, query() here). Kept: the codec, the state query with its reply wait, the batched
refresh, single and batched MIT frames, enable and disable. Every wait is the source's loop with the source's numbers,
so the frames and their timing are the source's; `tests/test_metal_parity.py` holds the port to the source's frames
byte for byte.

The wire:

- a command frame is `arbitration_id = motor id`, standard 11-bit, 8 data bytes; `FF*7 FC` enables a motor and
  `FF*7 FD` disables it;
- a state query (a "refresh") and a register read or write go to #PARAM_ID: `[id lo, id hi, 0xCC, 0, 0, 0, 0, 0]`,
  or `<HBB` (id, #OP_READ or #OP_WRITE, register) and the 4-byte value;
- the MIT command packs `[q16, dq12, kp12, kd12, tau12]`, position first;
- motor `i` replies from `0x10 + i` (#REPLY_OFFSET). A state frame carries `status << 4 | id` in byte 0 (status 0
  disabled, 1 enabled, 8 to 14 a fault), then `[q16, dq12, tau12]`, then the MOSFET's and the rotor's temperatures in
  degrees C; a register reply echoes the request's first four bytes and carries the value.

Units at this module's boundary are degrees, degrees per second and N.m, as in the source; radians are internal.

@see DamiaoChain, DamiaoBus, encode_mit(), decode_state()
"""

import dataclasses
import logging
import math
import struct
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from robots import errors, motor_arm, robstride

if TYPE_CHECKING:
    # Annotations only: the Metal gripper drives its motor through this module's bus.
    from robots import metal_gripper

## Command byte (`data[7]` of an `FF*7 xx` frame): energize the motor.
CMD_ENABLE = 0xFC
## Command byte: de-energize the motor.
CMD_DISABLE = 0xFD
## Third byte of a state query to #PARAM_ID.
CMD_REFRESH = 0xCC
## Arbitration id of state queries and register access.
PARAM_ID = 0x7FF
## Register access operation: read.
OP_READ = 0x33
## Register access operation: write (volatile; nothing here saves to flash).
OP_WRITE = 0x55
## Register access operation: save to flash; never sent, only recognized in replies.
_OP_SAVE = 0xAA
## Motor `i` replies from `REPLY_OFFSET + i`.
REPLY_OFFSET = 0x10

## Status nibble of a state frame: de-energized.
STATUS_DISABLED = 0
## Status nibble: energized.
STATUS_ENABLED = 1
## Status nibbles of a fault (overvoltage, undervoltage, overcurrent, overheating, communication loss, overload).
FAULT_STATUSES = frozenset(range(8, 15))
FAULT_TEXT: dict[int, str] = {
    0x8: "overvoltage (the supply is too high, or a joint was pushed fast by hand and fed power back): check the "
    "arm's power supply, then switch the arm off and on with the arm supported",
    0x9: "undervoltage (the supply sagged or its cable is loose): check the arm's power supply and its cable, then "
    "switch the arm off and on with the arm supported",
    0xA: "overcurrent (the joint was blocked or pushed hard): clear whatever blocks it, then switch the arm off and "
    "on with the arm supported",
    0xB: "driver over-temperature (the motor's driver board is too hot): switch the arm off and let it cool for 15 "
    "minutes",
    0xC: "coil over-temperature (the motor's winding is too hot, usually from pushing against something or holding a "
    "load too long): switch the arm off and let it cool for 15 minutes, then check nothing blocks that joint",
    0xD: "lost contact with the computer (no command for too long: the program stopped, or the CAN cable dropped): "
    "check the CAN cable, then switch the arm off and on with the arm supported",
    0xE: "overload (it held too much torque for too long): clear whatever loads the joint, then switch the arm off "
    "and on with the arm supported",
}
## Every status nibble a state frame may carry.
_KNOWN_STATUSES = frozenset({STATUS_DISABLED, STATUS_ENABLED}) | FAULT_STATUSES

## Register: the CAN timeout, in 50 us units; with no frame for that long the motor disables itself. 0 is off.
REG_TIMEOUT = 9
## Register: the control mode (#MODE_MIT, #MODE_FORCE_POSITION).
REG_CTRL_MODE = 10
## Registers: the MIT scaling `PMAX` (rad), `VMAX` (rad/s), `TMAX` (N.m), floats; a frame means what they say.
REG_PMAX, REG_VMAX, REG_TMAX = 21, 22, 23
## Control mode: MIT.
MODE_MIT = 1
## Control mode: force-position (MakerMods' current-limited gripper mode).
MODE_FORCE_POSITION = 4
## Arbitration offset of an `FF*7 xx` command per control mode (MIT, position-velocity, velocity, force-position).
MODE_OFFSETS = {1: 0x000, 2: 0x100, 3: 0x200, 4: 0x300}

## `(PMAX rad, VMAX rad/s, TMAX N.m)` per motor model, the fixed-point scaling of every frame: MakerMods flash each
## Metal motor with the ranges of its joint.
MOTOR_LIMITS: dict[str, tuple[float, float, float]] = {
    "metal_jlo": (6.28, 10.0, 30.0),
    "metal_j2": (6.28, 10.0, 120.0),
    "metal_jhi": (6.28, 30.0, 20.0),
}
## The MIT frame's `kp` range, N.m per rad; 12 bits over it.
KP_RANGE = (0.0, 500.0)
## The MIT frame's `kd` range, N.m s per rad; 12 bits over it.
KD_RANGE = (0.0, 5.0)

## How long a state query, an MIT frame or an enable or disable waits for its reply, seconds: MakerMods' console
## patches the source bus's 1 ms to this.
REPLY_TIMEOUT_S = 0.05
## How long the batched refresh collects its replies, seconds.
BATCH_REFRESH_TIMEOUT_S = 0.01
## How long a batch of MIT frames collects its replies, seconds. The source measured 9% of replies lost at 1 ms on a
## Metal arm at 200 Hz, and 0.02% at this.
MIT_BATCH_TIMEOUT_S = 0.005
## Receive poll granularity, seconds.
POLL_S = 0.0001

# query(): the gripper motor answers late, see query()

## One attempt's reply wait, seconds.
QUERY_TIMEOUT_S = 0.12
## Attempts (sends of the request) per query.
QUERY_ATTEMPTS = 2
## Bus silence after which a query puts a flush frame on the bus, seconds: a punctual reply arrives well inside it, a
## held one needs the next frame.
FLUSH_AFTER_S = 0.003
## Flush frames per attempt.
FLUSHES_PER_ATTEMPT = 3
## Frames a query drops at most before its request: they answer earlier requests.
DRAIN_LIMIT = 64
## Seconds a frame may wait in the adapter's send queue.
_SEND_TIMEOUT_S = 0.1

# The arm (#DamiaoChain), MakerMods' follower and console

## How far past a soft limit a motor may read when the arm opens, degrees (MakerMods' console): at rest with torque
## off, the Metal arm's shoulder lift and elbow sit on their 0 limits and settle a degree or two past them.
LIMIT_TOLERANCE_DEG = 5.0
## The startup sync: a joint moves at most this per command from its measured angle, degrees,
SYNC_STEP_DEG = 1.0
## ... until it is this close to its target, degrees.
SYNC_TOL_DEG = 3.0
## A syncing joint that moves less than this, degrees,
SYNC_PROGRESS_DEG = 0.2
## ... for this long is let go to the full rate, seconds (parked past a limit, or stuck above `kp x step`).
SYNC_STALL_S = 1.0
## The velocity feedforward's filter weight on each new finite difference.
FF_ALPHA = 0.04
## The velocity feedforward's ceiling, degrees per second.
FF_MAX_DEG_S = 120.0
## Commands further apart than this restart the feedforward from zero, seconds.
FF_RESET_S = 0.5
## The feedforward's time step is clamped to this range, seconds.
FF_DT_RANGE_S = (0.005, 0.1)
## Tries per joint of a disable that is not confirmed: leaving a motor energized is not acceptable.
DISABLE_RETRIES = 5

_log = logging.getLogger(__name__)


def encode_mit(
    model: str, kp: float, kd: float, position_deg: float, velocity_deg_s: float, torque_nm: float
) -> bytes:
    """The 8-byte MIT command payload for one motor.

    @param model The motor model, a key of #MOTOR_LIMITS.
    @param kp Position gain, N.m per rad.
    @param kd Velocity gain, N.m s per rad.
    @param position_deg The target, degrees.
    @param velocity_deg_s The target velocity (the feedforward), degrees per second.
    @param torque_nm Feedforward torque, N.m.
    @return The payload.
    """
    pmax, vmax, tmax = MOTOR_LIMITS[model]
    kp_u = robstride.float_to_uint(kp, *KP_RANGE, 12)
    kd_u = robstride.float_to_uint(kd, *KD_RANGE, 12)
    q = robstride.float_to_uint(math.radians(position_deg), -pmax, pmax, 16)
    dq = robstride.float_to_uint(math.radians(velocity_deg_s), -vmax, vmax, 12)
    tau = robstride.float_to_uint(torque_nm, -tmax, tmax, 12)
    return bytes(
        [
            (q >> 8) & 0xFF,
            q & 0xFF,
            dq >> 4,
            ((dq & 0xF) << 4) | ((kp_u >> 8) & 0xF),
            kp_u & 0xFF,
            kd_u >> 4,
            ((kd_u & 0xF) << 4) | ((tau >> 8) & 0xF),
            tau & 0xFF,
        ]
    )


def refresh_payload(motor_id: int) -> bytes:
    """The payload of a state query to #PARAM_ID.

    @param motor_id The motor's CAN id.
    @return `[id lo, id hi, 0xCC, 0, 0, 0, 0, 0]`.
    """
    return bytes([motor_id & 0xFF, (motor_id >> 8) & 0xFF, CMD_REFRESH, 0, 0, 0, 0, 0])


@dataclasses.dataclass(frozen=True)
class MotorState:
    """One decoded state frame.

    Attributes:
    - status: The status nibble (#STATUS_DISABLED, #STATUS_ENABLED, one of #FAULT_STATUSES).
    - position_deg: Position, degrees, in the motor's own frame (its stored zero).
    - velocity_deg_s: Velocity, degrees per second.
    - torque_nm: Torque, N.m.
    - temp_mos_c: The driver MOSFET's temperature, degrees C.
    - temp_rotor_c: The rotor's temperature, degrees C.
    - t: The bus clock's reading when the frame was decoded, seconds.
    """

    status: int
    position_deg: float
    velocity_deg_s: float
    torque_nm: float
    temp_mos_c: float
    temp_rotor_c: float
    t: float


def decode_state(data: bytes, model: str, t: float) -> MotorState:
    """Decodes a state frame: byte 0 the status and id, then q16, dq12, tau12 and the two temperatures.

    @param data The frame's payload.
    @param model The motor model, a key of #MOTOR_LIMITS.
    @param t The bus clock's reading now, seconds.
    @return The state.
    @throws ValueError The payload is shorter than 8 bytes.
    """
    if len(data) < 8:
        raise ValueError(f"state frame needs 8 bytes, got {len(data)}")
    pmax, vmax, tmax = MOTOR_LIMITS[model]
    q = (data[1] << 8) | data[2]
    dq = (data[3] << 4) | (data[4] >> 4)
    tau = ((data[4] & 0x0F) << 8) | data[5]
    return MotorState(
        status=data[0] >> 4,
        position_deg=math.degrees(robstride.uint_to_float(q, -pmax, pmax, 16)),
        velocity_deg_s=math.degrees(robstride.uint_to_float(dq, -vmax, vmax, 12)),
        torque_nm=robstride.uint_to_float(tau, -tmax, tmax, 12),
        temp_mos_c=float(data[6]),
        temp_rotor_c=float(data[7]),
        t=t,
    )


def is_state_frame(data: bytes, motor_id: int) -> bool:
    """Whether a reply is a state frame of `motor_id` and not a register reply, which shares its arbitration id.

    @param data The frame's payload, 8 bytes.
    @param motor_id The motor's CAN id.
    @return True for the motor's id in the low nibble, a known status in the high one, and no register operation in
        bytes 1 and 2.
    """
    return (
        data[0] & 0x0F == motor_id
        and data[0] >> 4 in _KNOWN_STATUSES
        and not (data[1] == 0 and data[2] in (OP_READ, OP_WRITE, _OP_SAVE))
    )


class MotorSilent(errors.ArmError):
    """A motor did not answer within its reply wait."""


class StartupPose(errors.ArmError):
    """The arm refuses to open: a motor reads outside its soft limits, or is still energized or faulted."""


@dataclasses.dataclass
class Motor:
    """One Damiao motor on the bus.

    Attributes:
    - name: The joint it drives, for messages.
    - can_id: Its CAN id; it replies from `REPLY_OFFSET + can_id`.
    - model: Its model, a key of #MOTOR_LIMITS.
    - state: Its last decoded state, or None before the first.
    - drops: Replies that never came since the bus was built.
    """

    name: str
    can_id: int
    model: str
    state: MotorState | None = None
    drops: int = 0

    @property
    def reply_id(self) -> int:
        """The arbitration id its replies come from."""
        return REPLY_OFFSET + self.can_id


class DamiaoBus:
    """Seven Damiao motors on one CAN adapter, as the source's bus drives them.

    Every public method takes the bus's lock, since one thread drives an arm but another may close it.

    Attributes:
    - channel: The adapter, as #robots.robstride.open_can_bus takes it.
    - motors: The motors by name, in joint order.

    @see open(), close(), read(), refresh_all(), mit(), mit_batch(), command(), query()
    """

    def __init__(
        self, channel: str, motors: list[Motor], *, opener: Callable[[str], Any], clock: Callable[[], float]
    ) -> None:
        """Binds the motors to an adapter; nothing is opened.

        @param channel The adapter.
        @param motors The motors, base to gripper.
        @param opener Opens the adapter and returns a bus with python-can's `send`, `recv` and `shutdown`
            (#robots.robstride.open_can_bus in production, a fake bus in tests).
        @param clock Monotonic clock, seconds: every reply wait.
        @throws ValueError Two motors share a CAN id.
        """
        self.channel = channel
        self.motors = {motor.name: motor for motor in motors}
        if len({motor.can_id for motor in motors}) != len(motors):
            raise ValueError(f"duplicate CAN ids on {channel}: {[motor.can_id for motor in motors]}")
        self._opener = opener
        self._clock = clock
        self._bus: Any = None
        self._lock = threading.RLock()

    def open(self) -> None:
        """Opens the adapter; sends nothing (the source connects without its handshake, which would energize).

        @throws Exception Whatever the opener raises.
        """
        with self._lock:
            self._bus = self._opener(self.channel)

    def close(self) -> None:
        """Shuts the adapter down; a failure is logged, and an slcan port is still closed under it."""
        with self._lock:
            bus, self._bus = self._bus, None
            if bus is None:
                return
            try:
                bus.shutdown()
            except Exception as exc:
                _log.warning("%s: adapter shutdown failed: %s", self.channel, exc)
                # python-can's slcan shutdown writes "C" first; when that write fails it never reaches the serial
                # close, and the port would stay open under the next session's.
                serial_port = getattr(bus, "serialPortOrig", None)
                if serial_port is not None:
                    serial_port.close()

    @property
    def is_open(self) -> bool:
        """Whether the adapter is open."""
        return self._bus is not None

    def _require(self) -> Any:
        """The open adapter, or ArmError when it is closed."""
        if self._bus is None:
            raise errors.ArmError(f"{self.channel}: the CAN adapter is not open")
        return self._bus

    def _send(self, arbitration_id: int, data: bytes) -> None:
        """Sends one standard data frame."""
        message = robstride.python_can().Message(arbitration_id=arbitration_id, data=data, is_extended_id=False)
        self._require().send(message, timeout=_SEND_TIMEOUT_S)

    def _decode(self, motor: Motor, data: bytes) -> MotorState:
        """Decodes a reply into the motor's state and stores it."""
        motor.state = decode_state(data, motor.model, self._clock())
        return motor.state

    def _recv_reply(self, reply_id: int, timeout: float) -> bytes | None:
        """The first frame from `reply_id` within `timeout`, polled every #POLL_S; other frames are dropped."""
        bus = self._require()
        start = self._clock()
        while self._clock() - start < timeout:
            message = bus.recv(timeout=POLL_S)
            if message is not None and message.arbitration_id == reply_id:
                return bytes(message.data)
        return None

    def _recv_all(self, reply_ids: list[int], timeout: float) -> dict[int, bytes]:
        """The frames from `reply_ids` that arrive within `timeout`, the last one per id; stops once each has one."""
        bus = self._require()
        responses: dict[int, bytes] = {}
        expected = set(reply_ids)
        start = self._clock()
        while len(responses) < len(reply_ids) and (self._clock() - start) < timeout:
            message = bus.recv(timeout=POLL_S)
            if message is not None and message.arbitration_id in expected:
                responses[message.arbitration_id] = bytes(message.data)
                if len(responses) == len(reply_ids):
                    break
        return responses

    def read(self, name: str) -> MotorState:
        """Queries one motor's state and waits #REPLY_TIMEOUT_S for its reply.

        @param name The motor.
        @return Its state, also stored as `Motor.state`.
        @throws MotorSilent No reply came (counted in `drops`).
        """
        with self._lock:
            motor = self.motors[name]
            self._send(PARAM_ID, refresh_payload(motor.can_id))
            data = self._recv_reply(motor.reply_id, REPLY_TIMEOUT_S)
            if data is None:
                motor.drops += 1
                raise MotorSilent(
                    f"{self.channel}: {name} (id {motor.can_id}, reply id 0x{motor.reply_id:02X}) did not answer; "
                    "check the arm's 24 V power and its CAN wiring"
                )
            return self._decode(motor, data)

    def refresh_all(self, names: list[str]) -> dict[str, float]:
        """Queries every motor at once and collects the replies for #BATCH_REFRESH_TIMEOUT_S.

        A motor that did not answer keeps its last state (counted in `drops`).

        @param names The motors.
        @return Each motor's position, degrees.
        @throws ArmError A motor has never answered.
        """
        with self._lock:
            motors = [self.motors[name] for name in names]
            for motor in motors:
                self._send(PARAM_ID, refresh_payload(motor.can_id))
            responses = self._recv_all([motor.reply_id for motor in motors], BATCH_REFRESH_TIMEOUT_S)
            positions: dict[str, float] = {}
            for motor in motors:
                data = responses.get(motor.reply_id)
                if data is not None:
                    self._decode(motor, data)
                else:
                    motor.drops += 1
                    _log.debug("%s: %s missed a refresh, keeping its last state", self.channel, motor.name)
                if motor.state is None:
                    raise errors.ArmError(
                        f"{self.channel}: {motor.name} has never answered; switch the arm's power on, release its "
                        "emergency stop and check the CAN cable to that motor"
                    )
                positions[motor.name] = motor.state.position_deg
            return positions

    def mit(
        self, name: str, kp: float, kd: float, position_deg: float, velocity_deg_s: float, torque_nm: float
    ) -> None:
        """Sends one MIT frame and waits #REPLY_TIMEOUT_S for the reply, which it folds into the motor's state.

        A missed reply is counted in `drops` and costs nothing else.

        @param name The motor.
        @param kp Position gain.
        @param kd Velocity gain.
        @param position_deg The target, degrees.
        @param velocity_deg_s The target velocity, degrees per second.
        @param torque_nm Feedforward torque, N.m.
        """
        with self._lock:
            motor = self.motors[name]
            self._send(motor.can_id, encode_mit(motor.model, kp, kd, position_deg, velocity_deg_s, torque_nm))
            data = self._recv_reply(motor.reply_id, REPLY_TIMEOUT_S)
            if data is None:
                motor.drops += 1
                return
            self._decode(motor, data)

    def mit_batch(self, commands: dict[str, tuple[float, float, float, float, float]]) -> None:
        """Sends one MIT frame per motor back to back, then collects the replies for #MIT_BATCH_TIMEOUT_S.

        Missed replies are counted in `drops`; the motors keep their last state.

        @param commands `(kp, kd, position deg, velocity deg/s, torque N.m)` per motor.
        """
        with self._lock:
            motors = [self.motors[name] for name in commands]
            for motor, (kp, kd, position, velocity, torque) in zip(motors, commands.values(), strict=True):
                self._send(motor.can_id, encode_mit(motor.model, kp, kd, position, velocity, torque))
            responses = self._recv_all([motor.reply_id for motor in motors], MIT_BATCH_TIMEOUT_S)
            for motor in motors:
                data = responses.get(motor.reply_id)
                if data is None:
                    motor.drops += 1
                else:
                    self._decode(motor, data)

    def command(self, name: str, command: int) -> MotorState | None:
        """Sends an `FF*7 xx` command (MIT mode) and waits #REPLY_TIMEOUT_S for the reply.

        @param name The motor.
        @param command #CMD_ENABLE or #CMD_DISABLE.
        @return The state the reply carries, or None when none came (counted in `drops`).
        """
        with self._lock:
            motor = self.motors[name]
            self._send(motor.can_id, b"\xff" * 7 + bytes([command]))
            data = self._recv_reply(motor.reply_id, REPLY_TIMEOUT_S)
            if data is None:
                motor.drops += 1
                return None
            return self._decode(motor, data)

    def query(
        self,
        name: str,
        arbitration_id: int,
        data: bytes,
        matches: Callable[[bytes], bool],
        want: Callable[[bytes], bool] | None = None,
    ) -> bytes:
        """Sends one request to a motor that may answer late, and returns the reply that answers it.

        MakerMods measured the Metal gripper (id 7) on two arms and two adapters: it often sends its reply to a request
        only when the next frame appears on the bus, sometimes twice, and occasionally never; motors 1 to 6 reply
        within 0.25 ms. So frames already queued, which answer earlier requests, are dropped first (at most
        #DRAIN_LIMIT); after #FLUSH_AFTER_S of bus silence a state query of another motor flushes a held reply
        (#FLUSHES_PER_ATTEMPT per attempt); the request is sent up to #QUERY_ATTEMPTS times, each waiting
        #QUERY_TIMEOUT_S. `matches` is the reply's identity (a register echo, a state frame); `want` narrows it to the
        content this request must produce, so a late duplicate of an earlier answer is skipped. A reply that matches
        but is not wanted is returned only when nothing better came, and the caller's own check then fails on it.

        @param name The motor; its replies come from its reply id.
        @param arbitration_id The request's arbitration id.
        @param data The request's payload.
        @param matches Whether a reply's payload is of the kind this request gets.
        @param want Whether a matching payload is the answer this request must produce; None takes any match.
        @return The reply's payload, 8 bytes.
        @throws MotorSilent No matching reply came (counted in `drops`).
        """
        with self._lock:
            bus = self._require()
            motor = self.motors[name]
            for _ in range(DRAIN_LIMIT):
                if bus.recv(timeout=0) is None:
                    break
            flush_id = next((other.can_id for other in self.motors.values() if other is not motor), motor.can_id)
            flush = struct.pack("<HB", flush_id, CMD_REFRESH) + bytes(5)
            unwanted: bytes | None = None
            for _ in range(QUERY_ATTEMPTS):
                self._send(arbitration_id, data)
                flushes = 0
                now = quiet_since = self._clock()
                deadline = now + QUERY_TIMEOUT_S
                while now < deadline:
                    if flushes < FLUSHES_PER_ATTEMPT and now - quiet_since >= FLUSH_AFTER_S:
                        self._send(PARAM_ID, flush)
                        flushes += 1
                        quiet_since = now
                    message = bus.recv(timeout=max(0.0, min(FLUSH_AFTER_S, deadline - now)))
                    now = self._clock()
                    if message is None:
                        continue
                    quiet_since = now  # any frame on the bus also releases a held reply
                    reply = bytes(message.data)
                    if (
                        not message.is_extended_id
                        and message.arbitration_id == motor.reply_id
                        and len(reply) == 8
                        and matches(reply)
                    ):
                        if want is None or want(reply):
                            return reply
                        unwanted = reply
            if unwanted is not None:
                return unwanted
            motor.drops += 1
            raise MotorSilent(
                f"{self.channel}: no fresh reply from {name} (id {motor.can_id}); check the arm's 24 V power and its "
                "CAN wiring"
            )

    def store(self, name: str, data: bytes) -> MotorState:
        """Decodes a state frame query() returned into the motor's state.

        @param name The motor.
        @param data The payload.
        @return The state, also stored as `Motor.state`.
        """
        with self._lock:
            return self._decode(self.motors[name], data)

    def drops(self) -> int:
        """Replies that never came, every motor together, since the bus was built.

        @return The count.
        """
        return sum(motor.drops for motor in self.motors.values())


class DamiaoChain:
    """One arm's Damiao motors on its adapter, as #robots.motor_arm.MotorArm drives them (a #robots.motor_arm.Chain).

    MakerMods' `MetalFollower` (their LeRobot fork at `5ce2fe4`) with their console's reads, and every check it makes:

    - open: the gripper's setup, every motor read, a pose more than #LIMIT_TOLERANCE_DEG outside the soft limits
      refused (#StartupPose), and every joint preloaded with a zero-gain MIT frame at its own angle, so enabling it
      holds it there; a refused open switches no joint off, since a powered one holds the arm;
    - every tick reads all seven motors, re-queries a motor once when it misses its reply, never serves a missed one
      from the cache, and requires every motor energized (read());
    - every command (send_action()): the soft limits, the gripper's included; the startup sync, which moves a joint at
      most #SYNC_STEP_DEG per command from its measured angle until it is within #SYNC_TOL_DEG of its target (a joint
      that makes no progress for #SYNC_STALL_S is let go); the velocity feedforward, a filtered finite difference of
      the commands; and one MIT frame per motor at the table's gains, the gripper's through its holding control.

    One departure: each joint's MIT frame carries the arm's gravity torque at this tick's pose
    (`gravity`), as MakerMods' SDK follower does (`metal_sdk`); their LeRobot
    follower sends 0, and the arm then sags onto the table and cannot lift off it.

    Attributes:
    - name: `"left"` or `"right"`, for messages.
    - names: The motors' names, base to gripper.
    - limits_deg: Soft limits per motor, degrees, the gripper's included.
    - transient: Empty.
    - holds_grip: True: the gripper holds an object at its hold torque (#robots.makers_hold).
    - alarm: Why the gripper stopped, empty while it runs.
    - bus: The arm's #DamiaoBus.
    - gripper: The arm's gripper (#robots.metal_gripper.MetalGripper).
    - enabled: Whether the joints are energized, or may be: set by enable(), cleared only once every joint confirmed
      its switch-off (relax(), close() with `disable_torque`).

    @see open(), enable(), read(), send_action(), rearm(), relax(), close()
    """

    transient: tuple[type[Exception], ...] = ()
    holds_grip = True

    def __init__(
        self,
        name: str,
        channel: str,
        joints: tuple[motor_arm.Joint, ...],
        gripper: "Callable[[DamiaoBus], metal_gripper.MetalGripper]",
        *,
        gravity: Callable[[tuple[float, ...]], tuple[float, ...]],
        opener: Callable[[str], Any],
        clock: Callable[[], float],
    ) -> None:
        """Builds the arm's bus and gripper; nothing is opened.

        @param name `"left"` or `"right"`, for messages.
        @param channel The arm's CAN adapter (#robots.robstride.open_can_bus).
        @param joints The arm's motors, base to gripper, the gripper last.
        @param gripper Builds the gripper on the arm's bus.
        @param gravity The joints' gravity torques, N.m, from their angles, degrees, base to wrist: each joint's MIT
            torque.
        @param opener Opens the adapter.
        @param clock Monotonic clock, seconds.
        """
        self.name = name
        self.names = tuple(joint.name for joint in joints)
        self.limits_deg = {joint.name: joint.limits_deg for joint in joints}
        self._gains = {joint.name: (joint.kp, joint.kd) for joint in joints}
        self._gravity = gravity
        self._joints = self.names[:-1]
        self._gripper = self.names[-1]
        self.bus = DamiaoBus(
            channel, [Motor(name=j.name, can_id=j.motor_id, model=j.model) for j in joints], opener=opener, clock=clock
        )
        self.gripper = gripper(self.bus)
        self._clock = clock
        self.alarm = ""
        self.enabled = False
        self._reading: motor_arm.MotorReading | None = None
        self._synced = False
        self._synced_motors: set[str] = set()
        self._sync_progress: dict[str, tuple[float, float]] = {}
        self._ff_previous: dict[str, float] | None = None
        self._ff_velocities: dict[str, float] = {}
        self._ff_time: float | None = None

    def open(self) -> None:
        """Opens the adapter, sets the gripper up, reads every motor, checks the pose, and preloads every joint.

        @throws StartupPose A motor reads more than #LIMIT_TOLERANCE_DEG outside its soft limits, or is energized or
            faulted.
        @throws GripperSafetyError The gripper's setup refused (#robots.metal_gripper.MetalGripper.connect).
        @throws MotorSilent A motor did not answer.
        """
        with robstride.adapter_words(self.name, self.bus.channel):
            self.bus.open()
            try:
                self.gripper.connect()
                for joint in self._joints:
                    self.bus.read(joint)
                reading = self.read()
                self._check_startup_pose(reading)
                # The joints are enabled next: clear their stored MIT targets first, at the angle each reads now.
                for joint, position in zip(self._joints, reading.q_deg, strict=False):
                    self.gripper.check()
                    self.bus.mit(joint, 0.0, 0.0, position, 0.0, 0.0)
            except MotorSilent as exc:
                self._close_after_failure()
                raise MotorSilent(f"arm '{self.name}' did not open: {exc}") from exc
            except BaseException:
                self._close_after_failure()
                raise
        _log.info("arm '%s': open on %s, %d motors", self.name, self.bus.channel, len(self.names))

    def _check_startup_pose(self, reading: motor_arm.MotorReading) -> None:
        """Refuses a motor outside its soft limits by more than #LIMIT_TOLERANCE_DEG, energized, or faulted.

        An energized joint would go limp under the zero-gain preload, so an arm left energized by an earlier session
        (closed without torque off) is refused rather than dropped, and left powered for the operator to support.
        """
        problems: list[str] = []
        for joint, position in zip(self.names, reading.q_deg, strict=True):
            status = self._state(joint).status
            if status in FAULT_STATUSES:
                problems.append(f"{joint} is in fault 0x{status:X}, {FAULT_TEXT[status]}")
            elif status == STATUS_ENABLED:
                problems.append(
                    f"{joint} is still powered from an earlier session, holding the arm with nothing driving it: "
                    "support the arm, then switch its power off and on"
                )
            low, high = self.limits_deg[joint]
            if not low - LIMIT_TOLERANCE_DEG <= position <= high + LIMIT_TOLERANCE_DEG:
                problems.append(
                    f"{joint} reads {position:+.1f} deg, outside its range {low:g} to {high:g}: if the arm really "
                    "sits inside its range, that motor's zero has moved (a reseated horn, a motor swapped); zero it "
                    "again with MakerMods' calibration"
                )
        if problems:
            raise StartupPose(f"arm '{self.name}' refuses to start: {'; '.join(problems)}.")

    def _close_after_failure(self) -> None:
        """Closes after a failed open, the gripper restored and every joint left as it is: open() energizes none, so a
        powered one holds the arm for an earlier session, and switching it off would drop it. A failure of the close is
        logged, the original raised."""
        try:
            self.close(disable_torque=False)
        except Exception as exc:
            _log.warning("arm '%s': cleanup after a failed open: %s", self.name, exc)

    def _state(self, joint: str) -> MotorState:
        """A motor's last state; every motor has one once the arm is open."""
        state = self.bus.motors[joint].state
        if state is None:
            raise errors.ArmError(f"arm '{self.name}': {joint} has no state")
        return state

    def enable(self) -> None:
        """Energizes the joints, then the gripper (preloaded, then enabled); each holds where it was preloaded.

        @throws GripperSafetyError The gripper did not enable; it is tripped.
        """
        self.gripper.check()
        for joint in self._joints:
            self.bus.command(joint, CMD_ENABLE)
        self.gripper.enable()
        self.enabled = True
        self.rearm()

    def rearm(self) -> None:
        """Re-arms the startup sync and restarts the feedforward: the next command eases in from where the arm is."""
        self._synced = False
        self._synced_motors = set()
        self._sync_progress = {}
        self._ff_previous = None
        self._ff_velocities = {}
        self._ff_time = None

    def read(self) -> motor_arm.MotorReading:
        """Reads every motor; a missed reply is asked once more, never served from the cache.

        @return The reading, every motor's two temperatures included.
        """
        self._gripper_checked(age=False)
        for joint in self._joints:
            try:
                self.bus.read(joint)
            except MotorSilent:
                # One fresh re-query tolerates a dropped USB-CAN reply; a second miss stops the arm.
                self.bus.read(joint)
        self._refresh_gripper()
        states = [self._state(joint) for joint in self.names]
        q = tuple(state.position_deg for state in states)
        if not all(math.isfinite(x) for x in q):
            raise errors.ArmError(f"arm '{self.name}': non-finite motor feedback {q}")
        if self.enabled:
            for joint, state in zip(self._joints, states, strict=False):
                if state.status != STATUS_ENABLED:
                    detail = (
                        "switched itself off (lost power or contact): check the arm's power and its CAN cable"
                        if state.status not in FAULT_STATUSES
                        else f"in fault 0x{state.status:X}, {FAULT_TEXT[state.status]}"
                    )
                    raise errors.ArmError(f"arm '{self.name}' {joint}: motor {detail}; then reopen the arm")
        self._reading = motor_arm.MotorReading(
            q_deg=q,
            velocity_deg_s=tuple(state.velocity_deg_s for state in states),
            torque_nm=tuple(state.torque_nm for state in states),
            temperatures_c=tuple({"coil": state.temp_rotor_c, "driver board": state.temp_mos_c} for state in states),
        )
        return self._reading

    def _gripper_checked(self, *, age: bool) -> bool:
        if self.alarm:
            return False
        try:
            self.gripper.check(age=age)
        except errors.ArmError as exc:
            self._stop_gripper(exc)
            return False
        return True

    def _refresh_gripper(self) -> None:
        try:
            self.gripper.refresh()
        except MotorSilent as exc:
            if not self.alarm:
                self._stop_gripper(exc)
            if self.bus.motors[self._gripper].state is None:
                raise
            return
        self._gripper_checked(age=False)

    def _stop_gripper(self, exc: BaseException) -> None:
        if not self.gripper.error:
            self.gripper.trip(exc)
        self.alarm = f"the gripper stopped: {exc}"
        _log.error("arm '%s': %s; the joints hold and the run must stop", self.name, self.alarm)

    def command(self, goal_deg: list[float]) -> motor_arm.MotorReading:
        """send_action() on a goal per motor, base to gripper.

        @param goal_deg The target per motor, degrees, base to gripper.
        @return This tick's read(): the pose published is the one the tick started from.
        """
        assert self._reading is not None, "a tick reads before it commands"
        self.send_action(dict(zip(self.names, goal_deg, strict=True)))
        return self._reading

    def send_action(self, goal_deg: dict[str, float]) -> dict[str, float]:
        """One command: clamped, synced, with its velocity feedforward and gravity torque, one MIT frame per motor.

        @param goal_deg The target per motor, degrees, every name of #names.
        @return The goal sent per motor, degrees (before the gripper's holding law).
        """
        goal: dict[str, float] = {}
        for joint in self.names:
            low, high = self.limits_deg[joint]
            goal[joint] = max(low, min(high, float(goal_deg[joint])))
        if not self._synced:
            self._sync(goal, self.bus.refresh_all(list(self.names)))
        velocities = self._feedforward(goal)
        assert self._reading is not None, "a tick reads before it commands"
        torques = dict(zip(self._joints, self._gravity(self._reading.q_deg[: len(self._joints)]), strict=True))
        if self._gripper_checked(age=True):
            try:
                self.gripper.write(goal[self._gripper], velocities[self._gripper], 0.0)
            except errors.ArmError as exc:
                self._stop_gripper(exc)
        try:
            self.bus.mit_batch(
                {
                    joint: (*self._gains[joint], goal[joint], velocities[joint], torques[joint])
                    for joint in self._joints
                }
            )
        except Exception as exc:
            if not self.gripper.error:
                self.gripper.trip(exc)
            raise
        return goal

    def _sync(self, goal: dict[str, float], present: dict[str, float]) -> None:
        """The startup sync, in place on `goal`: each joint not yet synced moves at most #SYNC_STEP_DEG from its
        measured angle, and is synced once within #SYNC_TOL_DEG of its goal or after #SYNC_STALL_S without progress."""
        now = self._clock()
        for joint, position in goal.items():
            if joint in self._synced_motors:
                continue
            measured = present[joint]
            err = position - measured
            if abs(err) <= SYNC_TOL_DEG:
                self._synced_motors.add(joint)
                continue
            last_position, last_progress = self._sync_progress.get(joint, (measured, now))
            if abs(measured - last_position) > SYNC_PROGRESS_DEG:
                self._sync_progress[joint] = (measured, now)
            elif now - last_progress > SYNC_STALL_S:
                _log.warning(
                    "arm '%s': startup sync stalled on %s (%.1f deg from its goal); letting it go to full rate",
                    self.name,
                    joint,
                    abs(err),
                )
                self._synced_motors.add(joint)
                continue
            else:
                self._sync_progress.setdefault(joint, (measured, now))
            goal[joint] = measured + max(-SYNC_STEP_DEG, min(SYNC_STEP_DEG, err))
        if len(self._synced_motors) >= len(goal):
            self._synced = True
            _log.info("arm '%s': synced, tracking at full rate", self.name)

    def _feedforward(self, goal: dict[str, float]) -> dict[str, float]:
        """The velocity feedforward per motor: the commands' finite difference, clamped to #FF_MAX_DEG_S and filtered
        (#FF_ALPHA); zero after a gap of #FF_RESET_S."""
        now = self._clock()
        if self._ff_time is None or now - self._ff_time > FF_RESET_S:
            self._ff_previous, self._ff_velocities, self._ff_time = None, {}, None
        if self._ff_previous is None or self._ff_time is None:
            velocities = dict.fromkeys(goal, 0.0)
        else:
            dt = max(FF_DT_RANGE_S[0], min(FF_DT_RANGE_S[1], now - self._ff_time))
            velocities = {}
            for joint, position in goal.items():
                raw = max(-FF_MAX_DEG_S, min(FF_MAX_DEG_S, (position - self._ff_previous[joint]) / dt))
                velocities[joint] = (1.0 - FF_ALPHA) * self._ff_velocities[joint] + FF_ALPHA * raw
        self._ff_previous = dict(goal)
        self._ff_velocities = velocities
        self._ff_time = now
        return dict(velocities)

    def close(self, *, disable_torque: bool) -> None:
        """Closes the adapter; the gripper is released either way, the joints switched off when asked.

        The gripper is disabled and its mode and CAN timeout restored whatever the joints do (MakerMods': a gripper
        left energized would lose its object to its own timeout half a second later anyway). Every step is tried; the
        adapter is closed last. A close without the switch-off leaves the joints #enabled, holding their last target
        with nothing driving them, so a later close that is asked to switch them off opens the adapter once more to do
        it (as `robots.robstride.RobstrideBus.close` does).

        @param disable_torque Disable every joint too, asking one that does not confirm again (#DISABLE_RETRIES).
        """
        if not self.bus.is_open:
            if not (disable_torque and self.enabled):
                return
            try:
                self.bus.open()
            except Exception as exc:
                raise errors.ArmError(
                    f"arm '{self.name}': torque off not verified: the joints are still powered and the CAN adapter "
                    f"{self.bus.channel} did not open again to switch them off ({exc}); support the arm, then switch "
                    "its power off"
                ) from exc
        problems: list[str] = []
        try:
            if disable_torque:
                for joint in self._joints:
                    if not self._disable(joint):
                        problems.append(f"{joint} did not confirm its disable")
                self.enabled = bool(problems)
            problems += self.gripper.release()
        finally:
            self.bus.close()
        if problems:
            raise errors.ArmError(f"arm '{self.name}': torque off not verified: {'; '.join(problems)}")

    def relax(self) -> None:
        """Switches every joint and the gripper off and keeps the adapter open; read() goes on.

        @throws ArmError A motor did not confirm its switch-off.
        """
        problems = [f"{joint} did not confirm its disable" for joint in self._joints if not self._disable(joint)]
        self.enabled = bool(problems)
        problems += self.gripper.release()
        if problems:
            raise errors.ArmError(f"arm '{self.name}': limp not verified: {'; '.join(problems)}")

    def _disable(self, joint: str) -> bool:
        """Disables one joint; True once a reply says it is not energized."""
        for _ in range(DISABLE_RETRIES + 1):
            state = self.bus.command(joint, CMD_DISABLE)
            if state is not None and state.status != STATUS_ENABLED:
                return True
        return False

    def drops(self) -> int:
        """Replies that never came, every motor together.

        @return The count.
        """
        return self.bus.drops()
