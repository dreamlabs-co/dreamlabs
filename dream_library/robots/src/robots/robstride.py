"""RobStride motors in MIT mode on one CAN adapter: the frame codec and a request/reply bus; the only module that
knows python-can.

A Makers Arm (#robots.makers) has seven RobStride motors, six joints and the gripper, on one classic CAN bus at
1 Mbit/s per arm, addressed by CAN id 1 to 7 and driven in MIT position mode. Both arms of a bimanual station answer
to the same ids, which is why each arm has its own adapter and a bus is chosen by adapter, never by motor id.

Ported from the Maker Arm's reference driver (its RobStride module, cross-checked there against the vendor's driver),
with what a policy run needs and nothing else: the codec, the handshake, state reads, position
commands, torque on and off.
`tests/test_makers_parity.py` holds the port to that driver's frames byte for byte.

The wire (RS00/RS02 manual, MIT mode):

- a command frame is `arbitration_id = motor id`, standard 11-bit, 8 data bytes;
- `FF*7 FC` enables a motor, `FF*7 FD` disables it, `FF*7 FB` clears a fault and doubles as the state query (the
  reply is a feedback frame);
- the MIT command packs `[q16, dq12, kp12, kd12, tau12]`, position first, while the feedback frame carries the
  motor id in byte 0 and then `[q16, dq12, tau12, temp16]`; the asymmetry is easy to get wrong, so both codecs live
  here;
- a reply's arbitration id is the host id (#HOST_ID), so replies are told apart by payload byte 0.

Units at this module's boundary are degrees, degrees per second and N.m; the radians are internal. The gains ride
in every MIT frame, so `kp` and `kd` are per-motor state on the bus, not a register write.

The adapter (open_can_bus()) is a SocketCAN link (a candleLight under the kernel's gs_usb driver, brought up at
1 Mbit/s the way `dreamcli robot up` brings a YAM's links up) or a serial device path (a CANable on slcan firmware,
named by its `/dev/serial/by-id/...` path, which carries its serial). A CANable is opened directly over its serial
device, with no root and no `slcand`, so it comes back by itself after a replug; one an `slcand` already holds is
opened through the kernel link that `slcand` made of it (slcan_holder()).
The source's raw-USB candleLight transport (`gs_usb:<serial>`) is not ported: on Linux the kernel driver already
makes such an adapter a SocketCAN link. python-can is imported on first use, so the package imports without the
`makers` extra.

@see RobstrideChain, RobstrideBus, open_can_bus(), encode_mit(), decode_feedback()
"""

import contextlib
import dataclasses
import logging
import math
import os
import pathlib
import threading
from collections.abc import Callable, Iterator
from typing import Any

from robots import errors, makers_hold, motor_arm

## MIT-mode command byte (`data[7]` of an `FF*7 xx` frame): energize the motor.
CMD_ENABLE = 0xFC
## MIT-mode command byte: de-energize the motor.
CMD_DISABLE = 0xFD
## MIT-mode command byte: clear a fault; also the state query, whose reply is a feedback frame.
CMD_CLEAR_FAULT = 0xFB

## Arbitration id every RobStride motor replies from (the host id).
HOST_ID = 0xFD

## `(PMAX rad, VMAX rad/s, TMAX N.m)` per motor model, the fixed-point scaling of every frame; RS00 is `"O0"`,
## RS02 `"O1"` in the vendor's naming.
MOTOR_LIMITS: dict[str, tuple[float, float, float]] = {
    "O0": (12.57, 33.0, 14.0),
    "O1": (12.57, 44.0, 17.0),
}

## The MIT frame's `kp` range, N.m per rad; 12 bits over it.
KP_RANGE = (0.0, 500.0)
## The MIT frame's `kd` range, N.m s per rad; 12 bits over it.
KD_RANGE = (0.0, 5.0)

## How long one state query waits for its reply, seconds (the vendor's RUNNING_TIMEOUT).
QUERY_TIMEOUT_S = 0.003
## Per-motor reply wait after an MIT command, seconds: a ceiling, not a delay, since the wait ends when the reply
## lands. 5 ms where the vendor has 1 ms: a reply can take 1.4-2.8 ms instead of 0.5 ms after a re-plug, a 1 ms
## ceiling then sends the seven frames of a tick back to back, and the CANable stops taking writes (the arm
## freezes mid-move). One frame in flight at a time on any USB topology.
COMMAND_REPLY_TIMEOUT_S = 0.005
## Handshake: a motor silent this long is missing, seconds.
HANDSHAKE_TIMEOUT_S = 0.05
## Pause after each motor's handshake query, seconds.
HANDSHAKE_GAP_S = 0.01
## Receive poll granularity, seconds.
_POLL_S = 0.0003
## A reading younger than this is served from the cache rather than queried again, seconds, so an observation and a
## command in the same tick share one bus round trip.
STATE_CACHE_TTL_S = 0.02
## Frames flush_rx() drops at most when opening, the leftovers of a previous session.
_FLUSH_MAX_FRAMES = 4096
## Tries per motor of a disable: leaving a motor energized is not acceptable, so a silent one is asked again.
DISABLE_RETRIES = 5
## Tries per motor of an enable: some firmware does not answer it, and the handshake already proved the motor is there.
ENABLE_RETRIES = 2
## Seconds between two tries of an enable or a disable.
_RETRY_GAP_S = 0.01
## How long an enable waits for its reply, seconds.
_ENABLE_REPLY_TIMEOUT_S = 0.01
## Seconds a frame may wait in the adapter's send queue.
_SEND_TIMEOUT_S = 0.1
## The classic CAN bit rate every Makers Arm bus runs at.
BITRATE = 1_000_000
## A channel that starts with this is a serial device (slcan); anything else is a SocketCAN link name.
SERIAL_DEVICE_PREFIX = "/dev/"
PROC_DIR = pathlib.Path("/proc")
NET_DIR = pathlib.Path("/sys/class/net")
IFF_UP = 0x1
_SLCAND_VALUE_OPTIONS = frozenset({"-s", "-S", "-t", "-b"})
## Every motor of an arm unanswered this long latches the arm down, seconds (arm power, e-stop, CAN cable, adapter).
ARM_SILENT_S = 0.3
## One motor unanswered this long is named in the log, seconds.
JOINT_SILENT_S = 1.0
## A whole motor turn, degrees.
_FULL_TURN_DEG = 360.0
## Widens the wrap test, degrees, so a joint parked just past a limit is not taken for a wrapped one.
_WRAP_GRACE_DEG = 20.0

_log = logging.getLogger(__name__)


def float_to_uint(x: float, x_min: float, x_max: float, bits: int) -> int:
    """Clamps `x` to `[x_min, x_max]` and quantizes it onto `bits` bits, truncating.

    @param x The value.
    @param x_min The low end of the range.
    @param x_max The high end of the range.
    @param bits The field's width.
    @return The unsigned field value.
    """
    x = max(x_min, min(x_max, x))
    return int((x - x_min) / (x_max - x_min) * ((1 << bits) - 1))


def uint_to_float(x: int, x_min: float, x_max: float, bits: int) -> float:
    """The inverse of float_to_uint(), up to its quantization.

    @param x The unsigned field value.
    @param x_min The low end of the range.
    @param x_max The high end of the range.
    @param bits The field's width.
    @return The value.
    """
    return x / ((1 << bits) - 1) * (x_max - x_min) + x_min


def encode_mit(model: str, kp: float, kd: float, position_deg: float) -> bytes:
    """The 8-byte MIT command payload for one motor: a position target at zero velocity and zero feed-forward torque.

    @param model The motor model, a key of #MOTOR_LIMITS.
    @param kp Position gain, N.m per rad.
    @param kd Velocity gain, N.m s per rad.
    @param position_deg The target, degrees.
    @return The payload.
    """
    pmax, vmax, tmax = MOTOR_LIMITS[model]
    q = float_to_uint(math.radians(position_deg), -pmax, pmax, 16)
    dq = float_to_uint(0.0, -vmax, vmax, 12)
    kp_u = float_to_uint(kp, *KP_RANGE, 12)
    kd_u = float_to_uint(kd, *KD_RANGE, 12)
    tau = float_to_uint(0.0, -tmax, tmax, 12)
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


@dataclasses.dataclass(frozen=True)
class MotorState:
    """One decoded feedback frame.

    Attributes:
    - motor_id: The motor's CAN id, payload byte 0.
    - position_deg: Position, degrees, in the motor's own frame (its stored zero).
    - velocity_deg_s: Velocity, degrees per second.
    - torque_nm: Torque, N.m.
    - t: The bus clock's reading when the frame was decoded, seconds.
    - fault_flags: Byte 6 bits 5..4, the motor's fault and warning flags; non-zero only on firmware that packs them.
    - temp_c: The motor driver's temperature, degrees C.
    """

    motor_id: int
    position_deg: float
    velocity_deg_s: float
    torque_nm: float
    t: float
    fault_flags: int
    temp_c: float


def decode_feedback(data: bytes, model: str, t: float) -> MotorState:
    """Decodes a feedback frame: byte 0 the motor id, then q16, dq12, tau12 and the temperature word.

    @param data The frame's payload.
    @param model The motor model, a key of #MOTOR_LIMITS.
    @param t The bus clock's reading now, seconds.
    @return The state.
    @throws ValueError The payload is shorter than 8 bytes.
    """
    if len(data) < 8:
        raise ValueError(f"feedback frame needs 8 bytes, got {len(data)}")
    pmax, vmax, tmax = MOTOR_LIMITS[model]
    q = (data[1] << 8) | data[2]
    dq = (data[3] << 4) | (data[4] >> 4)
    tau = ((data[4] & 0x0F) << 8) | data[5]
    return MotorState(
        motor_id=data[0],
        position_deg=math.degrees(uint_to_float(q, -pmax, pmax, 16)),
        velocity_deg_s=math.degrees(uint_to_float(dq, -vmax, vmax, 12)),
        torque_nm=uint_to_float(tau, -tmax, tmax, 12),
        t=t,
        fault_flags=data[6] & 0x30,
        temp_c=(((data[6] & 0x0F) << 8) | data[7]) / 10.0,
    )


def is_fault_frame(data: bytes) -> bool:
    """Whether a reply reports a fault rather than state (the vendor's test).

    @param data The frame's payload.
    @return True for a non-zero little-endian fault word in bytes 1..4 with bytes 5..7 zero.
    """
    return len(data) >= 8 and int.from_bytes(data[1:5], "little") != 0 and data[5] == data[6] == data[7] == 0


class MotorFault(errors.ArmError):
    """A motor answered with a fault word instead of its state."""


class MotorSilent(errors.ArmError):
    """A motor did not answer a state query within its timeout."""


class StaleZero(errors.ArmError):
    """The arm refuses to move: a motor's zero no longer matches the calibration pose."""


@dataclasses.dataclass
class Motor:
    """One RobStride motor on the bus.

    Attributes:
    - name: The joint it drives, for messages.
    - can_id: Its CAN id.
    - model: Its model, a key of #MOTOR_LIMITS.
    - kp: The position gain its frames carry, N.m per rad.
    - kd: The velocity gain its frames carry, N.m s per rad.
    - state: Its last decoded feedback, or None before the first.
    - drops: Replies that never came since the bus was built.
    """

    name: str
    can_id: int
    model: str
    kp: float
    kd: float
    state: MotorState | None = None
    drops: int = 0


def open_can_bus(channel: str) -> Any:
    """Opens the CAN adapter behind `channel` at #BITRATE with python-can.

    @param channel A serial device path (#SERIAL_DEVICE_PREFIX; a CANable on slcan firmware) or a SocketCAN link
        name (`ip -br link show type can`), which must already be up at #BITRATE.
    @return The python-can bus.
    @throws Exception Whatever python-can raises when the adapter cannot be opened.
    """
    can = python_can()
    if channel.startswith(SERIAL_DEVICE_PREFIX):
        holder = slcan_holder(channel)
        if holder is None:
            try:
                return can.interface.Bus(channel=channel, interface="slcan", bitrate=BITRATE)
            except can.CanInterfaceNotImplementedError as exc:
                raise errors.ConfigError(
                    f"{channel} is a CANable, opened directly over USB, which needs pyserial ({exc}): in the dream "
                    "environment run pip install pyserial, then start dreamcli setup or dreamcli robot up again"
                ) from exc
        channel = held_link(channel, holder)
    return can.interface.Bus(channel=channel, interface="socketcan")


@dataclasses.dataclass(frozen=True)
class SlcanDaemon:
    """An `slcand` running on this box: the serial device it holds and the kernel CAN link it made of it."""

    pid: int
    tty: str
    link: str | None


def slcan_daemons() -> list[SlcanDaemon]:
    """Every `slcand` on this box, read from `/proc` (#PROC_DIR) without touching any device.

    @return One #SlcanDaemon per process, sorted by pid.
    """
    found: list[SlcanDaemon] = []
    try:
        entries = [entry for entry in PROC_DIR.iterdir() if entry.name.isdigit()]
    except OSError:
        return found
    for entry in entries:
        try:
            if (entry / "comm").read_text().strip() != "slcand":
                continue
            argv = [word.decode(errors="replace") for word in (entry / "cmdline").read_bytes().split(b"\0") if word]
        except OSError:
            continue
        words: list[str] = []
        skip = False
        for word in argv[1:]:
            if skip:
                skip = False
            elif word in _SLCAND_VALUE_OPTIONS:
                skip = True
            elif not word.startswith("-"):
                words.append(word)
        if not words:
            continue
        tty = words[0] if words[0].startswith("/") else f"/dev/{words[0]}"
        found.append(SlcanDaemon(int(entry.name), os.path.realpath(tty), words[1] if len(words) > 1 else None))
    return sorted(found, key=lambda daemon: daemon.pid)


def slcan_holder(device: str) -> SlcanDaemon | None:
    """The `slcand` holding a serial device, if one does.

    @param device The device's path.
    @return Its #SlcanDaemon, or None.
    """
    real = os.path.realpath(device)
    return next((daemon for daemon in slcan_daemons() if daemon.tty == real), None)


def held_link(device: str, holder: SlcanDaemon) -> str:
    """The link an `slcand` made of a serial device, checked to be up.

    @param device The device, for the message.
    @param holder The `slcand` holding it.
    @return The link's name.
    """
    stop = f"stop that slcand (sudo kill {holder.pid}) and the adapter is opened directly"
    if holder.link is None:
        raise errors.ArmError(f"{device} is held by slcand (pid {holder.pid}) under a link it did not name: {stop}")
    try:
        state = link_state(NET_DIR / holder.link)
    except OSError:
        state = "missing"
    if state != "up":
        raise errors.ArmError(
            f"{device} is held by slcand (pid {holder.pid}) as CAN link {holder.link}, which is {state}: run "
            f"sudo ip link set {holder.link} up, or {stop}"
        )
    return holder.link


def link_state(entry: pathlib.Path) -> str:
    """Whether a network link is up, from its sysfs folder.

    @param entry The link's folder under #NET_DIR.
    @return `up`; else its operstate.
    """
    operstate = (entry / "operstate").read_text().strip()
    if int((entry / "flags").read_text().strip(), 16) & IFF_UP and operstate not in ("down", "lowerlayerdown"):
        return "up"
    return "down" if operstate in ("up", "unknown") else operstate


class RobstrideBus:
    """Seven RobStride motors on one CAN adapter, request then reply, one frame in flight at a time.

    Deliberately not a burst: USB-CAN adapters with a shallow receive queue drop much of a seven-frame burst, and a
    dropped reply would read as a frozen joint. One request, that motor's reply, the next motor: about a millisecond
    per joint. Every public method takes the bus's lock, since one thread drives an arm but another may close it.

    Attributes:
    - channel: The adapter, as open_can_bus() takes it.
    - motors: The motors by name, in joint order.
    - torque_enabled: Whether enable_torque() ran and no disable_torque() since.

    @see open(), close(), read_position(), command_position(), enable_torque(), disable_torque()
    """

    def __init__(
        self,
        channel: str,
        motors: list[Motor],
        *,
        opener: Callable[[str], Any],
        clock: Callable[[], float],
        sleep: Callable[[float], Any],
    ) -> None:
        """Binds the motors to an adapter; nothing is opened.

        @param channel The adapter.
        @param motors The motors, base to gripper.
        @param opener Opens the adapter and returns a bus with python-can's `send`, `recv` and `shutdown`
            (open_can_bus() in production, a fake bus in tests).
        @param clock Monotonic clock, seconds: reply deadlines, the state cache's age.
        @param sleep Sleeps that many seconds: the handshake's gaps and the retries.
        @throws ValueError Two motors share a CAN id.
        """
        self.channel = channel
        self.motors = {motor.name: motor for motor in motors}
        self._by_id = {motor.can_id: motor for motor in motors}
        if len(self._by_id) != len(motors):
            raise ValueError(f"duplicate CAN ids on {channel}: {[motor.can_id for motor in motors]}")
        self._opener = opener
        self._clock = clock
        self._sleep = sleep
        self._bus: Any = None
        self._lock = threading.RLock()
        self.torque_enabled = False

    def open(self) -> None:
        """Opens the adapter, drops stale frames and checks that every motor answers; energizes nothing.

        The handshake is a state query per motor. A silent or faulted motor closes the adapter again and raises.

        @throws ArmError A motor did not answer, or answered with a fault.
        @throws Exception Whatever the opener raises.
        """
        with self._lock:
            self._bus = self._opener(self.channel)
            try:
                self._flush_rx()
                self._handshake()
            except BaseException:
                self._close_adapter()
                raise

    def _handshake(self) -> None:
        """Queries every motor once; raises naming the silent and the faulted ones."""
        missing: list[str] = []
        faulted: list[str] = []
        for motor in self.motors.values():
            try:
                self.query_state(motor.name, timeout=HANDSHAKE_TIMEOUT_S)
            except MotorSilent:
                missing.append(motor.name)
            except MotorFault:
                faulted.append(motor.name)
            self._sleep(HANDSHAKE_GAP_S)
        if missing or faulted:
            parts = []
            if missing:
                parts.append(
                    f"{', '.join(missing)} did not answer (switch the arm's power on, release its emergency stop, "
                    "check the CAN cable and every motor's connector)"
                )
            if faulted:
                parts.append(
                    f"{', '.join(faulted)} reported a fault (switch the arm's power off and on, arm supported)"
                )
            raise errors.ArmError(f"{self.channel}: " + "; ".join(parts))

    def close(self, *, disable_torque: bool) -> None:
        """Closes the adapter, dropping torque first when asked.

        When an earlier disable failed, the adapter is closed but the motors may still be energized: a close that
        is asked to drop torque then opens the adapter once more (no handshake) and tries again.

        @param disable_torque Send the disable to every motor before closing.
        @throws Exception The disable's failure, raised after the adapter is closed: a caller that reports
            "torque off" must not do so while the motors may still be energized.
        """
        with self._lock:
            if self._bus is None:
                if not (disable_torque and self.torque_enabled):
                    return
                self._bus = self._opener(self.channel)
            failure: Exception | None = None
            try:
                if disable_torque:
                    try:
                        self.disable_torque()
                    except Exception as exc:  # the adapter is never kept open over a failed disable
                        failure = exc
            finally:
                self._close_adapter()
            if failure is not None:
                raise failure

    def _close_adapter(self) -> None:
        """Shuts the adapter down; a failure is logged, and an slcan port is still closed under it."""
        bus, self._bus = self._bus, None
        if bus is None:
            return
        try:
            bus.shutdown()
        except Exception as exc:
            _log.warning("%s: adapter shutdown failed: %s", self.channel, exc)
            # python-can's slcan shutdown writes "C" first; when that write fails it never reaches the serial close,
            # and the port would stay open under the next session's.
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

    def _send(self, can_id: int, data: bytes) -> None:
        """Sends one standard data frame to a motor."""
        message = python_can().Message(arbitration_id=can_id, data=data, is_extended_id=False)
        self._require().send(message, timeout=_SEND_TIMEOUT_S)

    def _recv_from(self, can_id: int, timeout: float) -> bytes | None:
        """The next frame whose payload byte 0 names `can_id`, or None after `timeout`; another motor's state frame
        seen meanwhile is folded into that motor's cache rather than dropped."""
        bus = self._require()
        deadline = self._clock() + timeout
        while True:
            remaining = deadline - self._clock()
            if remaining <= 0:
                return None
            message = bus.recv(timeout=min(_POLL_S, remaining))
            if message is None or message.is_error_frame or message.is_remote_frame or len(message.data) < 1:
                continue
            data = bytes(message.data)
            if data[0] == can_id:
                return data
            other = self._by_id.get(data[0])
            if other is not None and len(data) >= 8 and not is_fault_frame(data):
                other.state = decode_feedback(data, other.model, self._clock())

    def _flush_rx(self) -> None:
        """Drops the frames a previous session left queued, so the first read is not stale."""
        bus = self._require()
        for _ in range(_FLUSH_MAX_FRAMES):
            if bus.recv(timeout=0.0005) is None:
                return

    def query_state(self, name: str, timeout: float = QUERY_TIMEOUT_S) -> MotorState:
        """Sends the clear-fault query to one motor and decodes its reply.

        @param name The motor.
        @param timeout How long to wait for the reply, seconds.
        @return The motor's state, also stored as `Motor.state`.
        @throws MotorSilent Nothing came back in time (counted in `drops`).
        @throws MotorFault The reply is a fault frame.
        """
        with self._lock:
            motor = self.motors[name]
            self._send(motor.can_id, b"\xff" * 7 + bytes([CMD_CLEAR_FAULT]))
            data = self._recv_from(motor.can_id, timeout)
            if data is None:
                motor.drops += 1
                raise MotorSilent(f"{self.channel}: {name} (id {motor.can_id}) did not answer the state query")
            if is_fault_frame(data):
                raise MotorFault(f"{self.channel}: {name} fault word 0x{int.from_bytes(data[1:5], 'little'):08X}")
            motor.state = decode_feedback(data, motor.model, self._clock())
            return motor.state

    def read_position(self, name: str) -> float:
        """One motor's position, from the cache when younger than #STATE_CACHE_TTL_S, else queried.

        A missed reply keeps the last known position (logged at DEBUG; the drop is counted, and the arm's driver
        watches the counts); only a motor that has never answered raises.

        @param name The motor.
        @return Its position, degrees, in its own frame.
        @throws MotorSilent The motor has never answered.
        @throws MotorFault The reply is a fault frame.
        """
        with self._lock:
            motor = self.motors[name]
            state = motor.state
            if state is not None and self._clock() - state.t <= STATE_CACHE_TTL_S:
                return state.position_deg
            try:
                return self.query_state(name).position_deg
            except MotorSilent:
                if state is None:
                    raise
                _log.debug("%s: %s missed a reply, keeping %.2f deg", self.channel, name, state.position_deg)
                return state.position_deg

    def command_position(self, name: str, position_deg: float, *, kp: float | None, kd: float | None) -> None:
        """Sends one MIT position frame and folds the reply, the motor's new state, into its cache.

        A missed reply is counted in `drops` and costs nothing else: the cache keeps the last state.

        @param name The motor.
        @param position_deg The target, degrees, in the motor's own frame.
        @param kp The position gain for this frame only, or None for the motor's own.
        @param kd The velocity gain for this frame only, or None for the motor's own.
        @throws MotorFault The reply is a fault frame.
        """
        with self._lock:
            motor = self.motors[name]
            self._send(
                motor.can_id,
                encode_mit(motor.model, motor.kp if kp is None else kp, motor.kd if kd is None else kd, position_deg),
            )
            data = self._recv_from(motor.can_id, COMMAND_REPLY_TIMEOUT_S)
            if data is None:
                motor.drops += 1
                return
            if is_fault_frame(data):
                raise MotorFault(f"{self.channel}: {name} fault word 0x{int.from_bytes(data[1:5], 'little'):08X}")
            motor.state = decode_feedback(data, motor.model, self._clock())

    def _simple_command(self, name: str, command: int, timeout: float) -> bool:
        """Sends an `FF*7 xx` command; True when the motor answered, whose state reply is then cached."""
        motor = self.motors[name]
        self._send(motor.can_id, b"\xff" * 7 + bytes([command]))
        data = self._recv_from(motor.can_id, timeout)
        if data is not None and len(data) >= 8 and not is_fault_frame(data):
            motor.state = decode_feedback(data, motor.model, self._clock())
        return data is not None

    def enable_torque(self) -> None:
        """Energizes every motor; each holds where it is under its gains until its first position frame.

        A silent motor is asked again (#ENABLE_RETRIES) and then passed over: some firmware does not answer the
        enable, and the handshake already proved the motor is there.
        """
        with self._lock:
            for name in self.motors:
                for attempt in range(ENABLE_RETRIES + 1):
                    if self._simple_command(name, CMD_ENABLE, _ENABLE_REPLY_TIMEOUT_S) or attempt == ENABLE_RETRIES:
                        break
                    self._sleep(_RETRY_GAP_S)
            self.torque_enabled = True

    def disable_torque(self) -> None:
        """De-energizes every motor, asking a silent one again up to #DISABLE_RETRIES times."""
        with self._lock:
            for name in self.motors:
                for attempt in range(DISABLE_RETRIES + 1):
                    if self._simple_command(name, CMD_DISABLE, COMMAND_REPLY_TIMEOUT_S) or attempt == DISABLE_RETRIES:
                        break
                    self._sleep(_RETRY_GAP_S)
            self.torque_enabled = False

    def drops(self) -> int:
        """Replies that never came, every motor together, since the bus was built.

        @return The count.
        """
        return sum(motor.drops for motor in self.motors.values())


def python_can() -> Any:
    """Imports python-can on first use, so the package imports without it.

    The Metal arm's bus (#robots.damiao) imports it through here too.

    @return The `can` module.
    @throws ConfigError It is not installed.
    """
    try:
        import can
    except ImportError as exc:
        raise errors.ConfigError(
            "python-can is not installed: the dream library brings it; in your dream env "
            "(source ~/dream-env/bin/activate) run pip install --upgrade dreamlabs"
        ) from exc
    return can


@contextlib.contextmanager
def adapter_words(name: str, channel: str) -> Iterator[None]:
    """Puts the adapter's own failure in words: which arm's adapter, and what to do.

    @param name The arm, `"left"` or `"right"`.
    @param channel The adapter's channel.
    @return A context manager for the block.
    """
    try:
        yield
    except python_can().CanError as exc:
        raise errors.ArmError(
            f"the {name} arm's CAN adapter {channel} failed ({type(exc).__name__}: {exc}): unplug the adapter and "
            "plug it back in, or switch the arm's power off and on, then try again (on a station, press Reset)"
        ) from exc


class RobstrideChain:
    """One arm's RobStride motors on its adapter, as #robots.motor_arm.MotorArm drives them (a
    #robots.motor_arm.Chain).

    The Makers Arm follower and the safety checks its driver makes on every tick:

    - full-turn wrap: after a power cycle a motor can report its angle a whole turn off; a joint outside its soft
      limits that a 360 degree shift brings inside gets that shift for the session;
    - stale zero: a joint outside its limits that no whole turn explains has a zero that no longer matches the
      calibration pose, and the arm refuses to energize or move (#StaleZero);
    - the soft limits on every frame;
    - a tick reads the previous tick's command replies, no extra bus traffic; a missed reply keeps the last reading;
    - the gripper through its hold (#robots.makers_hold.GripperHold) on the last replies' gripper feedback;
    - a joint unanswered for #JOINT_SILENT_S raises #alarm, naming it (its reading is the last it sent, so a run on it
      would drive blind), and an arm whose every motor is unanswered for #ARM_SILENT_S is latched down (a motor that
      lost power comes back disabled, so carrying on would command an arm that cannot move). A motor's fault word
      (#MotorFault) holds the arm for the tick only.

    Attributes:
    - name: `"left"` or `"right"`, for messages.
    - names: The motors' names, base to gripper.
    - limits_deg: Soft limits per motor, degrees.
    - transient: #MotorFault.
    - holds_grip: True: a closed gripper squeezes as the config asks, at its hold torque
      (#robots.makers_hold.GripperHold), or at the full position gain when `gripper_hold_torque_nm` is null.
    - alarm: The joints unanswered for #JOINT_SILENT_S while the others answer, and what to do; empty otherwise.
    - bus: The arm's #RobstrideBus.
    - stale_zero: The joints whose zero is stale, with the angle each read at open; empty when none.
    - enabled: Whether the motors are energized.

    @see open(), enable(), read(), command(), relax(), close()
    """

    transient: tuple[type[Exception], ...] = (MotorFault,)
    holds_grip = True

    def __init__(
        self,
        name: str,
        channel: str,
        joints: tuple[motor_arm.Joint, ...],
        hold_torque_nm: float | None,
        *,
        opener: Callable[[str], Any],
        clock: Callable[[], float],
        sleep: Callable[[float], Any],
    ) -> None:
        """Builds the arm's bus; nothing is opened.

        @param name `"left"` or `"right"`, for messages.
        @param channel The arm's CAN adapter (open_can_bus()).
        @param joints The arm's motors, base to gripper, the gripper last.
        @param hold_torque_nm The gripper's holding torque, N.m, or None for plain position control at full gain.
        @param opener Opens the adapter.
        @param clock Monotonic clock, seconds.
        @param sleep Sleeps that many seconds.
        """
        self.name = name
        self.names = tuple(joint.name for joint in joints)
        self.limits_deg = {joint.name: joint.limits_deg for joint in joints}
        motors = [Motor(name=j.name, can_id=j.motor_id, model=j.model, kp=j.kp, kd=j.kd) for j in joints]
        self.bus = RobstrideBus(channel, motors, opener=opener, clock=clock, sleep=sleep)
        self._gripper = joints[-1]
        self._clock = clock
        self._turn_offset: dict[str, float] = dict.fromkeys(self.names, 0.0)
        self.stale_zero: dict[str, float] = {}
        self.enabled = False
        self._hold = (
            None
            if hold_torque_nm is None
            else makers_hold.GripperHold(self._gripper.kp, self._gripper.kd, self._gripper.limits_deg, hold_torque_nm)
        )
        self.alarm = ""
        self._was_holding = False
        self._last: motor_arm.MotorReading | None = None
        self._drops_seen: dict[str, int] = {}
        self._joint_silent_since: dict[str, float] = {}
        self._joint_warned: set[str] = set()

    def open(self) -> None:
        """Opens the adapter, checks every motor answers (no torque), then classifies the pose."""
        with adapter_words(self.name, self.bus.channel):
            try:
                self.bus.open()
            except errors.ArmError as exc:
                raise errors.ArmError(f"arm '{self.name}' did not open: {exc}") from exc
            try:
                self._detect_full_turn_offsets()
            except BaseException:
                self.bus.close(disable_torque=False)
                raise
        _log.info("arm '%s': open on %s, %d motors", self.name, self.bus.channel, len(self.names))

    def enable(self) -> None:
        """Energizes the arm; each motor holds its present pose under its gains.

        @throws StaleZero A motor's zero is stale; nothing is energized.
        """
        if self.stale_zero:
            raise StaleZero(self._stale_zero_message())
        self.bus.enable_torque()
        self.enabled = True

    def close(self, *, disable_torque: bool) -> None:
        """Closes the adapter, dropping torque first when asked.

        @param disable_torque Send the disable to every motor first.
        @throws Exception The disable's failure, after the adapter is closed (`RobstrideBus.close`).
        """
        self.bus.close(disable_torque=disable_torque)
        self.enabled = False

    def rearm(self) -> None:
        """Nothing: the ramp is the thread's."""

    def relax(self) -> None:
        """Switches every motor off and keeps the adapter open; read() then asks each motor for its pose."""
        self.bus.disable_torque()
        self.enabled = False
        self._last = None

    def drops(self) -> int:
        """Replies that never came, every motor together.

        @return The count.
        """
        return self.bus.drops()

    def _detect_full_turn_offsets(self) -> None:
        """Gives a joint that reads a whole turn outside its limits that turn as an offset, and marks the rest of
        those outside their limits as stale."""
        raw = {joint: self.bus.read_position(joint) for joint in self.names}
        self.stale_zero = {}
        for joint, position in raw.items():
            self._turn_offset[joint] = 0.0
            low, high = self.limits_deg[joint]
            low, high = low - _WRAP_GRACE_DEG, high + _WRAP_GRACE_DEG
            if low <= position <= high:
                continue
            for shift in (-_FULL_TURN_DEG, _FULL_TURN_DEG):
                if low <= position + shift <= high:
                    self._turn_offset[joint] = shift
                    _log.warning(
                        "arm '%s': %s reads %+.1f deg, a full turn outside %s; correcting by %+.0f deg this session",
                        self.name,
                        joint,
                        position,
                        self.limits_deg[joint],
                        shift,
                    )
                    break
            else:
                self.stale_zero[joint] = position
                _log.error(
                    "arm '%s': %s reads %+.1f deg, outside its soft limits %s by more than %.0f deg and not by a "
                    "whole turn; the motor zero no longer matches the calibration pose",
                    self.name,
                    joint,
                    position,
                    self.limits_deg[joint],
                    _WRAP_GRACE_DEG,
                )

    def _stale_zero_message(self) -> str:
        """Names every stale joint and the way out."""
        stale = ", ".join(f"{joint} at {angle:+.1f} deg" for joint, angle in self.stale_zero.items())
        return (
            f"arm '{self.name}' refuses to move: {stale} is outside the soft limits and not by a whole turn. "
            "Move the arm to its zero pose and recalibrate it."
        )

    def read(self) -> motor_arm.MotorReading:
        """The pose the last command's replies carried; before any command, every motor's pose from the bus.

        @return The reading.
        @throws MotorSilent A motor has never answered.
        @throws MotorFault A motor answered with a fault.
        """
        return self._last if self._last is not None else self._read()

    def _read(self) -> motor_arm.MotorReading:
        """Every motor's pose, turn-corrected, with its velocity, torque and temperature; from the cache within its
        age."""
        q: list[float] = []
        velocity: list[float] = []
        torque: list[float] = []
        temps: list[dict[str, float]] = []
        for joint in self.names:
            position = self.bus.read_position(joint)
            state = self.bus.motors[joint].state
            if state is None:
                raise errors.ArmError(f"arm '{self.name}': {joint} has no state after a read")
            q.append(position + self._turn_offset[joint])
            velocity.append(state.velocity_deg_s)
            torque.append(state.torque_nm)
            temps.append({"driver": state.temp_c})
        return motor_arm.MotorReading(
            q_deg=tuple(q), velocity_deg_s=tuple(velocity), torque_nm=tuple(torque), temperatures_c=tuple(temps)
        )

    def command(self, goal_deg: list[float]) -> motor_arm.MotorReading:
        """Sends one MIT position frame per motor, the gripper's through its hold, clamped and turn-corrected.

        The replies refresh the pose, so the reading returned is the post-command state at no extra bus cost.

        @param goal_deg The target per motor, degrees, base to gripper; the gripper's entry becomes the hold's target.
        @return The reading after the frames.
        @throws StaleZero The arm must not move.
        @throws ArmError The arm is not energized, or no motor has answered for #ARM_SILENT_S.
        @throws MotorFault A motor answered with a fault.
        """
        gains = self._gripper_frame(goal_deg)
        if self.stale_zero:
            raise StaleZero(self._stale_zero_message())
        if not self.enabled:
            raise errors.ArmError(f"arm '{self.name}': command before enable")
        for joint, position in zip(self.names, goal_deg, strict=True):
            low, high = self.limits_deg[joint]
            kp, kd = gains if gains is not None and joint == self._gripper.name else (None, None)
            self.bus.command_position(
                joint, max(low, min(high, float(position))) - self._turn_offset[joint], kp=kp, kd=kd
            )
        self._last = self._read()
        self._watch_replies()
        return self._last

    def _gripper_frame(self, goal_deg: list[float]) -> tuple[float, float] | None:
        """Passes the gripper goal through the hold, which may pull it back to the object and scale the gripper's
        gains for this tick; None (plain position control) with no hold or no gripper reply yet."""
        name = self._gripper.name
        state = self.bus.motors[name].state
        if self._hold is None or state is None:
            return None
        position = state.position_deg + self._turn_offset[name]
        age = self._clock() - state.t
        target, kp, kd = self._hold.update(
            goal_deg[-1], position, state.velocity_deg_s, state.torque_nm, age, self._clock()
        )
        goal_deg[-1] = target
        if self._hold.holding != self._was_holding:
            self._was_holding = self._hold.holding
            rest = self._hold.rest_effort_nm
            if self._was_holding:
                _log.info(
                    "arm '%s': gripper holding (resting effort %.2f N.m excluded)",
                    self.name,
                    0.0 if rest is None else rest,
                )
            else:
                _log.info("arm '%s': gripper released", self.name)
        return kp, kd

    def _watch_replies(self) -> None:
        """Notices an arm that stopped answering: the bus keeps the last reading over a missed reply, which is right
        for one frame and wrong for an arm that lost power, whose pose would stay fresh forever. One joint silent for
        #JOINT_SILENT_S raises #alarm; it clears when the joint answers again.

        @throws ArmError Every motor has been silent for #ARM_SILENT_S.
        """
        now = self._clock()
        motors = self.bus.motors
        for name, motor in motors.items():
            if motor.drops > self._drops_seen.get(name, 0):
                self._joint_silent_since.setdefault(name, now)
            else:
                self._joint_silent_since.pop(name, None)
                self._joint_warned.discard(name)
            self._drops_seen[name] = motor.drops
        silent = self._joint_silent_since
        if len(silent) == len(motors) and now - max(silent.values()) >= ARM_SILENT_S:
            raise errors.ArmError(
                f"no motor has answered for {now - max(silent.values()):.1f} s "
                "(arm power or e-stop, CAN cable, or adapter); support the arm, fix the cause, then reset the arms"
            )
        lost = [name for name, since in silent.items() if now - since >= JOINT_SILENT_S]
        joints = f"the {' and '.join(lost)} joint{'s' if len(lost) > 1 else ''}"
        self.alarm = (
            f"{', '.join(lost)} has not answered for {now - min(silent[name] for name in lost):.1f} s while the "
            "others do (its power or its CAN cable); its reading is stale, so the run stops and the arm parks, which "
            f"cannot drive that motor: support the arm at {joints} while it parks, then check that motor before the "
            "next run"
            if lost
            else ""
        )
        for name, since in silent.items():
            if now - since >= JOINT_SILENT_S and name not in self._joint_warned:
                self._joint_warned.add(name)
                _log.warning(
                    "arm '%s': %s has not answered for %.1f s while the others do; check that motor's CAN and power",
                    self.name,
                    name,
                    now - since,
                )
