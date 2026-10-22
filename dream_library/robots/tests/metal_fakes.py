"""Hardware fakes for the Metal arm: a CAN adapter with seven simulated Damiao motors, and a clock.

Standalone, python-can and the standard library only, so the parity snapshot (`data/metal_parity_snapshot.py`) runs
MakerMods' own driver on the same motors these tests run #robots.metal on. A motor answers every frame at once
unless told otherwise: a state query, an MIT frame, an enable or a disable with its state frame from `0x10 + id`
(status, position, velocity, torque, both temperatures), a register read or write with the register's echo. An MIT
frame moves an enabled motor to its target within the frame; a rigid obstacle stops it in the closing (decreasing
angle) direction, a rigid stop in the opening (increasing) one, and either pushes back with `kp x error` of torque.
A command frame counts only at the arbitration offset of the motor's control mode, as on the motor. The registers are
the ones the gripper's setup reads and writes: the CAN timeout, the control mode and the MIT scaling. The bus takes
python-can's place directly (send, recv, shutdown), with recv written the way python-can's `BusABC.recv` waits, on the
injected clock, so the clock advances by the same steps under both drivers.
"""

import collections
import math
import struct
import time
from typing import Any

import can

## Poll step of a waiting recv, seconds.
POLL_S = 0.0001
## Replies come from `REPLY_OFFSET + id`.
REPLY_OFFSET = 0x10
## State queries and register access go to this id.
PARAM_ID = 0x7FF
## `(PMAX rad, VMAX rad/s, TMAX N.m)` per Damiao model, MakerMods' Metal tables.
MOTOR_LIMITS = {"metal_jlo": (6.28, 10.0, 30.0), "metal_j2": (6.28, 10.0, 120.0), "metal_jhi": (6.28, 30.0, 20.0)}
## The Metal arm's motor model per CAN id.
MODELS = {
    1: "metal_jlo",
    2: "metal_j2",
    3: "metal_jlo",
    4: "metal_jlo",
    5: "metal_jhi",
    6: "metal_jhi",
    7: "metal_jhi",
}
## Arbitration offset of a command frame per control mode.
MODE_OFFSETS = {1: 0x000, 2: 0x100, 3: 0x200, 4: 0x300}
## Register ids: CAN timeout, control mode, MIT scaling.
REG_TIMEOUT, REG_MODE, REG_PMAX, REG_VMAX, REG_TMAX = 9, 10, 21, 22, 23
## Made-up gripper ends, motor degrees, for tests only: closed at 0, fully open (100 mm, the grip's 1.0) at 120.
GRIPPER_CLOSED_DEG, GRIPPER_OPEN_DEG = 0.0, 120.0
## The fixture ends as an arm section carries them.
GRIPPER_ENDS = {"gripper_open_deg": GRIPPER_OPEN_DEG, "gripper_closed_deg": GRIPPER_CLOSED_DEG}


class FakeClock:
    """A monotonic clock that moves only when something sleeps on it.

    Attributes:
    - now: The time, seconds.
    """

    def __init__(self, start: float = 1000.0) -> None:
        """@param start The first reading."""
        self.now = start

    def __call__(self) -> float:
        """The time, seconds."""
        return self.now

    def sleep(self, seconds: float) -> None:
        """Moves the clock forward by `seconds` (a negative step is none)."""
        self.now += max(0.0, seconds)


class RealClock:
    """The wall clock in the #FakeClock shape, for a fake bus under a driver thread that runs in real time."""

    def __call__(self) -> float:
        """`time.monotonic()`."""
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        """`time.sleep()`."""
        time.sleep(seconds)


def float_to_uint(x: float, low: float, high: float, bits: int) -> int:
    """Clamps `x` to `[low, high]` and scales it to an unsigned `bits`-wide integer, truncating."""
    return int((max(low, min(high, x)) - low) / (high - low) * ((1 << bits) - 1))


def uint_to_float(x: int, low: float, high: float, bits: int) -> float:
    """The inverse of float_to_uint()."""
    return float(x) / ((1 << bits) - 1) * (high - low) + low


def encode_state(motor: "FakeMotor") -> bytes:
    """A motor's state frame: `status << 4 | id`, q16, dq12, tau12, then the MOSFET's and the rotor's temperatures."""
    pmax, vmax, tmax = MOTOR_LIMITS[motor.model]
    q = float_to_uint(math.radians(motor.position_deg), -pmax, pmax, 16)
    dq = float_to_uint(math.radians(motor.velocity_deg_s), -vmax, vmax, 12)
    tau = float_to_uint(motor.torque_nm, -tmax, tmax, 12)
    status = motor.fault or (1 if motor.enabled else 0)
    return bytes(
        [
            (status << 4) | (motor.can_id & 0x0F),
            (q >> 8) & 0xFF,
            q & 0xFF,
            dq >> 4,
            ((dq & 0xF) << 4) | ((tau >> 8) & 0xF),
            tau & 0xFF,
            motor.temp_mos_c & 0xFF,
            motor.temp_rotor_c & 0xFF,
        ]
    )


def decode_mit(data: bytes, model: str) -> tuple[float, float, float]:
    """`(position deg, velocity deg/s, kp)` out of an MIT command frame."""
    pmax, vmax, _ = MOTOR_LIMITS[model]
    q = (data[0] << 8) | data[1]
    dq = (data[2] << 4) | (data[3] >> 4)
    kp = ((data[3] & 0xF) << 8) | data[4]
    return (
        math.degrees(uint_to_float(q, -pmax, pmax, 16)),
        math.degrees(uint_to_float(dq, -vmax, vmax, 12)),
        uint_to_float(kp, 0.0, 500.0, 12),
    )


class FakeMotor:
    """One simulated Damiao motor.

    Attributes:
    - can_id, model: Its id and model.
    - position_deg: Where it is, in its own frame.
    - velocity_deg_s: The velocity it reports.
    - enabled: MIT frames move it only while enabled.
    - fault: A fault status (8 to 14) every reply carries instead of enabled or disabled; 0 for none.
    - temp_mos_c, temp_rotor_c: The temperatures it reports, degrees C.
    - registers: Register id to its 4 raw bytes.
    - obstacle_deg: A rigid object: the motor cannot pass this angle in the closing (decreasing) direction, and
      pushing on it gives `kp x error` of torque.
    - stop_deg: A rigid stop the same way in the opening (increasing) direction: a gripper's real open end short of
      its configured one, a joint against the camera mount.
    - torque_nm, max_torque_nm: The torque it reports, and the largest closing torque it reported.
    - commands, kps: Every MIT frame's target and kp, in order.
    - silent: Never replies.
    - holds_replies: Holds each reply until the next frame appears on the bus, as MakerMods measured the Metal
      gripper doing.
    """

    def __init__(self, can_id: int, position_deg: float = 0.0) -> None:
        self.can_id = can_id
        self.model = MODELS[can_id]
        self.position_deg = position_deg
        self.velocity_deg_s = 0.0
        self.enabled = False
        self.fault = 0
        self.temp_mos_c = 35
        self.temp_rotor_c = 33
        pmax, vmax, tmax = MOTOR_LIMITS[self.model]
        self.registers = {
            REG_TIMEOUT: struct.pack("<I", 0),
            REG_MODE: struct.pack("<I", 1),
            REG_PMAX: struct.pack("<f", pmax),
            REG_VMAX: struct.pack("<f", vmax),
            REG_TMAX: struct.pack("<f", tmax),
        }
        self.obstacle_deg: float | None = None
        self.stop_deg: float | None = None
        self.torque_nm = 0.0
        self.max_torque_nm = 0.0
        self.commands: list[float] = []
        self.kps: list[float] = []
        self.silent = False
        self.holds_replies = False

    @property
    def mode(self) -> int:
        """The control mode register."""
        return int(struct.unpack("<I", self.registers[REG_MODE])[0])


class FakeCanBus:
    """An adapter whose frames to motor ids get Damiao replies queued on its receive side.

    Attributes:
    - channel: What it was opened as.
    - motors: The simulated motors by CAN id.
    - sent: Frames sent so far.
    - frames: Every frame sent, `(arbitration_id, payload)`.
    - drop_every: Drop the reply to every n-th frame sent; 0 drops none.
    - drop_ids: The motors whose replies drop_every may drop; None for every motor.
    - fail_sends: Every send raises, as a CANable whose serial side stopped taking writes.
    - closed: Whether shutdown() ran.
    """

    def __init__(self, channel: str, clock: Any) -> None:
        """Seven motors, ids 1 to 7, the Metal models, all at 0 degrees and disabled."""
        self.channel = channel
        self._clock = clock
        self.motors = {i: FakeMotor(i) for i in range(1, 8)}
        self._rx: collections.deque[can.Message] = collections.deque()
        self._held: list[can.Message] = []
        self.sent = 0
        self.frames: list[tuple[int, bytes]] = []
        self.drop_every = 0
        self.drop_ids: set[int] | None = None
        self.fail_sends = False
        self.closed = False

    def _queue(self, motor: FakeMotor, data: bytes) -> None:
        """Queues one reply of `motor`, or holds it, or drops it as the motor and the bus are set to."""
        if motor.silent:
            return
        dropping = self.drop_every and self.sent % self.drop_every == 0
        if dropping and (self.drop_ids is None or motor.can_id in self.drop_ids):
            return
        message = can.Message(arbitration_id=REPLY_OFFSET + motor.can_id, data=data, is_extended_id=False, is_rx=True)
        message.timestamp = self._clock()
        (self._held if motor.holds_replies else self._rx).append(message)

    def send(self, message: can.Message, timeout: float | None = None) -> None:
        """Applies the frame to its motor and queues the reply; any held reply goes out first."""
        if self.fail_sends:
            raise can.CanOperationError("Could not write to serial device")
        self.sent += 1
        data = bytes(message.data)
        self.frames.append((message.arbitration_id, data))
        for held in self._held:
            held.timestamp = self._clock()
            self._rx.append(held)
        self._held = []
        if message.arbitration_id == PARAM_ID:
            self._param(data)
            return
        motor = self.motors.get(message.arbitration_id & 0xFF)
        if motor is None or message.arbitration_id - motor.can_id != MODE_OFFSETS[motor.mode]:
            return
        if data[:7] == b"\xff" * 7:
            if data[7] == 0xFC:
                motor.enabled = True
            elif data[7] == 0xFD:
                motor.enabled = False
            elif data[7] == 0xFE:
                motor.position_deg = 0.0
            else:
                return
        elif motor.mode == 1:
            target, _, kp = decode_mit(data, motor.model)
            motor.commands.append(target)
            motor.kps.append(kp)
            if motor.enabled:
                tmax = MOTOR_LIMITS[motor.model][2]
                if motor.obstacle_deg is not None and target < motor.obstacle_deg:
                    motor.position_deg = motor.obstacle_deg
                    motor.torque_nm = max(-tmax, kp * math.radians(target - motor.obstacle_deg))
                elif motor.stop_deg is not None and target > motor.stop_deg:
                    motor.position_deg = motor.stop_deg
                    motor.torque_nm = min(tmax, kp * math.radians(target - motor.stop_deg))
                else:
                    motor.position_deg, motor.torque_nm = target, 0.0
                motor.max_torque_nm = max(motor.max_torque_nm, -motor.torque_nm)
        else:
            return
        self._queue(motor, encode_state(motor))

    def _param(self, data: bytes) -> None:
        """A frame to #PARAM_ID: a state query (0xCC) or a register read (0x33) or write (0x55)."""
        motor = self.motors.get(data[0] | (data[1] << 8))
        if motor is None:
            return
        if data[2] == 0xCC:
            self._queue(motor, encode_state(motor))
        elif data[2] == 0x33 and data[3] in motor.registers:
            self._queue(motor, data[:4] + motor.registers[data[3]])
        elif data[2] == 0x55 and data[3] in motor.registers:
            motor.registers[data[3]] = data[4:8]
            self._queue(motor, data[:8])

    def _recv_internal(self, timeout: float | None) -> can.Message | None:
        """The next queued reply, polled every #POLL_S until `timeout`; any wait takes at least one poll, so a wait
        shorter than the clock's float resolution still moves it, as real time moves under a real one."""
        deadline = self._clock() + (timeout or 0.0)
        waited = False
        while True:
            if self._rx and self._rx[0].timestamp <= self._clock():
                return self._rx.popleft()
            if not timeout or (waited and self._clock() >= deadline):
                return None
            self._clock.sleep(POLL_S)
            waited = True

    def recv(self, timeout: float | None = None) -> can.Message | None:
        """python-can's `BusABC.recv` over #_recv_internal, on the injected clock."""
        start = self._clock()
        time_left = timeout
        while True:
            message = self._recv_internal(time_left)
            if message is not None:
                return message
            if timeout is None:
                continue
            time_left = timeout - (self._clock() - start)
            if time_left <= 0:
                return None

    def shutdown(self) -> None:
        """Marks the adapter closed."""
        self.closed = True


class FakeCanFactory:
    """The opener a `DamiaoBus` takes: one #FakeCanBus per channel, kept so a test can reach the motors.

    Reopening a channel keeps its motors, as reopening an adapter leaves the motors behind it as they were.

    Attributes:
    - buses: The bus per channel.
    - opens: Times each channel was opened.
    - fail_opens: Channels whose opens raise.
    """

    def __init__(self, clock: Any = None) -> None:
        """@param clock The clock the buses wait on (#FakeClock); the wall clock when None."""
        self._clock = clock if clock is not None else RealClock()
        self.buses: dict[str, FakeCanBus] = {}
        self.opens: dict[str, int] = {}
        self.fail_opens: set[str] = set()

    def __call__(self, channel: str) -> FakeCanBus:
        """Opens `channel`."""
        if channel in self.fail_opens:
            raise can.CanInitializationError(f"could not open {channel}")
        self.opens[channel] = self.opens.get(channel, 0) + 1
        bus = self.buses.get(channel)
        if bus is None or bus.closed:
            old, bus = bus, FakeCanBus(channel, self._clock)
            if old is not None:
                bus.motors = old.motors
            self.buses[channel] = bus
        return bus

    def bus(self, channel: str) -> FakeCanBus:
        """The bus of `channel`, built (closed) before its first open so a test can pose its motors."""
        if channel not in self.buses:
            bus = FakeCanBus(channel, self._clock)
            bus.closed = True
            self.buses[channel] = bus
        return self.buses[channel]
