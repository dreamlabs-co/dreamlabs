"""Hardware fakes for the Makers Arm: a CAN adapter with seven simulated RobStride motors, and a clock.

A port of the fake bus the source Makers driver's own tests and bring-up run on, so the two drivers are tested
against the same motors: a commanded position is reached within one frame, a rigid obstacle stops the gripper and
pushes back with `kp x error` of torque, every frame is answered unless drops are injected. Two changes: the clock is
injected (#FakeClock, so a test steps time instead of sleeping, and the parity test steps it exactly as the source's
run does), and the bus takes python-can's place directly (send, recv, shutdown) instead of subclassing it, with recv
written the way python-can's `BusABC.recv` waits, so the clock advances by the same steps.
"""

import collections
import math
import time
from typing import Any

import can

from robots import robstride

## Poll step of a waiting recv, seconds: the source fake's.
POLL_S = 0.0001


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


def encode_feedback(
    motor_id: int,
    model: str,
    position_deg: float,
    velocity_deg_s: float = 0.0,
    torque: float = 0.0,
    temp_c: float = 35.0,
    mode: int = 0,
    fault_flags: int = 0,
) -> bytes:
    """The inverse of `robstride.decode_feedback`, for simulated motors."""
    pmax, vmax, tmax = robstride.MOTOR_LIMITS[model]
    q = robstride.float_to_uint(math.radians(position_deg), -pmax, pmax, 16)
    dq = robstride.float_to_uint(math.radians(velocity_deg_s), -vmax, vmax, 12)
    tau = robstride.float_to_uint(torque, -tmax, tmax, 12)
    # Low 12 bits the temperature (tenths), bits 15..14 the mode, bits 13..12 the fault and warning flags.
    temp = (int(round(temp_c * 10)) & 0x0FFF) | ((mode & 0x3) << 14) | ((fault_flags & 0x30) << 8)
    return bytes(
        [
            motor_id & 0xFF,
            (q >> 8) & 0xFF,
            q & 0xFF,
            dq >> 4,
            ((dq & 0xF) << 4) | ((tau >> 8) & 0xF),
            tau & 0xFF,
            (temp >> 8) & 0xFF,
            temp & 0xFF,
        ]
    )


def decode_mit_position(data: bytes, model: str) -> float:
    """The position, degrees, out of an MIT command frame."""
    pmax = robstride.MOTOR_LIMITS[model][0]
    q = (data[0] << 8) | data[1]
    return math.degrees(q / 65535.0 * 2 * pmax - pmax)


def decode_mit_kp(data: bytes) -> float:
    """The position gain out of an MIT command frame."""
    return (((data[3] & 0xF) << 8) | data[4]) / 4095.0 * robstride.KP_RANGE[1]


class FakeMotor:
    """One simulated motor.

    Attributes:
    - can_id, model: Its id and model.
    - position_deg: Where it is, in its own frame.
    - enabled: Commands move it only while enabled.
    - fault_word: Non-zero makes every reply a fault frame.
    - commands, kps: Every position frame's target and kp, in order.
    - obstacle_deg: A rigid object: the motor cannot pass this angle in the closing (+) direction, and pushing on
      it gives `kp x error` of torque.
    - torque_nm, max_torque_nm: The torque it reports, and the most it reported.
    """

    def __init__(self, can_id: int, model: str, position_deg: float = 0.0) -> None:
        self.can_id = can_id
        self.model = model
        self.position_deg = position_deg
        self.enabled = False
        self.fault_word = 0
        self.commands: list[float] = []
        self.kps: list[float] = []
        self.obstacle_deg: float | None = None
        self.torque_nm = 0.0
        self.max_torque_nm = 0.0


class FakeCanBus:
    """An adapter whose frames to motor ids get RobStride replies queued on its receive side.

    Attributes:
    - channel: What it was opened as.
    - motors: The simulated motors by CAN id.
    - sent: Frames sent so far.
    - frames: Every frame sent, `(arbitration_id, payload)`.
    - drop_every: Drop the reply to every n-th frame sent; 0 drops none.
    - fail_sends: Every send raises, as a CANable whose serial side stopped taking writes.
    - closed: Whether shutdown() ran.
    """

    def __init__(self, channel: str, clock: Any) -> None:
        """Seven motors, ids 1 to 7, the Makers models (ids 2 and 3 are RS02s), all at 0 degrees."""
        self.channel = channel
        self._clock = clock
        self.motors = {i + 1: FakeMotor(i + 1, "O1" if i + 1 in (2, 3) else "O0") for i in range(7)}
        self._rx: collections.deque[can.Message] = collections.deque()
        self.sent = 0
        self.frames: list[tuple[int, bytes]] = []
        self.drop_every = 0
        self.fail_sends = False
        self.closed = False

    def _reply(self, motor: FakeMotor) -> None:
        if motor.fault_word:
            data = bytes([motor.can_id]) + motor.fault_word.to_bytes(4, "little") + b"\x00\x00\x00"
        else:
            data = encode_feedback(
                motor.can_id, motor.model, motor.position_deg, torque=motor.torque_nm, mode=2 if motor.enabled else 0
            )
        message = can.Message(arbitration_id=robstride.HOST_ID, data=data, is_extended_id=False, is_rx=True)
        message.timestamp = self._clock()
        self._rx.append(message)

    def send(self, message: can.Message, timeout: float | None = None) -> None:
        """Applies the frame to its motor and queues the reply."""
        if self.fail_sends:
            raise can.CanOperationError("Could not write to serial device")
        self.sent += 1
        data = bytes(message.data)
        self.frames.append((message.arbitration_id, data))
        motor = self.motors.get(message.arbitration_id)
        if motor is None:
            return
        if data[:7] == b"\xff" * 7:
            command = data[7]
            if command == robstride.CMD_ENABLE:
                motor.enabled = True
            elif command == robstride.CMD_DISABLE:
                motor.enabled = False
            elif command != robstride.CMD_CLEAR_FAULT:
                return
        else:
            target = decode_mit_position(data, motor.model)
            kp = decode_mit_kp(data)
            motor.commands.append(target)
            motor.kps.append(kp)
            if motor.enabled:
                if motor.obstacle_deg is not None and target > motor.obstacle_deg:
                    motor.position_deg = motor.obstacle_deg
                    limit = robstride.MOTOR_LIMITS[motor.model][2]
                    motor.torque_nm = min(limit, kp * math.radians(target - motor.obstacle_deg))
                else:
                    motor.position_deg, motor.torque_nm = target, 0.0
                motor.max_torque_nm = max(motor.max_torque_nm, motor.torque_nm)
        if self.drop_every and self.sent % self.drop_every == 0:
            return
        self._reply(motor)

    def _recv_internal(self, timeout: float | None) -> can.Message | None:
        """The source fake's wait: the next queued reply, polled every #POLL_S until `timeout`."""
        deadline = self._clock() + (timeout or 0.0)
        while True:
            if self._rx and self._rx[0].timestamp <= self._clock():
                return self._rx.popleft()
            if self._clock() >= deadline:
                return None
            self._clock.sleep(POLL_S)

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
    """The opener a `RobstrideBus` takes: one #FakeCanBus per channel, kept so a test can reach the motors.

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
