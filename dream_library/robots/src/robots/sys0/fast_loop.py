"""A faster CAN transport under the arms: i2rt's driver threads sped up at runtime, the vendored i2rt left untouched.

This is not a second control loop. The station's control loop (`robots.control`) and System 0's 250 Hz thread
(`robots.sys0.motors`) decide what each motor is told; i2rt's two threads per arm are the transport that carries
that block onto the CAN bus and brings the motors' state back. Stock i2rt takes 16.5 ms to do that round trip.
For a position target that delay hardly matters, because the motor's own PD acts on it in the firmware. For a
torque it is most of the problem: System 0 answers what it measured one round trip ago, and it expects
this faster transport, so with System 0 the arms are opened with it.

Measured on a YAM, a torque written into i2rt's
command block shows up in the state the host reads 16.5 ms later, about 9 ms of it in i2rt itself:

- the driver loop (`DMChainCanInterface._set_commands`) sends each motor its MIT frame and waits for that motor's
  reply before the next: 7 x 0.41 ms = 3.2 ms a cycle. Sending all seven back to back and then collecting the
  replies takes 1.83 ms (the 1 Mbps bus's own limit), with no reply lost in 7,000 cycles;
- the robot loop (`MotorChainRobot.start_server`) runs `update()` then sleeps 4 ms: 6.35 ms a cycle, and a new
  command waits for it before the driver sees it;
- the driver stores a cycle's replies only after releasing its command lock, so the robot loop, which was waiting
  on that lock, reads the cycle before.

install_before_robot() swaps the robot loop for one with a #ROBOT_LOOP_SLEEP_S sleep, and the driver loop for one
that stores its state before it releases the lock and pauses #DRIVER_HANDOFF_S after it, so the robot loop gets the
lock between cycles (both before the robot is built: their threads start in the constructors).
install_pipelined_bus() swaps the driver's per-cycle function for #PipelinedBus, the back-to-back
one; a motor whose reply does not arrive in #REPLY_WAIT_S is re-sent through i2rt's own send-and-wait call, which
keeps i2rt's retries and its error when a motor is really gone. With both, a torque reaches the host's view of the
joint in 8.6 to 10 ms. The frames are packed exactly as i2rt v1.1.1's `DMSingleMotorCanInterface.set_control` packs
them in MIT mode (#mit_frame()), the version `vendor/i2rt` carries.

@see install_before_robot(), install_pipelined_bus()
"""

import logging
import time
from typing import Any

import numpy as np

_log = logging.getLogger(__name__)

## Sleep after each robot-loop `update()`, seconds (i2rt v1.1.1: 0.004).
ROBOT_LOOP_SLEEP_S: float = 0.0005
## How long a pipelined cycle waits for the seven replies before falling back per motor, seconds.
REPLY_WAIT_S: float = 0.004
## The driver loop's pause between cycles, outside the command lock, seconds. On Python 3.10, a station's Python
## (Ubuntu 22.04), i2rt's own `time.sleep(0)` takes the lock back before the robot loop waiting on it gets it: with the
## 1.8 ms pipelined cycle the robot loop got it about 40 times a second on a station (stock i2rt: 140), and since
## `update()` waits for it holding the robot's state lock, every read of the arm and the tracker's every tick waited 20
## to 100 ms. Python 3.12 hands it over; a real pause does on both, for about a tenth of the
## driver's rate.
DRIVER_HANDOFF_S: float = 0.0002


def install_before_robot() -> None:
    """Replaces i2rt's robot loop and driver loop with the fast ones, for every robot built from here on.

    The patch is on i2rt's classes, so it reaches both arms and is the same whether it runs once or twice.

    @throws ImportError i2rt is not installed.
    """
    from i2rt.motor_drivers.dm_driver import DMChainCanInterface  # type: ignore[import-not-found]
    from i2rt.robots.motor_chain_robot import MotorChainRobot  # type: ignore[import-not-found]

    DMChainCanInterface._set_torques_and_update_state = _driver_loop
    MotorChainRobot.start_server = _robot_loop


def install_pipelined_bus(robot: Any) -> "PipelinedBus":
    """Swaps one robot's driver cycle onto the back-to-back bus; the driver looks the method up every cycle.

    @param robot An i2rt `MotorChainRobot`, already built.
    @return The bus, which counts its cycles and fallbacks.
    @throws ValueError The motor chain is not in MIT mode.
    """
    chain = robot.motor_chain
    with chain.command_lock:  # between two cycles
        bus = PipelinedBus(chain)
        chain._set_commands = bus
    return bus


def mit_frame(motor_type: str, pos: float, vel: float, kp: float, kd: float, torque: float) -> bytes:
    """The 8 data bytes `set_control` sends in MIT mode (i2rt v1.1.1, `dm_driver.py`).

    @param motor_type The motor's i2rt type name, which sets each field's range.
    @param pos Position target, motor space, radians.
    @param vel Velocity target, rad/s.
    @param kp Stiffness.
    @param kd Damping.
    @param torque Torque, newton metres.
    @return The frame's data.
    """
    from i2rt.motor_drivers.utils import MotorType, float_to_uint  # type: ignore[import-not-found]

    c = MotorType.get_motor_constants(motor_type)
    p = float_to_uint(pos, c.POSITION_MIN, c.POSITION_MAX, 16)
    v = float_to_uint(vel, c.VELOCITY_MIN, c.VELOCITY_MAX, 12)
    kp_ = float_to_uint(kp, c.KP_MIN, c.KP_MAX, 12)
    kd_ = float_to_uint(kd, c.KD_MIN, c.KD_MAX, 12)
    t = float_to_uint(torque, c.TORQUE_MIN, c.TORQUE_MAX, 12)
    return bytes(
        [
            (p >> 8) & 0xFF,
            p & 0xFF,
            (v >> 4) & 0xFF,
            ((v & 0xF) << 4) | (kp_ >> 8),
            kp_ & 0xFF,
            (kd_ >> 4) & 0xFF,
            ((kd_ & 0xF) << 4) | (t >> 8),
            t & 0xFF,
        ]
    )


class PipelinedBus:
    """The driver's per-cycle `_set_commands` with every motor's frame sent back to back.

    Attributes:
    - fallbacks: Replies that did not arrive in time and were re-sent through i2rt's own call.
    - cycles: Cycles run.
    """

    def __init__(self, chain: Any) -> None:
        """Binds the chain's interface and its motors' frame and reply ids.

        @param chain The robot's i2rt `DMChainCanInterface`.
        @throws ValueError The chain is not in MIT mode.
        """
        import can
        from i2rt.motor_drivers.dm_driver import ControlMode

        iface = chain.motor_interface
        if getattr(iface, "control_mode", ControlMode.MIT) != ControlMode.MIT:
            raise ValueError("the pipelined bus packs MIT frames; this motor chain is not in MIT mode")
        self._chain = chain
        self._iface = iface
        self._can = can
        self._frame_ids = [iface._get_frame_id(m) for m, _ in chain.motor_list]
        self._reply_index = {iface.receive_mode.get_receive_id(m): k for k, (m, _) in enumerate(chain.motor_list)}
        self.fallbacks = 0
        self.cycles = 0

    def __call__(self, commands: list[Any]) -> list[Any]:
        """Sends one cycle's commands and returns the motors' replies, in chain order.

        @param commands One i2rt `MotorCmd` per motor.
        @return One i2rt `MotorInfo` per motor.
        """
        chain, iface = self._chain, self._iface
        bus = iface.bus
        count = len(chain.motor_list)
        args = []
        for idx, (_, motor_type) in enumerate(chain.motor_list):
            c = commands[idx]
            d = chain.motor_direction[idx]
            args.append(
                (motor_type, chain._joint_position_sim_to_real_idx(c.pos, idx), c.vel * d, c.kp, c.kd, c.torque * d)
            )
        while bus.recv(timeout=0.0) is not None:  # a late reply from an earlier fallback is not this cycle's state
            pass
        for idx, (motor_type, pos, vel, kp, kd, torque) in enumerate(args):
            data = mit_frame(motor_type, pos, vel, kp, kd, torque)
            bus.send(self._can.Message(arbitration_id=self._frame_ids[idx], data=data, is_extended_id=False))
        replies: list[Any] = [None] * count
        got, deadline = 0, time.monotonic() + REPLY_WAIT_S
        while got < count and (left := deadline - time.monotonic()) > 0:
            message = bus.recv(timeout=left)
            if message is None:
                break
            k = self._reply_index.get(message.arbitration_id)
            if k is not None and replies[k] is None:
                replies[k] = iface.parse_recv_message(message, chain.motor_list[k][1])
                got += 1
        for k in range(count):  # a lost reply: i2rt's own send-and-wait for that motor (its retries, its error)
            if replies[k] is None:
                self.fallbacks += 1
                motor_id, motor_type = chain.motor_list[k]
                replies[k] = iface.set_control(
                    motor_id=motor_id,
                    motor_type=motor_type,
                    pos=args[k][1],
                    vel=args[k][2],
                    kp=args[k][3],
                    kd=args[k][4],
                    torque=args[k][5],
                )
        self.cycles += 1
        return replies


def _robot_loop(self: Any) -> None:
    """i2rt v1.1.1's robot loop (update, check the chain, sleep, report the rate) with a #ROBOT_LOOP_SLEEP_S sleep."""
    last_time, iteration_count = time.time(), 0
    self.update()
    while not self._stop_event.is_set():
        current_time = time.time()
        elapsed_time = current_time - last_time
        self.update()
        if not self.motor_chain.running:
            raise RuntimeError(f"{self}: motor_chain_robot's motor chain is not running, exiting the robot server")
        time.sleep(ROBOT_LOOP_SLEEP_S)
        iteration_count += 1
        if elapsed_time >= 10.0:
            _log.info("%s: robot loop at %.2f Hz (fast loop)", self, iteration_count / elapsed_time)
            last_time, iteration_count = current_time, 0


def _driver_loop(self: Any) -> None:
    """i2rt v1.1.1's `DMChainCanInterface._set_torques_and_update_state`, storing each cycle's state before it
    releases the command lock (and without its step-time statistics). Recovery and errors are unchanged."""
    with self._rate_recorder:
        while self.running:
            try:
                with self.command_lock:
                    try:
                        motor_feedback = self._set_commands(self.commands)
                    except RuntimeError as e:
                        if "Motor error detected" in str(e):
                            _log.warning("Motor error in control loop, attempting recovery: %s", e)
                            if self._try_recover_motors():
                                _log.warning("Motor recovery successful, continuing control loop")
                                continue
                            self.running = False
                        raise
                    errors = np.array([fb.error_code != "0x1" for fb in motor_feedback])
                    if np.any(errors):
                        _log.warning("Motor errors detected in feedback: %s", errors)
                        if self._try_recover_motors(motor_feedback):
                            _log.warning("Motor recovery successful, continuing control loop")
                            continue
                        self.running = False
                        _log.error("motor errors: %s", errors)
                        raise Exception(
                            "motors have unrecoverable errors after recovery attempts, stopping control loop"
                        )
                    with self.state_lock:  # inside the command lock: whoever takes it next reads this cycle
                        self.state = motor_feedback
                        self._update_absolute_positions(motor_feedback)
                if self.same_bus_device_driver is not None:
                    time.sleep(0.001)
                    with self.same_bus_device_lock:
                        self.same_bus_device_states = self.same_bus_device_driver.read_states()
                time.sleep(DRIVER_HANDOFF_S)  # long enough for the robot loop, waiting on the lock, to take it
                self._rate_recorder.track()
            except Exception as e:
                _log.error("DM error in control loop: %s", e)
                self.running = False
                raise e
