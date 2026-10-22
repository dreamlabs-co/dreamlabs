# Ported from MakerMods Lab (`makermodslab/metal_gripper.py`, `metal_gripper_hold.py` and `gripper_settings.py` at
# f3ee37c), the project #robots.makers_hold vendors from, Apache License 2.0. Copyright 2026
# MakerMods contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""The Metal arm's gripper: MakerMods' holding control and its safety stops, on the arm's #robots.damiao bus.

The gripper is the arm's seventh Damiao motor, but MakerMods drive it apart from the joints, and so does this module:
every request to it is a `DamiaoBus.query()` (it answers late), it is set up and checked on its own, and its target
goes through the holding law (#robots.makers_hold.HoldingController: closing toward decreasing angles, no closing cap,
no resting-effort baseline, MakerMods' Metal settings) before each MIT frame.

Ported from MakerMods Lab: `MetalGripperBus` (the wrapper's watchdog, temperature stop and fault latch) and
`MetalHoldingBus` (holding in MIT mode, the gripper mode their Metal console runs). Kept, as they run it:

- setup (connect()): the motor must start in MIT or force-position mode, disabled, both temperature sensors under
  #RESTART_C, with the MIT scaling registers this driver encodes with; then its CAN timeout is set to #WATCHDOG_TICKS
  (a stopped process leaves the gripper energized at most 500 ms), its mode to MIT, and it is preloaded at its own
  angle with zero gains. All volatile: nothing is saved to flash;
- every command (write()): the target clamped to the soft limits, a fresh state query, the holding law, then one MIT
  frame at the configured gains with the arm's velocity feedforward, zeroed while holding or after #PAUSE_S without a
  command;
- the checks around each (check()): a latched fault, #STOP_C on either sensor, a fault status, or an energized gripper
  that reads disabled or has not answered for #STALE_S, counted from the tick's own refresh (MakerMods' holding thread
  refreshed it every 20 ms apart from the servo loop; here a tick held up by the station's main thread is not a silent
  gripper, and the motor's timeout still bounds a stopped tick). The arm's chain trips the gripper on any failure
  (trip(): it is disabled and the fault latched, so every later call raises #GripperSafetyError until the arm is
  reopened);
- release (release()): the gripper is disabled and its mode and timeout restored, whatever the joints do.

Not ported: the current-limited force-position mode (MakerMods' other gripper mode), the registry of live grippers and
their live settings, and the holding thread, which wrote the gripper at 50 Hz apart from the servo loop: here the arm's
thread writes it on every tick (#robots.metal.ARM, #robots.damiao.DamiaoChain), with or without a target, and when the
thread stops the gripper's own timeout disables it.

@see MetalGripper, GripperSafetyError
"""

import logging
import math
import struct
from collections.abc import Callable

from robots import damiao, errors, makers_hold

## The gripper's motor name on the arm's bus.
NAME = "gripper"
## The gripper's soft limits, motor degrees: 0 with the jaws closed (the calibration zero), 137.5 fully open, the
## vendor's 0 to 2.4 rad.
LIMITS_DEG = (0.0, 137.5)
## The CAN timeout set while the arm runs, in the motor's 50 us units: 500 ms.
WATCHDOG_TICKS = 10_000
## Either temperature sensor at this latches the gripper down, degrees C.
STOP_C = 55.0
## Setup refuses a gripper this warm, degrees C.
RESTART_C = 45.0
## An energized gripper whose last state is older than this latches down, seconds.
STALE_S = 0.3
## No command for this long zeroes the feedforward, seconds.
PAUSE_S = 0.1

_log = logging.getLogger(__name__)


class GripperSafetyError(errors.ArmError):
    """The gripper is latched down; the arm must be reopened deliberately."""


class MetalGripper:
    """The Metal arm's gripper motor on the arm's bus, in motor degrees; driven by the arm's one thread.

    Attributes:
    - kp: The gripper's MIT position gain.
    - kd: The gripper's MIT velocity gain.
    - limits_deg: Its soft limits, degrees.
    - controller: The holding law.
    - error: Why the gripper is latched down; empty while it runs.
    - enabled: Whether it is energized.
    - temperature_c: The warmer of its two sensors at its last reply, degrees C.
    - temperatures_c: Both sensors at its last reply, degrees C: `"coil"` (the motor's rotor) and `"driver board"`
      (its MOSFETs).

    @see connect(), enable(), write(), refresh(), check(), trip(), release()
    """

    def __init__(
        self,
        bus: damiao.DamiaoBus,
        *,
        gains: tuple[float, float],
        hold_torque_nm: float,
        clock: Callable[[], float],
    ) -> None:
        """Binds the gripper to its arm's bus; nothing is sent.

        @param bus The arm's bus; its motor #NAME is the gripper.
        @param gains `(kp, kd)` of every frame but the zero-gain preloads.
        @param hold_torque_nm The hold torque, N.m (#robots.makers_hold.MIN_HOLD_TORQUE_NM to `MAX_HOLD_TORQUE_NM`).
        @param clock Monotonic clock, seconds.
        """
        self._bus = bus
        self._motor = bus.motors[NAME]
        self.kp, self.kd = gains
        self.limits_deg = LIMITS_DEG
        self.controller = makers_hold.HoldingController(
            hold_torque_nm,
            (math.radians(LIMITS_DEG[0]), math.radians(LIMITS_DEG[1])),
            closing_direction=-1,
            cap_closing_torque=False,
            rest_baseline=False,
        )
        self._clock = clock
        self.error = ""
        self.enabled = False
        self.temperature_c = 0.0
        self.temperatures_c: dict[str, float] = {}
        self._prepared = False
        self._mode = damiao.MODE_MIT
        self._original_mode: int | None = None
        self._original_timeout: int | None = None
        self._status = damiao.STATUS_DISABLED
        self._last_feedback = 0.0
        self._goal = 0.0
        self._velocity = 0.0
        self._feedforward = 0.0
        self._last_action = 0.0
        self._was_holding = False

    def _is_state(self, data: bytes) -> bool:
        """Whether a reply is the gripper's state frame (a register reply shares its arbitration id)."""
        return damiao.is_state_frame(data, self._motor.can_id)

    def _state(self, data: bytes) -> None:
        """Takes a state frame: the motor's state, its status, its temperature and the time it came."""
        state = self._bus.store(NAME, data)
        self._status = state.status
        self.temperatures_c = {"coil": state.temp_rotor_c, "driver board": state.temp_mos_c}
        self.temperature_c = max(state.temp_mos_c, state.temp_rotor_c)
        self._last_feedback = self._clock()

    def _register(self, rid: int, value: int | None = None, fmt: str = "I", expect: int | None = None) -> float:
        """Reads a register, or writes one and reads it back.

        A write's echo must carry the value, and its read-back too; a late duplicate of an earlier read is skipped,
        but a register that really kept another value is returned, and fails the write.

        @param rid The register.
        @param value The value to write, or None to read.
        @param fmt Its `struct` format, `"I"` or `"f"`.
        @param expect The value a read must carry to be the answer (a write's read-back), or None.
        @return The value.
        @throws GripperSafetyError A write did not keep its value.
        @throws MotorSilent No reply came.
        """
        op = damiao.OP_READ if value is None else damiao.OP_WRITE
        header = struct.pack("<HBB", self._motor.can_id, op, rid)
        payload = bytes(4) if value is None else struct.pack("<" + fmt, value)
        target = payload if value is not None else None
        if expect is not None:
            target = struct.pack("<" + fmt, expect)
        want = None if target is None else (lambda data: data[4:] == target)
        reply = self._bus.query(NAME, damiao.PARAM_ID, header + payload, lambda data: data[:4] == header, want)
        result = float(struct.unpack("<" + fmt, reply[4:])[0])
        if value is not None and (result != value or self._register(rid, fmt=fmt, expect=value) != value):
            raise GripperSafetyError(f"{self._bus.channel}: gripper register {rid} did not keep the value {value}")
        return result

    def _sensors(self) -> str:
        (hot, hot_c), (other, other_c) = sorted(self.temperatures_c.items(), key=lambda item: -item[1])
        return f"{hot} reads {hot_c:.0f} C ({other} {other_c:.0f} C)"

    def refresh(self) -> None:
        """Queries the gripper's state.

        @throws MotorSilent No reply came.
        """
        self._state(self._bus.query(NAME, damiao.PARAM_ID, damiao.refresh_payload(self._motor.can_id), self._is_state))

    def _command(self, command: int) -> None:
        """Sends an `FF*7 xx` command at the present mode's offset; an enable wants any status but disabled, a disable
        any but enabled, so a held earlier state frame is never taken for the answer (the callers check the status)."""
        wants: dict[int, Callable[[bytes], bool]] = {
            damiao.CMD_ENABLE: lambda data: data[0] >> 4 != damiao.STATUS_DISABLED,
            damiao.CMD_DISABLE: lambda data: data[0] >> 4 != damiao.STATUS_ENABLED,
        }
        arbitration_id = self._motor.can_id + damiao.MODE_OFFSETS[self._mode]
        self._state(
            self._bus.query(NAME, arbitration_id, b"\xff" * 7 + bytes([command]), self._is_state, wants[command])
        )

    def _send_mit(self, kp: float, kd: float, goal_deg: float, velocity_deg_s: float, torque_nm: float) -> None:
        """One MIT frame; its reply is the gripper's new state, checked."""
        packet = damiao.encode_mit(self._motor.model, kp, kd, goal_deg, velocity_deg_s, torque_nm)
        self._state(self._bus.query(NAME, self._motor.can_id, packet, self._is_state))
        self.check()

    def _preload(self) -> None:
        """A zero-gain MIT frame at the goal: the motor's stored target is where it is, not a stale one."""
        self._send_mit(0.0, 0.0, self._goal, 0.0, 0.0)

    def check(self, *, age: bool = True) -> None:
        """Raises when the gripper must stop; sends nothing.

        @param age Whether the last reply's age is tested too. The arm's read() skips it before the tick's own
            refresh: there the age measures how late the tick is, not the gripper (MakerMods' holding thread tests it
            right after its refresh).
        @throws GripperSafetyError A latched fault, #STOP_C reached, a fault status, or an energized gripper that
            reads disabled or (with `age`) whose last reply is older than #STALE_S.
        """
        if self.error:
            raise GripperSafetyError(self.error)
        if self.temperature_c >= STOP_C:
            raise GripperSafetyError(
                f"{self._bus.channel}: the gripper's {self._sensors()}, at or over its {STOP_C:.0f} C stop; grip "
                f"released. Let it cool below {RESTART_C:.0f} C before restarting. A gripper that heats up every run "
                "is pushing on something: its open end (gripper_open_deg) may be set past its real stop"
            )
        if self._status not in (damiao.STATUS_DISABLED, damiao.STATUS_ENABLED):
            raise GripperSafetyError(
                f"{self._bus.channel}: gripper motor fault 0x{self._status:X}, "
                f"{damiao.FAULT_TEXT.get(self._status, 'an unknown fault')}; the arm must stop"
            )
        stale = age and self._clock() - self._last_feedback > STALE_S
        if self.enabled and (self._status != damiao.STATUS_ENABLED or stale):
            raise GripperSafetyError(f"{self._bus.channel}: the gripper is disabled or its feedback stale")

    def trip(self, exc: BaseException) -> None:
        """Latches the gripper down: the fault is kept and the gripper disabled.

        The disable is best effort: the gripper's timeout disables it anyway once the frames stop.

        @param exc Why.
        """
        self.error = str(exc)
        self.enabled = False
        try:
            self._command(damiao.CMD_DISABLE)
        except Exception:
            _log.exception(
                "%s: the gripper's disable could not be confirmed; its timeout stays armed", self._bus.channel
            )
        _log.error("%s: gripper stopped: %s", self._bus.channel, exc)

    def connect(self) -> None:
        """Sets the gripper up for holding control on an open bus; energizes nothing.

        @throws GripperSafetyError It is in neither MIT nor force-position mode, is not disabled, is warm, its MIT
            scaling is not the driver's, or a register write did not keep its value.
        @throws MotorSilent It did not answer.
        """
        self._mode = int(self._register(damiao.REG_CTRL_MODE))
        self._original_mode = self._mode
        if self._mode not in (damiao.MODE_MIT, damiao.MODE_FORCE_POSITION):
            raise GripperSafetyError(f"{self._bus.channel}: the gripper must start in MIT or force-position mode")
        self.refresh()
        if self._status != damiao.STATUS_DISABLED:
            raise GripperSafetyError(
                f"{self._bus.channel}: the gripper motor is not disabled (status {self._status}), so the arm is "
                "likely still powered from an earlier session: support the arm, then switch its power off and on"
            )
        if self.temperature_c >= RESTART_C:
            raise GripperSafetyError(
                f"{self._bus.channel}: the gripper is still warm, its {self._sensors()}; let it cool below "
                f"{RESTART_C:.0f} C first"
            )
        scaling = zip(
            (damiao.REG_PMAX, damiao.REG_VMAX, damiao.REG_TMAX), damiao.MOTOR_LIMITS[self._motor.model], strict=True
        )
        for rid, scale in scaling:
            if not math.isclose(self._register(rid, fmt="f"), scale, rel_tol=1e-5):
                raise GripperSafetyError(
                    f"{self._bus.channel}: the gripper's MIT scaling (register {rid}) is not the Metal driver's"
                )
        assert self._motor.state is not None, "refresh() stored a state"
        self._goal = float(self._motor.state.position_deg)
        self._original_timeout = int(self._register(damiao.REG_TIMEOUT))
        self._register(damiao.REG_TIMEOUT, WATCHDOG_TICKS)
        if self._mode != damiao.MODE_MIT:
            self._register(damiao.REG_CTRL_MODE, damiao.MODE_MIT)
        self._mode = damiao.MODE_MIT
        self._prepared = True
        self._preload()

    def enable(self) -> None:
        """Energizes the gripper where it is: preloaded at its goal, then enabled; a failure trips it.

        @throws GripperSafetyError Setup did not complete, or it did not enable.
        """
        if not self._prepared:
            raise GripperSafetyError(f"{self._bus.channel}: the gripper's setup did not complete")
        if self.enabled:
            return
        try:
            self._preload()
            self._command(damiao.CMD_ENABLE)
            if self._status != damiao.STATUS_ENABLED:
                raise GripperSafetyError(f"{self._bus.channel}: the gripper did not enable in MIT mode")
            self.enabled = True
        except Exception as exc:
            self.trip(exc)
            raise

    def write(self, goal_deg: float, velocity_deg_s: float, torque_nm: float) -> None:
        """One command: the goal through the holding law, then an MIT frame at the gripper's gains.

        @param goal_deg The requested angle, degrees; clamped to the soft limits.
        @param velocity_deg_s The arm's velocity feedforward for the gripper, degrees per second.
        @param torque_nm Feedforward torque, N.m.
        @throws GripperSafetyError A value is not finite, or a check failed.
        @throws MotorSilent The gripper did not answer.
        """
        if not all(math.isfinite(x) for x in (goal_deg, velocity_deg_s, torque_nm)):
            raise GripperSafetyError(f"{self._bus.channel}: invalid gripper command")
        self._velocity, self._feedforward = velocity_deg_s, torque_nm
        self._last_action = self._clock()
        self._force(goal_deg)

    def _force(self, goal_deg: float) -> None:
        """Regulates on a fresh state: the holding law's target, sent at the gripper's own gains (no closing cap)."""
        self.check()
        if not math.isfinite(goal_deg):
            raise GripperSafetyError(f"{self._bus.channel}: invalid gripper target")
        low, high = self.limits_deg
        self._goal = min(high, max(low, goal_deg))
        # A fresh state every time: the holding equation must use a sample that answers this request.
        self.refresh()
        self.check()
        state = self._motor.state
        assert state is not None, "refresh() stored a state"
        now = self._clock()
        target = self.controller.update(
            math.radians(self._goal),
            math.radians(state.position_deg),
            math.radians(state.velocity_deg_s),
            state.torque_nm,
            self.kp,
            now,
        )
        holding = self.controller.holding
        if holding != self._was_holding:
            self._was_holding = holding
            _log.info("%s: gripper %s", self._bus.channel, "holding" if holding else "released")
        paused = now - self._last_action > PAUSE_S
        velocity = 0.0 if holding or paused else self._velocity
        feedforward = 0.0 if holding or paused else self._feedforward
        self._send_mit(self.kp, self.kd, math.degrees(target), velocity, feedforward)

    def _disable(self) -> None:
        """Disables the gripper and requires the reply to say so."""
        self.enabled = False
        self._command(damiao.CMD_DISABLE)
        if self._status == damiao.STATUS_ENABLED:
            raise GripperSafetyError(f"{self._bus.channel}: the gripper's disable was not confirmed")

    def release(self) -> list[str]:
        """Disables the gripper and restores the mode and CAN timeout it had before connect(); on an open bus.

        The mode is read first (a mode write may have landed although its reply was lost; an unreadable one leaves the
        disable at the last known mode). A gripper that did not confirm its disable keeps its settings.

        @return What could not be verified, empty when everything was.
        """
        problems: list[str] = []
        try:
            self._mode = int(self._register(damiao.REG_CTRL_MODE))
        except Exception as exc:
            problems.append(f"gripper mode unreadable: {exc}")
        try:
            self._disable()
            if self._original_mode is not None:
                self._register(damiao.REG_CTRL_MODE, self._original_mode)
                self._mode = self._original_mode
            if self._original_timeout is not None:
                self._register(damiao.REG_TIMEOUT, self._original_timeout)
        except Exception as exc:
            problems.append(str(exc))
        self._prepared = False
        return problems
