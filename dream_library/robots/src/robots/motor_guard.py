"""What every arm on a motor bus watches on every tick of its thread: #MotorGuard, blocked motors and hot ones."""

import dataclasses
import logging
import math
from collections.abc import Callable

_log = logging.getLogger(__name__)

JOINT_BLOCK_DEG = 12.0
GRIP_BLOCK_MM = 6.0
STILL_DEG = 1.5
GRIP_STILL_MM = 1.0
BLOCK_S = 0.5
JOINT_HOLD_DEG = 5.0
GRIP_HOLD_MM = 1.0
GRIP_SQUEEZE_MM = 3.0
FREE_DEG = 1.0
BLOCK_STOP_S = 5.0
TEMP_STOP_S = 1.0


@dataclasses.dataclass
class _Slot:
    since: float | None = None
    ref: float = 0.0
    hold: float | None = None
    direction: float = 0.0
    held_since: float = 0.0
    told: bool = False


class MotorGuard:
    """Watches one arm's motors; run by its thread, read from any (#alarm(), #motor_temps_c)."""

    def __init__(
        self,
        arm: str,
        names: tuple[str, ...],
        joints: int,
        *,
        temp_warn_c: float,
        temp_stop_c: float,
        holds_grip: bool,
        to_deg: Callable[[float], float],
        clock: Callable[[], float],
    ) -> None:
        """Nothing is watched until the first tick.

        @param arm The arm, for messages.
        @param names The motors, base to gripper, the gripper last.
        @param joints The joint slots the arm drives.
        @param temp_warn_c The body's warning temperature, degrees C.
        @param temp_stop_c The body's stop temperature, degrees C.
        @param holds_grip Whether the chain sets how a closed gripper squeezes.
        @param to_deg The gripper's motor angle at a stroke, for the log.
        @param clock Monotonic clock, seconds.
        """
        self.arm = arm
        self.names = names
        self.joints = joints
        self.temp_warn_c = temp_warn_c
        self.temp_stop_c = temp_stop_c
        self.holds_grip = holds_grip
        self.to_deg = to_deg
        self._clock = clock
        self._slots = [_Slot() for _ in range(joints)]
        self._grip = _Slot()
        self._blocked_alarm: str | None = None
        self._hot: dict[tuple[str, str], float] = {}
        self._warned: set[tuple[str, str]] = set()
        self._temp_alarm: str | None = None
        self.motor_temps_c: tuple[float, ...] | None = None

    def alarm(self, chain_alarm: str = "") -> str | None:
        """Why the run must stop, if it must: the chain's own alarm, a motor too hot, a joint blocked too long.

        @param chain_alarm The chain's alarm, empty when it has none.
        @return The reason, or None.
        """
        return chain_alarm or self._temp_alarm or self._blocked_alarm or None

    def adjust(self, goal: list[float], measured: list[float], command: list[float]) -> list[float]:
        """This tick's goal with every blocked motor held (see the module docstring); watches the pushes.

        @param goal `[6 joints rad, gripper mm]`.
        @param measured The pose this tick read, same units.
        @param command The command sent last tick, same units.
        @return The goal to step toward, a new list.
        """
        now = self._clock()
        out = list(goal)
        blocked: list[str] = []
        for i, slot in enumerate(self._slots):
            out[i] = self._joint(i, slot, out[i], measured[i], command[i], now)
            if slot.hold is not None and now - slot.held_since >= BLOCK_STOP_S:
                blocked.append(
                    f"{self.names[i]} has been blocked for {now - slot.held_since:.0f} s "
                    f"(asked to go {math.degrees(abs(goal[i] - measured[i])):.0f} deg further)"
                )
        g = len(goal) - 1
        out[g] = self._gripper(out[g], measured[g], command[g], now)
        self._blocked_alarm = (
            None
            if not blocked
            else f"{'; '.join(blocked)}: something is in its way or it presses on the table, and the policy keeps "
            "asking; the run stops and the arm parks. Clear the way before the next run"
        )
        return out

    def _joint(self, i: int, slot: _Slot, goal: float, measured: float, command: float, now: float) -> float:
        if slot.hold is not None:
            back = (goal - slot.hold) * slot.direction <= 0.0
            reached = abs(measured - slot.hold) < math.radians(FREE_DEG)
            if not (back or reached):
                return slot.hold
            _log.info("arm '%s': %s is free again", self.arm, self.names[i])
            slot.hold, slot.since = None, None
            return goal
        push = command - measured
        if self._pushing(slot, push, measured, math.radians(JOINT_BLOCK_DEG), math.radians(STILL_DEG), now):
            slot.direction = math.copysign(1.0, push)
            slot.hold = measured + slot.direction * math.radians(JOINT_HOLD_DEG)
            slot.held_since = now
            _log.warning(
                "arm '%s': %s is blocked: it has not moved for %.1f s while asked to go %.0f deg further; holding it "
                "%.0f deg past where it stopped (something in its way, or pressing on the table)",
                self.arm,
                self.names[i],
                BLOCK_S,
                math.degrees(abs(push)),
                JOINT_HOLD_DEG,
            )
            return slot.hold if (goal - slot.hold) * slot.direction > 0.0 else goal
        return goal

    def _gripper(self, goal: float, measured: float, command: float, now: float) -> float:
        slot = self._grip
        if slot.hold is not None:
            if (goal - slot.hold) * slot.direction > 0.0 and abs(measured - slot.hold) >= GRIP_STILL_MM / 2:
                return slot.hold
            slot.hold, slot.since = None, None
            return goal
        push = command - measured
        if not self._pushing(slot, push, measured, GRIP_BLOCK_MM, GRIP_STILL_MM, now):
            return goal
        if push > 0.0:
            slot.direction, slot.hold = 1.0, measured + GRIP_HOLD_MM
            if not slot.told:
                slot.told = True
                _log.warning(
                    "arm '%s': the gripper stops opening at %.0f deg, short of its open end %.0f deg (%.0f mm "
                    "short); holding it there. Something stops it, or gripper_open_deg is past its real stop: set "
                    "gripper_open_deg to about %.0f for this arm (dreamcli setup measures it)",
                    self.arm,
                    self.to_deg(measured),
                    self.to_deg(command),
                    push,
                    self.to_deg(measured),
                )
        elif not self.holds_grip:
            slot.direction, slot.hold = -1.0, measured - GRIP_SQUEEZE_MM
            _log.info("arm '%s': the gripper closed on something; holding it with a gentle squeeze", self.arm)
        else:
            return goal
        return slot.hold if (goal - slot.hold) * slot.direction > 0.0 else goal

    @staticmethod
    def _pushing(slot: _Slot, push: float, measured: float, block: float, still: float, now: float) -> bool:
        if abs(push) < block or (slot.since is not None and abs(measured - slot.ref) > still):
            slot.since = None
            return False
        if slot.since is None:
            slot.since, slot.ref = now, measured
            return False
        return now - slot.since >= BLOCK_S

    def temperatures(self, temps: tuple[dict[str, float], ...] | None) -> None:
        """Takes a tick's temperatures, logs the warm ones and sets or clears the alarm.

        @param temps Per motor, base to gripper, its sensors by name, degrees C.
        """
        if temps is None:
            return
        now = self._clock()
        self.motor_temps_c = tuple(max(sensors.values(), default=math.nan) for sensors in temps)
        hottest: tuple[float, str, str] | None = None
        for motor, sensors in zip(self.names, temps, strict=True):
            for sensor, value in sensors.items():
                key = (motor, sensor)
                if value >= self.temp_stop_c:
                    since = self._hot.setdefault(key, now)
                    if now - since >= TEMP_STOP_S and (hottest is None or value > hottest[0]):
                        hottest = (value, motor, sensor)
                else:
                    self._hot.pop(key, None)
                if value >= self.temp_warn_c and key not in self._warned:
                    self._warned.add(key)
                    _log.warning(
                        "arm '%s': %s %s at %.0f C (warns at %.0f C, the run stops at %.0f C); let the arm rest "
                        "between runs",
                        self.arm,
                        motor,
                        sensor,
                        value,
                        self.temp_warn_c,
                        self.temp_stop_c,
                    )
                elif value < self.temp_warn_c - 5.0:
                    self._warned.discard(key)
        if hottest is not None and self._temp_alarm is None:
            value, motor, sensor = hottest
            self._temp_alarm = (
                f"{motor} {sensor} reached {value:.0f} C (the run stops at {self.temp_stop_c:.0f} C); the arm parks. "
                f"Let it cool below {self.temp_warn_c:.0f} C before the next run"
            )
            _log.error("arm '%s': %s", self.arm, self._temp_alarm)
        elif self._temp_alarm is not None and all(t < self.temp_warn_c for t in self.motor_temps_c):
            _log.info("arm '%s': every motor is under %.0f C again", self.arm, self.temp_warn_c)
            self._temp_alarm = None
