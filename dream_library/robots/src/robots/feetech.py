"""One arm's Feetech servos on their serial bus, for #robots.motor_arm: #FeetechChain, on LeRobot's own bus.

The SO-101 (#robots.so101) moves on STS3215 servos daisy-chained on one serial bus per arm. The bus is LeRobot's
(`lerobot.motors.feetech.FeetechMotorsBus`, the `so101` extra, imported only by open_bus()): the protocol, the
handshake and the calibration registers are LeRobot's. On top of it this module does what a policy run needs, the way
an SO-101 follower does:

- the arm's LeRobot calibration file must match what its servos hold, or the arm does not start: the file of another
  arm (a swapped cable or adapter) would put every joint somewhere else;
- the servos' gains and the gripper's torque caps are set at open, as LeRobot's SO-101 follower sets them;
- a tick reads every position in one sync read and writes every goal in one sync write, from the one thread that
  owns the bus (#robots.motor_arm.MotorArm's);
- a read that fails holds the arm for the tick (a dropped status packet is common and harmless); the arm is latched
  down after #MAX_MISSED_S without one;
- every servo's temperature is read once every #TEMP_READ_S, for #robots.motor_guard;
- the switch-off goes servo by servo, so one that refuses (a gripper in overload, which answers every write with its
  alarm) leaves the others switched off, and the error names it and what to do.

Units: a servo reads 0 to 4095 ticks a turn, and LeRobot's calibration moves each servo's zero (its homing offset)
so the arm's middle pose reads #CENTER_TICK. That middle pose is the URDF's zero (dream's SO-101 bake used the same
rule), so a joint in degrees is `(ticks - 2048) * 360 / 4096` on every SO-101. The calibration file is what makes it
true for an arm, which is why it is checked. The servos report no velocity or torque the tick reads, so both read 0.

@see FeetechChain, open_bus(), read_calibration()
"""

import dataclasses
import json
import logging
import pathlib
import re
from collections.abc import Callable
from typing import Any

from robots import errors, motor_arm

## Ticks a servo reads for one turn.
TICKS_PER_TURN = 4096
## The tick LeRobot's calibration puts the arm's middle pose at: the URDF's zero.
CENTER_TICK = 2048
## A run of failed reads this long latches the arm down, seconds.
MAX_MISSED_S = 0.3
## Reads tried at open before the arm is refused: the first status packet after connect is often lost.
FIRST_READ_ATTEMPTS = 5
## Wait between those reads, seconds.
FIRST_READ_RETRY_S = 0.2
## Retries of the switch-off, which has to land.
DISABLE_RETRIES = 5
TEMP_READ_S = 1.0
## The servos' position mode (`lerobot.motors.feetech.OperatingMode.POSITION`).
_POSITION_MODE = 0
## The gripper's torque caps, as LeRobot's SO-101 follower sets them, so a gripper closed on an object stalls safely.
_GRIPPER_CAPS = (("Max_Torque_Limit", 500), ("Protection_Current", 250), ("Overload_Torque", 25))
## The fields of a calibration entry, in LeRobot's order.
_CALIBRATION_FIELDS = ("id", "drive_mode", "homing_offset", "range_min", "range_max")

_log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Calibration:
    """One servo's entry of a LeRobot calibration file (`lerobot.motors.MotorCalibration`).

    Attributes:
    - id: The servo's id on the bus.
    - drive_mode: 0 (LeRobot's SO-101 has no other).
    - homing_offset: What the servo subtracts from its raw angle, ticks: where its middle pose reads #CENTER_TICK.
    - range_min: The low end of the joint's calibrated travel, ticks (the gripper's closed end).
    - range_max: The high end, ticks (the gripper's open end).
    """

    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int


class MissedRead(Exception):
    """A position read got no answer; the arm holds for this tick."""


def ticks_to_deg(ticks: float) -> float:
    """A servo's reading in the URDF's degrees.

    @param ticks The reading, ticks.
    @return Degrees from the middle pose.
    """
    return (ticks - CENTER_TICK) * 360.0 / TICKS_PER_TURN


def deg_to_ticks(deg: float) -> int:
    """An angle in the URDF's degrees as a servo goal.

    @param deg Degrees from the middle pose.
    @return The goal, whole ticks.
    """
    return round(deg * TICKS_PER_TURN / 360.0) + CENTER_TICK


def read_calibration(path: str, joints: tuple[motor_arm.Joint, ...]) -> dict[str, Calibration]:
    """Reads an arm's LeRobot calibration file for its servos.

    @param path The file (`lerobot-calibrate` writes it, under `~/.cache/huggingface/lerobot/calibration/`).
    @param joints The arm's servos, base to gripper.
    @return Each servo's #Calibration by its name.
    """
    try:
        raw = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise errors.ConfigError(
            f"the calibration file {path} is not there: calibrate the arm with dreamcli setup (step 1 calibrates an "
            "SO-101) or lerobot-calibrate, which write it under ~/.cache/huggingface/lerobot/calibration/, and give "
            "that file's path"
        ) from exc
    except (OSError, ValueError) as exc:
        raise errors.ConfigError(f"the calibration file {path} does not load: {exc}") from exc
    if not isinstance(raw, dict):
        raise errors.ConfigError(f"the calibration file {path} is not a LeRobot calibration: it is not a mapping")
    out = {}
    for joint in joints:
        entry = raw.get(joint.name)
        if not isinstance(entry, dict) or not all(isinstance(entry.get(key), int) for key in _CALIBRATION_FIELDS):
            raise errors.ConfigError(
                f"the calibration file {path} has no LeRobot entry for {joint.name!r} (integers "
                f"{', '.join(_CALIBRATION_FIELDS)}): is it an SO-101 follower's?"
            )
        calibration = Calibration(**{key: entry[key] for key in _CALIBRATION_FIELDS})
        if calibration.id != joint.motor_id or calibration.drive_mode != 0:
            raise errors.ConfigError(
                f"the calibration file {path} gives {joint.name!r} id {calibration.id} and drive mode "
                f"{calibration.drive_mode}; an SO-101 has it at id {joint.motor_id}, drive mode 0"
            )
        if not 0 <= calibration.range_min < calibration.range_max < TICKS_PER_TURN:
            raise errors.ConfigError(
                f"the calibration file {path} gives {joint.name!r} the range {calibration.range_min} to "
                f"{calibration.range_max}, not a range of ticks (0 to {TICKS_PER_TURN - 1}, low to high)"
            )
        out[joint.name] = calibration
    return out


def open_bus(port: str, joints: tuple[motor_arm.Joint, ...], calibration: dict[str, Calibration] | None) -> Any:
    """Opens an arm's servos on LeRobot's Feetech bus and checks they all answer.

    @param port The serial device (`/dev/serial/by-id/...`).
    @param joints The arm's servos, base to gripper.
    @param calibration Each servo's #Calibration, from the arm's file.
    @return The connected `FeetechMotorsBus`.
    @throws ConfigError LeRobot is not installed.
    @throws ArmError The port does not open, or a servo does not answer.
    """
    try:
        # LeRobot is the `so101` extra, which the dev venv leaves out, so mypy cannot see it.
        from lerobot.motors import Motor, MotorCalibration, MotorNormMode  # type: ignore[import-not-found]
        from lerobot.motors.feetech import FeetechMotorsBus  # type: ignore[import-not-found]
    except ImportError as exc:
        raise errors.ConfigError(
            f"an SO-101 runs on LeRobot's Feetech bus, which is not installed here ({exc}). In the dream environment "
            '(`source ~/dream-env/bin/activate`, Python 3.12 or newer) run: pip install "dreamlabs[so101]"'
        ) from exc
    # Every read and write here is raw ticks; the norm modes are LeRobot's for an SO-101 and go unused.
    modes = [MotorNormMode.RANGE_M100_100] * (len(joints) - 1) + [MotorNormMode.RANGE_0_100]
    bus = FeetechMotorsBus(
        port=port,
        motors={
            joint.name: Motor(joint.motor_id, joint.model, mode) for joint, mode in zip(joints, modes, strict=True)
        },
        calibration=None
        if calibration is None
        else {name: MotorCalibration(**dataclasses.asdict(entry)) for name, entry in calibration.items()},
    )
    try:
        bus.connect()
    except RuntimeError as exc:
        missing = missing_servos(str(exc), joints)
        if not missing:
            raise errors.ArmError(f"the servos on {port} did not answer: {exc}") from exc
        raise errors.ArmError(
            f"{' and '.join(missing)} on {port} did not answer while the others did: reseat that servo's cable at "
            "both ends (a loose 3-pin plug is the usual cause), check the arm's power supply, then try again"
        ) from exc
    except (ConnectionError, OSError) as exc:
        raise errors.ArmError(f"the servos on {port} did not answer: {exc}") from exc
    return bus


def held_calibration(port: str, joints: tuple[motor_arm.Joint, ...]) -> dict[str, Calibration]:
    """The calibration an arm's servos hold in their EEPROM, read once without writing a register.

    @param port The serial device.
    @param joints The arm's servos.
    @return Each servo's #Calibration as the servos hold it.
    """
    bus = open_bus(port, joints, None)
    try:
        held = bus.read_calibration()
    finally:
        bus.disconnect(disable_torque=False)
    return {
        joint.name: Calibration(
            id=joint.motor_id,
            drive_mode=int(held[joint.name].drive_mode),
            homing_offset=int(held[joint.name].homing_offset),
            range_min=int(held[joint.name].range_min),
            range_max=int(held[joint.name].range_max),
        )
        for joint in joints
    }


def calibration_differences(file: dict[str, Calibration], held: dict[str, Any]) -> list[str]:
    """Where a calibration file and what the servos hold disagree: homing offset and range, per servo.

    @param file The file's #Calibration per servo.
    @param held What the servos hold per servo.
    @return `"<servo> <field> <held> (file <value>)"` per difference.
    """
    return [
        f"{name} {field} {getattr(held[name], field)} (file {getattr(entry, field)})"
        for name, entry in file.items()
        for field in ("homing_offset", "range_min", "range_max")
        if getattr(held[name], field) != getattr(entry, field)
    ]


def missing_servos(message: str, joints: tuple[motor_arm.Joint, ...]) -> list[str]:
    """The servos LeRobot's handshake reports missing, by id and name.

    @param message The handshake's error text.
    @param joints The arm's servos.
    @return `servo 3 (elbow_flex)` per missing servo, in LeRobot's order.
    """
    names = {joint.motor_id: joint.name for joint in joints}
    found = []
    for line in message.partition("Missing motor IDs:")[2].splitlines()[1:]:
        listed = re.match(r"\s*- (\d+)", line)
        if listed is None:
            break
        motor_id = int(listed.group(1))
        found.append(f"servo {motor_id} ({names[motor_id]})" if motor_id in names else f"servo {motor_id}")
    return found


class FeetechChain:
    """One arm's servos on their serial bus: the #robots.motor_arm.Chain of an SO-101.

    Attributes:
    - name: `"left"` or `"right"`, for messages.
    - port: The serial device.
    - names: The servos' names, base to gripper.
    - limits_deg: Soft limits per servo, degrees: the table's, narrowed to the calibrated range.
    - transient: #MissedRead.
    - calibration: Each servo's #Calibration, from the arm's file.
    - enabled: Whether the servos may be energized.
    - holds_grip: False: the gripper is a servo in position mode, so #robots.motor_guard holds it in a gentle squeeze
      on an object.
    - alarm: Empty: the chain raises nothing of its own.

    @see open(), enable(), read(), command(), relax(), close()
    """

    transient: tuple[type[Exception], ...] = (MissedRead,)
    holds_grip = False
    alarm = ""

    def __init__(
        self,
        name: str,
        port: str,
        joints: tuple[motor_arm.Joint, ...],
        calibration_path: str,
        *,
        opener: Callable[..., Any],
        clock: Callable[[], float],
        sleep: Callable[[float], Any],
    ) -> None:
        """Reads the arm's calibration file; nothing is opened.

        @param name `"left"` or `"right"`, for messages.
        @param port The serial device.
        @param joints The arm's servos, base to gripper, the gripper last.
        @param calibration_path The arm's LeRobot calibration file.
        @param opener Opens the bus: open_bus(), or a fake bus in tests.
        @param clock Monotonic clock, seconds.
        @param sleep Sleeps that many seconds.
        @throws ConfigError The calibration file does not fit the arm (read_calibration()).
        """
        self.name = name
        self.port = port
        self.names = tuple(joint.name for joint in joints)
        self.calibration = read_calibration(calibration_path, joints)
        self._calibration_path = calibration_path
        self.limits_deg = {
            joint.name: (
                max(joint.limits_deg[0], ticks_to_deg(self.calibration[joint.name].range_min)),
                min(joint.limits_deg[1], ticks_to_deg(self.calibration[joint.name].range_max)),
            )
            for joint in joints
        }
        self.enabled = False
        self._joints = joints
        self._opener = opener
        self._clock = clock
        self._sleep = sleep
        self._bus: Any = None
        self._zeros = (0.0,) * len(joints)
        self._last: motor_arm.MotorReading | None = None
        self._missed = 0
        self._missed_since: float | None = None
        self._temps_at: float | None = None

    def open(self) -> None:
        """Opens the bus, checks the calibration, configures the servos and reads the pose.

        The servos are switched off while they are configured (LeRobot writes their registers that way), so an arm
        left energized drops: open it resting.

        @throws ArmError A servo does not answer, the servos' calibration is not the file's, or no pose is read; the
            bus is closed again.
        """
        bus = self._opener(self.port, self._joints, self.calibration)
        try:
            self._check_calibration(bus)
            self._disable(bus)
            bus.configure_motors()
            for joint in self._joints:
                bus.write("Operating_Mode", joint.name, _POSITION_MODE)
                bus.write("P_Coefficient", joint.name, int(joint.kp))
                bus.write("I_Coefficient", joint.name, 0)
                bus.write("D_Coefficient", joint.name, int(joint.kd))
            for register, value in _GRIPPER_CAPS:
                bus.write(register, self.names[-1], value)
        except ConnectionError as exc:
            bus.disconnect(disable_torque=False)
            raise errors.ArmError(f"arm '{self.name}': the servos on {self.port} stopped answering: {exc}") from exc
        except BaseException:
            bus.disconnect(disable_torque=False)
            raise
        self._bus = bus
        for attempt in range(FIRST_READ_ATTEMPTS):
            try:
                self._last = self._read()
                break
            except ConnectionError as exc:
                _log.warning("arm '%s': first read %d failed: %s", self.name, attempt + 1, exc)
                self._sleep(FIRST_READ_RETRY_S)
        if self._last is None:
            self.close(disable_torque=False)
            raise errors.ArmError(f"arm '{self.name}': no pose from {self.port} in {FIRST_READ_ATTEMPTS} reads")
        _log.info("arm '%s': open on %s, %d servos", self.name, self.port, len(self.names))

    def enable(self) -> None:
        """Energizes the servos, each holding where it is: its goal is set to its reading first."""
        raw = self._bus.sync_read("Present_Position", list(self.names), normalize=False)
        self._bus.sync_write("Goal_Position", raw, normalize=False)
        self.enabled = True
        self._bus.enable_torque(num_retry=2)

    def read(self) -> motor_arm.MotorReading:
        """Every servo's position in one sync read, and every #TEMP_READ_S their temperatures.

        @return The pose, degrees.
        @throws MissedRead The read got no answer; the arm holds this tick.
        @throws ArmError No read has answered for #MAX_MISSED_S.
        """
        try:
            self._last = self._read()
        except ConnectionError as exc:
            now = self._clock()
            self._missed += 1
            if self._missed_since is None:
                self._missed_since = now
            if now - self._missed_since > MAX_MISSED_S:
                raise errors.ArmError(
                    f"arm '{self.name}': no position from {self.port} for {now - self._missed_since:.2f} s: {exc}"
                ) from exc
            raise MissedRead(f"a position read on {self.port} got no answer") from exc
        self._missed_since = None
        now = self._clock()
        if self._temps_at is None or now - self._temps_at >= TEMP_READ_S:
            self._temps_at = now
            try:
                raw = self._bus.sync_read("Present_Temperature", list(self.names), normalize=False)
            except ConnectionError as exc:
                _log.debug("arm '%s': a temperature read got no answer: %s", self.name, exc)
                return self._last
            return dataclasses.replace(
                self._last, temperatures_c=tuple({"servo": float(raw[name])} for name in self.names)
            )
        return self._last

    def command(self, goal_deg: list[float]) -> motor_arm.MotorReading:
        """Every servo's goal in one sync write (no reply comes back).

        @param goal_deg One goal per servo, degrees, base to gripper.
        @return The pose this tick read.
        """
        self._bus.sync_write(
            "Goal_Position",
            {name: deg_to_ticks(deg) for name, deg in zip(self.names, goal_deg, strict=True)},
            normalize=False,
        )
        assert self._last is not None, "a tick reads before it commands"
        return self._last

    def rearm(self) -> None:
        """Nothing: the ramp is the thread's."""

    def relax(self) -> None:
        """Switches every servo off and keeps the bus open; a servo reads its position with torque off."""
        self._disable(self._bus)
        self.enabled = False

    def close(self, *, disable_torque: bool) -> None:
        """Closes the bus, switching the servos off first when asked.

        A close asked to switch off after the bus was closed with the servos on opens it once more to do so.

        @param disable_torque Switch every servo off first.
        @throws Exception The switch-off's failure, after the bus is closed.
        """
        if self._bus is None:
            if not (disable_torque and self.enabled):
                return
            self._bus = self._opener(self.port, self._joints, self.calibration)
        bus, self._bus = self._bus, None
        try:
            if disable_torque:
                self._disable(bus)
                self.enabled = False
        finally:
            try:
                bus.disconnect(disable_torque=False)
            except Exception as exc:
                _log.warning("arm '%s': closing %s: %s", self.name, self.port, exc)

    def drops(self) -> int:
        """Position reads that got no answer.

        @return The count.
        """
        return self._missed

    def _disable(self, bus: Any) -> None:
        failed = []
        for joint in self._joints:
            try:
                bus.disable_torque(joint.name, num_retry=DISABLE_RETRIES)
            except (ConnectionError, RuntimeError) as exc:
                _log.error("arm '%s': switching %s off: %s", self.name, joint.name, exc)
                failed.append(f"{joint.name} (id {joint.motor_id}): {_advice(str(exc))}")
        if failed:
            others = "the other servos are off" if len(failed) < len(self._joints) else "no servo confirmed it"
            raise errors.ArmError(
                f"arm '{self.name}' switch-off: {others}; {'; '.join(failed)}. Support the arm, then switch its power "
                "off and on"
            )

    def _read(self) -> motor_arm.MotorReading:
        """One sync read of every position, in degrees."""
        raw = self._bus.sync_read("Present_Position", list(self.names), normalize=False)
        return motor_arm.MotorReading(tuple(ticks_to_deg(raw[name]) for name in self.names), self._zeros, self._zeros)

    def _check_calibration(self, bus: Any) -> None:
        """Refuses an arm whose servos hold another calibration than its file: the file of another arm, or one
        recalibrated since.

        @throws ArmError A servo's homing offset or range differs from the file's.
        """
        differ = calibration_differences(self.calibration, bus.read_calibration())
        if differ:
            raise errors.ArmError(
                f"arm '{self.name}' on {self.port} holds another calibration than {self._calibration_path}: "
                f"{', '.join(differ)}. Is that file this arm's (a swapped cable puts the other arm here)? Name this "
                "arm's file, or calibrate it again: dreamcli setup calibrates an SO-101 on its step 1"
            )


def _advice(error: str) -> str:
    if "Overload" in error:
        return (
            "it reports an overload: it pushed too hard for too long, into its stop or on an object, and cut its "
            f"own torque. Check that nothing holds it and that its calibrated range is inside its real stops ({error})"
        )
    return f"it did not confirm the switch-off ({error})"
