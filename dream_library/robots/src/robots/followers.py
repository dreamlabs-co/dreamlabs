"""One follower arm as the loop sees it: #FollowerArm over a #Driver, in the arm's URDF units.

Every robot reaches the loop through the same two classes. A #Driver is the small piece that knows one robot's
hardware: it reads the arm and takes a target, both in the units of the arm's URDF (joints in radians, with the
URDF's zero and direction; the gripper from 0 closed to 1 open), and it can close and switch the arm off. #FollowerArm
wraps any driver with what every arm needs and no driver should repeat: the command checks (one finite target per
joint, a gripper in `[0, 1]`), the last command for `commanded_*`, hold(), and the closed flag.

An arm has as many joints as its `home` in the config lists (`FollowerArm.dof`), 1 to #MAX_JOINTS. Its driver reads
that many and takes that many, and the loop, the policy's wire and the live view carry that many; a read or a command
of another width fails naming the arm.

The drivers this package ships:

- `robots.yam.I2rtDriver`: the i2rt YAM.
- `robots.motor_arm.MotorArm`: an arm on a CAN motor bus (a Makers Arm, a Metal arm), one shared thread over the
  robot's table.
- `robots.driver_template.TemplateArm`: the commented template a custom robot starts from (`body: custom`).

A driver reports errors by raising; they propagate unchanged. `ArmError` is raised here only for what this wrapper
itself detects.

System 0 (#robots.sys0) writes the arm a level below write(): write_motor_command() swaps in a whole MIT-mode block
(position, stiffness, damping and torque per motor) and set_motor_loop_hook() runs a check on the driver's motor loop
before every update, both passed to a driver that has them (`robots.yam.I2rtDriver`). An arm System 0 drives may also
be served from a process of its own (`robots.sys0.arm_process.RemoteArm`); what the station, the loop and the home
move use of either is #Arm.

@see Driver, Reading, FollowerArm, Arm
"""

import abc
import dataclasses
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
from shared.contracts import live

from robots import errors, types

## The most joints an arm may have: what the live view's tick carries per arm.
MAX_JOINTS = live.TICK_MAX_ARM_DOF

## Deprecated: the joints every arm had before each arm counted its own (`FollowerArm.dof`). Drivers copied from the
## earlier template or the trossen example size their readings by it, so a six-joint arm's copy still runs; a driver
## for another arm uses its own count.
JOINT_COUNT = 6


@dataclasses.dataclass(frozen=True)
class Reading:
    """One read of an arm, in its URDF units.

    Attributes:
    - positions_rad: Joint positions, shape `[dof]` (the arm's joints, `FollowerArm.dof`), float64.
    - velocities_rad_s: Joint velocities, shape `[dof]`, float64.
    - gripper: Gripper opening, 0 closed to 1 open.
    - gripper_effort: The gripper motor's effort as the driver reports it (for `robots.gripper_force`), or None.
    - motor_temps_c: The hottest sensor of each motor, base to gripper, degrees C, as of the driver's last report;
      None when the driver reports none.
    """

    positions_rad: npt.NDArray[np.float64]
    velocities_rad_s: npt.NDArray[np.float64]
    gripper: float
    gripper_effort: float | None = None
    motor_temps_c: tuple[float, ...] | None = None


class Driver(abc.ABC):
    """What one robot's arm must do for the loop: four methods in the arm's URDF units.

    read() and write() are called from the loop's tick and its command thread, up to a few hundred times a second, so
    each should cost little (a cached reading, a target handed to the robot's own thread or bus). write() takes a
    target; how the arm gets there (its gains, its rate cap) is the driver's business.
    """

    drops_when_limp = False

    @abc.abstractmethod
    def read(self) -> Reading:
        """The arm's latest state.

        @return The #Reading.
        @throws Exception The driver's own error when the arm cannot be read.
        """

    @abc.abstractmethod
    def write(self, positions_rad: npt.NDArray[np.float64], gripper: float) -> None:
        """Takes a new target.

        @param positions_rad A finite target per joint, `[dof]`, radians, float64 (checked by #FollowerArm).
        @param gripper The gripper target, 0 closed to 1 open (checked by #FollowerArm).
        @throws Exception The driver's own error when the arm cannot take it.
        """

    @abc.abstractmethod
    def close(self) -> None:
        """Releases the hardware without moving the arm; the motors keep their last target where they can."""

    @abc.abstractmethod
    def torque_off(self) -> None:
        """Switches the arm's motors off and releases the hardware; an arm without brakes parks first."""

    def is_alive(self) -> bool:
        """Whether the driver still answers; read() and write() are refused once it is False.

        @return True unless the driver knows its arm stopped answering.
        """
        return True

    def disengage(self) -> bool:
        """Puts the arm in a passive mode it has (gravity compensation, zero torque), if it has one.

        @return False when it has none: the wrapper then holds the arm.
        """
        return False

    def why_stopped(self) -> str:
        """Why the driver stopped answering (is_alive() False), for messages.

        @return The reason, or empty when the driver does not know.
        """
        return ""

    def alarm(self) -> str | None:
        """Why the run must stop although the arm still answers (a motor too hot, a joint blocked), if it must.

        @return The reason, naming the motor and what to do.
        """
        return None

    def gripper_deg(self) -> float | None:
        """The gripper motor's angle as of the last read(), degrees, before it is mapped to 0 closed to 1 open.

        Only the setup wizard asks, from its own thread, to measure the gripper's ends by hand into the arm's
        `gripper_closed_deg` and `gripper_open_deg` (`robots.driver_template`); return what read() kept.

        @return None when the driver reports none: its gripper's range is its own, and the wizard skips that step.
        """
        return None


class Arm(Protocol):
    """What the station, the loop and the home move use of an arm: a #FollowerArm, or one served from a process of
    its own (`robots.sys0.arm_process.RemoteArm`, with System 0).

    @see FollowerArm
    """

    name: str
    home_rad: npt.NDArray[np.float64]
    dof: int

    def read(self) -> types.ArmState:
        """The measured state and the last commanded targets (`FollowerArm.read()`)."""
        ...

    def write(self, command: types.ArmCommand) -> None:
        """Commands joint targets and the gripper (`FollowerArm.write()`).

        @param command A target per joint in radians and a gripper value in `[0, 1]`.
        """
        ...

    def hold(self) -> None:
        """Re-commands the last target, or the measured pose (`FollowerArm.hold()`)."""
        ...

    def disengage(self) -> None:
        """Puts the arm in its driver's passive mode (`FollowerArm.disengage()`)."""
        ...

    def is_alive(self) -> bool:
        """Whether the arm still answers (`FollowerArm.is_alive()`)."""
        ...

    def close(self) -> None:
        """Releases the driver (`FollowerArm.close()`)."""
        ...

    def torque_off(self) -> None:
        """Switches the arm off through its driver (`FollowerArm.torque_off()`)."""
        ...

    @property
    def drops_when_limp(self) -> bool:
        """Whether disengage() lets the arm fall (`FollowerArm.drops_when_limp`)."""
        ...

    def alarm(self) -> str | None:
        """Why the run must stop although the arm answers (`FollowerArm.alarm()`)."""
        ...

    def holds_dead(self) -> bool:
        """Whether the arm stopped answering while it held (`FollowerArm.holds_dead()`)."""
        ...

    def why_stopped(self) -> str:
        """Why the arm reads dead (`FollowerArm.why_stopped()`)."""
        ...

    def close_for_reopen(self) -> None:
        """Lets the arm go so it can be opened again (`FollowerArm.close_for_reopen()`)."""
        ...


class FollowerArm:
    """One follower arm: read its state, write joint targets, hold, disengage, switch off.

    Attributes:
    - name: `"left"` or `"right"`; used in messages only.
    - home_rad: The arm's home joints, shape `[dof]`, radians (`home` in its config section); `robots.home` moves the
      arm there with the gripper open.
    - dof: The arm's joints: as many as its home lists, 1 to #MAX_JOINTS. Every read and command has this many.
    """

    def __init__(self, name: str, driver: Driver, *, home_rad: npt.ArrayLike) -> None:
        """Binds a name and a home to an opened driver; nothing is commanded here.

        @param name `"left"` or `"right"`, for messages.
        @param driver The arm's opened #Driver.
        @param home_rad The arm's home joints, radians: one per joint, which is how many joints the arm has.
        @throws ArmError `home_rad` is not 1 to #MAX_JOINTS finite values.
        """
        self.name = name
        self.home_rad = _joints(name, "home", home_rad)
        self.dof: int = self.home_rad.shape[0]
        self._driver = driver
        # None until the first write(); read() reports it as commanded_* so consumers can
        # tell "never commanded" from "commanded to zero".
        self._last_command: types.ArmCommand | None = None
        self._is_closed = False

    def read(self) -> types.ArmState:
        """Reads the arm in one driver call.

        @return The measured state plus the last commanded targets (None before any write).
        @throws ArmError The arm is closed or its driver stopped, or the driver's reading is not #dof joints.
        """
        self._require_alive()
        reading = self._driver.read()
        if reading.positions_rad.shape != (self.dof,) or reading.velocities_rad_s.shape != (self.dof,):
            raise errors.ArmError(
                f"arm '{self.name}' driver read {reading.positions_rad.shape} positions and "
                f"{reading.velocities_rad_s.shape} velocities, expected ({self.dof},) each: the arm has {self.dof} "
                "joints, as many as its home in the config lists"
            )
        last = self._last_command
        return types.ArmState(
            positions_rad=reading.positions_rad,
            velocities_rad_s=reading.velocities_rad_s,
            gripper=reading.gripper,
            commanded_rad=None if last is None else last.positions_rad,
            commanded_gripper=None if last is None else last.gripper,
            gripper_effort=reading.gripper_effort,
            motor_temps_c=reading.motor_temps_c,
        )

    def write(self, command: types.ArmCommand) -> None:
        """Hands the driver joint targets and the gripper, and remembers them for `commanded_*`.

        The gripper value is written as it is: what the loop hands down is what reaches the driver. How hard a
        commanded close ends up squeezing is regulated a level up, by #robots.gripper_force.GripperForceCap on
        the measured effort, not by anything this wrapper does to the number.

        @param command A target per joint in radians and a gripper value in `[0, 1]`.
        @throws ArmError The arm is closed or its driver stopped, the targets are not a finite `[dof]` vector, or the
            gripper is out of range.
        """
        command = validated(self.name, self.dof, command)
        self._require_alive()
        self._driver.write(command.positions_rad, command.gripper)
        self._last_command = command

    def write_motor_command(
        self,
        target: npt.NDArray[np.float64],
        kp: npt.NDArray[np.float64],
        kd: npt.NDArray[np.float64],
        torque: npt.NDArray[np.float64],
    ) -> None:
        """Hands the driver one whole MIT-mode block, System 0's command, and remembers its target.

        The target is remembered as the last command, so `commanded_*` reports it and hold() holds there on the PD.

        @param target Position targets, `[dof + 1]`: the joints in radians, then the gripper in `[0, 1]`.
        @param kp Stiffness per joint, `[dof]`.
        @param kd Damping per joint, `[dof]`.
        @param torque Torque per joint on top of gravity compensation, `[dof]`, newton metres.
        @throws ArmError The arm is closed or its driver stopped, or the driver takes no MIT block (not an i2rt YAM).
        """
        self._require_alive()
        write: Callable[..., None] | None = getattr(self._driver, "write_motor_command", None)
        if write is None:
            raise errors.ArmError(f"arm '{self.name}': the driver has no MIT command block to write")
        write(target, kp, kd, torque)
        self._last_command = types.ArmCommand(
            positions_rad=np.array(target[: self.dof]), gripper=min(max(float(target[self.dof]), 0.0), 1.0)
        )

    def set_motor_loop_hook(self, hook: Callable[[], None] | None) -> None:
        """Runs `hook` on the driver's motor loop before every update, or removes it; a driver without one takes none.

        @param hook A fast check to run before each update, or None to remove the one installed.
        """
        set_hook: Callable[[Any], None] | None = getattr(self._driver, "set_motor_loop_hook", None)
        if set_hook is not None:
            set_hook(hook)

    def hold(self) -> None:
        """Re-commands the last written targets, or the current position if none was written.

        Safe to call on every stall tick: a target that is already set costs one message.

        @throws ArmError The driver's reading is malformed (only on the never-written path).
        """
        if self._last_command is not None:
            self.write(self._last_command)
            return
        state = self.read()
        # A measured gripper can sit a hair outside [0, 1] from sensor noise; holding must
        # never fail validation, so clip rather than raise.
        gripper = min(max(state.gripper, 0.0), 1.0)
        self.write(types.ArmCommand(positions_rad=state.positions_rad, gripper=gripper))

    @property
    def drops_when_limp(self) -> bool:
        """Whether disengage() lets the arm fall (`Driver.drops_when_limp`)."""
        return self._driver.drops_when_limp

    def alarm(self) -> str | None:
        """Why the run must stop although the arm answers, from its driver (`Driver.alarm()`).

        @return `"arm '<name>': <reason>"`, or None while the arm may run.
        """
        reason = self._driver.alarm()
        return None if reason is None else f"arm '{self.name}': {reason}"

    def gripper_deg(self) -> float | None:
        """The gripper motor's angle as of the driver's last read, degrees (`Driver.gripper_deg()`).

        @return None when the driver reports none.
        @throws ArmError The arm is closed or its driver stopped.
        """
        self._require_alive()
        return self._driver.gripper_deg()

    def is_alive(self) -> bool:
        """Whether the arm still answers: not closed, and its driver alive.

        @return False once close() or torque_off() ran or the driver stopped (a motor stopped answering, the bus went
            away); read() and write() raise then, and a station reports it per arm and offers a reset.
        """
        return not self._is_closed and self._driver.is_alive()

    def holds_dead(self) -> bool:
        """Whether the arm stopped answering while it held and was not let go here.

        @return True when it drops when limp.
        """
        return self.drops_when_limp and not self._is_closed and not self._driver.is_alive()

    def why_stopped(self) -> str:
        """Why the arm reads dead (is_alive() False), for messages.

        @return Its driver's reason, or "its driver stopped answering" when the driver does not know.
        """
        return self._driver.why_stopped() or "its driver stopped answering"

    def close(self) -> None:
        """Releases the driver without moving the arm, so the arm can be reopened.

        The arm reads dead from here on and refuses reads and writes; `StationController.reset()` opens a fresh one.
        """
        self._is_closed = True
        self._driver.close()

    def torque_off(self) -> None:
        """Switches the arm off through its driver: the motors go limp (a brakeless arm parks first).

        For process exit, never for a stop: a stopped run holds (#robots.control.ControlLoop) so the arm stays where
        the policy left it. A brakeless arm that stopped answering cannot park, so it is left holding under power and
        the driver raises to say so (switched off where it holds, it would drop).
        """
        self._is_closed = True
        self._driver.torque_off()

    def close_for_reopen(self) -> None:
        """Lets the arm go so it can be opened again: switched off when it would drop (torque_off()), else closed."""
        if self.drops_when_limp and self.is_alive():
            self.torque_off()
        else:
            self.close()

    def disengage(self) -> None:
        """Puts the arm in its driver's passive mode, or holds it when the driver has none."""
        if not self._driver.disengage():
            self.hold()

    def _require_alive(self) -> None:
        """Raises unless the arm is open and its driver alive."""
        if self._is_closed:
            raise errors.ArmError(f"arm '{self.name}' driver is closed; reset the arms to reopen it")
        if not self._driver.is_alive():
            raise errors.ArmError(
                f"arm '{self.name}' stopped: {self.why_stopped()}. Check the arm's power and the cable to its "
                "adapter, then open the arm again"
            )


def validated(name: str, dof: int, command: types.ArmCommand) -> types.ArmCommand:
    """Checks a command as `FollowerArm.write()` does, so an arm served from another process refuses it here.

    @param name `"left"` or `"right"`, for the message.
    @param dof The arm's joints (`FollowerArm.dof`).
    @param command A target per joint in radians and a gripper value in `[0, 1]`.
    @return The command with its targets as a float64 `[dof]` array and its gripper as a float.
    @throws ArmError The targets are not a finite `[dof]` vector, or the gripper is out of range.
    """
    positions_rad = np.asarray(command.positions_rad, dtype=np.float64)
    if positions_rad.shape != (dof,):
        raise errors.ArmError(
            f"arm '{name}' command has shape {positions_rad.shape}, expected ({dof},): the arm has {dof} joints"
        )
    if not np.all(np.isfinite(positions_rad)):
        raise errors.ArmError(f"arm '{name}' command contains a non-finite joint target: {positions_rad}")
    gripper = float(command.gripper)
    # A NaN gripper fails both comparisons, so the range check also rejects it.
    if not 0.0 <= gripper <= 1.0:
        raise errors.ArmError(f"arm '{name}' gripper command {gripper} is outside [0, 1]")
    return types.ArmCommand(positions_rad=positions_rad, gripper=gripper)


def _joints(name: str, what: str, value: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """1 to #MAX_JOINTS finite joint values as a read-only float64 array.

    @param name The arm, for the message.
    @param what What the values are, for the message.
    @param value The values.
    @return The array.
    @throws ArmError Not 1 to #MAX_JOINTS finite values.
    """
    joints = np.array(value, dtype=np.float64)
    if joints.ndim != 1 or not 1 <= joints.shape[0] <= MAX_JOINTS or not np.all(np.isfinite(joints)):
        raise errors.ArmError(
            f"arm '{name}' {what} must be 1 to {MAX_JOINTS} finite joint values, one per joint, got {joints}"
        )
    joints.setflags(write=False)
    return joints
