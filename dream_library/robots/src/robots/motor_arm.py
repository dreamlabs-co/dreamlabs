"""One thread per arm on a CAN motor bus: #MotorArm, the shared driver of every bus arm, over its robot's table.

A robot on a motor bus is a short table, an #ArmSpec (#robots.makers, #robots.metal, #robots.so101): its motors as
#Joint rows (name, bus id, motor model, gains, soft limits), its tick rate, its gripper's stroke and where it parks.
What every such arm does is here, once:

- the tick at the robot's rate: take the newest target, step the command one tick toward it, send one frame per motor
  through the arm's #Chain, publish the pose the chain reports;
- the startup ramp: the command is seeded from the measured pose and eases onto a new target at #SYNC_SPEED_DEG_S until
  every joint and the gripper are within tolerance (or its time budget runs out), then tracks at the per-tick cap
  #MAX_SPEED_DEG_S (#_Stepper); a target older than #TARGET_STALE_S is dropped and the ramp re-armed;
- the soft limits on every target, and the gripper's stroke, linear in the motor angle between the arm's
  configured ends (#LinearStroke); no gripper goal comes nearer its open end than `ArmSpec.grip_margin_mm`, so the
  gripper never rests pressed into its stop;
- faults: an error the chain calls transient (one motor's fault word) holds the arm and the tick goes on; anything else
  latches the arm down: it then reads dead and its motors hold their last frame, and torque-off leaves them so (it
  cannot park, and switched off it would drop; torque_off() says to support it and cut its power). A pose older than
  #POSE_STALE_S (a fault word that does not clear, a bus call that does not return) is refused, not read;
- the guard (#robots.motor_guard.MotorGuard): a blocked joint or gripper is held instead of pushed harder, and a motor
  too hot or a joint blocked too long raises the arm's alarm(), which ends a run the safe way while the arm answers;
  it watches an arm holding with no fresh target too (between runs, after a home move);
- the park before torque-off (`ArmSpec.rest_rad`), since these arms have no brakes;
- limp (disengage()): every motor switched off where the arm is and the pose read on, so a hand can move it (the
  setup wizard's steps by hand).

What differs per bus is its #Chain (#robots.robstride.RobstrideChain, #robots.damiao.DamiaoChain,
#robots.feetech.FeetechChain): the handshake and the pose checks at open, how a tick reads the arm, how the frames go
out (the gripper's holding law included), and how the motors switch off.

Units: the loop's `[slots joints rad, gripper 0..1]` at read() and write(); the tick's `[slots joints rad, gripper
stroke mm]` (the stroke is what the arm's datasets record, #LinearStroke); motor degrees at the chain. An arm has as
many joint slots as its `home` in the config lists (`MotorArm.slots`), at least as many as its chain has joints. A
joint is its motor's angle in radians, with no sign or offset: on every arm here the motor frame is the URDF's
(dream_ik's `maker_arm_ik.urdf`, the Metal URDF and the SO-101's), so the loop's units are the URDF's. The slots past
the chain's joints (the SO-101's sixth, after its five) read 0 and are held there, as dream's IK holds the SO-101's
dummy sixth joint.

The algorithm is the Makers Arm's reference driver (its follower and command stepper); `tests/test_makers_parity.py`
holds it to that driver's frames byte for byte.

@see open_arm(), MotorArm, ArmSpec, Chain
"""

import dataclasses
import logging
import math
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import numpy.typing as npt

from robots import errors, followers, motor_guard

if TYPE_CHECKING:
    # Annotations only: config imports the Metal gripper, which imports the buses, which import this module.
    from robots import config

## Deprecated: the revolute joints of a six-joint arm; an arm's own count is its config's.
ARM_DOF = followers.JOINT_COUNT

# The thread's rates, in physical units; the thread turns them into per-tick steps at the robot's tick rate.

## Startup ramp onto a new target, degrees per second (1 degree per 30 Hz tick).
SYNC_SPEED_DEG_S = 30.0
## Startup ramp of the gripper, mm per second.
SYNC_SPEED_MM_S = 40.0
## The ramp ends once every joint is this close to the target, degrees,
SYNC_TOL_DEG = 3.0
## ... and the gripper this close, mm.
SYNC_TOL_MM = 5.0
## Slack added to the ramp's time budget, seconds: a target that keeps moving gets the time the initial gap takes at
## the ramp speed plus this, then tracking goes to the full (still capped) rate.
SYNC_GRACE_S = 1.0
## The per-tick speed cap while tracking, degrees per second (the Makers Arm's policy cap; an SO-101 bridge
## caps at about the same).
## dream's IK holds the Makers and the Metal joints under the same 120 degrees per second.
MAX_SPEED_DEG_S = 120.0
## The gripper's cap while tracking, mm per second.
MAX_SPEED_MM_S = 120.0
## The gripper's speed on the way to rest, mm per second (the joints' is the robot's, `ArmSpec.rest_speed_deg_s`).
REST_SPEED_MM_S = 40.0
## "At rest" on the measured pose: every joint within this, radians,
REST_TOL_RAD = math.radians(5.0)
## ... and the gripper within this, mm.
REST_TOL_MM = 8.0
## A park's time limit is its own travel time at the rest speeds plus this, seconds,
REST_TIMEOUT_MARGIN_S = 5.0
## ... and never under this.
REST_TIMEOUT_S = 12.0
## A park that stops getting closer for this long ends there, seconds.
REST_STALL_S = 2.0
## A target older than this is dropped and the ramp re-armed, seconds.
TARGET_STALE_S = 0.5
POSE_STALE_S = 1.0
## How long open_arm() waits for the thread's first pose, seconds.
FIRST_POSE_TIMEOUT_S = 10.0
## How long stopping the thread waits for it, seconds.
_JOIN_TIMEOUT_S = 2.0

_log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Joint:
    """One motor of a bus arm: a row of its robot's table.

    Attributes:
    - name: The joint, for messages (`"shoulder_pan"`, ..., `"gripper"`).
    - motor_id: Its id on the bus (a CAN id, or a Feetech servo's id).
    - model: Its motor model, a key of its bus's `MOTOR_LIMITS` (`"sts3215"` on a Feetech bus).
    - kp: Its position gain: MIT, N.m per rad, on a CAN bus; the servo's P register on a Feetech bus.
    - kd: Its velocity gain: MIT, N.m s per rad, on a CAN bus; the servo's D register on a Feetech bus.
    - limits_deg: Its soft limits `(low, high)`, degrees in the URDF's frame.
    """

    name: str
    motor_id: int
    model: str
    kp: float
    kd: float
    limits_deg: tuple[float, float]


@dataclasses.dataclass(frozen=True)
class MotorReading:
    """One arm's pose in motor degrees, base to gripper, with each motor's velocity and torque.

    Attributes:
    - q_deg: Positions, degrees, the chain's joint order.
    - velocity_deg_s: Velocities, degrees per second.
    - torque_nm: Torques, N.m.
    - temperatures_c: Per motor, its sensors by name (`"coil"`, `"driver board"`, ...), degrees C; None when this read
      carried none (a chain may read them less often than the pose).
    """

    q_deg: tuple[float, ...]
    velocity_deg_s: tuple[float, ...]
    torque_nm: tuple[float, ...]
    temperatures_c: tuple[dict[str, float], ...] | None = None


class Chain(Protocol):
    """One arm's motors on its bus, in motor degrees: what #MotorArm drives.

    Attributes:
    - name: `"left"` or `"right"`, for messages.
    - names: The motors' names, base to gripper (the gripper last): the joints, then the gripper.
    - limits_deg: Soft limits per motor, degrees.
    - transient: The errors that hold the arm for a tick rather than latch it down.
    - holds_grip: Whether the chain sets how a closed gripper squeezes (a hold torque, a gain the config chose);
      without that the guard holds it in a gentle squeeze (#robots.motor_guard).
    - alarm: Why the run must stop while the arm still answers (a Metal gripper that tripped), empty while it may run.

    @see open(), enable(), read(), command(), rearm(), relax(), close(), drops()
    """

    name: str
    names: tuple[str, ...]
    limits_deg: dict[str, tuple[float, float]]
    transient: tuple[type[Exception], ...]
    holds_grip: bool
    alarm: str

    def open(self) -> None:
        """Opens the bus, checks the motors and the pose; energizes nothing, and closes the bus again on failure."""

    def enable(self) -> None:
        """Energizes the motors; each holds where it is."""

    def read(self) -> MotorReading:
        """The pose this tick starts from."""

    def command(self, goal_deg: list[float]) -> MotorReading:
        """Sends one frame per motor toward a goal.

        @param goal_deg One target per motor, degrees, base to gripper.
        @return The pose to publish.
        """

    def rearm(self) -> None:
        """A new stream is about to start: what the chain eases in by itself starts over."""

    def relax(self) -> None:
        """Switches every motor off and keeps the bus open: read() goes on, command() is not called again."""

    def close(self, *, disable_torque: bool) -> None:
        """Closes the bus.

        @param disable_torque Switch every motor off first.
        @throws Exception The switch-off is not confirmed, after the bus is closed.
        """

    def drops(self) -> int:
        """Replies that never came, every motor together."""


class LinearStroke:
    """The gripper's stroke, linear in its motor angle between the arm's two ends (`gripper_open_deg`,
    `gripper_closed_deg`): every bus arm's gripper.

    Attributes:
    - closed_mm: 0.
    - open_mm: The stroke at the open end.
    """

    def __init__(self, open_deg: float, closed_deg: float, open_mm: float) -> None:
        """@param open_deg The motor angle at the open end, degrees.
        @param closed_deg The motor angle at the closed end, degrees.
        @param open_mm The stroke at the open end, mm.
        """
        self.open_deg, self.closed_deg = open_deg, closed_deg
        self.closed_mm, self.open_mm = 0.0, open_mm

    def to_mm(self, deg: float) -> float:
        """The stroke at a motor angle, clipped to the ends.

        @param deg The gripper motor's angle, degrees.
        @return The stroke, mm.
        """
        span = self.open_deg - self.closed_deg
        fraction = (deg - self.closed_deg) / span if span else 0.0
        return float(min(1.0, max(0.0, fraction)) * self.open_mm)

    def to_deg(self, mm: float) -> float:
        """The motor angle of a stroke, the stroke clipped first.

        @param mm The stroke, mm.
        @return The gripper motor's angle, degrees.
        """
        fraction = min(1.0, max(0.0, float(mm) / self.open_mm))
        return self.closed_deg + fraction * (self.open_deg - self.closed_deg)


@dataclasses.dataclass(frozen=True)
class ArmSpec:
    """One robot on a motor bus, as a table: everything #MotorArm needs to know about it.

    Attributes:
    - name: The robot, for the thread's name and messages (`"makers"`).
    - ik: dream's IK model of the arm (`dream_ik`), the name the client's `ik` and #robots.kinematics.DreamKinematics
      take.
    - tick_hz: The thread's rate: one frame per motor per tick.
    - rest_speed_deg_s: How fast the arm parks before torque-off, degrees per second on every joint.
    - rest_rad: Where it parks: a joint angle per slot (`MotorArm.slots`) in radians, a pose it may go limp in; None
      parks it where it was opened.
    - grip_scale_mm: The stroke, mm, the loop's gripper 1.0 stands for (what the robot's datasets record): the
      stroke at the arm's `gripper_open_deg`.
    - grip_margin_mm: How far inside the open end every gripper goal stays, mm of the stroke: home, the park, a hold,
      the policy's 1.0. The open end is where the gripper stood pushed open by hand (the setup wizard, a LeRobot
      calibration) or the vendor's limit, so a goal on it presses the gripper into its stop for as long as it holds
      (an SO-101 gripper overloads after some minutes held open at home). The closed end stays reachable, since
      a grasp must close on a thin object; the hold law or the guard's squeeze bounds that push.
    - temp_warn_c: A motor sensor at this is logged once, degrees C (#robots.motor_guard), and the arm refuses to open
      until it is back under (open_arm()).
    - temp_stop_c: A motor sensor at this stops the run and the arm parks, degrees C: under the motors' own fault
      temperature, which switches a motor off wherever the arm is.
    - chain: The arm's #Chain: `chain(side, cfg, opener=..., clock=..., sleep=...)`.
    """

    name: str
    ik: str
    tick_hz: float
    rest_speed_deg_s: float
    rest_rad: tuple[float, ...] | None
    grip_scale_mm: float
    grip_margin_mm: float
    temp_warn_c: float
    temp_stop_c: float
    chain: Callable[..., Chain]


def _slots(spec: ArmSpec, side: str, home: tuple[float, ...], joints: int) -> int:
    """An arm's joint slots: as many as its home lists, checked against its chain and the robot's rest pose.

    @param spec The robot's table.
    @param side `"left"` or `"right"`, for messages.
    @param home The arm's home in the config (`config.ArmConfig.home`).
    @param joints The joints its chain drives.
    @return `len(home)`.
    @throws ConfigError The home lists fewer angles than the chain has joints or not 0 past them, or the robot's rest
        pose is not as long as the home.
    """
    slots = len(home)
    if slots < joints or any(home[joints:]):
        raise errors.ConfigError(
            f"arm '{side}': its home in the config lists {slots} angles {list(home)}, but the {spec.name} arm has "
            f"{joints} joints: list one angle per joint, and 0 for any slot past them, which the arm holds"
        )
    if spec.rest_rad is not None and len(spec.rest_rad) != slots:
        raise errors.ConfigError(
            f"arm '{side}': its home in the config lists {slots} angles, but the {spec.name} arm parks at "
            f"{len(spec.rest_rad)} (its rest pose): list {len(spec.rest_rad)}"
        )
    return slots


def _clamp(value: float, low: float, high: float) -> float:
    """`value` within `[low, high]`."""
    return low if value < low else (high if value > high else value)


class _Stepper:
    """The command state of one arm in `[slots joints rad, gripper mm]`, stepped one tick at a time.

    Attributes:
    - slots: The arm's joint slots (`MotorArm.slots`); the gripper is the one after them.
    """

    def __init__(self, joint_limits_rad: list[tuple[float, float]], stroke_mm: tuple[float, float], hz: float) -> None:
        """Sets the per-tick steps from the thread's rates.

        @param joint_limits_rad Each joint slot's soft limits, radians.
        @param stroke_mm The gripper's goal range, `(closed, most open)` mm.
        @param hz The tick rate the steps are per.
        """
        self.sync_step_rad = math.radians(SYNC_SPEED_DEG_S / hz)
        self.sync_step_mm = SYNC_SPEED_MM_S / hz
        self.sync_tol_rad = math.radians(SYNC_TOL_DEG)
        self.sync_tol_mm = SYNC_TOL_MM
        self.max_step_rad = math.radians(MAX_SPEED_DEG_S / hz)
        self.max_step_mm = MAX_SPEED_MM_S / hz
        self.joint_limits = joint_limits_rad
        self.slots = len(joint_limits_rad)
        self.stroke_mm = stroke_mm
        self.cmd: list[float] | None = None
        self.synced = False

    def seed(self, measured: list[float]) -> None:
        """Makes the measured pose the command, the gripper inside its goal range; a no-op once seeded.

        @param measured The arm's measured `[slots joints rad, gripper mm]`.
        """
        if self.cmd is None:
            g = self.slots
            self.cmd = [float(x) for x in measured[:g]] + [_clamp(float(measured[g]), *self.stroke_mm)]

    def mark_stale(self) -> None:
        """Re-arms the ramp; the command holds."""
        self.synced = False

    def sanitize(self, target: list[float]) -> list[float]:
        """The target clamped to the joints' soft limits and the gripper's goal range.

        @param target `[slots joints rad, gripper mm]`.
        @return The clamped target, a new list.
        """
        out = [_clamp(float(target[i]), *self.joint_limits[i]) for i in range(self.slots)]
        out.append(_clamp(float(target[self.slots]), *self.stroke_mm))
        return out

    def _advance(self, goal: list[float], step_rad: float, step_mm: float) -> tuple[float, float]:
        """Moves the command toward a clamped goal by at most one step; the errors before the step."""
        assert self.cmd is not None, "the stepper is seeded before it steps"
        max_err_rad = 0.0
        g = self.slots
        for i in range(g):
            err = goal[i] - self.cmd[i]
            max_err_rad = max(max_err_rad, abs(err))
            self.cmd[i] += _clamp(err, -step_rad, step_rad)
        grip_err = goal[g] - self.cmd[g]
        self.cmd[g] += _clamp(grip_err, -step_mm, step_mm)
        return max_err_rad, abs(grip_err)

    def step(self, target: list[float]) -> list[float]:
        """One tick toward `target`, at the ramp rate until synced and at the tracking rate after.

        @param target `[slots joints rad, gripper mm]`, clamped first.
        @return The command after the step.
        """
        goal = self.sanitize(target)
        step_rad = self.sync_step_rad if not self.synced else self.max_step_rad
        step_mm = self.sync_step_mm if not self.synced else self.max_step_mm
        max_err_rad, grip_err = self._advance(goal, step_rad, step_mm)
        if not self.synced and max_err_rad <= self.sync_tol_rad and grip_err <= self.sync_tol_mm:
            self.synced = True
        return self.hold()

    def step_toward(self, goal: list[float], step_rad: float, step_mm: float) -> list[float]:
        """One tick toward `goal` at an explicit rate (the park); leaves `synced` alone.

        @param goal `[slots joints rad, gripper mm]`, clamped first.
        @param step_rad The most a joint moves this tick, radians.
        @param step_mm The most the gripper moves this tick, mm.
        @return The command after the step.
        """
        self._advance(self.sanitize(goal), step_rad, step_mm)
        return self.hold()

    def hold(self) -> list[float]:
        """The command, unchanged.

        @return A copy of the command.
        """
        assert self.cmd is not None, "the stepper is seeded before it holds"
        return list(self.cmd)


class MotorArm(followers.Driver):
    """The #robots.followers.Driver of an arm on a motor bus: its own thread at the robot's rate over its #Chain.

    See the module docstring for the tick. Every method but the thread's own is safe from any thread.

    Attributes:
    - spec: The robot's #ArmSpec.
    - name: `"left"` or `"right"`, for messages.
    - chain: The arm's #Chain.
    - slots: The arm's joints as the loop counts them: as many as its home in the config lists
      (`robots.followers.FollowerArm.dof`). Every read and command has this many.
    - joints: How many of the slots the arm's motors turn (#Chain names them), the first ones; the rest read 0 and
      hold there.
    - stroke: The gripper's #LinearStroke.
    - open_goal_mm: The most open goal the gripper is given, mm: `ArmSpec.grip_margin_mm` inside its open end.
    - period: Seconds per tick.
    - stepper: The command state.
    - fault: Why the arm is latched down; empty while it runs.
    - rest: Where the arm parks, `[slots joints rad, gripper mm]` (`ArmSpec.rest_rad`, the gripper at
      #open_goal_mm); None until the first tick when the arm parks where it was opened.
    - rest_result: How the last park ended, `(reached, reason)`, or None.
    - rest_budget_s: The running (or last) park's time limit, seconds.
    - rested: Set when a park ends.
    - first_pose: Set when the first pose is published.
    - limp: Set once the motors are off after disengage(), or from the start on an arm opened read only.
    - guard: The arm's #robots.motor_guard.MotorGuard.
    """

    drops_when_limp = True

    def __init__(
        self,
        spec: ArmSpec,
        side: str,
        cfg: "config.ArmConfig",
        *,
        opener: Callable[..., Any],
        clock: Callable[[], float],
        sleep: Callable[[float], Any],
    ) -> None:
        """Builds the arm's chain and the tick's state; nothing is opened, and the thread starts with start().

        @param spec The robot's table.
        @param side `"left"` or `"right"`.
        @param cfg The arm's config section.
        @param opener Opens the arm's bus, called as its chain calls it (#robots.robstride.open_can_bus,
            #robots.feetech.open_bus, a fake bus in tests).
        @param clock Monotonic clock, seconds.
        @param sleep Sleeps that many seconds.
        @throws ConfigError The config lacks what the robot's chain needs, its home lists fewer angles than the chain
            has joints, or not 0 for the slots past them, or the robot's rest pose is not as long as the home.
        @throws ValueError The chain has no joint.
        """
        self.spec = spec
        self.name = side
        self.stroke = LinearStroke(cfg.gripper_open_deg, cfg.gripper_closed_deg, spec.grip_scale_mm)
        self.open_goal_mm = self.stroke.open_mm - spec.grip_margin_mm
        self.chain = spec.chain(side, cfg, opener=opener, clock=clock, sleep=sleep)
        self.joints = len(self.chain.names) - 1
        if self.joints < 1:
            raise ValueError(f"{spec.name}: a chain drives at least one joint and a gripper, got {self.chain.names}")
        self.slots = _slots(spec, side, cfg.home, self.joints)
        # The slots past the arm's joints hold at 0.
        self._pad = (0.0,) * (self.slots - self.joints)
        self.period = 1.0 / spec.tick_hz
        limits = [self.chain.limits_deg[joint] for joint in self.chain.names[: self.joints]]
        limits += [(0.0, 0.0)] * (self.slots - self.joints)
        self.stepper = _Stepper(
            [(math.radians(low), math.radians(high)) for low, high in limits],
            (self.stroke.closed_mm, self.open_goal_mm),
            spec.tick_hz,
        )
        self.guard = motor_guard.MotorGuard(
            side,
            self.chain.names,
            self.joints,
            temp_warn_c=spec.temp_warn_c,
            temp_stop_c=spec.temp_stop_c,
            holds_grip=self.chain.holds_grip,
            to_deg=self.stroke.to_deg,
            clock=clock,
        )
        self._rest_step_rad = math.radians(spec.rest_speed_deg_s / spec.tick_hz)
        self._rest_step_mm = REST_SPEED_MM_S / spec.tick_hz
        self.rest: list[float] | None = None if spec.rest_rad is None else [*spec.rest_rad, self.open_goal_mm]
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"{spec.name}-{side}", daemon=True)
        self._measured: tuple[npt.NDArray[np.float32], npt.NDArray[np.float64]] | None = None
        self._measured_at = 0.0
        self._target: npt.NDArray[np.float64] | None = None
        self._t_target = 0.0
        self._resting = False
        self._parked = False
        self._held: list[float] | None = None
        self.rest_result: tuple[bool, str] | None = None
        self._rest_started = 0.0
        self._rest_progressed = 0.0
        self._rest_best: tuple[float, float] | None = None
        self.rest_budget_s = REST_TIMEOUT_S
        self.rested = threading.Event()
        self.first_pose = threading.Event()
        self.fault = ""
        self._tick_error = ""
        self._sync_deadline: float | None = None
        self._sync_announced = False
        self._limp = False
        self._off = False
        self._off_tried = False
        self.limp = threading.Event()

    # The loop's side: followers.Driver

    def read(self) -> followers.Reading:
        """The newest published pose in the loop's units; the gripper is the stroke over `ArmSpec.grip_scale_mm`.

        @return The #robots.followers.Reading.
        """
        q, velocity = self.latest()
        return followers.Reading(
            positions_rad=np.asarray(q[: self.slots], dtype=np.float64),
            velocities_rad_s=velocity,
            gripper=float(q[self.slots]) / self.spec.grip_scale_mm,
            motor_temps_c=self.guard.motor_temps_c,
        )

    def write(self, positions_rad: npt.NDArray[np.float64], gripper: float) -> None:
        """Hands the thread a new target; its tick steps toward it.

        @param positions_rad A target per slot, radians.
        @param gripper The gripper target, 0 closed to 1 open.
        """
        self.set_target(np.append(positions_rad, gripper * self.spec.grip_scale_mm))

    def disengage(self) -> bool:
        """Lets the arm go limp where it is, with no park, so a hand can move it while read() follows it.

        The next tick switches every motor off (`Chain.relax`) and the ticks after only read. It has no brakes, so it
        drops unless supported. It stays limp until closed; opening it again energizes it.

        @return True once the motors are off.
        @throws ArmError The thread did not switch them off (latched down, or no tick in time).
        """
        with self._lock:
            self._limp = True
        if not self.limp.wait(timeout=_JOIN_TIMEOUT_S):
            raise errors.ArmError(f"arm '{self.name}' did not go limp: {self.fault or 'no tick in time'}")
        return True

    def alarm(self) -> str | None:
        """Why the run must stop while the arm answers, if it must.

        @return The reason, or None.
        """
        return self.guard.alarm(self.chain.alarm)

    def is_alive(self) -> bool:
        """Whether the tick runs and nothing latched the arm down.

        @return False once the thread stopped or #fault is set.
        """
        return self._thread.is_alive() and not self.fault

    def why_stopped(self) -> str:
        """#fault, else that the thread ended.

        @return The reason.
        """
        return self.fault or ("" if self._thread.is_alive() else "its motor thread ended")

    def close(self) -> None:
        """Stops the thread and closes the bus without a frame more, so the motors keep their last target."""
        self.stop()
        self.chain.close(disable_torque=False)

    def torque_off(self) -> None:
        """Parks the arm at rest, then switches every motor off.

        The arm has no brakes, so torque off anywhere else drops it: a running arm first travels to rest on its
        measured pose and parks (bounded by its own travel time, #REST_STALL_S without progress), then every motor is
        switched off; a limp one (disengage()) is switched off where it is. An arm that stopped answering while it
        held (latched down before or during the park, or closed under power for a reopen) cannot park, so it is never
        switched off here: its bus is closed with the motors still holding and the error says so, as a station's Reset
        leaves it. An arm already switched off is left as it is; one whose switch-off did
        not confirm is switched off again, with no park (the chain opens its bus once more for it).

        @throws Exception The disable's failure, after the bus is closed: support the arm.
        """
        if self._off:
            _log.info("arm '%s': already switched off", self.name)
            return
        if self._off_tried:
            _log.warning("arm '%s': its last switch-off did not confirm; switching it off again", self.name)
        elif self._limp:
            _log.info("arm '%s': limp, switching off where it is", self.name)
        elif self.is_alive():
            _log.info("arm '%s': parking at rest before torque off", self.name)
            self.request_rest()
            deadline = time.monotonic() + self.rest_budget_s + _JOIN_TIMEOUT_S
            while not self.rested.wait(timeout=0.05) and self.is_alive() and time.monotonic() < deadline:
                pass
            result = self.rest_result if self.rested.is_set() else None
            reached, reason = result if result is not None else (False, "no answer")
            (_log.info if reached else _log.warning)("arm '%s': %s, switching off there", self.name, reason)
        if not (self._limp or self._off_tried) and not self.is_alive():
            why = self.why_stopped()
            self.stop()
            try:
                self.chain.close(disable_torque=False)
            except Exception as exc:
                _log.warning("arm '%s': closing its bus: %s", self.name, exc)
            raise errors.ArmError(
                f"arm '{self.name}' stopped answering while it held ({why}); it still holds under power so it does "
                "not drop: support it, then switch its power off"
            )
        self._off_tried = True
        self.stop()
        self.chain.close(disable_torque=True)
        self._off = True
        _log.info("arm '%s': torque off (missed replies: %d)", self.name, self.chain.drops())

    # The thread

    def start(self) -> None:
        """Starts the thread."""
        self._thread.start()

    def read_only(self) -> None:
        """Makes the thread only read the pose and publish it: before start(), on an arm never energized."""
        with self._lock:
            self._limp = True
        self.limp.set()

    def stop(self) -> None:
        """Ends the thread after its current tick and waits for it."""
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=_JOIN_TIMEOUT_S)

    def set_target(self, target: npt.NDArray[np.float64]) -> None:
        """Takes a new target; any thread.

        @param target `[slots joints rad, gripper mm]`.
        """
        with self._lock:
            self._target = np.array(target, dtype=np.float64)
            self._t_target = self._clock()

    def latest(self) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float64]]:
        """The newest published pose; any thread.

        @return `([slots joints rad, gripper mm] float32, [slots] joint velocities rad/s)`.
        """
        with self._lock:
            measured, measured_at = self._measured, self._measured_at
        if measured is None:
            raise errors.ArmError(f"arm '{self.name}' has published no pose yet")
        age = self._clock() - measured_at
        if age > POSE_STALE_S:
            raise errors.ArmError(
                f"arm '{self.name}' has published no fresh pose for {age:.1f} s "
                f"({self._tick_error or self.fault or 'its thread is stuck in a bus call'}); support the arm, then "
                "reset the arms"
            )
        return measured

    def request_rest(self) -> None:
        """Starts the park at the rest speeds; `rested` is set when it ends, and the arm holds there after."""
        with self._lock:
            if self._parked or self._resting:
                return
            self._resting = True
            self._rest_started = self._rest_progressed = self._clock()
            self._rest_best = None
            if self._measured is not None and self.rest is not None:
                self.rest_budget_s = self._rest_timeout_s(self._measured[0], self.rest)
            self.rested.clear()
            self._target = None

    def _run(self) -> None:
        """Ticks at the robot's rate until stopped or latched down."""
        while not self._stop_event.is_set() and not self.fault:
            t0 = self._clock()
            self.tick()
            self._sleep(max(0.0, self.period - (self._clock() - t0)))

    def tick(self) -> None:
        """One tick, with the fault policy.

        An error the chain calls transient is logged once and the arm holds; any other latches #fault. Nothing runs
        once latched.
        """
        if self.fault:
            return
        try:
            self._tick()
        except self.chain.transient as exc:
            if self._tick_error != str(exc):
                _log.error("arm '%s': motor fault, holding: %s", self.name, exc)
            self._tick_error = str(exc)
        except Exception as exc:
            self.fault = f"{type(exc).__name__}: {exc}"
            _log.exception("arm '%s': latched down: %s", self.name, self.fault)
        else:
            if self._tick_error:
                _log.info("arm '%s': ticking again (was: %s)", self.name, self._tick_error)
            self._tick_error = ""

    def _tick(self) -> None:
        """One tick: take the target, step the command, send it, publish the pose the chain reports."""
        with self._lock:
            limp = self._limp
        if limp:
            self._limp_tick()
            return
        goal: npt.NDArray[np.float64] | None = None
        with self._lock:
            target, t_target = self._target, self._t_target
        if target is not None:
            if self._clock() - t_target > TARGET_STALE_S:
                with self._lock:
                    self._target = None
                self._rearm_sync()
            else:
                goal = target
        before = self.chain.read()
        self.guard.temperatures(before.temperatures_c)
        measured = self._to_loop(before)
        self.stepper.seed([float(x) for x in measured])
        if self.rest is None:
            self.rest = [float(x) for x in measured[: self.slots]] + [self.open_goal_mm]
        with self._lock:
            resting, parked = self._resting, self._parked
        if resting:
            command = self._rest_tick(measured)
        elif parked:
            command = self.stepper.hold()
        elif goal is not None:
            self._held = None
            self._sync_budget(goal)
            command = self.stepper.step(self._guarded(goal, measured))
        else:
            command = self._hold_tick(measured)
        reading = self.chain.command(self._to_motors(command))
        published = measured if reading is before else self._to_loop(reading)
        velocity = np.radians(np.asarray(reading.velocity_deg_s[: self.joints] + self._pad, dtype=np.float64))
        with self._lock:
            self._measured, self._measured_at = (published, velocity), self._clock()
        self.first_pose.set()

    def _limp_tick(self) -> None:
        """A limp arm's tick: the motors switched off once, then the pose read and published; nothing is sent."""
        first = not self.limp.is_set()
        if first:
            self.chain.relax()
        with self._lock:
            resting, self._resting = self._resting, False
        if resting:
            self.rest_result = (False, "let go limp before it reached rest")
            self.rested.set()
        if first:
            self.limp.set()
            _log.info("arm '%s': limp, motors off; support the arm", self.name)
        reading = self.chain.read()
        self.guard.temperatures(reading.temperatures_c)
        velocity = np.radians(np.asarray(reading.velocity_deg_s[: self.joints] + self._pad, dtype=np.float64))
        with self._lock:
            self._measured, self._measured_at = (self._to_loop(reading), velocity), self._clock()
        self.first_pose.set()

    def _hold_tick(self, measured: npt.NDArray[np.float32]) -> list[float]:
        if self._held is None:
            self._held = self.stepper.hold()
        guarded = self._guarded(self._held, measured)
        return self.stepper.step_toward(guarded, self.stepper.sync_step_rad, self.stepper.sync_step_mm)

    def _guarded(self, goal: Any, measured: npt.NDArray[np.float32]) -> list[float]:
        return self.guard.adjust(
            self.stepper.sanitize([float(x) for x in goal]), [float(x) for x in measured], self.stepper.hold()
        )

    def _to_loop(self, reading: MotorReading) -> npt.NDArray[np.float32]:
        """Motor degrees as `[slots joints rad, gripper mm]`, float32 as the reference driver publishes them."""
        out = np.zeros(self.slots + 1, dtype=np.float32)
        for i in range(self.joints):
            out[i] = math.radians(float(reading.q_deg[i]))
        out[self.slots] = self.stroke.to_mm(float(reading.q_deg[self.joints]))
        return out

    def _to_motors(self, command: list[float]) -> list[float]:
        """`[slots joints rad, gripper mm]` as degrees per motor."""
        out = [math.degrees(float(command[i])) for i in range(self.joints)]
        out.append(self.stroke.to_deg(float(command[self.slots])))
        return out

    def _rearm_sync(self) -> None:
        """Re-arms the ramp: the next target eases in from the present command."""
        self.stepper.mark_stale()
        self._sync_deadline = None
        self._sync_announced = False
        self.chain.rearm()

    def _sync_budget(self, goal: npt.NDArray[np.float64]) -> None:
        """Bounds the startup ramp in time: the initial gap at the ramp speed plus #SYNC_GRACE_S.

        The stepper leaves its ramp only when every joint and the gripper are within tolerance on the same tick,
        which a target that keeps moving may never give; the ramp exists to close the initial gap gently, so it gets
        exactly the time that takes, then tracking goes to the full (still capped) rate.
        """
        if self.stepper.synced:
            if not self._sync_announced:
                self._sync_announced = True
                _log.info("arm '%s': synced, tracking at full speed", self.name)
            return
        now = self._clock()
        if self._sync_deadline is None:
            command = np.asarray(self.stepper.hold(), dtype=np.float64)
            want = np.asarray(self.stepper.sanitize([float(x) for x in goal]), dtype=np.float64)
            g = self.slots
            t_joint = float(np.max(np.abs(want[:g] - command[:g]))) / max(1e-9, self.stepper.sync_step_rad)
            t_grip = abs(float(want[g] - command[g])) / max(1e-9, self.stepper.sync_step_mm)
            self._sync_deadline = now + max(t_joint, t_grip) * self.period + SYNC_GRACE_S
        elif now > self._sync_deadline:
            self.stepper.synced = True
            self._sync_announced = True
            _log.info("arm '%s': startup ramp done (the target kept moving), tracking at full speed", self.name)

    def _rest_tick(self, measured: npt.NDArray[np.float32]) -> list[float]:
        """One tick of the park; ends it when the measured pose is at rest, stops getting closer, or runs out of
        time."""
        rest = self.rest
        assert rest is not None, "the first tick sets it"
        now = self._clock()
        command = self.stepper.step_toward(self._guarded(rest, measured), self._rest_step_rad, self._rest_step_mm)
        g = self.slots
        err_rad = float(np.max(np.abs(measured[:g] - np.asarray(rest[:g]))))
        err_mm = abs(float(measured[g]) - rest[g])
        reached: bool | None = None
        reason = ""
        if err_rad <= REST_TOL_RAD and err_mm <= REST_TOL_MM:
            reached, reason = True, "at rest"
        else:
            if self._rest_best is None or err_rad < self._rest_best[0] - 0.01 or err_mm < self._rest_best[1] - 1.0:
                self._rest_best, self._rest_progressed = (err_rad, err_mm), now
            if now - self._rest_progressed > REST_STALL_S:
                reached, reason = False, f"stalled {math.degrees(err_rad):.1f} deg / {err_mm:.0f} mm from rest"
            elif now - self._rest_started > self.rest_budget_s:
                reached, reason = False, f"timed out {math.degrees(err_rad):.1f} deg / {err_mm:.0f} mm from rest"
        if reached is not None:
            _log.info("arm '%s': park %s", self.name, reason)
            self.rest_result = (reached, reason)
            with self._lock:
                self._resting = False
                self._parked = True
            self._rearm_sync()
            self.rested.set()
        return command

    def _rest_timeout_s(self, q: npt.NDArray[np.float32], rest: list[float]) -> float:
        """How long a park from `q` (`[slots joints rad, gripper mm]`) may take at the rest speeds.

        @return The farthest joint's or the gripper's travel time plus #REST_TIMEOUT_MARGIN_S, at least
            #REST_TIMEOUT_S.
        """
        q64 = np.asarray(q, dtype=np.float64)
        g = self.slots
        gap_deg = math.degrees(float(np.max(np.abs(q64[:g] - np.asarray(rest[:g])))))
        gap_mm = abs(float(q64[g]) - rest[g])
        travel_s = max(gap_deg / self.spec.rest_speed_deg_s, gap_mm / REST_SPEED_MM_S)
        return max(REST_TIMEOUT_S, travel_s + REST_TIMEOUT_MARGIN_S)


def open_arm(
    spec: ArmSpec,
    cfg: "config.ArmConfig",
    side: str,
    *,
    opener: Callable[..., Any],
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Any] = time.sleep,
    energize: bool = True,
) -> followers.FollowerArm:
    """Opens one arm on its bus, energizes it, starts its thread and wraps it in a FollowerArm.

    @param spec The robot's table.
    @param cfg The arm's config section.
    @param side `"left"` or `"right"`.
    @param opener Opens the arm's bus, called as its chain calls it: #robots.robstride.open_can_bus,
        #robots.feetech.open_bus, or a fake bus in tests.
    @param clock Monotonic clock, seconds.
    @param sleep Sleeps that many seconds.
    @param energize Switch the motors on.
    @return The #robots.followers.FollowerArm over a #MotorArm, with the config's home.
    @throws ConfigError The bus's library (python-can, LeRobot) is not installed, the config lacks what the chain
        needs, or its motor temperature limits leave the warning temperature at or above the stop temperature.
    """
    spec = with_temp_limits(spec, cfg)
    arm = MotorArm(spec, side, cfg, opener=opener, clock=clock, sleep=sleep)
    arm.chain.open()
    if energize:
        try:
            _refuse_warm(spec, arm.chain)
            arm.chain.enable()
        except BaseException:
            _close_after_failure(arm.chain)
            raise
    else:
        arm.read_only()
    arm.start()
    deadline = time.monotonic() + FIRST_POSE_TIMEOUT_S
    while not arm.first_pose.wait(timeout=0.05):
        if not arm.is_alive() or time.monotonic() > deadline:
            arm.stop()
            _close_after_failure(arm.chain, disable_torque=energize)
            raise errors.ArmError(f"arm '{side}' published no pose from {cfg.can_channel}: {arm.fault or 'timed out'}")
    return followers.FollowerArm(side, arm, home_rad=cfg.home)


def with_temp_limits(spec: ArmSpec, cfg: "config.ArmConfig") -> ArmSpec:
    """The robot's table with the station file's motor temperature limits in place of its own.

    The file's `motor_temp_warn_c` and `motor_temp_stop_c` replace the body's where set: the owner may raise or lower
    either.

    @param spec The robot's table.
    @param cfg The arm's config section, which carries the file's limits.
    @return The table with the limits the arm runs on.
    @throws ConfigError The warning temperature is not below the stop temperature.
    """
    warn = spec.temp_warn_c if cfg.temp_warn_c is None else cfg.temp_warn_c
    stop = spec.temp_stop_c if cfg.temp_stop_c is None else cfg.temp_stop_c
    if warn >= stop:
        raise errors.ConfigError(
            f"{spec.name}: the motor warning temperature ({warn:g} C) must be below the stop temperature "
            f"({stop:g} C); set 'motor_temp_warn_c' and 'motor_temp_stop_c' in the station file to fit"
        )
    return dataclasses.replace(spec, temp_warn_c=warn, temp_stop_c=stop)


def _refuse_warm(spec: ArmSpec, chain: Chain) -> None:
    temps = chain.read().temperatures_c
    if temps is None:
        return
    warm = [
        f"{motor} is at {value:.0f} C ({sensor})"
        for motor, sensors in zip(chain.names, temps, strict=True)
        for sensor, value in sensors.items()
        if value >= spec.temp_warn_c
    ]
    if warm:
        raise errors.ArmError(
            f"the {chain.name} arm's {'; '.join(warm)}: let it cool below {spec.temp_warn_c:.0f} C, then try again "
            "(on a station, press Reset)"
        )


def _close_after_failure(chain: Chain, *, disable_torque: bool = True) -> None:
    """Closes with torque off (unless the arm was opened read only) after a failed open; a failure of that is logged,
    the original raised."""
    try:
        chain.close(disable_torque=disable_torque)
    except Exception as exc:
        _log.warning("arm '%s': cleanup after a failed open: %s", chain.name, exc)
