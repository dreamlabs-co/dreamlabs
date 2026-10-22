"""The command path with System 0: each arm's tracker drives its motors at 250 Hz from the loop's plan.

#MotorPath stands where the loop's command writer stands (`robots.control`) and takes the same
calls. Each tick the loop submits its command (after the ramp, the grip force cap and the clamp) and, once the
start-up ramp has ended, each arm's #robots.sys0.tracker.Plan: that command at the tick's time, then the rows still
buffered, one per tick from then, as the loop will command them. Each arm's #ArmMotors then runs its
#robots.sys0.tracker.ArmTracker on the measured state every 4 ms and writes the MIT block it returns
(`FollowerArm.write_motor_command`).

On a station each arm's #ArmMotors runs in that arm's own process (#robots.sys0.arm_process), beside i2rt's motor
threads and nothing else. In the agent's process the tracker's 4 ms tick would share one interpreter lock with
three cameras, the control loop and the server, where it waits 50 to 70 ms for it, past #BLOCK_STALE_S, so
every run stops on the watchdog within a second of the ramp. An arm in this process (the tests) is driven by
#MotorPath's own thread.

Who drives is decided per submission, the way the YAM followers decide it per command:

- a submission without a plan (the ramp, and before it, nothing) goes to the command writer, which writes the PD
  targets as the loop does without System 0; that is also how the arms are handed back when a new ramp starts;
- a submission with a plan hands the motors to the trackers, which restart from the measured pose.

hold() keeps whoever drives: the writer re-commands its last target, a tracker keeps following the last command
and nothing past it. stop() leaves each arm on its PD at the last target the tracker wrote, so the loop's own
hold after it holds there.

Two things stop a run safely. A supervisor hold (an overspeed, a limit, the leash, a clamped torque, a late tick)
keeps that arm still on its PD and sets `error`, which the loop's next tick raises, so the station reports why the
run ended. And a check on i2rt's own motor loop (#BLOCK_STALE_S): when the tracker stops writing while it drives,
i2rt would keep sending the last torque, so the check puts a PD hold at the measured pose in its place and latches
the supervisor.

@see build_arm(), ArmMotors, MotorPath
"""

import dataclasses
import logging
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

import numpy as np
import numpy.typing as npt

from robots import config, errors, followers, kinematics, types
from robots.sys0 import policy, tracker

if TYPE_CHECKING:  # robots.control imports this module, so its writer is named for the type checker only
    from robots import control

_log = logging.getLogger(__name__)

## The arms, as the loop names them.
ARMS: tuple[str, ...] = ("left", "right")
## A tracker block older than this while the tracker drives is replaced by a PD hold, seconds: the supervisor's own
## watchdog, and about three times the longest stall measured in the follower.
BLOCK_STALE_S: float = 0.05
## The interpreter hands the lock over this often in an arm's process, seconds (CPython's default, 5 ms, is longer
## than a tick, beside i2rt's busy motor threads); the YAM follower sets the same.
SWITCH_INTERVAL_S: float = 0.0005
## A tick this long after the one before it counts as late in the line stop() logs, seconds (three periods).
LATE_TICK_S: float = 0.012
## Seconds stop() waits for the thread's current tick.
STOP_TIMEOUT_S: float = 2.0


@dataclasses.dataclass(frozen=True)
class Models:
    """What the trackers need besides the arms, for arms driven in this process; built once per station.

    An arm served from its own process (#robots.sys0.arm_process.RemoteArm) builds its own in that process, so a
    loop over such arms is handed #IN_ARM_PROCESSES, which carries none.

    Attributes:
    - net: The tracker network, shared by both arms; None in #IN_ARM_PROCESSES.
    - kinematics: Each arm's forward kinematics, keyed `"left"` and `"right"`: the trackers' own, apart from the
      loop's, since one MuJoCo model's data is not shared between two threads.
    - joint_range_rad: Each arm's joint limits, `[6, 2]`, radians, keyed the same.

    @see build_arm()
    """

    net: policy.Policy | None
    kinematics: dict[str, kinematics.Kinematics]
    joint_range_rad: dict[str, npt.NDArray[np.float64]]


## System 0 on, with every arm's tracker in that arm's own process: what a station hands its loops.
IN_ARM_PROCESSES = Models(net=None, kinematics={}, joint_range_rad={})


def build_arm(cfg: config.ArmConfig) -> tuple[policy.Policy, kinematics.ArmKinematics]:
    """Loads the bundled network and builds one arm's kinematics, in the process that drives that arm.

    @param cfg The arm's config; only its gripper type is read.
    @return The network, warmed up, and the arm's kinematics (its `joint_range_rad` is the tracker's limits).
    @throws ConfigError i2rt or mujoco is not installed.
    """
    arm_kinematics = kinematics.ArmKinematics.from_config(cfg)
    net = policy.load()
    policy.act(net, np.zeros(policy.OBS_DIM))  # the first call's one-off costs land here, not on the first 4 ms tick
    _log.info(
        "System 0: %s tracker, iteration %s",
        net.meta.get("actuator"),
        net.meta.get("iteration"),
    )
    return net, arm_kinematics


class ArmMotors:
    """One arm's tracker and the check on its motor loop; the caller ticks once() every 4 ms.

    Thread-safe: follow() comes from whoever relays the loop's plan, once() from the ticking thread, and the check
    from i2rt's motor loop.

    Attributes:
    - name: `"left"` or `"right"`.
    - is_driving: Whether the tracker has the motors.
    - error: Why the run must stop (a supervisor hold), or None; set once.
    - ticks: Ticks on which the tracker wrote the arm.

    @see follow(), once(), start(), stop()
    """

    def __init__(
        self,
        name: str,
        arm: followers.FollowerArm,
        net: policy.Policy,
        joint_range_rad: npt.NDArray[np.float64],
        arm_kinematics: kinematics.Kinematics,
        *,
        max_ee_speed_m_s: float,
        clock: Callable[[], float],
    ) -> None:
        """Binds the arm and builds its tracker; nothing drives until follow() is given a plan.

        @param name `"left"` or `"right"`.
        @param arm The arm, in this process.
        @param net The tracker network.
        @param joint_range_rad The arm's joint limits, `[6, 2]`, radians.
        @param arm_kinematics The arm's forward kinematics, this tracker's alone.
        @param max_ee_speed_m_s The loop's end-effector speed cap, which the tracker puts on the plan too.
        @param clock The loop's clock, seconds; the plans' times are on it.
        """
        self.name = name
        self._arm = arm
        self._tracker = tracker.ArmTracker(net, joint_range_rad, arm_kinematics, max_ee_speed_m_s)
        self._clock = clock
        self._lock = threading.Lock()
        self._plan: tracker.Plan | None = None
        self._last_block_s: float | None = None
        self._last_tick_s: float | None = None
        self.is_driving = False
        self.error: str | None = None
        self.ticks = 0
        self._late_ticks = 0
        self._worst_gap_s = 0.0

    @property
    def supervisor(self) -> tracker.Supervisor:
        """The tracker's supervisor, whose latch says why the arm holds."""
        return self._tracker.supervisor

    def follow(self, plan: tracker.Plan | None) -> None:
        """Gives the tracker a plan to follow, taking the motors if it does not have them; None hands them back.

        Handed back, the arm holds on its PD at the last target the tracker wrote until the command writer writes.

        @param plan What to follow, or None for the PD path.
        @throws ArmError The hold at a hand-back failed (see `FollowerArm.hold()`).
        """
        with self._lock:
            if plan is None:
                if self.is_driving:
                    self.is_driving = False
                    self._last_tick_s = None
                    self._arm.hold()
                    _log.info("System 0 hands the %s arm back to the PD path", self.name)
                return
            self._plan = plan
            if not self.is_driving:
                self._tracker.restart()
                self._last_block_s = self._clock()
                self.is_driving = True
                _log.info("System 0 drives the %s arm from %.3f s", self.name, self._clock())

    def once(self, now_s: float) -> None:
        """Runs one tick: reads the arm, runs the tracker, writes the MIT block; nothing while the PD path drives.

        @param now_s Time on the loop's clock, seconds.
        @throws ArmError The arm read or write failed.
        """
        with self._lock:
            plan = self._plan
            if not self.is_driving or plan is None:
                return
            state = self._arm.read()
            q = np.append(state.positions_rad, state.gripper)
            qd = np.append(state.velocities_rad_s, 0.0)
            command = self._tracker.tick(now_s, q, qd, plan)
            self._arm.write_motor_command(command.target, command.kp, command.kd, command.torque)
            self._last_block_s = self._clock()
            self.ticks += 1
            if self._last_tick_s is not None:
                gap_s = now_s - self._last_tick_s
                self._late_ticks += int(gap_s > LATE_TICK_S)
                self._worst_gap_s = max(self._worst_gap_s, gap_s)
            self._last_tick_s = now_s
            self._report_hold()

    def start(self) -> None:
        """Puts the stale-block check on the arm's motor loop."""
        self._arm.set_motor_loop_hook(self._guard)

    def stop(self) -> None:
        """Removes the check and leaves the arm on its PD at the last target the tracker wrote to it."""
        self._arm.set_motor_loop_hook(None)
        if self.ticks:
            _log.info(
                "System 0, %s arm: %d ticks driven, worst gap %.1f ms, %d ticks later than %.0f ms",
                self.name,
                self.ticks,
                1e3 * self._worst_gap_s,
                self._late_ticks,
                1e3 * LATE_TICK_S,
            )
        self.follow(None)

    def _report_hold(self) -> None:
        """Sets `error` the first time the supervisor holds the arm; the arm stays held on its PD meanwhile."""
        reason = self._tracker.supervisor.latched_reason
        if reason and self.error is None:
            self.error = f"System 0 safety hold on the {self.name} arm: {reason}; the arm holds where it tripped"
            _log.error("%s", self.error)

    def _guard(self) -> None:
        """On i2rt's motor loop, before each update: a tracker block older than #BLOCK_STALE_S becomes a PD hold.

        It runs on i2rt's thread, so nothing it raises may reach i2rt's loop: a failure is logged (the thread
        boundary, checklist rule 4) and the loop's own checks report the arm.
        """
        last_s = self._last_block_s
        if not self.is_driving or last_s is None:
            return
        age_s = self._clock() - last_s
        if age_s <= BLOCK_STALE_S:
            return
        try:
            self._hold_stale(age_s)
        except Exception:
            _log.exception("System 0: the %s arm's stale-block hold failed", self.name)

    def _hold_stale(self, age_s: float) -> None:
        """Writes a PD hold at the measured pose in place of a stale tracker block, and latches the supervisor."""
        state = self._arm.read()
        pose = np.append(state.positions_rad, min(max(state.gripper, 0.0), 1.0))
        self._arm.write_motor_command(
            pose, tracker.NOMINAL_KP.copy(), tracker.NOMINAL_KD.copy(), np.zeros(policy.ARM_JOINTS)
        )
        self._tracker.supervisor.latch(f"watchdog: the tracker's block was {1e3 * age_s:.0f} ms old", pose)
        # Stale from now until the tracker writes again: the hold is in place, nothing more to do each cycle.
        self._last_block_s = None
        _log.error("System 0: %s arm's block %.0f ms old; PD hold in its place", self.name, 1e3 * age_s)
        self._report_hold()


class Handle(Protocol):
    """One arm's System 0 as #MotorPath drives it: #ArmMotors in this process, or a proxy to an arm's process.

    @see ArmMotors, robots.sys0.arm_process.RemoteMotors
    """

    @property
    def error(self) -> str | None:
        """Why the run must stop (a safety hold), or None."""
        ...

    def follow(self, plan: tracker.Plan | None) -> None:
        """Gives the tracker a plan, taking the motors; None hands them back to the PD path.

        @param plan What to follow, or None.
        """
        ...

    def once(self, now_s: float) -> None:
        """Runs one tick where the tracker is in this process; elsewhere, reads what its process reported.

        @param now_s Time on the loop's clock, seconds.
        """
        ...

    def start(self) -> None:
        """Starts the arm's System 0 for a run."""
        ...

    def stop(self) -> None:
        """Stops it, leaving the arm on its PD at the last target written."""
        ...


@dataclasses.dataclass(frozen=True)
class _Submission:
    """One tick's command, with the plan when the trackers are to follow it.

    Attributes:
    - action: The command after the ramp, the grip force cap and the clamp.
    - t_s: The tick's time, seconds.
    - plans: Each arm's plan, or None while the PD path has the arms.
    """

    action: types.Action
    t_s: float
    plans: dict[str, tracker.Plan] | None


class MotorPath:
    """Both arms' System 0 behind the loop's command writer: the writer for the ramp, the trackers after it.

    Its thread runs once() every 4 ms: the writer's period while the writer drives, and, while the trackers drive,
    the tick of every tracker in this process (an arm's process ticks its own).

    Attributes:
    - period_s: Seconds between ticks, the writer's.
    - written: Ticks on which both arms were written, by the trackers or the writer.
    - error: Why the run must stop: a supervisor hold, or what the thread raised; the loop's next tick raises it.
    - is_driving: Whether the trackers have the motors.

    @see submit(), hold(), once(), start(), stop()
    """

    def __init__(
        self,
        writer: "control._CommandWriter",
        arms: dict[str, followers.Arm],
        models: Models,
        *,
        max_ee_speed_m_s: float,
        clock: Callable[[], float],
    ) -> None:
        """Binds the writer and each arm's System 0; nothing drives until a submission with a plan.

        @param writer The loop's command writer, which drives the arms whenever the trackers do not.
        @param arms The two arms keyed `"left"` and `"right"`: each served from its own process brings its own
            System 0 (`RemoteArm.sys0_motors()`); one in this process gets an #ArmMotors over `models`.
        @param models The network and each arm's geometry, for the arms in this process.
        @param max_ee_speed_m_s The loop's end-effector speed cap, which the trackers put on the plan too.
        @param clock The loop's clock, seconds; the plans' times are on it.
        @throws ConfigError An arm in this process, and `models` without a network.
        """
        self._writer = writer
        self._motors = {arm: _motors_for(arm, arms[arm], models, max_ee_speed_m_s, clock) for arm in ARMS}
        self._clock = clock
        self.period_s: float = writer.period_s
        self.written = 0
        self._error: Exception | None = None
        self.is_driving = False
        self._submission: _Submission | None = None
        self._thread: threading.Thread | None = None
        self._is_running = False

    @property
    def error(self) -> Exception | None:
        """Why the run must stop: the first arm's safety hold, or what the thread raised; None while all is well."""
        if self._error is None:
            for arm in ARMS:
                reason = self._motors[arm].error
                if reason:
                    self._error = errors.ControlLoopError(reason)
                    break
        return self._error

    def submit(self, action: types.Action, now_s: float, plans: dict[str, tracker.Plan] | None = None) -> None:
        """Makes `action` the latest command, and `plans` what the trackers follow; None leaves the PD path driving.

        @param action The tick's command for both arms.
        @param now_s The tick's time, seconds.
        @param plans Each arm's plan, or None during the ramp.
        @throws ArmError An arm's process refused the plan, or its hold at a hand-back failed.
        """
        self._writer.submit(action, now_s)
        self._submission = _Submission(action=action, t_s=now_s, plans=plans)
        if plans is None:
            if self.is_driving:
                self._hand_back()
            return
        for arm in ARMS:
            self._motors[arm].follow(plans[arm])
        if not self.is_driving:
            self.is_driving = True
            _log.info("System 0 drives both arms from %.3f s", now_s)

    def hold(self) -> None:
        """Holds the arms where they are being taken: the writer's last target, or the trackers' last command.

        @throws ArmError A never-written arm's observation is malformed (see `FollowerArm.hold()`).
        """
        submission = self._submission
        if submission is None or submission.plans is None or not self.is_driving:
            self._writer.hold()
            return
        still = {}
        for arm, command in (("left", submission.action.left), ("right", submission.action.right)):
            row = np.append(command.positions_rad, command.gripper)
            still[arm] = tracker.Plan(t_s=np.array([submission.t_s]), rows=row[None, :])
            self._motors[arm].follow(still[arm])
        self._submission = dataclasses.replace(submission, plans=still)

    def once(self, now_s: float) -> None:
        """Runs one tick: the trackers in this process while they drive, the writer otherwise.

        @param now_s Time on the loop's clock, seconds.
        @throws ArmError An arm read or write failed.
        """
        if not self.is_driving:
            written = self._writer.written
            self._writer.once(now_s)
            self.written += self._writer.written - written
            return
        for arm in ARMS:
            self._motors[arm].once(now_s)
        self.written += 1

    def start(self) -> None:
        """Starts each arm's System 0 and the thread."""
        for arm in ARMS:
            self._motors[arm].start()
        self._is_running = True
        self._thread = threading.Thread(target=self._run, name="sys0-motors", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stops the thread and each arm's System 0, which leaves its arm on its PD at the last target written."""
        self._is_running = False
        if self._thread is not None:
            self._thread.join(STOP_TIMEOUT_S)
            if self._thread.is_alive():
                _log.warning("System 0 thread still ticking after %.1f s; holding the arms anyway", STOP_TIMEOUT_S)
            self._thread = None
        for arm in ARMS:
            self._motors[arm].stop()
        self.is_driving = False

    def _hand_back(self) -> None:
        """The writer takes the motors back for a new ramp, which starts at the measured pose: it starts afresh."""
        for arm in ARMS:
            self._motors[arm].follow(None)
        self.is_driving = False
        self._writer.restart()
        _log.info("System 0 hands the arms back to the PD path")

    def _run(self) -> None:
        """The thread's body: once() every `period_s`; what a tick raises ends the thread and is kept in `error`."""
        try:
            next_s = self._clock()
            while self._is_running:
                now_s = self._clock()
                if now_s < next_s:
                    time.sleep(min(next_s - now_s, self.period_s))
                    continue
                next_s = max(next_s + self.period_s, now_s - self.period_s)  # never burst to catch up
                self.once(now_s)
        except Exception as exc:
            # The thread boundary (checklist rule 4): the loop's next tick raises it and leaves the arms safe.
            self._error = exc


def _motors_for(
    name: str, arm: followers.Arm, models: Models, max_ee_speed_m_s: float, clock: Callable[[], float]
) -> Handle:
    """One arm's System 0: the arm's own when it is served from its own process, else an #ArmMotors here.

    @param name `"left"` or `"right"`.
    @param arm The arm.
    @param models The network and geometry for an arm in this process.
    @param max_ee_speed_m_s The loop's end-effector speed cap.
    @param clock The loop's clock, seconds.
    @return The handle #MotorPath drives.
    @throws ConfigError The arm is in this process and `models` has no network.
    """
    own: Callable[[float], Handle] | None = getattr(arm, "sys0_motors", None)
    if own is not None:
        return own(max_ee_speed_m_s)
    if models.net is None or not isinstance(arm, followers.FollowerArm):
        raise errors.ConfigError(f"System 0: the {name} arm is not in a process of its own and no network was loaded")
    return ArmMotors(
        name,
        arm,
        models.net,
        models.joint_range_rad[name],
        models.kinematics[name],
        max_ee_speed_m_s=max_ee_speed_m_s,
        clock=clock,
    )
