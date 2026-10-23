"""One arm's System 0 tick: the plan in, a motor command out, through a supervisor that can always hold the arm.

Pure logic, no I/O: the caller (`robots.sys0.motors`) reads the arm, passes the time, the measured state and the
plan, and writes what comes back. The code is the tracker the bundled policy expects, with its constants:

- **The reference.** #Plan is the loop's timeline for one tick: the command it just submitted, at its tick time,
  then the rows still buffered, one per tick from then. ArmTracker.tick() samples it (sample()) at now and every
  #robots.sys0.policy.FUTURE_DT_S ahead, so between two 20 Hz rows the arm is asked for the plan's own value at
  that instant (the follower's direct command path), and past the plan's end the last row holds. The rows are then
  shaped as the YAM follower shapes them (ArmTracker.capped()): #REF_LIMIT_MARGIN_RAD inside every joint
  limit, each joint at most #REF_SPEED_FRACTION of the supervisor's trip speed, and the grasp point at most
  `max_ee_speed_m_s`, the look-ahead included, so what the policy sees ahead is what it will be asked for.
- **The command** (decode()). The policy's first six outputs are joint torques, up to #TORQUE_FRACTION of each
  motor's limit; the firmware keeps its stock damping (#NOMINAL_KD) and no stiffness, so the torque does the
  tracking and the damper acts at the motor's own rate. Gravity compensation is i2rt's, added under it as always.
- **The gripper** stays the position servo it is without System 0, on the loop's own command: the command after
  the grip force cap (#robots.gripper_force), held, with the network's seventh output (a gripper
  residual) left unused. The cap regulates grip on the measured effort at the loop's tick; a residual moving the
  target every 4 ms, and a reference drifting toward the policy's raw close between ticks, fought it and made a
  closed grasp jitter.
- **The supervisor** (#Supervisor). With no stiffness in the firmware nothing else holds the arm, so every
  command passes through it: it clamps and rate-limits the torque, and latches a nominal PD hold at the measured
  pose on an overspeed, a joint past its limit, a tracking error held too long, a torque clamped too long, or a
  tick that did not come. A hold lasts until the run ends.

@see ArmTracker, Plan, sample(), Supervisor, decode()
"""

import dataclasses

import numpy as np
import numpy.typing as npt

from robots import kinematics
from robots.sys0 import policy

## The rate the tracker acts at, beside i2rt's motor loop.
MOTOR_HZ: float = 250.0
## The YAM followers' stock PD gains (i2rt `yam.yml`): what a hold applies.
NOMINAL_KP: npt.NDArray[np.float64] = np.array([80.0, 80.0, 80.0, 40.0, 10.0, 10.0])
NOMINAL_KD: npt.NDArray[np.float64] = np.array([5.0, 5.0, 5.0, 1.5, 1.5, 1.5])
## Motor torque limits: DM4340 on the three big joints, DM4310 on the wrist, newton metres.
TORQUE_LIMIT: npt.NDArray[np.float64] = np.array([28.0, 28.0, 28.0, 10.0, 10.0, 10.0])
## The policy's torque bound, as a fraction of each motor's limit.
TORQUE_FRACTION: float = 0.4
## The reference moves at most this fraction of the supervisor's trip speed per joint: a live plan can lift the
## shoulder 1.26 rad in 0.3 s, and followed as asked the arm reaches the trip speed and holds.
REF_SPEED_FRACTION: float = 0.8
## The reference is kept this far inside each joint's limits: with no stiffness the arm follows the reference
## itself, and a plan past a limit (a wrist asked 0.1 rad past its stop) would take it there.
REF_LIMIT_MARGIN_RAD: float = 0.03
## Proportional cuts per look-ahead step in the grasp-point speed cap; two reach 1 % on a 0.8 rad joint jump.
EE_CAP_PASSES: int = 4
## The longest step the reference's speed cap integrates over, seconds: a stalled thread does not let it jump.
MAX_REF_STEP_S: float = 0.05


@dataclasses.dataclass(frozen=True)
class MotorCommand:
    """One arm's MIT-mode command; i2rt adds gravity compensation to `torque`.

    Attributes:
    - target: Position targets, `[7]`: six joints in radians, then the gripper in `[0, 1]`.
    - kp: Firmware stiffness per joint, `[6]`.
    - kd: Firmware damping per joint, `[6]`.
    - torque: Feed-forward torque per joint, `[6]`, newton metres.
    """

    target: npt.NDArray[np.float64]
    kp: npt.NDArray[np.float64]
    kd: npt.NDArray[np.float64]
    torque: npt.NDArray[np.float64]


@dataclasses.dataclass(frozen=True)
class Plan:
    """What one arm is asked to follow: rows due at times, on the loop's clock.

    Attributes:
    - t_s: When each row is due, `[n]`, ascending, seconds.
    - rows: The rows, `[n, 7]`: six joints in radians, then the gripper in `[0, 1]`.

    @see sample()
    """

    t_s: npt.NDArray[np.float64]
    rows: npt.NDArray[np.float64]


def sample(plan: Plan, t_s: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """The plan at each of `t_s`, linear between rows and held at either end.

    @param plan The plan.
    @param t_s The times, `[m]`, seconds.
    @return The rows, `[m, 7]`.
    """
    return np.stack([np.interp(t_s, plan.t_s, plan.rows[:, d]) for d in range(plan.rows.shape[1])], axis=1)


@dataclasses.dataclass(frozen=True)
class SupervisorConfig:
    """When the supervisor holds the arm, and how far it lets the torque go (the default values).

    Attributes:
    - watchdog_s: Longest gap between two policy ticks before the hold.
    - max_speed_rad_s: Per-joint speed that trips the hold; about 1.5 times the peaks a YAM reaches tracking
      bottle-opening replays (2.95 rad/s on the big joints, 4.59 on the wrist).
    - limit_tolerance_rad: How far past a joint limit the arm may be measured before the hold; a tolerance outside
      the range, since the YAM rests with joint 2 at its model limit.
    - leash_rad: A tracking error above this, held for `leash_s`, trips the hold.
    - leash_s: See `leash_rad`.
    - max_torque_step: Most the torque may change in one tick, as a fraction of each motor's limit.
    - max_clamped_ticks: Consecutive ticks with a clamped torque before the hold.
    """

    watchdog_s: float = 0.05
    max_speed_rad_s: tuple[float, ...] = (4.5, 4.5, 4.5, 7.0, 7.0, 7.0)
    limit_tolerance_rad: float = 0.05
    leash_rad: float = 0.4
    leash_s: float = 0.2
    max_torque_step: float = 0.05
    max_clamped_ticks: int = 25


class Supervisor:
    """Clamps and rate-limits one arm's torque and, when needed, latches it to a PD hold.

    Attributes:
    - latched_reason: Why the hold is engaged, or `""` while the policy drives.
    - hold_pose: The pose the hold keeps, `[7]`, set when it latches.
    - clamped_ticks: Ticks whose torque was clamped to its bound, over the run.

    @see filter()
    """

    def __init__(self, joint_range_rad: npt.NDArray[np.float64], cfg: SupervisorConfig) -> None:
        """Starts with the policy driving.

        @param joint_range_rad Joint limits, `[6, 2]`, radians.
        @param cfg The limits.
        """
        self.joint_range_rad = joint_range_rad
        self.cfg = cfg
        self.latched_reason = ""
        self.hold_pose: npt.NDArray[np.float64] = np.zeros(policy.ARM_DOF)
        self.clamped_ticks = 0
        self._consecutive_clamped = 0
        self._last_torque: npt.NDArray[np.float64] = np.zeros(policy.ARM_JOINTS)
        self._last_tick_s: float | None = None
        self._leash_since_s: float | None = None

    def filter(
        self,
        now_s: float,
        q: npt.NDArray[np.float64],
        qd: npt.NDArray[np.float64],
        q_ref: npt.NDArray[np.float64],
        command: MotorCommand,
    ) -> MotorCommand:
        """What to send this tick: `command` with its torque clamped and rate-limited, or the hold once latched.

        @param now_s Monotonic time, seconds; each call is one policy tick.
        @param q Measured joints and gripper, `[7]`.
        @param qd Measured velocities, `[7]`.
        @param q_ref The reference now, `[7]`.
        @param command The policy's command for this tick.
        @return The command to send.
        """
        if not self.latched_reason:
            reason = self._trip(now_s, q, qd, q_ref)
            if reason:
                self.latched_reason = reason
                self.hold_pose = np.array(q, dtype=np.float64)
        self._last_tick_s = now_s
        if self.latched_reason:
            return MotorCommand(
                self.hold_pose.copy(), NOMINAL_KP.copy(), NOMINAL_KD.copy(), np.zeros(policy.ARM_JOINTS)
            )
        return self._shape(command)

    def resume(self) -> None:
        """A new stretch of driving: the watchdog and the leash start over; a latched hold stays latched."""
        self._last_tick_s = None
        self._leash_since_s = None

    def latch(self, reason: str, pose: npt.NDArray[np.float64]) -> None:
        """Engages the hold from outside the tick (the motor loop found the tick's block stale).

        @param reason Why, for the run's error.
        @param pose The pose to hold, `[7]`.
        """
        if not self.latched_reason:
            self.latched_reason = reason
            self.hold_pose = np.array(pose, dtype=np.float64)

    def _trip(
        self, now_s: float, q: npt.NDArray[np.float64], qd: npt.NDArray[np.float64], q_ref: npt.NDArray[np.float64]
    ) -> str:
        """The reason to hold now, or `""`."""
        c = self.cfg
        joints = slice(0, policy.ARM_JOINTS)
        if self._last_tick_s is not None and now_s - self._last_tick_s > c.watchdog_s:
            return f"watchdog: {1e3 * (now_s - self._last_tick_s):.0f} ms between two ticks"
        fast = np.abs(qd[joints]) > np.asarray(c.max_speed_rad_s)
        if fast.any():
            return f"overspeed on joint {int(np.argmax(fast)) + 1}"
        outside = (q[joints] < self.joint_range_rad[:, 0] - c.limit_tolerance_rad) | (
            q[joints] > self.joint_range_rad[:, 1] + c.limit_tolerance_rad
        )
        if outside.any():
            return f"joint {int(np.argmax(outside)) + 1} past its limit"
        if np.abs(q[joints] - q_ref[joints]).max() <= c.leash_rad:
            self._leash_since_s = None
        elif self._leash_since_s is None:
            self._leash_since_s = now_s
        elif now_s - self._leash_since_s > c.leash_s:
            return f"tracking error over {c.leash_rad:g} rad for {c.leash_s:g} s"
        if self._consecutive_clamped > c.max_clamped_ticks:
            return "torque clamped too long"
        return ""

    def _shape(self, command: MotorCommand) -> MotorCommand:
        """Clamps the torque to its bound, then limits its change since the last tick."""
        bound = TORQUE_FRACTION * TORQUE_LIMIT
        step = self.cfg.max_torque_step * TORQUE_LIMIT
        clamped = np.clip(command.torque, -bound, bound)
        limited = np.clip(clamped, self._last_torque - step, self._last_torque + step)
        was_clamped = bool(np.any(clamped != command.torque))
        self.clamped_ticks += int(was_clamped)
        self._consecutive_clamped = self._consecutive_clamped + 1 if was_clamped else 0
        self._last_torque = limited
        return dataclasses.replace(command, torque=limited)


def decode(
    action: npt.NDArray[np.float64], q_ref: npt.NDArray[np.float64], joint_range_rad: npt.NDArray[np.float64]
) -> MotorCommand:
    """The motor command a policy action stands for.

    @param action The action, `[7]`, in `[-1, 1]`: six joint torques as fractions of #TORQUE_FRACTION of the limit,
        then a gripper residual, which the station leaves unused (the module docstring, *The gripper*).
    @param q_ref The reference now, `[7]`; its gripper is the loop's command after the grip force cap.
    @param joint_range_rad Joint limits, `[6, 2]`, radians.
    @return The command; its joint targets are the reference (no stiffness acts on them), its gripper the
        reference's.
    """
    a = np.clip(action, -1.0, 1.0)
    target = np.array(q_ref, dtype=np.float64)
    target[: policy.ARM_JOINTS] = np.clip(target[: policy.ARM_JOINTS], joint_range_rad[:, 0], joint_range_rad[:, 1])
    target[policy.ARM_JOINTS] = min(max(float(target[policy.ARM_JOINTS]), 0.0), 1.0)
    torque = a[: policy.ARM_JOINTS] * TORQUE_FRACTION * TORQUE_LIMIT
    return MotorCommand(target, np.zeros(policy.ARM_JOINTS), NOMINAL_KD.copy(), torque)


class ArmTracker:
    """One arm's tracker: its policy state, its reference shaping and its supervisor.

    Attributes:
    - supervisor: The #Supervisor every command passes through.

    @see tick(), restart()
    """

    def __init__(
        self,
        net: policy.Policy,
        joint_range_rad: npt.NDArray[np.float64],
        arm_kinematics: kinematics.Kinematics,
        max_ee_speed_m_s: float,
        supervisor: SupervisorConfig | None = None,
    ) -> None:
        """Binds the network and the arm's geometry; restart() before the first tick.

        @param net The tracker network, shared by both arms.
        @param joint_range_rad Joint limits, `[6, 2]`, radians.
        @param arm_kinematics The arm's forward kinematics, for the grasp-point speed cap; this tracker's alone,
            since a MuJoCo model's data is not shared between threads.
        @param max_ee_speed_m_s The grasp point's speed cap on the reference, the loop's own cap.
        @param supervisor The supervisor's limits; the defaults unless given.
        """
        self._net = net
        self._joint_range_rad = np.asarray(joint_range_rad, dtype=np.float64)
        self._kinematics = arm_kinematics
        self._max_ee_speed_m_s = max_ee_speed_m_s
        self.supervisor = Supervisor(self._joint_range_rad, supervisor or SupervisorConfig())
        self._ref_cap_rad_s = REF_SPEED_FRACTION * np.asarray(self.supervisor.cfg.max_speed_rad_s)
        self._offsets_s: npt.NDArray[np.float64] = (
            np.arange(policy.FUTURE_FRAMES + 1, dtype=np.float64) * policy.FUTURE_DT_S
        )
        self._a_prev: npt.NDArray[np.float64] = np.zeros(policy.ARM_DOF)
        self._ref_prev: npt.NDArray[np.float64] | None = None
        self._ref_prev_s = 0.0

    def restart(self) -> None:
        """A new stretch of driving: the shaped reference re-seeds from the measured pose, the previous action is 0."""
        self._ref_prev = None
        self._a_prev = np.zeros(policy.ARM_DOF)
        self.supervisor.resume()

    def tick(self, now_s: float, q: npt.NDArray[np.float64], qd: npt.NDArray[np.float64], plan: Plan) -> MotorCommand:
        """One step: the shaped reference, the policy's action, the supervised command.

        @param now_s Time on the loop's clock, seconds (the plan's clock and the supervisor's).
        @param q Measured joints and gripper, `[7]`.
        @param qd Measured velocities, `[7]` (the gripper's is 0).
        @param plan What to follow.
        @return The command to write.
        """
        rows = self.capped(sample(plan, now_s + self._offsets_s), q, now_s)
        action = policy.act(self._net, policy.observation(q, qd, self._a_prev, rows))
        self._a_prev = action
        return self.supervisor.filter(now_s, q, qd, rows[0], decode(action, rows[0], self._joint_range_rad))

    def capped(
        self, rows: npt.NDArray[np.float64], q: npt.NDArray[np.float64], now_s: float
    ) -> npt.NDArray[np.float64]:
        """`rows` (the reference now, then every look-ahead step) inside the joint limits and the speed caps.

        Every row is clipped #REF_LIMIT_MARGIN_RAD inside each joint's range; the first then moves from the last
        tick's shaped value (seeded at the measured pose) and each later one from the row before it, at most
        #REF_SPEED_FRACTION of the trip speed per joint; then each of those steps is shortened along its own joint
        direction until the grasp point travels no faster than `max_ee_speed_m_s`.

        @param rows The sampled plan, `[FUTURE_FRAMES + 1, 7]`.
        @param q Measured joints and gripper, `[7]`.
        @param now_s The tick's time, seconds.
        @return The shaped rows (a new array); the gripper column passes through.
        """
        joints = slice(0, policy.ARM_JOINTS)
        out = np.array(rows, dtype=np.float64)
        low = self._joint_range_rad[:, 0] + REF_LIMIT_MARGIN_RAD
        high = self._joint_range_rad[:, 1] - REF_LIMIT_MARGIN_RAD
        out[:, joints] = np.clip(out[:, joints], low, high)
        if self._ref_prev is None:
            self._ref_prev, self._ref_prev_s = np.array(q[joints], dtype=np.float64), now_s - 1.0 / MOTOR_HZ
        dt_s = min(max(now_s - self._ref_prev_s, 0.0), MAX_REF_STEP_S)
        step = self._ref_cap_rad_s * dt_s
        out[0, joints] = self._ref_prev + np.clip(out[0, joints] - self._ref_prev, -step, step)
        ahead = self._ref_cap_rad_s * policy.FUTURE_DT_S
        for k in range(1, out.shape[0]):
            out[k, joints] = out[k - 1, joints] + np.clip(out[k, joints] - out[k - 1, joints], -ahead, ahead)
        steps_m = np.full(out.shape[0], self._max_ee_speed_m_s * policy.FUTURE_DT_S)
        steps_m[0] = self._max_ee_speed_m_s * dt_s
        out = _ee_rate_limited(self._kinematics, self._ref_prev, out, steps_m)
        self._ref_prev, self._ref_prev_s = out[0, joints].copy(), now_s
        return out


def _ee_rate_limited(
    arm_kinematics: kinematics.Kinematics,
    q_from: npt.NDArray[np.float64],
    rows: npt.NDArray[np.float64],
    max_step_m: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """`rows` with each step's grasp-point travel capped: row k moves at most `max_step_m[k]` from row k - 1.

    The grasp point's path over a long step is curved, so one proportional cut leaves it about 10 % over; the cut
    repeats up to #EE_CAP_PASSES times, until it is within 1 %.

    @param arm_kinematics The arm's forward kinematics.
    @param q_from The six joints the first step starts from.
    @param rows The rows, `[n, 7]`; the gripper column passes through.
    @param max_step_m The most the grasp point may travel into each row, `[n]`, metres.
    @return The capped rows (a new array).
    """
    out = np.array(rows, dtype=np.float64)
    prev = np.asarray(q_from[: policy.ARM_JOINTS], dtype=np.float64)
    p_prev = arm_kinematics.ee_position(prev)
    for k in range(out.shape[0]):
        p = arm_kinematics.ee_position(out[k, : policy.ARM_JOINTS])
        for _ in range(EE_CAP_PASSES):
            travel_m = float(np.linalg.norm(p - p_prev))
            if travel_m <= max_step_m[k] * 1.01:
                break
            out[k, : policy.ARM_JOINTS] = prev + (out[k, : policy.ARM_JOINTS] - prev) * (max_step_m[k] / travel_m)
            p = arm_kinematics.ee_position(out[k, : policy.ARM_JOINTS])
        prev, p_prev = out[k, : policy.ARM_JOINTS], p
    return out
