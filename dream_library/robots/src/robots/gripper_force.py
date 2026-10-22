"""The loop-side grip force cap: position control until the gripper touches something, then effort-regulated
compliance.

A policy commands a gripper *position*, and i2rt's motor PD turns whatever position error remains into force.
On a grasped object a commanded "closed" therefore keeps squeezing, and the force is whatever the leftover error
implies rather than a number anyone chose: a thin object is crushed, a heavy one is held by luck. #GripperForceCap
turns the gripper into a hybrid controller at the loop's level, as the YAM follower's grip force cap does:

- free motion: the command passes through untouched, so closing and opening are still position control;
- in contact: the command is offset back toward open by however much it takes to hold the measured gripper
  effort at `cap`. The offset integrates the *signed* effort error, so at `effort == cap` it freezes and the
  gripper parks where the effort reads a constant cap value. Over the cap it backs off, under it (an object
  compressing, or slipping) it re-grips. Nothing pulls the offset back toward the raw target while a grasp
  holds, so there is no hunting and no creep into an over-squeeze;
- in contact, a hard floor as well: the command never sits more than `cap / stiffness` closed of the *measured*
  jaw, so the motor PD can never be asked for more than the cap however far the policy closes. The offset above is
  measured from the raw target and lags; without the floor a closed target on a wide or rigid object pins the PD
  near its maximum torque, and the slow offset then swings the jaw between squeeze and release;
- in contact, a ceiling too: the command never opens past the measured jaw. There the squeeze is already zero,
  so opening further only lets go. Without it the effort filter's lag (3 Hz at the 20 Hz tick) keeps the offset
  growing after the floor has already eased the squeeze, the jaw lifts off the object, contact is lost, and the
  slow release closes it again onto the object: a squeeze-and-release cycle about twice a second;
- contact means a high effort on a near-stalled jaw. A fast free close draws high effort just accelerating the
  jaw; read as contact, the floor would throttle every close to the release rate.

The offset bleeds away only in free motion or on a deliberate open, so the next grasp starts from zero; a tick
with no effort reading keeps the back-off it has rather than snapping back to the raw close. The floor, the
stalled-jaw test and the feedback-gap hold are the YAM follower's; the ceiling is
this package's, for the slower tick the cap runs at here.

The effort i2rt reports is noisy (impedance-mode ripple on the DM motor), so it passes an #EffortFilter, a
median-of-three spike rejector and a one-pole low-pass, before the control law sees it.

Units are the driver's: `cap` and the measured effort are both i2rt's `gripper_eff`, the gripper motor's torque,
not newtons (#robots.types.ArmState.gripper_effort). Position is the loop's gripper convention, 0 closed and 1
open. Pure arithmetic, no I/O and no numpy; #robots.control owns one cap per arm and runs it once a tick.

@see GripperForceCap, GripperForceCap.update(), EffortFilter
"""

import math

## Effort cap the station holds a grasp at, in i2rt's `gripper_eff` units. 0.65: at 2.0 grasps come out harder
## than any task needs, and at 1.5 they crush cups.
DEFAULT_CAP: float = 0.65

## Low-pass corner of #EffortFilter, Hz. An 8 Hz corner suits a 250 Hz command thread; the cap runs on the tick
## here, at `control_hz` (20 Hz on the station), where 8 Hz sits above half the sample rate and would filter
## nothing. 3 Hz still settles inside about 50 ms and actually rejects the ripple.
DEFAULT_CUTOFF_HZ: float = 3.0

## Largest step `update()` integrates for, seconds: a tick that arrives late (a hold, a stall, a long refill wait)
## must not move the offset by the whole gap at once.
_MAX_STEP_S: float = 0.1
## The gripper's stiffness in effort per gripper unit: the linear_4310 gripper's kp (20 N*m/rad) times its motor
## stroke (6.57 rad from closed to open). What the PD makes of a target `x` gripper units closed of the jaw.
DEFAULT_STIFFNESS: float = 20.0 * 6.57


class EffortFilter:
    """A median-of-three spike rejector followed by a one-pole low-pass, over one gripper's effort signal.

    The median kills the isolated spikes that a CAN read of a motor under impedance control throws out; the
    low-pass takes the ripple off what is left. Both are needed: a low-pass alone smears a spike over its whole
    time constant instead of dropping it.

    Attributes:
    - cutoff_hz: The low-pass corner, Hz; the filter is time-based, so an irregular tick is handled correctly.

    @see update(), reset()
    """

    def __init__(self, cutoff_hz: float = DEFAULT_CUTOFF_HZ) -> None:
        """Builds an empty filter; the first sample passes through as its own output.

        @param cutoff_hz Low-pass corner, Hz.
        """
        self.cutoff_hz = float(cutoff_hz)
        self._window: list[float] = []
        self._value: float | None = None
        self._t_s: float | None = None

    def update(self, sample: float, now_s: float) -> float:
        """Takes one raw effort sample and returns the filtered signal.

        @param sample The driver's gripper effort for this tick.
        @param now_s Tick time on the loop's clock, seconds; the step between calls sets the filter weight.
        @return The filtered effort, in the same units as `sample`.
        """
        self._window.append(float(sample))
        if len(self._window) > 3:
            self._window.pop(0)
        median = sorted(self._window)[len(self._window) // 2]
        if self._value is None or self._t_s is None or now_s <= self._t_s:
            # The first sample, or a clock that did not advance: there is nothing to blend from.
            self._value, self._t_s = median, now_s
            return median
        alpha = 1.0 - math.exp(-2.0 * math.pi * self.cutoff_hz * (now_s - self._t_s))
        self._value += alpha * (median - self._value)
        self._t_s = now_s
        return self._value

    def reset(self) -> None:
        """Forgets every sample, so the next one passes through as if the filter were new."""
        self._window.clear()
        self._value = None
        self._t_s = None


class GripperForceCap:
    """One arm's grip force cap: the back-off added to that arm's commanded gripper target.

    See the module docstring for the control law. One instance per arm, updated once per tick with that tick's
    commanded target and measured gripper state; it holds the offset between ticks, so it is stateful and must
    be reset() whenever the run it was regulating ends.

    Attributes:
    - cap: Effort the grasp is held at, in i2rt's `gripper_eff` units; 0 or less passes every command through.
    - gain: Offset units per second per unit of relative effort error, the rate the back-off integrates at.
    - release: Offset units per second bled off in free motion or on a deliberate open.
    - deadband: How far above the measured position a target must sit to count as a deliberate open.
    - contact_frac: Fraction of `cap` the filtered effort must pass for the gripper to count as in contact.
    - contact_speed: The jaw must also move slower than this, gripper units per second, to count as in contact.
    - stiffness: Effort per gripper unit of squeeze (#DEFAULT_STIFFNESS); sets the hard floor's distance.
    - max_offset: Ceiling on the back-off, in gripper units; a cap that cannot be met stops here.
    - filter: The #EffortFilter the measured effort passes through.
    - offset: The current back-off, in gripper units; 0 in free motion.
    - effort: The filtered effort as of the last update(), for logs.
    - speed: The jaw's low-passed speed as of the last update(), gripper units per second.

    @see update(), reset()
    """

    def __init__(
        self,
        cap: float = DEFAULT_CAP,
        *,
        gain: float = 0.4,
        release: float = 0.5,
        deadband: float = 0.02,
        contact_frac: float = 0.25,
        contact_speed: float = 0.3,
        stiffness: float = DEFAULT_STIFFNESS,
        max_offset: float = 1.0,
        cutoff_hz: float = DEFAULT_CUTOFF_HZ,
    ) -> None:
        """Builds a cap with the default tuning; nothing is regulated until the first update().

        @param cap Effort to hold a grasp at, in i2rt's `gripper_eff` units; 0 or less makes update() a
            pass-through, which is how the config disables the cap.
        @param gain Offset units per second per unit of relative effort error.
        @param release Offset units per second bled off in free motion or on a deliberate open.
        @param deadband Gripper units a target must sit above the measured position to read as opening.
        @param contact_frac Fraction of `cap` the filtered effort passes for contact to be declared.
        @param contact_speed Jaw speed, gripper units per second, under which a high effort counts as contact.
        @param stiffness Effort per gripper unit of squeeze; the floor sits `cap / stiffness` closed of the jaw.
        @param max_offset Ceiling on the back-off, in gripper units; the whole range, since a thick object holds the
            jaw at 0.4 or more open while the policy commands 0.
        @param cutoff_hz Low-pass corner of the effort filter, Hz.
        """
        self.cap = float(cap)
        self.gain = float(gain)
        self.release = float(release)
        self.deadband = float(deadband)
        self.contact_frac = float(contact_frac)
        self.contact_speed = float(contact_speed)
        self.stiffness = float(stiffness)
        self.max_offset = float(max_offset)
        self.filter = EffortFilter(cutoff_hz)
        self.offset = 0.0
        self.effort = 0.0
        self.speed = 0.0
        self._t_s: float | None = None
        self._measured: float | None = None

    def update(self, target: float, measured: float, effort: float | None, now_s: float) -> float:
        """Regulates one tick's gripper command: the target in free motion, the target backed off in contact.

        @param target The commanded gripper opening this tick, in `[0, 1]`; anything outside is clipped in.
        @param measured The measured gripper opening this tick, in `[0, 1]`; compared with `target` to tell a
            deliberate open from a squeeze.
        @param effort The measured gripper effort, or None when there is no reading: the back-off held so far
            is kept, and a driver that never reports one passes the target through.
        @param now_s Tick time on the loop's clock, seconds.
        @return The gripper command to send, in `[0, 1]`: never below `target`, so the cap can only open the
            gripper, never close it further.
        """
        target = min(1.0, max(0.0, float(target)))
        if self.cap <= 0.0:
            return target
        if effort is None:
            return min(1.0, target + self.offset)
        self.effort = self.filter.update(abs(float(effort)), now_s)
        step_s = 0.0 if self._t_s is None or now_s <= self._t_s else min(_MAX_STEP_S, now_s - self._t_s)
        self._t_s = now_s
        if self._measured is not None and step_s > 0.0:
            jaw_speed = abs(measured - self._measured) / step_s
            self.speed += (1.0 - math.exp(-2.0 * math.pi * self.filter.cutoff_hz * step_s)) * (jaw_speed - self.speed)
        self._measured = measured
        is_opening = target > measured + self.deadband
        is_in_contact = self.effort > self.contact_frac * self.cap and self.speed < self.contact_speed
        if is_opening or not is_in_contact:
            # Free motion or a deliberate open: there is nothing to regulate, and the offset bleeds away so
            # the next grasp starts from zero.
            self.offset -= self.release * step_s
        else:
            # In contact: integrate the signed error. At `effort == cap` the error is zero and the offset
            # freezes, so the gripper parks where the effort reads the cap. No decay term fights that
            # equilibrium, which is what keeps it from hunting or creeping back into the raw target.
            self.offset += self.gain * (self.effort / self.cap - 1.0) * step_s
        self.offset = min(self.max_offset, max(0.0, self.offset))
        if is_in_contact and not is_opening:
            # The floor and the ceiling, folded into the offset so the integrator resumes from what was sent
            # (anti-windup): never squeezing harder than the cap needs, never letting go of the object.
            floor = measured - self.cap / self.stiffness
            command = min(measured, max(floor, target + self.offset))
            self.offset = min(self.max_offset, max(0.0, command - target))
        return min(1.0, target + self.offset)

    def reset(self) -> None:
        """Drops the back-off and the filter's history, so the next grasp is regulated from scratch.

        Called when the run the cap was regulating ends or restarts (#robots.control starts a ramp): the arm
        may be anywhere by the time the next grasp begins, and an offset carried across it would open the
        gripper for a squeeze that never happened.
        """
        self.offset = 0.0
        self.effort = 0.0
        self.speed = 0.0
        self.filter.reset()
        self._t_s = None
        self._measured = None
