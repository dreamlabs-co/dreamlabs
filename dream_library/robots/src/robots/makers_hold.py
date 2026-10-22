# HoldingController is vendored from MakerMods Lab (makermodslab/metal_gripper_hold.py at 64defd4),
# Apache License 2.0. Copyright 2026 MakerMods contributors. It keeps a resting-effort baseline on top of the
# original.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""Gripper holding torque and closing-effort cap for the Makers Arm gripper (a RobStride RS00).

The gripper is a position-controlled motor, so grip force is `kp x position error`: a close commanded onto an object
leaves the error large and the motor pushes with everything it has (the vendor measured 13.5 N.m before holding
engaged). The control law and its numbers are the vendor's, not tuned here:

- closing cap, every update, holding or not: while closing, `kp` is scaled so that `kp x error` never exceeds
  `max(1 N.m, 2 x hold torque)`, and `kd` drops to #CAPPED_CLOSING_KD so a free close is not damped to a crawl. The
  scaled gain is rounded down to the MIT frame's 12-bit grid, so quantization never lifts it over the cap.
- holding: contact is a closing effort over 0.5 N.m (100 ms mean) with the jaws stopped, for 100 ms. The target is
  then regulated so the measured torque sits at the hold torque, `q = q_last + (hold - |torque|) / kp`, smoothed
  (0.19 s), never closing past the request. Opening 0.25 deg past the jaws, or 0.3 s without contact, releases it.
- resting-effort baseline (this package's addition, not the vendor's): the closing effort a jaw reports while it sits
  at its request is tracked (2 s average, at most #REST_EFFORT_MAX_NM) and subtracted before every effort test above,
  and added to the cap. A gripper can read 0.3-0.4 N.m at rest; under the vendor's law alone a capped close
  then latches "contact" before the jaw moves, and the grasp is refused.

On a Makers Arm this is what keeps a grasp from crushing, the job the loop's `grip_force_cap` does on a YAM
(#robots.gripper_force), which the Makers driver leaves alone because it reports no gripper effort. The Maker closes
toward increasing angles.

The Metal arm's gripper (#robots.metal_gripper) runs the same law as MakerMods run it there: closing toward
decreasing angles, with neither the closing cap nor the resting-effort baseline, which leaves the vendor's law
exactly.

@see GripperHold, HoldingController
"""

import collections
import math

## Hold torque the vendor ships, N.m at the motor, and the range it takes.
DEFAULT_HOLD_TORQUE_NM = 0.5
MIN_HOLD_TORQUE_NM = 0.1
MAX_HOLD_TORQUE_NM = 2.0
## `kd` while the closing cap scales `kp` (a free close under the cap).
CAPPED_CLOSING_KD = 0.02
## The MIT frame's `kp` resolution: 12 bits over 0 to 500.
_KP_STEP = 500.0 / 4095.0
## Gripper feedback older than this is not regulated on, seconds; the last frame is repeated instead.
STALE_FEEDBACK_S = 0.25
## The resting-effort baseline's time constant while the jaw sits at its request, seconds.
REST_EFFORT_TAU_S = 2.0
## The most closing effort the baseline may explain away, N.m: a real grasp must still read above the cap plus 0.5.
REST_EFFORT_MAX_NM = 0.4
## "At its request": a jaw within this of the goal in the closing direction is not being asked to close, radians
## (the MIT position loop parks a few tenths of a degree short of its target).
_AT_REQUEST_RAD = math.radians(1.0)


def clamp_hold_torque(value: float) -> float:
    """Clamps a hold torque into the vendor's range.

    @param value N.m at the motor (motor-output effort, not fingertip force).
    @return The value within #MIN_HOLD_TORQUE_NM to #MAX_HOLD_TORQUE_NM.
    @throws ValueError The value is not finite.
    """
    if not math.isfinite(value):
        raise ValueError("gripper holding torque must be finite")
    return min(MAX_HOLD_TORQUE_NM, max(MIN_HOLD_TORQUE_NM, float(value)))


def capped_kp(kp: float, kp_scale: float) -> float:
    """`kp x kp_scale` on the MIT frame's 12-bit grid, rounded down when scaled.

    The frame encoder truncates; the 1e-6 step keeps an exact grid value from rounding one step lower first.

    @param kp The motor's own position gain.
    @param kp_scale The fraction of it the cap allows; 1 or more leaves `kp` as it is.
    @return The gain to send.
    """
    if kp_scale >= 1.0:
        return kp
    return (math.floor(kp * kp_scale / _KP_STEP) + 1e-6) * _KP_STEP


class HoldingController:
    """The vendor's holding law in radians; vendored as it is, see the module docstring.

    Attributes:
    - torque_nm: The hold torque, N.m.
    - closing_direction: +1 when the jaws close toward increasing angles (the Maker), -1 the other way (the Metal).
    - cap_closing_torque: Whether the closing cap is applied.
    - rest_baseline: Whether the resting-effort baseline is learned and subtracted.
    - limits: The gripper's soft limits, radians.
    - holding: Whether the jaws are regulating on an object now.
    - closing_capped: Whether the last command was limited by the closing cap (a free close, not holding).
    - kp_scale: The fraction of `kp` for the command update() returned; below 1 only while `closing_capped`.
    - rest_effort_nm: The closing effort the jaw reports at rest, N.m, at least 0; None until observed.

    @see update()
    """

    def __init__(
        self,
        torque_nm: float,
        limits_rad: tuple[float, float],
        *,
        closing_direction: int = 1,
        cap_closing_torque: bool = True,
        rest_baseline: bool = True,
    ) -> None:
        """Builds a controller that is not holding.

        @param torque_nm The hold torque, N.m; clamped into the vendor's range.
        @param limits_rad The gripper's soft limits, radians.
        @param closing_direction +1 or -1, see the class docstring.
        @param cap_closing_torque Apply the closing cap.
        @param rest_baseline Learn the resting-effort baseline; off, `rest_effort_nm` stays None and the law is the
            vendor's.
        @throws ValueError The direction is neither +1 nor -1, or the torque is not finite.
        """
        if closing_direction not in (-1, 1):
            raise ValueError("closing direction must be -1 or +1")
        self.torque_nm = clamp_hold_torque(torque_nm)
        self.closing_direction = closing_direction
        self.cap_closing_torque = cap_closing_torque
        self.rest_baseline = rest_baseline
        self.limits = limits_rad
        self.holding = False
        self.closing_capped = False
        self.kp_scale = 1.0
        self.last_command: float | None = None
        self.filtered: float | None = None
        self.history: collections.deque[tuple[float, float]] = collections.deque()
        self.last_time: float | None = None
        self.contact_started: float | None = None
        self.low_effort_since: float | None = None
        self.rest_effort_nm: float | None = None

    @property
    def rest_bias_nm(self) -> float:
        """The resting effort subtracted from every effort test, N.m; 0 until observed."""
        return self.rest_effort_nm or 0.0

    @property
    def closing_cap_nm(self) -> float:
        """The closing-effort ceiling, N.m: room above the hold target for contact detection and regulation."""
        return max(1.0, 2.0 * self.torque_nm) + self.rest_bias_nm

    def update(self, goal: float, position: float, velocity: float, effort: float, kp: float, now: float) -> float:
        """One update: the gripper target to send, with `kp_scale` and `closing_capped` set for its gains.

        @param goal The requested gripper angle, radians.
        @param position The measured gripper angle, radians.
        @param velocity The measured gripper velocity, radians per second.
        @param effort The measured gripper torque, N.m.
        @param kp The gripper motor's own position gain.
        @param now The clock's reading, seconds.
        @return The target to send, radians.
        @throws ValueError An input is not finite, or `kp` is not positive.
        """
        if not all(math.isfinite(x) for x in (goal, position, velocity, effort, kp, now)) or kp <= 0:
            raise ValueError("invalid gripper holding feedback/gain")
        goal = min(self.limits[1], max(self.limits[0], goal))
        dt = 0.0 if self.last_time is None else max(0.0, now - self.last_time)
        self.last_time = now
        self.history.append((now, effort))
        while self.history and self.history[0][0] < now - 0.1:
            self.history.popleft()
        # Signed closing effort avoids treating opening acceleration as a grasp.
        closing_effort = self.closing_direction * sum(e for _, e in self.history) / len(self.history)
        request = self.closing_direction * (goal - position)
        opening = request < -math.radians(0.25)
        if self.rest_baseline and not self.holding and abs(velocity) < 0.3 and request <= _AT_REQUEST_RAD:
            # At rest at its request: the closing effort it reports now is the jaw's own load (friction, preload,
            # cable drag), not a grasp. Learn it slowly; cap what it can excuse.
            sample = min(REST_EFFORT_MAX_NM, max(0.0, self.closing_direction * effort))
            if self.rest_effort_nm is None:
                self.rest_effort_nm = sample
            else:
                self.rest_effort_nm += (1.0 - math.exp(-dt / REST_EFFORT_TAU_S)) * (sample - self.rest_effort_nm)
        grip_effort = closing_effort - self.rest_bias_nm
        if self.holding:
            # Relaxing the target first lowers the effort below its final value; require a sustained loss of contact
            # so it cannot oscillate between full-close tracking and holding.
            if grip_effort < min(0.2, self.torque_nm * 0.4):
                if self.low_effort_since is None:
                    self.low_effort_since = now
            else:
                self.low_effort_since = None
            lost_contact = self.low_effort_since is not None and now - self.low_effort_since >= 0.3
            if opening or lost_contact:
                self.holding = False
                self.contact_started = None
                self.low_effort_since = None
        elif request > 0 and grip_effort > 0.5 and abs(velocity) < 0.3:
            if self.contact_started is None:
                self.contact_started = now
            # 100 ms of contact evidence, not one acceleration sample.
            if now - self.contact_started >= 0.1:
                self.holding = True
        else:
            self.contact_started = None

        if self.holding and self.last_command is not None and self.filtered is not None:
            # q_raw = q_last + s * (target_effort - measured_effort) / Kp
            measured = max(0.0, abs(effort) - self.rest_bias_nm)
            raw = self.last_command + self.closing_direction * (self.torque_nm - measured) / kp
            # Time-based smoothing, independent of the tick rate: alpha = 0.1 at a 20 ms update.
            alpha = 1.0 - math.exp(-min(dt, 0.05) / 0.19)
            self.filtered += alpha * (raw - self.filtered)
            # Never close beyond the request; the bounded command is stored so nothing winds up.
            bounded = max(goal, self.filtered) if self.closing_direction < 0 else min(goal, self.filtered)
            command = min(self.limits[1], max(self.limits[0], bounded))
            self.filtered = command
        else:
            command = goal
            self.filtered = position
        self.closing_capped = False
        self.kp_scale = 1.0
        sent = command
        if self.cap_closing_torque:
            lead = self.closing_cap_nm / kp
            error = self.closing_direction * (command - position)
            if error > lead:
                # kp x lead at the jaws, the same closing effort either way.
                equivalent = min(self.limits[1], max(self.limits[0], position + self.closing_direction * lead))
                if self.holding:
                    command = sent = equivalent
                    self.filtered = command
                else:
                    # Aim at the real goal with kp scaled so kp x error == cap: a free close is not rate-limited, and
                    # the error only shrinks while the jaws close. Holding regulates from the full-gain equivalent.
                    self.closing_capped = True
                    self.kp_scale = lead / error
                    command = equivalent
        self.last_command = command
        return sent


class GripperHold:
    """#HoldingController in the follower's units: degrees in, `(target deg, kp, kd)` out.

    One per arm, run from that arm's tick with the gripper motor's latest feedback, its position in the same frame
    as the goal (the follower's turn-corrected degrees).

    Attributes:
    - kp: The gripper motor's own position gain.
    - kd: The gripper motor's own velocity gain.
    - limits_deg: The gripper's soft limits, degrees.
    - controller: The #HoldingController.

    @see update(), reset()
    """

    def __init__(self, kp: float, kd: float, limits_deg: tuple[float, float], torque_nm: float) -> None:
        """Builds the controller, not holding.

        @param kp The gripper motor's position gain.
        @param kd The gripper motor's velocity gain.
        @param limits_deg The gripper's soft limits, degrees.
        @param torque_nm The hold torque, N.m.
        """
        self.kp, self.kd = float(kp), float(kd)
        self.limits_deg = (float(limits_deg[0]), float(limits_deg[1]))
        self._torque_nm = clamp_hold_torque(torque_nm)
        self._last: tuple[float, float, float] | None = None
        self.controller = self._fresh()

    def _fresh(self) -> HoldingController:
        """A controller in radians over the soft limits, closing toward increasing angles."""
        low, high = self.limits_deg
        return HoldingController(self._torque_nm, (math.radians(low), math.radians(high)), closing_direction=1)

    @property
    def holding(self) -> bool:
        """Whether the jaws are regulating on an object now."""
        return self.controller.holding

    @property
    def rest_effort_nm(self) -> float | None:
        """The jaw's learned resting closing effort, N.m; None until it has sat at a request."""
        return self.controller.rest_effort_nm

    def reset(self) -> None:
        """Starts over, not holding and with no frame to repeat."""
        self.controller = self._fresh()
        self._last = None

    def update(
        self,
        goal_deg: float,
        position_deg: float,
        velocity_deg_s: float,
        torque_nm: float,
        feedback_age_s: float,
        now: float,
    ) -> tuple[float, float, float]:
        """The gripper frame to send this tick.

        With stale feedback the last frame is repeated, since regulating on an old torque would walk the target;
        before any frame, the goal passes through at the closing cap's gain for a full-stroke error.

        @param goal_deg The requested gripper angle, degrees.
        @param position_deg The measured gripper angle, degrees.
        @param velocity_deg_s The measured gripper velocity, degrees per second.
        @param torque_nm The measured gripper torque, N.m.
        @param feedback_age_s How old the measurement is, seconds.
        @param now The clock's reading, seconds.
        @return `(target deg, kp, kd)`.
        """
        if feedback_age_s > STALE_FEEDBACK_S:
            if self._last is not None:
                return self._last
            span = math.radians(abs(self.limits_deg[1] - self.limits_deg[0])) or 1.0
            return goal_deg, capped_kp(self.kp, min(1.0, self.controller.closing_cap_nm / (self.kp * span))), self.kd
        target = self.controller.update(
            math.radians(goal_deg), math.radians(position_deg), math.radians(velocity_deg_s), torque_nm, self.kp, now
        )
        kp = capped_kp(self.kp, self.controller.kp_scale)
        kd = min(self.kd, CAPPED_CLOSING_KD) if self.controller.closing_capped else self.kd
        self._last = (math.degrees(target), kp, kd)
        return self._last
