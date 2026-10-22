"""The setup wizard's powered checks: the arms move under power, slowly, after the operator confirmed the safety popup.

#Motion runs on its own thread, on arms opened powered by their body's `open_arm(cfg, side)`
(`robots.setup_session.open_body_arm`) and used only through `read()`, `write()`, `hold()` and `disengage()`. It sends
each arm its own position (nothing may move), moves both to their home, nudges every joint 5 degrees and back and each
gripper a little, then runs one small end-effector sweep from home, the same for every robot, solved with dream's IK
(#robots.setup_model.ArmModel.solve), and goes back to home. Each step is judged by #robots.setup_checks. Stop, or a
failed check of how the arm moved, leaves both arms holding where they are, and the page asks the operator to hold
them up and let them go (`robots.setup_session.Session.release`). A failed check that says the write path itself is
wrong (#robots.setup_checks.LET_GO_CAUSES: a joint that runs far from what was sent, a joint farther from home than any
test move, the gripper going elsewhere) or an error lets both arms go at once with `disengage()`, since a hold goes
through that same write. A joint that stops short of what was sent and stays still is blocked: that arm is parked and
switched off and the other holds, as for its motor guard's alarm (#robots.setup_checks.PARK_CAUSES). The page draws
the arms from `Motion.latest` meanwhile, the joint being nudged (`Motion.nudging`) highlighted, and at the end the
operator says whether each joint and the hand moved like the drawing (Motion.eyes()).

@see Motion, SWEEP_MOVES
"""

import logging
import math
import threading
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from robots import control, setup_checks, setup_devices, setup_model, types
from robots.setup_checks import Finding, Reading

## The size of the nudge, radians (5 degrees).
NUDGE_RAD: float = math.radians(5.0)

## Seconds the nudge takes out, and back.
NUDGE_MOVE_S: float = 1.0

## Seconds a moved arm is given to settle before it is read back.
SETTLE_S: float = 0.6

## The gripper's nudge, on its 0 to 1 scale.
GRIPPER_NUDGE: float = 0.2

## Fastest any joint moves on the way to home, radians per second.
HOME_SPEED_RAD_S: float = 0.25

## Least time the move to home takes, seconds.
HOME_MIN_S: float = 2.0

## Each sweep move as the page words it, for its notes.
SWEEP_WORDS: dict[str, str] = {
    "+x": "Moving the hand 3 cm forward",
    "-x": "Moving the hand 3 cm back",
    "+y": "Moving the hand 3 cm left",
    "-y": "Moving the hand 3 cm right",
    "+z": "Moving the hand 3 cm up",
    "-z": "Moving the hand 3 cm down",
    "turn about x": "Rolling the hand 10 degrees",
    "turn about y": "Tilting the hand 10 degrees",
    "turn about z": "Turning the hand 10 degrees",
}

## Longest the move home before Switch off the motors may take (home_before_off()), seconds: well past the slowest
## such move (#HOME_SPEED_RAD_S over the farthest a held arm may be from home); past it the motors go off anyway.
HOME_OFF_TIMEOUT_S: float = 15.0

## The note a run gets when Switch off the motors took both arms home first.
HOMED_NOTE: str = "Moved the arms home, then switched the motors off."

## The sweep's moves from home: a name, a shift in metres and a turn as a rotation vector in radians, both in the arm's
## base frame through the tool point. The same for every robot.
SWEEP_MOVES: tuple[tuple[str, tuple[float, float, float], tuple[float, float, float]], ...] = (
    ("+x", (0.03, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ("-x", (-0.03, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ("+y", (0.0, 0.03, 0.0), (0.0, 0.0, 0.0)),
    ("-y", (0.0, -0.03, 0.0), (0.0, 0.0, 0.0)),
    ("+z", (0.0, 0.0, 0.03), (0.0, 0.0, 0.0)),
    ("-z", (0.0, 0.0, -0.03), (0.0, 0.0, 0.0)),
    ("turn about x", (0.0, 0.0, 0.0), (math.radians(10.0), 0.0, 0.0)),
    ("turn about y", (0.0, 0.0, 0.0), (0.0, math.radians(10.0), 0.0)),
    ("turn about z", (0.0, 0.0, 0.0), (0.0, 0.0, math.radians(10.0))),
)

## A lifting joint's sweep move counts toward its tracking advice when the move turned it at least this far, radians
## (2 degrees): under that, a joint's dead band is most of the move.
ADVICE_MIN_MOVE_RAD: float = math.radians(2.0)

## Seconds of each sweep move: out, held there, and back.
SWEEP_OUT_S: float = 1.0
SWEEP_HOLD_S: float = 0.6
SWEEP_BACK_S: float = 1.0

## A sweep target the IK misses by more than this is out of reach from home, metres and radians.
REACH_POS_M: float = 0.005
REACH_ANG_RAD: float = math.radians(3.0)

## Most a sweep move may turn any joint from home, radians (20 degrees). From a well-placed home every move needs far
## less (a Makers Arm at rest: 17 at most); a folded, near-singular home asks a base or wrist swing of 65 or more for a
## 3 cm or 10 degree move (the Metal arm's recorded start). A move past it is shrunk like an unreachable one.
SWEEP_MAX_TURN_RAD: float = math.radians(20.0)

TOO_FAR_MARGIN_RAD: float = math.radians(5.0)

## How many times a sweep move out of reach or past #SWEEP_MAX_TURN_RAD is halved before it is skipped.
SWEEP_SHRINKS: int = 2

## A powered joint this far from what was sent, for #WATCHDOG_TICKS ticks in a row, stops everything, radians.
WATCHDOG_RAD: float = math.radians(12.0)
WATCHDOG_TICKS: int = 6

STILL_RAD: float = math.radians(1.0)

## Most a joint may move when the wizard sends the arm its own position (the hold check), radians.
HOLD_CHECK_RAD: float = math.radians(3.0)

## Most the gripper may move in the hold check, on its 0 closed to 1 open scale.
HOLD_CHECK_GRIPPER: float = 0.15

_log = logging.getLogger(__name__)


class Stopped(Exception):
    """The operator pressed Stop, or a check failed: the run ends."""


class Motion:
    """The powered checks, on their own thread: the hold check, home, the nudges, then the sweep.

    Runs only after the operator confirmed the safety popup, on arms opened powered (each holds where it is):
    the arm is sent its own position and must not move (a driver that reads and writes differently moves here); both
    go slowly to their home; each joint of each arm is nudged #NUDGE_RAD and back, and each gripper a little; then
    both arms run the same small end-effector sweep from home (#SWEEP_MOVES), solved with dream's IK and judged by
    forward kinematics of where each arm got. Stop ends it with both arms holding. The first failed check ends it too,
    since nothing after it would mean anything: with both arms holding, or let go at once when the failure says the
    write path is wrong (the hold check, a joint #TOO_FAR_MARGIN_RAD past any test move, a YAM joint #WATCHDOG_RAD
    from what was sent; #robots.setup_checks.LET_GO_CAUSES) or the run raised. An arm that drops when limp with a
    joint #WATCHDOG_RAD from what was sent, or whose motor guard raises its alarm (a motor too hot), is parked and
    switched off (park()), and the other holds. The run stays on (`parking`) until every park ends, since a park
    drives the arm to rest under power; STOP then lets every arm go limp at once (press_stop(), browser B29).

    Attributes:
    - state: `idle`, `running`, `parking` (a failed check's parks still run), `done`, `failed` or `stopped`.
    - let_go: True once the run let both arms go (disengaged) after its failure; they hold otherwise.
    - parked: Per side, why the arm was parked and switched off (park()): its alarm, or the blocked joint.
    - off_failed: Per side, why an arm was to be parked whose switch-off failed (park()): it may still be powered.
    - limp_off: The parked arms that were let go limp first (STOP during their park, browser B37): off where they
      stood, not at rest.
    - stop_cut: True once STOP let every arm go during a park (press_stop()).
    - parking: The arms whose park runs or is still to run while #state is `parking`.
    - phase: What runs now.
    - results: Every check done: `{"group", "side", "label", "finding"}`.
    - notes: What the wizard changed and why, in plain words, each for one arm: `{"side", "text"}`, and `move`
      with `skipped` for a sweep move it shrank or skipped, or a lifting joint's tracking advice; a note about both
      arms (the move home before the motors go off) is `{"text"}` alone.
    - latest: Per side, the arm's last reading, which the page draws.
    - nudging: `{"side", "joint"}` while that joint is nudged, which the page highlights; else None.
    """

    def __init__(
        self,
        body: str,
        model: setup_model.ArmModel,
        arms: dict[str, Any],
        home: dict[str, npt.NDArray[np.float64]],
        *,
        clock: Callable[[], float],
        sleep: Callable[[float], Any],
    ) -> None:
        """Binds the opened arms and the home pose; nothing moves until run().

        @param body The config's `body`, for its #robots.setup_checks.Tolerance.
        @param model The arm's URDF.
        @param arms Per side, the arm, opened powered.
        @param home Per side, the home joints.
        @param clock Monotonic clock, seconds.
        @param sleep Blocking sleep, seconds.
        """
        self._model = model
        self._arms = arms
        self._home = home
        self._clock = clock
        self._sleep = sleep
        self._tolerance = setup_checks.tolerance_for(body)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._commanded: dict[str, types.ArmCommand] = {}
        self._reach: dict[str, npt.NDArray[np.float64]] = {}
        self._strays = 0
        self._stray_from: dict[str, npt.NDArray[np.float64]] = {}
        self._straying: set[str] = set()
        self._lifting: dict[str, frozenset[int]] = {}
        self._nudged: dict[str, npt.NDArray[np.float64]] = {}
        self._traces: dict[str, list[Reading]] = {}
        self.state = "idle"
        self.let_go = False
        self.parked: dict[str, str] = {}
        self.off_failed: dict[str, str] = {}
        self.limp_off: list[str] = []
        self.stop_cut = False
        self.parking: list[str] = []
        self.phase = ""
        self.results: list[dict[str, Any]] = []
        self.notes: list[dict[str, Any]] = []
        self.latest: dict[str, Reading] = {}
        self.nudging: dict[str, Any] | None = None
        self.error: str | None = None
        self._deadline_s: float | None = None
        self._ended: tuple[str, str] = ("", "")

    def stop(self) -> None:
        """Asks the run to stop; the arms hold where they are, and a park that runs ends at rest."""
        self._stop.set()

    def press_stop(self) -> None:
        """The page's STOP: stop(), and during a park every arm goes limp at once (browser B29)."""
        self._stop.set()
        with self._lock:
            if self.state == "parking":
                _log.warning("setup: STOP while the %s arm parks; letting every arm go", " and ".join(self.parking))
                self.stop_cut = True
                self._let_go_all()

    def run(self) -> None:
        """Runs every powered check in order; both arms end holding at home or where they stopped, or let go."""
        self.state = "running"
        try:
            for side, arm in self._arms.items():
                start = np.abs(setup_checks.reading_of(arm.read(), self._clock()).joints - self._home[side])
                self._reach[side] = np.maximum(start, SWEEP_MAX_TURN_RAD) + TOO_FAR_MARGIN_RAD
                self._lifting[side] = self._model.lifting(self._home[side], side)
            self._hold_check()
            self.phase = "moving to home"
            self._move_to({side: self._home[side] for side in self._arms}, None)
            self._pause(SETTLE_S)
            for side in self._arms:
                self._nudges(side)
            self._sweep()
            self.phase = "back to home"
            self._move_to({side: self._home[side] for side in self._arms}, None)
            self.phase = "done"
            self.state = "done"
        except Stopped as exc:
            last = self.results[-1]["finding"] if self.results else None
            ended = "failed" if last is not None and last.state == "fail" else "stopped"
            self.error = str(exc)
            if last is not None and last.cause in setup_checks.PARK_CAUSES:
                side = self.results[-1]["side"]
                self._park_then_hold(list(self._arms) if side == "both" else [side], last.message, ended)
                return
            self.state = ended
            if last is not None and last.cause in setup_checks.LET_GO_CAUSES:
                self._let_go_all()
            else:
                self._hold_all()
        except Exception as exc:
            _log.exception("setup: the powered checks failed")
            self.error = setup_devices.explain(exc)
            self.state = "failed"
            self._let_go_all()
        finally:
            self.nudging = None

    def begin_home_off(self) -> None:
        """Shows the run as moving again for home_before_off().

        Called before its thread starts, so the page's next poll already has the STOP bar.
        """
        self._ended = (self.state, self.phase)
        self.state, self.phase = "running", "moving the arms home, then switching the motors off"

    def home_before_off(self, sides: list[str]) -> str | None:
        """Drives the held arms home before Switch off the motors lets them go; the run then shows as ended.

        The same move as the checks' own way home (#HOME_SPEED_RAD_S, watched every tick: STOP, an alarm, a joint far
        from what was sent, a joint past its reach), cut off at #HOME_OFF_TIMEOUT_S. Whatever ends it, the caller
        switches the motors off next; a check that trips here is not added to the run's results, it is the note.

        @param sides The arms that hold.
        @return None when they got home, else why not.
        """
        results = len(self.results)
        self._stop.clear()
        self._strays, self._straying = 0, set()
        self._deadline_s = self._clock() + HOME_OFF_TIMEOUT_S
        missed: str | None = None
        try:
            self._move_to({side: self._home[side] for side in sides}, None)
        except Stopped as exc:
            missed = str(exc)
        except Exception as exc:
            _log.exception("setup: the move home before switching off failed")
            missed = setup_devices.explain(exc)
        finally:
            self._deadline_s = None
            del self.results[results:]
            self.state, self.phase = self._ended
        note = (
            HOMED_NOTE
            if missed is None
            else (
                f"The arms did not reach home before the motors went off ({missed}); they are off where they stopped."
            )
        )
        self.notes.append({"text": note})
        _log.info("setup: %s", note)
        return missed

    def eyes(self, same: bool) -> None:
        """Takes the operator's answer, after the run, to whether each joint and the hand moved like the drawing.

        A driver that reads and writes a joint the same wrong way passes every number, so only this catches it when
        home did not.

        @param same True when they did; False fails the run with the cause `eyes`.
        """
        label = "each joint and the hand moved like the drawing"
        said = label if same else "you saw a joint or the hand move unlike the drawing"
        finding = Finding("pass", said) if same else Finding("fail", said, "eyes")
        self.results.append({"group": "eyes", "side": "both", "label": label, "finding": finding})
        if not same:
            self.state, self.error = "failed", said

    def status(self) -> dict[str, Any]:
        """What the page shows while the checks run and after.

        @return `{"state", "let_go", "stop_cut", "phase", "nudging", "results", "notes", "error"}`.
        """
        return {
            "state": self.state,
            "let_go": self.let_go,
            "stop_cut": self.stop_cut,
            "phase": self.phase,
            "nudging": self.nudging,
            "results": [dict(row, finding=row["finding"].as_json()) for row in self.results],
            "notes": list(self.notes),
            "error": self.error,
        }

    def _record(self, group: str, side: str, label: str, finding: Finding) -> None:
        """Adds a check's result; a fail ends the run."""
        self.results.append({"group": group, "side": side, "label": label, "finding": finding})
        if finding.state == "fail":
            raise Stopped(f"{side} arm: {finding.message}")

    def _read(self, side: str) -> Reading:
        """One reading of one arm, kept as its #latest; a joint past its reach from home fails the run (`too_far`)."""
        reading = setup_checks.reading_of(self._arms[side].read(), self._clock())
        self.latest[side] = reading
        if side in self._traces:
            self._traces[side].append(reading)
        past = np.abs(reading.joints - self._home[side]) - self._reach[side]
        if float(np.max(past)) > 0.0:
            joint = int(np.argmax(past))
            if side in self._straying and self._arms[side].drops_when_limp:
                short = math.degrees(abs(reading.joints[joint] - self._commanded[side].positions_rad[joint]))
                self._blocked(side, f"joint {joint + 1}", joint, short, moved=True)
            said = f"joint {joint + 1} moved farther from home than any test move"
            self._record("limit", side, "stays near home", Finding("fail", said, "too_far", joint))
        return reading

    def park(self, side: str, why: str) -> bool:
        """Parks the arm and switches it off (`FollowerArm.torque_off()`): its motor guard's alarm, or a blocked joint.

        @param side The arm.
        @param why Its alarm in the guard's words, or what the blocked joint did.
        @return True when it is off.
        """
        _log.error("setup: %s; parking the %s arm and switching it off", why, side)
        off = control.run_teardown(f"park {side} arm", self._arms[side].torque_off)
        (self.parked if off else self.off_failed)[side] = why
        if off and self.let_go:
            self.limp_off.append(side)
        return off

    def _park_then_hold(self, sides: list[str], why: str, ended: str) -> None:
        self.nudging = None
        with self._lock:
            self.state, self.parking = "parking", sides
            arms = f"the {sides[0]} arm is" if len(sides) == 1 else "both arms are"
            self.phase = f"{arms} moving to rest under power, then switching off"
        for side in sides:
            self.park(side, why)
        with self._lock:
            self.state, self.parking = ended, []
            if not self.let_go:
                self._hold_all()

    def _hold_all(self) -> None:
        """Every arm not parked holds where it is; a failing hold is logged, since the run is over either way."""
        for side, arm in self._arms.items():
            if side in self.parked or side in self.off_failed:
                continue
            try:
                arm.hold()
            except Exception:
                _log.exception("setup: the %s arm did not take a hold", side)

    def _let_go_all(self) -> None:
        self.let_go = True
        for side, arm in self._arms.items():
            if side in self.parked or side in self.off_failed:
                continue
            try:
                arm.disengage()
            except Exception:
                _log.exception("setup: the %s arm did not let go", side)

    def _write(self, side: str, joints: npt.NDArray[np.float64], gripper: float) -> None:
        """Sends one arm one target and remembers it for the watchdog."""
        command = types.ArmCommand(positions_rad=np.asarray(joints, dtype=np.float64), gripper=float(gripper))
        self._arms[side].write(command)
        self._commanded[side] = command

    def _watch(self) -> None:
        """Stops the run on Stop, an arm's motor guard alarm, or a joint #WATCHDOG_RAD from its target too long."""
        if self._stop.is_set():
            raise Stopped("stopped by the operator")
        if self._deadline_s is not None and self._clock() > self._deadline_s:
            raise Stopped(f"not home within {HOME_OFF_TIMEOUT_S:g} s")
        for side, arm in self._arms.items():
            alarm = arm.alarm()
            if alarm is not None:
                self._record("alarm", side, "its motors stay within their guard", Finding("fail", alarm, "alarm"))
        joints = {side: self._read(side).joints for side in self._commanded}
        far = {side: np.abs(q - self._commanded[side].positions_rad) > WATCHDOG_RAD for side, q in joints.items()}
        self._straying = {side for side, mask in far.items() if mask.any()}
        if not self._straying:
            self._strays = 0
            return
        if self._strays == 0:
            self._stray_from = joints
        self._strays += 1
        if self._strays < WATCHDOG_TICKS:
            return
        moved = {side: np.abs(q - self._stray_from[side]) for side, q in joints.items() if far[side].any()}
        if not all(self._arms[side].drops_when_limp for side in moved):
            finding = Finding("fail", "a joint stayed far from where we sent it", "not_followed")
            self._record("watchdog", "both", "the arm follows what we send", finding)
        side = next(iter(moved)) if len(moved) == 1 else "both"
        first = next(iter(moved))
        joint = int(np.argmax(far[first]))
        short = math.degrees(abs(joints[first][joint] - self._commanded[first].positions_rad[joint]))
        which = f"joint {joint + 1}" if side != "both" else f"joint {joint + 1} of the {first} arm"
        self._blocked(side, which, joint, short, moved=float(np.max(moved[first][far[first]])) > STILL_RAD)

    def _blocked(self, side: str, which: str, joint: int, short: float, moved: bool) -> None:
        how = f"stopped {short:.0f} degrees short of where we sent it and stayed there"
        if moved:
            how = f"stayed {short:.0f} degrees from where we sent it while it moved"
        said = f"{which} {how}: something is in its way or presses on the arm, or it presses on the table or on itself"
        self._record("watchdog", side, "the arm follows what we send", Finding("fail", said, "blocked", joint))

    def _hold_check(self) -> None:
        """Sends each arm its own position, the first write of all: nothing may move, the gripper included, checked on
        every tick."""
        self.phase = "hold check"
        before = {side: self._read(side) for side in self._arms}
        for side, reading in before.items():
            self._write(side, reading.joints, _clip01(reading.gripper))
        moved = dict.fromkeys(before, 0.0)
        end = self._clock() + SETTLE_S
        while self._clock() < end:
            if self._stop.is_set():
                raise Stopped("stopped by the operator")
            for side, reading in before.items():
                now = self._read(side)
                gap = np.abs(now.joints - reading.joints)
                moved[side] = max(moved[side], float(np.max(gap)))
                grip = abs(now.gripper - reading.gripper)
                if grip > HOLD_CHECK_GRIPPER:
                    said = f"the gripper moved {grip:.2f} when we sent it where it was"
                    self._record(
                        "hold", side, "holds still when sent where it is", Finding("fail", said, "gripper_write")
                    )
                if moved[side] > HOLD_CHECK_RAD:
                    joint = int(np.argmax(gap))
                    said = (
                        f"joint {joint + 1} moved {math.degrees(moved[side]):.0f} degrees when we sent the arm where "
                        "it was. If it is still moving or pushing, press your emergency stop"
                    )
                    # A joint that lifts the arm and sank a few degrees sags under the arm's weight (a Maker Arm's
                    # shoulder, no gravity torque): a warning, not a write() that took the wrong joints.
                    lower = self._model.tool(now.joints, side)[2, 3] < self._model.tool(reading.joints, side)[2, 3]
                    sags = (
                        lower
                        and moved[side] <= 3 * HOLD_CHECK_RAD
                        and joint in self._model.lifting(reading.joints, side)
                    )
                    cause = "weight" if sags else "not_followed"
                    self._record(
                        "hold", side, "holds still when sent where it is", Finding("fail", said, cause, joint)
                    )
            self._sleep(1.0 / setup_model.MOTION_HZ)
        for side in before:
            said = f"moved at most {math.degrees(moved[side]):.1f} degrees"
            self._record("hold", side, "holds still when sent where it is", Finding("pass", said))

    def _move_to(self, targets: dict[str, npt.NDArray[np.float64]], gripper: dict[str, float] | None) -> None:
        """Moves arms in a straight line in joint space, slowly (#HOME_SPEED_RAD_S), watching every tick."""
        starts = {side: self._read(side) for side in targets}
        span = max(float(np.max(np.abs(targets[side] - starts[side].joints))) for side in targets)
        duration = max(HOME_MIN_S, span / HOME_SPEED_RAD_S)
        grips = gripper or {side: _clip01(starts[side].gripper) for side in targets}
        self._glide(
            {side: (starts[side].joints, targets[side], starts[side].gripper, grips[side]) for side in targets},
            duration,
        )

    def _glide(self, paths: dict[str, tuple[Any, Any, float, float]], duration: float) -> None:
        """Writes straight-line joint and gripper paths over `duration`, watching each tick.

        The rate is #robots.setup_model.MOTION_HZ.

        @param paths Per side, `(joints from, joints to, gripper from, gripper to)`.
        @param duration Seconds.
        """
        ticks = max(1, round(duration * setup_model.MOTION_HZ))
        start = self._clock()
        for tick in range(1, ticks + 1):
            share = _ease(tick / ticks)
            for side, (q_from, q_to, g_from, g_to) in paths.items():
                q = np.asarray(q_from) + share * (np.asarray(q_to) - np.asarray(q_from))
                self._write(side, q, _clip01(g_from + share * (g_to - g_from)))
            self._watch()
            self._sleep(max(0.0, start + tick / setup_model.MOTION_HZ - self._clock()))

    def _pause(self, seconds: float) -> None:
        """Holds still for `seconds`, watching."""
        end = self._clock() + seconds
        while self._clock() < end:
            self._watch()
            self._sleep(min(1.0 / setup_model.MOTION_HZ, max(0.0, end - self._clock())))

    def _nudges(self, side: str) -> None:
        """Each joint of one arm 5 degrees from home and back (not the joints it does not have), then the gripper a
        little."""
        home = self._home[side]
        followed = False
        self._nudged[side] = np.full(len(home), np.nan)
        for joint in range(len(home)):
            if joint in self._model.held:
                continue
            self.phase = f"nudging the {side} arm's joint {joint + 1}"
            self.nudging = {"side": side, "joint": joint}
            sent = NUDGE_RAD if home[joint] + NUDGE_RAD <= self._model.upper[joint] else -NUDGE_RAD
            before = self._read(side)
            self._traces[side] = [before]
            target = home.copy()
            target[joint] += sent
            grip = _clip01(before.gripper)
            self._glide({side: (home, target, grip, grip)}, NUDGE_MOVE_S)
            self._pause(SETTLE_S)
            self._read(side)
            path = self._traces.pop(side)
            lifts = joint in self._lifting[side]
            finding = setup_checks.judge_nudge(joint, sent, path, NUDGE_MOVE_S, lifts=lifts, followed=followed)
            self._nudged[side][joint] = float(path[-1].joints[joint] - path[0].joints[joint]) / sent
            self._glide({side: (target, home, grip, grip)}, NUDGE_MOVE_S)
            self._pause(SETTLE_S / 2)
            self.nudging = None
            self._record("nudge", side, f"joint {joint + 1} {math.degrees(sent):+.0f} degrees", finding)
            followed = True
        self.phase = f"nudging the {side} gripper"
        before = self._read(side)
        start = _clip01(before.gripper)
        sent = -GRIPPER_NUDGE if start > 0.5 else GRIPPER_NUDGE
        self._glide({side: (home, home, start, start + sent)}, NUDGE_MOVE_S)
        self._pause(SETTLE_S)
        after = self._read(side)
        finding = setup_checks.judge_gripper_nudge(sent, before.gripper, after.gripper)
        self._glide({side: (home, home, start + sent, start)}, NUDGE_MOVE_S)
        self._record("nudge", side, f"gripper {sent:+.1f}", finding)

    def _sweep(self) -> None:
        """Both arms through #SWEEP_MOVES from home, each solved with dream's IK and judged by how the tool moved; then
        a note per arm on how its lifting joints followed up against down (#robots.setup_checks.tracking_advice)."""
        sides = list(self._arms)
        home_pose = {side: self._model.tool(self._home[side], side) for side in self._model.sides}
        grips = {side: _clip01(self._read(side).gripper) for side in sides}
        rise = {side: self._model.rise(self._home[side], side) for side in sides}
        # Per side and lifting joint, how far short of where it was sent it held in each move that turned it up, and
        # down: from the target, not from where the move started, so a steady sag shows (down both ways) apart from
        # friction (behind the move both ways).
        fell: dict[str, dict[int, tuple[list[float], list[float]]]] = {
            side: {joint: ([], []) for joint in self._lifting[side]} for side in sides
        }
        for name, shift, turn in SWEEP_MOVES:
            self.phase = f"sweep: {name}"
            path = self._reachable_path(name, home_pose, np.asarray(shift), np.asarray(turn))
            if path is None:
                continue
            q, hold_end = path
            start_q = {side: self._read(side).joints for side in sides}
            before = {side: self._model.tool(start_q[side], side) for side in sides}
            start = self._clock()
            for tick in range(q[sides[0]].shape[0]):
                for side in sides:
                    self._write(side, q[side][tick], grips[side])
                self._watch()
                if tick == hold_end:
                    for side in sides:
                        now = self._read(side).joints
                        reached = (before[side], self._model.tool(now, side))
                        sent = (home_pose[side], self._model.tool(q[side][tick], side))
                        moved = q[side][tick] - self._home[side]
                        short = (now - start_q[side]) - moved
                        finding = setup_checks.judge_reach(
                            name,
                            bool(np.any(turn)),
                            sent,
                            reached,
                            self._tolerance,
                            short,
                            self._lifting[side],
                            moved,
                            self._nudged.get(side),
                        )
                        for joint, (up, down) in fell[side].items():
                            if abs(moved[joint]) >= ADVICE_MIN_MOVE_RAD:
                                way = up if moved[joint] * rise[side][joint] > 0 else down
                                off = float(now[joint] - q[side][tick][joint])
                                way.append(-off * math.copysign(1.0, moved[joint]))
                        self._record("sweep", side, name, finding)
                self._sleep(max(0.0, start + (tick + 1) / setup_model.MOTION_HZ - self._clock()))
            self._pause(SETTLE_S / 2)
        for side in sides:
            for joint, (up, down) in sorted(fell[side].items()):
                advice = setup_checks.tracking_advice(joint, up, down)
                if advice is not None:
                    self.notes.append({"side": side, "text": advice})

    def _reachable_path(
        self, name: str, home_pose: dict[str, Any], shift: npt.NDArray[np.float64], turn: npt.NDArray[np.float64]
    ) -> tuple[dict[str, npt.NDArray[np.float64]], int] | None:
        """One sweep move's joints for both arms (out, held, back), shrunk while it misses or turns a joint too far.

        Too far is more than #SWEEP_MAX_TURN_RAD from home.

        @return `(per side [ticks, 6], the last held tick)`, or None when even the smallest misses or turns too far
            (noted).
        """
        out = round(SWEEP_OUT_S * setup_model.MOTION_HZ)
        hold = round(SWEEP_HOLD_S * setup_model.MOTION_HZ)
        back = round(SWEEP_BACK_S * setup_model.MOTION_HZ)
        shares = np.concatenate(
            [_ease(np.arange(1, out + 1) / out), np.ones(hold), _ease(1.0 - np.arange(1, back + 1) / back)]
        )
        scale = 1.0
        reasons: dict[str, list[str]] = {side: [] for side in self._arms}  # per arm, why each size was too big
        for _ in range(SWEEP_SHRINKS + 1):
            targets = {
                side: np.stack([setup_model.pose_offset(pose, shift * scale * s, turn * scale * s) for s in shares])
                for side, pose in home_pose.items()
            }
            solved = self._model.solve(targets, {side: self._home[side] for side in self._model.sides})
            held = slice(out, out + hold)
            fits = True
            for side in self._arms:
                far = (
                    float(np.max(solved[side]["pos_miss"][held])) > REACH_POS_M
                    or float(np.max(solved[side]["ang_miss"][held])) > REACH_ANG_RAD
                )
                turned = float(np.max(np.abs(solved[side]["q"] - self._home[side])))
                if far or turned > SWEEP_MAX_TURN_RAD:
                    fits = False
                    reasons[side].append(
                        "it is out of the arm's reach from its home pose"
                        if far
                        else f"a joint would turn {math.degrees(turned):.0f} degrees, more than the "
                        f"{math.degrees(SWEEP_MAX_TURN_RAD):.0f} the test allows"
                    )
            if fits:
                for side, why in reasons.items():
                    if why:
                        text = f"{SWEEP_WORDS[name]}: ran at {_size(scale)}, because at full size {why[0]}."
                        self.notes.append({"side": side, "text": text, "move": name, "skipped": False})
                return {side: solved[side]["q"] for side in self._arms}, out + hold - 1
            scale /= 2.0
        for side, why in reasons.items():
            if why:
                text = f"{SWEEP_WORDS[name]}: skipped, because even at {_size(scale * 2)} {why[-1]}."
                self.notes.append({"side": side, "text": text, "move": name, "skipped": True})
        return None


def _size(scale: float) -> str:
    """A sweep move's size in words: `half size`, `a quarter of its size`, else a percentage."""
    return {0.5: "half size", 0.25: "a quarter of its size"}.get(scale, f"{scale:.0%} of its size")


def _ease(share: Any) -> Any:
    """Smoothstep of a share in [0, 1], so a move starts and ends slowly."""
    return share * share * (3.0 - 2.0 * share)


def _clip01(value: float) -> float:
    """A gripper value clipped to [0, 1], as a write requires; a reading can sit a hair outside."""
    return min(max(float(value), 0.0), 1.0)
