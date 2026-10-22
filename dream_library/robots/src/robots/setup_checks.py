"""The setup wizard's checks as pure judgments: readings in, a #Finding out, with the likely cause in plain words.

The wizard asks one question of every robot, whoever wrote its driver: does dream's URDF of the arm, drawn at the
numbers the driver reports, move like the real arm? A driver must read and write joints in URDF units (radians, the
URDF's zero and direction, in the URDF's order) and the gripper as 0 closed to 1 open. Each check here looks at what
the driver reported while something known happened, and says what is off:

- wiggle_scores(), moved_device(): which device's readings changed while the operator moved one arm by hand.
- judge_nudge(), judge_gripper_nudge(): the wizard sent one joint 5 degrees (or the gripper a little); it must move
  the same way, by a share of it (#NUDGE_MIN_FRACTION to #NUDGE_MAX_FRACTION). A joint that falls short is named by
  what else the readings say: whether any joint moved, whether it crept on, whether it lifts the arm.
- judge_reach(): the wizard moved the tool a little; forward kinematics of the arm's readings before and after must
  have moved the same way along the move sent by a share of it, as a nudge must, and no farther off it than the body's
  tolerance.
- tracking_advice(): what a lifting joint falling short up against down says about the arm's weight and its gravity
  compensation.

None of these open anything: the session (#robots.setup_session) records readings and calls them.

@see Finding, CAUSES
"""

import dataclasses
import math

import numpy as np
import numpy.typing as npt

from robots import setup_model, types

## A device whose joints changed this much since the baseline was moved, radians (8 degrees).
WIGGLE_MIN_RAD: float = math.radians(8.0)

## A nudged joint passes when it moved the way it was sent by at least this fraction of the move: 5x under (sent -5.0
## degrees, an SO-101 joint moved -1.8, and a stricter check said nothing moved).
NUDGE_MIN_FRACTION: float = 0.2

## A nudged joint that moved less than this fraction of what was sent did not move at all, as far as its readings tell.
NUDGE_STILL_FRACTION: float = 0.1

## Another joint than the one nudged that moved at least this fraction of the move, and more than the nudged one,
## took the move: the driver writes joints in another order than it reads them. Less is the nudged joint falling short
## while its neighbour drifts under the load (sent joint 3 +5.0 degrees on a Metal arm, it fell short
## and joint 4 drifted -1.1, which a looser check would call swapped wiring).
NUDGE_SWAP_FRACTION: float = 0.5

## A nudged gripper that moved less than this fraction of what was sent did not move.
GRIPPER_NUDGE_MIN_FRACTION: float = 0.4

## A nudged joint passes when it moved at most this many times the move: 2x over; farther, the driver writes in other
## units than it reads.
NUDGE_MAX_FRACTION: float = 2.0

## A joint that lifts the arm and fell this far short in the sweep's moves, on average one way, gets advice on how it
## tracks (a note, never a fail), radians (1 degree).
SWEEP_SHORT_RAD: float = math.radians(1.0)

## Falling this much farther short lifting than lowering is the arm's weight; lowering than lifting, gravity
## compensation pushing up too hard. Closer, the joint is soft or sticky. Radians (1 degree).
GRAVITY_GAP_RAD: float = math.radians(1.0)

## A lifting joint behind its sweep move is the arm's weight only when it fell this much more of its nudge short than a
## joint that lifts nothing did. Less, it follows like the rest of the arm (the DM4340P joints 1 to 3 moved
## 3.6 to 3.8 of their 5 degree nudge, the unloaded base yaw included).
WEIGHT_GAP_FRACTION: float = 0.1


@dataclasses.dataclass(frozen=True)
class Tolerance:
    """How closely a powered arm's sweep must follow what the wizard sends, per kind of arm.

    Attributes:
    - pos_m: The swept tool's move may differ from the move sent by this much.
    - ang_rad: And its turn from the turn sent by this much.
    """

    pos_m: float
    ang_rad: float


## Per body, how closely its sweep follows.
TOLERANCES: dict[str, Tolerance] = {
    "yam": Tolerance(pos_m=0.010, ang_rad=math.radians(3.0)),
    "makers": Tolerance(pos_m=0.012, ang_rad=math.radians(4.0)),
    "metal": Tolerance(pos_m=0.012, ang_rad=math.radians(4.0)),
    "so101": Tolerance(pos_m=0.015, ang_rad=math.radians(5.0)),
}

## A turn move may shift the tool this many times the body's #Tolerance pos_m: turning the wrist swings the tool point,
## and a lifting joint 2 degrees behind already moves it about 1.5 cm on a YAM (YAMs failed "turn about y" on
## 1.5 cm against 1.0 cm).
TURN_POS_FACTOR: float = 2.0

## The tolerance of a robot of the customer's own.
CUSTOM_TOLERANCE = Tolerance(pos_m=0.015, ang_rad=math.radians(5.0))


@dataclasses.dataclass(frozen=True)
class Cause:
    """The likely cause of a failed check, in the words the page shows.

    Attributes:
    - title: The cause, in plain words.
    - impact: What it does to the robot's runs.
    - fix: The one thing to change.
    - safe: Whether the robot can be saved and run with it as it is: it runs, less precisely, until it is fixed. False
      when it would move unlike what the model sends, or not at all, so it is fixed first.
    """

    title: str
    impact: str
    fix: str
    safe: bool = False


_MAPPING_IMPACT = "The robot moves the wrong joints, the wrong way or the wrong amount."
_NOT_FOLLOWING = "The robot will not follow the model."

## The likely causes a failed check names, by code: one the readings support, in as few words as a first-time user
## needs.
CAUSES: dict[str, Cause] = {
    "no_power": Cause(
        "The arm gets no power or no commands.",
        "The robot will not move.",
        "Check its power, its emergency stop and its cable.",
    ),
    "weight": Cause(
        "This arm cannot hold its own weight.",
        "It sags below where the model sends it.",
        "Add gravity compensation to your driver, or raise the stiffness of the lifting joints.",
        safe=True,
    ),
    "joint_stuck": Cause(
        "This joint is stuck.",
        _NOT_FOLLOWING,
        "Check that nothing blocks it and that its motor answers your driver.",
    ),
    "slow": Cause(
        "This joint follows too slowly.",
        "The robot lags behind the model.",
        "Make write() send every command at once, and raise the joint's stiffness.",
        safe=True,
    ),
    "overshoot": Cause(
        "This joint swings past where we send it.",
        "The robot overshoots and may knock things over.",
        "Lower the joint's stiffness or raise its damping.",
    ),
    "write_order": Cause(
        "Another joint moved instead of the one we sent.",
        _MAPPING_IMPACT,
        "Make write() take joints in the order read() returns them.",
    ),
    "write_flipped": Cause(
        "This joint moved the wrong way.",
        _MAPPING_IMPACT,
        "Flip this joint's sign in write() to match read().",
    ),
    "write_size": Cause(
        "This joint moved over twice as far as we sent.",
        _MAPPING_IMPACT,
        "Make write() take radians, like read().",
    ),
    "not_followed": Cause(
        "The arm did not follow what we sent.",
        _MAPPING_IMPACT,
        "Make write() take joints exactly as read() returns them.",
    ),
    "too_far": Cause(
        "This joint went farther from home than any test move.",
        _MAPPING_IMPACT,
        "Make write() take radians, in the directions read() uses.",
    ),
    "gripper_still": Cause(
        "The gripper did not move.",
        "The hand will not open or close.",
        "Check that write() sends the gripper and read() reports it, 0 closed to 1 open.",
    ),
    "gripper_write": Cause(
        "Your driver writes the gripper differently from how it reads it.",
        "The hand opens when the model closes it.",
        "Use 0 closed to 1 open in both write() and read().",
    ),
    "gripper_write_size": Cause(
        "Your driver writes the gripper in other units than it reads it.",
        "The hand opens or closes by the wrong amount.",
        "Make write() take 0 closed to 1 open, like read().",
    ),
    "blocked": Cause(
        "Something blocks the arm.",
        _NOT_FOLLOWING,
        "Clear its way, then click Scan again in Connect arms.",
    ),
    "alarm": Cause(
        "A motor got too hot, or a joint pushed too long.",
        "The robot stops in the middle of runs.",
        "Let the motor cool, or clear what it pressed against. Then click Scan again in Connect arms.",
    ),
    "eyes": Cause(
        "You saw the arm move unlike the drawing.",
        _MAPPING_IMPACT,
        "Check your driver's joint directions and order against the drawing.",
    ),
    "missed": Cause(
        "The hand missed where we sent it.",
        "The robot lands off target.",
        "Raise your joints' stiffness, and check that nothing presses on the arm.",
        safe=True,
    ),
}


## Whether to go on past a failed check, by `Cause.safe`, in the words the page shows under the fix.
GO_ON: dict[bool, str] = {
    True: "You can continue and save. Your robot will run, but less precisely, until you fix this and test again.",
    False: "Fix this before you continue. If you save now, your robot won't move the way the model expects.",
}

LET_GO_CAUSES = frozenset({"not_followed", "too_far", "gripper_write"})

PARK_CAUSES = frozenset({"blocked", "alarm"})


@dataclasses.dataclass(frozen=True)
class Reading:
    """One read of one arm, as the driver reported it.

    Attributes:
    - t_s: Monotonic time of the read, seconds.
    - joints: `[width]` joint values as read, one per joint of dream's model (radians when the driver is right).
    - gripper: The gripper value as read (0 closed to 1 open when the driver is right).
    """

    t_s: float
    joints: npt.NDArray[np.float64]
    gripper: float


def reading_of(state: types.ArmState, t_s: float) -> Reading:
    """A #Reading of what an arm's `read()` returned.

    @param state The arm's state.
    @param t_s When it was read, monotonic seconds.
    @return The reading.
    """
    return Reading(t_s=t_s, joints=np.asarray(state.positions_rad, dtype=np.float64), gripper=float(state.gripper))


@dataclasses.dataclass(frozen=True)
class Finding:
    """What one check found.

    Attributes:
    - state: `waiting` (not decided yet), `pass` or `fail`.
    - message: What was seen, in plain words.
    - cause: The code of the likely cause in #CAUSES when it failed, else None.
    - joint: The joint the finding is about, 0-based, when there is one.
    """

    state: str
    message: str
    cause: str | None = None
    joint: int | None = None

    def as_json(self) -> dict[str, object]:
        """The finding as the page reads it, with the cause's words (#Cause; empty when it passed).

        @return `{"state", "message", "cause", "title", "impact", "fix", "safe", "go_on", "joint"}`: `safe` and
            `go_on` (#GO_ON) say whether continuing past it is safe, and are True and empty when it passed.
        """
        words = CAUSES[self.cause] if self.cause is not None else Cause("", "", "", safe=True)
        return {
            "state": self.state,
            "message": self.message,
            "cause": self.cause,
            "title": words.title,
            "impact": words.impact,
            "fix": words.fix,
            "safe": words.safe,
            "go_on": GO_ON[words.safe] if self.cause is not None else "",
            "joint": self.joint,
        }


def tolerance_for(body: str) -> Tolerance:
    """How closely a body's arm must follow.

    @param body The config's `body`.
    @return Its #Tolerance, or #CUSTOM_TOLERANCE for a robot not in #TOLERANCES.
    """
    return TOLERANCES.get(body, CUSTOM_TOLERANCE)


def wiggle_scores(baseline: dict[str, Reading], latest: dict[str, Reading]) -> dict[str, float]:
    """How much each device's joints changed since the baseline: the largest change of any joint.

    @param baseline Per device, its reading when the operator was asked to move one arm.
    @param latest Per device, its reading now.
    @return Per device in both, the largest absolute joint change, in the units the driver reports.
    """
    return {
        device: float(np.max(np.abs(latest[device].joints - start.joints)))
        for device, start in baseline.items()
        if device in latest
    }


def moved_device(scores: dict[str, float]) -> tuple[str | None, bool]:
    """Which device the operator moved, from wiggle_scores().

    @param scores Per device, how much it changed.
    @return `(device, several)`: the one device past #WIGGLE_MIN_RAD, or None; `several` is True when more than one
        is past it (the operator moved two arms, or one arm bumped the other), and then the device is None.
    """
    moved = [device for device, score in scores.items() if score >= WIGGLE_MIN_RAD]
    if len(moved) > 1:
        return None, True
    return (moved[0] if moved else None), False


def judge_nudge(
    joint: int, sent_rad: float, path: list[Reading], move_s: float, *, lifts: bool, followed: bool
) -> Finding:
    """The wizard moved one joint by `sent_rad` under power: did that joint follow, the same way, by a share of it?

    Another joint that took the move (#NUDGE_SWAP_FRACTION of it, and more than the joint sent) is the driver writing
    joints in another order than it reads them (`write_order`); a neighbour drifting less while the joint sent falls
    short is that joint falling short, not the order. A joint that falls short is named by what else the readings
    say: nothing moved and no joint of this arm followed before it (`no_power`), it was still creeping on as it
    settled (`slow`), it lifts the arm (`weight`), else `joint_stuck`. A joint that settled right but swung past
    #NUDGE_MAX_FRACTION of the move on the way fails as `overshoot`.

    @param joint The joint sent, 0-based.
    @param sent_rad The move sent, radians, signed.
    @param path Every reading from before the move until it settled: the first is before, the last after.
    @param move_s Seconds the move took from the first reading; the readings after that are its settle.
    @param lifts Whether this joint lifts the arm (`robots.setup_model.ArmModel.lifting()`).
    @param followed Whether a joint of this arm nudged before followed: the arm has power and takes commands.
    @return The #Finding.
    """
    before, after = path[0].joints, path[-1].joints
    delta = after - before
    moved = float(delta[joint])
    others = np.abs(delta)
    others[joint] = 0.0
    stray = int(np.argmax(others))
    strayed = float(delta[stray])
    said = f"sent joint {joint + 1} {math.degrees(sent_rad):+.1f} degrees"
    took = abs(strayed) >= NUDGE_SWAP_FRACTION * abs(sent_rad) and abs(strayed) > abs(moved)
    if took:
        return Finding(
            "fail",
            f"{said}; joint {stray + 1} moved {math.degrees(strayed):+.1f} degrees instead",
            "write_order",
            joint,
        )
    if abs(moved) < NUDGE_MIN_FRACTION * abs(sent_rad):
        nothing = max(abs(moved), abs(strayed)) < NUDGE_STILL_FRACTION * abs(sent_rad)
        if nothing:
            words = "nothing moved"
        elif abs(moved) < NUDGE_STILL_FRACTION * abs(sent_rad):
            words = "it did not move"
        else:
            words = "it moved under a fifth of that"
        said = f"{said}; {words} ({math.degrees(moved):+.1f})"
        if abs(strayed) >= NUDGE_STILL_FRACTION * abs(sent_rad):
            said = f"{said}, while joint {stray + 1} drifted {math.degrees(strayed):+.1f}"
        settle = [reading.joints[joint] for reading in path if reading.t_s >= path[0].t_s + move_s]
        crept = float(settle[-1] - settle[0]) * math.copysign(1.0, sent_rad)
        if nothing and not followed:
            return Finding("fail", said, "no_power", joint)
        if crept >= NUDGE_STILL_FRACTION * abs(sent_rad):
            return Finding("fail", f"{said}, still creeping on as it settled", "slow", joint)
        return Finding("fail", said, "weight" if lifts else "joint_stuck", joint)
    if moved * sent_rad < 0:
        return Finding("fail", f"{said}; it moved {math.degrees(moved):+.1f}", "write_flipped", joint)
    if moved / sent_rad > NUDGE_MAX_FRACTION:
        return Finding("fail", f"{said}; it moved {math.degrees(moved):+.1f}", "write_size", joint)
    peak = max(float(reading.joints[joint] - before[joint]) / sent_rad for reading in path)
    if peak > NUDGE_MAX_FRACTION:
        swung = f"it swung to {math.degrees(peak * sent_rad):+.1f}, then settled at {math.degrees(moved):+.1f}"
        return Finding("fail", f"{said}; {swung}", "overshoot", joint)
    return Finding("pass", f"{said}; it moved {math.degrees(moved):+.1f}", joint=joint)


def judge_gripper_nudge(sent: float, before: float, after: float) -> Finding:
    """The wizard opened or closed the gripper a little under power: did it go that way, by about as much?

    @param sent The change sent, on the 0 to 1 scale, signed (positive opens).
    @param before The reading before.
    @param after The reading once it settled.
    @return The #Finding.
    """
    moved = after - before
    said = f"sent the gripper {sent:+.2f}; it moved {moved:+.2f}"
    if abs(moved) < GRIPPER_NUDGE_MIN_FRACTION * abs(sent):
        return Finding("fail", said, "gripper_still")
    if moved * sent < 0:
        return Finding("fail", said, "gripper_write")
    if abs(moved - sent) > 0.5 * abs(sent):
        return Finding("fail", said, "gripper_write_size")
    return Finding("pass", said)


def judge_reach(
    move: str,
    turning: bool,
    sent: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    reached: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    tolerance: Tolerance,
    short_rad: npt.NDArray[np.float64],
    lifting: frozenset[int],
    moved_rad: npt.NDArray[np.float64] | None = None,
    nudged: npt.NDArray[np.float64] | None = None,
) -> Finding:
    """How the arm's tool moved, by forward kinematics of its readings before and after, against the move sent.

    Judged like a nudge: along the move sent (its shift, or its turn's axis) the tool must have gone the same way by
    #NUDGE_MIN_FRACTION to #NUDGE_MAX_FRACTION of it, or, for a move smaller than the body's #Tolerance along that
    line, have ended within the tolerance of where it was sent; and what it did besides (off that line or axis, and
    the turn of a shift or the shift of a turn) must stay within the body's #Tolerance (a turn's shift within
    #TURN_POS_FACTOR of it). A miss is `weight` when the joint farthest off lifts the arm, fell behind its move and
    fell #WEIGHT_GAP_FRACTION more of its nudge short than any joint that lifts nothing, else `missed`: an arm that
    went past, drifted where it was not sent, or whose unloaded joints fell as short, is not
    sagging under its weight.

    @param move The move's name, for the message (`"+x 3 cm"`).
    @param turning True for a turn move, False for a shift.
    @param sent `([4, 4], [4, 4])` tool poses of the joints sent before the move and at its end.
    @param reached `([4, 4], [4, 4])` tool poses of the joints read before the move and at its end.
    @param tolerance The body's #Tolerance (`pos_m`, `ang_rad`).
    @param short_rad `[width]`, per joint, how far its move read differs from its move sent, radians.
    @param lifting The joints that lift the arm (`robots.setup_model.ArmModel.lifting()`).
    @param moved_rad `[6]` per joint, the move sent, radians; None judges a lifting joint's miss as `weight` as before.
    @param nudged `[width]` per joint, the share of its nudge it moved, NaN when not nudged; None (or NaN for the joint
        farthest off) leaves the nudges out.
    @return The #Finding.
    """
    shift_sent = sent[1][:3, 3] - sent[0][:3, 3]
    shift_reached = reached[1][:3, 3] - reached[0][:3, 3]
    turn_sent = setup_model.rotation_vector(sent[1][:3, :3] @ sent[0][:3, :3].T)
    turn_reached = setup_model.rotation_vector(reached[1][:3, :3] @ reached[0][:3, :3].T)
    line_sent, line_reached = (turn_sent, turn_reached) if turning else (shift_sent, shift_reached)
    size = float(np.linalg.norm(line_sent))
    along = float(line_reached @ line_sent) / size
    share = along / size
    beside = float(np.linalg.norm(line_reached - along * line_sent / size))
    if turning:
        off_m, off_rad = float(np.linalg.norm(shift_reached - shift_sent)), beside
        made = f"turned {math.degrees(along):.1f} of {math.degrees(size):.1f} degrees"
    else:
        off_m, off_rad = beside, float(np.linalg.norm(turn_reached - turn_sent))
        made = f"moved {along * 100:.1f} of {size * 100:.1f} cm"
    said = (
        f"{move}: {made} ({share:.0%}), {off_m * 100:.1f} cm and {math.degrees(off_rad):.1f} degrees off the move sent"
    )
    pos_m = tolerance.pos_m * (TURN_POS_FACTOR if turning else 1.0)
    # A move smaller than the tolerance along its own line (a shift the IK halved to a few millimetres from a home
    # with little room, on an SO-101) sits inside the motors' dead band: how much of it the hand
    # made says nothing, so it is judged by where the hand ended, within that tolerance of where it was sent.
    whole = tolerance.ang_rad if turning else tolerance.pos_m
    if size < whole:
        on_line = float(np.linalg.norm(line_reached - line_sent)) <= whole
    else:
        on_line = NUDGE_MIN_FRACTION <= share <= NUDGE_MAX_FRACTION
    if on_line and off_m <= pos_m and off_rad <= tolerance.ang_rad:
        return Finding("pass", said)
    worst = int(np.argmax(np.abs(short_rad)))
    said = f"{said}; joint {worst + 1} was {math.degrees(abs(float(short_rad[worst]))):.1f} degrees off"
    behind = moved_rad is None or float(short_rad[worst]) * float(moved_rad[worst]) < 0
    if nudged is not None and math.isfinite(nudged[worst]):
        unloaded = [1.0 - float(share) for j, share in enumerate(nudged) if j not in lifting and math.isfinite(share)]
        behind = behind and 1.0 - float(nudged[worst]) > max(unloaded, default=0.0) + WEIGHT_GAP_FRACTION
    return Finding("fail", said, "weight" if worst in lifting and behind else "missed", worst)


def tracking_advice(joint: int, lifting_rad: list[float], lowering_rad: list[float]) -> str | None:
    """Advice on how one joint that lifts the arm tracks, from how far short of where it was sent it held in the sweep.

    Friction holds a joint behind its move whichever way it went; the arm's weight holds it low, short when it lifts
    and past when it lowers. So falling shorter lifting than lowering is the arm's weight; the reverse, gravity
    compensation pushing up too hard; about the same both ways, a soft or sticky joint.

    @param joint The joint, 0-based.
    @param lifting_rad How far short of its target it held in each move that turned it to raise the tool, radians
        (negative past it).
    @param lowering_rad The same for each move that turned it to lower the tool.
    @return One line of advice, or None when it moved only one way or fell short by under #SWEEP_SHORT_RAD both ways.
    """
    if not lifting_rad or not lowering_rad:
        return None
    up, down = float(np.mean(lifting_rad)), float(np.mean(lowering_rad))
    if max(up, down) < SWEEP_SHORT_RAD:
        return None
    said = f"joint {joint + 1} fell {math.degrees(up):.1f} degrees short lifting and {math.degrees(down):.1f} lowering"
    if up - down >= GRAVITY_GAP_RAD:
        return (
            f"{said}: the arm sags under its weight. Add or raise gravity compensation in your driver, or stiffen the "
            "lifting joints."
        )
    if down - up >= GRAVITY_GAP_RAD:
        return (
            f"{said}: your gravity compensation pushes up too hard. Lower it, or check the end-effector mass it uses."
        )
    return f"{said}: the same both ways, so the joint is soft or sticky, not heavy. Raise its stiffness."
