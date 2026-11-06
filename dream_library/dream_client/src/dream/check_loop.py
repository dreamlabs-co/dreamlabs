"""`python -m dream.check_loop`: a stand-in for the model that checks your control loop, with no model in it.

When a robot does the wrong thing, either the model or the loop that runs it is at fault. This command takes the model
out. It serves the model's address on this computer; point your loop at it with one setting,
`DREAM_GATEWAY_URL=http://127.0.0.1:<port>` (in the environment or `~/.dreamlabs/credentials`), and run the loop as
you would with the model. Every get_action() is answered as the model answers it, a chunk of 48 rows, under an
`action.<part>` for each `state.<part>` the call carries (and `q_r_chunk`, `q_l_chunk` for a YAM's keys), but the rows
are a scripted motion from wherever the robot stands (#Plan). The loop's first call says which arm it drives
(#dream.check_arm.identify): the client's `ik` config key sends each hand's pose and, to this address only, names its
arm (an arm of your own by its arm file), and a YAM's parts name a YAM pair.

1. hold: the reported pose for one chunk; nothing should move.
2. every joint of every arm and every gripper at once, 240 rows: each hand's tool point traces a circle 8 cm across,
   twice, the arms opposite ways round, and each gripper closes and opens once a lap. Each joint the circle hardly
   moves (the wrist's, always) makes one 10-degree out-and-back on top, one after another, while the arm's first three
   joints keep the tool point on its circle (#dream.check_loop.Plan._circle). Each hand's pose in the answer is its
   arm's at the joints answered, so a loop that runs the model's tool poses (the client's `ik` head `eef`, a station's
   `dreamcli robot run`) moves the arm as one that runs its joints. On an arm the check does not know (a model trained
   on YAM arms run on another arm, which is sent zeros for the joints), the hands' poses circle, the joints hold, and
   the checks read the tool points. Every check below is judged from this one motion.
3. only when a check finds a PROBLEM: each joint, tool point axis or gripper it named, alone, one after another
   (#dream.check_loop.Plan.extend). A joint makes one smooth cycle, +10 degrees at 1 s and -10 at 2 s, 3 s, at rest at
   both ends (within 12 degrees of a limit, 10 degrees one way only, away from it); a tool point axis the same, 4 cm;
   a gripper opens, closes and opens, 1 s a stroke (0 closed, 1 open). The lines then judge the whole run, so they
   name the part at fault.

When the client's IK drives a known arm (its `ik` head `eef`, which it names to this address), the IK picks the joints
that reach each hand pose, and they need not be the plan's: the checks then read each hand whole, its tool point's move
and its turn. A hand that starts with a joint at its limit may settle in the hold where the IK keeps that joint, just
inside it. When a hand is at fault, each joint of its arm moves alone, and the lines read the hand there too.

The rows are 1/30 s apart, as the model's; row 0 of every chunk is the row due when the call came, at `--rate` rows a
second, the rate the loop plays rows at (the client's `action_hz`, 20 by default). A first call that sends no hand pose
and names no YAM is refused, and nothing moves: the check moves only an arm it knows the shape of, or its hands.

Each call carries the robot's state back, and at the end the command prints a PASS or PROBLEM line per check (#judge):

- hold: nothing moved while the robot was told to hold (unit mix-ups, offsets, an inverted gripper show here first);
- right part moved: each joint's command moved that joint and no other (parts in another order, swapped arms);
- direction and size: each part moved the way and as far as told (signs, degrees for radians, an inverted gripper);
- lag and tracking: how far behind its targets each part follows (a loop that delays, queues or drops commands, or
  joints that follow their commands slowly);
- play rate: rows played a second against `--rate` (a loop that plays the plan slower or faster than it means to);
- circle: each hand's tool point, read at the calls and fit with an ellipse, went at least 6 cm across its 8 cm circle
  both ways (#CIRCLE_PASS), and at least 0.7 as wide as long (#CIRCLE_ROUND), so a run whose hands hardly moved, or
  traced a flat band, never passes;
- camera rate: the frames new to each call's camera windows a second against the 5 the model reads (#camera_rate).

The loop only reports the state when it calls, so every check is a fit over all the calls of the run, not a trace.

The arms move: the command says so and waits for you to type "the workspace is clear" before it answers any call;
keep a hand on the e-stop. A loop that reads positions in one unit and sends commands in another moves on the very
first chunk, before any check can stop it; only the robot's own limits and the e-stop stop that. When anything moves
more than 30 degrees (10 cm) while the robot is told to hold, the stand-in stops answering, and every later call is
refused.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import http.server
import itertools
import json
import math
import pathlib
import sys
import threading
import time
import urllib.parse
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from dream import _window, check_arm, client_ik, input_check
from dream.debug_view import DebugView
from dream.errors import DreamConfigError

__all__ = ["CHECKS", "Channel", "Plan", "StandIn", "camera_rate", "confirm", "judge", "main", "watch"]

## Rows a second in the plan, as the model's rows: 48 rows are 1.6 s of motion at the recorded speed.
PLAN_HZ = 30
## Rows per chunk, the model's action horizon.
HORIZON = 48
## The port the stand-in serves on by default.
DEFAULT_PORT = 8642
## Rows a second the loop plays, by default: the client's `action_hz` default and our own loop's.
DEFAULT_RATE = 20.0
## The swing of each joint alone, radians (a tool point axis's, metres), and its rows: one cycle of 3 s, then 0.5 s at
## rest.
SWING, SWING_TOOL = math.radians(10), 0.04
SWING_ROWS, REST_ROWS = 90, 15
## The shape of one swing, sin(2 pi x) - sin(4 pi x) / 2: at rest at both ends, its peaks +-1.299 at x = 1/3 and 2/3.
SWING_PEAK = 1.299038
## A joint this close to a limit swings one way only, away from it.
LIMIT_MARGIN = math.radians(12)
## Every joint and gripper at once (#dream.check_loop.Plan._circle), rows: each tool point's circle, its radius
## (metres) and laps, in the upright plane this far round from straight out towards the arm's side, speeding up and
## slowing down over FADE_ROWS. A joint the circle moves less than CIRCLE_MIN makes an out-and-back of SWING over
## BUMP_ROWS on top; the circle's joints move at most CIRCLE_MAX (a smaller circle when they would move more).
TOGETHER_ROWS, FADE_ROWS, BUMP_ROWS = 240, 15, 90
CIRCLE_R, CIRCLE_LAPS, CIRCLE_TILT = 0.04, 2, math.radians(45)
CIRCLE_MIN, CIRCLE_MAX = math.radians(5), math.radians(20)
## A hand whose tool point traces less than this across (metres) fails the circle check: the check could not move it
## round its circle, or the loop did not play it.
CIRCLE_PASS = 0.06
## A hand whose traced path is narrower than this share of its length fails the circle check: a flat band, not a
## circle (an arm that reached its commands one way only).
CIRCLE_ROUND = 0.7
## The fewest calls in the circle the circle check fits a path to.
CIRCLE_CALLS = 5
## A tool point axis whose calls in the circle are further than this share of its range from its commands at its
## own lag and gain moved unlike them (#_misses): its gain and lag would mean nothing.
CIRCLE_FIT = 0.15
## The most a part's own lag is fit to (seconds, #_lag_fit): a part fit at it is at least this late.
OWN_LAG_MAX = 1.0
## What the circle check says may cause a flat band. From the hands alone it cannot tell a lag from joints that move
## the wrong amounts: either one turns when an axis gets there.
CIRCLE_CAUSE = (
    "the arm did not follow those commands (a joint at its limit, a joint that sags under its weight or tracks "
    "slowly, as with low position gains or friction, or an IK that cannot reach them)"
)
## How near the circle takes a joint to a limit it stands clear of.
LIMIT_GAP = math.radians(2)
## The angles round the circle a plane is tried at (#_fit_planes).
ROUND = np.linspace(0.0, 2 * math.pi, 49)
## The gripper alone: to open, close and open again, rows each; then it stays open.
GRIP_STROKES = (15, 30, 30, 15)
## While the robot is told to hold, a joint or a hand's turn further than this (radians), a tool point further than
## this (metres) or a gripper further than this from where it stood is a PROBLEM; past the stop limits the stand-in
## stops answering.
HOLD_JOINT, HOLD_TOOL, HOLD_GRIP = math.radians(1), 0.005, 0.05
STOP_JOINT, STOP_TOOL = math.radians(30), 0.10
## The checks' limits: a moving part is the one that moved most, and another moving more than this is a PROBLEM
## (radians, metres); a gain outside this range; a lag over this many seconds; a tracking error over this (radians,
## metres); a play rate this far from `--rate`, as a share of it.
OTHER_JOINT, OTHER_TOOL = math.radians(2), 0.01
GAIN_RANGE = (0.8, 1.2)
MAX_LAG_S = 0.25
## The longest lag the robot as a whole may have in the fit of the play rate (#_fit), seconds; and the error, in
## degrees of a joint or cm of a tool point, within which two fits are as good as each other.
MAX_FIT_LAG_S, FIT_SLACK = 0.5, 0.3
MAX_TRACK_JOINT, MAX_TRACK_TOOL = math.radians(1.0), 0.005
RATE_SHARE = 0.05
## A camera history this far from the frames a second the model reads (#dream._window.OBS_FPS), as a share of it.
FRAME_SHARE = 0.1
## How to fill each camera's history at the rate the client keeps it, as a PROBLEM or a reminder says.
FRAME_HINT = "hand set_observation() every camera tick, with its capture time in seconds (time.monotonic())"
## The checks, in the order their lines print.
CHECKS = ("hold", "right part moved", "direction and size", "lag and tracking", "play rate", "circle")
## The refusal every call gets once the stand-in has stopped or finished: never retryable, so a loop stops.
DONE_CODE = "DA-5000"
## The command, as its lines name it.
COMMAND = "python -m dream.check_loop"


@dataclasses.dataclass(frozen=True)
class Segment:
    """One step of the plan: rows `[start, start + rows)` move `parts` (columns `cols`), then `rest` rows hold.

    Attributes:
    - kind: `hold`, `together` (every joint and gripper at once), `joint` (one joint's swing), `axis` (one tool point
      axis's swing) or `grip`.
    - side: The arm it moves, the parts' common prefix (`left`, `right`, empty for a one-armed robot's `arm`).
    - parts: The state parts it moves.
    - cols: For `joint` and `axis`, the one column of `parts[0]` it swings; else empty.
    - start, rows, rest: Its first row, its moving rows, the rows at rest after them.
    """

    kind: str
    side: str
    parts: tuple[str, ...]
    cols: tuple[int, ...]
    start: int
    rows: int
    rest: int

    @property
    def end(self) -> int:
        """The row after its rest."""
        return self.start + self.rows + self.rest

    def say(self) -> str:
        """What the robot does in it, in words.

        @return The words, as the check prints them.
        """
        name = " and ".join(self.parts)
        find = "moves alone, to find the part at fault:"
        if self.kind == "joint":
            return f"{name} joint {self.cols[0] + 1} {find} one {math.degrees(SWING):.0f}-degree swing out and back"
        if self.kind == "axis":
            return f"{name} {'xyz'[self.cols[0]]} {find} {100 * SWING_TOOL:.0f} cm out and back"
        if self.kind == "grip":
            return f"{name} {find} open, close, open" if len(self.parts) == 1 else f"{name}: open, close, open"
        return {
            "hold": "holding still: nothing should move",
            "together": f"each hand draws an {200 * CIRCLE_R:.0f} cm circle twice while every joint turns; each "
            "gripper closes and opens",
        }[self.kind]


class Plan:
    """The scripted motion, as absolute targets per part for any row, from the robot's state at the first call.

    Parts are read from the state's keys: a part ending `_eef_xyz` is a tool point, `_eef_rot6` a tool rotation, one
    ending `gripper` or `grip` a gripper (one number, 0 closed to 1 open), and every other a group of joints. The
    prefix before `_arm`, `_gripper`, `_grip` or `_eef_` is the arm's side.

    Attributes:
    - start: Per part, the robot's reading at the first call.
    - settled: Per hand part on an arm the client's IK drives (`hands`), where the IK may hold it instead: at its
      joints moved inside the limits the IK keeps them in (`check_arm.Arm.ik_limits`), a hand that starts at a limit
      settles there; empty on any other arm.
    - arms: Per side, the arm the check knows it as (#dream.check_arm.Arm); None for an arm it does not know, whose
      hands' poses move while its joints hold.
    - hands: Whether the checks read each hand's pose, not the joints: the client's IK drives a known arm, picking
      joints of its own for each hand pose.
    - segments: The steps, in order; #extend adds to them.
    - rows: The row after the last step.
    """

    def __init__(
        self,
        state: Mapping[str, Sequence[float] | np.ndarray],
        *,
        arms: Mapping[str, check_arm.Arm] | None,
        hands: bool,
    ):
        """Builds the hold and the together-motion for this robot.

        @param state Per part (without `state.`), the robot's reading at the first call.
        @param arms Per side, its arm (#dream.check_arm.identify), or None for an arm the check does not know.
        @param hands The client's IK drives the known arms from each hand's pose (`<side>_eef_xyz`, `<side>_eef_rot6`).
        @throws ValueError An arm the check does not know and no hand pose to move.
        """
        self.start = {part: np.asarray(value, dtype=float).reshape(-1) for part, value in state.items()}
        self.settled: dict[str, np.ndarray] = {}
        for side, arm in (arms or {}).items() if hands else ():
            q = self.start[f"{side}_arm"]  # (joints,)
            inside = np.clip(q, arm.ik_limits[:, 0], arm.ik_limits[:, 1])  # (joints,)
            moved = arm.hand(inside[None])[0] - arm.hand(q[None])[0]  # (9,): the tool point, then rot6
            self.settled[f"{side}_eef_xyz"] = self.start[f"{side}_eef_xyz"] + moved[:3]
            self.settled[f"{side}_eef_rot6"] = self.start[f"{side}_eef_rot6"] + moved[3:]
        self.arms = arms
        self.hands = hands
        ## Per joint part of a known arm, its joints' `(lower, upper)` limits, radians.
        self.limits = {f"{side}_arm": arm.limits for side, arm in (arms or {}).items()}
        self.segments: list[Segment] = []
        self._table: dict[str, np.ndarray] | None = None
        row = self._add("hold", "", (), (), HORIZON, 0, 0)
        moves = ("joint", "grip") if arms is not None else ("xyz", "grip")
        parts = tuple(p for p in self.start if _kind(p) in moves)
        if arms is None and not any(_kind(p) == "xyz" for p in parts):
            raise ValueError(
                "the first call names no arm the check knows the shape of: send each hand's pose with the client's "
                "`ik` config key (its `arm`, the arm you drive), or name the robot `yam` on a YAM"
            )
        self.rows = self._add("together", "", parts, (), TOGETHER_ROWS, REST_ROWS, row)
        self.circle = self._circle(parts)
        ## Per side, how far across its hand's tool point goes round in the plan (metres), for the circle check.
        self.across: dict[str, float] = {}
        for part, move in self.circle.items():
            known = self._arm(part) if _kind(part) == "joint" else None
            if known is not None:
                self.across[_side(part)] = _across(known.hand(self.start[part] + move)[:, :3])
            elif _kind(part) == "xyz":
                self.across[_side(part)] = _across(move)
        # The swing's sign per joint, away from a limit it stands near; 0 swings both ways.
        self.away: dict[tuple[str, int], int] = {}
        for part, limits in self.limits.items():
            for col, (lower, upper) in enumerate(limits):
                near_low = self.start[part][col] - lower < LIMIT_MARGIN
                near_high = upper - self.start[part][col] < LIMIT_MARGIN
                self.away[(part, col)] = 1 if near_low and not near_high else -1 if near_high else 0

    def _add(
        self, kind: str, side: str, parts: tuple[str, ...], cols: tuple[int, ...], rows: int, rest: int, start: int
    ) -> int:
        """Appends one segment and returns the row after it."""
        self.segments.append(Segment(kind, side, parts, cols, start, rows, rest))
        self._table = None
        return start + rows + rest

    def extend(self, channels: Sequence[Channel], row: int) -> None:
        """Appends each channel's move alone, one after another from `row`.

        Each is a joint's or a tool point axis's swing, or a gripper's strokes.

        @param channels The joints, tool point axes and grippers to move alone.
        @param row The first row of the first move: the row due now, or later.
        """
        for ch in channels:
            if ch.kind in ("joint", "xyz"):
                kind = "joint" if ch.kind == "joint" else "axis"
                row = self._add(kind, _side(ch.part), (ch.part,), (ch.col,), SWING_ROWS, REST_ROWS, row)
            elif ch.kind == "grip":
                row = self._add("grip", _side(ch.part), (ch.part,), (), sum(GRIP_STROKES), REST_ROWS, row)
        self.rows = row

    def sides(self) -> list[str]:
        """The arms' sides, the left first.

        @return Each side's name (`left`, `right`, empty for a one-armed robot).
        """
        found = sorted({_side(p) for p in self.start if _kind(p) in ("joint", "grip", "xyz")})
        return sorted(found, key=lambda side: (side != "left", side != "right", side))

    def segment_at(self, row: float) -> Segment:
        """The segment a row falls in.

        @param row A row of the plan.
        @return Its segment; the last one past the end.
        """
        for segment in self.segments:
            if row < segment.end:
                return segment
        return self.segments[-1]

    def offsets(self, row: float) -> dict[str, np.ndarray]:
        """Per part, the target's offset from where the part stood (a gripper's target itself), at one row.

        A known arm's hand poses are left at 0 here: #table puts them where the arm's joints take them.

        @param row A row of the plan, fractions included.
        @return Per state part, its offset.
        """
        out = {part: np.zeros_like(value) for part, value in self.start.items()}
        for part, value in self.start.items():
            if _kind(part) == "grip":
                out[part] = value.copy()
        for segment in self.segments:
            u = row - segment.start
            if segment.kind == "grip" and u >= 0:
                for part in segment.parts:
                    out[part] = np.full_like(out[part], _grip_profile(u, float(self.start[part][0])))
            if u < 0 or u >= segment.rows:
                continue
            if segment.kind == "joint":
                part, col = segment.parts[0], segment.cols[0]
                out[part][col] = self._swing(u / segment.rows, self.away.get((part, col), 0))
            elif segment.kind == "axis":
                out[segment.parts[0]][segment.cols[0]] = SWING_TOOL / SWING * self._swing(u / segment.rows, 0)
            elif segment.kind == "together":
                i = int(u)
                for part in segment.parts:
                    table = self.circle[part]
                    out[part] = table[i] + (u - i) * (table[i + 1] - table[i])
        return out

    def table(self) -> dict[str, np.ndarray]:
        """Per part, its target at every row of the plan and a chunk past it.

        A hand on an arm the check knows is where its arm's joints are at that row (#dream.check_arm.Arm.hand), so
        the answer's joints and hand poses agree, whichever of them the loop runs.

        @return Per state part, `(plan rows + HORIZON, width)` absolute targets (a gripper's, the target itself).
        """
        if self._table is None:
            rows = [self.offsets(r) for r in range(self.rows + HORIZON)]
            out = {part: np.stack([r[part] for r in rows]) for part in self.start}  # per part (rows, width)
            for side, arm in (self.arms or {}).items():
                q = self.start[f"{side}_arm"]  # (joints,)
                hand = arm.hand(q + out[f"{side}_arm"]) - arm.hand(q[None])  # (rows, 9): xyz, then rot6
                for part, cols in ((f"{side}_eef_xyz", slice(0, 3)), (f"{side}_eef_rot6", slice(3, 9))):
                    if part in out:
                        out[part] = hand[:, cols]
            self._table = {p: o if _kind(p) == "grip" else self.start[p] + o for p, o in out.items()}
        return self._table

    def targets(self, row: int) -> dict[str, np.ndarray]:
        """Per part, the absolute target at one row (#table).

        @param row A row of the plan; past the end, the last.
        @return Per state part, its target.
        """
        return {part: rows[min(row, len(rows) - 1)] for part, rows in self.table().items()}

    def _swing(self, x: float, away: int) -> float:
        """One joint's swing at the share `x` of its rows: a full cycle, or a bump `away` from a limit."""
        if away:
            return away * SWING * (1 - math.cos(2 * math.pi * x)) / 2
        return SWING / SWING_PEAK * (math.sin(2 * math.pi * x) - math.sin(4 * math.pi * x) / 2)

    def _arm(self, part: str) -> check_arm.Arm | None:
        """The known arm whose joints a part is (`<side>_arm`); None for any other part, or an arm not known."""
        side = _side(part)
        return self.arms[side] if self.arms and part == f"{side}_arm" and side in self.arms else None

    def _circle(self, parts: tuple[str, ...]) -> dict[str, np.ndarray]:
        """The together-motion: per part, its offset (a gripper's target) at each of its rows and the row after.

        Each hand's tool point circles in the upright plane #CIRCLE_TILT round from straight out, through where it
        stands (#_ring); the left arm goes round the other way, its gripper half a lap apart, so no two arms' parts
        move alike. On a known arm the circle is solved to its first three joints with the arm's Jacobian there (to
        first order, so a near-circle). Each joint the circle moves less than #CIRCLE_MIN makes one out-and-back of
        #SWING on top, away from its nearer limit, the joints one after another; the first three joints take back
        what those moves do to the tool point. No joint comes within 2 degrees of a limit it stood clear of. An arm
        that cannot draw the circle so from where it stands (folded at its limits, as a Metal arm at home) draws it
        with every joint, in the plane it draws the largest circle in (#_circle_joints). On an arm the check does not
        know, the tool point itself circles.
        """
        n = TOGETHER_ROWS
        speed = np.array([min(_ease(r / FADE_ROWS), _ease((n - r) / FADE_ROWS)) for r in range(n + 1)])
        turn = _turn()  # (rows,) radians round the circle
        sides = self.sides()
        out, still, whole = {}, [], set()
        for part in parts:
            left = len(sides) > 1 and sides.index(_side(part)) == 0
            if _kind(part) == "grip":
                start = float(self.start[part][0])
                grip = start + speed * (0.55 - 0.4 * np.cos(turn + math.pi * left) - start)  # (rows,)
                out[part] = grip.reshape(n + 1, 1)  # (rows, 1)
                continue
            if _kind(part) == "xyz":
                out[part] = _ring(self.start[part], turn, left)  # (rows, 3 xyz)
                continue
            move = np.zeros((n + 1, self.start[part].size))  # (rows, joints)
            arm = self._arm(part)
            if arm is not None:
                inward, round_, every = _circle_joints(arm, self.start[part], self.limits[part])  # (joints,) each
                if every:
                    whole.add(part)
                way = -1.0 if left else 1.0
                move = CIRCLE_R * (np.outer(np.cos(turn) - 1, inward) + np.outer(way * np.sin(turn), round_))
            limits = self.limits.get(part)
            still += [
                (part, c)
                for c in range(move.shape[1])
                if np.max(np.abs(move[:, c])) < CIRCLE_MIN and (limits is None or limits[c][1] > limits[c][0])
            ]
            out[part] = move
        bumps = {part: np.zeros_like(move) for part, move in out.items()}
        gap = (n - 2 * FADE_ROWS - BUMP_ROWS) / max(len(still) - 1, 1)
        for k, (part, c) in enumerate(still):
            first = FADE_ROWS + k * gap
            x = np.clip((np.arange(n + 1) - first) / BUMP_ROWS, 0.0, 1.0)
            bumps[part][:, c] += self._away(part, c) * SWING * (1 - np.cos(2 * math.pi * x)) / 2
        for part, move in out.items():
            move = move + bumps[part]
            arm = self._arm(part)
            if arm is not None:
                q = self.start[part]
                # The tool point stays on its circle while a joint swings: the first three take back the swings of
                # the others; on a circle every joint draws, all the other joints take back each swing.
                jac = arm.jacobian(q)
                if part in whole:
                    for j in np.flatnonzero(np.any(bumps[part] != 0, axis=0)):
                        others = np.arange(q.size) != j
                        move[:, others] -= np.outer(bumps[part][:, j], _dls(jac[:, others]) @ jac[:, j])
                else:
                    move[:, :3] -= bumps[part][:, 3:] @ (_dls(jac[:, :3]) @ jac[:, 3:]).T
                low = np.minimum(q, self.limits[part][:, 0] + LIMIT_GAP)
                high = np.maximum(q, self.limits[part][:, 1] - LIMIT_GAP)
                move = np.clip(q + move, low, high) - q
            out[part] = move
        return out

    def _away(self, part: str, col: int) -> int:
        """The way a joint's out-and-back goes: towards the further of its limits, up without limits."""
        if part not in self.limits:
            return 1
        lower, upper = self.limits[part][col]
        return 1 if upper - self.start[part][col] >= self.start[part][col] - lower else -1


def _ring(tool: np.ndarray, turn: np.ndarray, left: bool) -> np.ndarray:
    """A tool point's circle of #CIRCLE_R from where it stands, `(rows, 3 xyz)` offsets: in the upright plane
    #CIRCLE_TILT round from straight out from the base towards the arm's side, at `turn` radians round each row; the
    left arm goes the other way round. The hand stands at the circle's bottom, so it never goes below where it
    stood: an arm at home is often folded onto the limits that lower it."""
    reach = np.array([tool[0], tool[1], 0.0]) / max(math.hypot(tool[0], tool[1]), 1e-6)
    across = np.cross([0.0, 0.0, 1.0], reach)
    e1, e2 = math.cos(CIRCLE_TILT) * reach + math.sin(CIRCLE_TILT) * across, np.array([0.0, 0.0, 1.0])
    a = -turn if left else turn
    return CIRCLE_R * (np.outer(np.sin(a), e1) + np.outer(1 - np.cos(a), e2))


def _circle_joints(arm: check_arm.Arm, q: np.ndarray, limits: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """The joint moves that take a hand round its circle from `q`: `(inward, round_)`, `(joints,)` each, the joints at
    #CIRCLE_R times `(cos t - 1) * inward + sin t * round_` at `t` radians round (to first order); and whether every
    joint draws it.

    First as #_ring puts the circle, solved to the first three joints. When that draws less than the whole circle
    inside #CIRCLE_MAX and the joints' limits (`(joints, 2)` radians), every joint draws it, in #_ring's plane if
    they can, else in the plane of #PLANES they draw the largest circle in: a joint near a limit only moves the
    circle's centre away from it, the joints clear of their limits take the hand round it. A circle no plane draws
    whole is drawn smaller, and the circle check fails it (#CIRCLE_PASS).
    """
    jac = arm.jacobian(q)  # (3 xyz, joints)
    low = np.minimum(q, limits[:, 0] + LIMIT_GAP) - q  # (joints,) the room below, <= 0
    high = np.maximum(q, limits[:, 1] - LIMIT_GAP) - q  # (joints,) the room above, >= 0
    tool = arm.hand(q[None])[0, :3]
    reach = np.array([tool[0], tool[1], 0.0]) / max(math.hypot(tool[0], tool[1]), 1e-6)
    out = math.cos(CIRCLE_TILT) * reach + math.sin(CIRCLE_TILT) * np.cross([0.0, 0.0, 1.0], reach)
    first = np.arange(q.size) < 3
    plane = np.array([[out, [0.0, 0.0, 1.0]]])  # (1 plane, 2 axes, 3 xyz)
    inward, round_, drawn = _fit_planes(jac, plane, first, first, low, high)
    if drawn[0] >= 0.95:
        return inward[0], round_[0], False
    every, clear = np.ones(q.size, dtype=bool), (high >= CIRCLE_MAX) & (-low >= CIRCLE_MAX)
    inward, round_, drawn = _fit_planes(jac, plane, every, clear, low, high)
    if drawn[0] >= 0.95:
        return inward[0], round_[0], True
    inward, round_, drawn = _fit_planes(jac, PLANES, every, clear, low, high)
    # Of the planes that draw nearly the largest circle, the one whose first three joints each turn #CIRCLE_MIN on
    # the circle, as many as can (a swing of one of them on top would throw the hand off its circle), then the one
    # nearest #_ring's.
    turned = np.abs(inward[:, :3]) * 2 * CIRCLE_R + np.abs(round_[:, :3]) * CIRCLE_R >= CIRCLE_MIN  # (planes, 3)
    near = drawn >= 0.95 * np.max(drawn)
    like = (np.abs(PLANES[:, 0] @ plane[0, 0]) + np.abs(PLANES[:, 1] @ plane[0, 1])) / 2  # (planes,) 0 to 1
    best = int(np.argmax(np.where(near, np.sum(turned, axis=1) + like, -1.0)))
    return inward[best], round_[best], True


def _fit_planes(
    jac: np.ndarray, planes: np.ndarray, centre: np.ndarray, turn: np.ndarray, low: np.ndarray, high: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per plane, #_circle_joints's two moves and how much of the circle they draw.

    @param jac The tool point's Jacobian, `(3 xyz, joints)`.
    @param planes `(planes, 2, 3 xyz)`: per plane, its axis towards where the hand stands from the centre (`e1`)
        and the one round (`e2`).
    @param centre, turn `(joints,)` masks: the joints that move the hand towards the centre, and round it.
    @param low, high `(joints,)` the room each joint has below (<= 0) and above (>= 0) where it stands.
    @return `inward`, `round_` `(planes, joints)` each, scaled inside #CIRCLE_MAX and the room; and `(planes,)` the
        share of the #CIRCLE_R circle they draw, to first order (0 to 1).
    """
    inward = np.zeros((len(planes), jac.shape[1]))  # (planes, joints)
    round_ = np.zeros_like(inward)
    inward[:, centre] = planes[:, 0] @ _dls(jac[:, centre]).T  # (planes, 3 xyz) @ (3 xyz, k joints)
    round_[:, turn] = planes[:, 1] @ _dls(jac[:, turn]).T
    cos, sin = (np.cos(ROUND) - 1)[None, :, None], np.sin(ROUND)[None, :, None]  # (1, angles, 1)
    path = CIRCLE_R * (cos * inward[:, None] + sin * round_[:, None])  # (planes, angles, joints)
    with np.errstate(divide="ignore", invalid="ignore"):
        room = np.where(path > 0, high / path, np.where(path < 0, low / path, np.inf))  # (planes, angles, joints)
    big = np.maximum(np.max(np.abs(path), axis=(1, 2)), 1e-9)  # (planes,)
    scale = np.minimum(1.0, np.minimum(CIRCLE_MAX / big, np.min(room, axis=(1, 2))))  # (planes,)
    to_centre, to_round = inward @ jac.T, round_ @ jac.T  # (planes, 3 xyz) the tool point's moves
    a, b = np.linalg.norm(to_centre, axis=1), np.linalg.norm(to_round, axis=1)  # (planes,)
    square = np.linalg.norm(np.cross(to_centre, to_round), axis=1) / np.maximum(a * b, 1e-12)  # (planes,) sine
    drawn = scale * np.minimum(np.minimum(a, b), 1.0) * square
    return inward * scale[:, None], round_ * scale[:, None], drawn


def _planes() -> np.ndarray:
    """#PLANES: `(planes, 2, 3 xyz)`, 200 axes towards the hand spread over the sphere, 12 axes round about each."""
    k = np.arange(200) + 0.5
    polar, azimuth = np.arccos(1 - 2 * k / 200), math.pi * (1 + 5**0.5) * k
    e1 = np.stack([np.cos(azimuth) * np.sin(polar), np.sin(azimuth) * np.sin(polar), np.cos(polar)], axis=1)
    side = np.cross(e1, [0.0, 0.0, 1.0])  # (200, 3 xyz)
    flat = np.linalg.norm(side, axis=1) < 0.1
    side[flat] = np.cross(e1[flat], [1.0, 0.0, 0.0])
    side /= np.linalg.norm(side, axis=1, keepdims=True)
    other = np.cross(e1, side)  # (200, 3 xyz)
    angle = np.linspace(0.0, math.pi, 12, endpoint=False)[None, :, None]  # (1, 12, 1)
    e2 = np.cos(angle) * side[:, None] + np.sin(angle) * other[:, None]  # (200, 12, 3 xyz)
    return np.stack([np.repeat(e1, 12, axis=0), e2.reshape(200 * 12, 3)], axis=1)  # (2400, 2, 3 xyz)


## The circles' planes #_circle_joints tries when the first three joints cannot draw the circle (#_planes).
PLANES = _planes()


def _turn() -> np.ndarray:
    """How far round its circle each hand is at each row of the together-motion, `(rows,)` radians: #CIRCLE_LAPS laps,
    speeding up and slowing down over #FADE_ROWS."""
    n = TOGETHER_ROWS
    speed = np.array([min(_ease(r / FADE_ROWS), _ease((n - r) / FADE_ROWS)) for r in range(n + 1)])
    turn = np.concatenate([[0.0], np.cumsum(speed[1:] + speed[:-1])])
    return np.asarray(2 * math.pi * CIRCLE_LAPS * turn / turn[n])


def _wave(points: np.ndarray, turn: np.ndarray) -> np.ndarray:
    """A hand's points `(n, 3 xyz)` fit with a centre plus `B @ (cos t, sin t)`, from how far round the circle the plan
    was at each `(n,)`: `B` as `(2 cos sin, 3 xyz)`, so a lag only turns it."""
    basis = np.stack([np.ones_like(turn), np.cos(turn), np.sin(turn)], axis=1)  # (n, 3)
    return np.asarray(np.linalg.lstsq(basis, points, rcond=None)[0][1:])  # (2 cos sin, 3 xyz)


def _sizes(wave: np.ndarray) -> tuple[float, float, np.ndarray]:
    """The ellipse a hand's #_wave `(2 cos sin, 3 xyz)` traces: its length and width (metres) and the width's direction
    `(3 xyz,)`, twice the wave's singular values. A band (a hand that reached its commands one way only, or one axis
    late) is long and thin."""
    u, s, _ = np.linalg.svd(wave.T)  # u (3 xyz, 2), s (2,) the semi-axes, longest first
    return 2 * float(s[0]), 2 * float(s[1]), u[:, 1]


def _misses(path: np.ndarray, points: np.ndarray, row_prev: np.ndarray, gap: np.ndarray, rate: float) -> list[str]:
    """How a hand missed its circle, in words: each axis the circle moves #CIRCLE_R / 2 or more, its points at the
    calls `(calls, 3 xyz)` fit to its planned path `(rows, 3 xyz)` with a lag and gain of its own (#_lag_fit), so
    calls far apart still read the path between them. Each call's point was read `gap` `(calls,)` seconds after the
    call before it, whose chunk started at `row_prev` `(calls,)` of the path, the loop playing `rate` rows a second.

    An axis left further than #CIRCLE_FIT from its commands, or fit at #OWN_LAG_MAX, moved unlike them; else its gain
    outside #GAIN_RANGE; and the axis #MAX_LAG_S or more out of step with the others, how far. Empty for none."""
    at = np.arange(len(path))

    def planned(k: int) -> Callable[[float], np.ndarray]:
        return lambda lag: np.interp(row_prev + (gap - lag) * rate, at, path[:, k])

    span = {k: float(np.ptp(path[:, k])) for k in range(3) if np.ptp(path[:, k]) >= CIRCLE_R / 2}
    fits = {k: _lag_fit(points[:, k], planned(k)) for k in span}  # lag (s), error, gain
    unlike = [
        k for k, (lag, rms, gain) in fits.items() if rms > CIRCLE_FIT * span[k] or gain < 0 or lag >= OWN_LAG_MAX
    ]
    like = [k for k in fits if k not in unlike]
    step = {k: fits[k][0] - float(np.mean([fits[j][0] for j in like if j != k])) for k in like if len(like) > 1}
    odd = max(step, key=lambda k: abs(step[k]), default=None)
    out = []
    for k, (_, _, gain) in fits.items():
        said = ["moved unlike its commands"] if k in unlike else []
        if k in like and not GAIN_RANGE[0] <= gain <= GAIN_RANGE[1]:
            said.append(f"moved x{gain:.2f} of its commands")
        if k == odd and abs(step[k]) >= MAX_LAG_S:
            others = " and ".join("xyz"[j] for j in like if j != k)
            said.append(f"ran {abs(step[k]):.2f} s {'behind' if step[k] > 0 else 'ahead of'} its {others}")
        if said:
            out.append(f"{'xyz'[k]} {' and '.join(said)}")
    return out


def _across(points: np.ndarray) -> float:
    """How far across a set of points `(n, 3 xyz)` is: the largest distance between two of them (metres)."""
    return float(np.max(np.linalg.norm(points[:, None] - points[None], axis=2))) if len(points) > 1 else 0.0


def _dls(jac: np.ndarray) -> np.ndarray:
    """The damped inverse of a (3 xyz, k joints) Jacobian, (k, 3): bounded near a singular pose (1 cm of damping)."""
    return jac.T @ np.linalg.inv(jac @ jac.T + 1e-4 * np.eye(3))


def _ease(u: float) -> float:
    """0 to 1 over `u` in [0, 1], at rest at both ends."""
    return 0.0 if u <= 0 else 1.0 if u >= 1 else 0.5 - 0.5 * math.cos(math.pi * u)


def _grip_profile(u: float, start: float) -> float:
    """The gripper's target `u` rows into its segment: from where it stood to open, closed, open, then open."""
    a, b, c, _ = GRIP_STROKES
    if u < a:
        return start + (1 - start) * _ease(u / a)
    if u < a + b:
        return 1 - _ease((u - a) / b)
    if u < a + b + c:
        return _ease((u - a - b) / c)
    return 1.0


def _kind(part: str) -> str:
    """A state part's kind: `xyz`, `rot`, `grip` or `joint`."""
    if part.endswith("eef_xyz"):
        return "xyz"
    if part.endswith("eef_rot6"):
        return "rot"
    if part.endswith("gripper") or part.endswith("grip"):
        return "grip"
    return "joint"


def _side(part: str) -> str:
    """A part's side: its name before `_arm`, `_gripper`, `_grip` or `_eef_`; empty when it has none."""
    for suffix in ("_eef_xyz", "_eef_rot6", "_gripper", "_grip", "_arm"):
        if part.endswith(suffix):
            return part[: -len(suffix)]
    return "" if part in ("arm", "gripper", "grip", "eef_xyz", "eef_rot6") else part


@dataclasses.dataclass
class Call:
    """One get_action() the stand-in answered: when it came, the state it carried, the row its chunk starts at, and
    its camera windows (#_frames)."""

    t: float
    state: dict[str, np.ndarray]
    row: int
    window: tuple[int, ...]


class StandIn:
    """The model's address on this computer, answering with the plan and recording the state every call carries.

    Attributes:
    - calls: Every call answered with rows, in order.
    - plan: The #Plan, from the first call's state; None before it.
    - stopped: Why it stopped answering (something moved while it held), or None.
    - finished: True once a call came after the plan's last row.
    - first: The lines of the together-motion, judged once it played (#_judge); None before.
    - alone: The joints and grippers the together-motion found at fault, each then moved alone
      (#dream.check_loop.Plan.extend).
    - checkpoint, cameras: The model the first call named and the camera roles it sent (`video.<role>`).
    - poses: Whether the first call sent each hand's pose (`state.<side>_eef_xyz`, which the client adds with the
      `ik` config key), so the model check sends them too (#dream.check).
    - inputs: The first call's inputs, as #dream.input_check.inputs_id() names them; None before it.
    """

    def __init__(
        self,
        *,
        rate: float = DEFAULT_RATE,
        clock: Callable[[], float] = time.monotonic,
        view: DebugView | None = None,
    ) -> None:
        """Nothing is served until serve().

        @param rate Rows a second the loop plays.
        @param clock The clock calls are timed on.
        @param view The page of what the model sees (#dream.DebugView), shown every call answered; or None.
        """
        self.rate, self.clock, self.view = rate, clock, view
        self.calls: list[Call] = []
        self.t0 = 0.0
        self.plan: Plan | None = None
        self.stopped: str | None = None
        self.finished = False
        self.first: list[Line] | None = None
        self.alone: list[Channel] = []
        self.checkpoint: str | None = None
        self.cameras: list[str] = []
        self.poses = False
        self.inputs: str | None = None
        self.done = threading.Event()
        self._lock = threading.Lock()
        self._server: http.server.ThreadingHTTPServer | None = None

    def answer(self, body: Mapping[str, Any], arm: str | None, head: str | None) -> dict[str, Any]:
        """The reply to one `/v1/get_action` body: the chunk due now, or a refusal once stopped or finished.

        @param body The request: `session_id` and `observation`.
        @param arm The arm the call names (`dream.client_ik.ARM_HEADER`, unquoted), or None.
        @param head The `ik` head the call names (`dream.client_ik.HEAD_HEADER`), or None.
        @return The reply: `session_id`, the status envelope and `actions`.
        """
        with self._lock:
            now = self.clock()
            sent = body.get("session_id")
            session = sent if isinstance(sent, str) and sent else uuid.uuid4().hex
            observation = body.get("observation")
            state = _state(observation if isinstance(observation, dict) else {})
            if self.stopped is None and not self.finished:
                if self.plan is None:
                    # The model the loop names and the cameras it sends, for the model check (#dream.check).
                    named = body.get("checkpoint")
                    self.checkpoint = named if isinstance(named, str) and named else None
                    self.cameras = sorted(
                        k[len("video.") :]
                        for k in (observation if isinstance(observation, dict) else {})
                        if k.startswith("video.")
                    )
                    self.poses = any(f"{side}_eef_xyz" in state for side in ("right", "left"))
                    self.inputs = input_check.inputs_id(observation if isinstance(observation, dict) else {})
                    robot = observation.get(input_check.ROBOT_KEY) if isinstance(observation, dict) else None
                    arms = check_arm.identify(state, robot, arm)
                    # The client's IK drives a known arm from each hand's pose: the checks read the hands.
                    hands = head == "eef" and arms is not None and all(f"{side}_eef_rot6" in state for side in arms)
                    try:
                        self.plan = Plan(state, arms=arms, hands=hands)
                    except ValueError as error:
                        self.stopped = f"stopped: {error}"
                    self.t0 = now
                else:
                    self.stopped = self._moved_while_holding(state, now)
            if self.stopped is not None or self.finished:
                self.done.set()
                return _refusal(session, self.stopped or "has finished: its lines are in the terminal it runs in")
            assert self.plan is not None
            row = int(round((now - self.t0) * self.rate))
            self.calls.append(Call(now, state, row, _frames(observation if isinstance(observation, dict) else {})))
            if row >= self.plan.rows and self.first is None:
                # The together-motion has played: judge it, then move alone each part it found at fault.
                self.first, self.alone = _judge(self)
                if self.alone:
                    self.plan.extend(self.alone, row)
            if row >= self.plan.rows:
                self.finished = True
                self.done.set()
                return _refusal(session, "has finished: its lines are in the terminal it runs in")
            chunk = {part: rows[row : row + HORIZON] for part, rows in self.plan.table().items()}  # (48, width)
            actions: dict[str, Any] = {"per_key": {f"action.{part}": chunk[part].tolist() for part in state}}
            for side, key in (("right", "q_r"), ("left", "q_l")):
                if f"{side}_arm" in state and f"{side}_gripper" in state:
                    rows_ = np.concatenate([chunk[f"{side}_arm"], chunk[f"{side}_gripper"]], axis=1).tolist()
                    actions[f"{key}_chunk"], actions[key] = rows_, rows_[0]
            code = "DA-0001" if sent is None else "DA-0000"
            return {
                "session_id": session,
                "status": _envelope(code, "new" if sent is None else None, None),
                "actions": actions,
            }

    def _moved_while_holding(self, state: Mapping[str, np.ndarray], now: float) -> str | None:
        """Why to stop: a part that moved past the stop limits while the robot was told to hold; None when none."""
        assert self.plan is not None
        if (now - self.t0) * self.rate >= HORIZON + 0.5 * self.rate:  # the hold, plus half a second
            return None
        for part, value in state.items():
            start = self.plan.start.get(part)
            if start is None or start.shape != value.shape or _kind(part) not in ("joint", "xyz"):
                continue
            limit = STOP_JOINT if _kind(part) == "joint" else STOP_TOOL
            moved = float(np.max(np.abs(value - start)))
            if moved > limit:
                size = f"{math.degrees(moved):.0f} degrees" if _kind(part) == "joint" else f"{moved * 100:.0f} cm"
                return (
                    f"stopped: state.{part} moved {size} during the hold; your loop changes the commands on the "
                    "way (units, part order or offsets)"
                )
        return None

    def serve(self, port: int = DEFAULT_PORT, host: str = client_ik.CHECK_HOST) -> str:
        """Starts answering on a thread of its own.

        @param port The port; 0 picks a free one.
        @param host The address to listen on.
        @return The URL a loop's `DREAM_GATEWAY_URL` names.
        """
        stand_in = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802  # http.server's spelling
                length = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads(self.rfile.read(length) or b"{}")
                except ValueError:
                    body = {}
                named = self.headers.get(client_ik.ARM_HEADER)
                reply = (
                    stand_in.answer(
                        body,
                        None if named is None else urllib.parse.unquote(named),
                        self.headers.get(client_ik.HEAD_HEADER),
                    )
                    if self.path == "/v1/get_action"
                    else {"status": _envelope("DA-0000", None, None)}
                )
                answered = self.path == "/v1/get_action" and reply["actions"] is not None
                if stand_in.view is not None and answered and isinstance(body.get("observation"), dict):
                    stand_in.view.add(body, reply)  # the calls answered; the refusals that end the motion are not
                payload = json.dumps(reply).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self) -> None:  # noqa: N802  # http.server's spelling
                # The plan so far (its segments grow once the together-motion finds a part at fault), for a viewer.
                with stand_in._lock:
                    plan = stand_in.plan
                    reply = (
                        None
                        if plan is None
                        else {
                            "rate": stand_in.rate,
                            "start": {p: v.tolist() for p, v in plan.start.items()},
                            "segments": [dataclasses.asdict(s) for s in plan.segments],
                        }
                    )
                payload = json.dumps(reply).encode()
                self.send_response(200 if self.path == "/v1/check/plan" else 404)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002  # http.server's signature
                pass

        self._server = http.server.ThreadingHTTPServer((host, port), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return f"http://{host}:{self._server.server_address[1]}"

    def close(self) -> None:
        """Stops answering."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()

    def arms(self) -> dict[str, Any] | None:
        """The motion as the page of what the model sees draws the arms (#dream.DebugView.show_arms).

        @return #trace(), with `now` (seconds since the first call, on the stand-in's clock), `done` (it has finished
            or stopped), `tcp_m` (the YAM's tool point along its last link, `dream_ik.specs.TCP_OFFSET_M`, where the
            page puts each YAM gripper's dot) and `chains`: per side of a known arm, its joints from the base to the
            tool point (#dream.check_arm.Arm.chain, each `origin` a 4x4 row by row, `axis`, `col` and `kind`) and its
            `tool`, from which the page draws any arm but a YAM; None before the first call.
        """
        with self._lock:
            if self.plan is None:
                return None
            chains = {
                side: {
                    "links": [
                        {"origin": origin.reshape(16).tolist(), "axis": axis.tolist(), "col": col, "kind": kind}
                        for origin, axis, col, kind in arm.chain
                    ],
                    "tool": arm.tool.reshape(16).tolist(),
                }
                for side, arm in (self.plan.arms or {}).items()
            }
            return {
                **self.trace(),
                "now": self.clock() - self.t0,
                "done": self.done.is_set(),
                "tcp_m": check_arm.ik_specs().TCP_OFFSET_M,
                "chains": chains,
            }

    def read_as(self) -> str:
        """The arm the first call named, in words, as the check prints it once the motion starts.

        @return The words.
        """
        assert self.plan is not None
        if self.plan.arms is None:
            return "an arm the check does not know: each hand's pose circles and the joints hold"
        posed = any(_kind(part) == "xyz" for part in self.plan.start)
        arms = f"{self.arm} arms" if self.arm in client_ik.ARMS else f"the arms of {self.arm}"
        if self.plan.hands:
            return f"{arms}, through the client's IK: the lines read the hands"
        return arms if posed else f"{arms} (joints only, no hand pose)"

    @property
    def arm(self) -> str | None:
        """The arm the first call named: dream's name for it (`yam`, `metal`, ...), or its arm file's path.

        From #dream.check_arm.identify; None before the first call, and for an arm the check does not know.
        """
        arms = None if self.plan is None else self.plan.arms
        return None if not arms else next(iter(arms.values())).name

    def trace(self) -> dict[str, Any]:
        """The run as JSON: the plan's targets for every row and each call; main() adds the lines judge() prints.

        @return The run, as JSON values.
        """
        assert self.plan is not None
        return {
            "plan_hz": PLAN_HZ,
            "rate": self.rate,
            "horizon": HORIZON,
            "arm": self.arm,
            "hands": self.plan.hands,
            "segments": [dataclasses.asdict(s) for s in self.plan.segments],
            "start": {p: v.tolist() for p, v in self.plan.start.items()},
            "targets": {p: rows.tolist() for p, rows in self.plan.table().items()},
            "calls": [
                {"t": c.t - self.t0, "row": c.row, "state": {p: v.tolist() for p, v in c.state.items()}}
                for c in self.calls
            ],
            "stopped": self.stopped,
        }


def _state(observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Per part, the observation's state values that are numbers.

    Its keys are `state.<part>`; each part is named without the prefix.
    """
    out = {}
    for key, value in observation.items():
        if not key.startswith("state."):
            continue
        try:
            array = np.asarray(value, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            continue
        if array.size and np.all(np.isfinite(array)):
            out[key[len("state.") :]] = array
    return out


def _frames(observation: Mapping[str, Any]) -> tuple[int, ...]:
    """One hash per frame of the call's camera windows, oldest first, each over that frame on every camera.

    Empty when the call sent no frames (no `video.` key).
    """
    windows = [value for key, value in sorted(observation.items()) if key.startswith("video.")]
    if not windows or not all(isinstance(frames, list) for frames in windows):
        return ()
    return tuple(hash(tuple(frames)) for frames in zip(*windows, strict=False))


def camera_rate(stand_in: StandIn) -> Line:
    """How fast your loop filled each camera's history, against the rate the model reads it (#dream._window.OBS_FPS).

    The rate is the frames new to each call's windows a second between calls. A frame is new to a call when the call
    before did not send it. Two calls whose windows share a frame count the frames between them exactly; two that share
    none (further apart than a window spans) count at least a window. A call whose window repeats a frame (a still
    scene, a stale camera) is not counted.

    @param stand_in The stand-in, after a run.
    @return PASS within #FRAME_SHARE of the rate, PROBLEM past it, SKIP when too little of the run was counted
        exactly to say, with why: frames that repeat, or calls further apart than one call's frames span.
    """
    fps, calls = _window.OBS_FPS, [call for call in stand_in.calls if call.window]
    if len(calls) < 2:
        return Line("SKIP", "camera rate", "your loop sent no camera frames on two calls")
    pairs = []  # (seconds between two calls, frames new to the later, whether their windows share a frame)
    for earlier, later in itertools.pairwise(calls):
        if len(set(later.window)) == len(later.window):
            shared = set(later.window) & set(earlier.window)
            pairs.append((later.t - earlier.t, len(later.window) - len(shared), bool(shared)))
    seconds = sum(dt for dt, _, _ in pairs)
    least = sum(new for _, new, _ in pairs) / seconds if seconds > 0 else 0.0
    exact = sum(dt for dt, _, shared in pairs if shared)
    read = f"the {fps:g} the model reads"
    if least > fps * (1 + FRAME_SHARE):
        took = f"at least {least:.1f} frames a second per camera"
        return Line("PROBLEM", "camera rate", f"{took}, not {read}: {FRAME_HINT}")
    if exact > 0 and exact >= (calls[-1].t - calls[0].t) / 2:
        rate = sum(new for _, new, shared in pairs if shared) / exact
        took = f"{rate:.1f} frames a second per camera"
        if abs(rate / fps - 1) <= FRAME_SHARE:
            return Line("PASS", "camera rate", f"{took}, {read}")
        return Line("PROBLEM", "camera rate", f"{took}, not {read}: {FRAME_HINT}")
    if len(pairs) < (len(calls) - 1) / 2:
        why = f"the frames repeat (a still scene or a stale camera): {FRAME_HINT}"
        return Line("SKIP", "camera rate", f"not counted: {why}")
    gap = float(np.median([later.t - earlier.t for earlier, later in itertools.pairwise(calls)]))
    span = (_window.OBS_WINDOW - 1) / fps  # seconds of frames one call carries
    return Line(
        "SKIP", "camera rate", f"not counted: calls came {gap:.1f} s apart, more than the {span:g} s each carries"
    )


def watch(stand_in: StandIn) -> None:
    """Prints each step of the motion as it starts, and what the together-motion found, until the stand-in is done.

    @param stand_in The stand-in, serving.
    """
    said, told, named = None, False, False
    while not stand_in.done.wait(0.25):
        plan = stand_in.plan
        if plan is None:
            continue
        t = stand_in.clock() - stand_in.t0
        if not named:
            named = True
            print(f"{t:5.1f} s  your loop drives {stand_in.read_as()}", flush=True)
        if stand_in.first is not None and not told:
            told = True
            if stand_in.alone:
                found = ", ".join(line.check for line in stand_in.first if line.status == "PROBLEM")
                names = ", ".join(c.name for c in stand_in.alone)
                print(
                    f"{t:5.1f} s  {found}: PROBLEM; next, to find the part at fault, each alone: {names}", flush=True
                )
        segment = plan.segment_at(t * stand_in.rate)
        if segment is not said and t * stand_in.rate < plan.rows:
            said = segment
            print(f"{t:5.1f} s  {segment.say()}", flush=True)


def _envelope(code: str, reason: str | None, detail: str | None) -> dict[str, Any]:
    """A status envelope as DreamAPI writes one; never retryable."""
    names = {"DA-0000": "OK", "DA-0001": "OK_NEW_SESSION", DONE_CODE: "CHECK_LOOP"}
    return {
        "code": code,
        "name": names[code],
        "reason": reason,
        "retryable": False,
        "retry_after_s": None,
        "trace_id": uuid.uuid4().hex[:16],
        "detail": detail,
    }


def _refusal(session: str, why: str) -> dict[str, Any]:
    """The reply once the stand-in stopped or finished: a refusal that is never retryable, so the loop stops."""
    return {
        "session_id": session,
        "status": _envelope(DONE_CODE, "check_loop", f"the loop check {why}"),
        "actions": None,
    }


# ---------------------------------------------------------------------------------------------------------------------
# The checks: fits over every call of the run.


@dataclasses.dataclass(frozen=True)
class Line:
    """One check's verdict: `PASS` or `PROBLEM` (`SKIP` when the run did not reach it), its name and why."""

    status: str
    check: str
    detail: str

    def __str__(self) -> str:
        return f"{self.status:<8} {self.check:<19} {self.detail}"


def judge(stand_in: StandIn) -> list[Line]:
    """The checks over the calls the stand-in answered, each a fit over the whole run.

    Hold, right part moved, direction and size, lag and tracking, play rate.

    @param stand_in The stand-in, after a run.
    @return One #Line per check, in that order.
    """
    return _judge(stand_in)[0]


def _judge(stand_in: StandIn) -> tuple[list[Line], list[Channel]]:
    """#judge's lines, and the joints and grippers they found at fault, in the plan's order: none when the fault is
    the loop's as a whole (the play rate) or the run stopped."""
    plan, calls = stand_in.plan, stand_in.calls
    if plan is None:
        return [Line("SKIP", name, stand_in.stopped or "the loop never called") for name in CHECKS], []
    if stand_in.stopped is not None:
        return [Line("PROBLEM", CHECKS[0], stand_in.stopped)] + [
            Line("SKIP", name, "not run: the run stopped during the hold") for name in CHECKS[1:]
        ], []
    if len(calls) < 2:
        return [Line("SKIP", name, "the loop called once") for name in CHECKS], []
    t0 = stand_in.t0
    channels = _channels(plan)
    moving = [c for c in channels if c.kind != "grip"]
    # The state each call carried was measured while the robot played the chunk of the call before it.
    t = np.array([c.t - t0 for c in calls[1:]])
    t_prev = np.array([c.t - t0 for c in calls[:-1]])
    row_prev = np.array([c.row for c in calls[:-1]], dtype=float)
    # Per channel, `(calls,)` readings (a hand's `(calls, 3)`), its start and its targets each row likewise.
    measured = {ch: np.array([_value(plan, c.state, ch) for c in calls[1:]]) for ch in channels}
    start = {ch: _value(plan, calls[0].state, ch) for ch in channels}
    table = _target_table(plan, channels)

    def commanded(ch: Channel, lag: float, rate: float) -> np.ndarray:
        return _interp(row_prev + (t - lag - t_prev) * rate, table[ch])

    hold, held = _hold(calls, channels, start, stand_in)
    if hold.status == "PASS" and moving and all(np.max(np.ptp(measured[ch], axis=0)) < _unit(ch) for ch in moving):
        return [hold, Line("PROBLEM", CHECKS[1], _nothing_moved(plan))] + [
            Line("SKIP", name, "not judged: nothing moved") for name in CHECKS[2:]
        ], []
    # Rate and lag together, robust to a few parts that move wrongly (the median over parts of each one's error).
    rate, lag = _fit(
        lambda lag, rate: float(
            np.median([np.mean((measured[ch] - commanded(ch, lag, rate)) ** 2) / _unit(ch) ** 2 for ch in moving])
        ),
        stand_in.rate,
    )
    rows_at = row_prev + (t - lag - t_prev) * rate
    rest = _at_rest(plan, rows_at)
    base = {ch: np.median(measured[ch][rest], axis=0) if rest.any() else start[ch] for ch in channels}
    order, match, crossed = _order(plan, channels, measured, base, rows_at, lambda ch, lag_: commanded(ch, lag_, rate))
    own = _own_lags(moving, measured, match, lambda ch, lag_: commanded(ch, lag_, rate))
    direction, bent = _direction(
        plan, channels, measured, base, match, lambda ch: commanded(ch, own[ch][0] if ch in own else lag, rate)
    )
    late, slow = _lag(own, lag)
    share = rate / stand_in.rate - 1
    played = f"the loop played {rate:.1f} rows a second"
    pace = (
        Line("PASS", "play rate", f"{played} (--rate {stand_in.rate:g})")
        if abs(share) <= RATE_SHARE
        else Line(
            "PROBLEM",
            "play rate",
            f"{played}, {abs(share):.0%} {'slower' if share < 0 else 'faster'} than --rate "
            f"{stand_in.rate:g}: a loop that overruns its tick or sleeps the wrong time (if it means to play "
            f"{rate:.0f}, run with --rate {rate:.0f})",
        )
    )
    at_fault = held | crossed | bent | slow
    lines = [hold, order, direction, late, pace, _circle_line(plan, calls, rate)]
    return lines, [] if pace.status == "PROBLEM" else _alone(plan, channels, at_fault)


def _circle_line(plan: Plan, calls: Sequence[Call], rate: float) -> Line:
    """The circle check: each hand's tool point, read at the calls during the circle and fit with an ellipse
    (#_sizes), at least #CIRCLE_PASS across both ways and #CIRCLE_ROUND as wide as long.

    @param plan The plan the run played.
    @param calls The calls, in order.
    @param rate The rows a second the loop played (#_fit).
    @return The line: a PROBLEM when the check could not draw the circle from where the arms stood, or a hand
        traced less or a flat band; a SKIP when the loop called too rarely in it (under #CIRCLE_CALLS times, or never
        where the circle is wide) to see the circle.
    """
    whole = f"{200 * CIRCLE_R:.0f} cm circle"
    small = [f"{_hand(side)} {100 * size:.1f} cm" for side, size in plan.across.items() if size < CIRCLE_PASS]
    if small:
        return Line(
            "PROBLEM",
            "circle",
            f"the check could not draw the {whole} from where the arms stood ({', '.join(small)} across): start them "
            "clear of their joint limits and run this again",
        )
    together = next(s for s in plan.segments if s.kind == "together")
    during = [c for c in calls if together.start <= c.row <= together.end]
    turn = _turn()[np.clip([c.row - together.start for c in during], 0, TOGETHER_ROWS)]  # (calls,) radians
    traced: dict[str, tuple[float, float, np.ndarray]] = {}
    misses: dict[str, list[str]] = {}
    for side in plan.across:
        read = [
            (p, a) for p, a in zip((_tool(plan, side, c.state) for c in during), turn, strict=True) if p is not None
        ]
        planned = _wave(np.array([_planned_tool(plan, side, c.row - together.start) for c in during]), turn)
        if len(read) < CIRCLE_CALLS or _sizes(planned)[1] < CIRCLE_PASS:
            return Line("SKIP", "circle", f"the loop called {len(during)} times in the circle, too rarely to see it")
        wave = _wave(np.array([p for p, _ in read]), np.array([a for _, a in read]))
        timed = [
            (a, b, p)
            for a, b in zip(calls[:-1], calls[1:], strict=True)
            if together.start <= b.row <= together.end and (p := _tool(plan, side, b.state)) is not None
        ]
        path = np.array([_planned_tool(plan, side, r) for r in range(TOGETHER_ROWS + 1)])  # (rows, 3 xyz)
        traced[side], misses[side] = (
            _sizes(wave),
            _misses(
                path,
                np.array([p for _, _, p in timed]),
                np.array([a.row - together.start for a, _, _ in timed], dtype=float),
                np.array([b.t - a.t for a, b, _ in timed]),
                rate,
            ),
        )
    fit = f"fit to the {len(during)} calls in it"
    said = ", ".join(f"{_hand(side)} {100 * a:.1f} by {100 * b:.1f} cm" for side, (a, b, _) in traced.items())
    small = [side for side, (long, _, _) in traced.items() if long < CIRCLE_PASS]
    if small:
        return Line(
            "PROBLEM",
            "circle",
            f"{' and '.join(_hand(side) for side in small)} traced under {100 * CIRCLE_PASS:.0f} cm of the {whole} "
            f"({said}, {fit}): the loop did not play the circle",
        )
    flat = {side: v for side, v in traced.items() if v[1] < CIRCLE_PASS or v[1] < CIRCLE_ROUND * v[0]}
    if flat:
        why = "; ".join(
            f"{_hand(side)}'s "
            + (
                ", ".join(misses[side])
                or f"path went only {100 * wide:.1f} cm {'up and down' if abs(way[2]) >= 0.7 else 'across'} of "
                f"{200 * CIRCLE_R:.0f}"
            )
            for side, (_, wide, way) in flat.items()
        )
        return Line(
            "PROBLEM",
            "circle",
            f"{' and '.join(_hand(side) for side in flat)} traced a flat band, not the {whole} ({said}, {fit}): "
            f"{why}; {CIRCLE_CAUSE}",
        )
    return Line("PASS", "circle", f"each hand traced its {whole} ({said}, {fit})")


def _hand(side: str) -> str:
    """A side's hand, in words."""
    return f"the {side} hand" if side else "the hand"


def _tool(plan: Plan, side: str, state: Mapping[str, np.ndarray]) -> np.ndarray | None:
    """A side's tool point `(3 xyz,)` in a reading: the hand pose the loop reports, else its known arm's at the
    joints; None for neither."""
    if f"{side}_eef_xyz" in state:
        return np.asarray(state[f"{side}_eef_xyz"], dtype=float)[:3]
    part = f"{side}_arm" if side else "arm"
    arm = plan.arms.get(side) if plan.arms else None
    if arm is None or part not in state:
        return None
    return arm.hand(np.asarray(state[part], dtype=float)[None])[0, :3]


def _planned_tool(plan: Plan, side: str, row: int) -> np.ndarray:
    """A side's tool point `(3 xyz,)` in the plan's circle at one of its rows (clamped to the circle's)."""
    for part, move in plan.circle.items():
        if _side(part) != side or _kind(part) not in ("joint", "xyz"):
            continue
        at = move[int(np.clip(row, 0, len(move) - 1))]
        arm = plan.arms.get(side) if plan.arms and _kind(part) == "joint" else None
        return plan.start[part][:3] + at if arm is None else arm.hand((plan.start[part] + at)[None])[0, :3]
    raise KeyError(side)


def trace_path() -> pathlib.Path:
    """Where a run's trace goes when no `--trace` names a file.

    @return `~/.dreamlabs/check-<date>-<time>.json`, at this second.
    """
    return pathlib.Path.home() / ".dreamlabs" / f"check-{time.strftime('%Y%m%d-%H%M%S')}.json"


def save_trace(stand_in: StandIn, lines: Sequence[Line], path: pathlib.Path) -> None:
    """Writes the run (the stand-in's `trace()`) and its lines to `path`, making its folder.

    @param stand_in The stand-in, after a run.
    @param lines The lines the run printed.
    @param path The JSON file.
    """
    trace = stand_in.trace()
    trace["lines"] = [dataclasses.asdict(line) for line in lines]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace))


def _alone(plan: Plan, channels: Sequence[Channel], at_fault: set[Channel]) -> list[Channel]:
    """The parts to move alone, in the plan's order: the channels at fault, but on an arm the client's IK drives,
    which picks the joints that carry each hand's move, every joint of each arm whose hand is at fault (a joint dream's
    IK holds stays still), then the grippers at fault."""
    if not plan.hands:
        return [ch for ch in channels if ch in at_fault]
    sides = {_side(ch.part) for ch in at_fault if ch.kind != "grip"}
    joints = [
        Channel(f"{side}_arm", col, "joint")
        for side in plan.sides()
        if side in sides
        for col, (lower, upper) in enumerate(plan.limits[f"{side}_arm"])
        if upper > lower
    ]
    return joints + [ch for ch in channels if ch.kind == "grip" and ch in at_fault]


@dataclasses.dataclass(frozen=True)
class Channel:
    """One number of the state (a part's column) or one hand as a whole: its kind, and its name in a line.

    On an arm the client's IK drives (`Plan.hands`), each hand is read whole: its tool point (kind `hand`, part
    `<side>_eef_xyz`, metres) and its turn from where it stood about the base's x, y and z (kind `turn`, part
    `<side>_eef_rot6`, radians, #dream.check_arm.turned), 3 numbers each.
    """

    part: str
    col: int
    kind: str

    @property
    def name(self) -> str:
        """The channel as a line names it.

        As `right_arm joint 3`, `right_eef_xyz x`, `right hand`, `right hand's turn` or `right_gripper`.
        """
        if self.kind == "joint":
            return f"{self.part} joint {self.col + 1}"
        if self.kind == "xyz":
            return f"{self.part} {'xyz'[self.col]}"
        if self.kind == "hand":
            return f"{_side(self.part)} hand".strip()
        if self.kind == "turn":
            return f"{_side(self.part)} hand's turn".strip()
        return self.part

    def show(self, value: float) -> str:
        """A size in this channel's unit: degrees, cm, or the gripper's own.

        @param value The size in the state's unit: radians, metres, the gripper's.
        @return It in words.
        """
        if self.kind in ("joint", "turn"):
            return f"{math.degrees(value):.1f} degrees"
        if self.kind in ("xyz", "hand"):
            return f"{value * 100:.1f} cm"
        return f"{value:.2f}"


def _nothing_moved(plan: Plan) -> str:
    """Why no joint (on an arm the check does not know, or one the client's IK drives, no hand) moved when told to, as
    the line says it."""
    if plan.arms is None:
        return (
            "no hand moved: your loop does not send the robot the commands it gets (on an arm the check does not "
            "know, it moves each hand's pose and holds the joints, as the client's `ik` head `eef` runs them)"
        )
    if plan.hands:
        return "no hand moved: your loop does not send the robot the commands it gets"
    return "no joint moved: your loop does not send the robot the commands it gets"


def _channels(plan: Plan) -> list[Channel]:
    """The channels the checks read: every joint (on an arm the check does not know, every tool point axis; on one the
    client's IK drives, each hand's tool point and turn, `Plan.hands`) and gripper.

    Beside a `<side>_grip` (0 closed to 1 open, which the client's `ik` adds and drives), that side's
    `<side>_gripper` is the arm's own reading in its own unit (a Metal arm's millimetres): not read.
    """
    out = []
    for part, value in plan.start.items():
        kind = _kind(part)
        if part.endswith("gripper") and part[: -len("gripper")] + "grip" in plan.start:
            continue
        if plan.hands and kind in ("xyz", "rot"):
            out.append(Channel(part, 0, "hand" if kind == "xyz" else "turn"))
        elif kind == "grip" or (not plan.hands and kind == ("joint" if plan.arms is not None else "xyz")):
            out += [Channel(part, col, kind) for col in range(value.size)]
    return out


def _unit(ch: Channel) -> float:
    """The size the fits weigh a channel's error in: a degree of a joint or a hand's turn, a cm of a tool point, all
    of a gripper."""
    return {"joint": math.radians(1), "turn": math.radians(1), "xyz": 0.01, "hand": 0.01, "grip": 1.0}[ch.kind]


def _least(ch: Channel) -> float:
    """The least move of a joint, a tool point or a hand's turn that its lines judge: 3 degrees, 1 cm."""
    return 0.01 if ch.kind in ("xyz", "hand") else math.radians(3)


def _value(plan: Plan, state: Mapping[str, np.ndarray], ch: Channel) -> np.ndarray:
    """A channel's reading in a state: `()`, or a hand's `(3,)` (its turn from where it stood at the first call); nan
    where the state lacks it."""
    whole = ch.kind in ("hand", "turn")
    value = state.get(ch.part)
    if value is None or value.size <= (5 if ch.kind == "turn" else 2 if whole else ch.col):
        return np.full(3, math.nan) if whole else np.asarray(math.nan)
    if ch.kind == "turn":
        return np.asarray(check_arm.turned(value[None], plan.start[ch.part])[0])
    return value[:3].copy() if whole else np.asarray(value[ch.col])


def _target_table(plan: Plan, channels: Sequence[Channel]) -> dict[Channel, np.ndarray]:
    """Per channel, its target at every row of the plan and a chunk past it: `(rows,)`, or a hand's `(rows, 3)`."""
    table = plan.table()
    out = {}
    for ch in channels:
        if ch.kind == "turn":
            out[ch] = check_arm.turned(table[ch.part], plan.start[ch.part])
        else:
            out[ch] = table[ch.part] if ch.kind == "hand" else table[ch.part][:, ch.col]
    return out


def _interp(rows: np.ndarray, table: np.ndarray) -> np.ndarray:
    """A channel's targets (#_target_table) at fractional rows, held at the table's ends: `(len(rows),)` or
    `(len(rows), 3)`."""
    at = np.arange(table.shape[0])
    if table.ndim == 1:
        return np.asarray(np.interp(rows, at, table))
    return np.stack([np.interp(rows, at, table[:, k]) for k in range(table.shape[1])], axis=1)


def _at_rest(plan: Plan, rows: np.ndarray) -> np.ndarray:
    """Which rows hold every joint and tool point still: the hold and the rests between moves."""
    return np.array([all(not np.any(o) for p, o in plan.offsets(r).items() if _kind(p) != "grip") for r in rows])


def _fit(error: Callable[[float, float], float], rate: float) -> tuple[float, float]:
    """The play rate and lag that make `error(lag, rate)` least, the lag at most #MAX_FIT_LAG_S.

    The loop reports the state only when it calls, and most loops call at a steady cadence, so a loop that plays slower
    than `rate` and one whose robot runs late look alike at the calls: both leave the robot the same rows behind the
    plan. Of the fits whose error is within #FIT_SLACK of the best (in the channels' units, #_unit), the one with the
    rate nearest `rate` wins; a robot further behind than #MAX_FIT_LAG_S can explain is a rate PROBLEM.
    """

    def best_of(grid: list[tuple[float, float]]) -> tuple[float, float]:
        scored = [(error(lag, r), r, lag) for r, lag in grid]
        floor = min(e for e, _, _ in scored)
        near = [(abs(r - rate), e, lag, r) for e, r, lag in scored if e <= floor + FIT_SLACK**2]
        _, _, lag, r = min(near)
        return r, lag

    r0, lag0 = best_of(
        [
            (r, lag)
            for r in np.linspace(0.5 * rate, 1.5 * rate, 41)
            for lag in np.arange(0.0, MAX_FIT_LAG_S + 1e-9, 0.05)
        ]
    )
    return best_of(
        [
            (r, lag)
            for r in np.linspace(r0 - 0.03 * rate, r0 + 0.03 * rate, 13)
            for lag in np.arange(max(0.0, lag0 - 0.05), min(MAX_FIT_LAG_S, lag0 + 0.05) + 1e-9, 0.01)
        ]
    )


def _hold(
    calls: Sequence[Call], channels: Sequence[Channel], start: Mapping[Channel, np.ndarray], stand_in: StandIn
) -> tuple[Line, set[Channel]]:
    """Nothing may move while the robot is told to hold: the calls in the hold, against the first call's state (a hand
    the client's IK drives, against it or where the IK may hold it instead, `Plan.settled`).

    @return The line, and the channels that moved.
    """
    plan = stand_in.plan
    assert plan is not None
    held = [c for c in calls[1:] if (c.t - stand_in.t0) * stand_in.rate <= HORIZON + 0.25 * stand_in.rate]
    moved = {}
    for ch in channels:
        refs = [start[ch]] + ([_value(plan, plan.settled, ch)] if ch.part in plan.settled else [])
        worst = max(
            (min(float(np.linalg.norm(_value(plan, c.state, ch) - ref)) for ref in refs) for c in held), default=0.0
        )
        limits = {"joint": HOLD_JOINT, "turn": HOLD_JOINT, "xyz": HOLD_TOOL, "hand": HOLD_TOOL, "grip": HOLD_GRIP}
        if worst > limits[ch.kind]:
            moved[ch] = f"{ch.name} {ch.show(worst)}"
    if not held:
        return Line("PASS", "hold", "no call came during the hold; nothing to judge (the loop calls rarely)"), set()
    if moved:
        said = ", ".join(list(moved.values())[:4])
        return Line("PROBLEM", "hold", "moved during the hold: " + said), set(moved)
    return Line("PASS", "hold", f"nothing moved during the hold (calls: {len(held)})"), set()


def _slots(plan: Plan) -> list[tuple[Segment, Channel, int, int]]:
    """Each move of one channel alone: its segment, the channel, its first and last row."""
    out = []
    for s in plan.segments:
        if s.kind in ("joint", "axis"):
            kind = "joint" if s.kind == "joint" else "xyz"
            out.append((s, Channel(s.parts[0], s.cols[0], kind), s.start, s.start + s.rows))
    return out


def _order(
    plan: Plan,
    channels: Sequence[Channel],
    measured: Mapping[Channel, np.ndarray],
    base: Mapping[Channel, np.ndarray],
    rows: np.ndarray,
    commanded: Callable[[Channel, float], np.ndarray],
) -> tuple[Line, dict[Channel, Channel], set[Channel]]:
    """Each move of one channel alone must move that channel most, and no other past #OTHER_JOINT (#OTHER_TOOL); in the
    together-motion, each joint must follow its own command, not another's far better (#_crossed). On an arm the
    client's IK drives, each hand must follow its own command so, in the together-motion and while each joint moves
    alone.

    @return The line; per channel commanded, the channel that moved for it (itself when right); the channels wrong.
    """
    wrong, unseen, match, bad = [], [], {}, set()
    moving = [c for c in channels if c.kind != "grip"]
    start = {c: _value(plan, plan.start, c) for c in moving}
    alone = {ch for _, ch, _, _ in _slots(plan)}
    for s in plan.segments:
        if s.kind == "together" or (plan.hands and s.kind == "joint"):
            inside = (rows >= s.start) & (rows < s.end + 0.3 * HORIZON)
            judged = [c for c in moving if c not in alone]
            for got, told in _crossed(judged, moving, measured, base, start, inside, commanded):
                match[told] = got
                bad |= {told, got}
                wrong.append(f"{told.name}'s command moved {got.name}")
    for _, ch, first, last in _slots(plan):
        inside = (rows >= first) & (rows < last + 0.3 * HORIZON)
        if not inside.any():
            unseen.append(ch.name)
            continue
        if plan.hands:
            continue  # the IK picks the joints: the hands are judged above
        size = {c: float(np.max(np.abs(measured[c][inside] - base[c]))) for c in moving}
        top = max(size, key=lambda c: size[c])
        match[ch] = top
        other = OTHER_JOINT if ch.kind == "joint" else OTHER_TOOL
        if top != ch:
            wrong.append(f"{ch.name}'s command moved {top.name} ({top.show(size[top])})")
            bad |= {ch, top}
        elif any(size[c] > other for c in moving if c != ch):
            also = max((c for c in moving if c != ch), key=lambda c: size[c])
            wrong.append(f"{ch.name}'s command also moved {also.name} ({also.show(size[also])})")
            bad |= {ch, also}
    if wrong:
        return (
            Line(
                "PROBLEM",
                "right part moved",
                "; ".join(_unique(wrong)[:4]) + ": the loop's parts are in another order "
                "than the model's (joint order, or the arms swapped)",
            ),
            match,
            bad,
        )
    seen = len(_slots(plan)) - len(unseen)
    together = any(s.kind == "together" for s in plan.segments)
    said = (
        "each hand's move and turn followed its own command"
        + (f", also while {seen} joints moved alone" if seen else "")
        if plan.hands
        else f"each joint followed its own command: {len(moving)} at once" + (f", then {seen} alone" if seen else "")
        if together
        else f"each command moved its own part ({seen} moves seen)"
    )
    note = f"; no call came while {', '.join(unseen[:3])} moved" if unseen else ""
    return Line("PASS", "right part moved", said + note), match, bad


def _crossed(
    judged: Sequence[Channel],
    moving: Sequence[Channel],
    measured: Mapping[Channel, np.ndarray],
    base: Mapping[Channel, np.ndarray],
    start: Mapping[Channel, np.ndarray],
    inside: np.ndarray,
    commanded: Callable[[Channel, float], np.ndarray],
) -> list[tuple[Channel, Channel]]:
    """The joints that followed another joint's command in the together-motion: `(the joint, the command's joint)`.

    Each joint's moves at the calls inside the motion (`inside`) are fitted against every joint's command, each with
    its best lag (0 to 1 s) and gain; it followed another's when that fit leaves under a quarter of its own's error.
    A hand's tool point is fitted against each hand's tool point command, its turn against each hand's turn.
    """
    if int(inside.sum()) < 3:
        return []
    lags = np.arange(0.0, 1.01, 0.05)
    told = {c: [commanded(c, lag)[inside] - start[c] for lag in lags] for c in moving}

    def miss(y: np.ndarray, c: Channel) -> float:
        out = 1.0
        for x in told[c]:
            xx = float(np.sum(x * x))
            if xx > 0:
                out = min(
                    out, float(np.sum((y - float(np.sum(x * y)) / xx * x) ** 2) / max(float(np.sum(y * y)), 1e-12))
                )
        return out

    out = []
    for ch in judged:
        y = measured[ch][inside] - base[ch]
        if float(np.max(np.abs(y))) < _unit(ch):
            continue  # it hardly moved: the direction and size line says so
        fits = {c: miss(y, c) for c in moving if c.kind == ch.kind}
        best = min(fits, key=lambda c: fits[c])
        if best != ch and fits[best] < 0.25 * fits[ch]:
            out.append((ch, best))
    return out


def _direction(
    plan: Plan,
    channels: Sequence[Channel],
    measured: Mapping[Channel, np.ndarray],
    base: Mapping[Channel, np.ndarray],
    match: Mapping[Channel, Channel],
    commanded: Callable[[Channel], np.ndarray],
) -> tuple[Line, set[Channel]]:
    """Each part must move the way and as far as told: over every call, the part that moved (#_order's match) against
    its command, a gain between #GAIN_RANGE; a gripper's against its targets, which must rise as it opens.

    @return The line, and the channels that moved the wrong way or size, or never far enough at a call to judge.
    """
    bad, gains, wrong = [], [], set()
    for ch in channels:
        c = commanded(ch)
        if ch.kind == "grip":
            x, y = c - c.mean(), measured[ch] - measured[ch].mean()
            if float(np.ptp(c)) < 0.3:
                continue
            gain = float(np.sum(x * y) / np.sum(x * x))
            if gain < 0:
                bad.append(f"{ch.part} closed when told to open (x{gain:.1f}): 0 must be closed and 1 open")
                wrong.add(ch)
            elif not GAIN_RANGE[0] <= gain <= GAIN_RANGE[1]:
                bad.append(f"{ch.part} moved x{gain:.1f} of its command: 0 closed to 1 open")
                wrong.add(ch)
            continue
        x = c - _value(plan, plan.start, ch)
        if float(np.max(np.abs(x))) < _least(ch):
            wrong.add(ch)  # no call came while it had moved far enough to judge: it moves alone next
            continue
        top = match.get(ch, ch)
        gain = float(np.sum((measured[top] - base[top]) * x) / np.sum(x * x))
        gains.append(gain)
        if not GAIN_RANGE[0] <= gain <= GAIN_RANGE[1]:
            hint = " (degrees for radians?)" if abs(gain) > 20 or abs(gain) < 0.05 else ""
            bad.append(
                f"{top.name} went the other way (x{gain:.1f})"
                if gain < 0
                else f"{top.name} moved x{gain:.1f} of its command{hint}"
            )
            wrong |= {ch, top}
    if bad:
        return Line("PROBLEM", "direction and size", "; ".join(_unique(bad)[:4])), wrong
    if not gains:
        return Line("PASS", "direction and size", "no call came during a move; nothing to judge"), wrong
    return Line(
        "PASS",
        "direction and size",
        f"every part moved x{min(gains):.2f} to x{max(gains):.2f} of its command; grippers as told",
    ), wrong


def _own_lags(
    moving: Sequence[Channel],
    measured: Mapping[Channel, np.ndarray],
    match: Mapping[Channel, Channel],
    commanded: Callable[[Channel, float], np.ndarray],
) -> dict[Channel, tuple[float, float]]:
    """Per part that moved far enough to judge, its own lag behind its targets and the error left after it.

    Fitted against the part that moved for it (#_order's match), with its own gain, so an order or sign PROBLEM is not
    counted again as lag.

    @return Per channel, `(lag in seconds, error in its unit after it)`.
    """
    out = {}
    for ch in moving:
        if float(np.max(np.ptp(commanded(ch, 0.0), axis=0))) < _least(ch):
            continue
        lag, rms, _ = _lag_fit(measured[match.get(ch, ch)], functools.partial(commanded, ch))
        out[ch] = (lag, rms)
    return out


def _lag_fit(y: np.ndarray, commanded: Callable[[float], np.ndarray]) -> tuple[float, float, float]:
    """The lag (seconds, 0 to #OWN_LAG_MAX in steps of 0.02) at which readings `y` `(calls,)` or `(calls, 3)` best
    fit their commands there (`commanded(lag)`, the same shape) with a gain of their own, about each one's mean.

    @return The lag, the error left after it (in y's unit; a hand's, its distance each call) and the gain.
    """
    y = y - y.mean(axis=0)
    fits = []
    for lag in np.arange(0.0, OWN_LAG_MAX + 0.01, 0.02):
        x = commanded(float(lag))
        x = x - x.mean(axis=0)
        gain = float(np.sum(x * y) / np.sum(x * x))
        e = y - gain * x
        fits.append((float(np.sqrt(np.sum(e * e) / len(e))), float(lag), gain))
    rms, lag, gain = min(fits)
    return lag, rms, gain


def _lag(own: Mapping[Channel, tuple[float, float]], lag: float) -> tuple[Line, set[Channel]]:
    """The slowest part's lag behind its targets, and the largest error left after each part's lag (#_own_lags).

    @return The line, and the parts over either limit.
    """
    if not own:
        return Line("PASS", "lag and tracking", "no call came during a move; nothing to judge"), set()
    slow = max(own, key=lambda ch: own[ch][0])
    worst = max(own, key=lambda ch: own[ch][1] / _unit(ch))
    track = {ch: MAX_TRACK_TOOL if ch.kind in ("xyz", "hand") else MAX_TRACK_JOINT for ch in own}
    limit = track[worst]
    over = {ch for ch, (late, miss) in own.items() if late > MAX_LAG_S or miss > track[ch]}
    if own[slow][0] > MAX_LAG_S:
        return Line(
            "PROBLEM",
            "lag and tracking",
            f"{slow.name} is {own[slow][0]:.2f} s behind (the whole robot {lag:.2f} s, limit {MAX_LAG_S:.2f} s): "
            + (
                "your loop delays, queues or drops its commands, or every joint follows them slowly"
                if lag > MAX_LAG_S
                else "the rest is on time, so the joints that move it follow their commands slowly (low position "
                "gains, friction, or a joint sagging under its weight), or your loop delays its commands"
            ),
        ), over
    if own[worst][1] > limit:
        return Line(
            "PROBLEM",
            "lag and tracking",
            f"{worst.name} misses its targets by {worst.show(own[worst][1])} "
            f"after its lag (limit {worst.show(limit)})",
        ), over
    return Line(
        "PASS",
        "lag and tracking",
        f"{lag:.2f} s behind (slowest part {own[slow][0]:.2f} s, limit {MAX_LAG_S:.2f} s), then off by at most "
        f"{worst.show(own[worst][1])}",
    ), over


def _unique(items: Sequence[str]) -> list[str]:
    seen: list[str] = []
    for item in items:
        if item not in seen:
            seen.append(item)
    return seen


# ---------------------------------------------------------------------------------------------------------------------
# The command.


START = "the workspace is clear"  # what the person at the robot types before anything moves, with an agent or not


def confirm(moving: str) -> bool:
    """Says what will move and asks the person at the robot to type START; nothing is served before it.

    @param moving What moves and for how long, a sentence.
    @return True when they typed it.
    """
    print(
        f"{moving}\nKeep a hand on the e-stop. An agent types these words only after the person at the robot has "
        "typed them to it.",
        flush=True,
    )
    try:
        answer = input(f'Type "{START}" to start: ')
    except EOFError:  # no terminal to type in
        answer = ""
    if answer.strip().lower() == START:
        return True
    print("Not started: nothing moved.", flush=True)
    return False


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m dream.check_loop`: serves the stand-in until the plan has played, then prints one line per check.

    @param argv The arguments; `sys.argv[1:]` when None.
    @return 0 when every check passes, 1 when one is a PROBLEM, 2 for a bad argument or no "the workspace is clear".
    """
    parser = argparse.ArgumentParser(prog=COMMAND, description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"the port to serve on ({DEFAULT_PORT})")
    parser.add_argument(
        "--rate",
        type=float,
        default=DEFAULT_RATE,
        help=f"rows a second your loop plays (the client's action_hz; {DEFAULT_RATE:g})",
    )
    parser.add_argument(
        "--trace",
        type=pathlib.Path,
        help="write the run (targets, calls, lines) to this JSON file (~/.dreamlabs/check-<date>-<time>.json)",
    )
    args = parser.parse_args(argv)
    try:
        check_arm.ik_specs()
    except DreamConfigError as error:
        print(f"{COMMAND}: {error}", file=sys.stderr)
        return 2
    moving = (
        f"Your loop moves the arms: each hand circles twice, {200 * CIRCLE_R:.0f} cm across, while every joint "
        f"turns, about {(HORIZON + TOGETHER_ROWS + REST_ROWS) / args.rate:.0f} s at {args.rate:g} rows a second, "
        "more if a part is at fault."
    )
    if not confirm(moving):
        return 2
    stand_in = StandIn(rate=args.rate)
    url = stand_in.serve(args.port)
    print(
        f"The stand-in for the model answers at {url}. Point your loop at it with\n\n"
        f"    DREAM_GATEWAY_URL={url}\n\n"
        "and run it as you would with the model.",
        flush=True,
    )
    try:
        watch(stand_in)
        time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nstopped by you: the lines judge the calls so far", flush=True)
    stand_in.close()
    lines = judge(stand_in) + [camera_rate(stand_in)]
    print("")
    for line in lines:
        print(line)
    if stand_in.plan is not None:
        path = args.trace or trace_path()
        save_trace(stand_in, lines, path)
        print(f"\nThe run's trace (the check's targets and every call): {path}")
    return 1 if any(line.status == "PROBLEM" for line in lines) else 0


if __name__ == "__main__":
    sys.exit(main())
