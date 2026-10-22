"""Fixed-rate inference control loop: observe, refill the chunk buffer, command both arms.

#ControlLoop is the loop's executor. It owns the tick clock, the action-chunk buffer and the single in-flight
`get_action` call; the arms, the cameras and the dream client (a `dream.DreamClient` with its policy open) are
built elsewhere and passed in. The client keeps the policy's frame window: the loop hands it each camera's newest
encoded frame and the arms' state on every tick that brings new frames (`set_observation()`), and a refill asks it
for the next chunk (`get_action()`). The loop converts to and from the serve's dicts through `robots.adapter` and
imports `dream` only for its error types, which the tick boundary sorts into "retry" and
"stop". The first commands after start and after a session reset are ramped
from the arms' measured pose into the policy's targets; a commanded close is held to
`grip_force_cap` on the measured gripper effort, so a grasp squeezes to a chosen force rather than to
whatever the motor PD's leftover position error implies (#robots.gripper_force); and every
command's end-effector speed is capped against the previous one, through the arms' kinematics, as the
last line of defence. Nothing here smooths: the blend between consecutive chunks
is the dream client's, done before
the chunk reaches the loop from the joints the loop last commanded each arm, which go beside every
request (the wire's `commanded`, `robots.adapter.commanded_joints`, never sent), and no filter sits on the command
path.
The tick does not write the arms itself: it submits the command to #_CommandWriter, which runs it out to both arms at
`command_hz` on its own thread, interpolated between rows as a YAM follower's command thread
does, or, with System 0 on, to #robots.sys0.motors.MotorPath, whose trackers drive the motors from
the plan once the start-up ramp has ended. A prompt change mid-run (set_task()) is applied at the next safe point,
when no refill is in flight: the client's episode is reset, the buffer dropped and the ramp
made due again, so the new task starts as a fresh episode. Every refill's round trip is kept
in the `timing` snapshot, the rows still to run and the last seconds of commanded ticks in the
`plan` snapshot (#PlanSnapshot, for a viewer's charts), and with `inference_log_path`
every inference, error, stall and run boundary goes to a JSON Lines file (#robots.inference_log).

@see ControlLoop, ControlLoop.run(), ControlLoop.step(), ControlLoop.set_task()
"""

import asyncio
import bisect
import collections
import csv
import dataclasses
import logging
import math
import pathlib
import threading
import time
from collections.abc import Awaitable, Callable
from types import TracebackType
from typing import Any

import dream
import numpy as np
import numpy.typing as npt
from shared.contracts import envelope

from robots import (
    adapter,
    cameras,
    errors,
    followers,
    gripper_force,
    inference_log,
    kinematics,
    rollout_episode,
    rollouts,
    types,
)
from robots.sys0 import motors as sys0_motors
from robots.sys0 import tracker as sys0_tracker

## Seconds `__aenter__` waits for every camera to publish its first frame.
FIRST_FRAME_TIMEOUT_S: float = 5.0

## Poll interval while waiting for first frames; short enough not to delay startup noticeably.
_FIRST_FRAME_POLL_S: float = 0.01

## Seconds a refill waits after a retryable refusal that carries no `retry_after_s` of its own, and after a failure
## that would repeat at once if asked again (a chunk of the wrong shape for the embodiment, a gateway nothing answers
## at), so a refusal is collected about once a second instead of thirty times a second.
RETRY_AFTER_DEFAULT_S: float = 1.0

RETRY_LOG_EVERY_S: float = 5.0

SERVE_FAULT_REPEATS: int = 3

POOL_WARMING_LIMIT_S: float = 1800.0

## Seconds after the first tick the first chunk may land and still be run: one that lands later (a model that was
## starting, a cold one for 15 minutes or more) stops the run before its first move (#robots.errors.ModelReady), so
## the arms never start moving on their own after a wait the operator may have walked away from.
MOVE_AFTER_WAIT_LIMIT_S: float = 60.0

## Trace rows buffered before a flush; one second of ticks at 30 Hz, so a crash loses at most that much.
_TRACE_FLUSH_ROWS: int = 30

## Seconds `__aexit__` waits for the command thread to finish its current write before holding the arms.
_COMMAND_STOP_TIMEOUT_S: float = 2.0

## Seconds of commanded ticks #PlanSnapshot keeps as history: two chunks' worth at the station's rate, enough to
## see whether the arms followed the plan they were given.
PLAN_HISTORY_S: float = 3.0

## The two arms, in the order the per-arm arrays of #PlanSnapshot and the trace columns are laid out.
ARMS: tuple[str, str] = ("left", "right")

## Commanded ticks the loop keeps in `ticks` for a reader that polls slower than the tick (the live agent's tick
## records): ten seconds at the station's 20 Hz, so a reader paused that long misses nothing.
TICK_RING: int = 200


def trace_columns(dofs: dict[str, int]) -> tuple[str, ...]:
    """Column order of the per-tick trace, documented in the README: a column per joint of each arm.

    The commanded columns are the tick's submitted command; `ee_speed_*` is the end-effector speed each arm asked for
    before the clamp; `commands_written` counts the command periods written so far.

    @param dofs Per arm, its joints (`robots.followers.FollowerArm.dof`).
    @return The columns.
    """
    return (
        ("t_s", "chunk_id", "row_index", "ramp_alpha", "clamped")
        + tuple(f"measured_l{i}" for i in range(dofs["left"]))
        + tuple(f"measured_r{i}" for i in range(dofs["right"]))
        + ("gripper_l", "gripper_r")
        + tuple(f"commanded_l{i}" for i in range(dofs["left"]))
        + tuple(f"commanded_r{i}" for i in range(dofs["right"]))
        + ("cmd_gripper_l", "cmd_gripper_r", "ee_speed_l_m_s", "ee_speed_r_m_s", "commands_written")
    )


_log = logging.getLogger(__name__)


@dataclasses.dataclass
class _Ramp:
    """One blend from the measured pose into the policy's targets.

    Attributes:
    - origin: The arms' measured pose on the tick the ramp started, as a command.
    - ticks: Commanded ticks the ramp lasts; the blend weight is `tick / ticks`.
    - tick: Commanded ticks so far.
    """

    origin: types.Action
    ticks: int
    tick: int = 0


@dataclasses.dataclass
class _EpisodeCounts:
    clamps: int = 0
    stalls: int = 0
    max_ee_speed_requested_m_s: float = 0.0
    left_max_motor_temp_c: float | None = None
    right_max_motor_temp_c: float | None = None

    def note_temps(self, temps: dict[str, list[float]] | None) -> None:
        """Keeps each arm's hottest reading.

        @param temps `_motor_temps()` of a tick's state.
        """
        if temps is None:
            return
        for arm, values in temps.items():
            field = f"{arm}_max_motor_temp_c"
            seen = getattr(self, field)
            setattr(self, field, max(values) if seen is None else max(seen, *values))


@dataclasses.dataclass(frozen=True)
class _Submission:
    """One tick's command for both arms, as handed to the command path.

    Attributes:
    - action: The command after the ramp and the clamp.
    - t_submit_s: Tick time it was submitted at, on the loop's clock, seconds.
    """

    action: types.Action
    t_submit_s: float


@dataclasses.dataclass
class _Interpolation:
    """One linear run from where the command path was to a newly submitted target.

    Attributes:
    - origin: Where the previous run had got to when the target arrived; the first run's is the target itself.
    - submission: The target and its submit time; the run is `origin + alpha * (target - origin)` with
      `alpha = (now + command period - t_submit_s) / window_s`, capped at 1.
    - window_s: Seconds the run spans: the gap between this submit and the previous one, clamped to
      `[1 / command_hz, command_interp_max_s]`.
    - position: Where the run has got to, the last value written.
    """

    origin: types.Action
    submission: _Submission
    window_s: float
    position: types.Action


@dataclasses.dataclass(frozen=True)
class RefillTiming:
    """Round-trip statistics of the refills collected so far in a run; `ControlLoop.timing` is rebound to a new
    one on every chunk that lands, so a concurrent reader always sees a consistent snapshot.

    `rtt_ms` is measured on the loop's clock, from the tick that started the refill to the tick that collected
    it, so it is quantised to a control period; the dream fields come from the client's optional `last_timing`
    and are None when the client has none.

    Attributes:
    - count: Refills that returned a chunk (installed or discarded) since the run started.
    - last_rtt_ms: Round trip of the last one, milliseconds.
    - p50_ms: Median round trip over the run, milliseconds (nearest rank).
    - p95_ms: 95th percentile round trip over the run, milliseconds (nearest rank).
    - serve_infer_ms: The serve's own inference time for the last refill, from `last_timing`.
    - gateway_ms: Time spent in DreamAPI for the last refill, from `last_timing`.
    - network_ms: The rest of the round trip for the last refill, from `last_timing`.
    - request_bytes: Size of the last request body, from `last_timing`.
    - response_bytes: Size of the last response body, from `last_timing`.
    - up_mbps: The upload rate the client achieved for the last request, megabits per second, from `last_timing`;
      None off Linux.
    """

    count: int = 0
    last_rtt_ms: float | None = None
    p50_ms: float | None = None
    p95_ms: float | None = None
    serve_infer_ms: float | None = None
    gateway_ms: float | None = None
    network_ms: float | None = None
    request_bytes: int | None = None
    response_bytes: int | None = None
    up_mbps: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """The snapshot as a plain dict, for a status endpoint.

        @return One key per attribute.
        """
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class PlanSnapshot:
    """The rows the loop is about to command and the ticks it just commanded, for a viewer.

    `ControlLoop.plan` is rebound to a new one on every commanded tick and never mutated, so a
    concurrent reader (a viewer's charts) always sees one consistent picture without a
    lock. Every history array is a fresh copy and every plan array a slice of arrays that are never
    written again; every time is seconds relative to `now_s`, at most zero for the history and
    ahead for the plan, so a chart puts both on one axis.

    Attributes:
    - now_s: Loop time the snapshot was taken at, seconds.
    - plan_t_s: When each remaining buffered row is due, `[rows]`, ascending: row `k` of a chunk is
      the target for `observed_at_s + k * period`.
    - plan: The remaining rows per arm (`"left"`, `"right"`), `[rows, dof + 1]`: the arm's joints in radians
      then the gripper, as the policy sent them, before the ramp and the clamp.
    - plan_ee_speed_m_s: The end effector's speed along the plan per arm, `[rows]`, from the
      kinematics when the chunk was installed; NaN for a row with no predecessor and without kinematics.
    - history_t_s: The last commanded ticks, `[n]`, ascending and at most 0; at most #PLAN_HISTORY_S seconds.
    - measured: The measured joints and gripper per arm on those ticks, `[n, dof + 1]`.
    - commanded: The commands submitted on those ticks per arm, after ramp and clamp, `[n, dof + 1]`.
    - clamped: Whether the clamp bounded the command on each of those ticks, `[n]`.
    - ee_speed_m_s: The end effector's measured speed per arm on those ticks, `[n]`, from consecutive
      measured poses; NaN on the first tick of a run and without kinematics.
    - max_ee_speed_m_s: The configured cap, so a chart can draw it.

    @see ControlLoop.plan, empty()
    """

    now_s: float
    plan_t_s: npt.NDArray[np.float64]
    plan: dict[str, npt.NDArray[np.float64]]
    plan_ee_speed_m_s: dict[str, npt.NDArray[np.float64]]
    history_t_s: npt.NDArray[np.float64]
    measured: dict[str, npt.NDArray[np.float64]]
    commanded: dict[str, npt.NDArray[np.float64]]
    clamped: npt.NDArray[np.bool_]
    ee_speed_m_s: dict[str, npt.NDArray[np.float64]]
    max_ee_speed_m_s: float

    @classmethod
    def empty(cls, max_ee_speed_m_s: float, dofs: dict[str, int]) -> "PlanSnapshot":
        """A snapshot with no rows and no history: what a loop publishes before its first commanded tick.

        @param max_ee_speed_m_s The configured cap.
        @param dofs Per arm, its joints.
        @return The empty snapshot.
        """
        return cls(
            now_s=0.0,
            plan_t_s=np.zeros(0),
            plan={arm: np.zeros((0, dofs[arm] + 1)) for arm in ARMS},
            plan_ee_speed_m_s={arm: np.zeros(0) for arm in ARMS},
            history_t_s=np.zeros(0),
            measured={arm: np.zeros((0, dofs[arm] + 1)) for arm in ARMS},
            commanded={arm: np.zeros((0, dofs[arm] + 1)) for arm in ARMS},
            clamped=np.zeros(0, dtype=bool),
            ee_speed_m_s={arm: np.zeros(0) for arm in ARMS},
            max_ee_speed_m_s=max_ee_speed_m_s,
        )


@dataclasses.dataclass(frozen=True)
class _ChunkPlan:
    """The installed rows of the current chunk as arrays, built once per install for #PlanSnapshot.

    Attributes:
    - rows: The installed rows per arm, `[n, dof + 1]`; the buffer is always their tail.
    - ee_speed_m_s: The end effector's speed per arm from each row's predecessor to it, `[n]`.
    - due_s: Loop time each row is due at, `[n]`.
    """

    rows: dict[str, npt.NDArray[np.float64]]
    ee_speed_m_s: dict[str, npt.NDArray[np.float64]]
    due_s: npt.NDArray[np.float64]


@dataclasses.dataclass(frozen=True)
class _ClampResult:
    """One tick's clamp outcome.

    Attributes:
    - action: The command to submit.
    - clamped: Whether either arm's joints or gripper were cut.
    - ee_speed_m_s: The end-effector speed each arm asked for before the clamp, keyed by arm; NaN without
      kinematics and on the first command.
    """

    action: types.Action
    clamped: bool
    ee_speed_m_s: dict[str, float]


class _History:
    """A fixed-size ring of the last commanded ticks: the history half of #PlanSnapshot.

    Rings rather than lists so a tick appends in constant time without allocating; snapshot()
    copies the rings out in order, which is where the per-tick cost goes (a few hundred floats).
    The measured end-effector speed is taken between consecutive appended ticks on the loop's
    clock, so a hold between two commanded ticks lowers it rather than skipping it.

    @see append(), snapshot()
    """

    def __init__(
        self, size: int, arm_kinematics: dict[str, kinematics.Kinematics] | None, dofs: dict[str, int]
    ) -> None:
        """Allocates the rings, a column per joint of each arm and one for its gripper.

        @param size Ticks kept.
        @param arm_kinematics Per-arm kinematics for the measured end-effector speed; None leaves it NaN.
        @param dofs Per arm, its joints.
        """
        self._kinematics = arm_kinematics
        self._size = size
        self._dofs = dofs
        self._t_s = np.full(size, np.nan)
        self._measured = {arm: np.full((size, dofs[arm] + 1), np.nan) for arm in ARMS}
        self._commanded = {arm: np.full((size, dofs[arm] + 1), np.nan) for arm in ARMS}
        self._clamped = np.zeros(size, dtype=bool)
        self._ee_speed_m_s = {arm: np.full(size, np.nan) for arm in ARMS}
        self._head = 0
        self._count = 0
        # The previous appended tick's measured end-effector position per arm and its time, for the speed.
        self._prev_ee: dict[str, npt.NDArray[np.float64] | None] = {arm: None for arm in ARMS}
        self._prev_t_s = 0.0

    def append(self, now_s: float, state: types.RobotState, action: types.Action, clamped: bool) -> None:
        """Records one commanded tick, overwriting the oldest once the ring is full.

        @param now_s The tick time, seconds.
        @param state The tick's measured arm state.
        @param action The command submitted, after ramp and clamp.
        @param clamped Whether the clamp cut it.
        """
        i = self._head
        measured = _by_arm_state(state)
        commanded = _by_arm(action)
        self._t_s[i] = now_s
        for arm in ARMS:
            dof = self._dofs[arm]
            self._measured[arm][i, :dof] = measured[arm].positions_rad
            self._measured[arm][i, dof] = measured[arm].gripper
            self._commanded[arm][i, :dof] = commanded[arm].positions_rad
            self._commanded[arm][i, dof] = commanded[arm].gripper
            self._ee_speed_m_s[arm][i] = self._measured_speed(arm, measured[arm].positions_rad, now_s)
        self._clamped[i] = clamped
        self._prev_t_s = now_s
        self._head = (i + 1) % self._size
        self._count = min(self._count + 1, self._size)

    def snapshot(
        self,
        now_s: float,
        plan_t_s: npt.NDArray[np.float64],
        plan: dict[str, npt.NDArray[np.float64]],
        plan_ee_speed_m_s: dict[str, npt.NDArray[np.float64]],
        max_ee_speed_m_s: float,
    ) -> PlanSnapshot:
        """Copies the rings out in chronological order, relative to `now_s`, around the given plan half.

        @param now_s The time the snapshot is taken at; the history times are relative to it.
        @param plan_t_s When each remaining row is due, relative to `now_s`.
        @param plan The remaining rows per arm.
        @param plan_ee_speed_m_s The planned end-effector speed per arm.
        @param max_ee_speed_m_s The configured cap.
        @return The snapshot.
        """
        return PlanSnapshot(
            now_s=now_s,
            plan_t_s=plan_t_s,
            plan=plan,
            plan_ee_speed_m_s=plan_ee_speed_m_s,
            history_t_s=self._ordered(self._t_s) - now_s,
            measured={arm: self._ordered(self._measured[arm]) for arm in ARMS},
            commanded={arm: self._ordered(self._commanded[arm]) for arm in ARMS},
            clamped=self._ordered(self._clamped),
            ee_speed_m_s={arm: self._ordered(self._ee_speed_m_s[arm]) for arm in ARMS},
            max_ee_speed_m_s=max_ee_speed_m_s,
        )

    def _measured_speed(self, arm: str, joints_rad: npt.NDArray[np.float64], now_s: float) -> float:
        """The end effector's speed since the previous appended tick, and remembers this tick's position.

        @param arm `"left"` or `"right"`.
        @param joints_rad The measured joints this tick.
        @param now_s The tick time.
        @return Metres per second; NaN without kinematics, on the first tick, or when no time has passed.
        """
        if self._kinematics is None:
            return math.nan
        position = self._kinematics[arm].ee_position(joints_rad)
        previous, self._prev_ee[arm] = self._prev_ee[arm], position
        if previous is None or now_s <= self._prev_t_s:
            return math.nan
        return float(np.linalg.norm(position - previous)) / (now_s - self._prev_t_s)

    def _ordered(self, ring: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """A chronological copy of one ring's filled part.

        @param ring One of the rings, indexed like `_t_s` on its first axis.
        @return A new array of the `count` recorded entries, oldest first.
        """
        if self._count < self._size:
            return ring[: self._count].copy()
        return np.concatenate([ring[self._head :], ring[: self._head]])


@dataclasses.dataclass(frozen=True)
class TickSnapshot:
    """One commanded tick, as the live agent records it beside the video: what
    the arms reported and what they were told, with the chunk the command came from. Rebound whole, never
    mutated; a tick that held or stalled publishes `state` but no snapshot, so a gap in `tick` is a hold.

    Attributes:
    - tick: The loop's `tick_count` at this tick.
    - t_s: Monotonic tick time, seconds.
    - state: The #robots.types.RobotState read this tick.
    - action: The #robots.types.Action written this tick, after the ramp and the clamp.
    - chunk: The chunk count the row came from.
    - row: The row of that chunk.
    - clamped: Whether the rate clamp bounded the command.
    - session_id: The DreamAPI session the chunk came from, or None before the client opened one.
    """

    tick: int
    t_s: float
    state: types.RobotState
    action: types.Action
    chunk: int
    row: int
    clamped: bool
    session_id: str | None = None


class ControlLoop:
    """Fixed-rate inference loop driving both follower arms from a dream policy.

    Drains one action chunk tick by tick and refills it from `get_action()` before it runs
    dry. The dream client keeps the policy's frame window: a tick on which every
    camera has a frame encoded since the last one handed over hands the client that set with the arms' state
    (`set_observation()`, timed by the frames' capture time), and a refill hands it the refill tick's state, with
    the newest frames, on its own thread just before it asks. A chunk answers the observation it was computed from,
    row `k` being the target for `observed_at_s + k * period`, so the rows the arms have already executed
    when it lands are dropped rather than replayed. The first commands after start, and again
    after a session reset, are blended from the arms' measured pose into the policy's targets
    over `ramp_s`. Every later chunk is commanded as the dream client returned it: the
    blend from one chunk into the next is the client's, before the loop sees the rows,
    and the loop adds no smoothing of its own. A commanded close is held to
    `grip_force_cap` on each arm's measured gripper effort (#robots.gripper_force). Every command is then
    clamped against the previous one: each arm's end-effector speed to `max_ee_speed_m_s`,
    through its `kinematics`, and the gripper to one `max_gripper_step`; and, with
    `trace_path` set, appended to a CSV trace. The clamped command is not written here but
    submitted to the command path, which
    runs it out to the arms at `command_hz`. Until the first chunk lands nothing is
    at stake, so a warming pool is waited out at the pace DreamAPI asks for (up to #POOL_WARMING_LIMIT_S), and
    `first_chunk_timeout_s` bounds only a wait with no answer from DreamAPI; a first chunk that lands more than
    #MOVE_AFTER_WAIT_LIMIT_S after the first tick is not run, the run stops with #robots.errors.ModelReady before
    its first move, so the arms never start moving after a long wait; after it, `max_stall_ticks` bounds every
    empty-buffer burst. A serve fault that repeats word for word (#SERVE_FAULT_REPEATS) stops the run with
    DreamAPI's message, and an arm's own alarm (a motor too hot, a joint blocked too long: `FollowerArm.alarm()`)
    stops it with #robots.errors.SafetyStop. The most recent snapshots
    stay readable on `state`, `video` and `plan`; all are rebound to new objects every tick and never
    mutated, so a concurrent reader always sees a consistent snapshot without a lock.

    Attributes:
    - state: Latest #robots.types.RobotState snapshot, or None before the first tick.
    - video: Latest frame per camera, keyed by camera name; empty before the first tick.
    - tick_count: Ticks that published a `state` so far, so a reader polling slower than the tick can tell how many
      ticks passed between two snapshots it saw (the live agent's tick guard divides by it).
    - steps_done: Ticks that wrote a command to the arms so far: a run's steps, which `max_steps` bounds.
    - plan: Latest #PlanSnapshot, the rows still to run and the last #PLAN_HISTORY_S seconds of commanded
      ticks; empty before the first commanded tick.
    - tick: The latest #TickSnapshot, or None before the first commanded tick.
    - tick_history: The last #TICK_RING snapshots, oldest first, appended once per commanded tick; a reader takes
      the ones newer than the last it saw.
    - refill_wait_ticks: The `tick_count` of each recent tick that waited for a chunk (the sync path of
      `_wait_for_refill`), oldest first, at most #TICK_RING of them; the live agent's tick guard leaves every
      interval that spans such a tick out of its measurement, since a wait is the policy's latency and not the
      agent's load. A ring rather than the last one alone, because two ticks in a row can wait and
      a reader that sees the second must still learn of the first.
    - clamp_count: Ticks on which the rate clamp bounded a command.
    - max_ee_speed_requested_m_s: The largest end-effector speed either arm asked for on any tick of the run,
      before the clamp: how hard the policy pushes against `max_ee_speed_m_s`. 0 until a second command.
    - timing: #RefillTiming over the run so far; rebound, never mutated.
    - refill_error: The message of the last refill's failure while it stands (DreamAPI's refusal, `DA-2000 ...
      (pool_warming)` while a serve boots; a timeout), or None: set when a refill fails, cleared when DreamAPI
      answers one again, so a reader can say why the arms hold while the loop still retries.
    - refill_error_code: `errors.error_code()` of that failure (DreamAPI's reason, or the fault's class), set and
      cleared with `refill_error`, for a reader that words its own message.

    @see run(), step(), set_task()
    """

    def __init__(
        self,
        followers: dict[str, followers.Arm],
        cameras: list[cameras.CameraReader],
        client: Any,
        *,
        task: str,
        control_hz: float,
        action_horizon: int,
        action_timeout_s: float,
        refill_at_remaining: int | None = None,
        stale_after_s: float = 0.2,
        max_stall_ticks: int = 240,
        first_chunk_timeout_s: float = 120.0,
        ramp_s: float = 2.0,
        ramp_max_rad_s: float = 0.5,
        max_ee_speed_m_s: float,
        max_gripper_step: float = 0.05,
        grip_force_cap: float = gripper_force.DEFAULT_CAP,
        kinematics: dict[str, kinematics.Kinematics] | None = None,
        trace_path: str | None = None,
        command_hz: float = 250.0,
        command_interp_max_s: float = 0.10,
        command_thread: bool = True,
        sys0: sys0_motors.Models | None = None,
        camera_owner: bool = True,
        inference_log_path: str | None = None,
        run_number: int = 1,
        recorder: rollouts.Recorder | None = None,
        max_steps: int | None = None,
        wire: adapter.Wire = adapter.YAM_VIDEO_WIRE,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        """Binds hardware, client and timing; nothing starts until `async with`.

        @param followers The two arms keyed `"left"` and `"right"`; written only by the command path.
        @param cameras Camera readers; each is started in `__aenter__` and stopped in `__aexit__`.
        @param client A `dream.DreamClient` whose policy is open (`client.policy(config)`), or a stand-in with its
            calls: `set_observation(frames, observation, *, timestamp)`, `get_action(*, commanded)`,
            `reset_episode()`, `close()`, `session_id`, and `refill_due(executed, remaining)` unless
            `refill_at_remaining` pins the refill point. The loop closes it on the way out.
        @param task The language prompt sent with every observation.
        @param control_hz Tick rate.
        @param action_horizon Rows per action chunk; the checkpoint's fixed horizon.
        @param action_timeout_s A refill older than this is late: the tick stalls and its result is discarded.
        @param refill_at_remaining None (the default) asks the client when to refill (`client.refill_due`, the
            client's execution horizon and sync/async pacing, _refill_due()); an int pins it, starting the next
            refill once the buffer holds that many rows or fewer. Tests pin it to read exact rows; a station never
            does.
        @param stale_after_s Frame age past which a tick fails instead of feeding stale pixels to the policy.
        @param max_stall_ticks Consecutive empty-buffer ticks tolerated after the first chunk before the loop stops.
        @param first_chunk_timeout_s Seconds the first chunk may take after the first tick.
        @param ramp_s Seconds to blend from the measured pose into the first target after start and after a reset.
        @param ramp_max_rad_s Joint speed cap during the blend; a larger first delta stretches the ramp.
        @param max_ee_speed_m_s Cap on each arm's end-effector speed from one command to the next, metres per
            second; a faster command is scaled back along its joint delta and counted.
        @param max_gripper_step Most the gripper command may move per tick, in `[0, 1]` units.
        @param grip_force_cap Gripper effort a grasp is held at, in i2rt's `gripper_eff` units (motor torque,
            not newtons); the commanded close is backed off until the measured effort sits there
            (#robots.gripper_force). 0 turns the cap off and leaves grip force to the motor PD.
        @param kinematics Forward kinematics per arm, keyed `"left"` and `"right"` (#robots.kinematics.ArmKinematics
            on a station); None leaves the joints unclamped and the end-effector speeds NaN, for tests without a
            model.
        @param trace_path CSV file appended with one row per commanded tick (trace_columns()), or None.
        @param command_hz Rate the command path writes the arms at; 250 Hz by default.
        @param command_interp_max_s Longest run from one written command to the next submitted target, seconds.
        @param command_thread Run the command path on its own thread (production); False writes each tick's
            command synchronously inside `step()` and leaves later periods to the caller, for tests.
        @param sys0 System 0's models (`robots.sys0.motors.Models`, or `IN_ARM_PROCESSES` for arms served from
            their own processes), or None to command the rows through the motors' PD as the command writer does.
            With them the command path is #robots.sys0.motors.MotorPath: the writer during the start-up ramp, the
            trackers on the plan after it.
        @param camera_owner Start the readers in `__aenter__` and stop them on the way out (the one-shot
            `robots.run`); False takes readers a station opened and keeps for its lifetime, so they are read, never
            started or stopped, here.
        @param inference_log_path JSON Lines file appended with one record per inference, error, stall and run
            boundary (#robots.inference_log), or None.
        @param run_number The run's ordinal, stamped into every inference log record.
        @param recorder The #robots.rollouts.Recorder that saves the run as an episode on disk (one per prompt), or
            None to save nothing. Its calls only queue work for its own threads, so the tick does not wait on it.
        @param max_steps Commands written to the arms after which run() returns; None runs until it is cancelled.
        @param wire The body's #robots.adapter.Wire, whose `to_dream` builds everything the client is handed and
            whose `from_dream` reads every reply: the YAM's (`robots.adapter.YAM_VIDEO_WIRE`) by default, the config's
            body's otherwise (`robots.bodies`). Nothing else in the loop differs between the bodies. It is bound to
            the arms (`Wire.for_arms()`), so every command is as wide as the arm it goes to.
        @param clock Monotonic clock, seconds; injected so tests can drive ticks without real time. The
            command thread reads the same clock.
        @param sleep Awaitable sleep, seconds; injected for the same reason.
        """
        self.state: types.RobotState | None = None
        self.tick_count = 0
        self.steps_done = 0
        self._max_steps = max_steps
        self.video: dict[str, types.Frame] = {}
        self.tick: TickSnapshot | None = None
        self.tick_history: collections.deque[TickSnapshot] = collections.deque(maxlen=TICK_RING)
        self.refill_wait_ticks: collections.deque[int] = collections.deque(maxlen=TICK_RING)
        self._followers = followers
        self._cameras = cameras
        self._is_camera_owner = camera_owner
        self._client = client
        self._task = task
        ## Per arm, its joints (`robots.followers.FollowerArm.dof`): every row, ring and trace column is this wide.
        self._dofs = {arm: followers[arm].dof for arm in ARMS}
        bound = wire.for_arms(self._dofs)
        self._to_dream = bound.to_dream
        self._from_dream = bound.from_dream
        self._commanded_joints = bound.commanded
        self._period_s = 1.0 / control_hz
        self._horizon = action_horizon
        self._action_timeout_s = action_timeout_s
        self._refill_at_remaining = refill_at_remaining
        ## Rows the current chunk put in the buffer at install: with the buffer's length, how many of it the arms
        ## have executed (_refill_due()).
        self._installed_rows = 0
        self._stale_after_s = stale_after_s
        self._max_stall_ticks = max_stall_ticks
        self._first_chunk_timeout_s = first_chunk_timeout_s
        self._ramp_s = ramp_s
        self._ramp_max_rad_s = ramp_max_rad_s
        self._max_ee_speed_m_s = max_ee_speed_m_s
        self._max_gripper_step = max_gripper_step
        # One cap per arm, each holding its own back-off between ticks; None when the config turned the cap off,
        # which is the only way a commanded close reaches the arms unregulated.
        self._grip_caps: dict[str, gripper_force.GripperForceCap] | None = (
            None if grip_force_cap <= 0 else {arm: gripper_force.GripperForceCap(grip_force_cap) for arm in ARMS}
        )
        self._kinematics = kinematics
        self.max_ee_speed_requested_m_s = 0.0
        self.plan = PlanSnapshot.empty(max_ee_speed_m_s, self._dofs)
        self._chunk_plan: _ChunkPlan | None = None
        self._history = _History(max(1, math.ceil(PLAN_HISTORY_S * control_hz)), kinematics, self._dofs)
        self._trace_path = trace_path
        self._trace_file: Any = None
        self._trace: Any = None
        self._trace_rows = 0
        self._inference_log = inference_log.InferenceLog(inference_log_path, run_number)
        self._recorder = recorder
        self._episode_counts = _EpisodeCounts()
        # A prompt handed to set_task() mid-run waits here until no refill is in flight.
        self._pending_task: str | None = None
        ## Whether the next refill starts a new session first (the client's reset_episode(), which releases the old
        ## session on the gateway with one bounded request), on the refill's own thread, never on the tick.
        self._reset_before_refill = False
        ## The prompt the in-flight refill was sent with; set_task() may change `_task` before it lands.
        self._refill_prompt = task
        ## Per camera, the capture time of the encoded frame last handed to the client: a tick hands a new set once
        ## every camera has a newer one (_handover()).
        self._handed_s: dict[str, float] = {}
        ## The frames last handed to the client, by window key, and the timestamp they went with; None before the
        ## first. A refill tick without new frames hands them again with its own state.
        self._handed: tuple[dict[str, str], float] | None = None
        ## True from a refill's start until its thread has handed the client the refill tick's observation: no tick
        ## hands one meanwhile, so the client's timestamps never go back and the request carries that tick's state.
        self._is_refill_handing_over = False
        self.timing = RefillTiming()
        self.refill_error: str | None = None
        self.refill_error_code: str | None = None
        # Every collected round trip, kept sorted for the percentiles; one refill per chunk, so it stays small.
        self._rtts_ms: list[float] = []
        writer = _CommandWriter(
            followers,
            command_hz=command_hz,
            command_interp_max_s=command_interp_max_s,
            clock=clock,
        )
        ## System 0's motor path when it is on: it drives the arms after the ramp and lends them to `writer` before.
        self._sys0_path = (
            None
            if sys0 is None
            else sys0_motors.MotorPath(writer, followers, sys0, max_ee_speed_m_s=max_ee_speed_m_s, clock=clock)
        )
        self._commands: _CommandWriter | sys0_motors.MotorPath = writer if self._sys0_path is None else self._sys0_path
        self._has_command_thread = command_thread
        ## Rows commanded since the in-flight refill's observation: how far the arm actually got along the plan.
        self._rows_since_observation = 0
        self._last_action: types.Action | None = None
        self.clamp_count = 0
        # Clamps come in bursts like stale frames; the WARNING goes out once per burst.
        self._has_logged_clamp = False
        # Every grasp trips the force cap, so the INFO that says it is working goes out once a run.
        self._has_logged_grip_cap = False
        self._clock = clock
        self._sleep = sleep
        self._buffer: collections.deque[types.Action] = collections.deque()
        # At most one get_action in flight: dream raises on a concurrent call.
        self._refill: asyncio.Task[Any] | None = None
        self._refill_started_s = 0.0
        self._refill_observed_at_s = 0.0
        # DreamAPI's retry hint after a retryable refusal: no refill starts before it.
        self._retry_not_before_s = 0.0
        self._first_tick_s: float | None = None
        self._first_chunk_from_s: float | None = None
        self._serve_fault = ""
        self._serve_fault_count = 0
        self._failed_refills = 0
        self._failing_since_s = 0.0
        self._failure_logged: tuple[str, float] = ("", -math.inf)
        self._stall_ticks = 0
        ## Every stall tick over the run, unlike `_stall_ticks`, which is the current burst.
        self._stall_count = 0
        # One inference log record per stall burst, like the WARNING for stale frames.
        self._has_logged_stall = False
        # Stale frames arrive in bursts; one log line per burst keeps the log readable.
        self._has_logged_stale = False
        ## Chunks collected so far; the first one is the loop's start of motion and is logged as such.
        self._chunk_count = 0
        ## Set by a tick that blocked waiting for a refill, so run() reports the wait instead of an overrun.
        self._waited_for_refill = False
        # The origin is measured on the first commanded tick, not here: the arms may move
        # between construction and the first chunk.
        self._is_ramp_due = True
        self._ramp: _Ramp | None = None

    async def __aenter__(self) -> "ControlLoop":
        """Opens the trace file, starts the camera threads, waits for every camera's first frame, then starts
        the command thread.

        @return The loop itself.
        @throws CameraError A camera thread died or published nothing within #FIRST_FRAME_TIMEOUT_S.
        @throws OSError The trace or the inference log file cannot be opened; nothing else has started yet.
        @throws RolloutError The recorder cannot start (no space, no folder); nothing else has started yet.
        """
        self._inference_log.open()
        try:
            self._begin_episode(self._task)
        except BaseException:
            self._end_episode("start_failed")
            self._inference_log.close()
            raise
        self._inference_log.write(
            "run_start",
            prompt=self._task,
            control_hz=1.0 / self._period_s,
            action_horizon=self._horizon,
            refill=self._refill_policy(),
            sys0=self._sys0_path is not None,
        )
        try:
            self._open_trace()
        except OSError:
            self._end_episode("start_failed")
            self._inference_log.close()
            raise
        if self._is_camera_owner:
            for reader in self._cameras:
                reader.start()
        try:
            await self._wait_for_first_frames()
        except BaseException:
            # The block was never entered, so __aexit__ will not run: nothing may be left
            # capturing or open. The dream client is the caller's to close.
            self._stop_cameras()
            self._close_trace()
            self._end_episode("start_failed")
            self._inference_log.close()
            raise
        if self._has_command_thread:
            self._commands.start()
        _log.info(
            "loop ready: %d cameras publishing, %.0f Hz ticks, %.0f Hz commands%s, %d-row chunks, "
            "refill %s, %s, task %r",
            len(self._cameras),
            1.0 / self._period_s,
            1.0 / self._commands.period_s,
            "" if self._sys0_path is None else " (System 0 drives the motors after the ramp)",
            self._horizon,
            "paced by the client"
            if self._refill_at_remaining is None
            else f"at {self._refill_at_remaining} remaining",
            f"end effector capped at {self._max_ee_speed_m_s:g} m/s"
            if self._kinematics is not None
            else "no kinematics so no end-effector speed cap",
            self._task,
        )
        return self

    async def _wait_for_first_frames(self) -> None:
        """Polls every camera's slot until each has published a frame.

        @throws CameraError A camera thread died or published nothing within #FIRST_FRAME_TIMEOUT_S.
        """
        deadline_s = self._clock() + FIRST_FRAME_TIMEOUT_S
        while True:
            # One pass over the slots: a frame landing between two reads must not turn the
            # message into "cameras [] published no first frame".
            waiting = [reader.name for reader in self._cameras if reader.frame is None]
            if not waiting:
                return
            dead = [reader.name for reader in self._cameras if not reader.is_alive()]
            timed_out = self._clock() > deadline_s
            if dead or timed_out:
                raise errors.CameraError(
                    f"cameras {waiting} published no first frame within {FIRST_FRAME_TIMEOUT_S} s"
                    + (f"; dead threads: {dead}" if dead else "")
                    + "; check the cables; dreamcli setup shows the cameras"
                )
            await self._sleep(_FIRST_FRAME_POLL_S)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Stops the command thread and the cameras, leaves the arms safe and closes the dream client.

        @param exc_type The exception type that ended the `async with` block, or None on a clean exit.
        @param exc The exception instance, or None.
        @param tb Its traceback, or None.
        """
        self._stop_cameras()
        # Before the hold or the disengage: nothing may write the arms after they are left safe.
        run_teardown("stop command thread", self._commands.stop)
        is_arm_fault = exc_type is not None and issubclass(exc_type, errors.ArmError)
        if exc_type is None or issubclass(exc_type, asyncio.CancelledError):
            _log.info("loop stopped after %d chunks; holding the arms", self._chunk_count)
            reason = "stopped" if exc_type is None else "cancelled"
        else:
            _log.error("loop stopped by %s after %d chunks: %s", exc_type.__name__, self._chunk_count, exc)
            reason = exc_type.__name__
            if exc is not None:
                self._inference_log.write("error", **inference_log.error_fields(exc, "stop"))
        self._inference_log.write(
            "run_stop",
            reason=reason,
            chunks=self._chunk_count,
            stalls=self._stall_count,
            clamps=self.clamp_count,
            session_id=self.session_id,
        )
        self._end_episode(reason, None if exc is None or reason == "cancelled" else str(exc))
        for name, arm in self._followers.items():
            if is_arm_fault and not arm.drops_when_limp:
                run_teardown(f"disengage {name} arm", arm.disengage)
            elif arm.is_alive():
                run_teardown(f"hold {name} arm", arm.hold)
            else:
                _log.error("%s arm does not answer; its motors keep their last target until it is switched off", name)
        if self._refill is not None:
            # Nothing can cancel the worker thread; claim its outcome so asyncio
            # does not warn about an unretrieved exception when the task is collected. A task
            # that already finished runs the callback immediately, which is exactly the case
            # where an uncollected failure would otherwise go unretrieved.
            self._refill.add_done_callback(_discard_refill)
        run_teardown("close dream client", self._client.close)
        run_teardown("close trace", self._close_trace)
        run_teardown("close inference log", self._inference_log.close)

    @property
    def chunk_count(self) -> int:
        """Chunks installed since the run started; the first one is the start of motion."""
        return self._chunk_count

    @property
    def stall_count(self) -> int:
        """Ticks on which the buffer was empty after the first chunk, over the whole run."""
        return self._stall_count

    @property
    def task(self) -> str:
        """The prompt refills are sent with; a set_task() not yet applied does not show here."""
        return self._task

    def set_task(self, task: str) -> None:
        """Changes the prompt and starts a fresh episode at the next safe point.

        The change waits until no refill is in flight (the client refuses a reset during a
        request): the next tick then drops the buffered rows (they answer the old task) and makes
        the ramp due again, and the refill it starts calls the client's `reset_episode()` on its
        own thread before it asks with the new prompt. A chunk that lands for the old prompt in the
        meantime is dropped too. Safe to call from any task on the loop's event loop, before or
        after `async with`; a second call before the first is applied replaces it. It is also how
        a new episode with the same prompt is asked for: the loop owns the client's episodes.

        @param task The new language prompt.
        """
        self._pending_task = task
        _log.info("task change queued: %r; applied once no refill is in flight", task)

    @property
    def dream_client(self) -> Any:
        """The run's dream client, which the live view reads the model's round trips and session from."""
        return self._client

    @property
    def session_id(self) -> str | None:
        """The dream client's current DreamAPI session id (`client.session_id`), or None before it opened one."""
        session_id: str | None = self._client.session_id
        return session_id

    def discard(self) -> None:
        """Releases a loop that was built but never entered: the readers' devices and the dream client.

        For a caller whose start was cancelled between the build and `async with`; nothing was
        started, so there is no command thread to stop and the arms are not touched. Safe after
        `__aexit__` too (every step is idempotent), but pointless.
        """
        self._stop_cameras()
        run_teardown("close dream client", self._client.close)

    async def run(self) -> None:
        """Ticks at `control_hz` until an exception, a cancellation or the last of `max_steps` stops it.

        A run that takes its `max_steps` commands returns with the arms holding the last one.

        Deadlines come from the monotonic clock, so jitter does not accumulate lag; an overrun
        is logged with the measured tick duration and the schedule restarts from now.

        @throws ControlLoopError No chunk landed within `first_chunk_timeout_s`, or the buffer stayed empty
            past `max_stall_ticks` after one had.
        @throws ModelReady The first chunk landed more than #MOVE_AFTER_WAIT_LIMIT_S after the first tick; nothing
            moved.
        @throws CameraError A camera thread died.
        @throws ArmError An arm read or write failed.
        """
        deadline_s = self._clock()
        while True:
            tick_started_s = deadline_s
            await self.step(tick_started_s)
            if self._max_steps is not None and self.steps_done >= self._max_steps:
                _log.info("run took its %d steps; ending it", self._max_steps)
                return
            deadline_s += self._period_s
            now_s = self._clock()
            if now_s > deadline_s and self._waited_for_refill:
                # The tick blocked on purpose (an empty buffer waits for its refill); not an overrun.
                deadline_s = now_s
            elif now_s > deadline_s:
                _log.warning(
                    "tick overrun: %.1f ms against a %.1f ms period; rescheduling from now",
                    (now_s - tick_started_s) * 1e3,
                    self._period_s * 1e3,
                )
                deadline_s = now_s
            # A zero-length sleep after an overrun still yields, so consumers of state/video get
            # a turn even when every tick runs long.
            await self._sleep(deadline_s - now_s)

    async def step(self, now_s: float) -> None:
        """Runs one tick: observe, publish, refill if due, command the arms.

        A stale frame or an empty buffer fails the tick softly: the arms hold and the next tick
        tries again. Everything else raises and stops the loop.

        @param now_s Monotonic tick deadline, seconds.
        @throws ControlLoopError No chunk landed within `first_chunk_timeout_s`, or the buffer stayed empty
            past `max_stall_ticks` after one had.
        @throws ModelReady The first chunk landed more than #MOVE_AFTER_WAIT_LIMIT_S after the first tick; nothing
            moved.
        @throws CameraError A camera thread died.
        @throws ArmError An arm read failed, or a write failed on the command thread since the last tick.
        """
        if self._commands.error is not None:
            # The command thread is the one place a write can fail off the tick; it stops the loop
            # from here, as an in-tick write failure did, so __aexit__ disengages the arms.
            raise self._commands.error
        self._waited_for_refill = False
        if self._first_tick_s is None:
            self._first_tick_s = self._first_chunk_from_s = now_s
        left = self._followers["left"].read()
        right = self._followers["right"].read()
        for arm in self._followers.values():
            alarm = arm.alarm()
            if alarm is not None:
                raise errors.SafetyStop(alarm)
        frames = self._read_frames(now_s)
        if frames is None:
            self._hold()
            return
        self._has_logged_stale = False
        state = types.RobotState(left=left, right=right, timestamp_s=now_s)
        observation = types.Observation(state=state, images=frames, timestamp_s=now_s)
        self.state = state
        self.video = frames
        self.tick_count += 1
        self._maybe_refill(observation, now_s)
        if not self._buffer:
            await self._wait_for_refill(now_s)
        if not self._buffer:
            self._stall(now_s)
            return
        row = self._buffer.popleft()
        self._rows_since_observation += 1
        ramped = self._ramp_toward(row, state)
        capped = self._cap_grip_force(ramped, state, now_s)
        clamp = self._clamp(capped)
        if self._sys0_path is None:
            self._commands.submit(clamp.action, now_s)
        else:
            self._sys0_path.submit(clamp.action, now_s, self._tracker_plans(clamp.action, now_s))
        if not self._has_command_thread:
            self._commands.once(now_s)
        self.steps_done += 1
        self._last_action = clamp.action
        self._stall_ticks = 0
        self._has_logged_stall = False
        self._trace_tick(now_s, state, clamp)
        self._history.append(now_s, state, clamp.action, clamp.clamped)
        self.plan = self._snapshot_plan(now_s)
        if self._recorder is not None:
            self._recorder.tick(
                now_s, self._chunk_count, self._horizon - len(self._buffer) - 1, state, clamp.action, frames
            )
        # The buffer is always the tail of the installed chunk, so its length says which row just went out.
        snapshot = TickSnapshot(
            tick=self.tick_count,
            t_s=now_s,
            state=state,
            action=clamp.action,
            chunk=self._chunk_count,
            row=self._horizon - len(self._buffer) - 1,
            clamped=clamp.clamped,
            session_id=self.session_id,
        )
        self.tick = snapshot
        self.tick_history.append(snapshot)

    def _read_frames(self, now_s: float) -> dict[str, types.Frame] | None:
        """Collects the latest frame per camera; None when any frame is stale.

        @param now_s Monotonic tick deadline, seconds; a frame older than `stale_after_s` at this time is stale.
        @return The frames keyed by camera name, or None when one is missing or stale (the arms hold).
        @throws CameraError A camera thread has died: stale video is the unsafe case.
        """
        frames: dict[str, types.Frame] = {}
        for reader in self._cameras:
            if not reader.is_alive():
                raise errors.CameraError(
                    f"camera '{reader.name}' stopped sending pictures ({reader.error}); check its cable; "
                    "dreamcli setup shows the cameras"
                )
            # One slot read per camera: the frame judged for staleness is the frame forwarded.
            frame = reader.frame
            if frame is None or now_s - frame.timestamp_s > self._stale_after_s:
                self._log_stale_once(reader.name, frame, now_s)
                return None
            frames[reader.name] = frame
        return frames

    def _log_stale_once(self, name: str, frame: types.Frame | None, now_s: float) -> None:
        """Logs the first stale frame of a burst and nothing for the rest of it.

        @param name The camera whose frame is stale.
        @param frame That frame, or None when the camera has published nothing yet.
        @param now_s Monotonic tick deadline, seconds, for the age in the message.
        """
        if self._has_logged_stale:
            return
        self._has_logged_stale = True
        age = "missing" if frame is None else f"{now_s - frame.timestamp_s:.3f} s old"
        _log.warning(
            "camera '%s' frame is %s, older than stale_after_s=%.3f; holding until frames are fresh",
            name,
            age,
            self._stale_after_s,
        )

    def _maybe_refill(self, observation: types.Observation, now_s: float) -> None:
        """Collects a finished refill, hands the client this tick, then starts a refill when one is due.

        A refill is due when the buffer has drained to the refill point (_refill_due()).

        A refill is not started while one is in flight (dream refuses concurrent calls) or while a
        retryable refusal's `retry_after_s` has not elapsed. A tick that starts none hands the client its frames and
        state when every camera has a new encoded frame (_handover()); a refill's thread hands over its own tick.

        @param observation This tick's observation; handed to the client, and when a refill starts, asked about with
            its arms' last commanded joints beside it (the wire's `commanded`, `robots.adapter.commanded_joints`,
            never sent).
        @param now_s Monotonic tick deadline, seconds; stamps the refill's start.
        """
        self._collect_refill(now_s)
        if self._refill is None and self._pending_task is not None:
            self._apply_pending_task()
        starts = self._refill is None and self._refill_due() and now_s >= self._retry_not_before_s
        if not starts:
            if not self._is_refill_handing_over:
                tick = self._handover(observation, is_refill=False)
                if tick is not None:
                    frames, state, timestamp_s = tick
                    self._client.set_observation(frames, state, timestamp=timestamp_s)
            return
        handover = self._handover(observation, is_refill=True)
        assert handover is not None, "a refill tick always hands over"
        # The joints each arm was last commanded, the row before this tick's: the client anchors the new chunk there.
        commanded = self._commanded_joints(self._last_action)
        reset, self._reset_before_refill = self._reset_before_refill, False
        self._is_refill_handing_over = True
        self._refill = asyncio.create_task(asyncio.to_thread(self._ask, handover, commanded, reset))
        self._refill_prompt = self._task
        self._refill_started_s = now_s
        self._refill_observed_at_s = observation.timestamp_s
        self._rows_since_observation = 0
        _log.debug("refill started with %d actions buffered", len(self._buffer))

    def _handover(
        self, observation: types.Observation, *, is_refill: bool
    ) -> tuple[dict[str, str], dict[str, Any], float] | None:
        """What this tick hands the client's `set_observation()`, or None when it hands nothing.

        The client keeps one set of frames every `1 / obs_fps` seconds of the timestamps it is handed and replaces
        its latest state on every call. The cameras encode one capture per slot of that period on one grid
        (`robots.cameras.SampleClock`), so a tick hands a new set once every camera has a frame newer than the one
        it last handed, timed by the newest capture among them; on the ticks in between nothing is handed, so the
        frame a slot encoded reaches the client once and the client's spacing of the window follows the slots. A
        refill tick hands its state in any case, with the frames last handed and their timestamp when no new set is
        complete, so the request carries the newest state.

        @param observation This tick's observation.
        @param is_refill Whether a refill starts on this tick.
        @return `(frames, observation, timestamp_s)` for `set_observation()`; None on a tick that hands nothing.
        @throws ControlLoopError A camera's frame carries no encoded frame, or a camera is not a role of the wire.
        """
        times = {name: frame.jpeg_timestamp_s for name, frame in observation.images.items()}
        is_new = self._handed is None or all(
            time_s > self._handed_s.get(name, -math.inf) for name, time_s in times.items()
        )
        if not (is_new or is_refill):
            return None
        frames, state = self._to_dream(observation, self._task)
        if is_new:
            self._handed = (frames, max(times.values(), default=observation.timestamp_s))
            self._handed_s = times
        assert self._handed is not None
        handed_frames, timestamp_s = self._handed
        return handed_frames, state, timestamp_s

    def _ask(
        self,
        handover: tuple[dict[str, str], dict[str, Any], float],
        commanded: dict[str, npt.NDArray[np.float64]],
        reset: bool,
    ) -> dict[str, Any]:
        """One refill, on its own thread: the reset when one is due, the refill tick's observation, then the request.

        The reset (the client's reset_episode()) releases the old session on the gateway with one bounded request
        and empties the client's window; the observation is then the new episode's first, and the request goes out
        with no id, so the gateway mints a fresh session with a reset. Ticks hand the client nothing until the
        observation is in (`_is_refill_handing_over`), so it lands before any later tick's.

        @param handover The refill tick's `(frames, observation, timestamp_s)` (_handover()).
        @param commanded The arms' last commanded joints per chunk key (`robots.adapter.commanded_joints`).
        @param reset Whether a new session starts first (_drop_session()).
        @return The action chunk, as get_action() returns it.
        @throws dream.DreamError From any of the three calls; a failed reset fails this refill, and the next one
            resets again.
        """
        try:
            if reset:
                self._client.reset_episode()
            frames, state, timestamp_s = handover
            self._client.set_observation(frames, state, timestamp=timestamp_s)
        finally:
            self._is_refill_handing_over = False
        chunk: dict[str, Any] = self._client.get_action(commanded=commanded)
        return chunk

    def _refill_due(self) -> bool:
        """Whether the buffer has drained to the refill point.

        Pinned (`refill_at_remaining` an int): due once the buffer holds that many rows or fewer. Otherwise the
        client decides (`client.refill_due`, `dream.pacing`): it measures every round trip, knows the execution
        horizon, and paces async (the request goes out so its reply lands as the arms finish the execution horizon)
        or sync (only once the buffer is empty, the arms holding through the inference) by that round trip. The loop
        tells it how many rows of the current chunk the arms executed since the install and how many are left.

        @return True when a refill should start, given none is in flight.
        """
        remaining = len(self._buffer)
        if self._refill_at_remaining is not None:
            return remaining <= self._refill_at_remaining
        return bool(self._client.refill_due(self._installed_rows - remaining, remaining))

    def _refill_policy(self) -> str | int:
        """The refill point as the inference log's `run_start` record carries it: `"client"`, or the pinned rows."""
        return "client" if self._refill_at_remaining is None else self._refill_at_remaining

    def _drop_session(self) -> None:
        """Makes the next refill start a new session: reset_episode() runs first on that refill's thread.

        The reset releases the old session on the gateway with one bounded request, so it runs beside the
        request it precedes and never on the tick (_maybe_refill(), _ask()).
        """
        self._reset_before_refill = True

    def _apply_pending_task(self) -> None:
        """Makes the queued prompt the task and starts a fresh episode; called only with no refill in flight.

        Mirrors the session-expired path: the buffered rows answer the old task and are dropped,
        and the new episode's first targets are near the home pose again, so a ramp carries
        the arms into them.

        @throws DreamError The client is closed (it cannot be mid-request here).
        """
        task = self._pending_task
        assert task is not None
        self._pending_task = None
        previous, self._task = self._task, task
        if self._recorder is not None:
            self._end_episode("task_changed")
            try:
                self._begin_episode(task)
            except (OSError, rollout_episode.RolloutError):
                _log.exception("rollouts: the new prompt's episode cannot be recorded; the run goes on without it")
            self._inference_log.write("episode_start", prompt=task)
        self._drop_session()
        self._buffer.clear()
        self._is_ramp_due = True
        self._retry_not_before_s = 0.0
        _log.info("task changed from %r to %r: episode reset, buffer dropped, ramp due", previous, task)

    def _collect_refill(self, now_s: float) -> None:
        """Moves a finished refill into the buffer, or drops it when it failed or arrived late.

        @param now_s Monotonic time the collection happens at, seconds; measures the refill's latency and the rows
            due since its observation.
        @throws DreamAuthError The credential was refused; a retry cannot make it valid.
        @throws DreamConfigError The checkpoint or the request is wrong; the operator must fix the config.
        @throws DreamObservationError The adapter and the embodiment disagree; a bug, not a transient.
        """
        task = self._refill
        if task is None or not task.done():
            return
        self._refill = None
        try:
            reply = task.result()
            chunk = self._from_dream(reply, self._refill_observed_at_s, self._horizon)
        except (dream.DreamAuthError, dream.DreamConfigError, dream.DreamObservationError):
            # Not retryable: every retry would fail the same way, so the loop stops and the
            # operator fixes the key, the config or the code; __aexit__ logs it.
            raise
        except dream.DreamSessionExpired as exc:
            # The buffered rows answer a session the gateway has already replaced; the client
            # has adopted the new one, so the next refill starts the episode over.
            _log.warning("episode restarted server-side, dropping the buffered chunk and refilling: %s", exc)
            self._inference_log.write("error", **inference_log.error_fields(exc, "reset"))
            self.refill_error = str(exc)
            self.refill_error_code = errors.error_code(exc)
            self._buffer.clear()
            # The new episode's first targets are near its home pose again, wherever the arms are
            # now: a ramp carries the arms into them.
            self._is_ramp_due = True
            return
        except (dream.DreamError, errors.ControlLoopError) as exc:
            if isinstance(exc, dream.DreamServerError) and not exc.retryable:
                raise errors.ControlLoopError(
                    f"{exc}. DreamAPI marks this refusal not retryable, so the run stops{_refusal_help(exc.reason)}",
                    reason=errors.error_code(exc),
                ) from exc
            # The one narrow except: a policy fault costs one refill, not the
            # loop. The buffer keeps draining and a later tick retries, after the hold _defer_retry
            # sets for a fault that would repeat at once. ControlLoopError here is from_dream's
            # shape check, the same fault strict mode reports as DreamActionError, so strict on
            # and off behave alike.
            self._inference_log.write("error", **inference_log.error_fields(exc, "retry"))
            self.refill_error = str(exc)
            # from_dream's shape check is the fault strict mode raises as DreamActionError; one code for both.
            self.refill_error_code = (
                "DreamActionError" if isinstance(exc, errors.ControlLoopError) else errors.error_code(exc)
            )
            self._count_serve_fault(exc)
            if self._chunk_count == 0 and self.refill_error_code == "pool_warming":
                self._first_chunk_from_s = now_s
            self._defer_retry(exc, now_s)
            # A retry after a fault is a new session: the client releases the old one on the gateway (its seat
            # frees now, not on idle eviction) and the retry carries no id, so the gateway mints a fresh one with
            # a reset rather than resuming a session whose serve may be the one that just failed. A refusal that
            # says the session is fine keeps it (_starts_new_session()).
            if _starts_new_session(exc):
                self._drop_session()
            return
        assert self._first_tick_s is not None, "step() sets it before the first refill"
        waited_s = now_s - self._first_tick_s
        if self._chunk_count == 0 and waited_s > MOVE_AFTER_WAIT_LIMIT_S:
            raise errors.ModelReady(
                f"the policy is ready: it first answered {waited_s:.0f} s after the run started, and a run never "
                "starts moving on its own after a wait that long, so the arms did not move; start the run again, and "
                "it moves at once"
            )
        if self._recorder is not None:
            self._recorder.reply(reply, self._refill_observed_at_s)
        self.refill_error = None
        self.refill_error_code = None
        self._serve_fault, self._serve_fault_count = "", 0
        self._failed_refills = 0
        elapsed_s = now_s - self._refill_started_s
        self._record_timing(elapsed_s)
        if elapsed_s > self._action_timeout_s:
            _log.warning(
                "discarding action chunk that took %.3f s, later than action_timeout_s=%.3f",
                elapsed_s,
                self._action_timeout_s,
            )
            self._log_inference(chunk, elapsed_s, due=None, skipped=None, discarded="late")
            return
        self._install_chunk(chunk, now_s, elapsed_s)

    def _record_timing(self, elapsed_s: float) -> None:
        """Rebinds `timing` with one more round trip and the client's `last_timing`, when it has one.

        @param elapsed_s Seconds the refill took on the loop's clock.
        """
        rtt_ms = elapsed_s * 1e3
        bisect.insort(self._rtts_ms, rtt_ms)
        client = inference_log.client_timing(self._client)
        self.timing = RefillTiming(
            count=len(self._rtts_ms),
            last_rtt_ms=rtt_ms,
            p50_ms=_percentile(self._rtts_ms, 0.50),
            p95_ms=_percentile(self._rtts_ms, 0.95),
            serve_infer_ms=client["serve_infer_ms"],
            gateway_ms=client["gateway_ms"],
            network_ms=client["network_ms"],
            request_bytes=client["request_bytes"],
            response_bytes=client["response_bytes"],
            up_mbps=client["up_mbps"],
        )

    def _log_inference(
        self,
        chunk: types.ActionChunk,
        elapsed_s: float,
        *,
        due: int | None,
        skipped: int | None,
        discarded: str | None,
    ) -> None:
        """Writes the `inference` record for one chunk that came back, installed or not.

        @param chunk The chunk `from_dream` built.
        @param elapsed_s Seconds the refill took on the loop's clock, the record's `rtt_ms`.
        @param due Rows due by wall time when it was collected; None when it was discarded as late.
        @param skipped Rows skipped on install; None when it was not installed.
        @param discarded Why it was not installed (`"late"`, `"horizon_past"`), or None when it was.
        """
        temps = None if self.state is None else _motor_temps(self.state)
        self._episode_counts.note_temps(temps)
        self._inference_log.write(
            "inference",
            prompt=self._refill_prompt,
            session_id=self.session_id,
            chunk=self._chunk_count if discarded is None else None,
            rtt_ms=round(elapsed_s * 1e3, 3),
            **inference_log.client_timing(self._client),
            rows=len(chunk.actions),
            mode=getattr(self._client, "mode", None),
            executed_since_obs=self._rows_since_observation,
            due=due,
            skipped=skipped,
            discarded=discarded,
            motor_temps_c=temps,
        )

    def _count_serve_fault(self, exc: Exception) -> None:
        if not (isinstance(exc, dream.DreamServerError) and exc.code == "DA-5000" and exc.reason == "error"):
            self._serve_fault, self._serve_fault_count = "", 0
            return
        message = str(exc)
        self._serve_fault_count = self._serve_fault_count + 1 if message == self._serve_fault else 1
        self._serve_fault = message
        if self._serve_fault_count >= SERVE_FAULT_REPEATS:
            raise errors.ControlLoopError(
                f"the policy server failed the request {self._serve_fault_count} times in a row with the same error, "
                f"so asking again will not help; the run stops. Fix what it names, then start again. {message}",
                reason=errors.error_code(exc),
            ) from exc

    def _defer_retry(self, exc: Exception, now_s: float) -> None:
        """Logs a failed refill; one that would come back at once also holds the next refill (#_retry_after_s).

        @param exc The exception the refill raised.
        @param now_s Monotonic tick deadline, seconds; the retry hint is measured from it.
        """
        retry_after_s = _retry_after_s(exc)
        if retry_after_s is not None:
            self._retry_not_before_s = now_s + retry_after_s
        first = self._failed_refills == 0
        if first:
            self._failing_since_s = now_s
        self._failed_refills += 1
        message = str(exc)
        warming = self.refill_error_code == "pool_warming"
        logged, logged_at_s = self._failure_logged
        if not (first or warming) and message == logged and now_s - logged_at_s < RETRY_LOG_EVERY_S:
            return
        self._failure_logged = (message, now_s)
        waiting = ""
        if warming and self._chunk_count == 0 and self._first_tick_s is not None:
            waiting = (
                f"the policy is still starting on DreamAPI ({now_s - self._first_tick_s:.0f} s so far; a cold one can "
                "take 15 minutes) and the arms wait where they are; past "
                f"{MOVE_AFTER_WAIT_LIMIT_S:.0f} s the run stops without moving once it answers. "
            )
        refused = isinstance(exc, (dream.DreamServerError, dream.DreamActionError, errors.ControlLoopError))
        so_far = ""
        if self._failed_refills > 1 and not warming:
            so_far = f" {self._failed_refills} times in a row over {now_s - self._failing_since_s:.0f} s"
        when = "on the next due tick" if retry_after_s is None else f"after {retry_after_s:.1f} s"
        _log.warning(
            "%sget_action %s%s, retrying %s: %s", waiting, "refused" if refused else "failed", so_far, when, exc
        )

    def _install_chunk(self, chunk: types.ActionChunk, now_s: float, elapsed_s: float) -> None:
        """Replaces the buffer with the rows of `chunk` the arm has not yet executed.

        Row `k` is the target for `observed_at_s + k * period`, so the rows due by wall time are
        past; but the arm only advances along the plan while rows are being commanded, and it
        holds while the buffer is empty. So the rows to skip are the ones actually commanded
        since the observation, never more than wall time says are due: after a hold the plan
        resumes where the arm stopped instead of jumping ahead to where it would have been.
        Installing every row instead rewound the target by the inference latency on each
        replacement, a sawtooth that on a YAM drove a current spike through the PD loop. A chunk is
        discarded only when the arm has executed every one of its rows' worth since the observation;
        a chunk that took longer than its horizon while the arms held (the client's sync pacing on a
        slow link) still has all of its rows ahead of the arm, and installs whole.

        @param chunk The chunk `from_dream` built for the observation the refill sent.
        @param now_s Monotonic time the chunk is being installed at, seconds.
        @param elapsed_s Seconds the refill took, for the log line.
        """
        due = round((now_s - chunk.observed_at_s) / self._period_s)
        skipped = min(self._rows_since_observation, due)
        if skipped >= len(chunk.actions):
            _log.warning(
                "discarding action chunk that landed %.0f ms after its observation: "
                "its whole %d-row horizon (%.0f ms) is already past; refilling on the next due tick",
                (now_s - chunk.observed_at_s) * 1e3,
                len(chunk.actions),
                len(chunk.actions) * self._period_s * 1e3,
            )
            self._log_inference(chunk, elapsed_s, due=due, skipped=None, discarded="horizon_past")
            return
        # Overlay, not append. The refill was requested with the observation of row p of the old
        # chunk, and `skipped` rows of the old chunk went out while the answer was computed, so the
        # arm is now at old row p + skipped. The new chunk's row 0 is the target for the observation
        # instant (old row p), which makes its row `skipped` the target for this very tick. The old
        # chunk's remaining tail and the new chunk's head therefore cover the same future ticks; the
        # new one is the fresher prediction, so it replaces the tail from here on.
        self._buffer = collections.deque(chunk.actions[skipped:])
        self._installed_rows = len(self._buffer)
        self._chunk_plan = _plan_chunk(chunk, skipped, self._period_s, self._kinematics, self._last_action)
        self._chunk_count += 1
        _log.info(
            "chunk %d: %d actions, %d executed since its observation, %d due by wall time, after %.0f ms, %s%s",
            self._chunk_count,
            len(chunk.actions),
            self._rows_since_observation,
            due,
            elapsed_s * 1e3,
            getattr(self._client, "mode", "unpaced"),
            " (first chunk; motion starts)" if self._chunk_count == 1 else "",
        )
        self._log_inference(chunk, elapsed_s, due=due, skipped=skipped, discarded=None)

    async def _wait_for_refill(self, now_s: float) -> None:
        """Sync path: waits for the pending refill until its deadline, then collects it as of the tick.

        The arms hold through the wait and no rows are executed, so the chunk is installed by the
        executed-row rule from where the arm stopped.

        @param now_s Monotonic tick deadline, seconds; the wait ends at `refill start + action_timeout_s`.
        """
        if self._refill is None:
            return
        timeout_s = max(0.0, self._refill_started_s + self._action_timeout_s - now_s)
        self._waited_for_refill = True
        self.refill_wait_ticks.append(self.tick_count)
        waited_from_s = self._clock()
        await asyncio.wait({self._refill}, timeout=timeout_s)
        # The arms held through the wait, so the rows due during it are past when the chunk
        # is installed; the tick's own time would install them as if no time had passed.
        self._collect_refill(now_s + self._clock() - waited_from_s)

    def _ramp_toward(self, target: types.Action, state: types.RobotState) -> types.Action:
        """Blends the target from the measured pose while a ramp runs; starts one when due.

        @param target The action popped from the buffer this tick.
        @param state This tick's measured arm state, the origin of a ramp that starts now.
        @return The command to send: the blend, or `target` itself once the ramp has ended.
        """
        if self._is_ramp_due:
            self._ramp = self._start_ramp(target, state)
            self._is_ramp_due = False
            # A ramp is the start of a run or of a fresh episode; a back-off carried across it would open the
            # gripper for a squeeze that is over (#robots.gripper_force.GripperForceCap.reset).
            for cap in (self._grip_caps or {}).values():
                cap.reset()
        if self._ramp is None:
            return target
        self._ramp.tick += 1
        alpha = self._ramp.tick / self._ramp.ticks
        if alpha >= 1.0:
            self._ramp = None
            return target
        return types.Action(
            left=_blend(self._ramp.origin.left, target.left, alpha),
            right=_blend(self._ramp.origin.right, target.right, alpha),
        )

    def _cap_grip_force(self, action: types.Action, state: types.RobotState, now_s: float) -> types.Action:
        """Holds each arm's commanded close to `grip_force_cap` on its measured gripper effort.

        The cap runs here, on the tick, rather than on the command path: the measured effort it regulates on is
        read once a tick with the rest of the arm state, and the command path never reads the arms. It runs
        before the clamp so `max_gripper_step` stays the last word on how fast the gripper may move, the
        back-off included.

        @param action The command after the ramp.
        @param state This tick's measured arm state, carrying each gripper's position and effort.
        @param now_s The tick time, seconds.
        @return The command with each gripper backed off as far as its cap asks; `action` itself when the cap
            is off or no driver reports an effort.
        """
        if self._grip_caps is None:
            return action
        wanted = _by_arm(action)
        measured = _by_arm_state(state)
        grippers = {
            arm: self._grip_caps[arm].update(
                wanted[arm].gripper, measured[arm].gripper, measured[arm].gripper_effort, now_s
            )
            for arm in ARMS
        }
        if any(grippers[arm] > wanted[arm].gripper for arm in ARMS):
            self._log_grip_cap_once()
        return types.Action(
            left=types.ArmCommand(positions_rad=action.left.positions_rad, gripper=grippers["left"]),
            right=types.ArmCommand(positions_rad=action.right.positions_rad, gripper=grippers["right"]),
        )

    def _log_grip_cap_once(self) -> None:
        """Logs the first grasp of the run the force cap backed off, and nothing for the rest.

        Every grasp of a run trips the cap, so this says once that the regulator is doing its job; what it did
        on any given tick is the trace's commanded gripper column.
        """
        if self._has_logged_grip_cap:
            return
        self._has_logged_grip_cap = True
        _log.info(
            "grip force cap engaged: a commanded close was backed off to hold the measured gripper effort at "
            "the cap; later grasps this run are not logged"
        )

    def _clamp(self, action: types.Action) -> _ClampResult:
        """Caps each arm's end-effector speed and gripper step from the last command: the last line of defence.

        The first command of a run has nothing to be measured from and passes; the ramp handles it.

        @param action The command after the ramp.
        @return The command to submit, whether anything was cut, and the speed each arm asked for.
        """
        if self._last_action is None:
            return _ClampResult(action, False, {arm: math.nan for arm in ARMS})
        wanted = _by_arm(action)
        last = _by_arm(self._last_action)
        commands: dict[str, types.ArmCommand] = {}
        speeds: dict[str, float] = {}
        cuts: list[str] = []
        for arm in ARMS:
            commands[arm], speeds[arm], cut = _clamp_arm(
                wanted[arm],
                last[arm],
                None if self._kinematics is None else self._kinematics[arm],
                self._max_ee_speed_m_s,
                1.0 / self._period_s,
                self._max_gripper_step,
            )
            if cut is not None:
                cuts.append(f"{arm} {cut}")
        if self._kinematics is not None:
            self.max_ee_speed_requested_m_s = max(self.max_ee_speed_requested_m_s, *speeds.values())
            counts = self._episode_counts
            counts.max_ee_speed_requested_m_s = max(counts.max_ee_speed_requested_m_s, *speeds.values())
        if cuts:
            self.clamp_count += 1
            self._episode_counts.clamps += 1
            self._log_clamp_once(cuts[0])
        return _ClampResult(types.Action(left=commands["left"], right=commands["right"]), bool(cuts), speeds)

    def _log_clamp_once(self, what: str) -> None:
        """Logs the first clamped command of the run and nothing for the rest: the counters carry the rest.

        @param what The clamp's description of the first cut (arm, asked and allowed speed or step).
        """
        if self._has_logged_clamp:
            return
        self._has_logged_clamp = True
        _log.warning(
            "command clamped: %s; later clamps this run are counted (clamp_count) and the largest request kept "
            "(max_ee_speed_requested_m_s), not logged",
            what,
        )

    def _tracker_plans(self, action: types.Action, now_s: float) -> dict[str, sys0_tracker.Plan] | None:
        """What System 0 follows from this tick: the command just submitted, then the rows still buffered.

        The rows are timed as the loop will command them, one per period from now, not by their chunk's due
        times: after a hold the loop installs a chunk from where the arm stopped, so those times can already be
        past. The gripper is the submitted one, after the grip force cap: the policy's raw gripper ahead would pull
        the target toward a full close between ticks and undo the cap on every grasp. It is reached over the tick,
        not at once: the plan's first row holds the gripper the loop submitted last tick, and the rows from one
        period on hold this tick's, so the tracker, which samples the plan every 4 ms, moves the gripper's target
        in a ramp, as the command writer does without System 0. A gripper held at this tick's value from now was a
        20 Hz staircase of steps up to `max_gripper_step`, heard on a station and felt in every grasp it closed.
        None while the start-up ramp runs, which the command writer drives from the measured pose.

        @param action The command submitted this tick.
        @param now_s The tick time, seconds.
        @return Each arm's plan, or None during the ramp.
        """
        if self._ramp is not None:
            return None
        chunk = self._chunk_plan
        remaining = len(self._buffer) if chunk is not None else 0
        t_s: npt.NDArray[np.float64] = now_s + np.arange(remaining + 1, dtype=np.float64) * self._period_s
        previous = _by_arm(self._last_action) if self._last_action is not None else {}
        plans: dict[str, sys0_tracker.Plan] = {}
        for arm, command in _by_arm(action).items():
            head = np.append(command.positions_rad, command.gripper)[None, :]
            rows = head if chunk is None or remaining == 0 else np.vstack([head, chunk.rows[arm][-remaining:]])
            rows[:, self._dofs[arm]] = command.gripper
            arm_t_s = t_s
            before = previous.get(arm)
            if before is not None and before.gripper != command.gripper:
                if len(rows) == 1:
                    # Nothing buffered: a second row one period on, where the ramp to this tick's gripper ends.
                    rows = np.vstack([rows, rows])
                    arm_t_s = np.append(t_s, now_s + self._period_s)
                rows[0, self._dofs[arm]] = before.gripper
            plans[arm] = sys0_tracker.Plan(t_s=arm_t_s, rows=rows)
        return plans

    def _snapshot_plan(self, now_s: float) -> PlanSnapshot:
        """Builds this tick's #PlanSnapshot: the buffer's tail of the installed chunk plus the history rings.

        The buffer is always the tail of the installed chunk (an install replaces it whole, a reset
        empties it), so the remaining rows are the last `len(buffer)` of the chunk's arrays: slices,
        no per-row work.

        @param now_s The tick time, seconds; every time in the snapshot is relative to it.
        @return The snapshot.
        """
        chunk = self._chunk_plan
        remaining = len(self._buffer)
        plan_t_s: npt.NDArray[np.float64]
        rows: dict[str, npt.NDArray[np.float64]]
        speeds: dict[str, npt.NDArray[np.float64]]
        if chunk is None or remaining == 0:
            plan_t_s = np.zeros(0)
            rows = {arm: np.zeros((0, self._dofs[arm] + 1)) for arm in ARMS}
            speeds = {arm: np.zeros(0) for arm in ARMS}
        else:
            start = len(chunk.due_s) - remaining
            plan_t_s = chunk.due_s[start:] - now_s
            rows = {arm: chunk.rows[arm][start:] for arm in ARMS}
            speeds = {arm: chunk.ee_speed_m_s[arm][start:] for arm in ARMS}
        return self._history.snapshot(now_s, plan_t_s, rows, speeds, self._max_ee_speed_m_s)

    def _begin_episode(self, task: str) -> None:
        """Starts the recorder's episode for `task` and copies the inference log into it; nothing without a recorder.

        @param task The prompt the episode is recorded under.
        @throws RolloutError The recorder cannot start.
        @throws OSError The episode's inference log cannot be opened.
        """
        if self._recorder is None:
            return
        self._episode_counts = _EpisodeCounts()
        folder = self._recorder.begin(task)
        self._inference_log.set_mirror(str(folder / rollout_episode.INFERENCE_LOG_NAME))

    def _end_episode(self, reason: str, error: str | None = None) -> None:
        """Stops copying the inference log and hands the recorder's episode to its finisher thread; returns at once.

        @param reason Why it ended: `"stopped"`, `"cancelled"`, an exception's class name, `"task_changed"`.
        @param error The message of the exception that ended it.
        """
        if self._recorder is None:
            return
        self._inference_log.set_mirror(None)
        self._recorder.end(reason, error, dataclasses.asdict(self._episode_counts))

    def _open_trace(self) -> None:
        """Opens the trace file for appending and writes the header when the file is new or empty.

        Nothing happens without `trace_path`. The file stays open for the loop's lifetime and is
        closed by _close_trace() from `__aexit__`.
        """
        if self._trace_path is None:
            return
        is_new = not pathlib.Path(self._trace_path).exists() or pathlib.Path(self._trace_path).stat().st_size == 0
        self._trace_file = open(self._trace_path, "a", newline="", encoding="utf-8")  # noqa: SIM115 (closed in __aexit__)
        self._trace = csv.writer(self._trace_file)
        if is_new:
            self._trace.writerow(trace_columns(self._dofs))

    def _close_trace(self) -> None:
        """Closes the trace file if one is open; safe to call twice."""
        if self._trace_file is not None:
            self._trace_file.close()
            self._trace_file = None
            self._trace = None

    def _trace_tick(self, now_s: float, state: types.RobotState, clamp: _ClampResult) -> None:
        """Appends one row per commanded tick; flushed every #_TRACE_FLUSH_ROWS rows.

        @param now_s Monotonic tick deadline, seconds, the row's `t_s`.
        @param state This tick's measured arm state, the `measured_*` columns.
        @param clamp The command submitted this tick (the `commanded_*` columns), whether the clamp cut it, and
            the end-effector speeds it asked for.
        """
        if self._trace is None:
            return
        action = clamp.action
        alpha = 1.0 if self._ramp is None else self._ramp.tick / self._ramp.ticks
        # The buffer is always the tail of the installed chunk, so its length says which row just went out.
        row_index = self._horizon - len(self._buffer) - 1
        row: list[Any] = [f"{now_s:.4f}", self._chunk_count, row_index, f"{alpha:.3f}", int(clamp.clamped)]
        row += [f"{v:.5f}" for v in state.left.positions_rad] + [f"{v:.5f}" for v in state.right.positions_rad]
        row += [f"{state.left.gripper:.4f}", f"{state.right.gripper:.4f}"]
        row += [f"{v:.5f}" for v in action.left.positions_rad] + [f"{v:.5f}" for v in action.right.positions_rad]
        row += [f"{action.left.gripper:.4f}", f"{action.right.gripper:.4f}"]
        row += [f"{clamp.ee_speed_m_s[arm]:.4f}" for arm in ARMS] + [self._commands.written]
        self._trace.writerow(row)
        self._trace_rows += 1
        if self._trace_rows % _TRACE_FLUSH_ROWS == 0:
            self._trace_file.flush()

    def _start_ramp(self, target: types.Action, state: types.RobotState) -> _Ramp:
        """Sizes a ramp from the measured pose to the first target and logs it once.

        The ramp lasts `max(ramp_s, largest joint delta / ramp_max_rad_s)` seconds, counted in
        commanded ticks so a stall stretches it instead of skipping ahead in it.

        @param target The first row to be commanded after the ramp starts.
        @param state This tick's measured arm state, the ramp's origin.
        @return The ramp, with `tick` at 0.
        """
        origin = types.Action(left=_measured_command(state.left), right=_measured_command(state.right))
        max_delta_rad = max(
            float(np.max(np.abs(target.left.positions_rad - origin.left.positions_rad))),
            float(np.max(np.abs(target.right.positions_rad - origin.right.positions_rad))),
        )
        duration_s = max(self._ramp_s, max_delta_rad / self._ramp_max_rad_s)
        # Counted in commanded ticks rather than seconds, so an overrun or a stall stretches the
        # ramp instead of skipping ahead in it; it only ever takes longer, never moves faster.
        ticks = max(1, math.ceil(duration_s / self._period_s))
        _log.info(
            "ramp started: %.2f s over %d ticks from the measured pose to the first target, max joint delta %.3f rad",
            duration_s,
            ticks,
            max_delta_rad,
        )
        return _Ramp(origin=origin, ticks=ticks)

    def _stall(self, now_s: float) -> None:
        """Holds the arms for an empty-buffer tick and stops the loop once the wait is too long.

        @param now_s Monotonic tick deadline, seconds.
        """
        self._hold()
        if not self._has_logged_stall:
            self._has_logged_stall = True
            self._inference_log.write("stall", chunk=self._chunk_count, max_stall_ticks=self._max_stall_ticks)
        if self._chunk_count == 0:
            # Before motion the arms rest, so a booting pool is waited out and max_stall_ticks does not count.
            assert self._first_tick_s is not None and self._first_chunk_from_s is not None, "step() sets them"
            if now_s - self._first_tick_s > POOL_WARMING_LIMIT_S:
                raise errors.ControlLoopError(
                    f"the policy has not started on DreamAPI after {POOL_WARMING_LIMIT_S / 60:.0f} minutes (it still "
                    "answers pool_warming); the run stops. Start again later, or ask Dream Labs support",
                    reason=self.refill_error_code,
                )
            if now_s - self._first_chunk_from_s > self._first_chunk_timeout_s:
                warmed = self._first_chunk_from_s > self._first_tick_s
                since = "the last pool_warming answer" if warmed else "the first tick"
                because = f"; the last get_action failed: {self.refill_error}" if self.refill_error else ""
                raise errors.ControlLoopError(
                    f"no action chunk within first_chunk_timeout_s={self._first_chunk_timeout_s:g} s of "
                    f"{since}{because}",
                    reason=self.refill_error_code,
                )
            return
        self._stall_ticks += 1
        self._stall_count += 1
        self._episode_counts.stalls += 1
        if self._stall_ticks > self._max_stall_ticks:
            # The stall names what emptied the buffer when a refill failure stands, so the station's `last_error`
            # and the page say "the serve limit" or "wrong shape for this body", not "the buffer ran dry".
            because = f"; the last get_action failed: {self.refill_error}" if self.refill_error else ""
            held_s = self._stall_ticks * self._period_s
            raise errors.ControlLoopError(
                f"action buffer empty for {self._stall_ticks} consecutive ticks ({held_s:.2f} s), "
                f"more than max_stall_ticks={self._max_stall_ticks}{because}",
                reason=self.refill_error_code,
            )
        _log.warning("stall %d/%d: no action chunk yet, holding", self._stall_ticks, self._max_stall_ticks)

    def _hold(self) -> None:
        """Clears the command path's target and re-holds both arms; the driver keeps its last PD target."""
        self._commands.hold()

    def _stop_cameras(self) -> None:
        """Stops every camera thread the loop owns; a failure to stop one is logged and the others are still stopped.
        Readers a station lent (`camera_owner=False`) are left running for the next run and the live view."""
        if not self._is_camera_owner:
            return
        for reader in self._cameras:
            run_teardown(f"stop camera '{reader.name}'", reader.stop)


class _CommandWriter:
    """The loop's command path: runs each tick's submitted command out to both arms at `command_hz`.

    i2rt applies every `command_joint_pos` straight to the motor PD, so a row re-sent as-is at
    the row rate is a position staircase, and the policies were trained on the motion a
    command thread makes of a row instead. This is that thread. Each command period it takes
    the newest submission, moves linearly from where the previous run had got to toward it over
    `window_s = clamp(gap between the last two submits, 1 / command_hz, command_interp_max_s)`
    seconds (a row submitted every 33 ms is reached in about eight 4 ms steps), and writes both
    arms. There is no filter on this path: the rows arrive smoothed from the dream client and are
    written as the run reaches them. A run always starts from where the last one got to, never
    from the target, so the command path cannot jump: after a hold the next target is reached
    from where the arms were left, at most `command_interp_max_s` later. The value written each
    period is where the run should be one period on, so a run whose window is one period is
    written whole at once.

    Between start() and stop() the periods run on the command thread; without one the loop calls
    once() itself (tests). Every arm write goes through here under one lock, the hold included,
    so the thread and the tick never write an arm at the same time.

    Attributes:
    - period_s: Seconds between writes, `1 / command_hz`.
    - written: Command periods on which both arms were written so far.
    - error: The exception a write on the command thread raised; the thread has ended when it is set.

    @see submit(), hold(), once()
    """

    def __init__(
        self,
        followers: dict[str, followers.Arm],
        *,
        command_hz: float,
        command_interp_max_s: float,
        clock: Callable[[], float],
    ) -> None:
        """Binds the arms and the timing; nothing is written until a submission arrives.

        @param followers The two arms keyed `"left"` and `"right"`.
        @param command_hz Write rate.
        @param command_interp_max_s Longest run window, seconds.
        @param clock The loop's clock; the thread paces itself and times the runs on it.
        """
        self._followers = followers
        self.period_s = 1.0 / command_hz
        self._max_window_s = command_interp_max_s
        self._clock = clock
        self.written = 0
        self.error: Exception | None = None
        # The latest-value slot: the tick rebinds it and the thread reads it, a reference swap each way.
        self._submission: _Submission | None = None
        self._interpolation: _Interpolation | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._is_running = False

    def submit(self, action: types.Action, now_s: float) -> None:
        """Makes `action` the target the next command periods run toward.

        @param action The tick's command for both arms.
        @param now_s The tick time, seconds; the gap to the previous submit sizes the run.
        """
        self._submission = _Submission(action=action, t_submit_s=now_s)

    def hold(self) -> None:
        """Clears the target, so no period writes until the next submit, and re-commands each arm's last target.

        The re-command is what `FollowerArm.hold()` does: cheap, explicit, and independent of whether
        the driver keeps its last PD target. The next submitted target is run from where the arms were left.

        @throws ArmError A never-written arm's observation is malformed (see `FollowerArm.hold()`).
        """
        self._submission = None
        with self._lock:
            for arm in self._followers.values():
                arm.hold()

    def restart(self) -> None:
        """Forgets the run in progress, so the next target is written as it is, as the first one of a loop is.

        The interpolation (the class docstring) carries the motor PD from where the last run got to toward each new
        row in 4 ms steps, so a 20 Hz row never reaches the motors as a 50 ms staircase. System 0's motor path
        drives the arms between runs of this writer (#robots.sys0.motors.MotorPath) and hands them back at a new
        ramp; the run this writer had going still points at where the arms were before that, and resuming it would
        step them back. The ramp's first target is one small step from the arms' measured pose, so writing it as it is
        moves them by no more than that step.
        """
        self._interpolation = None

    def once(self, now_s: float) -> None:
        """Runs one command period: writes the run's next value, filtered, to both arms; nothing while held.

        @param now_s Time on the loop's clock, seconds; the value written is where the run should be at
            `now_s + period_s`.
        @throws ArmError A write was refused (see `FollowerArm.write()`); a driver exception propagates unchanged.
        """
        submission = self._submission
        if submission is None:
            return
        if self._interpolation is None or self._interpolation.submission is not submission:
            self._interpolation = self._start_interpolation(submission)
        run = self._interpolation
        elapsed_s = now_s - submission.t_submit_s
        alpha = min(1.0, (elapsed_s + self.period_s) / run.window_s)
        run.position = submission.action if alpha >= 1.0 else _blend_action(run.origin, submission.action, alpha)
        command = run.position
        with self._lock:
            self._followers["left"].write(command.left)
            self._followers["right"].write(command.right)
        self.written += 1

    def start(self) -> None:
        """Starts the command thread; it runs once() every `period_s` until stop()."""
        self._is_running = True
        self._thread = threading.Thread(target=self._run, name="commands", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stops the command thread and waits for its current write to finish; a no-op when it never ran."""
        self._is_running = False
        if self._thread is None:
            return
        self._thread.join(_COMMAND_STOP_TIMEOUT_S)
        if self._thread.is_alive():
            _log.warning("command thread still writing after %.1f s; holding the arms anyway", _COMMAND_STOP_TIMEOUT_S)
        self._thread = None

    def _start_interpolation(self, submission: _Submission) -> _Interpolation:
        """Starts a run to a newly submitted target from where the previous run got to.

        The run's window is the measured gap between this submit and the previous one, clamped to
        `[period_s, command_interp_max_s]`, so a row submitted every 33 ms is reached in about eight
        4 ms steps. The first target ever has nothing to run from and is written as it is.

        @param submission The newest submission.
        @return The run, positioned where the previous run got to (or at the target for the first one).
        """
        previous = self._interpolation
        if previous is None:
            # The first target ever: there is nothing to run from, so it is written as it is.
            return _Interpolation(
                origin=submission.action, submission=submission, window_s=self.period_s, position=submission.action
            )
        gap_s = submission.t_submit_s - previous.submission.t_submit_s
        window_s = min(max(gap_s, self.period_s), self._max_window_s)
        return _Interpolation(
            origin=previous.position, submission=submission, window_s=window_s, position=previous.position
        )

    def _run(self) -> None:
        """The command thread's body: once() every `period_s` on the loop's clock until stop().

        Sleeps out the rest of each period rather than spinning, because the tick thread needs the
        GIL. A write that raises ends the thread and is stored in `error` for the next tick to raise
        (checklist rule 4: one boundary per thread).
        """
        try:
            while self._is_running:
                started_s = self._clock()
                self.once(started_s)
                # Sleep out the rest of the period rather than spin: the tick thread needs the GIL.
                time.sleep(max(0.0, self.period_s - (self._clock() - started_s)))
        except Exception as exc:
            # The thread boundary (checklist rule 4): a failed write ends the thread here, and the
            # next tick raises it, so the loop stops the way an in-tick write failure stopped it.
            self.error = exc


def _measured_command(arm: types.ArmState) -> types.ArmCommand:
    """The measured pose as a command; the gripper is clipped as `hold()` does, so a blend from it validates.

    @param arm The measured state of one arm.
    @return The equivalent command.
    """
    return types.ArmCommand(positions_rad=arm.positions_rad, gripper=min(max(arm.gripper, 0.0), 1.0))


def _blend(origin: types.ArmCommand, target: types.ArmCommand, alpha: float) -> types.ArmCommand:
    """`origin + alpha * (target - origin)` for the joints and the gripper alike.

    @param origin The command the ramp started from.
    @param target This tick's target.
    @param alpha Blend weight in `[0, 1)`.
    @return A new #robots.types.ArmCommand; neither argument is touched.
    """
    return types.ArmCommand(
        positions_rad=origin.positions_rad + alpha * (target.positions_rad - origin.positions_rad),
        gripper=origin.gripper + alpha * (target.gripper - origin.gripper),
    )


def _blend_action(origin: types.Action, target: types.Action, alpha: float) -> types.Action:
    """`origin + alpha * (target - origin)` for both arms.

    @param origin The action blended from.
    @param target The action blended toward.
    @param alpha Blend weight in `[0, 1]`.
    @return A new #robots.types.Action; neither argument is touched.
    """
    return types.Action(left=_blend(origin.left, target.left, alpha), right=_blend(origin.right, target.right, alpha))


def _clamp_arm(
    command: types.ArmCommand,
    last: types.ArmCommand,
    arm_kinematics: kinematics.Kinematics | None,
    max_ee_speed_m_s: float,
    control_hz: float,
    max_gripper_step: float,
) -> tuple[types.ArmCommand, float, str | None]:
    """Caps one arm's end-effector speed from the last command's pose, and its gripper to one step.

    The speed is `|fk(command) - fk(last)| * control_hz`: the distance the grasp point would travel
    in one tick. Over the cap, the joint delta is scaled by `cap / speed` and, because forward
    kinematics is nonlinear and the scaled step's speed is only about the cap, measured once more
    and scaled again by what it came to when still over; then accepted. Without kinematics the joints
    pass through and the speed is NaN.

    @param command The wanted command.
    @param last The last command written to the same arm.
    @param arm_kinematics The arm's forward kinematics, or None.
    @param max_ee_speed_m_s The cap, metres per second.
    @param control_hz The tick rate: one command per tick, so displacement per tick times this is the speed.
    @param max_gripper_step Gripper limit per tick.
    @return The clamped command, the speed asked for, and a description of the cut or None when nothing was cut.
    """
    gripper_delta = command.gripper - last.gripper
    gripper = last.gripper + min(max(gripper_delta, -max_gripper_step), max_gripper_step)
    cut: str | None = None
    if abs(gripper_delta) > max_gripper_step:
        cut = f"gripper asked {abs(gripper_delta):.3f} in one tick, limit {max_gripper_step:.3f}"
    if arm_kinematics is None:
        return types.ArmCommand(positions_rad=command.positions_rad, gripper=gripper), math.nan, cut
    p_last = arm_kinematics.ee_position(last.positions_rad)
    speed = float(np.linalg.norm(arm_kinematics.ee_position(command.positions_rad) - p_last)) * control_hz
    if speed <= max_ee_speed_m_s:
        return types.ArmCommand(positions_rad=command.positions_rad, gripper=gripper), speed, cut
    positions = last.positions_rad + (command.positions_rad - last.positions_rad) * (max_ee_speed_m_s / speed)
    scaled_speed = float(np.linalg.norm(arm_kinematics.ee_position(positions) - p_last)) * control_hz
    if scaled_speed > max_ee_speed_m_s:
        positions = last.positions_rad + (positions - last.positions_rad) * (max_ee_speed_m_s / scaled_speed)
    cut = f"end effector asked {speed:.2f} m/s, limit {max_ee_speed_m_s:.2f} m/s"
    return types.ArmCommand(positions_rad=positions, gripper=gripper), speed, cut


def _plan_chunk(
    chunk: types.ActionChunk,
    skipped: int,
    period_s: float,
    arm_kinematics: dict[str, kinematics.Kinematics] | None,
    last: types.Action | None,
) -> _ChunkPlan:
    """The installed rows of a chunk as arrays with their due times and end-effector speeds.

    The one place the plan is walked row by row, once per chunk, so the per-tick snapshot only
    slices. The first installed row's speed is measured from the last command when there is one:
    that is the pose it follows on the arm.

    @param chunk The chunk being installed.
    @param skipped Rows dropped from its head.
    @param period_s Seconds between rows.
    @param arm_kinematics Per-arm kinematics; None leaves every speed NaN.
    @param last The last command submitted, or None.
    @return The arrays for rows `skipped` onward.
    """
    installed = chunk.actions[skipped:]
    rows = {
        arm: np.array([[*_by_arm(action)[arm].positions_rad, _by_arm(action)[arm].gripper] for action in installed])
        for arm in ARMS
    }
    speeds: dict[str, npt.NDArray[np.float64]] = {arm: np.full(len(installed), np.nan) for arm in ARMS}
    if arm_kinematics is not None:
        for arm in ARMS:
            positions = np.array(
                [arm_kinematics[arm].ee_position(_by_arm(action)[arm].positions_rad) for action in installed]
            )
            speeds[arm][1:] = np.linalg.norm(np.diff(positions, axis=0), axis=1) / period_s
            if last is not None:
                p_last = arm_kinematics[arm].ee_position(_by_arm(last)[arm].positions_rad)
                speeds[arm][0] = float(np.linalg.norm(positions[0] - p_last)) / period_s
    due_s = np.asarray(chunk.observed_at_s + np.arange(skipped, len(chunk.actions)) * period_s, dtype=np.float64)
    return _ChunkPlan(rows=rows, ee_speed_m_s=speeds, due_s=due_s)


def _by_arm(action: types.Action) -> dict[str, types.ArmCommand]:
    """An action's two arm commands keyed as #ARMS names them.

    @param action The action.
    @return `{"left": action.left, "right": action.right}`.
    """
    return {"left": action.left, "right": action.right}


def _by_arm_state(state: types.RobotState) -> dict[str, types.ArmState]:
    """A robot state's two arm states keyed as #ARMS names them.

    @param state The state.
    @return `{"left": state.left, "right": state.right}`.
    """
    return {"left": state.left, "right": state.right}


def _motor_temps(state: types.RobotState) -> dict[str, list[float]] | None:
    temps = {arm: _by_arm_state(state)[arm].motor_temps_c for arm in ARMS}
    if all(value is None for value in temps.values()):
        return None
    return {arm: [round(t, 1) for t in value] for arm, value in temps.items() if value is not None}


def _percentile(sorted_values: list[float], q: float) -> float | None:
    """Nearest-rank percentile of an ascending list.

    @param sorted_values The values, ascending; may be empty.
    @param q The quantile in `(0, 1]`.
    @return The value at rank `ceil(q * n)`, or None for an empty list.
    """
    if not sorted_values:
        return None
    rank = max(1, math.ceil(q * len(sorted_values)))
    return sorted_values[rank - 1]


def _retry_after_s(exc: Exception) -> float | None:
    """Seconds to hold the next refill after a failed one, or None for a fault worth retrying at the next tick.

    @param exc The exception a refill raised.
    @return The envelope's `retry_after_s` for a refusal.
    """
    if isinstance(exc, dream.DreamServerError):
        return RETRY_AFTER_DEFAULT_S if exc.retry_after_s is None else exc.retry_after_s
    if isinstance(exc, (dream.DreamActionError, errors.ControlLoopError, dream.DreamProtocolError)):
        return RETRY_AFTER_DEFAULT_S
    return None


def run_teardown(what: str, action: Callable[[], object]) -> bool:
    """Runs one teardown step; a failure is logged so the remaining steps still run.

    @param what What the step does, for the log line.
    @param action The step.
    @return True when the step finished, False when it raised.
    """
    try:
        action()
    except errors.RobotsError as exc:
        _log.error("teardown step failed: %s: %s", what, exc)
        return False
    except Exception:
        # The one broad except: teardown must reach every arm and the dream client, and must
        # never raise over the exception that ended the loop.
        _log.exception("teardown step failed: %s", what)
        return False
    return True


def _refusal_help(reason: str | None) -> str:
    """DreamAPI's customer words for a refusal's reason (`shared.contracts.envelope.REASON_HELP`): what happened and
    what to do, so a run stopped by `max_serves` says that Dream-1 is at max capacity, and what then.

    @param reason The refusal's reason, e.g. `max_serves`; None when the answer carried none.
    @return `. <meaning> <action>`, or an empty string for a reason the contract does not know.
    """
    words = {str(known): text for known, text in envelope.REASON_HELP.items()}.get(reason or "")
    return "" if words is None else f". {words.meaning} {words.action}"


def _starts_new_session(exc: Exception) -> bool:
    """Whether the retry after a failed refill starts a new session; the rule of _collect_refill()'s retry branch.

    A refusal that says the session itself is fine keeps it: `rate_limited` and `queue_timeout` (the robot asked
    too fast, or the serve is answering another session) and a chunk of the wrong shape (the checkpoint answers
    the same shape every time), where a reset would throw away the episode's context on the serve for nothing.
    Every other failure (a serve fault, a timeout, an unreachable gateway, a pool refusal) may mean the serve or
    the session is gone, and the retry does not resume it.

    @param exc The exception the refill raised.
    @return True when the retry must carry no session id.
    """
    if isinstance(exc, dream.DreamServerError):
        return exc.reason not in ("rate_limited", "queue_timeout")
    return not isinstance(exc, (dream.DreamActionError, errors.ControlLoopError))


def _discard_refill(task: asyncio.Future[Any]) -> None:
    """Retrieves the outcome of a refill that finished after shutdown; nobody else will.

    @param task The refill task, done or cancelled.
    """
    if task.cancelled():
        return
    if task.exception() is not None:
        _log.debug("refill finished after shutdown with %r", task.exception())
