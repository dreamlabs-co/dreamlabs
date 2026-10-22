"""Test doubles for the control loop: a robot, a capture source, a dream client and a clock.

Each fake mirrors the shape the library consumes (i2rt's `Robot`, cv2's `VideoCapture`, `dream.DreamClient` with
its policy open, `time.monotonic` / `asyncio.sleep`) and records what was done to it, so a test can
assert on commands, observations and sleeps instead of on hardware. make_loop() assembles a
`robots.control.ControlLoop` from them.

The `robots` modules are imported lazily inside make_loop() and the fixtures that need them,
so the fakes themselves stay usable before the library exists (see `test_conftest_fakes.py`).
"""

import asyncio
import dataclasses
import math
import pathlib
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any

import dream
import numpy as np
import pytest
from dream import fine_tuning
from dream.errors import DreamServerError

from robots import cameras

## Joint count of one YAM arm.
ARM_DOF = 6

## Default frame shape (rows, columns, channels) the fake camera produces.
FRAME_SHAPE = (48, 64, 3)

## Longest a gated FakeClient call waits before giving up; a test that forgot to open the gate
## fails here instead of hanging pytest.
GATE_TIMEOUT_S = 5.0

## Sleep inside FakeCaptureSource.read(): a real capture blocks for a frame period, and without
## it the daemon capture thread spins and starves the test thread of the GIL.
READ_DELAY_S = 0.001

## Defaults make_loop() uses when the test does not override them. The ramp is one tick (shorter
## than any period used here, no speed cap), so the policy's rows reach the arms unchanged and a
## test that pins the buffer, the refill or the teardown reads the rows, not a blend.
LOOP_DEFAULTS: dict[str, Any] = {
    "control_hz": 30.0,
    "action_horizon": 4,
    "action_timeout_s": 1.0,
    "stale_after_s": 0.2,
    "max_stall_ticks": 3,
    "first_chunk_timeout_s": 120.0,
    "refill_at_remaining": 0,
    "ramp_s": 1e-3,
    "ramp_max_rad_s": math.inf,
    # Raw pass-through for the unit tests that pin exact rows; the clamp has its own tests. No
    # kinematics, so the joints are never clamped and the end-effector speeds are NaN; the clamp tests inject one.
    "max_ee_speed_m_s": math.inf,
    "max_gripper_step": math.inf,
    # Off, so the rows reach the arms unregulated; the fakes report no gripper effort anyway, which is a
    # pass-through of its own. The cap has its own tests (test_gripper_force.py) and one loop test.
    "grip_force_cap": 0.0,
    # The command path writes synchronously inside step(), one command per tick, reached whole (the run
    # window is capped at one command period), so the rows reach the arms unchanged; the
    # interpolation and the thread have their own tests.
    "command_hz": 30.0,
    "command_interp_max_s": 1.0 / 30.0,
    "command_thread": False,
}

## Camera names the default make_loop() wires up: the three serve roles, as the example config names them.
CAMERA_NAMES = ("top", "left", "right")

## Language prompt make_loop() hands the loop; the fake client records it under `prompt`.
TASK = "pick up the cup"


class FakeRobot:
    """Stand-in for an i2rt `Robot`: static observations and a log of every command.

    Attributes:
    - joint_pos: Positions reported by get_observations(), radians, shape [6].
    - joint_vel: Velocities reported by get_observations(), rad/s, shape [6].
    - gripper_pos: Normalised gripper reported by get_observations(), shape [1].
    - gripper_eff: Gripper effort reported by get_observations() as i2rt's `gripper_eff`, or None for a driver
      that reports none (the default, as `MotorChainRobot` is the only one that does).
    - commands: Every 7-vector passed to command_joint_pos(), in order.
    - read_error: Raised by the next get_observations() call, then cleared.
    - write_error: Raised by the next command_joint_pos() call, then cleared.
    - gravity_comp_calls: Times enter_gravity_comp_idle() was called (only when enabled).
    - zero_torque_calls: Times zero_torque_mode() was called (only when enabled).
    - closed: True once close() was called, as `MotorChainRobot.close()` would stop the driver's threads.
    """

    def __init__(
        self,
        joint_pos: np.ndarray | None = None,
        gripper_pos: float = 0.5,
        gripper_eff: float | None = None,
        has_gravity_comp: bool = False,
        has_zero_torque: bool = False,
    ) -> None:
        """Builds a robot at rest, optionally with the disengage methods the wrapper probes for.

        @param joint_pos Reported joint positions, radians; zeros when omitted.
        @param gripper_pos Reported gripper opening in [0, 1].
        @param gripper_eff Reported gripper effort, or None to report none.
        @param has_gravity_comp Expose `enter_gravity_comp_idle()`.
        @param has_zero_torque Expose `zero_torque_mode()`.
        """
        self.joint_pos = np.zeros(ARM_DOF) if joint_pos is None else np.asarray(joint_pos, dtype=np.float64)
        self.joint_vel = np.zeros(ARM_DOF)
        self.gripper_pos = np.array([gripper_pos])
        self.gripper_eff = gripper_eff
        self.commands: list[np.ndarray] = []
        self.read_error: BaseException | None = None
        self.write_error: BaseException | None = None
        self.gravity_comp_calls = 0
        self.zero_torque_calls = 0
        self.closed = False
        # FollowerArm.disengage() picks a method by `hasattr`, so an unsupported method must be
        # absent from the instance rather than a no-op; binding per instance keeps one class.
        if has_gravity_comp:
            self.enter_gravity_comp_idle = self._enter_gravity_comp_idle
        if has_zero_torque:
            self.zero_torque_mode = self._zero_torque_mode

    def get_observations(self) -> dict[str, np.ndarray]:
        """Returns the i2rt observation dict, or raises the queued `read_error` once.

        @return `{"joint_pos": [6], "joint_vel": [6], "gripper_pos": [1]}`, plus `gripper_eff` [1] when this
            robot reports one.
        @throws BaseException Whatever was queued in `read_error`.
        """
        if self.read_error is not None:
            error, self.read_error = self.read_error, None
            raise error
        observation = {
            "joint_pos": self.joint_pos.copy(),
            "joint_vel": self.joint_vel.copy(),
            "gripper_pos": self.gripper_pos.copy(),
        }
        if self.gripper_eff is not None:
            observation["gripper_eff"] = np.array([self.gripper_eff])
        return observation

    def command_joint_pos(self, positions: np.ndarray) -> None:
        """Records a 7-vector command `[*joints_rad, gripper]`, or raises the queued `write_error` once.

        @param positions Joint targets plus gripper, shape [7].
        @throws BaseException Whatever was queued in `write_error`.
        """
        if self.write_error is not None:
            error, self.write_error = self.write_error, None
            raise error
        self.commands.append(np.array(positions, dtype=np.float64))

    def close(self) -> None:
        """Marks the driver closed; a real one stops its threads and its CAN channel here."""
        self.closed = True

    def _enter_gravity_comp_idle(self) -> None:
        self.gravity_comp_calls += 1

    def _zero_torque_mode(self) -> None:
        self.zero_torque_calls += 1


class FakeCaptureSource:
    """Stand-in for `cv2.VideoCapture`: read() yields BGR frames until told to fail.

    Attributes:
    - ok: While True read() returns `(True, frame)`; set False (or call fail()) for `(False, None)`.
    - released: True once release() was called.
    - read_count: Number of read() calls so far.
    """

    def __init__(self, frames: Iterable[np.ndarray] | np.ndarray | None = None) -> None:
        """Builds a source over a fixed frame or a sequence of frames.

        A finite sequence is followed by repeats of its last frame, so a capture thread looping
        at full speed never runs dry by accident; use fail() to end the stream on purpose.

        @param frames One BGR frame, an iterable of them, or None for a default frame.
        """
        if frames is None:
            frames = make_bgr_frame()
        if isinstance(frames, np.ndarray):
            frames = [frames]
        self._frames: Iterator[np.ndarray] = iter(frames)
        self._last: np.ndarray | None = None
        self.ok = True
        self.released = False
        self.read_count = 0

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Returns the next frame the cv2 way, `(ok, frame_bgr)`.

        @return `(True, frame)` while `ok`, else `(False, None)`.
        """
        self.read_count += 1
        time.sleep(READ_DELAY_S)
        if not self.ok:
            return False, None
        self._last = next(self._frames, self._last)
        if self._last is None:
            return False, None
        return True, self._last

    def fail(self) -> None:
        """Ends the stream: every read() from now on returns `(False, None)`."""
        self.ok = False

    def release(self) -> None:
        """Records the release; a real driver would close the device here."""
        self.released = True


class FakeClient:
    """Stand-in for a `dream.DreamClient` with its policy open: fixed-horizon chunks, optional latency, optional fault.

    set_observation() keeps the tick it is handed as the latest, as the real client does, and refuses what the real
    one refuses before anything is kept: no frames, a frame key in the observation, a timestamp earlier than the
    episode's last. get_action() takes no observation: it records what the real client would send, the latest
    observation with each frame key's latest frame (a one-frame window: the fake keeps no window), and refuses with
    `DreamObservationError` when no set_observation() came since the episode started. reset_episode() starts a new
    episode, so the next get_action() needs a set_observation() first, as on the real client.

    Chunk `k` (zero-based call index) carries rows `k * horizon + i` in every joint column, so
    the sequence of commanded values across chunks is 0, 1, 2, ... and a test can tell which
    action of which chunk reached the arms. Grippers (the seventh column) cycle through
    `(row % 10) / 10`. Chunks are nested lists, as the real client returns them from JSON.

    Attributes:
    - horizon: Rows per chunk; settable between calls.
    - gate: When set, get_action() blocks until the event is set (bounded by GATE_TIMEOUT_S).
    - raise_once: Raised by the next get_action() call, then cleared.
    - ticks: Every `(frames, observation, timestamp)` set_observation() kept, in order, copies of the mappings.
    - observations: What each get_action() call sent, in order: the latest observation and, under each frame key,
      a list of its latest frame.
    - commanded: The `commanded` each get_action() call received, in the same order.
    - calls: get_action() calls that got past the single-flight and the observation checks, including ones that
      raised.
    - closed: True once close() was called.
    - resets: Number of reset_episode() calls so far.
    - reset_after_calls: The value of `calls` at each reset_episode(), in order, so a test can pin when it happened.
    - mode: The pacing mode the loop logs (`dream.DreamClient.mode`); `"async"` unless a test sets it.
    - session_id: The session id the loop reads (`dream.DreamClient.session_id`); None unless a test sets it.
    - last_timing: The timing block the loop reads (`dream.DreamClient.last_timing`); None unless a test sets it.
    - refill_answer: What refill_due() answers, given `(executed, remaining)`; settable. By default, only when
      nothing remains, the client's sync rule.
    - refill_asks: Every `(executed, remaining)` refill_due() was asked with, in order.
    """

    def __init__(
        self,
        horizon: int = 4,
        gate: threading.Event | None = None,
        raise_once: BaseException | None = None,
    ) -> None:
        """Builds a client answering chunks of `horizon` rows.

        @param horizon Rows per chunk.
        @param gate Event get_action() waits on before answering; None answers immediately.
        @param raise_once Exception the first get_action() raises instead of answering.
        """
        self.horizon = horizon
        self.gate = gate
        self.raise_once = raise_once
        self.ticks: list[tuple[dict[str, str], dict[str, Any], float | None]] = []
        self.observations: list[dict[str, Any]] = []
        self.commanded: list[Mapping[str, Any] | None] = []
        self.calls = 0
        self.closed = False
        self.resets = 0
        self.reset_after_calls: list[int] = []
        self.mode = "async"
        self.session_id: str | None = None
        self.last_timing: dict[str, Any] | None = None
        self.refill_answer: Callable[[int, int], bool] = lambda executed, remaining: remaining == 0
        self.refill_asks: list[tuple[int, int]] = []
        # The episode's latest tick and its timestamp; None until the episode's first set_observation().
        self._latest: tuple[dict[str, str], dict[str, Any]] | None = None
        self._latest_s: float | None = None
        # Non-blocking, like the real client: a second concurrent call is a bug, not a queue.
        self._in_flight = threading.Lock()

    def set_observation(
        self, frames: Mapping[str, str], observation: Mapping[str, Any], *, timestamp: float | None = None
    ) -> None:
        """Keeps one tick as the latest; refuses, before keeping anything, what the real client refuses.

        @param frames Per window key, one frame.
        @param observation Every other key; never a frame key.
        @param timestamp The frames' capture time; `time.monotonic()` when None.
        @throws dream.DreamObservationError No frames, a frame key in the observation, or a timestamp earlier than
            the episode's last.
        """
        if not frames:
            raise dream.DreamObservationError("set_observation() frames is empty: pass one frame per window key")
        both = sorted(set(frames) & set(observation))
        if both:
            raise dream.DreamObservationError(f"observation keys {both} are also frame keys")
        timestamp_s = time.monotonic() if timestamp is None else timestamp
        if self._latest_s is not None and timestamp_s < self._latest_s:
            raise dream.DreamObservationError(
                f"set_observation() timestamp {timestamp_s!r} is earlier than the previous call's, {self._latest_s!r}"
            )
        self._latest_s = timestamp_s
        self._latest = (dict(frames), dict(observation))
        self.ticks.append((dict(frames), dict(observation), timestamp))

    def get_action(self, *, commanded: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Records what the real client would send and the commanded joints, and returns the next chunk;
        single-flight like the real client.

        @param commanded The joints last commanded per chunk key, keyword only as on the real client; stored.
        @return A serve action dict: `q_r_chunk` and `q_l_chunk` as `[H][7]` nested lists.
        @throws dream.DreamError Another get_action() is still in flight on this instance.
        @throws dream.DreamObservationError No set_observation() since the episode started.
        @throws RuntimeError The gate stayed closed for GATE_TIMEOUT_S (a test bug, not a policy fault).
        @throws BaseException Whatever was queued in `raise_once`.
        """
        if not self._in_flight.acquire(blocking=False):
            raise dream.DreamError("FakeClient: another inference request is in flight")
        try:
            latest = self._latest
            if latest is None:
                raise dream.DreamObservationError(
                    "no observation since the episode started: call set_observation(...) before get_action()"
                )
            frames, observation = latest
            self.observations.append({**observation, **{key: [frame] for key, frame in frames.items()}})
            self.commanded.append(commanded)
            call_index = self.calls
            self.calls += 1
            if self.gate is not None and not self.gate.wait(timeout=GATE_TIMEOUT_S):
                raise RuntimeError(f"FakeClient gate stayed closed for {GATE_TIMEOUT_S} s")
            if self.raise_once is not None:
                error, self.raise_once = self.raise_once, None
                raise error
            return make_chunk(self.horizon, base=call_index * self.horizon)
        finally:
            self._in_flight.release()

    def refill_due(self, executed: int, remaining: int) -> bool:
        """Records the question and answers it with `refill_answer`, as the real client paces the loop.

        @param executed Rows of the current chunk executed since its install.
        @param remaining Rows of it still queued.
        @return `refill_answer(executed, remaining)`.
        """
        self.refill_asks.append((executed, remaining))
        return self.refill_answer(executed, remaining)

    def reset_episode(self) -> None:
        """Records the reset and starts a new episode; like the real client, refuses while a get_action() is in
        flight.

        @throws dream.DreamError Another get_action() is still in flight on this instance.
        """
        if not self._in_flight.acquire(blocking=False):
            raise dream.DreamError("FakeClient: reset_episode while an inference request is in flight")
        try:
            self.resets += 1
            self.reset_after_calls.append(self.calls)
            self._latest = None
            self._latest_s = None
        finally:
            self._in_flight.release()

    def close(self) -> None:
        """Records the close; idempotent like the real client."""
        self.closed = True


class ManualClock:
    """A clock the test advances by hand; also the `sleep` that advances it.

    Pass the instance as `clock=` (it is callable) and `sleep=clock.sleep` to `ControlLoop`.
    `sleep` yields to the event loop exactly once so worker-thread results can be delivered,
    but never waits on wall time.

    Attributes:
    - now_s: Current time, seconds; settable.
    - sleeps: Every duration passed to sleep(), in order.
    """

    def __init__(self, now_s: float = 0.0) -> None:
        """Starts the clock at `now_s`.

        @param now_s Initial time, seconds.
        """
        self.now_s = now_s
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        """Returns `now_s`, so the instance can stand in for `time.monotonic`."""
        return self.now_s

    async def sleep(self, dt_s: float) -> None:
        """Advances `now_s` by `dt_s` (never backwards) and yields once to the event loop.

        @param dt_s Seconds to advance; negative values are recorded and treated as zero.
        """
        self.sleeps.append(dt_s)
        self.now_s += max(dt_s, 0.0)
        await asyncio.sleep(0)


@dataclasses.dataclass
class LoopHarness:
    """A ControlLoop plus every fake it was built from, so tests can reach both sides.

    Attributes:
    - loop: The assembled `robots.control.ControlLoop`.
    - client: The FakeClient behind `loop`.
    - clock: The ManualClock injected as `clock` / `sleep`.
    - robots: FakeRobot per arm, keyed "left" / "right".
    - arms: The `FollowerArm` wrappers, keyed the same way.
    - sources: FakeCaptureSource per camera, keyed by camera name.
    - readers: The `CameraReader`s, in config order.
    """

    loop: Any
    client: FakeClient
    clock: ManualClock
    robots: dict[str, FakeRobot]
    arms: dict[str, Any]
    sources: dict[str, FakeCaptureSource]
    readers: list[Any]


def make_bgr_frame(shape: tuple[int, int, int] = FRAME_SHAPE, bgr: tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
    """Builds a solid uint8 frame in OpenCV's BGR order.

    The default is pure blue in BGR, so a correct BGR to RGB conversion puts the 255 in the
    last channel and a missing one leaves it in the first.

    @param shape `(rows, columns, 3)`.
    @param bgr Per-channel value in BGR order.
    @return The frame, `np.uint8`.
    """
    frame = np.empty(shape, dtype=np.uint8)
    frame[...] = bgr
    return frame


def make_chunk(horizon: int, base: float = 0.0) -> dict[str, list[list[float]]]:
    """Builds a serve action dict: `q_r_chunk` and `q_l_chunk`, each `[H][7]` nested lists of floats.

    Row `i` holds `base + i` in the six joint columns and `((base + i) % 10) / 10` in the
    seventh (gripper) column, so rows and chunks are distinguishable after they pass through
    the adapter. Lists rather than arrays because that is what JSON, and so the real client,
    hands back.

    @param horizon Rows per key.
    @param base Value of the first row.
    @return The chunk keyed `q_r_chunk`, `q_l_chunk`; the two are separate lists.
    """
    rows = [base + i for i in range(horizon)]
    chunk = [[float(row)] * ARM_DOF + [round((row % 10) / 10, 6)] for row in rows]
    return {"q_r_chunk": [list(row) for row in chunk], "q_l_chunk": [list(row) for row in chunk]}


def yam_arm(name: str, robot: FakeRobot) -> Any:
    """A `FollowerArm` over a #FakeRobot through the YAM's driver (`robots.yam.I2rtDriver`), its home at zeros.

    @param name `"left"` or `"right"`.
    @param robot The fake robot.
    @return The arm.
    """
    from robots import followers, yam

    return followers.FollowerArm(name, yam.I2rtDriver(name, robot), home_rad=np.zeros(ARM_DOF))


def make_loop(
    control_hz: float = LOOP_DEFAULTS["control_hz"],
    action_horizon: int = LOOP_DEFAULTS["action_horizon"],
    action_timeout_s: float = LOOP_DEFAULTS["action_timeout_s"],
    stale_after_s: float = LOOP_DEFAULTS["stale_after_s"],
    max_stall_ticks: int = LOOP_DEFAULTS["max_stall_ticks"],
    first_chunk_timeout_s: float = LOOP_DEFAULTS["first_chunk_timeout_s"],
    refill_at_remaining: int | None = LOOP_DEFAULTS["refill_at_remaining"],
    ramp_s: float = LOOP_DEFAULTS["ramp_s"],
    ramp_max_rad_s: float = LOOP_DEFAULTS["ramp_max_rad_s"],
    max_ee_speed_m_s: float = LOOP_DEFAULTS["max_ee_speed_m_s"],
    max_gripper_step: float = LOOP_DEFAULTS["max_gripper_step"],
    grip_force_cap: float = LOOP_DEFAULTS["grip_force_cap"],
    kinematics: dict[str, Any] | None = None,
    trace_path: str | None = None,
    command_hz: float = LOOP_DEFAULTS["command_hz"],
    command_interp_max_s: float = LOOP_DEFAULTS["command_interp_max_s"],
    command_thread: bool = LOOP_DEFAULTS["command_thread"],
    inference_log_path: str | None = None,
    run_number: int = 1,
    recorder: Any = None,
    client: FakeClient | None = None,
    clock: ManualClock | None = None,
    robots: dict[str, FakeRobot] | None = None,
    camera_names: tuple[str, ...] = CAMERA_NAMES,
    task: str = TASK,
    sys0: Any = None,
) -> LoopHarness:
    """Assembles a ControlLoop over fakes with the defaults the tests agree on.

    Imports `robots.control`, `robots.followers` and `robots.cameras` here rather than at module
    level so the fakes remain importable while the library is still being written.

    @param control_hz Tick rate passed to the loop.
    @param action_horizon Rows per chunk; also sets the client's horizon when the client is built here.
    @param action_timeout_s Refill deadline passed to the loop.
    @param stale_after_s Frame age limit passed to the loop and every reader.
    @param max_stall_ticks Stall budget passed to the loop.
    @param first_chunk_timeout_s First-chunk deadline passed to the loop.
    @param refill_at_remaining Refill point passed to the loop: pinned rows (0 here, so the unit tests read exact
        rows), or None to size it from the measured round trip.
    @param ramp_s Ramp length passed to the loop; one tick by default (see LOOP_DEFAULTS).
    @param ramp_max_rad_s Ramp speed cap passed to the loop; none by default.
    @param max_ee_speed_m_s End-effector speed cap passed to the loop; off by default.
    @param max_gripper_step Gripper rate clamp passed to the loop; off by default.
    @param grip_force_cap Grip force cap passed to the loop; off by default.
    @param kinematics Per-arm kinematics passed to the loop; none by default, so the joints are never clamped.
    @param trace_path Trace file passed to the loop; off by default.
    @param command_hz Command write rate passed to the loop; one per tick by default (see LOOP_DEFAULTS).
    @param command_interp_max_s Longest run window passed to the loop; one command period by default.
    @param command_thread Whether the loop writes on its own thread; off by default so step() writes synchronously.
    @param inference_log_path JSON Lines inference log passed to the loop; off by default.
    @param run_number Run ordinal passed to the loop, stamped into the inference log.
    @param recorder The `robots.rollouts.Recorder` passed to the loop; none by default, so no test writes an episode.
    @param client A prepared FakeClient; a fresh one at `action_horizon` when omitted.
    @param clock A prepared ManualClock; a fresh one at 0.0 when omitted.
    @param robots FakeRobot per arm keyed "left" / "right"; fresh ones when omitted.
    @param camera_names Cameras to wire up, one FakeCaptureSource each; serve roles unless a test wants otherwise.
    @param task Language prompt passed to the loop.
    @param sys0 System 0's models passed to the loop (`robots.sys0.motors.Models`); off by default.
    @return The #LoopHarness holding the loop and its fakes.
    """
    from robots import cameras, control

    if client is None:
        client = FakeClient(horizon=action_horizon)
    if clock is None:
        clock = ManualClock()
    if robots is None:
        robots = {"left": FakeRobot(), "right": FakeRobot()}
    arms = {name: yam_arm(name, robot) for name, robot in robots.items()}
    sources = {name: FakeCaptureSource() for name in camera_names}
    readers = [cameras.CameraReader(name, source, stale_after_s) for name, source in sources.items()]
    loop = control.ControlLoop(
        followers=arms,
        cameras=readers,
        client=client,
        task=task,
        control_hz=control_hz,
        action_horizon=action_horizon,
        action_timeout_s=action_timeout_s,
        refill_at_remaining=refill_at_remaining,
        stale_after_s=stale_after_s,
        max_stall_ticks=max_stall_ticks,
        first_chunk_timeout_s=first_chunk_timeout_s,
        ramp_s=ramp_s,
        ramp_max_rad_s=ramp_max_rad_s,
        max_ee_speed_m_s=max_ee_speed_m_s,
        max_gripper_step=max_gripper_step,
        grip_force_cap=grip_force_cap,
        kinematics=kinematics,
        trace_path=trace_path,
        command_hz=command_hz,
        command_interp_max_s=command_interp_max_s,
        command_thread=command_thread,
        sys0=sys0,
        inference_log_path=inference_log_path,
        run_number=run_number,
        recorder=recorder,
        clock=clock,
        sleep=clock.sleep,
    )
    return LoopHarness(
        loop=loop,
        client=client,
        clock=clock,
        robots=robots,
        arms=arms,
        sources=sources,
        readers=readers,
    )


def make_app_config(**overrides: Any) -> Any:
    """Builds a `robots.config.AppConfig` with every required key at a test value; `overrides` replace fields.

    Two arms on `can_left` / `can_right`, three OpenCV cameras named after the serve roles, the
    #TASK prompt and the station's 0.6 m/s end-effector cap, so `run.policy_config` accepts it.

    @param overrides Field values replacing the defaults, e.g. `task="..."` or `trace_path="..."`.
    @return The config.
    """
    from robots import config

    fields: dict[str, Any] = {
        "follower_l": config.ArmConfig(can_channel="can_left"),
        "follower_r": config.ArmConfig(can_channel="can_right"),
        "cameras": [
            config.CameraConfig(name=name, device=i, width=64, height=48) for i, name in enumerate(CAMERA_NAMES)
        ],
        "task": TASK,
        "control_hz": LOOP_DEFAULTS["control_hz"],
        "action_horizon": LOOP_DEFAULTS["action_horizon"],
        "action_timeout_s": LOOP_DEFAULTS["action_timeout_s"],
        "stale_after_s": LOOP_DEFAULTS["stale_after_s"],
        "max_stall_ticks": LOOP_DEFAULTS["max_stall_ticks"],
        "max_ee_speed_m_s": 0.6,
    }
    return config.AppConfig(**{**fields, **overrides})


## What the credentials file of every test says unless the test writes its own: a key and the three URLs on example
## hosts.
CREDENTIALS_TEXT = (
    "DREAM_GATEWAY_URL=https://api.example.com\nDREAM_API_URL=https://platform.example.com\n"
    "DREAM_DATA_URL=https://data.example.com\nDREAM_API_KEY=test-key\n"
)


@pytest.fixture(autouse=True)
def credentials_path(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Points `dream.credentials` at a file of this test's own (in a directory apart from `tmp_path`, which some
    tests expect empty), holding #CREDENTIALS_TEXT, and clears the four variables, so a developer's real
    `~/.dreamlabs/credentials` and their own settings never reach a test. A test that wants other credentials
    rewrites the file or sets a variable.

    @return The file.
    """
    for name in ("DREAM_GATEWAY_URL", "DREAM_API_URL", "DREAM_DATA_URL", "DREAM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path_factory.mktemp("account") / "credentials"
    path.write_text(CREDENTIALS_TEXT, encoding="utf-8")
    monkeypatch.setenv("DREAM_CREDENTIALS", str(path))
    return path


@pytest.fixture(autouse=True)
def home_dir(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Points `~` at an empty directory of this test's own, so a run that records (`rollouts.dir` defaults to
    `~/.dreamlabs/rollouts`) never writes into a developer's real home.

    @return The directory.
    """
    path = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(path))
    return path


@pytest.fixture
def fake_robot() -> FakeRobot:
    """A FakeRobot at rest with neither disengage method."""
    return FakeRobot()


@pytest.fixture
def fake_capture_source() -> FakeCaptureSource:
    """A FakeCaptureSource repeating the default blue BGR frame."""
    return FakeCaptureSource()


@pytest.fixture
def fake_client() -> FakeClient:
    """A FakeClient answering immediately with chunks of LOOP_DEFAULTS["action_horizon"] rows."""
    return FakeClient(horizon=LOOP_DEFAULTS["action_horizon"])


@pytest.fixture
def manual_clock() -> ManualClock:
    """A ManualClock starting at 0.0 s."""
    return ManualClock()


@pytest.fixture
def loop_harness() -> Iterator[LoopHarness]:
    """A default make_loop() harness; stops any camera reader still running on teardown."""
    harness = make_loop()
    yield harness
    # A test that raised before __aexit__ ran would otherwise leave daemon threads spinning;
    # a reader that never started has no thread to join.
    for reader in harness.readers:
        if reader.is_alive():
            reader.stop()


@pytest.fixture(autouse=True)
def quick_capture_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    """A source that stops delivering ends its reader within a few reads, not the production 2 s: the tests that
    fail a source on purpose (`FakeCaptureSource.fail()`, a timing-out fake pipeline) wait on `is_alive()`."""
    monkeypatch.setattr(cameras, "CAPTURE_MISS_TOLERANCE_S", 0.05)


@pytest.fixture(autouse=True)
def any_model_fits(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test asks a real account whether a model fits the body's embodiment (`robots.bodies.check_model`);
    `test_custom_body.py` tests the check itself."""
    from robots import bodies

    monkeypatch.setattr(bodies, "check_model", lambda app, checkpoint: None)


@pytest.fixture(autouse=True)
def no_training_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test asks a real data service where a fine-tune's training data is; the setup tests that want it pass
    `ask_training_data`."""

    def refuse(model: str, **_: object) -> dict[str, object]:
        raise DreamServerError(f"No fine-tuning job of yours made {model!r}.", code="not_found")

    monkeypatch.setattr(fine_tuning, "training_data", refuse)
