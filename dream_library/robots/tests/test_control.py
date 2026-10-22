"""Behaviour tests for `robots.control.ControlLoop` against the fakes in `conftest.py`.

Every test drives the loop with the injected `ManualClock`, so tick time never comes from the
wall clock. Worker threads (the fake policy in `asyncio.to_thread`, the fake capture threads)
still do real work, and a test waits for them through the event loop rather than by sleeping
for a control period. Each docstring names the behaviour it pins.
"""

import asyncio
import base64
import collections
import dataclasses
import logging
import pathlib
import re
import threading
import time
from collections.abc import Callable, Container, Iterator, Mapping
from typing import Any

import conftest
import cv2
import dream
import numpy as np
import pytest
from shared.contracts import envelope

from robots import adapter, cameras, control, errors, gripper_force, types

## Refill deadline for tests that let a gated refill run late: `_wait_for_refill` waits this long
## on the wall clock the first time round, so it stays small.
STALL_TIMEOUT_S = 0.1

## Longest a test waits for a worker thread to hand its result back to the event loop.
THREAD_WAIT_S = 2.0

## How long the blocking capture source below parks a read before giving up.
BLOCKING_READ_S = 0.3

## Hold vector a never-commanded FakeRobot at rest produces: zeros for the joints, its gripper.
HOLD_COMMAND = [0.0] * conftest.ARM_DOF + [0.5]

## Wall-clock slice each injected sleep gives the capture threads; see #YieldingClock.
THREAD_SLICE_S = 0.001

## Each arm's joints on the test station's arms, which the trace has a column per.
YAM_JOINTS = {"left": conftest.ARM_DOF, "right": conftest.ARM_DOF}

## Per-channel slack for a solid colour after a quality-85 JPEG round trip.
JPEG_TOLERANCE = 4

## The three dream errors the loop must not retry: a retry would fail the same way.
UNRECOVERABLE_ERRORS = [
    dream.DreamAuthError("DreamAPI refused the inference request: DA-1000 UNAUTHORIZED (unknown_key)", code="DA-1000"),
    dream.DreamConfigError("config key 'checkpoint' must be a non-empty str"),
    dream.DreamObservationError("observation is missing keys declared by the configured embodiment"),
]


## Retry hint the warming refusals below carry: two of the one-second ticks those tests step with.
WARMING_RETRY_S = 2.0


## Metres the linear stand-in kinematics moves the end effector per radian of each of the first three joints.
FAKE_ARM_M_PER_RAD = 1.0


class StopTicksError(Exception):
    """Raised from a patched `step` to end `run()` after a chosen number of ticks."""


class LinearKinematics:
    """A stand-in `Kinematics`: the end effector is the first three joints, as metres, so a joint delta of `d` in
    each of them is a displacement of `|d| * sqrt(3)` and the clamp's scaling is exact in one pass."""

    def ee_position(self, joints_rad: np.ndarray) -> np.ndarray:
        return np.asarray(joints_rad[:3], dtype=np.float64) * FAKE_ARM_M_PER_RAD


class StillKinematics:
    """A stand-in `Kinematics` whose end effector never moves, so the joints are never speed-clamped."""

    def ee_position(self, joints_rad: np.ndarray) -> np.ndarray:
        return np.zeros(3)


def linear_kinematics() -> dict[str, LinearKinematics]:
    """One `LinearKinematics` per arm, as `ControlLoop(kinematics=...)` takes them."""
    return {arm: LinearKinematics() for arm in control.ARMS}


class AnswerOnceClient(conftest.FakeClient):
    """A FakeClient that answers its first call at once and parks every later one behind `gate`.

    Stalls are counted only once a chunk has landed, so a stall test needs one answer
    before the policy goes silent; `run()` leaves no room to close a gate by hand between ticks.
    """

    def __init__(self, horizon: int, gate: threading.Event) -> None:
        super().__init__(horizon=horizon)
        self._gate_after_first = gate

    def get_action(self, *, commanded: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Answers as FakeClient does, then gates every later call.

        @param commanded The joints last commanded per chunk key; recorded.
        @return The next chunk.
        """
        chunk = super().get_action(commanded=commanded)
        self.gate = self._gate_after_first
        return chunk


class WarmingClient(conftest.FakeClient):
    """A FakeClient that refuses the calls whose zero-based index is in `refused` as a cold pool does.

    Each refusal is a retryable `DreamServerError` `DA-2000 NO_CAPACITY (pool_warming)` carrying
    `retry_after_s`; every other call answers a chunk. Refused calls still count in `calls` and
    still advance the chunk numbering, as the real client's requests do.
    """

    def __init__(self, horizon: int, refused: Container[int], retry_after_s: float | None) -> None:
        super().__init__(horizon=horizon)
        self.refused = refused
        self.retry_after_s = retry_after_s

    def get_action(self, *, commanded: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Refuses or answers call number `calls`.

        @param commanded The joints last commanded per chunk key; recorded.
        @return The next chunk.
        @throws dream.DreamServerError This call's index is in `refused`.
        """
        if self.calls in self.refused:
            self.raise_once = dream.DreamServerError(
                "DreamAPI refused the inference request: DA-2000 NO_CAPACITY (pool_warming)",
                code="DA-2000",
                reason="pool_warming",
                retryable=True,
                retry_after_s=self.retry_after_s,
            )
        return super().get_action(commanded=commanded)


class FailingClient(conftest.FakeClient):
    """A FakeClient whose call number `calls` fails with `fail(calls)` when that returns an exception."""

    def __init__(self, horizon: int, fail: Callable[[int], BaseException | None]) -> None:
        super().__init__(horizon=horizon)
        self.fail = fail

    def get_action(self, *, commanded: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Fails or answers call number `calls`."""
        self.raise_once = self.fail(self.calls)
        return super().get_action(commanded=commanded)


MISSING_CAMERA = dream.DreamServerError(
    "DreamAPI refused the inference request: DA-5000 SERVE_FAULT (error): inference error: observation is missing "
    "['video.overhead']",
    code="DA-5000",
    reason="error",
    retryable=True,
)


class YieldingClock(conftest.ManualClock):
    """A ManualClock whose sleep also parks the event loop for #THREAD_SLICE_S of wall time.

    Tick time still comes from `now_s`, never from the wall clock. The slice exists because the
    capture threads run on real time: `__aenter__` polls for first frames with the injected
    sleep, and a sleep that only yields once gives three threads no chance to publish before the
    500 polls of the (manual) first-frame timeout run out.
    """

    async def sleep(self, dt_s: float) -> None:
        await super().sleep(dt_s)
        await asyncio.sleep(THREAD_SLICE_S)


class BlockingCaptureSource:
    """A capture source whose read() parks for #BLOCKING_READ_S and then reports failure.

    Keeps the capture thread alive (blocked in read) long enough for `__aenter__` to time out on
    the ManualClock, and still lets `stop()`'s join return without waiting on a real device.
    """

    def __init__(self) -> None:
        self.released = False
        self._gate = threading.Event()

    def read(self) -> tuple[bool, np.ndarray | None]:
        self._gate.wait(BLOCKING_READ_S)
        return False, None

    def release(self) -> None:
        self.released = True


@pytest.fixture
def build() -> Iterator[Callable[..., conftest.LoopHarness]]:
    """A `make_loop` factory whose harnesses get their capture threads stopped on teardown."""
    built: list[conftest.LoopHarness] = []

    def factory(**kwargs: Any) -> conftest.LoopHarness:
        kwargs.setdefault("clock", YieldingClock())
        harness = conftest.make_loop(**kwargs)
        built.append(harness)
        return harness

    yield factory
    for harness in built:
        for reader in harness.readers:
            if reader.is_alive():
                reader.stop()


async def wait_for_refill_task() -> None:
    """Awaits every task the loop has in flight (the one refill), bounded by #THREAD_WAIT_S."""
    current = asyncio.current_task()
    pending = {task for task in asyncio.all_tasks() if task is not current}
    if pending:
        await asyncio.wait(pending, timeout=THREAD_WAIT_S)


def joint_values(robot: conftest.FakeRobot) -> list[float]:
    """First joint of every command the robot received; FakeClient chunks make this the row index."""
    return [float(command[0]) for command in robot.commands]


def stale_now_s() -> float:
    """A tick time far ahead of any frame the capture threads stamp with the wall clock."""
    return time.monotonic() + 10.0


def records_with(caplog: pytest.LogCaptureFixture, text: str) -> list[logging.LogRecord]:
    return [record for record in caplog.records if text in record.getMessage()]


# Tick order


def test_tick_reads_arms_then_cameras_then_publishes_then_refills_then_writes(
    build: Callable[..., conftest.LoopHarness], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Arms are read before the cameras, the snapshot is published before the refill,
    and both arms are written last."""
    calls: list[str] = []
    to_dream = adapter.YAM_VIDEO_WIRE.to_dream

    def to_dream_recording(obs: Any, task: str) -> Any:
        loop = harness.loop
        published = loop.state is not None and loop.state is obs.state and loop.video is obs.images
        calls.append("refill:published" if published else "refill:unpublished")
        return to_dream(obs, task)

    # The loop binds the wire's to_dream when it is built, so the wire is patched first.
    monkeypatch.setattr(adapter.YAM_VIDEO_WIRE, "to_dream", to_dream_recording)
    harness = build()
    loop = harness.loop

    def recording(label: str, original: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            calls.append(label)
            return original(*args, **kwargs)

        return wrapper

    for name, arm in harness.arms.items():
        arm.read = recording(f"read:{name}", arm.read)
        arm.write = recording(f"write:{name}", arm.write)
    for reader in harness.readers:
        reader.is_alive = recording(f"camera:{reader.name}", reader.is_alive)

    async def scenario() -> list[str]:
        async with loop:
            calls.clear()
            await loop.step(harness.clock())
            # Snapshot before __aexit__ adds its hold, which also goes through write().
            return list(calls)

    tick_calls = asyncio.run(scenario())

    assert tick_calls == [
        "read:left",
        "read:right",
        "camera:top",
        "camera:left",
        "camera:right",
        "refill:published",
        "write:left",
        "write:right",
    ]


def test_action_dispatch_writes_the_popped_action_to_both_arms(build: Callable[..., conftest.LoopHarness]) -> None:
    """The popped action is split into left and right targets and both arms are
    commanded with it, in chunk order."""
    harness = build()

    async def scenario() -> None:
        async with harness.loop:
            await harness.loop.step(0.0)
            await harness.loop.step(0.1)

    asyncio.run(scenario())

    for robot in harness.robots.values():
        # The tick writes come first; __aexit__ adds one hold at the end.
        np.testing.assert_allclose(robot.commands[0], [0.0] * 6 + [0.0])
        np.testing.assert_allclose(robot.commands[1], [1.0] * 6 + [0.1])
    assert harness.client.calls == 1


def test_state_carries_the_commanded_targets_after_a_write(build: Callable[..., conftest.LoopHarness]) -> None:
    """The published state reports the commanded joints and gripper once an arm was written."""
    harness = build()

    async def scenario() -> tuple[Any, Any]:
        async with harness.loop:
            await harness.loop.step(0.0)
            before = harness.loop.state
            await harness.loop.step(0.1)
            return before, harness.loop.state

    before, after = asyncio.run(scenario())

    assert before.left.commanded_rad is None and before.left.commanded_gripper is None
    np.testing.assert_allclose(after.left.commanded_rad, [0.0] * 6)
    assert after.left.commanded_gripper == 0.0
    np.testing.assert_allclose(after.right.commanded_rad, [0.0] * 6)


def commanded_values(client: conftest.FakeClient) -> list[dict[str, list[float]]]:
    """The `commanded` of every get_action() call as plain lists, per chunk key."""
    return [{key: np.asarray(joints).tolist() for key, joints in (sent or {}).items()} for sent in client.commanded]


def test_every_refill_carries_the_joints_of_the_last_written_row(build: Callable[..., conftest.LoopHarness]) -> None:
    """The client anchors each new chunk on the joints the loop last commanded: the observation is read before the
    tick submits its row, so a refill carries the previous tick's row per chunk key (`adapter.commanded_joints`),
    nothing before the first row, and the same on a refill that starts a new session first."""
    client = WarmingClient(horizon=1, refused={1}, retry_after_s=None)
    harness = build(action_horizon=1, max_stall_ticks=3, client=client)

    async def scenario() -> None:
        async with harness.loop:
            for tick in range(4):
                await harness.loop.step(float(tick))

    asyncio.run(scenario())

    # Tick 0 asks before any write; tick 1 asks with row 0 written and is refused, so tick 1 holds row 0; tick 2's
    # retry resets the session first and still carries row 0; it answers row 2, which tick 3's request carries.
    assert client.reset_after_calls == [2]
    row = {0: [0.0] * 6, 2: [2.0] * 6}
    assert commanded_values(client) == [
        {},
        {"q_r_chunk": row[0], "q_l_chunk": row[0]},
        {"q_r_chunk": row[0], "q_l_chunk": row[0]},
        {"q_r_chunk": row[2], "q_l_chunk": row[2]},
    ]
    assert joint_values(harness.robots["left"])[:3] == [0.0, 0.0, 2.0]


def test_commanded_joints_are_the_row_last_submitted() -> None:
    """Before the first row there is nothing to anchor on; after it, each arm's joints of that row, float64 radians."""
    right = types.ArmCommand(positions_rad=np.arange(6, dtype=np.float32), gripper=0.5)
    left = types.ArmCommand(positions_rad=-np.arange(6, dtype=np.float32), gripper=0.5)

    commanded = adapter.commanded_joints(types.Action(left=left, right=right))

    assert adapter.commanded_joints(None) == {}
    assert list(commanded) == ["q_r_chunk", "q_l_chunk"]
    assert commanded["q_r_chunk"].dtype == np.float64 and commanded["q_r_chunk"].tolist() == list(range(6))
    assert commanded["q_l_chunk"].tolist() == [-float(i) for i in range(6)]


# Prefetch threshold and single flight


def test_refill_starts_only_when_buffer_is_empty_at_threshold_zero(build: Callable[..., conftest.LoopHarness]) -> None:
    """With refill_at_remaining=0 the refill is due only when the buffer is empty."""
    harness = build(action_horizon=4, refill_at_remaining=0)
    calls_per_tick: list[int] = []

    async def scenario() -> None:
        async with harness.loop:
            for tick in range(5):
                await harness.loop.step(tick * 0.1)
                calls_per_tick.append(harness.client.calls)

    asyncio.run(scenario())

    assert calls_per_tick == [1, 1, 1, 1, 2]
    assert joint_values(harness.robots["left"])[:5] == [0, 1, 2, 3, 4]


def test_refill_starts_when_buffer_length_reaches_threshold_two(build: Callable[..., conftest.LoopHarness]) -> None:
    """With refill_at_remaining=2 the refill starts once len(buffer) <= 2 while the loop
    keeps draining, and the new chunk replaces the remainder when it lands, from the row due then."""
    harness = build(control_hz=10.0, action_horizon=4, refill_at_remaining=2)
    calls_per_tick: list[int] = []

    async def scenario() -> None:
        async with harness.loop:
            for tick in range(4):
                await harness.loop.step(tick * 0.1)
                # A prefetch runs in a worker thread; wait for it so `calls` reflects this tick.
                await wait_for_refill_task()
                calls_per_tick.append(harness.client.calls)

    asyncio.run(scenario())

    # Tick 1 pops with 3 left (3 > 2, no refill); tick 2 pops and sees 2 <= 2, so it prefetches.
    assert calls_per_tick == [1, 1, 2, 2]
    # Rows 0, 1, 2 of chunk 0, then row 1 of chunk 1 (value 5): the prefetched chunk replaced the
    # one action left over, and its row 0 was due on the tick it was still in flight.
    assert joint_values(harness.robots["left"])[:4] == [0, 1, 2, 5]


def test_only_one_refill_is_in_flight(
    build: Callable[..., conftest.LoopHarness], caplog: pytest.LogCaptureFixture
) -> None:
    """The loop never has more than one get_action outstanding; the FakeClient
    would raise DreamError on a concurrent call."""
    gate = threading.Event()
    gate.set()
    client = conftest.FakeClient(horizon=4, gate=gate)
    harness = build(action_horizon=4, refill_at_remaining=10, client=client)

    async def scenario() -> None:
        async with harness.loop:
            await harness.loop.step(0.0)
            gate.clear()
            # Every tick is a due tick at this threshold; only the first may start a call.
            for tick in range(1, 4):
                await harness.loop.step(tick * 0.1)
            gate.set()
            await wait_for_refill_task()

    with caplog.at_level(logging.WARNING, logger="robots.control"):
        asyncio.run(scenario())

    assert client.calls == 2
    assert not records_with(caplog, "get_action failed")
    assert joint_values(harness.robots["right"])[:4] == [0, 1, 2, 3]


@pytest.mark.parametrize("periods_late", [1, 2, 3])
def test_chunk_collected_after_its_observation_installs_only_the_rows_still_ahead(
    build: Callable[..., conftest.LoopHarness], caplog: pytest.LogCaptureFixture, periods_late: int
) -> None:
    """Row k of a chunk is the target for observed_at_s + k * period. With rows commanded every
    period while the refill is in flight, a chunk collected k periods after its observation has had k rows
    executed, so rows k..H-1 are installed and the first command from it is row k."""
    gate = threading.Event()
    gate.set()
    client = conftest.FakeClient(horizon=4, gate=gate)
    harness = build(control_hz=10.0, action_horizon=4, refill_at_remaining=3, client=client)

    async def scenario() -> None:
        async with harness.loop:
            # Chunk 0 lands on the tick that observed it; row 0 goes out.
            await harness.loop.step(0.0)
            gate.clear()
            # Three rows left: the prefetch starts, observing this tick; row 1 goes out.
            await harness.loop.step(0.1)
            # Rows keep going out every period while the refill is in flight.
            for period in range(1, periods_late):
                await harness.loop.step(0.1 + period * 0.1)
            gate.set()
            await wait_for_refill_task()
            await harness.loop.step(0.1 + periods_late * 0.1)

    with caplog.at_level(logging.INFO, logger="robots.control"):
        asyncio.run(scenario())

    assert client.calls >= 2
    # Rows 0..periods_late of chunk 0, then row `periods_late` of chunk 1 (values 4..7).
    expected = list(range(periods_late + 1)) + [4 + periods_late]
    assert joint_values(harness.robots["left"])[: len(expected)] == expected
    (landed,) = records_with(caplog, "chunk 2:")
    assert (
        f"4 actions, {periods_late} executed since its observation, {periods_late} due by wall time, "
        f"after {periods_late * 100} ms"
    ) in landed.getMessage()


def test_chunk_collected_after_a_hold_skips_only_the_rows_that_were_executed(
    build: Callable[..., conftest.LoopHarness], caplog: pytest.LogCaptureFixture
) -> None:
    """The arm advances along the plan only while rows are commanded. When the clock jumps three
    periods with no tick in between (the arm held), the chunk resumes one row in, not three: skipping by wall
    time would jump the arm to where it would have been had it kept moving."""
    gate = threading.Event()
    gate.set()
    client = conftest.FakeClient(horizon=4, gate=gate)
    harness = build(control_hz=10.0, action_horizon=4, refill_at_remaining=3, client=client)

    async def scenario() -> None:
        async with harness.loop:
            await harness.loop.step(0.0)
            gate.clear()
            await harness.loop.step(0.1)
            gate.set()
            await wait_for_refill_task()
            await harness.loop.step(0.4)

    with caplog.at_level(logging.INFO, logger="robots.control"):
        asyncio.run(scenario())

    assert joint_values(harness.robots["left"])[:3] == [0, 1, 5]
    (landed,) = records_with(caplog, "chunk 2:")
    assert "1 executed since its observation, 3 due by wall time" in landed.getMessage()


def test_chunk_whose_whole_horizon_is_past_is_discarded_with_a_warning(
    build: Callable[..., conftest.LoopHarness], caplog: pytest.LogCaptureFixture
) -> None:
    """A chunk whose every row the arms already executed while it was computed has no row still ahead; it
    is discarded with a WARNING naming the latency and the horizon, and the buffer it would have replaced is kept."""
    harness = build(control_hz=10.0, action_horizon=4)
    loop = harness.loop
    kept = adapter.from_dream(conftest.make_chunk(4, base=100.0), observed_at_s=0.0, horizon=4).actions
    loop._buffer = collections.deque(kept)
    loop._rows_since_observation = 4  # the arms ran four rows since the observation the late chunk answers
    late = adapter.from_dream(conftest.make_chunk(4), observed_at_s=0.0, horizon=4)

    with caplog.at_level(logging.INFO, logger="robots.control"):
        loop._install_chunk(late, now_s=0.4, elapsed_s=0.4)

    (discarded,) = records_with(caplog, "discarding action chunk that landed")
    assert discarded.levelno == logging.WARNING
    assert "400 ms after its observation" in discarded.getMessage()
    assert "4-row horizon (400 ms)" in discarded.getMessage()
    assert list(loop._buffer) == kept
    assert loop.chunk_count == 0


def test_a_chunk_slower_than_its_horizon_while_the_arms_held_installs_whole(
    build: Callable[..., conftest.LoopHarness], caplog: pytest.LogCaptureFixture
) -> None:
    """The client's sync pacing on a slow link. The inference takes longer than the chunk's whole horizon,
    but the arms held through it and executed none of its rows, so every row is still ahead of the arm: the chunk
    installs from row 0 instead of being discarded as past."""
    harness = build(control_hz=10.0, action_horizon=4, refill_at_remaining=0)
    answer = harness.client.get_action

    def slow_answer(*, commanded: Mapping[str, Any] | None = None) -> dict[str, Any]:
        # Six periods of inference against a four-row chunk, on the loop's own clock.
        harness.clock.now_s += 0.6
        return answer(commanded=commanded)

    harness.client.get_action = slow_answer

    async def scenario() -> None:
        async with harness.loop:
            for tick in range(8):
                await harness.loop.step(tick * 0.1)

    with caplog.at_level(logging.INFO, logger="robots.control"):
        asyncio.run(scenario())

    assert not records_with(caplog, "discarding action chunk")
    assert harness.client.calls == 2
    assert joint_values(harness.robots["left"])[:8] == [0, 1, 2, 3, 4, 5, 6, 7]


def test_sync_path_resumes_the_plan_where_the_arm_stopped(build: Callable[..., conftest.LoopHarness]) -> None:
    """With refill_at_remaining 0 the tick blocks on the refill and the arms hold. The rows due
    while they held were never executed, so the chunk is installed from row 0: the plan pauses for the
    inference latency and resumes where the arm is, instead of jumping ahead."""
    harness = build(control_hz=10.0, action_horizon=4)
    answer = harness.client.get_action

    def slow_answer(*, commanded: Mapping[str, Any] | None = None) -> dict[str, Any]:
        # Two periods of inference, on the loop's own clock.
        harness.clock.now_s += 0.2
        return answer(commanded=commanded)

    harness.client.get_action = slow_answer

    async def scenario() -> None:
        async with harness.loop:
            for tick in range(8):
                await harness.loop.step(tick * 0.1)

    asyncio.run(scenario())

    assert harness.client.calls == 2
    assert joint_values(harness.robots["left"])[:8] == [0, 1, 2, 3, 4, 5, 6, 7]


def test_observed_at_s_is_the_timestamp_of_the_observation_the_refill_sent(
    build: Callable[..., conftest.LoopHarness], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The chunk's observed_at_s is the tick time of the observation that was sent,
    not the tick on which the result was collected."""
    observed: list[float] = []
    from_dream = adapter.YAM_VIDEO_WIRE.from_dream

    def from_dream_recording(action: Any, observed_at_s: float, horizon: int) -> Any:
        observed.append(observed_at_s)
        return from_dream(action, observed_at_s, horizon)

    # The loop binds the wire's from_dream when it is built, so the wire is patched first.
    monkeypatch.setattr(adapter.YAM_VIDEO_WIRE, "from_dream", from_dream_recording)
    harness = build(action_horizon=4, refill_at_remaining=2)

    async def scenario() -> None:
        async with harness.loop:
            for now_s in (5.0, 5.1, 5.2):
                await harness.loop.step(now_s)
            await wait_for_refill_task()
            await harness.loop.step(5.3)

    asyncio.run(scenario())

    assert observed == [5.0, 5.2]
    assert harness.client.observations[1]["state.right_arm"].dtype == np.float32


# Sync path: stalls and the refill deadline


def test_empty_buffer_with_pending_refill_holds_and_counts_stalls(
    build: Callable[..., conftest.LoopHarness], caplog: pytest.LogCaptureFixture
) -> None:
    """Buffer empties before the next chunk arrives -> hold last command, log the stall, and
    stop with ControlLoopError once max_stall_ticks is exceeded."""
    gate = threading.Event()
    client = AnswerOnceClient(horizon=1, gate=gate)
    harness = build(action_horizon=1, action_timeout_s=STALL_TIMEOUT_S, max_stall_ticks=3, client=client)

    async def scenario() -> None:
        try:
            async with harness.loop:
                for now_s in (0.0, 1.0, 2.0, 3.0):
                    await harness.loop.step(now_s)
                with pytest.raises(errors.ControlLoopError, match="max_stall_ticks=3"):
                    await harness.loop.step(4.0)
        finally:
            gate.set()
            await wait_for_refill_task()

    with caplog.at_level(logging.WARNING, logger="robots.control"):
        asyncio.run(scenario())

    assert client.calls == 2
    stall_logs = [record.getMessage() for record in records_with(caplog, "stall ")]
    assert [message[:9] for message in stall_logs] == ["stall 1/3", "stall 2/3", "stall 3/3"]
    for robot in harness.robots.values():
        # Row 0, four stall holds of it (the raising tick holds before it raises) and the __aexit__ hold.
        assert len(robot.commands) == 6
        for command in robot.commands:
            np.testing.assert_allclose(command, [0.0] * 6 + [0.0])


def test_run_stops_with_control_loop_error_after_max_stall_ticks(
    build: Callable[..., conftest.LoopHarness],
) -> None:
    """Run() propagates the ControlLoopError from a stall burst and __aexit__ holds both arms."""
    gate = threading.Event()
    client = AnswerOnceClient(horizon=1, gate=gate)
    harness = build(action_horizon=1, action_timeout_s=STALL_TIMEOUT_S, max_stall_ticks=2, client=client)

    async def scenario() -> None:
        try:
            async with harness.loop:
                await harness.loop.run()
        finally:
            gate.set()
            await wait_for_refill_task()

    with pytest.raises(errors.ControlLoopError, match="3 consecutive ticks"):
        asyncio.run(scenario())

    for robot in harness.robots.values():
        # Row 0 of the one chunk, three stall holds of it, and the __aexit__ hold.
        assert len(robot.commands) == 5
        np.testing.assert_allclose(robot.commands[-1], [0.0] * 6 + [0.0])
    assert client.closed


def test_successful_tick_resets_the_stall_count(
    build: Callable[..., conftest.LoopHarness], caplog: pytest.LogCaptureFixture
) -> None:
    """Max_stall_ticks counts consecutive stalls; a successful tick resets the count.

    Two bursts of two stalls around one commanded tick stay under max_stall_ticks=3; a cumulative
    count would raise ControlLoopError on the fourth stall.
    """
    gate = threading.Event()
    client = AnswerOnceClient(horizon=1, gate=gate)
    harness = build(action_horizon=1, action_timeout_s=STALL_TIMEOUT_S, max_stall_ticks=3, client=client)

    async def scenario() -> None:
        try:
            async with harness.loop:
                # Chunk 0 lands and is drained by this tick; every later call is gated.
                await harness.loop.step(0.0)
                # Burst one: the gated refill never lands.
                await harness.loop.step(1.0)
                await harness.loop.step(2.0)
                gate.set()
                await wait_for_refill_task()
                # The late chunk is discarded; the fresh refill answers at once and the arms move.
                await harness.loop.step(3.0)
                gate.clear()
                # Burst two: the count starts again at one.
                await harness.loop.step(4.0)
                await harness.loop.step(5.0)
        finally:
            gate.set()
            await wait_for_refill_task()

    with caplog.at_level(logging.WARNING, logger="robots.control"):
        asyncio.run(scenario())

    assert client.calls == 4
    stall_logs = [record.getMessage() for record in records_with(caplog, "stall ")]
    assert [message[:9] for message in stall_logs] == ["stall 1/3", "stall 2/3", "stall 1/3", "stall 2/3"]
    # Row 0 of chunk 2 (value 2) is the one commanded tick between the two bursts of holds.
    assert joint_values(harness.robots["left"])[3] == 2.0


def test_cold_pool_refusals_are_retried_at_the_hinted_pace_and_count_no_stall(
    build: Callable[..., conftest.LoopHarness], caplog: pytest.LogCaptureFixture
) -> None:
    """A retryable DreamServerError (a warming pool) holds the next refill for its retry_after_s,
    is logged once at WARNING with its reason, and counts no stall before the first chunk; motion
    starts with the first answer."""
    client = WarmingClient(horizon=4, refused={0, 1, 2}, retry_after_s=WARMING_RETRY_S)
    harness = build(client=client)
    calls_per_tick: list[int] = []

    async def scenario() -> None:
        async with harness.loop:
            for tick in range(7):
                await harness.loop.step(float(tick))
                calls_per_tick.append(client.calls)

    with caplog.at_level(logging.WARNING, logger="robots.control"):
        asyncio.run(scenario())

    # One call per retry hint: ticks 0, 2, 4 refused, tick 6 answered.
    assert calls_per_tick == [1, 1, 2, 2, 3, 3, 4]
    refused = records_with(caplog, f"retrying after {WARMING_RETRY_S:.1f} s")
    assert len(refused) == 3
    assert all(record.levelno == logging.WARNING and "pool_warming" in record.getMessage() for record in refused)
    assert not records_with(caplog, "stall ")
    assert not records_with(caplog, "get_action failed")
    left = harness.robots["left"].commands
    # Six holds while the pool warmed, row 0 of the first chunk, and the __aexit__ hold of it. The
    # chunk is the fourth call's, so its rows start at 12 (WarmingClient numbers refused calls too).
    assert len(left) == 8
    for command in left[:6]:
        np.testing.assert_allclose(command, HOLD_COMMAND)
    np.testing.assert_allclose(left[6], [12.0] * 6 + [0.2])


def test_refill_error_stands_while_the_pool_warms_and_clears_when_a_chunk_lands(
    build: Callable[..., conftest.LoopHarness],
) -> None:
    """`refill_error` is DreamAPI's refusal, as the client worded it, from the tick that collects it until the tick
    that collects a chunk: ticks 0, 2 and 4 ask and are refused, tick 6 asks and is answered."""
    client = WarmingClient(horizon=4, refused={0, 1, 2}, retry_after_s=WARMING_RETRY_S)
    harness = build(client=client)
    seen: list[str | None] = []

    async def scenario() -> None:
        async with harness.loop:
            assert harness.loop.refill_error is None
            for tick in range(8):
                await harness.loop.step(float(tick))
                seen.append(harness.loop.refill_error)

    asyncio.run(scenario())

    assert seen == ["DreamAPI refused the inference request: DA-2000 NO_CAPACITY (pool_warming)"] * 6 + [None, None]


def test_a_dreamapi_that_never_answers_stops_the_loop_after_first_chunk_timeout_s(
    build: Callable[..., conftest.LoopHarness], caplog: pytest.LogCaptureFixture
) -> None:
    client = FailingClient(horizon=4, fail=lambda _: dream.DreamTimeout("get_action timed out after 3.0 s"))
    harness = build(client=client, first_chunk_timeout_s=5.0)

    async def scenario() -> None:
        async with harness.loop:
            for tick in range(6):
                await harness.loop.step(float(tick))
            with pytest.raises(errors.ControlLoopError, match="first_chunk_timeout_s=5 s"):
                await harness.loop.step(6.0)

    with caplog.at_level(logging.INFO, logger="robots.control"):
        asyncio.run(scenario())

    assert client.calls == 7
    assert not records_with(caplog, "chunk 1:")
    assert not records_with(caplog, "stall ")
    assert client.closed
    for robot in harness.robots.values():
        assert len(robot.commands) == 8
        for command in robot.commands:
            np.testing.assert_allclose(command, HOLD_COMMAND)


def test_retry_hint_is_respected_and_stalls_count_once_a_chunk_has_landed(
    build: Callable[..., conftest.LoopHarness], caplog: pytest.LogCaptureFixture
) -> None:
    """After the first chunk a retryable refusal still holds the next refill for retry_after_s,
    and the empty-buffer ticks meanwhile count against max_stall_ticks as before."""
    client = WarmingClient(horizon=1, refused={1}, retry_after_s=WARMING_RETRY_S)
    harness = build(action_horizon=1, max_stall_ticks=3, client=client)
    calls_per_tick: list[int] = []

    async def scenario() -> None:
        async with harness.loop:
            for tick in range(5):
                await harness.loop.step(float(tick))
                calls_per_tick.append(client.calls)

    with caplog.at_level(logging.WARNING, logger="robots.control"):
        asyncio.run(scenario())

    # Tick 1's call is refused; tick 2 is inside the hint; tick 3 asks again and is answered.
    assert calls_per_tick == [1, 2, 2, 3, 4]
    stall_logs = [record.getMessage() for record in records_with(caplog, "stall ")]
    assert [message[:9] for message in stall_logs] == ["stall 1/3", "stall 2/3"]
    # Row 0 of chunk 0, two holds of it, then row 0 of chunks 2 and 3 (values 2 and 3).
    assert joint_values(harness.robots["left"])[:5] == [0, 0, 0, 2, 3]


def test_a_failed_refill_starts_a_new_session_on_the_retry(build: Callable[..., conftest.LoopHarness]) -> None:
    """A refusal or a serve fault ends the episode's session: the retry's refill thread calls the client's
    reset_episode() first, which releases the old session on the gateway, so the retry carries no id and the
    gateway mints a fresh one; a run with no failure never resets."""
    client = WarmingClient(horizon=1, refused={1}, retry_after_s=None)
    harness = build(action_horizon=1, max_stall_ticks=3, client=client)

    async def scenario() -> None:
        async with harness.loop:
            for tick in range(4):
                await harness.loop.step(float(tick))

    asyncio.run(scenario())
    # Call 1 (tick 1) is refused; the retry at tick 2, after the 1 s default hold, resets first, then asks.
    assert client.resets == 1 and client.reset_after_calls == [2] and client.calls == 4


def test_result_that_lands_in_time_fills_the_buffer(build: Callable[..., conftest.LoopHarness]) -> None:
    """A result that lands within action_timeout_s of its start fills the buffer."""
    gate = threading.Event()
    client = conftest.FakeClient(horizon=4, gate=gate)
    harness = build(action_timeout_s=STALL_TIMEOUT_S, client=client)

    async def scenario() -> None:
        async with harness.loop:
            await harness.loop.step(0.0)
            gate.set()
            await wait_for_refill_task()
            # Within the deadline and under half a period after its observation, so no row is past.
            await harness.loop.step(0.01)

    asyncio.run(scenario())

    assert client.calls == 1
    assert joint_values(harness.robots["left"]) == [0.0, 0.0, 0.0]
    # One hold from the stall, then row 0 of the chunk, then the __aexit__ hold of that row.
    np.testing.assert_allclose(harness.robots["left"].commands[1], [0.0] * 6 + [0.0])
    assert harness.robots["left"].commands[0][6] == 0.5


def test_late_result_is_discarded_and_a_fresh_refill_starts(
    build: Callable[..., conftest.LoopHarness], caplog: pytest.LogCaptureFixture
) -> None:
    """A result that lands after action_timeout_s is discarded with a log line
    and a fresh refill starts on the next due tick."""
    gate = threading.Event()
    client = conftest.FakeClient(horizon=4, gate=gate)
    harness = build(action_timeout_s=STALL_TIMEOUT_S, client=client)

    async def scenario() -> None:
        async with harness.loop:
            await harness.loop.step(0.0)
            gate.set()
            await wait_for_refill_task()
            await harness.loop.step(1.0)

    with caplog.at_level(logging.WARNING, logger="robots.control"):
        asyncio.run(scenario())

    discarded = records_with(caplog, "discarding action chunk")
    assert len(discarded) == 1
    assert f"action_timeout_s={STALL_TIMEOUT_S:.3f}" in discarded[0].getMessage()
    assert client.calls == 2
    # Row 0 of chunk 1 (value 4) reached the arms; chunk 0 never did.
    assert joint_values(harness.robots["right"])[1] == 4.0
    assert 0.0 not in joint_values(harness.robots["right"])[1:]


def test_late_prefetch_keeps_draining_the_current_chunk(
    build: Callable[..., conftest.LoopHarness], caplog: pytest.LogCaptureFixture
) -> None:
    """While a prefetch exceeds action_timeout_s the loop keeps draining the chunk it
    has, and stalls only once that runs out."""
    gate = threading.Event()
    gate.set()
    client = conftest.FakeClient(horizon=4, gate=gate)
    harness = build(action_horizon=4, refill_at_remaining=3, action_timeout_s=STALL_TIMEOUT_S, client=client)

    async def scenario() -> None:
        try:
            async with harness.loop:
                await harness.loop.step(0.0)
                gate.clear()
                for now_s in (1.0, 2.0, 3.0, 4.0):
                    await harness.loop.step(now_s)
        finally:
            gate.set()
            await wait_for_refill_task()

    with caplog.at_level(logging.WARNING, logger="robots.control"):
        asyncio.run(scenario())

    assert client.calls == 2
    assert joint_values(harness.robots["left"])[:5] == [0, 1, 2, 3, 3]
    assert len(records_with(caplog, "stall 1/")) == 1


def test_dream_error_from_get_action_is_logged_and_retried_next_tick(
    build: Callable[..., conftest.LoopHarness], caplog: pytest.LogCaptureFixture
) -> None:
    """Get_action raising a DreamError costs one refill; it is logged at the tick
    boundary and the next due tick calls again."""
    client = conftest.FakeClient(horizon=4, raise_once=dream.DreamTimeout("no reply in 15000 ms"))
    harness = build(client=client)

    async def scenario() -> None:
        async with harness.loop:
            await harness.loop.step(0.0)
            await harness.loop.step(0.1)

    with caplog.at_level(logging.WARNING, logger="robots.control"):
        asyncio.run(scenario())

    failed = records_with(caplog, "get_action failed")
    assert len(failed) == 1
    assert "no reply in 15000 ms" in failed[0].getMessage()
    assert client.calls == 2
    # Tick 1 held (stall), tick 2 wrote row 0 of chunk 1 (value 4).
    assert joint_values(harness.robots["left"])[:2] == [0.0, 4.0]
    assert harness.robots["left"].commands[0][6] == 0.5


def test_mis_shaped_chunk_is_held_then_ends_the_run_naming_the_shape_like_a_dream_action_error(
    build: Callable[..., conftest.LoopHarness], caplog: pytest.LogCaptureFixture
) -> None:
    """From_dream's ControlLoopError (a chunk of the wrong shape) costs one refill, like the
    DreamActionError strict mode raises for the same fault, so strict on and off behave alike. A checkpoint for
    another body answers the same shape every time, so the next refill is held for RETRY_AFTER_DEFAULT_S instead
    of asked thirty times a second, the buffer runs dry, and the stall that ends the run names the shape fault as
    its `reason` (`error_code()`), so the page says which model to pick rather than that the loop stalled."""
    harness = build(action_horizon=4)

    async def scenario() -> None:
        async with harness.loop:
            await harness.loop.step(0.0)
            # The next chunk is one row short: from_dream rejects it after the client accepted it.
            harness.client.horizon = 3
            with pytest.raises(errors.ControlLoopError, match="q_r_chunk") as info:
                for step in range(1, 12):
                    await harness.loop.step(step * 0.1)
            assert info.value.reason == "DreamActionError"
            assert harness.loop.refill_error_code == "DreamActionError"
            assert errors.error_code(info.value) == "DreamActionError"

    with caplog.at_level(logging.WARNING, logger="robots.control"):
        asyncio.run(scenario())

    refused = records_with(caplog, f"retrying after {control.RETRY_AFTER_DEFAULT_S:.1f} s")
    assert len(refused) == 1 and "(3, 7)" in refused[0].getMessage()
    assert not records_with(caplog, "get_action failed")
    assert harness.client.calls == 2, "held: no second try within RETRY_AFTER_DEFAULT_S"
    # Chunk 0 drained, then the arms held row 3 through the stalls.
    values = joint_values(harness.robots["left"])
    assert values[:4] == [0, 1, 2, 3] and set(values[4:]) == {3}


NOT_RETRYABLE = [
    dream.DreamServerError(
        "DreamAPI refused the inference request: DA-2000 NO_CAPACITY (max_serves)",
        code="DA-2000",
        reason="max_serves",
        retryable=False,
    ),
    dream.DreamServerError(
        "DreamAPI refused the inference request: DA-6000 GATEWAY_FAULT (cannot_scale): weights could not be prepared",
        code="DA-6000",
        reason="cannot_scale",
        retryable=False,
    ),
]


@pytest.mark.parametrize("refusal", NOT_RETRYABLE, ids=lambda refusal: refusal.reason)
def test_a_refusal_the_contract_marks_not_retryable_ends_the_run_at_once_with_dreamapis_words(
    build: Callable[..., conftest.LoopHarness], caplog: pytest.LogCaptureFixture, refusal: dream.DreamServerError
) -> None:
    harness = build(action_horizon=4)

    async def scenario() -> None:
        async with harness.loop:
            await harness.loop.step(0.0)
            harness.client.raise_once = refusal
            with pytest.raises(errors.ControlLoopError, match="not retryable, so the run stops") as info:
                for step in range(1, 12):
                    await harness.loop.step(step * 0.1)
            assert info.value.reason == refusal.reason
            assert str(refusal) in str(info.value)
            words = next(text for known, text in envelope.REASON_HELP.items() if known == refusal.reason)
            assert str(info.value).endswith(f"so the run stops. {words.meaning} {words.action}")

    with caplog.at_level(logging.WARNING, logger="robots.control"):
        asyncio.run(scenario())

    assert harness.client.calls == 2, "no second try"
    assert not records_with(caplog, "retrying")


def test_a_wrong_shape_action_is_held_and_ends_the_run_under_its_code(
    build: Callable[..., conftest.LoopHarness], caplog: pytest.LogCaptureFixture
) -> None:
    refusal = dream.DreamActionError("action 'q_r_chunk' has shape (3, 7), expected (4, 7)")
    code = "DreamActionError"
    harness = build(action_horizon=4)

    async def scenario() -> None:
        async with harness.loop:
            await harness.loop.step(0.0)
            harness.client.raise_once = refusal
            with pytest.raises(errors.ControlLoopError, match="the last get_action failed") as info:
                for step in range(1, 12):
                    await harness.loop.step(step * 0.1)
            assert info.value.reason == code
            assert errors.error_code(info.value) == code
            assert str(refusal) in str(info.value)

    with caplog.at_level(logging.WARNING, logger="robots.control"):
        asyncio.run(scenario())

    assert harness.client.calls == 2, "held: no second try within RETRY_AFTER_DEFAULT_S"
    assert len(records_with(caplog, f"retrying after {control.RETRY_AFTER_DEFAULT_S:.1f} s")) == 1
    assert not records_with(caplog, "get_action failed")


@pytest.mark.parametrize("error", UNRECOVERABLE_ERRORS, ids=lambda error: type(error).__name__)
def test_unrecoverable_dream_error_stops_the_loop_and_holds(
    build: Callable[..., conftest.LoopHarness], error: dream.DreamError
) -> None:
    """DreamAuthError, DreamConfigError and DreamObservationError propagate out of the tick:
    a retry cannot fix a refused key, a wrong checkpoint or an adapter/embodiment disagreement."""
    client = conftest.FakeClient(horizon=4, raise_once=error)
    harness = build(client=client)

    async def scenario() -> None:
        async with harness.loop:
            await harness.loop.step(0.0)

    with pytest.raises(type(error)):
        asyncio.run(scenario())

    assert client.calls == 1
    assert client.closed
    for robot in harness.robots.values():
        # No tick write happened; __aexit__'s hold is the only command.
        assert len(robot.commands) == 1
        np.testing.assert_allclose(robot.commands[0], HOLD_COMMAND)


def test_session_expired_drops_the_buffer_and_refills(
    build: Callable[..., conftest.LoopHarness], caplog: pytest.LogCaptureFixture
) -> None:
    """DreamSessionExpired means the episode restarted server-side; the rows still buffered
    answer the dead session, so they are dropped, the restart is logged at WARNING, and the next
    refill starts the new episode instead of stopping the loop."""
    harness = build(action_horizon=4, refill_at_remaining=2)

    async def scenario() -> None:
        async with harness.loop:
            await harness.loop.step(0.0)
            await harness.loop.step(0.1)
            harness.client.raise_once = dream.DreamSessionExpired(
                "inference session s-1 was replaced by s-2 (unknown): the episode restarted server-side",
                code="DA-0001",
                reason="unknown",
            )
            # Two rows left after this pop: the prefetch that raises starts here.
            await harness.loop.step(0.2)
            await wait_for_refill_task()
            await harness.loop.step(0.3)

    with caplog.at_level(logging.WARNING, logger="robots.control"):
        asyncio.run(scenario())

    restarted = records_with(caplog, "episode restarted server-side")
    assert len(restarted) == 1
    assert restarted[0].levelno == logging.WARNING
    assert "s-1 was replaced by s-2" in restarted[0].getMessage()
    assert not records_with(caplog, "get_action failed")
    assert harness.client.calls == 3
    # Rows 0..2 of chunk 0, then row 0 of chunk 2 (value 8): row 3 was dropped with the dead session.
    assert joint_values(harness.robots["left"])[:4] == [0, 1, 2, 8]


def test_non_dream_exception_from_get_action_propagates(build: Callable[..., conftest.LoopHarness]) -> None:
    """The tick boundary catches dream.DreamError only; anything else stops the loop."""
    client = conftest.FakeClient(horizon=4, raise_once=RuntimeError("policy process crashed"))
    harness = build(client=client)

    async def scenario() -> None:
        async with harness.loop:
            await harness.loop.step(0.0)

    with pytest.raises(RuntimeError, match="policy process crashed"):
        asyncio.run(scenario())
    assert client.closed


# Cameras


def test_stale_frame_holds_and_logs_once_per_burst(
    build: Callable[..., conftest.LoopHarness], caplog: pytest.LogCaptureFixture
) -> None:
    """Camera frame older than stale_after_s -> fail the tick, hold last command, log once
    per burst; a new burst after recovery logs again."""
    harness = build()

    async def scenario() -> None:
        async with harness.loop:
            await harness.loop.step(stale_now_s())
            assert harness.loop.state is None
            await harness.loop.step(stale_now_s())
            await harness.loop.step(time.monotonic())
            assert harness.loop.state is not None
            await harness.loop.step(stale_now_s())

    with caplog.at_level(logging.WARNING, logger="robots.control"):
        asyncio.run(scenario())

    stale_logs = records_with(caplog, "older than stale_after_s=0.200")
    assert len(stale_logs) == 2
    assert "camera 'top'" in stale_logs[0].getMessage()
    assert harness.client.calls == 1
    # Two holds, the real write (row 0), one more hold of that row, and the __aexit__ hold.
    left = harness.robots["left"].commands
    assert len(left) == 5
    np.testing.assert_allclose(left[0], HOLD_COMMAND)
    np.testing.assert_allclose(left[1], HOLD_COMMAND)
    np.testing.assert_allclose(left[2], [0.0] * 6 + [0.0])
    np.testing.assert_allclose(left[3], [0.0] * 6 + [0.0])


def test_camera_thread_death_raises_camera_error_out_of_run(build: Callable[..., conftest.LoopHarness]) -> None:
    """Camera thread dies -> stop the loop; CameraError propagates out of run() and the
    arms are held (not disengaged)."""
    harness = build()
    reader = harness.readers[1]

    async def scenario() -> None:
        async with harness.loop:
            harness.sources["left"].fail()
            deadline_s = time.monotonic() + THREAD_WAIT_S
            while reader.is_alive() and time.monotonic() < deadline_s:
                await asyncio.sleep(0.001)
            await harness.loop.run()

    with pytest.raises(errors.CameraError, match=r"camera 'left' stopped sending pictures \(") as raised:
        asyncio.run(scenario())
    assert "Error(" not in str(raised.value)
    assert str(raised.value).endswith("; check its cable; dreamcli setup shows the cameras")

    for robot in harness.robots.values():
        assert len(robot.commands) == 1
        np.testing.assert_allclose(robot.commands[0], HOLD_COMMAND)
    assert harness.client.closed
    assert all(source.released for source in harness.sources.values())


def test_first_frame_timeout_in_aenter_raises_camera_error(build: Callable[..., conftest.LoopHarness]) -> None:
    """A camera that never publishes a first frame -> CameraError from __aenter__, with
    every camera stopped and released."""
    # A plain ManualClock: the 500 polls then take milliseconds of wall time, so the blocked
    # thread is still alive (not dead) when the manual deadline passes.
    harness = build(camera_names=("top",), clock=conftest.ManualClock())
    blocked = BlockingCaptureSource()
    readers = [harness.readers[0], cameras.CameraReader("left", blocked, stale_after_s=0.2)]
    loop = control.ControlLoop(
        followers=harness.arms,
        cameras=readers,
        client=harness.client,
        task=conftest.TASK,
        control_hz=30.0,
        action_horizon=4,
        action_timeout_s=1.0,
        max_ee_speed_m_s=float("inf"),
        clock=harness.clock,
        sleep=harness.clock.sleep,
    )

    async def scenario() -> None:
        async with loop:
            raise AssertionError("__aenter__ returned without every camera's first frame")

    with pytest.raises(
        errors.CameraError,
        match=r"cameras \[.*'left'\] published no first frame within 5.0 s.*; check the cables; dreamcli setup shows",
    ):
        asyncio.run(scenario())

    assert blocked.released
    assert harness.sources["top"].released
    assert not readers[0].is_alive()
    # The wait was paced by the injected sleep, never the wall clock.
    assert harness.clock() >= control.FIRST_FRAME_TIMEOUT_S


def test_camera_thread_dead_at_startup_raises_camera_error(build: Callable[..., conftest.LoopHarness]) -> None:
    """A capture thread that dies before its first frame -> CameraError from __aenter__."""
    harness = build()
    harness.sources["top"].fail()

    async def scenario() -> None:
        async with harness.loop:
            raise AssertionError("__aenter__ returned with a dead camera")

    with pytest.raises(errors.CameraError, match="top"):
        asyncio.run(scenario())

    assert all(source.released for source in harness.sources.values())
    assert not harness.client.closed


# Teardown


def test_aexit_holds_both_arms_and_closes_the_policy_on_normal_exit(
    build: Callable[..., conftest.LoopHarness],
) -> None:
    """On exit the loop stops the cameras, holds both arms and closes the policy client."""
    harness = build()

    async def scenario() -> None:
        async with harness.loop:
            pass

    asyncio.run(scenario())

    for robot in harness.robots.values():
        assert len(robot.commands) == 1
        np.testing.assert_allclose(robot.commands[0], HOLD_COMMAND)
    assert harness.client.closed
    assert all(source.released for source in harness.sources.values())
    assert not any(reader.is_alive() for reader in harness.readers)


def test_aexit_holds_both_arms_on_control_loop_error(build: Callable[..., conftest.LoopHarness]) -> None:
    """A ControlLoopError leaves both arms holding their last command."""
    harness = build()

    async def scenario() -> None:
        async with harness.loop:
            await harness.loop.step(0.0)
            raise errors.ControlLoopError("action buffer empty for 11 consecutive ticks")

    with pytest.raises(errors.ControlLoopError):
        asyncio.run(scenario())

    for robot in harness.robots.values():
        assert len(robot.commands) == 2
        np.testing.assert_allclose(robot.commands[1], robot.commands[0])
    assert harness.client.closed


def test_arm_read_error_disengages_both_arms(build: Callable[..., conftest.LoopHarness]) -> None:
    """Arm read error -> stop the loop and disengage both arms."""
    robots = {"left": conftest.FakeRobot(has_gravity_comp=True), "right": conftest.FakeRobot(has_gravity_comp=True)}
    harness = build(robots=robots)
    robots["right"].read_error = errors.ArmError("arm 'right' observation lacks 'joint_pos'")

    async def scenario() -> None:
        async with harness.loop:
            await harness.loop.step(0.0)

    with pytest.raises(errors.ArmError, match="lacks 'joint_pos'"):
        asyncio.run(scenario())

    for robot in robots.values():
        assert robot.gravity_comp_calls == 1
        assert robot.commands == []
    assert harness.client.closed


def test_arm_write_error_disengages_both_arms(build: Callable[..., conftest.LoopHarness]) -> None:
    """Arm write error -> stop the loop and disengage both arms, via zero torque when that
    is what the driver offers."""
    robots = {"left": conftest.FakeRobot(has_zero_torque=True), "right": conftest.FakeRobot(has_zero_torque=True)}
    harness = build(robots=robots)
    robots["left"].write_error = errors.ArmError("arm 'left' rejected the command")

    async def scenario() -> None:
        async with harness.loop:
            await harness.loop.step(0.0)

    with pytest.raises(errors.ArmError, match="rejected the command"):
        asyncio.run(scenario())

    assert robots["left"].zero_torque_calls == 1
    assert robots["right"].zero_torque_calls == 1
    assert robots["left"].commands == []
    assert robots["right"].commands == []


def test_driver_exception_propagates_unchanged_and_arms_are_held(build: Callable[..., conftest.LoopHarness]) -> None:
    """A driver exception that is not an ArmError propagates unchanged and
    the teardown holds rather than disengages."""
    robots = {"left": conftest.FakeRobot(has_gravity_comp=True), "right": conftest.FakeRobot(has_gravity_comp=True)}
    harness = build(robots=robots)
    robots["left"].read_error = OSError("CAN bus down")

    async def scenario() -> None:
        async with harness.loop:
            await harness.loop.step(0.0)

    with pytest.raises(OSError, match="CAN bus down"):
        asyncio.run(scenario())

    for robot in robots.values():
        assert robot.gravity_comp_calls == 0
        assert len(robot.commands) == 1


def test_teardown_failure_is_logged_never_raised_over_the_original(
    build: Callable[..., conftest.LoopHarness], caplog: pytest.LogCaptureFixture
) -> None:
    """__aexit__'s teardown logs its own failures and every step still runs."""
    harness = build()

    def close_and_fail() -> None:
        harness.client.closed = True
        raise RuntimeError("socket already gone")

    harness.client.close = close_and_fail
    harness.robots["left"].write_error = OSError("left hold refused")

    async def scenario() -> None:
        async with harness.loop:
            raise errors.ControlLoopError("original fault")

    with caplog.at_level(logging.ERROR, logger="robots.control"), pytest.raises(errors.ControlLoopError):
        asyncio.run(scenario())

    failures = [record.getMessage() for record in records_with(caplog, "teardown step failed")]
    assert failures == ["teardown step failed: hold left arm", "teardown step failed: close dream client"]
    assert len(harness.robots["right"].commands) == 1
    assert harness.client.closed


def test_a_teardown_failure_of_ours_is_one_line_without_a_traceback(
    build: Callable[..., conftest.LoopHarness], caplog: pytest.LogCaptureFixture
) -> None:
    harness = build()
    harness.robots["left"].write_error = errors.ArmError("arm 'left' may still be powered: cut its power")

    async def scenario() -> None:
        async with harness.loop:
            raise errors.ControlLoopError("original fault")

    with caplog.at_level(logging.ERROR, logger="robots.control"), pytest.raises(errors.ControlLoopError):
        asyncio.run(scenario())

    [record] = records_with(caplog, "teardown step failed")
    assert record.getMessage() == "teardown step failed: hold left arm: arm 'left' may still be powered: cut its power"
    assert record.exc_info is None


# Scheduling


def test_run_ticks_at_deadlines_spaced_by_the_period(build: Callable[..., conftest.LoopHarness]) -> None:
    """Deadlines advance by exactly one period from the monotonic clock."""
    harness = build(control_hz=10.0, clock=YieldingClock(now_s=100.0))
    ticks: list[float] = []

    async def fake_step(now_s: float) -> None:
        ticks.append(now_s)
        if len(ticks) == 3:
            raise StopTicksError

    harness.loop.step = fake_step

    with pytest.raises(StopTicksError):
        asyncio.run(harness.loop.run())

    assert ticks == pytest.approx([100.0, 100.1, 100.2])
    assert harness.clock.sleeps == pytest.approx([0.1, 0.1])


def test_overrun_is_logged_with_measured_duration_and_does_not_accumulate_lag(
    build: Callable[..., conftest.LoopHarness], caplog: pytest.LogCaptureFixture
) -> None:
    """Tick overrun -> log with the measured duration; the schedule restarts from now so
    the next sleep is a full period, not a catch-up."""
    harness = build(control_hz=10.0)
    ticks: list[float] = []

    async def slow_then_fast_step(now_s: float) -> None:
        ticks.append(now_s)
        if len(ticks) == 1:
            harness.clock.now_s += 0.25
        if len(ticks) == 4:
            raise StopTicksError

    harness.loop.step = slow_then_fast_step

    with caplog.at_level(logging.WARNING, logger="robots.control"), pytest.raises(StopTicksError):
        asyncio.run(harness.loop.run())

    overruns = records_with(caplog, "tick overrun")
    assert len(overruns) == 1
    assert "250.0 ms against a 100.0 ms period" in overruns[0].getMessage()
    # No lag: the tick after the overrun sleeps 0, then every later tick sleeps a whole period.
    assert harness.clock.sleeps == pytest.approx([0.0, 0.1, 0.1])
    assert ticks == pytest.approx([0.0, 0.25, 0.35, 0.45])


def test_run_stops_on_cancellation_and_leaves_the_arms_held(build: Callable[..., conftest.LoopHarness]) -> None:
    """Run() ends only by exception or cancellation; a cancelled loop still tears down."""
    harness = build()

    async def scenario() -> asyncio.Task[None]:
        async def session() -> None:
            async with harness.loop:
                await harness.loop.run()

        task = asyncio.create_task(session())
        deadline_s = time.monotonic() + THREAD_WAIT_S
        while len(harness.robots["left"].commands) < 3 and time.monotonic() < deadline_s:
            await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await wait_for_refill_task()
        return task

    task = asyncio.run(scenario())

    assert task.cancelled()
    assert harness.client.closed
    assert not any(reader.is_alive() for reader in harness.readers)
    left = harness.robots["left"].commands
    assert len(left) >= 4
    # The teardown hold re-commands the last written action.
    np.testing.assert_allclose(left[-1], left[-2])


# Start-up ramp


## Resting pose of both fake arms in the ramp tests: below every row FakeClient returns, so each
## blended command moves the same way and the sequence can be checked for monotonicity.
RAMP_ORIGIN_RAD = -10.0

## Session-expired error the reset test queues on the fake policy.
SESSION_EXPIRED = dream.DreamSessionExpired(
    "inference session s-1 was replaced by s-2 (unknown): the episode restarted server-side",
    code="DA-0001",
    reason="unknown",
)


def ramp_harness(build: Callable[..., conftest.LoopHarness], **overrides: Any) -> conftest.LoopHarness:
    """A 10 Hz loop over arms resting at #RAMP_ORIGIN_RAD (gripper 0.5); `ramp_s=0.4` is four ticks."""
    robots = {
        name: conftest.FakeRobot(joint_pos=np.full(conftest.ARM_DOF, RAMP_ORIGIN_RAD)) for name in ("left", "right")
    }
    settings: dict[str, Any] = {"control_hz": 10.0, "ramp_s": 0.4, "ramp_max_rad_s": 100.0, "robots": robots}
    return build(**{**settings, **overrides})


def test_ramp_first_command_is_one_step_from_the_measured_pose(
    build: Callable[..., conftest.LoopHarness], caplog: pytest.LogCaptureFixture
) -> None:
    """The first command after start is one ramp step from where the arms rest, not the
    target, and the ramp is announced once at INFO with its duration and largest joint delta."""
    harness = ramp_harness(build)

    async def scenario() -> None:
        async with harness.loop:
            await harness.loop.step(0.0)

    with caplog.at_level(logging.INFO, logger="robots.control"):
        asyncio.run(scenario())

    started = records_with(caplog, "ramp started")
    assert len(started) == 1 and started[0].levelno == logging.INFO
    assert "0.40 s over 4 ticks" in started[0].getMessage()
    assert "max joint delta 10.000 rad" in started[0].getMessage()
    for robot in harness.robots.values():
        # Row 0 targets joints 0.0 and gripper 0.0; one step of four from (-10, 0.5) is a quarter of the way.
        np.testing.assert_allclose(robot.commands[0], [-7.5] * 6 + [0.375])


def test_ramp_commands_rise_monotonically_and_end_exactly_on_the_target(
    build: Callable[..., conftest.LoopHarness],
) -> None:
    """While the ramp runs, each command is `origin + k/N * (row_k - origin)` with the buffer
    draining at the normal rate; the command at `k == N` is the row itself and later rows pass through."""
    harness = ramp_harness(build)

    async def scenario() -> None:
        async with harness.loop:
            for tick in range(6):
                await harness.loop.step(tick * 0.1)

    asyncio.run(scenario())

    left = harness.robots["left"].commands[:6]
    joints = [float(command[0]) for command in left]
    # Rows 0..5 blended at alpha 1/4, 2/4, 3/4, then rows 3, 4, 5 unchanged.
    assert joints == pytest.approx([-7.5, -4.5, -1.0, 3.0, 4.0, 5.0])
    assert joints == sorted(joints) and len(set(joints)) == len(joints)
    assert [float(command[6]) for command in left[:3]] == pytest.approx([0.375, 0.3, 0.275])
    np.testing.assert_array_equal(left[3], [3.0] * 6 + [0.3])
    np.testing.assert_array_equal(left[4], [4.0] * 6 + [0.4])
    assert harness.client.calls == 2


def test_ramp_speed_cap_stretches_the_ramp_past_ramp_s(
    build: Callable[..., conftest.LoopHarness], caplog: pytest.LogCaptureFixture
) -> None:
    """A first delta of 10 rad at 0.5 rad/s needs 20 s, so the ramp lasts 200 ticks instead of
    the four `ramp_s` asks for, and the first step moves each joint by what the cap allows per tick."""
    harness = ramp_harness(build, ramp_max_rad_s=0.5)

    async def scenario() -> None:
        async with harness.loop:
            await harness.loop.step(0.0)
            await harness.loop.step(0.1)

    with caplog.at_level(logging.INFO, logger="robots.control"):
        asyncio.run(scenario())

    (started,) = records_with(caplog, "ramp started")
    assert "20.00 s over 200 ticks" in started.getMessage()
    commands = harness.robots["right"].commands
    np.testing.assert_allclose(commands[0], [-9.95] * 6 + [0.4975])
    np.testing.assert_allclose(commands[1], [-9.89] * 6 + [0.496])
    # 0.5 rad/s at 10 Hz is 0.05 rad per tick; the slack is rounding, not a wider cap.
    np.testing.assert_array_less(np.abs(commands[0][:6] - RAMP_ORIGIN_RAD), 0.05 + 1e-9)


def test_ramp_restarts_from_the_measured_pose_after_a_session_reset(
    build: Callable[..., conftest.LoopHarness], caplog: pytest.LogCaptureFixture
) -> None:
    """After DreamSessionExpired drops the buffer, the next chunk is ramped again, from
    wherever the arms are measured on that tick rather than from the original origin; the ramp is the
    whole of that handoff."""
    harness = ramp_harness(build, ramp_s=0.2, action_horizon=4, refill_at_remaining=2)

    async def scenario() -> None:
        async with harness.loop:
            await harness.loop.step(0.0)
            await harness.loop.step(0.1)
            harness.client.raise_once = SESSION_EXPIRED
            # Two rows left after this pop: the prefetch that raises starts here.
            await harness.loop.step(0.2)
            await wait_for_refill_task()
            for robot in harness.robots.values():
                robot.joint_pos = np.full(conftest.ARM_DOF, 5.0)
            await harness.loop.step(0.3)
            await harness.loop.step(0.4)

    with caplog.at_level(logging.INFO, logger="robots.control"):
        asyncio.run(scenario())

    assert len(records_with(caplog, "ramp started")) == 2
    assert harness.client.calls == 3
    # Two-tick ramp: half way to row 0, then row 1 and row 2 as is. After the reset the arms sit at
    # 5.0 and chunk 2 starts at row 8: half way from 5 to 8, then row 9 as is.
    assert joint_values(harness.robots["left"])[:5] == pytest.approx([-5.0, 1.0, 2.0, 6.5, 9.0])


def test_ramp_shorter_than_a_period_passes_the_first_target_through(
    build: Callable[..., conftest.LoopHarness],
) -> None:
    """A ramp shorter than one tick with a speed cap the first delta does not reach is one
    tick long, so the first command is the row itself; what every other test in this file relies on."""
    harness = ramp_harness(build, ramp_s=0.01, ramp_max_rad_s=1000.0)

    async def scenario() -> None:
        async with harness.loop:
            await harness.loop.step(0.0)

    asyncio.run(scenario())

    for robot in harness.robots.values():
        np.testing.assert_array_equal(robot.commands[0], [0.0] * 6 + [0.0])


# Published snapshots


def test_state_and_video_are_new_objects_each_tick_and_never_mutated(
    build: Callable[..., conftest.LoopHarness],
) -> None:
    """State and video are rebound to fresh immutable snapshots every tick; a consumer
    holding the previous one sees it unchanged."""
    harness = build()

    async def scenario() -> tuple[Any, dict[str, Any], Any, dict[str, Any]]:
        async with harness.loop:
            assert harness.loop.state is None
            assert harness.loop.video == {}
            await harness.loop.step(1.0)
            first_state, first_video = harness.loop.state, harness.loop.video
            first_frames = dict(first_video)
            await harness.loop.step(2.0)
            assert dict(first_video) == first_frames
            return first_state, first_video, harness.loop.state, harness.loop.video

    first_state, first_video, second_state, second_video = asyncio.run(scenario())

    assert second_state is not first_state
    assert second_video is not first_video
    assert first_state.timestamp_s == 1.0
    assert second_state.timestamp_s == 2.0
    assert set(first_video) == set(conftest.CAMERA_NAMES)
    assert all(frame.camera == name for name, frame in second_video.items())
    with pytest.raises(dataclasses.FrozenInstanceError):
        first_state.timestamp_s = 3.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        first_video["top"].timestamp_s = 3.0  # type: ignore[misc]


def test_video_frames_are_the_published_frames_and_the_policy_gets_them_as_jpeg_windows(
    build: Callable[..., conftest.LoopHarness],
) -> None:
    """The frames on `video` are the capture threads' own RGB Frame objects; the
    client is handed each camera's encoded frame under `video.<role>`, the model's role for the camera's name, so the
    episode's first request carries a one-frame base64 JPEG window per role, BGR on the wire so the serve's cv2
    decode gives the colours back, plus the prompt and the gripper state."""
    harness = build(task="stack the blocks")

    async def scenario() -> dict[str, Any]:
        async with harness.loop:
            await harness.loop.step(0.0)
            return harness.loop.video

    video = asyncio.run(scenario())
    sent = harness.client.observations[0]

    for name in conftest.CAMERA_NAMES:
        assert video[name].data.shape == conftest.FRAME_SHAPE
        # The fake source is blue in BGR; the published frame must be RGB.
        assert video[name].data[0, 0].tolist() == [0, 0, 255]
        window = sent[f"video.{adapter.YAM_VIDEO_ROLES[name]}"]
        assert isinstance(window, list) and len(window) == 1 and isinstance(window[0], str)
        decoded = cv2.imdecode(np.frombuffer(base64.b64decode(window[0]), dtype=np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape == conftest.FRAME_SHAPE
        # Decoded by cv2 the pixel is BGR again: pure blue is [255, 0, 0].
        np.testing.assert_allclose(decoded[0, 0], [255, 0, 0], atol=JPEG_TOLERANCE)
    assert sent["prompt"] == "stack the blocks"
    assert sent["state.left_gripper"].tolist() == [0.5]


class StubReader:
    """A camera reader whose frame the test publishes by hand, with the encoded frame and capture time it chooses.

    Attributes:
    - name: The camera name, a serve role.
    - frame: The published frame, or None.
    - error: Always None: the stub never dies.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.frame: types.Frame | None = None
        self.error: BaseException | None = None

    def publish(self, now_s: float, jpeg: str, jpeg_s: float) -> None:
        """Publishes a fresh capture at `now_s` carrying `jpeg`, encoded from a capture at `jpeg_s`."""
        data = np.zeros(conftest.FRAME_SHAPE, dtype=np.uint8)
        self.frame = types.Frame(data=data, camera=self.name, timestamp_s=now_s, jpeg=jpeg, jpeg_timestamp_s=jpeg_s)

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def is_alive(self) -> bool:
        return True


def stub_cameras(harness: conftest.LoopHarness) -> dict[str, StubReader]:
    """Swaps the harness's readers for #StubReader s, one per camera, before the loop is entered."""
    stubs = {name: StubReader(name) for name in conftest.CAMERA_NAMES}
    harness.loop._cameras = list(stubs.values())
    return stubs


def test_a_tick_hands_the_client_a_set_once_every_camera_has_a_new_frame_and_a_refill_tick_always_hands_its_state(
    build: Callable[..., conftest.LoopHarness],
) -> None:
    """The client keeps the window and is handed each slot's encoded frames once, timed by the newest
    capture among them; a tick whose cameras have not all encoded a new frame since the last set hands nothing; a
    refill tick hands its own state, with the last set's frames and timestamp when no new set is complete, so the
    request carries the newest state and the timestamps never go back."""
    harness = build(control_hz=10.0, action_horizon=4, refill_at_remaining=0)
    stubs = stub_cameras(harness)
    right = harness.robots["right"]

    def publish(now_s: float, jpegs: dict[str, tuple[str, float]]) -> None:
        for name, stub in stubs.items():
            stub.publish(now_s, *jpegs[name])

    first = {name: (f"{name}-0", 1.0) for name in stubs}
    publish(0.0, first)

    async def scenario() -> None:
        async with harness.loop:
            await harness.loop.step(0.0)  # refill: the first set, on the refill's thread
            publish(0.1, first)
            await harness.loop.step(0.1)  # nothing new
            publish(0.2, {**first, "top": ("top-1", 1.19), "left": ("left-1", 1.2)})
            await harness.loop.step(0.2)  # two of three cameras new: nothing
            publish(0.3, {"top": ("top-1", 1.19), "left": ("left-1", 1.2), "right": ("right-1", 1.21)})
            await harness.loop.step(0.3)  # the set is complete: handed, timed by its newest capture
            right.joint_pos = np.full(conftest.ARM_DOF, 0.7)
            await harness.loop.step(0.4)  # refill with nothing new: this tick's state, the last set's frames

    asyncio.run(scenario())

    client = harness.client
    assert [timestamp for _, _, timestamp in client.ticks] == [1.0, 1.21, 1.21]
    assert [frames["video.overhead"] for frames, _, _ in client.ticks] == ["top-0", "top-1", "top-1"]
    second_set = {"video.overhead": "top-1", "video.wrist_l": "left-1", "video.wrist_r": "right-1"}
    assert client.ticks[1][0] == second_set and client.ticks[2][0] == second_set
    assert client.calls == 2
    np.testing.assert_allclose(client.observations[1]["state.right_arm"], [0.7] * conftest.ARM_DOF)
    assert client.observations[1]["video.wrist_r"] == ["right-1"]


class HeldHandoverClient(conftest.FakeClient):
    """A FakeClient whose refill-thread set_observation() waits for `release`: a refill whose thread is slow to hand
    its tick over while the loop ticks on."""

    def __init__(self, horizon: int, release: threading.Event) -> None:
        super().__init__(horizon=horizon)
        self.release = release
        self._tick_thread = threading.get_ident()

    def set_observation(self, frames: Mapping[str, str], observation: Mapping[str, Any], **kwargs: Any) -> None:
        if threading.get_ident() != self._tick_thread and not self.release.wait(GATE_WAIT_S):
            raise RuntimeError("HeldHandoverClient was never released")
        super().set_observation(frames, observation, **kwargs)


## Longest #HeldHandoverClient holds a refill's handover.
GATE_WAIT_S = 5.0


def test_no_tick_hands_the_client_a_set_before_the_refills_thread_has_handed_its_own(
    build: Callable[..., conftest.LoopHarness],
) -> None:
    """A refill hands the client the refill tick's observation on its own thread. A later tick with new frames
    must not get in first: its newer timestamp would make the client refuse the refill tick's (timestamps never go
    back) and the request would carry a later state than the one it answers. The new set waits, and is handed on
    the first tick after the refill's own."""
    release = threading.Event()
    client = HeldHandoverClient(horizon=4, release=release)
    harness = build(control_hz=10.0, action_horizon=4, refill_at_remaining=3, client=client)
    stubs = stub_cameras(harness)

    for name, stub in stubs.items():
        stub.publish(0.0, f"{name}-0", 1.0)

    async def scenario() -> None:
        try:
            async with harness.loop:
                release.set()
                await harness.loop.step(0.0)  # the first chunk, its handover let through
                release.clear()
                await harness.loop.step(0.1)  # three rows left: a prefetch starts and its handover is held
                for name, stub in stubs.items():
                    stub.publish(0.2, f"{name}-1", 1.2)
                await harness.loop.step(0.2)  # a new set, held back: the refill's handover is not in yet
                assert [timestamp for _, _, timestamp in client.ticks] == [1.0]
                release.set()
                await wait_for_refill_task()
                for stub in stubs.values():
                    assert stub.frame is not None
                    stub.publish(0.3, stub.frame.jpeg, stub.frame.jpeg_timestamp_s)
                await harness.loop.step(0.3)  # the held set goes now, after the refill's own
        finally:
            release.set()
            await wait_for_refill_task()

    asyncio.run(scenario())

    assert [timestamp for _, _, timestamp in client.ticks] == [1.0, 1.0, 1.2]
    assert [frames["video.overhead"] for frames, _, _ in client.ticks] == ["top-0", "top-0", "top-1"]
    # The tick at 0.3 installs the prefetch and starts the next one, whose thread hands the held set.
    assert client.calls == 3


def test_task_is_a_required_keyword(build: Callable[..., conftest.LoopHarness]) -> None:
    """Every observation carries the prompt, so a loop cannot be built without one."""
    harness = build()

    with pytest.raises(TypeError, match="task"):
        control.ControlLoop(
            followers=harness.arms,
            cameras=harness.readers,
            client=harness.client,
            control_hz=30.0,
            action_horizon=4,
            action_timeout_s=1.0,
            max_ee_speed_m_s=float("inf"),
        )  # type: ignore[call-arg]


# --- Lifecycle logging --------------------------------------------------------------------------


def test_the_loop_logs_ready_first_chunk_and_stop_at_info(
    build: Callable[..., conftest.LoopHarness], caplog: pytest.LogCaptureFixture
) -> None:
    """An operator watching the log sees the loop come up, the first chunk land, and why it stopped."""
    harness = build(action_horizon=4, refill_at_remaining=0)

    async def scenario() -> None:
        async with harness.loop:
            await harness.loop.step(0.0)
            await wait_for_refill_task()
            await harness.loop.step(0.1)

    with caplog.at_level(logging.INFO, logger="robots.control"):
        asyncio.run(scenario())

    assert len(records_with(caplog, "loop ready:")) == 1
    first = records_with(caplog, "chunk 1:")
    assert len(first) == 1 and "motion starts" in first[0].getMessage() and first[0].levelno == logging.INFO
    stopped = records_with(caplog, "loop stopped after")
    assert len(stopped) == 1 and stopped[0].levelno == logging.INFO
    assert not records_with(caplog, "tick overrun")


# --- Chunk handoff: rows as sent, and the rate clamp --------------------------


class _FlatClient:
    """Chunks whose every row is one value: chunk k is `values[k]` in every column, so a boundary is a step."""

    def __init__(self, horizon: int, values: list[float]) -> None:
        self.horizon = horizon
        self.values = values
        self.calls = 0
        self.session_id = None

    def set_observation(self, frames: Mapping[str, str], observation: Mapping[str, Any], **_: Any) -> None:
        return None

    def get_action(self, *, commanded: Mapping[str, Any] | None = None) -> dict[str, Any]:
        value = self.values[min(self.calls, len(self.values) - 1)]
        self.calls += 1
        chunk = np.full((self.horizon, 7), value)
        return {"q_r_chunk": chunk, "q_l_chunk": chunk}

    def reset_episode(self) -> None:
        return None

    def close(self) -> None:
        return None


def test_a_new_chunk_is_commanded_as_sent_across_the_boundary(
    build: Callable[..., conftest.LoopHarness],
) -> None:
    """The loop adds nothing at a chunk boundary. The blend from one chunk into the next is the dream
    client's, before the rows reach the loop, so a step between two chunks as they arrive here is commanded as
    a step: the rows go to the arm exactly as sent."""
    client = _FlatClient(horizon=8, values=[0.0, 0.4])
    harness = build(control_hz=10.0, action_horizon=8, client=client)

    async def scenario() -> None:
        async with harness.loop:
            for tick in range(16):
                await harness.loop.step(tick * 0.1)

    asyncio.run(scenario())

    values = joint_values(harness.robots["left"])
    assert values[:8] == [0.0] * 8
    assert values[8:16] == [0.4] * 8


def test_the_speed_clamp_scales_a_fast_command_to_the_cap_and_logs_once(
    build: Callable[..., conftest.LoopHarness], caplog: pytest.LogCaptureFixture, tmp_path: pathlib.Path
) -> None:
    """A 0.5 rad step in every joint moves the linear stand-in's end effector 0.866 m in one 100 ms
    tick, 8.66 m/s against a 0.866 m/s cap, so the joint delta is scaled by a tenth and the arm advances 0.05 rad
    per tick, each tick's end-effector speed exactly the cap; one WARNING for the run, a count, the largest
    request kept, and the trace's `clamped` and `ee_speed_*` columns filled."""
    client = _FlatClient(horizon=4, values=[0.0, 0.5])
    cap_m_s = np.sqrt(3.0) * 0.5
    trace = tmp_path / "trace.csv"
    harness = build(
        control_hz=10.0,
        action_horizon=4,
        client=client,
        max_ee_speed_m_s=cap_m_s,
        kinematics=linear_kinematics(),
        trace_path=str(trace),
    )

    async def scenario() -> None:
        async with harness.loop:
            for tick in range(8):
                await harness.loop.step(tick * 0.1)

    with caplog.at_level(logging.WARNING, logger="robots.control"):
        asyncio.run(scenario())

    values = joint_values(harness.robots["left"])
    np.testing.assert_allclose(values[4:8], [0.05, 0.10, 0.15, 0.20])
    speeds_m_s = np.sqrt(3.0) * np.diff(values) * 10.0
    np.testing.assert_allclose(speeds_m_s[3:7], cap_m_s, rtol=1e-12)
    assert harness.loop.clamp_count == 4
    assert harness.loop.max_ee_speed_requested_m_s == pytest.approx(np.sqrt(3.0) * 0.5 * 10.0)
    warnings = records_with(caplog, "command clamped")
    assert len(warnings) == 1 and "left end effector asked 8.66 m/s, limit 0.87 m/s" in warnings[0].getMessage()
    rows = [line.split(",") for line in trace.read_text().splitlines()][1:]
    columns = list(control.trace_columns(YAM_JOINTS))
    assert [row[columns.index("clamped")] for row in rows] == ["0", "0", "0", "0", "1", "1", "1", "1"]
    assert rows[0][columns.index("ee_speed_l_m_s")] == "nan"
    assert float(rows[4][columns.index("ee_speed_r_m_s")]) == pytest.approx(8.66, abs=0.005)


def test_a_command_under_the_cap_and_the_first_command_pass_untouched(
    build: Callable[..., conftest.LoopHarness],
) -> None:
    """The first command of a run has nothing to be measured from and passes whole (the ramp is one
    tick here); a later step of 0.02 rad per joint, 0.35 m/s on the stand-in, is under the 0.5 m/s cap and
    reaches the arm as sent, with no clamp counted and the request still recorded."""
    client = _FlatClient(horizon=4, values=[5.0, 5.02])
    harness = build(
        control_hz=10.0,
        action_horizon=4,
        client=client,
        max_ee_speed_m_s=0.5,
        kinematics=linear_kinematics(),
    )

    async def scenario() -> None:
        async with harness.loop:
            for tick in range(8):
                await harness.loop.step(tick * 0.1)

    asyncio.run(scenario())

    values = joint_values(harness.robots["right"])
    assert values[0] == 5.0 and values[4] == 5.02
    assert harness.loop.clamp_count == 0
    assert harness.loop.max_ee_speed_requested_m_s == pytest.approx(np.sqrt(3.0) * 0.02 * 10.0)


def test_the_gripper_is_still_step_clamped_while_the_joints_are_speed_clamped(
    build: Callable[..., conftest.LoopHarness],
) -> None:
    """The gripper keeps its per-tick step cap, `max_gripper_step`, independent of the end effector: with
    a kinematics that never moves, a 0 -> 1 gripper step is cut to 0.25 per tick while the joints pass whole,
    and the tick still counts as clamped."""
    client = _FlatClient(horizon=4, values=[0.0, 1.0])
    harness = build(
        control_hz=10.0,
        action_horizon=4,
        client=client,
        max_ee_speed_m_s=0.01,
        max_gripper_step=0.25,
        kinematics={arm: StillKinematics() for arm in control.ARMS},
    )

    async def scenario() -> None:
        async with harness.loop:
            for tick in range(8):
                await harness.loop.step(tick * 0.1)

    asyncio.run(scenario())

    commands = harness.robots["left"].commands
    assert [float(command[0]) for command in commands[4:8]] == [1.0, 1.0, 1.0, 1.0]
    np.testing.assert_allclose([float(command[6]) for command in commands[4:8]], [0.25, 0.5, 0.75, 1.0])
    assert harness.loop.clamp_count == 3


# The grip force cap (robots.gripper_force)


def _capped_robots(effort: float | None) -> dict[str, conftest.FakeRobot]:
    """Two fake arms whose driver reports `effort` as i2rt's `gripper_eff`, or none when it is None.

    @param effort The gripper effort both drivers report, or None for a driver that reports none.
    @return FakeRobot per arm, keyed as the loop keys its followers.
    """
    return {arm: conftest.FakeRobot(gripper_eff=effort) for arm in control.ARMS}


def test_a_commanded_close_is_backed_off_until_the_measured_effort_sits_at_the_cap(
    build: Callable[..., conftest.LoopHarness],
) -> None:
    """The driver reports five times the cap's effort against a commanded full close on a jaw measured at
    0.5, so from the first tick the hard floor holds the command `cap / stiffness` closed of the jaw, and the
    back-off then integrates on top of it (gain 0.4 per unit of relative error per second, 0.4 * 4 * 0.1 s a
    tick at 10 Hz) up to the jaw and no further: a fake effort that never falls is a cap that cannot be met, and
    opening past the jaw would only let go of the object."""
    client = _FlatClient(horizon=4, values=[0.0, 0.0])
    harness = build(
        control_hz=10.0,
        action_horizon=4,
        client=client,
        grip_force_cap=2.0,
        robots=_capped_robots(10.0),
    )

    async def scenario() -> None:
        async with harness.loop:
            for tick in range(8):
                await harness.loop.step(tick * 0.1)

    asyncio.run(scenario())

    # Eight ticks, then the hold `__aexit__` re-sends as the ninth write.
    grippers = [float(command[6]) for command in harness.robots["left"].commands]
    floor = 0.5 - 2.0 / gripper_force.DEFAULT_STIFFNESS
    np.testing.assert_allclose(grippers[:8], [min(0.5, floor + 0.16 * tick) for tick in range(8)])
    # The joints are no business of the cap's.
    assert [float(command[0]) for command in harness.robots["left"].commands[:8]] == [0.0] * 8


def test_a_driver_that_reports_no_gripper_effort_is_commanded_as_the_policy_sent_it(
    build: Callable[..., conftest.LoopHarness],
) -> None:
    """The cap has nothing to regulate on without feedback, so a station whose driver reports no `gripper_eff`
    (a sim robot, an older i2rt) runs as it did before the cap: the close reaches the arms whole."""
    client = _FlatClient(horizon=4, values=[0.0, 0.0])
    harness = build(
        control_hz=10.0,
        action_horizon=4,
        client=client,
        grip_force_cap=2.0,
        robots=_capped_robots(None),
    )

    async def scenario() -> None:
        async with harness.loop:
            for tick in range(8):
                await harness.loop.step(tick * 0.1)

    asyncio.run(scenario())

    assert [float(command[6]) for command in harness.robots["left"].commands[:8]] == [0.0] * 8


def test_a_task_change_drops_the_back_off_so_the_new_episode_grips_from_scratch(
    build: Callable[..., conftest.LoopHarness],
) -> None:
    """A task change ramps from the measured pose again, and the cap is reset with it: a back-off carried into the
    new episode would open the gripper for a squeeze that is over."""
    client = _FlatClient(horizon=4, values=[0.0, 0.0, 0.0])
    harness = build(
        control_hz=10.0,
        action_horizon=4,
        client=client,
        refill_at_remaining=2,
        grip_force_cap=2.0,
        robots=_capped_robots(10.0),
    )

    async def scenario() -> None:
        async with harness.loop:
            for tick in range(4):
                await harness.loop.step(tick * 0.1)
            harness.loop.set_task("stack the blocks")
            for tick in range(4, 8):
                await harness.loop.step(tick * 0.1)

    asyncio.run(scenario())

    # A refill was in flight when the task changed, so the change lands on the tick after that one;
    # the back-off goes with the ramp it brings, and the new episode builds its own from zero.
    grippers = [float(command[6]) for command in harness.robots["left"].commands]
    floor = 0.5 - 2.0 / gripper_force.DEFAULT_STIFFNESS
    expected = [min(0.5, floor + 0.16 * tick) for tick in range(5)] + [
        min(0.5, floor + 0.16 * tick) for tick in range(3)
    ]
    np.testing.assert_allclose(grippers[:8], expected)


def test_the_plan_snapshot_carries_the_remaining_rows_and_the_history(
    build: Callable[..., conftest.LoopHarness],
) -> None:
    """After the first tick of a 10 Hz loop with 4-row chunks, `plan` holds the three rows still to run,
    due 0.1, 0.2 and 0.3 s ahead (row k is due at observed_at + k / hz), their planned end-effector speed from the
    injected kinematics (sqrt(3) rad per row on the first three joints: 17.3 m/s; NaN for row 0 of the first chunk,
    which has no predecessor), and one tick of history whose measured speed is NaN. After four ticks the plan is
    empty and the history is four ticks at 0.3, 0.2, 0.1 s ago and now, the arms measured still (speed 0) and
    commanded the rows 0..3. Every snapshot is a new object, never mutated."""
    harness = build(control_hz=10.0, action_horizon=4, max_ee_speed_m_s=100.0, kinematics=linear_kinematics())
    snapshots: list[control.PlanSnapshot] = []

    async def scenario() -> None:
        async with harness.loop:
            for tick in range(4):
                await harness.loop.step(tick * 0.1)
                snapshots.append(harness.loop.plan)

    asyncio.run(scenario())

    first = snapshots[0]
    np.testing.assert_allclose(first.plan_t_s, [0.1, 0.2, 0.3], atol=1e-9)
    assert first.plan["left"].shape == (3, 7) and first.plan["right"][:, 0].tolist() == [1.0, 2.0, 3.0]
    np.testing.assert_allclose(first.plan_ee_speed_m_s["left"], np.sqrt(3.0) * 10.0)
    assert first.history_t_s.tolist() == [0.0] and first.clamped.tolist() == [False]
    assert first.measured["left"].shape == (1, 7) and first.commanded["right"][0, 0] == 0.0
    assert np.isnan(first.ee_speed_m_s["left"][0]) and first.max_ee_speed_m_s == 100.0

    last = snapshots[-1]
    assert last.plan_t_s.size == 0 and last.plan["left"].shape == (0, 7) and last.plan_ee_speed_m_s["right"].size == 0
    np.testing.assert_allclose(last.history_t_s, [-0.3, -0.2, -0.1, 0.0], atol=1e-9)
    assert last.commanded["left"][:, 0].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert last.measured["right"][:, 0].tolist() == [0.0] * 4 and last.measured["right"][0, 6] == 0.5
    assert np.isnan(last.ee_speed_m_s["right"][0]) and last.ee_speed_m_s["right"][1:].tolist() == [0.0, 0.0, 0.0]
    assert first is not last and first.history_t_s.tolist() == [0.0]


def test_the_plan_snapshot_history_is_bounded_and_the_second_chunk_plans_from_the_last_command(
    build: Callable[..., conftest.LoopHarness],
) -> None:
    """The history keeps at most PLAN_HISTORY_S seconds of ticks (30 at 10 Hz) however long the run; the
    second chunk's first row is planned from the last command (row 3 to row 4: one row's speed, no NaN)."""
    harness = build(control_hz=10.0, action_horizon=4, max_ee_speed_m_s=100.0, kinematics=linear_kinematics())
    second_chunk_plan: list[control.PlanSnapshot] = []

    async def scenario() -> None:
        async with harness.loop:
            for tick in range(40):
                await harness.loop.step(tick * 0.1)
                if tick == 4:
                    second_chunk_plan.append(harness.loop.plan)

    asyncio.run(scenario())

    plan = harness.loop.plan
    size = int(np.ceil(control.PLAN_HISTORY_S * 10.0))
    assert plan.history_t_s.shape == (size,) and plan.commanded["left"].shape == (size, 7)
    np.testing.assert_allclose(plan.history_t_s, np.arange(-size + 1, 1) * 0.1, atol=1e-9)
    assert plan.commanded["left"][-1, 0] == 39.0 and plan.commanded["left"][0, 0] == 39.0 - size + 1
    np.testing.assert_allclose(second_chunk_plan[0].plan_ee_speed_m_s["left"], np.sqrt(3.0) * 10.0)


def test_the_trace_file_has_one_row_per_commanded_tick(
    build: Callable[..., conftest.LoopHarness], tmp_path: pathlib.Path
) -> None:
    """With trace_path set the loop appends a header and one row per commanded tick, every row
    carrying every column of `control.trace_columns()` for the arms' joints."""
    trace = tmp_path / "trace.csv"
    harness = build(control_hz=10.0, action_horizon=4, trace_path=str(trace))

    async def scenario() -> None:
        async with harness.loop:
            for tick in range(4):
                await harness.loop.step(tick * 0.1)

    asyncio.run(scenario())

    rows = [line.split(",") for line in trace.read_text().splitlines()]
    assert rows[0] == list(control.trace_columns(YAM_JOINTS))
    assert len(rows) == 1 + 4
    assert all(len(row) == len(rows[0]) for row in rows)
    # chunk_id and row_index of the four rows of chunk 1; a one-tick ramp passes through (alpha 1), no clamp.
    assert [row[1:5] for row in rows[1:]] == [["1", str(i), "1.000", "0"] for i in range(4)]
    # No kinematics: the end-effector speed columns are NaN. The last column counts the command periods written so
    # far: one per tick with the synchronous switch.
    assert [row[-3:-1] for row in rows[1:]] == [["nan", "nan"]] * 4
    assert [row[-1] for row in rows[1:]] == ["1", "2", "3", "4"]


# --- Command path: interpolation, thread --------------------------------------------

## Command period at 250 Hz, what the command-path tests step the run with.
COMMAND_PERIOD_S = 1.0 / 250.0


def test_a_submitted_row_is_run_out_over_the_submit_gap_at_command_hz(
    build: Callable[..., conftest.LoopHarness],
) -> None:
    """A row submitted 33 ms after the last is reached in about eight 4 ms steps, each one command
    period further along the line from the previous row, monotone, and then re-sent whole every period."""
    harness = build(control_hz=30.0, command_hz=250.0, command_interp_max_s=0.10)
    tick_s = 1.0 / 30.0

    async def scenario() -> None:
        async with harness.loop:
            await harness.loop.step(0.0)
            # Submits row 1 and writes the first period of its run itself (the synchronous switch).
            await harness.loop.step(tick_s)
            for period in range(1, 10):
                harness.loop._commands.once(tick_s + period * COMMAND_PERIOD_S)

    asyncio.run(scenario())

    left = harness.robots["left"].commands
    joints = [float(command[0]) for command in left]
    # Row 0 as is (nothing to run from), then row 1 at 0.12, 0.24, ... 0.96 of the way, then the row itself.
    assert joints[0] == 0.0
    assert joints[1:9] == pytest.approx([0.12 * step for step in range(1, 9)])
    assert joints[9:12] == [1.0, 1.0, 1.0]
    assert joints[:10] == sorted(joints[:10]) and len(set(joints[:10])) == 10
    # The gripper runs the same line: row 1's 0.1 reached in the same steps.
    assert [float(command[6]) for command in left[1:3]] == pytest.approx([0.012, 0.024])


def test_a_hold_clears_the_target_and_the_next_row_runs_from_where_the_arms_were_left(
    build: Callable[..., conftest.LoopHarness],
) -> None:
    """A held tick clears the target, so later command periods write nothing beyond the hold's own
    re-command, and the row after the hold runs from the last written command rather than jumping."""
    harness = build()

    async def scenario() -> None:
        async with harness.loop:
            await harness.loop.step(0.0)
            await harness.loop.step(stale_now_s())
            for period in range(1, 4):
                harness.loop._commands.once(period * COMMAND_PERIOD_S)
            assert len(harness.robots["left"].commands) == 2
            await harness.loop.step(time.monotonic())

    asyncio.run(scenario())

    # Row 0, the hold's re-command of it, row 1 (reached whole: the run window is one period), the __aexit__ hold.
    assert joint_values(harness.robots["left"]) == [0.0, 0.0, 1.0, 1.0]


def test_the_command_thread_writes_on_its_own_and_stops_with_the_loop(
    build: Callable[..., conftest.LoopHarness],
) -> None:
    """With the thread on, one submitted row is re-sent every command period without the tick doing
    anything, and nothing is written once the loop has exited."""
    harness = build(command_thread=True, command_hz=250.0)
    left = harness.robots["left"]

    async def scenario() -> None:
        async with harness.loop:
            await harness.loop.step(0.0)
            deadline_s = time.monotonic() + THREAD_WAIT_S
            while len(left.commands) < 3 and time.monotonic() < deadline_s:
                await asyncio.sleep(0.001)

    asyncio.run(scenario())

    written = len(left.commands)
    assert written >= 4
    time.sleep(0.02)
    assert len(left.commands) == written
    for command in left.commands:
        np.testing.assert_allclose(command, [0.0] * 6 + [0.0])


def test_a_write_failure_on_the_command_thread_stops_the_loop_and_disengages(
    build: Callable[..., conftest.LoopHarness],
) -> None:
    """A write that fails on the command thread ends the thread, is raised by the next step() as
    the ArmError it was, and __aexit__ disengages both arms as for an in-tick write failure."""
    robots = {"left": conftest.FakeRobot(has_zero_torque=True), "right": conftest.FakeRobot(has_zero_torque=True)}
    harness = build(robots=robots, command_thread=True, command_hz=250.0)
    robots["left"].write_error = errors.ArmError("arm 'left' rejected the command")

    async def scenario() -> None:
        async with harness.loop:
            await harness.loop.step(0.0)
            deadline_s = time.monotonic() + THREAD_WAIT_S
            while harness.loop._commands.error is None and time.monotonic() < deadline_s:
                await asyncio.sleep(0.001)
            await harness.loop.step(0.1)

    with pytest.raises(errors.ArmError, match="rejected the command"):
        asyncio.run(scenario())

    assert robots["left"].zero_torque_calls == 1
    assert robots["right"].zero_torque_calls == 1
    assert robots["left"].commands == []
    assert robots["right"].commands == []


# Task change mid-run (set_task) and the timing snapshot


def test_set_task_mid_run_resets_the_episode_once_and_the_next_request_carries_the_new_prompt(
    build: Callable[..., conftest.LoopHarness], caplog: pytest.LogCaptureFixture
) -> None:
    """set_task() between refills: the next tick resets the client's episode exactly once, drops the buffered
    rows, ramps again from the measured pose, and the request that follows carries the new prompt."""
    harness = build(action_horizon=4, refill_at_remaining=2)

    async def scenario() -> None:
        async with harness.loop:
            await harness.loop.step(0.0)
            # Three rows buffered after this pop: no prefetch is out when the task changes.
            await harness.loop.step(0.1)
            harness.loop.set_task("stack the blocks")
            await harness.loop.step(0.2)
            await harness.loop.step(0.3)

    with caplog.at_level(logging.INFO, logger="robots.control"):
        asyncio.run(scenario())

    assert harness.client.resets == 1
    # One request with the old prompt went out before the reset; the second is the new episode's first, and the
    # refill's thread handed the client that tick after the reset (the fake refuses a request with none).
    assert harness.client.reset_after_calls == [1]
    assert [obs["prompt"] for obs in harness.client.observations] == [conftest.TASK, "stack the blocks"]
    assert harness.client.ticks[-1][1]["prompt"] == "stack the blocks"
    # Rows 0 and 1 of chunk 0; rows 2 and 3 were dropped with the old task; then chunk 1 from its row 0.
    assert joint_values(harness.robots["left"])[:4] == [0, 1, 4, 5]
    assert len(records_with(caplog, "ramp started")) == 2
    changed = records_with(caplog, "task changed from")
    assert len(changed) == 1 and changed[0].levelno == logging.INFO
    assert "'pick up the cup' to 'stack the blocks'" in changed[0].getMessage()
    assert harness.loop.task == "stack the blocks"


def test_set_task_waits_for_the_refill_in_flight_and_is_applied_once_it_lands(
    build: Callable[..., conftest.LoopHarness],
) -> None:
    """A reset during a request is refused by the client, so the change waits: ticks while the refill is out
    keep draining the old chunk with no reset; the tick after it lands resets, drops the rows and refills."""
    gate = threading.Event()
    client = AnswerOnceClient(horizon=4, gate=gate)
    harness = build(action_horizon=4, refill_at_remaining=2, client=client)

    async def scenario() -> None:
        try:
            async with harness.loop:
                await harness.loop.step(0.0)
                await harness.loop.step(0.1)
                # Two rows buffered before this tick: the gated prefetch starts here.
                await harness.loop.step(0.2)
                harness.loop.set_task("stack the blocks")
                await harness.loop.step(0.3)
                assert client.resets == 0 and harness.loop.task == conftest.TASK
                gate.set()
                await wait_for_refill_task()
                await harness.loop.step(0.4)
        finally:
            gate.set()
            await wait_for_refill_task()

    asyncio.run(scenario())

    assert client.resets == 1 and client.reset_after_calls == [2]
    assert [obs["prompt"] for obs in client.observations] == [conftest.TASK, conftest.TASK, "stack the blocks"]
    # All of chunk 0 while the prefetch was out; chunk 1 landed and was dropped with the old task; chunk 2 row 0.
    assert joint_values(harness.robots["left"])[:5] == [0, 1, 2, 3, 8]


def test_timing_snapshot_carries_the_last_rtt_the_percentiles_and_the_counters(
    build: Callable[..., conftest.LoopHarness],
) -> None:
    """Round trips are measured on the loop's clock between the tick that started a refill and the one that
    collected it: 0, 100 and 200 ms here (a 10 Hz loop, so both prefetches land inside the horizon); `timing`
    is rebound per chunk with count, last, p50 and p95, and `chunk_count` counts the installs."""
    harness = build(control_hz=10.0, action_horizon=4, refill_at_remaining=2)
    snapshots = []

    async def scenario() -> None:
        async with harness.loop:
            # The prefetches start at 0.2 and 0.4 (two rows left); each is let land before the collecting tick.
            for now_s in (0.0, 0.1, 0.2, 0.3, 0.4, 0.6):
                if now_s in (0.3, 0.6):
                    await wait_for_refill_task()
                await harness.loop.step(now_s)
                snapshots.append(harness.loop.timing)

    asyncio.run(scenario())

    timing = harness.loop.timing
    assert timing.count == 3 and harness.loop.chunk_count == 3 and harness.loop.stall_count == 0
    assert timing.last_rtt_ms == pytest.approx(200.0)
    assert timing.p50_ms == pytest.approx(100.0) and timing.p95_ms == pytest.approx(200.0)
    assert timing.serve_infer_ms is None and timing.response_bytes is None
    # Rebound, never mutated: the snapshot taken after the first chunk still says one refill.
    assert snapshots[0].count == 1 and snapshots[0].last_rtt_ms == 0.0
    assert snapshots[0] is not snapshots[-1]
    assert timing.as_dict()["count"] == 3 and control.RefillTiming().as_dict()["p95_ms"] is None


def test_a_commanded_tick_publishes_a_snapshot_of_state_and_action_in_the_ring_S7(
    build: Callable[..., conftest.LoopHarness],
) -> None:
    """Every commanded tick rebinds `loop.tick` and appends it to `loop.tick_history`; a held tick appends nothing."""
    harness = build()
    loop = harness.loop

    async def scenario() -> None:
        async with loop:
            assert loop.tick is None and len(loop.tick_history) == 0
            await loop.step(harness.clock())
            await loop.step(harness.clock())

    asyncio.run(scenario())
    assert loop.tick is not None and len(loop.tick_history) == 2
    last = loop.tick_history[-1]
    assert last is loop.tick and last.tick == loop.tick_count and last.t_s == loop.state.timestamp_s
    assert last.state is loop.state, "the snapshot carries the very state the tick published"
    assert last.action.left.positions_rad.shape == (6,) and 0.0 <= last.action.left.gripper <= 1.0
    assert last.chunk == loop.chunk_count and last.row >= 0 and isinstance(last.clamped, bool)
    assert loop.tick_history.maxlen == control.TICK_RING == 200


def test_a_snapshot_names_the_policy_session_and_a_refill_wait_marks_its_tick_S8_I13a(
    build: Callable[..., conftest.LoopHarness],
) -> None:
    """The snapshot carries the policy client's session id, and the tick that waited for a chunk is remembered
    for the live agent's guard."""
    harness = build()
    loop = harness.loop
    harness.client.session_id = "711c847e-c28d-4f4e-885c-9d473bd4b79d"

    async def scenario() -> None:
        async with loop:
            await loop.step(harness.clock())
            await loop.step(harness.clock())

    asyncio.run(scenario())
    assert loop.session_id == "711c847e-c28d-4f4e-885c-9d473bd4b79d"
    assert loop.tick is not None and loop.tick.session_id == "711c847e-c28d-4f4e-885c-9d473bd4b79d"
    assert list(loop.refill_wait_ticks) == [1], "the first tick waited for the first chunk; the second did not"


# The refill point is the client's (dream.pacing)


def test_a_sync_client_gets_every_row_of_the_chunk_run_before_the_next_request(
    build: Callable[..., conftest.LoopHarness],
) -> None:
    """In sync the client answers "due" only once nothing is left (dream.pacing), so the loop runs the
    whole action horizon, not an execution horizon, before it asks again."""
    client = conftest.FakeClient(horizon=6)
    harness = build(control_hz=10.0, action_horizon=6, refill_at_remaining=None, client=client)

    async def scenario() -> None:
        async with harness.loop:
            for tick in range(7):
                await harness.loop.step(tick * 0.1)
                await wait_for_refill_task()

    asyncio.run(scenario())

    assert client.calls == 2, "one request per whole chunk"
    assert joint_values(harness.robots["left"])[:7] == [0, 1, 2, 3, 4, 5, 6]


def test_the_first_sync_chunk_after_a_switch_runs_past_the_rows_run_in_flight_then_to_its_end(
    build: Callable[..., conftest.LoopHarness],
) -> None:
    """The request behind the first sync chunk went out in async while the arms moved, so the loop installs
    that chunk past the rows they ran during the round trip; then, the client now sync, it runs the rest of the chunk
    to its last row before the next request (the station's async-to-sync transition: about 48 - 15 = 33 rows)."""
    gate = threading.Event()
    gate.set()
    client = conftest.FakeClient(horizon=8, gate=gate)
    client.refill_answer = lambda executed, remaining: remaining == 0 or executed >= 2  # async: an early request
    harness = build(control_hz=10.0, action_horizon=8, refill_at_remaining=None, client=client)

    async def scenario() -> None:
        async with harness.loop:
            await harness.loop.step(0.0)  # chunk 0 installs whole; row 0 runs
            gate.clear()
            for tick in range(1, 6):  # the async request goes out at tick 2, after two rows; rows 2..5 run meanwhile
                await harness.loop.step(tick * 0.1)
            client.mode = "sync"  # the slow reply switched the client
            client.refill_answer = lambda executed, remaining: remaining == 0
            gate.set()
            await wait_for_refill_task()
            for tick in range(6, 12):
                await harness.loop.step(tick * 0.1)
                await wait_for_refill_task()

    asyncio.run(scenario())

    # Chunk 1 (rows 8..15) answered the observation of tick 2; ticks 2..5 ran rows 2..5 of chunk 0 meanwhile, so it
    # installs from its row 4 (value 12) and runs to its last row (15). Only then, at tick 10 with nothing left, does
    # the third request go out; chunk 2 (rows 16..23) installs whole after the hold.
    assert joint_values(harness.robots["left"])[:12] == [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17]
    assert client.calls == 3
    # The loop asked a sync client on every row of chunk 1's four and was told "due" only at the last.
    start = client.refill_asks.index((0, 4))
    assert client.refill_asks[start : start + 5] == [(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)]


def test_unpinned_loop_asks_the_client_with_rows_executed_and_rows_left(
    build: Callable[..., conftest.LoopHarness],
) -> None:
    """Without a pinned refill point the loop asks `client.refill_due(executed, remaining)` on every tick
    with none in flight, counting rows executed of the chunk since its install, and refills when it says so."""
    client = conftest.FakeClient(horizon=6)
    client.refill_answer = lambda executed, remaining: remaining == 0 or executed >= 4
    harness = build(control_hz=10.0, action_horizon=6, refill_at_remaining=None, client=client)

    async def scenario() -> None:
        async with harness.loop:
            for tick in range(6):
                await harness.loop.step(tick * 0.1)
                await wait_for_refill_task()

    asyncio.run(scenario())

    # Tick 0: empty buffer. Ticks 1..4: rows 1..4 executed before each ask. The fifth ask says yes.
    assert client.refill_asks[:5] == [(0, 0), (1, 5), (2, 4), (3, 3), (4, 2)]
    assert client.calls == 2


def test_a_client_that_always_says_due_keeps_exactly_one_refill_in_flight(
    build: Callable[..., conftest.LoopHarness],
) -> None:
    """A client that answers "due" on every ask (a round trip past the whole chunk) gets one more call per
    tick after the install, never two in flight."""
    client = conftest.FakeClient(horizon=4)
    client.refill_answer = lambda executed, remaining: True
    harness = build(control_hz=10.0, action_horizon=4, refill_at_remaining=None, client=client)
    calls_per_tick: list[int] = []

    async def scenario() -> None:
        async with harness.loop:
            await harness.loop.step(0.0)
            for tick in range(1, 5):
                await harness.loop.step(tick * 0.1)
                await wait_for_refill_task()
                calls_per_tick.append(client.calls)

    asyncio.run(scenario())

    # Every tick after the install collects the previous refill and starts the next: one more call per tick, never two.
    assert calls_per_tick == [2, 3, 4, 5]


CONNECTION_REFUSED = dream.DreamProtocolError(
    "inference request to http://127.0.0.1:9/v1/infer failed: [Errno 111] Connection refused; "
    "is DreamAPI running there?"
)


def test_a_gateway_nothing_answers_at_is_asked_once_a_second_and_ends_the_run_at_first_chunk_timeout_s(
    build: Callable[..., conftest.LoopHarness], caplog: pytest.LogCaptureFixture
) -> None:
    client = FailingClient(horizon=4, fail=lambda _: CONNECTION_REFUSED)
    harness = build(client=client, first_chunk_timeout_s=12.0)

    async def scenario() -> None:
        async with harness.loop:
            with pytest.raises(errors.ControlLoopError, match="first_chunk_timeout_s=12 s") as raised:
                for tick in range(200):
                    await harness.loop.step(tick * 0.1)
            assert str(CONNECTION_REFUSED) in str(raised.value)

    with caplog.at_level(logging.WARNING, logger="robots.control"):
        asyncio.run(scenario())

    assert 11 <= client.calls <= 13, "about one try a second"
    lines = [record.getMessage() for record in records_with(caplog, "get_action failed")]
    assert len(lines) == 3, lines
    assert "retrying after 1.0 s" in lines[0] and "in a row" not in lines[0]
    assert re.search(r"failed \d+ times in a row over 1\d s, retrying after 1.0 s: inference request", lines[2])


def test_a_warming_pool_is_waited_out_past_first_chunk_timeout_s_up_to_the_limit(
    build: Callable[..., conftest.LoopHarness], caplog: pytest.LogCaptureFixture
) -> None:
    client = WarmingClient(horizon=4, refused=range(10_000), retry_after_s=30.0)
    harness = build(client=client, first_chunk_timeout_s=120.0, stale_after_s=32 * 60.0)

    async def scenario() -> None:
        async with harness.loop:
            for tick in range(31):
                await harness.loop.step(tick * 60.0)
            with pytest.raises(
                errors.ControlLoopError, match="has not started on DreamAPI after 30 minutes"
            ) as raised:
                await harness.loop.step(31 * 60.0)
            assert errors.error_code(raised.value) == "pool_warming"

    with caplog.at_level(logging.WARNING, logger="robots.control"):
        asyncio.run(scenario())

    assert client.calls == 32
    progress = records_with(caplog, "the policy is still starting on DreamAPI (600 s so far")
    assert len(progress) == 1 and "retrying after 30.0 s" in progress[0].getMessage()
    assert not records_with(caplog, "first_chunk_timeout_s")


def test_a_first_chunk_after_a_long_start_stops_the_run_and_never_moves_the_arms(
    build: Callable[..., conftest.LoopHarness],
) -> None:
    """A client that answers only past MOVE_AFTER_WAIT_LIMIT_S after the first tick (a cold pool) is not run: the
    run stops with ModelReady, its code for the page, and every command the arms got was a hold."""
    client = WarmingClient(horizon=4, refused=range(3), retry_after_s=30.0)
    harness = build(client=client, first_chunk_timeout_s=120.0, stale_after_s=10 * 60.0)

    async def scenario() -> None:
        async with harness.loop:
            with pytest.raises(errors.ModelReady, match=r"first answered \d+ s after the run started") as raised:
                for tick in range(10):
                    await harness.loop.step(tick * 30.0)
            assert errors.error_code(raised.value) == "ModelReady"

    asyncio.run(scenario())

    assert client.calls == 4
    left = harness.robots["left"].commands
    assert left
    for command in left:
        np.testing.assert_allclose(command, HOLD_COMMAND)


def test_a_serve_fault_that_repeats_word_for_word_ends_the_run_with_dreamapis_message(
    build: Callable[..., conftest.LoopHarness],
) -> None:
    client = FailingClient(horizon=4, fail=lambda _: MISSING_CAMERA)
    harness = build(client=client)

    async def scenario() -> None:
        async with harness.loop:
            for tick in range(control.SERVE_FAULT_REPEATS - 1):
                await harness.loop.step(float(tick))
            with pytest.raises(errors.ControlLoopError, match="will not help") as raised:
                await harness.loop.step(float(control.SERVE_FAULT_REPEATS - 1))
            assert "observation is missing ['video.overhead']" in str(raised.value)
            assert errors.error_code(raised.value) == "error"

    asyncio.run(scenario())

    assert client.calls == control.SERVE_FAULT_REPEATS


def test_a_silent_serve_is_held_through_and_never_counted_as_a_repeated_fault(
    build: Callable[..., conftest.LoopHarness],
) -> None:
    silent = dream.DreamServerError(
        "DreamAPI refused the inference request: DA-5000 SERVE_FAULT (timeout): tcp://127.0.0.1:5555: "
        "no reply in 0.5s",
        code="DA-5000",
        reason="timeout",
        retryable=True,
    )
    fails = set(range(3 * control.SERVE_FAULT_REPEATS))
    client = FailingClient(horizon=1, fail=lambda call: silent if call in fails else None)
    harness = build(action_horizon=1, max_stall_ticks=100, client=client)

    async def scenario() -> None:
        async with harness.loop:
            for tick in range(3 * control.SERVE_FAULT_REPEATS + 3):
                await harness.loop.step(float(tick))

    asyncio.run(scenario())

    assert client.calls > 3 * control.SERVE_FAULT_REPEATS
    assert harness.loop.chunk_count >= 1 and harness.loop.refill_error is None


def test_a_serve_fault_that_passes_is_retried_and_the_run_goes_on(build: Callable[..., conftest.LoopHarness]) -> None:
    fails = set(range(control.SERVE_FAULT_REPEATS - 1))
    client = FailingClient(horizon=1, fail=lambda call: MISSING_CAMERA if call in fails else None)
    harness = build(action_horizon=1, max_stall_ticks=100, client=client)

    async def scenario() -> None:
        async with harness.loop:
            for tick in range(6):
                await harness.loop.step(float(tick))

    asyncio.run(scenario())

    assert harness.loop.chunk_count >= 1 and harness.loop.refill_error is None


def test_a_sync_hold_through_an_inference_counts_no_stall(build: Callable[..., conftest.LoopHarness]) -> None:
    harness = build(action_horizon=4, max_stall_ticks=3, action_timeout_s=3.0)
    answer = harness.client.get_action

    def slow_answer(*, commanded: Mapping[str, Any] | None = None) -> dict[str, Any]:
        harness.clock.now_s += 1.0
        return answer(commanded=commanded)

    harness.client.get_action = slow_answer

    async def scenario() -> None:
        async with harness.loop:
            for tick in range(13):
                await harness.loop.step(tick / 30.0)

    asyncio.run(scenario())

    assert harness.client.calls >= 3
    assert harness.loop.stall_count == 0


def test_an_arm_fault_holds_brakeless_arms_instead_of_letting_them_drop(
    build: Callable[..., conftest.LoopHarness], caplog: pytest.LogCaptureFixture
) -> None:
    robots = {"left": conftest.FakeRobot(has_zero_torque=True), "right": conftest.FakeRobot(has_zero_torque=True)}
    harness = build(robots=robots)
    for arm in harness.arms.values():
        arm._driver.drops_when_limp = True
    robots["right"].read_error = errors.ArmError("arm 'right' wrist_yaw reports fault 0x9, under voltage")

    async def scenario() -> None:
        async with harness.loop:
            await harness.loop.step(0.0)

    with pytest.raises(errors.ArmError, match="under voltage"):
        asyncio.run(scenario())

    for robot in robots.values():
        assert robot.zero_torque_calls == 0
        assert len(robot.commands) == 1


def test_an_arm_alarm_stops_the_loop_with_a_safety_stop_and_both_arms_hold(
    build: Callable[..., conftest.LoopHarness],
) -> None:
    robots = {"left": conftest.FakeRobot(has_gravity_comp=True), "right": conftest.FakeRobot(has_gravity_comp=True)}
    harness = build(robots=robots)
    alarm = "elbow_flex coil reached 66 C (the run stops at 65 C); the arm parks. Let it cool below 55 C"

    async def scenario() -> None:
        async with harness.loop:
            await harness.loop.step(0.0)
            harness.arms["left"]._driver.alarm = lambda: alarm
            await harness.loop.step(1.0 / 30)

    with pytest.raises(errors.SafetyStop, match="arm 'left': elbow_flex coil reached 66 C"):
        asyncio.run(scenario())

    for robot in robots.values():
        assert robot.gravity_comp_calls == 0
        assert len(robot.commands) == 2
