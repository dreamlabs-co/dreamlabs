"""End-to-end tests for `robots.run_report`: a loop over a simulated follower writes its trace, the report reads it.

The follower (#SimArm) tracks each command with a time constant, optionally late, slow or wobbling, which is how a
bridge, a driver or a controller of a customer's goes wrong; the client (#SineClient) plans smooth sine waves; the
cameras (#ScriptedCamera) hand the loop the frames each test draws. Ticks come from a `ManualClock`, so each run is
the same every time.
"""

import asyncio
import base64
import collections
import io
import math
import pathlib
import shutil
from collections.abc import Callable
from typing import Any

import conftest
import cv2
import numpy as np
import numpy.typing as npt
import pytest

from robots import cameras, cli, control, errors, run_report, types

## Ticks per second of every run here, the station's rate.
HZ = 30.0

## Rows per chunk, and the rows left when the next refill starts.
HORIZON = 16
REFILL_AT = 8

## Each joint's sine wave: amplitude, radians, and frequency, Hz; slow enough that a follower that keeps up stays
## well inside #robots.run_report.DIVERGE_RAD.
AMPLITUDE = (0.3, 0.25, 0.2, 0.3, 0.15, 0.2)
FREQUENCY = (0.25, 0.2, 0.3, 0.25, 0.35, 0.2)

## Capture size of the test cameras, and the policy size their frames are sent at.
CAPTURE = (128, 72)
SENT = (64, 36)

## The follower's time constant, seconds: a stiff arm.
TAU_S = 0.03


class SimArm(conftest.FakeRobot):
    """A follower that tracks its commands: each tick it moves `dt / tau_s` of the way to the command it got `delay`
    ticks ago, no faster than `max_speed` rad/s per joint, plus a `wobble` (amplitude, Hz) on joint 0."""

    def __init__(self, delay: int, max_speed: npt.NDArray[np.float64], wobble: tuple[float, float]) -> None:
        """@param delay Ticks between a command and the arm starting for it.
        @param max_speed Per joint, the fastest the arm moves, rad/s.
        @param wobble Amplitude, radians, and frequency, Hz, of an oscillation on joint 0; 0 amplitude for none.
        """
        super().__init__()
        self._late: collections.deque[npt.NDArray[np.float64]] = collections.deque()
        self._delay = delay
        self._max_speed = max_speed
        self._wobble = wobble
        self._tracked = np.zeros(conftest.ARM_DOF)
        self._ticks = 0

    def command_joint_pos(self, positions: np.ndarray) -> None:
        """Records the command and moves the arm one tick along.

        @param positions Joint targets plus gripper, shape [7].
        """
        super().command_joint_pos(positions)
        self._late.append(np.asarray(positions[: conftest.ARM_DOF], dtype=np.float64))
        target = self._late.popleft() if len(self._late) > self._delay else self._tracked
        dt = 1.0 / HZ
        step = np.clip((target - self._tracked) * min(1.0, dt / TAU_S), -self._max_speed * dt, self._max_speed * dt)
        self._tracked = self._tracked + step
        self._ticks += 1
        amplitude, hz = self._wobble
        self.joint_pos = self._tracked.copy()
        self.joint_pos[0] += amplitude * math.sin(2 * math.pi * hz * self._ticks / HZ)


class SineClient(conftest.FakeClient):
    """A client whose chunks follow #AMPLITUDE and #FREQUENCY from the observation's time, each joint scaled by
    `speed` (1 for the defaults); the gripper holds at 0.5."""

    def __init__(self, speed: float) -> None:
        """@param speed How many times faster than #FREQUENCY the waves run."""
        super().__init__(horizon=HORIZON)
        self._speed = speed
        self.refill_answer = lambda executed, remaining: remaining <= REFILL_AT

    def get_action(self, *, commanded: Any = None) -> dict[str, Any]:
        """Records the request as the fake does, then plans from the latest observation's time.

        @param commanded The joints last commanded, stored by the fake.
        @return `q_r_chunk` and `q_l_chunk`, `[H][7]` nested lists.
        """
        super().get_action(commanded=commanded)
        assert self._latest_s is not None
        times = self._latest_s + np.arange(HORIZON) / HZ  # [H]
        waves = np.sin(2 * np.pi * self._speed * np.outer(times, FREQUENCY)) * AMPLITUDE  # [H, 6]
        rows = [[*row, 0.5] for row in waves.tolist()]
        return {"q_r_chunk": rows, "q_l_chunk": [list(row) for row in rows]}


class ScriptedCamera:
    """A camera reader the test feeds: show() publishes a frame encoded at the policy size, captured at `now_s`."""

    def __init__(self, name: str) -> None:
        """@param name The camera's serve role."""
        self.name = name
        self.frame: types.Frame | None = None
        self.error: BaseException | None = None

    def show(self, rgb: npt.NDArray[np.uint8], now_s: float) -> None:
        """Publishes one capture and its encoded frame.

        @param rgb The capture, `[H, W, 3]`.
        @param now_s Its capture time.
        """
        jpeg = base64.b64encode(cameras.encode_jpeg(rgb, SENT)).decode()
        self.frame = types.Frame(data=rgb, camera=self.name, timestamp_s=now_s, jpeg=jpeg, jpeg_timestamp_s=now_s)

    def is_alive(self) -> bool:
        """@return True: a scripted camera never dies."""
        return True

    def start(self) -> None:
        """Nothing to start: the test publishes each frame."""

    def stop(self) -> None:
        """Nothing to stop."""


def scene(t: float, size: tuple[int, int]) -> npt.NDArray[np.uint8]:
    """A capture of a moving block on a lit table, with sensor noise: a different picture every tick.

    @param t Capture time, seconds.
    @param size `(width, height)`.
    @return RGB pixels, `[height, width, 3]`.
    """
    width, height = size
    rng = np.random.default_rng(round(t * 1000))
    image = np.full((height, width, 3), 120, dtype=np.float64)
    x = int((0.5 + 0.3 * math.sin(t)) * width)
    image[height // 3 : height // 2, max(0, x - width // 10) : x + width // 10] = (200, 60, 40)
    image += rng.normal(0.0, 3.0, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


## Draws each camera's capture at a time: `(camera, t) -> pixels`.
Draw = Callable[[str, float], npt.NDArray[np.uint8]]


def fly(
    tmp_path: pathlib.Path,
    seconds: float,
    arms: dict[str, SimArm],
    speed: float,
    draw: Draw,
) -> pathlib.Path:
    """Runs the loop for `seconds` on the arms, the trace on, and returns the trace's path.

    Each tick shows every camera a new capture, steps the loop, then waits for a refill the tick started, so the
    chunk lands before the next tick, as on a fast link.
    """
    trace = tmp_path / "trace.csv"
    readers = [ScriptedCamera(name) for name in conftest.CAMERA_NAMES]
    harness = conftest.make_loop(
        control_hz=HZ,
        action_horizon=HORIZON,
        refill_at_remaining=None,
        ramp_s=1.0,
        trace_path=str(trace),
        client=SineClient(speed),
        robots=dict(arms),
        readers=readers,
    )

    async def scenario() -> None:
        harness.clock.now_s = 100.0
        for reader in readers:
            reader.show(draw(reader.name, 0.0), harness.clock.now_s)
        async with harness.loop:
            for tick in range(round(seconds * HZ)):
                now_s = 100.0 + tick / HZ
                harness.clock.now_s = now_s
                for reader in readers:
                    reader.show(draw(reader.name, tick / HZ), now_s)
                await harness.loop.step(now_s)
                pending = {task for task in asyncio.all_tasks() if task is not asyncio.current_task()}
                if pending:
                    await asyncio.wait(pending, timeout=2.0)

    asyncio.run(scenario())
    return trace


def follower(delay: int, slow: dict[int, float], wobble: tuple[float, float]) -> SimArm:
    """A #SimArm, fast on every joint but the ones `slow` caps, rad/s."""
    max_speed = np.full(conftest.ARM_DOF, 100.0)
    for joint, cap in slow.items():
        max_speed[joint] = cap
    return SimArm(delay, max_speed, wobble)


def healthy() -> SimArm:
    """A #SimArm that keeps up."""
    return follower(1, {}, (0.0, 0.0))


def plain(name: str, t: float) -> npt.NDArray[np.uint8]:
    """Every camera's capture is the #scene()."""
    return scene(t, CAPTURE)


def failed(result: run_report.Report) -> dict[str, run_report.Check]:
    """The checks with a problem, by name."""
    return {check.name: check for check in result.checks if check.kind is not None}


def test_a_follower_that_keeps_up_passes_every_check(tmp_path: pathlib.Path) -> None:
    """A stiff arm one tick late, live cameras: five PASS lines, one for the cameras, the state, the loop and one per
    arm with its joints' worst error and lag, and no worst problem."""
    trace = fly(tmp_path, 12.0, {"left": healthy(), "right": healthy()}, 1.0, plain)

    result = run_report.report(trace)

    cameras = f"cameras {', '.join(conftest.CAMERA_NAMES)}"
    assert [check.name for check in result.checks] == [cameras, "state", "loop", "left arm", "right arm"]
    assert failed(result) == {} and run_report.worst(result.checks) is None
    by_name = {check.name: check for check in result.checks}
    # A command reaches the arm one tick late, and is read back on the tick after: two ticks of lag.
    assert (
        by_name["right arm"]
        .line()
        .startswith(f"PASS     right arm: all {conftest.ARM_DOF} joints follow their commands, worst error ")
    )
    assert by_name["right arm"].text.endswith(f" rad, lag {2000 / HZ:.0f} ms")
    assert by_name[cameras].text == f"sent {SENT[0]}x{SENT[1]}, no black or frozen frames"
    assert result.run == f'12 s at {HZ:g} Hz, "{conftest.TASK}"'


def test_a_joint_its_bridge_slows_is_the_one_problem_and_the_worst(tmp_path: pathlib.Path) -> None:
    """A bridge that caps the right arm's joint 3 at 0.2 rad/s, under the 0.47 rad/s its wave asks: that joint, and
    only it, diverges, and it is the worst problem."""
    arms = {"left": healthy(), "right": follower(1, {3: 0.2}, (0.0, 0.0))}
    trace = fly(tmp_path, 12.0, arms, 1.0, plain)

    result = run_report.report(trace)

    assert list(failed(result)) == ["right arm joint 3"]
    check = failed(result)["right arm joint 3"]
    assert check.kind == "diverging" and check.text.startswith("off its command by ")
    assert run_report.worst(result.checks) == check
    names = [c.name for c in result.checks]
    assert names[names.index("right arm joint 3") + 1] == "right arm"
    others = {c.name: c for c in result.checks}["right arm"].text
    assert others.startswith(f"other {conftest.ARM_DOF - 1} joints follow their commands, worst error ")


def test_a_late_follower_shows_its_lag_and_diverges(tmp_path: pathlib.Path) -> None:
    """A left arm that starts each command 10 ticks late: its fastest joint leaves its command by more than
    #robots.run_report.DIVERGE_RAD, with about 11 ticks of lag, and the right arm passes."""
    arms = {"left": follower(10, {}, (0.0, 0.0)), "right": healthy()}
    trace = fly(tmp_path, 12.0, arms, 1.0, plain)

    result = run_report.report(trace)

    assert "left arm joint 0" in failed(result) and not any(name.startswith("right") for name in failed(result))
    lag_ms = int(failed(result)["left arm joint 0"].text.split("lag ")[1].removesuffix(" ms"))
    assert 10000 / HZ <= lag_ms <= 12000 / HZ


def test_a_wobbling_joint_is_flagged_as_oscillating(tmp_path: pathlib.Path) -> None:
    """A right arm whose joint 0 wobbles 0.02 rad at 6 Hz around its command: what it achieved oscillates."""
    arms = {"left": healthy(), "right": follower(1, {}, (0.02, 6.0))}
    trace = fly(tmp_path, 6.0, arms, 1.0, plain)

    result = run_report.report(trace)

    assert [(c.name, c.kind, c.text.split(",")[0]) for c in failed(result).values()] == [
        ("right arm joint 0", "oscillating", "achieved oscillates at 6 Hz")
    ]


class NanArm(SimArm):
    """A #SimArm whose driver reads joint 5 as NaN from tick `after` on."""

    def __init__(self, after: int) -> None:
        """@param after The first tick read as NaN."""
        super().__init__(1, np.full(conftest.ARM_DOF, 10.0), (0.0, 0.0))
        self._after = after

    def command_joint_pos(self, positions: np.ndarray) -> None:
        """Moves the arm one tick along, then spoils joint 5's reading once past `after`.

        @param positions Joint targets plus gripper, shape [7].
        """
        super().command_joint_pos(positions)
        if self._ticks > self._after:
            self.joint_pos[5] = math.nan


def test_a_nan_reading_is_flagged_on_its_joint_and_in_the_state_sent(tmp_path: pathlib.Path) -> None:
    """A right arm whose joint 5 reads NaN from 1.5 s on: that joint's line and the state line say so, with the
    counts, and NaN is the worst problem."""
    trace = fly(tmp_path, 4.0, {"left": healthy(), "right": NanArm(round(1.5 * HZ))}, 1.0, plain)

    result = run_report.report(trace)

    found = failed(result)
    assert set(found) == {"state", "right arm joint 5"}
    assert found["state"].text == "state.right_arm not a number in 2 of 4 sets sent"
    # The first spoiled reading is the one after tick 45's command, so the last 2.5 s less one tick are NaN.
    assert found["right arm joint 5"].text == f"{round(2.5 * HZ) - 1} ticks are not a number, first at t=1.5 s"
    assert run_report.worst(result.checks) == found["state"]


def test_black_frozen_and_stretched_cameras_are_flagged(tmp_path: pathlib.Path) -> None:
    """The top camera goes black for 3 s, the left one freezes for 5 s, and the right one captures 4:3 that is sent
    16:9: one PROBLEM line each, with the number and when."""

    def draw(name: str, t: float) -> npt.NDArray[np.uint8]:
        if name == "top" and 4.0 <= t < 7.0:
            return np.zeros((CAPTURE[1], CAPTURE[0], 3), dtype=np.uint8)
        if name == "left":
            return scene(3.0 if 3.0 <= t < 8.0 else t, CAPTURE)
        if name == "right":
            return scene(t, (64, 48))
        return scene(t, CAPTURE)

    trace = fly(tmp_path, 12.0, {"left": healthy(), "right": healthy()}, 1.0, draw)

    result = run_report.report(trace)

    found = failed(result)
    assert set(found) == {"camera top", "camera left", "camera right"}
    assert (found["camera top"].kind, found["camera top"].text) == ("black", "3 of 12 frames black, first at t=4.0 s")
    assert found["camera left"].kind == "frozen" and found["camera left"].text.startswith("the same picture for ")
    assert found["camera right"].text == "captured 64x48, sent 64x36: the model sees it 1.33x wider"
    assert run_report.worst(result.checks) == found["camera top"]


def test_each_run_replaces_the_last_one_s_trace(tmp_path: pathlib.Path) -> None:
    """Two runs on one trace path: the trace and its inputs hold the second, shorter run alone."""
    fly(tmp_path, 8.0, {"left": healthy(), "right": healthy()}, 1.0, plain)
    trace = fly(tmp_path, 4.0, {"left": healthy(), "right": healthy()}, 1.0, plain)

    result = run_report.report(trace)

    assert len(trace.read_text().splitlines()) == 1 + 4 * HZ
    assert len(result.trace.t_s) == 4 * HZ
    assert len(list((control.trace_inputs_dir(str(trace)) / "top").iterdir())) == 4


def test_a_trace_without_its_inputs_is_refused(tmp_path: pathlib.Path) -> None:
    """A trace whose model inputs are gone (an older release's, or deleted) is refused, not reported in part."""
    trace = fly(tmp_path, 2.0, {"left": healthy(), "right": healthy()}, 1.0, plain)
    shutil.rmtree(control.trace_inputs_dir(str(trace)))

    with pytest.raises(errors.ConfigError, match="no model inputs beside the trace"):
        run_report.report(trace)


def test_dreamcli_robot_report_prints_a_line_per_check_the_worst_and_one_plot(tmp_path: pathlib.Path) -> None:
    """`dreamcli robot report` with no argument on the slowed-joint run the station config traces: the run, a line
    per check (the slowed joint alone on its own), the worst problem with what to do, and one PNG of the joints;
    naming the trace prints the same."""
    arms = {"left": healthy(), "right": follower(1, {3: 0.2}, (0.0, 0.0))}
    trace = fly(tmp_path, 12.0, arms, 1.0, plain)
    example = pathlib.Path(__file__).parents[1] / "configs" / "yam.example.yaml"
    (tmp_path / cli.CONFIG_NAME).write_text(example.read_text() + f"\ntrace_path: {trace}\n")
    out, named = io.StringIO(), io.StringIO()

    code = cli.run(["--home", str(tmp_path), "robot", "report"], out=out)

    printed = out.getvalue().splitlines()
    assert code == 0
    assert printed[0].startswith("last run: 12 s at ")
    # The run; the cameras, the state, the loop and the left arm; the slow joint and the rest of the right arm; the
    # worst problem and the plot.
    assert len(printed) == 1 + 4 + 2 + 2
    assert [line for line in printed if line.startswith("PROBLEM")][0].startswith(
        "PROBLEM  right arm joint 3: off its command by "
    )
    assert printed[-2] == f"WORST    right arm joint 3: {run_report.FIXES['diverging']}"
    assert printed[-1] == f"PLOT     {trace.with_suffix('.png')}"
    image = cv2.imread(str(trace.with_suffix(".png")))
    assert image.shape == (2 * 150 + 30, conftest.ARM_DOF * 320, 3)
    assert cli.run(["--home", str(tmp_path / "elsewhere"), "robot", "report", str(trace)], out=named) == 0
    assert named.getvalue() == out.getvalue()


def test_dreamcli_robot_report_says_when_the_station_turned_the_trace_off(tmp_path: pathlib.Path) -> None:
    """A station config with `trace_path: null` records nothing to report on, and `dreamcli robot report` says which
    line to delete."""
    example = pathlib.Path(__file__).parents[1] / "configs" / "yam.example.yaml"
    (tmp_path / cli.CONFIG_NAME).write_text(example.read_text() + "\ntrace_path: null\n")
    out = io.StringIO()

    assert cli.run(["--home", str(tmp_path), "robot", "report"], out=out) == 1
    assert out.getvalue().startswith(f"report: {tmp_path / cli.CONFIG_NAME} turns the trace off")
