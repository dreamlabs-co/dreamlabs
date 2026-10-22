"""Behaviour tests for `robots.station.StationController` over the conftest fakes; no hardware, no sockets.

The controller is driven on a real asyncio loop: `start()` returns at once and the run task
builds a loop over fresh fakes and ticks it on a manual clock that yields a millisecond of wall
time per tick, so a run makes chunks within milliseconds and every transition can be awaited
with `wait_for`. What the arms were told is read from the fake robots' command logs.
"""

import asyncio
import dataclasses
import json
import logging
import os
import pathlib
import signal
import socket
import subprocess
import sys
import threading
import time
import types
from collections.abc import Callable, Iterator
from typing import Any

import conftest
import dream
import numpy as np
import pytest
import test_control

from robots import arms_lock, cameras, cli, config, control, errors, followers, kinematics, run, station, yam

## Longest a test waits for the run task to reach a state.
WAIT_S = 5.0

## Seconds the test home move takes; 120 steps of 0.4 ms.
HOME_DURATION_S = 0.05


@dataclasses.dataclass
class Station:
    """A controller plus every fake behind it.

    Attributes:
    - controller: The `StationController`.
    - robots: FakeRobot per arm.
    - clients: Every FakeClient `make_client` built, in order (one per run).
    - checkpoints: The checkpoint each `make_client` call was asked for, in order.
    - action_spaces: The action space each `make_client` call was asked for, in order.
    - camera_sets: Every set of FakeCaptureSources `make_cameras` built, keyed by camera name, in order: one at the
      fixture's open_cameras(), one more per reopen of a dead set.
    - readers: Every list of CameraReaders built, in order.
    - log_path: The inference log the runs append to.
    """

    controller: station.StationController
    robots: dict[str, conftest.FakeRobot]
    clients: list[conftest.FakeClient]
    checkpoints: list[str | None]
    action_spaces: list[str | None]
    camera_sets: list[dict[str, conftest.FakeCaptureSource]]
    readers: list[list[cameras.CameraReader]]
    log_path: pathlib.Path


@pytest.fixture
def build(tmp_path: pathlib.Path) -> Iterator[Callable[..., Station]]:
    """A `Station` factory; capture threads still alive at teardown are stopped."""
    built: list[Station] = []

    def factory(
        client_factory: Callable[[], conftest.FakeClient] | None = None,
        camera_factory: Callable[[], dict[str, conftest.FakeCaptureSource]] | None = None,
        task: str = conftest.TASK,
        rollouts_dir: pathlib.Path | None = None,
        action_space: str | None = None,
        home_on_key_refused: bool = True,
    ) -> Station:
        robots = {"left": conftest.FakeRobot(), "right": conftest.FakeRobot()}
        arms = {name: conftest.yam_arm(name, robot) for name, robot in robots.items()}
        parts = Station(
            controller=None,  # type: ignore[arg-type]
            robots=robots,
            clients=[],
            checkpoints=[],
            action_spaces=[],
            camera_sets=[],
            readers=[],
            log_path=tmp_path / "inference.jsonl",
        )

        def make_cameras() -> list[cameras.CameraReader]:
            sources = (
                camera_factory()
                if camera_factory
                else {name: conftest.FakeCaptureSource() for name in conftest.CAMERA_NAMES}
            )
            readers = [
                cameras.CameraReader(name, source, conftest.LOOP_DEFAULTS["stale_after_s"])
                for name, source in sources.items()
            ]
            parts.camera_sets.append(sources)
            parts.readers.append(readers)
            return readers

        def make_client(checkpoint: str | None, action_space: str | None) -> conftest.FakeClient:
            client = (
                client_factory()
                if client_factory
                else conftest.FakeClient(horizon=conftest.LOOP_DEFAULTS["action_horizon"])
            )
            parts.checkpoints.append(checkpoint)
            parts.action_spaces.append(action_space)
            parts.clients.append(client)
            return client

        def make_arms() -> dict[str, followers.FollowerArm]:
            fresh = {"left": conftest.FakeRobot(), "right": conftest.FakeRobot()}
            parts.robots = fresh
            return {name: conftest.yam_arm(name, robot) for name, robot in fresh.items()}

        clock = test_control.YieldingClock()
        parts.controller = station.StationController(
            arms,
            task=task,
            make_cameras=make_cameras,
            make_client=make_client,
            loop_settings={**conftest.LOOP_DEFAULTS, "clock": clock, "sleep": clock.sleep},
            inference_log_path=str(parts.log_path),
            checkpoint="dreamlabs/dream-1-beta",
            action_space=action_space,
            make_arms=make_arms,
            make_recorder=None
            if rollouts_dir is None
            else lambda checkpoint, run_number: run.make_recorder(
                conftest.make_app_config(rollouts=config.RolloutsConfig(enabled=True, dir=str(rollouts_dir))),
                checkpoint,
                run_number,
            ),
            home_duration_s=HOME_DURATION_S,
            home_on_key_refused=home_on_key_refused,
        )
        parts.controller.open_cameras()
        built.append(parts)
        return parts

    yield factory
    for parts in built:
        for readers in parts.readers:
            for reader in readers:
                if reader.is_alive():
                    reader.stop()


async def wait_for(predicate: Callable[[], bool], what: str, timeout_s: float = WAIT_S) -> None:
    """Polls `predicate` on the event loop until it holds, failing the test after `timeout_s`."""
    deadline_s = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline_s:
            raise AssertionError(f"timed out after {timeout_s} s waiting for {what}")
        await asyncio.sleep(0.005)


async def start_and_wait_for_chunks(
    parts: Station, chunks: int = 2, prompt: str | None = None, checkpoint: str | None = None
) -> None:
    """Starts a run and waits until its loop has installed `chunks` chunks."""
    status = await parts.controller.start(prompt, checkpoint)
    assert status["state"] == station.STARTING
    await wait_for(lambda: parts.controller.state == station.RUNNING, "running")
    await wait_for(lambda: parts.controller.loop is not None and parts.controller.loop.chunk_count >= chunks, "chunks")


# Start / stop


def test_start_returns_at_once_then_the_run_ticks_and_stop_leaves_the_arms_holding(
    build: Callable[..., Station], caplog: pytest.LogCaptureFixture
) -> None:
    """start() accepts and returns `starting`; the run builds a client and cameras, enters `running` and makes
    chunks; stop() returns `idle` only once the loop's __aexit__ has closed the client, stopped the cameras and
    re-held both arms; every transition is one INFO line."""
    parts = build()

    async def scenario() -> dict[str, Any]:
        await start_and_wait_for_chunks(parts)
        running = parts.controller.status()
        stopped = await parts.controller.stop()
        return {"running": running, "stopped": stopped}

    with caplog.at_level(logging.INFO, logger="robots.station"):
        result = asyncio.run(scenario())

    running = result["running"]
    assert running["state"] == station.RUNNING and running["run"] == 1 and running["prompt"] == conftest.TASK
    assert running["chunks"] >= 2 and running["timing"]["count"] >= 2 and running["timing"]["last_rtt_ms"] is not None
    assert running["started_at"] is not None and running["run_elapsed_s"] is not None and running["uptime_s"] >= 0
    assert running["last_error"] is None and running["stalls"] == 0
    # No kinematics in the test settings: nothing is clamped and the largest request stays at zero.
    assert running["clamps"] == 0 and running["max_ee_speed_requested_m_s"] == 0.0
    assert running["max_ee_speed_m_s"] == conftest.LOOP_DEFAULTS["max_ee_speed_m_s"]
    stopped = result["stopped"]
    assert stopped["state"] == station.IDLE and stopped["run"] == 1
    assert len(parts.clients) == 1 and parts.clients[0].closed
    # The cameras are the station's: still publishing after the run, released only by close().
    assert not any(source.released for source in parts.camera_sets[0].values())
    for robot in parts.robots.values():
        # __aexit__'s hold re-sends the last command: the arms stay where the run left them.
        assert len(robot.commands) >= 3
        np.testing.assert_allclose(robot.commands[-1], robot.commands[-2])
    transitions = [
        r.getMessage() for r in caplog.records if "station: " in r.getMessage() and " -> " in r.getMessage()
    ]
    assert [t.split(" (")[0] for t in transitions] == [
        "station: idle -> starting",
        "station: starting -> running",
        "station: running -> stopping",
        "station: stopping -> idle",
    ]


def test_every_run_is_saved_as_its_own_episode_named_for_the_run_and_the_checkpoint(
    build: Callable[..., Station], tmp_path: pathlib.Path
) -> None:
    """With a recorder factory, a run's ticks land in an episode folder carrying the station's checkpoint and run
    number; a second run is a second folder."""
    parts = build(rollouts_dir=tmp_path / "rollouts")

    async def scenario() -> None:
        for _ in range(2):
            await start_and_wait_for_chunks(parts)
            await parts.controller.stop()

    asyncio.run(scenario())
    deadline_s = time.monotonic() + WAIT_S
    infos: list[dict[str, Any]] = []
    while time.monotonic() < deadline_s and len(infos) < 2:
        infos = [
            json.loads(path.read_text())
            for path in sorted((tmp_path / "rollouts").glob("*/rollout.json"))
            if json.loads(path.read_text())["status"] == "complete"
        ]
        time.sleep(0.05)
    assert [(info["run"], info["checkpoint"], info["task"]) for info in infos] == [
        (1, "dreamlabs/dream-1-beta", conftest.TASK),
        (2, "dreamlabs/dream-1-beta", conftest.TASK),
    ]
    assert all(info["frames"] > 0 for info in infos)


def test_a_second_start_is_refused_while_a_run_exists_and_the_run_is_untouched(build: Callable[..., Station]) -> None:
    """Exactly one run at a time: the refusal is a StationError naming the state, the run number does not move."""
    parts = build()

    async def scenario() -> None:
        await start_and_wait_for_chunks(parts, chunks=1)
        with pytest.raises(errors.StationError, match="start refused: station is running"):
            await parts.controller.start()
        assert parts.controller.run_number == 1 and len(parts.clients) == 1
        await parts.controller.stop()

    asyncio.run(scenario())


def test_stop_with_nothing_running_is_a_no_op(build: Callable[..., Station]) -> None:
    """Stop is idempotent: the status comes back idle and nothing is written to the arms."""
    parts = build()

    status = asyncio.run(parts.controller.stop())

    assert status["state"] == station.IDLE
    assert all(robot.commands == [] for robot in parts.robots.values())


def test_the_cameras_stay_open_across_runs_and_a_fresh_client_is_built_per_run(build: Callable[..., Station]) -> None:
    """The station opens its cameras once and keeps them publishing between runs (`frames` answers while idle), so
    the second run reuses the same readers, none of which is released before close(); the client is fresh per
    run, and the run number and the inference log's `run` field follow."""
    parts = build()
    assert parts.controller.state == station.IDLE
    # Each reader publishes its first frame once its window thread has encoded the first sample; on a slow runner
    # that is more than a fixed pause, so wait for all three rather than assume them.
    deadline_s = time.monotonic() + WAIT_S
    while sorted(parts.controller.frames) != sorted(conftest.CAMERA_NAMES):
        assert time.monotonic() < deadline_s, f"cameras never all published: {sorted(parts.controller.frames)}"
        time.sleep(0.005)
    assert all(age is not None for age in parts.controller.status()["cameras"].values())

    async def scenario() -> None:
        await start_and_wait_for_chunks(parts, chunks=1)
        await parts.controller.stop()
        assert sorted(parts.controller.frames) == sorted(conftest.CAMERA_NAMES)
        await start_and_wait_for_chunks(parts, chunks=1)
        await parts.controller.stop()

    asyncio.run(scenario())

    assert parts.controller.run_number == 2
    assert len(parts.clients) == 2 and len(parts.camera_sets) == 1
    assert parts.clients[0] is not parts.clients[1]
    assert not any(source.released for source in parts.camera_sets[0].values())
    assert all(reader.is_alive() for reader in parts.controller.cameras)
    asyncio.run(parts.controller.close())
    assert all(source.released for source in parts.camera_sets[0].values()) and parts.controller.cameras == []
    runs = sorted({__import__("json").loads(line)["run"] for line in parts.log_path.read_text().splitlines()})
    assert runs == [1, 2]


def test_a_stop_during_the_build_discards_what_the_build_opened(build: Callable[..., Station]) -> None:
    """Stop while the client is still being built on the worker thread: the controller is idle at once, and when
    the build finishes its client is closed rather than leaked; the station's cameras stay up throughout."""
    release = threading.Event()
    entered = threading.Event()

    def slow_client() -> conftest.FakeClient:
        entered.set()
        release.wait(WAIT_S)
        return conftest.FakeClient(horizon=conftest.LOOP_DEFAULTS["action_horizon"])

    parts = build(client_factory=slow_client)

    async def scenario() -> None:
        await parts.controller.start()
        await wait_for(entered.is_set, "the build to start")
        status = await parts.controller.stop()
        assert status["state"] == station.IDLE
        assert parts.clients == []
        release.set()
        await wait_for(lambda: len(parts.clients) == 1 and parts.clients[0].closed, "the abandoned build discarded")
        assert all(reader.is_alive() for reader in parts.controller.cameras)

    asyncio.run(scenario())

    assert all(reader.is_alive() for reader in parts.readers[0])
    assert all(robot.commands == [] for robot in parts.robots.values())


def test_a_close_during_the_build_waits_it_out_and_leaves_no_thread_behind(build: Callable[..., Station]) -> None:
    """Close (process exit) while the client is still building on the worker thread: the close waits the build out,
    so the build neither reopens the cameras the close stops nor leaves its client open, and no thread the station
    started outlives the close. Before, the close returned at once and the build then reopened the cameras: with the
    real client on one core, 8 threads ran on at 1.0 CPU s/s and the process could abort at exit (134)."""
    release = threading.Event()
    entered = threading.Event()

    def slow_client() -> conftest.FakeClient:
        entered.set()
        release.wait(WAIT_S)
        return conftest.FakeClient(horizon=conftest.LOOP_DEFAULTS["action_horizon"])

    before = set(threading.enumerate())
    parts = build(client_factory=slow_client)

    async def scenario() -> None:
        await parts.controller.start()
        await wait_for(entered.is_set, "the build to start")
        closing = asyncio.create_task(parts.controller.close())
        await wait_for(lambda: parts.controller.state == station.IDLE, "the run stopped")
        await asyncio.sleep(0.05)
        assert not closing.done(), "the close returned with the build still going"
        release.set()
        assert await closing

    asyncio.run(scenario())

    assert len(parts.readers) == 1 and not any(reader.is_alive() for reader in parts.readers[0])
    assert len(parts.clients) == 1 and parts.clients[0].closed
    assert [thread.name for thread in threading.enumerate() if thread not in before] == []


@pytest.mark.skipif(not hasattr(os, "sched_setaffinity"), reason="pins a process to one core (Linux only)")
def test_a_close_during_the_build_on_one_core_exits_cleanly() -> None:
    """The close-during-build test in a fresh process pinned to one core, as on a slow CI runner: it passes and the
    process exits 0, never with "terminate called without an active exception" (exit 134)."""
    core = min(os.sched_getaffinity(0))
    test = f"{__file__}::test_a_close_during_the_build_waits_it_out_and_leaves_no_thread_behind"
    child = (
        f"import os, sys, pytest; os.sched_setaffinity(0, {{{core}}}); "
        f"sys.exit(pytest.main(['-q', '-p', 'no:cacheprovider', {test!r}]))"
    )
    done = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0 and "terminate called" not in done.stderr, done.stdout[-2000:] + done.stderr[-2000:]


# Failures


def test_an_unrecoverable_policy_error_ends_the_run_in_error_with_the_envelope(build: Callable[..., Station]) -> None:
    """The loop stops on DreamAuthError; the controller records type, message, code and reason, holds the arms,
    and accepts the next start from `error`."""
    parts = build(
        client_factory=lambda: conftest.FakeClient(horizon=4, raise_once=test_control.UNRECOVERABLE_ERRORS[0])
    )

    async def scenario() -> None:
        await parts.controller.start()
        await wait_for(lambda: parts.controller.state == station.ERROR, "error")
        status = parts.controller.status()
        assert status["last_error"]["type"] == "DreamAuthError"
        assert status["last_error"]["code"] == "DA-1000" and status["last_error"]["reason"] is None
        assert "UNAUTHORIZED" in status["last_error"]["message"]
        assert parts.clients[0].closed
        await parts.controller.start()
        assert parts.controller.run_number == 2 and parts.controller.last_error is None
        await parts.controller.stop()

    asyncio.run(scenario())


def test_a_run_refused_over_the_budget_cap_homes_the_arms_and_keeps_the_refusal(build: Callable[..., Station]) -> None:
    """DreamAPI refusing `over_cap` ends the run like any refusal, then the station homes the arms itself, since no
    run can start until the owner raises the cap and nobody may be watching; `last_error` still names the cap."""
    over_cap = dream.DreamAuthError(
        "DreamAPI refused the inference request: DA-1000 UNAUTHORIZED (over_cap)", code="DA-1000", reason="over_cap"
    )
    parts = build(client_factory=lambda: conftest.FakeClient(horizon=4, raise_once=over_cap))

    async def scenario() -> None:
        written_before = {name: len(robot.commands) for name, robot in parts.robots.items()}
        await parts.controller.start()
        controller = parts.controller
        await wait_for(lambda: controller.state == station.IDLE and controller.last_error is not None, "homed")
        assert all(len(robot.commands) > written_before[name] for name, robot in parts.robots.items())
        assert parts.controller.last_error is not None
        assert parts.controller.last_error["error_code"] == "over_cap"

    asyncio.run(scenario())


# A refused key

## The home pose move_home ends on: every joint at 0 rad, the gripper fully open.
HOME_POSE = [0.0] * 6 + [1.0]


def key_refused(reason: str = station.KEY_REFUSED_REASON) -> dream.DreamAuthError:
    """The refusal the gateway answers for a key the platform's export marks `disabled` (out of credit, over cap)."""
    return dream.DreamAuthError(
        f"DreamAPI refused the inference request: DA-1000 UNAUTHORIZED ({reason})", code="DA-1000", reason=reason
    )


async def refuse_mid_run(parts: Station, error: BaseException) -> None:
    """Starts a run, lets it drive the arms, then has the next get_action raise `error`; waits for #ERROR."""
    await start_and_wait_for_chunks(parts, chunks=2)
    parts.clients[0].raise_once = error
    await wait_for(lambda: parts.controller.state == station.ERROR, "error after the refusal")


def assert_holding(parts: Station) -> None:
    """Each arm still holds the run's last command: nothing wrote it after the loop let go."""
    for name, robot in parts.robots.items():
        assert len(robot.commands) >= 2
        np.testing.assert_allclose(robot.commands[-1], robot.commands[-2])
        assert not np.allclose(robot.commands[-1], HOME_POSE), f"{name} went home"


def test_a_run_whose_key_is_refused_brings_the_arms_home_and_ends_in_key_refused(
    build: Callable[..., Station], caplog: pytest.LogCaptureFixture
) -> None:
    """`home_on_key_refused` on: the gateway refusing the key mid-run (`DA-1000 disabled`: out of credit or
    over the spend cap) ends the loop, which holds the arms and closes the client; then the station moves both arms
    home through `homing` and ends in `error` with `error_code` `key_refused`, the envelope kept. The next start is
    accepted from there."""
    parts = build(home_on_key_refused=True)

    async def scenario() -> None:
        await refuse_mid_run(parts, key_refused())
        status = parts.controller.status()
        assert status["last_error"]["error_code"] == errors.KEY_REFUSED == "key_refused"
        assert status["last_error"]["type"] == "DreamAuthError"
        assert status["last_error"]["code"] == "DA-1000" and status["last_error"]["reason"] == "disabled"
        assert "out of credit" in status["last_error"]["message"]
        assert status["last_error"]["message"].endswith("the arms went home")
        assert parts.clients[0].closed
        await parts.controller.start()
        assert parts.controller.last_error is None
        await parts.controller.stop()

    with caplog.at_level(logging.INFO, logger="robots.station"):
        asyncio.run(scenario())

    for robot in parts.robots.values():
        np.testing.assert_allclose(robot.commands[-1], HOME_POSE)
    transitions = [r.getMessage().split(" (")[0] for r in caplog.records if " -> " in r.getMessage()]
    assert transitions[:4] == [
        "station: idle -> starting",
        "station: starting -> running",
        "station: running -> homing",
        "station: homing -> error",
    ]


def test_with_the_setting_off_a_refused_key_holds_the_arms_as_before(build: Callable[..., Station]) -> None:
    """`home_on_key_refused: false` in the station file: the refusal is a fault like any other, the arms hold the
    run's last command and `error_code` is DreamAPI's reason."""
    parts = build(home_on_key_refused=False)

    async def scenario() -> None:
        await refuse_mid_run(parts, key_refused())
        await asyncio.sleep(HOME_DURATION_S * 4)

    asyncio.run(scenario())
    assert parts.controller.state == station.ERROR
    assert parts.controller.last_error is not None
    assert parts.controller.last_error["error_code"] == "disabled"
    assert_holding(parts)


def test_a_key_refused_home_move_refuses_start_stop_and_home_until_it_ends_and_close_waits_for_it(
    build: Callable[..., Station],
) -> None:
    """The move after the refusal is a home move like any other, so the arms are written from one place: a start, a
    stop or a home while it runs is refused, and close() waits it out before switching the arms off."""
    parts = build(home_on_key_refused=True)
    parts.controller._home_duration_s = 0.5

    async def scenario() -> None:
        await start_and_wait_for_chunks(parts, chunks=2)
        parts.clients[0].raise_once = key_refused()
        await wait_for(lambda: parts.controller.state == station.HOMING, "homing after the refusal")
        with pytest.raises(errors.StationError, match="start refused: station is homing"):
            await parts.controller.start()
        with pytest.raises(errors.StationError, match="home refused: station is homing"):
            await parts.controller.home()
        with pytest.raises(errors.StationError, match="stop refused: a home move is in progress"):
            await parts.controller.stop()
        await parts.controller.close()
        assert parts.controller.state == station.ERROR
        assert parts.controller.last_error is not None
        assert parts.controller.last_error["error_code"] == errors.KEY_REFUSED

    asyncio.run(scenario())
    for robot in parts.robots.values():
        np.testing.assert_allclose(robot.commands[-1], HOME_POSE)


@pytest.mark.parametrize(
    "error",
    [
        key_refused("unknown_key"),
        key_refused("not_entitled"),
        dream.DreamConfigError("config key 'checkpoint' must be a non-empty str"),
    ],
    ids=["unknown_key", "not_entitled", "DreamConfigError"],
)
def test_every_other_fault_still_holds_the_arms_with_the_flag_on(
    build: Callable[..., Station], error: dream.DreamError
) -> None:
    """Only `disabled` is the refusal of a key out of credit: another `DA-1000` reason and every other fault that
    ends a run leave the arms holding the run's last command, as before."""
    parts = build(home_on_key_refused=True)

    async def scenario() -> None:
        await refuse_mid_run(parts, error)
        await asyncio.sleep(HOME_DURATION_S * 4)

    asyncio.run(scenario())
    assert parts.controller.last_error is not None
    assert parts.controller.last_error["error_code"] == errors.error_code(error) != errors.KEY_REFUSED
    assert_holding(parts)


def test_a_refused_key_with_a_dead_arm_never_homes_and_says_the_arms_hold(
    build: Callable[..., Station], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Home is refused while an arm's driver is dead, so the move after a refusal is too: neither arm moves, the run
    ends with `error_code` `disabled`, and the message says the arms hold and why, until a reset."""
    parts = build(home_on_key_refused=True)

    async def scenario() -> None:
        await start_and_wait_for_chunks(parts, chunks=2)
        # A driver dying under a ticking loop ends the run with ArmError first; the controller's own view of the arms
        # is what the guard reads, so it is the one that says an arm died between the loop's exit and the move.
        monkeypatch.setattr(parts.controller, "_dead_arms", lambda: ["left"])
        parts.clients[0].raise_once = key_refused()
        await wait_for(lambda: parts.controller.state == station.ERROR, "error after the refusal")
        await asyncio.sleep(HOME_DURATION_S * 4)

    asyncio.run(scenario())
    assert parts.controller.last_error is not None
    assert parts.controller.last_error["error_code"] == "disabled"
    assert "the arms were not homed" in parts.controller.last_error["message"]
    assert "'left'" in parts.controller.last_error["message"]
    assert_holding(parts)


def test_a_key_refused_home_move_that_fails_on_an_arm_ends_in_home_error_naming_the_refusal(
    build: Callable[..., Station], monkeypatch: pytest.MonkeyPatch
) -> None:
    """When an arm cannot be read for the move after the refusal, the other arm still goes home and the station ends
    in `error` with `HomeError` (the arms need a reset), the message naming the refusal too."""
    parts = build(home_on_key_refused=True)
    real_home = station.home.move_home

    def failing_home(arms: dict[str, followers.Arm], **kwargs: Any) -> dict[str, str]:
        # The loop's ticks read the arms too, so the fault is queued only once the loop is gone and the move is next.
        parts.robots["left"].read_error = RuntimeError("no observation")
        return real_home(arms, **kwargs)

    monkeypatch.setattr(station.home, "move_home", failing_home)

    async def scenario() -> None:
        await refuse_mid_run(parts, key_refused())

    asyncio.run(scenario())
    assert parts.controller.last_error is not None
    assert parts.controller.last_error["error_code"] == "HomeError"
    assert parts.controller.last_error["message"].startswith("the API key was refused")
    np.testing.assert_allclose(parts.robots["right"].commands[-1], HOME_POSE)
    assert not np.allclose(parts.robots["left"].commands[-1], HOME_POSE)


def test_only_a_disabled_auth_error_is_a_refused_key() -> None:
    """refused_key() is the DreamAuthError of a disabled key and nothing else."""
    refusal = key_refused()
    assert station.refused_key(refusal) is refusal
    assert station.refused_key(key_refused("unknown_key")) is None
    assert station.refused_key(dream.DreamServerError("x", code="DA-2000", reason="disabled")) is None
    assert station.refused_key(RuntimeError("disabled")) is None


def test_a_dead_camera_is_reopened_at_the_next_start_and_a_reopen_that_fails_ends_the_run_in_error(
    build: Callable[..., Station],
) -> None:
    """A reader whose thread died while idle is rebuilt by the next start (a second camera set), so the run goes on;
    when the rebuild itself raises CameraError the start lands in `last_error` with the client closed and the arms
    untouched."""
    calls = 0

    def factory() -> dict[str, conftest.FakeCaptureSource]:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise errors.CameraError("camera 'top' could not open device 0")
        return {name: conftest.FakeCaptureSource() for name in conftest.CAMERA_NAMES}

    parts = build(camera_factory=factory)

    async def scenario() -> None:
        parts.camera_sets[0]["top"].fail()
        await wait_for(lambda: not parts.controller.cameras[0].is_alive(), "the dead reader")
        await start_and_wait_for_chunks(parts, chunks=1)
        assert len(parts.camera_sets) == 2 and parts.camera_sets[0]["top"].released
        await parts.controller.stop()
        parts.camera_sets[1]["left"].fail()
        await wait_for(lambda: not all(reader.is_alive() for reader in parts.controller.cameras), "the dead reader")
        await parts.controller.start()
        await wait_for(lambda: parts.controller.state == station.ERROR, "error")

    asyncio.run(scenario())

    assert parts.controller.last_error["type"] == "CameraError"  # type: ignore[index]
    assert parts.clients[-1].closed
    assert all(robot.commands == [] for robot in parts.robots.values()) or parts.controller.run_number == 2


def test_a_camera_that_dies_while_idle_is_reopened_by_the_watchdog_and_left_alone_during_a_run(
    build: Callable[..., Station],
) -> None:
    """`reopen_dead_cameras()` (what serve()'s watchdog calls every CAMERA_WATCH_S) rebuilds a set with a dead reader
    while nothing runs, so the live view shows the scene again without a start; with every reader alive, or with a
    run on, it does nothing."""
    parts = build()

    async def scenario() -> None:
        assert await parts.controller.reopen_dead_cameras() is False, "nothing dead"
        parts.camera_sets[0]["top"].fail()
        await wait_for(lambda: not parts.readers[0][0].is_alive(), "the reader to die")
        assert await parts.controller.reopen_dead_cameras() is True
        assert len(parts.camera_sets) == 2 and all(reader.is_alive() for reader in parts.readers[1])
        assert parts.controller.status()["cameras"].keys() == {"top", "left", "right"}
        await start_and_wait_for_chunks(parts)
        parts.camera_sets[1]["left"].fail()
        await wait_for(lambda: parts.controller.state == station.ERROR, "the run to end on the dead camera")
        assert await parts.controller.reopen_dead_cameras() is True, "idle again after the run ended"
        assert len(parts.camera_sets) == 3

    asyncio.run(scenario())


def test_an_arm_alarm_ends_the_run_in_error_and_parks_that_arm_alone(build: Callable[..., Station]) -> None:
    parts = build()
    alarm = "elbow_flex coil reached 66 C (the run stops at 65 C); the arm parks. Let it cool below 55 C"

    async def scenario() -> None:
        await start_and_wait_for_chunks(parts)
        parts.controller._arms["left"]._driver.alarm = lambda: alarm
        await wait_for(lambda: parts.controller.state == station.ERROR, "the run to end on the alarm")

    asyncio.run(scenario())

    status = parts.controller.status()
    assert status["last_error"]["type"] == "SafetyStop"
    assert status["last_error"]["message"] == f"arm 'left': {alarm}"
    assert status["arms"] == {"left": False, "right": True}
    assert parts.robots["left"].closed and not parts.robots["right"].closed


def test_an_arm_alarm_while_idle_parks_that_arm_and_reports_it(build: Callable[..., Station]) -> None:
    parts = build()
    alarm = "wrist_flex has been blocked for 5 s (asked to go 30 deg further)"

    async def scenario() -> None:
        assert await parts.controller.park_alarmed_arms() is False
        parts.controller._arms["right"]._driver.alarm = lambda: alarm
        assert await parts.controller.park_alarmed_arms() is True
        assert await parts.controller.park_alarmed_arms() is False

    asyncio.run(scenario())

    status = parts.controller.status()
    assert status["state"] == station.ERROR and status["last_error"]["message"] == f"arm 'right': {alarm}"
    assert status["arms"] == {"left": True, "right": False}


def test_an_arm_that_stops_answering_while_idle_is_reported_by_name_with_what_to_do(
    build: Callable[..., Station],
) -> None:
    parts = build()
    driver = parts.controller._arms["left"]._driver
    assert parts.controller.report_dead_arms() is False
    driver.is_alive = lambda: False
    driver.why_stopped = lambda: "motor 3 stopped answering"
    assert parts.controller.report_dead_arms() is True
    assert parts.controller.report_dead_arms() is False

    status = parts.controller.status()
    assert status["state"] == station.ERROR and status["last_error"]["type"] == "ArmError"
    assert status["last_error"]["message"] == (
        "arm 'left' stopped answering while the station was idle (motor 3 stopped answering). It may still hold under "
        "power with nothing driving it: support it, check its power and the cable to its adapter, then press Reset"
    )
    assert status["arms"] == {"left": False, "right": True}


# Home


def test_home_is_refused_while_a_run_exists_and_moves_the_arms_once_it_is_stopped(
    build: Callable[..., Station],
) -> None:
    """The arms are never written from two places: home while running is a StationError and writes nothing; after
    stop it drives both arms to zeros with the grippers open and the controller is idle again."""
    parts = build()

    async def scenario() -> None:
        await start_and_wait_for_chunks(parts, chunks=1)
        with pytest.raises(errors.StationError, match="home refused: station is running"):
            await parts.controller.home()
        await parts.controller.stop()
        written_before = {name: len(robot.commands) for name, robot in parts.robots.items()}
        status = await parts.controller.home()
        assert status["state"] == station.IDLE
        for name, robot in parts.robots.items():
            assert len(robot.commands) == written_before[name] + 121
            np.testing.assert_allclose(robot.commands[-1], [0.0] * 6 + [1.0])

    asyncio.run(scenario())


def test_home_moves_through_homing_and_a_failed_arm_leaves_the_controller_in_error(
    build: Callable[..., Station], caplog: pytest.LogCaptureFixture
) -> None:
    """A read fault on one arm: state `error`, `last_error` HomeError naming that arm, why and what to do, never
    just "an arm"; the other arm still reached home. Start is refused while the move
    runs."""
    parts = build()
    parts.robots["left"].read_error = RuntimeError("no observation")

    async def scenario() -> None:
        home_task = asyncio.create_task(parts.controller.home())
        await wait_for(lambda: parts.controller.state == station.HOMING, "homing")
        with pytest.raises(errors.StationError, match="start refused: station is homing"):
            await parts.controller.start()
        with pytest.raises(errors.StationError, match="stop refused: a home move is in progress"):
            await parts.controller.stop()
        status = await home_task
        assert status["state"] == station.ERROR
        assert status["last_error"]["type"] == "HomeError"
        message = status["last_error"]["message"]
        assert message.startswith("arm 'left' did not reach home: its read failed: ") and "no observation" in message
        assert message.endswith("; press Reset to open it again"), message

    with caplog.at_level(logging.INFO, logger="robots.station"):
        asyncio.run(scenario())

    np.testing.assert_allclose(parts.robots["right"].commands[-1], [0.0] * 6 + [1.0])
    assert parts.robots["left"].commands == []
    assert any("idle -> homing" in r.getMessage() for r in caplog.records)
    assert any("homing -> error" in r.getMessage() for r in caplog.records)


# Reset


def test_a_dead_arm_refuses_a_start_and_reset_reopens_both_arms_then_homes_P6(
    build: Callable[..., Station], caplog: pytest.LogCaptureFixture
) -> None:
    """An arm whose motor chain stopped reads False in `arms`, and a start is refused naming it; `reset()` closes
    every driver, builds fresh ones with the factory (`resetting`), homes them (`homing`) and ends `idle` with
    both arms alive; the first run after it drives the new arms. A reset is refused while a run exists."""
    parts = build()
    old_left = parts.robots["left"]
    old_left.motor_chain = types.SimpleNamespace(running=False)
    old_left.closed = False
    old_left.close = lambda: setattr(old_left, "closed", True)  # type: ignore[attr-defined]

    async def scenario() -> None:
        assert parts.controller.status()["arms"] == {"left": False, "right": True}
        with pytest.raises(
            errors.StationError, match="start refused: arm 'left' driver is dead; reset the arms first"
        ):
            await parts.controller.start()
        with pytest.raises(errors.StationError, match="home refused: arm 'left' driver is dead; reset the arms first"):
            await parts.controller.home()
        accepted = await parts.controller.reset()
        assert accepted["state"] == station.RESETTING, "a reset is accepted at once, like a start"
        with pytest.raises(errors.StationError, match="start refused: station is (resetting|homing)"):
            await parts.controller.start()
        with pytest.raises(errors.StationError, match="home refused: station is (resetting|homing)"):
            await parts.controller.home()
        with pytest.raises(errors.StationError, match="stop refused: a (reset|home move) is in progress"):
            await parts.controller.stop()
        with pytest.raises(errors.StationError, match="reset refused: station is (resetting|homing)"):
            await parts.controller.reset()
        await wait_for(lambda: parts.controller.state == station.IDLE, "idle after the reset")
        status = parts.controller.status()
        assert status["arms"] == {"left": True, "right": True}
        assert status["last_error"] is None
        assert old_left.closed, "the dead driver was closed before a fresh one was opened"
        assert parts.robots["left"] is not old_left
        np.testing.assert_allclose(parts.robots["left"].commands[-1], [0.0] * 6 + [1.0])
        await start_and_wait_for_chunks(parts)
        assert parts.robots["left"].commands, "the run drives the reopened arm"
        with pytest.raises(errors.StationError, match="reset refused: station is running; stop the run first"):
            await parts.controller.reset()
        await parts.controller.stop()

    with caplog.at_level(logging.INFO, logger="robots.station"):
        asyncio.run(scenario())
    assert any("idle -> resetting" in r.getMessage() for r in caplog.records)
    assert any("resetting -> homing" in r.getMessage() for r in caplog.records)
    assert any("homing -> idle" in r.getMessage() for r in caplog.records)


def test_a_reset_whose_reopen_fails_ends_in_error_with_both_arms_dead_until_a_reset_succeeds(
    build: Callable[..., Station],
) -> None:
    """The factory raising (i2rt cannot open the channel) is `error` with the exception recorded under its class as
    `error_code`; the old drivers were closed before the reopen, so both arms read dead and a start or a home is
    refused rather than written to a closed driver; the next reset, with the factory working, brings both back;
    a controller built without `make_arms` refuses the reset outright."""
    parts = build()
    working = parts.controller._make_arms
    parts.controller._make_arms = lambda: (_ for _ in ()).throw(RuntimeError("can_follower_l: no such device"))

    async def scenario() -> None:
        assert (await parts.controller.reset())["state"] == station.RESETTING
        await wait_for(lambda: parts.controller.state == station.ERROR, "error after the failed reopen")
        status = parts.controller.status()
        assert status["last_error"]["message"] == "can_follower_l: no such device"
        assert status["last_error"]["error_code"] == "RuntimeError"
        assert status["arms"] == {"left": False, "right": False}
        assert all(robot.closed for robot in parts.robots.values())
        with pytest.raises(errors.StationError, match="start refused: arm 'left' driver is dead"):
            await parts.controller.start()
        with pytest.raises(errors.StationError, match="home refused: arm 'left' driver is dead"):
            await parts.controller.home()
        parts.controller._make_arms = working
        await parts.controller.reset()
        await wait_for(lambda: parts.controller.state == station.IDLE, "idle after the second reset")
        assert parts.controller.status()["arms"] == {"left": True, "right": True}
        parts.controller._make_arms = None
        with pytest.raises(errors.StationError, match="reset refused: this station cannot reopen its arms"):
            await parts.controller.reset()

    asyncio.run(scenario())


def test_a_home_whose_caller_is_cancelled_still_ends_idle_and_close_waits_for_a_reset_in_flight(
    build: Callable[..., Station],
) -> None:
    """The live agent's link dropping cancels its wait on `home()` (and the station API cancels a call that timed
    out): the move must still finish and set the state, never strand it in `homing`; likewise close() waits out a
    reset in flight and then closes every driver, so no driver thread keeps the process alive after exit."""
    parts = build()

    async def scenario() -> None:
        waiter = asyncio.create_task(parts.controller.home())
        await wait_for(lambda: parts.controller.state == station.HOMING, "homing")
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert parts.controller.state == station.HOMING, "the move goes on after the caller left"
        await wait_for(lambda: parts.controller.state == station.IDLE, "idle after the abandoned home")
        for robot in parts.robots.values():
            np.testing.assert_allclose(robot.commands[-1], [0.0] * 6 + [1.0])

        assert (await parts.controller.reset())["state"] == station.RESETTING
        await parts.controller.close()
        assert parts.controller.state == station.IDLE
        assert all(robot.closed for robot in parts.robots.values()), "close() released the reopened drivers"
        assert parts.controller.status()["arms"] == {"left": False, "right": False}

    asyncio.run(scenario())


def test_a_start_names_the_checkpoint_the_client_is_built_with_and_status_reports_it_P6(
    build: Callable[..., Station],
) -> None:
    """The config's checkpoint is what `make_client` gets until a start names another, which sticks for the runs
    after it; a blank checkpoint is rejected like a blank prompt."""
    parts = build()

    async def scenario() -> None:
        assert parts.controller.status()["checkpoint"] == "dreamlabs/dream-1-beta"
        await start_and_wait_for_chunks(parts)
        await parts.controller.stop()
        await start_and_wait_for_chunks(parts, checkpoint=" dreamlabs/dream-1-ft:abc ")
        assert parts.controller.status()["checkpoint"] == "dreamlabs/dream-1-ft:abc"
        await parts.controller.stop()
        with pytest.raises(ValueError, match="checkpoint must be a non-empty string"):
            await parts.controller.start(checkpoint="  ")
        await start_and_wait_for_chunks(parts)
        await parts.controller.stop()
        assert parts.checkpoints == ["dreamlabs/dream-1-beta", "dreamlabs/dream-1-ft:abc", "dreamlabs/dream-1-ft:abc"]

    asyncio.run(scenario())


def test_a_start_names_the_action_space_the_client_is_built_with_and_a_yam_takes_none_P9(
    build: Callable[..., Station],
) -> None:
    """On a body on the client's IK the config's action space is what `make_client` gets until a start names another,
    which sticks like a checkpoint; an unknown one is a ValueError. A YAM reports none and refuses a start naming one
    before anything changes."""
    parts = build(action_space="eef")
    yam = build()

    async def scenario() -> None:
        assert parts.controller.status()["action_space"] == "eef"
        await start_and_wait_for_chunks(parts)
        await parts.controller.stop()
        status = await parts.controller.start(None, None, "joint")
        assert status["action_space"] == "joint"
        await wait_for(lambda: parts.controller.state == station.RUNNING, "running")
        await parts.controller.stop()
        with pytest.raises(ValueError, match=r"action_space must be one of \['eef', 'joint'\], got 'joints'"):
            await parts.controller.start(None, None, "joints")
        await start_and_wait_for_chunks(parts)
        await parts.controller.stop()
        assert parts.action_spaces == ["eef", "joint", "joint"]

        assert yam.controller.status()["action_space"] is None
        with pytest.raises(errors.StationError, match="takes no action space"):
            await yam.controller.start("fold", "dreamlabs/dream-1-ft:abc", "joint")
        assert (yam.controller.state, yam.controller.prompt, yam.controller.checkpoint) == (
            station.IDLE,
            conftest.TASK,
            "dreamlabs/dream-1-beta",
        )

    asyncio.run(scenario())


def test_a_start_with_a_step_count_ends_after_that_many_steps_and_the_arms_go_home_P6(
    build: Callable[..., Station],
) -> None:
    """A run asked for N steps writes N commands, ends on its own and moves the arms home; the count sticks for the
    next start like a checkpoint, and a count below one is a ValueError before anything changes."""
    parts = build()

    async def scenario() -> None:
        assert parts.controller.status()["steps"] == 0
        with pytest.raises(ValueError, match="steps must be a whole number of at least 1, got 0"):
            await parts.controller.start("fold", None, None, 0)
        states: list[str] = []

        def homing() -> bool:
            states.append(parts.controller.state)
            return parts.controller.state == station.HOMING

        status = await parts.controller.start("fold", None, None, 30)
        assert status["steps"] == 30
        await wait_for(homing, "the home move after the last step")
        await wait_for(lambda: parts.controller.state == station.IDLE, "idle after the home move")
        assert parts.controller.loop is not None and parts.controller.loop.steps_done == 30
        assert station.RUNNING in states and parts.controller.last_error is None
        for robot in parts.robots.values():
            np.testing.assert_allclose(robot.commands[-1], [0.0] * 6 + [1.0])
        await parts.controller.start()
        await wait_for(lambda: parts.controller.state == station.HOMING, "the second run's home")
        assert parts.controller.status()["steps"] == 30

    asyncio.run(scenario())


# Prompt


def test_set_prompt_while_running_resets_the_episode_and_the_next_request_carries_it(
    build: Callable[..., Station],
) -> None:
    """The running loop is told the new task: exactly one reset_episode, the requests after it carry the new
    prompt, and status() shows it."""
    parts = build()

    async def scenario() -> None:
        await start_and_wait_for_chunks(parts, chunks=2)
        status = await parts.controller.set_prompt("stack the blocks")
        assert status["prompt"] == "stack the blocks"
        client = parts.clients[0]
        await wait_for(lambda: client.resets == 1, "the episode reset")
        await wait_for(
            lambda: any(obs["prompt"] == "stack the blocks" for obs in client.observations),
            "a request with the new prompt",
        )
        await asyncio.sleep(0.05)
        assert client.resets == 1
        reset_at = client.reset_after_calls[0]
        prompts = [obs["prompt"] for obs in client.observations]
        assert set(prompts[:reset_at]) == {conftest.TASK}
        assert set(prompts[reset_at:]) == {"stack the blocks"}
        await parts.controller.stop()

    asyncio.run(scenario())


def test_set_prompt_while_idle_is_stored_for_the_next_run(build: Callable[..., Station]) -> None:
    """Idle: the prompt is kept and the next run's first request carries it; no client exists to reset."""
    parts = build()

    async def scenario() -> None:
        status = await parts.controller.set_prompt("  fold the towel  ")
        assert status["prompt"] == "fold the towel"
        await start_and_wait_for_chunks(parts, chunks=1)
        await parts.controller.stop()

    asyncio.run(scenario())

    assert parts.clients[0].observations[0]["prompt"] == "fold the towel"
    assert parts.clients[0].resets == 0


def test_start_with_a_prompt_uses_it_and_a_blank_prompt_is_rejected(build: Callable[..., Station]) -> None:
    """`start(prompt)` sets the prompt for that run; a blank or non-string prompt is a ValueError and no run starts."""
    parts = build()

    async def scenario() -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            await parts.controller.start("   ")
        with pytest.raises(ValueError, match="non-empty string"):
            await parts.controller.set_prompt(42)  # type: ignore[arg-type]
        assert parts.controller.state == station.IDLE and parts.controller.run_number == 0
        await start_and_wait_for_chunks(parts, chunks=1, prompt="wipe the table")
        await parts.controller.stop()

    asyncio.run(scenario())

    assert parts.clients[0].observations[0]["prompt"] == "wipe the table"


# Status and close


def test_status_before_any_run_and_after_close(build: Callable[..., Station]) -> None:
    """status() is complete before the first run (zeros and Nones), and close() stops a run for process exit."""
    parts = build()

    idle = parts.controller.status()
    assert idle["state"] == station.IDLE and idle["run"] == 0 and idle["chunks"] == 0
    assert idle["session_id"] is None and idle["started_at"] is None and idle["run_elapsed_s"] is None
    assert idle["timing"] == control.RefillTiming().as_dict()
    assert idle["clamps"] == 0 and idle["max_ee_speed_requested_m_s"] == 0.0
    assert idle["max_ee_speed_m_s"] == conftest.LOOP_DEFAULTS["max_ee_speed_m_s"]

    async def scenario() -> None:
        await start_and_wait_for_chunks(parts, chunks=1)
        await parts.controller.close()

    asyncio.run(scenario())

    assert parts.controller.state == station.IDLE
    assert parts.clients[0].closed


def test_close_switches_both_arms_off_after_the_cameras(
    build: Callable[..., Station], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Process exit switches the arms off (`FollowerArm.torque_off()`), never just closes them: a closed driver
    leaves the motors holding; and it does so after the readers, so the live view shows the scene until the end.
    One arm's failure does not stop the other's switch-off."""
    parts = build()
    events: list[str] = []
    original_close_cameras = parts.controller.close_cameras

    def close_cameras() -> None:
        events.append("cameras")
        original_close_cameras()

    def torque_off(arm: followers.FollowerArm) -> None:
        events.append(arm.name)
        if arm.name == "left":
            raise RuntimeError("can_left refused")

    monkeypatch.setattr(parts.controller, "close_cameras", close_cameras)
    monkeypatch.setattr(followers.FollowerArm, "torque_off", torque_off)

    asyncio.run(parts.controller.close())

    assert events == ["cameras", "left", "right"]


def test_robot_down_while_a_stop_is_in_progress_still_switches_both_arms_off(build: Callable[..., Station]) -> None:
    parts = build()

    async def scenario() -> bool:
        await start_and_wait_for_chunks(parts, chunks=1)
        stopping = asyncio.ensure_future(parts.controller.stop())
        await asyncio.sleep(0)
        assert parts.controller.state == station.STOPPING
        switched_off = await parts.controller.close()
        await stopping
        return switched_off

    assert asyncio.run(scenario()) is True
    assert parts.controller.state == station.IDLE
    assert all(robot.closed for robot in parts.robots.values())


def test_from_config_checks_the_camera_roles_before_opening_any_arm() -> None:
    """A camera that is not a serve role fails in from_config with ConfigError, before i2rt is even imported."""
    cameras = [config.CameraConfig(name=name, device=i, width=64, height=48) for i, name in enumerate(("top", "back"))]
    cfg = conftest.make_app_config(cameras=cameras)

    with pytest.raises(errors.ConfigError, match=r"\['back'\]"):
        station.StationController.from_config(cfg, inference_log_path=None)


def test_every_runs_client_records_the_run_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    """Live View only shows the station, so every run's dream client records the run (no `_record` key turns its
    recorder off), with or without a `live` section; a start that names another checkpoint keeps it."""
    robots = {"left": conftest.FakeRobot(), "right": conftest.FakeRobot()}
    opened: list[dict[str, Any]] = []

    def open_client(dream_config: dict[str, Any]) -> conftest.FakeClient:
        opened.append(dream_config)
        return conftest.FakeClient()

    monkeypatch.setattr(yam, "open_arm", lambda cfg, name: conftest.yam_arm(name, robots[name]))
    monkeypatch.setattr(kinematics.ArmKinematics, "from_config", lambda cfg: test_control.StillKinematics())
    monkeypatch.setattr(run, "open_client", open_client)
    live_section = config.LiveConfig(robot_id="rbt_test", cameras=conftest.CAMERA_NAMES)
    controller = station.StationController.from_config(
        conftest.make_app_config(live=live_section), inference_log_path=None
    )

    controller._make_client(None, None)
    controller._make_client("alice/towels-ft", None)

    assert [config_.get("checkpoint") for config_ in opened] == [run.DEFAULT_CHECKPOINT, "alice/towels-ft"]
    assert all("_record" not in config_ for config_ in opened)


@pytest.mark.parametrize(("sys0", "threshold_ms"), [(False, 900), (True, 1000)], ids=["plain", "sys0"])
def test_every_runs_client_gets_the_sync_threshold_of_the_stations_mode(
    sys0: bool, threshold_ms: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A YAM station's clients go sync above 900 ms, and above 1000 ms on a station started with System 0
    (`run.policy_config`, `sys0`); a start that names another checkpoint keeps it."""
    from robots.sys0 import arm_process

    robots = {"left": conftest.FakeRobot(), "right": conftest.FakeRobot()}
    arms = {name: conftest.yam_arm(name, robot) for name, robot in robots.items()}
    opened: list[dict[str, Any]] = []

    def open_client(dream_config: dict[str, Any]) -> conftest.FakeClient:
        opened.append(dream_config)
        return conftest.FakeClient()

    monkeypatch.setattr(yam, "open_arm", lambda cfg, name: arms[name])
    monkeypatch.setattr(arm_process, "open_arms", lambda cfgs: dict(arms))
    monkeypatch.setattr(kinematics.ArmKinematics, "from_config", lambda cfg: test_control.StillKinematics())
    monkeypatch.setattr(run, "open_client", open_client)
    controller = station.StationController.from_config(
        conftest.make_app_config(action_horizon=48, control_hz=20.0), inference_log_path=None, sys0=sys0
    )

    controller._make_client(None, None)
    controller._make_client("alice/towels-ft", None)

    assert [config_["sync_above_ms"] for config_ in opened] == [threshold_ms, threshold_ms]


def test_a_boot_whose_right_arm_cannot_open_switches_the_opened_left_arm_off(monkeypatch: pytest.MonkeyPatch) -> None:
    robot = conftest.FakeRobot()
    switched: list[str] = []
    real_torque_off = followers.FollowerArm.torque_off

    def open_arm(cfg: config.ArmConfig, name: str) -> followers.FollowerArm:
        if name == "right":
            raise errors.ArmError("arm 'right': motor 3 on can_right did not answer")
        return conftest.yam_arm(name, robot)

    def torque_off(arm: followers.FollowerArm) -> None:
        switched.append(arm.name)
        real_torque_off(arm)

    monkeypatch.setattr(yam, "open_arm", open_arm)
    monkeypatch.setattr(kinematics.ArmKinematics, "from_config", lambda cfg: test_control.StillKinematics())
    monkeypatch.setattr(followers.FollowerArm, "torque_off", torque_off)

    with pytest.raises(errors.ArmError, match="motor 3 on can_right did not answer"):
        station.StationController.from_config(conftest.make_app_config(), inference_log_path=None)

    assert switched == ["left"] and robot.closed


def test_a_dream_error_from_the_client_build_is_recorded(build: Callable[..., Station]) -> None:
    """Opening the run's client raising (a bad checkpoint string) is a failed start in `error`, with no cameras
    opened."""

    def bad_client() -> conftest.FakeClient:
        raise dream.DreamConfigError("config key 'checkpoint' must be a non-empty str")

    parts = build(client_factory=bad_client)

    async def scenario() -> None:
        await parts.controller.start()
        await wait_for(lambda: parts.controller.state == station.ERROR, "error")

    asyncio.run(scenario())

    assert parts.controller.last_error["type"] == "DreamConfigError"  # type: ignore[index]
    assert len(parts.camera_sets) == 1


def test_managed_exposure_is_held_for_the_length_of_a_run_and_released_after(build: Callable[..., Station]) -> None:
    """The station freezes every reader's managed exposure from the start request (before frame 0) and releases it
    when the run ends, so a brightness change is never written into an episode; between runs the controller
    converges again. Readers without managed exposure (a cv2 device) are simply not asked."""
    parts = build()
    holds: list[tuple[str, bool]] = []
    for reader in parts.controller.cameras[:2]:
        reader._hold = lambda hold, name=reader.name: holds.append((name, hold))  # what a RealSense source has

    async def scenario() -> None:
        await start_and_wait_for_chunks(parts)
        assert holds == [("top", True), ("left", True)]
        assert parts.controller.status()["exposure"] == {}, "fake sources report no controller state"
        capture_fps = parts.controller.status()["capture_fps"]
        assert set(capture_fps) == {reader.name for reader in parts.controller.cameras}, (
            "every camera, measured or not"
        )
        await parts.controller.stop()
        assert holds[2:] == [("top", False), ("left", False)]
        await start_and_wait_for_chunks(parts)
        await parts.controller.stop()
        assert holds[4:] == [("top", True), ("left", True), ("top", False), ("left", False)]

    asyncio.run(scenario())


def test_start_without_any_prompt_is_refused_until_one_is_typed(build: Callable[..., Station]) -> None:
    """A config without `task` gives an empty prompt: Start alone is refused and nothing runs; Start with a prompt, or
    after Apply prompt, runs with it."""
    parts = build(task="")

    async def scenario() -> None:
        with pytest.raises(errors.StationError, match="no task prompt"):
            await parts.controller.start()
        assert parts.controller.state == station.IDLE and parts.controller.run_number == 0
        await parts.controller.set_prompt("fold the towel")
        await start_and_wait_for_chunks(parts, chunks=1)
        await parts.controller.stop()

    asyncio.run(scenario())

    assert parts.clients[0].observations[0]["prompt"] == "fold the towel"


def serve_on_fakes(
    monkeypatch: pytest.MonkeyPatch,
    make_cameras: Callable[[], list[cameras.CameraReader]] | None = None,
) -> dict[str, conftest.FakeRobot]:
    """Makes serve()'s StationController.from_config open two YAM arms over FakeRobots."""
    robots = {"left": conftest.FakeRobot(), "right": conftest.FakeRobot()}

    def fake_cameras() -> list[cameras.CameraReader]:
        stale_after_s = conftest.LOOP_DEFAULTS["stale_after_s"]
        return [
            cameras.CameraReader(name, conftest.FakeCaptureSource(), stale_after_s) for name in conftest.CAMERA_NAMES
        ]

    def from_config(
        cfg: config.AppConfig,
        *,
        inference_log_path: str | None,
        sys0: bool = False,
        skip_tcp_calibration: bool = False,
    ) -> station.StationController:
        return station.StationController(
            {name: conftest.yam_arm(name, robot) for name, robot in robots.items()},
            task=conftest.TASK,
            make_cameras=make_cameras or fake_cameras,
            make_client=lambda checkpoint, action_space: conftest.FakeClient(
                horizon=conftest.LOOP_DEFAULTS["action_horizon"]
            ),
            loop_settings=dict(conftest.LOOP_DEFAULTS),
            inference_log_path=inference_log_path,
        )

    monkeypatch.setattr(station.StationController, "from_config", staticmethod(from_config))
    return robots


def serve_fakes(
    cfg: config.AppConfig, tmp_path: pathlib.Path, api_port: int = 0, arms_lock_path: pathlib.Path | None = None
) -> None:
    """Runs serve() on the fakes on this (the main) thread, as `python -m robots.station` does."""
    log_path = str(tmp_path / "inference.jsonl")
    serving = station.serve(
        cfg, host="127.0.0.1", api_port=api_port, inference_log_path=log_path, arms_lock_path=arms_lock_path
    )
    asyncio.run(asyncio.wait_for(serving, WAIT_S))


def switched_off(robots: dict[str, conftest.FakeRobot], caplog: pytest.LogCaptureFixture) -> bool:
    """Both arms closed (a driver without a motor chain is only closed) and serve()'s last line says so."""
    return all(robot.closed for robot in robots.values()) and caplog.records[-1].message == station.SWITCHED_OFF_LINE


def test_a_camera_that_cannot_open_at_boot_switches_the_opened_arms_off(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:

    def no_camera() -> list[cameras.CameraReader]:
        raise errors.CameraError("camera 'top' cannot open /dev/video4")

    robots = serve_on_fakes(monkeypatch, no_camera)
    with caplog.at_level(logging.INFO), pytest.raises(errors.CameraError, match="video4"):
        serve_fakes(conftest.make_app_config(), tmp_path)
    assert switched_off(robots, caplog)


def test_an_api_port_in_use_switches_the_opened_arms_off(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    robots = serve_on_fakes(monkeypatch)
    with socket.socket() as holder:
        holder.bind(("127.0.0.1", 0))
        holder.listen()
        taken = holder.getsockname()[1]
        with caplog.at_level(logging.INFO), pytest.raises(OSError):
            serve_fakes(conftest.make_app_config(), tmp_path, api_port=taken)
    assert switched_off(robots, caplog)


def test_a_live_view_client_that_cannot_start_switches_the_opened_arms_off(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:

    def no_live(*args: Any, **kwargs: Any) -> None:
        raise dream.DreamConfigError("Live View needs PyAV, OpenCV and websockets")

    monkeypatch.setattr(dream, "LiveViewClient", no_live)
    robots = serve_on_fakes(monkeypatch)
    cfg = conftest.make_app_config(live=config.LiveConfig(robot_id="rbt_test", cameras=conftest.CAMERA_NAMES))
    with caplog.at_level(logging.INFO), pytest.raises(dream.DreamConfigError, match="websockets"):
        serve_fakes(cfg, tmp_path)
    assert switched_off(robots, caplog)
    assert not any("station API listening" in record.message for record in caplog.records)


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGHUP], ids=["robot-down", "hangup"])
def test_a_stop_while_the_station_opens_is_honored_once_it_has(
    signum: int, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    received: list[int] = []

    def opening_while_stopped() -> list[cameras.CameraReader]:
        os.kill(os.getpid(), signum)
        time.sleep(0.2)
        stale_after_s = conftest.LOOP_DEFAULTS["stale_after_s"]
        return [
            cameras.CameraReader(name, conftest.FakeCaptureSource(), stale_after_s) for name in conftest.CAMERA_NAMES
        ]

    robots = serve_on_fakes(monkeypatch, opening_while_stopped)
    previous = signal.signal(signum, lambda number, frame: received.append(number))
    try:
        with caplog.at_level(logging.INFO):
            serve_fakes(conftest.make_app_config(), tmp_path)
    finally:
        signal.signal(signum, previous)
    assert received == [], "serve()'s own handler took the signal"
    assert switched_off(robots, caplog)
    assert not any("station ready" in record.message for record in caplog.records)


def test_a_stop_while_the_arms_open_ends_the_station_before_the_cameras_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    received: list[int] = []
    opened_cameras: list[bool] = []

    def cameras_opened() -> list[cameras.CameraReader]:
        opened_cameras.append(True)
        return []

    robots = serve_on_fakes(monkeypatch, cameras_opened)
    opens = station.StationController.from_config

    def stopped_while_opening(
        cfg: config.AppConfig,
        *,
        inference_log_path: str | None,
        sys0: bool = False,
        skip_tcp_calibration: bool = False,
    ) -> station.StationController:
        os.kill(os.getpid(), signal.SIGINT)
        time.sleep(0.2)
        return opens(cfg, inference_log_path=inference_log_path, sys0=sys0)

    monkeypatch.setattr(station.StationController, "from_config", staticmethod(stopped_while_opening))
    previous = signal.signal(signal.SIGINT, lambda number, frame: received.append(number))
    try:
        with caplog.at_level(logging.INFO):
            serve_fakes(conftest.make_app_config(), tmp_path)
    finally:
        signal.signal(signal.SIGINT, previous)
    assert received == [], "serve()'s own handler took the signal"
    assert opened_cameras == []
    assert switched_off(robots, caplog)


def test_the_station_holds_the_arms_lock_from_before_they_open_until_they_are_switched_off(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    lock = tmp_path / arms_lock.LOCK_NAME

    def no_camera() -> list[cameras.CameraReader]:
        raise errors.CameraError("camera 'top' cannot open /dev/video4")

    robots = serve_on_fakes(monkeypatch, no_camera)
    opens = station.StationController.from_config
    holders: list[str | None] = []

    def from_config(
        cfg: config.AppConfig,
        *,
        inference_log_path: str | None,
        sys0: bool = False,
        skip_tcp_calibration: bool = False,
    ) -> station.StationController:
        holders.append(arms_lock.held_by(lock))
        return opens(cfg, inference_log_path=inference_log_path, sys0=sys0)

    monkeypatch.setattr(station.StationController, "from_config", staticmethod(from_config))
    with caplog.at_level(logging.INFO), pytest.raises(errors.CameraError):
        serve_fakes(conftest.make_app_config(), tmp_path, arms_lock_path=lock)
    assert holders == [
        f"the arms are held by the station agent (pid {os.getpid()}): 'dreamcli robot down' stops it, "
        "then run this again"
    ]
    assert switched_off(robots, caplog) and arms_lock.held_by(lock) is None

    with arms_lock.acquire(lock, arms_lock.SETUP), pytest.raises(errors.StationError, match="held by dreamcli setup"):
        serve_fakes(conftest.make_app_config(), tmp_path, arms_lock_path=lock)
    assert len(holders) == 1, "no arm was opened"


def test_main_takes_the_arms_lock_it_is_given_else_the_station_directorys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    seen: list[object] = []

    async def fake_serve(cfg: config.AppConfig, **kwargs: object) -> None:
        seen.append(kwargs["arms_lock_path"])

    monkeypatch.setattr(station, "serve", fake_serve)
    monkeypatch.setattr(config, "load", lambda path: conftest.make_app_config())
    monkeypatch.setenv(cli.HOME_ENV, str(tmp_path / "home"))
    station.main(["--config", "station.yaml", "--arms-lock", str(tmp_path / "given.lock")])
    station.main(["--config", "station.yaml"])
    assert seen == [tmp_path / "given.lock", (tmp_path / "home" / arms_lock.LOCK_NAME).resolve()]


def test_main_passes_the_unlisted_skip_of_the_tool_point_check_to_serve(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """`dreamcli robot up --skip-tcp-calibration` hands the flag to the station as is; main() gives it to serve(),
    which runs `robots.run.check_tool_point()` with it at the start and at every Start. --help does not list it."""
    seen: list[object] = []

    async def fake_serve(cfg: config.AppConfig, **kwargs: object) -> None:
        seen.append(kwargs["skip_tcp_calibration"])

    monkeypatch.setattr(station, "serve", fake_serve)
    monkeypatch.setattr(config, "load", lambda path: conftest.make_app_config())
    monkeypatch.setenv(cli.HOME_ENV, str(tmp_path / "home"))
    station.main(["--config", "station.yaml"])
    station.main(["--config", "station.yaml", "--skip-tcp-calibration"])
    assert seen == [False, True]


# Live View: the station feeds dream.LiveViewClient, as any control loop does


class FakeLive:
    """Stands in for the station's `dream.LiveViewClient`: records what the feed streams and says what a viewer chose.

    Attributes:
    - streamed: `(obs, status, keywords)` per stream() call.
    - dream_client: What the feed last set the attribute to.
    - prompt: What get_prompt() returns.
    - model: What get_model() returns.
    - steps: What get_steps() returns.
    """

    def __init__(self) -> None:
        self.streamed: list[tuple[dict[str, Any], str, dict[str, Any]]] = []
        self.dream_client: Any = None
        self.prompt: str | None = "fold the towel"
        self.model: tuple[str | None, str | None] = (None, None)
        self.steps: int | None = None

    def stream(self, obs: dict[str, Any], status: str, **keywords: Any) -> None:
        self.streamed.append((obs, status, {**keywords, "dream_client": self.dream_client}))

    def stats(self) -> dict[str, Any]:
        return {"linked": True}

    def get_prompt(self) -> str | None:
        return self.prompt

    def get_model(self) -> tuple[str | None, str | None]:
        return self.model

    def get_steps(self) -> int | None:
        return self.steps


def test_the_feed_streams_the_station_to_live_view_and_the_runs_dream_client_while_running(
    build: Callable[..., Station],
) -> None:
    """Between runs the feed streams the cameras, `idle`, the arms' reads, the task and the model, with no dream
    client; during a run `running`, the run number, and the run's own dream client for the round trips and session;
    status() reports the client's counters under `live`."""
    parts = build()
    live = FakeLive()

    every_camera = {f"video.{name}" for name in conftest.CAMERA_NAMES}

    async def scenario() -> None:
        feed = asyncio.create_task(station._feed_live(parts.controller, live))  # type: ignore[arg-type]
        try:
            # Each camera's reader brings its first frame in its own time and the feed leaves out a camera with no
            # frame yet, so wait for a call that carries all of them rather than for a number of calls.
            await wait_for(lambda: bool(live.streamed) and set(live.streamed[-1][0]) == every_camera, "the idle feed")
            obs, status, keywords = live.streamed[-1]
            assert set(obs) == every_camera and status == station.IDLE
            assert set(keywords["arms"]) == {"left", "right"}
            assert all(len(read[0]) == 6 and 0.0 <= read[1] <= 1.0 for read in keywords["arms"].values())
            assert keywords["prompt"] == conftest.TASK and keywords["run"] is None and keywords["error"] is None
            assert keywords["model"] == (parts.controller.checkpoint, parts.controller.action_space)
            assert keywords["dream_client"] is None
            await start_and_wait_for_chunks(parts)
            await wait_for(lambda: live.streamed[-1][1] == station.RUNNING, "the running feed")
            _, _, running = live.streamed[-1]
            assert running["run"] == 1 and running["dream_client"] is parts.clients[0]
            assert parts.controller.status()["live"] == {"linked": True}
            await parts.controller.stop()
        finally:
            feed.cancel()

    asyncio.run(scenario())


def test_the_pages_buttons_run_the_stations_own_calls_with_what_the_viewer_chose(
    build: Callable[..., Station], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Start runs start() with the viewer's task, model and steps; Stop is stop() then home(); a refusal is the
    station's own words; a home that ends in `error` is a failure with the arm's message."""
    parts = build()
    live = FakeLive()

    async def scenario() -> list[dict[str, Any]]:
        press = station._live_press(parts.controller, live, asyncio.get_running_loop())  # type: ignore[arg-type]
        answers = [await asyncio.wrap_future(press(dream.EventChange.start))]
        await wait_for(lambda: parts.controller.state == station.RUNNING, "running")
        answers.append(await asyncio.wrap_future(press(dream.EventChange.end)))

        async def refused(*args: Any) -> dict[str, Any]:
            raise errors.StationError("a home move is in progress")

        monkeypatch.setattr(parts.controller, "start", refused)
        answers.append(await asyncio.wrap_future(press(dream.EventChange.start)))

        async def stranded() -> dict[str, Any]:
            return {"state": station.ERROR, "last_error": {"message": "left: did not reach home"}}

        monkeypatch.setattr(parts.controller, "home", stranded)
        answers.append(await asyncio.wrap_future(press(dream.EventChange.home)))
        return answers

    answers = asyncio.run(scenario())
    assert answers[0] == {"state": station.STARTING} and parts.controller.prompt == "fold the towel"
    assert answers[1]["state"] == station.IDLE and "error" not in answers[1]
    assert answers[2]["error"] == "a home move is in progress"
    assert answers[3] == {"state": station.ERROR, "error": "left: did not reach home"}
