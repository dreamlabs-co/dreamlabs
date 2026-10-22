"""Behaviour tests for the rollout recorder (`robots.rollouts`, `robots.rollout_episode`).

A real `ControlLoop` over the fakes runs into a real `Recorder` under `tmp_path`: the episode that comes out is read
back with pyarrow and PyAV, the way a LeRobot v2.1 reader reads it. Failure paths (a full disk, an encoder that cannot
keep up, a killed run) are driven by patching the writer or killing a real child process.
"""

import asyncio
import errno
import json
import pathlib
import signal
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import av
import conftest
import numpy as np
import pyarrow.parquet as pq
import pytest
import test_control
from shared.datasets import embodiments, validation

from robots import config, control, errors, rollout_episode, rollouts, run, types

## Ticks the loop tests run: eight at 30 Hz, two chunks of four.
TICKS = 8


def make_recorder(
    directory: pathlib.Path,
    *,
    cameras: tuple[str, ...] = conftest.CAMERA_NAMES,
    fps: float = 30.0,
    run_number: int = 3,
    max_ee_speed_m_s: float = conftest.LOOP_DEFAULTS["max_ee_speed_m_s"],
    joints: tuple[int, int] = (6, 6),
) -> rollouts.Recorder:
    """A recorder writing into `directory`, named like a test station, with the test loop's settings; `joints` is
    each arm's, left then right."""
    return rollouts.Recorder(
        config.RolloutsConfig(dir=str(directory), robot_name="test-bot"),
        body="yam",
        checkpoint="dreamlabs/test-ckpt",
        run_number=run_number,
        fps=fps,
        camera_names=cameras,
        joints={"left": joints[0], "right": joints[1]},
        robot_id="rbt_test",
        settings={key: conftest.LOOP_DEFAULTS[key] for key in rollouts.SETTINGS}
        | {"max_ee_speed_m_s": max_ee_speed_m_s},
    )


@pytest.fixture
def build() -> Iterator[Callable[..., conftest.LoopHarness]]:
    """A `make_loop` factory on a YieldingClock; capture threads stopped on teardown."""
    built: list[conftest.LoopHarness] = []

    def factory(**kwargs: Any) -> conftest.LoopHarness:
        kwargs.setdefault("clock", test_control.YieldingClock())
        harness = conftest.make_loop(**kwargs)
        built.append(harness)
        return harness

    yield factory
    for harness in built:
        for reader in harness.readers:
            if reader.is_alive():
                reader.stop()


def run_loop(harness: conftest.LoopHarness, ticks: int = TICKS) -> None:
    """Runs `ticks` ticks of the loop, letting each refill land before the next tick."""

    async def scenario() -> None:
        async with harness.loop:
            for i in range(ticks):
                await harness.loop.step(i / 30.0)
                await test_control.wait_for_refill_task()

    asyncio.run(scenario())


def only_episode(directory: pathlib.Path) -> pathlib.Path:
    """The one episode folder under `directory`."""
    folders = [path for path in directory.iterdir() if path.is_dir()]
    assert len(folders) == 1, folders
    return folders[0]


def video_frames(path: pathlib.Path) -> list[Any]:
    """Every decoded frame of a video as RGB arrays."""
    with av.open(str(path)) as container:
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(container.streams.video[0])]


def tick_arguments(i: int, shape: tuple[int, int, int] = (48, 64, 3)) -> tuple[Any, ...]:
    """The arguments of `Recorder.tick` for tick `i`, with frames that change every tick."""
    arm = types.ArmState(np.full(6, 0.01 * i), np.zeros(6), 0.5, None, None)
    state = types.RobotState(arm, arm, i / 30.0)
    command = types.ArmCommand(np.full(6, 0.02 * i), 0.25)
    frames = {
        name: types.Frame(np.full(shape, (i * 7 + k) % 256, dtype=np.uint8), name, i / 30.0, ())
        for k, name in enumerate(conftest.CAMERA_NAMES)
    }
    return i / 30.0, 1, i % 4, state, types.Action(command, command), frames


def feed(recorder: rollouts.Recorder, ticks: int) -> None:
    """Begins an episode and records `ticks` ticks into it."""
    recorder.begin("pick up the cup")
    for i in range(ticks):
        recorder.tick(*tick_arguments(i))


def test_a_run_saves_a_loadable_lerobot_v21_episode(
    build: Callable[..., conftest.LoopHarness], tmp_path: pathlib.Path
) -> None:
    """Eight ticks over three cameras: the folder holds parquet rows, one VP9 video per camera with a frame per
    row, the v2.1 meta files, and `rollout.json` with the prompt, checkpoint and robot name."""
    recorder = make_recorder(tmp_path)
    harness = build(recorder=recorder, inference_log_path=str(tmp_path / "station.jsonl"))
    run_loop(harness)
    recorder.wait()

    root = only_episode(tmp_path)
    info = json.loads((root / "rollout.json").read_text())
    assert info["status"] == "complete" and info["frames"] == TICKS and info["stop_reason"] == "stopped"
    assert (info["task"], info["checkpoint"], info["robot_name"], info["robot_id"]) == (
        conftest.TASK,
        "dreamlabs/test-ckpt",
        "test-bot",
        "rbt_test",
    )
    assert not (root / "INCOMPLETE.txt").exists() and not (root / "ticks.f64").exists()
    meta = json.loads((root / "meta/info.json").read_text())
    assert meta["codebase_version"] == "v2.1" and meta["total_frames"] == TICKS and meta["fps"] == 30.0
    assert meta["data_path"].format(episode_chunk=0, episode_index=0) == "data/chunk-000/episode_000000.parquet"
    assert set(meta["features"]) >= {
        "observation.state",
        "action",
        *(f"observation.images.{n}" for n in conftest.CAMERA_NAMES),
    }
    assert meta["features"]["observation.state"]["names"][:2] == ["left_joint_0", "left_joint_1"]
    assert meta["features"][f"observation.images.{conftest.CAMERA_NAMES[0]}"]["info"]["video.codec"] == "vp9"
    table = pq.read_table(root / "data/chunk-000/episode_000000.parquet")
    assert table.num_rows == TICKS
    assert table["frame_index"].to_pylist() == list(range(TICKS))
    assert table["timestamp"].to_pylist() == pytest.approx([i / 30.0 for i in range(TICKS)])
    assert len(table["observation.state"][0]) == 14 and len(table["action"][0]) == 14
    for name in conftest.CAMERA_NAMES:
        path = root / f"videos/chunk-000/observation.images.{name}/episode_000000.mp4"
        frames = video_frames(path)
        assert len(frames) == TICKS and frames[0].shape == conftest.FRAME_SHAPE
        # The fake source is pure blue; RGB order puts the 255 in the last channel.
        assert frames[0][..., 2].mean() > 200 and frames[0][..., 0].mean() < 60
    assert (root / "meta/tasks.jsonl").read_text() == json.dumps({"task_index": 0, "task": conftest.TASK}) + "\n"
    episodes = json.loads((root / "meta/episodes.jsonl").read_text())
    assert episodes == {"episode_index": 0, "tasks": [conftest.TASK], "length": TICKS}
    stats = json.loads((root / "meta/episodes_stats.jsonl").read_text())["stats"]
    assert np.array(stats["observation.images.top"]["mean"]).shape == (3, 1, 1)
    assert stats["action"]["count"] == [TICKS] and len(stats["action"]["mean"]) == 14


def test_the_rollouts_folder_passes_the_dream_dataset_check_as_it_is(
    build: Callable[..., conftest.LoopHarness], tmp_path: pathlib.Path
) -> None:
    """The check `dream.datasets.upload_dataset` runs before a byte is sent (one directory per session, each a
    LeRobot v2.1 dataset) accepts the rollouts folder untouched, extra files and all, so a run can be uploaded as
    training data: a run as long as one training sample (`validation.min_episode_frames()`), the shortest it takes."""
    ticks = validation.min_episode_frames(30)
    recorder = make_recorder(tmp_path)
    run_loop(build(recorder=recorder), ticks)
    recorder.wait()

    body = embodiments.parse(
        {
            "state": {"arms": [0, 14]},
            "action": {"arms": [0, 14]},
            "video": {name: f"observation.images.{name}" for name in conftest.CAMERA_NAMES},
        }
    )
    summary = validation.check_dataset(validation.list_files(tmp_path), body, validation.DirectoryReader(tmp_path))
    assert (summary["sessions"], summary["episodes"], summary["frames"]) == (1, 1, ticks)
    assert summary["state_dim"] == summary["action_dim"] == 14


def test_the_action_column_is_what_the_arms_were_commanded(
    build: Callable[..., conftest.LoopHarness], tmp_path: pathlib.Path
) -> None:
    """The saved commanded vector of each tick is the 7-vector the fake arm received; the measured vector is what it
    reported."""
    recorder = make_recorder(tmp_path)
    harness = build(recorder=recorder)
    harness.robots["left"].joint_pos = np.full(6, 0.3)
    run_loop(harness)
    recorder.wait()

    table = pq.read_table(only_episode(tmp_path) / "data/chunk-000/episode_000000.parquet")
    action = np.array(table["action"].to_pylist())
    state = np.array(table["observation.state"].to_pylist())
    sent = np.array(harness.robots["left"].commands)[:TICKS]  # the last write is the hold at exit
    assert action[:, :7] == pytest.approx(sent, abs=1e-5)
    assert state[:, :6] == pytest.approx(np.full((TICKS, 6), 0.3), abs=1e-6)
    assert state[:, 6] == pytest.approx(np.full(TICKS, 0.5), abs=1e-6)


def test_every_model_reply_is_saved_raw_and_the_inference_log_is_copied(
    build: Callable[..., conftest.LoopHarness], tmp_path: pathlib.Path
) -> None:
    """One `chunks/reply_N.npz` per inference record, holding the client's arrays as it returned them; the episode's
    `inference.jsonl` carries run_start, the inferences and run_stop, beside the station-wide log."""
    recorder = make_recorder(tmp_path / "rollouts")
    harness = build(recorder=recorder, inference_log_path=str(tmp_path / "station.jsonl"))
    run_loop(harness)
    recorder.wait()

    root = only_episode(tmp_path / "rollouts")
    records = [json.loads(line) for line in (root / "inference.jsonl").read_text().splitlines()]
    kinds = [record["kind"] for record in records]
    assert kinds[0] == "run_start" and kinds[-1] == "run_stop" and kinds.count("inference") == 2
    assert (tmp_path / "station.jsonl").read_text() == (root / "inference.jsonl").read_text()
    replies = sorted((root / "chunks").glob("reply_*.npz"))
    assert [path.name for path in replies] == ["reply_000000.npz", "reply_000001.npz"]
    with np.load(replies[1]) as reply:
        assert reply["q_r_chunk"].shape == (4, 7) and reply["q_r_chunk"][0, 0] == 4.0
        assert float(reply["_observed_at_s"]) >= 0.0 and int(reply["_after_tick"]) <= TICKS


def test_a_run_without_a_station_log_still_gets_its_inference_log_in_the_episode(
    build: Callable[..., conftest.LoopHarness], tmp_path: pathlib.Path
) -> None:
    """The mirror works on its own: no `inference_log_path`, and the episode still has `inference.jsonl`."""
    recorder = make_recorder(tmp_path)
    run_loop(build(recorder=recorder))
    recorder.wait()
    assert (only_episode(tmp_path) / "inference.jsonl").read_text().count('"kind": "inference"') == 2


def test_a_task_change_ends_the_episode_and_starts_another(
    build: Callable[..., conftest.LoopHarness], tmp_path: pathlib.Path
) -> None:
    """set_task() mid-run: the first prompt's ticks are one episode, the new prompt's ticks another."""
    recorder = make_recorder(tmp_path)
    harness = build(recorder=recorder)

    async def scenario() -> None:
        async with harness.loop:
            for i in range(4):
                await harness.loop.step(i / 30.0)
                await test_control.wait_for_refill_task()
            harness.loop.set_task("wipe the table")
            for i in range(4, 8):
                await harness.loop.step(i / 30.0)
                await test_control.wait_for_refill_task()

    asyncio.run(scenario())
    recorder.wait()

    infos = [json.loads((path / "rollout.json").read_text()) for path in recorder.episodes]
    assert [info["task"] for info in infos] == [conftest.TASK, "wipe the table"]
    assert [info["stop_reason"] for info in infos] == ["task_changed", "stopped"]
    assert all(info["status"] == "complete" and info["frames"] > 0 for info in infos)
    assert sum(info["frames"] for info in infos) == TICKS
    assert recorder.episodes[1].name.endswith("_run3_2")


def test_a_full_disk_stops_recording_not_the_run(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each camera's encoder hits ENOSPC after its fifth frame: recording stops, the error is logged and written to
    rollout.json, and the rows and videos that were saved still agree and load. The limit is per camera (the frame's
    `pts`), since the camera threads race: a count shared by all three let one camera take every frame first."""
    real_write = rollouts._write_frame

    def write(container: Any, stream: Any, rgb: Any, pts: int) -> None:
        if pts >= 5:
            raise OSError(errno.ENOSPC, "No space left on device")
        real_write(container, stream, rgb, pts)

    monkeypatch.setattr(rollouts, "_write_frame", write)
    recorder = make_recorder(tmp_path)
    feed(recorder, 30)
    recorder.end("stopped")
    recorder.wait()

    root = only_episode(tmp_path)
    info = json.loads((root / "rollout.json").read_text())
    assert info["status"] == "partial" and info["stop_reason"] == "disk_full" and "No space" in info["error"]
    assert info["frames"] == 5
    assert pq.read_table(root / "data/chunk-000/episode_000000.parquet").num_rows == info["frames"]
    assert len(video_frames(root / "videos/chunk-000/observation.images.top/episode_000000.mp4")) == info["frames"]


def test_a_run_ended_by_a_dreamapi_refusal_keeps_the_refusal_in_rollout_json(
    build: Callable[..., conftest.LoopHarness], tmp_path: pathlib.Path
) -> None:
    recorder = make_recorder(tmp_path)
    client = test_control.FailingClient(horizon=4, fail=lambda _: test_control.MISSING_CAMERA)
    harness = build(recorder=recorder, client=client)

    async def scenario() -> None:
        async with harness.loop:
            for tick in range(control.SERVE_FAULT_REPEATS):
                await harness.loop.step(float(tick))

    with pytest.raises(errors.ControlLoopError, match="will not help"):
        asyncio.run(scenario())
    recorder.wait()

    info = json.loads((only_episode(tmp_path) / "rollout.json").read_text())
    assert info["stop_reason"] == "ControlLoopError" and info["status"] == "empty"
    assert "observation is missing ['video.overhead']" in info["error"]


def test_rollout_json_says_what_the_speed_cap_did_to_the_plan(
    build: Callable[..., conftest.LoopHarness], tmp_path: pathlib.Path
) -> None:
    cap_m_s = np.sqrt(3.0) * 0.5
    recorder = make_recorder(tmp_path, fps=10.0, max_ee_speed_m_s=cap_m_s)
    harness = build(
        control_hz=10.0,
        client=test_control._FlatClient(horizon=4, values=[0.0, 0.5]),
        max_ee_speed_m_s=cap_m_s,
        kinematics=test_control.linear_kinematics(),
        recorder=recorder,
    )

    async def scenario() -> None:
        async with harness.loop:
            for tick in range(8):
                await harness.loop.step(tick * 0.1)

    asyncio.run(scenario())
    recorder.wait()

    info = json.loads((only_episode(tmp_path) / "rollout.json").read_text())
    assert info["max_ee_speed_m_s"] == cap_m_s and info["fps"] == 10.0
    assert info["max_stall_ticks"] == conftest.LOOP_DEFAULTS["max_stall_ticks"]
    assert info["clamps"] == harness.loop.clamp_count > 0
    assert info["max_ee_speed_requested_m_s"] == pytest.approx(np.sqrt(3.0) * 0.5 * 10.0)
    assert info["stalls"] == 0
    assert info["left_max_motor_temp_c"] is None and info["right_max_motor_temp_c"] is None


def test_rollout_json_counts_the_episodes_stalls(
    build: Callable[..., conftest.LoopHarness], tmp_path: pathlib.Path
) -> None:
    recorder = make_recorder(tmp_path)
    gate = threading.Event()
    client = test_control.AnswerOnceClient(horizon=1, gate=gate)
    harness = build(action_horizon=1, action_timeout_s=test_control.STALL_TIMEOUT_S, client=client, recorder=recorder)

    async def scenario() -> None:
        try:
            async with harness.loop:
                for tick in range(3):
                    await harness.loop.step(float(tick))
                gate.set()
                await test_control.wait_for_refill_task()
                await harness.loop.step(3.0)
                gate.clear()
                await harness.loop.step(4.0)
                await harness.loop.step(5.0)
        finally:
            gate.set()
            await test_control.wait_for_refill_task()

    asyncio.run(scenario())
    recorder.wait()

    info = json.loads((only_episode(tmp_path) / "rollout.json").read_text())
    assert info["stalls"] == harness.loop.stall_count == 4
    assert info["clamps"] == 0


def test_the_episode_keeps_each_arms_hottest_motor() -> None:
    counts = control._EpisodeCounts()
    counts.note_temps(None)
    counts.note_temps({"right": [35.0, 41.0, 38.0]})
    counts.note_temps({"right": [36.0, 39.0, 40.0]})
    assert (counts.left_max_motor_temp_c, counts.right_max_motor_temp_c) == (None, 41.0)


def test_a_station_records_its_configs_loop_settings(tmp_path: pathlib.Path) -> None:
    app = conftest.make_app_config(
        rollouts=config.RolloutsConfig(enabled=True, dir=str(tmp_path)), max_ee_speed_m_s=0.15
    )
    recorder = run.make_recorder(app, "ckpt", 1)
    assert recorder is not None
    folder = recorder.begin("task")
    recorder.end("stopped")
    recorder.wait()

    info = json.loads((folder / "rollout.json").read_text())
    assert {key: info[key] for key in rollouts.SETTINGS} == {key: getattr(app, key) for key in rollouts.SETTINGS}
    assert info["max_ee_speed_m_s"] == 0.15


def test_a_run_ended_by_a_refusal_marked_not_retryable_keeps_it_in_rollout_json(
    build: Callable[..., conftest.LoopHarness], tmp_path: pathlib.Path
) -> None:
    recorder = make_recorder(tmp_path)
    refusal = test_control.NOT_RETRYABLE[1]
    client = test_control.FailingClient(horizon=4, fail=lambda _: refusal)
    harness = build(recorder=recorder, client=client)

    async def scenario() -> None:
        async with harness.loop:
            await harness.loop.step(0.0)

    with pytest.raises(errors.ControlLoopError, match="not retryable"):
        asyncio.run(scenario())
    recorder.wait()

    info = json.loads((only_episode(tmp_path) / "rollout.json").read_text())
    assert info["stop_reason"] == "ControlLoopError" and "DA-6000 GATEWAY_FAULT (cannot_scale)" in info["error"]


def test_a_run_ended_by_a_gateway_nothing_answers_at_keeps_dreams_words_in_rollout_json(
    build: Callable[..., conftest.LoopHarness], tmp_path: pathlib.Path
) -> None:
    recorder = make_recorder(tmp_path)
    client = test_control.FailingClient(horizon=4, fail=lambda _: test_control.CONNECTION_REFUSED)
    harness = build(recorder=recorder, client=client, first_chunk_timeout_s=3.0)

    async def scenario() -> None:
        async with harness.loop:
            for tick in range(50):
                await harness.loop.step(tick * 0.1)

    with pytest.raises(errors.ControlLoopError, match="first_chunk_timeout_s=3 s"):
        asyncio.run(scenario())
    recorder.wait()

    info = json.loads((only_episode(tmp_path) / "rollout.json").read_text())
    assert info["stop_reason"] == "ControlLoopError" and "Connection refused" in info["error"]


def test_a_start_without_room_raises(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Under 2 GB free at the start: begin() raises naming the folder and the way out, and writes nothing."""
    monkeypatch.setattr(rollouts.shutil, "disk_usage", lambda _: type("U", (), {"free": 10 * 1024**2})())
    recorder = make_recorder(tmp_path / "rollouts")
    with pytest.raises(rollout_episode.RolloutError, match="rollouts.enabled: false"):
        recorder.begin("task")
    assert not list((tmp_path / "rollouts").iterdir())


def test_a_loop_whose_recorder_cannot_start_does_not_start(
    build: Callable[..., conftest.LoopHarness], tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The error reaches the caller before any camera or arm is touched, and the inference log is closed."""
    monkeypatch.setattr(rollouts.shutil, "disk_usage", lambda _: type("U", (), {"free": 0})())
    harness = build(recorder=make_recorder(tmp_path), inference_log_path=str(tmp_path / "station.jsonl"))

    async def scenario() -> None:
        async with harness.loop:
            pytest.fail("the loop must not start")

    with pytest.raises(rollout_episode.RolloutError):
        asyncio.run(scenario())
    assert not any(reader.is_alive() for reader in harness.readers)
    assert harness.robots["left"].commands == []


def test_an_encoder_that_cannot_keep_up_stops_recording_and_never_blocks_the_tick(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stuck encoder with a tiny queue allowance: tick() returns at once every time, the episode is marked
    `encoder_overloaded`, and what was encoded before is kept as a partial episode."""
    real_write = rollouts._write_frame
    monkeypatch.setattr(rollouts, "_write_frame", lambda *args: (time.sleep(0.2), real_write(*args)))
    monkeypatch.setattr(rollouts, "MAX_QUEUE_BYTES", 48 * 64 * 3 * 5)
    recorder = make_recorder(tmp_path)
    recorder.begin("task")
    started = time.perf_counter()
    for i in range(40):
        recorder.tick(*tick_arguments(i))
    assert time.perf_counter() - started < 0.5
    recorder.end("stopped")
    recorder.wait()
    info = json.loads((only_episode(tmp_path) / "rollout.json").read_text())
    assert info["status"] == "partial" and info["stop_reason"] == "encoder_overloaded"
    assert 1 <= info["frames"] < 40


def test_a_run_that_stopped_before_any_tick_leaves_an_empty_marker_and_no_dataset(tmp_path: pathlib.Path) -> None:
    """No commanded tick: status `empty`, the inference log kept, no parquet and no videos."""
    recorder = make_recorder(tmp_path)
    recorder.begin("task")
    recorder.end("stopped")
    recorder.wait()
    root = only_episode(tmp_path)
    assert json.loads((root / "rollout.json").read_text())["status"] == "empty"
    assert not (root / "data").exists() and not (root / "INCOMPLETE.txt").exists()


def test_a_killed_run_is_recovered_by_the_next_run(tmp_path: pathlib.Path) -> None:
    """A child process records 90 ticks and is killed with SIGKILL: its folder keeps INCOMPLETE.txt and the raw
    files, and the next recorder's begin() finishes it into a loadable partial episode whose rows and videos agree."""
    script = textwrap.dedent(
        f"""
        import os, signal, sys, time
        sys.path.insert(0, {str(pathlib.Path(__file__).parent)!r})
        import test_rollouts
        recorder = test_rollouts.make_recorder(__import__("pathlib").Path({str(tmp_path)!r}))
        test_rollouts.feed(recorder, 90)
        time.sleep(3.0)
        os.kill(os.getpid(), signal.SIGKILL)
        """
    )
    child = subprocess.run([sys.executable, "-c", script], check=False, timeout=60)
    assert child.returncode == -signal.SIGKILL
    root = only_episode(tmp_path)
    assert (root / "INCOMPLETE.txt").exists() and (root / "ticks.f64").exists()
    assert json.loads((root / "rollout.json").read_text())["status"] == "recording"

    follower = make_recorder(tmp_path)
    follower.begin("next run")
    follower.end("stopped")
    follower.wait()

    info = json.loads((root / "rollout.json").read_text())
    assert info["status"] == "partial" and info["stop_reason"] == "recovered_after_crash"
    # A killed encoder loses its look-ahead and the open fragment: well under a second of a 3 s run.
    assert 60 <= info["frames"] <= 90
    assert not (root / "INCOMPLETE.txt").exists() and not (root / "ticks.f64").exists()
    assert pq.read_table(root / "data/chunk-000/episode_000000.parquet").num_rows == info["frames"]
    for name in conftest.CAMERA_NAMES:
        path = root / f"videos/chunk-000/observation.images.{name}/episode_000000.mp4"
        assert len(video_frames(path)) == info["frames"]


def test_a_live_run_is_not_recovered_from_under_itself(tmp_path: pathlib.Path) -> None:
    """The marker names this process, which is alive, so a second recorder leaves the folder alone."""
    first = make_recorder(tmp_path)
    folder = first.begin("task")
    assert rollout_episode.unfinished(tmp_path) == []
    first.end("stopped")
    first.wait()
    assert (folder / "rollout.json").exists()


def test_the_config_turns_rollouts_on_and_rejects_typos(tmp_path: pathlib.Path) -> None:
    """Off by default (the live link records the run), `rollouts: {enabled: true}` gives a recorder, an unknown key
    fails by name."""
    off = conftest.make_app_config()
    assert not off.rollouts.enabled and off.rollouts.dir == "~/.dreamlabs/rollouts"
    assert run.make_recorder(off, "ckpt", 1) is None
    on = conftest.make_app_config(rollouts=config.RolloutsConfig(enabled=True))
    assert run.make_recorder(on, "ckpt", 1) is not None

    base = (pathlib.Path(__file__).parent.parent / "configs/yam.example.yaml").read_text()
    path = tmp_path / "station.yaml"
    path.write_text(base)
    assert not config.load(path).rollouts.enabled
    path.write_text(base + "rollouts: {enabled: true, dir: /data/rollouts, robot_name: bench}\n")
    loaded = config.load(path).rollouts
    assert (loaded.enabled, loaded.dir, loaded.robot_name) == (True, "/data/rollouts", "bench")
    path.write_text(base + "rollouts: {enable: false}\n")
    with pytest.raises(config.errors.ConfigError, match="rollouts"):
        config.load(path)
