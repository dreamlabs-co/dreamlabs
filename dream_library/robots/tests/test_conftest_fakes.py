"""Pins the behaviour of the conftest fakes without touching the library under test.

Every other test file relies on these doubles; if one of them drifts (a fake that queues instead
of failing, a clock that waits on wall time) the loop tests start passing or failing for the
wrong reason. These tests deliberately import nothing from `robots`.
"""

import asyncio
import threading
import time

import conftest
import dream
import numpy as np
import pytest

## Bounded wait for the worker threads below; a hang here is a fake bug, not a slow machine.
JOIN_TIMEOUT_S = 5.0


def test_fake_robot_reports_i2rt_shape_and_records_commands() -> None:
    robot = conftest.FakeRobot(joint_pos=np.arange(6), gripper_pos=0.25)

    obs = robot.get_observations()
    robot.command_joint_pos(np.array([1, 2, 3, 4, 5, 6, 0.5]))

    assert set(obs) == {"joint_pos", "joint_vel", "gripper_pos"}
    assert obs["joint_pos"].shape == (6,)
    assert obs["joint_vel"].shape == (6,)
    assert obs["gripper_pos"].shape == (1,)
    assert obs["gripper_pos"][0] == 0.25
    assert len(robot.commands) == 1
    np.testing.assert_array_equal(robot.commands[0], [1, 2, 3, 4, 5, 6, 0.5])


def test_fake_robot_raises_queued_errors_once() -> None:
    robot = conftest.FakeRobot()
    robot.read_error = OSError("bus down")
    robot.write_error = OSError("bus down")

    with pytest.raises(OSError):
        robot.get_observations()
    with pytest.raises(OSError):
        robot.command_joint_pos(np.zeros(7))

    robot.get_observations()
    robot.command_joint_pos(np.zeros(7))
    assert len(robot.commands) == 1


def test_fake_robot_exposes_disengage_methods_only_when_asked() -> None:
    plain = conftest.FakeRobot()
    gravity = conftest.FakeRobot(has_gravity_comp=True)
    torque = conftest.FakeRobot(has_zero_torque=True)

    assert not hasattr(plain, "enter_gravity_comp_idle")
    assert not hasattr(plain, "zero_torque_mode")
    assert not hasattr(gravity, "zero_torque_mode")
    assert not hasattr(torque, "enter_gravity_comp_idle")

    gravity.enter_gravity_comp_idle()
    torque.zero_torque_mode()
    assert gravity.gravity_comp_calls == 1
    assert torque.zero_torque_calls == 1


def test_fake_capture_source_yields_frames_then_repeats_the_last() -> None:
    first = conftest.make_bgr_frame(bgr=(1, 2, 3))
    second = conftest.make_bgr_frame(bgr=(4, 5, 6))
    source = conftest.FakeCaptureSource([first, second])

    reads = [source.read() for _ in range(3)]

    assert all(ok for ok, _ in reads)
    assert reads[0][1] is first
    assert reads[1][1] is second
    assert reads[2][1] is second
    assert source.read_count == 3


def test_fake_capture_source_default_frame_is_bgr_uint8() -> None:
    ok, frame = conftest.FakeCaptureSource().read()

    assert ok
    assert frame is not None
    assert frame.dtype == np.uint8
    assert frame.shape == conftest.FRAME_SHAPE
    # Blue in BGR: the 255 sits in channel 0 until someone converts to RGB.
    assert frame[0, 0].tolist() == [255, 0, 0]


def test_fake_capture_source_switches_to_failure_and_records_release() -> None:
    source = conftest.FakeCaptureSource()
    assert source.read()[0]

    source.fail()
    source.release()

    assert source.read() == (False, None)
    assert source.read() == (False, None)
    assert source.released


## A one-camera tick the fake client is handed before it is asked.
FRAMES = {"image_window.top": "anBlZw=="}


def ready_client(**kwargs: object) -> conftest.FakeClient:
    """A FakeClient that was handed one tick, so get_action() has an observation to send."""
    client = conftest.FakeClient(**kwargs)  # type: ignore[arg-type]
    client.set_observation(FRAMES, {"prompt": conftest.TASK}, timestamp=1.0)
    return client


def test_fake_client_returns_chunks_of_the_set_horizon() -> None:
    client = conftest.FakeClient(horizon=3)
    client.set_observation(FRAMES, {"state.arm_r.joint_pos": np.zeros(6, dtype=np.float32)}, timestamp=1.0)

    chunk = client.get_action()

    assert set(chunk) == {"q_r_chunk", "q_l_chunk"}
    assert np.shape(chunk["q_r_chunk"]) == (3, 7)
    assert np.shape(chunk["q_l_chunk"]) == (3, 7)
    # Nested lists of floats, the way JSON hands a chunk to the real client.
    assert isinstance(chunk["q_r_chunk"], list) and isinstance(chunk["q_r_chunk"][0], list)
    assert all(type(value) is float for value in chunk["q_r_chunk"][0])
    assert client.calls == 1
    # What the real client would send: the observation, and each frame key's frame as a one-frame window.
    assert len(client.observations) == 1
    assert set(client.observations[0]) == {"state.arm_r.joint_pos", "image_window.top"}
    assert client.observations[0]["image_window.top"] == [FRAMES["image_window.top"]]

    client.horizon = 5
    assert np.shape(client.get_action()["q_l_chunk"]) == (5, 7)


def test_fake_client_rows_are_sequential_across_chunks() -> None:
    client = ready_client(horizon=4)

    first = np.array(client.get_action()["q_r_chunk"])
    second = np.array(client.get_action()["q_r_chunk"])

    assert first[:, 0].tolist() == [0, 1, 2, 3]
    assert second[:, 0].tolist() == [4, 5, 6, 7]
    assert first[:, 6].tolist() == pytest.approx([0.0, 0.1, 0.2, 0.3])
    assert (first[:, 6] >= 0).all() and (first[:, 6] <= 1).all()


def test_fake_client_refuses_what_the_real_client_refuses() -> None:
    """No get_action() before the episode's first set_observation(), nor after a reset_episode() until the next;
    no frame key in the observation; no timestamp earlier than the episode's last."""
    client = conftest.FakeClient()
    with pytest.raises(dream.DreamObservationError, match="no observation since the episode started"):
        client.get_action()
    with pytest.raises(dream.DreamObservationError, match="also frame keys"):
        client.set_observation(FRAMES, dict(FRAMES))
    client.set_observation(FRAMES, {}, timestamp=2.0)
    with pytest.raises(dream.DreamObservationError, match="earlier than the previous"):
        client.set_observation(FRAMES, {}, timestamp=1.0)
    client.get_action()

    client.reset_episode()

    with pytest.raises(dream.DreamObservationError, match="no observation since the episode started"):
        client.get_action()
    client.set_observation(FRAMES, {}, timestamp=1.0)
    client.get_action()
    assert client.calls == 2 and client.resets == 1 and client.reset_after_calls == [1]
    assert [timestamp for _, _, timestamp in client.ticks] == [2.0, 1.0]


def test_fake_client_raises_once_then_recovers() -> None:
    client = ready_client(raise_once=dream.DreamTimeout("slow"))

    with pytest.raises(dream.DreamTimeout):
        client.get_action()
    chunk = client.get_action()

    assert np.shape(chunk["q_r_chunk"]) == (4, 7)
    assert client.calls == 2
    assert client.raise_once is None


def test_fake_client_blocks_on_gate_and_rejects_a_concurrent_call() -> None:
    gate = threading.Event()
    client = ready_client(gate=gate)
    results: list[object] = []

    def worker() -> None:
        results.append(client.get_action())

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    # `calls` moves only inside the lock, so once it does the worker is parked on the gate.
    deadline_s = time.monotonic() + JOIN_TIMEOUT_S
    while client.calls == 0 and time.monotonic() < deadline_s:
        time.sleep(0.001)
    assert client.calls == 1
    # A second call must fail, not queue.
    with pytest.raises(dream.DreamError):
        client.get_action()
    assert not results

    gate.set()
    thread.join(JOIN_TIMEOUT_S)

    assert not thread.is_alive()
    assert len(results) == 1
    # The rejected call never got past the lock: nothing counted, nothing recorded.
    assert client.calls == 1
    assert len(client.observations) == 1
    # Single-flight is per call, not per instance: the lock is free again once the worker returns.
    client.get_action()


def test_fake_client_records_close() -> None:
    client = conftest.FakeClient()
    assert not client.closed
    client.close()
    client.close()
    assert client.closed


def test_manual_clock_sleep_advances_time_and_yields() -> None:
    clock = conftest.ManualClock(now_s=10.0)
    seen: list[float] = []

    async def observer() -> None:
        seen.append(clock())

    async def scenario() -> None:
        task = asyncio.create_task(observer())
        await clock.sleep(0.5)
        await clock.sleep(-1.0)
        await task

    asyncio.run(scenario())

    assert clock() == 10.5
    assert clock.sleeps == [0.5, -1.0]
    # The observer ran during the first sleep, after now_s had already moved.
    assert seen == [10.5]


def test_manual_clock_is_settable() -> None:
    clock = conftest.ManualClock()
    clock.now_s = 3.25
    assert clock() == 3.25


def test_make_chunk_shapes_and_values() -> None:
    chunk = conftest.make_chunk(2, base=7.0)

    assert np.shape(chunk["q_r_chunk"]) == (2, 7)
    assert np.shape(chunk["q_l_chunk"]) == (2, 7)
    assert chunk["q_r_chunk"][1][:6] == [8.0] * 6
    assert [row[6] for row in chunk["q_r_chunk"]] == pytest.approx([0.7, 0.8])
    assert chunk["q_l_chunk"] is not chunk["q_r_chunk"]
    assert chunk["q_l_chunk"][0] is not chunk["q_r_chunk"][0]
