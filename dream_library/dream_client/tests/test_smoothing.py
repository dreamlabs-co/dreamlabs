"""Seam smoothing tests C-42..C-42e and C-42l: always on, no knob, and what it does.

The contract tests run the client against `FakeDreamAPI` with a handler that returns chosen chunks, so the blend
is checked on the dict `get_action` hands back and on the bytes the fake recorded; the unit tests at the end drive
`dream.smoothing` directly on synthetic chunks.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from conftest import FakeDreamAPI, make_observation

import dream
from dream import seam_join, smoothing
from dream.client import PolicyClient
from dream.errors import DreamConfigError, DreamObservationError, DreamSessionExpired

## Rows per chunk in these tests, as the YAM serve returns them.
ROWS = 48

## Rows the join spans after the clock row; every row from the clock row plus this on is the filtered chunk's own.
RAMP = seam_join.RAMP_ROWS

## Joint columns of a chunk; the seventh column is the gripper.
JOINTS = 6

## The row of the previous chunk the observations below put the arms on.
ANCHOR = 12


def line_chunk(start: float, stop: float, gripper: float) -> npt.NDArray[np.float64]:
    """A `[ROWS, 7]` chunk whose joint columns ramp from `start` to `stop` and whose gripper column is constant."""
    joints = np.repeat(np.linspace(start, stop, ROWS)[:, None], JOINTS, axis=1)
    return np.column_stack([joints, np.full(ROWS, gripper)])


def serve_chunks(server: FakeDreamAPI, chunk: npt.NDArray[np.float64]) -> None:
    """Makes the fake answer every request with `chunk` for both arms, sessions handled as by default."""

    def handler(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        status, reply = server.default_reply(body)
        if reply.get("actions") is not None:
            reply["actions"]["q_r_chunk"] = chunk.tolist()
            reply["actions"]["q_l_chunk"] = chunk.tolist()
        return status, reply

    server.handler = handler


def observation_on(chunk: npt.NDArray[np.float64], row: int) -> dict[str, Any]:
    """An observation whose measured joint poses are row `row` of `chunk`, for both arms."""
    observation = make_observation()
    pose = chunk[row, :JOINTS].astype(np.float32)
    observation["state.arm_r.joint_pos"] = pose
    observation["state.arm_l.joint_pos"] = pose.copy()
    return observation


@pytest.fixture
def long_config(config: dict[str, Any]) -> dict[str, Any]:
    """The shared config with `[ROWS, 7]` chunks declared."""
    for key in ("q_r_chunk", "q_l_chunk"):
        config["embodiment"]["action"][key]["shape"] = [ROWS, 7]
    return config


@pytest.fixture
def client(long_config: dict[str, Any]) -> Iterator[PolicyClient]:
    """A client against the fake, closed on teardown."""
    with dream._open_policy(long_config) as client:
        yield client


def test_c42_there_is_no_smoothing_knob(long_config: dict[str, Any]) -> None:
    """C-42 C-27a: a `smoothing` config key is an unknown key, so the join cannot be turned off or tuned."""
    long_config["smoothing"] = {}
    with pytest.raises(DreamConfigError, match="smoothing"):
        dream._open_policy(long_config)


def test_c42_the_blend_never_reaches_the_wire(
    server: FakeDreamAPI, client: PolicyClient, observation: dict[str, Any]
) -> None:
    """C-42 C-27: the body is still exactly the four contract fields; the join is local."""
    serve_chunks(server, line_chunk(0.0, 1.0, 0.0))
    client.get_action(observation)
    body = json.loads(server.requests[0].raw)
    assert sorted(body) == ["checkpoint", "observation", "reset", "session_id"]


def test_c42_the_first_chunk_of_an_episode_is_filtered_only(server: FakeDreamAPI, client: PolicyClient) -> None:
    """C-42: with no previous plan there is no motion to continue: the chunk is the filter's output alone."""
    first = line_chunk(0.0, 1.0, 0.0)
    serve_chunks(server, first)
    action = client.get_action(observation_on(first, 0))
    for key in ("q_r_chunk", "q_l_chunk"):
        assert isinstance(action[key], list)
        assert np.allclose(action[key], first, atol=1e-9)


def test_c42a_the_clock_row_is_the_rows_executed_during_the_round_trip() -> None:
    """C-42a: the clock row is the round trip in rows; an arm that ran out of rows is at rest on the plan's last."""
    previous = line_chunk(0.0, 1.0, 0.0)[:, :JOINTS]
    clock, rows, times = smoothing.old_motion(previous, ANCHOR, rows_in_flight=6.4)
    assert clock == 6
    # The last commanded row, then the three the old plan would command next.
    assert times.tolist() == [-1.0, 0.0, 1.0, 2.0]
    assert np.array_equal(rows, previous[ANCHOR + 5 : ANCHOR + 9])
    # Near the plan's end only the rows that exist are taken.
    clock, rows, times = smoothing.old_motion(previous, ROWS - 4, rows_in_flight=2.0)
    assert (clock, times.tolist()) == (2, [-1.0, 0.0, 1.0])
    assert np.array_equal(rows, previous[ROWS - 3 :])
    # A round trip longer than the rows left (and a loop that drained the chunk, the anchor on its last row): the
    # arm holds the last row, which is the whole old motion.
    for anchor, in_flight in ((ROWS - 4, 9.0), (ROWS - 1, 0.0)):
        clock, rows, times = smoothing.old_motion(previous, anchor, in_flight)
        assert clock == ROWS - 1 - anchor
        assert np.array_equal(rows, previous[-1:])


def test_c42b_the_join_continues_the_previous_plan() -> None:
    """C-42b: from the clock row the chunk leaves the old motion with its velocity and is the plan's own after it."""
    first, second = line_chunk(0.0, 1.0, 0.0), line_chunk(0.3, 2.0, 1.0)
    smoother = smoothing.SeamSmoother(action_hz=20.0)
    smoother._previous["q_r_chunk"] = first
    action = smoother.smooth(observation_on(first, ANCHOR), {"q_r_chunk": second.tolist()}, 0.0, 0.3)
    joined = np.asarray(action["q_r_chunk"])
    clock = 6  # 0.3 s at 20 rows per second
    assert joined.shape == (ROWS, 7)
    # Rows before the clock row are the previous plan's; the clock row is where it goes next, not the new chunk's.
    assert np.allclose(joined[:clock, :JOINTS], first[ANCHOR : ANCHOR + clock, :JOINTS], atol=1e-12)
    assert np.allclose(joined[clock, :JOINTS], first[ANCHOR + clock, :JOINTS], atol=1e-12)
    # No velocity step at the seam: the first step is the old plan's.
    old_step = first[1, 0] - first[0, 0]
    assert abs((joined[clock + 1, 0] - joined[clock, 0]) - old_step) < 0.25 * old_step
    # From the end of the ramp on, and for the gripper from the clock row on, the rows are the filtered chunk's.
    expected = smoothing.filter_from(second, clock)
    assert np.array_equal(joined[clock + RAMP :], expected[clock + RAMP :])
    assert np.array_equal(joined[clock:, JOINTS], expected[clock:, JOINTS])
    assert np.abs(np.diff(joined[clock:, 0], n=2)).max() < 0.01


def test_c42b_a_models_own_wire_is_joined_on_its_own_joints(server: FakeDreamAPI, config: dict[str, Any]) -> None:
    """C-42b on the model's own wire (an embodiment config, what a YAM station sends): the join
    anchors on `state.<side>_arm`, so the second chunk continues the first exactly as it does on the station wire."""
    yam_body = {
        "state": {"right_arm": [0, 6], "right_gripper": [6, 7], "left_arm": [7, 13], "left_gripper": [13, 14]},
        "action": {"right_arm": [0, 6], "right_gripper": [6, 7], "left_arm": [7, 13], "left_gripper": [13, 14]},
        "video": {"overhead": "observation.images.overhead", "wrist_l": "observation.images.wrist_l"},
    }
    first, second = line_chunk(0.0, 1.0, 0.0), line_chunk(0.3, 2.0, 1.0)
    chunks = [first, second]

    def handler(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        status, reply = server.default_reply(body)
        if reply.get("actions") is not None:
            chunk = chunks.pop(0)
            reply["actions"]["q_r_chunk"] = chunk.tolist()
            reply["actions"]["q_l_chunk"] = chunk.tolist()
            reply["actions"]["per_key"] = {
                "action.right_arm": chunk[:, :JOINTS].tolist(),
                "action.right_gripper": chunk[:, JOINTS:].tolist(),
                "action.left_arm": chunk[:, :JOINTS].tolist(),
                "action.left_gripper": chunk[:, JOINTS:].tolist(),
            }
        return status, reply

    server.handler = handler
    frame = make_observation()["image_window.top"]

    def observation(row: int) -> dict[str, Any]:
        pose = first[row, :JOINTS].astype(np.float32)
        return {
            "state.right_arm": pose,
            "state.right_gripper": np.ones(1, dtype=np.float32),
            "state.left_arm": pose,
            "state.left_gripper": np.ones(1, dtype=np.float32),
            "video.overhead": frame,
            "video.wrist_l": frame,
            "prompt": "pick up the cup",
        }

    with dream._open_policy({**config, "embodiment": yam_body, "action_hz": 20.0}) as client:
        assert client._smoother._pose_keys == {"q_r_chunk": "state.right_arm", "q_l_chunk": "state.left_arm"}
        client.get_action(observation(0))
        joined = np.asarray(client.get_action(observation(ANCHOR))["q_r_chunk"])
    smoother = smoothing.SeamSmoother(action_hz=20.0)
    smoother._previous["q_r_chunk"] = smoothing.filter_from(first, 0)
    rtt_s = client.last_timing["rtt_ms"] / 1000.0 if client.last_timing else 0.0
    expected = smoother.smooth(observation_on(first, ANCHOR), {"q_r_chunk": second.tolist()}, 0.0, rtt_s)
    assert np.allclose(joined, np.asarray(expected["q_r_chunk"]), atol=1e-9), "joined, not merely filtered"
    assert not np.allclose(joined, smoothing.filter_from(second, 0), atol=1e-6)


def test_c42b_an_arm_holding_at_the_end_of_the_plan_starts_from_rest() -> None:
    """C-42b: an arm on the previous plan's last row (sync pacing) is joined from that row at rest, from row 0."""
    first, second = line_chunk(0.0, 1.0, 0.0), line_chunk(0.3, 2.0, 1.0)
    smoother = smoothing.SeamSmoother(action_hz=20.0)
    smoother._previous["q_r_chunk"] = first
    action = smoother.smooth(observation_on(first, ROWS - 1), {"q_r_chunk": second.tolist()}, 0.0, 1.2)
    joined = np.asarray(action["q_r_chunk"])
    assert np.allclose(joined[0, :JOINTS], first[-1, :JOINTS], atol=1e-12)
    assert abs(joined[1, 0] - joined[0, 0]) < 0.25 * (second[1, 0] - second[0, 0])


def test_a_fence_runs_on_the_joined_rows_and_is_the_plan_the_next_chunk_joins() -> None:
    """The reach area's fence (`dream.client_ik.ClientIK.fence`) gets each arm's joined chunk, after the join that
    carries the old motion on; what it gives is the reply and the previous chunk the next join continues."""
    first, second = line_chunk(0.0, 1.0, 0.0), line_chunk(0.3, 2.0, 1.0)
    seen: list[tuple[str, npt.NDArray[np.float64]]] = []

    def hold_from_10(key: str, rows: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        seen.append((key, rows.copy()))
        held = rows.copy()
        held[10:, :JOINTS] = rows[9, :JOINTS]
        return held

    plain, fenced = smoothing.SeamSmoother(action_hz=20.0), smoothing.SeamSmoother(action_hz=20.0, fence=hold_from_10)
    replies = []
    for smoother in (plain, fenced):
        smoother._previous["q_r_chunk"] = first
        action = smoother.smooth(observation_on(first, ANCHOR), {"q_r_chunk": second.tolist()}, 0.0, 0.3)
        replies.append(np.asarray(action["q_r_chunk"]))
    joined, held = replies
    assert [key for key, _ in seen] == ["q_r_chunk"] and np.array_equal(seen[0][1], joined)
    assert np.array_equal(held[:10], joined[:10]) and np.array_equal(
        held[10:, :JOINTS], np.repeat(joined[9:10, :JOINTS], ROWS - 10, axis=0)
    )
    assert np.array_equal(fenced._previous["q_r_chunk"], held)


def test_c42l_the_commanded_joints_anchor_the_join_whatever_the_arm_trails_by() -> None:
    """C-42l: with `commanded` the chunk takes over at the row after the joints last commanded, though the measured
    pose trails them by three rows; without, it takes over three rows back along the plan."""
    first, second = line_chunk(0.0, 1.0, 0.0), line_chunk(0.3, 2.0, 1.0)
    lagging = observation_on(first, ANCHOR - 3)
    clock = 6  # 0.3 s at 20 rows per second
    smoother = smoothing.SeamSmoother(action_hz=20.0)
    smoother._previous["q_r_chunk"] = first
    commanded = {"q_r_chunk": first[ANCHOR - 1, :JOINTS]}
    action = smoother.smooth(lagging, {"q_r_chunk": second.tolist()}, 0.0, 0.3, commanded)
    joined = np.asarray(action["q_r_chunk"])
    assert np.allclose(joined[: clock + 1, :JOINTS], first[ANCHOR : ANCHOR + clock + 1, :JOINTS], atol=1e-12)
    smoother._previous["q_r_chunk"] = first
    behind = np.asarray(smoother.smooth(lagging, {"q_r_chunk": second.tolist()}, 0.0, 0.3)["q_r_chunk"])
    assert np.allclose(behind[clock, :JOINTS], first[ANCHOR - 3 + clock, :JOINTS], atol=1e-12)


def test_c42l_anchor_row_holds_on_the_last_row_and_drops_an_arm_far_from_its_command() -> None:
    """C-42l: joints on the last row of a plan that ends at rest anchor there (the sync hold), while a hold mid-plan
    anchors at its first row; a whole written row is read for its joints; a measured pose far from the joints
    commanded is off the plan."""
    previous = line_chunk(0.0, 1.0, 0.0)[:, :JOINTS]
    at_rest = np.concatenate([previous[:40], np.repeat(previous[40:41], ROWS - 40, axis=0)])
    assert smoothing.nearest_row(at_rest, at_rest[-1], smoothing.MAX_OFFSET_RAD) == ROWS - 1
    assert smoothing.anchor_row(at_rest, at_rest[-1], at_rest[-1]) == ROWS - 1
    held = np.concatenate([previous[:10], np.repeat(previous[10:11], 10, axis=0), previous[11 : ROWS - 9]])
    assert smoothing.nearest_row(held, held[15], smoothing.MAX_OFFSET_RAD) == 10
    written = np.append(previous[ANCHOR - 1], 0.5)  # the joints, then the gripper
    assert smoothing.anchor_row(previous, previous[ANCHOR - 3], written) == ANCHOR
    assert smoothing.anchor_row(previous, previous[ANCHOR - 3], previous[ANCHOR - 1]) == ANCHOR
    assert smoothing.anchor_row(previous, previous[ANCHOR - 3], None) == ANCHOR - 3
    assert smoothing.anchor_row(previous, previous[ANCHOR] + 1.0, previous[ANCHOR - 1]) is None


@pytest.mark.parametrize(
    ("commanded", "message"),
    [
        ({"q_x_chunk": np.zeros(JOINTS)}, "q_x_chunk"),
        ({"q_r_chunk": np.zeros((2, JOINTS))}, "q_r_chunk"),
        ({"q_l_chunk": np.zeros(3)}, "q_l_chunk"),
    ],
)
def test_c42l_a_bad_commanded_is_refused_before_anything_is_sent(
    server: FakeDreamAPI, client: PolicyClient, commanded: dict[str, Any], message: str
) -> None:
    """C-42l: a `commanded` key that is not a chunk key, or a value that is not six (or seven) numbers, is refused."""
    with pytest.raises(DreamObservationError, match=message):
        client.get_action(observation_on(line_chunk(0.0, 1.0, 0.0), 0), commanded=commanded)
    assert not server.requests


@pytest.mark.parametrize("pose", [np.zeros((2, JOINTS)), 0.5], ids=["history", "scalar"])
def test_c42l_with_strict_off_a_pose_that_is_not_flat_takes_a_commanded_of_six_as_before(
    server: FakeDreamAPI, long_config: dict[str, Any], pose: Any
) -> None:
    """C-42l: with strict off, an arm's pose that is not one flat run (a history, a scalar) gives the arm no joint
    count, so a `commanded` of six is sent as given, not refused against 12 or 1."""
    long_config["strict"] = False
    first = line_chunk(0.0, 1.0, 0.0)
    serve_chunks(server, first)
    observation = observation_on(first, 0)
    observation["state.arm_r.joint_pos"] = pose
    with dream._open_policy(long_config) as client:
        client.get_action(observation, commanded={"q_r_chunk": first[0, :JOINTS]})
    assert len(server.requests) == 1


def test_c42c_an_arm_off_the_plan_is_not_joined(server: FakeDreamAPI, client: PolicyClient) -> None:
    """C-42c: a pose farther than the offset limit from every row of the previous chunk gets the filter but no join."""
    first, second = line_chunk(0.0, 1.0, 0.0), line_chunk(0.3, 2.0, 1.0)
    serve_chunks(server, first)
    client.get_action(observation_on(first, 0))

    serve_chunks(server, second)
    off_plan = make_observation()
    off_plan["state.arm_r.joint_pos"] = np.full(JOINTS, 5.0, dtype=np.float32)
    off_plan["state.arm_l.joint_pos"] = first[ANCHOR, :JOINTS].astype(np.float32)
    action = client.get_action(off_plan)
    assert np.allclose(action["q_r_chunk"], second, atol=1e-9)
    assert not np.allclose(action["q_l_chunk"], second, atol=0.01)


def test_c42d_reset_and_session_loss_forget_the_previous_plan(server: FakeDreamAPI, client: PolicyClient) -> None:
    """C-42d: after `reset_episode()` and after a session replaced mid-episode, the next chunk is filtered only."""
    first, second = line_chunk(0.0, 1.0, 0.0), line_chunk(0.3, 2.0, 1.0)
    observation = observation_on(first, ANCHOR)
    expected = {"q_r_chunk": smoothing.filter_from(second, 0).tolist()}
    serve_chunks(server, first)
    client.get_action(observation_on(first, 0))
    client.reset_episode()
    serve_chunks(server, second)
    assert client.get_action(observation)["q_r_chunk"] == expected["q_r_chunk"]

    serve_chunks(server, first)
    client.get_action(observation_on(first, 0))
    server.restart()
    with pytest.raises(DreamSessionExpired):
        client.get_action(observation)
    serve_chunks(server, second)
    assert client.get_action(observation)["q_r_chunk"] == expected["q_r_chunk"]


def test_c42e_the_filter_runs_at_twice_the_row_rate_from_the_clock_row() -> None:
    """C-42e: from the clock row every column is the upsampled filter's, with earlier rows as context; jitter goes."""
    rng = np.random.default_rng(1)
    jittery = line_chunk(0.3, 2.0, 1.0) + rng.normal(scale=0.02, size=(ROWS, 7))
    clock = 9
    filtered = smoothing.filter_from(jittery, clock)
    segment = jittery[clock - smoothing.FILTER_CONTEXT_ROWS :]
    fine = np.arange(2 * segment.shape[0] - 1) / 2.0
    upsampled = np.column_stack([np.interp(fine, np.arange(segment.shape[0]), column) for column in segment.T])
    expected = smoothing.savgol_filter(upsampled, smoothing.SAVGOL_WINDOW, smoothing.SAVGOL_POLYORDER)[::2]
    assert np.allclose(filtered[clock:], expected[smoothing.FILTER_CONTEXT_ROWS :], atol=1e-12)
    assert np.array_equal(filtered[:clock], jittery[:clock])
    for column in (0, JOINTS):  # a joint and the gripper: every column is filtered
        raw = np.abs(np.diff(jittery[clock:, column], n=2)).sum()
        assert np.abs(np.diff(filtered[clock:, column], n=2)).sum() < raw / 2


def test_c42e_savgol_passes_a_cubic_and_a_short_chunk_unchanged() -> None:
    """C-42e: an order-3 fit reproduces a cubic exactly; fewer rows than the window come back untouched."""
    t = np.arange(ROWS, dtype=np.float64)[:, None]
    cubic = 0.1 * t**3 - t**2 + 3.0 * t + np.arange(7)
    assert np.allclose(smoothing.savgol_filter(cubic, 11, 3), cubic, atol=1e-8)
    short = cubic[:5]
    assert smoothing.savgol_filter(short, 11, 3) is short
    line = line_chunk(0.0, 1.0, 0.0)
    assert np.allclose(smoothing.filter_from(line, 0), line, atol=1e-12)


def test_c42c_nearest_row_is_none_beyond_the_offset() -> None:
    """C-42c: the nearest row is returned only within the offset limit."""
    chunk = line_chunk(0.0, 1.0, 0.0)[:, :JOINTS]
    assert smoothing.nearest_row(chunk, chunk[ANCHOR] + 0.001, smoothing.MAX_OFFSET_RAD) == ANCHOR
    assert smoothing.nearest_row(chunk, chunk[ANCHOR] + 1.0, smoothing.MAX_OFFSET_RAD) is None
