"""Tests of the model check (`dream.check_model`): a recorded episode through the real client to a fake model.

The episode is a one-armed session in the layout fine-tuning takes, 6 s at 30 rows a second, with one camera. The fake
model finds the recorded state nearest the one it is sent and answers the 48 actions recorded from there: a model that
learned these demos perfectly. A fault is planted in the recording or in the model, and the line meant to catch it
must print PROBLEM.
"""

from __future__ import annotations

import json
import math
import pathlib
import re
from collections.abc import Callable
from typing import Any

import av
import numpy as np
import pyarrow
import pyarrow.parquet
import pytest
from conftest import FakeDreamAPI

from dream import check_model

## The body: one arm and its gripper, one camera.
EMBODIMENT = {
    "state": {"right_arm": [0, 6], "right_gripper": [6, 7]},
    "action": {"right_arm": [0, 6], "right_gripper": [6, 7]},
    "video": {"overhead": "observation.images.overhead"},
}
## Rows a second and rows in the episode.
FPS, ROWS = 30, 180


def motion(t: np.ndarray) -> np.ndarray:
    """The demonstrated commands at times `t`: a slow sine per joint (radians), the gripper opening and closing."""
    joints = [0.3 * np.sin(2 * math.pi * 0.25 * t + j) for j in range(6)]
    return np.stack(joints + [0.5 + 0.5 * np.sin(2 * math.pi * 0.2 * t)], axis=1)


def write_session(root: pathlib.Path, scale: float) -> tuple[pathlib.Path, np.ndarray, np.ndarray]:
    """A LeRobot v2.1 session of one episode; the arm follows its commands 3 rows late.

    @param scale What the recording multiplies the joints by: 1, or degrees per radian for a recording in degrees.
    @return The session folder, and the true (radian) states and actions the fake model learned.
    """
    action = motion(np.arange(ROWS) / FPS)
    state = np.concatenate([np.repeat(action[:1], 3, axis=0), action[:-3]])
    units = np.array([scale] * 6 + [1.0])

    def column(values: np.ndarray) -> pyarrow.Array:
        return pyarrow.array(
            [row.tolist() for row in (values * units).astype(np.float32)], pyarrow.list_(pyarrow.float32())
        )

    (root / "data" / "chunk-000").mkdir(parents=True)
    pyarrow.parquet.write_table(
        pyarrow.table(
            {"observation.state": column(state), "action": column(action), "timestamp": np.arange(ROWS) / FPS}
        ),
        root / "data" / "chunk-000" / "episode_000000.parquet",
    )
    (root / "meta").mkdir()
    info = {
        "codebase_version": "v2.1",
        "fps": FPS,
        "chunks_size": 1000,
        "total_episodes": 1,
        "total_frames": ROWS,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {"observation.state": {"shape": [7]}, "action": {"shape": [7]}},
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    (root / "meta" / "episodes.jsonl").write_text(json.dumps({"episode_index": 0, "tasks": ["wave"], "length": ROWS}))
    video = root / "videos" / "chunk-000" / "observation.images.overhead" / "episode_000000.mp4"
    video.parent.mkdir(parents=True)
    with av.open(str(video), "w") as container:
        stream = container.add_stream("mpeg4", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for index in range(ROWS):
            image = np.full((48, 64, 3), index % 256, dtype=np.uint8)
            container.mux(stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")))
        container.mux(stream.encode())
    return root, state, action


def run(
    tmp_path: pathlib.Path,
    server: FakeDreamAPI,
    capsys: pytest.CaptureFixture[str],
    scale: float = 1.0,
    model: Callable[[np.ndarray], np.ndarray] = lambda rows: rows,
) -> list[str]:
    """Runs the check on a recorded session against the fake model and returns the lines it prints.

    @param scale The recording's joint unit per radian.
    @param model What the fake model does to the recorded rows it answers.
    """
    session, state, action = write_session(tmp_path / "session", scale)
    seen: list[dict[str, Any]] = []

    def answer(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        observation = body["observation"]
        seen.append(observation)
        sent = np.concatenate([observation["state.right_arm"], observation["state.right_gripper"]])
        row = int(np.argmin(np.sum((state - sent) ** 2, axis=1)))
        rows = model(action[np.minimum(row + np.arange(48), ROWS - 1)])
        status, reply = server.default_reply(body)
        reply["actions"] = {
            "per_key": {"action.right_arm": rows[:, :6].tolist(), "action.right_gripper": rows[:, 6:].tolist()}
        }
        return status, reply

    server.handler = answer
    config = tmp_path / "policy.json"
    config.write_text(json.dumps({"checkpoint": "dreamlabs/dream-1-beta", "embodiment": EMBODIMENT}))
    check_model.main([str(session), "--config", str(config), "--moments", "6"])
    assert len(seen) == 6 and all(len(o["video.overhead"]) == 9 for o in seen)  # each moment's window of frames
    out = capsys.readouterr().out
    return [line for line in out.splitlines() if line.startswith(("PASS", "PROBLEM", "SKIP"))]


def _status(lines: list[str]) -> dict[str, str]:
    return {check: line.split()[0] for line in lines for check in check_model.CHECKS if line[9:].startswith(check)}


def test_a_model_that_learned_the_demos_passes(tmp_path: pathlib.Path, server: FakeDreamAPI, capsys: Any) -> None:
    lines = run(tmp_path, server, capsys)
    assert set(_status(lines).values()) == {"PASS"}, lines


def test_a_recording_in_degrees_starts_away(tmp_path: pathlib.Path, server: FakeDreamAPI, capsys: Any) -> None:
    lines = run(tmp_path, server, capsys, scale=180 / math.pi)
    assert _status(lines)["start"] == "PROBLEM" and "right_arm" in lines[1], lines


def test_a_model_moving_against_the_demos_is_a_direction_problem(
    tmp_path: pathlib.Path, server: FakeDreamAPI, capsys: Any
) -> None:
    lines = run(tmp_path, server, capsys, model=lambda rows: 2 * rows[:1] - rows)
    status = _status(lines)
    assert status["direction"] == "PROBLEM" and status["start"] == "PASS", lines


def test_a_model_that_barely_moves_is_a_size_problem(
    tmp_path: pathlib.Path, server: FakeDreamAPI, capsys: Any
) -> None:
    lines = run(tmp_path, server, capsys, model=lambda rows: rows[:1] + 0.1 * (rows - rows[:1]))
    assert _status(lines)["size"] == "PROBLEM", lines


def test_a_refusal_stops_the_check(tmp_path: pathlib.Path, server: FakeDreamAPI, capsys: Any) -> None:
    server.fail_next("DA-1000", "unauthorized")
    session, _, _ = write_session(tmp_path / "session", 1.0)
    config = tmp_path / "policy.json"
    config.write_text(json.dumps({"checkpoint": "dreamlabs/dream-1-beta", "embodiment": EMBODIMENT}))
    assert check_model.main([str(session), "--config", str(config)]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert any(line.startswith("PROBLEM  answers") for line in lines), lines


@pytest.mark.parametrize(
    ("break_it", "said"),
    [
        (lambda info, root: info.pop("chunks_size"), "meta/info.json has no chunks_size; the check reads the LeRobot"),
        (lambda info, root: info["features"].pop("action"), "meta/info.json has no features.action;"),
        (lambda info, root: (root / "meta" / "episodes.jsonl").unlink(), "No such file or directory"),
        (lambda info, root: info.update(chunks_size=1), None),  # still readable: episode 0 is in chunk 0
        (
            lambda info, root: info.update(video_path="videos/{video_key}/episode_{episode_index:06d}.mp4"),
            "episode_000000.mp4: no such video (info.json's video_path",
        ),
        (
            lambda info, root: (root / "data" / "chunk-000" / "episode_000000.parquet").unlink(),
            "no data/chunk-000/episode_000000.parquet, where info.json's data_path puts episode 0;",
        ),
    ],
)
def test_a_recording_in_another_layout_is_named_before_it_is_read(
    break_it: Callable[[dict[str, Any], pathlib.Path], object], said: str | None, tmp_path: pathlib.Path
) -> None:
    """A session the check cannot read is one Unreadable naming what is missing (a recorder that writes no
    `chunks_size`, say), never a KeyError from inside the reader."""
    session, _, _ = write_session(tmp_path / "session", 1.0)
    info = json.loads((session / "meta" / "info.json").read_text())
    break_it(info, session)
    (session / "meta" / "info.json").write_text(json.dumps(info))
    if said is None:
        check_model.read_episode(session, 0, EMBODIMENT["video"])
        return
    with pytest.raises(check_model.Unreadable, match=re.escape(said)):
        check_model.read_episode(session, 0, EMBODIMENT["video"])
    with pytest.raises(check_model.Unreadable, match="lists no episode 1 with its tasks; give one with --episode"):
        check_model.readable(write_session(tmp_path / "other", 1.0)[0], 1)
