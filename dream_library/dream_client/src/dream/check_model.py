"""`python -m dream.check_model`: your recorded demos through the model, compared with what you did, nothing moving.

When the robot does the wrong thing, either what the model is given or the loop is at fault; this check leaves the
robot out. It reads one session of your recordings in the layout fine-tuning takes (LeRobot v2.1, as
#dream.datasets checks it), and at a few moments spread over one episode hands the client what your loop would at that
moment: each camera's frames of the window before it, the state then and the episode's task. The model's chunk is set
against the actions you recorded over the same 48 rows, and one PASS or PROBLEM line is printed per check (#judge):

- answers: the model answered every call;
- start: each part's first row is near what you commanded at that moment (a state in other units, in another part
  order or with an offset sends the model somewhere else);
- direction: where you moved a part, the model moves it the same way;
- size: and about as far, a third to three times.

    python -m dream.check_model SESSION --config policy.json [--episode 0] [--moments 8]

`policy.json` is the dict your loop passes to `policy()`, with its `embodiment`: the spans that cut the dataset's
`observation.state` and `action` into parts, and the camera each video is. Each moment is one get_action() on that
model, billed as any call. It reads the videos with PyAV and OpenCV and the tables with pyarrow (`pip install --upgrade
dreamlabs` carries them). The lines judge the inputs and the model together: a model that has never seen your robot
may move less like your demos than a fine-tune of them.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import json
import math
import pathlib
import re
import sys
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from dream import _window
from dream.check_loop import HORIZON, PLAN_HZ, Line, _kind
from dream.errors import DreamError, DreamServerError

__all__ = ["CHECKS", "Episode", "Unreadable", "judge", "main", "read_episode", "readable"]

## The command, as its lines name it.
COMMAND = "python -m dream.check_model"
## The checks, in the order their lines print.
CHECKS = ("answers", "start", "direction", "size")
## The frame each moment sends, width x height, and its JPEG quality.
FRAME_SIZE, JPEG_QUALITY = (640, 320), 85
## How far each part's first row may start from what you commanded then: radians, metres, a gripper's own unit.
START_LIMIT = {"joint": math.radians(15), "xyz": 0.05, "grip": 0.5}
## How far you must have moved a part over the chunk for its direction and size to count.
MOVED = {"joint": math.radians(3), "xyz": 0.01, "grip": 0.3}
## The model's move against yours, as a ratio of their lengths, that `size` passes.
SIZE_RANGE = (1 / 3, 3.0)
## Calls refused while the model starts (`retryable`) are made again for this long.
RETRY_S = 300.0
## Fields a LeRobot path template may name, each `{name}` or `{name:0Nd}`.
PATH_FIELD_RE = re.compile(r"\{(episode_chunk|episode_index|video_key)(?::0(\d+)d)?\}")


@dataclasses.dataclass
class Episode:
    """One recorded episode, as the check reads it.

    Attributes:
    - fps: Rows a second.
    - task: The episode's task, the prompt.
    - state, action: `observation.state` and `action`, `[rows, width]`.
    - videos: Per camera role of the config, its episode's mp4.
    """

    fps: float
    task: str
    state: np.ndarray
    action: np.ndarray
    videos: dict[str, pathlib.Path]


class Unreadable(ValueError):
    """A session the check cannot read (#readable, #read_episode); `str(error)` is one line naming what is missing."""


def readable(folder: pathlib.Path, episode: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """A session folder's `meta/info.json` and its episode's line of `meta/episodes.jsonl`, checked for #read_episode.

    They hold what #read_episode reads: the info fields and features an upload needs (`shared.datasets.validation`),
    the episode with its task, and its parquet where `info.json` says.

    @param folder The session folder.
    @param episode Its `episode_index`.
    @return `(info, the episode's line)`.
    @throws Unreadable What is missing, and the layout read (LeRobot v2.1, as an upload takes it).
    """
    from shared.datasets import validation

    layout = f"the check reads the LeRobot {validation.CODEBASE_VERSION} layout an upload takes"
    try:
        info = json.loads((folder / validation.INFO_FILE).read_text())
    except (OSError, ValueError) as error:
        raise Unreadable(f"{error}; {layout}") from None
    missing = [field for field in validation.REQUIRED_INFO_FIELDS if field not in info]
    if "features" in info:
        missing += [f"features.{name}" for name in validation.REQUIRED_FEATURES if name not in info["features"]]
    if missing:
        raise Unreadable(f"{validation.INFO_FILE} has no {', '.join(missing)}; {layout}")
    try:
        lines = (folder / validation.EPISODES_FILE).read_text().splitlines()
        rows = [json.loads(line) for line in lines if line.strip()]
    except (OSError, ValueError) as error:
        raise Unreadable(f"{error}; {layout}") from None
    meta = next((row for row in rows if row.get("episode_index") == episode and row.get("tasks")), None)
    if meta is None:
        raise Unreadable(
            f"{validation.EPISODES_FILE} lists no episode {episode} with its tasks; give one with --episode"
        )
    data = _fill(info["data_path"], _fields(info, episode))
    if not (folder / data).is_file():
        raise Unreadable(f"no {data}, where info.json's data_path puts episode {episode}; {layout}")
    return info, meta


def _fields(info: Mapping[str, Any], episode: int) -> dict[str, int]:
    """The path fields of one episode: its chunk and its index."""
    return {"episode_chunk": episode // int(info["chunks_size"]), "episode_index": episode}


def read_episode(folder: pathlib.Path, episode: int, video: Mapping[str, str]) -> Episode:
    """One episode of a session folder.

    @param folder The session folder: `meta/info.json`, `meta/episodes.jsonl`, the parquets and the videos.
    @param episode Its `episode_index`.
    @param video Per camera role, the dataset's video key (the config's `embodiment.video`).
    @return The episode.
    @throws Unreadable The folder is not one #readable reads, or a video is not where `info.json` says.
    """
    import pyarrow.parquet  # type: ignore[import-untyped]  # the `datasets` extra

    info, meta = readable(folder, episode)
    fields = _fields(info, episode)
    table = pyarrow.parquet.read_table(
        folder / _fill(info["data_path"], fields), columns=["observation.state", "action"]
    )
    videos = {role: folder / _fill(info["video_path"], {**fields, "video_key": key}) for role, key in video.items()}
    for path in videos.values():
        if not path.is_file():
            raise Unreadable(f"{path}: no such video (info.json's video_path and the config's embodiment.video)")
    return Episode(
        fps=float(info["fps"]),
        task=str(meta["tasks"][0]),
        state=np.array(table.column("observation.state").to_pylist(), dtype=float),
        action=np.array(table.column("action").to_pylist(), dtype=float),
        videos=videos,
    )


def _fill(template: str, fields: Mapping[str, Any]) -> str:
    """A LeRobot path template with its fields filled; nothing else in it is read as a format."""
    return PATH_FIELD_RE.sub(lambda m: str(fields[m.group(1)]).zfill(int(m.group(2) or 0)), template)


def _frames(path: pathlib.Path, indices: set[int]) -> dict[int, str]:
    """The frames at `indices` of one video, each as the check sends it: 640x320 JPEG, base64."""
    import av
    import cv2

    out, last = {}, max(indices)
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in indices:
                small = cv2.resize(frame.to_ndarray(format="bgr24"), FRAME_SIZE, interpolation=cv2.INTER_AREA)
                ok, jpeg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if not ok:
                    raise RuntimeError(f"{path}: frame {index} did not encode as JPEG")
                out[index] = base64.b64encode(jpeg.tobytes()).decode("ascii")
            if index >= last:
                break
    missing = indices - set(out)
    if missing:
        raise ValueError(f"{path}: {len(out)} frames, none at {min(missing)}; the video is shorter than its episode")
    return out


def moments(rows: int, fps: float, count: int) -> list[int]:
    """The rows the check asks at: `count` spread over the episode, each with 48 recorded model rows after it.

    @param rows The episode's rows.
    @param fps Its rows a second.
    @param count How many moments to ask at; fewer when the episode is short.
    @return The rows, in order.
    @throws ValueError The episode is shorter than one chunk.
    """
    last = rows - 1 - math.ceil((HORIZON - 1) * fps / PLAN_HZ)
    if last < 0:
        raise ValueError(f"the episode has {rows} rows, fewer than one chunk of {HORIZON} at {fps:g} a second")
    return sorted({round(x) for x in np.linspace(min(last, round(0.1 * rows)), last, count)})


def ask(
    client: Any,
    episode: Episode,
    parts: Mapping[str, Sequence[int]],
    row: int,
    ticks: Sequence[float],
    frames: Mapping[str, Mapping[int, str]],
) -> dict[str, Any]:
    """The model's action at one moment, from a new episode on the client.

    @param client The DreamClient, its policy open.
    @param episode The recording.
    @param parts Per state part, its span of `observation.state`.
    @param row The moment's row.
    @param ticks The window's times before the moment, seconds, oldest first, the moment last.
    @param frames Per camera role, the frames by row.
    @return The action dict.
    """
    client.reset_episode()
    base = time.monotonic() - row / episode.fps
    for tick in ticks:
        index = max(0, round(tick * episode.fps))
        observation: dict[str, Any] = {"prompt": episode.task}
        for part, (lo, hi) in parts.items():
            observation[f"state.{part}"] = episode.state[index, lo:hi].astype(np.float32)
        client.set_observation(
            {f"video.{role}": by_row[index] for role, by_row in frames.items()}, observation, timestamp=base + tick
        )
    give_up = time.monotonic() + RETRY_S
    while True:
        try:
            action: dict[str, Any] = client.get_action()
            return action
        except DreamServerError as error:
            if not error.retryable or time.monotonic() > give_up:
                raise
            time.sleep(error.retry_after_s or 2.0)


def predicted(action: Mapping[str, Any], parts: Sequence[str]) -> dict[str, np.ndarray]:
    """Per action part the chunk carries, its rows.

    They are `per_key`'s `action.<part>`, or a YAM's `q_r_chunk` and `q_l_chunk` cut into arm and gripper.

    @param action The action dict get_action() returned.
    @param parts The config's action parts.
    @return Per part the chunk carries, `[48, width]`.
    """
    out = {}
    per_key = action["per_key"] if "per_key" in action else {}
    for part in parts:
        if f"action.{part}" in per_key:
            out[part] = np.asarray(per_key[f"action.{part}"], dtype=float)
    for side, key in (("right", "q_r_chunk"), ("left", "q_l_chunk")):
        if key in action:
            rows = np.asarray(action[key], dtype=float)
            out.setdefault(f"{side}_arm", rows[:, :-1])
            out.setdefault(f"{side}_gripper", rows[:, -1:])
    return {part: rows for part, rows in out.items() if part in parts}


def recorded(episode: Episode, span: Sequence[int], row: int) -> np.ndarray:
    """What you commanded over the model's 48 rows from `row`, at the model's 30 rows a second.

    @param episode The recording.
    @param span The part's span of `action`.
    @param row The moment's row.
    @return The recorded commands, `[48, width]`.
    """
    at = row + np.arange(HORIZON) * episode.fps / PLAN_HZ
    lo, hi = span
    return np.stack([np.interp(at, np.arange(len(episode.action)), episode.action[:, c]) for c in range(lo, hi)], 1)


def judge(
    chunks: Sequence[tuple[dict[str, np.ndarray], dict[str, np.ndarray]]], asked: int, error: str | None
) -> list[Line]:
    """One line per check, from each moment's predicted and recorded rows per part.

    @param chunks Per moment answered, `(predicted, recorded)`, each per part `[48, width]`.
    @param asked How many moments were asked.
    @param error Why the model stopped answering, or None.
    @return One #Line per check, in #CHECKS order.
    """
    if error is not None:
        return [Line("PROBLEM", "answers", f"{len(chunks)} of {asked} calls answered, then: {error}")] + [
            Line("SKIP", name, "not run: the model did not answer") for name in CHECKS[1:]
        ]
    lines = [Line("PASS", "answers", f"{asked} of {asked} calls answered")]
    parts = sorted({part for got, _ in chunks for part in got if _kind(part) != "rot"})
    if not parts:
        return lines + [
            Line("PROBLEM", name, "the chunks carry none of the config's action parts") for name in CHECKS[1:]
        ]
    start, cos, size = {}, {}, {}
    for part in parts:
        kind = _kind(part)
        pairs = [(got[part], want[part]) for got, want in chunks if part in got]
        start[part] = float(np.median([np.max(np.abs(p[0] - r[0])) for p, r in pairs]))
        moved = [(p - p[0], r - r[0]) for p, r in pairs if np.max(np.ptp(r, axis=0)) >= MOVED[kind]]
        if moved:
            cos[part] = float(
                np.median([np.sum(p * r) / (np.linalg.norm(p) * np.linalg.norm(r) + 1e-12) for p, r in moved])
            )
            size[part] = float(np.median([np.linalg.norm(p) / np.linalg.norm(r) for p, r in moved]))
    far = [f"{part} {_show(part, start[part])}" for part in parts if start[part] > START_LIMIT[_kind(part)]]
    worst = max(parts, key=lambda part: start[part] / START_LIMIT[_kind(part)])
    lines.append(
        Line(
            "PROBLEM",
            "start",
            "starts away from what you commanded: " + ", ".join(far) + " (a state in other units, "
            "another part order or an offset)",
        )
        if far
        else Line("PASS", "start", f"every part starts within {_show(worst, start[worst])} of what you commanded")
    )
    if not cos:
        return lines + [Line("SKIP", name, "you held every part still at the moments asked") for name in CHECKS[2:]]
    against = [f"{part} ({cos[part]:.2f})" for part in cos if cos[part] <= 0]
    lines.append(
        Line("PROBLEM", "direction", "moves against what you did: " + ", ".join(against))
        if against
        else Line(
            "PASS",
            "direction",
            f"each part moves the way you did (agreement {min(cos.values()):.2f} to "
            f"{max(cos.values()):.2f}, 1 is the same way)",
        )
    )
    off = [f"{part} x{size[part]:.2f}" for part in size if not SIZE_RANGE[0] <= size[part] <= SIZE_RANGE[1]]
    lines.append(
        Line("PROBLEM", "size", "moves much more or less than you did: " + ", ".join(off))
        if off
        else Line(
            "PASS", "size", f"each part moves x{min(size.values()):.2f} to x{max(size.values()):.2f} as far as you did"
        )
    )
    return lines


def _show(part: str, value: float) -> str:
    """A size in the part's unit: degrees, cm, or the gripper's own."""
    kind = _kind(part)
    return (
        f"{math.degrees(value):.1f} degrees"
        if kind == "joint"
        else f"{value * 100:.1f} cm"
        if kind == "xyz"
        else (f"{value:.2f}")
    )


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m dream.check_model`: asks the model at a few moments of one recorded episode, then prints the lines.

    @param argv The arguments; `sys.argv[1:]` when None.
    @return 0 when every check passes, 1 when one is a PROBLEM, 2 for a bad argument or dataset.
    """
    import dream

    parser = argparse.ArgumentParser(prog=COMMAND, description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "session", type=pathlib.Path, help="a session folder of your recordings (meta/, data/, videos/)"
    )
    parser.add_argument(
        "--config", type=pathlib.Path, required=True, help="the policy() config, as JSON, with embodiment"
    )
    parser.add_argument("--episode", type=int, default=0, help="the episode_index to ask about (0)")
    parser.add_argument("--moments", type=int, default=8, help="how many moments to ask the model about (8)")
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text())
        embodiment = config["embodiment"]
        episode = read_episode(args.session, args.episode, embodiment["video"])
        rows = moments(len(episode.state), episode.fps, args.moments)
        # The window the client keeps (its own defaults when the config sets none), ending at each moment.
        window = int(config.get("obs_window", _window.OBS_WINDOW))
        step = 1 / float(config.get("obs_fps", _window.OBS_FPS))
        ticks = {row: [row / episode.fps - k * step for k in range(window - 1, -1, -1)] for row in rows}
        needed = {max(0, round(tick * episode.fps)) for times in ticks.values() for tick in times}
        print(
            f"{COMMAND}: {len(rows)} moments of episode {args.episode} ({episode.task!r}), {config['checkpoint']}",
            flush=True,
        )
        frames = {role: _frames(path, needed) for role, path in episode.videos.items()}
    except ImportError as error:
        print(
            f"{COMMAND}: {error}; PyAV, OpenCV and pyarrow come with `pip install --upgrade dreamlabs`",
            file=sys.stderr,
        )
        return 2
    except (OSError, KeyError, ValueError) as error:
        print(
            f"{COMMAND}: missing key {error}" if isinstance(error, KeyError) else f"{COMMAND}: {error}",
            file=sys.stderr,
        )
        return 2
    chunks: list[tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = []
    stopped = None
    with dream.DreamClient() as client:
        client.policy(config)
        for row in rows:
            try:
                action = ask(client, episode, embodiment["state"], row, ticks[row], frames)
            except DreamError as failure:
                stopped = str(failure)
                break
            got = predicted(action, list(embodiment["action"]))
            chunks.append((got, {part: recorded(episode, embodiment["action"][part], row) for part in got}))
    lines = judge(chunks, len(rows), stopped)
    print("")
    for line in lines:
        print(line)
    return 1 if any(line.status == "PROBLEM" for line in lines) else 0


if __name__ == "__main__":
    sys.exit(main())
