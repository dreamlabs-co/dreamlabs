"""A saved rollout on disk: finishing it into a LeRobot v2.1 dataset, and recovering it.

An episode folder is a LeRobot v2.1 dataset of one episode (`meta/`, `data/`, `videos/`) plus `rollout.json` (prompt,
checkpoint, robot, status), `inference.jsonl` and `chunks/reply_N.npz` (the model's raw replies). While a run goes the
recorder (`robots.rollouts`) writes crash-safe raw files: `ticks.f64` (`ticks_wide.f64` unless both arms have six
joints, #journal_file()), one float64 row per commanded tick, and one fragmented MP4 per camera. #finish() turns them
into the parquet, the final MP4s (a remux, no re-encode) and `meta/`, and deletes the raw ones. A killed run keeps them
with an `INCOMPLETE.txt` holding its pid; a later run finishes it. `timestamp` is `frame_index / fps`;
`rollout.t_monotonic_s` is the loop's clock. `rollout.json` carries each arm's joint count (`joints`), which sets the
width of the journal's rows and of the state and action vectors.

@see finish(), unfinished()
"""

import itertools
import json
import os
import pathlib
from collections.abc import Mapping
from typing import Any

import av
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

from robots import errors

## Journal columns per tick before the measured vector and the commanded vector: loop time, chunk, row in the chunk.
JOURNAL_HEAD = 3
## Each arm's joints in an episode whose `rollout.json` does not carry them (an older recording): six per arm.
JOINTS_BEFORE_WIDTHS = {"left": 6, "right": 6}

JOURNAL_NAME = "ticks.f64"
## The journal of an episode whose arms are not both six joints. A robots release from before each arm counted its own
## joints reads #JOURNAL_NAME as rows of two six-joint arms, so it finds no journal here (an empty episode) instead of
## misreading one.
WIDE_JOURNAL_NAME = "ticks_wide.f64"
INFO_NAME = "rollout.json"
MARKER_NAME = "INCOMPLETE.txt"
INFERENCE_LOG_NAME = "inference.jsonl"
CHUNKS_DIR = "chunks"

## What `rollout.json` says an episode ended as.
STATUS_RECORDING = "recording"
STATUS_COMPLETE = "complete"
STATUS_PARTIAL = "partial"
STATUS_EMPTY = "empty"

_DATA_FILE = "data/chunk-000/episode_000000.parquet"


class RolloutError(errors.RobotsError):
    """An episode cannot be started, or finished."""


def journal_file(root: pathlib.Path, joints: Mapping[str, int]) -> pathlib.Path:
    """Where an episode's journal lives: #JOURNAL_NAME when both arms have six joints, else #WIDE_JOURNAL_NAME.

    @param root The episode folder.
    @param joints Per arm, `"left"` and `"right"`, its joints (`rollout.json`'s `joints`).
    @return The path.
    """
    return root / (JOURNAL_NAME if dict(joints) == JOINTS_BEFORE_WIDTHS else WIDE_JOURNAL_NAME)


def video_file(root: pathlib.Path, camera: str, *, partial: bool) -> pathlib.Path:
    """Where a camera's video lives, as LeRobot's `video_path` lays it out.

    @param root The episode folder.
    @param camera The camera name.
    @param partial True for the fragmented file written while recording, False for the finished one.
    @return The path.
    """
    name = "episode_000000" + (".partial.mp4" if partial else ".mp4")
    return root / "videos/chunk-000" / f"observation.images.{camera}" / name


def write_json(path: pathlib.Path, value: Any) -> None:
    """Writes JSON through a temporary file and a rename, so a crash never leaves half a file.

    @param path The file.
    @param value The JSON value.
    """
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def read_info(root: pathlib.Path) -> dict[str, Any]:
    """Reads an episode's `rollout.json`.

    @param root The episode folder.
    @return The mapping.
    """
    info: dict[str, Any] = json.loads((root / INFO_NAME).read_text(encoding="utf-8"))
    return info


def vector_names(joints: Mapping[str, int]) -> list[str]:
    """The names of the measured and of the commanded vector: per arm, left then right, its joints then its gripper.

    @param joints Per arm, `"left"` and `"right"`, its joints (`robots.followers.FollowerArm.dof`).
    @return The names, `joint_<i>` in radians and `gripper` in `[0, 1]`, each prefixed by its arm.
    """
    return [
        f"{arm}_{part}"
        for arm in ("left", "right")
        for part in [*(f"joint_{i}" for i in range(joints[arm])), "gripper"]
    ]


def finish(
    root: pathlib.Path,
    *,
    status: str,
    stop_reason: str,
    error: str | None = None,
    loop_counts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Turns the journal and the partial videos into the LeRobot dataset and updates `rollout.json`.

    The episode is as long as the shortest of the journal and the videos; with no frames only the logs stay (empty).

    @param root The episode folder.
    @param status #STATUS_COMPLETE, or #STATUS_PARTIAL for an episode that stopped early or was recovered.
    @param stop_reason Why recording ended.
    @param error What went wrong.
    @param loop_counts What the loop did to the plan over the episode.
    @return The updated `rollout.json` mapping.
    """
    info = read_info(root)
    cameras: list[str] = info["cameras"]
    joints = info["joints"] if "joints" in info else JOINTS_BEFORE_WIDTHS
    names = vector_names(joints)
    width = JOURNAL_HEAD + 2 * len(names)
    journal = journal_file(root, joints)
    raw = np.fromfile(journal, dtype=np.float64) if journal.exists() else np.zeros(0)
    rows = raw[: len(raw) // width * width].reshape(-1, width)
    counts = [_video_frames(video_file(root, name, partial=True)) for name in cameras]
    frames = min([len(rows), *counts])
    if frames == 0:
        info.update(status=STATUS_EMPTY, frames=0)
    else:
        fps = float(info["fps"])
        columns = _columns(rows[:frames], fps, len(names))
        (root / _DATA_FILE).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table({k: pa.array(list(v) if v.shape[1] > 1 else v[:, 0]) for k, v in columns.items()}),
            root / _DATA_FILE,
        )
        videos = {name: _remux(root, name, frames) for name in cameras}
        info.update(status=status, frames=frames, duration_s=frames / fps)
        _write_meta(root, info, columns, videos, names)
    info.update(stop_reason=stop_reason, error=error, **(loop_counts or {}))
    for raw_file in [journal, *(video_file(root, name, partial=True) for name in cameras)]:
        raw_file.unlink(missing_ok=True)
    write_json(root / INFO_NAME, info)
    (root / MARKER_NAME).unlink(missing_ok=True)
    return info


def unfinished(rollouts_dir: pathlib.Path) -> list[pathlib.Path]:
    """The episode folders whose INCOMPLETE.txt names a process that is gone.

    @param rollouts_dir The folder episodes are saved into; may not exist.
    @return The folders.
    """
    found = []
    for marker in sorted(rollouts_dir.glob(f"*/{MARKER_NAME}")):
        words = marker.read_text(encoding="utf-8").split()
        try:
            os.kill(int(words[1]), 0)
        except PermissionError:
            continue
        except (ProcessLookupError, IndexError, ValueError):
            found.append(marker.parent)
        # a process with that pid exists: the run is still going
    return found


def _video_frames(path: pathlib.Path) -> int:
    """Frames in a partial video, counted from its packets; zero when there is no file."""
    if not path.exists():
        return 0
    with av.open(str(path)) as container:
        return sum(1 for packet in container.demux(container.streams.video[0]) if packet.pts is not None)


def _columns(rows: npt.NDArray[np.float64], fps: float, vector: int) -> dict[str, npt.NDArray[Any]]:
    """The parquet columns, each `[frames, dims]`, from the journal rows, whose two vectors are `vector` wide."""
    n = len(rows)
    index = np.arange(n)[:, None]
    head = JOURNAL_HEAD
    return {
        "observation.state": rows[:, head : head + vector].astype(np.float32),
        "action": rows[:, head + vector :].astype(np.float32),
        "timestamp": (index / fps).astype(np.float32),
        "frame_index": index,
        "episode_index": np.zeros((n, 1), dtype=np.int64),
        "index": index,
        "task_index": np.zeros((n, 1), dtype=np.int64),
        "rollout.t_monotonic_s": rows[:, :1],
        "rollout.chunk": rows[:, 1:2].astype(np.int64),
        "rollout.row": rows[:, 2:3].astype(np.int64),
    }


def _remux(root: pathlib.Path, camera: str, frames: int) -> dict[str, Any]:
    """Copies the first `frames` packets of a camera's fragmented MP4 into a faststart MP4 and samples its pixels.

    There are no B-frames, so cutting after any packet leaves a valid stream.

    @return The feature entry for `info.json`, with the per-channel image statistics in `stats`.
    """
    target = video_file(root, camera, partial=False)
    partial = av.open(str(video_file(root, camera, partial=True)))
    with partial as src, av.open(str(target), "w", options={"movflags": "faststart"}) as dst:
        # `opaque`: a stream copy, with the source's codec parameters and no encoder of its own; without it PyAV looks
        # the decoder's name up as an encoder, which fails for a codec whose two names differ.
        stream = dst.add_stream_from_template(src.streams.video[0], opaque=True)
        packets = (p for p in src.demux(src.streams.video[0]) if p.dts is not None)
        for packet in itertools.islice(packets, frames):
            packet.stream = stream
            dst.mux(packet)
    # A sample, as LeRobot's own image statistics are: key frames only, at most about 100 of them, every fourth pixel.
    with av.open(str(target)) as container:
        video = container.streams.video[0]
        video.codec_context.skip_frame = "NONKEY"
        sampled = itertools.islice(container.decode(video), 0, None, max(1, frames // 1000))
        pixels = [f.to_ndarray(format="rgb24")[::4, ::4].reshape(-1, 3) / 255.0 for f in sampled]
        width, height = video.codec_context.width, video.codec_context.height
    stats = {
        k: v if k == "count" else [[[x]] for x in v] for k, v in _stats(np.concatenate(pixels), len(pixels)).items()
    }
    meta = {"video.height": height, "video.width": width, "video.channels": 3, "video.codec": "vp9"}
    meta |= {"video.pix_fmt": "yuv420p", "video.is_depth_map": False, "has_audio": False}
    return {
        "dtype": "video",
        "shape": [height, width, 3],
        "names": ["height", "width", "channels"],
        "info": meta,
        "stats": stats,
    }


def _stats(values: npt.NDArray[Any], count: int) -> dict[str, Any]:
    """LeRobot's per-feature statistics over `[samples, dims]`."""
    values = values.astype(np.float64)
    stats = [values.min(axis=0), values.max(axis=0), values.mean(axis=0), values.std(axis=0)]
    return {k: v.tolist() for k, v in zip(("min", "max", "mean", "std"), stats, strict=True)} | {"count": [count]}


def _write_meta(
    root: pathlib.Path,
    info: dict[str, Any],
    columns: dict[str, npt.NDArray[Any]],
    videos: dict[str, dict[str, Any]],
    vector_names: list[str],
) -> None:
    """Writes `meta/info.json`, `tasks.jsonl`, `episodes.jsonl` and `episodes_stats.jsonl`."""
    frames, fps, task = info["frames"], float(info["fps"]), info["task"]
    features: dict[str, Any] = {}
    for name, values in columns.items():
        names = vector_names if name in ("observation.state", "action") else None
        features[name] = {"dtype": str(values.dtype), "shape": [values.shape[1]], "names": names}
    stats = {name: _stats(values, frames) for name, values in columns.items()}
    for name, video in videos.items():
        key = f"observation.images.{name}"
        features[key] = {k: video[k] for k in ("dtype", "shape", "names")} | {
            "info": video["info"] | {"video.fps": fps}
        }
        stats[key] = video["stats"]
    meta = root / "meta"
    meta.mkdir(exist_ok=True)
    chunk = "chunk-{episode_chunk:03d}"
    write_json(
        meta / "info.json",
        {
            "codebase_version": "v2.1",
            "robot_type": info["body"],
            "total_episodes": 1,
            "total_frames": frames,
            "total_tasks": 1,
            "total_videos": len(videos),
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": fps,
            "splits": {"train": "0:1"},
            "data_path": f"data/{chunk}/episode_{{episode_index:06d}}.parquet",
            "video_path": f"videos/{chunk}/{{video_key}}/episode_{{episode_index:06d}}.mp4",
            "features": features,
        },
    )
    for file, record in (
        ("tasks", {"task_index": 0, "task": task}),
        ("episodes", {"episode_index": 0, "tasks": [task], "length": frames}),
        ("episodes_stats", {"episode_index": 0, "stats": stats}),
    ):
        (meta / f"{file}.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
