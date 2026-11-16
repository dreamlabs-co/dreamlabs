"""LeRobot v3.0 sessions read as the v2.1 sessions fine-tuning trains on.

LeRobot 3.0 stores many episodes per file: `data/chunk-<NNN>/file-<NNN>.parquet` holds the rows of several episodes
back to back, `videos/<video key>/chunk-<NNN>/file-<NNN>.mp4` their frames back to back (LeRobot joins the episodes'
own videos by stream copy, so each episode starts on a key frame), and the episodes are rows of
`meta/episodes/chunk-<NNN>/file-<NNN>.parquet` saying which data file and row range, and which video file and time
range per camera, each one is. The tasks are `meta/tasks.parquet`, a `task_index` per task string. Fine-tuning reads
v2.1, one parquet and one mp4 per camera per episode (`validation`), so a v3.0 session is converted before anything
else reads it: the data service converts each one of an upload, and the dream client
checks one before it is sent through the same reading (`validation.check_dataset()`).

read_session() parses a session's v3.0 meta into a #Session: every episode's length, tasks, rows and video ranges,
held to each other and to the files present. v21_meta() writes the v2.1 meta files of it (`meta/info.json`,
`meta/episodes.jsonl`, `meta/tasks.jsonl` and, when the episodes carry statistics, `meta/episodes_stats.jsonl`);
episode_tables() cuts one data file into its episodes' v2.1 parquets. Cutting the videos is the data service's, in its
limited child. Every path a meta file names is a template filled here (fill()), never with
`str.format`, and every failure is a `content_checks.DatasetInvalidError` naming the v3.0 file as the user named it.
"""

import dataclasses
import io
import json
import math
import re
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pyarrow
import pyarrow.parquet

from shared.datasets import content_checks

## The codebase version this module reads.
CODEBASE_VERSION: str = "v3.0"

## The version it writes, the one fine-tuning reads (`validation.CODEBASE_VERSION`, held equal by a test).
WRITES: str = "v2.1"

INFO_FILE: str = "meta/info.json"
TASKS_FILE: str = "meta/tasks.parquet"
EPISODES_DIR: str = "meta/episodes/"

## The v2.1 files a converted session holds, as LeRobot 2.1 names them.
V21_EPISODES_FILE: str = "meta/episodes.jsonl"
V21_TASKS_FILE: str = "meta/tasks.jsonl"
V21_STATS_FILE: str = "meta/episodes_stats.jsonl"
V21_DATA_PATH: str = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
V21_VIDEO_PATH: str = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

## LeRobot 3.0's own templates, for the failure lines.
DATA_PATH_EXAMPLE: str = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH_EXAMPLE: str = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"

## The `info.json` fields read, top level; LeRobot 3.0 writes them all.
REQUIRED_INFO_FIELDS: tuple[str, ...] = (
    "codebase_version",
    "fps",
    "chunks_size",
    "total_episodes",
    "total_frames",
    "data_path",
    "video_path",
    "features",
)

## The largest meta parquet read into memory (`meta/tasks.parquet`, one `meta/episodes` file), as `validation`'s
## META_MAX_BYTES: an episodes file of LeRobot's is a few megabytes with every statistic.
META_MAX_BYTES: int = 64 * 1024**2

## The largest data file read into memory to be cut: LeRobot 3.0 writes them at 100 MB; ten times that is a choice.
DATA_FILE_MAX_BYTES: int = 1024**3

## The fields a v3.0 path template may name, with an optional `0<width>d` of at most two digits.
_FIELD_RE: re.Pattern[str] = re.compile(r"\{(\w+)(?::(0?\d{1,2})d)?\}")

## A v3.0 data file and video file where LeRobot puts them; a template that fills to anything else is refused, so a
## path can never leave `data/` or `videos/` (or climb out of the session).
_DATA_FILE_RE: re.Pattern[str] = re.compile(r"data/[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*\.parquet")
_VIDEO_FILE_RE: re.Pattern[str] = re.compile(r"videos/[^/]+(?:/[A-Za-z0-9_-]+)+\.mp4")

## The prefix of an episode's statistics columns in `meta/episodes`: `stats/<feature>/<statistic>`.
_STATS: str = "stats/"

## The task limits the v2.1 validator applies after conversion; refuse them before any upload too.
TASK_MAX_CHARS: int = 512
_CONTROL_RE: re.Pattern[str] = re.compile(r"[\x00-\x1f\x7f]")

## What a malformed meta file raises while its fields are typed; each turns into the file's one line.
_MALFORMED: tuple[type[Exception], ...] = (KeyError, TypeError, IndexError, ValueError, AttributeError, OverflowError)


@dataclasses.dataclass(frozen=True)
class Video:
    """Where one camera's frames of one episode sit.

    Attributes:
    - path: The v3.0 video file, relative to the session.
    - start_s: The episode's first frame in it, seconds (`videos/<key>/from_timestamp`).
    - end_s: Just past its last frame (`to_timestamp`).
    - whole: Whether the file holds this episode alone, from its start: it is the episode's video as it is.
    """

    path: str
    start_s: float
    end_s: float
    whole: bool


@dataclasses.dataclass(frozen=True)
class Episode:
    """One episode of a v3.0 session.

    Attributes:
    - index: Its `episode_index`, 0, 1, 2, ... in order.
    - length: Its frames.
    - tasks: Its task strings.
    - data: The v3.0 data file holding its rows, relative to the session.
    - videos: Each video key to where its frames sit.
    - stats: Its statistics as LeRobot 2.1 writes them per episode (`{feature: {statistic: value}}`), or None.
    """

    index: int
    length: int
    tasks: list[str]
    data: str
    videos: dict[str, Video]
    stats: dict[str, dict[str, Any]] | None


@dataclasses.dataclass(frozen=True)
class Session:
    """A v3.0 session's meta, read and held together (read_session()).

    Attributes:
    - name: The session, as the failure lines name it.
    - info: Its `meta/info.json`.
    - tasks: Each `task_index` to its task string.
    - episodes: Its episodes in index order.
    - video_keys: Its cameras, sorted.
    """

    name: str
    info: dict[str, Any]
    tasks: dict[int, str]
    episodes: list[Episode]
    video_keys: list[str]

    def data_files(self) -> list[str]:
        """The data files its episodes name, each once, in episode order.

        @return Their paths, relative to the session.
        """
        return list(dict.fromkeys(episode.data for episode in self.episodes))

    def video_files(self) -> list[str]:
        """The video files its episodes name, each once, by camera then episode.

        @return Their paths, relative to the session.
        """
        return list(dict.fromkeys(e.videos[key].path for key in self.video_keys for e in self.episodes))

    def v21_data_path(self, episode: Episode) -> str:
        """Where the converted session holds an episode's rows (#V21_DATA_PATH).

        @param episode The episode.
        @return Its parquet's path, relative to the session.
        """
        return fill(V21_DATA_PATH, {"episode_chunk": self.chunk(episode), "episode_index": episode.index})

    def v21_video_path(self, episode: Episode, key: str) -> str:
        """Where the converted session holds an episode's frames of one camera (#V21_VIDEO_PATH).

        @param episode The episode.
        @param key The camera's video key.
        @return Its mp4's path, relative to the session.
        """
        return fill(
            V21_VIDEO_PATH,
            {"episode_chunk": self.chunk(episode), "episode_index": episode.index, "video_key": key},
        )

    def chunk(self, episode: Episode) -> int:
        """The v2.1 chunk of an episode: its index over `chunks_size`.

        @param episode The episode.
        @return The chunk number.
        """
        return episode.index // int(self.info["chunks_size"])


def is_v3_layout(files: set[str] | list[str]) -> bool:
    """Whether a session's files are laid out as LeRobot 3.0's, by their names alone.

    @param files Every file of the session, relative to it.
    @return True when it holds `meta/tasks.parquet` or a parquet under `meta/episodes/` and no `meta/episodes.jsonl`.
    """
    names = set(files)
    has_v3 = TASKS_FILE in names or any(n.startswith(EPISODES_DIR) and n.endswith(".parquet") for n in names)
    return has_v3 and V21_EPISODES_FILE not in names


def is_v3_info(info: Any) -> bool:
    """Whether a parsed `meta/info.json` is LeRobot 3.0's.

    @param info The parsed file.
    @return True when its `codebase_version` is `v3.0` (or `v3.<n>`: LeRobot keeps the layout across 3.x).
    """
    return isinstance(info, dict) and str(info.get("codebase_version", "")).startswith("v3.")


def fill(template: str, values: dict[str, Any]) -> str:
    """A path template with its fields filled (#_FIELD_RE), never through `str.format`.

    A field not in `values`, a format on a text field, or a brace left over raises ValueError.

    @param template The template, as a meta file gave it.
    @param values The fields it may name.
    @return The path.
    """

    def one(match: re.Match[str]) -> str:
        if match[1] not in values:
            raise ValueError(f"the template names an unknown field {match[1]!r}")
        value = values[match[1]]
        if isinstance(value, str):
            if match[2] is not None:
                raise ValueError(f"the field {match[1]!r} is text and takes no format")
            return value
        return format(value, f"{match[2] or ''}d")

    path = _FIELD_RE.sub(one, template)
    if "{" in path or "}" in path:
        raise ValueError("the template has a brace that is not a field")
    return path


def read_session(name: str, files: dict[str, int], read: Callable[[str], bytes]) -> Session:
    """A v3.0 session's meta parsed and held together: every episode's rows and frames named and present.

    @param name The session, as the failure lines name it.
    @param files Every file of the session, relative to it, to its size in bytes.
    @param read Reads a file of the session, by its path relative to it.
    @return The #Session.
    @throws content_checks.DatasetInvalidError When a meta file is missing, too large or not what it should be, the
        episodes are not numbered 0, 1, 2, ... or disagree with `info.json`, or an episode names a data or video file
        the session does not hold, or a range that does not fit it.
    """
    where = f"Directory '{name}'"
    info = _info(where, files, read)
    keys = sorted(
        k for k, feature in info["features"].items() if isinstance(feature, dict) and feature.get("dtype") == "video"
    )
    tasks = _tasks(where, files, read)
    rows = _episode_rows(where, files, read, keys)
    episodes = [_episode(where, info, keys, row, files) for row in rows]
    for position, episode in enumerate(episodes):
        if episode.index != position:
            raise content_checks.DatasetInvalidError(
                f"{where}: meta/episodes lists episode {episode.index} where episode {position} belongs; number the "
                "episodes 0, 1, 2, ... with no gaps, as LeRobot writes them."
            )
    if len(episodes) != int(info["total_episodes"]):
        raise content_checks.DatasetInvalidError(
            f"{where}: meta/episodes lists {len(episodes)} episodes; {INFO_FILE} says {info['total_episodes']}."
        )
    known = set(tasks.values())
    for episode in episodes:
        unknown = [task for task in episode.tasks if task not in known]
        if unknown:
            raise content_checks.DatasetInvalidError(
                f"{where}: episode {episode.index} names the task {unknown[0][:80]!r}, "
                f"which {TASKS_FILE} does not list."
            )
    _check_whole(episodes, keys)
    return Session(name=name, info=info, tasks=tasks, episodes=_with_whole(episodes, keys), video_keys=keys)


def v21_meta(session: Session) -> dict[str, bytes]:
    """The v2.1 meta files of a v3.0 session, by their path in the session, as LeRobot 2.1 writes them.

    `info.json` keeps every field of the v3.0 one but the layout's: its version, the two path templates, the chunk and
    video counts LeRobot 2.1 keeps and not the file sizes 3.0 does.

    @param session The session (read_session()).
    @return `meta/info.json`, `meta/episodes.jsonl`, `meta/tasks.jsonl`, and `meta/episodes_stats.jsonl` when every
        episode carries statistics.
    """
    info = json.loads(json.dumps(session.info))
    info["codebase_version"] = WRITES
    info["data_path"] = V21_DATA_PATH
    info["video_path"] = V21_VIDEO_PATH
    info.pop("data_files_size_in_mb", None)
    info.pop("video_files_size_in_mb", None)
    episodes = len(session.episodes)
    info["total_chunks"] = max(1, math.ceil(episodes / int(info["chunks_size"])))
    info["total_videos"] = episodes * len(session.video_keys)
    info["total_tasks"] = len(session.tasks)
    out = {
        INFO_FILE: json.dumps(info, indent=4).encode(),
        V21_EPISODES_FILE: _lines(
            {"episode_index": e.index, "tasks": e.tasks, "length": e.length} for e in session.episodes
        ),
        V21_TASKS_FILE: _lines({"task_index": i, "task": task} for i, task in sorted(session.tasks.items())),
    }
    if session.episodes and all(e.stats is not None for e in session.episodes):
        out[V21_STATS_FILE] = _lines({"episode_index": e.index, "stats": e.stats} for e in session.episodes)
    return out


def episode_tables(session: Session, data_file: str, data: bytes) -> Iterator[tuple[Episode, bytes]]:
    """One v3.0 data file cut into the v2.1 parquets of the episodes it holds, each its own rows in their order.

    Each episode is the rows whose `episode_index` is its index, which must be as many as its length and sit together;
    the columns, their types and the file's schema metadata are kept as they are.

    @param session The session (read_session()).
    @param data_file The data file, relative to the session.
    @param data Its bytes.
    @return Each episode the file holds, in index order, with its parquet's bytes.
    @throws content_checks.DatasetInvalidError When the file is not a parquet, has no `episode_index`, or an episode's
        rows are not as many as its length or do not sit together.
    """
    where = f"Directory '{session.name}': {data_file}"
    try:
        table = pyarrow.parquet.read_table(io.BytesIO(data))
    except (pyarrow.ArrowException, OSError) as error:
        raise content_checks.DatasetInvalidError(
            f"{where} could not be read as a parquet file; it may be damaged."
        ) from error
    if "episode_index" not in table.column_names:
        raise content_checks.DatasetInvalidError(
            f"{where} has no episode_index column, so its episodes cannot be told apart."
        )
    column = table.column("episode_index")
    if not pyarrow.types.is_integer(column.type) or column.null_count:
        raise content_checks.DatasetInvalidError(f"{where}: the episode_index column is not an integer per frame.")
    indices = column.to_numpy()
    for episode in (e for e in session.episodes if e.data == data_file):
        rows = np.flatnonzero(indices == episode.index)
        if rows.size != episode.length:
            raise content_checks.DatasetInvalidError(
                f"{where} holds {rows.size} frames of episode {episode.index}; meta/episodes says {episode.length}."
            )
        if rows.size and rows[-1] - rows[0] + 1 != rows.size:
            raise content_checks.DatasetInvalidError(
                f"{where}: the frames of episode {episode.index} do not sit together, as LeRobot writes them."
            )
        start = int(rows[0]) if rows.size else 0
        buffer = io.BytesIO()
        pyarrow.parquet.write_table(table.slice(start, int(rows.size)), buffer)
        yield episode, buffer.getvalue()


def _lines(rows: Iterator[dict[str, Any]]) -> bytes:
    """JSON lines, one object per line, as LeRobot 2.1 writes its `.jsonl` files."""
    return "".join(json.dumps(row) + "\n" for row in rows).encode()


def _info(where: str, files: dict[str, int], read: Callable[[str], bytes]) -> dict[str, Any]:
    """`meta/info.json`, a LeRobot 3.0 one with every #REQUIRED_INFO_FIELDS."""
    if INFO_FILE not in files:
        raise content_checks.DatasetInvalidError(f"{where}: missing {INFO_FILE}.")
    if files[INFO_FILE] > META_MAX_BYTES:
        raise content_checks.DatasetInvalidError(f"{where}: {INFO_FILE} is over {META_MAX_BYTES // 1024**2} MB.")
    try:
        info = json.loads(read(INFO_FILE))
    except (ValueError, RecursionError) as error:
        raise content_checks.DatasetInvalidError(f"{where}: {INFO_FILE} is not valid JSON.") from error
    if not isinstance(info, dict) or not is_v3_info(info):
        raise content_checks.DatasetInvalidError(
            f"{where}: its files are laid out as LeRobot {CODEBASE_VERSION}'s, but {INFO_FILE} says "
            f"{info.get('codebase_version') if isinstance(info, dict) else None!r}."
        )
    missing = [field for field in REQUIRED_INFO_FIELDS if field not in info or info[field] is None]
    if missing:
        raise content_checks.DatasetInvalidError(
            f"{where}: {INFO_FILE} has no {', '.join(missing)}; a LeRobot {CODEBASE_VERSION} info file needs "
            f"{', '.join(REQUIRED_INFO_FIELDS)}."
        )
    try:
        if not isinstance(info["features"], dict) or int(info["chunks_size"]) < 1 or int(info["total_episodes"]) < 0:
            raise ValueError("features, chunks_size or total_episodes")
        str(info["data_path"]), str(info["video_path"])
    except _MALFORMED as error:
        raise content_checks.DatasetInvalidError(
            f"{where}: {INFO_FILE} is not a LeRobot {CODEBASE_VERSION} info file."
        ) from error
    return info


def _read_meta_parquet(where: str, files: dict[str, int], read: Callable[[str], bytes], path: str) -> pyarrow.Table:
    """A meta parquet of the session, within #META_MAX_BYTES."""
    if path not in files:
        raise content_checks.DatasetInvalidError(f"{where}: missing {path}.")
    if files[path] > META_MAX_BYTES:
        raise content_checks.DatasetInvalidError(f"{where}: {path} is over {META_MAX_BYTES // 1024**2} MB.")
    try:
        return pyarrow.parquet.read_table(io.BytesIO(read(path)))
    except (pyarrow.ArrowException, OSError) as error:
        raise content_checks.DatasetInvalidError(f"{where}: {path} could not be read as a parquet file.") from error


def _tasks(where: str, files: dict[str, int], read: Callable[[str], bytes]) -> dict[int, str]:
    """`meta/tasks.parquet`: each `task_index` to its task, the task the one text column (pandas writes it as the
    frame's index: `task` or `__index_level_0__`)."""
    table = _read_meta_parquet(where, files, read, TASKS_FILE)
    text = [
        name
        for name, kind in zip(table.column_names, table.schema.types, strict=True)
        if pyarrow.types.is_string(kind) or pyarrow.types.is_large_string(kind)
    ]
    if "task_index" not in table.column_names or len(text) != 1:
        raise content_checks.DatasetInvalidError(
            f"{where}: {TASKS_FILE} needs a task_index column and one column of task text."
        )
    out: dict[int, str] = {}
    for index, task in zip(table.column("task_index").to_pylist(), table.column(text[0]).to_pylist(), strict=True):
        if not isinstance(index, int) or not isinstance(task, str):
            raise content_checks.DatasetInvalidError(
                f"{where}: {TASKS_FILE} has a row without an integer task_index and its text."
            )
        if index in out:
            raise content_checks.DatasetInvalidError(f"{where}: {TASKS_FILE} lists task_index {index} twice.")
        out[index] = task
    if not out:
        raise content_checks.DatasetInvalidError(f"{where}: {TASKS_FILE} lists no task.")
    return out


def _episode_rows(
    where: str, files: dict[str, int], read: Callable[[str], bytes], keys: list[str]
) -> list[dict[str, Any]]:
    """Every row of every `meta/episodes` parquet, in file then row order, sorted by episode index."""
    paths = sorted(
        (p for p in files if p.startswith(EPISODES_DIR) and p.endswith(".parquet")), key=lambda p: p.split("/")
    )
    if not paths:
        raise content_checks.DatasetInvalidError(f"{where}: no parquet under {EPISODES_DIR}.")
    needed = [
        "episode_index",
        "tasks",
        "length",
        "data/chunk_index",
        "data/file_index",
        *(
            f"videos/{key}/{field}"
            for key in keys
            for field in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")
        ),
    ]
    rows: list[dict[str, Any]] = []
    for path in paths:
        table = _read_meta_parquet(where, files, read, path)
        missing = [column for column in needed if column not in table.column_names]
        if missing:
            raise content_checks.DatasetInvalidError(
                f"{where}: {path} has no {missing[0]} column; LeRobot {CODEBASE_VERSION} writes it for every episode."
            )
        wanted = [*needed, *(c for c in table.column_names if c.startswith(_STATS))]
        rows.extend(table.select(wanted).to_pylist())
    try:
        rows.sort(key=lambda row: int(row["episode_index"]))
    except _MALFORMED as error:
        raise content_checks.DatasetInvalidError(
            f"{where}: meta/episodes has an episode_index that is not a number."
        ) from error
    return rows


def _episode(where: str, info: dict[str, Any], keys: list[str], row: dict[str, Any], files: dict[str, int]) -> Episode:
    """One `meta/episodes` row as an #Episode, its files filled from `info.json`'s templates and present."""
    try:
        index, length, tasks = int(row["episode_index"]), int(row["length"]), list(row["tasks"])
        data_fields = {"chunk_index": int(row["data/chunk_index"]), "file_index": int(row["data/file_index"])}
        ranges = {
            key: (
                {
                    "chunk_index": int(row[f"videos/{key}/chunk_index"]),
                    "file_index": int(row[f"videos/{key}/file_index"]),
                    "video_key": key,
                },
                float(row[f"videos/{key}/from_timestamp"]),
                float(row[f"videos/{key}/to_timestamp"]),
            )
            for key in keys
        }
    except _MALFORMED as error:
        raise content_checks.DatasetInvalidError(
            f"{where}: a row of meta/episodes is not a LeRobot {CODEBASE_VERSION} episode."
        ) from error
    if length < 1 or not all(isinstance(task, str) for task in tasks):
        raise content_checks.DatasetInvalidError(
            f"{where}: episode {index} in meta/episodes has no frames or a task that is not text."
        )
    if any(not task.strip() for task in tasks):
        raise content_checks.DatasetInvalidError(f"{where}: episode {index} in meta/episodes has an empty task.")
    if any(len(task) > TASK_MAX_CHARS for task in tasks):
        raise content_checks.DatasetInvalidError(
            f"{where}: episode {index} in meta/episodes has a task over {TASK_MAX_CHARS} characters."
        )
    if any(_CONTROL_RE.search(task) for task in tasks):
        raise content_checks.DatasetInvalidError(
            f"{where}: episode {index} in meta/episodes has a task with a control character."
        )
    try:
        data = fill(str(info["data_path"]), data_fields)
    except ValueError:
        data = ""
    if not _DATA_FILE_RE.fullmatch(data):
        raise content_checks.DatasetInvalidError(
            f"{where}: {INFO_FILE}'s data_path is not a template like {DATA_PATH_EXAMPLE}."
        )
    if data not in files:
        raise content_checks.DatasetInvalidError(f"{where}: episode {index}'s rows are in {data}, which is missing.")
    if files[data] > DATA_FILE_MAX_BYTES:
        raise content_checks.DatasetInvalidError(
            f"{where}: {data} is over {DATA_FILE_MAX_BYTES // 1024**2} MB; write the data in smaller files "
            "(LeRobot's data_files_size_in_mb)."
        )
    videos = {}
    for key, (fields, start, end) in ranges.items():
        try:
            path = fill(str(info["video_path"]), fields)
        except ValueError:
            path = ""
        if not _VIDEO_FILE_RE.fullmatch(path) or ".." in path.split("/"):
            raise content_checks.DatasetInvalidError(
                f"{where}: {INFO_FILE}'s video_path is not a template like {VIDEO_PATH_EXAMPLE}."
            )
        if path not in files:
            raise content_checks.DatasetInvalidError(
                f"{where}: episode {index}'s {key} frames are in {path}, which is missing."
            )
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise content_checks.DatasetInvalidError(
                f"{where}: episode {index}'s {key} frames run from {start} s to {end} s in {path}, which is no range."
            )
        frames = (end - start) * float(info["fps"])
        if abs(frames - length) > 0.5:
            raise content_checks.DatasetInvalidError(
                f"{where}: episode {index}'s {key} range in {path} spans {frames:.2f} frames at "
                f"{info['fps']} fps; its length is {length}."
            )
        videos[key] = Video(path=path, start_s=start, end_s=end, whole=False)
    return Episode(index=index, length=length, tasks=tasks, data=data, videos=videos, stats=_stats(row))


def _stats(row: dict[str, Any]) -> dict[str, dict[str, Any]] | None:
    """An episode's `stats/<feature>/<statistic>` columns as LeRobot 2.1's per-episode statistics, or None."""
    stats: dict[str, dict[str, Any]] = {}
    for column, value in row.items():
        if not column.startswith(_STATS) or value is None:
            continue
        feature, _, statistic = column[len(_STATS) :].rpartition("/")
        if feature and statistic:
            stats.setdefault(feature, {})[statistic] = value
    return stats or None


def _check_whole(episodes: list[Episode], keys: list[str]) -> None:
    """Refuses two episodes whose frames of one camera overlap in one file: each episode's must be its own."""
    for key in keys:
        spans: dict[str, list[tuple[float, float, int]]] = {}
        for episode in episodes:
            video = episode.videos[key]
            spans.setdefault(video.path, []).append((video.start_s, video.end_s, episode.index))
        for path, ranges in spans.items():
            ranges.sort()
            for (_, end, first), (start, _, second) in zip(ranges, ranges[1:], strict=False):
                if start < end - 1e-6:
                    raise content_checks.DatasetInvalidError(
                        f"Episodes {first} and {second} share frames of {path}; meta/episodes must give each episode "
                        "its own time range."
                    )


def _with_whole(episodes: list[Episode], keys: list[str]) -> list[Episode]:
    """The episodes with `whole` set on each video that is the only episode in its file and starts it."""
    holders: dict[str, int] = {}
    for episode in episodes:
        for key in keys:
            path = episode.videos[key].path
            holders[path] = holders.get(path, 0) + 1
    out = []
    for episode in episodes:
        videos = {
            key: dataclasses.replace(video, whole=holders[video.path] == 1 and video.start_s == 0)
            for key, video in episode.videos.items()
        }
        out.append(dataclasses.replace(episode, videos=videos))
    return out
