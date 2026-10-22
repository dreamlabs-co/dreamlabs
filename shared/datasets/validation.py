"""Dataset validation on the customer's machine: the layout from the file list alone, then the content through a
reader, both against a parsed embodiment (`embodiments.parse()`), summarised in a few counts.

A dataset is a set of files, each named by its path relative to the dataset root: one top-level session folder per
session, each a LeRobot v2.1 dataset (`meta/info.json`, `meta/episodes.jsonl`, `meta/tasks.jsonl`, optionally
dream's `meta/modality.json`, `data/<chunk>/<episode>.parquet`, `videos/<chunk>/<video key>/<episode>.mp4`). The
root picked is either one session folder (it holds `meta/info.json`: the dream client names the session after it,
as the site's pages do) or a folder of session folders. A note at the root that is not data (is_stray(): a
README, a `.txt` or a `.md`) is skipped and never sent; any other file at the root is refused.
Two passes, both run by the dream client before a byte is sent:

- check_layout() reads names and sizes only (#FileEntry): strays skipped, every other path plain and relative
  inside a session folder named per #SESSION_NAME_RE, the required `meta/` files, at least one parquet, every mp4
  under `videos/` at LeRobot's standard `videos/chunk-<NNN>/<video key>/<episode>.mp4` (#_VIDEO_PATH_RE; training
  renames cameras and reads videos in that layout only), and one video directory per video key the embodiment
  names, matched by key (_check_video_keys(): every stream named, except that a session with more streams than a
  fine-tune trains names exactly `embodiments.MAX_CAMERAS` of them, and training leaves the others out); a body
  naming more cameras than that is refused first (`embodiments.check_cameras()`).
- check_dataset() runs check_layout() and then reads through a #Reader (#DirectoryReader over the local directory):
  the `meta/` files parsed, `info.json`'s `video_path` held to the same chunk-first layout, every listed episode's
  parquet opened by `content_checks`, every mp4's first bytes probed and its format read by FFmpeg and held to the
  service's whitelist (`video_formats`, the check the data worker runs too) and its timing to its episode
  (`video_formats.refuse_off_timing()`, the data worker's too), the session's rate to the rates training takes
  (`content_checks.check_fps()`), each camera's axes named as training reads them
  (`content_checks.check_camera_axes()`, the data worker's too), the vector widths (and the count of their `names`,
  when given) against the embodiment's spans, every episode at least one training sample long
  (min_episode_frames()), and a shipped `meta/modality.json` read as a modality file (its parts
  need not be the embodiment's: training writes its own from the embodiment config). It answers the summary the
  client's report shows, with a note for each session whose timestamps drift from its videos
  (`content_checks.timestamp_drift()`), which the data worker re-times rather than refuses.

check_tables() runs the same checks of one session's meta files and parquets on disk, without the body or the videos:
the data worker's, over an upload that may never have met the dream client.

Nothing in a path or a meta file is trusted: an absolute path, a `..` or `.` segment, a control character or a name
that is not UTF-8 fails the dataset (a path becomes an S3 key); `info.json`'s `data_path` and `video_path` are
templates of named fields filled here (#PATH_FIELD_RE), never handed to `str.format`; and no name in a meta
file may hold a control character. Every failure is a `content_checks.DatasetInvalidError` whose message is the one
line the user sees, naming directories and files as the user named them.
"""

import dataclasses
import json
import math
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO, Protocol

from shared.datasets import content_checks, embodiments, video_formats

## The LeRobot codebase version the recipe's loader reads.
CODEBASE_VERSION: str = "v2.1"

## The per-episode file of a LeRobot dataset: one `{"episode_index", "tasks", "length"}` object per line.
EPISODES_FILE: str = "meta/episodes.jsonl"

## The info file of a LeRobot dataset: features, counts, rate and the data path template.
INFO_FILE: str = "meta/info.json"

## The task file of a LeRobot dataset: one `{"task_index", "task"}` object per line, the tasks the parquets'
## `task_index` values point at and the trainer resolves them through.
TASKS_FILE: str = "meta/tasks.jsonl"

## The files every dataset directory must carry, relative to it.
REQUIRED_META: tuple[str, ...] = (INFO_FILE, EPISODES_FILE, TASKS_FILE)

## The `info.json` fields the recipe reads (_info_fields()), top level then the two vector features; a missing one is
## named in the failure line, so a customer whose exporter leaves one out (`chunks_size`) knows which to add.
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
REQUIRED_FEATURES: tuple[str, ...] = ("action", "observation.state")

## dream's per-dataset modality file, optional. Training regenerates it from the embodiment config, so one that is
## not a modality file, or that describes another body (other parts, spans or cameras), fails the dataset here: the
## customer hears at upload that their metadata contradicts the body instead of never.
MODALITY_FILE: str = "meta/modality.json"

## The per-session fields every directory of a dataset must agree on beyond what the embodiment already pins (the
## vector widths and the video keys): the recipe trains one body at one rate, and the feature names are the body's.
SHARED_FIELDS: tuple[str, ...] = ("fps", "state_names", "action_names")

## What a top-level directory may be called: it becomes the session name in the mixture spec and in S3 keys, so
## letters, digits, dot, underscore and hyphen only, 64 at most, the first a letter or digit.
SESSION_NAME_RE: re.Pattern[str] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

## The most recording one dataset may hold, in hours: every session's frames at its rate, together. The data service
## serves it to the Upload page, whose browser checks refuse the same.
MAX_DATASET_HOURS: int = 10

## The largest `meta/` file read into memory (a line of `episodes.jsonl` is a few hundred bytes; 64 MB is a hundred
## thousand episodes with long prompts, and the other meta files are far smaller).
META_MAX_BYTES: int = 64 * 1024**2

## The frames of one training sample's action chunk at the training rate (`content_checks.USABLE_FPS`, the first): the
## training recipe's `action_horizon` (held to it by a data service test). dream draws a
## sample only when the chunk and the frame after it lie inside the episode, so a shorter episode trains on nothing
## (min_episode_frames()).
SAMPLE_FRAMES: int = 48

## The short episodes a refusal names by index before it counts the rest.
NAMED_EPISODES: int = 5

## The longest task string kept; the prompts the recipe trains on are a sentence.
TASK_MAX_CHARS: int = 512

## The fields `info.json`'s `data_path` template may name, as dream fills them, with an optional `0<width>d` format
## of at most two digits (LeRobot writes `03d` and `06d`; a width in the millions would build a path that long).
PATH_FIELD_RE: re.Pattern[str] = re.compile(r"\{(\w+)(?::(0?\d{1,2})d)?\}")

## The template `data_path` is expected to look like, for the failure line.
_DATA_PATH_EXAMPLE: str = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"

## Where an mp4 of a session must sit: LeRobot v2.1's default, chunk first, the video key's directory inside it.
## Training's camera renames and the prefix copy assume it, so a key-first recording is refused here by name
## rather than failing a training run.
_VIDEO_PATH_RE: re.Pattern[str] = re.compile(r"videos/chunk-\d+/[^/]+/[^/]+\.mp4")

## The template `video_path` is expected to look like, for the failure lines.
_VIDEO_PATH_EXAMPLE: str = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

## What a session directory directly contains, for the failure lines.
_LAYOUT: str = "meta/, data/ and videos/"

## A character no path, task string, feature name or block name may hold: it breaks the prompt, the S3 key or the
## failure line.
_CONTROL_RE: re.Pattern[str] = re.compile(r"[\x00-\x1f\x7f]")

## What a malformed meta file raises while it is parsed and its fields are typed: each parser turns these into its
## one line, so nothing but a `content_checks.DatasetInvalidError` leaves check_dataset() for a file that is not
## what it should be. RecursionError is JSON nested deeper than the parser recurses (`[[[...]]]`).
_MALFORMED_ERRORS: tuple[type[Exception], ...] = (
    RecursionError,
    KeyError,
    TypeError,
    IndexError,
    ValueError,
    AttributeError,
    OverflowError,
)

## How a disagreeing field reads in the failure line.
_FIELD_WORDS: dict[str, str] = {
    "fps": "fps",
    "state_names": "observation.state names",
    "action_names": "action names",
}


@dataclasses.dataclass(frozen=True)
class FileEntry:
    """One file of a dataset as the upload declares it.

    Attributes:
    - path: Relative to the dataset root, `/`-separated: `<session>/meta/info.json`.
    - size_bytes: Its size.
    """

    path: str
    size_bytes: int


class Reader(Protocol):
    """Where check_dataset() reads a dataset's files from, by their #FileEntry path (#DirectoryReader on disk)."""

    def read(self, path: str) -> bytes:
        """The whole file.

        @param path The file's path relative to the dataset root.
        @return Its bytes.
        """
        ...

    def read_head(self, path: str, size: int) -> bytes:
        """The file's first bytes.

        @param path The file's path relative to the dataset root.
        @param size How many bytes, at most.
        @return Up to `size` bytes (fewer when the file is shorter).
        """
        ...

    def open(self, path: str) -> BinaryIO:
        """The file, open for reading, for FFmpeg to read only what it needs of it; the caller closes it.

        @param path The file's path relative to the dataset root.
        @return A binary file object.
        """
        ...


class DirectoryReader:
    """A #Reader over a local directory, the dataset root.

    Attributes:
    - root: The directory holding one directory per session.
    """

    def __init__(self, root: Path) -> None:
        """Keeps the root; nothing is read until asked.

        @param root The dataset root.
        """
        self.root = root

    def read(self, path: str) -> bytes:
        """The whole file under the root.

        @param path The file's path relative to the root.
        @return Its bytes.
        """
        return (self.root / path).read_bytes()

    def read_head(self, path: str, size: int) -> bytes:
        """The file's first bytes.

        @param path The file's path relative to the root.
        @param size How many bytes, at most.
        @return Up to `size` bytes.
        """
        with (self.root / path).open("rb") as file:
            return file.read(size)

    def open(self, path: str) -> BinaryIO:
        """The file under the root, open for reading.

        @param path The file's path relative to the root.
        @return A binary file object; the caller closes it.
        """
        return (self.root / path).open("rb")


def list_files(root: Path) -> list[FileEntry]:
    """Every regular file under a local dataset root, sorted by path, as the upload declares them.

    Hidden files are left out: a `.DS_Store` or a `.git/` a desktop drops into the directory is neither part of the
    dataset nor something to send, and the service refuses a hidden path anyway. Links are followed, to folders as to
    files, as the site's folder pick follows them: a session folder that links to one stored elsewhere is sent, not
    silently left out. A folder reached twice through links, or one that cannot be read, fails the listing instead of
    leaving data out.

    @param root The dataset root.
    @return One #FileEntry per file, hidden ones (any path component starting with `.`) skipped.
    @throws OSError For a folder that cannot be read.
    @throws ValueError For a folder reached twice through links (a link back up the tree, or two to one folder).
    """
    files: list[FileEntry] = []
    seen: set[str] = set()
    for folder, dirs, names in os.walk(root, onerror=_raise, followlinks=True):
        real = os.path.realpath(folder)
        if real in seen:
            relative = Path(folder).relative_to(root).as_posix()
            raise ValueError(f"the folder {relative!r} is reached twice through links")
        seen.add(real)
        dirs[:] = [name for name in dirs if not name.startswith(".")]
        for name in names:
            path = Path(folder) / name
            if not name.startswith(".") and path.is_file():
                files.append(FileEntry(path=path.relative_to(root).as_posix(), size_bytes=path.stat().st_size))
    return sorted(files, key=lambda entry: entry.path.split("/"))


def _raise(error: OSError) -> None:
    """os.walk()'s `onerror`: a folder it cannot list fails the listing (it skips it silently otherwise)."""
    raise error


def is_stray(path: str) -> bool:
    """Whether a file is a note at the dataset root rather than data, which the upload skips instead of refusing.

    A desktop export often leaves a `README.txt` or notes beside its session folders; they are not part of any
    session, so the dream client and the site's pages leave them out of the upload and say so, and check_layout()
    skips them in a list that still holds them.

    @param path The file's path relative to the dataset root.
    @return True for a file directly at the root whose name, lowercased, starts with `readme` or ends with `.txt` or
        `.md`.
    """
    name = path.lower()
    return "/" not in path and (name.startswith("readme") or name.endswith((".txt", ".md")))


def check_layout(files: list[FileEntry], embodiment: embodiments.Embodiment) -> dict[str, dict[str, FileEntry]]:
    """Refuses, from names and sizes alone, a file list that is not a LeRobot v2.1 dataset of this body.

    @param files The dataset's files; strays (is_stray()) are skipped.
    @param embodiment The body the dataset was recorded on; its video keys are matched to the video directories.
    @return Session folder to its files, keyed by their path within the session.
    @throws content_checks.DatasetInvalidError With the one line the user sees; for a body naming more cameras than a
        fine-tune trains, `embodiments.check_cameras()`'s.
    """
    try:
        embodiments.check_cameras(embodiment)
    except embodiments.EmbodimentInvalidError as error:
        raise content_checks.DatasetInvalidError(str(error)) from None
    kept = [entry for entry in files if not is_stray(entry.path)]
    if not kept:
        raise content_checks.DatasetInvalidError(
            "The dataset has no files; it must hold at least one session folder (a LeRobot v2.1 dataset with meta/, "
            "data/ and videos/ inside)."
        )
    sessions: dict[str, dict[str, FileEntry]] = {}
    for entry in kept:
        top, rest = _split(entry.path)
        if not rest:
            raise content_checks.DatasetInvalidError(f"{top!r} is not inside a session folder.")
        if top not in sessions and not SESSION_NAME_RE.match(top):
            raise content_checks.DatasetInvalidError(
                f"Session folder {top!r}: a session folder is named with letters, digits, '.', '_' and '-' only, "
                "starting with a letter or digit, 64 characters at most."
            )
        session = sessions.setdefault(top, {})
        if rest in session:
            raise content_checks.DatasetInvalidError(f"Directory '{top}': {rest} is listed twice.")
        _check_size(top, rest, entry.size_bytes)
        session[rest] = entry
    for name, session in sorted(sessions.items()):
        _check_session_layout(name, session, embodiment)
    return sessions


def check_dataset(files: list[FileEntry], embodiment: embodiments.Embodiment, reader: Reader) -> dict[str, Any]:
    """The layout, then every meta file, episode parquet and mp4 header read through `reader`; the summary.

    @param files The dataset's files; strays (is_stray()) are skipped and not counted.
    @param embodiment The body the dataset was recorded on; the vector widths must be its spans' and the video
        directories its video keys.
    @param reader Reads a file by its #FileEntry path.
    @return The summary: `sessions` (how many directories), `episodes`, `frames`, `files`, `bytes`, `state_dim`,
        `action_dim`, `fps` (as `meta/info.json` records it), `video_keys` (sorted) and `notes` (a line per session
        whose timestamps processing will re-time, _drift_note()).
    @throws content_checks.DatasetInvalidError For anything the recipe cannot train on, with the one line the user
        sees.
    """
    layout = check_layout(files, embodiment)
    sessions = [_check_session(name, session, embodiment, reader) for name, session in sorted(layout.items())]
    first = sessions[0]
    for session in sessions[1:]:
        for field in SHARED_FIELDS:
            if session[field] != first[field]:
                raise content_checks.DatasetInvalidError(
                    f"Directory '{session['dir']}': its {_FIELD_WORDS[field]} differ from '{first['dir']}'."
                )
    episodes = sum(session["episodes"] for session in sessions)
    if episodes < 1:
        raise content_checks.DatasetInvalidError("The dataset has no episodes.")
    frames = sum(session["frames"] for session in sessions)
    check_hours(frames, first["fps"])
    return {
        "sessions": len(sessions),
        "episodes": episodes,
        "frames": frames,
        "files": sum(len(session) for session in layout.values()),
        "bytes": sum(entry.size_bytes for session in layout.values() for entry in session.values()),
        "state_dim": embodiment.state_dim,
        "action_dim": embodiment.action_dim,
        "fps": first["fps"],
        "video_keys": sorted(embodiment.video.values()),
        "notes": [session["note"] for session in sessions if session["note"] is not None],
    }


def check_hours(frames: int, fps: int) -> None:
    """Refuses a dataset holding more than #MAX_DATASET_HOURS of recording.

    check_dataset() calls it before an upload, and the data worker before any media work,
    which old clients and direct calls do not skip.

    @param frames Every session's frames together.
    @param fps The rate they share.
    @throws content_checks.DatasetInvalidError Saying the hours, rounded up to a tenth so that just past the limit
        reads past it. It never says to split the dataset: the terms of service forbid splitting data to evade a
        limit.
    """
    if frames > MAX_DATASET_HOURS * 3600 * fps:
        tenths = -(-frames * 10 // (fps * 3600))
        raise content_checks.DatasetInvalidError(
            f"The dataset holds {tenths // 10}.{tenths % 10} h of recording ({frames:,} frames at {fps} fps); a "
            f"dataset holds at most {MAX_DATASET_HOURS} h. Upload fewer sessions or episodes."
        )


def parse_episodes(top: str, data: bytes) -> list[dict[str, Any]]:
    """`meta/episodes.jsonl` as the checks read it.

    @param top The dataset directory, for the message.
    @param data The whole file.
    @return `{episode_index, length, tasks}` per line, in file order.
    @throws content_checks.DatasetInvalidError When it is not an episodes file, a task is empty or whitespace only,
        is longer than #TASK_MAX_CHARS or holds a control character, an index is listed twice, or the indices are not
        0, 1, 2, ... in file order.
    """
    try:
        rows = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
    except (ValueError, RecursionError) as error:
        raise content_checks.DatasetInvalidError(
            f"Directory '{top}': {EPISODES_FILE} is not valid JSON lines."
        ) from error
    episodes: list[dict[str, Any]] = []
    for row in rows:
        try:
            index, frames, tasks = int(row["episode_index"]), int(row["length"]), row["tasks"]
            if not isinstance(tasks, list):
                raise TypeError("tasks must be a list")
        except _MALFORMED_ERRORS as error:
            raise content_checks.DatasetInvalidError(
                f"Directory '{top}': a line of {EPISODES_FILE} lacks episode_index, length or tasks."
            ) from error
        if not all(isinstance(task, str) for task in tasks):
            raise content_checks.DatasetInvalidError(
                f"Directory '{top}': episode {index} lists a task that is not a string."
            )
        # LeRobot writes the same text into tasks.jsonl, so this also catches an empty task there; whitespace alone
        # is empty too, as the trainer's prompt would carry no instruction.
        if any(not task.strip() for task in tasks):
            raise content_checks.DatasetInvalidError(f"Directory '{top}': episode {index} has an empty task.")
        if any(len(task) > TASK_MAX_CHARS for task in tasks):
            raise content_checks.DatasetInvalidError(
                f"Directory '{top}': episode {index} has a task over {TASK_MAX_CHARS} characters."
            )
        if any(_CONTROL_RE.search(task) for task in tasks):
            raise content_checks.DatasetInvalidError(
                f"Directory '{top}': episode {index} has a task with a control character (a newline, a tab or NUL)."
            )
        episodes.append({"episode_index": index, "length": frames, "tasks": tasks})
    if len({episode["episode_index"] for episode in episodes}) != len(episodes):
        raise content_checks.DatasetInvalidError(f"Directory '{top}': {EPISODES_FILE} lists an episode index twice.")
    # dream's loader reads an episode's length at its index in this file (`_get_step_filter`): a gap left by a deleted
    # episode crashes the job on the GPU node, and an out-of-order line gives an episode another's length.
    for position, episode in enumerate(episodes):
        if episode["episode_index"] != position:
            raise content_checks.DatasetInvalidError(
                f"Directory '{top}': {EPISODES_FILE} lists episode {episode['episode_index']} where episode "
                f"{position} belongs; number the episodes 0, 1, 2, ... in file order with no gaps, as LeRobot writes "
                "them."
            )
    return episodes


def parse_tasks(top: str, data: bytes) -> set[int]:
    """The task indices `meta/tasks.jsonl` defines: the set a parquet's `task_index` values must fall in.

    @param top The dataset directory, for the message.
    @param data The whole file.
    @return The `task_index` values it lists.
    @throws content_checks.DatasetInvalidError When it is not a tasks file, a line lacks an integer task_index, or it
        lists no task.
    """
    try:
        rows = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
    except (ValueError, RecursionError) as error:
        raise content_checks.DatasetInvalidError(
            f"Directory '{top}': {TASKS_FILE} is not valid JSON lines."
        ) from error
    indices: set[int] = set()
    for row in rows:
        try:
            indices.add(int(row["task_index"]))
        except _MALFORMED_ERRORS as error:
            raise content_checks.DatasetInvalidError(
                f"Directory '{top}': a line of {TASKS_FILE} lacks an integer task_index."
            ) from error
    if not indices:
        raise content_checks.DatasetInvalidError(f"Directory '{top}': {TASKS_FILE} lists no task.")
    return indices


def _split(path: str) -> tuple[str, str]:
    """`<top-level dir>` and `<rest>` of a file path, refusing anything that is not a plain relative path."""
    if _CONTROL_RE.search(path) or not _is_utf8(path):
        raise content_checks.DatasetInvalidError(
            "The dataset holds a file whose name has a control character or is not UTF-8."
        )
    if path.startswith("/") or "\\" in path:
        raise content_checks.DatasetInvalidError(f"{path!r} is an absolute path; every file must be relative.")
    parts = path.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise content_checks.DatasetInvalidError(
            f"{path!r} is not a plain relative path (an empty, '.' or '..' segment)."
        )
    return parts[0], "/".join(parts[1:])


def _is_utf8(path: str) -> bool:
    """Whether a path encodes as UTF-8 (a name read from disk keeps undecodable bytes as surrogates, which an S3 key
    cannot carry)."""
    try:
        path.encode()
    except UnicodeEncodeError:
        return False
    return True


def _check_size(top: str, rest: str, size_bytes: int) -> None:
    """Refuses a meta file or an episode parquet too large to read into memory, and an empty data file."""
    if rest in (*REQUIRED_META, MODALITY_FILE) and size_bytes > META_MAX_BYTES:
        raise content_checks.DatasetInvalidError(f"Directory '{top}': {rest} is over {META_MAX_BYTES // 1024**2} MB.")
    if _is_parquet(rest) and size_bytes > content_checks.PARQUET_MAX_BYTES:
        raise content_checks.DatasetInvalidError(
            f"Directory '{top}': {rest} is over {content_checks.PARQUET_MAX_BYTES // 1024**2} MB; an episode parquet "
            "holds one episode."
        )
    if (_is_parquet(rest) or _is_mp4(rest)) and size_bytes == 0:
        raise content_checks.DatasetInvalidError(f"Directory '{top}': {rest} is empty.")


def _check_session_layout(name: str, session: dict[str, FileEntry], embodiment: embodiments.Embodiment) -> None:
    """One directory's required files and its video directories against the body's video keys, from the names
    alone."""
    _check_session_info(name, session)
    for required in REQUIRED_META:
        if required not in session:
            raise content_checks.DatasetInvalidError(f"Directory '{name}': missing {required}.")
    if not any(_is_parquet(rest) for rest in session):
        raise content_checks.DatasetInvalidError(f"Directory '{name}': no parquet file under data/.")
    for rest in sorted(session):
        if _is_mp4(rest) and not _VIDEO_PATH_RE.fullmatch(rest):
            raise content_checks.DatasetInvalidError(
                f"Directory '{name}': {rest} is not at videos/chunk-<NNN>/<video key>/<episode>.mp4; fine-tuning "
                f"reads LeRobot {CODEBASE_VERSION}'s standard video layout: move the videos there and set "
                f"{INFO_FILE}'s video_path to {_VIDEO_PATH_EXAMPLE}."
            )
    _check_video_keys(name, sorted({_video_key(rest) for rest in session if _is_mp4(rest)}), embodiment)


def _check_session_info(name: str, session: dict[str, FileEntry]) -> None:
    """A directory without `meta/info.json`: the directories one level down that hold one (a dataset picked one folder
    too far out), named so the user can pick the right one, else the layout a session directory must have. A session
    that also holds session folders is refused: they would be sent inside it and never trained."""
    inner = sorted(rest.split("/")[0] for rest in session if rest == f"{rest.split('/')[0]}/{INFO_FILE}")
    if INFO_FILE in session and inner:
        listed = ", ".join(f"'{folder}'" for folder in inner)
        raise content_checks.DatasetInvalidError(
            f"Directory '{name}' is a session folder and also holds session folders: {listed}; they would not be "
            "trained."
        )
    if INFO_FILE in session:
        return
    if len(inner) == 1:
        raise content_checks.DatasetInvalidError(
            f"Directory '{name}' holds its LeRobot dataset one folder down, in '{inner[0]}'."
        )
    if inner:
        listed = ", ".join(f"'{folder}'" for folder in inner)
        raise content_checks.DatasetInvalidError(
            f"Directory '{name}' holds its LeRobot datasets one folder down, in {listed}."
        )
    raise content_checks.DatasetInvalidError(
        f"Directory '{name}': missing {INFO_FILE}; a session folder (one LeRobot {CODEBASE_VERSION} dataset) "
        f"directly contains {_LAYOUT}."
    )


def _check_video_keys(name: str, video_keys: list[str], embodiment: embodiments.Embodiment) -> None:
    """The video streams a directory has against the ones the body's roles read, matched by key: a stream for every
    role, and a role for every stream, except that a directory with more streams than a fine-tune trains
    (`embodiments.MAX_CAMERAS`) has exactly that many named, and training leaves the others out."""
    if len(embodiment.video) != min(len(video_keys), embodiments.MAX_CAMERAS):
        pick = (
            f" Name {embodiments.MAX_CAMERAS} of them in its video; training leaves the others out."
            if len(video_keys) > embodiments.MAX_CAMERAS
            else ""
        )
        raise content_checks.DatasetInvalidError(
            f"Directory '{name}': {len(video_keys)} video streams; the embodiment names {len(embodiment.video)}.{pick}"
        )
    for role, key in embodiment.video.items():
        if key not in video_keys:
            streams = ", ".join(video_keys)
            raise content_checks.DatasetInvalidError(
                f"Directory '{name}': no video stream {key} for role {role!r}; the dataset has {streams}."
            )


def check_tables(name: str, directory: Path) -> None:
    """Holds one session's meta files and parquets on disk to check_dataset()'s checks, except the body and videos.

    The data worker runs it over every session of an upload before its table stages: an old client, the browser or a
    direct call skips the dream client's checks, and a dataset they would refuse otherwise reaches the trainer and
    fails the job on the GPU node.

    @param name The session as the customer's line names it.
    @param directory The session directory.
    @throws content_checks.DatasetInvalidError For anything check_dataset() refuses in these files, with its line.
    """
    session = {entry.path: entry for entry in list_files(directory)}
    for rest, entry in session.items():
        _check_size(name, rest, entry.size_bytes)

    def read(rest: str) -> bytes:
        return (directory / rest).read_bytes()

    fields, episodes, tasks = _session_meta(name, session, read)
    _check_episode_count(name, fields, episodes)
    _check_parquets(name, fields, episodes, session, read, tasks)
    _check_episode_lengths(name, fields["fps"], episodes)


def _check_session(
    name: str, session: dict[str, FileEntry], embodiment: embodiments.Embodiment, reader: Reader
) -> dict[str, Any]:
    """One directory's facts (`dir`, `episodes`, `frames`, `note` and the #SHARED_FIELDS), after every check of its
    meta files, data files and the body."""

    def read(rest: str) -> bytes:
        return reader.read(f"{name}/{rest}")

    fields, episodes, tasks = _session_meta(name, session, read)
    _check_embodiment(name, fields, embodiment)
    _check_episode_count(name, fields, episodes)
    _check_videos(name, fields, episodes, session, reader)
    frames, drifted = _check_parquets(name, fields, episodes, session, read, tasks)
    _check_episode_lengths(name, fields["fps"], episodes)
    note = _drift_note(name, fields["fps"], drifted, len(episodes)) if drifted else None
    return {
        "dir": name,
        "episodes": fields["episodes"],
        "frames": frames,
        "note": note,
        **{k: fields[k] for k in SHARED_FIELDS},
    }


def _session_meta(
    name: str, session: dict[str, FileEntry], read: Callable[[str], bytes]
) -> tuple[dict[str, Any], list[dict[str, Any]], set[int]]:
    """One directory's meta files parsed and checked on their own: its `info.json` fields (_info_fields()), episodes
    (parse_episodes()) and task indices (parse_tasks()); `read` reads a file by its path within the directory."""
    info = _parse_info(name, read(INFO_FILE))
    episodes = parse_episodes(name, read(EPISODES_FILE))
    tasks = parse_tasks(name, read(TASKS_FILE))
    if MODALITY_FILE in session:
        _check_modality(name, read(MODALITY_FILE))
    _check_info_fields(name, info)
    try:
        fields = _info_fields(info)
    except _MALFORMED_ERRORS as error:
        raise content_checks.DatasetInvalidError(
            f"Directory '{name}': {INFO_FILE} is not a LeRobot {CODEBASE_VERSION} info file."
        ) from error
    _check_names(
        name,
        INFO_FILE,
        [
            fields["codebase_version"],
            fields["data_path"],
            fields["video_path"],
            *fields["video_keys"],
            *(fields["state_names"] or []),
            *(fields["action_names"] or []),
        ],
    )
    if fields["codebase_version"] != CODEBASE_VERSION:
        raise content_checks.DatasetInvalidError(
            f"Directory '{name}': LeRobot {fields['codebase_version']} is not supported; export {CODEBASE_VERSION}."
        )
    try:
        _fill_path(fields["data_path"], fields, 0)
    except ValueError as error:
        raise content_checks.DatasetInvalidError(
            f"Directory '{name}': {INFO_FILE}'s data_path is not a template like {_DATA_PATH_EXAMPLE}."
        ) from error
    for key in fields["video_keys"]:
        try:
            video_path = _fill_path(fields["video_path"], fields, 0, key)
        except ValueError:
            video_path = ""
        if not _VIDEO_PATH_RE.fullmatch(video_path) or _video_key(video_path) != key:
            raise content_checks.DatasetInvalidError(
                f"Directory '{name}': {INFO_FILE}'s video_path is not a template like {_VIDEO_PATH_EXAMPLE}."
            )
    fields["fps"] = content_checks.check_fps(name, fields["fps"])
    for key in fields["video_keys"]:
        content_checks.check_camera_axes(name, key, info["features"][key])
    return fields, episodes, tasks


def min_episode_frames(fps: int) -> int:
    """The fewest frames an episode recorded at `fps` may hold.

    One training sample: #SAMPLE_FRAMES frames at the training rate and the frame after them, which spans 1.6 s from
    the first frame to the last at any rate (49 frames at 30 fps, 25 at 15, where dream's resample makes each recorded
    frame two).

    @param fps The session's rate, one of `content_checks.USABLE_FPS`.
    @return The frames.
    """
    return SAMPLE_FRAMES * fps // content_checks.USABLE_FPS[0] + 1


def _check_episode_lengths(name: str, fps: int, episodes: list[dict[str, Any]]) -> None:
    """Refuses a session with an episode shorter than min_episode_frames(), after the parquets confirmed the lengths;
    names the short episodes."""
    least = min_episode_frames(fps)
    short = [episode["episode_index"] for episode in episodes if episode["length"] < least]
    if not short:
        return
    shown = [str(index) for index in short[:NAMED_EPISODES]]
    if len(short) > NAMED_EPISODES:
        named = f"episodes {', '.join(shown)} and {len(short) - NAMED_EPISODES} more are"
    elif len(short) > 1:
        named = f"episodes {', '.join(shown[:-1])} and {shown[-1]} are"
    else:
        named = f"episode {shown[0]} is"
    seconds = SAMPLE_FRAMES / content_checks.USABLE_FPS[0]
    raise content_checks.DatasetInvalidError(
        f"Directory '{name}': {named} shorter than one training sample: at {fps} fps an episode needs at least "
        f"{least} frames, {seconds:g} s from its first frame to its last."
    )


def _check_episode_count(name: str, fields: dict[str, Any], episodes: list[dict[str, Any]]) -> None:
    """`episodes.jsonl` lists as many episodes as `info.json` says."""
    if len(episodes) != fields["episodes"]:
        raise content_checks.DatasetInvalidError(
            f"Directory '{name}': {EPISODES_FILE} lists {len(episodes)} episodes; {INFO_FILE} says "
            f"{fields['episodes']}."
        )


def _drift_note(name: str, fps: int, drifted: list[tuple[str, content_checks.Drift]], episodes: int) -> str:
    """The note for a session whose timestamps drift from its videos; the browser's line, word for word."""
    path, worst = max(drifted, key=lambda item: item[1].frames)
    return (
        f"Directory '{name}': the timestamps of {len(drifted)} of {episodes} episodes drift from {fps} fps, by up to "
        f"{worst.frames:.1f} frames ({path}, frame {worst.row}); processing re-times them as frame_index / fps, so "
        "each frame of data is paired with its own video frame."
    )


def _check_embodiment(name: str, fields: dict[str, Any], embodiment: embodiments.Embodiment) -> None:
    """The directory's vector widths and video keys, as `meta/info.json` declares them, against the body's spans and
    roles, and each vector's `names` (when given) one per value; the parquets are then held to the same widths
    (_check_parquets())."""
    for what, declared, spans in (
        ("actions have", fields["action_dim"], embodiment.action_dim),
        ("observation.state has", fields["state_dim"], embodiment.state_dim),
    ):
        if declared != spans:
            raise content_checks.DatasetInvalidError(
                f"Directory '{name}': {what} {declared} dimensions; the embodiment's spans cover {spans}."
            )
    for feature, names, width in (
        ("observation.state", fields["state_names"], fields["state_dim"]),
        ("action", fields["action_names"], fields["action_dim"]),
    ):
        if names is not None and len(names) != width:
            raise content_checks.DatasetInvalidError(
                f"Directory '{name}': {INFO_FILE} lists {len(names)} {feature} names for {width} values."
            )
    _check_video_keys(name, fields["video_keys"], embodiment)


def _check_videos(
    name: str, fields: dict[str, Any], episodes: list[dict[str, Any]], session: dict[str, FileEntry], reader: Reader
) -> None:
    """Every video key of `info.json` has an mp4 directory, every listed episode has an mp4 per key where `video_path`
    puts it (the loader reads it there), every mp4 starts like one, every mp4 is in a format the service accepts
    (`video_formats`: read by FFmpeg, which needs PyAV installed), and every episode's video holds one frame per frame
    of the episode at the session's rate (`video_formats.refuse_off_timing()`), so a dataset the data worker would
    refuse is refused here, before a byte is uploaded."""
    mp4s = sorted(rest for rest in session if _is_mp4(rest))
    present = {_video_key(rest) for rest in mp4s}
    lengths: dict[str, int] = {}
    for key in fields["video_keys"]:
        if key not in present:
            raise content_checks.DatasetInvalidError(f"Directory '{name}': no mp4 for video key {key}.")
        for episode in episodes:
            path = _fill_path(fields["video_path"], fields, episode["episode_index"], key)
            if path not in session:
                raise content_checks.DatasetInvalidError(
                    f"Directory '{name}': episode {episode['episode_index']} has no {key} video at {path}."
                )
            if path in lengths:  # the loader reads it from its start for each: all would get the first one's images
                raise content_checks.DatasetInvalidError(
                    f"Directory '{name}': episode {episode['episode_index']} shares its {key} video {path} with "
                    f"another episode or camera; set {INFO_FILE}'s video_path to {_VIDEO_PATH_EXAMPLE}, one file each."
                )
            lengths[path] = episode["length"]
    for rest in mp4s:
        head = reader.read_head(f"{name}/{rest}", content_checks.MP4_PROBE_BYTES)
        content_checks.check_mp4_head(name, rest, head)
        with reader.open(f"{name}/{rest}") as file:
            found = video_formats.answer(file)
        where = f"Directory '{name}': {rest}"
        video_formats.refuse_unless_whitelisted(where, found)
        if rest in lengths:
            video_formats.refuse_off_timing(where, found, fields["fps"], lengths[rest])


def _check_parquets(
    name: str,
    fields: dict[str, Any],
    episodes: list[dict[str, Any]],
    session: dict[str, FileEntry],
    read: Callable[[str], bytes],
    tasks: set[int],
) -> tuple[int, list[tuple[str, content_checks.Drift]]]:
    """Every listed episode's parquet against its index line, the vector widths, and its `task_index` and integer
    annotation values against `meta/tasks.jsonl`; answers the frames the parquets hold, which must be what
    `meta/info.json` says, and each parquet whose timestamps drift from its videos (`content_checks.timestamp_drift()`)
    with its drift."""
    frames = 0
    drifted = []
    for episode in episodes:
        index = episode["episode_index"]
        path = _fill_path(fields["data_path"], fields, index)
        if path not in session:
            raise content_checks.DatasetInvalidError(f"Directory '{name}': episode {index} has no parquet at {path}.")
        facts = content_checks.check_parquet(name, path, read(path), fields["fps"])
        if facts.rows != episode["length"]:
            raise content_checks.DatasetInvalidError(
                f"Directory '{name}': {path} has {facts.rows} frames; {EPISODES_FILE} says episode {index} has "
                f"{episode['length']}."
            )
        for what, width, declared in (
            ("action", facts.action_dim, fields["action_dim"]),
            ("observation.state", facts.state_dim, fields["state_dim"]),
        ):
            if width != declared:
                raise content_checks.DatasetInvalidError(
                    f"Directory '{name}': {path} has {width} {what} values per frame; {INFO_FILE} says {declared}."
                )
        unknown = facts.task_indices - tasks
        if unknown:
            raise content_checks.DatasetInvalidError(
                f"Directory '{name}': {path} has a task_index {min(unknown)} that {TASKS_FILE} does not define."
            )
        unknown = facts.annotation_indices - tasks
        if unknown:
            raise content_checks.DatasetInvalidError(
                f"Directory '{name}': {path} has a task {min(unknown)} in {content_checks.ANNOTATION_COLUMN} that "
                f"{TASKS_FILE} does not define."
            )
        frames += facts.rows
        if facts.drift is not None:
            drifted.append((path, facts.drift))
    if frames != fields["frames"]:
        raise content_checks.DatasetInvalidError(
            f"Directory '{name}': the episode parquets hold {frames} frames; {INFO_FILE} says {fields['frames']}."
        )
    return frames, drifted


def _is_parquet(rest: str) -> bool:
    """Whether a file is an episode parquet by where it sits and what it is called."""
    return rest.startswith("data/") and rest.endswith(".parquet")


def _is_mp4(rest: str) -> bool:
    """Whether a file is a video by where it sits and what it is called: an mp4 anywhere under `videos/`;
    check_layout() then holds every one to #_VIDEO_PATH_RE, so _video_key() reads the right segment."""
    return rest.startswith("videos/") and rest.endswith(".mp4")


def _video_key(rest: str) -> str:
    """The video key of an mp4 path: the directory it sits in."""
    return rest.split("/")[-2]


def _check_info_fields(top: str, info: dict[str, Any]) -> None:
    """Refuses an `info.json` without every #REQUIRED_INFO_FIELDS and, when `features` is an object, the
    #REQUIRED_FEATURES, naming each missing one; a field present but malformed is _info_fields()'s to refuse."""
    missing = [field for field in REQUIRED_INFO_FIELDS if field not in info]
    if "features" in info and isinstance(info["features"], dict):  # only an object's keys can be checked
        missing += [f"features.{feature}" for feature in REQUIRED_FEATURES if feature not in info["features"]]
    if missing:
        raise content_checks.DatasetInvalidError(
            f"Directory '{top}': {INFO_FILE} has no {', '.join(missing)}; a LeRobot {CODEBASE_VERSION} info file "
            f"needs {', '.join(REQUIRED_INFO_FIELDS)}, with {' and '.join(REQUIRED_FEATURES)} in features."
        )


def _parse_info(top: str, data: bytes) -> dict[str, Any]:
    """`meta/info.json` as a dict, or the one-line reason it is not one."""
    try:
        info = json.loads(data)
    except (ValueError, RecursionError) as error:
        raise content_checks.DatasetInvalidError(f"Directory '{top}': {INFO_FILE} is not valid JSON.") from error
    if not isinstance(info, dict):
        raise content_checks.DatasetInvalidError(f"Directory '{top}': {INFO_FILE} must hold a JSON object.")
    return info


def _check_modality(top: str, data: bytes) -> None:
    """Refuses a `meta/modality.json` that is not a modality file (the state/action spans and video map dream reads)
    or a name in it with a control character. Its parts, spans and cameras need not be the embodiment's: training
    replaces every session's own file with one written from the embodiment config,
    and nothing else reads it, so a customer's file that names other parts uploads as it is."""
    try:
        modality = json.loads(data)
        names = [
            *modality["state"],
            *modality["action"],
            *modality["video"],
            *(str(entry["original_key"]) for entry in modality["video"].values()),
        ]
        for span in [*modality["state"].values(), *modality["action"].values()]:
            int(span["start"]), int(span["end"])
    except _MALFORMED_ERRORS as error:
        raise content_checks.DatasetInvalidError(
            f"Directory '{top}': {MODALITY_FILE} is not a modality file."
        ) from error
    _check_names(top, MODALITY_FILE, names)


def _check_names(top: str, file: str, names: list[str]) -> None:
    """Refuses a name of a meta file that holds a control character; it would reach a failure line or the
    training run."""
    if any(_CONTROL_RE.search(name) for name in names):
        raise content_checks.DatasetInvalidError(
            f"Directory '{top}': a name in {file} holds a control character (a newline, a tab or NUL)."
        )


def _fill_path(template: str, fields: dict[str, Any], index: int, video_key: str = "") -> str:
    """Where the loader reads an episode's file: an `info.json` path template (`data_path`, or `video_path` with a
    `video_key`) with the fields dream fills (#PATH_FIELD_RE), never `str.format`, which would let the template
    reach attributes; a field outside those (`{video_key}` without a key, or with a format), a conversion or a brace
    left over raises ValueError."""
    chunk = index // fields["chunks_size"]
    values = {"episode_chunk": chunk, "chunk_index": chunk, "episode_index": index}

    def fill(match: re.Match[str]) -> str:
        if match[1] == "video_key" and video_key and match[2] is None:
            return video_key
        if match[1] not in values:
            raise ValueError(f"the template names an unknown field {match[1]!r}")
        return format(values[match[1]], f"{match[2] or ''}d")

    path = PATH_FIELD_RE.sub(fill, template)
    if "{" in path or "}" in path:
        raise ValueError("the template has a brace that is not a field")
    return path


def _number(value: Any) -> int | float:
    """A JSON number as JSON gave it, for a check to judge (check_fps(): `15.0` is 15, `30.5` no rate); text,
    true/false or one too large for a float (`1e400`) raises, as the browser's infoFields() does."""
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise TypeError(f"{value!r} is not a number")
    return value


def _info_fields(info: dict[str, Any]) -> dict[str, Any]:
    """The fields the recipe reads from `meta/info.json`, typed; a missing or malformed one raises."""
    features = info["features"]
    # Sorted: a recorder writes the features in whatever order the cameras came up, so the same cameras appear in
    # different orders across sessions. Identity is the set.
    video_keys = sorted(key for key, feature in features.items() if feature["dtype"] == "video")
    chunks_size = int(info["chunks_size"])
    if chunks_size < 1:
        raise ValueError("chunks_size must be positive")
    return {
        "codebase_version": str(info["codebase_version"]),
        "episodes": int(info["total_episodes"]),
        "frames": int(info["total_frames"]),
        "fps": _number(info["fps"]),
        "action_dim": _width(features["action"]),
        "state_dim": _width(features["observation.state"]),
        "video_keys": video_keys,
        "state_names": _names(features["observation.state"]),
        "action_names": _names(features["action"]),
        "data_path": str(info["data_path"]),
        # Absent or null reads as "", which the video_path check refuses by name; dream's loader needs it.
        "video_path": str(info.get("video_path") or ""),
        "chunks_size": chunks_size,
    }


def _width(feature: dict[str, Any]) -> int:
    """A vector feature's width: its `shape` is `[width]`, as training reads it; any other shape raises."""
    (width,) = feature["shape"]
    return int(width)


def _names(feature: dict[str, Any]) -> list[str] | None:
    """A feature's `names`: LeRobot writes a list, a dict of lists (`{"axes": [...]}`, `{"motors": [...]}`;
    concatenated in order) or null; anything else raises for the caller's one-line message."""
    if "names" not in feature or feature["names"] is None:
        return None
    names = feature["names"]
    if isinstance(names, dict):
        names = [name for group in names.values() for name in group]
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError("names must be a list of strings")
    return names
