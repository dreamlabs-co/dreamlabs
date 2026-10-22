"""What is inside a dataset's files: every episode parquet and every mp4 header, and the one refusal type.

The layout checks of `validation` read file names and `meta/` files; these read the data, so a NaN action, a
timestamp that runs backwards, a parquet with the wrong number of rows for its episode or a video that is not an MP4
fails the dataset at upload, not the training run hours later. check_parquet() opens one episode parquet (held in
memory, at most #PARQUET_MAX_BYTES) with pyarrow and answers its #ParquetFacts (row count and vector widths, from
which the validator counts the dataset's frames instead of trusting `meta/info.json`); check_mp4_head() looks at the
first #MP4_PROBE_BYTES of a video for the `ftyp` box every MP4 starts with. check_fps() refuses a rate training cannot
take, and check_camera_axes() a camera whose axes training cannot find by name; timestamp_drift() measures how far an
episode's timestamps run from its video frames, which processing corrects (re-timing it) rather than refuses, so the
validators report it as a note. Every refusal, here and in
`validation`, is a #DatasetInvalidError whose message is the one line the user sees: it names the directory and the
file as the user named them, never a key or an exception. Nothing else escapes: a parquet whose footer parses but
whose pages do not, or whose clock is not a number, is a #DatasetInvalidError too, never the reader's own error.
"""

import dataclasses
import io
from typing import Any

import numpy as np
import pyarrow
import pyarrow.parquet

from shared.datasets import video_formats

## The largest episode parquet checked (one episode of a 64-wide float32 vector at 30 fps for an hour is under
## 40 MB); a larger one is refused rather than read into memory.
PARQUET_MAX_BYTES: int = 256 * 1024**2

## The columns every episode parquet must carry: the two vectors the recipe trains on and the frame clock.
REQUIRED_COLUMNS: tuple[str, ...] = ("action", "observation.state", "timestamp")

## The language column dream's loader reads an episode's task from. The data service's task annotation writes it
## from `task_index` but keeps one a dataset brings: text is the prompt itself, an integer an index into
## `meta/tasks.jsonl`, held there like `task_index`, as the trainer resolves it on the GPU node.
ANNOTATION_COLUMN: str = "annotation.human.task_description"

## The bytes of an mp4 probed: the size of the first box, then its type, which is `ftyp`.
MP4_PROBE_BYTES: int = 8

## The box type at bytes 4..8 of every MP4 file.
MP4_MAGIC: bytes = b"ftyp"

## The frame rates training takes, fastest first: 30, and each rate it resamples up to 30 (the training service's
## list and the browser's `ACCEPTED_FPS`; a test holds the three to one list).
USABLE_FPS: tuple[int, ...] = (30, 15, 10, 5)

## The axes dream's loader finds by name in a camera's `shape`: height and width, and its channels under either name
## (check_camera_axes()).
CAMERA_AXES: tuple[str, ...] = ("height", "width")
CHANNEL_AXES: tuple[str, ...] = ("channel", "channels")


class DatasetInvalidError(ValueError):
    """A dataset the recipe cannot train on (its layout, a meta file or a data file); `str(error)` is the one line
    the user sees."""


@dataclasses.dataclass(frozen=True)
class Drift:
    """The frame of an episode whose timestamp sits furthest from its own video frame (timestamp_drift()).

    Attributes:
    - row: The frame, counted from 0 in its parquet.
    - frames: How far its timestamp is from `row / fps`, in frames.
    """

    row: int
    frames: float


@dataclasses.dataclass(frozen=True)
class ParquetFacts:
    """What one episode parquet holds, for the validator's per-episode checks.

    Attributes:
    - rows: The frames (rows) in it.
    - action_dim: The width of every `action` row.
    - state_dim: The width of every `observation.state` row.
    - drift: How far its timestamps run from its video frames when processing will re-time it (timestamp_drift()),
      else None.
    - task_indices: The distinct `task_index` values the parquet holds, empty when it carries no such column.
    - annotation_indices: The distinct values of an integer #ANNOTATION_COLUMN, empty when it carries none or text.
    """

    rows: int
    action_dim: int
    state_dim: int
    drift: Drift | None
    task_indices: frozenset[int] = frozenset()
    annotation_indices: frozenset[int] = frozenset()


def check_parquet(directory: str, path: str, data: bytes, fps: int) -> ParquetFacts:
    """Opens one episode parquet and refuses anything the recipe would choke on.

    @param directory The dataset directory, for the message.
    @param path The file's path within it, for the message.
    @param data The whole file.
    @param fps The rate its session's `meta/info.json` records, which its timestamps are measured against.
    @return Its #ParquetFacts.
    @throws DatasetInvalidError When the file is not a parquet or its pages cannot be read, it lacks a column of
        #REQUIRED_COLUMNS, a vector column is not a list of one width, a value is not finite, the timestamp column
        is not numeric, the timestamps do not strictly increase, a `task_index` column is not an integer, or an
        #ANNOTATION_COLUMN is neither an integer nor text.
    """
    where = f"Directory '{directory}': {path}"
    # A damaged footer or page comes out of pyarrow as a plain OSError (a thrift error), not an ArrowException.
    try:
        schema = pyarrow.parquet.read_schema(io.BytesIO(data))
    except pyarrow.ArrowException as error:
        raise DatasetInvalidError(f"{where} is not a parquet file.") from error
    except OSError as error:
        raise DatasetInvalidError(f"{where} could not be read as a parquet file; it may be damaged.") from error
    for column in REQUIRED_COLUMNS:
        if column not in schema.names:
            raise DatasetInvalidError(f"{where} has no {column} column.")
    # task_index is not required (an episode's task can come from episodes.jsonl), but when it is present its values
    # are read so the caller can hold them to meta/tasks.jsonl, which is what the trainer resolves them through.
    has_task_index = "task_index" in schema.names
    has_annotation = ANNOTATION_COLUMN in schema.names
    read_columns = [
        *REQUIRED_COLUMNS,
        *(["task_index"] if has_task_index else []),
        *([ANNOTATION_COLUMN] if has_annotation else []),
    ]
    try:
        table = pyarrow.parquet.read_table(io.BytesIO(data), columns=read_columns)
    except (pyarrow.ArrowException, OSError) as error:
        raise DatasetInvalidError(f"{where} could not be read as a parquet file; it may be damaged.") from error
    widths = [_vector_width(where, table, name) for name in ("action", "observation.state")]
    clock = table.column("timestamp")
    if not (pyarrow.types.is_integer(clock.type) or pyarrow.types.is_floating(clock.type)):
        raise DatasetInvalidError(f"{where}: the timestamp column is not a number per frame.")
    timestamps = clock.to_numpy()
    if not np.isfinite(timestamps).all():
        raise DatasetInvalidError(f"{where}: timestamp holds a value that is not a finite number.")
    if timestamps.size > 1 and not (np.diff(timestamps) > 0).all():
        raise DatasetInvalidError(f"{where}: the timestamps do not increase from one frame to the next.")
    task_indices = _task_indices(where, table) if has_task_index else frozenset()
    annotation_indices = _annotation_indices(where, table) if has_annotation else frozenset()
    return ParquetFacts(
        rows=table.num_rows,
        action_dim=widths[0],
        state_dim=widths[1],
        drift=timestamp_drift(timestamps, fps),
        task_indices=task_indices,
        annotation_indices=annotation_indices,
    )


def check_fps(directory: str, fps: object) -> int:
    """Refuses a session recorded at a rate training cannot take (#USABLE_FPS); the browser's line, word for word.

    @param directory The dataset directory, for the message.
    @param fps The rate its `meta/info.json` records, as JSON gave it: `15.0` is 15 (the browser cannot tell them
        apart); `30.5`, text or true/false is no rate.
    @return The rate as a whole number.
    @throws DatasetInvalidError When the rate is not one of #USABLE_FPS.
    """
    if isinstance(fps, bool) or not isinstance(fps, int | float) or fps not in USABLE_FPS:
        slower = ", ".join(str(rate) for rate in USABLE_FPS[1:])
        shown = int(fps) if isinstance(fps, float) and fps.is_integer() else fps  # 24.0 reads 24, as the browser's
        raise DatasetInvalidError(
            f"Directory '{directory}': meta/info.json records {shown} fps; training runs at {USABLE_FPS[0]} fps and "
            f"resamples only {slower} fps up to it."
        )
    return int(fps)


def check_camera_axes(directory: str, key: str, feature: dict[str, Any]) -> None:
    """Refuses a camera whose `meta/info.json` entry does not name its shape's axes; the browser's line, word for word.

    dream's loader finds a camera's height, width and channels by name in its `names` (`names.index("height")`, and
    `channel` or `channels`: LeRobot versions differ) and fails the job, on the GPU node, without them. LeRobot writes
    them; an entry assembled by hand can give a `shape` alone.

    @param directory The dataset directory, for the message.
    @param key The camera's video key.
    @param feature Its entry in `meta/info.json`'s features.
    @throws DatasetInvalidError When `names` is not one name per axis of `shape` holding height, width and channels.
    """
    names, shape = feature.get("names"), feature.get("shape")
    if not (
        isinstance(names, list)
        and isinstance(shape, list)
        and len(names) == len(shape)
        and all(axis in names for axis in CAMERA_AXES)
        and any(axis in names for axis in CHANNEL_AXES)
    ):
        raise DatasetInvalidError(
            f"Directory '{directory}': camera {key} in meta/info.json has no axis names training can read; give it "
            '"names": ["height", "width", "channels"], one per axis of its shape, in the shape\'s order.'
        )


def timestamp_drift(timestamps: np.ndarray, fps: int) -> Drift | None:
    """How far an episode's timestamps run from its video frames, when that would change a frame the trainer picks.

    The trainer gives each frame of data the video frame nearest its timestamp (dream's `get_frames_by_timestamps()`),
    and a frame's own video frame is the one at `row / fps`: LeRobot writes `timestamp = frame_index / fps`. A
    recorder that stamps rows with its own clock instead (a 28.4 Hz loop labelled 30 fps) drifts a little more each
    frame, until every frame is paired with another's image. The browser computes the same (`timestampDrift()`; a
    test runs both over the same cases).

    @param timestamps The episode's `timestamp` column, seconds per frame.
    @param fps The rate its session's `meta/info.json` records.
    @return The furthest frame (the first of equals) when it sits more than `video_formats.HALF_FRAME` from its own
        video frame; None when every frame is paired with its own.
    """
    offsets = np.abs(timestamps.astype(np.float64) * fps - np.arange(timestamps.size))
    if not offsets.size or offsets.max() <= video_formats.HALF_FRAME:
        return None
    row = int(offsets.argmax())
    return Drift(row=row, frames=float(offsets[row]))


def _task_indices(where: str, table: pyarrow.Table) -> frozenset[int]:
    """The distinct values of a parquet's `task_index` column; refuses a column that is not an integer per frame."""
    column = table.column("task_index")
    if not pyarrow.types.is_integer(column.type):
        raise DatasetInvalidError(f"{where}: the task_index column is not an integer per frame.")
    return frozenset(value for value in column.to_pylist() if value is not None)


def _annotation_indices(where: str, table: pyarrow.Table) -> frozenset[int]:
    """The distinct values of an integer #ANNOTATION_COLUMN, none for text; refuses a column that is neither."""
    column = table.column(ANNOTATION_COLUMN)
    if pyarrow.types.is_string(column.type) or pyarrow.types.is_large_string(column.type):
        return frozenset()
    if not pyarrow.types.is_integer(column.type):
        raise DatasetInvalidError(f"{where}: the {ANNOTATION_COLUMN} column is neither an integer nor text per frame.")
    return frozenset(value for value in column.to_pylist() if value is not None)


def _vector_width(where: str, table: pyarrow.Table, name: str) -> int:
    """The one width of a `list<float>` / `fixed_size_list<float>` column whose values are all finite."""
    column = table.column(name).combine_chunks()
    if pyarrow.types.is_fixed_size_list(column.type):
        width = int(column.type.list_size)
    elif pyarrow.types.is_list(column.type) or pyarrow.types.is_large_list(column.type):
        lengths = np.diff(column.offsets.to_numpy())
        if lengths.size == 0 or (lengths != lengths[0]).any():
            raise DatasetInvalidError(f"{where}: the {name} rows are empty or not all the same length.")
        width = int(lengths[0])
    else:
        raise DatasetInvalidError(f"{where}: the {name} column is not a list of numbers.")
    if column.null_count or width < 1:
        raise DatasetInvalidError(f"{where}: the {name} column has empty rows.")
    values = column.flatten().to_numpy(zero_copy_only=False)
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise DatasetInvalidError(f"{where}: {name} holds a value that is not a finite number.")
    return width


def check_mp4_head(directory: str, path: str, head: bytes) -> None:
    """Refuses a video whose first bytes are not an MP4's.

    @param directory The dataset directory, for the message.
    @param path The file's path within it, for the message.
    @param head The file's first #MP4_PROBE_BYTES bytes (fewer when the file is shorter).
    @throws DatasetInvalidError When the `ftyp` box is not where every MP4 has it.
    """
    if head[4:MP4_PROBE_BYTES] != MP4_MAGIC:
        raise DatasetInvalidError(f"Directory '{directory}': {path} is not an MP4 file.")
