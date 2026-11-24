"""Data-quality warnings for a dataset that passed the checks: what is almost always a recorder bug.

`validation.check_dataset()` answers "will training run on this". These look at what is inside the episodes and
answer one line per finding, each a guess that refuses nothing: the summary's `warnings`, which the dream client's
report carries beside its `notes` and the site's Upload page lists under each session (its own copy of these rules,
held to these lines by the cases of `shared/tests/datasets/fixtures/quality_warnings.json`).

In the tables (#Tables, one per session, fed each episode's parquet):

- a vector that never changes in an episode, one of its values that never changes in the session, or one that never
  changes in an episode while the rest of its part (its arm) moves: an arm or a gripper an episode does not use is not
  a finding
- values above #MAX_ABS, a step between two frames above #MAX_JUMP (at #REFERENCE_FPS), a first action further than
  #MAX_START_GAP from the first state
- two episodes with exactly the same actions
- a `<side>_grip` part outside 0 to 1 and a `<side>_eef_rot6` part that is not two unit vectors at right angles

In the videos (video_lines(), LeRobot v2.1 sessions only): a size other than `meta/info.json`'s, a shape that is not
16:9, one unchanging or black picture, and one file on two cameras of an episode. In the whole dataset
(dataset_lines()): fewer than #MIN_EPISODES episodes.

The limits are guesses for an arm recorded in radians and metres. Nothing here judges whether a demonstration
succeeded or whether the task text describes it.
"""

import collections
import concurrent.futures
import hashlib
import io
import itertools
from collections.abc import Callable
from typing import Any, BinaryIO

import numpy as np
import pyarrow
import pyarrow.parquet

## The vector columns checked, as LeRobot names them.
VECTORS: tuple[str, ...] = ("action", "observation.state")

## The largest sensible action or state value.
MAX_ABS: float = 10.0

## The largest sensible step of a value between two frames at #REFERENCE_FPS; a session at another rate has it scaled
## by `REFERENCE_FPS / fps`.
MAX_JUMP: float = 1.0
REFERENCE_FPS: float = 30.0

## The largest sensible distance from an episode's first state to its first action.
MAX_START_GAP: float = 0.5

## The fewest episodes a dataset should hold.
MIN_EPISODES: int = 10

## A part (an arm) is idle in an episode when none of its values moves this much: sensor noise on a resting arm is
## a few ten-thousandths of a radian. A value that never changes is reported for an episode only inside a part that
## moves, since an episode that uses one arm, or leaves a gripper alone, is not a recording mistake.
IDLE_BELOW: float = 0.01

## Episodes or values named in one warning before the rest are counted.
NAMED: int = 5

## The longest name quoted in a warning; a longer one is cut.
QUOTED_CHARS: int = 60

## How far a `<side>_grip` value may lie outside 0 to 1, and a `<side>_eef_rot6` length or dot product from exact.
GRIP_SLACK: float = 0.05
ROTATION_SLACK: float = 0.05

## Bytes of a video hashed, with its size, to tell two files apart; enough to separate two real recordings.
HASH_BYTES: int = 1 << 16

## Videos read at once: decoding and reading release the interpreter, and more than this many mostly wait on the disk.
JOBS: int = 8

## The most frames read from each video to tell a frozen or black camera, as shares of its length in the order they
## are read. Reading stops at the first frame that differs from the start, so a live camera costs two frames.
SAMPLED_SHARES: tuple[float, ...] = (0.0, 0.5, 0.95, 0.25, 0.75, 0.1)

## The frames decoded from the start of a video whose keyframes are too few to sample by seeking.
DECODED_FROM_START: int = 120

## The size each sampled frame is shrunk to, width then height, before two are compared.
SAMPLE_SIZE: tuple[int, int] = (64, 36)

## Two sampled frames are one picture when no pixel differs by this much, of 255: a small object moving in a still
## scene reads as live, and a camera that stalled repeats its picture exactly.
FROZEN_BELOW: float = 3.0

## A video is black when its sampled frames average below this brightness, of 255.
BLACK_BELOW: float = 8.0

## The shape the model reads, width over height, and how far a video may be from it. The model's own frame, 320 x
## 176, is 2.3% wider than 16:9 because its height is a multiple of 16, and must pass.
ASPECT: float = 16 / 9
ASPECT_SLACK: float = 0.03

## What reading a file a recorder or a disk damaged can raise here; such a file is left out, never a failure.
_READ_ERRORS: tuple[type[Exception], ...] = (OSError, ValueError, IndexError, ArithmeticError, pyarrow.ArrowException)


def never_failing(lines: Callable[[], list[str]]) -> list[str]:
    """Warning lines from a call that may not fail the checks around it.

    @param lines Answers the lines.
    @return Its lines, or none when it raises anything: a dataset that passed the checks is never refused, or left
        unchecked, because a warning could not be worked out.
    """
    try:
        return lines()
    except Exception:  # noqa: BLE001 - no failure of a warning may fail the checks
        return []


def number(value: float) -> str:
    """A measured value as a warning prints it, the same in the browser's lines.

    @param value The value.
    @return It at four significant digits without trailing zeros (`1.401`, `57.3`, `12`).
    """
    rounded = float(f"{value:.4g}")
    return str(int(rounded)) if rounded.is_integer() and abs(rounded) < 1e15 else repr(rounded)


def quoted(text: str) -> str:
    """A name or text from the dataset, safe to print in one warning line.

    @param text The name or text as the dataset wrote it.
    @return It with every character that is not printable replaced by `?`, cut to #QUOTED_CHARS.
    """
    clean = "".join(char if char.isprintable() else "?" for char in text)
    return clean if len(clean) <= QUOTED_CHARS else clean[: QUOTED_CHARS - 3] + "..."


def named(episodes: list[int]) -> str:
    """Episodes as a warning names them.

    @param episodes Their numbers, at least one.
    @return The first #NAMED and a count of the rest, e.g. `episodes 0, 1, 2 and 4 more`.
    """
    shown = ", ".join(str(episode) for episode in episodes[:NAMED])
    rest = len(episodes) - NAMED
    word = "episode" if len(episodes) == 1 else "episodes"
    return f"{word} {shown}" + (f" and {rest} more" if rest > 0 else "")


def vector(table: pyarrow.Table, name: str) -> np.ndarray | None:
    """One vector column as an array, without building Python lists.

    @param table The episode's table.
    @param name The column, a list of numbers per frame.
    @return A (frames, width) float64 array, or None for a column the checks refuse (absent, not a list of numbers,
        holding a null, rows of different widths).
    """
    if name not in table.column_names:
        return None
    column = table.column(name).combine_chunks()
    kind = column.type
    fixed = pyarrow.types.is_fixed_size_list(kind)
    if not (fixed or pyarrow.types.is_list(kind) or pyarrow.types.is_large_list(kind)) or column.null_count:
        return None
    if not (pyarrow.types.is_floating(kind.value_type) or pyarrow.types.is_integer(kind.value_type)):
        return None
    flat = column.flatten()
    if flat.null_count or len(column) == 0 or len(flat) % len(column):
        return None
    width = len(flat) // len(column)
    if width == 0 or (not fixed and len(set(column.value_lengths().to_pylist())) > 1):
        return None
    values: np.ndarray = flat.to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
    return values.reshape(len(column), width)


def value_names(spans: dict[str, tuple[int, int]], width: int) -> list[str]:
    """Each value of a vector as a warning names it.

    @param spans The vector's parts, each to its `[start, end)` span.
    @param width The vector's width.
    @return One name per value: the part's name for a part one wide, `right_arm[2]` otherwise, `value 9` outside
        every part.
    """
    names = [f"value {index}" for index in range(width)]
    for part, (start, end) in spans.items():
        for index in range(start, min(end, width)):
            names[index] = quoted(part) if end - start == 1 else f"{quoted(part)}[{index - start}]"
    return names


def in_moving_part(index: int, spans: dict[str, tuple[int, int]], moved: np.ndarray) -> bool:
    """Whether a value that never changed in an episode sits in a part that moved in it.

    @param index The value's position in the vector.
    @param spans The vector's parts.
    @param moved How far each value of the vector moved in the episode, highest minus lowest.
    @return False for a part one value wide (a gripper left alone) and for a part none of whose values moved
        #IDLE_BELOW (an arm the episode did not use); True otherwise, and for a value outside every part.
    """
    for start, end in spans.values():
        if start <= index < end:
            return end - start > 1 and float(moved[start:end].max()) >= IDLE_BELOW
    return True


def unmoved_values(indices: list[int], spans: dict[str, tuple[int, int]], labels: list[str]) -> str:
    """The values that never change, as one warning names them together.

    @param indices The values' positions in the vector, in order, at least one.
    @param spans The vector's parts.
    @param labels Every value's name (value_names()).
    @return The first #NAMED names and a count of the rest, then `never changes` or `never change`; a part wider than
        one value whose values are all among them is named once, as `right_arm (all 6 values)`.
    """
    left = set(indices)
    names: dict[int, str] = {}
    for part, (start, end) in spans.items():
        if end - start > 1 and end <= len(labels) and left.issuperset(range(start, end)):
            names[start] = f"{quoted(part)} (all {end - start} values)"
            left -= set(range(start, end))
    names.update({index: labels[index] for index in left})
    ordered = [names[index] for index in sorted(names)]
    more = f" and {len(ordered) - NAMED} more" if len(ordered) > NAMED else ""
    verb = "never changes" if len(indices) == 1 else "never change"
    return f"{', '.join(ordered[:NAMED])}{more} {verb}"


def part_problems(values: np.ndarray, spans: dict[str, tuple[int, int]]) -> list[str]:
    """The parts whose values the model reads in a fixed form, when an episode's are not in it.

    @param values One episode's vector, every frame finite, (frames, width).
    @param spans The vector's parts.
    @return `<part> is outside 0 (closed) to 1 (open)` for a `<side>_grip` part and `<part> is not the first two
        columns of a rotation matrix` for a `<side>_eef_rot6` part.
    """
    problems = []
    for part, (start, end) in spans.items():
        if end > values.shape[1]:
            continue
        block = values[:, start:end]
        if part.endswith("_grip") and (block.min() < -GRIP_SLACK or block.max() > 1 + GRIP_SLACK):
            problems.append(f"{quoted(part)} is outside 0 (closed) to 1 (open)")
        if part.endswith("_eef_rot6") and end - start == 6:
            first, second = block[:, :3], block[:, 3:]
            lengths = np.abs(np.linalg.norm(np.concatenate([first, second]), axis=1) - 1).max()
            if lengths > ROTATION_SLACK or np.abs((first * second).sum(axis=1)).max() > ROTATION_SLACK:
                problems.append(f"{quoted(part)} is not the first two columns of a rotation matrix")
    return problems


class Tables:
    """One session's episode tables, read one at a time, and the warnings they add up to.

    Attributes:
    - episodes: The episodes read so far.
    """

    def __init__(self, fps: float, spans: dict[str, dict[str, tuple[int, int]]]) -> None:
        """Starts a session with nothing read.

        @param fps The session's frame rate, which #MAX_JUMP is scaled to.
        @param spans Each of #VECTORS to its parts' `[start, end)` spans (the embodiment's `action` and `state`).
        """
        self.episodes = 0
        self._max_jump = MAX_JUMP * REFERENCE_FPS / fps
        self._spans = spans
        self._still: dict[str, list[int]] = {name: [] for name in VECTORS}
        self._large: dict[str, list[int]] = {name: [] for name in VECTORS}
        self._jumps: dict[str, list[int]] = {name: [] for name in VECTORS}
        self._peak: dict[str, float] = dict.fromkeys(VECTORS, 0.0)
        ## The widest step of each vector: its size, episode, frame and value.
        self._step: dict[str, tuple[float, int, int, int]] = dict.fromkeys(VECTORS, (0.0, 0, 0, 0))
        self._stuck: dict[str, dict[int, list[int]]] = {name: collections.defaultdict(list) for name in VECTORS}
        self._lowest: dict[str, np.ndarray] = {}
        self._highest: dict[str, np.ndarray] = {}
        self._parts: dict[str, list[int]] = collections.defaultdict(list)
        self._apart: list[int] = []
        self._gap = 0.0
        self._by_hash: dict[bytes, list[int]] = {}

    def add_parquet(self, episode: int, data: bytes) -> None:
        """Reads one episode's parquet; one that cannot be read adds nothing (the checks refused it, or will).

        Neither does one these rules trip over: a warning is never the reason a dataset fails.

        @param episode The episode's number.
        @param data The whole file.
        """
        try:
            file = pyarrow.parquet.ParquetFile(io.BytesIO(data))
            table = file.read(columns=[name for name in VECTORS if name in file.schema_arrow.names])
            vectors = {name: vector(table, name) for name in VECTORS}
            self.add(episode, vectors)
        except Exception:  # noqa: BLE001, S110 - no failure of a warning may fail the checks
            pass

    def add(self, episode: int, vectors: dict[str, np.ndarray | None]) -> None:
        """Takes one episode's vectors.

        @param episode The episode's number.
        @param vectors Each of #VECTORS as a (frames, width) array, or None for one that could not be read.
        """
        self.episodes += 1
        first: dict[str, np.ndarray] = {}
        for name in VECTORS:
            frames = vectors.get(name)
            if frames is None:
                continue
            # A frame holding a NaN or an infinity is the checks' to refuse; it is left out here, and so is the step
            # across it, which would otherwise read as a jump between two frames that never met.
            finite = np.isfinite(frames).all(axis=1)
            values = frames[finite]
            if len(values) < 2:
                continue
            first[name] = values[0]
            low, high = values.min(axis=0), values.max(axis=0)
            same_width = name in self._lowest and len(self._lowest[name]) == len(low)
            self._lowest[name] = np.minimum(self._lowest[name], low) if same_width else low
            self._highest[name] = np.maximum(self._highest[name], high) if same_width else high
            unmoved = np.flatnonzero(high == low)
            if len(unmoved) == len(low):
                self._still[name].append(episode)
            else:
                for index in unmoved:
                    if in_moving_part(int(index), self._spans.get(name, {}), high - low):
                        self._stuck[name][int(index)].append(episode)
            top = float(np.abs(values).max())
            if top > MAX_ABS:
                self._large[name].append(episode)
                self._peak[name] = max(self._peak[name], top)
            steps = np.abs(np.diff(frames, axis=0))
            steps[~(finite[1:] & finite[:-1])] = 0.0
            if steps.size:
                frame, index = np.unravel_index(int(steps.argmax()), steps.shape)
                widest = float(steps[frame, index])
                if widest > self._max_jump:
                    self._jumps[name].append(episode)
                    if widest > self._step[name][0]:
                        self._step[name] = (widest, episode, int(frame) + 1, int(index))
            for problem in part_problems(values, self._spans.get(name, {})):
                self._parts[f"{name}: {problem}"].append(episode)
            if name == "action":
                self._by_hash.setdefault(hashlib.sha256(values.tobytes()).digest(), []).append(episode)
        if len(first) == 2 and len(first["action"]) == len(first["observation.state"]):
            distance = float(np.abs(first["action"] - first["observation.state"]).max())
            if distance > MAX_START_GAP:
                self._apart.append(episode)
                self._gap = max(self._gap, distance)

    def lines(self) -> list[str]:
        """The warnings of the episodes read.

        @return One line per finding, without the session's name.
        """
        lines = []
        for name in VECTORS:
            labels = value_names(self._spans.get(name, {}), len(self._lowest.get(name, ())))
            if self._still[name]:
                lines.append(
                    f"{name} never changes in {named(self._still[name])}: the robot did not move, or it was not "
                    "recorded."
                )
            never = [index for index in range(len(labels)) if self._lowest[name][index] == self._highest[name][index]]
            if never and len(never) < len(labels) and self.episodes > 1:
                unmoved = unmoved_values(never, self._spans.get(name, {}), labels)
                lines.append(f"{name}: {unmoved} in any episode of the session.")
            # One line per set of episodes, naming every value stuck in exactly those: an arm that sat idle through
            # an episode is one finding, not one per joint.
            together: dict[tuple[int, ...], list[int]] = {}
            for index, where in sorted(self._stuck[name].items()):
                if index not in never and index < len(labels):
                    together.setdefault(tuple(where), []).append(index)
            for episodes, indices in together.items():
                unmoved = unmoved_values(indices, self._spans.get(name, {}), labels)
                lines.append(f"{name}: {unmoved} in {named(list(episodes))}.")
            if self._large[name]:
                lines.append(
                    f"{name} reaches {number(self._peak[name])} in {named(self._large[name])}, above "
                    f"{number(MAX_ABS)}: check the units."
                )
            if self._jumps[name]:
                widest, episode, frame, index = self._step[name]
                lines.append(
                    f"{name} jumps between two frames in {named(self._jumps[name])}, by up to {number(widest)} "
                    f"({labels[index]}, episode {episode}, frame {frame}); more than {number(self._max_jump)} in one "
                    "frame is usually a dropped or glitched reading."
                )
        if self._apart:
            lines.append(
                f"the first action is up to {number(self._gap)} from the first state in {named(self._apart)}, more "
                f"than {number(MAX_START_GAP)}: the two vectors may be in different units or orders."
            )
        for problem, where in self._parts.items():
            lines.append(f"{problem}, in {named(where)}.")
        for copies in self._by_hash.values():
            # An episode that stands still is already named above; two of them are not worth a second line.
            moving = [episode for episode in copies if episode not in self._still["action"]]
            if len(moving) > 1:
                lines.append(f"{named(moving)} hold exactly the same actions: one looks like a copy.")
        return lines


def declared_sizes(info: dict[str, Any]) -> dict[str, tuple[int, int]]:
    """Each camera's size as `meta/info.json` declares it.

    @param info The session's `meta/info.json`.
    @return (height, width) by video key; a camera whose entry does not name both axes with whole numbers is left out.
    """
    features = info.get("features")
    sizes: dict[str, tuple[int, int]] = {}
    for key, feature in features.items() if isinstance(features, dict) else ():
        if not isinstance(feature, dict) or feature.get("dtype") != "video":
            continue
        shape, names = feature.get("shape"), feature.get("names")
        if not isinstance(shape, list) or not isinstance(names, list) or len(shape) != len(names):
            continue
        axes = {name: size for name, size in zip(names, shape, strict=True) if isinstance(name, str)}
        height, width = axes.get("height"), axes.get("width")
        if type(height) is int and type(width) is int:
            sizes[key] = (height, width)
    return sizes


def size_lines(sizes: dict[str, list[tuple[int, tuple[int, int]]]], declared: dict[str, tuple[int, int]]) -> list[str]:
    """The warnings of the videos' sizes, which the browser reads from the MP4's header too.

    @param sizes Each video key to its videos' (episode, (height, width)).
    @param declared Each video key's (height, width) from `meta/info.json` (declared_sizes()).
    @return One line per camera whose videos are not the declared size, and one per camera that is not 16:9.
    """
    wrong: dict[str, tuple[tuple[int, int], list[int]]] = {}
    shaped: dict[str, tuple[tuple[int, int], list[int]]] = {}
    for key, found in sizes.items():
        for episode, size in found:
            if key in declared and size != declared[key]:
                wrong.setdefault(key, (size, []))[1].append(episode)
            if size[0] > 0 and abs(size[1] / size[0] - ASPECT) > ASPECT_SLACK * ASPECT:
                shaped.setdefault(key, (size, []))[1].append(episode)
    lines = []
    for key, (size, episodes) in wrong.items():
        want = declared[key]
        lines.append(
            f"{quoted(key)} is {size[1]}x{size[0]} in {named(episodes)}, but meta/info.json says {want[1]}x{want[0]}."
        )
    for key, (size, episodes) in shaped.items():
        lines.append(
            f"{quoted(key)} is {size[1]}x{size[0]} in {named(episodes)}, not 16:9: the model squashes it to 16:9."
        )
    return lines


def same_file_lines(digests: list[tuple[str, int, bytes]]) -> list[str]:
    """The warning for one video file on two cameras of an episode.

    @param digests Each video's key, episode and digest, in the order of the files.
    @return One line per pair of cameras, naming the episodes.
    """
    same: dict[tuple[str, str], list[int]] = {}
    seen: dict[tuple[int, bytes], str] = {}
    for key, episode, digest in digests:
        other = seen.setdefault((episode, digest), key)
        if other != key:
            same.setdefault((other, key), []).append(episode)
    return [
        f"{quoted(first)} and {quoted(second)} hold the same video file in {named(episodes)}."
        for (first, second), episodes in same.items()
    ]


def _small_grey(frame: Any) -> np.ndarray:
    """A decoded frame at #SAMPLE_SIZE in grey, as signed integers so two can be subtracted."""
    grey: np.ndarray = frame.to_ndarray(width=SAMPLE_SIZE[0], height=SAMPLE_SIZE[1], format="gray")
    return grey.astype(np.int16)


def _moved(first: np.ndarray, frame: np.ndarray) -> bool:
    """Whether two sampled frames are two pictures: some pixel differs by #FROZEN_BELOW or more."""
    return float(np.abs(frame - first).max()) >= FROZEN_BELOW


def _sampled_frames(container: Any, stream: Any) -> list[np.ndarray]:
    """Frames spread over a video, small and grey, read until one differs from the first.

    A seek lands on the keyframe before the moment asked for, and that keyframe is the sample: decoding on to the
    exact moment costs ten times as much over a dataset. A video with too few keyframes for that gives the same frame
    every time, so it is read from its start instead, #DECODED_FROM_START frames at most.
    """
    length = float(stream.duration * stream.time_base) if stream.duration else float(container.duration or 0) / 1e6
    frames: dict[int, np.ndarray] = {}
    first = None
    for share in SAMPLED_SHARES:
        container.seek(int(share * length / stream.time_base) + (stream.start_time or 0), stream=stream)
        for frame in container.decode(stream):
            grey = frames.setdefault(frame.pts if frame.pts is not None else len(frames), _small_grey(frame))
            first = grey if first is None else first
            break
        else:
            continue
        if len(frames) > 1 and _moved(first, grey):
            return list(frames.values())
    if len(frames) > 1:
        return list(frames.values())
    container.seek(stream.start_time or 0, stream=stream)
    decoded = [_small_grey(frame) for frame in itertools.islice(container.decode(stream), DECODED_FROM_START)]
    picked = sorted({int(index) for index in np.linspace(0, len(decoded) - 1, len(SAMPLED_SHARES))}) if decoded else []
    return [decoded[index] for index in picked]


def video_facts(file: BinaryIO, size_bytes: int) -> tuple[tuple[int, int], bytes, bool, bool] | None:
    """What the video warnings need from one file.

    @param file The video, open for reading.
    @param size_bytes Its size.
    @return Its (height, width), a digest of its first #HASH_BYTES and its size, whether its sampled frames are one
        picture, and whether they are black; None for a file that cannot be opened as a video. A video whose frames
        cannot be decoded here (a codec this machine lacks) is neither frozen nor black.
    """
    import av  # noqa: PLC0415 - only where a video is read, as `video_formats` imports it

    unreadable: tuple[type[Exception], ...] = (*_READ_ERRORS, av.FFmpegError)
    try:
        digest = hashlib.sha256(file.read(HASH_BYTES) + str(size_bytes).encode()).digest()
        file.seek(0)
        frozen = black = False
        with av.open(file) as container:
            stream = container.streams.video[0]
            size = (int(stream.height), int(stream.width))
            try:
                frames = _sampled_frames(container, stream)
            except unreadable:
                frames = []
            if len(frames) > 1:
                frozen = not any(_moved(frames[0], frame) for frame in frames[1:])
                black = float(np.mean(frames)) < BLACK_BELOW
    except unreadable:
        return None
    return size, digest, frozen, black


def video_lines(
    videos: list[tuple[str, int, int, Callable[[], BinaryIO]]], declared: dict[str, tuple[int, int]]
) -> list[str]:
    """The warnings of one session's videos, read #JOBS at once.

    @param videos Each video's key, episode, size in bytes and a call that opens it for reading.
    @param declared Each video key's (height, width) from `meta/info.json` (declared_sizes()).
    @return One line per finding, without the session's name; a video that cannot be read adds nothing.
    """

    def read(video: tuple[str, int, int, Callable[[], BinaryIO]]) -> tuple[tuple[int, int], bytes, bool, bool] | None:
        try:
            with video[3]() as file:
                return video_facts(file, video[2])
        except OSError:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=JOBS) as pool:
        facts = list(pool.map(read, videos))
    sizes: dict[str, list[tuple[int, tuple[int, int]]]] = collections.defaultdict(list)
    digests: list[tuple[str, int, bytes]] = []
    frozen: dict[str, list[int]] = collections.defaultdict(list)
    black: dict[str, list[int]] = collections.defaultdict(list)
    for (key, episode, _, _), fact in zip(videos, facts, strict=True):
        if fact is None:
            continue
        size, digest, is_frozen, is_black = fact
        sizes[key].append((episode, size))
        digests.append((key, episode, digest))
        if is_black:
            black[key].append(episode)
        elif is_frozen:
            frozen[key].append(episode)
    lines = size_lines(sizes, declared)
    lines += [f"{quoted(key)} is black in {named(episodes)}." for key, episodes in black.items()]
    lines += [
        f"{quoted(key)} shows one unchanging picture in {named(episodes)}: a stalled camera."
        for key, episodes in frozen.items()
    ]
    return lines + same_file_lines(digests)


def dataset_lines(episodes: int) -> list[str]:
    """The warnings only the whole dataset shows.

    @param episodes The dataset's episodes.
    @return One line per finding.
    """
    lines = []
    if 0 < episodes < MIN_EPISODES:
        lines.append(
            f"The dataset holds {episodes} episode{'' if episodes == 1 else 's'}; fewer than {MIN_EPISODES} rarely "
            "teach a task."
        )
    return lines
