"""The setup wizard's cameras: each role's live feed, the frame as the model gets it, and a reference to compare with.

The model learned from views like the ones in the data it was trained on, so the wizard shows each camera role next to
a reference frame: the first frame of the operator's own LeRobot v2.1 dataset when they give its path (#Dataset; a
fine-tuning dataset), else a frame from our YAM data that ships with the package (#SAMPLE_DIR). Beside both it
shows what the model sees: the live frame through the client's own resize (`robots.cameras.encode_jpeg` at the
policy size), which stretches a 4:3 camera to 16:9.

#CameraBank holds one capture thread per role, opened the way the station opens it. swapped_roles() compares the two
wrist cameras' live frames with their references, and says when they match each other's better than their own: two
wrist cameras given each other's roles.

@see CameraBank, Dataset, sample_jpeg(), swapped_roles(), camera_roles()
"""

import dataclasses
import io
import itertools
import json
import pathlib
import threading
import time
import urllib.request
import zipfile
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from robots import cameras, config, errors

## Our YAM reference frames, shipped with the package: the first head-camera frame of three episodes of an in-house
## YAM pick-and-place dataset (episodes 0 to 2), and the wrist
## frames the earlier alignment step ghosted (`camcal_reference`, from a YAM station), all at 640x360.
SAMPLE_DIR = pathlib.Path(__file__).with_name("setup_reference")

## The shipped frames per kind of camera, in the order the page flips through them.
SAMPLES: dict[str, tuple[str, ...]] = {
    "head": ("yam_head_0.jpg", "yam_head_1.jpg", "yam_head_2.jpg"),
    "wrist_left": ("yam_wrist_left.jpg",),
    "wrist_right": ("yam_wrist_right.jpg",),
}

ROLE_KINDS: dict[str, str] = {
    "top": "head",
    "scene": "head",
    "overhead": "head",
    "head": "head",
    "left": "wrist_left",
    "wrist_l": "wrist_left",
    "left_wrist": "wrist_left",
    "right": "wrist_right",
    "wrist_r": "wrist_right",
    "right_wrist": "wrist_right",
}

## What the page calls each kind of camera.
KIND_LABELS: dict[str, str] = {
    "head": "Head camera",
    "wrist_left": "Left wrist camera",
    "wrist_right": "Right wrist camera",
}

## A known kind's role button in the camera step: where the camera sits.
KIND_BUTTONS: dict[str, str] = {"head": "Head", "wrist_left": "Left", "wrist_right": "Right"}

## Words in a dataset's video keys that say which kind of camera each is, for the first guess the page shows.
KEY_HINTS: dict[str, tuple[str, ...]] = {
    "head": ("overhead", "top", "scene", "head", "front", "high", "main"),
    "wrist_left": ("wrist_l", "left_wrist", "left"),
    "wrist_right": ("wrist_r", "right_wrist", "right"),
}

## The dataset format read: LeRobot v2.1, one video per episode and camera.
DATASET_VERSION = "v2.1"

## Seconds one byte-range read of a fine-tune's training data may take.
FETCH_TIMEOUT_S: float = 20.0

## Width of a live feed or reference frame on the page, pixels.
PREVIEW_WIDTH: int = 640

## Seconds a camera may take to deliver its first frame; a RealSense starts in about two.
FIRST_FRAME_S: float = 8.0

## Size every frame is shrunk to before two are compared, `(width, height)`.
COMPARE_SIZE: tuple[int, int] = (64, 36)

## How much better the two wrist cameras' frames must match each other's references than their own to count as swapped.
## 0.2 warned on cameras that were not swapped: on footage from three YAM stations, correctly
## named wrist cameras crossed it on 19% of frames against the station's own first frame. 0.6 crossed on none of 852.
SWAP_MARGIN: float = 0.6

## The loop's stale-frame limit, for the capture threads the wizard opens (the station's own value).
STALE_AFTER_S: float = 0.2

REOPEN_BACKOFF_S: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
WATCH_S: float = 0.5


@dataclasses.dataclass(frozen=True)
class CameraRole:
    """One of a robot's camera roles as the wizard and its page treat it.

    Attributes:
    - role: Its name, the station file's camera name: one the embodiment names, any string.
    - kind: `head`, `wrist_left` or `wrist_right` when #ROLE_KINDS knows the name, else None (an embodiment's own
      name, such as `cam_high`): then it has no reference picture of ours and no dataset key is guessed for it.
    - label: What the page calls it: "Head camera (top)", or "Camera cam_high".
    - short: Its button in the role step: "Head", "Left" or "Right" (#KIND_BUTTONS), or its name.
    """

    role: str
    kind: str | None
    label: str
    short: str


def camera_roles(roles: tuple[str, ...]) -> dict[str, CameraRole]:
    """Each camera role's kind and labels: the one place the wizard and its page decide them.

    A name outside #ROLE_KINDS is never an error: a customer's embodiment names its cameras as it likes.

    @param roles The robot's camera roles.
    @return Per role, in its order, the #CameraRole.
    """
    found = {}
    for role in roles:
        kind = ROLE_KINDS.get(role)
        if kind is None:
            found[role] = CameraRole(role, None, f"Camera {role}", role)
        else:
            label = KIND_LABELS[kind]
            found[role] = CameraRole(role, kind, f"{label} ({role})", KIND_BUTTONS[kind])
    return found


def sample_jpeg(kind: str, index: int) -> bytes:
    """One of our shipped YAM reference frames.

    @param kind `head`, `wrist_left` or `wrist_right`.
    @param index Which one, wrapping round the ones there are.
    @return The JPEG bytes.
    @throws ValueError The kind is unknown.
    """
    if kind not in SAMPLES:
        raise ValueError(f"no reference frames of a {kind!r} camera")
    names = SAMPLES[kind]
    return (SAMPLE_DIR / names[index % len(names)]).read_bytes()


class Dataset:
    """The operator's own LeRobot v2.1 dataset, read for the first frame of its episodes.

    Only `meta/info.json` (the video keys, the path templates, the chunk size) and `meta/episodes.jsonl` (which
    episodes there are) are read up front; a frame is decoded from its episode's video when asked, with PyAV, whose
    wheels decode the AV1 LeRobot records by default (OpenCV's do not).

    Attributes:
    - path: The dataset's root.
    - video_keys: Its cameras, by feature name (`observation.images.wrist_l`).
    - episodes: Its episode indices, in file order.
    """

    def __init__(self, path: pathlib.Path | zipfile.Path) -> None:
        """Reads the dataset's metadata.

        @param path The dataset's root: the directory holding `meta/` and `videos/`, on disk or in a zip.
        @throws ValueError It is not a LeRobot v2.1 dataset with videos and episodes; the message says what is
            missing.
        """
        info_path = path / "meta" / "info.json"
        if not info_path.is_file():
            raise ValueError(f"{path} is not a LeRobot dataset: there is no meta/info.json")
        info = json.loads(info_path.read_text(encoding="utf-8"))
        version = info.get("codebase_version")
        if version != DATASET_VERSION:
            raise ValueError(f"{path} is a LeRobot {version} dataset; the wizard reads {DATASET_VERSION}")
        features = info.get("features")
        if not isinstance(features, dict):
            raise ValueError(f"{info_path} has no features")
        self.video_keys: list[str] = [
            key for key, feature in features.items() if isinstance(feature, dict) and feature.get("dtype") == "video"
        ]
        if not self.video_keys:
            raise ValueError(f"{path} records no video")
        episodes_path = path / "meta" / "episodes.jsonl"
        if not episodes_path.is_file():
            raise ValueError(f"{path} has no meta/episodes.jsonl")
        lines = [line for line in episodes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.episodes: list[int] = [int(json.loads(line)["episode_index"]) for line in lines]
        if not self.episodes:
            raise ValueError(f"{episodes_path} lists no episode")
        self.path = path
        self._template: str = info["video_path"]
        self._chunk_size: int = int(info["chunks_size"])
        self._cache: dict[tuple[str, int], bytes] = {}

    def first_frame(self, key: str, slot: int) -> bytes:
        """The first frame of one episode's video for one camera, as a JPEG no wider than #PREVIEW_WIDTH.

        @param key One of #video_keys.
        @param slot Which episode, counted from the first in the file and wrapping round.
        @return The JPEG bytes.
        @throws ValueError The key is not one of the dataset's cameras, the video is missing, or it cannot be decoded.
        """
        if key not in self.video_keys:
            raise ValueError(f"the dataset has no camera {key!r}; it has {self.video_keys}")
        episode = self.episode(slot)
        cached = self._cache.get((key, episode))
        if cached is not None:
            return cached
        video = self.path / self._template.format(
            video_key=key, episode_chunk=episode // self._chunk_size, episode_index=episode
        )
        if not video.is_file():
            raise ValueError(f"the dataset's episode {episode} has no video at {video}")
        jpeg = encode_preview(decode_first_frame(video))
        self._cache[(key, episode)] = jpeg
        return jpeg

    def episode(self, slot: int) -> int:
        """The episode a slot stands for.

        @param slot Counted from the first episode in the file, wrapping round.
        @return Its `episode_index`.
        """
        return self.episodes[slot % len(self.episodes)]

    def guess_keys(self, roles: dict[str, CameraRole]) -> dict[str, str | None]:
        """A first guess at which video key is which role, from the words in the keys (#KEY_HINTS).

        @param roles camera_roles() of the robot.
        @return Per role, a key no other role took, or None (always for a role of no kind).
        """
        guess: dict[str, str | None] = {}
        taken: set[str] = set()
        # Wrists first: "left" and "right" are surer words than a head camera's.
        for cam in sorted(roles.values(), key=lambda cam: cam.kind == "head"):
            pick = None
            for hint in KEY_HINTS[cam.kind] if cam.kind is not None else ():
                pick = next((key for key in self.video_keys if hint in key.lower() and key not in taken), None)
                if pick is not None:
                    break
            guess[cam.role] = pick
            if pick is not None:
                taken.add(pick)
        return guess


def training_dataset(url: str) -> Dataset:
    """A fine-tune's training data, read from its zip by byte ranges: only the metadata and the videos asked for.

    @param url The zip's download URL (`dream.fine_tuning.training_data()`).
    @return The #Dataset of its first session.
    @throws ValueError The zip holds no LeRobot session.
    @throws OSError The zip could not be read.
    """
    archive = zipfile.ZipFile(io.BufferedReader(_RangeFile("your training data", _http_range(url)), 1 << 16))
    infos = sorted(name for name in archive.namelist() if name == "meta/info.json" or name.endswith("/meta/info.json"))
    if not infos:
        raise ValueError("your training data holds no meta/info.json")
    return Dataset(zipfile.Path(archive, infos[0][: -len("meta/info.json")]))


def _http_range(url: str) -> Callable[[int, int], tuple[bytes, int]]:
    def fetch(start: int, stop: int) -> tuple[bytes, int]:
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{stop - 1}"})
        with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT_S) as answer:
            return answer.read(), int(answer.headers["Content-Range"].rsplit("/", 1)[1])

    return fetch


class _RangeFile(io.RawIOBase):
    """A read-only file whose bytes come from `fetch(start, stop) -> (bytes, size)`, one call per read."""

    def __init__(self, name: str, fetch: Callable[[int, int], tuple[bytes, int]]) -> None:
        self.name = name
        self._fetch = fetch
        self._pos = 0
        self._size = fetch(0, 1)[1]

    def readable(self) -> bool:
        """It can be read.

        @return True.
        """
        return True

    def seekable(self) -> bool:
        """It can seek.

        @return True.
        """
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        """Moves to a byte.

        @param offset Bytes from where `whence` says.
        @param whence `io.SEEK_SET`, `io.SEEK_CUR` or `io.SEEK_END`.
        @return The new position.
        """
        self._pos = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self._size}[whence] + offset
        return self._pos

    def readinto(self, buffer: Any) -> int:
        """Fills a buffer from the position with one fetch.

        @param buffer The buffer.
        @return The bytes read, 0 at the end.
        """
        stop = min(self._pos + len(buffer), self._size)
        if stop <= self._pos:
            return 0
        data = self._fetch(self._pos, stop)[0]
        buffer[: len(data)] = data
        self._pos += len(data)
        return len(data)


def decode_first_frame(video: pathlib.Path | zipfile.Path) -> npt.NDArray[np.uint8]:
    """The first frame of a video as RGB pixels, decoded with PyAV.

    @param video The file, on disk or in a zip.
    @return `[H, W, 3]` uint8.
    @throws ValueError PyAV is not installed, or the file has no decodable video frame.
    """
    try:
        import av
    except ImportError as exc:
        raise ValueError(
            "reading your dataset's videos needs PyAV, which the dream library carries: install the dream library "
            "with pip install --upgrade dreamlabs"
        ) from exc
    try:
        source = str(video) if isinstance(video, pathlib.Path) else io.BytesIO(video.read_bytes())
    except OSError as exc:
        raise ValueError(f"{video} could not be read: {exc}") from exc
    try:
        with av.open(source, "r") as container:
            for frame in container.decode(video=0):
                return np.asarray(frame.to_ndarray(format="rgb24"), dtype=np.uint8)
    except av.error.FFmpegError as exc:
        raise ValueError(f"{video} could not be decoded: {exc}") from exc
    raise ValueError(f"{video} has no video frame")


def encode_preview(rgb: npt.NDArray[np.uint8]) -> bytes:
    """An RGB frame as a JPEG no wider than #PREVIEW_WIDTH, for the page.

    @param rgb `[H, W, 3]` uint8.
    @return The JPEG bytes.
    """
    height, width = rgb.shape[:2]
    if width <= PREVIEW_WIDTH:
        return cameras.encode_jpeg(rgb)
    return cameras.encode_jpeg(rgb, (PREVIEW_WIDTH, height * PREVIEW_WIDTH // width))


def decode_jpeg(data: bytes) -> npt.NDArray[np.uint8]:
    """A JPEG as RGB pixels.

    @param data The JPEG bytes.
    @return `[H, W, 3]` uint8.
    @throws ValueError The bytes are not an image.
    """
    import cv2

    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("not an image")
    return np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)


def similarity(a: npt.NDArray[np.uint8], b: npt.NDArray[np.uint8]) -> float:
    """How alike two frames look at a glance: the correlation of their small grey images.

    Both are shrunk to #COMPARE_SIZE whatever their shape (as the policy resize stretches them), so what counts is
    where the bright and dark parts are: a wrist camera's own gripper fingers, the table's edge.

    @param a `[H, W, 3]` RGB.
    @param b `[H, W, 3]` RGB.
    @return -1 to 1.
    """
    import cv2

    def small(rgb: npt.NDArray[np.uint8]) -> npt.NDArray[np.float64]:
        grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        tiny = np.asarray(cv2.resize(grey, COMPARE_SIZE, interpolation=cv2.INTER_AREA), dtype=np.float64)
        spread = float(tiny.std())
        return (tiny - tiny.mean()) / spread if spread > 1e-6 else np.zeros_like(tiny)

    return float(np.mean(small(a) * small(b)))


def swapped_roles(
    live: dict[str, npt.NDArray[np.uint8]], references: dict[str, npt.NDArray[np.uint8]]
) -> list[dict[str, Any]]:
    """The left and right wrist cameras, when their live frames match each other's references better than their own.

    Only that pair is compared: a head camera given a wrist camera's role is plain to see on the page, and a head and a
    wrist frame shrunk to #COMPARE_SIZE match by chance often enough to warn on cameras that were not swapped.

    @param live Per role, its live frame.
    @param references Per role, its reference frame.
    @return `[{"roles": [a, b], "own": float, "crossed": float}]`: the summed similarity each way, for the wrist pair
        when its crossed sum beats its own by #SWAP_MARGIN.
    """
    roles = sorted(role for role in live if role in references)
    found = []
    for a, b in itertools.combinations(roles, 2):
        if {ROLE_KINDS.get(a), ROLE_KINDS.get(b)} != {"wrist_left", "wrist_right"}:
            continue
        own = similarity(live[a], references[a]) + similarity(live[b], references[b])
        crossed = similarity(live[a], references[b]) + similarity(live[b], references[a])
        if crossed > own + SWAP_MARGIN:
            found.append({"roles": [a, b], "own": round(own, 3), "crossed": round(crossed, 3)})
    return found


def camera_config(row: dict[str, Any]) -> config.CameraConfig:
    """The `robots.config.CameraConfig` for one camera row of the form, as Save writes it.

    @param row A checked camera row: `name`, `source`, `value`, `width`, `height`, `fps`, `rotate`.
    @return The config.
    """
    value = str(row["value"])
    size = {"name": row["name"], "width": row["width"], "height": row["height"], "fps": row["fps"]}
    if row["source"] == "serial":
        return config.CameraConfig(**size, serial=value, rotate=row["rotate"])
    if row["source"] == "class":
        return config.CameraConfig(**size, source=value, rotate=row["rotate"])
    device: int | str = int(value) if value.isdigit() else value
    return config.CameraConfig(**size, device=device, rotate=row["rotate"])


def open_camera_reader(camera: config.CameraConfig) -> Any:
    """Opens and starts one camera the way the station agent does.

    Through `robots.cameras.CameraReader.from_config` with the station's policy size, policy frame rate and managed
    exposure, so the frame is the station's own: serial, size, rate, rotation and colour. Face blur is off: it touches
    only the frames encoded for the policy, which the wizard never sends anywhere.

    @param camera The camera.
    @return The started reader.
    @throws CameraError The camera cannot be opened.
    """
    reader = cameras.CameraReader.from_config(
        camera,
        STALE_AFTER_S,
        obs_fps=config.AppConfig.obs_fps,
        policy_image_size=config.AppConfig.policy_image_size,
        face_blur=False,
        exposure=config.ExposureConfig(),
    )
    reader.start()
    return reader


def _stopped(reader: Any) -> str:
    return str(reader.error) if reader.error is not None else "its capture stopped"


class CameraBank:
    """The cameras held open by role while the page shows them: one capture thread each.

    A camera stays open from open() until release(), because a live feed reads it on and on; a picture of every
    camera opens them one at a time, so the server releases every camera here before one. A camera whose capture
    stops meanwhile (a USB drop stops it, `robots.cameras.CAPTURE_MISS_TOLERANCE_S`) is reopened by a watch thread,
    as the station reopens its own, and states() says it is reconnecting until it delivers again.
    """

    def __init__(self, open_reader: Callable[[config.CameraConfig], Any], policy_size: tuple[int, int]) -> None:
        """Binds how a camera is opened; nothing opens until open().

        @param open_reader `CameraConfig -> started reader` (open_camera_reader(); tests pass a stand-in with the
            `CameraReader` shape: `frame`, `error`, `is_alive()`, `stop()`).
        @param policy_size `(width, height)` the client sends frames at (`AppConfig.policy_image_size`).
        """
        self._open_reader = open_reader
        self._policy_size = policy_size
        self._lock = threading.Lock()
        self._readers: dict[str, tuple[dict[str, Any], Any]] = {}
        self._down: dict[str, dict[str, Any]] = {}
        self._watcher: threading.Thread | None = None

    def open(self, row: dict[str, Any]) -> dict[str, Any]:
        """Opens the row's camera for its role and waits for a frame; a camera already open as the row says stays.

        @param row A checked camera row; `name` is the role.
        @return `{"role", "width", "height", "warning"}`: the frame size after rotation, and
            `robots.cameras.shape_warning()` of it (None when the model sees it unstretched).
        @throws ValueError The camera cannot be opened or delivers no frame within #FIRST_FRAME_S.
        """
        role = row["name"]
        with self._lock:
            held = self._readers.get(role)
            if held is not None and (held[0] != row or not held[1].is_alive()):
                held[1].stop()
                del self._readers[role]
                held = None
            if held is None:
                self._readers[role] = (dict(row), self._start(row))
                self._down.pop(role, None)
                if self._watcher is None or not self._watcher.is_alive():
                    self._watcher = threading.Thread(target=self._watch, name="setup-cameras-watch", daemon=True)
                    self._watcher.start()
            reader = self._readers[role][1]
        deadline = time.monotonic() + FIRST_FRAME_S
        while reader.frame is None and reader.is_alive() and time.monotonic() < deadline:
            time.sleep(0.05)
        frame = reader.frame
        if frame is None:
            self.release(role)
            reason = f"it stopped: {reader.error}" if reader.error is not None else f"none in {FIRST_FRAME_S:g} s"
            raise ValueError(f"the {role} camera {row['value']} opened but delivered no frame ({reason})")
        height, width = frame.data.shape[:2]
        warning = cameras.shape_warning(width, height, self._policy_size)
        return {"role": role, "width": width, "height": height, "warning": warning}

    def _start(self, row: dict[str, Any]) -> Any:
        try:
            return self._open_reader(camera_config(row))
        except errors.CameraError as exc:
            raise ValueError(str(exc)) from exc
        except RuntimeError as exc:
            raise ValueError(f"the {row['name']} camera {row['value']} did not open: {exc}") from exc

    def _watch(self) -> None:
        while True:
            time.sleep(WATCH_S)
            with self._lock:
                if not self._readers:
                    self._watcher = None
                    return
            self.revive()

    def revive(self) -> None:
        """Reopens each held camera whose capture stopped, each no sooner than its next try (#REOPEN_BACKOFF_S)."""
        due = []
        with self._lock:
            now = time.monotonic()
            for role, (row, reader) in self._readers.items():
                down = self._down.get(role)
                if reader.is_alive():
                    if down is not None and reader.frame is not None:
                        del self._down[role]
                    continue
                if down is None:
                    down = self._down[role] = {
                        "error": _stopped(reader),
                        "tries": 0,
                        "next_s": now + REOPEN_BACKOFF_S[0],
                    }
                if now >= down["next_s"]:
                    down["tries"] += 1
                    down["next_s"] = now + REOPEN_BACKOFF_S[min(down["tries"], len(REOPEN_BACKOFF_S) - 1)]
                    due.append((role, row, reader))
        for role, row, reader in due:
            reader.stop()
            try:
                fresh = self._start(row)
            except ValueError as exc:
                with self._lock:
                    if role in self._down:
                        self._down[role]["error"] = str(exc)
                continue
            with self._lock:
                held = self._readers.get(role)
                if held is not None and held[1] is reader:
                    self._readers[role] = (row, fresh)
                    continue
            fresh.stop()

    def states(self) -> dict[str, dict[str, Any]]:
        """How each held camera is doing, for the page.

        @return Per role `{"state": "live" | "reconnecting", "error": why it stopped or None, "tries": reopens so
            far}`.
        """
        with self._lock:
            states = {}
            for role, (_, reader) in self._readers.items():
                down = self._down.get(role)
                if down is not None:
                    states[role] = {"state": "reconnecting", "error": down["error"], "tries": down["tries"]}
                elif not reader.is_alive():
                    states[role] = {"state": "reconnecting", "error": _stopped(reader), "tries": 0}
                else:
                    states[role] = {"state": "live", "error": None, "tries": 0}
            return states

    def rows(self) -> dict[str, dict[str, Any]]:
        """The row each held role was opened with (open()), the page's own, by role.

        @return Copies of the rows.
        """
        with self._lock:
            return {role: dict(row) for role, (row, _reader) in self._readers.items()}

    def is_open(self, role: str) -> bool:
        """Whether the role's camera is held open and capturing.

        @param role The role.
        @return True while open() has it and its capture thread runs.
        """
        held = self._readers.get(role)
        return held is not None and held[1].is_alive()

    def frame(self, role: str) -> npt.NDArray[np.uint8] | None:
        """The role's latest frame.

        @param role The role.
        @return RGB `[H, W, 3]` after the station's rotation, or None when the role is not open or has no frame.
        """
        held = self._readers.get(role)
        frame = held[1].frame if held is not None else None
        return None if frame is None else np.asarray(frame.data, dtype=np.uint8)

    def live_jpeg(self, role: str) -> bytes | None:
        """The role's latest frame as a JPEG no wider than #PREVIEW_WIDTH, for its live feed.

        @param role The role.
        @return JPEG bytes, or None when the role has no frame.
        """
        rgb = self.frame(role)
        return None if rgb is None else encode_preview(rgb)

    def model_jpeg(self, role: str) -> bytes | None:
        """The role's latest frame exactly as the client sends it to the model: resized to the policy size.

        @param role The role.
        @return JPEG bytes, or None when the role has no frame.
        """
        rgb = self.frame(role)
        return None if rgb is None else cameras.encode_jpeg(rgb, self._policy_size)

    def roles(self) -> list[str]:
        """The roles held open now.

        @return The roles.
        """
        return [role for role in self._readers if self.is_open(role)]

    def release(self, role: str | None = None) -> None:
        """Stops one role's camera, or every one, and frees the device for anything else.

        @param role The role, or None for all.
        """
        with self._lock:
            names = [role] if role is not None else list(self._readers)
            for name in names:
                held = self._readers.pop(name, None)
                self._down.pop(name, None)
                if held is not None:
                    held[1].stop()
