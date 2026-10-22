"""Pins `robots.cameras.CameraReader`: the capture thread, the latest-frame slot, staleness and the encode thread.

Every test drives the reader through `conftest.FakeCaptureSource`, which has the
`cv2.VideoCapture` shape. The encode thread encodes the policy's frames with the real cv2; the
from_config tests reach cv2 through `sys.modules` stand-ins instead, and the encoder tests spy
on the real module's calls. Waits are bounded polls on the reader's public attributes, never
fixed sleeps.
"""

import base64
import dataclasses
import logging
import pathlib
import sys
import threading
import time
import types as pytypes
from collections.abc import Callable
from typing import Any

import conftest
import cv2
import numpy as np
import pytest

from robots import cameras, config, errors, types

## Longest a test waits for the capture thread to do something before failing.
WAIT_TIMEOUT_S = 3.0

## Staleness limit the readers here are built with.
STALE_AFTER_S = 0.2

## Per-channel slack for a solid colour after a quality-85 JPEG round trip.
JPEG_TOLERANCE = 4

## Encoding keywords the from_config tests pass: every capture encoded at the capture size.
FROM_CONFIG_ENCODING: dict[str, Any] = {
    "obs_fps": 30.0,
    "policy_image_size": (conftest.FRAME_SHAPE[1], conftest.FRAME_SHAPE[0]),
    # These tests drive from_config with a fake cv2; the real face detector has its own tests below, so keep it
    # out of the fake-cv2 path (building it there would bind faceblur to the stand-in module).
    "face_blur": False,
}


def _wait_until(predicate: Callable[[], bool], timeout_s: float = WAIT_TIMEOUT_S) -> bool:
    """Polls `predicate` until it is true or `timeout_s` passes; returns the last value."""
    deadline_s = time.monotonic() + timeout_s
    while time.monotonic() < deadline_s:
        if predicate():
            return True
        time.sleep(0.001)
    return predicate()


def _reader(source: conftest.FakeCaptureSource, name: str = "overhead", rotate: int = 0) -> cameras.CameraReader:
    # face_blur off: these tests exercise capture and encoding mechanics; the blur has its own tests below.
    return cameras.CameraReader(name, source, STALE_AFTER_S, rotate=rotate, face_blur=False)


def _started(source: conftest.FakeCaptureSource, name: str = "overhead", rotate: int = 0) -> cameras.CameraReader:
    """Builds and starts a reader, and waits for its first frame."""
    reader = _reader(source, name, rotate)
    reader.start()
    assert _wait_until(lambda: reader.frame is not None), "capture thread never published a first frame"
    return reader


def _marked_frame(shape: tuple[int, int, int] = conftest.FRAME_SHAPE) -> np.ndarray:
    """A black BGR frame with a single white pixel at the top-left corner, to track rotation."""
    frame = np.zeros(shape, dtype=np.uint8)
    frame[0, 0] = (255, 255, 255)
    return frame


# --- lifecycle --------------------------------------------------------------------------------


def test_a_brief_miss_keeps_the_thread_and_a_sustained_one_stops_it(
    fake_capture_source: conftest.FakeCaptureSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One read that returns no frame (a USB stall, a driver timeout) is retried, and the frame the source delivers
    next is published as if nothing happened; only a source that stays silent for `CAPTURE_MISS_TOLERANCE_S` stops
    the thread, with the outage in the error."""
    monkeypatch.setattr(cameras, "CAPTURE_MISS_TOLERANCE_S", 0.3)
    reader = cameras.CameraReader("top", fake_capture_source, STALE_AFTER_S)
    reader.start()
    try:
        assert _wait_until(lambda: reader.frame is not None)
        fake_capture_source.ok = False
        time.sleep(0.1)
        assert reader.is_alive(), "a miss shorter than the tolerance is a hiccup, not a death"
        fake_capture_source.ok = True
        first = reader.frame
        assert _wait_until(lambda: reader.frame is not first)
        assert reader.is_alive()
        fake_capture_source.ok = False
        assert _wait_until(lambda: not reader.is_alive())
        assert isinstance(reader.error, errors.CameraError)
        assert "no frame for 0.3 s" in str(reader.error)
    finally:
        reader.stop()


def test_reader_has_no_frame_and_is_not_alive_before_start(fake_capture_source: conftest.FakeCaptureSource) -> None:
    """Nothing is captured until the thread starts; the slot is None and the reader is not alive."""
    reader = _reader(fake_capture_source)

    assert reader.frame is None
    assert reader.error is None
    assert not reader.is_alive()
    assert fake_capture_source.read_count == 0


def test_start_spawns_a_daemon_thread_and_publishes_a_first_frame(
    fake_capture_source: conftest.FakeCaptureSource,
) -> None:
    """The capture thread does one thing: capture, publish, repeat."""
    reader = _started(fake_capture_source)
    try:
        assert reader.is_alive()
        assert isinstance(reader.frame, types.Frame)
        assert reader.frame.camera == "overhead"
        capture_threads = [thread for thread in threading.enumerate() if thread.name == "camera-overhead"]
        assert len(capture_threads) == 1
        assert capture_threads[0].daemon
    finally:
        reader.stop()


def test_stop_joins_the_thread_and_releases_the_source(fake_capture_source: conftest.FakeCaptureSource) -> None:
    """Lifetime follows the loop; stop joins the thread and releases the capture device."""
    reader = _started(fake_capture_source)

    reader.stop()

    assert not reader.is_alive()
    assert fake_capture_source.released
    assert not any(thread.name == "camera-overhead" for thread in threading.enumerate())
    # The last published frame stays readable after stop; consumers still hold a valid snapshot.
    assert reader.frame is not None


def test_stop_is_idempotent(fake_capture_source: conftest.FakeCaptureSource) -> None:
    """Stop() is idempotent; a second call neither raises nor blocks."""
    reader = _started(fake_capture_source)

    reader.stop()
    reader.stop()

    assert fake_capture_source.released
    assert not reader.is_alive()


def test_stop_before_start_only_releases(fake_capture_source: conftest.FakeCaptureSource) -> None:
    """A reader that never started has no thread to join; stop still releases the source."""
    reader = _reader(fake_capture_source)

    reader.stop()

    assert fake_capture_source.released
    assert reader.frame is None


def test_start_twice_raises_camera_error(fake_capture_source: conftest.FakeCaptureSource) -> None:
    """One thread per camera; a second start on a running reader is a CameraError."""
    reader = _started(fake_capture_source)
    try:
        with pytest.raises(errors.CameraError, match="already running"):
            reader.start()
    finally:
        reader.stop()


def test_reader_can_be_restarted_after_stop(fake_capture_source: conftest.FakeCaptureSource) -> None:
    """Start after stop spawns a fresh thread and keeps publishing."""
    reader = _started(fake_capture_source)
    reader.stop()
    first = reader.frame

    reader.start()
    try:
        assert _wait_until(lambda: reader.frame is not first)
        assert reader.is_alive()
    finally:
        reader.stop()


def test_hundred_start_stop_cycles_leave_thread_count_unchanged(
    fake_capture_source: conftest.FakeCaptureSource,
) -> None:
    """No camera thread outlives the reader that owns it; 100 cycles leak nothing."""
    reader = _reader(fake_capture_source)
    baseline = threading.active_count()

    for _ in range(100):
        reader.start()
        reader.stop()

    assert threading.active_count() == baseline
    assert reader.error is None


# --- the latest-frame slot ---------------------------------------------------------------------


def test_slot_holds_only_the_latest_of_several_frames() -> None:
    """Latest-frame slot, not a queue; after five frames only the last is visible."""
    frames = [conftest.make_bgr_frame(bgr=(0, 0, value)) for value in (10, 20, 30, 40, 50)]
    source = conftest.FakeCaptureSource(frames)
    plain_read = source.read

    def read_five_then_end() -> tuple[bool, np.ndarray | None]:
        # Exactly five publishes, then the stream ends, so the slot cannot be refreshed by repeats.
        if source.read_count >= 5:
            source.fail()
        return plain_read()

    source.read = read_five_then_end  # type: ignore[method-assign]
    reader = _reader(source)
    reader.start()
    assert _wait_until(lambda: not reader.is_alive())
    reader.stop()

    # Five frames, then the ended stream is retried (no frame each time) until the miss tolerance runs out.
    assert source.read_count >= 6
    assert reader.frame is not None
    assert reader.frame.data[0, 0].tolist() == [50, 0, 0]
    # Nothing but the slot holds pixels: the frame carries one encoded string, the reader keeps the newest
    # encode alone, and no other attribute keeps a list, tuple or dict of anything.
    assert isinstance(reader.frame.jpeg, str) and reader.frame.jpeg
    assert not any(
        isinstance(value, (list, tuple, dict)) and value for name, value in vars(reader).items() if name != "_encoded"
    )


def test_each_publish_is_a_new_frame_object(fake_capture_source: conftest.FakeCaptureSource) -> None:
    """Publishing rebinds one reference; the old Frame is never mutated in place."""
    reader = _started(fake_capture_source)
    try:
        first = reader.frame
        assert first is not None
        first_data = first.data.copy()
        assert _wait_until(lambda: reader.frame is not first)
        # The frame the consumer held is untouched by later captures.
        np.testing.assert_array_equal(first.data, first_data)
    finally:
        reader.stop()


def test_frame_is_immutable(fake_capture_source: conftest.FakeCaptureSource) -> None:
    """Frame is never mutated after construction; it is a frozen dataclass."""
    reader = _started(fake_capture_source)
    reader.stop()
    frame = reader.frame
    assert frame is not None

    with pytest.raises(dataclasses.FrozenInstanceError):
        frame.timestamp_s = 0.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        frame.data = np.zeros((1, 1, 3), dtype=np.uint8)  # type: ignore[misc]


def test_frame_data_does_not_alias_the_source_buffer() -> None:
    """The published pixels are the reader's own copy; a driver reusing its buffer cannot change them."""
    bgr = conftest.make_bgr_frame(bgr=(7, 8, 9))
    reader = _started(conftest.FakeCaptureSource(bgr))
    reader.stop()
    frame = reader.frame
    assert frame is not None

    bgr[...] = 0

    assert frame.data[0, 0].tolist() == [9, 8, 7]


# --- pixels -----------------------------------------------------------------------------------


def test_frame_is_rgb_uint8_converted_from_bgr() -> None:
    """Frame.data is RGB uint8 [H, W, 3]; BGR (1, 2, 3) becomes RGB (3, 2, 1)."""
    reader = _started(conftest.FakeCaptureSource(conftest.make_bgr_frame(bgr=(1, 2, 3))))
    reader.stop()
    frame = reader.frame
    assert frame is not None

    assert frame.data.dtype == np.uint8
    assert frame.data.shape == conftest.FRAME_SHAPE
    assert frame.data[0, 0].tolist() == [3, 2, 1]
    assert frame.data[-1, -1].tolist() == [3, 2, 1]


def test_default_blue_bgr_frame_lands_in_the_red_channel_last() -> None:
    """Pure blue in BGR must end up with the 255 in the last (blue) RGB channel."""
    reader = _started(conftest.FakeCaptureSource())
    reader.stop()
    assert reader.frame is not None
    assert reader.frame.data[0, 0].tolist() == [0, 0, 255]


def test_missing_cv2_kills_the_capture_thread_visibly(monkeypatch: pytest.MonkeyPatch) -> None:
    """The capture and encode threads both need cv2, so without it the reader dies with the
    ImportError in `error` instead of publishing frames the policy could never receive."""
    monkeypatch.setitem(sys.modules, "cv2", None)
    reader = _reader(conftest.FakeCaptureSource())

    reader.start()
    assert _wait_until(lambda: not reader.is_alive())
    reader.stop()

    assert isinstance(reader.error, ImportError)
    assert reader.frame is None


def test_rotate_90_turns_the_frame_clockwise() -> None:
    """The configured rotation is applied at capture; 90° clockwise moves (0, 0) to (0, H-1)."""
    rows, cols, _ = conftest.FRAME_SHAPE
    reader = _started(conftest.FakeCaptureSource(_marked_frame()), rotate=90)
    reader.stop()
    frame = reader.frame
    assert frame is not None

    assert frame.data.shape == (cols, rows, 3)
    assert frame.data[0, rows - 1].tolist() == [255, 255, 255]
    assert int(frame.data.sum()) == 3 * 255
    assert frame.data.flags["C_CONTIGUOUS"]


def test_rotate_180_moves_the_corner_pixel_to_the_opposite_corner() -> None:
    """180° keeps the shape and mirrors both axes."""
    rows, cols, _ = conftest.FRAME_SHAPE
    reader = _started(conftest.FakeCaptureSource(_marked_frame()), rotate=180)
    reader.stop()
    frame = reader.frame
    assert frame is not None

    assert frame.data.shape == (rows, cols, 3)
    assert frame.data[rows - 1, cols - 1].tolist() == [255, 255, 255]
    assert int(frame.data.sum()) == 3 * 255


def test_rotate_270_turns_the_frame_counter_clockwise() -> None:
    """270° clockwise (90° counter-clockwise) moves (0, 0) to (W-1, 0)."""
    rows, cols, _ = conftest.FRAME_SHAPE
    reader = _started(conftest.FakeCaptureSource(_marked_frame()), rotate=270)
    reader.stop()
    frame = reader.frame
    assert frame is not None

    assert frame.data.shape == (cols, rows, 3)
    assert frame.data[cols - 1, 0].tolist() == [255, 255, 255]
    assert int(frame.data.sum()) == 3 * 255


def test_rotate_0_leaves_the_frame_untouched() -> None:
    """Rotation 0 is the identity, still RGB-converted."""
    reader = _started(conftest.FakeCaptureSource(_marked_frame()), rotate=0)
    reader.stop()
    assert reader.frame is not None
    assert reader.frame.data.shape == conftest.FRAME_SHAPE
    assert reader.frame.data[0, 0].tolist() == [255, 255, 255]


def test_constructor_rejects_a_rotation_that_is_not_a_quarter_turn(
    fake_capture_source: conftest.FakeCaptureSource,
) -> None:
    """Rotation is 0/90/180/270; anything else is a CameraError at construction."""
    with pytest.raises(errors.CameraError, match="rotate=45"):
        cameras.CameraReader("overhead", fake_capture_source, STALE_AFTER_S, rotate=45)


def test_constructor_rejects_a_non_positive_stale_after(fake_capture_source: conftest.FakeCaptureSource) -> None:
    """Stale_after_s is a positive age; zero would mark every frame stale."""
    with pytest.raises(errors.CameraError, match="stale_after_s"):
        cameras.CameraReader("overhead", fake_capture_source, 0.0)


@pytest.mark.parametrize(
    "keywords, message",
    [
        ({"obs_period_s": -0.2}, "obs_period_s=-0.2"),
        ({"obs_period_s": 0.01, "expected_fps": 30.0}, "obs_period_s=0.01"),
    ],
)
def test_constructor_rejects_a_period_it_cannot_sample(
    fake_capture_source: conftest.FakeCaptureSource, keywords: dict[str, Any], message: str
) -> None:
    with pytest.raises(errors.CameraError, match=message):
        cameras.CameraReader("overhead", fake_capture_source, STALE_AFTER_S, **keywords)


# --- timestamps and staleness -----------------------------------------------------------------


def test_timestamps_are_monotonic_capture_times(fake_capture_source: conftest.FakeCaptureSource) -> None:
    """The thread stamps timestamp_s at capture from the monotonic clock; successive frames never go back."""
    before_s = time.monotonic()
    reader = _started(fake_capture_source)
    seen: list[types.Frame] = []
    try:
        deadline_s = time.monotonic() + WAIT_TIMEOUT_S
        while len(seen) < 5 and time.monotonic() < deadline_s:
            frame = reader.frame
            if frame is not None and (not seen or frame is not seen[-1]):
                seen.append(frame)
            time.sleep(0.001)
    finally:
        reader.stop()
    after_s = time.monotonic()

    assert len(seen) == 5
    stamps = [frame.timestamp_s for frame in seen]
    assert stamps == sorted(stamps)
    assert before_s <= stamps[0] and stamps[-1] <= after_s


def test_is_stale_without_a_frame(fake_capture_source: conftest.FakeCaptureSource) -> None:
    """No frame yet counts as stale, whatever the clock says."""
    reader = _reader(fake_capture_source)

    assert reader.is_stale(0.0)
    assert reader.is_stale(1e9)


def test_is_stale_compares_frame_age_against_stale_after_s(fake_capture_source: conftest.FakeCaptureSource) -> None:
    """Camera frame older than stale_after_s fails the tick; the comparison uses the caller's now_s."""
    reader = _reader(fake_capture_source)
    # The slot is a public attribute. The frame sits at t=0 so the equal-age boundary is exact:
    # `STALE_AFTER_S - 0.0 == STALE_AFTER_S`, whereas `10.0 + 0.2 - 10.0` is a hair under 0.2 and
    # would let `>=` pass for `>`.
    reader.frame = types.Frame(data=np.zeros((1, 1, 3), dtype=np.uint8), camera="overhead", timestamp_s=0.0)

    assert not reader.is_stale(0.0)
    assert not reader.is_stale(STALE_AFTER_S / 2)
    assert not reader.is_stale(STALE_AFTER_S)
    assert reader.is_stale(STALE_AFTER_S + 1e-6)
    assert reader.is_stale(1.0)


def test_is_stale_uses_a_manual_clock(fake_capture_source: conftest.FakeCaptureSource) -> None:
    """checklist rule 11: staleness takes now_s in, so a ManualClock decides it, not wall time."""
    clock = conftest.ManualClock(now_s=100.0)
    reader = _reader(fake_capture_source)
    reader.frame = types.Frame(data=np.zeros((1, 1, 3), dtype=np.uint8), camera="overhead", timestamp_s=clock())

    assert not reader.is_stale(clock())
    clock.now_s += STALE_AFTER_S * 2
    assert reader.is_stale(clock())


def test_a_live_reader_is_fresh_at_the_current_monotonic_time(fake_capture_source: conftest.FakeCaptureSource) -> None:
    """A healthy capture thread keeps the frame younger than stale_after_s."""
    reader = _started(fake_capture_source)
    try:
        assert not reader.is_stale(time.monotonic())
    finally:
        reader.stop()


# --- failure ----------------------------------------------------------------------------------


def test_source_failure_sets_error_and_kills_the_thread(fake_capture_source: conftest.FakeCaptureSource) -> None:
    """A capture thread that dies is visible through error and is_alive(); the thread exits."""
    reader = _started(fake_capture_source)
    last = reader.frame

    fake_capture_source.fail()

    assert _wait_until(lambda: not reader.is_alive())
    assert isinstance(reader.error, errors.CameraError)
    assert "overhead" in str(reader.error)
    assert _wait_until(lambda: not any(thread.name == "camera-overhead" for thread in threading.enumerate()))
    # The last good frame is still there for anyone holding the snapshot, but the loop sees a dead reader.
    assert reader.frame is not None
    assert reader.frame.timestamp_s >= (last.timestamp_s if last is not None else 0.0)
    reader.stop()
    assert fake_capture_source.released


def test_failure_on_the_first_read_leaves_the_slot_empty() -> None:
    """A camera that never publishes a first frame stays at frame None with error set."""
    source = conftest.FakeCaptureSource()
    source.fail()
    reader = _reader(source)

    reader.start()
    assert _wait_until(lambda: not reader.is_alive())
    reader.stop()

    assert reader.frame is None
    assert isinstance(reader.error, errors.CameraError)
    assert reader.is_stale(time.monotonic())


def test_malformed_frame_from_the_source_is_a_camera_error() -> None:
    """The reader only publishes [H, W, 3]; a grayscale frame kills the thread instead of leaking."""
    source = conftest.FakeCaptureSource(np.zeros((8, 8), dtype=np.uint8))
    reader = _reader(source)

    reader.start()
    assert _wait_until(lambda: not reader.is_alive())
    reader.stop()

    assert isinstance(reader.error, errors.CameraError)
    assert "shape" in str(reader.error)
    assert reader.frame is None


def test_source_exception_propagates_into_error_unchanged() -> None:
    """checklist rule 4: a driver exception is stored as-is at the thread boundary, never wrapped."""
    original = OSError("device unplugged")
    source = conftest.FakeCaptureSource()

    def read() -> tuple[bool, np.ndarray | None]:
        raise original

    source.read = read  # type: ignore[method-assign]
    reader = _reader(source)

    reader.start()
    assert _wait_until(lambda: not reader.is_alive())
    reader.stop()

    assert reader.error is original


def test_restart_after_failure_clears_error(fake_capture_source: conftest.FakeCaptureSource) -> None:
    """`error` describes the current thread; a restart begins healthy."""
    reader = _started(fake_capture_source)
    fake_capture_source.fail()
    assert _wait_until(lambda: not reader.is_alive())
    reader.stop()
    assert reader.error is not None

    fake_capture_source.ok = True
    reader.start()
    try:
        assert reader.error is None
        assert _wait_until(reader.is_alive)
    finally:
        reader.stop()


def test_camera_error_is_a_robots_error() -> None:
    """Every package exception derives from RobotsError."""
    assert issubclass(errors.CameraError, errors.RobotsError)


# --- from_config ------------------------------------------------------------------------------


class _FakeCapture:
    """Stand-in for `cv2.VideoCapture`: records the device and every property set."""

    def __init__(self, device: int | str, opened: bool) -> None:
        self.device = device
        self.opened = opened
        self.props: dict[int, float] = {}

    def isOpened(self) -> bool:  # noqa: N802 - cv2's own spelling
        return self.opened

    def set(self, prop: int, value: float) -> bool:
        self.props[prop] = value
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        return True, conftest.make_bgr_frame()

    def release(self) -> None:
        self.opened = False


def _fourcc(*code: str) -> int:
    """`cv2.VideoWriter.fourcc`: four characters packed little-endian."""
    return sum(ord(char) << (8 * index) for index, char in enumerate(code))


## MJPG as the fake packs it, what from_config() asks of every OpenCV device before its size.
MJPG = _fourcc(*"MJPG")


def _install_fake_cv2(monkeypatch: pytest.MonkeyPatch, opened: bool) -> list[_FakeCapture]:
    """Puts a minimal cv2 in sys.modules and returns the list every VideoCapture is appended to."""
    captures: list[_FakeCapture] = []
    fake_cv2 = pytypes.ModuleType("cv2")

    def video_capture(device: int | str) -> _FakeCapture:
        capture = _FakeCapture(device, opened)
        captures.append(capture)
        return capture

    fake_cv2.VideoCapture = video_capture  # type: ignore[attr-defined]
    fake_cv2.CAP_PROP_FRAME_WIDTH = 3  # type: ignore[attr-defined]
    fake_cv2.CAP_PROP_FRAME_HEIGHT = 4  # type: ignore[attr-defined]
    fake_cv2.CAP_PROP_FPS = 5  # type: ignore[attr-defined]
    fake_cv2.CAP_PROP_FOURCC = 6  # type: ignore[attr-defined]
    fake_cv2.VideoWriter = pytypes.SimpleNamespace(fourcc=_fourcc)  # type: ignore[attr-defined]
    fake_cv2.COLOR_BGR2RGB = 4  # type: ignore[attr-defined]
    fake_cv2.COLOR_RGB2BGR = 4  # type: ignore[attr-defined]
    fake_cv2.IMWRITE_JPEG_QUALITY = 1  # type: ignore[attr-defined]
    fake_cv2.INTER_AREA = 3  # type: ignore[attr-defined]
    fake_cv2.cvtColor = lambda bgr, _code: bgr[..., ::-1].copy()  # type: ignore[attr-defined]
    # A rotated frame is not at the policy size; the stub only has to return one of the size asked for.
    fake_cv2.resize = lambda _img, dsize, interpolation: np.zeros((dsize[1], dsize[0], 3), dtype=np.uint8)  # type: ignore[attr-defined]
    # The encode thread encodes the policy's frame; the bytes are never decoded in these tests.
    fake_cv2.imencode = lambda _ext, _img, _params: (True, np.frombuffer(b"jpeg", dtype=np.uint8))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    return captures


def test_from_config_raises_camera_error_when_the_device_cannot_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """From_config raises CameraError naming the camera if the device cannot be opened."""
    _install_fake_cv2(monkeypatch, opened=False)
    cfg = config.CameraConfig(name="wrist_l", device="/dev/video9", width=640, height=480)

    with pytest.raises(errors.CameraError, match="wrist_l.*/dev/video9"):
        cameras.CameraReader.from_config(cfg, STALE_AFTER_S, **FROM_CONFIG_ENCODING)


def test_from_config_opens_the_device_and_requests_size_and_fps(monkeypatch: pytest.MonkeyPatch) -> None:
    """From_config opens cv2.VideoCapture(device), asks for MJPG (V4L2's default YUYV cannot carry
    720p30 over USB 2), sets width/height/fps and keeps the rotation."""
    captures = _install_fake_cv2(monkeypatch, opened=True)
    cfg = config.CameraConfig(name="overhead", device=2, width=640, height=480, fps=15, rotate=90)

    reader = cameras.CameraReader.from_config(cfg, STALE_AFTER_S, **FROM_CONFIG_ENCODING)

    assert reader.name == "overhead"
    assert not reader.is_alive()
    assert len(captures) == 1
    assert captures[0].device == 2
    assert captures[0].props == {6: MJPG, 3: 640, 4: 480, 5: 15}
    assert list(captures[0].props)[0] == 6, "the format before the size: a size set first may pick a YUYV mode"
    # The rotation from the config is live: a started reader turns the frame.
    reader.start()
    try:
        assert _wait_until(lambda: reader.frame is not None)
        assert reader.frame is not None
        rows, cols, _ = conftest.FRAME_SHAPE
        assert reader.frame.data.shape == (cols, rows, 3)
    finally:
        reader.stop()
    assert not captures[0].opened


def test_from_config_opens_a_camera_class_of_your_own(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """A camera's `source` names a class of the operator's own (an SDK camera OpenCV cannot open): from_config builds
    it from the camera's config and reads it as it reads an OpenCV device, rotation and all; stop() releases it."""
    _install_fake_cv2(monkeypatch, opened=True)
    (tmp_path / "my_cams.py").write_text(
        "import numpy as np\n"
        "released = []\n"
        "class Cam:\n"
        "    def __init__(self, cfg):\n"
        "        self.cfg = cfg\n"
        "    def read(self):\n"
        "        return True, np.zeros((self.cfg.height, self.cfg.width, 3), dtype=np.uint8)\n"
        "    def release(self):\n"
        "        released.append(self.cfg.name)\n"
    )
    cfg = config.CameraConfig(name="scene", source=f"{tmp_path / 'my_cams.py'}:Cam", width=64, height=48, rotate=90)

    reader = cameras.CameraReader.from_config(cfg, STALE_AFTER_S, **FROM_CONFIG_ENCODING)
    reader.start()
    try:
        assert _wait_until(lambda: reader.frame is not None)
        assert reader.frame is not None and reader.frame.data.shape == (64, 48, 3)
    finally:
        reader.stop()
    assert sys.modules["_robots_camera_my_cams"].released == ["scene"]


def test_module_does_not_import_cv2_at_load() -> None:
    """Cv2 is reached lazily, so the module's namespace carries no cv2 binding."""
    assert "cv2" not in vars(cameras)


@pytest.mark.parametrize(
    "fps, obs_fps, period_s",
    [(30, 5.0, 0.2), (30, 100.0, 0.0), (30, 7.0, 1 / 7), (15, 5.0, 0.2), (30, 30.0, 0.0)],
)
def test_from_config_spaces_the_encoded_frames_by_the_training_rate_not_the_camera_fps(
    monkeypatch: pytest.MonkeyPatch, fps: int, obs_fps: float, period_s: float
) -> None:
    _install_fake_cv2(monkeypatch, opened=True)
    cfg = config.CameraConfig(name="top", device=0, width=640, height=360, fps=fps)

    reader = cameras.CameraReader.from_config(cfg, STALE_AFTER_S, obs_fps=obs_fps, policy_image_size=(640, 360))

    assert reader.obs_period_s == pytest.approx(period_s)


# --- RealSenseSource ---------------------------------------------------------------------------

## The serial of a D405, as a station would write it.
REALSENSE_SERIAL = "260322275394"


class _FakeColorFrame:
    """Stand-in for a librealsense video frame: get_data() hands out the BGR buffer."""

    def __init__(self, bgr: np.ndarray) -> None:
        self.bgr = bgr

    def get_data(self) -> np.ndarray:
        return self.bgr


class _FakeFrameset:
    """Stand-in for the composite frame wait_for_frames() returns."""

    def __init__(self, bgr: np.ndarray) -> None:
        self.bgr = bgr

    def get_color_frame(self) -> _FakeColorFrame:
        return _FakeColorFrame(self.bgr)


class _FakeStreamConfig:
    """Stand-in for `rs.config`: records the device and the streams enabled on it."""

    def __init__(self) -> None:
        self.serial: str | None = None
        self.streams: list[tuple[object, ...]] = []

    def enable_device(self, serial: str) -> None:
        self.serial = serial

    def enable_stream(self, *args: object) -> None:
        self.streams.append(args)


class _FakePipeline:
    """Stand-in for `rs.pipeline`: streams one BGR frame per wait until told to time out."""

    def __init__(self) -> None:
        self.started_with: _FakeStreamConfig | None = None
        self.is_timing_out = False
        self.is_stopped_already = False
        self.timeouts_ms: list[int] = []
        self.stop_count = 0

    def start(self, stream: _FakeStreamConfig) -> object:
        self.started_with = stream
        return object()

    def wait_for_frames(self, timeout_ms: int) -> _FakeFrameset:
        self.timeouts_ms.append(timeout_ms)
        time.sleep(conftest.READ_DELAY_S)
        if self.is_timing_out:
            raise RuntimeError(f"Frame didn't arrive within {timeout_ms}")
        return _FakeFrameset(conftest.make_bgr_frame())

    def stop(self) -> None:
        self.stop_count += 1
        if self.is_stopped_already:
            raise RuntimeError("stop() cannot be called before start()")


def _install_fake_realsense(monkeypatch: pytest.MonkeyPatch) -> list[_FakePipeline]:
    """Puts a minimal pyrealsense2 in sys.modules and returns the list every pipeline is appended to."""
    pipelines: list[_FakePipeline] = []
    fake_rs = pytypes.ModuleType("pyrealsense2")

    def pipeline() -> _FakePipeline:
        created = _FakePipeline()
        pipelines.append(created)
        return created

    fake_rs.pipeline = pipeline  # type: ignore[attr-defined]
    fake_rs.config = _FakeStreamConfig  # type: ignore[attr-defined]
    fake_rs.stream = pytypes.SimpleNamespace(color="color")  # type: ignore[attr-defined]
    fake_rs.format = pytypes.SimpleNamespace(bgr8="bgr8")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake_rs)
    return pipelines


def test_realsense_source_starts_the_colour_stream_and_reads_a_bgr_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """The source enables the device by serial and the colour stream as bgr8; read() is the cv2 shape."""
    pipelines = _install_fake_realsense(monkeypatch)
    rows, cols, _ = conftest.FRAME_SHAPE

    source = cameras.RealSenseSource(REALSENSE_SERIAL, cols, rows, 30)
    ok, bgr = source.read()

    assert len(pipelines) == 1
    started_with = pipelines[0].started_with
    assert started_with is not None
    assert started_with.serial == REALSENSE_SERIAL
    assert started_with.streams == [("color", cols, rows, "bgr8", 30)]
    assert ok
    assert bgr is not None
    assert bgr.shape == conftest.FRAME_SHAPE
    assert bgr.dtype == np.uint8
    assert bgr[0, 0].tolist() == [255, 0, 0]


def test_realsense_source_waits_for_a_first_frame_with_a_long_timeout_then_a_few_periods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first frame after start is slow; every read() after that waits a few frame periods only."""
    pipelines = _install_fake_realsense(monkeypatch)

    source = cameras.RealSenseSource(REALSENSE_SERIAL, 640, 360, 30)
    source.read()

    expected_read_ms = int(cameras.REALSENSE_READ_TIMEOUT_PERIODS * 1000 / 30)
    assert pipelines[0].timeouts_ms == [cameras.REALSENSE_START_TIMEOUT_MS, expected_read_ms]
    assert 2 * 1000 / 30 <= expected_read_ms <= 10 * 1000 / 30


def test_realsense_timeout_reads_as_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """No frame within the timeout is `(False, None)`, the same failure a cv2 read reports."""
    pipelines = _install_fake_realsense(monkeypatch)
    source = cameras.RealSenseSource(REALSENSE_SERIAL, 640, 360, 30)

    pipelines[0].is_timing_out = True

    assert source.read() == (False, None)


def test_realsense_release_stops_the_pipeline_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Release() stops the pipeline; a second release (stop() is idempotent) does not stop it again."""
    pipelines = _install_fake_realsense(monkeypatch)
    source = cameras.RealSenseSource(REALSENSE_SERIAL, 640, 360, 30)

    source.release()
    source.release()

    assert pipelines[0].stop_count == 1


def test_realsense_first_frame_timeout_stops_the_pipeline_and_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """A camera whose first frame never arrives is let go before the error propagates, so a retry can open it."""
    pipelines = _install_fake_realsense(monkeypatch)
    fake_rs = sys.modules["pyrealsense2"]
    make_pipeline = fake_rs.pipeline

    def timing_out() -> _FakePipeline:
        created: _FakePipeline = make_pipeline()
        created.is_timing_out = True
        return created

    monkeypatch.setattr(fake_rs, "pipeline", timing_out)

    with pytest.raises(RuntimeError, match="Frame didn't arrive within"):
        cameras.RealSenseSource(REALSENSE_SERIAL, 640, 360, 30)

    assert pipelines[0].stop_count == 1


def test_realsense_release_of_a_camera_lost_from_usb_logs_and_returns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """After a device loss librealsense has stopped the pipeline already; release() logs its stop() error."""
    pipelines = _install_fake_realsense(monkeypatch)
    source = cameras.RealSenseSource(REALSENSE_SERIAL, 640, 360, 30)
    pipelines[0].is_stopped_already = True
    caplog.set_level(logging.WARNING, logger="robots.cameras")

    source.release()
    source.release()

    assert pipelines[0].stop_count == 1
    assert any("cannot be called before start()" in r.message for r in caplog.records)


def test_realsense_source_without_pyrealsense2_is_a_camera_error_naming_the_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A station without the wheel gets a CameraError that says what to install."""
    monkeypatch.setitem(sys.modules, "pyrealsense2", None)

    with pytest.raises(errors.CameraError, match="pip install pyrealsense2") as info:
        cameras.RealSenseSource(REALSENSE_SERIAL, 640, 360, 30)

    assert REALSENSE_SERIAL in str(info.value)


class _FakeRealSenseDevice:
    """Stand-in for a librealsense device in a context's list: a serial and the USB link it enumerated at."""

    def __init__(self, serial: str, usb: str | None) -> None:
        self.info = {"serial_number": serial, "usb_type_descriptor": usb}

    def supports(self, key: str) -> bool:
        return self.info.get(key) is not None

    def get_info(self, key: str) -> str:
        value = self.info[key]
        assert value is not None, "librealsense raises for an info the device does not support"
        return value


def _install_realsense_on_usb(monkeypatch: pytest.MonkeyPatch, usb: str | None, start_error: str | None) -> None:
    """A fake pyrealsense2 whose one camera enumerated at `usb` and whose pipeline start raises `start_error`."""
    _install_fake_realsense(monkeypatch)
    fake_rs = sys.modules["pyrealsense2"]
    make_pipeline = fake_rs.pipeline

    def refusing() -> _FakePipeline:
        created: _FakePipeline = make_pipeline()

        def start(stream: _FakeStreamConfig) -> object:
            raise RuntimeError(start_error)

        if start_error is not None:
            created.start = start  # type: ignore[method-assign]
        return created

    monkeypatch.setattr(fake_rs, "pipeline", refusing)
    fake_rs.camera_info = pytypes.SimpleNamespace(  # type: ignore[attr-defined]
        serial_number="serial_number", usb_type_descriptor="usb_type_descriptor", name="name"
    )
    devices = [_FakeRealSenseDevice(REALSENSE_SERIAL, usb)]
    fake_rs.context = lambda: pytypes.SimpleNamespace(query_devices=lambda: devices)  # type: ignore[attr-defined]


def test_a_realsense_on_usb2_that_cannot_open_its_mode_says_it_needs_a_usb3_port_and_cable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """librealsense's "Couldn't resolve requests" on a USB2 link names the link and the fix, same type."""
    _install_realsense_on_usb(monkeypatch, "2.1", "Couldn't resolve requests")

    with pytest.raises(RuntimeError) as info:
        cameras.RealSenseSource(REALSENSE_SERIAL, 1280, 720, 30)

    said = str(info.value)
    assert "Couldn't resolve requests" in said, "librealsense's own words stay in the message"
    assert REALSENSE_SERIAL in said
    assert "USB 2.1" in said and "1280x720 at 30 fps" in said
    assert cameras.USB3_FIX in said
    assert type(info.value) is RuntimeError, "every caller that catches RuntimeError still does"


def test_a_realsense_on_usb3_that_cannot_open_keeps_librealsense_s_error_as_it_was(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On a USB3 link the cause is something else, so nothing about USB is added."""
    _install_realsense_on_usb(monkeypatch, "3.2", "Couldn't resolve requests")

    with pytest.raises(RuntimeError) as info:
        cameras.RealSenseSource(REALSENSE_SERIAL, 1280, 720, 30)

    assert str(info.value) == "Couldn't resolve requests"


def test_usb_link_reads_the_camera_s_usb_type_and_is_none_when_librealsense_cannot_say(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The link is the device's `usb_type_descriptor`; another serial, no descriptor, or no package is None."""
    _install_realsense_on_usb(monkeypatch, "2.1", None)
    assert cameras.usb_link(REALSENSE_SERIAL) == "2.1"
    assert cameras.usb_link("000000000000") is None

    _install_realsense_on_usb(monkeypatch, None, None)
    assert cameras.usb_link(REALSENSE_SERIAL) is None

    monkeypatch.setitem(sys.modules, "pyrealsense2", None)
    assert cameras.usb_link(REALSENSE_SERIAL) is None


def test_a_realsense_that_opens_on_usb2_logs_its_link(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A camera that opened at a mode USB2 carries still says which link it is on, for the next look at the log."""
    _install_realsense_on_usb(monkeypatch, "2.1", None)
    caplog.set_level(logging.INFO, logger="robots.cameras")

    cameras.RealSenseSource(REALSENSE_SERIAL, 640, 360, 15).release()

    assert any(REALSENSE_SERIAL in r.message and "USB 2.1" in r.message for r in caplog.records)


def test_from_config_with_serial_builds_a_reader_over_a_realsense_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """From_config picks the source by which of device/serial is set; the reader is otherwise unchanged."""
    pipelines = _install_fake_realsense(monkeypatch)
    cfg = config.CameraConfig(name="top", serial=REALSENSE_SERIAL, width=640, height=360, fps=15)

    reader = cameras.CameraReader.from_config(cfg, STALE_AFTER_S, **FROM_CONFIG_ENCODING)

    assert "cv2" not in vars(cameras)
    assert len(pipelines) == 1
    reader.start()
    try:
        assert _wait_until(lambda: reader.frame is not None)
        assert reader.frame is not None
        assert reader.frame.data.shape == conftest.FRAME_SHAPE
        assert reader.frame.data[0, 0].tolist() == [0, 0, 255]
        # A camera that stops delivering kills the thread the way a failed cv2 read does.
        pipelines[0].is_timing_out = True
        assert _wait_until(lambda: not reader.is_alive())
        assert isinstance(reader.error, errors.CameraError)
        assert "no frame" in str(reader.error)
    finally:
        reader.stop()
    assert pipelines[0].stop_count == 1


def test_from_config_with_no_device_serial_or_source_is_a_camera_error() -> None:
    """Config.load never builds such a config; a hand-built one fails by name, not in cv2."""
    cfg = config.CameraConfig(name="top", width=640, height=360)

    with pytest.raises(errors.CameraError, match="'top' has no device, serial or source"):
        cameras.CameraReader.from_config(cfg, STALE_AFTER_S, **FROM_CONFIG_ENCODING)


def test_module_does_not_import_pyrealsense2_at_load() -> None:
    """Pyrealsense2 is reached lazily, so the module's namespace carries no binding for it."""
    assert "pyrealsense2" not in vars(cameras)


# --- the frames encoded for the policy -----------------------------------------------------------

## Gray level of capture `i` in the encoding tests: distinct, and far enough apart to survive a JPEG round trip.
GRAY_STEP = 15


def _gray_frames(count: int) -> list[np.ndarray]:
    """`count` solid gray frames, capture `i` at level `i * GRAY_STEP`."""
    return [conftest.make_bgr_frame(bgr=(i * GRAY_STEP,) * 3) for i in range(count)]


def _ended_after(source: conftest.FakeCaptureSource, reads: int) -> conftest.FakeCaptureSource:
    """Makes the source fail after exactly `reads` successful reads, so a capture count is exact, not a race."""
    plain_read = source.read

    def read() -> tuple[bool, np.ndarray | None]:
        if source.read_count >= reads:
            source.fail()
        return plain_read()

    source.read = read  # type: ignore[method-assign]
    return source


def _paced(source: conftest.FakeCaptureSource, period_s: float = 0.003) -> conftest.FakeCaptureSource:
    """Makes the source deliver a frame per `period_s`, like a camera: the encode thread encodes a tiny test
    frame in well under that, so every capture due lands and the encoding tests stay exact."""
    paced_read = source.read

    def read() -> tuple[bool, np.ndarray | None]:
        time.sleep(period_s)
        return paced_read()

    source.read = read  # type: ignore[method-assign]
    return source


CAMERA_FPS = 30.0


@dataclasses.dataclass
class Encoded:
    """What a reader encoded over a run of distinct gray captures.

    Attributes:
    - captures: The capture index of every frame the encode thread encoded, in order, read from its gray level.
    - frame: The last published frame, its `jpeg` replaced by the newest encoded one (a capture still being encoded
      when the last frame was published lands in a later publish, and there is none).
    """

    captures: list[int]
    frame: types.Frame


def _encoded_after(
    captures: int,
    every: int,
    policy_image_size: tuple[int, int] | None = None,
    times_s: list[float] | None = None,
) -> Encoded:
    """Runs a reader over `captures` distinct gray frames until the source ends: what it encoded, and its last
    frame."""
    source = _paced(_ended_after(conftest.FakeCaptureSource(_gray_frames(captures)), captures))
    ticks = iter(times_s if times_s is not None else [i / CAMERA_FPS for i in range(captures)])
    reader = cameras.CameraReader(
        "overhead",
        source,
        STALE_AFTER_S,
        obs_period_s=0.0 if every == 1 else every / CAMERA_FPS,
        policy_image_size=policy_image_size,
        face_blur=False,
        expected_fps=CAMERA_FPS,
        clock=lambda: next(ticks),
    )
    encoded: list[int] = []
    encode = reader._encode_sample

    def recording(cv2_: Any, blur: Any, bgr: np.ndarray) -> str:
        encoded.append(round(int(bgr[0, 0, 0]) / GRAY_STEP))
        return encode(cv2_, blur, bgr)

    reader._encode_sample = recording  # type: ignore[method-assign]
    reader.start()
    assert _wait_until(lambda: not reader.is_alive())
    # The encode thread may still be encoding the last capture due when the capture thread ends; it is done once
    # it is idle.
    assert _wait_until(lambda: reader.is_encoder_idle)
    reader.stop()
    assert source.read_count >= captures + 1
    assert reader.frame is not None and reader.encoded is not None
    jpeg, jpeg_s = reader.encoded
    return Encoded(encoded, dataclasses.replace(reader.frame, jpeg=jpeg, jpeg_timestamp_s=jpeg_s))


def _decode(entry: str) -> np.ndarray:
    """Decodes one encoded frame the way the serve does: base64, then cv2, so the pixels come back BGR."""
    decoded = cv2.imdecode(np.frombuffer(base64.b64decode(entry), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    return np.asarray(decoded)


def _capture_of(entry: str) -> int:
    """The capture index an encoded frame came from, read back from its gray level."""
    return round(int(_decode(entry)[0, 0, 0]) / GRAY_STEP)


def test_one_capture_per_period_is_encoded_and_the_frame_carries_the_newest() -> None:
    """One capture every period, the first included, is encoded for the policy, and each published
    frame carries the newest as `jpeg` with its capture time: every third capture of fourteen is 0, 3, 6, 9, 12."""
    run = _encoded_after(captures=14, every=3)

    assert run.captures == [0, 3, 6, 9, 12]
    assert isinstance(run.frame.jpeg, str) and _capture_of(run.frame.jpeg) == 12
    assert run.frame.jpeg_timestamp_s == pytest.approx(12 / CAMERA_FPS)
    # The frame itself is the last capture, at full size and RGB; the encoded frame is the policy's, not the loop's.
    assert run.frame.data[0, 0].tolist() == [13 * GRAY_STEP] * 3


def test_the_first_capture_is_encoded_before_the_first_frame_is_published() -> None:
    """A published frame always carries an encoded frame, so the first capture is encoded and the
    first publish waits for it: one capture gives capture 0, a second that is not due adds nothing, the fourth is
    due."""
    assert _encoded_after(captures=1, every=3).captures == [0]
    assert _encoded_after(captures=2, every=3).captures == [0]
    assert _encoded_after(captures=4, every=3).captures == [0, 3]


def test_period_zero_encodes_every_capture() -> None:
    run = _encoded_after(captures=5, every=1)

    assert run.captures == [0, 1, 2, 3, 4]
    assert _capture_of(run.frame.jpeg) == 4


def test_the_encoded_frames_keep_their_spacing_when_the_camera_slows() -> None:
    times_s = [i / 30 for i in range(7)] + [0.2 + 0.05 * i for i in range(1, 9)]

    run = _encoded_after(captures=15, every=6, times_s=times_s)

    assert run.captures == [0, 6, 10, 14]


def test_a_camera_that_stalls_encodes_its_first_capture_after_the_gap() -> None:
    """Nothing is encoded for the slots of a gap; the first capture after it is, and the grid goes on from there
    (the dream client's window then starts its spacing again from that frame)."""
    times_s = [i / 30 for i in range(7)] + [1.0 + i / 30 for i in range(7)]

    run = _encoded_after(captures=14, every=6, times_s=times_s)

    assert run.captures == [0, 6, 7, 13]


@pytest.mark.parametrize("fps", [30.0, 24.0, 20.0, 15.0])
def test_encoded_captures_are_the_training_spacing_apart_at_any_camera_rate(fps: float) -> None:
    clock = cameras.SampleClock(period_s=0.2, early_s=0.5 / 30.0)
    rng = np.random.default_rng(int(fps))
    captures = int(6 * fps)
    times_s = np.arange(captures) / fps + rng.uniform(-0.003, 0.003, captures)
    taken = [t for t in times_s if clock.take(float(t))]

    gaps = np.diff(taken)
    assert np.all(np.abs(gaps - 0.2) <= 1 / fps + 0.006)
    spans = np.asarray(taken[8:]) - np.asarray(taken[:-8])
    assert np.all(np.abs(spans - 1.6) <= 1 / fps + 0.006)


def test_a_steady_30_fps_camera_with_jitter_gives_every_sixth_capture() -> None:
    clock = cameras.SampleClock(period_s=0.2, early_s=0.5 / 30.0)
    rng = np.random.default_rng(3)
    times_s = np.arange(90) / 30 + rng.uniform(-0.005, 0.005, 90)

    taken = [i for i, t in enumerate(times_s) if clock.take(float(t))]

    assert taken == list(range(0, 90, 6))


def test_every_camera_encodes_on_the_same_slots_whatever_its_phase() -> None:
    """The grid is the monotonic clock's own, not each camera's first capture's, so two cameras running out of phase
    (here by 13 ms and with jitter) encode one capture per slot each, within a frame period of each other: the set
    the loop hands the dream client is one moment of the scene. A reset forgets the last slot, not the grid."""
    rng = np.random.default_rng(7)
    first, second = (cameras.SampleClock(period_s=0.2, early_s=0.5 / 30.0) for _ in range(2))
    times_a = 5.0 + np.arange(90) / 30 + rng.uniform(-0.003, 0.003, 90)
    times_b = times_a + 0.013

    taken_a = [t for t in times_a if first.take(float(t))]
    taken_b = [t for t in times_b if second.take(float(t))]

    assert len(taken_a) == len(taken_b) == 15
    assert np.all(np.abs(np.asarray(taken_a) - np.asarray(taken_b)) < 1 / 30)
    first.reset()
    assert first.take(float(times_a[-1])) and not first.take(float(times_a[-1]) + 0.01)


def test_each_capture_due_is_encoded_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Encode once per capture due, on the encode thread; publishing and reading the frame cost no
    further encodes, and captures off the grid are never encoded."""
    encodes = 0
    real_imencode = cv2.imencode

    def counting_imencode(*args: Any, **kwargs: Any) -> Any:
        nonlocal encodes
        encodes += 1
        return real_imencode(*args, **kwargs)

    monkeypatch.setattr(cv2, "imencode", counting_imencode)

    run = _encoded_after(captures=14, every=3)
    before = encodes
    _ = (run.frame.jpeg, run.frame.jpeg)

    # Captures 0, 3, 6, 9, 12 were due: five encodes for fourteen captures, none for the reads.
    assert encodes == 5
    assert encodes == before


def test_the_encoded_frame_is_downscaled_to_the_policy_size() -> None:
    """The frame is published at the capture size; the encoded frame is the policy size."""
    source = _ended_after(conftest.FakeCaptureSource(conftest.make_bgr_frame(shape=(720, 1280, 3))), 1)
    reader = cameras.CameraReader("overhead", source, STALE_AFTER_S, policy_image_size=(640, 360), face_blur=False)
    reader.start()
    assert _wait_until(lambda: not reader.is_alive())
    assert _wait_until(lambda: reader.is_encoder_idle)
    reader.stop()
    frame = reader.frame
    assert frame is not None and reader.encoded is not None

    assert frame.data.shape == (720, 1280, 3)
    assert _decode(reader.encoded[0]).shape == (360, 640, 3)
    assert _decode(frame.jpeg).shape == (360, 640, 3)


def test_start_after_stop_publishes_nothing_encoded_before_the_stop() -> None:
    """A frame encoded before a stop is a picture from before the gap; a restarted reader publishes
    only what the new thread encoded, starting with its own first capture."""
    source = _paced(_ended_after(conftest.FakeCaptureSource(_gray_frames(5)), 5))
    reader = cameras.CameraReader("overhead", source, STALE_AFTER_S, face_blur=False)
    reader.start()
    assert _wait_until(lambda: not reader.is_alive())
    assert _wait_until(lambda: reader.is_encoder_idle)
    reader.stop()
    assert reader.encoded is not None and _capture_of(reader.encoded[0]) == 4
    stopped_s = time.monotonic()

    source.ok = True
    source.read_count = 3  # Two more reads before the wrapper ends the stream again.
    reader.start()
    assert reader.encoded is None, "nothing from before the stop survives a start"
    assert _wait_until(lambda: not reader.is_alive())
    assert _wait_until(lambda: reader.is_encoder_idle)
    reader.stop()

    assert reader.frame is not None and reader.encoded is not None
    assert reader.frame.jpeg_timestamp_s > stopped_s and reader.encoded[1] > stopped_s


# --- encode_jpeg ---------------------------------------------------------------------------------


def _rgb(shape: tuple[int, int, int], rgb: tuple[int, int, int]) -> np.ndarray:
    frame = np.empty(shape, dtype=np.uint8)
    frame[...] = rgb
    return frame


def test_encode_jpeg_downscales_a_720p_frame_to_640x360_with_inter_area(monkeypatch: pytest.MonkeyPatch) -> None:
    """1280x720 in, 640x360 out, through `cv2.resize(..., INTER_AREA)`, the station's downscale."""
    calls: list[tuple[tuple[int, int], int]] = []
    real_resize = cv2.resize

    def spying_resize(image: np.ndarray, dsize: tuple[int, int], interpolation: int) -> np.ndarray:
        calls.append((dsize, interpolation))
        return real_resize(image, dsize, interpolation=interpolation)

    monkeypatch.setattr(cv2, "resize", spying_resize)

    jpeg = cameras.encode_jpeg(_rgb((720, 1280, 3), (255, 0, 0)), (640, 360))

    assert calls == [((640, 360), cv2.INTER_AREA)]
    decoded = _decode(base64.b64encode(jpeg).decode("ascii"))
    assert decoded.shape == (360, 640, 3)
    # Pure red RGB comes back as cv2's BGR [0, 0, 255]: the wire is BGR-encoded, as the serve expects.
    np.testing.assert_allclose(decoded[0, 0], [0, 0, 255], atol=JPEG_TOLERANCE)


def test_encode_jpeg_uses_quality_85(monkeypatch: pytest.MonkeyPatch) -> None:
    """Frames are encoded at JPEG quality 85; the constant and the cv2 call agree."""
    params: list[list[int]] = []
    real_imencode = cv2.imencode

    def spying_imencode(ext: str, image: np.ndarray, encode_params: list[int]) -> Any:
        params.append(list(encode_params))
        return real_imencode(ext, image, encode_params)

    monkeypatch.setattr(cv2, "imencode", spying_imencode)

    cameras.encode_jpeg(_rgb(conftest.FRAME_SHAPE, (0, 255, 0)))

    assert cameras.JPEG_QUALITY == 85
    assert params == [[cv2.IMWRITE_JPEG_QUALITY, 85]]


@pytest.mark.parametrize("image_size", [None, (conftest.FRAME_SHAPE[1], conftest.FRAME_SHAPE[0])])
def test_encode_jpeg_does_not_resample_a_frame_already_at_size(
    monkeypatch: pytest.MonkeyPatch, image_size: tuple[int, int] | None
) -> None:
    """No size, or the frame's own size, means no resize call at all."""

    def refusing_resize(*_args: Any, **_kwargs: Any) -> np.ndarray:
        raise AssertionError("resize called for a frame already at the policy size")

    monkeypatch.setattr(cv2, "resize", refusing_resize)
    rows, cols, _ = conftest.FRAME_SHAPE

    decoded = _decode(
        base64.b64encode(cameras.encode_jpeg(_rgb(conftest.FRAME_SHAPE, (1, 2, 3)), image_size)).decode()
    )

    assert decoded.shape == (rows, cols, 3)


def test_encode_jpeg_leaves_the_frame_untouched() -> None:
    """rule 11: the pixels handed in are never modified; the BGR flip and the resize work on copies."""
    rgb = _rgb((720, 1280, 3), (10, 20, 30))
    before = rgb.copy()

    cameras.encode_jpeg(rgb, (640, 360))

    np.testing.assert_array_equal(rgb, before)
    assert rgb.shape == (720, 1280, 3)


# --- face blur (shared.faceblur, applied at capture) ---------------------------------------------------------


def _encoded_frame(frame: np.ndarray, *, face_blur: bool) -> np.ndarray:
    """Runs a reader over one capture (blur on or off) and returns what it encoded for the policy, decoded back to
    BGR."""
    source = _ended_after(conftest.FakeCaptureSource([frame]), 1)
    reader = cameras.CameraReader("overhead", source, STALE_AFTER_S, face_blur=face_blur)
    reader.start()
    # _ended_after stops the loop by failing the read after the last frame (a CameraError), the same way the
    # other capture tests end; the frame before it was captured and published.
    assert _wait_until(lambda: not reader.is_alive())
    assert _wait_until(lambda: reader.is_encoder_idle)
    reader.stop()
    assert reader.frame is not None and reader.encoded is not None
    assert reader.frame.jpeg == reader.encoded[0], "the first publish carries the first capture's encoding"
    return _decode(reader.encoded[0])


def _has_face(bgr: np.ndarray) -> bool:
    """Whether a fresh detector still finds a face in `bgr` - what the blur must destroy."""
    from shared import faceblur

    return faceblur.FaceBlurrer()._boxes(bgr) != []


def test_face_blur_removes_the_face_before_it_is_encoded_for_the_policy() -> None:
    """A frame with a face, captured with blur on, is encoded for the policy (what is sent to the API) faceless."""
    from shared import faceblur

    face = faceblur._self_check_frame()  # a BGR frame the detector reliably finds
    assert _has_face(face), "the fixture must contain a detectable face to begin with"
    assert not _has_face(_encoded_frame(face, face_blur=True))


def test_face_blur_off_sends_the_face_through() -> None:
    """The switch really switches: with blur off the same frame is encoded with its face intact."""
    from shared import faceblur

    face = faceblur._self_check_frame()
    assert _has_face(_encoded_frame(face, face_blur=False))


class _FixedDetector:
    """A YuNet stand-in that reports one detection, `(x, y, w, h)` and `score`, on every frame."""

    def __init__(self, box: tuple[float, float, float, float], score: float) -> None:
        self._rows = np.zeros((1, 15), np.float32)
        self._rows[0, :4] = box
        self._rows[0, 14] = score

    def setInputSize(self, size: tuple[int, int]) -> None:  # noqa: N802 - cv2.FaceDetectorYN's name
        """Accepted and ignored."""

    def detect(self, frame: np.ndarray) -> tuple[int, np.ndarray]:
        """The canned detection."""
        return 0, self._rows


@pytest.mark.parametrize(
    ("bgr", "score", "blurred"),
    [
        ((170, 140, 110), 0.84, False),  # a station's toy steering wheel: grey-blue plastic below the sure-face score
        ((120, 150, 200), 0.78, True),  # a real face at a hard angle: low score, but skin-coloured
        ((120, 150, 200), 0.70, False),  # skin-coloured clutter or a hand under the score floor
        ((170, 140, 110), 0.85, True),  # a sure face under a colour cast: blurred whatever its colour
    ],
)
def test_face_blur_at_capture_leaves_a_face_like_prop_alone(
    monkeypatch: pytest.MonkeyPatch, bgr: tuple[int, int, int], score: float, blurred: bool
) -> None:
    """What the robot sends the API follows shared.faceblur's rule: below the sure-face score a detection is
    blurred only when its box is mostly skin-coloured, so a face-shaped prop is encoded as captured."""
    from shared import faceblur

    real_blurrer = faceblur.FaceBlurrer

    def stubbed() -> faceblur.FaceBlurrer:
        blurrer = real_blurrer()
        blurrer._detector = _FixedDetector((100.0, 50.0, 60.0, 60.0), score)  # type: ignore[assignment]
        blurrer._size = None
        return blurrer

    # Texture the frame so a blur shows; +-8 keeps every pixel's chroma on its side of the skin bounds.
    noise = np.random.default_rng(0).integers(-8, 9, (240, 320, 3))
    frame = np.clip(np.asarray(bgr) + noise, 0, 255).astype(np.uint8)
    raw = _encoded_frame(frame, face_blur=False)
    monkeypatch.setattr(faceblur, "FaceBlurrer", stubbed)
    entry = _encoded_frame(frame, face_blur=True)
    assert (not np.array_equal(entry, raw)) == blurred


def test_frames_are_encoded_off_the_capture_thread_and_a_slow_sample_is_replaced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first publish waits for the first encode (a frame always carries an encoded frame); after that the
    capture thread keeps publishing while the encode thread is stuck in a slow blur, each frame carries the newest
    encode finished so far, and captures due that land while one is still waiting replace it and are counted, so
    the policy skips encoded frames rather than the camera skipping frames."""
    from shared import faceblur

    release = threading.Event()
    seen: list[int] = []

    class SlowBlurrer:
        def scrub_frame(self, frame: np.ndarray) -> np.ndarray:
            seen.append(int(frame[0, 0, 0]))
            if len(seen) > 1:
                release.wait(WAIT_TIMEOUT_S)
            return frame

    monkeypatch.setattr(faceblur, "FaceBlurrer", SlowBlurrer)
    frames = _gray_frames(6)
    source = _ended_after(conftest.FakeCaptureSource(frames), 6)
    reader = cameras.CameraReader("overhead", source, STALE_AFTER_S, face_blur=True)
    reader.start()
    assert _wait_until(lambda: not reader.is_alive()), "the capture thread ran on while the blur blocked"
    assert reader.frame is not None and _capture_of(reader.frame.jpeg) == 0, "only the first encode landed"
    assert reader.samples_dropped == 3, "captures 2 to 4 were each replaced by the next while 1 was being blurred"
    release.set()
    assert _wait_until(lambda: reader.is_encoder_idle)
    assert [level // GRAY_STEP for level in seen] == [0, 1, 5], "the first, the blocked one and the last offered"
    assert reader.encoded is not None and _capture_of(reader.encoded[0]) == 5
    reader.stop()


def test_a_detector_that_will_not_load_stops_the_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail closed: if the detector cannot be built the capture thread stops rather than emit raw faces."""
    from shared import faceblur

    def refuse() -> faceblur.FaceBlurrer:
        raise faceblur.FaceBlurError("detector unavailable")

    monkeypatch.setattr(faceblur, "FaceBlurrer", refuse)
    source = _ended_after(conftest.FakeCaptureSource(_gray_frames(3)), 3)
    reader = cameras.CameraReader("overhead", source, STALE_AFTER_S, face_blur=True)
    reader.start()
    assert _wait_until(lambda: not reader.is_alive())
    reader.stop()
    assert isinstance(reader.error, faceblur.FaceBlurError)
    assert reader.frame is None  # no frame was ever published


def test_limit_opencv_threads_caps_the_pool_unless_the_environment_sizes_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenCV's default pool (every hardware thread, spinning when idle) is capped at OPENCV_THREADS; an operator's
    OPENCV_FOR_THREADS_NUM wins, and a cv2 stand-in without setNumThreads is left alone."""
    calls: list[int] = []
    fake = pytypes.SimpleNamespace(setNumThreads=calls.append)
    monkeypatch.delenv("OPENCV_FOR_THREADS_NUM", raising=False)
    cameras.limit_opencv_threads(fake)
    assert calls == [cameras.OPENCV_THREADS] == [2]
    monkeypatch.setenv("OPENCV_FOR_THREADS_NUM", "4")
    cameras.limit_opencv_threads(fake)
    assert calls == [2], "the environment sizes the pool; the cap defers to it"
    cameras.limit_opencv_threads(pytypes.SimpleNamespace())


# --- managed exposure (a YAM station's controller) -----------------------------------------------


class _GainRange:
    """`rs.option_range` for gain: the D4xx colour sensor's default and maximum."""

    default = 64.0
    max = 128.0


class _FakeColorSensor:
    """Records every option write; gain optional so a sensor without one keeps the exposure-only behaviour."""

    def __init__(self, has_gain: bool = True) -> None:
        self.writes: list[tuple[str, float]] = []
        self.has_gain = has_gain

    def supports(self, option: str) -> bool:
        return option in ("enable_auto_exposure", "exposure") or (self.has_gain and option == "gain")

    def get_option_range(self, option: str) -> _GainRange:
        assert option == "gain"
        return _GainRange()

    def set_option(self, option: str, value: float) -> None:
        self.writes.append((option, float(value)))


class _FakeDevice:
    def __init__(self, sensor: _FakeColorSensor | None) -> None:
        self._sensor = sensor

    def first_color_sensor(self) -> _FakeColorSensor:
        # The D405 has no dedicated RGB sensor and raises here; the source then scans the sensors.
        raise RuntimeError("no color sensor")

    def query_sensors(self) -> list[Any]:
        return [] if self._sensor is None else [self._sensor]


def _install_fake_realsense_with_sensor(
    monkeypatch: pytest.MonkeyPatch, sensor: _FakeColorSensor | None
) -> list[_FakePipeline]:
    """The fake pyrealsense2 whose started pipeline exposes a device with `sensor` as the colour sensor."""
    pipelines = _install_fake_realsense(monkeypatch)
    fake_rs = sys.modules["pyrealsense2"]
    fake_rs.option = pytypes.SimpleNamespace(  # type: ignore[attr-defined]
        enable_auto_exposure="enable_auto_exposure", exposure="exposure", gain="gain"
    )
    if sensor is not None:
        sensor.get_stream_profiles = lambda: [pytypes.SimpleNamespace(stream_type=lambda: "color")]  # type: ignore[attr-defined]
    profile = pytypes.SimpleNamespace(get_device=lambda: _FakeDevice(sensor))
    monkeypatch.setattr(
        _FakePipeline, "start", lambda self, stream: (setattr(self, "started_with", stream), profile)[1]
    )
    return pipelines


_TARGET = 100.0
_FPS = 30
_CEILING_US = cameras.MANAGED_CEILING_FRACTION * 1_000_000.0 / _FPS


def _managed(sensor: _FakeColorSensor, **overrides: Any) -> cameras.RealSenseSource:
    # smooth_idle is on by default; the tests of the full-speed steps ask for it off, the smooth_idle tests turn it on.
    exposure = config.ExposureConfig(target_median=_TARGET, **{"smooth_idle": False, **overrides})
    return cameras.RealSenseSource(REALSENSE_SERIAL, 640, 360, _FPS, exposure=exposure)


def _step(source: cameras.RealSenseSource, level: int) -> None:
    source._next_track_s = 0.0  # skip the half-second pacing
    source.track(np.full((360, 640, 3), level, dtype=np.uint8))


def test_managed_exposure_switches_the_firmware_ae_off_and_seeds_from_a_known_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The colour sensor's auto-exposure is switched off, exposure seeds at half the ceiling (the frame period)
    and gain at the sensor's default, never where the firmware left it."""
    sensor = _FakeColorSensor()
    _install_fake_realsense_with_sensor(monkeypatch, sensor)
    source = _managed(sensor)
    assert sensor.writes == [("enable_auto_exposure", 0.0), ("exposure", _CEILING_US / 2), ("gain", 64.0)]
    assert source.exposure == {
        "exposure_us": round(_CEILING_US / 2, 1),
        "gain": 64.0,
        "held": False,
        "target_median": _TARGET,
    }


def test_unmanaged_exposure_hands_the_sensor_back_to_the_firmware(monkeypatch: pytest.MonkeyPatch) -> None:
    """`managed: false` switches the firmware's own auto-exposure on and tracks nothing."""
    sensor = _FakeColorSensor()
    _install_fake_realsense_with_sensor(monkeypatch, sensor)
    source = cameras.RealSenseSource(REALSENSE_SERIAL, 640, 360, _FPS, exposure=config.ExposureConfig(managed=False))
    assert sensor.writes == [("enable_auto_exposure", 1.0)]
    assert source.exposure is None
    _step(source, 20)
    assert sensor.writes == [("enable_auto_exposure", 1.0)]


def test_the_seed_and_the_ceiling_honour_the_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """`seed_us` is where the controller starts and `limit_us` caps what it may reach, under the frame period."""
    sensor = _FakeColorSensor()
    _install_fake_realsense_with_sensor(monkeypatch, sensor)
    source = _managed(sensor, seed_us=5000.0, limit_us=18000.0)
    assert ("exposure", 5000.0) in sensor.writes
    for _ in range(6):
        _step(source, 20)
    assert source.exposure is not None and source.exposure["exposure_us"] == 18000.0


def test_dark_scene_climbs_exposure_to_the_ceiling_then_gain(monkeypatch: pytest.MonkeyPatch) -> None:
    sensor = _FakeColorSensor()
    _install_fake_realsense_with_sensor(monkeypatch, sensor)
    source = _managed(sensor)
    seeded = len(sensor.writes)
    for _ in range(2):
        _step(source, 60)  # far under target: a max 1.6x step each time, half the ceiling to pinned
    assert source._exposure_us == _CEILING_US
    assert source._gain == 64.0, "gain untouched while exposure had room"
    assert all(knob == "exposure" for knob, _ in sensor.writes[seeded:])
    _step(source, 60)
    assert source._exposure_us == _CEILING_US
    assert source._gain > 64.0
    assert sensor.writes[-1][0] == "gain"
    for _ in range(10):
        _step(source, 60)
    assert source._gain == 128.0, "pinned at the sensor max, nothing further to command"
    writes_at_max = len(sensor.writes)
    _step(source, 60)
    assert len(sensor.writes) == writes_at_max


def test_bright_scene_unwinds_gain_before_shortening_exposure(monkeypatch: pytest.MonkeyPatch) -> None:
    sensor = _FakeColorSensor()
    _install_fake_realsense_with_sensor(monkeypatch, sensor)
    source = _managed(sensor)
    source._exposure_us = _CEILING_US
    source._gain = 100.0
    _step(source, 255)  # clipped: back off by feel
    assert source._gain < 100.0
    assert source._exposure_us == _CEILING_US
    assert sensor.writes[-1][0] == "gain"
    for _ in range(5):
        _step(source, 255)
    assert source._gain == 64.0, "floor, never below the sensor default"
    assert source._exposure_us < _CEILING_US, "only then does exposure give"
    assert sensor.writes[-1][0] == "exposure"


def test_the_run_hold_freezes_both_knobs_and_the_deadband_holds_still(monkeypatch: pytest.MonkeyPatch) -> None:
    sensor = _FakeColorSensor()
    _install_fake_realsense_with_sensor(monkeypatch, sensor)
    source = _managed(sensor)
    source.set_exposure_hold(True)
    writes = len(sensor.writes)
    _step(source, 60)
    assert len(sensor.writes) == writes and source.exposure is not None and source.exposure["held"] is True
    source.set_exposure_hold(False)
    _step(source, int(_TARGET) + 2)  # inside the deadband: nothing moves
    assert len(sensor.writes) == writes
    _step(source, 60)
    assert len(sensor.writes) == writes + 1


def test_a_sensor_without_gain_keeps_the_exposure_only_behaviour(monkeypatch: pytest.MonkeyPatch) -> None:
    sensor = _FakeColorSensor(has_gain=False)
    _install_fake_realsense_with_sensor(monkeypatch, sensor)
    source = _managed(sensor)
    for _ in range(6):
        _step(source, 60)
    assert source._exposure_us == _CEILING_US
    assert all(knob in ("exposure", "enable_auto_exposure") for knob, _ in sensor.writes)


def test_the_reader_tracks_from_the_capture_thread_and_relays_the_hold(monkeypatch: pytest.MonkeyPatch) -> None:
    """`CameraReader` calls the source's `track()` with every frame it reads and forwards `set_exposure_hold()`;
    a cv2 source has neither and the reader's calls are no-ops."""
    sensor = _FakeColorSensor()
    _install_fake_realsense_with_sensor(monkeypatch, sensor)
    source = _managed(sensor)
    tracked: list[tuple[int, ...]] = []
    monkeypatch.setattr(source, "track", lambda bgr: tracked.append(bgr.shape))
    reader = cameras.CameraReader("top", source, STALE_AFTER_S, face_blur=False)
    reader.start()
    assert _wait_until(lambda: len(tracked) >= 3)
    reader.set_exposure_hold(True)
    assert source.exposure is not None and source.exposure["held"] is True
    assert reader.exposure == source.exposure
    reader.stop()
    plain = cameras.CameraReader("plain", conftest.FakeCaptureSource(), STALE_AFTER_S, face_blur=False)
    plain.set_exposure_hold(True)
    assert plain.exposure is None


def _backdrop_and_table(dark_share: float, dark: int = 20, bright: int = 255) -> np.ndarray:
    """A wrist camera at the home pose: the top `dark_share` of the frame a black backdrop, the rest a table."""
    frame = np.full((360, 640, 3), bright, dtype=np.uint8)
    frame[: int(360 * dark_share)] = dark
    return frame


def test_the_exposure_ceiling_leaves_the_sensor_time_to_read_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dark scene pins exposure at `MANAGED_CEILING_FRACTION` of the frame period (28.3 ms at 30 fps), never the
    whole 33.3 ms: a sensor asked to expose for the whole period stretches every frame to two and delivers 15 fps."""
    sensor = _FakeColorSensor()
    _install_fake_realsense_with_sensor(monkeypatch, sensor)
    source = _managed(sensor)
    for _ in range(8):
        _step(source, 20)
    exposures = [value for knob, value in sensor.writes if knob == "exposure"]
    assert max(exposures) == pytest.approx(0.85 * 1_000_000 / _FPS)
    assert max(exposures) < 1_000_000 / _FPS


def test_clipped_highlights_darken_the_frame_whatever_its_median(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sixty percent black backdrop and forty percent white table: the median is the backdrop, far under target,
    which on its own would brighten the table further out of range. Clipping past `MANAGED_CLIP_MAX` darkens instead,
    gain first, then exposure, until the table holds its detail."""
    sensor = _FakeColorSensor()
    _install_fake_realsense_with_sensor(monkeypatch, sensor)
    source = _managed(sensor)
    source._exposure_us, source._gain = _CEILING_US, 100.0
    frame = _backdrop_and_table(0.6)
    assert float(np.median(frame[::4, ::4])) < _TARGET, "the median alone asks for more light"
    source._next_track_s = 0.0
    source.track(frame)
    assert source._gain < 100.0 and source._exposure_us == _CEILING_US, "gain gives first"
    for _ in range(6):
        source._next_track_s = 0.0
        source.track(frame)
    assert source._gain == 64.0 and source._exposure_us < _CEILING_US, "then exposure"


def test_a_frame_clipping_at_the_margin_is_never_brightened(monkeypatch: pytest.MonkeyPatch) -> None:
    """Under the clip limit but over `MANAGED_CLIP_RESUME` with a dark median: nothing moves, so a scene sitting at
    the limit does not swing between brightening into it and darkening out of it."""
    sensor = _FakeColorSensor()
    _install_fake_realsense_with_sensor(monkeypatch, sensor)
    source = _managed(sensor)
    frame = np.full((360, 640, 3), 40, dtype=np.uint8)
    frame[:, :3] = 255  # one column of the quarter-scale view: 0.6 percent clipped
    writes = len(sensor.writes)
    _step(source, 40)
    assert len(sensor.writes) == writes + 1, "a uniformly dark frame brightens"
    writes = len(sensor.writes)
    source._next_track_s = 0.0
    source.track(frame)
    assert len(sensor.writes) == writes, "the same darkness with clipped highlights holds still"


class _FakeMonotonic:
    """A monotonic clock the test moves, standing in for `cameras.time` so the controller's pacing, its
    persistence and its settle window run on test time."""

    def __init__(self) -> None:
        self.now_s = 1000.0

    def monotonic(self) -> float:
        return self.now_s

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


def _smooth(monkeypatch: pytest.MonkeyPatch) -> tuple[cameras.RealSenseSource, _FakeColorSensor, _FakeMonotonic]:
    """A managed camera with `smooth_idle` on, its sensor, and the clock it runs on."""
    sensor = _FakeColorSensor()
    _install_fake_realsense_with_sensor(monkeypatch, sensor)
    clock = _FakeMonotonic()
    monkeypatch.setattr(cameras, "time", clock)
    return _managed(sensor, smooth_idle=True), sensor, clock


def _track_for(
    source: cameras.RealSenseSource, clock: _FakeMonotonic, level: int | np.ndarray, seconds: float
) -> None:
    """Feeds one frame per controller period for `seconds` of test time: a flat `level`, or a whole frame."""
    frame = level if isinstance(level, np.ndarray) else np.full((360, 640, 3), level, dtype=np.uint8)
    for _ in range(round(seconds / cameras.MANAGED_PERIOD_S)):
        source.track(frame)
        clock.now_s += cameras.MANAGED_PERIOD_S


def _knob_ratios(sensor: _FakeColorSensor, since: int) -> list[float]:
    """Each write's factor over the previous write of the same knob, from write `since` on."""
    last: dict[str, float] = {}
    ratios = []
    for index, (knob, value) in enumerate(sensor.writes):
        if index >= since and knob in last:
            ratios.append(value / last[knob])
        last[knob] = value
    return ratios


def test_smooth_idle_settles_at_full_speed_after_the_camera_opens(monkeypatch: pytest.MonkeyPatch) -> None:
    """Right after the camera opens, `smooth_idle` steps at full speed with no wait, as the controller always did,
    so `robots run`'s two-second settle before a run still lands on target."""
    source, sensor, clock = _smooth(monkeypatch)
    seeded = len(sensor.writes)
    _track_for(source, clock, 60, 0.5)
    assert _knob_ratios(sensor, seeded) == [pytest.approx(cameras.MANAGED_MAX_STEP)]


def test_smooth_idle_ignores_a_change_shorter_than_the_persistence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settled, a scene darker or clipping for less than `MANAGED_PERSIST_S` (a hand past the lens) writes nothing,
    and a median within `MANAGED_IDLE_DEADBAND` of the target writes nothing however long it stays."""
    source, sensor, clock = _smooth(monkeypatch)
    _track_for(source, clock, int(_TARGET), 0.5)  # on target: settled
    writes = len(sensor.writes)
    _track_for(source, clock, 40, cameras.MANAGED_PERSIST_S - cameras.MANAGED_PERIOD_S)
    _track_for(source, clock, int(_TARGET), 0.5)
    _track_for(source, clock, 255, cameras.MANAGED_PERSIST_S - cameras.MANAGED_PERIOD_S)
    _track_for(source, clock, int(_TARGET) - 7, 30.0)
    assert len(sensor.writes) == writes


def test_smooth_idle_follows_lasting_drift_in_small_steps_back_to_the_deadband(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Settled, a scene off target the same way for `MANAGED_PERSIST_S` is corrected at most `MANAGED_SLOW_STEP` per
    step, and the correction runs until the median is back inside `MANAGED_DEADBAND`, not just the idle one."""
    source, sensor, clock = _smooth(monkeypatch)
    _track_for(source, clock, int(_TARGET), 0.5)
    writes = len(sensor.writes)
    _track_for(source, clock, 60, cameras.MANAGED_PERSIST_S)
    assert len(sensor.writes) == writes, "nothing until the drift has lasted"
    _track_for(source, clock, 60, 2.0)
    ratios = _knob_ratios(sensor, writes)
    assert ratios and all(1.0 < ratio <= cameras.MANAGED_SLOW_STEP + 1e-9 for ratio in ratios)
    writes = len(sensor.writes)
    _track_for(source, clock, int(_TARGET) - 5, 1.0)  # inside the idle deadband, outside the tight one: still going
    assert len(sensor.writes) > writes
    writes = len(sensor.writes)
    _track_for(source, clock, int(_TARGET) - 2, 0.5)  # back inside: stops
    _track_for(source, clock, int(_TARGET) - 5, 30.0)  # and the idle deadband holds it still from here
    assert len(sensor.writes) == writes


def test_smooth_idle_darkens_lasting_clipping_slowly_and_stops_on_an_overshoot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Settled, highlights clipping for `MANAGED_PERSIST_S` darken by at most `1 / MANAGED_SLOW_STEP` a step, never
    the full-speed `MANAGED_CLIPPED_RATIO`; a frame that then wants brightening stops the correction and has to last
    again before anything moves the other way."""
    source, sensor, clock = _smooth(monkeypatch)
    source._exposure_us, source._gain = _CEILING_US / 2, 64.0
    _track_for(source, clock, int(_TARGET), 0.5)
    writes = len(sensor.writes)
    _track_for(source, clock, _backdrop_and_table(0.6), cameras.MANAGED_PERSIST_S + 2.0)
    ratios = _knob_ratios(sensor, writes)
    assert ratios and all(1.0 / cameras.MANAGED_SLOW_STEP - 1e-9 <= ratio < 1.0 for ratio in ratios)
    writes = len(sensor.writes)
    _track_for(source, clock, 60, cameras.MANAGED_PERSIST_S - cameras.MANAGED_PERIOD_S)
    assert len(sensor.writes) == writes


def test_smooth_idle_stops_full_speed_steps_when_the_settle_window_ends(monkeypatch: pytest.MonkeyPatch) -> None:
    """A camera that never finds its target (a scene the knobs cannot reach, or one that keeps changing) stops taking
    full-speed steps after `MANAGED_SETTLE_S`; past it every step is a slow one."""
    source, sensor, clock = _smooth(monkeypatch)
    source._exposure_us = cameras.MANAGED_MIN_US
    frames = [np.full((360, 640, 3), level, dtype=np.uint8) for level in (40, 160)]
    for index in range(round(cameras.MANAGED_SETTLE_S / cameras.MANAGED_PERIOD_S)):
        source.track(frames[index % 2])
        clock.now_s += cameras.MANAGED_PERIOD_S
    writes = len(sensor.writes)
    _track_for(source, clock, 40, 20.0)
    ratios = _knob_ratios(sensor, writes)
    assert ratios and max(ratios) <= cameras.MANAGED_SLOW_STEP + 1e-9


def test_a_camera_under_its_configured_rate_is_measured_and_logged_once(
    fake_capture_source: conftest.FakeCaptureSource, caplog: pytest.LogCaptureFixture
) -> None:
    """`capture_fps` is the rate over the last `CAPTURE_RATE_WINDOW_S`; a camera under `CAPTURE_RATE_WARN_FRACTION`
    of its configured rate is logged at WARNING once, and its recovery at INFO."""
    caplog.set_level(logging.INFO, logger="robots.cameras")
    reader = cameras.CameraReader("wrist", fake_capture_source, STALE_AFTER_S, face_blur=False, expected_fps=30)
    now = 1000.0
    for _ in range(2):
        for _ in range(int(15 * cameras.CAPTURE_RATE_WINDOW_S) + 1):
            reader._measure_rate(now)
            now += 1 / 15
    assert reader.capture_fps == pytest.approx(15, abs=0.5)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "delivers" in r.message]
    assert len(warnings) == 1 and "15.0 fps of the 30 configured" in warnings[0].message
    for _ in range(int(30 * cameras.CAPTURE_RATE_WINDOW_S) + 2):
        reader._measure_rate(now)
        now += 1 / 30
    assert reader.capture_fps == pytest.approx(30, abs=0.5)
    assert any("back at" in r.message for r in caplog.records)


def test_a_camera_of_another_shape_is_said_to_look_stretched() -> None:
    assert cameras.shape_warning(1280, 720, (640, 360)) is None
    assert cameras.shape_warning(1920, 1080, (640, 360)) is None
    assert cameras.shape_warning(640, 480, (640, 360)) == (
        "This camera sends 640x480 (4:3) pictures, which is fine for setup: what matters is that the model gets 16:9 "
        "frames, cropped rather than squished. Running the robot squishes these to 640x360, so the model sees "
        "everything 1.33x wider; a 16:9 camera size such as 1280x720 when recording and running keeps it unstretched, "
        "as does cropping to 16:9 yourself if you send your own frames."
    )
    assert "2.00x taller" in (cameras.shape_warning(2560, 720, (640, 360)) or "")


def test_a_reader_says_once_that_the_model_will_see_its_frames_stretched(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="robots.cameras")
    source = conftest.FakeCaptureSource()  # 64x48, 4:3
    reader = cameras.CameraReader("overhead", source, STALE_AFTER_S, policy_image_size=(640, 360), face_blur=False)
    reader.start()
    assert _wait_until(lambda: source.read_count > 5)
    reader.stop()
    said = [r.message for r in caplog.records if "pictures" in r.message]
    assert said == [f"camera 'overhead': {cameras.shape_warning(64, 48, (640, 360))}"]
