"""The setup wizard's cameras (`robots.setup_cameras`): the operator's dataset, our samples, the swap check, and the
frame as the model gets it.

What is pinned: a LeRobot v2.1 dataset is read for the first frame of any episode of any camera, from AV1 (what
LeRobot records by default, which OpenCV cannot decode) and from H.264; a dataset of another version, or without
videos or episodes, is refused saying what is missing; the first guess of which key is which role follows the words
in the keys; our shipped samples cover every kind of camera; two wrist cameras given each other's roles are found by
their pictures and a correct pair is not; the live feed holds a camera open by role and the model's view is the
policy size; a camera that stops while held is opened again; a fine-tune's own training data is read from its zip by
byte ranges and becomes the reference: the account's newest fine-tune on the robot's embodiment.
"""

import dataclasses
import datetime
import http.server
import json
import pathlib
import sys
import threading
import time
import zipfile
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest
import setup_fakes
from dream import credentials
from dream.errors import DreamServerError
from setup_fakes import FakeCameras, FakeReader, sample_rgb

from robots import cameras, config, errors, setup_cameras, setup_wizard

av = pytest.importorskip("av", reason="the dataset's videos are read with PyAV")

## The dataset's cameras, by LeRobot feature name.
KEYS = ("observation.images.top", "observation.images.left", "observation.images.right")


def write_video(path: pathlib.Path, codec: str, colour: tuple[int, int, int]) -> None:
    """A three-frame 64x48 video of one colour."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream(codec, rate=30)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for _ in range(3):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), colour, dtype=np.uint8), format="rgb24")
            container.mux(stream.encode(frame))
        container.mux(stream.encode())


def write_dataset(root: pathlib.Path, codec: str, version: str = "v2.1", episodes: int = 2) -> pathlib.Path:
    """A LeRobot dataset with `episodes` episodes, each camera's video a colour of its own per episode."""
    info = {
        "codebase_version": version,
        "chunks_size": 1000,
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {key: {"dtype": "video", "shape": [48, 64, 3]} for key in KEYS}
        | {"action": {"dtype": "float32", "shape": [14]}},
    }
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    lines = [json.dumps({"episode_index": i, "length": 3}) for i in range(episodes)]
    (root / "meta" / "episodes.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for episode in range(episodes):
        for k, key in enumerate(KEYS):
            colour = (40 + 80 * k, 200 - 100 * episode, 60)
            write_video(root / f"videos/chunk-000/{key}/episode_{episode:06d}.mp4", codec, colour)
    return root


@pytest.mark.parametrize("codec", ["libsvtav1", "libx264"])
def test_a_lerobot_dataset_gives_the_first_frame_of_any_episode_and_camera(tmp_path: pathlib.Path, codec: str) -> None:
    if codec not in av.codecs_available:
        pytest.skip(f"this PyAV has no {codec} encoder")
    dataset = setup_cameras.Dataset(write_dataset(tmp_path / "ds", codec))
    assert dataset.video_keys == list(KEYS) and dataset.episodes == [0, 1]
    first = setup_cameras.decode_jpeg(dataset.first_frame("observation.images.left", 0))
    second = setup_cameras.decode_jpeg(dataset.first_frame("observation.images.left", 1))
    np.testing.assert_allclose(first.reshape(-1, 3).mean(axis=0), (120, 200, 60), atol=12)
    np.testing.assert_allclose(second.reshape(-1, 3).mean(axis=0), (120, 100, 60), atol=12)
    wrapped = dataset.first_frame("observation.images.left", 2)
    assert wrapped == dataset.first_frame("observation.images.left", 0)
    with pytest.raises(ValueError, match="no camera"):
        dataset.first_frame("observation.images.wrist", 0)


def test_a_dataset_that_is_not_lerobot_v21_says_what_is_missing(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match="no meta/info.json"):
        setup_cameras.Dataset(tmp_path)
    old = write_dataset(tmp_path / "old", "libx264", version="v2.0", episodes=1)
    with pytest.raises(ValueError, match="v2.0 dataset"):
        setup_cameras.Dataset(old)
    empty = write_dataset(tmp_path / "empty", "libx264", episodes=1)
    (empty / "meta" / "episodes.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="lists no episode"):
        setup_cameras.Dataset(empty)
    gone = write_dataset(tmp_path / "gone", "libx264", episodes=1)
    (gone / "videos/chunk-000/observation.images.top/episode_000000.mp4").unlink()
    with pytest.raises(ValueError, match="has no video"):
        setup_cameras.Dataset(gone).first_frame("observation.images.top", 0)


def test_the_first_guess_of_keys_follows_their_words_and_a_camera_of_no_kind_gets_none(tmp_path: pathlib.Path) -> None:
    dataset = setup_cameras.Dataset(write_dataset(tmp_path / "ds", "libx264", episodes=1))
    yam = setup_cameras.camera_roles(("top", "left", "right"))
    assert dataset.guess_keys(yam) == dict(zip(("top", "left", "right"), KEYS, strict=True))
    makers = setup_cameras.camera_roles(("scene", "wrist_l", "wrist_r"))
    assert dataset.guess_keys(makers) == dict(zip(("scene", "wrist_l", "wrist_r"), KEYS, strict=True))
    mixed = setup_cameras.camera_roles(("cam_high", "wrist_l", "belly"))
    assert dataset.guess_keys(mixed) == {"cam_high": None, "wrist_l": KEYS[1], "belly": None}


def test_every_camera_name_gets_a_kind_or_none_and_labels_and_none_is_an_error() -> None:
    """The one place a role's kind and its labels are decided; the page shows them as sent. An embodiment's own names
    (LeRobot's cam_high, cam_left_wrist, cam_right_wrist) are labelled by themselves; a second head is a head."""
    cams = setup_cameras.camera_roles(("top", "wrist_l", "cam_right_wrist", "overhead"))
    assert [dataclasses.astuple(cam) for cam in cams.values()] == [
        ("top", "head", "Head camera (top)", "Head"),
        ("wrist_l", "wrist_left", "Left wrist camera (wrist_l)", "Left"),
        ("cam_right_wrist", None, "Camera cam_right_wrist", "cam_right_wrist"),
        ("overhead", "head", "Head camera (overhead)", "Head"),
    ]


def test_a_box_without_pyav_is_sent_to_the_platform_for_the_library(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = setup_cameras.Dataset(write_dataset(tmp_path / "ds", "libx264", episodes=1))
    monkeypatch.setitem(sys.modules, "av", None)
    with pytest.raises(ValueError, match="install the dream library with pip install --upgrade dreamlabs"):
        dataset.first_frame("observation.images.left", 0)


def test_our_samples_cover_every_kind_and_wrap_round() -> None:
    for kind in ("head", "wrist_left", "wrist_right"):
        rgb = setup_cameras.decode_jpeg(setup_cameras.sample_jpeg(kind, 0))
        assert rgb.shape == (360, 640, 3)
    assert setup_cameras.sample_jpeg("head", 3) == setup_cameras.sample_jpeg("head", 0)
    assert setup_cameras.sample_jpeg("head", 1) != setup_cameras.sample_jpeg("head", 0)
    with pytest.raises(ValueError):
        setup_cameras.sample_jpeg("belly", 0)


def test_wrist_cameras_given_each_others_roles_are_found_by_their_pictures() -> None:
    left, right, head = sample_rgb("wrist_left"), sample_rgb("wrist_right"), sample_rgb("head")
    references = {"top": head, "left": left, "right": right}
    assert setup_cameras.swapped_roles({"top": head, "left": left, "right": right}, references) == []
    swapped = setup_cameras.swapped_roles({"top": head, "left": right, "right": left}, references)
    assert [pair["roles"] for pair in swapped] == [["left", "right"]]
    # A live view that is a little off (brighter, shifted) is still matched to its own reference.
    shifted = np.clip(np.roll(left, 20, axis=1).astype(int) + 25, 0, 255).astype(np.uint8)
    assert setup_cameras.swapped_roles({"left": shifted, "right": right}, references) == []


def test_a_head_camera_is_never_called_swapped_with_a_wrist_camera() -> None:
    # A head and a wrist frame shrunk to COMPARE_SIZE matched by chance, and setup offered to swap top and
    # left on cameras that were right. Only the two wrist cameras are compared.
    left, right, head = sample_rgb("wrist_left"), sample_rgb("wrist_right"), sample_rgb("head")
    references = {"top": head, "left": left, "right": right}
    assert setup_cameras.swapped_roles({"top": left, "left": head, "right": right}, references) == []
    assert setup_cameras.swapped_roles({"top": right, "left": left, "right": head}, references) == []


def test_the_bank_holds_a_role_open_and_shows_what_the_model_sees() -> None:
    cams = FakeCameras({"cam-a": np.zeros((480, 640, 3), dtype=np.uint8)})
    bank = setup_cameras.CameraBank(cams, (640, 360))
    row = {
        "name": "wrist_l",
        "source": "device",
        "value": "cam-a",
        "width": 640,
        "height": 480,
        "fps": 30,
        "rotate": 0,
    }
    warning = cameras.shape_warning(640, 480, (640, 360))
    assert bank.open(row) == {"role": "wrist_l", "width": 640, "height": 480, "warning": warning}
    assert bank.roles() == ["wrist_l"]
    model = setup_cameras.decode_jpeg(bank.model_jpeg("wrist_l") or b"")
    assert model.shape == (360, 640, 3)
    assert bank.open(row) and cams.opened == ["cam-a"]  # the same row keeps the camera open
    bank.release()
    assert bank.roles() == [] and bank.model_jpeg("wrist_l") is None


def test_a_camera_that_stops_is_opened_again_and_said_to_be_reconnecting_meanwhile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(setup_cameras, "WATCH_S", 0.01)
    monkeypatch.setattr(setup_cameras, "REOPEN_BACKOFF_S", (0.3, 0.02))
    cams = FakeCameras({"cam-a": np.zeros((480, 640, 3), dtype=np.uint8)})
    readers: list[FakeReader] = []
    unplugged = threading.Event()

    def open_reader(camera: config.CameraConfig) -> FakeReader:
        if unplugged.is_set():
            raise errors.CameraError("camera 'wrist_l' could not open device 'cam-a'")
        readers.append(cams(camera))
        return readers[-1]

    def wait_for(done: Callable[[], bool]) -> None:
        deadline = time.monotonic() + 5.0
        while not done():
            assert time.monotonic() < deadline, bank.states()
            time.sleep(0.01)

    bank = setup_cameras.CameraBank(open_reader, (640, 360))
    row = {
        "name": "wrist_l",
        "source": "device",
        "value": "cam-a",
        "width": 640,
        "height": 480,
        "fps": 30,
        "rotate": 0,
    }
    bank.open(row)
    assert bank.states() == {"wrist_l": {"state": "live", "error": None, "tries": 0}}
    unplugged.set()
    readers[0].error = errors.CameraError("camera 'wrist_l' returned no frame for 2 s")
    readers[0].stopped = True
    assert bank.states()["wrist_l"] == {"state": "reconnecting", "error": str(readers[0].error), "tries": 0}
    wait_for(lambda: bank.states()["wrist_l"]["tries"] >= 2)
    assert "could not open device" in bank.states()["wrist_l"]["error"]
    assert not bank.is_open("wrist_l") and bank.roles() == [] and len(readers) == 1
    unplugged.clear()
    wait_for(lambda: bank.states()["wrist_l"]["state"] == "live")
    assert bank.is_open("wrist_l") and len(readers) == 2 and cams.opened == ["cam-a", "cam-a"]
    bank.release()
    assert bank.states() == {} and all(reader.stopped for reader in readers)


def test_a_camera_that_opens_but_delivers_nothing_is_reported_and_released(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(setup_cameras, "FIRST_FRAME_S", 0.2)
    bank = setup_cameras.CameraBank(FakeCameras({"cam-b": None}), (640, 360))
    row = {"name": "top", "source": "device", "value": "cam-b", "width": 640, "height": 480, "fps": 30, "rotate": 0}
    with pytest.raises(ValueError, match="delivered no frame"):
        bank.open(row)
    assert bank.roles() == []
    assert isinstance(setup_cameras.camera_config(dict(row, source="serial")), config.CameraConfig)


class RangeServer:
    """One zip served by byte ranges, as S3 serves a presigned URL; counts the bytes it sent."""

    def __init__(self, blob: bytes) -> None:
        self.sent = 0
        fake = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                start, stop = (int(x) for x in self.headers["Range"].removeprefix("bytes=").split("-"))
                part = blob[start : stop + 1]
                fake.sent += len(part)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{start + len(part) - 1}/{len(blob)}")
                self.send_header("Content-Length", str(len(part)))
                self.end_headers()
                self.wfile.write(part)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                pass

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}/dataset.zip"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def training_zip(tmp_path: pathlib.Path) -> Iterator[tuple[RangeServer, pathlib.Path, int]]:
    """A dataset zipped under a session directory beside 4 MB of other files, served by ranges; yields the server,
    the dataset on disk and the zip's size."""
    codec = next((c for c in ("libx264", "libsvtav1", "mpeg4") if c in av.codecs_available), None)
    if codec is None:
        pytest.skip("this PyAV has no video encoder")
    root = write_dataset(tmp_path / "ds", codec)
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as out:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                out.write(path, "session_a/" + path.relative_to(root).as_posix())
        out.writestr("session_a/data/chunk-000/episode_000000.parquet", np.random.default_rng(0).bytes(4 << 20))
    blob = archive.read_bytes()
    served = RangeServer(blob)
    yield served, root, len(blob)
    served.close()


def test_a_fine_tunes_training_data_is_read_from_its_zip_by_ranges(
    training_zip: tuple[RangeServer, pathlib.Path, int],
) -> None:
    served, root, size = training_zip
    dataset = setup_cameras.training_dataset(served.url)
    assert dataset.video_keys == list(KEYS) and dataset.episodes == [0, 1]
    assert dataset.first_frame(KEYS[1], 1) == setup_cameras.Dataset(root).first_frame(KEYS[1], 1)
    assert served.sent < size / 10, "only the metadata and the one video are read"


def test_the_wizard_compares_a_robot_with_its_newest_fine_tunes_training_data(
    tmp_path: pathlib.Path, training_zip: tuple[RangeServer, pathlib.Path, int]
) -> None:
    """Step 0 picks no model, so the camera step's references come from the data of the account's
    newest fine-tune on the robot's embodiment; another account's fine-tune is never read."""
    pytest.importorskip("dream_ik", reason="dream's IK needs Python 3.11 or newer")
    served, root, _ = training_zip
    account = credentials.Credentials(
        gateway_url="https://api.example.com",
        platform_url="https://platform.example.com",
        data_url="https://data.example.com",
        api_key="k",
        path=pathlib.Path(),
    )
    asked: list[str] = []

    def ask(model: str) -> dict[str, Any]:
        asked.append(model)
        # The platform's Makers builtin carries no video roles: the station's names are guessed from the keys.
        video = (
            {}
            if model == "dream-1-ft:tester-makers"
            else dict(zip(("scene", "wrist_l", "wrist_r"), KEYS, strict=True))
        )
        return {"url": served.url, "video": video}

    newer = setup_fakes.model("dream-1-ft:tester-metal-v2", "mine", "makermods_metal", None)
    newer = dataclasses.replace(newer, created_at=setup_fakes.CREATED + datetime.timedelta(days=1))
    server = setup_wizard.SetupServer(
        tmp_path / "station.yaml",
        account,
        ask_models=lambda: [*setup_fakes.MODELS, newer],
        ask_embodiments=lambda: setup_fakes.EMBODIMENTS,
        ask_training_data=ask,
    )
    server.choose({"body": "metal"})
    cams = server.status()["cameras"]
    assert asked == ["dream-1-ft:tester-metal-v2"] and cams["reference_note"] is None
    assert cams["references"]["wrist_l"] == {"source": "dataset", "key": KEYS[1], "episodes": 2}
    assert server.reference_jpeg("wrist_l", 0) == setup_cameras.Dataset(root).first_frame(KEYS[1], 0)
    server.choose({"body": "makers"})
    assert asked[-1] == "dream-1-ft:tester-makers"
    assert server.status()["cameras"]["references"]["scene"]["key"] == KEYS[0]

    def refuse(model: str) -> dict[str, Any]:
        raise DreamServerError("The dataset was deleted.", code="not_found")

    server._ask_training_data = refuse
    server.choose({"body": "metal"})
    assert server.status()["cameras"]["reference_note"] == (
        "Could not read the data dream-1-ft:tester-metal-v2 was trained on (The dataset was deleted.); showing our YAM"
        " pictures."
    )
    assert server.reference_jpeg("scene", 0) == setup_cameras.sample_jpeg("head", 0)
    server._models = [m for m in setup_fakes.MODELS if m.owner != "mine"]  # only shared and base models
    server.choose({"body": "metal"})
    assert len(asked) == 2 and server.status()["cameras"]["reference_note"] is None
    assert server.reference_jpeg("scene", 0) == setup_cameras.sample_jpeg("head", 0)
