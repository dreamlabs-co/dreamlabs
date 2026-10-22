"""Tests of `dream.datasets` against an in-process fake of the data service, over real HTTP.

`FakeDataService` speaks the data service's `/v1` routes (the embodiments, registration answering an S3 target with
credentials, completion, failure, retry, the dataset) and holds a dict "bucket" the stubbed transfer copies the zip
into, so what the client zips, declares, completes and reports is asserted on the wire. The boto3 transfer itself is
stubbed (`transfer` fixture): the data service's e2e runs it for real against moto.
"""

from __future__ import annotations

import builtins
import dataclasses
import http.server
import io
import json
import os
import pathlib
import re
import secrets
import shutil
import tempfile
import threading
import zipfile
from collections.abc import Iterator
from typing import Any

import numpy as np
import pyarrow
import pyarrow.parquet
import pytest
from shared.datasets import validation

from dream import datasets
from dream.errors import (
    DreamAuthError,
    DreamConfigError,
    DreamDatasetError,
    DreamServerError,
    DreamUploadError,
)

## The YAM body as a customer writes it: 14-wide state and action in four parts, three cameras.
YAM_CONFIG = {
    "state": {"right_arm": [0, 6], "right_gripper": [6, 7], "left_arm": [7, 13], "left_gripper": [13, 14]},
    "action": {"right_arm": [0, 6], "right_gripper": [6, 7], "left_arm": [7, 13], "left_gripper": [13, 14]},
    "video": {
        "overhead": "observation.images.overhead",
        "wrist_l": "observation.images.wrist_l",
        "wrist_r": "observation.images.wrist_r",
    },
}

## An embodiment the fake knows, as the service answers it.
EMBODIMENT = {"name": "alice/arm", "embodiment_config": YAM_CONFIG, "created_at": "2026-09-24T11:00:00Z"}

## The api key of the fake's one account.
KEY = "dl_live_test"

## The bucket the fake's targets name.
BUCKET = "customer-data"

## The credentials the fake answers; the stubbed transfer never signs with them.
CREDENTIALS = {
    "access_key_id": "ASIAFAKE",
    "secret_access_key": "secret",
    "session_token": "token",
    "expires_at": "2099-01-01T00:00:00Z",
}


class FakeDataService:
    """The data service in one threaded HTTP server, with the bucket the transfer stub writes into.

    Attributes:
    - url: The base URL.
    - embodiments: Every `POST /v1/embodiments` body that reached it.
    - datasets: Dataset id to its row (`name`, `embodiment`, `size_bytes`, `episodes`, `status`, `error`, and
      `error_detail` when the pipeline failed it).
    - bucket: S3 key to the bytes held under it.
    - registered: Every `POST /v1/datasets` body that reached it.
    - retried: Every `POST /v1/datasets/{id}/retry` that reached it, as its method and path.
    """

    def __init__(self) -> None:
        """Starts the server on a free port."""
        self.embodiments: list[dict[str, Any]] = []
        self.datasets: dict[str, dict[str, Any]] = {}
        self.bucket: dict[str, bytes] = {}
        self.registered: list[dict[str, Any]] = []
        self.retried: list[tuple[str, str]] = []
        self._lock = threading.Lock()
        fake = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:  # noqa: N802  # http.server API
                fake.handle(self)

            def do_POST(self) -> None:  # noqa: N802  # http.server API
                fake.handle(self)

            def do_DELETE(self) -> None:  # noqa: N802  # http.server API
                fake.handle(self)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002  # http.server API
                pass

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        """Stops the server."""
        self._server.shutdown()
        self._server.server_close()

    def handle(self, request: http.server.BaseHTTPRequestHandler) -> None:
        """Routes one request."""
        length = int(request.headers.get("Content-Length") or 0)
        raw = request.rfile.read(length)
        if request.headers.get("Authorization") != f"Bearer {KEY}":
            _reply(request, 401, json.dumps(_json_error("invalid_api_key", "This api key is not valid.")).encode())
            return
        body = json.loads(raw) if raw else None
        with self._lock:
            status, answer = self._route(request.command, request.path, body)
        _reply(request, status, json.dumps(answer).encode())

    def _route(self, method: str, path: str, body: Any) -> tuple[int, Any]:
        parts = path.strip("/").split("/")
        if (method, path) == ("GET", "/v1/embodiments/alice/arm"):
            return 200, EMBODIMENT
        if (method, path) == ("GET", "/v1/embodiments"):
            return 200, [EMBODIMENT]
        if (method, path) == ("POST", "/v1/embodiments"):
            if any(added["name"] == body["name"] for added in self.embodiments):
                return 409, _json_error("conflict", f"You already have an embodiment named {body['name']}.")
            self.embodiments.append(body)
            added = {**EMBODIMENT, "name": f"alice/{body['name']}", "embodiment_config": body["embodiment_config"]}
            return 201, added
        if (method, path) == ("POST", "/v1/datasets"):
            return self._register(body)
        if (method, path) == ("GET", "/v1/datasets"):
            return 200, [self._present(dataset_id) for dataset_id in reversed(list(self.datasets))]
        if method == "POST" and parts[-1] == "retry":
            self.retried.append((method, path))
        dataset = self.datasets.get(parts[2]) if len(parts) > 2 else None
        if dataset is None:
            return 404, _json_error("not_found", f"No {path}.")
        if method == "POST" and parts[-1] == "complete":
            return self._complete(parts[2])
        if method == "POST" and parts[-1] == "fail":
            if dataset["status"] == "ready_to_process":
                return 409, _json_error("conflict", "This dataset is ready_to_process.")
            dataset.update(status="failed", error=body["error"])
        if method == "POST" and parts[-1] == "retry":
            return self._retry(parts[2])
        if method == "DELETE":
            dataset["status"] = "deleted"
        return 200, self._present(parts[2])

    def _register(self, body: dict[str, Any]) -> tuple[int, Any]:
        self.registered.append(body)
        if body["embodiment"] != EMBODIMENT["name"]:
            return 404, _json_error("not_found", f"You have no embodiment {body['embodiment']!r}.")
        if not (isinstance(body["size_bytes"], int) and body["size_bytes"] >= 1 and body["episodes"] >= 0):
            return 422, _json_error("invalid_request", "size_bytes is at least 1 and episodes at least 0.")
        existing = next((id for id, row in self.datasets.items() if row["name"] == body["name"]), None)
        if existing is not None and self.datasets[existing]["status"] == "ready_to_process":
            return 409, _json_error("conflict", f"You already have a dataset named {body['name']!r}.")
        dataset_id = existing or f"ds_{secrets.token_hex(4)}"
        self.datasets[dataset_id] = {
            "name": body["name"],
            "embodiment": body["embodiment"],
            "size_bytes": body["size_bytes"],
            "episodes": body["episodes"],
            "status": "uploading",
            "error": None,
        }
        key = _key(body["name"])
        upload = {
            "dataset": self._present(dataset_id),
            "s3_uri": f"s3://{BUCKET}/{key}",
            "bucket": BUCKET,
            "key": key,
            "region": "us-west-2",
            "endpoint_url": None,
            "credentials": CREDENTIALS,
        }
        return 201, upload

    def _complete(self, dataset_id: str) -> tuple[int, Any]:
        dataset = self.datasets[dataset_id]
        if dataset["status"] == "failed":
            return 409, _json_error("conflict", "This dataset failed.")
        if dataset["status"] == "uploading":
            held = self.bucket.get(_key(dataset["name"]))
            if held is None:
                return 400, _json_error("validation", f"no object at s3://{BUCKET}/{_key(dataset['name'])}")
            if len(held) != dataset["size_bytes"]:
                message = f"the object is {len(held)} bytes, {dataset['size_bytes']} were declared"
                return 400, _json_error("validation", message)
            dataset["status"] = "ready_to_process"
        return 200, self._present(dataset_id)

    def _retry(self, dataset_id: str) -> tuple[int, Any]:
        dataset = self.datasets[dataset_id]
        if dataset["status"] != "failed":
            message = f"Dataset {dataset_id} is {dataset['status']}; only a failed dataset can be retried."
            return 409, _json_error("conflict", message)
        held = self.bucket.get(_key(dataset["name"]))
        if dataset.get("error_detail") is None and (held is None or len(held) != dataset["size_bytes"]):
            message = f"Dataset {dataset_id} failed during its upload; register the name again to upload it."
            return 409, _json_error("conflict", message)
        dataset.update(status="ready_to_process", error=None, error_detail=None)
        return 200, self._present(dataset_id)

    def _present(self, dataset_id: str) -> dict[str, Any]:
        dataset = self.datasets[dataset_id]
        return {
            "id": dataset_id,
            "name": f"alice/{dataset['name']}",
            "embodiment": dataset["embodiment"],
            "status": dataset["status"],
            "size_bytes": dataset["size_bytes"],
            "episodes": dataset["episodes"],
            "error": dataset["error"],
            "created_at": "2026-09-24T12:00:00Z",
        }


@dataclasses.dataclass
class Transfer:
    """What the stubbed transfer saw and does.

    Attributes:
    - zips: The zip's path, per call.
    - uploads: The service's registration answer, per call.
    - fault: Raised instead of copying, when set.
    - short: Bytes left off the end of the copy, so the completion finds another size than declared.
    """

    zips: list[pathlib.Path] = dataclasses.field(default_factory=list)
    uploads: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    fault: BaseException | None = None
    short: int = 0


@pytest.fixture
def service(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeDataService]:
    """The fake, named by `DREAM_DATA_URL` so every call reaches it; stopped after the test."""
    fake = FakeDataService()
    monkeypatch.setenv("DREAM_DATA_URL", fake.url)
    yield fake
    fake.close()


@pytest.fixture
def transfer(service: FakeDataService, monkeypatch: pytest.MonkeyPatch) -> Transfer:
    """The boto3 transfer replaced by a copy of the zip into the fake's bucket."""
    seen = Transfer()

    def copy(zip_path: pathlib.Path, upload: dict[str, Any]) -> None:
        seen.zips.append(zip_path)
        seen.uploads.append(upload)
        if seen.fault is not None:
            raise seen.fault
        data = zip_path.read_bytes()
        service.bucket[upload["key"]] = data[: len(data) - seen.short]

    monkeypatch.setattr(datasets, "_send_zip", copy)
    return seen


@pytest.fixture
def dataset_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """A one-session LeRobot v2.1 dataset of the fake's YAM embodiment: seven files, three small videos."""
    root = tmp_path / "towels"
    session = root / "s"
    frames, dim = 49, 14  # the fewest an episode at 30 fps may hold, one training sample
    vector = np.arange(frames * dim, dtype=np.float32).reshape(frames, dim)
    column = pyarrow.ListArray.from_arrays(
        pyarrow.array(np.arange(0, vector.size + 1, dim, dtype=np.int32)), pyarrow.array(vector.reshape(-1))
    )
    table = pyarrow.table({"action": column, "observation.state": column, "timestamp": np.arange(frames) / 30})
    buffer = io.BytesIO()
    pyarrow.parquet.write_table(table, buffer)
    names = [f"joint_{index}" for index in range(dim - 1)] + ["gripper"]
    feature = {"dtype": "float32", "shape": [dim], "names": names}
    video = {"dtype": "video", "shape": [48, 64, 3], "names": ["height", "width", "channels"]}
    info = {
        "codebase_version": "v2.1",
        "total_episodes": 1,
        "total_frames": frames,
        "fps": 30,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "action": feature,
            "observation.state": feature,
            "observation.images.overhead": video,
            "observation.images.wrist_l": video,
            "observation.images.wrist_r": video,
        },
    }
    clip = (pathlib.Path(__file__).parent / "fixtures" / "avc1.mp4").read_bytes()  # real H.264, 49 frames of 32 x 32
    videos = "videos/chunk-000"
    files = {
        "meta/info.json": json.dumps(info).encode(),
        "meta/episodes.jsonl": json.dumps({"episode_index": 0, "tasks": ["fold"], "length": frames}).encode() + b"\n",
        "meta/tasks.jsonl": b'{"task_index": 0, "task": "fold"}\n',
        "data/chunk-000/episode_000000.parquet": buffer.getvalue(),
        f"{videos}/observation.images.overhead/episode_000000.mp4": clip,
        f"{videos}/observation.images.wrist_l/episode_000000.mp4": clip,
        f"{videos}/observation.images.wrist_r/episode_000000.mp4": clip,
    }
    for path, data in files.items():
        (session / path).parent.mkdir(parents=True, exist_ok=True)
        (session / path).write_bytes(data)
    return root


def _upload_to(service: FakeDataService, root: pathlib.Path) -> Any:
    return datasets.upload_dataset(root, "towels", "alice/arm", api_key=KEY)


def test_an_upload_zips_the_files_as_listed_registers_the_size_and_episodes_and_answers_the_ready_dataset(
    service: FakeDataService, transfer: Transfer, dataset_dir: pathlib.Path
) -> None:
    dataset = _upload_to(service, dataset_dir)
    assert (dataset.name, dataset.embodiment, dataset.status, dataset.episodes, dataset.error) == (
        "alice/towels",
        "alice/arm",
        "ready_to_process",
        1,
        None,
    )
    (held,) = service.bucket.values()
    (registration,) = service.registered
    assert registration == {"name": "towels", "embodiment": "alice/arm", "size_bytes": len(held), "episodes": 1}
    assert dataset.size_bytes == len(held) and (transfer.uploads[0]["key"], *service.bucket) == (
        "alice/towels/dataset.zip",
        "alice/towels/dataset.zip",
    )
    with zipfile.ZipFile(io.BytesIO(held)) as archive:
        listed = validation.list_files(dataset_dir)
        assert archive.namelist() == [entry.path for entry in listed], "the checks' order, every file, nothing else"
        assert {info.compress_type for info in archive.infolist()} == {zipfile.ZIP_STORED}
        assert all(archive.read(entry.path) == (dataset_dir / entry.path).read_bytes() for entry in listed)
    (zip_path,) = transfer.zips
    assert not zip_path.exists(), "the temporary zip is removed"
    assert service.datasets[dataset.id]["status"] == "ready_to_process"


@pytest.mark.parametrize("name", ["Towels", "towels_v1", "alice/towels", "", "a" * 65])
def test_a_name_outside_the_services_rule_is_a_config_error_and_nothing_is_read_or_sent(
    name: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check runs first: the directory does not exist and the host refuses connections, so any other step would
    raise something else."""
    monkeypatch.setenv("DREAM_DATA_URL", "http://127.0.0.1:9")
    with pytest.raises(DreamConfigError, match=r"is not valid: .*a-z, 0-9 and '-' only, 1 to 64 .*e\.g\. 'towels-v1'"):
        datasets.upload_dataset(tmp_path / "absent", name, "alice/arm", api_key=KEY)


def test_a_64_character_name_is_uploaded(
    service: FakeDataService, transfer: Transfer, dataset_dir: pathlib.Path
) -> None:
    assert datasets.upload_dataset(dataset_dir, "a" * 64, "alice/arm", api_key=KEY).status == ("ready_to_process")


## The data service's dataset-name rule, which the client copies (it cannot import the service).
SERVICE_EMBODIMENTS = (
    pathlib.Path(__file__).resolve().parents[3] / "data_service" / "src" / "data_service" / "data" / "embodiments.py"
)


def test_the_dataset_name_rule_is_the_services() -> None:
    """A change to the service's `NAME_RE` fails here until the client's copy says the same."""
    found = re.search(r'^NAME_RE: .*re\.compile\(r"\^(.*)\$"\)$', SERVICE_EMBODIMENTS.read_text(), re.MULTILINE)
    assert found, "no NAME_RE pattern in the service embodiments module"
    assert datasets._NAME_RE.pattern == found.group(1)


def test_hidden_files_are_neither_checked_nor_sent(
    service: FakeDataService, transfer: Transfer, dataset_dir: pathlib.Path
) -> None:
    (dataset_dir / ".DS_Store").write_bytes(b"\x00" * 16)
    (dataset_dir / "s" / "meta" / ".DS_Store").write_bytes(b"\x00" * 16)
    assert _upload_to(service, dataset_dir).status == "ready_to_process"
    (held,) = service.bucket.values()
    with zipfile.ZipFile(io.BytesIO(held)) as archive:
        names = archive.namelist()
    assert len(names) == 7 and not any(".DS_Store" in name for name in names)


def test_a_note_beside_the_session_folders_is_skipped_with_a_warning_and_other_files_there_are_refused(
    service: FakeDataService, transfer: Transfer, dataset_dir: pathlib.Path
) -> None:
    """A README beside the session folders passes and stays home, as on the site's pages; a file there that is not a
    note is refused with the same line as the pages', and nothing is sent."""
    (dataset_dir / "README.txt").write_text("towels, day 2")
    (dataset_dir / "notes.md").write_text("camera moved")
    assert datasets.validate_dataset(dataset_dir, YAM_CONFIG).ok
    skipped = "Not uploaded: README.txt, notes.md (not inside a session folder)."
    with pytest.warns(UserWarning, match=re.escape(skipped)):
        assert _upload_to(service, dataset_dir).status == "ready_to_process"
    (held,) = service.bucket.values()
    with zipfile.ZipFile(io.BytesIO(held)) as archive:
        names = archive.namelist()
    assert len(names) == 7 and "README.txt" not in names and "notes.md" not in names
    (dataset_dir / "junk.bin").write_bytes(b"\x00")
    report = datasets.validate_dataset(dataset_dir, YAM_CONFIG)
    assert not report.ok and report.error == "'junk.bin' is not inside a session folder."


def test_a_session_folder_that_is_a_link_is_sent(
    service: FakeDataService, transfer: Transfer, dataset_dir: pathlib.Path
) -> None:
    """A folder of sessions put together with links sends every session: a linked one was left out without a word,
    and the rest uploaded as the whole dataset."""
    shutil.copytree(dataset_dir / "s", dataset_dir.parent / "store" / "s2")
    (dataset_dir / "s2").symlink_to(dataset_dir.parent / "store" / "s2", target_is_directory=True)
    assert _upload_to(service, dataset_dir).status == "ready_to_process"
    (held,) = service.bucket.values()
    with zipfile.ZipFile(io.BytesIO(held)) as archive:
        names = archive.namelist()
    assert len(names) == 14 and sum(name.startswith("s2/") for name in names) == 7


def test_a_file_that_cannot_be_read_fails_the_report_with_its_name(dataset_dir: pathlib.Path) -> None:
    """A file the listing found but this process cannot read is the dataset's one failing line, not a PermissionError
    out of the checks; upload_dataset() raises the same line as DreamDatasetError."""
    info = dataset_dir / "s" / "meta" / "info.json"
    info.chmod(0)
    try:
        if os.access(info, os.R_OK):  # root reads it anyway, so there is no failure to see
            pytest.skip("this process reads files without permission")
        report = datasets.validate_dataset(dataset_dir, YAM_CONFIG)
    finally:
        info.chmod(0o644)
    assert not report.ok and report.error is not None
    assert report.error.startswith(f"Cannot read {info}")


def test_timestamps_that_drift_from_the_videos_pass_with_a_note_the_upload_shows_as_a_warning(
    service: FakeDataService, transfer: Transfer, dataset_dir: pathlib.Path
) -> None:
    """A recorder that stamps rows with its own clock passes: processing re-times its timestamps as frame_index / fps,
    the report's note says so, and upload_dataset() shows it as a UserWarning and uploads."""
    path = dataset_dir / "s" / "data" / "chunk-000" / "episode_000000.parquet"
    table = pyarrow.parquet.read_table(path)
    drifting = pyarrow.array(np.arange(table.num_rows) / 20)  # a 30 fps session's rows, 20 a second
    pyarrow.parquet.write_table(table.set_column(table.column_names.index("timestamp"), "timestamp", drifting), path)
    note = (
        "Directory 's': the timestamps of 1 of 1 episodes drift from 30 fps, by up to 24.0 frames "
        "(data/chunk-000/episode_000000.parquet, frame 48); processing re-times them as frame_index / fps, so each "
        "frame of data is paired with its own video frame."
    )
    assert datasets.validate_dataset(dataset_dir, YAM_CONFIG).notes == [note]
    with pytest.warns(UserWarning, match=re.escape(note)):
        assert _upload_to(service, dataset_dir).status


def test_a_session_directory_is_uploaded_as_its_own_dataset_under_its_name(
    service: FakeDataService, transfer: Transfer, dataset_dir: pathlib.Path
) -> None:
    assert datasets.upload_dataset(dataset_dir / "s", "towels", "alice/arm", api_key=KEY).status
    (held,) = service.bucket.values()
    with zipfile.ZipFile(io.BytesIO(held)) as archive:
        assert archive.namelist()[0] == "s/data/chunk-000/episode_000000.parquet"


def test_upload_without_boto3_is_a_config_error_and_registers_nothing(
    service: FakeDataService, dataset_dir: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real `_send_zip` path is what is guarded here (no `transfer` fixture); the check runs before the first
    request, so the service sees nothing, not even the embodiment fetch."""
    real_import = builtins.__import__

    def without_boto3(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith(("boto3", "botocore")):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_boto3)
    with pytest.raises(DreamConfigError, match=re.escape("install the library: pip install --upgrade dreamlabs")):
        _upload_to(service, dataset_dir)
    assert service.registered == [] and service.datasets == {} and service.bucket == {}


def test_a_file_dated_before_1980_is_an_upload_error_and_registers_nothing(
    service: FakeDataService, transfer: Transfer, dataset_dir: pathlib.Path
) -> None:
    """A zip cannot record a time before 1980, and zipfile refused it with a ValueError that escaped the upload."""
    old = dataset_dir / "s" / "meta" / "tasks.jsonl"
    os.utime(old, (0, 0))
    with pytest.raises(DreamUploadError, match="cannot write the dataset zip: ZIP does not support timestamps"):
        _upload_to(service, dataset_dir)
    assert service.registered == [] and transfer.zips == []


@pytest.mark.parametrize("platform_url", ["https://platform.dreamlabs.ai", "https://staging.platform.example.com"])
def test_a_missing_dependency_is_the_pypi_install_whatever_the_platform(
    platform_url: str, dataset_dir: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_import = builtins.__import__

    def without_validation(name: str, *args: Any, **kwargs: Any) -> Any:
        fromlist = args[2] if len(args) > 2 else kwargs.get("fromlist") or ()
        if name == "shared.datasets" and "validation" in (fromlist or ()):
            raise ImportError("No module named 'pyarrow'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setenv("DREAM_API_URL", platform_url)
    monkeypatch.setattr(builtins, "__import__", without_validation)
    with pytest.raises(DreamConfigError) as raised:
        datasets.validate_dataset(dataset_dir, YAM_CONFIG)
    assert str(raised.value).endswith("; install the library: pip install --upgrade dreamlabs")


def test_a_temporary_directory_the_zip_cannot_be_made_in_is_an_upload_error_and_registers_nothing(
    service: FakeDataService,
    transfer: Transfer,
    dataset_dir: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "missing"))
    with pytest.raises(DreamUploadError, match="^cannot write the dataset zip: ") as refused:
        _upload_to(service, dataset_dir)
    assert isinstance(refused.value.__cause__, OSError)
    assert service.registered == [] and service.datasets == {} and not transfer.zips


def test_a_temporary_disk_smaller_than_the_zip_is_a_config_error_with_both_sizes_and_TMPDIR_before_any_write(
    service: FakeDataService,
    transfer: Transfer,
    dataset_dir: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The zip is stored uncompressed, so a temporary disk with less room than the dataset failed part way through
    writing it; now a disk even one byte smaller than the zip is refused before the zip is opened."""
    _upload_to(service, dataset_dir)
    (written,) = service.bucket.values()
    size = len(written)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    disk = shutil.disk_usage(scratch)
    monkeypatch.setattr(shutil, "disk_usage", lambda path: disk._replace(free=size - 1))
    with pytest.raises(DreamConfigError) as refused:
        datasets.upload_dataset(dataset_dir, "towels-2", "alice/arm", api_key=KEY)
    needed = re.fullmatch(
        rf"the upload first writes the dataset as one uncompressed zip in {re.escape(str(scratch))}, which needs "
        rf"\d+\.\d\d GiB \(([\d,]+) bytes\) free there and has 0\.00 GiB \({size - 1:,} bytes\)\. Set TMPDIR to a "
        r"directory on a disk with room \(TMPDIR=/path/with/space python upload\.py, or tempfile\.tempdir = "
        r"'/path/with/space' in a running Python\) and run the upload again\.",
        str(refused.value),
    )
    assert needed is not None, str(refused.value)
    assert int(needed[1].replace(",", "")) >= size
    assert list(scratch.iterdir()) == [] and len(transfer.zips) == 1 and len(service.registered) == 1


def test_an_info_file_without_chunks_size_is_refused_naming_it_and_nothing_is_sent(
    service: FakeDataService, transfer: Transfer, dataset_dir: pathlib.Path
) -> None:
    info_file = dataset_dir / "s/meta/info.json"
    fields = json.loads(info_file.read_bytes())
    del fields["chunks_size"]
    info_file.write_text(json.dumps(fields))
    named = "Directory 's': meta/info.json has no chunks_size; a LeRobot v2.1 info file needs codebase_version, fps, "
    assert datasets.validate_dataset(dataset_dir, YAM_CONFIG).error.startswith(named)
    with pytest.raises(DreamDatasetError, match=re.escape(named)):
        _upload_to(service, dataset_dir)
    assert not service.registered and not transfer.zips


def test_a_dataset_the_local_checks_refuse_sends_nothing(
    service: FakeDataService, transfer: Transfer, dataset_dir: pathlib.Path
) -> None:
    (dataset_dir / "s/meta/tasks.jsonl").unlink()
    with pytest.raises(DreamDatasetError, match="tasks.jsonl"):
        _upload_to(service, dataset_dir)
    assert not service.registered and not transfer.zips


def test_a_transfer_that_fails_reports_the_failure_and_raises_it_and_the_name_is_reused_by_the_rerun(
    service: FakeDataService, transfer: Transfer, dataset_dir: pathlib.Path
) -> None:
    transfer.fault = OSError("connection reset by peer\n  while sending part 3")
    with pytest.raises(DreamUploadError, match="^connection reset by peer while sending part 3$"):
        _upload_to(service, dataset_dir)
    (dataset_id,) = service.datasets
    row = service.datasets[dataset_id]
    assert (row["status"], row["error"]) == ("failed", "connection reset by peer while sending part 3")
    assert not service.bucket and not transfer.zips[0].exists()
    transfer.fault = None
    dataset = _upload_to(service, dataset_dir)
    assert (dataset.id, dataset.status, dataset.error, len(service.datasets)) == (
        dataset_id,
        "ready_to_process",
        None,
        1,
    )


def test_a_completion_the_service_refuses_reports_the_failure_with_the_services_line(
    service: FakeDataService, transfer: Transfer, dataset_dir: pathlib.Path
) -> None:
    transfer.short = 10
    with pytest.raises(DreamUploadError, match=r"^the object is \d+ bytes, \d+ were declared$") as refused:
        _upload_to(service, dataset_dir)
    (row,) = service.datasets.values()
    assert (row["status"], row["error"]) == ("failed", str(refused.value))
    assert isinstance(refused.value.__cause__, DreamServerError) and refused.value.__cause__.code == "validation"


def test_a_run_the_process_does_not_survive_leaves_the_dataset_uploading_and_the_rerun_reuses_it(
    service: FakeDataService, transfer: Transfer, dataset_dir: pathlib.Path
) -> None:
    transfer.fault = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        _upload_to(service, dataset_dir)
    (dataset_id,) = service.datasets
    assert service.datasets[dataset_id]["status"] == "uploading" and not transfer.zips[0].exists()
    transfer.fault = None
    assert (_upload_to(service, dataset_dir).id, len(service.datasets)) == (dataset_id, 1)


def test_the_failure_line_is_at_most_the_services_limit(
    service: FakeDataService, transfer: Transfer, dataset_dir: pathlib.Path
) -> None:
    transfer.fault = RuntimeError("x" * 1000)
    with pytest.raises(DreamUploadError) as refused:
        _upload_to(service, dataset_dir)
    (row,) = service.datasets.values()
    assert len(str(refused.value)) == datasets._ERROR_LINE_MAX == len(row["error"])


def test_an_embodiment_is_fetched_by_splitting_its_name_at_the_last_slash(monkeypatch: pytest.MonkeyPatch) -> None:
    """A username is an email's local part, which may hold a slash; a name never does."""
    asked: list[str] = []

    def answers(method: str, host: str, path: str, api_key: str, *args: Any, **kwargs: Any) -> Any:
        asked.append(path)
        return {**EMBODIMENT, "name": "a/b/arm"}

    monkeypatch.setattr(datasets, "_call", answers)
    assert datasets._fetch_embodiment("a/b/arm", KEY, "http://x").name == "a/b/arm"
    assert asked == ["/v1/embodiments/a%2Fb/arm"]


def test_refusals_carry_the_services_code_and_a_bad_key_is_an_auth_error(
    service: FakeDataService,
    transfer: Transfer,
    dataset_dir: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    _upload_to(service, dataset_dir)
    with pytest.raises(DreamServerError) as refused:
        _upload_to(service, dataset_dir)
    assert refused.value.code == "conflict" and "towels" in str(refused.value)
    assert len(transfer.zips) == 1 and not list(tmp_path.glob("dream-dataset-*.zip")), "a refused zip is removed"
    with pytest.raises(DreamServerError) as missing:
        datasets.upload_dataset(dataset_dir, "other", "alice/leg", api_key=KEY)
    assert missing.value.code == "not_found"
    with pytest.raises(DreamAuthError):
        datasets.list_embodiments(api_key="dl_live_other")


def test_validate_dataset_runs_offline_against_a_config_and_takes_a_session_directory_as_its_own_dataset(
    dataset_dir: pathlib.Path,
) -> None:
    report = datasets.validate_dataset(dataset_dir / "s", YAM_CONFIG)
    assert report == datasets.Report(
        ok=True,
        sessions=1,
        episodes=1,
        frames=49,
        files=7,
        bytes=sum(path.stat().st_size for path in (dataset_dir / "s").rglob("*") if path.is_file()),
        state_dim=14,
        action_dim=14,
        fps=30,
        video_keys=["observation.images.overhead", "observation.images.wrist_l", "observation.images.wrist_r"],
    )
    one_camera = {**YAM_CONFIG, "video": {"overhead": "observation.images.overhead"}}
    refused = datasets.validate_dataset(dataset_dir, one_camera)
    assert refused == datasets.Report(ok=False, error="Directory 's': 3 video streams; the embodiment names 1.")
    with pytest.raises(DreamConfigError, match="Unknown key 'fps'"):
        datasets.validate_dataset(dataset_dir, {**YAM_CONFIG, "fps": 30})
    # A body stored before `state_space` was retired still reads: the config a listing answers validates as the same
    # body, the retired key ignored (registering it anew is what add_embodiment refuses).
    stored = {**YAM_CONFIG, "state_space": {"space": "joint", "frame": "abs"}}
    assert datasets.validate_dataset(dataset_dir / "s", stored) == report


def test_add_embodiment_parses_the_config_before_sending_and_answers_the_embodiment(
    service: FakeDataService,
) -> None:
    with pytest.raises(DreamConfigError, match="state: nothing covers"):
        datasets.add_embodiment("gap", {**YAM_CONFIG, "state": {"arm": [1, 7]}}, api_key=KEY)
    retired = {**YAM_CONFIG, "state_space": {"space": "joint", "frame": "abs"}}  # an older config's key
    with pytest.raises(DreamConfigError, match="state_space is no longer part of the embodiment config"):
        datasets.add_embodiment("retired", retired, api_key=KEY)
    assert not service.embodiments
    added = datasets.add_embodiment("wrist", YAM_CONFIG, api_key=KEY)
    assert (added.name, added.embodiment_config) == ("alice/wrist", YAM_CONFIG)
    assert service.embodiments == [{"name": "wrist", "embodiment_config": YAM_CONFIG}]
    (listed,) = datasets.list_embodiments(api_key=KEY)
    assert listed == datasets.Embodiment("alice/arm", YAM_CONFIG, datasets._timestamp(EMBODIMENT["created_at"]))


def test_a_refused_embodiment_names_itself_and_the_code(service: FakeDataService) -> None:
    """A custom body's creation failed and the report could not say why; the refusal now names the embodiment
    and the error code, and says nothing of the service's internals ("data service", the URL it called)."""
    datasets.add_embodiment("wrist", YAM_CONFIG, api_key=KEY)
    with pytest.raises(DreamServerError) as taken:
        datasets.add_embodiment("wrist", YAM_CONFIG, api_key=KEY)
    assert (taken.value.code, taken.value.retryable, type(taken.value)) == ("conflict", False, DreamServerError)
    assert (
        str(taken.value)
        == "could not create the embodiment 'wrist' (conflict): You already have an embodiment named wrist."
    )
    assert "data service" not in str(taken.value) and service.url not in str(taken.value)
    with pytest.raises(DreamAuthError) as refused:
        datasets.add_embodiment("wrist", YAM_CONFIG, api_key="dl_live_from_another_site")
    assert refused.value.code == "invalid_api_key"
    assert str(refused.value) == (
        "could not create the embodiment 'wrist' (invalid_api_key): This api key is not valid.; "
        "an API key works only on the site that made it"
    )


def test_get_dataset_returns_the_row_the_service_holds(service: FakeDataService) -> None:
    """get_dataset answers the #Dataset for one of the caller's datasets by its id."""
    service.datasets["ds_abcd1234"] = {
        "name": "towels-v1",
        "embodiment": "alice/arm",
        "size_bytes": 4096,
        "episodes": 12,
        "status": "processed",
        "error": None,
    }
    found = datasets.get_dataset("ds_abcd1234", api_key=KEY)
    assert found == datasets.Dataset(
        id="ds_abcd1234",
        name="alice/towels-v1",
        embodiment="alice/arm",
        status="processed",
        size_bytes=4096,
        episodes=12,
        error=None,
        created_at=datasets._timestamp("2026-09-24T12:00:00Z"),
    )


def test_list_datasets_answers_every_dataset_of_the_account_newest_first(service: FakeDataService) -> None:
    """list_datasets wraps `GET /v1/datasets`: each row as a #Dataset, in the service's order, a failed one with its
    line; an account with none answers an empty list."""
    assert datasets.list_datasets(api_key=KEY) == []
    service.datasets["ds_old"] = {"name": "towels-v1", "embodiment": "alice/arm", "size_bytes": 1, "episodes": 1}
    service.datasets["ds_old"].update(status="failed", error="the upload is not a zip file; zip the dataset again")
    service.datasets["ds_new"] = {"name": "towels-v2", "embodiment": "alice/arm", "size_bytes": 2, "episodes": 2}
    service.datasets["ds_new"].update(status="processed", error=None)
    listed = datasets.list_datasets(api_key=KEY)
    assert [(found.id, found.name, found.status, found.error) for found in listed] == [
        ("ds_new", "alice/towels-v2", "processed", None),
        ("ds_old", "alice/towels-v1", "failed", "the upload is not a zip file; zip the dataset again"),
    ]


def test_delete_dataset_asks_first_and_deletes_only_on_yes(
    service: FakeDataService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """delete_dataset asks on the terminal; `y` sends `DELETE /v1/datasets/{id}` and answers the dataset `deleted`,
    anything else sends nothing and answers None, and `confirm=False` asks nothing."""
    service.datasets["ds_1"] = {"name": "towels-v1", "embodiment": "alice/arm", "size_bytes": 1, "episodes": 1}
    service.datasets["ds_1"].update(status="processed", error=None)
    asked: list[str] = []
    monkeypatch.setattr("builtins.input", lambda question: asked.append(question) or "n")
    assert datasets.delete_dataset("ds_1", api_key=KEY) is None
    assert service.datasets["ds_1"]["status"] == "processed", "nothing was sent"
    assert asked == ["Delete dataset ds_1? This cannot be undone. [y/N] "]
    monkeypatch.setattr("builtins.input", lambda question: "y")
    assert datasets.delete_dataset("ds_1", api_key=KEY).status == "deleted"  # type: ignore[union-attr]
    service.datasets["ds_1"]["status"] = "processed"
    monkeypatch.setattr("builtins.input", lambda question: pytest.fail("asked"))
    assert datasets.delete_dataset("ds_1", confirm=False, api_key=KEY).status == "deleted"  # type: ignore[union-attr]


def test_get_dataset_that_is_not_the_callers_is_not_found(service: FakeDataService) -> None:
    """A dataset id that is not the caller's is `not_found`, the service's own code."""
    with pytest.raises(DreamServerError, match="No /v1/datasets/ds_missing") as raised:
        datasets.get_dataset("ds_missing", api_key=KEY)
    assert raised.value.code == "not_found"


def _failed(service: FakeDataService, dataset_id: str, **row: Any) -> dict[str, Any]:
    """Puts a `failed` dataset in the fake as the pipeline leaves one (both error lines), `row` over it."""
    service.datasets[dataset_id] = {
        "name": "towels-v1",
        "embodiment": "alice/arm",
        "size_bytes": 4096,
        "episodes": 12,
        "status": "failed",
        "error": "Processing failed.",
        "error_detail": "convert: episode 3 has no frames",
        **row,
    }
    return service.datasets[dataset_id]


def test_retry_dataset_posts_to_the_datasets_retry_route_and_answers_it_ready_to_process(
    service: FakeDataService,
) -> None:
    """retry_dataset is one POST of the id's retry route, with no body, and answers the #Dataset the service does."""
    row = _failed(service, "ds_abcd1234")
    retried = datasets.retry_dataset("ds_abcd1234", api_key=KEY)
    assert service.retried == [("POST", "/v1/datasets/ds_abcd1234/retry")]
    assert retried == datasets.Dataset(
        id="ds_abcd1234",
        name="alice/towels-v1",
        embodiment="alice/arm",
        status="ready_to_process",
        size_bytes=4096,
        episodes=12,
        error=None,
        created_at=datasets._timestamp("2026-09-24T12:00:00Z"),
    )
    assert (row["status"], row["error"], row["error_detail"]) == ("ready_to_process", None, None)
    assert not service.registered and not service.bucket, "nothing is registered or uploaded again"


def test_retry_dataset_sends_the_id_as_one_path_segment_and_is_not_repeated(monkeypatch: pytest.MonkeyPatch) -> None:
    """An id is quoted whole, so a slash in it cannot name another route; a retry is not safe to repeat."""
    asked: list[tuple[str, str, tuple[Any, ...], dict[str, Any]]] = []

    def answers(method: str, host: str, path: str, api_key: str, *args: Any, **kwargs: Any) -> Any:
        asked.append((method, path, args, kwargs))
        return {
            "id": "ds_1/x",
            "name": "alice/towels-v1",
            "embodiment": "alice/arm",
            "status": "ready_to_process",
            "size_bytes": 4096,
            "episodes": 12,
            "error": None,
            "created_at": "2026-09-24T12:00:00Z",
        }

    monkeypatch.setattr(datasets, "_call", answers)
    assert datasets.retry_dataset("ds_1/x", api_key=KEY).status == "ready_to_process"
    assert asked == [("POST", "/v1/datasets/ds_1%2Fx/retry", (), {})]


def test_retry_dataset_that_is_not_the_callers_is_not_found(service: FakeDataService) -> None:
    """A dataset id that is not the caller's is `not_found`, the service's own code."""
    with pytest.raises(DreamServerError, match="No /v1/datasets/ds_missing/retry") as raised:
        datasets.retry_dataset("ds_missing", api_key=KEY)
    assert raised.value.code == "not_found"
    assert service.retried == [("POST", "/v1/datasets/ds_missing/retry")]


@pytest.mark.parametrize("status", ["uploading", "ready_to_process", "processing", "processed"])
def test_retry_dataset_that_is_not_failed_is_a_conflict_naming_its_status(
    service: FakeDataService, status: str
) -> None:
    """Only a `failed` dataset is retried: any other is `conflict`, with the service's line, and is left as it is."""
    row = _failed(service, "ds_abcd1234", status=status, error=None, error_detail=None)
    message = f"Dataset ds_abcd1234 is {status}; only a failed dataset can be retried."
    with pytest.raises(DreamServerError, match=f"^{re.escape(message)}$") as raised:
        datasets.retry_dataset("ds_abcd1234", api_key=KEY)
    assert raised.value.code == "conflict" and not raised.value.retryable
    assert row["status"] == status


def test_retry_dataset_whose_upload_never_finished_is_a_conflict_saying_to_upload_it_again(
    service: FakeDataService,
) -> None:
    """A dataset the client failed has no zip to process again: `conflict`, with the service's line, still failed."""
    row = _failed(service, "ds_abcd1234", error="the object is 4086 bytes, 4096 were declared", error_detail=None)
    message = "Dataset ds_abcd1234 failed during its upload; register the name again to upload it."
    with pytest.raises(DreamServerError, match=f"^{re.escape(message)}$") as raised:
        datasets.retry_dataset("ds_abcd1234", api_key=KEY)
    assert raised.value.code == "conflict" and not raised.value.retryable
    assert (row["status"], row["error"]) == ("failed", "the object is 4086 bytes, 4096 were declared")


def test_a_key_and_a_host_left_out_come_from_the_credentials(
    service: FakeDataService, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DREAM_CREDENTIALS", str(tmp_path / "credentials"))
    monkeypatch.delenv("DREAM_API_KEY", raising=False)
    monkeypatch.delenv("DREAM_DATA_URL", raising=False)
    with pytest.raises(DreamConfigError, match="no api key"):
        datasets.list_embodiments()
    (tmp_path / "credentials").write_text(f"DREAM_API_KEY={KEY}\nDREAM_DATA_URL={service.url}\n")
    # The key and the data service's URL both come from the credentials when neither is passed.
    assert [found.name for found in datasets.list_embodiments()] == ["alice/arm"]


def _key(name: str) -> str:
    return f"alice/{name}/dataset.zip"


def _json_error(code: str, message: str) -> dict[str, Any]:
    return {"error": {"code": code, "message": message}}


def _reply(request: http.server.BaseHTTPRequestHandler, status: int, body: bytes) -> None:
    request.send_response(status)
    request.send_header("Content-Length", str(len(body)))
    request.end_headers()
    request.wfile.write(body)
