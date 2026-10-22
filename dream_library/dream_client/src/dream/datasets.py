"""Embodiments and datasets on the Dream Labs data service: describe a robot body, check a dataset locally, upload it.

`dream.DreamClient` carries the same calls with its key bound (add_embodiment(), get_embodiments(),
validate_dataset(), upload_dataset(), get_dataset_status(), delete_dataset()); the functions here are the same calls
on their own, for a caller that passes `api_key` itself. The functions:

- add_embodiment() and list_embodiments(): the robot bodies an account records datasets on. An embodiment is
  `<username>/<name>` plus one config the user writes, `{"state": {part: [start, end]}, "action": {part: [start, end]},
  "video": {role: video_key}}`: the named spans of the state and action vectors, and the camera roles mapped to the
  dataset's video keys (add_embodiment() spells out every rule). The client checks the config before sending it, and
  the service refuses the same configs with the same line.
- validate_dataset(): the checks a dataset must pass, run on a local directory against a config, answering a #Report.
  Nothing is sent and no key is needed.
- upload_dataset(): fetches the embodiment's config, runs the same checks, writes the dataset as one zip (the
  directory tree as it is, stored uncompressed: mp4 and parquet are compressed already), registers it, sends the zip
  straight to S3 on temporary credentials the service grants for that one object, and completes it; the service
  checks the zip is there at the declared size and answers the dataset `ready_to_process`. Whatever fails after the
  registration is reported to the service, which marks the dataset `failed`, and raised as #DreamUploadError; the
  name is free again, so running the same call again starts over on the same dataset. A run the process did not
  survive (Ctrl+C, a crash) leaves the dataset `uploading`, and running it again reuses it too. Nothing is kept on
  the machine between runs.
- get_dataset(): a dataset's status as the service holds it now: `ready_to_process` until processing starts,
  `processing` while it runs, `processed` once it has (the one status a fine-tuning job is queued on), or `failed`
  when the upload or the processing could not finish (`error` says why). list_datasets() answers every one of the
  account's datasets, so an id upload_dataset() answered and was lost is found there by the dataset's name.

A dataset directory holds one LeRobot v2.1 dataset per subdirectory (`<session>/meta/info.json`, `data/`, `videos/`;
each subdirectory name letters, digits, `.`, `_` and `-`, at most 64), or is itself one (it has `meta/info.json`; its
name is then the session's). Hidden files (a `.DS_Store`, a `.git/`) are neither checked nor sent, nor is a note
beside the session folders (a README, a `.txt` or a `.md`, `validation.is_stray()`): upload_dataset() warns
`Not uploaded: README.txt (not inside a session folder).` for those.

Every call to the service takes your API key (`api_key`); left out, it is the credentials' (`DREAM_API_KEY`). Where the
service is comes from your credentials and is not a parameter. The local checks need pyarrow and the upload needs
boto3. The library the platform offers carries both; an install without them raises #DreamConfigError on the first call
that needs them, saying how to install the library.

Errors: a refusal by the service is a #DreamServerError whose `code` is the service's (`conflict` for a name that is
`ready_to_process`, `processing` or `processed`, `not_found`, `invalid_request`, `validation`; HTTP 5xx ones are
`retryable`: make the same call again after a short wait). A key it does not accept is #DreamAuthError, `forbidden`
for an account with no username yet (ask Dream Labs support to set one). A service that cannot be reached, or an
answer that is not the Dream API's, is #DreamProtocolError. A config that is not valid, a directory that is not one,
or a missing key is #DreamConfigError. A dataset the local checks refuse is #DreamDatasetError; an upload that did not
finish is #DreamUploadError. Those two live in `dream.errors`, not the `dream` namespace.

@par Example
```python
from dream import datasets

config = {
    "state": {"arm": [0, 6], "gripper": [6, 7]},
    "action": {"arm": [0, 6], "gripper": [6, 7]},
    "video": {"overhead": "observation.images.top", "wrist": "observation.images.wrist"},
}
datasets.add_embodiment("metal-arm", config, api_key=KEY)
dataset = datasets.upload_dataset("~/data/towels", "towels-v1", "<your-username>/metal-arm", api_key=KEY)
print(dataset.id, dataset.status, dataset.episodes, dataset.size_bytes)
```
"""

import contextlib
import dataclasses
import datetime
import http.client
import json
import os
import pathlib
import random
import re
import shutil
import tempfile
import time
import urllib.parse
import warnings
import zipfile
from typing import TYPE_CHECKING, Any

from shared.datasets import embodiments

if TYPE_CHECKING:
    import boto3
    from boto3.s3.transfer import TransferConfig
    from botocore.config import Config

from dream import credentials, transport
from dream.errors import (
    DreamAuthError,
    DreamConfigError,
    DreamDatasetError,
    DreamError,
    DreamProtocolError,
    DreamServerError,
    DreamUploadError,
)

## Seconds a call to the data service may take.
_TIMEOUT_S = 60.0

## Attempts of a call that is safe to repeat (a read, a completion, a failure report) before it gives up.
_ATTEMPTS = 8

## The first retry's longest wait, seconds; each later one doubles it, up to #_BACKOFF_MAX_S.
_BACKOFF_BASE_S = 0.5

## The longest wait between two attempts, seconds.
_BACKOFF_MAX_S = 30.0

## The zip goes up in one PUT below this many bytes and in parts of this many at or above it; boto3 enlarges the parts
## when a zip would need more than S3's 10,000, so the ceiling is S3's 5 TiB object.
_PART_BYTES = 64 * 1024**2

## Parts of the zip in flight at once.
_TRANSFER_CONCURRENCY = 8

## Characters of the one line a failure is reported with; the service's limit.
_ERROR_LINE_MAX = 500

## What the zip adds to each file beyond its bytes and twice its name: a local and a central header (30 + 46 bytes),
## their zip64 extras and a data descriptor, under 150 together; and once, the end records.
_ZIP_FILE_OVERHEAD_BYTES = 256
_ZIP_END_BYTES = 1024

## What a dataset name may be, applied with fullmatch(): the data service's rule
## (which this package cannot import), checked here so a bad name is refused
## before the dataset is read and zipped.
_NAME_RE = re.compile(r"[a-z0-9-]{1,64}")

## How #_NAME_RE reads in a refusal.
_NAME_RULE = "a dataset name uses a-z, 0-9 and '-' only, 1 to 64 characters, e.g. 'towels-v1'"

## How every environment, production included, gets the library: `dreamlabs` from PyPI, which brings this client
## (`dreamlabs-client`) with pyarrow, boto3 and dream's IK (_install_hint()). Dream Labs holds that name on PyPI,
## so the line installs no one else's code; a site that serves its own build installs it the same way
## through its Setup API page.
_INSTALL = "pip install --upgrade dreamlabs"


@dataclasses.dataclass(frozen=True)
class Embodiment:
    """A robot body as the service records it; pass its `name` to upload_dataset().

    Attributes:
    - name: `<username>/<name>`: what upload_dataset() and a dataset name it by.
    - embodiment_config: The config as given to add_embodiment(): `state`, `action` and `video`, and any of the
      optional keys; pass it to validate_dataset().
    - created_at: When it was added, UTC.
    - id: The embodiment's id (`emb_...`), what a fine-tune's #dream.models.Model names as its `embodiment`; None
      from a data service that does not answer it.
    """

    name: str
    embodiment_config: dict[str, Any]
    created_at: datetime.datetime
    id: str | None = None


@dataclasses.dataclass(frozen=True)
class Report:
    """What the local checks found: `ok` and the counts, or the one line that refused the dataset.

    Attributes:
    - ok: Whether the dataset passed; when False, fix `error` and check again before uploading.
    - error: Why it did not pass, one line naming the directory and file; None when it passed.
    - sessions: Dataset directories (one LeRobot dataset each).
    - episodes: Episodes in all.
    - frames: Frames in all.
    - files: Files in all, hidden ones not counted.
    - bytes: Their total size, in bytes.
    - state_dim: The width of `observation.state`, the end of the embodiment's last `state` span.
    - action_dim: The width of `action`, the end of the embodiment's last `action` span.
    - fps: The recording rate `meta/info.json` records, frames per second.
    - video_keys: The video streams of every directory, sorted.
    - notes: What processing will change, and why, one line per directory; nothing to fix. Namely timestamps that
      drift from the videos, which processing re-times as `frame_index / fps` so each frame is paired with its own
      image.
    """

    ok: bool
    error: str | None = None
    sessions: int = 0
    episodes: int = 0
    frames: int = 0
    files: int = 0
    bytes: int = 0
    state_dim: int = 0
    action_dim: int = 0
    fps: int = 0
    video_keys: list[str] = dataclasses.field(default_factory=list)
    notes: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class Dataset:
    """One of your datasets on the service; keep its `id` to follow its status.

    Attributes:
    - id: `ds_` + hex; what get_dataset() (and `DreamClient.get_dataset_status()`) takes.
    - name: `<username>/<name>`; what a fine-tuning job names it by.
    - embodiment: The embodiment's `<username>/<name>`.
    - status: `uploading` until the service has found the zip at the declared size, then `ready_to_process`;
      `processing` while the service processes it and `processed` once it has (the one status a fine-tuning job is
      queued on); `failed` when the upload reported a failure or the processing could not finish.
    - size_bytes: The zip's size as declared, in bytes.
    - episodes: Episodes in the dataset, as the local checks counted them.
    - error: Why the upload or the processing failed, one line; None otherwise.
    - created_at: When it was registered, UTC (again, when a `failed` or `uploading` name was reused).
    """

    id: str
    name: str
    embodiment: str
    status: str
    size_bytes: int
    episodes: int
    error: str | None
    created_at: datetime.datetime


def add_embodiment(name: str, config: dict[str, Any], *, api_key: str | None = None) -> Embodiment:
    """Records a robot body for your account; the config is checked here before it is sent.

    The config is a dict:

    - `state`: each part of the `observation.state` vector, by name, to its `[start, end)` span. The parts are your
      robot's: their names, their widths and how many there are follow its arms. A body that records joints names
      them, e.g. `{"right_arm": [0, 6], "right_gripper": [6, 7]}` for one six-joint arm; one that records the
      end-effector pose names the EEF parts, `{"right_eef_xyz": [0, 3], "right_eef_rot6": [3, 9], "right_grip":
      [9, 10]}`, alone or beside the joints. The spans tile `[0, width)` with no gap or overlap, 128 wide at most.
    - `action`: the same for the `action` vector, which is what the model learns to answer: joint targets for the
      joint parts and end-effector targets for the EEF parts, as the dataset recorded them.
    - `video`: each camera role to the dataset's video key it reads, e.g. `{"overhead": "observation.images.top"}`;
      at least one.
    - `head_camera` (optional when a role is named `overhead`): the role of the fixed head camera, the view the
      service labels the dataset's tasks from; left out, the `overhead` role is used, and a body with neither is
      refused at registration.

    Part names and roles are lowercase letters, digits and underscores, starting with a letter. No other key is
    taken; `state_space` and `action_space`, which older configs carried, are refused with a line saying so
    (`shared.datasets.embodiments.check_registration`). A body that records its end-effector pose names the six EEF
    parts (`right_eef_xyz`, `right_eef_rot6`, `right_grip` and the left three) among its parts like any other;
    whether a fine-tune trains them is `dream.fine_tuning.create`'s `use_eef`, not the config's.

    @param name `[a-z0-9-]{1,64}`, unique among your embodiments.
    @param config The embodiment config above.
    @param api_key Your API key; the credentials' when None.
    @return The new #Embodiment, named `<username>/<name>`: pass `name` to upload_dataset().
    @throws DreamConfigError The config is not valid (the message says what is wrong), or there is no key.
    @throws DreamAuthError The key was refused, or the account has no username yet (`forbidden`).
    @throws DreamServerError `conflict` when you have an embodiment of that name; `invalid_request` for a bad name;
        a `retryable` one (HTTP 5xx) may be made again after a short wait. Every refusal's message names the
        embodiment and the error code, and nothing of how the service is built.
    @throws DreamProtocolError The service could not be reached, or did not answer as the Dream API.
    """
    _parse(config, registering=True)
    api_key, host = _resolve(api_key)
    body = {"name": name, "embodiment_config": config}
    try:
        answer = _call("POST", host, "/v1/embodiments", api_key, body)
    except DreamServerError as error:
        code = f" ({error.code})" if error.code else ""
        # The service's own line only: a refused key's message also names the URL called and where it was set.
        line = error.message.split("; called ", 1)[0]
        hint = "; an API key works only on the site that made it" if isinstance(error, DreamAuthError) else ""
        raise type(error)(
            f"could not create the embodiment {name!r}{code}: {line}{hint}",
            code=error.code,
            reason=error.reason,
            retryable=error.retryable,
            retry_after_s=error.retry_after_s,
        ) from None
    return _embodiment_record(answer)


def list_embodiments(*, api_key: str | None = None) -> list[Embodiment]:
    """Your account's embodiments.

    @param api_key Your API key; the credentials' when None.
    @return A list of #Embodiment, newest first; an `embodiment_config` is what validate_dataset() takes, and an
        `id` is what a fine-tune's #dream.models.Model names as its `embodiment`.
    @throws DreamConfigError There is no key (and the credentials name none).
    @throws DreamAuthError The key was refused, or the account has no username yet (`forbidden`).
    @throws DreamServerError The service refused the call; a `retryable` one (HTTP 5xx, already retried by the
        client) may be made again after a short wait.
    @throws DreamProtocolError The service could not be reached, or did not answer as the Dream API.
    """
    api_key, host = _resolve(api_key)
    return [_embodiment_record(found) for found in _list(_call("GET", host, "/v1/embodiments", api_key, retry=True))]


def validate_dataset(path: str | os.PathLike[str], config: dict[str, Any]) -> Report:
    """Runs the dataset checks on a local directory against an embodiment config; nothing is sent, no key needed.

    These are the checks upload_dataset() runs before it sends anything, so a dataset that passes here passes there.
    They include each video's format, read with FFmpeg: H.264, HEVC, AV1 or VP9 in MP4, without a subtitle track, as
    the service requires.

    @param path The dataset directory: one LeRobot v2.1 dataset per subdirectory, or one such dataset itself; `~` is
        expanded, and hidden files are skipped.
    @param config The embodiment config of the body it was recorded on (an #Embodiment's `embodiment_config`).
    @return The #Report: `ok`, the counts and the `notes` on what processing will change, or `ok=False` and `error`,
        the one line to fix before uploading (a file that cannot be read is such a line too).
    @throws DreamConfigError The directory does not exist or cannot be listed, pyarrow is not installed (the message
        says how to install the library), or the config is not valid.
    """
    return _validate(path, _parse(config))


def upload_dataset(
    path: str | os.PathLike[str],
    name: str,
    embodiment: str,
    *,
    api_key: str | None = None,
) -> Dataset:
    """Checks a local dataset against its embodiment, sends it to S3 as one zip and completes it.

    The zip is written to the system's temporary directory (`tempfile.gettempdir()`: `TMPDIR` when set) and removed
    before this returns or raises. It is stored uncompressed, so that directory needs free space about the dataset's
    size; the call checks it before writing. A failure after the registration is reported to the service (the dataset
    is `failed` with the same line) before it is raised, and the name may be used again: running the same call again
    reuses the dataset. A name that is `ready_to_process`, `processing` or `processed` is taken for good. Each of the
    checks' notes (#Report `notes`: what processing will change) is shown as a `UserWarning` before the upload.

    @param path The dataset directory: one LeRobot v2.1 dataset per subdirectory, or one such dataset itself.
    @param name `[a-z0-9-]{1,64}`; free unless one of your datasets of that name is `ready_to_process`,
        `processing` or `processed`.
    @param embodiment The body it was recorded on: one of your own, as `<username>/<name>` (an #Embodiment's
        `name`); its config is fetched from the service.
    @param api_key Your API key; the credentials' when None.
    @return The #Dataset as the service answered the completion, `ready_to_process`. Keep its `id` and poll
        get_dataset() until `status` is `processed` (then queue a fine-tuning job on its `name`) or `failed` (`error`
        says why).
    @throws DreamConfigError `name` is not `[a-z0-9-]{1,64}` (checked before anything is read or sent), the
        directory does not exist or cannot be listed, pyarrow or boto3 is not installed (the message says how to
        install the library), there is no key, or the temporary directory has less free space than the zip needs (the
        message gives both sizes and says to set `TMPDIR`; nothing was written or sent).
    @throws DreamDatasetError The local checks refused the dataset; nothing was sent.
    @throws DreamUploadError The zip could not be written, the transfer did not finish, or the service did not find
        the zip at the declared size; after the registration the dataset is `failed`. Run the same call again.
    @throws DreamAuthError The key was refused, or the account has no username yet (`forbidden`).
    @throws DreamServerError The service refused the registration: `conflict` for a name that is
        `ready_to_process`, `processing` or `processed`, `not_found` for an embodiment that is not yours; a
        `retryable` one (HTTP 5xx) may be made again after a short wait.
    @throws DreamProtocolError The service could not be reached, or did not answer as the Dream API.
    """
    if not _NAME_RE.fullmatch(name):
        raise DreamConfigError(f"dataset name {name!r} is not valid: {_NAME_RULE}.")
    api_key, host = _resolve(api_key)
    _transfer()
    body = _fetch_embodiment(embodiment, api_key, host)
    report = _validate(path, _parse(body.embodiment_config))
    if not report.ok:
        raise DreamDatasetError(report.error)
    for note in report.notes:
        warnings.warn(note, stacklevel=2)
    root, entries, strays = _local_files(path)
    if strays:
        warnings.warn(f"Not uploaded: {', '.join(strays)} (not inside a session folder).", stacklevel=2)
    zip_path = _temporary_zip()
    try:
        _check_temporary_space(entries)
        try:
            _write_zip(zip_path, root, entries)
        except (OSError, ValueError) as error:  # ValueError: a file dated before 1980, which a zip cannot record
            raise DreamUploadError(f"cannot write the dataset zip: {error}") from error
        registration = {
            "name": name,
            "embodiment": body.name,
            "size_bytes": zip_path.stat().st_size,
            "episodes": report.episodes,
        }
        upload = _call("POST", host, "/v1/datasets", api_key, registration)
        return _deliver(zip_path, upload, api_key, host)
    finally:
        zip_path.unlink(missing_ok=True)


def get_dataset(dataset_id: str, *, api_key: str | None = None) -> Dataset:
    """One of your datasets by its id, with the status the service holds now.

    @param dataset_id The dataset's id (`ds_` + hex), as #upload_dataset answered it.
    @param api_key Your API key; the credentials' when None.
    @return The #Dataset: the same fields upload answered, `status` and `error` as they stand now
        (`ready_to_process`, `processing`, `processed` or `failed`). Poll until `processed` (a fine-tuning job can
        then be queued on it) or `failed` (`error` says why).
    @throws DreamConfigError There is no key (and the credentials name none).
    @throws DreamAuthError The key was refused, or the account has no username yet (`forbidden`).
    @throws DreamServerError `not_found` when no dataset of that id is yours; a `retryable` one (HTTP 5xx, already
        retried by the client) may be made again after a short wait.
    @throws DreamProtocolError The service could not be reached, or did not answer as the Dream API.
    """
    api_key, host = _resolve(api_key)
    return _dataset(_call("GET", host, f"/v1/datasets/{urllib.parse.quote(dataset_id, safe='')}", api_key, retry=True))


def list_datasets(*, api_key: str | None = None) -> list[Dataset]:
    """Your account's datasets, newest first, each with the status the service holds now.

    @param api_key Your API key; the credentials' when None.
    @return A list of #Dataset, newest first: every dataset you registered, a `failed` or `deleted` one included,
        with its `id` (what #get_dataset, #retry_dataset and #delete_dataset take) and its `name` (what a
        fine-tuning job names it by).
    @throws DreamConfigError There is no key (and the credentials name none).
    @throws DreamAuthError The key was refused, or the account has no username yet (`forbidden`).
    @throws DreamServerError The service refused the call; a `retryable` one (HTTP 5xx, already retried by the
        client) may be made again after a short wait.
    @throws DreamProtocolError The service could not be reached, or did not answer as the Dream API.
    """
    api_key, host = _resolve(api_key)
    return [_dataset(found) for found in _list(_call("GET", host, "/v1/datasets", api_key, retry=True))]


def retry_dataset(dataset_id: str, *, api_key: str | None = None) -> Dataset:
    """Has a dataset whose processing failed processed again, from the zip already uploaded; nothing is sent again.

    @param dataset_id The dataset's id (`ds_` + hex), as #upload_dataset answered it.
    @param api_key Your API key; the credentials' when None.
    @return The #Dataset, `ready_to_process`; poll #get_dataset as after an upload. A retry that raced another call
        on the same dataset (a second retry, an upload under its name) answers it as that call left it.
    @throws DreamConfigError There is no key (and the credentials name none).
    @throws DreamAuthError The key was refused, or the account has no username yet (`forbidden`).
    @throws DreamServerError `not_found` when no dataset of that id is yours, or the deployment offers no retries;
        `conflict` when it is not `failed`, or its upload never finished (upload it again under the same name
        instead).
    @throws DreamProtocolError The service could not be reached, or did not answer as the Dream API.
    """
    api_key, host = _resolve(api_key)
    path = f"/v1/datasets/{urllib.parse.quote(dataset_id, safe='')}/retry"
    return _dataset(_call("POST", host, path, api_key))


def delete_dataset(dataset_id: str, *, confirm: bool = True, api_key: str | None = None) -> Dataset | None:
    """Deletes a dataset by its id after asking you to confirm.

    A dataset with a fine-tuning job queued or running on it is refused until the job is over.

    The answer is the dataset `deleted`, and get_dataset() answers it the same from then on.

    @param dataset_id The dataset's id (`ds_` + hex), as #upload_dataset answered it.
    @param confirm Ask `Delete dataset <id>? This cannot be undone. [y/N]` on the terminal first; False skips it.
    @param api_key Your API key; the credentials' when None.
    @return The #Dataset, `deleted`; None when you did not confirm, and nothing was sent.
    @throws DreamConfigError There is no key (and the credentials name none).
    @throws DreamAuthError The key was refused, or the account has no username yet (`forbidden`).
    @throws DreamServerError `not_found` when no dataset has that id; `conflict` while a fine-tuning job is queued or
        running on it where fine-tuning is priced.
    @throws DreamProtocolError The service could not be reached, or did not answer as the Dream API.
    """
    if confirm and not _confirmed(f"Delete dataset {dataset_id}? This cannot be undone. [y/N] "):
        return None
    api_key, host = _resolve(api_key)
    return _dataset(_call("DELETE", host, f"/v1/datasets/{urllib.parse.quote(dataset_id, safe='')}", api_key))


def _confirmed(question: str) -> bool:
    """Whether the user typed `y` or `yes` to the question; no terminal to ask on counts as no."""
    try:
        return input(question).strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _deliver(zip_path: pathlib.Path, upload: Any, api_key: str, host: str) -> Dataset:
    """Sends the zip and completes the dataset; whatever fails on the way is reported with `fail` first. The report
    is best effort: a service that cannot be reached leaves the dataset `uploading`, which the rerun reuses too."""
    try:
        dataset_id = upload["dataset"]["id"]
    except (KeyError, TypeError) as error:
        raise _unreadable("a registration", error) from None
    try:
        _send_zip(zip_path, upload)
        completed = _call("POST", host, f"/v1/datasets/{dataset_id}/complete", api_key, retry=True)
    except Exception as error:
        line = _one_line(error)
        with contextlib.suppress(DreamError):
            _call("POST", host, f"/v1/datasets/{dataset_id}/fail", api_key, {"error": line}, retry=True)
        raise DreamUploadError(line) from error
    return _dataset(completed)


def _check_temporary_space(entries: list[Any]) -> None:
    """Refuses an upload whose zip will not fit in the temporary directory, before a byte of it is written (only the
    empty file _temporary_zip() made is there): the zip is stored uncompressed, so it needs the files' bytes plus
    their headers free."""
    directory = tempfile.gettempdir()
    needed = _ZIP_END_BYTES + sum(e.size_bytes + _ZIP_FILE_OVERHEAD_BYTES + 2 * len(e.path.encode()) for e in entries)
    free = shutil.disk_usage(directory).free
    if free < needed:
        raise DreamConfigError(
            f"the upload first writes the dataset as one uncompressed zip in {directory}, which needs {_size(needed)} "
            f"free there and has {_size(free)}. Set TMPDIR to a directory on a disk with room "
            "(TMPDIR=/path/with/space python upload.py, or tempfile.tempdir = '/path/with/space' in a running Python) "
            "and run the upload again."
        )


def _size(n: int) -> str:
    """A byte count as the space line reads it: GiB to two places, then the exact bytes."""
    return f"{n / 1024**3:.2f} GiB ({n:,} bytes)"


def _temporary_zip() -> pathlib.Path:
    """An empty file for the zip in the system's temporary directory; one that cannot be made there is the same
    failure as a zip that cannot be written."""
    try:
        handle, temporary = tempfile.mkstemp(prefix="dream-dataset-", suffix=".zip")
    except OSError as error:
        raise DreamUploadError(f"cannot write the dataset zip: {error}") from error
    os.close(handle)
    return pathlib.Path(temporary)


def _write_zip(zip_path: pathlib.Path, root: pathlib.Path, entries: list[Any]) -> None:
    """The dataset's files, in the order the checks listed them, stored uncompressed under their paths."""
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for entry in entries:
            archive.write(root / entry.path, arcname=entry.path)


def _send_zip(zip_path: pathlib.Path, upload: dict[str, Any]) -> None:
    """Puts the zip at the key the service answered, on its credentials, with boto3's transfer manager: one PUT below
    `_PART_BYTES`, parts of that size from there up, `_TRANSFER_CONCURRENCY` at once. A transfer that fails aborts
    its own multipart upload; one the process did not survive leaves its parts for the bucket's lifecycle rule."""
    Session, TransferConfig, Config = _transfer()
    granted = upload["credentials"]
    session = Session(
        aws_access_key_id=granted["access_key_id"],
        aws_secret_access_key=granted["secret_access_key"],
        aws_session_token=granted["session_token"],
        region_name=upload["region"],
    )
    endpoint_url = upload["endpoint_url"]
    # Path-style addressing for a custom endpoint (moto on one port has no per-bucket hostnames).
    config = Config(s3={"addressing_style": "path" if endpoint_url else "virtual"})
    client = session.client("s3", endpoint_url=endpoint_url, config=config)
    transfer = TransferConfig(
        multipart_threshold=_PART_BYTES, multipart_chunksize=_PART_BYTES, max_concurrency=_TRANSFER_CONCURRENCY
    )
    client.upload_file(str(zip_path), upload["bucket"], upload["key"], Config=transfer)


def _one_line(error: BaseException) -> str:
    """What an exception says, on one line of at most `_ERROR_LINE_MAX` characters; its type when it says nothing."""
    text = " ".join(str(error).split()) or type(error).__name__
    return text[:_ERROR_LINE_MAX]


def _validate(path: str | os.PathLike[str], body: embodiments.Embodiment) -> Report:
    """The checks over a local directory against a parsed body, as a #Report; a file that cannot be read fails the
    dataset with its own line, as a malformed one does."""
    validation = _validation()
    root, entries, _ = _local_files(path)
    files = [validation.FileEntry(path=entry.path, size_bytes=entry.size_bytes) for entry in entries]
    try:
        summary = validation.check_dataset(files, body, validation.DirectoryReader(root))
    except validation.content_checks.DatasetInvalidError as error:
        return Report(ok=False, error=str(error))
    except OSError as error:
        unreadable = error.filename or "a file of the dataset"
        return Report(ok=False, error=f"Cannot read {unreadable}: {error.strerror or error}.")
    return Report(ok=True, **summary)


def _local_files(path: str | os.PathLike[str]) -> tuple[pathlib.Path, list[Any], list[str]]:
    """The dataset root, its files and the names of the strays left out of them (`validation.is_stray()`), hidden
    files left out too; a directory that is one LeRobot dataset is its own single session. A directory that does not
    exist, or cannot be listed, is the caller's to fix."""
    validation = _validation()
    try:
        given = pathlib.Path(os.path.expanduser(path)).resolve()
        is_directory = given.is_dir()
        is_session = is_directory and (given / validation.INFO_FILE).is_file()
        listed = validation.list_files(given) if is_directory else []
    except (OSError, ValueError) as error:  # no permission, or a path holding a NUL byte
        raise DreamConfigError(f"cannot list the dataset directory {os.fspath(path)!r}: {error}") from None
    if not is_directory:
        raise DreamConfigError(f"{given} is not a directory")
    root, entries = given, listed
    if is_session:
        root, entries = given.parent, [dataclasses.replace(e, path=f"{given.name}/{e.path}") for e in listed]
    strays = [entry.path for entry in entries if validation.is_stray(entry.path)]
    return root, [entry for entry in entries if not validation.is_stray(entry.path)], strays


def _fetch_embodiment(name: str, api_key: str, host: str) -> Embodiment:
    """One of the caller's embodiments by its `<username>/<name>`, from the service. The split is at the last slash:
    a name never holds one, a username (an email's local part) may."""
    username, _, rest = name.rpartition("/")
    quoted = f"{urllib.parse.quote(username, safe='')}/{urllib.parse.quote(rest, safe='')}"
    return _embodiment_record(_call("GET", host, f"/v1/embodiments/{quoted}", api_key, retry=True))


def _parse(config: dict[str, Any], *, registering: bool = False) -> embodiments.Embodiment:
    """The body a config describes; a config that describes none is the caller's mistake. A config being registered
    is also held to `check_registration` (the retired keys are refused); a stored one is read with `parse` alone."""
    try:
        return embodiments.check_registration(config) if registering else embodiments.parse(config)
    except embodiments.EmbodimentInvalidError as error:
        raise DreamConfigError(str(error)) from None
    except TypeError:
        # The parser reads JSON, whose keys are strings; a Python dict keyed by anything else fails its name match.
        raise DreamConfigError("Part names and camera roles must be strings.") from None


def _resolve(api_key: str | None) -> tuple[str, str]:
    """The key a call uses, the argument's or else the credentials', and the data service's URL, always the
    credentials'."""
    loaded = credentials.load()
    api_key = loaded.api_key if api_key is None else api_key
    host = loaded.data_url
    if api_key is None:
        raise DreamConfigError(
            f"no api key: pass api_key, or set {credentials.API_KEY_ENV} in the environment or in "
            f"{credentials.file_path()}"
        )
    return api_key, host


def _install_hint() -> str:
    """How to install the library, which carries pyarrow, boto3 and dream's IK: `dreamlabs` from PyPI (`_INSTALL`)."""
    return f"install the library: {_INSTALL}"


def _validation() -> Any:
    """`shared.datasets.validation`, imported on first use: it opens parquets with pyarrow and reads videos' formats
    with PyAV, both in an optional extra; PyAV is imported here so its absence reads as a missing install, not as a
    video it cannot read."""
    try:
        import av  # noqa: F401, PLC0415 - the video-format check imports it per file; this fails once, up front
        from shared.datasets import validation
    except ImportError as error:
        raise DreamConfigError(f"the dataset checks need pyarrow and av ({error}); {_install_hint()}") from None
    return validation


def _transfer() -> "tuple[type[boto3.Session], type[TransferConfig], type[Config]]":
    """boto3's `Session`, its `TransferConfig` and botocore's `Config`, imported on first use: the transfer needs
    boto3, an optional extra. upload_dataset() calls this before its first request, so a machine that cannot send
    makes none."""
    try:
        from boto3 import Session
        from boto3.s3.transfer import TransferConfig
        from botocore.config import Config
    except ImportError as error:
        raise DreamConfigError(f"the upload needs boto3 ({error}); {_install_hint()}") from None
    return Session, TransferConfig, Config


def _call(
    method: str, host: str, path: str, api_key: str, body: dict[str, Any] | None = None, *, retry: bool = False
) -> Any:
    """One JSON request to the data service or the platform; `retry` repeats one that is safe to repeat on a 5xx or a
    lost connection, with backoff. A host, a key or a body HTTP cannot carry is the caller's to fix, never retried.

    Every request names this library as `User-Agent: dream-client/<dream.__version__>`: the platform lists a body's
    `parts` only to a client that converts them (`GET /v1/models`)."""
    try:
        parts = urllib.parse.urlsplit(host)
        _ = parts.port  # A port that is not a number in 0..65535 raises here.
    except ValueError as error:
        raise DreamConfigError(f"host must be an http(s) URL, got {host!r} ({error})") from None
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise DreamConfigError(f"host must be an http(s) URL, got {host!r}")
    headers = {
        **credentials._authorization(api_key),
        "Accept": "application/json",
        "User-Agent": f"{transport.USER_AGENT_PRODUCT}/{transport._client_version()}",
    }
    try:
        payload = None if body is None else json.dumps(body).encode()
    except (TypeError, ValueError) as error:  # a numpy number, say, where the service takes a JSON number
        raise DreamConfigError(
            f"the request to {host} cannot be sent as JSON ({error}); pass plain Python values"
        ) from None
    if payload is not None:
        headers["Content-Type"] = "application/json"
    attempts = _ATTEMPTS if retry else 1
    kind = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    for attempt in range(attempts):
        try:
            connection = kind(parts.netloc, timeout=_TIMEOUT_S)
        except http.client.InvalidURL as error:  # a host name with a space or a control character
            raise DreamConfigError(f"host must be an http(s) URL, got {host!r} ({error})") from None
        try:
            connection.request(method, parts.path.rstrip("/") + path, body=payload, headers=headers)
            response = connection.getresponse()
            status, reply = response.status, response.read()
        except ValueError as error:  # UnicodeError: a host name the resolver cannot encode (an empty label)
            raise DreamConfigError(f"cannot send to {host}: {error}") from None
        except (OSError, http.client.HTTPException) as error:
            if attempt + 1 == attempts:
                raise DreamProtocolError(f"cannot reach the Dream service at {host}: {error}") from None
            time.sleep(_backoff_s(attempt))
            continue
        finally:
            connection.close()
        if status >= 500 and attempt + 1 < attempts:
            time.sleep(_backoff_s(attempt))
            continue
        return _answer(status, reply, host)
    raise AssertionError("unreachable")


def _backoff_s(attempt: int) -> float:
    """The wait before retry number `attempt + 1`: exponential with full jitter."""
    return random.uniform(0, min(_BACKOFF_MAX_S, _BACKOFF_BASE_S * 2**attempt))


def _answer(status: int, reply: bytes, host: str) -> Any:
    """The decoded body of a success, or the error its refusal is."""
    try:
        decoded = json.loads(reply) if reply else None
    except ValueError:
        raise DreamProtocolError(f"the Dream service at {host} answered {status}, not with JSON") from None
    if status < 400:
        return decoded
    envelope = decoded.get("error") if isinstance(decoded, dict) else None
    code = envelope.get("code") if isinstance(envelope, dict) else None
    message = envelope.get("message") if isinstance(envelope, dict) else None
    message = message or f"the Dream service answered {status}"
    if status in (401, 403):
        where = credentials.describe_url(host)
        raise DreamAuthError(f"{message}; called {where}; {credentials.KEY_PER_SITE}", code=code)
    raise DreamServerError(message, code=code, retryable=status >= 500)


def _list(answer: Any) -> list[Any]:
    """The service's answer to a listing, which is a JSON list; anything else is the wrong thing listening."""
    if not isinstance(answer, list):
        raise DreamProtocolError(f"the Dream service answered {type(answer).__name__}, not a JSON list")
    return answer


def _unreadable(what: str, error: Exception) -> DreamProtocolError:
    """The error for an answer that is JSON but not the shape the Dream API answers: another service at the URL."""
    return DreamProtocolError(
        f"the Dream service answered {what} the client cannot read ({type(error).__name__}: {error}); is the URL "
        "the Dream service's?"
    )


def _embodiment_record(found: Any) -> Embodiment:
    """An embodiment as the service answers it."""
    try:
        return Embodiment(
            name=found["name"],
            embodiment_config=found["embodiment_config"],
            created_at=_timestamp(found["created_at"]),
            id=found.get("id"),
        )
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise _unreadable("an embodiment", error) from None


def _dataset(found: Any) -> Dataset:
    """A dataset as the service answers it."""
    try:
        return Dataset(
            id=found["id"],
            name=found["name"],
            embodiment=found["embodiment"],
            status=found["status"],
            size_bytes=found["size_bytes"],
            episodes=found["episodes"],
            error=found["error"],
            created_at=_timestamp(found["created_at"]),
        )
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise _unreadable("a dataset", error) from None


def _timestamp(text: str) -> datetime.datetime:
    """An ISO 8601 time as the service writes it; 3.10's parser does not take the `Z` suffix."""
    return datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
