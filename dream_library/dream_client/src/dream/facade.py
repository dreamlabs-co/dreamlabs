"""DreamClient: one object for the whole Dream Labs lifecycle, from a robot body to a running policy.

`DreamClient` is the public surface for everything an account does over the hosted services: record an embodiment,
check and upload a dataset, fine-tune Dream-1 on it, list the models it may run, and call a model for actions. It is a
thin, stateful front over the module functions (`dream.datasets`, `dream.fine_tuning`, `dream.models`) and the
model itself: the modules stay usable on their own, and this class carries the key so a caller does not pass it on
every call.

The key comes from the argument, else `DREAM_API_KEY` in the environment or in `~/.dreamlabs/credentials`. Where the
Dream Labs services are is set by Dream Labs, not by your code: the data service answers the embodiment, dataset and
fine-tuning calls, the platform answers get_models(), and the model answers get_action().

Running a model is three calls: policy() chooses the checkpoint and starts an empty frame window, set_observation()
hands the client every camera tick (one frame per camera and the latest state), and get_action() sends the frames the
window kept with the newest state and returns the next chunk of actions. Opening a policy is the one call that needs a
checkpoint; the rest of the surface does not.

@par Example
```python
import dream

client = dream.DreamClient()  # the key comes from ~/.dreamlabs/credentials

config = {
    "state": {"arm": [0, 6], "gripper": [6, 7]},
    "action": {"arm": [0, 6], "gripper": [6, 7]},
    "video": {"overhead": "observation.images.top", "wrist": "observation.images.wrist"},
}
client.add_embodiment("metal-arm", config)
report = client.validate_dataset("~/data/towels", config)
assert report.ok, report.error
dataset = client.upload_dataset("~/data/towels", "towels-v1", "<your-username>/metal-arm")

# ... poll client.get_dataset_status(dataset.id) until its status is "processed" ...

job = client.finetune(dataset.name, "towels")

# ... poll client.get_finetuning_status(job["id"]) until job["fine_tuned_model"] is set ...

client.policy({"checkpoint": job["fine_tuned_model"]})
while running:
    client.set_observation(frames, observation)  # every camera tick: one base64 JPEG per camera, and the state
    if due:
        chunk = client.get_action()  # the frames kept, with the newest state
client.close()
```
"""

from __future__ import annotations

import math
import time
import uuid
from collections.abc import Mapping
from types import TracebackType
from typing import TYPE_CHECKING, Any

from shared.datasets import embodiments

from dream import _recorder, _window
from dream import credentials as _credentials
from dream import datasets as _datasets
from dream import fine_tuning as _fine_tuning
from dream import models as _models
from dream.datasets import Dataset, Embodiment, Report
from dream.errors import DreamConfigError, DreamError, DreamProtocolError
from dream.models import Model

if TYPE_CHECKING:
    from dream.client import PolicyClient

__all__ = ["DreamClient", "Dataset", "Embodiment", "Model", "Report"]

## The `policy()` config keys that would say where the model is; refused, since only the credentials say that.
_GATEWAY_KEYS = frozenset({"host", "port", "scheme"})


class DreamClient:
    """One client for embodiments, datasets, fine-tuning, models and inference, with the key bound once.

    Construct it directly: `dream.DreamClient()`. The key comes from the argument or your credentials; where the
    Dream Labs services are is set by your credentials and cannot be changed from code. The embodiment, dataset,
    fine-tuning and model methods forward to the matching module function with this client's key, and hold nothing
    open. policy(), set_observation(), get_action() and reset_episode() run a model.

    policy() opens a session on the model, which close() releases; the class is also a context manager. One policy is
    held at a time: a second policy() replaces the first, closing it. set_observation() and get_action() may run on
    different threads at once, a control loop's tick and a worker asking for the next chunk: a lock guards the window
    only while a tick goes in or the observation is copied out, never during a request, so set_observation() never
    waits for the model. Two get_action() at once are refused.
    """

    def __init__(self, api_key: str | None = None) -> None:
        """Binds the key; nothing is sent and no session is opened.

        A missing key is not an error here: the first call that needs it raises #DreamConfigError.

        @param api_key Your API key; the credentials' (`DREAM_API_KEY`) when None.
        @throws DreamConfigError The credentials file exists but cannot be read, or names a service that is not a URL.
        """
        loaded = _credentials.load()
        self._api_key = loaded.api_key if api_key is None else api_key
        # Where the platform and the model are, the credentials' alone: read once here, never set from code.
        self._platform_url = loaded.platform_url
        self._gateway_url = loaded.gateway_url
        self._policy: PolicyClient | None = None
        ## The open policy's frame window: None until policy(), which starts a new one, and after close().
        self._window: _window.Window | None = None
        ## The open policy's recording: None without a policy, after close(), and when it could not start.
        self._recording: _recorder.Recorder | None = None
        ## Why the open policy is not recording, one line, when its recording could not start; None otherwise.
        self._recording_off: str | None = None
        ## The robot id every recording of this client is sent under, `rec_` and 32 hex digits, made by the first
        ## policy() that records; None before.
        self._robot_id: str | None = None
        ## The episode recordings are numbered by: 0 from the first policy(), then one more at each reset_episode()
        ## and each later policy(), so one robot id's episodes never share a number.
        self._episode = -1
        ## Checkpoint id to the embodiment config its observations are checked against (None: nothing to check),
        ## filled by #_embodiment_config_for so a checkpoint is looked up once per client.
        self._embodiment_configs: dict[str, dict[str, Any] | None] = {}

    # -- embodiments -------------------------------------------------------------------------------------------

    def add_embodiment(self, name: str, config: dict[str, Any]) -> Embodiment:
        """Records a robot body for the account; the config is checked here before it is sent.

        The config is a dict:

        - `state`: each part of the `observation.state` vector, by name, to its `[start, end)` span. The parts are your
          robot's: their names, their widths and how many there are follow its arms. A body that records joints names
          them, e.g. `{"right_arm": [0, 6], "right_gripper": [6, 7]}` for one six-joint arm; one that records the
          end-effector pose names the EEF parts, `{"right_eef_xyz": [0, 3], "right_eef_rot6": [3, 9], "right_grip":
          [9, 10]}`, alone or beside the joints. The spans tile `[0, width)` with no gap or overlap, 128 wide at most.
        - `action`: the same for the `action` vector, which is what the model learns to answer: joint targets for the
          joint parts and end-effector targets for the EEF parts, as the dataset recorded them.
        - `video`: each camera role to the dataset's video key it reads, e.g.
          `{"overhead": "observation.images.top"}`; at least one.
        - `head_camera` (optional when a role is named `overhead`): the role of the fixed head camera, the view
          the service labels the dataset's tasks from; left out, the `overhead` role is used, and a body with
          neither is refused at registration.

        Part names and roles are lowercase letters, digits and underscores, starting with a letter. No other key is
        taken; `state_space` and `action_space`, which older configs carried, are refused with a line saying so. A
        body that records its end-effector pose names the six EEF parts (`right_eef_xyz`, `right_eef_rot6`,
        `right_grip` and the left three) among its parts like any other; whether a fine-tune trains them is
        finetune()'s `use_eef`, not the config's.

        @param name `[a-z0-9-]{1,64}`, unique among the account's embodiments.
        @param config The embodiment config above.
        @return The new #dream.datasets.Embodiment, named `<username>/<name>`: pass its `name` to upload_dataset(),
            and its `embodiment_config` to validate_dataset().
        @throws DreamConfigError The config is not valid (the message says what is wrong), or there is no key.
        @throws DreamAuthError The key was refused, or the account has no username yet (`forbidden`: ask Dream Labs
            support to set one).
        @throws DreamServerError `conflict` when a body of that name exists; `invalid_request` for a bad name; a
            `retryable` one (HTTP 5xx) may be made again after a short wait.
        @throws DreamProtocolError The data service could not be reached, or did not answer as the Dream API.
        """
        return _datasets.add_embodiment(name, config, api_key=self._api_key)

    def get_embodiments(self) -> list[Embodiment]:
        """The account's own embodiments, newest first.

        @return A list of #dream.datasets.Embodiment. An `embodiment_config` is what validate_dataset() takes; an `id`
            is what a fine-tune's #dream.models.Model names as its `embodiment`.
        @throws DreamConfigError There is no key.
        @throws DreamAuthError The key was refused, or the account has no username yet (`forbidden`: ask Dream Labs
            support to set one).
        @throws DreamServerError The data service refused the call; a `retryable` one (HTTP 5xx, already retried by
            the client) may be made again after a short wait.
        @throws DreamProtocolError The data service could not be reached, or did not answer as the Dream API.
        """
        return _datasets.list_embodiments(api_key=self._api_key)

    # -- models ------------------------------------------------------------------------------------------------

    def get_models(self) -> list[Model]:
        """The checkpoints the account may run: the public ones and its own fine-tunes.

        @return A list of #dream.models.Model, public ones first. A model's `id` is the `checkpoint` policy() takes.
            `servable` means it can be started, not that it is running: the first get_action() to a model that is not
            running is refused `pool_warming` while it starts.
        @throws DreamConfigError There is no key.
        @throws DreamAuthError The key was refused.
        @throws DreamServerError The platform refused the call; a `retryable` one (HTTP 5xx, already retried by the
            client) may be made again after a short wait.
        @throws DreamProtocolError The platform could not be reached, or did not answer as the Dream API.
        """
        return _models.list_models(api_key=self._api_key)

    # -- datasets ----------------------------------------------------------------------------------------------

    def validate_dataset(self, path: str, config: dict[str, Any]) -> Report:
        """Runs the dataset checks on a local directory against an embodiment config; nothing is sent, no key needed.

        These are the checks upload_dataset() runs before it sends anything, so a dataset that passes here passes
        there.

        @param path The dataset directory: one LeRobot v2.1 dataset per subdirectory (`<session>/meta/info.json`,
            `data/`, `videos/`), or one such dataset itself; `~` is expanded, and hidden files are skipped.
        @param config The embodiment config of the body it was recorded on (an #Embodiment's `embodiment_config`).
        @return The #dream.datasets.Report: `ok` and the counts, or `ok=False` and `error`, the one line to fix
            before uploading (a file that cannot be read is such a line too); fix it and check again.
        @throws DreamConfigError The directory does not exist or cannot be listed, pyarrow is missing (the message
            says how to install the library), or the config is not valid.
        """
        return _datasets.validate_dataset(path, config)

    def upload_dataset(self, path: str, name: str, embodiment: str) -> Dataset:
        """Checks a local dataset against its embodiment, sends it to S3 as one zip and completes it.

        The zip is written to the system's temporary directory and removed before this returns or raises. Running the
        same call again after a #dream.errors.DreamUploadError reuses the same dataset.

        @param path The dataset directory: one LeRobot v2.1 dataset per subdirectory, or one such dataset itself.
        @param name `[a-z0-9-]{1,64}`; free unless a dataset of that name is already `ready_to_process` or beyond.
        @param embodiment The body it was recorded on, one of the account's own, as `<username>/<name>` (an
            #Embodiment's `name`).
        @return The #dream.datasets.Dataset as the service answered the completion, `ready_to_process`. Keep its `id`
            (get_datasets() finds it again by `name`) and poll get_dataset_status() until `status` is `processed`
            (then finetune() on its `name`) or `failed` (`error` says why).
        @throws DreamConfigError `name` is not `[a-z0-9-]{1,64}` (checked before anything is read or sent), the
            directory does not exist or cannot be listed, pyarrow or boto3 is missing (the message says how to
            install the library), or there is no key.
        @throws DreamDatasetError The local checks refused the dataset; nothing was sent. Import it from
            `dream.errors`.
        @throws DreamUploadError The zip could not be written, the transfer did not finish, or the service did not
            find the zip at the declared size. Run the same call again. Import it from `dream.errors`.
        @throws DreamAuthError The key was refused, or the account has no username yet (`forbidden`: ask Dream Labs
            support to set one).
        @throws DreamServerError `conflict` for a name in use, `not_found` for an embodiment that is not the
            account's; a `retryable` one (HTTP 5xx) may be made again after a short wait.
        @throws DreamProtocolError The data service could not be reached, or did not answer as the Dream API.
        """
        return _datasets.upload_dataset(path, name, embodiment, api_key=self._api_key)

    def get_dataset_status(self, dataset_id: str) -> Dataset:
        """A dataset's status as the service holds it now; poll it after upload_dataset().

        @param dataset_id The dataset's id (`ds_` + hex), as upload_dataset() answered it.
        @return The #dream.datasets.Dataset: `status` is `ready_to_process`, `processing`, `processed` or `failed`.
            Once it is `processed`, finetune() can queue a job on its `name`; when it is `failed`, `error` says why.
        @throws DreamConfigError There is no key.
        @throws DreamAuthError The key was refused, or the account has no username yet (`forbidden`: ask Dream Labs
            support to set one).
        @throws DreamServerError `not_found` when no dataset of that id is the account's; a `retryable` one (HTTP
            5xx, already retried by the client) may be made again after a short wait.
        @throws DreamProtocolError The data service could not be reached, or did not answer as the Dream API.
        """
        return _datasets.get_dataset(dataset_id, api_key=self._api_key)

    def get_datasets(self) -> list[Dataset]:
        """The account's datasets, newest first, each with the status the service holds now.

        @return A list of #dream.datasets.Dataset, a `failed` or `deleted` one included; its `id` is what
            get_dataset_status(), retry_dataset() and delete_dataset() take, its `name` what finetune() names.
        @throws DreamConfigError There is no key.
        @throws DreamAuthError The key was refused, or the account has no username yet (`forbidden`: ask Dream Labs
            support to set one).
        @throws DreamServerError The data service refused the call; a `retryable` one (HTTP 5xx, already retried by
            the client) may be made again after a short wait.
        @throws DreamProtocolError The data service could not be reached, or did not answer as the Dream API.
        """
        return _datasets.list_datasets(api_key=self._api_key)

    def retry_dataset(self, dataset_id: str) -> Dataset:
        """Has a dataset whose processing failed processed again, from the zip already uploaded.

        @param dataset_id The dataset's id (`ds_` + hex), as upload_dataset() answered it.
        @return The #dream.datasets.Dataset, `ready_to_process`; poll get_dataset_status() as after an upload. A
            retry that raced another call on the same dataset (a second retry, an upload under its name) answers
            it as that call left it.
        @throws DreamConfigError There is no key.
        @throws DreamAuthError The key was refused, or the account has no username yet (`forbidden`).
        @throws DreamServerError `not_found` when no dataset of that id is the account's, or the deployment offers
            no retries; `conflict` when it is not `failed`, or its upload never finished (upload it again under the
            same name instead).
        @throws DreamProtocolError The data service could not be reached, or did not answer as the Dream API.
        """
        return _datasets.retry_dataset(dataset_id, api_key=self._api_key)

    def delete_dataset(self, dataset_id: str, *, confirm: bool = True) -> Dataset | None:
        """Deletes a dataset by its id after asking you to confirm; refused while a job on it is queued or running.

        @param dataset_id The dataset's id (`ds_` + hex), as upload_dataset() answered it.
        @param confirm Ask on the terminal first; False skips the question.
        @return The #dream.datasets.Dataset, `deleted`; None when you did not confirm.
        @throws DreamConfigError There is no key.
        @throws DreamAuthError The key was refused, or the account has no username yet (`forbidden`).
        @throws DreamServerError `not_found` when no dataset has that id.
        @throws DreamProtocolError The data service could not be reached, or did not answer as the Dream API.
        """
        return _datasets.delete_dataset(dataset_id, confirm=confirm, api_key=self._api_key)

    # -- fine-tuning -------------------------------------------------------------------------------------------

    def finetune(
        self,
        dataset: str,
        suffix: str,
        *,
        learning_rate: float | None = None,
        batch_size: int | None = None,
        use_eef: bool = False,
        lr_schedule: str | None = None,
    ) -> dict[str, Any]:
        """Queues a fine-tuning job on one of the account's `processed` datasets.

        A knob left as None is not sent, and the service fills in its default; the job's `hyperparameters` says what it
        trains with. A knob out of range is refused with `invalid_request`, naming it. With `use_eef`, the dataset's
        embodiment is fetched and checked here first (`dream.fine_tuning.check_eef`), so a body that cannot train the
        end-effector pose is refused before anything is queued.

        @param dataset The dataset's `<username>/<name>` (a #dream.datasets.Dataset's `name`), one of the account's
            own and `processed`.
        @param suffix `[a-z0-9-]{1,40}`: the tail of the checkpoint id `dream-1-ft:<uid8>-<suffix>`; free among the
            account's models and active jobs.
        @param learning_rate The learning rate, 1e-6 to 1e-4; None takes the service's default of 1e-5.
        @param batch_size Leave it as None: the service sets it.
        @param use_eef Whether the job trains the end-effector pose beside the joints. The dataset's embodiment must
            then carry the six EEF parts, `right_eef_xyz` (3 wide), `right_eef_rot6` (6), `right_grip` (1) and the
            same three for `left`, in both `state` and `action`. False, the default, trains every part as
            joints.
            A body whose `action` carries `right_eef_rot6` or `left_eef_rot6` trains only with it; without it the
            service refuses the job with `invalid_request`. So does one whose `action` carries `right_eef_xyz` or
            `left_eef_xyz` without that side's `_eef_xyz` (3 wide) and `_eef_rot6` (6 wide) in `state`.
            The pose follows the base model's convention, per arm: `<side>_eef_xyz` is the tool centre point in
            metres (on YAM 0.1347 m past the wrist flange along the last link's z axis), `<side>_eef_rot6` the last
            link's orientation as the first two columns of its rotation matrix, `[R00, R10, R20, R01, R11, R21]`,
            both in that arm's own base frame, and `<side>_grip` the gripper opening, 0 closed to 1 open. A pose
            recorded another way passes every check and trains a mismatched model.
        @param lr_schedule The learning-rate schedule, `"constant"` or `"cosine"`; None takes the service's default,
            `"constant"`. Both warm up first; `"cosine"` then decays the rate to 0 by the last step.
            Cosine decay works best for specialization (a fine-tune on a single task).
            You may not get high reliability on specialist tasks yet.
        @return The job dict, `status` `queued`, with the keys `id` (`ftj_` + hex), `dataset`, `embodiment`,
            `base_model`, `suffix`, `fine_tuned_model` (None until the model is published), `status`,
            `hyperparameters` (the training settings as filled in),
            `queue_position`, `created_at`, `finished_at` and `error` (get_finetuning_status() says what each
            holds). Keep `job["id"]` and poll get_finetuning_status() with it.
        @throws DreamConfigError There is no key, a knob is not a plain number, or `use_eef` is asked of a dataset
            whose embodiment lacks an EEF part or has one at another width (the message names the part and its
            side).
        @throws DreamAuthError The key was refused, or the account has no username yet (`forbidden`: ask Dream Labs
            support to set one).
        @throws DreamServerError `not_found` for a dataset that is not the account's; `conflict` for one that is not
            `processed`, a suffix in use, or the account at its limit of active jobs; `invalid_request` for a suffix
            or a knob out of range, naming it, or for a dataset whose embodiment's action carries an end-effector
            rotation, or a position without its state pose, without `use_eef`; a `retryable` one (HTTP
            5xx) may be made again after a short wait.
        @throws DreamProtocolError The data service could not be reached, or did not answer as the Dream API.
        """
        return _fine_tuning.create(
            dataset,
            suffix,
            learning_rate=learning_rate,
            batch_size=batch_size,
            use_eef=use_eef,
            lr_schedule=lr_schedule,
            api_key=self._api_key,
        )

    def get_finetuning_status(self, job_id: str) -> dict[str, Any]:
        """A fine-tuning job as the service holds it now; poll it after finetune().

        The job dict's keys:

        - `id`: `ftj_` + hex.
        - `dataset`, `embodiment`: the dataset it trains on and that dataset's embodiment, each `<username>/<name>`.
        - `base_model`: the id of the checkpoint it starts from.
        - `suffix`: the suffix it was created with.
        - `fine_tuned_model`: the new checkpoint's id, `dream-1-ft:<uid8>-<suffix>`, once the job has succeeded and the
          checkpoint is published; None before.
        - `status`: `queued` (waiting for a training slot), `running`, `succeeded`, `failed` or `cancelled`.
        - `hyperparameters`: the training settings, as the service filled them in.
        - `queue_position`: 1 for the next job to start, while `queued`; None otherwise.
        - `created_at`, `finished_at`: ISO 8601 times in UTC; `finished_at` is None until the job ends.
        - `error`: one line saying the training did not finish, when `failed`; None otherwise.

        Poll until `fine_tuned_model` is set, not until `succeeded` alone: the checkpoint is published in the same
        pass that marks the job `succeeded`, or a little later. A `failed` or `cancelled` job never gets one, so stop
        polling there.

        @param job_id The job's id (`ftj_` + hex), as finetune() answered it.
        @return The job dict above. Once `fine_tuned_model` is set, `client.policy({"checkpoint":
            job["fine_tuned_model"]})` runs the model.
        @throws DreamConfigError There is no key.
        @throws DreamAuthError The key was refused, or the account has no username yet (`forbidden`: ask Dream Labs
            support to set one).
        @throws DreamServerError `not_found` for a job that is not the account's; a `retryable` one (HTTP 5xx, already
            retried by the client) may be made again after a short wait.
        @throws DreamProtocolError The data service could not be reached, or did not answer as the Dream API.
        """
        return _fine_tuning.get(job_id, api_key=self._api_key)

    # -- inference ---------------------------------------------------------------------------------------------

    def policy(self, config: dict[str, Any]) -> None:
        """Chooses the model and opens an empty frame window; nothing is sent until the first get_action().

        The checkpoint is a model's id: a #dream.models.Model's `id` from get_models(), or a fine-tuning job's
        `fine_tuned_model`. Opening a second policy closes the first, which ends its session and empties its window.

        For a fine-tune of a custom body (the account's own, or a shared or public one whose body another account
        registered), when `config` names no `embodiment` and `strict` is on (the default), the embodiment config of
        the body it was trained on is loaded here, from get_models(), and kept for the session: every get_action()
        then checks the observation before anything is sent, and the model's reply before it is returned, and a body
        with `parts` is converted to the slots its model trained on and back. `strict: False` with no `embodiment`
        loads nothing, so it skips both: calls are sent unchecked and unconverted, in the slot names the model
        trained on, never a robot's own parts. A model of a builtin body (a Dream-1 base model) has no config to load,
        so its calls are not checked. With `strict` on, a lookup that fails raises (`DreamProtocolError`; call
        policy() again), and so do a checkpoint get_models() does not name and a listed config this library cannot
        read: nothing runs unchecked by accident.

        @param config The session config, a dict (yours is not changed): `checkpoint` (required); optionally
            `embodiment` for the checks (the embodiment config the body was registered with, the dict add_embodiment()
            took, or a shape spec; loaded for a fine-tune of a custom body when left out); `obs_window` (int, at least
            1, default 9: frames kept per frame key) and `obs_fps` (a positive, finite number, default 5: frames kept
            per second of set_observation()'s timestamps); `timeout_ms` (milliseconds, default 15000) and `strict`
            (default True); the pacing keys `execution_horizon` (rows, default 30), `action_hz` (rows per second,
            default 20) and `sync_above_ms` (default 700), read by refill_due(); `kinematics`; `robot`, the robot the
            model runs on as a model trained on YAM arms reads it, `"yam"` (a YAM) or `"other_embodiment"` (any other
            arm), which every get_action() sends as the observation's `embodiment` (never put that key in your
            observation): on `"other_embodiment"` the model is given none of the arm's state, neither its joints, its
            hand pose nor its grip, and answers no joints, only each hand's pose, composed on the hand pose the
            observation carries (`per_key`'s `action.right_arm` and `action.left_arm` are None); left out, it is
            `"other_embodiment"` when `ik` runs a model trained on YAM arms on an arm that is not a YAM, and nothing is
            sent otherwise; set it to `"other_embodiment"` to ask the same with IK of your own; and `ik`, which runs a
            model that answers hand poses on an arm that is not a YAM: `{"arm": ..., "head": "eef"}`, the arm being
            `"metal"`, `"makers"` or the path of an arm file, a JSON object with every one of these keys: `urdf` (your
            arm's URDF, a path relative to the arm file), `base` (the link the hand's pose is measured from, oriented x
            forward, y to the arm's left, z up), `flange` (the link the gripper is mounted on), `tcp` (`[x, y, z]`,
            metres in the flange link's frame: the point between the closed fingertips), `approach` (`[x, y, z]` in the
            flange link's frame: the way the fingers point), `jaw` (`[x, y, z]` in the flange link's frame: the line
            the fingers close along, pointing to the arm's left when the hand points straight forward) and `rest` (the
            arm's joints in a working pose, in radians and inside the joint limits: the pose the solve leans toward).
            The user measures and confirms `tcp`, `approach` and `jaw` on the real gripper (`dreamcli tcp` writes the
            arm file); an agent writing this config never picks or estimates them. Both arms run the same arm file,
            each from its own base; the arm's joints are the moving joints from `base` to `flange`, in that order. The
            client then adds each hand's pose to the observation, from `state.<side>_arm`, and returns the reply's
            poses as `q_r_chunk` and `q_l_chunk`: your arm's joints, then the grip (the docs' "On another arm").
            policy() warns once, never refusing (one `UserWarning` line), when the arm file's tool point is not one
            a person set on the real gripper with `dreamcli tcp`: no record of it, a skipped one, one edited since, or
            `tcp` at the flange itself. Every key but `checkpoint`, `embodiment` and `robot` is consumed here and
            never sent.
        @throws DreamConfigError `checkpoint` is missing, a key has the wrong type or range, the config carries a key
            the client does not take (`host`, `port` and `scheme` among them: where the model is comes from your
            credentials), the `embodiment` given is not valid, `robot` is not `"yam"` or `"other_embodiment"`, `ik`
            cannot be served (`dream.client_ik`; its head `joint` with `robot` `"other_embodiment"` neither); a
            `checkpoint` get_models() does not name, a fine-tune of a robot registered with parts the platform lists
            only to the account that registered it (`Model.parts_withheld`), or one whose listed config this library
            cannot read, with `strict` on and no `embodiment` (the message says to pass it). `obs_window` and `obs_fps`
            are checked before anything else, so a policy already open stays open when they are wrong.
        @throws DreamProtocolError With `strict` on and no `embodiment`, the model lookup (get_models()) failed, so
            nothing is checked or converted and nothing opens; call policy() again. A policy already open stays open.
        """
        import dream  # deferred: the opener is defined in the package __init__, which imports this module

        # Every policy also records every tick set_observation() is given, in the background, to the platform's live
        # service (the credentials' platform, `dream._recorder` says how): each tick's state and each kept tick's
        # frames, under one `rec_` robot id per client, stored for Dream Labs and never shown live. It is best effort
        # (nothing is sent while a get_action() is in flight, the oldest ticks are dropped when it falls behind, a
        # platform that does not take recordings ends it for the policy), never raises into the caller, logs at DEBUG
        # only, and stays off without websockets, an API key or an https platform (`_recording_stats` says why). The
        # internal `_record: False` key turns it off for one policy, for episodes that are not the account's (the
        # MuJoCo example's simulated ones). Not in the docstrings, which the API reference renders.
        merged = dict(config)
        gateway_keys = sorted(_GATEWAY_KEYS & set(merged))
        if gateway_keys:
            raise DreamConfigError(
                f"config keys {gateway_keys} are not taken: where the model is comes from your credentials"
            )
        obs_window = merged.pop("obs_window", _window.OBS_WINDOW)
        obs_fps = merged.pop("obs_fps", _window.OBS_FPS)
        records = merged.pop("_record", True)
        if isinstance(obs_window, bool) or not isinstance(obs_window, int) or obs_window < 1:
            raise DreamConfigError(f"config key 'obs_window' must be an int of frames, at least 1, got {obs_window!r}")
        if isinstance(obs_fps, bool) or not isinstance(obs_fps, (int, float)) or not 0 < obs_fps < math.inf:
            raise DreamConfigError(
                f"config key 'obs_fps' must be a positive, finite number of frames per second, got {obs_fps!r}"
            )
        if not isinstance(records, bool):
            raise DreamConfigError(f"config key '_record' must be True or False, got {records!r}")
        if "embodiment" not in merged and merged.get("strict", True):
            loaded = self._embodiment_config_for(merged.get("checkpoint"))
            if loaded is not None:
                merged["embodiment"] = loaded
        # Where the model is is the credentials' alone, never the config's: the gateway is read-only from code.
        merged.update(_credentials.gateway_parts(self._gateway_url))
        recording, recording_off = (
            self._new_recording(round(obs_fps)) if records else (None, "recording is off for this policy")
        )
        self._close_policy()
        self._policy = dream._open_policy(merged, api_key=self._api_key)
        self._window = _window.Window(obs_window, 1.0 / obs_fps)
        self._episode += 1
        self._recording_off = recording_off
        if recording is not None:
            self._recording = recording
            recording.start()

    def set_observation(
        self, frames: Mapping[str, str], observation: Mapping[str, Any], *, timestamp: float | None = None
    ) -> None:
        """Hands the client one camera tick: its frames go into the window when due, its observation replaces the last.

        Call it on every camera tick, at whatever rate the cameras run. The first tick of an episode is kept, then one
        every `1 / obs_fps` seconds of `timestamp`, the tick nearest each due time; each frame key keeps its last
        `obs_window` frames. A tick whose frames are not kept still replaces the observation, so get_action() always
        sends the newest state. It costs a few operations per frame key, never waits for a get_action() running on
        another thread, and never waits on the network. The client keeps a copy of the mapping as it is at the call.

        @param frames Per window key, one frame. The key is the observation key the checkpoint reads the window under,
            `video.<role>`: one per role of a fine-tuned model's embodiment `video`; `video.overhead`,
            `video.wrist_l` and `video.wrist_r` for `dreamlabs/dream-1-beta` on a YAM (the deprecated
            `image_window.<role>` are still accepted). The frame is one base64 JPEG `str`, resized and encoded by you;
            the client never decodes it. The same keys on every call of an episode.
        @param observation Every other key the checkpoint reads: one `state.<part>` per part of a fine-tuned model's
            embodiment `state`; `state.right_arm`, `state.right_gripper`, `state.left_arm` and `state.left_gripper`
            for `dreamlabs/dream-1-beta` on a YAM, whose hand poses `ik` adds (or, deprecated, the YAM station's
            `state.arm_r.*` and `state.arm_l.*`); and `prompt`; never `embodiment`, which the client sends from
            policy()'s `robot` (get_action() refuses it). It replaces the last call's whole, nothing merged. The
            client keeps a copy of the mapping, so changing yours later changes nothing sent; the values (a numpy
            array) are not copied. Values are what JSON carries (numbers, strings, lists, numpy arrays), never bytes.
        @param timestamp When the frames were captured, seconds on a monotonic clock; `time.monotonic()` at the call
            when None. Never earlier than the last call's within an episode.
        @throws DreamConfigError No policy is open: call policy({"checkpoint": ...}) first.
        @throws DreamObservationError `frames` is not a non-empty mapping of non-empty `str` keys to one non-empty
            `str` frame each (bytes: send base64), `observation` is not a mapping or carries a frame key, the frame
            keys are not those of the episode's earlier calls, or `timestamp` is not a finite number or is earlier
            than the last call's. The message names the key, and the window is unchanged.
        """
        window = self._window
        policy = self._policy
        if policy is None or window is None:
            raise _no_policy("set_observation()")
        timestamp_s = time.monotonic() if timestamp is None else timestamp
        is_kept = window.add(frames, observation, timestamp_s)
        recording = self._recording
        if recording is not None:
            recording.offer(frames, observation, timestamp_s, is_kept, self._episode, policy._session_id)

    def get_action(self, *, commanded: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Sends the window and the latest observation to the model and returns its action chunk.

        The observation sent is the last set_observation()'s, plus, under each frame key, the frames kept so far in
        the episode, oldest first: `obs_window` of them once the window is full, fewer at the start of an episode,
        never none. The first call of an episode opens a session on the model, and later calls continue it. The first
        call to a model that is not running yet is refused `DA-2000` `pool_warming` while the model starts: wait
        `retry_after_s` and call again. When the round trips stay slow (a median of 1000 ms or more over the last
        five), it issues one `dream.DreamSlowNetworkWarning`, a Python warning that is never raised, and returns the
        action as usual. One call at a time: run it on a thread of its own and keep calling set_observation() from
        your camera loop meanwhile.

        @param commanded Per chunk key (`q_r_chunk`, `q_l_chunk`), the joints your loop last commanded that arm before
            the observation, or None; never sent. With them each chunk takes over at the row after those joints,
            where the loop is, instead of the row nearest the measured joints, which a real arm trails by a few rows.
        @return The action dict. A fine-tuned model's actions are under `per_key`: one `action.<part>` per part of its
            config's `action`, as rows, one per step (48 for Dream-1). A YAM body's are `q_r_chunk` and `q_l_chunk`,
            `[H, joints + 1]` (the arm's joint targets, then the gripper), already blended into the previous chunk so
            the control loop meets no step where one chunk replaces the last. Any other key the model adds is passed
            through.
        @throws DreamConfigError No policy is open: call policy({"checkpoint": ...}) first. Also: the host is not a
            valid host name, or the API key holds a character an HTTP header cannot carry.
        @throws DreamObservationError Before anything is sent: no set_observation() since the episode started, the
            observation carries `embodiment` (set policy()'s `robot` instead), the observation built does not match the
            embodiment config (checked for a fine-tune of a custom body, or the `embodiment` given), or a value cannot
            be sent as JSON. The message names the key.
        @throws DreamActionError The model's reply is not a usable action: it does not match the embodiment config,
            or it carries no action dict or a chunk that is not rows of numbers. Nothing is returned.
        @throws DreamTimeout No answer within `timeout_ms` (15000 ms by default); calling again is safe.
        @throws DreamProtocolError The model could not be reached, or what answered is not the Dream API; check the
            URL and the network, then call again.
        @throws DreamAuthError The key was refused (`DA-1000`): missing, unknown, revoked, or the checkpoint is not
            available to run on this account. Never retryable.
        @throws DreamBadRequest The model refused the request as malformed (`DA-4000`): `reason` is
            `unknown_checkpoint` (use an id get_models() lists), `too_large` (send smaller frames) or `malformed`.
            Never retryable.
        @throws DreamSessionExpired The model replaced the session mid-episode (`DA-0001` with a reason other than
            `new`). The new session is already in use and the action is not returned: call again to continue on it,
            or reset_episode() to start the episode over.
        @throws DreamServerError Any other refusal, with `code`, `reason`, `retryable` and `retry_after_s`: `DA-2000`
            (`pool_warming`: the model is starting; `rate_limited`: the calls came too fast; `pool_full`,
            `queue_timeout`: it is at max capacity; `max_serves`: it is at max capacity and no more copies of it may
            start, so `retryable` is False, though a gateway that sets its `retry_after_s` may have a seat free after
            that wait), `DA-5000` (the model failed the call) or `DA-6000` (a fault on the Dream Labs
            side; `cannot_scale`: the model cannot be started, the message says why). When `retryable` is True,
            wait `retry_after_s` seconds (a moment when it is None) and call again.
        @throws DreamError The client is closed, or another get_action() is running on it.
        """
        window = self._window
        policy = self._policy
        if policy is None or window is None:
            raise _no_policy("get_action()")
        recording = self._recording
        if recording is None:
            return policy.get_action(window.build_observation(), commanded=commanded)
        # The request has the uplink to itself: the recording sends nothing until it returns.
        recording.hold()
        try:
            return policy.get_action(window.build_observation(), commanded=commanded)
        finally:
            recording.release()

    def reset_episode(self) -> None:
        """Ends the episode: ends the session on the model, then empties the window.

        The next set_observation() is the first tick of a new episode, and the next get_action() starts a new
        session. Call it between episodes, when the robot is reset.

        @throws DreamConfigError No policy is open: call policy({"checkpoint": ...}) first.
        @throws DreamError A get_action() is running on another thread, or the client is closed; the session and the
            window are both left as they were, so calling it again once the get_action() returns is safe.
        """
        window = self._window
        policy = self._policy
        if policy is None or window is None:
            raise _no_policy("reset_episode()")
        # The session first: when it is refused, the episode goes on whole, its window included.
        policy.reset_episode()
        window.clear()
        self._episode += 1

    def refill_due(self, executed: int, remaining: int) -> bool:
        """Whether your loop should call get_action() now, by the pacing of the open policy (`dream.pacing`).

        It measures every round trip and paces async (the request goes out so its reply lands as `execution_horizon`
        rows have run) or sync (only once the chunk is used up, when a call is slower than `sync_above_ms`).

        @param executed Rows of the current chunk your loop has run since the chunk took over.
        @param remaining Rows of it still to run.
        @return True when a get_action() should start, given none is running.
        @throws DreamConfigError No policy is open: call policy({"checkpoint": ...}) first.
        """
        policy = self._policy
        if policy is None:
            raise _no_policy("refill_due()")
        return policy.refill_due(executed, remaining)

    @property
    def execution_horizon(self) -> int:
        """Rows of each chunk your loop runs before the next one takes over, the open policy's `execution_horizon`.

        @throws DreamConfigError No policy is open.
        """
        return self._open_policy("execution_horizon").execution_horizon

    @property
    def action_hz(self) -> float:
        """Rows per second your loop runs, the open policy's `action_hz`.

        @throws DreamConfigError No policy is open.
        """
        return self._open_policy("action_hz").action_hz

    @property
    def mode(self) -> str:
        """How the open policy paces get_action() now: `async` or `sync` (`dream.pacing`).

        @throws DreamConfigError No policy is open.
        """
        return self._open_policy("mode").mode

    @property
    def last_timing(self) -> dict[str, Any] | None:
        """Where the last get_action()'s time went (round trip, bytes, the model's timing), or None before the first.

        None when no policy is open.
        """
        policy = self._policy
        return None if policy is None else policy.last_timing

    @property
    def checkpoint(self) -> str | None:
        """The model the open policy runs, the `checkpoint` policy() was given; None when no policy is open."""
        policy = self._policy
        return None if policy is None else policy._checkpoint

    @property
    def session_id(self) -> str | None:
        """The session the model answers the episode under, once get_action() opened it; else None."""
        policy = self._policy
        return None if policy is None else policy._session_id

    @property
    def _recording_stats(self) -> dict[str, Any]:
        """How recording is going, read from any thread: a new dict on every read (internal; see policy()'s comment).

        - `state`: `off` (no policy is open, or its recording could not start: `last_error` says why), `connecting`
          (no tick yet, or dialing, or waiting for the platform to accept the link), `linked`, `backoff` (waiting to
          dial again after the link failed) or `disabled` (the platform does not take recordings; off until the next
          policy()).
        - `robot_id`: the id the recordings are sent under, `rec_` and 32 hex digits; None when `off`.
        - `frames_sent`, `states_sent`: frames and tick states sent.
        - `frames_dropped`, `states_dropped`: frames and tick states that will never be sent: dropped when the queue
          was full or the link stayed down, at a stop, or because they could not be sent (a state JSON cannot carry,
          a frame that is not base64 or is over 1 MiB).
        - `bytes_sent`: bytes of messages sent, framing left out.
        - `queue_bytes`: bytes waiting to be sent, frames counted at their decoded size.
        - `last_error`: the last link failure, or why recording could not start, one line; or None.

        Every count is 0 when `off`.
        """
        recording = self._recording
        if recording is None:
            return _recorder.off_stats(self._recording_off if self._policy is not None else None)
        return recording.stats()

    def _open_policy(self, what: str) -> PolicyClient:
        """The open policy client, or the no-policy error naming what was asked.

        @param what The property or method asked for, by name.
        @return The policy client.
        @throws DreamConfigError No policy is open.
        """
        policy = self._policy
        if policy is None:
            raise _no_policy(what)
        return policy

    def _new_recording(self, fps: int) -> tuple[_recorder.Recorder | None, str | None]:
        """A recorder for the policy being opened, not started, or why there is none; never raises.

        Makes the client's robot id the first time a recorder is made.

        @param fps The policy's `obs_fps`, rounded; the hello says at least 1.
        @return The recorder and None; or None and one line saying why recording is off for this policy: the
            websockets package is not installed, there is no API key or it holds a character an HTTP header cannot
            carry, or the platform URL is not `https://<host>[:<port>]` (`http://` only to a loopback host).
        """
        import dream  # deferred: the package __init__ imports this module

        missing = _recorder.websockets_missing()
        if missing is not None:
            return None, missing
        if not self._api_key:
            return None, (
                f"recording needs an API key: set {_credentials.API_KEY_ENV}, or pass api_key to DreamClient"
            )
        try:
            headers = _credentials._authorization(self._api_key)
            url = _recorder.link_url(self._platform_url)
        except DreamConfigError as error:
            return None, str(error)
        if self._robot_id is None:
            self._robot_id = f"rec_{uuid.uuid4().hex}"
        agent_version = f"dream-client/{dream.__version__}"[: _recorder.HELLO_MAX_FIELD_CHARS]
        return _recorder.Recorder(url, headers, self._robot_id, max(1, fps), agent_version), None

    def _embodiment_config_for(self, checkpoint: Any) -> dict[str, Any] | None:
        """The embodiment config a fine-tune the account runs was trained on, looked up once per checkpoint.

        get_models() answers each model with the config of the custom body it was trained on, whichever account
        holds that body (a company runs a fine-tune whose body Dream Labs registered). A builtin body has none:
        None, and the checks stay off as before. Runs only with `strict` on, and then never runs a robot unchecked:
        a lookup that fails, a checkpoint the listing does not name, or a listed config this library cannot read
        raises, since a robot with `parts` would otherwise run unconverted (its gripper in the trained 0-1 grip, not
        its own units). Nothing is cached then, so the next policy() looks again.

        @param checkpoint The `checkpoint` of the policy config.
        @return The config, or None when there is none to check against.
        @throws DreamProtocolError get_models() failed (the platform unreachable or refusing, as during a deploy), so
            the check could not run: the message says to call policy() again; the cause is the lookup's error.
        @throws DreamConfigError The listing does not name the checkpoint (a typo, or a model this account may not
            run), the model's body was registered with parts the listing withholds (`Model.parts_withheld`), or its
            listed config is not one this library's parser reads (a key a later platform lists), so its calls could
            be neither checked nor converted.
        """
        if not isinstance(checkpoint, str) or not checkpoint:
            return None
        if checkpoint in self._embodiment_configs:
            return self._embodiment_configs[checkpoint]
        try:
            model = next((m for m in self.get_models() if m.id == checkpoint), None)
            found = None if model is None or model.embodiment_config is None else dict(model.embodiment_config)
        except DreamError as error:
            raise DreamProtocolError(
                f"could not look up checkpoint {checkpoint!r} on the platform (GET /v1/models), so its embodiment "
                "config, the check of its calls and the conversion of a robot's parts cannot run and nothing is "
                "sent; call policy() again once the platform answers, or pass the robot's embodiment config as "
                f"config['embodiment']: {error}"
            ) from error
        if model is None:
            raise DreamConfigError(
                f"checkpoint {checkpoint!r} is not among the models this account may run (get_models(), GET "
                "/v1/models), so its embodiment config could not be looked up and its calls could be neither checked "
                "nor converted, and nothing is sent; check the name against get_models(), or pass the robot's "
                "embodiment config as config['embodiment']"
            )
        if model.parts_withheld:
            raise DreamConfigError(
                f"checkpoint {checkpoint!r} was trained on robot {model.embodiment!r}, registered with parts (what "
                "each part of its state and action is), which the platform lists only to the account that registered "
                "it, on a current client; pass that robot's embodiment config, the dict add_embodiment() took, as "
                "config['embodiment'], or run the model from that account on the current client "
                "(pip install --upgrade dreamlabs)"
            )
        if found is not None:
            try:
                embodiments.parse(found)
            except (embodiments.EmbodimentInvalidError, TypeError) as error:
                raise DreamConfigError(
                    f"the embodiment config listed for {checkpoint!r} is not one this library reads, so its calls "
                    "could be neither checked nor converted; install the current dream client (pip install --upgrade "
                    "dreamlabs), or pass the robot's embodiment config as "
                    f"config['embodiment']: {error}"
                ) from error
        self._embodiment_configs[checkpoint] = found
        return found

    def close(self) -> None:
        """Empties the window and releases the open inference session, if any; calling it again does nothing.

        Leaving a `with` block calls it. The embodiment, dataset, fine-tuning and model calls hold nothing open.
        """
        self._close_policy()

    def _close_policy(self) -> None:
        """Stops the open policy's recording, empties its window and releases its session; harmless when none is open.

        Called by policy() and close().
        """
        recording, self._recording = self._recording, None
        self._recording_off = None
        if recording is not None:
            recording.stop()
        window, self._window = self._window, None
        if window is not None:
            window.clear()
        if self._policy is not None:
            self._policy.close()
            self._policy = None

    def __enter__(self) -> DreamClient:
        """Returns the client itself; nothing is opened here."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Closes the client; the exception, if any, propagates unchanged."""
        self.close()

    def __repr__(self) -> str:
        """Names the class only; never the key."""
        return "<dream.DreamClient>"


def _no_policy(call: str) -> DreamConfigError:
    """The error for a call made before policy().

    @param call The method or property called, by name.
    @return The error to raise.
    """
    return DreamConfigError(f"no policy is open: call policy({{'checkpoint': ...}}) before {call}")
