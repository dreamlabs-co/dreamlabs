"""Fine-tuning jobs on the Dream Labs data service: train Dream-1 on an uploaded dataset, then run the result.

`dream.DreamClient` carries the two calls most callers need with its key bound (finetune(), get_finetuning_status());
the functions here are the whole set on their own, for a caller that passes `api_key` itself. Each answers the
service's JSON as it came: a job is a dict, a listing a list of them.

- create(): queues a job on one of your `processed` datasets under a suffix; the fine-tuned model's checkpoint id will
  be `dream-1-ft:<uid8>-<suffix>`, in the job as `fine_tuned_model` once it is published. The training knob
  `learning_rate` is optional: a knob left out takes the service's default, and the service refuses one outside its
  range with `invalid_request` naming the knob. `use_eef` (default False) asks the job to train the end-effector pose
  beside the joints; the dataset's embodiment must then carry the six EEF parts
  (`shared.datasets.embodiments.EEF_PARTS`) in both `state` and `action`, which create() checks before sending and the
  service checks again. `lr_schedule` is `"constant"` (the default) or `"cosine"`; cosine decay works best for
  specialization (a fine-tune on a single task), and specialist tasks may not be highly reliable yet.
- list() and get(): your jobs, newest first, and one by its id.
- cancel(): a job runs to its end once queued, so the cancel is refused with `conflict`.
- queue(): the line as a job queued now would see it.
- training_data(): where one of your fine-tunes' training data is, for comparing camera views.

A job is a dict with these keys:

- `id`: `ftj_` + hex; what get() and cancel() take.
- `dataset`, `embodiment`: the dataset it trains on and that dataset's embodiment, each `<username>/<name>`.
- `base_model`: the id of the checkpoint it starts from.
- `suffix`: the suffix it was created with.
- `fine_tuned_model`: the new checkpoint's id, `dream-1-ft:<uid8>-<suffix>`, once the job has succeeded and the
  checkpoint is published; None before. It is what `DreamClient.policy()` takes as `checkpoint`.
- `status`: `queued` (waiting for a training slot, or for a GPU once it has one), `running`, `succeeded`, `failed` or
  `cancelled`.
- `hyperparameters`: the training settings, as the service filled them in.
- `queue_position`: 1 for the next job to start, while `queued` behind other jobs; None otherwise, a job waiting for
  a GPU included.
- `created_at`, `started_at`, `finished_at`: ISO 8601 times in UTC; `started_at` is when training started on a GPU
  (None before), `finished_at` is None until the job ends.
- `progress`: where the job is, `{"phase", "step", "total_steps", "loss", "updated_at"}`, on a deployment that reports
  it, else None: `phase` is `waiting_for_gpu`, `preparing` (the dataset and the base are set up), `training` (with
  the last logged `step` and the action `loss` over your own data) or `saving` (training is over and the model is
  checked and published).
- `error`: when `failed`, one line: what to fix in your dataset when the cause was in it, else that the training did
  not finish; None otherwise.

Poll get() until `fine_tuned_model` is set, not until `succeeded` alone: the checkpoint is published in the same
pass that marks the job `succeeded`, or a little later when that publication is retried. A `failed` or `cancelled`
job never gets one, so stop polling there. Then `DreamClient.policy({"checkpoint": job["fine_tuned_model"]})` runs
the model; its first get_action() is refused `pool_warming` (a retryable #DreamServerError with `retry_after_s`)
while the model starts, and a model that cannot be started is refused `cannot_scale` with the reason.

Errors: the URL and key are `dream.datasets`'s (either one left out is the credentials'), and so are the errors. A
refusal is a #DreamServerError whose `code` is the service's: `not_found` for a dataset or job that is not yours;
`conflict` for a suffix in use, a dataset that is not `processed`, a cancel where fine-tuning is priced or of a
job already over, `payment_required` for a job the account's credits do not cover, or an account at
its limit of jobs queued or running at once; `invalid_request` for a suffix or a knob out of range, naming it;
`validation` for a knob of the wrong type; an HTTP 5xx one is `retryable`: make the same call again after a short
wait. A key the service does not accept is #DreamAuthError (`forbidden`: the account has no username yet, ask Dream
Labs support to set one); a service that cannot be reached, or an answer that is not the Dream API's, is
#DreamProtocolError; no key (and the credentials name none) is #DreamConfigError.

@par Example
```python
import time

from dream import fine_tuning

job = fine_tuning.create("<your-username>/towels-v1", "towels", api_key=KEY)
while job["fine_tuned_model"] is None and job["status"] in ("queued", "running", "succeeded"):
    time.sleep(60)
    job = fine_tuning.get(job["id"], api_key=KEY)
```
"""

import builtins
import urllib.parse
from typing import Any

from shared.datasets import embodiments

from dream import datasets
from dream.errors import DreamConfigError, DreamProtocolError


def create(
    dataset: str,
    suffix: str,
    *,
    learning_rate: float | None = None,
    batch_size: int | None = None,
    use_eef: bool = False,
    lr_schedule: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Queues a fine-tuning job on one of your `processed` datasets.

    A knob left as None is not sent, and the service fills in its default; the job's `hyperparameters` says what it
    trains with. A knob out of range is refused with `invalid_request`, naming it. With `use_eef`, the dataset's
    embodiment is fetched and checked here first (check_eef()), so a body that cannot train the end-effector pose is
    refused before anything is queued; a dataset the service does not list is left for the service to refuse.

    @param dataset The dataset's `<username>/<name>` (a #dream.datasets.Dataset's `name`), one of your own and
        `processed`.
    @param suffix `[a-z0-9-]{1,40}`: the tail of the checkpoint id `dream-1-ft:<uid8>-<suffix>`; free among your
        models and active jobs.
    @param learning_rate The learning rate, 1e-6 to 1e-4; None takes the service's default of 1e-5.
    @param batch_size Leave it as None: the service sets it.
    @param use_eef Whether the job trains the end-effector pose beside the joints. The dataset's embodiment must
        then carry the six EEF parts, `right_eef_xyz` (3 wide), `right_eef_rot6` (6), `right_grip` (1) and the same
        three for `left`, in both `state` and `action`. False, the default, trains every part as joints.
        A body whose `action` carries `right_eef_rot6` or `left_eef_rot6` trains only with it; without it the
        service refuses the job with `invalid_request`. So does one whose `action` carries `right_eef_xyz` or
        `left_eef_xyz` without that side's `_eef_xyz` (3 wide) and `_eef_rot6` (6 wide) in `state`.
        The pose follows the base model's convention, per arm: `<side>_eef_xyz` is the tool centre point in metres
        (on YAM 0.1347 m past the wrist flange along the last link's z axis), `<side>_eef_rot6` the last link's
        orientation as the first two columns of its rotation matrix, `[R00, R10, R20, R01, R11, R21]`, both in that
        arm's own base frame, and `<side>_grip` the gripper opening, 0 closed to 1 open. A pose recorded another way
        passes every check and trains a mismatched model.
        A robot registered with `parts` (what each part is) decides `use_eef` itself, so leave it out: the service
        trains its gripper poses when its parts can. Asking it of one whose poses are recorded, not trained, is
        refused here.
    @param lr_schedule The learning-rate schedule, `"constant"` or `"cosine"`; None takes the service's default,
        `"constant"`. Both warm up first; `"cosine"` then decays the rate to 0 by the last step.
        Cosine decay works best for specialization (a fine-tune on a single task).
        You may not get high reliability on specialist tasks yet.
    @param api_key Your API key; the credentials' when None.
    @return The job dict (its keys are listed above), `status` `queued`. Keep `job["id"]` and poll get() until
        `fine_tuned_model` is set, or the job is `failed` or `cancelled`.
    @throws DreamConfigError There is no key (and the credentials name none), a knob is not a plain number, or
        `use_eef` is asked of a dataset whose embodiment lacks an EEF part or has one at another width (the message
        names the part and its side).
    @throws DreamAuthError The key was refused, or the account has no username yet (`forbidden`).
    @throws DreamServerError `not_found` for a dataset that is not yours; `conflict` for one that is not
        `processed`, a suffix in use, or an account at its limit of active jobs; `invalid_request` for a suffix or a
        knob out of range, naming it, or for a dataset whose embodiment's action carries an end-effector rotation, or
        a position without its state pose, without `use_eef`; a
        `retryable` one (HTTP 5xx) may be made again after a short wait.
    @throws DreamProtocolError The service could not be reached, or did not answer as the Dream API.
    """
    api_key, host = datasets._resolve(api_key)
    body: dict[str, Any] = {"dataset": dataset, "suffix": suffix}
    named = {"learning_rate": learning_rate, "batch_size": batch_size, "lr_schedule": lr_schedule}
    knobs: dict[str, Any] = {name: value for name, value in named.items() if value is not None}
    if use_eef:
        check_eef(dataset, api_key=api_key)
        knobs["use_eef"] = True
    if knobs:
        body["hyperparameters"] = knobs
    return _object(datasets._call("POST", host, "/v1/fine_tuning/jobs", api_key, body))


def check_eef(dataset: str, *, api_key: str | None = None) -> None:
    """Checks that a dataset's embodiment can train the end-effector pose, the way create() does before sending.

    The dataset is found among your datasets by its name and its embodiment fetched; the config is then held to the
    rule the service applies (`shared.datasets.embodiments.check_eef`): the six EEF parts in both `state` and
    `action` at their widths. A config with `parts` is held to it as training reads
    it (`embodiments.trained`), and passes only when its parts train the gripper poses. A dataset the service does
    not list passes here, and the service answers `not_found` on the create.

    @param dataset The dataset's `<username>/<name>`.
    @param api_key Your API key; the credentials' when None.
    @throws DreamConfigError The embodiment cannot train the end-effector pose; the message names the first missing
        or wrong part and its side, or the action width. Or there is no key.
    @throws DreamAuthError The key was refused, or the account has no username yet (`forbidden`).
    @throws DreamServerError The service refused a read; a `retryable` one (HTTP 5xx, already retried by the client)
        may be made again after a short wait.
    @throws DreamProtocolError The service could not be reached, or did not answer as the Dream API.
    """
    api_key, host = datasets._resolve(api_key)
    listed = datasets._list(datasets._call("GET", host, "/v1/datasets", api_key, retry=True))
    mine = [datasets._dataset(found) for found in listed]
    matching = [found for found in mine if found.name == dataset]
    if not matching:
        return
    body = datasets._fetch_embodiment(matching[0].embodiment, api_key, host)
    try:
        parsed = embodiments.parse(body.embodiment_config)
        if parsed.plan is not None and not parsed.plan.use_eef:
            raise embodiments.EmbodimentInvalidError(
                "its robot's parts leave the gripper poses recorded, not trained, so it cannot take use_eef; leave "
                "use_eef out and the parts decide it."
            )
        embodiments.check_eef(embodiments.trained(parsed))
    except embodiments.EmbodimentInvalidError as error:
        raise DreamConfigError(f"{dataset}: {error}") from None


def list(*, api_key: str | None = None) -> builtins.list[dict[str, Any]]:
    """Your account's fine-tuning jobs.

    @param api_key Your API key; the credentials' when None.
    @return A list of job dicts (the keys are listed above), newest first.
    @throws DreamConfigError There is no key (and the credentials name none).
    @throws DreamAuthError The key was refused, or the account has no username yet (`forbidden`).
    @throws DreamServerError The service refused the call; a `retryable` one (HTTP 5xx, already retried by the
        client) may be made again after a short wait.
    @throws DreamProtocolError The service could not be reached, or did not answer as the Dream API.
    """
    api_key, host = datasets._resolve(api_key)
    answer = datasets._call("GET", host, "/v1/fine_tuning/jobs", api_key, retry=True)
    return [_object(found) for found in datasets._list(answer)]


def get(job_id: str, *, api_key: str | None = None) -> dict[str, Any]:
    """One of your jobs by its id, as the service holds it now; poll it to follow a job.

    @param job_id The job's `id` (`ftj_` + hex), as create() answered it.
    @param api_key Your API key; the credentials' when None.
    @return The job dict (the keys are listed above). `status` is `queued`, `running`, `succeeded`, `failed` or
        `cancelled`. Poll until `fine_tuned_model` is set, then pass it to `DreamClient.policy()` as `checkpoint`;
        stop polling on `failed` (`error` says so) or `cancelled`, which never get one.
    @throws DreamConfigError There is no key (and the credentials name none).
    @throws DreamAuthError The key was refused, or the account has no username yet (`forbidden`).
    @throws DreamServerError `not_found` for a job that is not yours; a `retryable` one (HTTP 5xx, already retried
        by the client) may be made again after a short wait.
    @throws DreamProtocolError The service could not be reached, or did not answer as the Dream API.
    """
    api_key, host = datasets._resolve(api_key)
    return _object(datasets._call("GET", host, _job_path(job_id), api_key, retry=True))


def cancel(job_id: str, *, api_key: str | None = None) -> dict[str, Any]:
    """Asks to cancel one of your jobs; a job runs to its end once queued, so the service refuses with `conflict`.

    @param job_id The job's `id` (`ftj_` + hex), as create() answered it.
    @param api_key Your API key; the credentials' when None.
    @return The job dict as the service answered the cancel, `status` `cancelled`; it never gets a model.
    @throws DreamConfigError There is no key (and the credentials name none).
    @throws DreamAuthError The key was refused, or the account has no username yet (`forbidden`).
    @throws DreamServerError `not_found` for a job that is not yours, `conflict` for one already finished or where
        fine-tuning is priced; a `retryable` one (HTTP 5xx) may be made again after a short wait.
    @throws DreamProtocolError The service could not be reached, or did not answer as the Dream API.
    """
    api_key, host = datasets._resolve(api_key)
    return _object(datasets._call("POST", host, _job_path(job_id) + "/cancel", api_key))


def queue(*, api_key: str | None = None) -> dict[str, Any]:
    """The line as a job queued now would see it, counted over every account.

    @param api_key Your API key; the credentials' when None.
    @return `{"jobs_ahead_of_new", "running", "slots_free", "slots_total", "updated_at"}`: the jobs queued ahead of a
        new one, the jobs training now, the training slots free and in all (a job takes one slot while it runs),
        and when the counts were taken, an ISO 8601 time in UTC.
    @throws DreamConfigError There is no key (and the credentials name none).
    @throws DreamAuthError The key was refused, or the account has no username yet (`forbidden`).
    @throws DreamServerError The service refused the call; a `retryable` one (HTTP 5xx, already retried by the
        client) may be made again after a short wait.
    @throws DreamProtocolError The service could not be reached, or did not answer as the Dream API.
    """
    api_key, host = datasets._resolve(api_key)
    return _object(datasets._call("GET", host, "/v1/fine_tuning/queue", api_key, retry=True))


def training_data(model: str, *, api_key: str | None = None) -> dict[str, Any]:
    """Where to read the dataset one of your fine-tunes was trained on; `dreamcli setup` compares camera views with it.

    @param model The fine-tune's id (`fine_tuned_model` of its job).
    @param api_key Your API key; the credentials' when None.
    @return `{"url", "video"}`: a one-hour download URL of the dataset's zip, and camera role to its video key.
    @throws DreamServerError `not_found` when none of your jobs made the model, or its dataset was deleted.
    @throws DreamProtocolError The service could not be reached, or did not answer as the Dream API.
    """
    api_key, host = datasets._resolve(api_key)
    path = f"/v1/fine_tuning/models/{urllib.parse.quote(model, safe='')}/training_data"
    return _object(datasets._call("GET", host, path, api_key, retry=True))


def _job_path(job_id: str) -> str:
    return f"/v1/fine_tuning/jobs/{urllib.parse.quote(job_id, safe='')}"


def _object(answer: Any) -> dict[str, Any]:
    """The service's answer, which is a JSON object on every route here; anything else is the wrong thing
    listening."""
    if not isinstance(answer, dict):
        raise DreamProtocolError(f"the data service answered {type(answer).__name__}, not a JSON object")
    return answer
