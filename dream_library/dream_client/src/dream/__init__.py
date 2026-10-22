"""Dream Labs client: record a robot body, fine-tune Dream-1 on a dataset, and call the result for actions.

The public namespace is `dream.DreamClient`, the error types, the one warning type (`dream.DreamSlowNetworkWarning`,
when the round trips to the model stay slow) and `dream.__version__`, plus three submodules `DreamClient` is a front
over: `dream.datasets` (embodiments and datasets on the data service), `dream.fine_tuning` (fine-tuning jobs on the
data service) and `dream.models` (the checkpoints an account may run, on the platform). `DreamClient` is the one object
every caller uses: it carries the key, forwards to those modules, and runs the model: policy() chooses it,
set_observation() hands it every camera tick, and get_action() asks it for the next chunk. The two errors only the
dataset upload raises, `DreamDatasetError` and `DreamUploadError`, are imported from `dream.errors`.

`dream.LiveViewClient` (`dream.live`) shows the robot on the platform's Live View page: stream() its observations and
status, get_prompt() the task a viewer sent, and register a callback for the #EventChange a viewer makes.

`dream.__version__` is the version of the installed library, read from its package metadata when `dream` is imported.
A library installed from PyPI reports its release, such as `"0.0.6"`; one built by the Dream Labs platform also
carries the commit it was built from, as in `"0.0.6+g<commit>"` (PyPI takes no such suffix). Quote it in a bug
report.

@par Example
```python
import dream

client = dream.DreamClient()  # the key comes from ~/.dreamlabs/credentials
client.add_embodiment("metal-arm", config)
dataset = client.upload_dataset("~/data/towels", "towels-v1", "<your-username>/metal-arm")
# ... poll client.get_dataset_status(dataset.id) until its status is "processed" ...
job = client.finetune(dataset.name, "towels")
# ... poll client.get_finetuning_status(job["id"]) until job["fine_tuned_model"] is set ...
client.policy({"checkpoint": job["fine_tuned_model"]})
while running:
    client.set_observation(frames, observation)  # every camera tick
    if due:
        chunk = client.get_action()
```
"""

import importlib.metadata
import pathlib
from typing import Any

from dream import client as _client
from dream import client_ik as _client_ik
from dream import credentials as _credentials
from dream import logs as _logs
from dream import models as _models
from dream import smoothing as _smoothing
from dream.errors import (
    DreamActionError,
    DreamAuthError,
    DreamBadRequest,
    DreamConfigError,
    DreamError,
    DreamObservationError,
    DreamProtocolError,
    DreamServerError,
    DreamSessionExpired,
    DreamSlowNetworkWarning,
    DreamTimeout,
)
from dream.facade import DreamClient
from dream.live import EventChange, LiveViewClient

# Diagnostic logging is set up once, on import, so `import dream` is all it takes for the
# on-by-default file at ~/.dream/logs/ to exist; the environment controls it (see dream.logs).
# It never raises: a home that cannot be written falls back to silence.
_logs.configure()

## `__version__` when no installed distribution carries the package, as when it runs from a source checkout.
_SOURCE_VERSION = "0+unknown"

## The distribution that installs the package, tried by name when no file list beside the package names it (an
## editable install points into the source tree): the client, which the library (`dreamlabs`) depends on.
_DISTRIBUTIONS = ("dreamlabs-client",)


def _installed_version(site: pathlib.Path) -> str:
    """The version of the distribution that installed this package, from its metadata.

    That distribution is the one installed in `site` whose file list (`RECORD`) holds this package's `__init__.py`,
    whatever its name; without one, the first of `_DISTRIBUTIONS` installed; without that, `_SOURCE_VERSION`.

    @param site The directory the package was imported from.
    @return The version, such as `0.0.6` from PyPI, or `0.0.6+g<commit>` for a library the platform built.
    """
    # The same answer as importlib.metadata.packages_distributions(), which parses the file list of every installed
    # distribution: about 125 ms in a fresh install of the library, where all of `import dream` takes about 175 ms.
    # Searching the lists beside the package as text adds about 4 ms there.
    entry = f"{__name__}/__init__.py,"
    for distribution in importlib.metadata.distributions(path=[str(site)]):
        record = distribution.read_text("RECORD") or ""
        if record.startswith(entry) or f"\n{entry}" in record:
            return distribution.version
    for name in _DISTRIBUTIONS:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return _SOURCE_VERSION


## Package version, described in the package docstring: the installed distribution's, never written in the source.
__version__ = _installed_version(pathlib.Path(__file__).resolve().parent.parent)

__all__ = [
    "DreamClient",
    "EventChange",
    "LiveViewClient",
    "__version__",
    "DreamError",
    "DreamConfigError",
    "DreamBadRequest",
    "DreamObservationError",
    "DreamActionError",
    "DreamTimeout",
    "DreamProtocolError",
    "DreamServerError",
    "DreamAuthError",
    "DreamSessionExpired",
    "DreamSlowNetworkWarning",
]

## Config keys consumed locally and never sent; the remainder is `checkpoint`, `embodiment` and `robot`.
_CONNECTION_KEYS = _client._CONNECTION_KEYS

## Config keys that pace the caller's loop (`dream.pacing`); consumed locally and never sent.
_PACING_KEYS = _client._PACING_KEYS

## Largest `timeout_ms` accepted: a socket timeout is a C int of milliseconds on every platform.
_MAX_TIMEOUT_MS = 2**31 - 1

## Documented defaults for the connection keys the credentials do not carry; `host`, `port` and
## `scheme` come from the credentials' gateway URL (`dream.credentials`).
_CONNECTION_DEFAULTS: dict[str, Any] = {"timeout_ms": 15000, "strict": True}


def _open_policy(config: dict[str, Any] | None = None, api_key: str | None = None) -> _client.PolicyClient:
    """Opens the inference client `DreamClient.policy()` runs on, for one checkpoint; the session opens on first use.

    Internal: callers use `DreamClient.policy()`, which adds the frame window and the recording, and whose
    get_action() builds each observation from set_observation(). No request reaches the model here. With `ik`, the
    platform is asked once for the checkpoint's record (`dream.models`), because what the model was trained on decides
    how the arm is sent. The config keys:

    - `checkpoint` (str, required): the model to run, a `Model.id` from `DreamClient.get_models()` or a fine-tuning
      job's `fine_tuned_model`.
    - `embodiment` (dict, optional): what strict mode checks every observation and reply against, the embodiment
      config the body was registered with (the dict `add_embodiment()` took) or a shape spec
      `{group: {key: {"shape": [...], "dtype": ...}}}`. Without it nothing is checked.
    - `strict` (bool, default True): check observations and replies against `embodiment`.
    - `timeout_ms` (int, milliseconds, 1 to 2147483647, default 15000): how long one call waits for the model.
    - `execution_horizon` (int, rows, default 30), `action_hz` (rows per second, default 20) and `sync_above_ms`
      (milliseconds, default 700): how the client paces your control loop (`dream.pacing`). Calls slower than
      `sync_above_ms` (the median of the last five) make the pacing sync (run the whole chunk, then ask again);
      otherwise it is async (ask while the chunk runs, so the reply lands as `execution_horizon` rows have run).
      `action_hz` is your loop's rate.
      Read by refill_due(); never sent.
    - `kinematics` (dict, optional): per chunk key (`q_r_chunk`, `q_l_chunk`), an object with `pose(joints)`, the
      end effector's `(position [3], rotation [3, 3])`, and `jacobian(joints)`, its `[6, joints]` Jacobian (linear
      rows, then angular, in the frame `pose` reports). Each chunk takes over from the motion the arm is on through a
      velocity blend (`dream.seam_join`); an arm with kinematics has it done on the end effector's pose, the path the
      gripper actually traces, and every other arm in joint space. Called from the thread that runs get_action();
      never sent.
    - `ik` (dict, optional): for a model fine-tuned with `use_eef`, `{"arm": <robot>, "head": "eef" or "joint"}`,
      the robot being `"yam"` (a YAM, for a Dream-1 model that reads each hand's pose beside the joints, such
      as `dreamlabs/dream-1-beta`: the client adds the poses and `state.<side>_grip` from
      `state.<side>_gripper`, sends the arm's own joints, and head `joint` runs the model's joint answer, this arm's
      own), `"metal"` (MakerMods Metal), `"makers"` (Makers Arm) or a FaaS beta robot (`dream.client_ik` `ARMS`,
      e.g. `"so101_faas"`). The client adds each arm's tool pose to every observation (`state.<side>_eef_xyz`
      and `state.<side>_eef_rot6`, from forward kinematics of `state.<side>_arm` in radians; do not send them
      yourself), and with head `eef` turns the model's tool poses into `q_r_chunk` and `q_l_chunk` (the arm's joints
      in radians, then the grip from 0 closed to 1 open); head `joint` keeps a Metal or Makers model's joint chunks.
      A beta robot's model reads its joints and gripper in the robot's recorded units: send `state.<side>_grip` too,
      and the client converts both ways and sends the gripper drive under the key `embodiment` names for it (so it
      needs `embodiment`); with head `joint` its joint answer comes back in radians. What the model reads follows from
      the body it was trained on, the `embodiment` of its record in `DreamClient.get_models()`: a model trained on
      YAM arms (the Dream-1 base models, `yam`) is sent zeros in its joint slots (another arm's joints are out of its
      range), each hand's tool pose, `state.<side>_gripper` from 0 closed to 1 open (send `state.<side>_grip`) and
      the body's cameras as `overhead`, `wrist_l` and `wrist_r` (the one scene camera as `overhead`; so it needs
      `embodiment`), with `robot` `"other_embodiment"` (below), so it reads none of that state, the hand pose and the
      grip included: the hand pose sent is only the anchor its answer is composed on. It refuses head `joint`, whose
      answer would be another robot's joints, for safety. The kinematics are the same that computed the poses the
      model trained on, so every pose is in the training frame. Needs the library from the platform on Python 3.11
      or newer; opening then takes a few seconds, compiling the solver once. Never sent.
    - `robot` (str, optional): the robot the model runs on, as a model trained on YAM arms reads it: `"yam"` (a YAM,
      the arm it was trained on) or `"other_embodiment"` (any other arm), sent on every request as the observation's
      `embodiment` (never put it in your observation yourself). On `"other_embodiment"` the model is given none of
      the arm's state, neither its joints, its hand pose nor its grip (it reads all zeros, as it sometimes did in
      training), and answers no joints, only each hand's pose, composed on the hand pose the request carries;
      `per_key`'s `action.right_arm` and `action.left_arm` come back as None. Left out, it is `"other_embodiment"`
      when `ik` runs a model trained on YAM arms on an arm that is not a YAM (the client then sends that model zeros
      for the joints), and nothing is sent otherwise. Given, it is sent as given: `"other_embodiment"` with IK of
      your own (no `ik`) asks for the same, and refuses `ik` head `joint`.
    - `host` (str), `port` (int, 1 to 65535) and `scheme` (`"http"` or `"https"`): where the model is; by default
      those of `DREAM_GATEWAY_URL` in the credentials (`dream.credentials`: the environment, then
      `~/.dreamlabs/credentials`, then production, which is `https` on port 443).

    Any other key is an error. The client copies the config, so yours is not changed.

    @param config The settings above, as a dict.
    @param api_key Your API key, sent with every request; the credentials' (`DREAM_API_KEY`) when None. Never part
        of `config`.
    @return The inference client (`PolicyClient`): get_action(observation) per request, reset_episode() between
        episodes, close() to release the session.
    @throws DreamConfigError A config key has the wrong type or is out of range, `checkpoint` is missing, the config
        carries a key the client does not take, `robot` is not `"yam"` or `"other_embodiment"`, `strict` is on and
        `embodiment` is not a valid embodiment config or shape spec, `ik` names an arm or a head the client does not
        take, head `joint` for a model trained on YAM arms or with `robot` `"other_embodiment"`, an arm or cameras
        that cannot be sent the way such a model reads them (`dream.client_ik`), or the
        install has no IK, the checkpoint is not one the key may run (with `ik`), or the credentials cannot be read
        or name a `DREAM_GATEWAY_URL` that is not `http(s)://<host>[:<port>]`.
    @throws DreamAuthError, DreamServerError, DreamProtocolError With `ik`: the platform refused the key, failed
        the call or could not be reached (`dream.models.list_models`).
    """
    # Shallow copy: the caller keeps their dict, and the remainder we store is never mutated.
    remainder = dict(config or {})
    conn = {key: remainder.pop(key) for key in _CONNECTION_KEYS if key in remainder}
    _validate_connection(conn)
    pace = {key: remainder.pop(key) for key in _PACING_KEYS if key in remainder}
    _validate_pacing(pace)
    kinematics = remainder.pop("kinematics", None)
    _validate_kinematics(kinematics)
    ik = remainder.pop("ik", None)
    credentials = _credentials.load()
    settings = {**_CONNECTION_DEFAULTS, **_credentials.gateway_parts(credentials.gateway_url), **conn}
    key = credentials.api_key if api_key is None else api_key
    checkpoint = remainder.get("checkpoint")
    model = None
    if ik is not None and isinstance(checkpoint, str) and checkpoint:
        _client_ik.check_config(ik)
        model = _models._record(checkpoint, api_key=key)
    return _client.PolicyClient(
        host=settings["host"],
        port=settings["port"],
        timeout_ms=settings["timeout_ms"],
        strict=settings["strict"],
        scheme=settings["scheme"],
        api_token=key,
        session_config=remainder,
        kinematics=kinematics,
        ik=ik,
        model=model,
        **pace,
    )


def __dir__() -> list[str]:
    """Lists only `__all__`, so tab completion never suggests a submodule or an internal."""
    return sorted(__all__)


def _validate_connection(conn: dict[str, Any]) -> None:
    """Checks the types and ranges of the connection keys present, before any socket exists.

    Only the connection keys are checked here; `PolicyClient` rejects any key that is
    neither a connection key nor `checkpoint` / `embodiment` / `robot`, naming it, because the
    request has no field that could carry it.

    @param conn The connection keys the user supplied.
    @throws DreamConfigError A key has the wrong type or is out of range; the message names it.
    """
    if "host" in conn and not (isinstance(conn["host"], str) and conn["host"]):
        raise DreamConfigError(f"config key 'host' must be a non-empty str, got {conn['host']!r}")
    if "port" in conn and not (_is_int(conn["port"]) and 1 <= conn["port"] <= 65535):
        raise DreamConfigError(f"config key 'port' must be an int in 1..65535, got {conn['port']!r}")
    if "timeout_ms" in conn and not (_is_int(conn["timeout_ms"]) and 1 <= conn["timeout_ms"] <= _MAX_TIMEOUT_MS):
        raise DreamConfigError(
            f"config key 'timeout_ms' must be an int of milliseconds in 1..{_MAX_TIMEOUT_MS}, "
            f"got {conn['timeout_ms']!r}"
        )
    if "strict" in conn and not isinstance(conn["strict"], bool):
        raise DreamConfigError(f"config key 'strict' must be a bool, got {conn['strict']!r}")
    if "scheme" in conn and conn["scheme"] not in ("http", "https"):
        raise DreamConfigError(f"config key 'scheme' must be 'http' or 'https', got {conn['scheme']!r}")


def _validate_pacing(pace: dict[str, Any]) -> None:
    """Checks the types and ranges of the pacing keys present.

    @param pace The pacing keys the user supplied.
    @throws DreamConfigError A key has the wrong type or is out of range; the message names it.
    """
    if "execution_horizon" in pace and not (_is_int(pace["execution_horizon"]) and pace["execution_horizon"] >= 1):
        raise DreamConfigError(
            f"config key 'execution_horizon' must be an int of rows, at least 1, got {pace['execution_horizon']!r}"
        )
    for key in ("action_hz", "sync_above_ms"):
        value = pace.get(key)
        if key in pace and not (isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0):
            raise DreamConfigError(f"config key '{key}' must be a positive number, got {value!r}")


def _validate_kinematics(kinematics: Any) -> None:
    """Checks the `kinematics` key: a dict from chunk keys to objects with `pose` and `jacobian` methods.

    @param kinematics The value supplied, or None when the key is absent.
    @throws DreamConfigError The value is not a dict, names a key other than an arm's chunk key, or holds an object
        without callable `pose` and `jacobian`; the message names it.
    """
    if kinematics is None:
        return
    if not isinstance(kinematics, dict):
        raise DreamConfigError(f"config key 'kinematics' must be a dict of chunk key to object, got {kinematics!r}")
    for key, arm in kinematics.items():
        if key not in _smoothing.ARM_POSE_KEYS:
            raise DreamConfigError(
                f"config key 'kinematics' names {key!r}; it takes the chunk keys {sorted(_smoothing.ARM_POSE_KEYS)}"
            )
        if not (callable(getattr(arm, "pose", None)) and callable(getattr(arm, "jacobian", None))):
            raise DreamConfigError(
                f"config key 'kinematics' entry {key!r} must have callable pose() and jacobian(), got {arm!r}"
            )


def _is_int(value: Any) -> bool:
    """True for an int that is not a bool (`True` is an int in Python, but not a port)."""
    return isinstance(value, int) and not isinstance(value, bool)
