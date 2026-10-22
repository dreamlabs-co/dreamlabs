"""Models on the Dream Labs platform: the checkpoints your account may run.

`dream.DreamClient.get_models()` is the same call with its key bound; `from dream import models` reaches it on its
own, for a caller that passes `api_key` itself.

- list_models(): the checkpoints your account may run, the public ones first and then your own fine-tunes. A
  model's `id` is the `checkpoint` that `DreamClient.policy()` takes. `servable` means it can be
  started, not that it is running: the first get_action() to a model that is not running is refused `pool_warming`
  while it starts.

The key and the errors are those of `dream.datasets`.

@par Example
```python
from dream import models

for model in models.list_models(api_key=KEY):
    if model.servable:
        print(model.id, model.embodiment)
```
"""

import dataclasses
import datetime
from typing import Any

from dream import credentials, datasets
from dream.errors import DreamConfigError, DreamProtocolError


@dataclasses.dataclass(frozen=True)
class Model:
    """One checkpoint your account may run; pass its `id` to `DreamClient.policy()` as `checkpoint`.

    Attributes:
    - id: The checkpoint id, what a run names as `checkpoint`: `dreamlabs/dream-1-beta` for a public model,
      `dream-1-ft:<uid8>-<suffix>` for a fine-tune (a fine-tuning job's `fine_tuned_model`).
    - owner: `public` for a checkpoint every account may run, `mine` for one of your fine-tunes.
    - base_model: The model it was trained from; the same as `id` for a base model.
    - embodiment: For a fine-tune, the id of the embodiment it was trained on (`emb_...`; one of
      `DreamClient.get_embodiments()` only when your account holds that body); for a public model, the body it drives
      (`yam`); None while the model is not available to run yet.
    - servable: Whether the model can be started for get_action(); False while it is not available to run yet. A
      servable model may still be stopped, so its first call is refused `pool_warming` while it starts.
    - created_at: When your fine-tune was registered, UTC; None for a Dream Labs model, whose date the platform does
      not list.
    - cameras: The camera roles the model reads, each sent as `video.<role>`: `overhead`, `wrist_l` and `wrist_r`
      for a model trained on YAM arms, the roles of the embodiment a fine-tune was trained on; None when the platform
      does not know them. With `ik` the client sends the body's cameras in these roles (`dream.client_ik`).
    - embodiment_config: For a fine-tune of a custom body, the embodiment config it was trained on (what
      `DreamClient.policy()` checks calls against); None for a builtin body.
    - parts_withheld: True when that body was registered with parts (what each part of its state and action is) that
      `embodiment_config` leaves out: they are listed only to the account that registered the body, so
      `DreamClient.policy()` needs the body's config as `config['embodiment']` to run the model.
    """

    id: str
    owner: str
    base_model: str
    embodiment: str | None
    servable: bool
    created_at: datetime.datetime | None
    # Last and None by default, so a Model built by position before the field existed still builds.
    cameras: list[str] | None = None
    embodiment_config: dict[str, Any] | None = None
    parts_withheld: bool = False


def list_models(*, api_key: str | None = None) -> list[Model]:
    """The checkpoints your account may run: the public ones and your own fine-tunes.

    @param api_key Your API key; the credentials' when None.
    @return A list of #Model, the public ones first, then your own, each group by id. A model's `id` is the
        `checkpoint` for `DreamClient.policy()`.
    @throws DreamConfigError There is no key (and the credentials name none).
    @throws DreamAuthError The key was refused.
    @throws DreamServerError The platform refused the call; a `retryable` one (HTTP 5xx, already retried by the
        client) may be made again after a short wait.
    @throws DreamProtocolError The platform could not be reached, or did not answer as the Dream API.
    """
    api_key, host = _resolve(api_key)
    return [_model(found) for found in datasets._list(datasets._call("GET", host, "/v1/models", api_key, retry=True))]


def _record(checkpoint: str, *, api_key: str | None) -> Model:
    """One checkpoint's record, for a client that must know what the model was trained on (`dream.client_ik`).

    @param checkpoint The checkpoint id a run names.
    @param api_key Your API key; the credentials' when None.
    @return The #Model whose `id` is `checkpoint`.
    @throws DreamConfigError The account may not run `checkpoint` (the platform does not list it), or it has no
        embodiment yet (not available to run).
    @throws DreamAuthError, DreamServerError, DreamProtocolError As list_models().
    """
    listed = list_models(api_key=api_key)
    found = next((model for model in listed if model.id == checkpoint), None)
    if found is None:
        raise DreamConfigError(
            f"checkpoint {checkpoint!r} is not a model your key may run; the Models page and "
            f"DreamClient().get_models() list those: {[model.id for model in listed]}"
        )
    if found.embodiment is None:
        raise DreamConfigError(f"checkpoint {checkpoint!r} is not available to run yet: it has no embodiment")
    return found


def _resolve(api_key: str | None) -> tuple[str, str]:
    """The key a call uses, the argument's or else the credentials', and the platform's URL, always the
    credentials'."""
    loaded = credentials.load()
    api_key = loaded.api_key if api_key is None else api_key
    host = loaded.platform_url
    if api_key is None:
        raise DreamConfigError(
            f"no api key: pass api_key, or set {credentials.API_KEY_ENV} in the environment or in "
            f"{credentials.file_path()}"
        )
    return api_key, host


def _model(found: Any) -> Model:
    """A model as the platform answers it.

    A platform from before models carried their body config lists none (`embodiment_config` absent): the model reads
    None there, so `policy()` runs without the body's checks, as it did against that platform before. One from before
    `parts_withheld` lists no parts to anyone: False. A model listed without `created_at` (a Dream Labs model, whose
    date the platform does not list) reads None there.

    @throws DreamProtocolError The answer is not a model, or it is a model without `cameras`: a platform older than
        this library, which reads each model's camera roles.
    """
    if isinstance(found, dict) and "id" in found and "cameras" not in found:
        raise DreamProtocolError(
            f"the Dream platform answered model {found['id']!r} without its camera roles ('cameras'): the platform "
            "is older than this library; install the library it serves (paste the install block from its Setup API "
            "page again)"
        )
    try:
        return Model(
            id=found["id"],
            owner=found["owner"],
            base_model=found["base_model"],
            embodiment=found["embodiment"],
            servable=found["servable"],
            created_at=None if found.get("created_at") is None else datasets._timestamp(found["created_at"]),
            cameras=None if found["cameras"] is None else [str(role) for role in found["cameras"]],
            embodiment_config=found["embodiment_config"] if "embodiment_config" in found else None,
            parts_withheld="parts_withheld" in found and found["parts_withheld"] is True,
        )
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise datasets._unreadable("a model", error) from None
