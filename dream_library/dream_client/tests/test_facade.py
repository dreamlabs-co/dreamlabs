"""Tests of `dream.DreamClient`: the key and the hosts it binds, and that each method forwards to its module.

DreamClient is a thin front, so these pin two things: the construction (the key defaults from the credentials and the
constructor may override it; the hosts are the credentials' alone) and the delegation (every control-plane method
calls the matching module function with this client's key, and the inference pair opens and forwards to a policy
client on the credentials' gateway). The
module functions themselves are tested against fakes in `test_datasets.py`, `test_fine_tuning.py` and `test_models.py`;
here they are replaced by recorders so the facade's wiring is what is asserted.
"""

from __future__ import annotations

import datetime
from typing import Any

import pytest

import dream
from dream import datasets, fine_tuning, models

## The checkpoints these tests open a policy on, each listed to the account on a builtin body.
LISTED: tuple[str, ...] = ("dream-1-ft:x-nb1", "c")


@pytest.fixture(autouse=True)
def no_lookup_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """policy() looks up a fine-tune's embodiment config; here the account's models are #LISTED, each on a builtin
    body, and it has no embodiments, so the lookup finds no config and sends nothing. A test that exercises the lookup
    replaces these."""
    made = datetime.datetime(2026, 10, 1, tzinfo=datetime.timezone.utc)
    listed = [models.Model(slug, "mine", "dreamlabs/dream-1-beta", "yam", True, made) for slug in LISTED]
    monkeypatch.setattr(models, "list_models", lambda **kwargs: listed)
    monkeypatch.setattr(datasets, "list_embodiments", lambda **kwargs: [])


@pytest.fixture(autouse=True)
def isolated_credentials(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """No credentials file and no DREAM_* in the environment, so a default is the production default, not a dev box."""
    monkeypatch.setenv("DREAM_CREDENTIALS", str(tmp_path / "no-credentials"))
    for name in ("DREAM_API_KEY", "DREAM_GATEWAY_URL", "DREAM_API_URL", "DREAM_DATA_URL"):
        monkeypatch.delenv(name, raising=False)


def test_the_key_and_hosts_default_from_the_credentials(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """With nothing passed, the key and the hosts are the credentials'."""
    path = tmp_path / "credentials"
    path.write_text(
        "DREAM_GATEWAY_URL=https://gw.example\n"
        "DREAM_API_URL=https://plat.example\n"
        "DREAM_DATA_URL=https://data.example\n"
        "DREAM_API_KEY=dl_live_x\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DREAM_CREDENTIALS", str(path))
    client = dream.DreamClient()
    assert (client._api_key, client._platform_url, client._gateway_url) == (
        "dl_live_x",
        "https://plat.example",
        "https://gw.example",
    )


def test_nothing_set_anywhere_is_the_production_hosts_and_no_key() -> None:
    """The isolated environment leaves the production hosts and no key."""
    client = dream.DreamClient()
    assert (client._api_key, client._platform_url, client._gateway_url) == (
        None,
        "https://platform.dreamlabs.ai",
        "https://api.dreamlabs.ai",
    )


def test_the_constructor_overrides_the_key_and_takes_no_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """A key passed to the constructor wins over the credentials'; where the services are cannot be passed at all,
    and the repr names neither the key nor a host."""
    monkeypatch.setenv("DREAM_API_KEY", "dl_live_x")
    client = dream.DreamClient("dl_live_y")
    assert client._api_key == "dl_live_y"
    assert repr(client) == "<dream.DreamClient>"
    for name in ("data_url", "platform_url", "gateway_url"):
        with pytest.raises(TypeError, match=name):
            dream.DreamClient("dl_live_y", **{name: "https://elsewhere.example"})


def test_every_control_plane_method_forwards_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each control-plane method calls its module function with this client's key and nothing saying where."""
    calls: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}

    def recorder(name: str, result: Any) -> Any:
        def fake(*args: Any, **kwargs: Any) -> Any:
            calls[name] = (args, kwargs)
            return result

        return fake

    monkeypatch.setattr(datasets, "add_embodiment", recorder("add_embodiment", "EMB"))
    monkeypatch.setattr(datasets, "list_embodiments", recorder("list_embodiments", ["EMB"]))
    monkeypatch.setattr(datasets, "validate_dataset", recorder("validate_dataset", "REPORT"))
    monkeypatch.setattr(datasets, "upload_dataset", recorder("upload_dataset", "DATASET"))
    monkeypatch.setattr(datasets, "get_dataset", recorder("get_dataset", "STATUS"))
    monkeypatch.setattr(datasets, "list_datasets", recorder("list_datasets", ["DATASET"]))
    monkeypatch.setattr(datasets, "retry_dataset", recorder("retry_dataset", "RETRIED"))
    monkeypatch.setattr(datasets, "delete_dataset", recorder("delete_dataset", "DELETED"))
    monkeypatch.setattr(fine_tuning, "create", recorder("create", "JOB"))
    monkeypatch.setattr(fine_tuning, "get", recorder("get", "JOB_STATUS"))
    monkeypatch.setattr(models, "list_models", recorder("list_models", ["MODEL"]))

    client = dream.DreamClient("dl_live_x")
    data = {"api_key": "dl_live_x"}

    assert client.add_embodiment("arm", {"c": 1}) == "EMB"
    assert calls["add_embodiment"] == (("arm", {"c": 1}), data)

    assert client.get_embodiments() == ["EMB"]
    assert calls["list_embodiments"] == ((), data)

    assert client.validate_dataset("~/d", {"c": 1}) == "REPORT"
    assert calls["validate_dataset"] == (("~/d", {"c": 1}), {})  # local only: no key, no host

    assert client.upload_dataset("~/d", "towels-v1", "alice/arm") == "DATASET"
    assert calls["upload_dataset"] == (("~/d", "towels-v1", "alice/arm"), data)

    assert client.get_dataset_status("ds_1") == "STATUS"
    assert calls["get_dataset"] == (("ds_1",), data)

    assert client.get_datasets() == ["DATASET"]
    assert calls["list_datasets"] == ((), data)

    assert client.retry_dataset("ds_1") == "RETRIED"
    assert calls["retry_dataset"] == (("ds_1",), data)
    assert client.delete_dataset("ds_1") == "DELETED"
    assert calls["delete_dataset"] == (("ds_1",), {"confirm": True, **data})

    assert client.finetune("alice/towels-v1", "nb1", batch_size=16, lr_schedule="cosine") == "JOB"
    assert calls["create"] == (
        ("alice/towels-v1", "nb1"),
        {"learning_rate": None, "batch_size": 16, "use_eef": False, "lr_schedule": "cosine", **data},
    )

    assert client.get_finetuning_status("ftj_1") == "JOB_STATUS"
    assert calls["get"] == (("ftj_1",), data)

    assert client.get_models() == ["MODEL"]
    assert calls["list_models"] == ((), data)


def test_get_action_needs_a_policy_first() -> None:
    """get_action before policy() is a clear config error, not a crash."""
    client = dream.DreamClient("dl_live_x")
    with pytest.raises(dream.DreamConfigError, match="no policy is open"):
        client.get_action()


def test_policy_opens_over_the_gateway_and_get_action_forwards(monkeypatch: pytest.MonkeyPatch) -> None:
    """policy() opens a client on the credentials' gateway and this DreamClient's key and answers nothing;
    get_action() sends it the window and the latest observation set_observation() was given; close releases it."""
    opened: dict[str, Any] = {}

    class FakePolicy:
        def __init__(self) -> None:
            self.observation: Any = None
            self.commanded: Any = None
            self.closed = False
            self._session_id = None

        def get_action(self, observation: dict[str, Any], *, commanded: Any = None) -> dict[str, Any]:
            self.observation = observation
            self.commanded = commanded
            return {"q_r_chunk": [[0.0] * 7]}

        def close(self) -> None:
            self.closed = True

    fake = FakePolicy()

    def fake_init(config: dict[str, Any], api_key: str | None = None) -> FakePolicy:
        opened["config"] = config
        opened["api_key"] = api_key
        return fake

    monkeypatch.setattr(dream, "_open_policy", fake_init)
    monkeypatch.setenv("DREAM_GATEWAY_URL", "https://gw.example:8443")

    client = dream.DreamClient("dl_live_x")
    assert client.policy({"checkpoint": "dream-1-ft:x-nb1"}) is None  # type: ignore[func-returns-value]
    assert opened["api_key"] == "dl_live_x"
    assert opened["config"]["checkpoint"] == "dream-1-ft:x-nb1"
    # The gateway URL is threaded in as the connection keys the policy client consumes.
    assert (opened["config"]["host"], opened["config"]["port"], opened["config"]["scheme"]) == (
        "gw.example",
        8443,
        "https",
    )

    client.set_observation({"video.top": "frame1"}, {"o": 1}, timestamp=10.0)
    assert client.get_action() == {"q_r_chunk": [[0.0] * 7]}
    assert fake.observation == {"o": 1, "video.top": ["frame1"]}
    client.set_observation({"video.top": "frame2"}, {"o": 2}, timestamp=10.05)
    client.get_action(commanded={"q_r_chunk": [0.0] * 6})
    assert fake.observation == {"o": 2, "video.top": ["frame1"]}, "the newest state, the frames kept at 5 fps"
    assert fake.commanded == {"q_r_chunk": [0.0] * 6}

    client.close()
    assert fake.closed is True


def test_a_config_saying_where_the_model_is_is_refused_and_opens_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """`host`, `port` and `scheme` in the policy config are a config error naming them: the gateway is the
    credentials' alone, so a caller cannot redirect a call."""
    opened: list[dict[str, Any]] = []
    monkeypatch.setattr(dream, "_open_policy", lambda config, api_key=None: opened.append(config) or _Noop())
    client = dream.DreamClient("dl_live_x")
    with pytest.raises(dream.DreamConfigError, match=r"config keys \['host', 'port'\] are not taken"):
        client.policy({"checkpoint": "c", "host": "other.example", "port": 9000})
    with pytest.raises(dream.DreamConfigError, match=r"\['scheme'\]"):
        client.policy({"checkpoint": "c", "scheme": "http"})
    assert opened == []


def test_it_is_a_context_manager_that_closes_the_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Leaving a `with` block closes any open inference session."""
    fake = _Noop()
    monkeypatch.setattr(dream, "_open_policy", lambda config, api_key=None: fake)
    with dream.DreamClient("dl_live_x") as client:
        client.policy({"checkpoint": "c"})
    assert fake.closed is True


class _Noop:
    """A stand-in policy client: records close(), answers nothing else."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True
