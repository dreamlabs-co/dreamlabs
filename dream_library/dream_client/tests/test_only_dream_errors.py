"""Every error a public call raises is a DreamError: no input or answer escapes as a builtin.

`dream.errors` promises that `except dream.DreamError` catches everything the client raises. Each test here is one
input or answer that would otherwise escape as a plain builtin (a `TypeError` from the JSON encoder, a `ValueError`
from a URL parser, a `KeyError` from an answer of the wrong shape), and pins the Dream error raised instead. The
errors keep the builtins their classes derive from, so `DreamObservationError` and `DreamConfigError` are still
`ValueError`s; a caller catching `TypeError` for bytes in an observation catches `DreamObservationError` instead.
"""

from __future__ import annotations

import http.server
import json
import os
import pathlib
import re
import threading
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from conftest import CHECKPOINT, FakeDreamAPI, envelope

import dream
from dream import credentials, datasets, fine_tuning, models
from dream.errors import (
    DreamActionError,
    DreamConfigError,
    DreamObservationError,
    DreamProtocolError,
)

## A key with a line break, which HTTP refuses in a header; it must never appear in a message.
BROKEN_KEY = "dl_live_secret\nInjected: 1"

## The minimal YAM body the shape tests send: valid enough to pass the client's checks.
CONFIG = {"state": {"arm": [0, 1]}, "action": {"arm": [0, 1]}, "video": {"top": "observation.images.top"}}


class JsonService:
    """A threaded HTTP server that answers every request with one configurable JSON body, as a misplaced service would.

    Attributes:
    - url: The base URL.
    - body: The JSON value every request is answered with.
    - status: The HTTP status every request is answered with.
    """

    def __init__(self) -> None:
        """Starts the server on a free port."""
        self.body: Any = None
        self.status = 200
        fake = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def _answer(self) -> None:
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                raw = json.dumps(fake.body).encode()
                self.send_response(fake.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            do_GET = _answer  # noqa: N815  # http.server API
            do_POST = _answer  # noqa: N815  # http.server API

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002  # http.server API
                pass

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        """Stops the server."""
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def service(monkeypatch: pytest.MonkeyPatch) -> Iterator[JsonService]:
    """The one-answer service, the data service and the platform both (`DREAM_DATA_URL`, `DREAM_API_URL`); stopped
    after the test."""
    fake = JsonService()
    monkeypatch.setenv("DREAM_DATA_URL", fake.url)
    monkeypatch.setenv("DREAM_API_URL", fake.url)
    yield fake
    fake.close()


@pytest.fixture
def loose(server: FakeDreamAPI) -> Iterator[dream.client.PolicyClient]:
    """An inference client with no embodiment checks, so only the JSON encoder and the reply are tested."""
    config = {"host": server.host, "port": server.port, "scheme": "http", "timeout_ms": 2000}
    with dream._open_policy({**config, "checkpoint": CHECKPOINT, "strict": False}) as opened:
        yield opened


def _answer_ok(actions: Any) -> Any:
    """A fake-gateway handler answering `DA-0000` with these actions on every request."""
    return lambda body: (200, {"session_id": "s", "status": envelope("DA-0000"), "actions": actions})


# -- get_action: the observation -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("video.top", [b"\xff\xd8 jpeg bytes, not base64"]),
        ("prompt", {"a set", "is not JSON"}),
        ("state.arm", np.array([object()])),
        (("a", "tuple"), 1),
    ],
)
def test_a_value_json_cannot_carry_is_an_observation_error_naming_the_key(
    server: FakeDreamAPI, loose: dream.client.PolicyClient, key: Any, value: Any
) -> None:
    """Bytes, a set, an object array or a tuple key were a TypeError from the encoder; nothing is sent either way."""
    with pytest.raises(
        DreamObservationError, match=re.escape(f"observation key {key!r} cannot be sent as JSON")
    ) as raised:
        loose.get_action({key: value, "prompt_ok": "fine"})
    assert isinstance(raised.value, ValueError)
    assert server.requests == []
    server.handler = None
    assert loose.get_action({"prompt": "the client still works"})["q_r_chunk"]


def test_a_circular_observation_is_an_observation_error(
    server: FakeDreamAPI, loose: dream.client.PolicyClient
) -> None:
    """A structure that holds itself was a ValueError from the encoder."""
    looped: dict[str, Any] = {}
    looped["self"] = looped
    with pytest.raises(DreamObservationError, match="'self' cannot be sent as JSON"):
        loose.get_action(looped)
    assert server.requests == []


def test_a_ragged_list_under_a_checked_shape_is_an_observation_error(
    server: FakeDreamAPI, config: dict[str, Any], observation: dict[str, Any]
) -> None:
    """numpy refuses to shape a ragged list with a ValueError; strict mode names the key instead."""
    observation["state.arm_r.joint_pos"] = [[0.0, 0.0, 0.0], [0.0]]
    with dream._open_policy(config) as client, pytest.raises(DreamObservationError, match="'state.arm_r.joint_pos'"):
        client.get_action(observation)
    assert server.requests == []


def test_a_key_that_is_not_a_string_passes_the_strict_check(
    server: FakeDreamAPI, config: dict[str, Any], observation: dict[str, Any]
) -> None:
    """The unexpected-key check read `"." in key`, a TypeError for an int key; JSON sends an int key as a string."""
    observation[7] = "a number key"
    with dream._open_policy(config) as client:
        assert client.get_action(observation)["q_r_chunk"]
    assert server.requests[0].body["observation"]["7"] == "a number key"


def test_a_measured_pose_that_is_not_numbers_only_skips_the_blend(
    server: FakeDreamAPI, loose: dream.client.PolicyClient
) -> None:
    """The blend read the pose as floats, a ValueError for strings; the arm now gets the filter without the seam."""
    for _ in range(2):
        action = loose.get_action({"state.arm_r.joint_pos": ["not", "numbers"]})
    assert np.asarray(action["q_r_chunk"]).shape == (4, 7)


# -- get_action: the reply -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("actions", [None, [1, 2]])
def test_a_reply_with_no_action_dict_is_an_action_error(
    server: FakeDreamAPI, loose: dream.client.PolicyClient, actions: Any
) -> None:
    """Without strict checks, a reply whose `actions` is not a dict was a TypeError from the blend."""
    server.handler = _answer_ok(actions)
    with pytest.raises(DreamActionError, match="no action dict"):
        loose.get_action({"prompt": "go"})


@pytest.mark.parametrize("chunk", [[[1.0, 2.0], [3.0]], [["a", "b"], ["c", "d"]], [[{}, {}]]])
def test_a_chunk_that_is_not_rows_of_numbers_is_an_action_error_naming_it(
    server: FakeDreamAPI, loose: dream.client.PolicyClient, chunk: Any
) -> None:
    """A ragged chunk, or one of strings or objects, was a ValueError or TypeError from the blend."""
    server.handler = _answer_ok({"q_r_chunk": chunk})
    with pytest.raises(DreamActionError, match="'q_r_chunk' is not rows of numbers"):
        loose.get_action({"prompt": "go"})


# -- get_action: the key and the host ------------------------------------------------------------------------------


@pytest.mark.parametrize("key", [BROKEN_KEY, "dl_live_’quoted"])
def test_a_key_http_cannot_carry_is_a_config_error_that_never_shows_the_key(
    server: FakeDreamAPI, config: dict[str, Any], observation: dict[str, Any], key: str
) -> None:
    """http.client refused it with a ValueError (whose text is the whole header) or a UnicodeEncodeError."""
    with dream._open_policy(config, api_key=key) as client:
        with pytest.raises(DreamConfigError, match="api key holds a character") as raised:
            client.get_action(observation)
        assert key not in str(raised.value) and "secret" not in str(raised.value)
        client.reset_episode()
    assert server.requests == []


@pytest.mark.parametrize("host", ["a..b", "with space", "x" * 70 + ".example"])
def test_a_host_that_is_no_host_name_is_a_config_error_naming_it(host: str, observation: dict[str, Any]) -> None:
    """An empty or overlong label was a UnicodeError from the resolver, a space an InvalidURL from http.client."""
    with dream._open_policy(
        {"checkpoint": CHECKPOINT, "host": host, "port": 8000, "scheme": "http", "strict": False}
    ) as client:
        with pytest.raises(DreamConfigError, match="cannot be sent"):
            client.get_action(observation)


# -- URLs ----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("url", ["http://127.0.0.1:abc", "http://127.0.0.1:99999", "http://[::1"])
def test_a_gateway_url_with_a_bad_port_is_a_config_error_naming_the_url(
    url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """credentials.gateway_parts read the port unchecked, a ValueError from urllib; the credentials refuse the URL
    first, naming the variable."""
    monkeypatch.setenv("DREAM_GATEWAY_URL", url)
    with pytest.raises(DreamConfigError, match=re.escape("DREAM_GATEWAY_URL (from the environment) is not a URL")):
        dream.DreamClient(api_key="k")
    with pytest.raises(DreamConfigError, match="is not a URL"):
        credentials.gateway_parts(url)


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("http://127.0.0.1:abc", "is not a URL"),
        ("http://127.0.0.1:99999", "is not a URL"),
        ("http://[::1", "is not a URL"),
        ("http://a..b", "cannot send to"),
        ("http://with space", "host must be an http"),
    ],
)
def test_a_data_or_platform_url_that_is_no_url_is_a_config_error_naming_it(
    url: str, message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An InvalidURL or a ValueError from the URL parser, the connection or the resolver, before any retry; a URL
    the credentials can already tell is none is refused there."""
    monkeypatch.setenv("DREAM_DATA_URL", url)
    with pytest.raises(DreamConfigError, match=message):
        dream.DreamClient(api_key="k").get_embodiments()
    monkeypatch.delenv("DREAM_DATA_URL")
    monkeypatch.setenv("DREAM_API_URL", url)
    with pytest.raises(DreamConfigError, match=message):
        dream.DreamClient(api_key="k").get_models()


def test_a_data_call_with_a_key_http_cannot_carry_is_a_config_error(service: JsonService) -> None:
    """The data calls built the same header, with the same ValueError holding the key."""
    with pytest.raises(DreamConfigError, match="api key holds a character") as raised:
        datasets.list_embodiments(api_key=BROKEN_KEY)
    assert "secret" not in str(raised.value)


# -- answers of the wrong shape ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "call"),
    [
        (None, lambda: datasets.list_embodiments(api_key="k")),
        ({"not": "a list"}, lambda: datasets.list_embodiments(api_key="k")),
        ([{}], lambda: datasets.list_embodiments(api_key="k")),
        (
            [{"name": "a", "embodiment_config": {}, "created_at": "yesterday"}],
            lambda: datasets.list_embodiments(api_key="k"),
        ),
        ({}, lambda: datasets.get_dataset("ds_1", api_key="k")),
        ([], lambda: datasets.get_dataset("ds_1", api_key="k")),
        ({}, lambda: datasets.retry_dataset("ds_1", api_key="k")),
        ([], lambda: datasets.retry_dataset("ds_1", api_key="k")),
        ([{}], lambda: models.list_models(api_key="k")),
        (None, lambda: models.list_models(api_key="k")),
        (None, lambda: fine_tuning.list(api_key="k")),
        ({"id": "e"}, lambda: datasets.add_embodiment("arm", CONFIG, api_key="k")),
    ],
)
def test_an_answer_of_another_shape_is_a_protocol_error(service: JsonService, body: Any, call: Any) -> None:
    """A KeyError, TypeError or ValueError from reading a record: another service is listening at the URL."""
    service.body = body
    with pytest.raises(DreamProtocolError, match="answered"):
        call()


# -- request values ------------------------------------------------------------------------------------------------


def test_a_numpy_number_as_a_knob_is_a_config_error(service: JsonService) -> None:
    """json.dumps raised TypeError for a numpy int; nothing is sent."""
    with pytest.raises(DreamConfigError, match="cannot be sent as JSON"):
        fine_tuning.create("alice/towels", "towels", batch_size=np.int64(16), api_key="k")  # type: ignore[arg-type]  # the point of the test


def test_an_embodiment_config_keyed_by_numbers_is_a_config_error(service: JsonService) -> None:
    """The parser's name check raised TypeError for a part name that is not a string."""
    config = {**CONFIG, "state": {1: [0, 1]}}
    with pytest.raises(DreamConfigError, match="must be strings"):
        datasets.add_embodiment("arm", config, api_key="k")
    with pytest.raises(DreamConfigError, match="must be str"):
        dream._open_policy({"checkpoint": CHECKPOINT, "embodiment": config})


# -- local files ---------------------------------------------------------------------------------------------------


def test_a_dataset_path_holding_a_nul_byte_is_a_config_error() -> None:
    """The path check raised ValueError (embedded null byte) from the operating system call."""
    with pytest.raises(DreamConfigError, match="cannot list the dataset directory"):
        datasets.validate_dataset("data\x00set", CONFIG)


def test_a_dataset_directory_that_cannot_be_read_is_a_config_error(tmp_path: pathlib.Path) -> None:
    """Looking for `meta/info.json` in a directory without read permission raised PermissionError."""
    locked = tmp_path / "locked"
    (locked / "meta").mkdir(parents=True)
    locked.chmod(0)
    if os.access(locked, os.R_OK | os.X_OK):  # root reads it anyway, so there is no failure to see
        locked.chmod(0o755)
        pytest.skip("this process reads directories without permission")
    try:
        with pytest.raises(DreamConfigError, match="cannot list the dataset directory"):
            datasets.validate_dataset(locked, CONFIG)
    finally:
        locked.chmod(0o755)


def test_a_credentials_file_that_is_not_text_is_a_config_error_naming_it(tmp_path: pathlib.Path) -> None:
    """Reading it as UTF-8 raised UnicodeDecodeError."""
    path = tmp_path / "credentials"
    path.write_bytes(b"\xff\xfe binary")
    with pytest.raises(DreamConfigError, match=f"cannot read the credentials file {path}"):
        credentials.load({credentials.FILE_ENV: str(path)})
