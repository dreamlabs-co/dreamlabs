"""Contract tests C-1..C-17 and C-27..C-38: wire shape, config split, sessions, validation.

Every test is named by its requirement id and runs against the in-process `FakeDreamAPI` from
`conftest.py`, which speaks DreamAPI's `/v1/get_action` contract independently of the library.
Bytes on the wire are compared as the fake recorded them.
"""

from __future__ import annotations

import importlib
import json
import pathlib
import time
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from conftest import CHECKPOINT, EMBODIMENT, HORIZON, TIMING, FakeDreamAPI, envelope, make_observation

import dream
from dream.client import PolicyClient
from dream.errors import (
    DreamActionError,
    DreamAuthError,
    DreamConfigError,
    DreamError,
    DreamObservationError,
    DreamServerError,
    DreamSessionExpired,
    DreamTimeout,
)

## Frozen module namespace: `dir(dream)` must equal exactly this, sorted.
MODULE_SURFACE = [
    "DreamClient",
    "EventChange",
    "LiveViewClient",
    "DreamActionError",
    "DreamAuthError",
    "DreamBadRequest",
    "DreamConfigError",
    "DreamError",
    "DreamObservationError",
    "DreamProtocolError",
    "DreamServerError",
    "DreamSessionExpired",
    "DreamSlowNetworkWarning",
    "DreamTimeout",
    "__version__",
]

## Frozen client surface: the only non-underscore members a user can discover.
CLIENT_SURFACE = [
    "get_action",
    "reset_episode",
    "close",
    "last_timing",
    "mode",
    "refill_due",
    "execution_horizon",
    "action_hz",
]

## Frozen public instance attributes; anything more is a surface change.
CLIENT_ATTRIBUTES = ["api_token", "host", "port", "scheme", "timeout_ms"]

## Every request body carries exactly these keys (the gateway's request model).
BODY_KEYS = {"checkpoint", "session_id", "reset", "observation"}

## Anything slower than this for a local, no-I/O call is blocking on the network.
INSTANT_S = 0.5

## Number of round trips for the "checkpoint on every request" sweep (C-33, the default run's budget).
SWEEP_REQUESTS = 1_000


@pytest.fixture
def client(config: dict[str, Any]) -> Iterator[PolicyClient]:
    """A strict client from `dream.init` against `server`, closed on teardown."""
    with dream._open_policy(config) as opened:
        yield opened


# ----------------------------------------------------------------------------- round trip


def test_c01_round_trip_returns_action_dict(client: PolicyClient, observation: dict[str, Any]) -> None:
    """C-1 C-1a: get_action returns the serve's action dict itself; no `info` anywhere."""
    action = client.get_action(observation)

    assert type(action) is dict
    assert np.asarray(action["q_r_chunk"]).shape == (HORIZON, 7)
    assert np.asarray(action["q_l_chunk"]).shape == (HORIZON, 7)
    assert "info" not in action


def test_c01a_no_info_on_the_wire(server: FakeDreamAPI, client: PolicyClient, observation: dict[str, Any]) -> None:
    """C-1a: no `info` key in the request body or the reply."""
    client.get_action(observation)

    assert "info" not in server.requests[0].body
    assert b"info" not in server.requests[0].raw


def test_c02_request_body_is_the_v1_schema(
    server: FakeDreamAPI, client: PolicyClient, observation: dict[str, Any]
) -> None:
    """C-2: every request is `POST /v1/get_action` with exactly `{checkpoint, session_id, reset, observation}`."""
    client.get_action(observation)
    client.get_action(observation)

    assert server.other_paths == []
    for request in server.requests:
        assert set(request.body) == BODY_KEYS
        assert request.body["checkpoint"] == CHECKPOINT
        assert request.body["reset"] is False
        assert request.headers["content-type"] == "application/json"


def test_c03_get_action_takes_the_observation_only(
    server: FakeDreamAPI, client: PolicyClient, observation: dict[str, Any]
) -> None:
    """C-3: `get_action(observation)` has no other positional parameter, and its one keyword (`commanded`, C-42l)
    stays local; key and session id are packaged internally."""
    import inspect

    parameters = inspect.signature(client.get_action).parameters
    assert list(parameters) == ["observation", "commanded"]
    assert parameters["commanded"].kind is inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(TypeError):
        client.get_action(observation, {"temperature": 0.1})  # type: ignore[call-arg]  # the point of the test
    assert server.requests == []


def test_c04_not_a_policy(client: PolicyClient) -> None:
    """C-4 C-13: no `BasePolicy` in the MRO; the policy hooks do not exist."""
    assert type(client).__mro__ == (PolicyClient, object)
    for name in ("check_observation", "check_action", "_get_action"):
        assert not hasattr(client, name), name


# ----------------------------------------------------------------------------- api key


def test_c05_no_credential_without_key(
    server: FakeDreamAPI, client: PolicyClient, observation: dict[str, Any]
) -> None:
    """C-5 C-32: `api_key=None` means no Authorization header; the key is never in the body."""
    client.get_action(observation)

    assert "authorization" not in server.requests[0].headers
    assert "api_token" not in server.requests[0].body


def test_c06_bearer_token_on_every_request(
    server: FakeDreamAPI, config: dict[str, Any], observation: dict[str, Any]
) -> None:
    """C-6 C-32: `api_key="x"` travels as `Authorization: Bearer x` on every request, never in the body."""
    with dream._open_policy(config, api_key="x") as client:
        for _ in range(3):
            client.get_action(observation)

    assert [request.headers["authorization"] for request in server.requests] == ["Bearer x"] * 3
    assert all(b'"api_token"' not in request.raw for request in server.requests)


# ----------------------------------------------------------------------------- validation


def test_c07_strict_passes_and_bytes_match_non_strict(
    server: FakeDreamAPI, config: dict[str, Any], observation: dict[str, Any]
) -> None:
    """C-7 C-7f C-7h: a valid observation passes; identical body to `strict=False`; no extra requests."""
    with dream._open_policy(config) as strict_client:
        strict_client.get_action(observation)
    with dream._open_policy({**config, "strict": False}) as loose_client:
        loose_client.get_action(observation)

    assert len(server.requests) == 2
    assert server.requests[0].body["observation"] == server.requests[1].body["observation"]


def test_c07a_missing_key_sends_nothing(
    server: FakeDreamAPI, client: PolicyClient, observation: dict[str, Any]
) -> None:
    """C-7a: a missing observation key is DreamObservationError naming it, and nothing is sent."""
    del observation["state.arm_l.gripper_pos"]

    with pytest.raises(DreamObservationError, match="state.arm_l.gripper_pos"):
        client.get_action(observation)
    assert server.requests == []


def test_c07b_wrong_shape_names_both_shapes(client: PolicyClient, observation: dict[str, Any]) -> None:
    """C-7b: the message carries the actual and the declared shape."""
    observation["state.arm_r.joint_pos"] = np.zeros(7, dtype=np.float32)

    with pytest.raises(DreamObservationError, match=r"\(7,\).*\(6,\)"):
        client.get_action(observation)


def test_c07c_wrong_dtype_is_caught(client: PolicyClient, observation: dict[str, Any]) -> None:
    """C-7c: float64 where float32 is declared is DreamObservationError."""
    observation["state.arm_r.joint_pos"] = np.zeros(6, dtype=np.float64)

    with pytest.raises(DreamObservationError, match="float64"):
        client.get_action(observation)


def test_c07d_unexpected_dotted_key_is_rejected(client: PolicyClient, observation: dict[str, Any]) -> None:
    """C-7d: a typo'd grouped key is rejected, not silently forwarded."""
    observation["state.arm_r.joint_poss"] = observation["state.arm_r.joint_pos"]

    with pytest.raises(DreamObservationError, match="joint_poss"):
        client.get_action(observation)


def test_c07e_mis_shaped_action_is_not_returned(
    server: FakeDreamAPI, client: PolicyClient, observation: dict[str, Any]
) -> None:
    """C-7e: a chunk with the wrong row count is DreamActionError and never returned."""
    server.horizon = HORIZON + 1

    with pytest.raises(DreamActionError, match="q_r_chunk"):
        client.get_action(observation)


def test_c07g_no_embodiment_skips_validation(server: FakeDreamAPI, config: dict[str, Any]) -> None:
    """C-7g: `strict=True` with no `embodiment` validates nothing and is not an error."""
    del config["embodiment"]
    with dream._open_policy(config) as client:
        client.get_action({"anything": [1, 2, 3]})

    assert server.requests[0].body["observation"] == {"anything": [1, 2, 3]}


def test_c07i_validation_cost_is_independent_of_array_size(config: dict[str, Any]) -> None:
    """C-7i: validating a 4-camera 1080p observation costs the same as a tiny one."""
    big_config = {
        **config,
        "embodiment": {"video": {f"cam{i}": {"shape": [1080, 1920, 3], "dtype": "uint8"} for i in range(4)}},
    }
    small_config = {**config, "embodiment": {"video": {"c": {"shape": [2, 2, 3], "dtype": "uint8"}}}}
    big = {f"video.cam{i}": np.zeros((1080, 1920, 3), dtype=np.uint8) for i in range(4)}
    small = {"video.c": np.zeros((2, 2, 3), dtype=np.uint8)}
    with dream._open_policy(big_config) as big_client, dream._open_policy(small_config) as small_client:
        big_spec = dream.client._observation_spec(big_client._embodiment)
        small_spec = dream.client._observation_spec(small_client._embodiment)
    started = time.perf_counter()
    for _ in range(200):
        dream.client._check(big, big_spec, DreamObservationError)
    big_s = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(200):
        dream.client._check(small, small_spec, DreamObservationError)
    small_s = time.perf_counter() - started

    assert big_s < 0.05 and big_s < 5 * small_s + 0.01


def test_c07j_prompt_and_extra_action_fields_pass(
    server: FakeDreamAPI, client: PolicyClient, observation: dict[str, Any]
) -> None:
    """C-7j: an ungrouped observation key (`prompt`) is metadata; extra action fields from the serve are tolerated."""
    action = client.get_action(observation)

    assert server.requests[0].body["observation"]["prompt"] == "pick up the cup"
    assert "per_key" in action and "q_r" in action


# ----------------------------------------------------------------------------- surface


def test_c08_public_surface_snapshot(client: PolicyClient) -> None:
    """C-8 C-21: module and client surfaces match the frozen lists."""
    assert sorted(dir(dream)) == sorted(MODULE_SURFACE)
    assert sorted(dream.__all__) == sorted(MODULE_SURFACE)
    public = sorted(name for name in vars(PolicyClient) if not name.startswith("_"))
    assert public == sorted(CLIENT_SURFACE)
    attributes = sorted(name for name in vars(client) if not name.startswith("_"))
    assert attributes == sorted(CLIENT_ATTRIBUTES)


def test_c09_numpy_arrays_travel_as_json_lists(
    server: FakeDreamAPI, client: PolicyClient, observation: dict[str, Any]
) -> None:
    """C-9: float32 arrays are sent as JSON lists with their values intact; the reply's lists come back as sent."""
    observation["state.arm_r.joint_pos"] = np.arange(6, dtype=np.float32) / 4
    action = client.get_action(observation)

    sent = json.loads(server.requests[0].raw)["observation"]
    assert sent["state.arm_r.joint_pos"] == [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
    assert sent["state.arm_r.gripper_pos"] == [1.0]
    assert action["q_l_chunk"] == [[0.1] * 7] * HORIZON


def test_c10_no_lifecycle_or_hook_members(client: PolicyClient) -> None:
    """C-10 C-20: `kill_server` and the other GR00T members are gone, not aliased."""
    for name in ("kill_server", "ping", "call_endpoint", "reset", "get_modality_config"):
        assert not hasattr(client, name), name


def test_c11_only_get_action_is_contacted(
    server: FakeDreamAPI, client: PolicyClient, observation: dict[str, Any]
) -> None:
    """C-11: every public call touches `/v1/get_action` and nothing else."""
    client.get_action(observation)
    client.reset_episode()
    client.get_action(observation)
    client.close()

    assert len(server.requests) == 2
    assert server.other_paths == []


def test_c12_module_exports() -> None:
    """C-12: `dream` exposes `DreamClient`, the errors and `__version__`; never `init`, `PolicyClient` or an
    `experimental` module: `DreamClient` is the one way to run a model."""
    assert callable(dream.DreamClient)
    assert isinstance(dream.__version__, str)
    assert not hasattr(dream, "PolicyClient")
    assert not hasattr(dream, "init")
    with pytest.raises(ImportError):
        importlib.import_module("dream.experimental")


def test_c14_init_and_direct_construction_send_the_same_bytes(
    server: FakeDreamAPI, config: dict[str, Any], observation: dict[str, Any]
) -> None:
    """C-14: `init` adds no behaviour beyond the config split."""
    with dream._open_policy(config) as from_init:
        from_init.get_action(observation)
    direct = PolicyClient(
        host=config["host"],
        port=config["port"],
        timeout_ms=config["timeout_ms"],
        session_config={"checkpoint": CHECKPOINT, "embodiment": EMBODIMENT},
    )
    with direct:
        direct.get_action(observation)

    assert server.requests[0].raw == server.requests[1].raw


@pytest.mark.parametrize(
    "key, value",
    [
        ("timeout_ms", "soon"),
        ("timeout_ms", 0),
        ("timeout_ms", 2**31),
        ("port", 70000),
        ("host", ""),
        ("strict", 1),
        ("scheme", "ftp"),
    ],
)
def test_c15_bad_connection_key_names_the_key(fd_guard: None, key: str, value: Any) -> None:
    """C-15 C-17: a wrong-typed connection key is DreamConfigError naming it, before any connection exists."""
    with pytest.raises(DreamConfigError, match=key):
        dream._open_policy({"checkpoint": CHECKPOINT, key: value})


def test_c16_defaults() -> None:
    """C-16: `dream._open_policy({"checkpoint": ...})` uses the documented defaults: production's gateway, no key."""
    with dream._open_policy({"checkpoint": CHECKPOINT}) as client:
        assert (client.host, client.port, client.timeout_ms, client.scheme, client.api_token) == (
            "api.dreamlabs.ai",
            443,
            15000,
            "https",
            None,
        )


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://api.example.com", ("api.example.com", 443, "https")),
        ("https://api.example.com/", ("api.example.com", 443, "https")),
        ("http://127.0.0.1:8000", ("127.0.0.1", 8000, "http")),
        ("http://gw.example", ("gw.example", 80, "http")),
        ("https://api.example.com:8443", ("api.example.com", 8443, "https")),
    ],
)
def test_c16c_gateway_url_env_replaces_the_gateway_defaults(
    monkeypatch: pytest.MonkeyPatch, url: str, expected: tuple[str, int, str]
) -> None:
    """C-16c: `DREAM_GATEWAY_URL` replaces the `host`, `port` and `scheme` defaults; `timeout_ms` keeps its own."""
    monkeypatch.setenv("DREAM_GATEWAY_URL", url)
    with dream._open_policy({"checkpoint": CHECKPOINT}) as client:
        assert (client.host, client.port, client.scheme, client.timeout_ms) == (*expected, 15000)


def test_c16c_config_keys_win_over_the_gateway_url_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """C-16c: a key the config sets wins over `DREAM_GATEWAY_URL`, key by key, as it wins over a default."""
    monkeypatch.setenv("DREAM_GATEWAY_URL", "https://api.example.com")
    with dream._open_policy({"checkpoint": CHECKPOINT, "host": "127.0.0.1", "port": 8000}) as client:
        assert (client.host, client.port, client.scheme) == ("127.0.0.1", 8000, "https")


@pytest.mark.parametrize(
    "url",
    ["api.example.com", "ftp://api.example.com", "https://", "https://api.example.com/v1", "https://h:99999"],
)
def test_c16d_bad_gateway_url_env_names_the_variable(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    """C-16d: a `DREAM_GATEWAY_URL` that is not http(s)://<host>[:<port>] is DreamConfigError naming it."""
    monkeypatch.setenv("DREAM_GATEWAY_URL", url)
    with pytest.raises(DreamConfigError, match="DREAM_GATEWAY_URL"):
        dream._open_policy({"checkpoint": CHECKPOINT})


def test_c16d_empty_gateway_url_env_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """C-16d: an empty `DREAM_GATEWAY_URL` is the same as unset: the built-in defaults hold."""
    monkeypatch.setenv("DREAM_GATEWAY_URL", "  ")
    with dream._open_policy({"checkpoint": CHECKPOINT}) as client:
        assert (client.host, client.port, client.scheme) == ("api.dreamlabs.ai", 443, "https")


def test_c16e_credentials_file_supplies_the_gateway_and_the_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """C-16e: `~/.dreamlabs/credentials` (here `DREAM_CREDENTIALS`) supplies the gateway and the key `init` uses."""
    path = tmp_path / "credentials"
    path.write_text(
        "DREAM_GATEWAY_URL=https://staging.api.example.com\nDREAM_API_KEY=dl_live_file\n", encoding="utf-8"
    )
    monkeypatch.setenv("DREAM_CREDENTIALS", str(path))
    with dream._open_policy({"checkpoint": CHECKPOINT}) as client:
        assert (client.host, client.port, client.scheme, client.api_token) == (
            "staging.api.example.com",
            443,
            "https",
            "dl_live_file",
        )


def test_c16e_env_and_the_api_key_argument_win_over_the_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """C-16e: name by name, the environment wins over the file, and an `api_key` passed to `init` wins over both."""
    path = tmp_path / "credentials"
    path.write_text("DREAM_GATEWAY_URL=https://file.example.com\nDREAM_API_KEY=dl_live_file\n", encoding="utf-8")
    monkeypatch.setenv("DREAM_CREDENTIALS", str(path))
    monkeypatch.setenv("DREAM_GATEWAY_URL", "http://127.0.0.1:8000")
    with dream._open_policy({"checkpoint": CHECKPOINT}) as client:
        assert (client.host, client.port, client.scheme, client.api_token) == (
            "127.0.0.1",
            8000,
            "http",
            "dl_live_file",
        )
    with dream._open_policy({"checkpoint": CHECKPOINT}, api_key="dl_live_arg") as client:
        assert client.api_token == "dl_live_arg"


def test_c16f_bad_gateway_url_in_the_file_names_the_variable_and_the_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """C-16f: a malformed `DREAM_GATEWAY_URL` in the file is `DreamConfigError` naming the variable and the file."""
    path = tmp_path / "credentials"
    path.write_text("DREAM_GATEWAY_URL=api.example.com\n", encoding="utf-8")
    monkeypatch.setenv("DREAM_CREDENTIALS", str(path))
    with pytest.raises(DreamConfigError, match=r"DREAM_GATEWAY_URL.*credentials"):
        dream._open_policy({"checkpoint": CHECKPOINT})


def test_c16b_https_scheme_speaks_tls() -> None:
    """The `scheme` key sets the transport: `https` opens an HTTPS connection to `host:port`."""
    import http.client

    with dream._open_policy(
        {"checkpoint": CHECKPOINT, "host": "gw.example", "port": 443, "scheme": "https"}
    ) as client:
        assert client.scheme == "https"
        # No socket is opened until a request; _connect() builds the connection object to inspect its class.
        assert isinstance(client._transport._connect(), http.client.HTTPSConnection)
        assert client._transport._endpoint() == "https://gw.example:443"


def test_c16a_checkpoint_is_required() -> None:
    """C-16a: without `checkpoint` there is nothing to route to; DreamConfigError at init."""
    with pytest.raises(DreamConfigError, match="checkpoint"):
        dream._open_policy()
    with pytest.raises(DreamConfigError, match="checkpoint"):
        dream._open_policy({"checkpoint": ""})


# ----------------------------------------------------------------------------- config split


def test_c27_connection_keys_never_reach_the_wire(
    server: FakeDreamAPI, client: PolicyClient, observation: dict[str, Any]
) -> None:
    """C-27: host/port/timeout_ms/strict and the embodiment are consumed locally; only `checkpoint` is sent."""
    client.get_action(observation)

    body = server.requests[0].body
    for key in ("host", "port", "timeout_ms", "strict", "embodiment", "config"):
        assert key not in body, key
    assert body["checkpoint"] == CHECKPOINT


def test_c27a_unknown_config_key_fails_at_init() -> None:
    """C-27a C-28: a key DreamAPI has no field for is DreamConfigError at init, naming it."""
    with pytest.raises(DreamConfigError, match="temperature"):
        dream._open_policy({"checkpoint": CHECKPOINT, "temperature": 0.7})


def test_c29_first_request_carries_null_session(
    server: FakeDreamAPI, client: PolicyClient, observation: dict[str, Any]
) -> None:
    """C-29: the first request has the `session_id` key, present and null."""
    client.get_action(observation)

    assert "session_id" in server.requests[0].body
    assert server.requests[0].body["session_id"] is None
    assert b'"session_id":null' in server.requests[0].raw


def test_c30_session_adoption(server: FakeDreamAPI, client: PolicyClient, observation: dict[str, Any]) -> None:
    """C-30: after `DA-0001 new` mints an id, every later request carries it and is answered `DA-0000`."""
    for _ in range(4):
        client.get_action(observation)

    minted = server.requests[1].body["session_id"]
    assert minted and minted in server.sessions
    assert [request.body["session_id"] for request in server.requests] == [None, minted, minted, minted]


def test_c31_bad_request_is_config_error(
    server: FakeDreamAPI, client: PolicyClient, observation: dict[str, Any]
) -> None:
    """C-31: `DA-4000` is DreamConfigError carrying the reason and DreamAPI's detail verbatim."""
    server.fail_next("DA-4000", "malformed", detail="observation: field required")

    with pytest.raises(DreamConfigError) as info:
        client.get_action(observation)

    assert "DA-4000" in str(info.value) and "malformed" in str(info.value)
    assert "observation: field required" in str(info.value)


def test_c33_checkpoint_on_every_request(
    server: FakeDreamAPI, client: PolicyClient, observation: dict[str, Any]
) -> None:
    """C-33: `checkpoint` is present and identical on request 1 and request N."""
    for _ in range(SWEEP_REQUESTS):
        client.get_action(observation)

    assert len(server.requests) == SWEEP_REQUESTS
    assert server.requests[0].body["checkpoint"] == server.requests[-1].body["checkpoint"] == CHECKPOINT


def test_c34_init_does_no_io(blackhole_port: int) -> None:
    """C-34: `init` returns at once against a port nothing listens on; the first call is what fails."""
    started = time.perf_counter()
    client = dream._open_policy({"host": "127.0.0.1", "port": blackhole_port, "checkpoint": CHECKPOINT})
    assert time.perf_counter() - started < INSTANT_S
    with client, pytest.raises(dream.DreamProtocolError):
        client.get_action({})


def test_c35_session_replaced_mid_episode_raises_and_adopts(
    server: FakeDreamAPI, client: PolicyClient, observation: dict[str, Any]
) -> None:
    """C-35: `DA-0001` with a reason other than `new` on a live session is DreamSessionExpired; new id adopted."""
    client.get_action(observation)
    client.get_action(observation)
    old = server.requests[-1].body["session_id"]
    server.restart()

    with pytest.raises(DreamSessionExpired) as info:
        client.get_action(observation)
    client.get_action(observation)

    assert info.value.code == "DA-0001" and info.value.reason == "unknown"
    new = server.requests[-1].body["session_id"]
    assert new != old and new in server.sessions


def test_c36_other_codes_pass_through_with_the_envelope(
    server: FakeDreamAPI, client: PolicyClient, observation: dict[str, Any]
) -> None:
    """C-36: `DA-2000` (and any future code) is DreamServerError carrying code, reason, retryable, retry_after_s."""
    server.fail_next("DA-2000", "pool_warming", retry_after_s=3.5, detail="serve booting")

    with pytest.raises(DreamServerError) as info:
        client.get_action(observation)

    error = info.value
    assert (error.code, error.reason, error.retryable, error.retry_after_s) == ("DA-2000", "pool_warming", True, 3.5)
    assert "serve booting" in str(error)


def test_c37_unauthorized_is_never_retried(
    server: FakeDreamAPI, client: PolicyClient, observation: dict[str, Any]
) -> None:
    """C-37: `DA-1000` is DreamAuthError and the request is sent exactly once."""
    server.fail_next("DA-1000", "unknown_key")

    with pytest.raises(DreamAuthError) as refused:
        client.get_action(observation)

    assert len(server.requests) == 1
    assert "; called http" in str(refused.value), "a refused key names the URL it was sent to and where that came from"
    assert "keys work only on the site that made them (dev, staging and production are separate)" in str(refused.value)


# ----------------------------------------------------------------------------- episodes


def test_c38_reset_episode_starts_a_new_session(
    server: FakeDreamAPI, client: PolicyClient, observation: dict[str, Any]
) -> None:
    """C-38: after a session was minted, `reset_episode()` sends null next and adopts the new id without error."""
    client.get_action(observation)
    client.get_action(observation)
    client.reset_episode()
    client.get_action(observation)
    client.get_action(observation)

    ids = [request.body["session_id"] for request in server.requests]
    assert ids[0] is None and ids[2] is None
    assert ids[1] and ids[3] and ids[3] != ids[1]


def test_c38a_reset_episode_is_local(server: FakeDreamAPI, client: PolicyClient, observation: dict[str, Any]) -> None:
    """C-38a: `reset_episode()` sends nothing; harmless before the first request or twice in a row."""
    client.reset_episode()
    client.reset_episode()
    client.get_action(observation)
    client.reset_episode()
    client.reset_episode()

    assert len(server.requests) == 1


def test_c38c_reset_episode_on_a_closed_client(config: dict[str, Any]) -> None:
    """C-38c: `reset_episode()` on a closed client is DreamError naming the operation."""
    client = dream._open_policy(config)
    client.close()

    with pytest.raises(DreamError, match="episode reset on a closed client"):
        client.reset_episode()


def test_c39_timeouts_are_dream_timeout(
    server: FakeDreamAPI, config: dict[str, Any], observation: dict[str, Any]
) -> None:
    """C-39: a held reply is DreamTimeout after about `timeout_ms`, and the next call works."""
    config["timeout_ms"] = 300
    server.hold_next = 1
    with dream._open_policy(config) as client:
        started = time.perf_counter()
        with pytest.raises(DreamTimeout):
            client.get_action(observation)
        assert 0.2 < time.perf_counter() - started < 1.5
        assert client.get_action(make_observation())["q_r_chunk"]


def test_c40_refusal_with_actions_is_still_a_refusal(
    server: FakeDreamAPI, client: PolicyClient, observation: dict[str, Any]
) -> None:
    """C-40: a non-OK status that carries an `actions` field never returns it."""

    def handler(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        return 502, {"session_id": "s", "status": envelope("DA-5000", "error"), "actions": {"q_r_chunk": []}}

    server.handler = handler
    with pytest.raises(DreamServerError):
        client.get_action(observation)


# ----------------------------------------------------------------------------- timing


def test_c41_last_timing_is_none_before_the_first_request(client: PolicyClient) -> None:
    """C-41: nothing has reached the wire, so there is nothing to report."""
    assert client.last_timing is None


def test_c41a_last_timing_splits_the_round_trip(
    server: FakeDreamAPI, client: PolicyClient, observation: dict[str, Any]
) -> None:
    """C-41a: after a success, `last_timing` carries the wire cost, DreamAPI's block and the network remainder."""
    reply = {
        "session_id": "11111111-1111-4111-8111-111111111111",
        "status": envelope("DA-0001", "new"),
        "actions": {"q_r_chunk": [[0.0] * 7] * HORIZON, "q_l_chunk": [[0.1] * 7] * HORIZON},
        "timing": dict(TIMING),
    }
    raw = json.dumps(reply).encode("utf-8")
    server.reply_override = (200, raw)
    client.get_action(observation)

    timing = client.last_timing
    assert timing is not None
    assert set(timing) == {
        "rtt_ms",
        "serve_infer_ms",
        "gateway_ms",
        "network_ms",
        "up_mbps",
        "tcp_cwnd_kb",
        "tcp_retrans",
        "tcp_rtt_ms",
        "tcp_app_limited",
        "request_bytes",
        "response_bytes",
    }
    assert isinstance(timing["rtt_ms"], float) and timing["rtt_ms"] > 0
    assert (timing["serve_infer_ms"], timing["gateway_ms"]) == (TIMING["serve_infer_ms"], TIMING["gateway_ms"])
    assert timing["network_ms"] == pytest.approx(timing["rtt_ms"] - TIMING["gateway_ms"])
    assert timing["request_bytes"] == len(server.requests[0].raw)
    assert timing["response_bytes"] == len(raw)


def test_c41b_last_timing_is_read_only(client: PolicyClient, observation: dict[str, Any]) -> None:
    """C-41b: every read is a fresh dict; mutating one changes nothing, and the property has no setter."""
    client.get_action(observation)
    first = client.last_timing
    assert first is not None
    first["rtt_ms"] = -1.0
    second = client.last_timing
    assert second is not None and second["rtt_ms"] > 0
    with pytest.raises(AttributeError):
        client.last_timing = None  # type: ignore[misc]


def test_c41c_last_timing_survives_a_refusal_and_a_timeout(
    server: FakeDreamAPI, config: dict[str, Any], observation: dict[str, Any]
) -> None:
    """C-41c: a refusal carries no block, so the split is unknown; a timeout still reports what it cost."""
    with dream._open_policy({**config, "timeout_ms": 200}) as client:
        server.fail_next("DA-2000", "pool_full", retry_after_s=1.0)
        with pytest.raises(DreamServerError):
            client.get_action(observation)
        refused = client.last_timing
        assert refused is not None
        assert refused["rtt_ms"] > 0 and refused["request_bytes"] > 0 and refused["response_bytes"] > 0
        assert (refused["serve_infer_ms"], refused["gateway_ms"], refused["network_ms"]) == (None, None, None)

        server.hold_next = 1
        with pytest.raises(DreamTimeout):
            client.get_action(observation)
        timed_out = client.last_timing
        assert timed_out is not None
        assert timed_out["rtt_ms"] >= 200 * 0.9, "the timeout's cost is visible"
        assert timed_out["response_bytes"] == 0 and timed_out["request_bytes"] > 0
        assert timed_out["gateway_ms"] is None and timed_out["network_ms"] is None


def test_c41d_a_reply_without_a_timing_block_leaves_the_split_unknown(
    server: FakeDreamAPI, client: PolicyClient, observation: dict[str, Any]
) -> None:
    """C-41d: a gateway older than the block sends none; the action still comes back and the split is `None`."""

    def without_timing(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        status, reply = server.default_reply(body)
        return status, {key: value for key, value in reply.items() if key != "timing"}

    server.handler = without_timing
    action = client.get_action(observation)
    assert "q_r_chunk" in action
    timing = client.last_timing
    assert timing is not None and timing["rtt_ms"] > 0
    assert (timing["serve_infer_ms"], timing["gateway_ms"], timing["network_ms"]) == (None, None, None)


def test_a_camera_window_must_be_a_list_of_strings(
    server: FakeDreamAPI, client: PolicyClient, observation: dict[str, Any]
) -> None:
    """A bare base64 string under `image_window.<role>` is refused locally, before any request.

    The serve treats a non-list window as an observation without cameras and answers with an unrelated
    "key not found" error; the client names the real mistake instead.
    """
    observation["image_window.top"] = observation["image_window.top"][0]
    with pytest.raises(DreamObservationError, match="image_window.top.*frame window"):
        client.get_action(observation)
    assert server.requests == []


def test_session_boundaries_are_logged_once_each(
    server: FakeDreamAPI, client: PolicyClient, observation: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """A run log names every session the gateway minted: one line at the start, none on the echoes."""
    import logging

    with caplog.at_level(logging.INFO, logger="dream.client"):
        client.get_action(observation)
        client.get_action(observation)
        client.get_action(observation)
    lines = [record.getMessage() for record in caplog.records if record.name == "dream.client"]
    assert len(lines) == 1 and lines[0].startswith("session ") and "started" in lines[0]
